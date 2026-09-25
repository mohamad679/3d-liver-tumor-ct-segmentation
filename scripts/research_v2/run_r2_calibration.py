#!/usr/bin/env python3
"""Run the preregistered R2 real-train timing and memory calibration only."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ["MPLBACKEND"] = "Agg"

import nibabel as nib
import numpy as np

from protoem_ct.artifacts import sha256_file
from protoem_ct.research_v2.r1_audit import (
    grid_for_spacing,
    reorient_array_to_ras,
    resample_array_to_grid,
)
from protoem_ct.research_v2.r2_overfit import (
    R2_CALIBRATION_MAX_STEPS,
    R2_PRIMARY_MAX_STEPS,
    R2_SELECTED_CASES,
    R2_THRESHOLD,
    choose_random_center,
    choose_random_positive_center,
    positive_case_index_for_step,
    training_role_for_step,
)
from protoem_ct.research_v2.r2_preflight import (
    R2_PATCH_SIZE,
    R2_SEED,
    R2_TARGET_SPACING_MM,
    extract_centered_patch,
    normalize_r2_ct,
    r2_binary_tumor_target,
)

EXPECTED_LOCK_SHA256 = "c9ee200234eebf8c67fd97ff2c96ca0765f4067a8a69a7ef439deafdda3530a8"
EXPECTED_PREFLIGHT_SHA256 = "0d55bfe51604518e1e2c5c7245779217b689be3207eee83783a9709f1924911f"
CONFIG_RELATIVE_PATH = Path("configs/research_v2/r2_overfit_real_v1.json")
LOCK_RELATIVE_PATH = Path("environments/research-v2-linux-cuda/uv.lock")


@dataclass(slots=True)
class PreparedCase:
    anonymous_case_id: str
    role: str
    image: Any
    tumor: Any


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _verify_preflight(path: Path) -> None:
    if sha256_file(path) != EXPECTED_PREFLIGHT_SHA256:
        raise RuntimeError("R2 preflight SHA-256 differs from the verified artifact")
    evidence = _load_json(path)
    if evidence.get("status") != "PASS":
        raise RuntimeError("R2 preflight evidence is not PASS")
    if evidence.get("main_training_started") is not False:
        raise RuntimeError("R2 preflight evidence is not pre-training evidence")

    environment = evidence.get("environment")
    gpu = evidence.get("gpu")
    step = evidence.get("forward_backward_optimizer_step")
    access = evidence.get("array_access")
    if not isinstance(environment, dict):
        raise RuntimeError("R2 preflight environment evidence is missing")
    if environment.get("uv_lock_sha256_verified") is not True:
        raise RuntimeError("R2 preflight did not verify the frozen uv.lock")
    if not isinstance(gpu, dict) or gpu.get("cuda_available") is not True:
        raise RuntimeError("R2 preflight did not verify CUDA")
    if gpu.get("training_device") != "cuda:0":
        raise RuntimeError("R2 preflight training device changed")
    if gpu.get("cpu_fallback_allowed") is not False:
        raise RuntimeError("R2 preflight allowed CPU fallback")
    if not isinstance(step, dict) or step.get("status") != "PASS":
        raise RuntimeError("R2 preflight optimizer step did not PASS")
    finite_fields = ("finite_logits", "finite_loss", "finite_gradients")
    if not all(step.get(field) is True for field in finite_fields):
        raise RuntimeError("R2 preflight finite checks did not all PASS")
    if not isinstance(access, dict):
        raise RuntimeError("R2 preflight array-access evidence is missing")
    forbidden_fields = (
        "validation_arrays_opened",
        "internal_test_arrays_opened",
        "external_arrays_opened",
    )
    for field in forbidden_fields:
        if int(access.get(field, -1)) != 0:
            raise RuntimeError(f"R2 preflight leakage boundary failed: {field}")


def _verify_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != "research_v2_r2_overfit_real.v1":
        raise RuntimeError("unexpected R2 config schema")
    training = config.get("training")
    inference = config.get("inference")
    budget = config.get("budget")
    if not isinstance(training, dict):
        raise RuntimeError("R2 training config is missing")
    if not isinstance(inference, dict):
        raise RuntimeError("R2 inference config is missing")
    if not isinstance(budget, dict):
        raise RuntimeError("R2 budget config is missing")

    expected_training = {
        "seed": R2_SEED,
        "augmentation": "disabled",
        "target_spacing_mm": list(R2_TARGET_SPACING_MM),
        "raw_tumor_label": 2,
        "patch_size": list(R2_PATCH_SIZE),
        "batch_size": 1,
        "positive_patch_fraction": 0.75,
        "learning_rate": 0.001,
        "weight_decay": 0.00001,
        "threshold": R2_THRESHOLD,
    }
    for field, expected in expected_training.items():
        if training.get(field) != expected:
            raise RuntimeError(f"R2 locked training field changed: {field}")

    if inference.get("sliding_window_roi_size") != list(R2_PATCH_SIZE):
        raise RuntimeError("R2 sliding-window ROI changed")
    if inference.get("sliding_window_overlap") != 0.5:
        raise RuntimeError("R2 sliding-window overlap changed")
    calibration = budget.get("timing_memory_calibration")
    primary = budget.get("primary")
    if not isinstance(calibration, dict) or not isinstance(primary, dict):
        raise RuntimeError("R2 budget entries are missing")
    if calibration.get("max_steps") != R2_CALIBRATION_MAX_STEPS:
        raise RuntimeError("R2 calibration budget changed")
    if primary.get("max_training_steps") != R2_PRIMARY_MAX_STEPS:
        raise RuntimeError("R2 primary step budget changed")


def _verify_selected_files(data_root: Path) -> None:
    expected_paths: set[Path] = set()
    for spec in R2_SELECTED_CASES:
        root = data_root / spec.anonymous_case_id
        image_path = root / "image.nii.gz"
        label_path = root / "label.nii.gz"
        expected_paths.update({image_path.resolve(), label_path.resolve()})
        if sha256_file(image_path) != spec.image_sha256:
            raise RuntimeError(f"R2 image hash mismatch: {spec.anonymous_case_id}")
        if sha256_file(label_path) != spec.label_sha256:
            raise RuntimeError(f"R2 label hash mismatch: {spec.anonymous_case_id}")

    observed_paths = {
        path.resolve()
        for pattern in ("*.nii", "*.nii.gz")
        for path in data_root.rglob(pattern)
        if path.is_file()
    }
    if observed_paths != expected_paths:
        raise RuntimeError("R2 data root must contain exactly eight selected train files")


def _prepare_case(data_root: Path, spec: Any) -> PreparedCase:
    root = data_root / spec.anonymous_case_id
    image_nifti = nib.load(str(root / "image.nii.gz"))
    label_nifti = nib.load(str(root / "label.nii.gz"))
    image = np.asarray(image_nifti.dataobj)
    label = np.asarray(label_nifti.dataobj)
    if image.ndim != 3 or label.ndim != 3 or image.shape != label.shape:
        raise RuntimeError(f"R2 source shape mismatch: {spec.anonymous_case_id}")
    if not bool(np.isfinite(image).all()):
        raise RuntimeError(f"R2 source CT contains NaN/Inf: {spec.anonymous_case_id}")

    image_affine = np.asarray(image_nifti.affine, dtype=np.float64)
    label_affine = np.asarray(label_nifti.affine, dtype=np.float64)
    if float(np.max(np.abs(image_affine - label_affine))) > 1e-4:
        raise RuntimeError(f"R2 source affine mismatch: {spec.anonymous_case_id}")

    native_tumor = r2_binary_tumor_target(label)
    if int(np.count_nonzero(native_tumor)) != int(spec.tumor_voxel_count):
        raise RuntimeError(f"R2 tumor voxel mismatch: {spec.anonymous_case_id}")

    ras_image, ras_image_affine = reorient_array_to_ras(image, image_affine)
    ras_tumor, ras_label_affine = reorient_array_to_ras(native_tumor, label_affine)
    target_shape, target_affine = grid_for_spacing(
        shape=ras_image.shape,
        affine=ras_image_affine,
        target_spacing_mm=R2_TARGET_SPACING_MM,
    )
    image_resampled = resample_array_to_grid(
        ras_image,
        source_affine=ras_image_affine,
        target_shape=target_shape,
        target_affine=target_affine,
        is_label_or_mask=False,
    )
    tumor_resampled = resample_array_to_grid(
        ras_tumor,
        source_affine=ras_label_affine,
        target_shape=target_shape,
        target_affine=target_affine,
        is_label_or_mask=True,
    ).astype(bool, copy=False)
    if spec.tumor_voxel_count > 0 and not bool(np.any(tumor_resampled)):
        raise RuntimeError(f"R2 positive tumor vanished: {spec.anonymous_case_id}")
    if spec.tumor_voxel_count == 0 and bool(np.any(tumor_resampled)):
        raise RuntimeError(f"R2 empty case became positive: {spec.anonymous_case_id}")

    return PreparedCase(
        anonymous_case_id=spec.anonymous_case_id,
        role=spec.role,
        image=normalize_r2_ct(image_resampled),
        tumor=np.asarray(tumor_resampled, dtype=bool),
    )


def _require_gpu(torch: Any) -> dict[str, Any]:
    if not bool(torch.cuda.is_available()):
        raise RuntimeError("R2 calibration requires CUDA; CPU fallback is forbidden")
    if int(torch.cuda.device_count()) != 2:
        raise RuntimeError("R2 calibration requires the registered two-GPU environment")

    devices: list[dict[str, Any]] = []
    for index in range(2):
        properties = torch.cuda.get_device_properties(index)
        if str(properties.name) != "Tesla T4":
            raise RuntimeError("R2 calibration requires Tesla T4 devices")
        devices.append(
            {
                "index": index,
                "name": str(properties.name),
                "total_memory_bytes": int(properties.total_memory),
                "total_memory_gib": float(properties.total_memory / (1024**3)),
            }
        )
    return {
        "cuda_available": True,
        "visible_device_count": 2,
        "training_device": "cuda:0",
        "cpu_fallback_allowed": False,
        "multi_gpu_memory_pooling_assumed": False,
        "devices": devices,
    }


def _build_model(monai: Any, device: Any) -> Any:
    return monai.networks.nets.SegResNet(
        spatial_dims=3,
        init_filters=8,
        in_channels=1,
        out_channels=2,
        dropout_prob=None,
        blocks_down=(1, 1, 1),
        blocks_up=(1, 1),
        upsample_mode="deconv",
    ).to(device)


def _sample_patch(
    cases: list[PreparedCase],
    *,
    step: int,
    rng: np.random.Generator,
) -> tuple[Any, Any, dict[str, Any]]:
    role = training_role_for_step(step)
    if role == "positive":
        case = cases[positive_case_index_for_step(step)]
        center = choose_random_positive_center(case.tumor, rng=rng)
    else:
        case = cases[3]
        center = choose_random_center(case.image.shape, rng=rng)

    image_patch = extract_centered_patch(
        case.image,
        center=center,
        patch_size=R2_PATCH_SIZE,
        pad_value=-1.0,
    ).astype(np.float32, copy=False)
    target_patch = extract_centered_patch(
        case.tumor,
        center=center,
        patch_size=R2_PATCH_SIZE,
        pad_value=False,
    ).astype(np.int64, copy=False)
    tumor_voxels = int(np.count_nonzero(target_patch))
    if role == "positive" and tumor_voxels <= 0:
        raise RuntimeError("R2 positive calibration patch contains no tumor")
    if role == "empty" and tumor_voxels != 0:
        raise RuntimeError("R2 empty calibration patch contains tumor")
    return (
        image_patch,
        target_patch,
        {
            "anonymous_case_id": case.anonymous_case_id,
            "sampling_role": role,
            "tumor_voxels": tumor_voxels,
        },
    )


def _optimizer_step(
    *,
    torch: Any,
    model: Any,
    optimizer: Any,
    scaler: Any,
    loss_function: Any,
    image_patch: Any,
    target_patch: Any,
    device: Any,
) -> dict[str, Any]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    image_tensor = torch.from_numpy(image_patch[None, None]).to(
        device=device,
        dtype=torch.float32,
    )
    target_tensor = torch.from_numpy(target_patch[None, None]).to(
        device=device,
        dtype=torch.long,
    )
    started = time.perf_counter()
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
        logits = model(image_tensor)
        loss = loss_function(logits, target_tensor)
    if not bool(torch.isfinite(logits).all().item()):
        raise RuntimeError("R2 calibration produced non-finite logits")
    if not bool(torch.isfinite(loss).item()):
        raise RuntimeError("R2 calibration produced non-finite loss")

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gradient_count = 0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        gradient_count += 1
        if not bool(torch.isfinite(parameter.grad).all().item()):
            raise RuntimeError("R2 calibration produced non-finite gradients")
    if gradient_count == 0:
        raise RuntimeError("R2 calibration produced no gradients")
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize(device)

    with torch.no_grad():
        probability = torch.softmax(logits.float(), dim=1)[:, 1]
    return {
        "loss": float(loss.detach().float().cpu().item()),
        "probability_min": float(probability.min().cpu().item()),
        "probability_max": float(probability.max().cpu().item()),
        "probability_mean": float(probability.mean().cpu().item()),
        "finite_logits": True,
        "finite_loss": True,
        "finite_gradients": True,
        "gradient_tensor_count": gradient_count,
        "step_time_seconds": float(time.perf_counter() - started),
    }


def _full_volume_probe(
    *,
    torch: Any,
    monai: Any,
    model: Any,
    case: PreparedCase,
    device: Any,
) -> float:
    model.eval()
    input_tensor = torch.from_numpy(case.image[None, None]).to(dtype=torch.float32)
    started = time.perf_counter()
    with torch.no_grad():
        logits = monai.inferers.sliding_window_inference(
            inputs=input_tensor,
            roi_size=R2_PATCH_SIZE,
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
            mode="gaussian",
            sw_device=device,
            device=torch.device("cpu"),
            progress=False,
        )
        probability = torch.softmax(logits.float(), dim=1)[:, 1]
    if not bool(torch.isfinite(probability).all().item()):
        raise RuntimeError("R2 calibration full-volume probe is non-finite")
    return float(time.perf_counter() - started)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-evidence", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    repo_root = args.repo_root.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    preflight_path = args.preflight_evidence.expanduser().resolve()
    if output_dir.exists():
        raise RuntimeError(f"refusing to overwrite R2 calibration dir: {output_dir}")
    output_dir.mkdir(parents=True)

    try:
        if sha256_file(repo_root / LOCK_RELATIVE_PATH) != EXPECTED_LOCK_SHA256:
            raise RuntimeError("R2 uv.lock differs from the frozen R0 CUDA lock")
        config_path = repo_root / CONFIG_RELATIVE_PATH
        config = _load_json(config_path)
        _verify_config(config)
        _verify_preflight(preflight_path)
        _verify_selected_files(data_root)

        torch = importlib.import_module("torch")
        monai = importlib.import_module("monai")
        gpu = _require_gpu(torch)
        cases = [_prepare_case(data_root, spec) for spec in R2_SELECTED_CASES]

        device = torch.device("cuda:0")
        torch.manual_seed(R2_SEED)
        torch.cuda.manual_seed_all(R2_SEED)
        rng = np.random.default_rng(R2_SEED)
        model = _build_model(monai, device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=0.001,
            weight_decay=0.00001,
        )
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        loss_function = monai.losses.DiceCELoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
            lambda_dice=1.0,
            lambda_ce=1.0,
        )
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        records: list[dict[str, Any]] = []
        for step in range(1, R2_CALIBRATION_MAX_STEPS + 1):
            image_patch, target_patch, sampling = _sample_patch(
                cases,
                step=step,
                rng=rng,
            )
            metrics = _optimizer_step(
                torch=torch,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                loss_function=loss_function,
                image_patch=image_patch,
                target_patch=target_patch,
                device=device,
            )
            records.append({"step": step, **sampling, **metrics})

        inference_seconds = _full_volume_probe(
            torch=torch,
            monai=monai,
            model=model,
            case=cases[1],
            device=device,
        )
        step_times = [float(record["step_time_seconds"]) for record in records]
        mean_step_seconds = float(statistics.fmean(step_times))
        summary = {
            "schema_version": "research_v2_r2_calibration.v1",
            "status": "PASS",
            "mode": "timing_memory_calibration",
            "main_training_started": False,
            "calibration_optimizer_steps": R2_CALIBRATION_MAX_STEPS,
            "seed": R2_SEED,
            "config_sha256": sha256_file(config_path),
            "preflight_sha256": sha256_file(preflight_path),
            "gpu": gpu,
            "mean_step_time_seconds": mean_step_seconds,
            "max_step_time_seconds": float(max(step_times)),
            "projected_2500_step_hours_excluding_evaluations": float(
                mean_step_seconds * R2_PRIMARY_MAX_STEPS / 3600.0
            ),
            "full_volume_probe_case_id": cases[1].anonymous_case_id,
            "full_volume_probe_seconds": inference_seconds,
            "full_volume_probe_finite": True,
            "peak_vram_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_vram_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "positive_patch_count": sum(
                record["sampling_role"] == "positive" for record in records
            ),
            "empty_patch_count": sum(record["sampling_role"] == "empty" for record in records),
            "threshold_changed": False,
            "gate_threshold_changed": False,
            "checkpoint_created": False,
            "training_warm_start_allowed": False,
            "validation_arrays_opened": 0,
            "internal_test_arrays_opened": 0,
            "external_arrays_opened": 0,
            "interpretation": (
                "Transient selected-train timing/memory calibration only; "
                "the primary R2 run must restart from seed 1729 and random initialization."
            ),
        }
        _write_json(output_dir / "r2_calibration_summary.json", summary)
        with (output_dir / "r2_calibration_steps.jsonl").open(
            "w",
            encoding="utf-8",
        ) as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except RuntimeError as exc:
        failure = {
            "schema_version": "research_v2_r2_calibration_failure.v1",
            "status": "FAILED",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "main_training_started": False,
            "checkpoint_created": False,
            "threshold_changed": False,
            "gate_threshold_changed": False,
            "validation_arrays_opened": 0,
            "internal_test_arrays_opened": 0,
            "external_arrays_opened": 0,
        }
        _write_json(output_dir / "r2_calibration_failure.json", failure)
        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)
        return 40


if __name__ == "__main__":
    sys.exit(main())
