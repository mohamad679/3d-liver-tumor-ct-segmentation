#!/usr/bin/env python3
"""Run the preregistered R2 2500-step real-train primary overfit diagnostic."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ["MPLBACKEND"] = "Agg"

import nibabel as nib
import numpy as np

from protoem_ct.artifacts import sha256_file
from protoem_ct.research_v2.r1_audit import (
    affine_spacing_mm,
    grid_for_spacing,
    reorient_array_to_ras,
    resample_array_to_grid,
    validate_matching_geometry,
)
from protoem_ct.research_v2.r2_metrics import compute_r2_native_case_metrics
from protoem_ct.research_v2.r2_overfit import (
    R2_FULL_VOLUME_EVAL_EVERY_STEPS,
    R2_PATCH_LOG_EVERY_STEPS,
    R2_PRIMARY_MAX_GPU_WALL_HOURS,
    R2_PRIMARY_MAX_STEPS,
    R2_SELECTED_CASES,
    R2_THRESHOLD,
    binary_dice,
    choose_random_center,
    choose_random_positive_center,
    positive_case_index_for_step,
    restore_probability_to_native,
    threshold_probability,
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
EXPECTED_CONFIG_SHA256 = "dfe50f52110bd77861f52fc4683712e0549a1f0f77c363f64f09da05940f3d1a"
EXPECTED_PREFLIGHT_SHA256 = "0d55bfe51604518e1e2c5c7245779217b689be3207eee83783a9709f1924911f"
EXPECTED_CALIBRATION_SUMMARY_SHA256 = (
    "f1866ea3d5028ad8e73d8e81df0dfeac80bd35646ea1a34472f3f5781aa2deb1"
)
EXPECTED_CALIBRATION_STEPS_SHA256 = (
    "4b3b738ae1bc82b0b6db0850f9b9f88a0076c655ecf41dff3951c1e84b6a8b09"
)
CONFIG_RELATIVE_PATH = Path("configs/research_v2/r2_overfit_real_v1.json")
LOCK_RELATIVE_PATH = Path("environments/research-v2-linux-cuda/uv.lock")


@dataclass(slots=True)
class PreparedCase:
    anonymous_case_id: str
    role: str
    image: Any
    tumor: Any
    resampled_affine: Any
    native_tumor: Any
    native_affine: Any
    native_shape: tuple[int, int, int]


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _verify_prerequisites(
    *,
    repo_root: Path,
    data_root: Path,
    preflight_path: Path,
    calibration_summary_path: Path,
    calibration_steps_path: Path,
) -> dict[str, Any]:
    if sha256_file(repo_root / LOCK_RELATIVE_PATH) != EXPECTED_LOCK_SHA256:
        raise RuntimeError("R2 uv.lock differs from the frozen R0 CUDA lock")
    config_path = repo_root / CONFIG_RELATIVE_PATH
    if sha256_file(config_path) != EXPECTED_CONFIG_SHA256:
        raise RuntimeError("R2 config differs from the preregistered primary contract")
    if sha256_file(preflight_path) != EXPECTED_PREFLIGHT_SHA256:
        raise RuntimeError("R2 preflight differs from the verified artifact")
    if sha256_file(calibration_summary_path) != EXPECTED_CALIBRATION_SUMMARY_SHA256:
        raise RuntimeError("R2 calibration summary differs from verified evidence")
    if sha256_file(calibration_steps_path) != EXPECTED_CALIBRATION_STEPS_SHA256:
        raise RuntimeError("R2 calibration steps differ from verified evidence")

    preflight = _load_json(preflight_path)
    calibration = _load_json(calibration_summary_path)
    if preflight.get("status") != "PASS":
        raise RuntimeError("R2 preflight evidence is not PASS")
    if calibration.get("status") != "PASS":
        raise RuntimeError("R2 calibration evidence is not PASS")
    if calibration.get("main_training_started") is not False:
        raise RuntimeError("R2 calibration evidence indicates main training started")
    if calibration.get("checkpoint_created") is not False:
        raise RuntimeError("R2 calibration must not create a checkpoint")
    if calibration.get("training_warm_start_allowed") is not False:
        raise RuntimeError("R2 calibration must not permit warm start")
    if int(calibration.get("calibration_optimizer_steps", -1)) != 20:
        raise RuntimeError("R2 calibration step count changed")
    if int(calibration.get("positive_patch_count", -1)) != 15:
        raise RuntimeError("R2 calibration positive patch count changed")
    if int(calibration.get("empty_patch_count", -1)) != 5:
        raise RuntimeError("R2 calibration empty patch count changed")
    if calibration.get("threshold_changed") is not False:
        raise RuntimeError("R2 threshold changed during calibration")
    if calibration.get("gate_threshold_changed") is not False:
        raise RuntimeError("R2 Gate threshold changed during calibration")
    for field in (
        "validation_arrays_opened",
        "internal_test_arrays_opened",
        "external_arrays_opened",
    ):
        if int(calibration.get(field, -1)) != 0:
            raise RuntimeError(f"R2 calibration leakage boundary failed: {field}")

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
    return _load_json(config_path)


def _require_gpu(torch: Any) -> dict[str, Any]:
    if not bool(torch.cuda.is_available()):
        raise RuntimeError("R2 primary requires CUDA; CPU fallback is forbidden")
    if int(torch.cuda.device_count()) != 2:
        raise RuntimeError("R2 primary requires the registered two-GPU environment")
    devices: list[dict[str, Any]] = []
    for index in range(2):
        properties = torch.cuda.get_device_properties(index)
        if str(properties.name) != "Tesla T4":
            raise RuntimeError("R2 primary requires Tesla T4 devices")
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
        resampled_affine=np.asarray(target_affine, dtype=np.float64),
        native_tumor=np.asarray(native_tumor, dtype=bool),
        native_affine=label_affine,
        native_shape=tuple(int(value) for value in label.shape),
    )


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
        raise RuntimeError("R2 positive primary patch contains no tumor")
    if role == "empty" and tumor_voxels != 0:
        raise RuntimeError("R2 empty primary patch contains tumor")
    return image_patch, target_patch, {
        "anonymous_case_id": case.anonymous_case_id,
        "sampling_role": role,
        "tumor_voxels": tumor_voxels,
    }


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
    image_tensor = torch.from_numpy(image_patch[None, None]).to(device=device, dtype=torch.float32)
    target_tensor = torch.from_numpy(target_patch[None, None]).to(device=device, dtype=torch.long)
    started = time.perf_counter()
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
        logits = model(image_tensor)
        loss = loss_function(logits, target_tensor)
    if not bool(torch.isfinite(logits).all().item()):
        raise RuntimeError("R2 primary produced non-finite logits")
    if not bool(torch.isfinite(loss).item()):
        raise RuntimeError("R2 primary produced non-finite loss")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gradient_count = 0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        gradient_count += 1
        if not bool(torch.isfinite(parameter.grad).all().item()):
            raise RuntimeError("R2 primary produced non-finite gradients")
    if gradient_count == 0:
        raise RuntimeError("R2 primary produced no gradients")
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize(device)
    with torch.no_grad():
        probability = torch.softmax(logits.float(), dim=1)[:, 1]
    probability_np = np.asarray(probability[0].detach().cpu().numpy(), dtype=np.float32)
    prediction_patch = threshold_probability(probability_np)
    patch_dice = binary_dice(
        np.asarray(target_patch, dtype=bool),
        prediction_patch,
    )
    return {
        "loss": float(loss.detach().float().cpu().item()),
        "patch_dice": patch_dice,
        "probability_min": float(probability.min().cpu().item()),
        "probability_max": float(probability.max().cpu().item()),
        "probability_mean": float(probability.mean().cpu().item()),
        "finite_logits": True,
        "finite_loss": True,
        "finite_gradients": True,
        "gradient_tensor_count": gradient_count,
        "step_time_seconds": float(time.perf_counter() - started),
    }


def _infer_resampled_probability(
    *,
    torch: Any,
    monai: Any,
    model: Any,
    case: PreparedCase,
    device: Any,
) -> np.ndarray:
    model.eval()
    input_tensor = torch.from_numpy(case.image[None, None]).to(dtype=torch.float32)

    def predictor(window: Any) -> Any:
        logits = model(window)
        if not bool(torch.isfinite(logits).all().item()):
            raise RuntimeError("R2 full-volume inference produced non-finite logits")
        return torch.softmax(logits.float(), dim=1)[:, 1:2]

    with torch.no_grad():
        probability_tensor = monai.inferers.sliding_window_inference(
            inputs=input_tensor,
            roi_size=R2_PATCH_SIZE,
            sw_batch_size=1,
            predictor=predictor,
            overlap=0.5,
            mode="gaussian",
            sw_device=device,
            device=torch.device("cpu"),
            progress=False,
        )
    probability = np.asarray(probability_tensor[0, 0].numpy(), dtype=np.float32)
    if not bool(np.isfinite(probability).all()):
        raise RuntimeError("R2 full-volume probability contains NaN/Inf")
    return probability


def _evaluate_case(
    *,
    torch: Any,
    monai: Any,
    model: Any,
    case: PreparedCase,
    device: Any,
    step: int,
    final: bool,
    reference_dir: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    probability_resampled = _infer_resampled_probability(
        torch=torch,
        monai=monai,
        model=model,
        case=case,
        device=device,
    )
    probability_native = restore_probability_to_native(
        probability_resampled,
        resampled_affine=case.resampled_affine,
        native_shape=case.native_shape,
        native_affine=case.native_affine,
    )
    prediction_native = threshold_probability(probability_native)
    validate_matching_geometry(
        ground_truth_shape=case.native_tumor.shape,
        prediction_shape=prediction_native.shape,
        ground_truth_affine=case.native_affine,
        prediction_affine=case.native_affine,
    )
    spacing = affine_spacing_mm(case.native_affine)
    voxel_volume_mm3 = float(spacing[0] * spacing[1] * spacing[2])
    record: dict[str, Any] = {
        "step": step,
        "anonymous_case_id": case.anonymous_case_id,
        "role": case.role,
        "native_full_volume_tumor_dice": binary_dice(case.native_tumor, prediction_native),
        "ground_truth_tumor_voxels": int(np.count_nonzero(case.native_tumor)),
        "predicted_tumor_voxels": int(np.count_nonzero(prediction_native)),
        "predicted_tumor_volume_mm3": float(
            np.count_nonzero(prediction_native) * voxel_volume_mm3
        ),
        "probability_min": float(probability_native.min()),
        "probability_max": float(probability_native.max()),
        "probability_mean": float(probability_native.mean()),
        "native_shape": list(case.native_shape),
        "prediction_ground_truth_affine_max_abs_delta": 0.0,
        "inference_and_native_restore_seconds": float(time.perf_counter() - started),
        "false_positive_lesion_count": None,
    }
    if final:
        metrics = compute_r2_native_case_metrics(
            ground_truth_mask=case.native_tumor.astype(np.uint8, copy=False),
            prediction_mask=prediction_native.astype(np.uint8, copy=False),
            ground_truth_affine=case.native_affine,
            prediction_affine=case.native_affine,
        )
        record.update(asdict(metrics))
        reference_dir.mkdir(parents=True, exist_ok=True)
        reference_path = reference_dir / f"{case.anonymous_case_id}.native_probability.npy"
        np.save(reference_path, probability_native, allow_pickle=False)
        record["private_reference_probability_filename"] = reference_path.name
        record["private_reference_probability_sha256"] = sha256_file(reference_path)
    del probability_resampled
    del probability_native
    return record


def _evaluate_all_cases(
    *,
    torch: Any,
    monai: Any,
    model: Any,
    cases: list[PreparedCase],
    device: Any,
    step: int,
    final: bool,
    reference_dir: Path,
) -> list[dict[str, Any]]:
    return [
        _evaluate_case(
            torch=torch,
            monai=monai,
            model=model,
            case=case,
            device=device,
            step=step,
            final=final,
            reference_dir=reference_dir,
        )
        for case in cases
    ]


def _save_checkpoint(
    *,
    torch: Any,
    path: Path,
    model: Any,
    optimizer: Any,
    scaler: Any,
    rng: np.random.Generator,
    step: int,
) -> str:
    checkpoint = {
        "schema_version": "research_v2_r2_checkpoint.v1",
        "step": step,
        "seed": R2_SEED,
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "amp_scaler_state_dict": scaler.state_dict(),
        "python_random_state": random.getstate(),
        "numpy_generator_state": rng.bit_generator.state,
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state_all": torch.cuda.get_rng_state_all(),
    }
    torch.save(checkpoint, path)
    return sha256_file(path)


def _write_private_overlays(
    *,
    data_root: Path,
    final_records: list[dict[str, Any]],
    reference_dir: Path,
    overlay_dir: Path,
) -> dict[str, Any]:
    plt = importlib.import_module("matplotlib.pyplot")
    overlay_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for spec, record in zip(R2_SELECTED_CASES, final_records, strict=True):
        root = data_root / spec.anonymous_case_id
        image_nifti = nib.load(str(root / "image.nii.gz"))
        label_nifti = nib.load(str(root / "label.nii.gz"))
        image = np.asarray(image_nifti.dataobj, dtype=np.float32)
        ground_truth = r2_binary_tumor_target(np.asarray(label_nifti.dataobj))
        probability = np.load(
            reference_dir / str(record["private_reference_probability_filename"]),
            allow_pickle=False,
        )
        prediction = threshold_probability(np.asarray(probability, dtype=np.float32))
        union = ground_truth | prediction
        if bool(np.any(union)):
            coordinates = np.argwhere(union)
            center = tuple(int(value) for value in np.median(coordinates, axis=0))
        else:
            center = tuple(int(value // 2) for value in image.shape)
        figure, axes = plt.subplots(1, 3, figsize=(15, 5))
        for axis, index in enumerate(center):
            image_slice = np.take(image, index, axis=axis)
            gt_slice = np.take(ground_truth, index, axis=axis)
            pred_slice = np.take(prediction, index, axis=axis)
            axes[axis].imshow(np.rot90(image_slice), cmap="gray", vmin=-200, vmax=300)
            if bool(np.any(gt_slice)):
                axes[axis].contour(np.rot90(gt_slice), levels=[0.5], linewidths=1.0)
            if bool(np.any(pred_slice)):
                axes[axis].contour(
                    np.rot90(pred_slice),
                    levels=[0.5],
                    linewidths=1.0,
                    linestyles="--",
                )
            axes[axis].set_title(f"native axis {axis} @ {index}")
            axes[axis].axis("off")
        figure.suptitle(spec.anonymous_case_id)
        figure.tight_layout()
        output_path = overlay_dir / f"{spec.anonymous_case_id}.png"
        figure.savefig(output_path, dpi=120, bbox_inches="tight")
        plt.close(figure)
        entries.append(
            {
                "anonymous_case_id": spec.anonymous_case_id,
                "filename": output_path.name,
                "sha256": sha256_file(output_path),
                "manual_nonclinical_geometry_review_required": True,
            }
        )
    return {
        "schema_version": "research_v2_r2_overlay_manifest.v1",
        "overlay_count": len(entries),
        "medical_images_in_manifest": False,
        "manual_nonclinical_geometry_review_required": True,
        "entries": entries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-evidence", type=Path, required=True)
    parser.add_argument("--calibration-summary", type=Path, required=True)
    parser.add_argument("--calibration-steps", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    repo_root = args.repo_root.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    preflight_path = args.preflight_evidence.expanduser().resolve()
    calibration_summary_path = args.calibration_summary.expanduser().resolve()
    calibration_steps_path = args.calibration_steps.expanduser().resolve()
    if output_dir.exists():
        raise RuntimeError(f"refusing to overwrite R2 primary dir: {output_dir}")
    output_dir.mkdir(parents=True)
    steps_path = output_dir / "r2_primary_steps.jsonl"
    evaluations_path = output_dir / "r2_primary_full_volume_evaluations.jsonl"
    reference_dir = output_dir / "private_reference_probabilities"
    overlay_dir = output_dir / "private_overlays"
    checkpoint_path = output_dir / "r2_primary_checkpoint.pt"
    started_wall = time.perf_counter()
    current_step = 0

    try:
        config = _verify_prerequisites(
            repo_root=repo_root,
            data_root=data_root,
            preflight_path=preflight_path,
            calibration_summary_path=calibration_summary_path,
            calibration_steps_path=calibration_steps_path,
        )
        torch = importlib.import_module("torch")
        monai = importlib.import_module("monai")
        gpu = _require_gpu(torch)
        device = torch.device("cuda:0")
        random.seed(R2_SEED)
        np.random.seed(R2_SEED)
        torch.manual_seed(R2_SEED)
        torch.cuda.manual_seed_all(R2_SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        rng = np.random.default_rng(R2_SEED)
        cases = [_prepare_case(data_root, spec) for spec in R2_SELECTED_CASES]
        model = _build_model(monai, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.00001)
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

        all_step_records: list[dict[str, Any]] = []
        final_records: list[dict[str, Any]] = []
        with steps_path.open("w", encoding="utf-8") as steps_handle, evaluations_path.open(
            "w", encoding="utf-8"
        ) as evaluation_handle:
            for step in range(1, R2_PRIMARY_MAX_STEPS + 1):
                elapsed_hours = (time.perf_counter() - started_wall) / 3600.0
                if elapsed_hours >= R2_PRIMARY_MAX_GPU_WALL_HOURS:
                    raise RuntimeError("R2 primary GPU wall-hour budget exhausted")
                current_step = step
                image_patch, target_patch, sampling = _sample_patch(cases, step=step, rng=rng)
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
                record = {
                    "step": step,
                    **sampling,
                    **metrics,
                    "elapsed_time_seconds": float(time.perf_counter() - started_wall),
                    "peak_vram_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                }
                steps_handle.write(json.dumps(record, sort_keys=True) + "\n")
                steps_handle.flush()
                all_step_records.append(record)
                if step % R2_PATCH_LOG_EVERY_STEPS == 0:
                    print(json.dumps(record, sort_keys=True))

                if step % R2_FULL_VOLUME_EVAL_EVERY_STEPS == 0:
                    final = step == R2_PRIMARY_MAX_STEPS
                    evaluation_records = _evaluate_all_cases(
                        torch=torch,
                        monai=monai,
                        model=model,
                        cases=cases,
                        device=device,
                        step=step,
                        final=final,
                        reference_dir=reference_dir,
                    )
                    for evaluation in evaluation_records:
                        evaluation_handle.write(json.dumps(evaluation, sort_keys=True) + "\n")
                    evaluation_handle.flush()
                    if final:
                        final_records = evaluation_records

        if len(all_step_records) != R2_PRIMARY_MAX_STEPS:
            raise RuntimeError("R2 primary did not complete exactly 2500 optimizer steps")
        if len(final_records) != len(R2_SELECTED_CASES):
            raise RuntimeError("R2 final full-volume evaluation is incomplete")
        checkpoint_sha256 = _save_checkpoint(
            torch=torch,
            path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            rng=rng,
            step=R2_PRIMARY_MAX_STEPS,
        )
        overlay_manifest = _write_private_overlays(
            data_root=data_root,
            final_records=final_records,
            reference_dir=reference_dir,
            overlay_dir=overlay_dir,
        )
        _write_json(output_dir / "r2_overlay_manifest.json", overlay_manifest)

        per_case_patch: dict[str, dict[str, Any]] = {}
        for spec in R2_SELECTED_CASES:
            records = [
                record
                for record in all_step_records
                if record["anonymous_case_id"] == spec.anonymous_case_id
            ]
            per_case_patch[spec.anonymous_case_id] = {
                "patch_count": len(records),
                "mean_patch_dice": float(statistics.fmean(record["patch_dice"] for record in records)),
                "final_patch_dice": float(records[-1]["patch_dice"]),
                "mean_loss": float(statistics.fmean(record["loss"] for record in records)),
            }

        positives = [record for record in final_records if record["role"].startswith("positive_")]
        empty_records = [
            record for record in final_records if record["role"] == "empty_lexicographic_first"
        ]
        if len(positives) != 3 or len(empty_records) != 1:
            raise RuntimeError("R2 final case-role accounting is invalid")
        positive_dice_gate = all(
            float(record["native_full_volume_tumor_dice"]) >= 0.95 for record in positives
        )
        empty_prediction_gate = int(empty_records[0]["predicted_tumor_voxels"]) == 0
        elapsed_seconds = float(time.perf_counter() - started_wall)
        summary = {
            "schema_version": "research_v2_r2_primary.v1",
            "status": "READY_FOR_RELOAD_AND_OVERLAY_REVIEW"
            if positive_dice_gate and empty_prediction_gate
            else "PRIMARY_GATE_METRICS_FAILED",
            "interpretation": (
                "Diagnostic overfit results on selected training patients only; "
                "not an estimate of unseen-patient performance."
            ),
            "main_training_started": True,
            "main_training_completed": True,
            "optimizer_steps_completed": R2_PRIMARY_MAX_STEPS,
            "seed": R2_SEED,
            "fresh_random_initialization": True,
            "warm_start_used": False,
            "extension_used": False,
            "config_sha256": EXPECTED_CONFIG_SHA256,
            "preflight_sha256": EXPECTED_PREFLIGHT_SHA256,
            "calibration_summary_sha256": EXPECTED_CALIBRATION_SUMMARY_SHA256,
            "calibration_steps_sha256": EXPECTED_CALIBRATION_STEPS_SHA256,
            "gpu": gpu,
            "fixed_threshold": R2_THRESHOLD,
            "threshold_changed": False,
            "gate_threshold_changed": False,
            "elapsed_seconds": elapsed_seconds,
            "elapsed_gpu_wall_hours": elapsed_seconds / 3600.0,
            "primary_max_gpu_wall_hours": R2_PRIMARY_MAX_GPU_WALL_HOURS,
            "peak_vram_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_vram_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "positive_patch_count": sum(
                record["sampling_role"] == "positive" for record in all_step_records
            ),
            "empty_patch_count": sum(
                record["sampling_role"] == "empty" for record in all_step_records
            ),
            "all_logged_losses_finite": all(
                bool(record["finite_loss"]) for record in all_step_records
            ),
            "all_logged_gradients_finite": all(
                bool(record["finite_gradients"]) for record in all_step_records
            ),
            "all_logged_logits_finite": all(
                bool(record["finite_logits"]) for record in all_step_records
            ),
            "per_case_patch_metrics": per_case_patch,
            "final_native_full_volume_metrics": final_records,
            "positive_case_native_dice_min_each_required": 0.95,
            "positive_case_native_dice_gate_pass": positive_dice_gate,
            "empty_case_prediction_must_be_empty": True,
            "empty_case_prediction_gate_pass": empty_prediction_gate,
            "checkpoint_filename": checkpoint_path.name,
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_reload_status": "PENDING_FRESH_PROCESS",
            "private_reference_probability_count": len(list(reference_dir.glob("*.npy"))),
            "overlay_manifest_filename": "r2_overlay_manifest.json",
            "overlay_manual_review_status": "PENDING",
            "validation_arrays_opened": 0,
            "internal_test_arrays_opened": 0,
            "external_arrays_opened": 0,
            "next_action": "fresh_process_checkpoint_reload_and_manual_overlay_geometry_review",
        }
        _write_json(output_dir / "r2_primary_summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if positive_dice_gate and empty_prediction_gate else 40
    except (RuntimeError, ValueError) as exc:
        failure = {
            "schema_version": "research_v2_r2_primary_failure.v1",
            "status": "FAILED",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "step_reached": current_step,
            "main_training_started": current_step > 0,
            "threshold_changed": False,
            "gate_threshold_changed": False,
            "validation_arrays_opened": 0,
            "internal_test_arrays_opened": 0,
            "external_arrays_opened": 0,
            "elapsed_seconds": float(time.perf_counter() - started_wall),
        }
        _write_json(output_dir / "r2_primary_failure.json", failure)
        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)
        return 40


if __name__ == "__main__":
    sys.exit(main())
