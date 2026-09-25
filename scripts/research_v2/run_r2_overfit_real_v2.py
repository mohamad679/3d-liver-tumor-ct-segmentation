#!/usr/bin/env python3
"""Run the locked R2 timing calibration or primary real-train overfit experiment."""

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
    compute_strict_case_metrics,
    grid_for_spacing,
    reorient_array_to_ras,
    resample_array_to_grid,
)
from protoem_ct.research_v2.r2_overfit import (
    R2_CALIBRATION_MAX_STEPS,
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
    choose_positive_center,
    extract_centered_patch,
    normalize_r2_ct,
    r2_binary_tumor_target,
)

EXPECTED_LOCK_SHA256 = (
    "c9ee200234eebf8c67fd97ff2c96ca0765f4067a8a69a7ef439deafdda3530a8"
)
EXPECTED_PREFLIGHT_SHA256 = (
    "0d55bfe51604518e1e2c5c7245779217b689be3207eee83783a9709f1924911f"
)
CONFIG_RELATIVE_PATH = Path("configs/research_v2/r2_overfit_real_v1.json")
LOCK_RELATIVE_PATH = Path("environments/research-v2-linux-cuda/uv.lock")


@dataclass(slots=True)
class PreparedCase:
    anonymous_case_id: str
    role: str
    normalized_image: np.ndarray
    resampled_tumor: np.ndarray
    resampled_affine: np.ndarray
    native_tumor: np.ndarray
    native_affine: np.ndarray
    native_shape: tuple[int, int, int]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def _verify_locked_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != "research_v2_r2_overfit_real.v1":
        raise RuntimeError("unexpected R2 experiment config schema")
    training = config.get("training")
    inference = config.get("inference")
    budget = config.get("budget")
    if not isinstance(training, dict):
        raise RuntimeError("R2 config training section is missing")
    if not isinstance(inference, dict):
        raise RuntimeError("R2 config inference section is missing")
    if not isinstance(budget, dict):
        raise RuntimeError("R2 config budget section is missing")

    locked_training = {
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
    for field, expected in locked_training.items():
        if training.get(field) != expected:
            raise RuntimeError(f"R2 locked training field changed: {field}")

    if inference.get("sliding_window_roi_size") != list(R2_PATCH_SIZE):
        raise RuntimeError("R2 sliding-window ROI changed")
    if inference.get("sliding_window_overlap") != 0.5:
        raise RuntimeError("R2 sliding-window overlap changed")
    if inference.get("threshold") != R2_THRESHOLD:
        raise RuntimeError("R2 inference threshold changed")

    calibration = budget.get("timing_memory_calibration")
    primary = budget.get("primary")
    if not isinstance(calibration, dict) or not isinstance(primary, dict):
        raise RuntimeError("R2 config budget entries are missing")
    if calibration.get("max_steps") != R2_CALIBRATION_MAX_STEPS:
        raise RuntimeError("R2 calibration step budget changed")
    if primary.get("max_training_steps") != R2_PRIMARY_MAX_STEPS:
        raise RuntimeError("R2 primary step budget changed")
    if primary.get("max_gpu_wall_hours") != R2_PRIMARY_MAX_GPU_WALL_HOURS:
        raise RuntimeError("R2 primary wall-time budget changed")


def _verify_preflight(path: Path) -> dict[str, Any]:
    observed_hash = sha256_file(path)
    if observed_hash != EXPECTED_PREFLIGHT_SHA256:
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
    return evidence


def _gpu_evidence(torch: Any) -> dict[str, Any]:
    if not bool(torch.cuda.is_available()):
        raise RuntimeError("R2 requires CUDA; CPU fallback is forbidden")
    if int(torch.cuda.device_count()) != 2:
        raise RuntimeError("R2 requires the registered two-device Kaggle environment")

    devices: list[dict[str, Any]] = []
    for index in range(2):
        properties = torch.cuda.get_device_properties(index)
        if str(properties.name) != "Tesla T4":
            raise RuntimeError("R2 requires the registered Tesla T4 environment")
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


def _verify_selected_files(data_root: Path) -> None:
    expected_paths: set[Path] = set()
    for spec in R2_SELECTED_CASES:
        case_root = data_root / spec.anonymous_case_id
        image_path = case_root / "image.nii.gz"
        label_path = case_root / "label.nii.gz"
        expected_paths.update({image_path.resolve(), label_path.resolve()})
        if sha256_file(image_path) != spec.image_sha256:
            raise RuntimeError(f"image SHA-256 mismatch: {spec.anonymous_case_id}")
        if sha256_file(label_path) != spec.label_sha256:
            raise RuntimeError(f"label SHA-256 mismatch: {spec.anonymous_case_id}")

    observed_paths = {
        path.resolve()
        for pattern in ("*.nii", "*.nii.gz")
        for path in data_root.rglob(pattern)
        if path.is_file()
    }
    if observed_paths != expected_paths:
        raise RuntimeError("R2 data root must contain exactly eight selected train files")


def _prepare_case(data_root: Path, spec: Any) -> PreparedCase:
    case_root = data_root / spec.anonymous_case_id
    image_nifti = nib.load(str(case_root / "image.nii.gz"))
    label_nifti = nib.load(str(case_root / "label.nii.gz"))
    image = np.asarray(image_nifti.dataobj)
    label = np.asarray(label_nifti.dataobj)
    if image.ndim != 3 or label.ndim != 3 or image.shape != label.shape:
        raise RuntimeError(f"source geometry mismatch: {spec.anonymous_case_id}")
    if not bool(np.isfinite(image).all()):
        raise RuntimeError(f"source CT contains NaN/Inf: {spec.anonymous_case_id}")

    image_affine = np.asarray(image_nifti.affine, dtype=np.float64)
    label_affine = np.asarray(label_nifti.affine, dtype=np.float64)
    affine_delta = float(np.max(np.abs(image_affine - label_affine)))
    if affine_delta > 1e-4:
        raise RuntimeError(f"source affine mismatch: {spec.anonymous_case_id}")

    native_tumor = r2_binary_tumor_target(label)
    native_voxels = int(np.count_nonzero(native_tumor))
    if native_voxels != int(spec.tumor_voxel_count):
        raise RuntimeError(f"native tumor voxel mismatch: {spec.anonymous_case_id}")

    ras_image, ras_image_affine = reorient_array_to_ras(image, image_affine)
    ras_tumor, ras_label_affine = reorient_array_to_ras(
        native_tumor,
        label_affine,
    )
    target_shape, target_affine = grid_for_spacing(
        shape=ras_image.shape,
        affine=ras_image_affine,
        target_spacing_mm=R2_TARGET_SPACING_MM,
    )
    resampled_image = resample_array_to_grid(
        ras_image,
        source_affine=ras_image_affine,
        target_shape=target_shape,
        target_affine=target_affine,
        is_label_or_mask=False,
    )
    resampled_tumor = resample_array_to_grid(
        ras_tumor,
        source_affine=ras_label_affine,
        target_shape=target_shape,
        target_affine=target_affine,
        is_label_or_mask=True,
    ).astype(bool, copy=False)
    if spec.tumor_voxel_count > 0 and not bool(np.any(resampled_tumor)):
        raise RuntimeError(f"positive tumor vanished: {spec.anonymous_case_id}")
    if spec.tumor_voxel_count == 0 and bool(np.any(resampled_tumor)):
        raise RuntimeError(f"empty case became positive: {spec.anonymous_case_id}")

    return PreparedCase(
        anonymous_case_id=spec.anonymous_case_id,
        role=spec.role,
        normalized_image=normalize_r2_ct(resampled_image),
        resampled_tumor=np.asarray(resampled_tumor, dtype=bool),
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


def _build_loss(monai: Any) -> Any:
    return monai.losses.DiceCELoss(
        include_background=False,
        to_onehot_y=True,
        softmax=True,
        lambda_dice=1.0,
        lambda_ce=1.0,
    )


def _sample_patch(
    cases: list[PreparedCase],
    *,
    step: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    sampling_role = training_role_for_step(step)
    if sampling_role == "positive":
        case = cases[positive_case_index_for_step(step)]
        center = choose_random_positive_center(case.resampled_tumor, rng=rng)
    else:
        case = cases[3]
        center = choose_random_center(case.normalized_image.shape, rng=rng)

    image_patch = extract_centered_patch(
        case.normalized_image,
        center=center,
        patch_size=R2_PATCH_SIZE,
        pad_value=-1.0,
    ).astype(np.float32, copy=False)
    target_patch = extract_centered_patch(
        case.resampled_tumor,
        center=center,
        patch_size=R2_PATCH_SIZE,
        pad_value=False,
    ).astype(np.int64, copy=False)
    tumor_voxels = int(np.count_nonzero(target_patch))
    if sampling_role == "positive" and tumor_voxels <= 0:
        raise RuntimeError("R2 positive patch contains no tumor")
    if sampling_role == "empty" and tumor_voxels != 0:
        raise RuntimeError("R2 empty patch contains tumor")

    metadata = {
        "anonymous_case_id": case.anonymous_case_id,
        "sampling_role": sampling_role,
        "center": [int(value) for value in center],
        "tumor_voxels": tumor_voxels,
    }
    return image_patch, target_patch, metadata


def _optimizer_step(
    *,
    torch: Any,
    model: Any,
    optimizer: Any,
    scaler: Any,
    loss_function: Any,
    image_patch: np.ndarray,
    target_patch: np.ndarray,
    device: Any,
) -> dict[str, Any]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    input_tensor = torch.from_numpy(image_patch[None, None]).to(
        device=device,
        dtype=torch.float32,
    )
    target_tensor = torch.from_numpy(target_patch[None, None]).to(
        device=device,
        dtype=torch.long,
    )
    started = time.perf_counter()
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
        logits = model(input_tensor)
        loss = loss_function(logits, target_tensor)
    if not bool(torch.isfinite(logits).all().item()):
        raise RuntimeError("R2 optimizer step produced non-finite logits")
    if not bool(torch.isfinite(loss).item()):
        raise RuntimeError("R2 optimizer step produced non-finite loss")

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gradient_tensors = 0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        gradient_tensors += 1
        if not bool(torch.isfinite(parameter.grad).all().item()):
            raise RuntimeError("R2 optimizer step produced non-finite gradients")
    if gradient_tensors == 0:
        raise RuntimeError("R2 optimizer step produced no gradients")

    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize(device)
    step_seconds = float(time.perf_counter() - started)

    with torch.no_grad():
        probability = torch.softmax(logits.float(), dim=1)[:, 1]
    probability_np = probability.detach().cpu().numpy()[0]
    prediction = probability_np >= R2_THRESHOLD
    return {
        "loss": float(loss.detach().float().cpu().item()),
        "patch_dice": binary_dice(
            np.asarray(target_patch, dtype=bool),
            prediction,
        ),
        "probability_min": float(probability_np.min()),
        "probability_max": float(probability_np.max()),
        "probability_mean": float(probability_np.mean()),
        "finite_logits": True,
        "finite_loss": True,
        "finite_gradients": True,
        "gradient_tensor_count": gradient_tensors,
        "step_time_seconds": step_seconds,
    }


def _sliding_window_probability(
    *,
    torch: Any,
    monai: Any,
    model: Any,
    case: PreparedCase,
    device: Any,
) -> tuple[np.ndarray, float]:
    model.eval()
    input_cpu = torch.from_numpy(case.normalized_image[None, None]).to(
        dtype=torch.float32
    )
    started = time.perf_counter()
    with torch.no_grad():
        logits_cpu = monai.inferers.sliding_window_inference(
            inputs=input_cpu,
            roi_size=R2_PATCH_SIZE,
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
            mode="gaussian",
            sw_device=device,
            device=torch.device("cpu"),
            progress=False,
        )
        probability = torch.softmax(logits_cpu.float(), dim=1)[0, 1].numpy()
    inference_seconds = float(time.perf_counter() - started)
    if not bool(np.isfinite(probability).all()):
        raise RuntimeError(
            f"non-finite full-volume probability: {case.anonymous_case_id}"
        )
    return np.asarray(probability, dtype=np.float32), inference_seconds


def _patch_metrics(
    *,
    torch: Any,
    model: Any,
    case: PreparedCase,
    device: Any,
    index: int,
) -> dict[str, Any]:
    if bool(np.any(case.resampled_tumor)):
        center = choose_positive_center(
            case.resampled_tumor,
            seed=R2_SEED + index,
        )
    else:
        center = tuple(int(value // 2) for value in case.resampled_tumor.shape)

    image_patch = extract_centered_patch(
        case.normalized_image,
        center=center,
        patch_size=R2_PATCH_SIZE,
        pad_value=-1.0,
    ).astype(np.float32, copy=False)
    target_patch = extract_centered_patch(
        case.resampled_tumor,
        center=center,
        patch_size=R2_PATCH_SIZE,
        pad_value=False,
    ).astype(bool, copy=False)

    model.eval()
    input_tensor = torch.from_numpy(image_patch[None, None]).to(
        device=device,
        dtype=torch.float32,
    )
    with torch.no_grad():
        logits = model(input_tensor)
        probability = torch.softmax(logits.float(), dim=1)[0, 1].cpu().numpy()
    prediction = threshold_probability(np.asarray(probability, dtype=np.float32))
    return {
        "center": [int(value) for value in center],
        "ground_truth_tumor_voxels": int(np.count_nonzero(target_patch)),
        "predicted_tumor_voxels": int(np.count_nonzero(prediction)),
        "patch_dice": binary_dice(
            np.asarray(target_patch, dtype=bool),
            prediction,
        ),
        "probability_min": float(probability.min()),
        "probability_max": float(probability.max()),
        "probability_mean": float(probability.mean()),
    }


def _evaluate_case(
    *,
    torch: Any,
    monai: Any,
    model: Any,
    case: PreparedCase,
    device: Any,
    index: int,
    reference_dir: Path | None,
) -> dict[str, Any]:
    patch_record = _patch_metrics(
        torch=torch,
        model=model,
        case=case,
        device=device,
        index=index,
    )
    probability_resampled, inference_seconds = _sliding_window_probability(
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
    metrics = compute_strict_case_metrics(
        case_identifier=case.anonymous_case_id,
        ground_truth_mask=case.native_tumor,
        prediction_mask=prediction_native,
        ground_truth_affine=case.native_affine,
        prediction_affine=case.native_affine,
    )

    if reference_dir is not None:
        case_dir = reference_dir / case.anonymous_case_id
        case_dir.mkdir(parents=True, exist_ok=False)
        np.save(
            case_dir / "native_probability.npy",
            probability_native,
            allow_pickle=False,
        )
        np.save(
            case_dir / "native_binary.npy",
            prediction_native.astype(np.uint8),
            allow_pickle=False,
        )

    return {
        "anonymous_case_id": case.anonymous_case_id,
        "role": case.role,
        "patch": patch_record,
        "native_full_volume": {
            "tumor_dice": float(metrics.tumor_dice),
            "tumor_iou": float(metrics.tumor_iou),
            "ground_truth_tumor_voxels": int(
                metrics.ground_truth_foreground_voxels
            ),
            "predicted_tumor_voxels": int(metrics.predicted_foreground_voxels),
            "ground_truth_lesion_count": int(metrics.ground_truth_lesion_count),
            "predicted_lesion_count": int(metrics.predicted_lesion_count),
            "false_positive_lesion_count": int(
                metrics.false_positive_lesion_count
            ),
            "false_negative_lesion_count": int(
                metrics.false_negative_lesion_count
            ),
            "probability_min": float(probability_native.min()),
            "probability_max": float(probability_native.max()),
            "probability_mean": float(probability_native.mean()),
            "inference_seconds": inference_seconds,
            "native_shape": [int(value) for value in case.native_shape],
        },
    }


def _evaluate_all(
    *,
    torch: Any,
    monai: Any,
    model: Any,
    cases: list[PreparedCase],
    device: Any,
    step: int,
    reference_dir: Path | None,
) -> dict[str, Any]:
    started = time.perf_counter()
    records = [
        _evaluate_case(
            torch=torch,
            monai=monai,
            model=model,
            case=case,
            device=device,
            index=index,
            reference_dir=reference_dir,
        )
        for index, case in enumerate(cases)
    ]
    return {
        "step": step,
        "elapsed_seconds": float(time.perf_counter() - started),
        "cases": records,
    }


def _checkpoint_payload(
    *,
    torch: Any,
    model: Any,
    optimizer: Any,
    scaler: Any,
    rng: np.random.Generator,
    step: int,
    config_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": "research_v2_r2_checkpoint.v1",
        "step": step,
        "seed": R2_SEED,
        "config_sha256": config_sha256,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "amp_scaler_state_dict": scaler.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state_all": torch.cuda.get_rng_state_all(),
        "numpy_rng_state": rng.bit_generator.state,
    }


def _build_training_objects(
    *,
    torch: Any,
    monai: Any,
    device: Any,
) -> tuple[Any, Any, Any, Any]:
    model = _build_model(monai, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.00001,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    loss_function = _build_loss(monai)
    return model, optimizer, scaler, loss_function


def _run_calibration(
    *,
    torch: Any,
    monai: Any,
    cases: list[PreparedCase],
    output_dir: Path,
    config_sha256: str,
    gpu_record: dict[str, Any],
    preflight_sha256: str,
) -> int:
    device = torch.device("cuda:0")
    torch.manual_seed(R2_SEED)
    torch.cuda.manual_seed_all(R2_SEED)
    rng = np.random.default_rng(R2_SEED)
    model, optimizer, scaler, loss_function = _build_training_objects(
        torch=torch,
        monai=monai,
        device=device,
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

    probe_probability, probe_seconds = _sliding_window_probability(
        torch=torch,
        monai=monai,
        model=model,
        case=cases[1],
        device=device,
    )
    if not bool(np.isfinite(probe_probability).all()):
        raise RuntimeError("R2 calibration inference probe is non-finite")

    step_times = [float(record["step_time_seconds"]) for record in records]
    mean_step_seconds = float(statistics.fmean(step_times))
    summary = {
        "schema_version": "research_v2_r2_calibration.v1",
        "status": "PASS",
        "mode": "timing_memory_calibration",
        "main_training_started": False,
        "calibration_steps": R2_CALIBRATION_MAX_STEPS,
        "seed": R2_SEED,
        "config_sha256": config_sha256,
        "preflight_sha256": preflight_sha256,
        "gpu": gpu_record,
        "mean_step_time_seconds": mean_step_seconds,
        "max_step_time_seconds": float(max(step_times)),
        "projected_2500_optimizer_step_hours_excluding_evaluations": float(
            mean_step_seconds * R2_PRIMARY_MAX_STEPS / 3600.0
        ),
        "full_volume_inference_probe_case_id": cases[1].anonymous_case_id,
        "full_volume_inference_probe_seconds": probe_seconds,
        "full_volume_inference_probe_finite": True,
        "peak_vram_allocated_bytes": int(
            torch.cuda.max_memory_allocated(device)
        ),
        "peak_vram_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "positive_patch_count": sum(
            record["sampling_role"] == "positive" for record in records
        ),
        "empty_patch_count": sum(
            record["sampling_role"] == "empty" for record in records
        ),
        "threshold_changed": False,
        "gate_threshold_changed": False,
        "training_warm_start_allowed": False,
        "interpretation": (
            "Transient selected-train timing/memory calibration only. "
            "The primary R2 run must restart from the locked seed and random initialization."
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


def _verify_calibration(
    path: Path,
    *,
    config_sha256: str,
    preflight_sha256: str,
) -> dict[str, Any]:
    calibration = _load_json(path)
    if calibration.get("status") != "PASS":
        raise RuntimeError("R2 primary requires PASS calibration evidence")
    if calibration.get("mode") != "timing_memory_calibration":
        raise RuntimeError("R2 calibration mode is invalid")
    if calibration.get("main_training_started") is not False:
        raise RuntimeError("R2 calibration claims main training started")
    if calibration.get("training_warm_start_allowed") is not False:
        raise RuntimeError("R2 calibration violates fresh-start policy")
    if calibration.get("config_sha256") != config_sha256:
        raise RuntimeError("R2 calibration config hash differs from primary")
    if calibration.get("preflight_sha256") != preflight_sha256:
        raise RuntimeError("R2 calibration preflight hash differs from primary")
    return calibration


def _save_checkpoint(
    *,
    torch: Any,
    path: Path,
    model: Any,
    optimizer: Any,
    scaler: Any,
    rng: np.random.Generator,
    step: int,
    config_sha256: str,
) -> None:
    payload = _checkpoint_payload(
        torch=torch,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        rng=rng,
        step=step,
        config_sha256=config_sha256,
    )
    torch.save(payload, path)


def _run_primary(
    *,
    torch: Any,
    monai: Any,
    cases: list[PreparedCase],
    output_dir: Path,
    config_sha256: str,
    gpu_record: dict[str, Any],
    preflight_sha256: str,
    calibration_path: Path,
) -> int:
    _verify_calibration(
        calibration_path,
        config_sha256=config_sha256,
        preflight_sha256=preflight_sha256,
    )

    device = torch.device("cuda:0")
    torch.manual_seed(R2_SEED)
    torch.cuda.manual_seed_all(R2_SEED)
    rng = np.random.default_rng(R2_SEED)
    model, optimizer, scaler, loss_function = _build_training_objects(
        torch=torch,
        monai=monai,
        device=device,
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    checkpoint_dir = output_dir / "private_checkpoints"
    checkpoint_dir.mkdir()
    checkpoint_path = checkpoint_dir / "r2_primary_latest.pt"
    evaluations: list[dict[str, Any]] = []
    positive_patches = 0
    final_step = 0
    termination_reason = "max_training_steps"
    training_started = time.perf_counter()

    log_path = output_dir / "r2_training_log.jsonl"
    with log_path.open("w", encoding="utf-8") as log_handle:
        for step in range(1, R2_PRIMARY_MAX_STEPS + 1):
            elapsed_before = time.perf_counter() - training_started
            if elapsed_before >= R2_PRIMARY_MAX_GPU_WALL_HOURS * 3600.0:
                termination_reason = "max_gpu_wall_hours"
                break

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
            final_step = step
            if sampling["sampling_role"] == "positive":
                positive_patches += 1

            record = {
                "step": step,
                **sampling,
                **metrics,
                "positive_patch_ratio_cumulative": float(
                    positive_patches / step
                ),
                "elapsed_time_seconds": float(
                    time.perf_counter() - training_started
                ),
                "peak_vram_allocated_bytes": int(
                    torch.cuda.max_memory_allocated(device)
                ),
                "peak_vram_reserved_bytes": int(
                    torch.cuda.max_memory_reserved(device)
                ),
            }
            log_handle.write(json.dumps(record, sort_keys=True) + "\n")
            log_handle.flush()

            if step % R2_PATCH_LOG_EVERY_STEPS == 0:
                print(
                    f"R2 step={step} loss={record['loss']:.6f} "
                    f"patch_dice={record['patch_dice']:.6f} "
                    f"positive_ratio="
                    f"{record['positive_patch_ratio_cumulative']:.4f}"
                )

            scheduled_eval = step % R2_FULL_VOLUME_EVAL_EVERY_STEPS == 0
            if scheduled_eval and step < R2_PRIMARY_MAX_STEPS:
                evaluations.append(
                    _evaluate_all(
                        torch=torch,
                        monai=monai,
                        model=model,
                        cases=cases,
                        device=device,
                        step=step,
                        reference_dir=None,
                    )
                )
                _save_checkpoint(
                    torch=torch,
                    path=checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    rng=rng,
                    step=step,
                    config_sha256=config_sha256,
                )

    if final_step <= 0:
        raise RuntimeError("R2 primary ended before any optimizer step completed")

    reference_dir = output_dir / "private_reload_reference"
    final_evaluation = _evaluate_all(
        torch=torch,
        monai=monai,
        model=model,
        cases=cases,
        device=device,
        step=final_step,
        reference_dir=reference_dir,
    )
    evaluations.append(final_evaluation)
    _save_checkpoint(
        torch=torch,
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        rng=rng,
        step=final_step,
        config_sha256=config_sha256,
    )

    _write_json(
        output_dir / "r2_evaluations.json",
        {
            "schema_version": "research_v2_r2_evaluations.v1",
            "evaluations": evaluations,
        },
    )

    final_cases = final_evaluation["cases"]
    positive_dice = [
        float(record["native_full_volume"]["tumor_dice"])
        for record in final_cases
        if str(record["role"]).startswith("positive_")
    ]
    empty_records = [
        record
        for record in final_cases
        if record["role"] == "empty_lexicographic_first"
    ]
    if len(positive_dice) != 3 or len(empty_records) != 1:
        raise RuntimeError("R2 final evaluation case-role cardinality is invalid")
    empty_prediction_voxels = int(
        empty_records[0]["native_full_volume"]["predicted_tumor_voxels"]
    )

    wall_seconds = float(time.perf_counter() - training_started)
    summary = {
        "schema_version": "research_v2_r2_primary_run.v1",
        "status": "COMPLETED_PENDING_RELOAD_AND_OVERLAY_REVIEW",
        "mode": "primary",
        "seed": R2_SEED,
        "config_sha256": config_sha256,
        "preflight_sha256": preflight_sha256,
        "calibration_sha256": sha256_file(calibration_path),
        "gpu": gpu_record,
        "completed_optimizer_steps": final_step,
        "termination_reason": termination_reason,
        "gpu_wall_seconds": wall_seconds,
        "max_gpu_wall_hours": R2_PRIMARY_MAX_GPU_WALL_HOURS,
        "positive_patch_count": positive_patches,
        "positive_patch_ratio": float(positive_patches / final_step),
        "peak_vram_allocated_bytes": int(
            torch.cuda.max_memory_allocated(device)
        ),
        "peak_vram_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "final_positive_native_full_volume_dice": positive_dice,
        "all_positive_native_full_volume_dice_at_least_0_95": all(
            value >= 0.95 for value in positive_dice
        ),
        "empty_native_predicted_tumor_voxels": empty_prediction_voxels,
        "empty_binary_prediction_is_empty": empty_prediction_voxels == 0,
        "checkpoint_relative_path": "private_checkpoints/r2_primary_latest.pt",
        "reload_reference_relative_path": "private_reload_reference",
        "medical_images_in_summary": False,
        "raw_labels_in_summary": False,
        "checkpoint_in_summary": False,
        "validation_arrays_opened": 0,
        "internal_test_arrays_opened": 0,
        "external_arrays_opened": 0,
        "interpretation": (
            "Diagnostic training-set overfit run only; these results are not "
            "evidence of unseen-patient performance."
        ),
    }
    _write_json(output_dir / "r2_primary_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"R2_PRIMARY_SUMMARY={output_dir / 'r2_primary_summary.json'}")
    return 0


def _failure_payload(mode: str, exc: RuntimeError) -> dict[str, Any]:
    return {
        "schema_version": "research_v2_r2_run_failure.v1",
        "status": "FAILED",
        "mode": mode,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "main_training_started": mode == "primary",
        "threshold_changed": False,
        "gate_threshold_changed": False,
        "validation_arrays_opened": 0,
        "internal_test_arrays_opened": 0,
        "external_arrays_opened": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("calibration", "primary"),
        required=True,
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-evidence", type=Path, required=True)
    parser.add_argument("--calibration-evidence", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    repo_root = args.repo_root.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    preflight_path = args.preflight_evidence.expanduser().resolve()

    if not data_root.is_dir():
        raise RuntimeError(f"R2 selected train data root is missing: {data_root}")
    if output_dir.exists():
        raise RuntimeError(f"refusing to overwrite R2 run directory: {output_dir}")
    output_dir.mkdir(parents=True)

    try:
        lock_path = repo_root / LOCK_RELATIVE_PATH
        if sha256_file(lock_path) != EXPECTED_LOCK_SHA256:
            raise RuntimeError("R2 uv.lock differs from the frozen R0 CUDA lock")

        config_path = repo_root / CONFIG_RELATIVE_PATH
        config = _load_json(config_path)
        _verify_locked_config(config)
        config_sha256 = sha256_file(config_path)
        _verify_preflight(preflight_path)
        _verify_selected_files(data_root)

        torch = importlib.import_module("torch")
        monai = importlib.import_module("monai")
        gpu_record = _gpu_evidence(torch)
        cases = [_prepare_case(data_root, spec) for spec in R2_SELECTED_CASES]
        preflight_sha256 = sha256_file(preflight_path)

        if args.mode == "calibration":
            return _run_calibration(
                torch=torch,
                monai=monai,
                cases=cases,
                output_dir=output_dir,
                config_sha256=config_sha256,
                gpu_record=gpu_record,
                preflight_sha256=preflight_sha256,
            )

        if args.calibration_evidence is None:
            raise RuntimeError("R2 primary requires --calibration-evidence")
        return _run_primary(
            torch=torch,
            monai=monai,
            cases=cases,
            output_dir=output_dir,
            config_sha256=config_sha256,
            gpu_record=gpu_record,
            preflight_sha256=preflight_sha256,
            calibration_path=args.calibration_evidence.expanduser().resolve(),
        )
    except RuntimeError as exc:
        failure = _failure_payload(args.mode, exc)
        _write_json(output_dir / "r2_failure.json", failure)
        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)
        return 40


if __name__ == "__main__":
    sys.exit(main())
