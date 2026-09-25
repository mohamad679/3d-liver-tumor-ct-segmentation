#!/usr/bin/env python3
"""Replay R2 primary attempt #1 and diagnose the step-227 gradient failure."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
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
from protoem_ct.research_v2.r2_numerical_diagnostic import (
    R2_DIAGNOSTIC_FAILURE_STEP,
    R2_DIAGNOSTIC_REPLAY_STEPS,
    R2_GRAD_SCALER_GROWTH_INTERVAL,
    R2_GRAD_SCALER_INITIAL_SCALE,
    R2_REPLAY_LOSS_ATOL,
    R2_REPLAY_PATCH_DICE_ATOL,
    classify_probe,
    expected_loss_scale_before_step,
)
from protoem_ct.research_v2.r2_overfit import (
    R2_SELECTED_CASES,
    binary_dice,
    choose_random_center,
    choose_random_positive_center,
    positive_case_index_for_step,
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
EXPECTED_PRIMARY_CONFIG_SHA256 = "dfe50f52110bd77861f52fc4683712e0549a1f0f77c363f64f09da05940f3d1a"
EXPECTED_ATTEMPT1_STEPS_SHA256 = "e9288e11aeafbd3214e8dc9e6c35b68432be92e9a1f01c6ae2cba8aea1e3e4b2"
EXPECTED_ATTEMPT1_FAILURE_SHA256 = "5a4ee93ab5ceafef673f4d7af20cdeff6cecf5374df1e45b9f6d7cb87246b1f2"
LOCK_RELATIVE_PATH = Path("environments/research-v2-linux-cuda/uv.lock")
PRIMARY_CONFIG_RELATIVE_PATH = Path("configs/research_v2/r2_overfit_real_v1.json")
DIAGNOSTIC_CONFIG_RELATIVE_PATH = Path("configs/research_v2/r2_numerical_stability_diag_v1.json")


@dataclass(slots=True)
class PreparedCase:
    anonymous_case_id: str
    image: Any
    tumor: Any


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise RuntimeError(f"expected JSON object per line: {path}")
        records.append(payload)
    return records


def _verify_prerequisites(
    *,
    repo_root: Path,
    data_root: Path,
    attempt1_steps: Path,
    attempt1_failure: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if sha256_file(repo_root / LOCK_RELATIVE_PATH) != EXPECTED_LOCK_SHA256:
        raise RuntimeError("R2 diagnostic uv.lock differs from frozen R0 lock")
    if sha256_file(repo_root / PRIMARY_CONFIG_RELATIVE_PATH) != EXPECTED_PRIMARY_CONFIG_SHA256:
        raise RuntimeError("R2 diagnostic primary config differs from preregistered v1 contract")
    if sha256_file(attempt1_steps) != EXPECTED_ATTEMPT1_STEPS_SHA256:
        raise RuntimeError("R2 diagnostic attempt-1 steps hash mismatch")
    if sha256_file(attempt1_failure) != EXPECTED_ATTEMPT1_FAILURE_SHA256:
        raise RuntimeError("R2 diagnostic attempt-1 failure hash mismatch")

    records = _load_jsonl(attempt1_steps)
    failure = _load_json(attempt1_failure)
    if len(records) != R2_DIAGNOSTIC_REPLAY_STEPS:
        raise RuntimeError("R2 diagnostic expected exactly 226 attempt-1 step records")
    if [int(record["step"]) for record in records] != list(
        range(1, R2_DIAGNOSTIC_REPLAY_STEPS + 1)
    ):
        raise RuntimeError("R2 diagnostic attempt-1 step records are not contiguous")
    if int(failure.get("step_reached", -1)) != R2_DIAGNOSTIC_FAILURE_STEP:
        raise RuntimeError("R2 diagnostic attempt-1 failure step differs from 227")
    if failure.get("error") != "R2 primary produced non-finite gradients":
        raise RuntimeError("R2 diagnostic attempt-1 failure reason changed")

    expected_paths: set[Path] = set()
    for spec in R2_SELECTED_CASES:
        case_root = data_root / spec.anonymous_case_id
        image_path = case_root / "image.nii.gz"
        label_path = case_root / "label.nii.gz"
        expected_paths.update({image_path.resolve(), label_path.resolve()})
        if sha256_file(image_path) != spec.image_sha256:
            raise RuntimeError(f"R2 diagnostic image hash mismatch: {spec.anonymous_case_id}")
        if sha256_file(label_path) != spec.label_sha256:
            raise RuntimeError(f"R2 diagnostic label hash mismatch: {spec.anonymous_case_id}")
    observed_paths = {
        path.resolve()
        for pattern in ("*.nii", "*.nii.gz")
        for path in data_root.rglob(pattern)
        if path.is_file()
    }
    if observed_paths != expected_paths:
        raise RuntimeError("R2 diagnostic data root must contain exactly eight selected train files")
    return records, failure


def _require_gpu(torch: Any) -> dict[str, Any]:
    if not bool(torch.cuda.is_available()):
        raise RuntimeError("R2 diagnostic requires CUDA; CPU fallback is forbidden")
    if int(torch.cuda.device_count()) != 2:
        raise RuntimeError("R2 diagnostic requires the registered two-GPU environment")
    devices: list[dict[str, Any]] = []
    for index in range(2):
        properties = torch.cuda.get_device_properties(index)
        if str(properties.name) != "Tesla T4":
            raise RuntimeError("R2 diagnostic requires Tesla T4 devices")
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
        raise RuntimeError(f"R2 diagnostic source shape mismatch: {spec.anonymous_case_id}")
    if not bool(np.isfinite(image).all()):
        raise RuntimeError(f"R2 diagnostic CT contains NaN/Inf: {spec.anonymous_case_id}")
    image_affine = np.asarray(image_nifti.affine, dtype=np.float64)
    label_affine = np.asarray(label_nifti.affine, dtype=np.float64)
    if float(np.max(np.abs(image_affine - label_affine))) > 1e-4:
        raise RuntimeError(f"R2 diagnostic source affine mismatch: {spec.anonymous_case_id}")
    native_tumor = r2_binary_tumor_target(label)
    if int(np.count_nonzero(native_tumor)) != int(spec.tumor_voxel_count):
        raise RuntimeError(f"R2 diagnostic tumor voxel mismatch: {spec.anonymous_case_id}")

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
    return PreparedCase(
        anonymous_case_id=spec.anonymous_case_id,
        image=normalize_r2_ct(image_resampled),
        tumor=np.asarray(tumor_resampled, dtype=bool),
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
        raise RuntimeError("R2 diagnostic positive patch contains no tumor")
    if role == "empty" and tumor_voxels != 0:
        raise RuntimeError("R2 diagnostic empty patch contains tumor")
    return (
        image_patch,
        target_patch,
        {
            "anonymous_case_id": case.anonymous_case_id,
            "sampling_role": role,
            "tumor_voxels": tumor_voxels,
        },
    )


def _gradient_audit(model: Any) -> dict[str, Any]:
    tensor_count = 0
    nonfinite_tensor_count = 0
    nonfinite_value_count = 0
    max_finite_abs = 0.0
    for parameter in model.parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        tensor_count += 1
        finite_mask = gradient.isfinite()
        if not bool(finite_mask.all().item()):
            nonfinite_tensor_count += 1
            nonfinite_value_count += int((~finite_mask).sum().item())
        if bool(finite_mask.any().item()):
            finite_values = gradient.detach()[finite_mask]
            max_finite_abs = max(max_finite_abs, float(finite_values.abs().max().item()))
    if tensor_count == 0:
        raise RuntimeError("R2 diagnostic produced no gradients")
    return {
        "gradient_tensor_count": tensor_count,
        "nonfinite_gradient_tensor_count": nonfinite_tensor_count,
        "nonfinite_gradient_value_count": nonfinite_value_count,
        "gradients_finite": nonfinite_tensor_count == 0,
        "max_finite_abs_gradient": max_finite_abs,
    }


def _replay_step(
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
    scale_before = float(scaler.get_scale())
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
        logits = model(image_tensor)
        loss = loss_function(logits, target_tensor)
    logits_finite = bool(torch.isfinite(logits).all().item())
    loss_finite = bool(torch.isfinite(loss).item())
    if not logits_finite or not loss_finite:
        raise RuntimeError("R2 diagnostic replay produced non-finite forward values")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gradients = _gradient_audit(model)
    if not gradients["gradients_finite"]:
        raise RuntimeError("R2 diagnostic replay diverged before the original failure step")
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize(device)
    with torch.no_grad():
        probability = torch.softmax(logits.float(), dim=1)[:, 1]
    probability_np = np.asarray(probability[0].detach().cpu().numpy(), dtype=np.float32)
    patch_dice = binary_dice(np.asarray(target_patch, dtype=bool), threshold_probability(probability_np))
    return {
        "loss": float(loss.detach().float().cpu().item()),
        "patch_dice": patch_dice,
        "scale_before": scale_before,
        "scale_after": float(scaler.get_scale()),
        **gradients,
    }


def _probe_backward(
    *,
    torch: Any,
    model: Any,
    optimizer: Any,
    loss_function: Any,
    image_patch: Any,
    target_patch: Any,
    device: Any,
    autocast_enabled: bool,
    loss_scale: float,
) -> dict[str, Any]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    image_tensor = torch.from_numpy(image_patch[None, None]).to(device=device, dtype=torch.float32)
    target_tensor = torch.from_numpy(target_patch[None, None]).to(device=device, dtype=torch.long)
    with torch.cuda.amp.autocast(enabled=autocast_enabled, dtype=torch.float16):
        logits = model(image_tensor)
        loss = loss_function(logits, target_tensor)
    logits_finite = bool(torch.isfinite(logits).all().item())
    loss_finite = bool(torch.isfinite(loss).item())
    if loss_scale == 1.0:
        loss.backward()
    else:
        (loss * loss_scale).backward()
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(loss_scale)
    gradients = _gradient_audit(model)
    return {
        "autocast_enabled": autocast_enabled,
        "loss_scale": loss_scale,
        "logits_finite": logits_finite,
        "loss_finite": loss_finite,
        "loss": float(loss.detach().float().cpu().item()),
        "optimizer_step_performed": False,
        **gradients,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--attempt1-steps", type=Path, required=True)
    parser.add_argument("--attempt1-failure", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    repo_root = args.repo_root.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    attempt1_steps = args.attempt1_steps.expanduser().resolve()
    attempt1_failure = args.attempt1_failure.expanduser().resolve()
    if output_dir.exists():
        raise RuntimeError(f"refusing to overwrite R2 diagnostic dir: {output_dir}")
    output_dir.mkdir(parents=True)
    output_path = output_dir / "r2_numerical_stability_diagnostic.json"
    started = time.perf_counter()

    try:
        attempt_records, _ = _verify_prerequisites(
            repo_root=repo_root,
            data_root=data_root,
            attempt1_steps=attempt1_steps,
            attempt1_failure=attempt1_failure,
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

        max_loss_abs_difference = 0.0
        max_patch_dice_abs_difference = 0.0
        scale_trace: list[dict[str, Any]] = []
        for step in range(1, R2_DIAGNOSTIC_REPLAY_STEPS + 1):
            image_patch, target_patch, sampling = _sample_patch(cases, step=step, rng=rng)
            expected = attempt_records[step - 1]
            if sampling["anonymous_case_id"] != expected["anonymous_case_id"]:
                raise RuntimeError(f"REPLAY_MISMATCH: case id at step {step}")
            if sampling["sampling_role"] != expected["sampling_role"]:
                raise RuntimeError(f"REPLAY_MISMATCH: sampling role at step {step}")
            if int(sampling["tumor_voxels"]) != int(expected["tumor_voxels"]):
                raise RuntimeError(f"REPLAY_MISMATCH: tumor voxels at step {step}")
            observed = _replay_step(
                torch=torch,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                loss_function=loss_function,
                image_patch=image_patch,
                target_patch=target_patch,
                device=device,
            )
            loss_difference = abs(float(observed["loss"]) - float(expected["loss"]))
            dice_difference = abs(float(observed["patch_dice"]) - float(expected["patch_dice"]))
            max_loss_abs_difference = max(max_loss_abs_difference, loss_difference)
            max_patch_dice_abs_difference = max(max_patch_dice_abs_difference, dice_difference)
            if loss_difference > R2_REPLAY_LOSS_ATOL:
                raise RuntimeError(f"REPLAY_MISMATCH: loss at step {step}")
            if dice_difference > R2_REPLAY_PATCH_DICE_ATOL:
                raise RuntimeError(f"REPLAY_MISMATCH: patch Dice at step {step}")
            if step in {1, 199, 200, 201, 226}:
                scale_trace.append(
                    {
                        "step": step,
                        "scale_before": observed["scale_before"],
                        "scale_after": observed["scale_after"],
                    }
                )

        expected_scale = expected_loss_scale_before_step(R2_DIAGNOSTIC_FAILURE_STEP)
        observed_scale = float(scaler.get_scale())
        if observed_scale != expected_scale:
            raise RuntimeError(
                f"REPLAY_MISMATCH: expected step-227 scale {expected_scale}, observed {observed_scale}"
            )

        probe_image, probe_target, probe_sampling = _sample_patch(
            cases,
            step=R2_DIAGNOSTIC_FAILURE_STEP,
            rng=rng,
        )
        if probe_sampling["sampling_role"] != "positive":
            raise RuntimeError("R2 diagnostic step 227 is not a positive patch")
        if probe_sampling["anonymous_case_id"] != R2_SELECTED_CASES[2].anonymous_case_id:
            raise RuntimeError("R2 diagnostic step 227 is not the preregistered q80 case")

        observed_scale_probe = _probe_backward(
            torch=torch,
            model=model,
            optimizer=optimizer,
            loss_function=loss_function,
            image_patch=probe_image,
            target_patch=probe_target,
            device=device,
            autocast_enabled=True,
            loss_scale=observed_scale,
        )
        baseline_scale_probe = _probe_backward(
            torch=torch,
            model=model,
            optimizer=optimizer,
            loss_function=loss_function,
            image_patch=probe_image,
            target_patch=probe_target,
            device=device,
            autocast_enabled=True,
            loss_scale=R2_GRAD_SCALER_INITIAL_SCALE,
        )
        fp32_probe = _probe_backward(
            torch=torch,
            model=model,
            optimizer=optimizer,
            loss_function=loss_function,
            image_patch=probe_image,
            target_patch=probe_target,
            device=device,
            autocast_enabled=False,
            loss_scale=1.0,
        )
        classification = classify_probe(
            observed_scale_amp_gradients_finite=bool(observed_scale_probe["gradients_finite"]),
            baseline_scale_amp_gradients_finite=bool(baseline_scale_probe["gradients_finite"]),
            fp32_gradients_finite=bool(fp32_probe["gradients_finite"]),
        )
        result = {
            "schema_version": "research_v2_r2_numerical_stability_diagnostic.v1",
            "status": "COMPLETE",
            "classification": classification,
            "interpretation": (
                "Root-cause diagnostic only; this result cannot PASS R2 and does not estimate "
                "unseen-patient performance."
            ),
            "gpu": gpu,
            "seed": R2_SEED,
            "primary_attempt1_steps_sha256": EXPECTED_ATTEMPT1_STEPS_SHA256,
            "primary_attempt1_failure_sha256": EXPECTED_ATTEMPT1_FAILURE_SHA256,
            "diagnostic_config_sha256": sha256_file(repo_root / DIAGNOSTIC_CONFIG_RELATIVE_PATH),
            "replay": {
                "optimizer_steps_replayed": R2_DIAGNOSTIC_REPLAY_STEPS,
                "all_sampling_records_matched_exactly": True,
                "max_loss_abs_difference": max_loss_abs_difference,
                "loss_abs_tolerance": R2_REPLAY_LOSS_ATOL,
                "max_patch_dice_abs_difference": max_patch_dice_abs_difference,
                "patch_dice_abs_tolerance": R2_REPLAY_PATCH_DICE_ATOL,
                "scale_trace": scale_trace,
                "expected_scale_before_step_227": expected_scale,
                "observed_scale_before_step_227": observed_scale,
            },
            "probe_step": R2_DIAGNOSTIC_FAILURE_STEP,
            "probe_sampling": probe_sampling,
            "probe_optimizer_steps_performed": 0,
            "probe_branches": {
                "amp_observed_scale": observed_scale_probe,
                "amp_baseline_65536": baseline_scale_probe,
                "fp32": fp32_probe,
            },
            "main_training_started": False,
            "diagnostic_optimizer_steps_performed": R2_DIAGNOSTIC_REPLAY_STEPS,
            "checkpoint_created": False,
            "warm_start_allowed": False,
            "threshold_changed": False,
            "gate_threshold_changed": False,
            "full_volume_performance_evaluation_performed": False,
            "validation_arrays_opened": 0,
            "internal_test_arrays_opened": 0,
            "external_arrays_opened": 0,
            "elapsed_seconds": float(time.perf_counter() - started),
        }
        _write_json(output_path, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (RuntimeError, ValueError) as exc:
        failure = {
            "schema_version": "research_v2_r2_numerical_stability_diagnostic_failure.v1",
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
            "elapsed_seconds": float(time.perf_counter() - started),
        }
        _write_json(output_dir / "r2_numerical_stability_diagnostic_failure.json", failure)
        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)
        return 40


if __name__ == "__main__":
    sys.exit(main())
