#!/usr/bin/env python3
"""Replay R2 primary attempt #1 and diagnose the step-227 gradient failure, v2."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

os.environ["MPLBACKEND"] = "Agg"

import numpy as np

from protoem_ct.artifacts import sha256_file
from protoem_ct.research_v2.r2_numerical_diagnostic_v2 import (
    R2_DIAGNOSTIC_FAILURE_STEP,
    R2_DIAGNOSTIC_REPLAY_STEPS,
    R2_GRAD_SCALER_BACKOFF_FACTOR,
    R2_GRAD_SCALER_GROWTH_FACTOR,
    R2_GRAD_SCALER_GROWTH_INTERVAL,
    R2_GRAD_SCALER_INITIAL_SCALE,
    R2_REPLAY_LOSS_ATOL,
    R2_REPLAY_PATCH_DICE_ATOL,
    R2_SINGLE_BACKOFF_SCALE,
    classify_probe,
    expected_loss_scale_before_step,
)
from protoem_ct.research_v2.r2_overfit import R2_SELECTED_CASES
from protoem_ct.research_v2.r2_preflight import R2_SEED

EXPECTED_LOCK_SHA256 = "c9ee200234eebf8c67fd97ff2c96ca0765f4067a8a69a7ef439deafdda3530a8"
EXPECTED_PRIMARY_CONFIG_SHA256 = "dfe50f52110bd77861f52fc4683712e0549a1f0f77c363f64f09da05940f3d1a"
EXPECTED_ATTEMPT1_STEPS_SHA256 = "e9288e11aeafbd3214e8dc9e6c35b68432be92e9a1f01c6ae2cba8aea1e3e4b2"
EXPECTED_ATTEMPT1_FAILURE_SHA256 = (
    "5a4ee93ab5ceafef673f4d7af20cdeff6cecf5374df1e45b9f6d7cb87246b1f2"
)
LOCK_PATH = Path("environments/research-v2-linux-cuda/uv.lock")
PRIMARY_CONFIG_PATH = Path("configs/research_v2/r2_overfit_real_v1.json")
DIAGNOSTIC_CONFIG_PATH = Path("configs/research_v2/r2_numerical_stability_diag_v2.json")
PRIMARY_RUNNER_PATH = Path("scripts/research_v2/run_r2_primary.py")


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


def _load_primary_runtime(repo_root: Path) -> Any:
    path = repo_root / PRIMARY_RUNNER_PATH
    spec = importlib.util.spec_from_file_location("r2_primary_runtime_v2diag", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load the locked R2 primary runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _verify_inputs(
    *,
    repo_root: Path,
    data_root: Path,
    attempt1_steps: Path,
    attempt1_failure: Path,
) -> list[dict[str, Any]]:
    if sha256_file(repo_root / LOCK_PATH) != EXPECTED_LOCK_SHA256:
        raise RuntimeError("R2 diagnostic v2 uv.lock differs from frozen R0 lock")
    if sha256_file(repo_root / PRIMARY_CONFIG_PATH) != EXPECTED_PRIMARY_CONFIG_SHA256:
        raise RuntimeError("R2 primary config differs from the locked v1 contract")
    if sha256_file(attempt1_steps) != EXPECTED_ATTEMPT1_STEPS_SHA256:
        raise RuntimeError("R2 attempt-1 steps hash mismatch")
    if sha256_file(attempt1_failure) != EXPECTED_ATTEMPT1_FAILURE_SHA256:
        raise RuntimeError("R2 attempt-1 failure hash mismatch")

    records = _load_jsonl(attempt1_steps)
    failure = _load_json(attempt1_failure)
    if len(records) != R2_DIAGNOSTIC_REPLAY_STEPS:
        raise RuntimeError("expected exactly 226 attempt-1 step records")
    if [int(record["step"]) for record in records] != list(
        range(1, R2_DIAGNOSTIC_REPLAY_STEPS + 1)
    ):
        raise RuntimeError("attempt-1 step records are not contiguous")
    if int(failure.get("step_reached", -1)) != R2_DIAGNOSTIC_FAILURE_STEP:
        raise RuntimeError("attempt-1 failure step differs from 227")
    if failure.get("error") != "R2 primary produced non-finite gradients":
        raise RuntimeError("attempt-1 failure reason changed")

    expected_paths: set[Path] = set()
    for case in R2_SELECTED_CASES:
        root = data_root / case.anonymous_case_id
        image_path = root / "image.nii.gz"
        label_path = root / "label.nii.gz"
        expected_paths.update({image_path.resolve(), label_path.resolve()})
        if sha256_file(image_path) != case.image_sha256:
            raise RuntimeError(f"image hash mismatch: {case.anonymous_case_id}")
        if sha256_file(label_path) != case.label_sha256:
            raise RuntimeError(f"label hash mismatch: {case.anonymous_case_id}")
    observed_paths = {
        path.resolve()
        for pattern in ("*.nii", "*.nii.gz")
        for path in data_root.rglob(pattern)
        if path.is_file()
    }
    if observed_paths != expected_paths:
        raise RuntimeError("diagnostic data root must contain exactly eight selected train files")
    return records


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
        finite = gradient.isfinite()
        if not bool(finite.all().item()):
            nonfinite_tensor_count += 1
            nonfinite_value_count += int((~finite).sum().item())
        if bool(finite.any().item()):
            finite_values = gradient.detach()[finite]
            max_finite_abs = max(max_finite_abs, float(finite_values.abs().max().item()))
    if tensor_count == 0:
        raise RuntimeError("diagnostic produced no gradients")
    return {
        "gradient_tensor_count": tensor_count,
        "nonfinite_gradient_tensor_count": nonfinite_tensor_count,
        "nonfinite_gradient_value_count": nonfinite_value_count,
        "gradients_finite": nonfinite_tensor_count == 0,
        "max_finite_abs_gradient": max_finite_abs,
    }


def _probe_backward(
    *,
    torch: Any,
    model: Any,
    loss_function: Any,
    image_patch: Any,
    target_patch: Any,
    device: Any,
    autocast_enabled: bool,
    loss_scale: float,
) -> dict[str, Any]:
    model.train()
    model.zero_grad(set_to_none=True)
    image = torch.from_numpy(image_patch[None, None]).to(device=device, dtype=torch.float32)
    target = torch.from_numpy(target_patch[None, None]).to(device=device, dtype=torch.long)
    with torch.cuda.amp.autocast(enabled=autocast_enabled, dtype=torch.float16):
        logits = model(image)
        loss = loss_function(logits, target)
    if loss_scale == 1.0:
        loss.backward()
    else:
        (loss * loss_scale).backward()
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(loss_scale)
    torch.cuda.synchronize(device)
    return {
        "autocast_enabled": autocast_enabled,
        "loss_scale": loss_scale,
        "logits_finite": bool(torch.isfinite(logits).all().item()),
        "loss_finite": bool(torch.isfinite(loss).item()),
        "loss": float(loss.detach().float().cpu().item()),
        "optimizer_step_performed": False,
        **_gradient_audit(model),
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
        raise RuntimeError(f"refusing to overwrite R2 diagnostic v2 dir: {output_dir}")
    output_dir.mkdir(parents=True)
    output_path = output_dir / "r2_numerical_stability_diagnostic_v2.json"
    started = time.perf_counter()

    try:
        attempt_records = _verify_inputs(
            repo_root=repo_root,
            data_root=data_root,
            attempt1_steps=attempt1_steps,
            attempt1_failure=attempt1_failure,
        )
        primary = _load_primary_runtime(repo_root)
        torch = importlib.import_module("torch")
        monai = importlib.import_module("monai")
        gpu = primary._require_gpu(torch)
        device = torch.device("cuda:0")

        random.seed(R2_SEED)
        np.random.seed(R2_SEED)
        torch.manual_seed(R2_SEED)
        torch.cuda.manual_seed_all(R2_SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        rng = np.random.default_rng(R2_SEED)
        cases = [primary._prepare_case(data_root, case) for case in R2_SELECTED_CASES]
        model = primary._build_model(monai, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.00001)
        scaler = torch.cuda.amp.GradScaler(
            enabled=True,
            init_scale=R2_GRAD_SCALER_INITIAL_SCALE,
            growth_factor=R2_GRAD_SCALER_GROWTH_FACTOR,
            backoff_factor=R2_GRAD_SCALER_BACKOFF_FACTOR,
            growth_interval=R2_GRAD_SCALER_GROWTH_INTERVAL,
        )
        loss_function = monai.losses.DiceCELoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
            lambda_dice=1.0,
            lambda_ce=1.0,
        )

        max_loss_difference = 0.0
        max_dice_difference = 0.0
        for step in range(1, R2_DIAGNOSTIC_REPLAY_STEPS + 1):
            image_patch, target_patch, sampling = primary._sample_patch(cases, step=step, rng=rng)
            expected = attempt_records[step - 1]
            if sampling["anonymous_case_id"] != expected["anonymous_case_id"]:
                raise RuntimeError(f"REPLAY_MISMATCH: case id at step {step}")
            if sampling["sampling_role"] != expected["sampling_role"]:
                raise RuntimeError(f"REPLAY_MISMATCH: sampling role at step {step}")
            if int(sampling["tumor_voxels"]) != int(expected["tumor_voxels"]):
                raise RuntimeError(f"REPLAY_MISMATCH: tumor voxels at step {step}")
            observed = primary._optimizer_step(
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
            max_loss_difference = max(max_loss_difference, loss_difference)
            max_dice_difference = max(max_dice_difference, dice_difference)
            if loss_difference > R2_REPLAY_LOSS_ATOL:
                raise RuntimeError(f"REPLAY_MISMATCH: loss at step {step}")
            if dice_difference > R2_REPLAY_PATCH_DICE_ATOL:
                raise RuntimeError(f"REPLAY_MISMATCH: patch Dice at step {step}")

        observed_scale = float(scaler.get_scale())
        expected_scale = expected_loss_scale_before_step(R2_DIAGNOSTIC_FAILURE_STEP)
        if observed_scale != expected_scale:
            raise RuntimeError(
                "REPLAY_MISMATCH: step-227 scale expected "
                f"{expected_scale}, observed {observed_scale}"
            )

        probe_image, probe_target, probe_sampling = primary._sample_patch(
            cases,
            step=R2_DIAGNOSTIC_FAILURE_STEP,
            rng=rng,
        )
        if probe_sampling["anonymous_case_id"] != R2_SELECTED_CASES[2].anonymous_case_id:
            raise RuntimeError("step-227 probe did not select the q80 case")

        probe_state = {key: value.detach().clone() for key, value in model.state_dict().items()}

        model.load_state_dict(probe_state)
        observed_probe = _probe_backward(
            torch=torch,
            model=model,
            loss_function=loss_function,
            image_patch=probe_image,
            target_patch=probe_target,
            device=device,
            autocast_enabled=True,
            loss_scale=observed_scale,
        )

        model.load_state_dict(probe_state)
        backoff_probe = _probe_backward(
            torch=torch,
            model=model,
            loss_function=loss_function,
            image_patch=probe_image,
            target_patch=probe_target,
            device=device,
            autocast_enabled=True,
            loss_scale=R2_SINGLE_BACKOFF_SCALE,
        )

        model.load_state_dict(probe_state)
        fp32_probe = _probe_backward(
            torch=torch,
            model=model,
            loss_function=loss_function,
            image_patch=probe_image,
            target_patch=probe_target,
            device=device,
            autocast_enabled=False,
            loss_scale=1.0,
        )

        classification = classify_probe(
            observed_scale_amp_gradients_finite=bool(observed_probe["gradients_finite"]),
            backoff_scale_amp_gradients_finite=bool(backoff_probe["gradients_finite"]),
            fp32_gradients_finite=bool(fp32_probe["gradients_finite"]),
        )
        result = {
            "schema_version": "research_v2_r2_numerical_stability_diagnostic.v2",
            "status": "COMPLETE",
            "classification": classification,
            "interpretation": (
                "Root-cause diagnostic only; cannot PASS R2 or estimate unseen-patient performance."
            ),
            "gpu": gpu,
            "seed": R2_SEED,
            "main_training_started": False,
            "checkpoint_created": False,
            "threshold_changed": False,
            "gate_threshold_changed": False,
            "validation_arrays_opened": 0,
            "internal_test_arrays_opened": 0,
            "external_arrays_opened": 0,
            "attempt1_steps_sha256": EXPECTED_ATTEMPT1_STEPS_SHA256,
            "attempt1_failure_sha256": EXPECTED_ATTEMPT1_FAILURE_SHA256,
            "diagnostic_config_sha256": sha256_file(repo_root / DIAGNOSTIC_CONFIG_PATH),
            "replay": {
                "optimizer_steps_replayed": R2_DIAGNOSTIC_REPLAY_STEPS,
                "all_sampling_records_matched_exactly": True,
                "max_loss_abs_difference": max_loss_difference,
                "loss_abs_tolerance": R2_REPLAY_LOSS_ATOL,
                "max_patch_dice_abs_difference": max_dice_difference,
                "patch_dice_abs_tolerance": R2_REPLAY_PATCH_DICE_ATOL,
                "scale_before_step_227": observed_scale,
                "expected_scale_before_step_227": expected_scale,
            },
            "probe": {
                "step": R2_DIAGNOSTIC_FAILURE_STEP,
                "anonymous_case_id": probe_sampling["anonymous_case_id"],
                "sampling_role": probe_sampling["sampling_role"],
                "tumor_voxels": int(probe_sampling["tumor_voxels"]),
                "optimizer_updates": 0,
                "amp_observed_scale": observed_probe,
                "amp_single_backoff_scale": backoff_probe,
                "fp32": fp32_probe,
            },
            "elapsed_seconds": float(time.perf_counter() - started),
        }
        _write_json(output_path, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (RuntimeError, ValueError) as exc:
        failure = {
            "schema_version": "research_v2_r2_numerical_stability_diagnostic_failure.v2",
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
        _write_json(output_dir / "r2_numerical_stability_diagnostic_failure_v2.json", failure)
        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)
        return 40


if __name__ == "__main__":
    sys.exit(main())
