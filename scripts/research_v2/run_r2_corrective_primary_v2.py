#!/usr/bin/env python3
"""Run R2 corrective primary v2 with standard AMP overflow recovery."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ["MPLBACKEND"] = "Agg"

import numpy as np

from protoem_ct.artifacts import sha256_file
from protoem_ct.research_v2.r2_amp_recovery import (
    R2_AMP_MAX_OVERFLOW_RETRIES_PER_STEP,
    R2_DIAGNOSTIC_V2_JSON_SHA256,
    overflow_retry_allowed,
)

CORRECTIVE_CONFIG_RELATIVE_PATH = Path("configs/research_v2/r2_corrective_primary_v2.json")
PRIMARY_RUNNER_PATH = Path(__file__).with_name("run_r2_primary.py")
primary: Any


def _load_primary_runtime() -> Any:
    spec = importlib.util.spec_from_file_location(
        "r2_primary_corrective_v2_runtime",
        PRIMARY_RUNNER_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load locked R2 primary runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _pop_required_cli_path(flag: str) -> Path:
    matches = [index for index, value in enumerate(sys.argv) if value == flag]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one {flag} argument")
    index = matches[0]
    if index + 1 >= len(sys.argv):
        raise RuntimeError(f"missing value for {flag}")
    path = Path(sys.argv[index + 1]).expanduser().resolve()
    del sys.argv[index : index + 2]
    return path


def _peek_required_cli_path(flag: str) -> Path:
    matches = [index for index, value in enumerate(sys.argv) if value == flag]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one {flag} argument")
    index = matches[0]
    if index + 1 >= len(sys.argv):
        raise RuntimeError(f"missing value for {flag}")
    return Path(sys.argv[index + 1]).expanduser().resolve()


def _optimizer_step_counter(optimizer: Any) -> int:
    values: list[int] = []
    for state in optimizer.state.values():
        if not isinstance(state, dict) or "step" not in state:
            continue
        step = state["step"]
        if hasattr(step, "item"):
            values.append(int(step.item()))
        else:
            values.append(int(step))
    return max(values, default=0)


def _gradient_audit(torch: Any, model: Any) -> tuple[int, bool]:
    count = 0
    finite = True
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        count += 1
        if not bool(torch.isfinite(parameter.grad).all().item()):
            finite = False
    if count == 0:
        raise RuntimeError("R2 corrective primary produced no gradients")
    return count, finite


def _recovering_optimizer_step(
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
    image_tensor = torch.from_numpy(image_patch[None, None]).to(
        device=device,
        dtype=torch.float32,
    )
    target_tensor = torch.from_numpy(target_patch[None, None]).to(
        device=device,
        dtype=torch.long,
    )
    started = time.perf_counter()
    overflow_scales: list[float] = []
    retries_started = 0

    while True:
        optimizer.zero_grad(set_to_none=True)
        scale_before = float(scaler.get_scale())
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
            logits = model(image_tensor)
            loss = loss_function(logits, target_tensor)
        if not bool(torch.isfinite(logits).all().item()):
            raise RuntimeError("R2 corrective primary produced non-finite logits")
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError("R2 corrective primary produced non-finite loss")

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient_count, gradients_finite = _gradient_audit(torch, model)
        optimizer_step_before = _optimizer_step_counter(optimizer)

        if not gradients_finite:
            overflow_scales.append(scale_before)
            scaler.step(optimizer)
            scaler.update()
            optimizer_step_after = _optimizer_step_counter(optimizer)
            scale_after = float(scaler.get_scale())
            if optimizer_step_after != optimizer_step_before:
                raise RuntimeError("AMP overflow unexpectedly changed optimizer state step")
            if not scale_after < scale_before:
                raise RuntimeError("AMP overflow did not reduce GradScaler scale")
            if not overflow_retry_allowed(retries_started):
                raise RuntimeError(
                    "R2 corrective primary exceeded the preregistered AMP overflow retry limit"
                )
            retries_started += 1
            continue

        scaler.step(optimizer)
        scaler.update()
        optimizer_step_after = _optimizer_step_counter(optimizer)
        if optimizer_step_after != optimizer_step_before + 1:
            raise RuntimeError("successful R2 optimizer update did not advance exactly one step")
        scale_after_success = float(scaler.get_scale())
        torch.cuda.synchronize(device)

        with torch.no_grad():
            probability = torch.softmax(logits.float(), dim=1)[:, 1]
        probability_np = np.asarray(
            probability[0].detach().cpu().numpy(),
            dtype=np.float32,
        )
        prediction_patch = primary.threshold_probability(probability_np)
        patch_dice = primary.binary_dice(
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
            "amp_overflow_retry_count": retries_started,
            "amp_overflow_scales": overflow_scales,
            "amp_success_scale_before_update": scale_before,
            "amp_scale_after_successful_update": scale_after_success,
            "amp_max_overflow_retries_per_step": R2_AMP_MAX_OVERFLOW_RETRIES_PER_STEP,
        }


def _verify_diagnostic_evidence(path: Path) -> None:
    if sha256_file(path) != R2_DIAGNOSTIC_V2_JSON_SHA256:
        raise RuntimeError("R2 diagnostic v2 JSON hash mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("R2 diagnostic v2 evidence must be a JSON object")
    if payload.get("status") != "COMPLETE":
        raise RuntimeError("R2 diagnostic v2 evidence is not COMPLETE")
    if payload.get("classification") != "AMP_BACKOFF_RECOVERY_SUPPORTED":
        raise RuntimeError("R2 diagnostic v2 does not justify AMP recovery primary")
    if payload.get("main_training_started") is not False:
        raise RuntimeError("R2 diagnostic v2 unexpectedly started main training")
    if payload.get("checkpoint_created") is not False:
        raise RuntimeError("R2 diagnostic v2 unexpectedly created a checkpoint")
    for field in (
        "validation_arrays_opened",
        "internal_test_arrays_opened",
        "external_arrays_opened",
    ):
        if int(payload.get(field, -1)) != 0:
            raise RuntimeError(f"R2 diagnostic v2 leakage boundary failed: {field}")


def _postprocess_output(output_dir: Path, *, corrective_config_sha256: str) -> None:
    steps_path = output_dir / "r2_primary_steps.jsonl"
    summary_path = output_dir / "r2_primary_summary.json"
    failure_path = output_dir / "r2_primary_failure.json"
    total_overflow_retries = 0
    recovered_steps = 0
    max_retries = 0
    if steps_path.is_file():
        for line in steps_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            retries = int(record.get("amp_overflow_retry_count", 0))
            total_overflow_retries += retries
            max_retries = max(max_retries, retries)
            if retries > 0:
                recovered_steps += 1

    metadata = {
        "corrective_primary_version": 2,
        "corrective_config_path": str(CORRECTIVE_CONFIG_RELATIVE_PATH),
        "corrective_config_sha256": corrective_config_sha256,
        "source_diagnostic_v2_sha256": R2_DIAGNOSTIC_V2_JSON_SHA256,
        "amp_overflow_policy": (
            "skip optimizer update, back off GradScaler, retry identical patch "
            "without consuming sampling RNG"
        ),
        "amp_total_overflow_retries": total_overflow_retries,
        "amp_steps_requiring_recovery": recovered_steps,
        "amp_max_overflow_retries_observed_single_step": max_retries,
    }
    target = summary_path if summary_path.is_file() else failure_path
    if not target.is_file():
        return
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {target}")
    payload.update(metadata)
    if target == summary_path:
        payload["schema_version"] = "research_v2_r2_corrective_primary.v2"
    else:
        payload["schema_version"] = "research_v2_r2_corrective_primary_failure.v2"
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


def main() -> int:
    diagnostic_evidence = _pop_required_cli_path("--diagnostic-evidence")
    output_dir = _peek_required_cli_path("--output-dir")
    repo_root = _peek_required_cli_path("--repo-root")
    corrective_config = repo_root / CORRECTIVE_CONFIG_RELATIVE_PATH
    _verify_diagnostic_evidence(diagnostic_evidence)
    corrective_config_sha256 = sha256_file(corrective_config)

    global primary
    primary = _load_primary_runtime()
    primary._optimizer_step = _recovering_optimizer_step
    result = int(primary.main())
    _postprocess_output(output_dir, corrective_config_sha256=corrective_config_sha256)
    return result


if __name__ == "__main__":
    sys.exit(main())
