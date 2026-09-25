#!/usr/bin/env python3
"""Verify the final R2 checkpoint in a fresh process against private native probabilities."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ["MPLBACKEND"] = "Agg"

import nibabel as nib
import numpy as np

from protoem_ct.artifacts import sha256_file
from protoem_ct.research_v2.r1_audit import grid_for_spacing, reorient_array_to_ras, resample_array_to_grid
from protoem_ct.research_v2.r2_overfit import (
    R2_RELOAD_PROBABILITY_ATOL,
    R2_SELECTED_CASES,
    restore_probability_to_native,
    threshold_probability,
)
from protoem_ct.research_v2.r2_preflight import (
    R2_PATCH_SIZE,
    R2_TARGET_SPACING_MM,
    normalize_r2_ct,
    r2_binary_tumor_target,
)

EXPECTED_LOCK_SHA256 = "c9ee200234eebf8c67fd97ff2c96ca0765f4067a8a69a7ef439deafdda3530a8"
EXPECTED_CONFIG_SHA256 = "dfe50f52110bd77861f52fc4683712e0549a1f0f77c363f64f09da05940f3d1a"
LOCK_RELATIVE_PATH = Path("environments/research-v2-linux-cuda/uv.lock")


@dataclass(slots=True)
class ReloadCase:
    anonymous_case_id: str
    image: Any
    resampled_affine: Any
    native_affine: Any
    native_shape: tuple[int, int, int]


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _prepare_case(data_root: Path, spec: Any) -> ReloadCase:
    root = data_root / spec.anonymous_case_id
    image_path = root / "image.nii.gz"
    label_path = root / "label.nii.gz"
    if sha256_file(image_path) != spec.image_sha256:
        raise RuntimeError(f"R2 reload image hash mismatch: {spec.anonymous_case_id}")
    if sha256_file(label_path) != spec.label_sha256:
        raise RuntimeError(f"R2 reload label hash mismatch: {spec.anonymous_case_id}")
    image_nifti = nib.load(str(image_path))
    label_nifti = nib.load(str(label_path))
    image = np.asarray(image_nifti.dataobj)
    label = np.asarray(label_nifti.dataobj)
    if image.shape != label.shape or image.ndim != 3:
        raise RuntimeError(f"R2 reload source shape mismatch: {spec.anonymous_case_id}")
    image_affine = np.asarray(image_nifti.affine, dtype=np.float64)
    label_affine = np.asarray(label_nifti.affine, dtype=np.float64)
    if float(np.max(np.abs(image_affine - label_affine))) > 1e-4:
        raise RuntimeError(f"R2 reload source affine mismatch: {spec.anonymous_case_id}")
    native_tumor = r2_binary_tumor_target(label)
    if int(np.count_nonzero(native_tumor)) != int(spec.tumor_voxel_count):
        raise RuntimeError(f"R2 reload tumor voxel mismatch: {spec.anonymous_case_id}")
    ras_image, ras_image_affine = reorient_array_to_ras(image, image_affine)
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
    return ReloadCase(
        anonymous_case_id=spec.anonymous_case_id,
        image=normalize_r2_ct(image_resampled),
        resampled_affine=np.asarray(target_affine, dtype=np.float64),
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


def _infer_native_probability(
    *,
    torch: Any,
    monai: Any,
    model: Any,
    case: ReloadCase,
    device: Any,
) -> np.ndarray:
    model.eval()
    input_tensor = torch.from_numpy(case.image[None, None]).to(dtype=torch.float32)

    def predictor(window: Any) -> Any:
        logits = model(window)
        if not bool(torch.isfinite(logits).all().item()):
            raise RuntimeError("R2 reload produced non-finite logits")
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
    probability_resampled = np.asarray(probability_tensor[0, 0].numpy(), dtype=np.float32)
    return np.asarray(
        restore_probability_to_native(
            probability_resampled,
            resampled_affine=case.resampled_affine,
            native_shape=case.native_shape,
            native_affine=case.native_affine,
        ),
        dtype=np.float32,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--primary-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    repo_root = args.repo_root.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    primary_dir = args.primary_dir.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    started = time.perf_counter()
    try:
        if output_path.exists():
            raise RuntimeError(f"refusing to overwrite R2 reload evidence: {output_path}")
        if sha256_file(repo_root / LOCK_RELATIVE_PATH) != EXPECTED_LOCK_SHA256:
            raise RuntimeError("R2 reload uv.lock differs from frozen R0 lock")
        summary_path = primary_dir / "r2_primary_summary.json"
        checkpoint_path = primary_dir / "r2_primary_checkpoint.pt"
        reference_dir = primary_dir / "private_reference_probabilities"
        summary = _load_json(summary_path)
        if summary.get("main_training_completed") is not True:
            raise RuntimeError("R2 primary did not complete")
        if summary.get("config_sha256") != EXPECTED_CONFIG_SHA256:
            raise RuntimeError("R2 primary config hash differs from locked contract")
        expected_checkpoint_hash = str(summary.get("checkpoint_sha256", ""))
        if sha256_file(checkpoint_path) != expected_checkpoint_hash:
            raise RuntimeError("R2 checkpoint SHA-256 differs from primary summary")

        torch = importlib.import_module("torch")
        monai = importlib.import_module("monai")
        if not bool(torch.cuda.is_available()) or int(torch.cuda.device_count()) != 2:
            raise RuntimeError("R2 reload requires the registered CUDA environment")
        device = torch.device("cuda:0")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        model = _build_model(monai, device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if checkpoint.get("schema_version") != "research_v2_r2_checkpoint.v1":
            raise RuntimeError("R2 checkpoint schema changed")
        if int(checkpoint.get("step", -1)) != 2500:
            raise RuntimeError("R2 checkpoint is not the final primary step")
        if checkpoint.get("config_sha256") != EXPECTED_CONFIG_SHA256:
            raise RuntimeError("R2 checkpoint config hash changed")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)

        cases = [_prepare_case(data_root, spec) for spec in R2_SELECTED_CASES]
        records: list[dict[str, Any]] = []
        for case in cases:
            reference_path = reference_dir / f"{case.anonymous_case_id}.native_probability.npy"
            if not reference_path.is_file():
                raise RuntimeError(f"R2 reload reference missing: {case.anonymous_case_id}")
            reference = np.load(reference_path, allow_pickle=False)
            reference_array = np.asarray(reference, dtype=np.float32)
            observed = _infer_native_probability(
                torch=torch,
                monai=monai,
                model=model,
                case=case,
                device=device,
            )
            if reference_array.shape != observed.shape:
                raise RuntimeError(f"R2 reload shape mismatch: {case.anonymous_case_id}")
            max_abs_difference = float(np.max(np.abs(reference_array - observed), initial=0.0))
            probability_match = max_abs_difference <= R2_RELOAD_PROBABILITY_ATOL
            binary_match = bool(
                np.array_equal(
                    threshold_probability(reference_array),
                    threshold_probability(observed),
                )
            )
            records.append(
                {
                    "anonymous_case_id": case.anonymous_case_id,
                    "reference_probability_sha256": sha256_file(reference_path),
                    "max_abs_probability_difference": max_abs_difference,
                    "probability_tolerance": R2_RELOAD_PROBABILITY_ATOL,
                    "probability_match": probability_match,
                    "binary_prediction_exact_match": binary_match,
                }
            )
            if not probability_match or not binary_match:
                raise RuntimeError(f"R2 checkpoint reload mismatch: {case.anonymous_case_id}")

        evidence = {
            "schema_version": "research_v2_r2_checkpoint_reload.v1",
            "status": "PASS",
            "process_id": os.getpid(),
            "fresh_process_required": True,
            "optimizer_steps_performed": 0,
            "checkpoint_sha256": expected_checkpoint_hash,
            "reload_probability_max_abs_tolerance": R2_RELOAD_PROBABILITY_ATOL,
            "reload_binary_prediction_exact_match_required": True,
            "case_records": records,
            "validation_arrays_opened": 0,
            "internal_test_arrays_opened": 0,
            "external_arrays_opened": 0,
            "elapsed_seconds": float(time.perf_counter() - started),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(output_path, evidence)
        print(json.dumps(evidence, indent=2, sort_keys=True))
        return 0
    except (RuntimeError, ValueError) as exc:
        failure = {
            "schema_version": "research_v2_r2_checkpoint_reload_failure.v1",
            "status": "FAILED",
            "process_id": os.getpid(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "optimizer_steps_performed": 0,
            "validation_arrays_opened": 0,
            "internal_test_arrays_opened": 0,
            "external_arrays_opened": 0,
            "elapsed_seconds": float(time.perf_counter() - started),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(output_path.with_name("r2_checkpoint_reload_failure.json"), failure)
        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)
        return 40


if __name__ == "__main__":
    sys.exit(main())
