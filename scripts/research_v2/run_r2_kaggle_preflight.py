#!/usr/bin/env python3
"""Run the real R2 post-transfer hash and single-GPU CUDA preflight on Kaggle.

The script is fail-closed. It accepts only the four preregistered train cases, verifies all eight
transferred NIfTI files by SHA-256 before decoding arrays, validates raw-label semantics, records
per-device VRAM, performs a single-GPU allocation probe, and executes exactly one transient
forward/backward/optimizer step on cuda:0. It does not run the preregistered main training budget.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from protoem_ct.artifacts import sha256_file
from protoem_ct.research_v2.r1_audit import (
    grid_for_spacing,
    reorient_array_to_ras,
    resample_array_to_grid,
)
from protoem_ct.research_v2.r2_preflight import (
    R2_ALLOCATION_PROBE_BYTES,
    R2_PATCH_SIZE,
    R2_SEED,
    R2_TARGET_SPACING_MM,
    choose_positive_center,
    extract_centered_patch,
    normalize_r2_ct,
    r2_binary_tumor_target,
)

EXPECTED_LOCK_SHA256 = "c9ee200234eebf8c67fd97ff2c96ca0765f4067a8a69a7ef439deafdda3530a8"
EXPECTED_CASES: dict[str, dict[str, Any]] = {
    "case_65917a1f3a47339a558042dac093bf1a": {
        "role": "positive_q20",
        "image_sha256": "fb2c353bc4cc35aedd9c8070883791c2241beb36bf7e5f884cd8a0f418226b9d",
        "label_sha256": "56282d27ae73a4ea808311dc54363fa344eee473c9af184f069ab94e97a70dae",
        "tumor_voxel_count": 2261,
    },
    "case_f232be7b649d2166e5158aa636b38766": {
        "role": "positive_q50",
        "image_sha256": "7efcb4f28d9cc19b2c306260b5b75f13a8ab772d309a211dbdb4a8ef8b30a124",
        "label_sha256": "b4946039c62db1b69c4d0bc810e4a3a51effdf4fe380b7b53def160cb3c6bc18",
        "tumor_voxel_count": 12265,
    },
    "case_6e4782e999691662f69fdf25f0e2e423": {
        "role": "positive_q80",
        "image_sha256": "7489d81f32ad48ce33e086cbc0105d65d1db4c9122c97af8d3ceef45eb9cfa47",
        "label_sha256": "3b12ff9755d6719311bcc74b825a45c6ce5ee56ef7a33e5c294289043dd58dbf",
        "tumor_voxel_count": 179093,
    },
    "case_25928d2451a7d948a79e60084e3c5d32": {
        "role": "empty_lexicographic_first",
        "image_sha256": "9f0fac0d26bfab882941e840a7a4eef6dab9d35d19bc64e1e79e7ec5738503bc",
        "label_sha256": "e5ec79c42475d308595a6108b3f48d231bfcba2d3692bbe773584f2d458d2fad",
        "tumor_voxel_count": 0,
    },
}
SMOKE_CASE_ID = "case_f232be7b649d2166e5158aa636b38766"


def _run_text_command(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {completed.returncode}: {' '.join(command)}"
        )
    return completed.stdout.strip()


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def _expected_paths(data_root: Path) -> dict[str, tuple[Path, Path]]:
    paths: dict[str, tuple[Path, Path]] = {}
    for case_id in EXPECTED_CASES:
        case_root = data_root / case_id
        paths[case_id] = (case_root / "image.nii.gz", case_root / "label.nii.gz")
    return paths


def _verify_transferred_files(data_root: Path) -> list[dict[str, Any]]:
    expected_paths = _expected_paths(data_root)
    expected_medical_paths = {
        path.resolve()
        for pair in expected_paths.values()
        for path in pair
    }
    observed_medical_paths = {
        path.resolve()
        for pattern in ("*.nii", "*.nii.gz")
        for path in data_root.rglob(pattern)
        if path.is_file()
    }
    extra = sorted(str(path.relative_to(data_root)) for path in observed_medical_paths - expected_medical_paths)
    missing = sorted(str(path.relative_to(data_root)) for path in expected_medical_paths - observed_medical_paths)
    if missing:
        raise RuntimeError(f"R2 private transfer is missing expected medical files: {missing}")
    if extra:
        raise RuntimeError(f"R2 private transfer contains extra medical files: {extra}")

    evidence: list[dict[str, Any]] = []
    for case_id, (image_path, label_path) in expected_paths.items():
        expected = EXPECTED_CASES[case_id]
        image_hash = sha256_file(image_path)
        label_hash = sha256_file(label_path)
        if image_hash != expected["image_sha256"]:
            raise RuntimeError(f"post-transfer image SHA-256 mismatch for {case_id}")
        if label_hash != expected["label_sha256"]:
            raise RuntimeError(f"post-transfer label SHA-256 mismatch for {case_id}")
        evidence.append(
            {
                "anonymous_case_id": case_id,
                "role": expected["role"],
                "image_sha256": image_hash,
                "label_sha256": label_hash,
                "post_transfer_hash_verified": True,
            }
        )
    return evidence


def _validate_selected_labels(data_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for case_id, expected in EXPECTED_CASES.items():
        label_path = data_root / case_id / "label.nii.gz"
        label_nifti = nib.load(str(label_path))
        label = np.asarray(label_nifti.dataobj)
        tumor = r2_binary_tumor_target(label)
        tumor_voxels = int(np.count_nonzero(tumor))
        if tumor_voxels != int(expected["tumor_voxel_count"]):
            raise RuntimeError(f"transferred label tumor voxel count changed for {case_id}")
        records.append(
            {
                "anonymous_case_id": case_id,
                "observed_raw_label_values": [int(value) for value in np.unique(label)],
                "tumor_voxel_count": tumor_voxels,
                "raw_label_two_mapping_verified": True,
            }
        )
    return records


def _prepare_smoke_patch(data_root: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    image_path = data_root / SMOKE_CASE_ID / "image.nii.gz"
    label_path = data_root / SMOKE_CASE_ID / "label.nii.gz"
    image_nifti = nib.load(str(image_path))
    label_nifti = nib.load(str(label_path))
    image = np.asarray(image_nifti.dataobj)
    label = np.asarray(label_nifti.dataobj)
    if image.ndim != 3 or label.ndim != 3:
        raise RuntimeError("R2 smoke image and label must both be exactly 3D")
    if image.shape != label.shape:
        raise RuntimeError("R2 smoke image and label shapes differ")
    if not bool(np.isfinite(image).all()):
        raise RuntimeError("R2 smoke image contains NaN or Inf")

    image_affine = np.asarray(image_nifti.affine, dtype=np.float64)
    label_affine = np.asarray(label_nifti.affine, dtype=np.float64)
    affine_max_abs_delta_mm = float(np.max(np.abs(image_affine - label_affine)))
    if affine_max_abs_delta_mm > 1e-4:
        raise RuntimeError("R2 smoke image/label effective affine delta exceeds 1e-4 mm")

    tumor = r2_binary_tumor_target(label)
    ras_image, ras_image_affine = reorient_array_to_ras(image, image_affine)
    ras_tumor, ras_label_affine = reorient_array_to_ras(tumor, label_affine)
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
    if not bool(np.any(resampled_tumor)):
        raise RuntimeError("R2 smoke positive case lost all tumor voxels after nearest resampling")

    normalized = normalize_r2_ct(resampled_image)
    center = choose_positive_center(resampled_tumor, seed=R2_SEED)
    image_patch = extract_centered_patch(
        normalized,
        center=center,
        patch_size=R2_PATCH_SIZE,
        pad_value=-1.0,
    ).astype(np.float32, copy=False)
    target_patch = extract_centered_patch(
        resampled_tumor,
        center=center,
        patch_size=R2_PATCH_SIZE,
        pad_value=False,
    ).astype(np.int64, copy=False)
    tumor_voxels_in_patch = int(np.count_nonzero(target_patch))
    if tumor_voxels_in_patch <= 0:
        raise RuntimeError("R2 preflight positive patch does not contain tumor voxels")
    metadata = {
        "anonymous_case_id": SMOKE_CASE_ID,
        "source_shape": [int(value) for value in image.shape],
        "resampled_shape": [int(value) for value in target_shape],
        "patch_size": list(R2_PATCH_SIZE),
        "positive_center_resampled": list(center),
        "tumor_voxels_in_patch": tumor_voxels_in_patch,
        "image_label_effective_affine_max_abs_delta_mm": affine_max_abs_delta_mm,
        "image_interpolation": "continuous_order_1",
        "label_interpolation": "nearest_order_0",
        "augmentation": "disabled",
    }
    return image_patch, target_patch, metadata


def _gpu_runtime_evidence(torch: Any) -> dict[str, Any]:
    if not bool(torch.cuda.is_available()):
        raise RuntimeError("R2 requires CUDA; CPU fallback is forbidden")
    device_count = int(torch.cuda.device_count())
    if device_count < 1:
        raise RuntimeError("R2 requires at least one visible CUDA device")
    devices: list[dict[str, Any]] = []
    for index in range(device_count):
        properties = torch.cuda.get_device_properties(index)
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        devices.append(
            {
                "index": index,
                "name": str(properties.name),
                "total_memory_bytes": int(properties.total_memory),
                "total_memory_gib": float(properties.total_memory / (1024**3)),
                "free_memory_bytes_at_preflight": int(free_bytes),
                "torch_mem_get_info_total_bytes": int(total_bytes),
                "compute_capability": [int(properties.major), int(properties.minor)],
            }
        )
    return {
        "cuda_available": True,
        "visible_device_count": device_count,
        "training_device": "cuda:0",
        "cpu_fallback_allowed": False,
        "multi_gpu_memory_pooling_assumed": False,
        "devices": devices,
    }


def _allocation_probe(torch: Any) -> dict[str, Any]:
    device = torch.device("cuda:0")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    before_free, before_total = torch.cuda.mem_get_info(device)
    element_count = R2_ALLOCATION_PROBE_BYTES // 4
    probe = torch.empty(element_count, dtype=torch.float32, device=device)
    probe.zero_()
    torch.cuda.synchronize(device)
    allocated = int(torch.cuda.memory_allocated(device))
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    del probe
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    after_free, after_total = torch.cuda.mem_get_info(device)
    return {
        "requested_bytes": R2_ALLOCATION_PROBE_BYTES,
        "before_free_bytes": int(before_free),
        "before_total_bytes": int(before_total),
        "allocated_bytes_during_probe": allocated,
        "peak_allocated_bytes": peak_allocated,
        "after_free_bytes": int(after_free),
        "after_total_bytes": int(after_total),
        "status": "PASS",
    }


def _single_optimizer_step(
    *,
    torch: Any,
    monai: Any,
    image_patch: np.ndarray,
    target_patch: np.ndarray,
) -> dict[str, Any]:
    device = torch.device("cuda:0")
    torch.manual_seed(R2_SEED)
    torch.cuda.manual_seed_all(R2_SEED)
    model = monai.networks.nets.SegResNet(
        spatial_dims=3,
        init_filters=8,
        in_channels=1,
        out_channels=2,
        dropout_prob=None,
        blocks_down=(1, 1, 1),
        blocks_up=(1, 1),
        upsample_mode="deconv",
    ).to(device)
    if {parameter.device.type for parameter in model.parameters()} != {"cuda"}:
        raise RuntimeError("R2 model parameters are not exclusively on CUDA")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_function = monai.losses.DiceCELoss(
        include_background=False,
        to_onehot_y=True,
        softmax=True,
        lambda_dice=1.0,
        lambda_ce=1.0,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    input_tensor = torch.from_numpy(image_patch[None, None]).to(device=device, dtype=torch.float32)
    target_tensor = torch.from_numpy(target_patch[None, None]).to(device=device, dtype=torch.long)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
        logits = model(input_tensor)
        loss = loss_function(logits, target_tensor)
    if not bool(torch.isfinite(logits).all().item()):
        raise RuntimeError("R2 preflight logits contain NaN or Inf")
    if not bool(torch.isfinite(loss).item()):
        raise RuntimeError("R2 preflight loss is NaN or Inf")

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gradient_tensor_count = 0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        gradient_tensor_count += 1
        if not bool(torch.isfinite(parameter.grad).all().item()):
            raise RuntimeError("R2 preflight gradient contains NaN or Inf")
    if gradient_tensor_count == 0:
        raise RuntimeError("R2 preflight produced no gradients")
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    with torch.no_grad():
        probabilities = torch.softmax(logits.float(), dim=1)[:, 1]
    if not bool(torch.isfinite(probabilities).all().item()):
        raise RuntimeError("R2 preflight probabilities contain NaN or Inf")
    return {
        "status": "PASS",
        "device": str(device),
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.00001,
        "loss": "MONAI DiceCELoss",
        "amp_fp16": True,
        "loss_value": float(loss.detach().float().cpu().item()),
        "finite_logits": True,
        "finite_loss": True,
        "finite_gradients": True,
        "gradient_tensor_count": gradient_tensor_count,
        "probability_min": float(probabilities.min().cpu().item()),
        "probability_max": float(probabilities.max().cpu().item()),
        "probability_mean": float(probabilities.mean().cpu().item()),
        "elapsed_seconds": float(elapsed),
        "peak_vram_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_vram_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "transient_preflight_step_count": 1,
        "main_training_started": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    repo_root = args.repo_root.expanduser().resolve()
    if not data_root.is_dir():
        raise RuntimeError(f"R2 private data root does not exist: {data_root}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise RuntimeError(f"refusing to overwrite R2 preflight evidence: {output_path}")

    lock_path = repo_root / "environments/research-v2-linux-cuda/uv.lock"
    lock_hash = sha256_file(lock_path)
    if lock_hash != EXPECTED_LOCK_SHA256:
        raise RuntimeError("R2 Linux CUDA uv.lock SHA-256 differs from the R0 lock")

    transfer_evidence = _verify_transferred_files(data_root)
    label_evidence = _validate_selected_labels(data_root)
    image_patch, target_patch, patch_evidence = _prepare_smoke_patch(data_root)

    torch = importlib.import_module("torch")
    monai = importlib.import_module("monai")
    gpu_evidence = _gpu_runtime_evidence(torch)
    allocation_evidence = _allocation_probe(torch)
    step_evidence = _single_optimizer_step(
        torch=torch,
        monai=monai,
        image_patch=image_patch,
        target_patch=target_patch,
    )

    driver_versions = sorted(
        {
            line.strip()
            for line in _run_text_command(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader",
                ]
            ).splitlines()
            if line.strip()
        }
    )
    evidence = {
        "schema_version": "research_v2_r2_kaggle_preflight.v1",
        "status": "PASS",
        "main_training_started": False,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "uv_version": _run_text_command(["uv", "--version"]),
            "torch": str(torch.__version__),
            "torch_cuda_runtime": str(torch.version.cuda),
            "cudnn_version": int(torch.backends.cudnn.version()),
            "monai": str(monai.__version__),
            "nnunetv2": _package_version("nnunetv2"),
            "nvidia_driver_versions": driver_versions,
            "uv_lock_sha256": lock_hash,
            "uv_lock_sha256_verified": True,
        },
        "gpu": gpu_evidence,
        "post_transfer_hashes": transfer_evidence,
        "selected_label_contract": label_evidence,
        "smoke_patch": patch_evidence,
        "allocation_probe": allocation_evidence,
        "forward_backward_optimizer_step": step_evidence,
        "array_access": {
            "selected_train_label_arrays_opened": 4,
            "selected_train_image_arrays_opened": 1,
            "validation_arrays_opened": 0,
            "internal_test_arrays_opened": 0,
            "external_arrays_opened": 0,
        },
        "medical_images_in_evidence": False,
        "raw_labels_in_evidence": False,
        "checkpoint_created": False,
        "interpretation": (
            "Real post-transfer train-only hash and CUDA preflight evidence. This is not the R2 "
            "main training run and is not evidence of unseen-patient performance."
        ),
    }
    output_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(evidence, indent=2, sort_keys=True))
    print(f"R2_KAGGLE_PREFLIGHT_EVIDENCE={output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
