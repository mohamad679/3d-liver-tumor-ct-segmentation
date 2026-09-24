#!/usr/bin/env python3
"""Run the real Research-v2 R1 audit on a Mac with the Lexar development data attached.

The runner is intentionally read-only with respect to manifest, split, images, and labels. It
opens arrays only for cases authorized as train/validation by the locked split. Internal-test and
external arrays are never opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import time
import zlib
from collections import Counter
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from scipy import ndimage

from protoem_ct.artifacts import (
    DatasetManifest,
    DevelopmentSplitManifest,
    hash_dataset_manifest,
    hash_development_split,
    phase2_artifact_from_json,
    sha256_file,
)
from protoem_ct.data.phase2_paths import resolve_regular_file_beneath_root
from protoem_ct.research_v2.r1_audit import (
    R1_AFFINE_ABS_TOLERANCE,
    R1_ALLOWED_RAW_LABEL_VALUES,
    R1_EXPECTED_TRAIN_CASES,
    R1_EXPECTED_VALIDATION_CASES,
    R1_TUMOR_RAW_LABEL_VALUE,
    affine_orientation,
    affine_spacing_mm,
    authorized_r1_cases,
    grid_for_spacing,
    reorient_array_to_ras,
    resample_array_to_grid,
    restore_crop_to_full_grid,
    validate_raw_lits_label_array,
)

EXPECTED_MANIFEST_HASH = "c24244951e050050cf25c4b321f67d61c2087fc0c93fdcf9d112e0e488e1384b"
EXPECTED_MANIFEST_FILE_SHA256 = (
    "0e21a7d555e20b6011091bc18e462bc150cc23a8f46e522dbd8c01b014a44ae3"
)
EXPECTED_SPLIT_HASH = "936376cd7b5e6070397c2fef16e5125c60fd6569ff3188d7e9bb5428a46ffadb"
EXPECTED_SPLIT_FILE_SHA256 = "416ca83e85c8598fc4f7065316153193211bf6b57875fbda4cafc01c360445f7"
DEFAULT_MANIFEST = Path("/Volumes/Lexar/ProtoEM-CT/runs/phase2_real_lits_v2/manifest.json")
DEFAULT_SPLIT = Path("/Volumes/Lexar/ProtoEM-CT/runs/phase2_real_lits_v2/split.json")
DEFAULT_SEARCH_ROOT = Path("/Volumes/Lexar/ProtoEM-CT")
OVERLAY_COUNT = 10
VISUAL_WINDOW_HU = (-200.0, 300.0)


def _jsonable_matrix(matrix: np.ndarray | None) -> list[list[float]] | None:
    if matrix is None:
        return None
    return [[float(value) for value in row] for row in np.asarray(matrix)]


def _load_locked_artifacts(
    manifest_path: Path, split_path: Path
) -> tuple[DatasetManifest, DevelopmentSplitManifest]:
    if sha256_file(manifest_path) != EXPECTED_MANIFEST_FILE_SHA256:
        raise RuntimeError("manifest raw SHA-256 does not match the R0 lock")
    if sha256_file(split_path) != EXPECTED_SPLIT_FILE_SHA256:
        raise RuntimeError("split raw SHA-256 does not match the R0 lock")
    manifest = phase2_artifact_from_json(manifest_path.read_text(encoding="utf-8"))
    split = phase2_artifact_from_json(split_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, DatasetManifest):
        raise RuntimeError("locked manifest did not parse as DatasetManifest")
    if not isinstance(split, DevelopmentSplitManifest):
        raise RuntimeError("locked split did not parse as DevelopmentSplitManifest")
    if manifest.manifest_hash != EXPECTED_MANIFEST_HASH:
        raise RuntimeError("manifest logical artifact hash does not match the R0 lock")
    if split.split_hash != EXPECTED_SPLIT_HASH:
        raise RuntimeError("split logical artifact hash does not match the R0 lock")
    if hash_dataset_manifest(manifest) != EXPECTED_MANIFEST_HASH:
        raise RuntimeError("independent manifest logical-hash recomputation failed")
    if hash_development_split(split) != EXPECTED_SPLIT_HASH:
        raise RuntimeError("independent split logical-hash recomputation failed")
    if split.source_manifest_hash != manifest.manifest_hash:
        raise RuntimeError("split source_manifest_hash does not match manifest")
    return manifest, split


def _candidate_root_for_match(match: Path, relative_path: str) -> Path:
    candidate = match
    for _ in Path(relative_path).parts:
        candidate = candidate.parent
    return candidate


def _detect_dataset_root(search_root: Path, relative_path: str) -> Path:
    if not search_root.is_dir():
        raise RuntimeError(f"dataset search root does not exist: {search_root}")
    basename = Path(relative_path).name
    candidates: set[Path] = set()
    for match in search_root.rglob(basename):
        if not match.is_file():
            continue
        candidate = _candidate_root_for_match(match, relative_path)
        try:
            expected = (candidate / relative_path).resolve(strict=True)
            resolved_match = match.resolve(strict=True)
        except OSError:
            continue
        if expected == resolved_match:
            candidates.add(candidate.resolve())
    if len(candidates) != 1:
        formatted = [str(path) for path in sorted(candidates)]
        raise RuntimeError(
            "could not uniquely auto-detect dataset root from one authorized train path; "
            f"candidates={formatted}. Re-run with --dataset-root."
        )
    return next(iter(candidates))


def _nifti_arrays(path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    image = nib.load(str(path))
    array = np.asanyarray(image.dataobj)
    return image, np.asarray(array)


def _form_record(image: nib.Nifti1Image, prefix: str) -> dict[str, Any]:
    qform, qcode = image.get_qform(coded=True)
    sform, scode = image.get_sform(coded=True)
    return {
        f"{prefix}_qform_code": int(qcode),
        f"{prefix}_qform": _jsonable_matrix(qform),
        f"{prefix}_sform_code": int(scode),
        f"{prefix}_sform": _jsonable_matrix(sform),
    }


def _forms_conflict(image: nib.Nifti1Image) -> bool:
    qform, qcode = image.get_qform(coded=True)
    sform, scode = image.get_sform(coded=True)
    if int(qcode) <= 0 or int(scode) <= 0 or qform is None or sform is None:
        return False
    return not bool(np.allclose(qform, sform, rtol=0.0, atol=R1_AFFINE_ABS_TOLERANCE))


def _audit_case(case: Any, *, dataset_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    mismatches: list[dict[str, Any]] = []
    image_path = resolve_regular_file_beneath_root(dataset_root, case.relative_image_path)
    label_path = resolve_regular_file_beneath_root(dataset_root, case.relative_label_path)
    if sha256_file(image_path) != case.image_sha256:
        raise RuntimeError("source image SHA-256 changed from locked manifest")
    if sha256_file(label_path) != case.label_sha256:
        raise RuntimeError("source label SHA-256 changed from locked manifest")

    image_nifti, image = _nifti_arrays(image_path)
    label_nifti, label = _nifti_arrays(label_path)
    image_affine = np.asarray(image_nifti.affine, dtype=np.float64)
    label_affine = np.asarray(label_nifti.affine, dtype=np.float64)

    def mismatch(reason: str) -> None:
        mismatches.append(
            {
                "anonymous_case_id": case.anonymous_case_id,
                "partition": case.partition,
                "reason": reason,
                "resolution_status": "unresolved",
                "resolution": None,
            }
        )

    if image.ndim != 3:
        mismatch("image_not_3d")
    if label.ndim != 3:
        mismatch("label_not_3d")
    image_finite = bool(np.isfinite(image).all())
    label_finite = bool(np.isfinite(label).all())
    if not image_finite:
        mismatch("image_nonfinite")
    if not label_finite:
        mismatch("label_nonfinite")
    if tuple(image.shape) != tuple(label.shape):
        mismatch("image_label_shape_mismatch")
    affine_match = bool(
        np.allclose(
            image_affine,
            label_affine,
            rtol=0.0,
            atol=R1_AFFINE_ABS_TOLERANCE,
        )
    )
    if not affine_match:
        mismatch("image_label_affine_mismatch")
    if _forms_conflict(image_nifti):
        mismatch("image_qform_sform_conflict")
    if _forms_conflict(label_nifti):
        mismatch("label_qform_sform_conflict")

    tumor = None
    observed_values: list[int | float] = []
    if label.ndim == 3 and label_finite:
        observed_values = [
            int(value) if float(value).is_integer() else float(value)
            for value in np.unique(label)
        ]
        try:
            tumor = validate_raw_lits_label_array(label)
        except ValueError as exc:
            mismatch(f"label_contract:{exc}")

    try:
        image_spacing = affine_spacing_mm(image_affine)
        label_spacing = affine_spacing_mm(label_affine)
        image_orientation = affine_orientation(image_affine)
        label_orientation = affine_orientation(label_affine)
    except ValueError as exc:
        mismatch(f"geometry_contract:{exc}")
        image_spacing = (float("nan"),) * 3
        label_spacing = (float("nan"),) * 3
        image_orientation = ("?", "?", "?")
        label_orientation = ("?", "?", "?")

    if (
        all(np.isfinite(image_spacing))
        and all(np.isfinite(label_spacing))
        and not np.allclose(image_spacing, label_spacing, rtol=0.0, atol=1e-6)
    ):
        mismatch("image_label_spacing_mismatch")

    lesion_count = 0
    tumor_voxels = 0
    tumor_volume_mm3 = 0.0
    if tumor is not None:
        _, lesion_count = ndimage.label(tumor, structure=np.ones((3, 3, 3), dtype=np.uint8))
        tumor_voxels = int(np.count_nonzero(tumor))
        if all(np.isfinite(image_spacing)):
            tumor_volume_mm3 = float(tumor_voxels * np.prod(np.asarray(image_spacing)))

    finite_image_values = image[np.isfinite(image)] if image_finite else np.array([], dtype=float)
    if finite_image_values.size:
        image_stats = {
            "min": float(np.min(finite_image_values)),
            "p01": float(np.percentile(finite_image_values, 1)),
            "median": float(np.median(finite_image_values)),
            "p99": float(np.percentile(finite_image_values, 99)),
            "max": float(np.max(finite_image_values)),
            "mean": float(np.mean(finite_image_values)),
            "std": float(np.std(finite_image_values)),
        }
    else:
        image_stats = None

    record: dict[str, Any] = {
        "anonymous_patient_id": case.anonymous_patient_id,
        "anonymous_case_id": case.anonymous_case_id,
        "partition": case.partition,
        "image_shape": [int(v) for v in image.shape],
        "label_shape": [int(v) for v in label.shape],
        "image_dtype": str(image_nifti.get_data_dtype()),
        "label_dtype": str(label_nifti.get_data_dtype()),
        "image_affine": _jsonable_matrix(image_affine),
        "label_affine": _jsonable_matrix(label_affine),
        "image_orientation": list(image_orientation),
        "label_orientation": list(label_orientation),
        "image_spacing_mm": [float(v) for v in image_spacing],
        "label_spacing_mm": [float(v) for v in label_spacing],
        "image_finite": image_finite,
        "label_finite": label_finite,
        "image_label_shape_match": tuple(image.shape) == tuple(label.shape),
        "image_label_affine_match": affine_match,
        "observed_label_values": observed_values,
        "allowed_label_values": list(R1_ALLOWED_RAW_LABEL_VALUES),
        "tumor_raw_label_value": R1_TUMOR_RAW_LABEL_VALUE,
        "tumor_voxel_count": tumor_voxels,
        "tumor_lesion_count_26c": int(lesion_count),
        "tumor_volume_mm3": tumor_volume_mm3,
        "image_intensity_stats": image_stats,
        "qa_passed": not mismatches,
    }
    record.update(_form_record(image_nifti, "image"))
    record.update(_form_record(label_nifti, "label"))
    return record, mismatches


def _ranked_quantile_indices(length: int, count: int) -> list[int]:
    if length < count:
        raise RuntimeError(f"need at least {count} successful train cases for overlay selection")
    return sorted({int(round(index * (length - 1) / (count - 1))) for index in range(count)})


def _select_diverse_train(records: list[dict[str, Any]], count: int) -> list[str]:
    train = [record for record in records if record["partition"] == "train" and record["qa_passed"]]
    if len(train) < count:
        raise RuntimeError("fewer than 10 QA-passing train cases are available for overlays")
    features = (
        "tumor_volume_mm3",
        "tumor_lesion_count_26c",
        "tumor_voxel_count",
    )
    selected: list[str] = []
    per_feature = max(3, count // len(features))
    for feature in features:
        ordered = sorted(train, key=lambda item: (float(item[feature]), item["anonymous_case_id"]))
        for index in _ranked_quantile_indices(len(ordered), per_feature):
            case_id = str(ordered[index]["anonymous_case_id"])
            if case_id not in selected:
                selected.append(case_id)
    spacing_order = sorted(
        train,
        key=lambda item: (
            float(np.prod(np.asarray(item["image_spacing_mm"], dtype=float))),
            item["anonymous_case_id"],
        ),
    )
    for index in _ranked_quantile_indices(len(spacing_order), count):
        case_id = str(spacing_order[index]["anonymous_case_id"])
        if case_id not in selected:
            selected.append(case_id)
        if len(selected) == count:
            break
    if len(selected) < count:
        for record in sorted(train, key=lambda item: item["anonymous_case_id"]):
            case_id = str(record["anonymous_case_id"])
            if case_id not in selected:
                selected.append(case_id)
            if len(selected) == count:
                break
    return selected[:count]


def _crop_bounds(mask: np.ndarray) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    shape = tuple(int(v) for v in mask.shape)
    coords = np.argwhere(mask)
    if coords.size == 0:
        center = tuple(value // 2 for value in shape)
        starts = tuple(max(0, center[i] - min(16, shape[i] // 2)) for i in range(3))
        stops = tuple(min(shape[i], starts[i] + min(32, shape[i])) for i in range(3))
        return starts, stops
    low = coords.min(axis=0)
    high = coords.max(axis=0) + 1
    starts = tuple(max(0, int(low[i]) - 8) for i in range(3))
    stops = tuple(min(shape[i], int(high[i]) + 8) for i in range(3))
    return starts, stops


def _roundtrip_trace(
    *,
    image: np.ndarray,
    tumor: np.ndarray,
    native_affine: np.ndarray,
    target_spacing: tuple[float, ...],
) -> tuple[dict[str, Any], np.ndarray]:
    ras_image, ras_affine = reorient_array_to_ras(image, native_affine)
    ras_tumor, ras_tumor_affine = reorient_array_to_ras(tumor.astype(np.uint8), native_affine)
    if not np.allclose(ras_affine, ras_tumor_affine, rtol=0.0, atol=1e-9):
        raise RuntimeError("image/mask RAS reorientation produced different affines")
    pure_orientation_restored = resample_array_to_grid(
        ras_tumor,
        source_affine=ras_affine,
        target_shape=tumor.shape,
        target_affine=native_affine,
        is_label_or_mask=True,
    ).astype(bool)
    pure_orientation_exact = bool(np.array_equal(pure_orientation_restored, tumor))

    target_shape, target_affine = grid_for_spacing(
        shape=ras_image.shape,
        affine=ras_affine,
        target_spacing_mm=target_spacing,
    )
    resampled_image = resample_array_to_grid(
        ras_image,
        source_affine=ras_affine,
        target_shape=target_shape,
        target_affine=target_affine,
        is_label_or_mask=False,
    )
    resampled_tumor = resample_array_to_grid(
        ras_tumor,
        source_affine=ras_affine,
        target_shape=target_shape,
        target_affine=target_affine,
        is_label_or_mask=True,
    ).astype(bool)
    starts, stops = _crop_bounds(resampled_tumor)
    crop = resampled_tumor[
        starts[0] : stops[0],
        starts[1] : stops[1],
        starts[2] : stops[2],
    ]
    restored_preprocessed = restore_crop_to_full_grid(
        crop,
        full_shape=resampled_tumor.shape,
        starts=starts,
    ).astype(bool)
    crop_restore_exact = bool(np.array_equal(restored_preprocessed, resampled_tumor))
    restored_native = resample_array_to_grid(
        restored_preprocessed.astype(np.uint8),
        source_affine=target_affine,
        target_shape=tumor.shape,
        target_affine=native_affine,
        is_label_or_mask=True,
    ).astype(bool)
    gt_voxels = int(np.count_nonzero(tumor))
    pred_voxels = int(np.count_nonzero(restored_native))
    intersection = int(np.count_nonzero(tumor & restored_native))
    denominator = gt_voxels + pred_voxels
    roundtrip_dice = 1.0 if denominator == 0 else 2.0 * intersection / denominator
    trace = {
        "geometry_probe_prediction_source": "ground_truth_tumor_geometry_only_not_model_output",
        "native_shape": [int(v) for v in tumor.shape],
        "native_orientation": list(affine_orientation(native_affine)),
        "native_spacing_mm": [float(v) for v in affine_spacing_mm(native_affine)],
        "ras_shape": [int(v) for v in ras_tumor.shape],
        "ras_orientation": list(affine_orientation(ras_affine)),
        "target_spacing_mm_train_median": [float(v) for v in target_spacing],
        "resampled_shape": [int(v) for v in resampled_tumor.shape],
        "resampled_label_values": [int(v) for v in np.unique(resampled_tumor)],
        "crop_starts": list(starts),
        "crop_stops": list(stops),
        "crop_restore_exact": crop_restore_exact,
        "pure_orientation_inverse_exact": pure_orientation_exact,
        "restored_native_shape_match": tuple(restored_native.shape) == tuple(tumor.shape),
        "restored_native_affine_is_requested_native_affine": True,
        "nearest_neighbor_used_for_mask": True,
        "continuous_linear_used_for_image": True,
        "roundtrip_geometry_probe_dice_not_performance": float(roundtrip_dice),
        "resampled_image_finite": bool(np.isfinite(resampled_image).all()),
    }
    return trace, restored_native


def _normalize_ct_slice(image_slice: np.ndarray) -> np.ndarray:
    low, high = VISUAL_WINDOW_HU
    clipped = np.clip(np.asarray(image_slice, dtype=np.float32), low, high)
    scaled = (clipped - low) / (high - low)
    return np.asarray(np.rint(scaled * 255.0), dtype=np.uint8)


def _overlay_panel(image_slice: np.ndarray, mask_slice: np.ndarray) -> np.ndarray:
    gray = _normalize_ct_slice(image_slice)
    rgb = np.stack((gray, gray, gray), axis=-1)
    mask = np.asarray(mask_slice, dtype=bool)
    rgb[mask, 0] = 255
    rgb[mask, 1] = (rgb[mask, 1].astype(np.float32) * 0.35).astype(np.uint8)
    rgb[mask, 2] = (rgb[mask, 2].astype(np.float32) * 0.35).astype(np.uint8)
    return np.rot90(rgb)


def _pad_panel(panel: np.ndarray, height: int, width: int) -> np.ndarray:
    output = np.zeros((height, width, 3), dtype=np.uint8)
    y = (height - panel.shape[0]) // 2
    x = (width - panel.shape[1]) // 2
    output[y : y + panel.shape[0], x : x + panel.shape[1]] = panel
    return output


def _png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    body = chunk_type + payload
    return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)


def _write_rgb_png(path: Path, rgb: np.ndarray) -> None:
    array = np.ascontiguousarray(rgb, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] != 3:
        raise RuntimeError("PNG writer requires HxWx3 uint8 RGB")
    height, width, _ = array.shape
    raw = b"".join(b"\x00" + array[row].tobytes() for row in range(height))
    payload = b"\x89PNG\r\n\x1a\n"
    payload += _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    payload += _png_chunk(b"IDAT", zlib.compress(raw, level=9))
    payload += _png_chunk(b"IEND", b"")
    path.write_bytes(payload)


def _write_triplanar_overlay(path: Path, image: np.ndarray, tumor: np.ndarray) -> dict[str, Any]:
    coords = np.argwhere(tumor)
    if coords.size:
        center = tuple(int(round(float(value))) for value in coords.mean(axis=0))
    else:
        center = tuple(int(value // 2) for value in image.shape)
    panels = [
        _overlay_panel(image[center[0], :, :], tumor[center[0], :, :]),
        _overlay_panel(image[:, center[1], :], tumor[:, center[1], :]),
        _overlay_panel(image[:, :, center[2]], tumor[:, :, center[2]]),
    ]
    height = max(panel.shape[0] for panel in panels)
    width = max(panel.shape[1] for panel in panels)
    separator = np.zeros((height, 4, 3), dtype=np.uint8)
    canvas_parts: list[np.ndarray] = []
    for index, panel in enumerate(panels):
        if index:
            canvas_parts.append(separator)
        canvas_parts.append(_pad_panel(panel, height, width))
    canvas = np.concatenate(canvas_parts, axis=1)
    _write_rgb_png(path, canvas)
    return {
        "slice_indices_native_axes": list(center),
        "panel_order": ["axis0", "axis1", "axis2"],
        "ct_window_hu": list(VISUAL_WINDOW_HU),
        "overlay_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--search-root", type=Path, default=DEFAULT_SEARCH_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise RuntimeError("output directory already exists; R1 audit never overwrites evidence")
    output_dir.mkdir(parents=True)
    overlays_dir = output_dir / "overlays_local_only"
    overlays_dir.mkdir()

    manifest, split = _load_locked_artifacts(args.manifest, args.split)
    allowed = authorized_r1_cases(manifest, split)
    first_train = next(case for case in allowed if case.partition == "train")
    dataset_root = (
        args.dataset_root.expanduser().resolve()
        if args.dataset_root is not None
        else _detect_dataset_root(args.search_root, first_train.relative_image_path)
    )

    records: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    attempted = Counter()
    successful = Counter()
    opened_case_ids: set[str] = set()
    for case in allowed:
        attempted[case.partition] += 1
        try:
            record, case_mismatches = _audit_case(case, dataset_root=dataset_root)
        except Exception as exc:  # noqa: BLE001 - persist every real-data case failure.
            mismatches.append(
                {
                    "anonymous_case_id": case.anonymous_case_id,
                    "partition": case.partition,
                    "reason": f"case_audit_exception:{type(exc).__name__}:{exc}",
                    "resolution_status": "unresolved",
                    "resolution": None,
                }
            )
            continue
        opened_case_ids.add(case.anonymous_case_id)
        successful[case.partition] += 1
        records.append(record)
        mismatches.extend(case_mismatches)

    if attempted["train"] != R1_EXPECTED_TRAIN_CASES:
        raise RuntimeError("attempted train audit count is not exactly 91")
    if attempted["validation"] != R1_EXPECTED_VALIDATION_CASES:
        raise RuntimeError("attempted validation audit count is not exactly 20")

    train_spacings = np.asarray(
        [record["image_spacing_mm"] for record in records if record["partition"] == "train"],
        dtype=np.float64,
    )
    if train_spacings.shape[0] != R1_EXPECTED_TRAIN_CASES:
        target_spacing: tuple[float, ...] | None = None
    else:
        target_spacing = tuple(float(value) for value in np.median(train_spacings, axis=0))

    overlay_ids: list[str] = []
    transform_traces: dict[str, Any] = {}
    overlay_review: list[dict[str, Any]] = []
    if target_spacing is not None and successful["train"] >= OVERLAY_COUNT:
        overlay_ids = _select_diverse_train(records, OVERLAY_COUNT)
        allowed_by_id = {case.anonymous_case_id: case for case in allowed}
        for case_id in overlay_ids:
            case = allowed_by_id[case_id]
            image_path = resolve_regular_file_beneath_root(dataset_root, case.relative_image_path)
            label_path = resolve_regular_file_beneath_root(dataset_root, case.relative_label_path)
            image_nifti, image = _nifti_arrays(image_path)
            _, label = _nifti_arrays(label_path)
            tumor = validate_raw_lits_label_array(label)
            trace, _ = _roundtrip_trace(
                image=image,
                tumor=tumor,
                native_affine=np.asarray(image_nifti.affine, dtype=np.float64),
                target_spacing=target_spacing,
            )
            overlay_path = overlays_dir / f"{case_id}_triplanar.png"
            overlay_meta = _write_triplanar_overlay(overlay_path, image, tumor)
            trace.update(overlay_meta)
            transform_traces[case_id] = trace
            overlay_review.append(
                {
                    "anonymous_case_id": case_id,
                    "overlay_filename": overlay_path.name,
                    "overlay_sha256": overlay_meta["overlay_sha256"],
                    "alignment_confirmed": None,
                    "notes": None,
                }
            )

    unresolved = [item for item in mismatches if item["resolution_status"] != "resolved"]
    transform_failures = [
        case_id
        for case_id, trace in transform_traces.items()
        if not (
            trace["crop_restore_exact"]
            and trace["pure_orientation_inverse_exact"]
            and trace["restored_native_shape_match"]
            and trace["resampled_image_finite"]
            and set(trace["resampled_label_values"]).issubset({0, 1})
        )
    ]
    summary = {
        "schema_version": "research_v2_r1_real_audit.v1",
        "status": "READY_FOR_HUMAN_OVERLAY_REVIEW"
        if (
            successful["train"] == R1_EXPECTED_TRAIN_CASES
            and successful["validation"] == R1_EXPECTED_VALIDATION_CASES
            and not unresolved
            and len(overlay_ids) == OVERLAY_COUNT
            and not transform_failures
        )
        else "BLOCKED",
        "manifest_artifact_hash": manifest.manifest_hash,
        "manifest_file_sha256": EXPECTED_MANIFEST_FILE_SHA256,
        "split_artifact_hash": split.split_hash,
        "split_file_sha256": EXPECTED_SPLIT_FILE_SHA256,
        "array_access": {
            "attempted_train_cases": attempted["train"],
            "attempted_validation_cases": attempted["validation"],
            "successfully_opened_train_cases": successful["train"],
            "successfully_opened_validation_cases": successful["validation"],
            "unique_authorized_case_arrays_opened": len(opened_case_ids),
            "internal_test_arrays_opened": 0,
            "external_arrays_opened": 0,
        },
        "train_only_target_spacing_median_mm": list(target_spacing)
        if target_spacing is not None
        else None,
        "observed_label_contract": {
            "allowed_raw_values": list(R1_ALLOWED_RAW_LABEL_VALUES),
            "tumor_raw_value": R1_TUMOR_RAW_LABEL_VALUE,
        },
        "case_record_count": len(records),
        "unresolved_mismatch_count": len(unresolved),
        "unresolved_mismatch_case_ids": sorted(
            {str(item["anonymous_case_id"]) for item in unresolved}
        ),
        "overlay_train_case_count": len(overlay_ids),
        "overlay_train_case_ids": overlay_ids,
        "transform_trace_failures": transform_failures,
        "human_overlay_review_complete": False,
        "training_performed": False,
        "threshold_tuning_performed": False,
        "internal_test_array_access_authorized": False,
        "external_array_access_authorized": False,
    }
    _write_json(output_dir / "r1_case_records_no_paths.json", records)
    _write_json(output_dir / "r1_mismatches.json", mismatches)
    _write_json(output_dir / "r1_transform_traces.json", transform_traces)
    _write_json(output_dir / "r1_overlay_review.json", overlay_review)
    _write_json(output_dir / "r1_real_audit_summary.json", summary)

    evidence_hashes = {}
    for filename in (
        "r1_case_records_no_paths.json",
        "r1_mismatches.json",
        "r1_transform_traces.json",
        "r1_overlay_review.json",
        "r1_real_audit_summary.json",
    ):
        evidence_hashes[filename] = hashlib.sha256((output_dir / filename).read_bytes()).hexdigest()
    _write_json(output_dir / "r1_evidence_hashes.json", evidence_hashes)

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"EVIDENCE_DIR={output_dir}")
    print("Review local overlays before R1 can PASS:")
    print(overlays_dir)
    return 0 if summary["status"] == "READY_FOR_HUMAN_OVERLAY_REVIEW" else 40


if __name__ == "__main__":
    started = time.time()
    try:
        exit_code = main()
    except Exception as exc:  # noqa: BLE001 - top-level audit fails closed with a visible reason.
        print(f"R1_AUDIT_FATAL={type(exc).__name__}:{exc}", file=sys.stderr)
        exit_code = 50
    print(f"R1_AUDIT_RUNTIME_SECONDS={time.time() - started:.3f}")
    raise SystemExit(exit_code)
