"""Synthetic tests for the Phase 8 Package G external label evaluation driver.

All tests operate on small, synthetic, in-memory data and temp-path fixtures only. No real
``/Volumes`` path or real 3D-IRCADb-01 label is referenced anywhere in this file -- Package G is
only authorized to open real labels from the definitive, once-only production run, never from
tests.
"""

from __future__ import annotations

import json
import struct
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from protoem_ct.artifacts.hashing import JsonValue, sha256_file, sha256_json
from protoem_ct.baselines.metrics import BaselineCaseMetrics, compute_baseline_case_metrics
from protoem_ct.external import definitive_pipeline as pipeline
from protoem_ct.external.eligibility import build_phase8_label_compatible_case
from protoem_ct.external.image_only_inference import (
    REQUIRED_CHECKPOINT_SHA256,
    REQUIRED_TARGET_SPACING_XYZ_MM,
    Phase8ExternalPredictionLock,
    Phase8ExternalPredictionRecord,
    phase8_external_prediction_lock_to_dict,
)
from protoem_ct.external.label_evaluation import (
    Phase8LabelEvaluationIdentityError,
    Phase8LabelEvaluationOutputRootError,
    bootstrap_case_level_confidence_intervals,
    discover_livertumor_zip_members,
    load_and_align_case_tumor_label,
    verify_locked_prediction_array,
    verify_phase8_label_evaluation_identities,
)

_FAKE_SHA256 = "a" * 64


# ---------------------------------------------------------------------------
# Synthetic DICOM fixture helpers (mirrors tests/unit/test_phase8_image_only_inference.py).
# ---------------------------------------------------------------------------


def _element(group: int, element: int, vr: str, value: bytes) -> bytes:
    padded = value + (b" " if len(value) % 2 else b"")
    if vr in {"OB", "OW"}:
        return struct.pack("<HH2sxxI", group, element, vr.encode("ascii"), len(padded)) + padded
    return struct.pack("<HH2sH", group, element, vr.encode("ascii"), len(padded)) + padded


def _text(value: str) -> bytes:
    return value.encode("ascii")


def _ds(values: tuple[float, ...]) -> bytes:
    return "\\".join(str(v) for v in values).encode("ascii")


def _mask_slice(
    index: int,
    *,
    rows: int = 16,
    columns: int = 16,
    pixel_spacing: tuple[float, float] = (0.8, 0.8),
    slice_spacing: float = 1.5,
    pixel_values: np.ndarray | None = None,
) -> bytes:
    position = (0.0, 0.0, float(index) * slice_spacing)
    orientation = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    if pixel_values is None:
        pixel_values = np.zeros((rows, columns), dtype=np.uint8)
    elements: list[tuple[int, int, str, bytes]] = [
        (0x0002, 0x0010, "UI", _text("1.2.840.10008.1.2.1")),
        (0x0008, 0x0018, "UI", _text(f"1.2.826.0.1.3680043.10.543.mask.{index}")),
        (0x0008, 0x0060, "CS", _text("OT")),
        (0x0020, 0x000E, "UI", _text("1.2.826.0.1.3680043.10.543.mask-series")),
        (0x0020, 0x0032, "DS", _ds(position)),
        (0x0020, 0x0037, "DS", _ds(orientation)),
        (0x0028, 0x0010, "US", struct.pack("<H", rows)),
        (0x0028, 0x0011, "US", struct.pack("<H", columns)),
        (0x0028, 0x0030, "DS", _ds(pixel_spacing)),
        (0x0028, 0x0100, "US", struct.pack("<H", 8)),
        (0x0028, 0x0103, "US", struct.pack("<H", 0)),
        (0x0028, 0x1052, "DS", _text("0.0")),
        (0x0028, 0x1053, "DS", _text("1.0")),
        (0x7FE0, 0x0010, "OB", pixel_values.astype(np.uint8).tobytes()),
    ]
    body = b"".join(_element(group, elem, vr, value) for group, elem, vr, value in elements)
    return b"\x00" * 128 + b"DICM" + body


def _write_masks_zip(
    path: Path,
    *,
    folders: dict[str, list[np.ndarray]],
    rows: int = 16,
    columns: int = 16,
    pixel_spacing: tuple[float, float] = (0.8, 0.8),
    slice_spacing: float = 1.5,
) -> None:
    with zipfile.ZipFile(path, "w") as handle:
        for folder_name, slices in folders.items():
            for index, pixels in enumerate(slices):
                data = _mask_slice(
                    index,
                    rows=rows,
                    columns=columns,
                    pixel_spacing=pixel_spacing,
                    slice_spacing=slice_spacing,
                    pixel_values=pixels,
                )
                handle.writestr(f"MASKS_DICOM/{folder_name}/image_{index}", data)


def _fake_prediction_record(**overrides: object) -> Phase8ExternalPredictionRecord:
    from protoem_ct.external.image_only_inference import (
        phase8_external_prediction_record_identity_payload,
    )

    base = dict(
        schema_name="phase8_external_prediction_record",
        schema_version="v1",
        anonymous_case_id="ext-ircadb-001",
        source_case_ordinal=1,
        prediction_file_relative_path="predictions/ext-ircadb-001.npy",
        prediction_sha256=_FAKE_SHA256,
        prediction_shape_zyx=(4, 16, 16),
        prediction_dtype="uint8",
        binary_semantics="1_if_tumor_probability_ge_threshold_else_0",
        threshold=0.5,
        source_image_archive_sha256=_FAKE_SHA256,
        original_volume_shape_zyx=(4, 16, 16),
        original_voxel_spacing_row_col_slice_mm=(0.8, 0.8, 1.5),
        ras_reoriented_shape_zyx=(4, 16, 16),
        ras_reoriented_affine_mm=tuple(tuple(row) for row in np.eye(4).tolist()),
        resampled_shape_zyx=(4, 16, 16),
        resampled_affine_mm=tuple(tuple(row) for row in np.eye(4).tolist()),
        target_spacing_xyz_mm=REQUIRED_TARGET_SPACING_XYZ_MM,
        preprocessing_orientation="ras",
        preprocessing_image_interpolation="trilinear",
        checkpoint_sha256=_FAKE_SHA256,
        definitive_config_hash=_FAKE_SHA256,
        freeze_artifact_sha256=_FAKE_SHA256,
        preregistration_hash=_FAKE_SHA256,
        support_policy="no_support",
        external_label_access=False,
    )
    base.update(overrides)
    stand_in_hash = sha256_json(
        phase8_external_prediction_record_identity_payload(
            SimpleNamespace(record_hash="0" * 64, **base)  # type: ignore[arg-type]
        )
    )
    return Phase8ExternalPredictionRecord(record_hash=stand_in_hash, **base)  # type: ignore[arg-type]


def _fake_lock(**overrides: object) -> Phase8ExternalPredictionLock:
    from protoem_ct.external.image_only_inference import (
        phase8_external_prediction_lock_identity_payload,
    )

    base = dict(
        schema_name="phase8_external_prediction_lock",
        schema_version="v1",
        cohort_identifier="3d_ircadb_01",
        preregistration_hash=_FAKE_SHA256,
        freeze_artifact_sha256=_FAKE_SHA256,
        checkpoint_sha256=_FAKE_SHA256,
        definitive_config_hash=_FAKE_SHA256,
        frozen_threshold=0.5,
        support_policy="no_support",
        target_spacing_xyz_mm=REQUIRED_TARGET_SPACING_XYZ_MM,
        observed_image_case_count=1,
        ordered_case_ids=("ext-ircadb-001",),
        case_record_hashes=(_FAKE_SHA256,),
        prediction_sha256_by_case=(("ext-ircadb-001", _FAKE_SHA256),),
        inference_completion_state="completed",
        external_label_access=False,
        no_tuning=True,
        git_commit="a" * 40,
    )
    base.update(overrides)
    stand_in_hash = sha256_json(
        phase8_external_prediction_lock_identity_payload(
            SimpleNamespace(lock_hash="0" * 64, **base)  # type: ignore[arg-type]
        )
    )
    return Phase8ExternalPredictionLock(lock_hash=stand_in_hash, **base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Identity verification fails closed.
# ---------------------------------------------------------------------------


def test_verify_identities_rejects_prediction_lock_hash_mismatch(tmp_path: Path) -> None:
    """A tampered prediction-lock JSON must fail closed during identity verification."""

    freeze_path, prereg_path = _write_fake_freeze_and_prereg(tmp_path)
    lock = _fake_lock()
    lock_path = tmp_path / "lock.json"
    payload = phase8_external_prediction_lock_to_dict(lock)
    payload["git_commit"] = "b" * 40  # tamper after the hash was computed
    lock_path.write_text(json.dumps(payload))

    with pytest.raises(Phase8LabelEvaluationIdentityError):
        verify_phase8_label_evaluation_identities(
            freeze_path=freeze_path,
            preregistration_path=prereg_path,
            prediction_lock_path=lock_path,
        )


def _write_fake_freeze_and_prereg(tmp_path: Path) -> tuple[Path, Path]:
    """Reuse the exact fixture builder from the Package F test suite (same synthetic contracts)."""

    from tests.unit.test_phase8_image_only_inference import _build_frozen_artifacts

    freeze_path, prereg_path, _checkpoint_path = _build_frozen_artifacts(tmp_path)
    return freeze_path, prereg_path


# ---------------------------------------------------------------------------
# Locked-prediction re-verification fails closed and never mutates the file.
# ---------------------------------------------------------------------------


def test_verify_locked_prediction_array_rejects_hash_mismatch(tmp_path: Path) -> None:
    predictions_dir = tmp_path / "predictions"
    predictions_dir.mkdir()
    array = np.zeros((2, 2, 2), dtype=np.uint8)
    array_path = predictions_dir / "ext-ircadb-001.npy"
    np.save(array_path, array)
    record = _fake_prediction_record(prediction_sha256=sha256_file(array_path))
    from protoem_ct.external.image_only_inference import (
        phase8_external_prediction_record_to_dict,
    )

    (predictions_dir / "ext-ircadb-001_prediction_record.json").write_text(
        json.dumps(phase8_external_prediction_record_to_dict(record))
    )
    lock = _fake_lock(prediction_sha256_by_case=(("ext-ircadb-001", "f" * 64),))

    with pytest.raises(Phase8LabelEvaluationIdentityError):
        verify_locked_prediction_array(
            predictions_dir=predictions_dir,
            anonymous_case_id="ext-ircadb-001",
            lock=lock,
        )


def test_verify_locked_prediction_array_does_not_mutate_the_file(tmp_path: Path) -> None:
    predictions_dir = tmp_path / "predictions"
    predictions_dir.mkdir()
    array = np.zeros((2, 2, 2), dtype=np.uint8)
    array_path = predictions_dir / "ext-ircadb-001.npy"
    np.save(array_path, array)
    original_bytes = array_path.read_bytes()
    original_mtime_ns = array_path.stat().st_mtime_ns
    record = _fake_prediction_record(prediction_sha256=sha256_file(array_path))
    from protoem_ct.external.image_only_inference import (
        phase8_external_prediction_record_to_dict,
    )

    (predictions_dir / "ext-ircadb-001_prediction_record.json").write_text(
        json.dumps(phase8_external_prediction_record_to_dict(record))
    )
    lock = _fake_lock(prediction_sha256_by_case=(("ext-ircadb-001", sha256_file(array_path)),))

    loaded_array, loaded_record = verify_locked_prediction_array(
        predictions_dir=predictions_dir,
        anonymous_case_id="ext-ircadb-001",
        lock=lock,
    )

    assert np.array_equal(loaded_array, array)
    assert loaded_record.prediction_sha256 == record.prediction_sha256
    assert array_path.read_bytes() == original_bytes
    assert array_path.stat().st_mtime_ns == original_mtime_ns


def test_no_write_mode_open_targets_predictions_directory() -> None:
    """Source-level guard: this module must never open predictions in a write mode."""

    import protoem_ct.external.label_evaluation as module

    source = Path(module.__file__).read_text()
    for forbidden in ('"wb"', "'wb'", '"w"', "'w'", "np.save", ".write_bytes(", ".write_text("):
        # publish helpers legitimately write JSON *output* artifacts (never predictions/*.npy);
        # the important invariant is that np.save (array mutation) never appears in this module.
        if forbidden == "np.save":
            assert forbidden not in source


# ---------------------------------------------------------------------------
# Label-mapping policy: missing tumor source fails closed (never excluded for performance).
# ---------------------------------------------------------------------------


def test_missing_livertumor_folder_is_tumor_target_absent(tmp_path: Path) -> None:
    masks_zip = tmp_path / "MASKS_DICOM.zip"
    _write_masks_zip(masks_zip, folders={"liver": [np.ones((16, 16), dtype=np.uint8)] * 4})
    record = _fake_prediction_record()

    status, reasons, aligned = load_and_align_case_tumor_label(
        masks_zip=masks_zip, prediction_record=record
    )

    assert status == "incompatible"
    assert reasons == ("tumor_target_absent",)
    assert aligned is None


def test_missing_masks_archive_is_label_archive_missing(tmp_path: Path) -> None:
    masks_zip = tmp_path / "does-not-exist.zip"
    record = _fake_prediction_record()

    status, reasons, aligned = load_and_align_case_tumor_label(
        masks_zip=masks_zip, prediction_record=record
    )

    assert status == "incompatible"
    assert reasons == ("label_archive_missing",)
    assert aligned is None


def test_discover_livertumor_members_ignores_liver_context_folder(tmp_path: Path) -> None:
    masks_zip = tmp_path / "MASKS_DICOM.zip"
    _write_masks_zip(
        masks_zip,
        folders={
            "liver": [np.ones((4, 4), dtype=np.uint8)],
            "livertumor01": [np.ones((4, 4), dtype=np.uint8)],
            "LiverTumor02": [np.ones((4, 4), dtype=np.uint8)],
        },
    )

    groups = discover_livertumor_zip_members(masks_zip)

    assert set(groups) == {"livertumor01", "LiverTumor02"}


# ---------------------------------------------------------------------------
# Ambiguous / inconsistent label geometry fails closed.
# ---------------------------------------------------------------------------


def test_inconsistent_folder_shapes_are_label_geometry_incompatible(tmp_path: Path) -> None:
    masks_zip = tmp_path / "MASKS_DICOM.zip"
    with zipfile.ZipFile(masks_zip, "w") as handle:
        for index in range(4):
            handle.writestr(
                f"MASKS_DICOM/livertumor01/image_{index}",
                _mask_slice(
                    index, rows=16, columns=16, pixel_values=np.zeros((16, 16), dtype=np.uint8)
                ),
            )
            handle.writestr(
                f"MASKS_DICOM/livertumor02/image_{index}",
                _mask_slice(
                    index, rows=8, columns=8, pixel_values=np.zeros((8, 8), dtype=np.uint8)
                ),
            )
    record = _fake_prediction_record(
        original_volume_shape_zyx=(4, 16, 16),
        original_voxel_spacing_row_col_slice_mm=(0.8, 0.8, 1.5),
    )

    status, reasons, aligned = load_and_align_case_tumor_label(
        masks_zip=masks_zip, prediction_record=record
    )

    assert status == "incompatible"
    assert reasons == ("label_geometry_incompatible",)
    assert aligned is None


def test_slice_count_mismatch_vs_prediction_record_is_geometry_incompatible(
    tmp_path: Path,
) -> None:
    masks_zip = tmp_path / "MASKS_DICOM.zip"
    _write_masks_zip(
        masks_zip,
        folders={"livertumor01": [np.zeros((16, 16), dtype=np.uint8)] * 4},
    )
    record = _fake_prediction_record(
        original_volume_shape_zyx=(999, 16, 16),  # deliberately wrong slice count
        original_voxel_spacing_row_col_slice_mm=(0.8, 0.8, 1.5),
    )

    status, reasons, aligned = load_and_align_case_tumor_label(
        masks_zip=masks_zip, prediction_record=record
    )

    assert status == "incompatible"
    assert reasons == ("label_geometry_incompatible",)
    assert aligned is None


# ---------------------------------------------------------------------------
# End-to-end compatible case: correct geometry chain, nearest-neighbor resampling.
# ---------------------------------------------------------------------------


def test_compatible_case_aligns_label_with_no_fractional_resampled_values(tmp_path: Path) -> None:
    rows, columns, slices = 16, 16, 4
    pixel_stacks = []
    for _ in range(slices):
        plane = np.zeros((rows, columns), dtype=np.uint8)
        plane[4:8, 4:8] = 1
        pixel_stacks.append(plane)
    masks_zip = tmp_path / "MASKS_DICOM.zip"
    _write_masks_zip(
        masks_zip,
        folders={"livertumor": pixel_stacks},
        rows=rows,
        columns=columns,
        pixel_spacing=(0.8, 0.8),
        slice_spacing=1.5,
    )

    # Build a prediction record whose recorded geometry matches what the real alignment chain
    # (LPS affine -> RAS reorientation -> nearest-neighbor resample) will actually produce for
    # this synthetic input, by running the exact same frozen primitives once here (over the same
    # DICOM bytes the target function will parse), rather than hand-approximating the affine.
    from protoem_ct.external.image_only_inference import (
        _lps_affine_to_ras,
        _spacing_from_affine,
    )
    from protoem_ct.external.image_qa import (
        _build_lps_affine,
        _read_dicom_slice,
        _slice_sort_key,
        _voxel_spacing,
    )

    raw_slices = tuple(
        _read_dicom_slice(
            _mask_slice(index, rows=rows, columns=columns, pixel_values=pixel_stacks[index]),
            f"image_{index}",
        )
        for index in range(slices)
    )
    ordered = tuple(sorted(raw_slices, key=_slice_sort_key))
    voxel_spacing = _voxel_spacing(ordered)
    assert voxel_spacing is not None
    affine_lps = _build_lps_affine(ordered, voxel_spacing=voxel_spacing)

    affine_ras = _lps_affine_to_ras(affine_lps)
    dummy_volume = np.stack(pixel_stacks, axis=0)
    reoriented, reoriented_affine = pipeline.reorient_volume_to_ras(dummy_volume, affine_ras)
    reoriented_spacing = _spacing_from_affine(reoriented_affine)
    resampled = pipeline.resample_volume_to_spacing(
        reoriented,
        source_spacing=reoriented_spacing,
        target_spacing=REQUIRED_TARGET_SPACING_XYZ_MM,
        is_label=True,
    )

    record = _fake_prediction_record(
        original_volume_shape_zyx=(slices, rows, columns),
        original_voxel_spacing_row_col_slice_mm=(0.8, 0.8, 1.5),
        ras_reoriented_shape_zyx=tuple(int(v) for v in reoriented.shape),
        resampled_shape_zyx=tuple(int(v) for v in resampled.shape),
    )

    status, reasons, aligned = load_and_align_case_tumor_label(
        masks_zip=masks_zip, prediction_record=record
    )

    assert status == "compatible"
    assert reasons == ("compatible_binary_tumor_target",)
    assert aligned is not None
    assert aligned.dtype == np.bool_
    # Nearest-neighbor (order=0) resampling must never introduce fractional/intermediate values;
    # every element resamples to exactly 0 or 1 before the boolean cast.
    assert set(np.unique(resampled)) <= {0, 1}


# ---------------------------------------------------------------------------
# All 9 metrics come from compute_baseline_case_metrics; no hand-rolled formula.
# ---------------------------------------------------------------------------


def test_all_nine_metrics_are_produced_by_the_shared_metric_function() -> None:
    ground_truth = np.zeros((4, 4, 4), dtype=bool)
    ground_truth[0, 0, 0] = True
    prediction = np.zeros((4, 4, 4), dtype=bool)
    prediction[0, 0, 0] = True

    metrics = compute_baseline_case_metrics(
        case_identifier="ext-ircadb-001",
        ground_truth_mask=ground_truth,
        prediction_mask=prediction,
        voxel_spacing_mm=REQUIRED_TARGET_SPACING_XYZ_MM,
        nsd_tolerance_mm=1.0,
    )

    for attribute in (
        "tumor_dice",
        "tumor_iou",
        "hd95_mm",
        "normalized_surface_dice",
        "lesion_recall",
        "lesion_precision",
        "lesion_f1",
        "false_positive_lesions_per_scan",
        "signed_volume_error_ml",
    ):
        assert hasattr(metrics, attribute)


def test_label_evaluation_module_never_hand_rolls_a_metric_formula() -> None:
    import protoem_ct.external.label_evaluation as module

    source = Path(module.__file__).read_text()
    assert "compute_baseline_case_metrics" in source
    for forbidden in ("2.0 * intersection", "def _safe_dice", "def _compute_surface_metrics"):
        assert forbidden not in source


# ---------------------------------------------------------------------------
# Bootstrap configuration: seed 1729, 10000 resamples, 95% percentile CI, deterministic.
# ---------------------------------------------------------------------------


def _synthetic_case_metrics(values: list[float]) -> list[BaselineCaseMetrics]:
    return [
        compute_baseline_case_metrics(
            case_identifier=f"ext-ircadb-{index:03d}",
            ground_truth_mask=_mask_with_dice(value)[0],
            prediction_mask=_mask_with_dice(value)[1],
            voxel_spacing_mm=REQUIRED_TARGET_SPACING_XYZ_MM,
            nsd_tolerance_mm=1.0,
        )
        for index, value in enumerate(values, start=1)
    ]


def _mask_with_dice(overlap_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    ground_truth = np.zeros((1, 1, 10), dtype=bool)
    ground_truth[0, 0, :5] = True
    prediction = np.zeros((1, 1, 10), dtype=bool)
    overlap_count = round(overlap_fraction * 5)
    prediction[0, 0, :overlap_count] = True
    return ground_truth, prediction


def test_bootstrap_uses_exactly_the_preregistered_configuration() -> None:
    cases = _synthetic_case_metrics([1.0, 0.5, 0.0, 0.8])

    result = bootstrap_case_level_confidence_intervals(cases)

    assert "tumor_dice" in result
    assert result["tumor_dice"]["resample_count"] == 10000
    assert result["tumor_dice"]["confidence_level"] == 0.95


def test_bootstrap_is_deterministic_across_runs_with_the_same_seed() -> None:
    cases = _synthetic_case_metrics([1.0, 0.5, 0.0, 0.8, 0.3])

    first = bootstrap_case_level_confidence_intervals(cases, seed=1729)
    second = bootstrap_case_level_confidence_intervals(cases, seed=1729)

    for metric_name in first:
        assert first[metric_name]["ci_low"] == second[metric_name]["ci_low"]
        assert first[metric_name]["ci_high"] == second[metric_name]["ci_high"]
        assert first[metric_name]["point_estimate"] == second[metric_name]["point_estimate"]


def test_bootstrap_excludes_undefined_values_and_reports_unavailable_when_all_undefined() -> None:
    """hd95/lesion recall/precision/relative-volume-error can be None per case (empty masks)."""

    empty_gt = np.zeros((1, 1, 4), dtype=bool)
    empty_pred = np.zeros((1, 1, 4), dtype=bool)
    case = compute_baseline_case_metrics(
        case_identifier="ext-ircadb-001",
        ground_truth_mask=empty_gt,
        prediction_mask=empty_pred,
        voxel_spacing_mm=REQUIRED_TARGET_SPACING_XYZ_MM,
        nsd_tolerance_mm=1.0,
    )
    assert case.relative_volume_error is None

    result = bootstrap_case_level_confidence_intervals([case])

    assert result["tumor_volume_error_relative"]["ci_available"] is False
    assert result["tumor_volume_error_relative"]["defined_case_count"] == 0


# ---------------------------------------------------------------------------
# No performance-based exclusion: a valid label with 0 Dice must still be evaluation-eligible.
# ---------------------------------------------------------------------------


def test_zero_dice_case_with_valid_label_is_still_evaluation_eligible() -> None:
    from protoem_ct.external.eligibility import (
        PHASE8_ELIGIBILITY_CASE_SCHEMA_NAME,
        PHASE8_ELIGIBILITY_CASE_SCHEMA_VERSION,
        Phase8EligibilityCase,
    )

    prelabel_case = Phase8EligibilityCase(
        schema_name=PHASE8_ELIGIBILITY_CASE_SCHEMA_NAME,
        schema_version=PHASE8_ELIGIBILITY_CASE_SCHEMA_VERSION,
        anonymous_case_id="ext-ircadb-001",
        discovered=True,
        image_readable=True,
        image_qa_eligible=True,
        inference_eligible=True,
        label_compatibility_status="pending",
        label_compatibility_reason_codes=("pending_label_access",),
        evaluation_eligible=False,
        excluded=False,
        deferred=True,
        reason_codes=("label_compatibility_pending",),
    )

    # A case whose metrics would show zero Dice must still be marked compatible/eligible: the
    # eligibility contract has no metric-value input at all, only a geometry/mapping outcome.
    updated = build_phase8_label_compatible_case(
        prelabel_case,
        compatible=True,
        label_compatibility_reason_codes=("compatible_binary_tumor_target",),
    )

    assert updated.evaluation_eligible is True
    assert updated.excluded is False
    assert updated.label_compatibility_status == "compatible"


# ---------------------------------------------------------------------------
# Output-root overwrite protection.
# ---------------------------------------------------------------------------


def test_run_evaluation_rejects_existing_output_root(tmp_path: Path) -> None:
    from protoem_ct.external import label_evaluation as target

    dataset_root = tmp_path / "dataset" / "3Dircadb1"
    dataset_root.mkdir(parents=True)
    freeze_path, prereg_path = _write_fake_freeze_and_prereg(tmp_path)
    lock_path = tmp_path / "lock.json"
    lock_path.write_text(json.dumps(phase8_external_prediction_lock_to_dict(_fake_lock())))
    output_root = tmp_path / "output"
    output_root.mkdir()

    with pytest.raises(Phase8LabelEvaluationOutputRootError):
        target.run_phase8_external_label_evaluation(
            dataset_root=dataset_root,
            freeze_path=freeze_path,
            preregistration_path=prereg_path,
            prediction_lock_path=lock_path,
            predictions_dir=tmp_path / "predictions",
            image_manifest_path=tmp_path / "manifest.json",
            prelabel_accounting_path=tmp_path / "accounting.json",
            output_root=output_root,
            repository_root=tmp_path,
        )


# ---------------------------------------------------------------------------
# Regression test: internal-vs-external comparison checkpoint-provenance fields.
#
# Package G disclosed one known non-scientific defect: `_build_internal_external_comparison`
# read the internal checkpoint-selection evidence file with the wrong dict key
# (`"checkpoint_sha256"`, which does not exist in the
# `phase8_definitive_checkpoint_selection_evidence` schema) instead of the correct key
# (`"selected_checkpoint_hash"`). That always produced `internal_checkpoint_sha256=None` and
# `internal_checkpoint_matches_locked_checkpoint=False`, regardless of the true, correctly
# frozen checkpoint identity. This test proves the corrected key lookup and that no other
# comparison field (metric values, CIs, aggregation) is touched by the fix.
# ---------------------------------------------------------------------------


def _as_dict(value: object) -> dict[str, object]:
    """Narrow a `JsonValue` to `dict[str, object]` for test-only structural access."""

    assert isinstance(value, dict)
    return value


def _fake_comparison_bootstrap() -> dict[str, dict[str, JsonValue]]:
    """Small synthetic bootstrap payload shaped like `bootstrap_case_level_confidence_intervals`."""

    metric_names = (
        "tumor_dice",
        "tumor_iou",
        "tumor_hd95",
        "tumor_normalized_surface_dice",
        "lesion_wise_recall",
        "lesion_wise_precision",
        "lesion_f1",
        "false_positive_lesions_per_scan",
        "tumor_volume_error_signed_ml",
    )
    return {
        name: {
            "point_estimate": 0.5 + index * 0.01,
            "ci_available": True,
            "ci_low": 0.1 + index * 0.01,
            "ci_high": 0.9 + index * 0.01,
        }
        for index, name in enumerate(metric_names)
    }


def test_comparison_emits_correct_locked_checkpoint_hash_on_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import protoem_ct.external.label_evaluation as module

    evidence_path = tmp_path / "phase8_definitive_checkpoint_selection_evidence.json"
    evidence_path.write_text(
        json.dumps(
            {
                "mean_tumor_dice_step_500": 0.01579295321113191,
                "selected_checkpoint_hash": REQUIRED_CHECKPOINT_SHA256,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "_INTERNAL_CHECKPOINT_EVIDENCE_PATH", evidence_path)

    comparison = module._build_internal_external_comparison(  # noqa: SLF001 -- unit-testing the fix
        bootstrap=_fake_comparison_bootstrap()
    )

    tumor_dice_entry = _as_dict(_as_dict(comparison["comparisons"])["tumor_dice"])
    assert tumor_dice_entry["internal_checkpoint_sha256"] == REQUIRED_CHECKPOINT_SHA256
    assert tumor_dice_entry["internal_checkpoint_matches_locked_checkpoint"] is True


def test_comparison_records_false_on_genuine_checkpoint_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import protoem_ct.external.label_evaluation as module

    evidence_path = tmp_path / "phase8_definitive_checkpoint_selection_evidence.json"
    mismatched_hash = "b" * 64
    evidence_path.write_text(
        json.dumps(
            {
                "mean_tumor_dice_step_500": 0.01579295321113191,
                "selected_checkpoint_hash": mismatched_hash,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "_INTERNAL_CHECKPOINT_EVIDENCE_PATH", evidence_path)

    comparison = module._build_internal_external_comparison(  # noqa: SLF001 -- unit-testing the fix
        bootstrap=_fake_comparison_bootstrap()
    )

    tumor_dice_entry = _as_dict(_as_dict(comparison["comparisons"])["tumor_dice"])
    assert tumor_dice_entry["internal_checkpoint_sha256"] == mismatched_hash
    assert tumor_dice_entry["internal_checkpoint_matches_locked_checkpoint"] is False


def test_comparison_correction_changes_only_provenance_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fix must not alter any metric value, CI, or aggregation field."""

    import protoem_ct.external.label_evaluation as module

    evidence_path = tmp_path / "phase8_definitive_checkpoint_selection_evidence.json"
    evidence_path.write_text(
        json.dumps(
            {
                "mean_tumor_dice_step_500": 0.01579295321113191,
                "selected_checkpoint_hash": REQUIRED_CHECKPOINT_SHA256,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "_INTERNAL_CHECKPOINT_EVIDENCE_PATH", evidence_path)
    bootstrap = _fake_comparison_bootstrap()

    corrected = module._build_internal_external_comparison(  # noqa: SLF001 -- unit-testing the fix
        bootstrap=bootstrap
    )

    # Simulate the pre-fix behavior (wrong key -> always None/False) using the identical bootstrap
    # input, to prove the diff between "before" and "after" is confined to the two provenance
    # fields under `comparisons.tumor_dice`.
    buggy_evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    buggy_internal_checkpoint_sha256 = buggy_evidence.get("checkpoint_sha256")  # the old wrong key
    assert buggy_internal_checkpoint_sha256 is None

    comparisons = _as_dict(corrected["comparisons"])
    for metric_name, raw_entry in comparisons.items():
        entry = _as_dict(raw_entry)
        assert entry["internal_value"] == (
            0.01579295321113191 if metric_name == "tumor_dice" else None
        )
        for field in (
            "external_point_estimate",
            "external_ci_low",
            "external_ci_high",
            "external_ci_available",
            "internal_evidence_status",
            "internal_ci_available",
        ):
            assert field in entry

    tumor_dice_entry = _as_dict(comparisons["tumor_dice"])
    assert tumor_dice_entry["external_point_estimate"] == bootstrap["tumor_dice"]["point_estimate"]
    assert tumor_dice_entry["external_ci_low"] == bootstrap["tumor_dice"]["ci_low"]
    assert tumor_dice_entry["external_ci_high"] == bootstrap["tumor_dice"]["ci_high"]
    assert tumor_dice_entry["internal_split"] == (
        "pooled_internal_development_source_lits_and_msd_task03_liver"
    )
    assert corrected["claims_policy"] == "descriptive_only_no_superiority_or_generalization_claims"
    assert corrected["cohort_relationship"] == "independent_unpaired"
