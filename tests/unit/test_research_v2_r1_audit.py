from __future__ import annotations

import hashlib

import numpy as np
import pytest

from protoem_ct.artifacts import (
    DATASET_MANIFEST_TYPE,
    DEVELOPMENT_SPLIT_MANIFEST_TYPE,
    PHASE2_DEVELOPMENT_COHORT_ROLE,
    PHASE2_SCHEMA_VERSION,
    DatasetCaseRecord,
    DatasetManifest,
    DevelopmentSplitManifest,
    SplitAssignment,
)
from protoem_ct.research_v2.r1_audit import (
    R1AccessBoundaryError,
    R1GeometryError,
    R1LabelValidationError,
    R1MetricInputError,
    authorized_r1_cases,
    compute_strict_case_metrics,
    grid_for_spacing,
    independent_reference_metrics,
    reorient_array_to_ras,
    require_r1_array_access,
    resample_array_to_grid,
    restore_crop_to_full_grid,
    threshold_fixture_probability,
    validate_raw_lits_label_array,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _identity_affine(spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> np.ndarray:
    affine = np.eye(4, dtype=np.float64)
    affine[0, 0], affine[1, 1], affine[2, 2] = spacing
    return affine


def _locked_shape_synthetic_artifacts() -> tuple[DatasetManifest, DevelopmentSplitManifest]:
    cases = []
    assignments = []
    for index in range(131):
        patient_id = f"pt-{index:03d}"
        case_id = f"case-{index:03d}"
        cases.append(
            DatasetCaseRecord(
                anonymous_patient_id=patient_id,
                anonymous_case_id=case_id,
                relative_image_path=f"images/{case_id}.nii.gz",
                relative_label_path=f"labels/{case_id}.nii.gz",
                image_sha256=_sha(f"image-{index}"),
                label_sha256=_sha(f"label-{index}"),
                cohort_role=PHASE2_DEVELOPMENT_COHORT_ROLE,
            )
        )
        if index < 91:
            partition = "train"
        elif index < 111:
            partition = "validation"
        else:
            partition = "internal_test"
        assignments.append(
            SplitAssignment(
                anonymous_patient_id=patient_id,
                anonymous_case_id=case_id,
                partition=partition,
            )
        )
    manifest_hash = _sha("manifest")
    manifest = DatasetManifest(
        schema_version=PHASE2_SCHEMA_VERSION,
        manifest_type=DATASET_MANIFEST_TYPE,
        dataset_id="lits-development",
        cohort_role=PHASE2_DEVELOPMENT_COHORT_ROLE,
        adapter_name="fixture-adapter",
        adapter_version="v1",
        generated_at_utc="2026-09-24T00:00:00Z",
        git_commit="abcdef0",
        dataset_root_fingerprint=_sha("root"),
        manifest_hash=manifest_hash,
        case_count=131,
        cases=tuple(cases),
    )
    split = DevelopmentSplitManifest(
        schema_version=PHASE2_SCHEMA_VERSION,
        manifest_type=DEVELOPMENT_SPLIT_MANIFEST_TYPE,
        generated_at_utc="2026-09-24T00:00:00Z",
        git_commit="abcdef0",
        source_manifest_hash=manifest_hash,
        split_policy_version="fixture-v1",
        split_seed=1,
        split_hash=_sha("split"),
        assignments=tuple(assignments),
        train_patient_count=91,
        validation_patient_count=20,
        internal_test_patient_count=20,
        train_case_count=91,
        validation_case_count=20,
        internal_test_case_count=20,
    )
    return manifest, split


def _cross_check(gt: np.ndarray, pred: np.ndarray, spacing: tuple[float, float, float] = (1, 1, 1)):
    affine = _identity_affine(tuple(float(v) for v in spacing))
    project = compute_strict_case_metrics(
        case_identifier="fixture-case",
        ground_truth_mask=gt,
        prediction_mask=pred,
        ground_truth_affine=affine,
        prediction_affine=affine,
    )
    reference = independent_reference_metrics(
        ground_truth_mask=gt,
        prediction_mask=pred,
        voxel_spacing_mm=spacing,
    )
    assert project.tumor_dice == pytest.approx(reference.dice, abs=1e-6)
    assert project.tumor_iou == pytest.approx(reference.iou, abs=1e-6)
    assert project.ground_truth_lesion_count == reference.ground_truth_lesions
    assert project.predicted_lesion_count == reference.predicted_lesions
    assert project.true_positive_lesion_count == reference.true_positive_lesions
    assert project.false_positive_lesion_count == reference.false_positive_lesions
    assert project.false_negative_lesion_count == reference.false_negative_lesions
    assert project.hd95_mm == pytest.approx(reference.hd95_mm, abs=1e-6)
    assert project.normalized_surface_dice == pytest.approx(reference.nsd, abs=1e-6)
    return project, reference


def test_r1_authorizes_exactly_train_and_validation() -> None:
    manifest, split = _locked_shape_synthetic_artifacts()
    allowed = authorized_r1_cases(manifest, split)
    assert len(allowed) == 111
    assert sum(case.partition == "train" for case in allowed) == 91
    assert sum(case.partition == "validation" for case in allowed) == 20
    assert not any(case.partition == "internal_test" for case in allowed)


def test_r1_rejects_internal_test_and_external_array_access() -> None:
    with pytest.raises(R1AccessBoundaryError):
        require_r1_array_access("internal_test")
    with pytest.raises(R1AccessBoundaryError):
        require_r1_array_access("external")


def test_raw_lits_label_two_is_tumor_and_unknown_value_is_rejected() -> None:
    label = np.zeros((4, 4, 4), dtype=np.uint8)
    label[1:3, 1:3, 1:3] = 1
    label[2, 2, 2] = 2
    tumor = validate_raw_lits_label_array(label)
    assert tumor.dtype == np.bool_
    assert int(tumor.sum()) == 1

    invalid = label.copy()
    invalid[0, 0, 0] = 3
    with pytest.raises(R1LabelValidationError):
        validate_raw_lits_label_array(invalid)


def test_raw_label_nan_is_rejected() -> None:
    label = np.zeros((3, 3, 3), dtype=np.float64)
    label[1, 1, 1] = np.nan
    with pytest.raises(R1LabelValidationError):
        validate_raw_lits_label_array(label)


def test_golden_both_empty_contract() -> None:
    gt = np.zeros((5, 5, 5), dtype=bool)
    pred = np.zeros_like(gt)
    project, reference = _cross_check(gt, pred)
    assert project.tumor_dice == 1.0
    assert project.tumor_iou == 1.0
    assert project.hd95_mm == 0.0
    assert project.normalized_surface_dice == 1.0
    assert reference.true_positive_lesions == 0


def test_golden_two_lesions_exact_match() -> None:
    gt = np.zeros((12, 12, 12), dtype=bool)
    gt[1:3, 1:3, 1:3] = True
    gt[8:10, 8:10, 8:10] = True
    project, reference = _cross_check(gt, gt.copy())
    assert project.tumor_dice == 1.0
    assert reference.ground_truth_lesions == 2
    assert reference.predicted_lesions == 2
    assert reference.true_positive_lesions == 2
    assert reference.false_positive_lesions == 0
    assert reference.false_negative_lesions == 0


def test_golden_small_shifted_lesion_known_answer() -> None:
    gt = np.zeros((5, 5, 5), dtype=bool)
    pred = np.zeros_like(gt)
    gt[2, 2, 2] = True
    pred[3, 2, 2] = True
    project, reference = _cross_check(gt, pred, spacing=(1.0, 1.0, 2.0))
    assert project.tumor_dice == 0.0
    assert project.tumor_iou == 0.0
    assert reference.true_positive_lesions == 0
    assert reference.false_positive_lesions == 1
    assert reference.false_negative_lesions == 1
    assert project.hd95_mm == pytest.approx(1.0, abs=1e-6)
    assert project.normalized_surface_dice == pytest.approx(1.0, abs=1e-6)


def test_shifted_prediction_with_partial_overlap_cross_checks_matching() -> None:
    gt = np.zeros((9, 9, 9), dtype=bool)
    pred = np.zeros_like(gt)
    gt[2:5, 2:5, 2:5] = True
    pred[3:6, 2:5, 2:5] = True
    project, reference = _cross_check(gt, pred)
    assert project.tumor_dice == pytest.approx(2.0 / 3.0, abs=1e-6)
    assert project.tumor_iou == pytest.approx(0.5, abs=1e-6)
    assert reference.true_positive_lesions == 1


def test_axis_flip_reorientation_and_native_restoration() -> None:
    native = np.zeros((5, 6, 7), dtype=np.uint8)
    native[1, 2, 3] = 1
    native_affine = np.array(
        [
            [-1.0, 0.0, 0.0, 4.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    ras, ras_affine = reorient_array_to_ras(native, native_affine)
    assert tuple(ras.shape) == tuple(native.shape)
    restored = resample_array_to_grid(
        ras,
        source_affine=ras_affine,
        target_shape=native.shape,
        target_affine=native_affine,
        is_label_or_mask=True,
    )
    assert np.array_equal(restored, native)


def test_affine_mismatch_is_rejected_before_metrics() -> None:
    gt = np.zeros((4, 4, 4), dtype=bool)
    pred = np.zeros_like(gt)
    gt_affine = _identity_affine()
    pred_affine = _identity_affine()
    pred_affine[0, 3] = 1.0
    with pytest.raises(R1GeometryError):
        compute_strict_case_metrics(
            case_identifier="fixture-case",
            ground_truth_mask=gt,
            prediction_mask=pred,
            ground_truth_affine=gt_affine,
            prediction_affine=pred_affine,
        )


def test_probability_and_wrong_threshold_are_rejected() -> None:
    probability = np.full((4, 4, 4), 0.75, dtype=np.float32)
    affine = _identity_affine()
    with pytest.raises(R1MetricInputError):
        compute_strict_case_metrics(
            case_identifier="fixture-case",
            ground_truth_mask=np.zeros((4, 4, 4), dtype=bool),
            prediction_mask=probability,
            ground_truth_affine=affine,
            prediction_affine=affine,
        )
    with pytest.raises(R1MetricInputError):
        threshold_fixture_probability(probability, threshold=0.6)
    thresholded = threshold_fixture_probability(probability, threshold=0.5)
    assert thresholded.dtype == np.bool_
    assert bool(thresholded.all())


def test_logits_are_rejected_as_probability() -> None:
    logits = np.full((3, 3, 3), 2.0, dtype=np.float32)
    with pytest.raises(R1MetricInputError):
        threshold_fixture_probability(logits, threshold=0.5)


def test_crop_restore_is_exact_and_out_of_bounds_rejected() -> None:
    crop = np.ones((2, 3, 4), dtype=np.uint8)
    restored = restore_crop_to_full_grid(crop, full_shape=(6, 7, 8), starts=(1, 2, 3))
    assert restored.shape == (6, 7, 8)
    assert int(restored.sum()) == int(crop.sum())
    assert np.array_equal(restored[1:3, 2:5, 3:7], crop)
    with pytest.raises(R1GeometryError):
        restore_crop_to_full_grid(crop, full_shape=(3, 3, 3), starts=(2, 2, 2))


def test_resampling_contract_nearest_for_mask_and_continuous_for_image() -> None:
    source_affine = _identity_affine((2.0, 2.0, 2.0))
    mask = np.zeros((3, 3, 3), dtype=np.uint8)
    mask[1, 1, 1] = 1
    image = mask.astype(np.float32)
    target_shape, target_affine = grid_for_spacing(
        shape=mask.shape,
        affine=source_affine,
        target_spacing_mm=(1.0, 1.0, 1.0),
    )
    resampled_mask = resample_array_to_grid(
        mask,
        source_affine=source_affine,
        target_shape=target_shape,
        target_affine=target_affine,
        is_label_or_mask=True,
    )
    resampled_image = resample_array_to_grid(
        image,
        source_affine=source_affine,
        target_shape=target_shape,
        target_affine=target_affine,
        is_label_or_mask=False,
    )
    assert set(np.unique(resampled_mask)).issubset({0, 1})
    assert np.any((resampled_image > 0.0) & (resampled_image < 1.0))
