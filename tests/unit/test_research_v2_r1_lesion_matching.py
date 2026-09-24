from __future__ import annotations

import numpy as np

from protoem_ct.research_v2.r1_audit import (
    compute_strict_case_metrics,
    independent_reference_metrics,
)


def _identity_affine() -> np.ndarray:
    return np.eye(4, dtype=np.float64)


def test_one_prediction_merging_two_gt_lesions_is_one_tp_one_fn() -> None:
    gt = np.zeros((10, 7, 7), dtype=bool)
    gt[1:3, 2:5, 2:5] = True
    gt[6:8, 2:5, 2:5] = True

    pred = np.zeros_like(gt)
    pred[1:8, 2:5, 2:5] = True

    affine = _identity_affine()
    project = compute_strict_case_metrics(
        case_identifier="fixture-merge",
        ground_truth_mask=gt,
        prediction_mask=pred,
        ground_truth_affine=affine,
        prediction_affine=affine,
    )
    reference = independent_reference_metrics(
        ground_truth_mask=gt,
        prediction_mask=pred,
        voxel_spacing_mm=(1.0, 1.0, 1.0),
    )

    assert reference.ground_truth_lesions == 2
    assert reference.predicted_lesions == 1
    assert reference.true_positive_lesions == 1
    assert reference.false_negative_lesions == 1
    assert reference.false_positive_lesions == 0
    assert project.true_positive_lesion_count == reference.true_positive_lesions
    assert project.false_negative_lesion_count == reference.false_negative_lesions
    assert project.false_positive_lesion_count == reference.false_positive_lesions


def test_two_predictions_splitting_one_gt_lesion_is_one_tp_one_fp() -> None:
    gt = np.zeros((10, 7, 7), dtype=bool)
    gt[1:8, 2:5, 2:5] = True

    pred = np.zeros_like(gt)
    pred[1:3, 2:5, 2:5] = True
    pred[6:8, 2:5, 2:5] = True

    affine = _identity_affine()
    project = compute_strict_case_metrics(
        case_identifier="fixture-split",
        ground_truth_mask=gt,
        prediction_mask=pred,
        ground_truth_affine=affine,
        prediction_affine=affine,
    )
    reference = independent_reference_metrics(
        ground_truth_mask=gt,
        prediction_mask=pred,
        voxel_spacing_mm=(1.0, 1.0, 1.0),
    )

    assert reference.ground_truth_lesions == 1
    assert reference.predicted_lesions == 2
    assert reference.true_positive_lesions == 1
    assert reference.false_negative_lesions == 0
    assert reference.false_positive_lesions == 1
    assert project.true_positive_lesion_count == reference.true_positive_lesions
    assert project.false_negative_lesion_count == reference.false_negative_lesions
    assert project.false_positive_lesion_count == reference.false_positive_lesions
