from __future__ import annotations

import numpy as np
import pytest

from protoem_ct.research_v2.r1_audit import R1GeometryError
from protoem_ct.research_v2.r2_metrics import compute_r2_native_case_metrics


def _identity_affine() -> np.ndarray:
    return np.eye(4, dtype=np.float64)


def test_r2_native_metrics_both_empty() -> None:
    empty = np.zeros((6, 6, 6), dtype=np.uint8)
    metrics = compute_r2_native_case_metrics(
        ground_truth_mask=empty,
        prediction_mask=empty,
        ground_truth_affine=_identity_affine(),
        prediction_affine=_identity_affine(),
    )
    assert metrics.tumor_dice == 1.0
    assert metrics.ground_truth_lesion_count == 0
    assert metrics.predicted_lesion_count == 0
    assert metrics.false_positive_lesion_count == 0
    assert metrics.predicted_tumor_volume_mm3 == 0.0


def test_r2_native_metrics_split_prediction_uses_one_to_one_matching() -> None:
    ground_truth = np.zeros((9, 9, 9), dtype=np.uint8)
    prediction = np.zeros_like(ground_truth)
    ground_truth[2:7, 2:7, 2:7] = 1
    prediction[2:4, 2:7, 2:7] = 1
    prediction[5:7, 2:7, 2:7] = 1

    metrics = compute_r2_native_case_metrics(
        ground_truth_mask=ground_truth,
        prediction_mask=prediction,
        ground_truth_affine=_identity_affine(),
        prediction_affine=_identity_affine(),
    )
    assert metrics.ground_truth_lesion_count == 1
    assert metrics.predicted_lesion_count == 2
    assert metrics.true_positive_lesion_count == 1
    assert metrics.false_positive_lesion_count == 1
    assert metrics.false_negative_lesion_count == 0


def test_r2_native_metrics_merged_prediction_uses_one_to_one_matching() -> None:
    ground_truth = np.zeros((10, 10, 10), dtype=np.uint8)
    prediction = np.zeros_like(ground_truth)
    ground_truth[2:4, 2:4, 2:4] = 1
    ground_truth[6:8, 6:8, 6:8] = 1
    prediction[2:8, 2:8, 2:8] = 1

    metrics = compute_r2_native_case_metrics(
        ground_truth_mask=ground_truth,
        prediction_mask=prediction,
        ground_truth_affine=_identity_affine(),
        prediction_affine=_identity_affine(),
    )
    assert metrics.ground_truth_lesion_count == 2
    assert metrics.predicted_lesion_count == 1
    assert metrics.true_positive_lesion_count == 1
    assert metrics.false_positive_lesion_count == 0
    assert metrics.false_negative_lesion_count == 1


def test_r2_native_metrics_reject_affine_mismatch_at_metric_grid() -> None:
    mask = np.zeros((4, 4, 4), dtype=np.uint8)
    prediction_affine = _identity_affine()
    prediction_affine[0, 3] = 2e-6
    with pytest.raises(R1GeometryError, match="affines do not match"):
        compute_r2_native_case_metrics(
            ground_truth_mask=mask,
            prediction_mask=mask,
            ground_truth_affine=_identity_affine(),
            prediction_affine=prediction_affine,
        )
