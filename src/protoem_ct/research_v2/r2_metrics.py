"""Native-grid metric subset for the Research-v2 R2 real-train overfit diagnostic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from scipy import ndimage  # type: ignore[import-untyped]
from scipy.optimize import linear_sum_assignment  # type: ignore[import-untyped]

from protoem_ct.research_v2.r1_audit import (
    R1_AFFINE_ABS_TOLERANCE,
    validate_binary_metric_mask,
    validate_matching_geometry,
)

Array = npt.NDArray[np.generic]
IntArray = npt.NDArray[np.integer[Any]]


@dataclass(frozen=True, slots=True)
class R2NativeCaseMetrics:
    """Gate-relevant native-grid metrics without unnecessary surface-distance allocation."""

    tumor_dice: float
    ground_truth_tumor_voxels: int
    predicted_tumor_voxels: int
    ground_truth_tumor_volume_mm3: float
    predicted_tumor_volume_mm3: float
    ground_truth_lesion_count: int
    predicted_lesion_count: int
    true_positive_lesion_count: int
    false_positive_lesion_count: int
    false_negative_lesion_count: int


def _maximum_overlap_true_positives(
    *,
    gt_labels: IntArray,
    gt_count: int,
    pred_labels: IntArray,
    pred_count: int,
) -> int:
    if gt_count == 0 or pred_count == 0:
        return 0

    overlaps = np.zeros((gt_count, pred_count), dtype=np.int64)
    for gt_index in range(1, gt_count + 1):
        gt_region = gt_labels == gt_index
        touched_predictions = np.unique(pred_labels[gt_region])
        for pred_index_raw in touched_predictions:
            pred_index = int(pred_index_raw)
            if pred_index == 0:
                continue
            overlaps[gt_index - 1, pred_index - 1] = int(
                np.count_nonzero(gt_region & (pred_labels == pred_index))
            )

    max_overlap = int(overlaps.max(initial=0))
    size = max(gt_count, pred_count)
    big = size * size + 1
    cost = np.full((size, size), max_overlap * big + size * size, dtype=np.int64)
    for row in range(gt_count):
        for col in range(pred_count):
            tie_break = row * size + col
            cost[row, col] = (max_overlap - int(overlaps[row, col])) * big + tie_break

    row_indices, col_indices = linear_sum_assignment(cost)
    matches = 0
    for row, col in zip(row_indices, col_indices, strict=True):
        if row < gt_count and col < pred_count and overlaps[row, col] > 0:
            matches += 1
    return matches


def compute_r2_native_case_metrics(
    *,
    ground_truth_mask: Array,
    prediction_mask: Array,
    ground_truth_affine: Array,
    prediction_affine: Array,
) -> R2NativeCaseMetrics:
    """Compute the R2 Gate subset on an exactly matched native grid."""

    ground_truth = validate_binary_metric_mask(
        ground_truth_mask,
        field_name="ground_truth_mask",
    )
    prediction = validate_binary_metric_mask(
        prediction_mask,
        field_name="prediction_mask",
    )
    spacing = validate_matching_geometry(
        ground_truth_shape=ground_truth.shape,
        prediction_shape=prediction.shape,
        ground_truth_affine=ground_truth_affine,
        prediction_affine=prediction_affine,
        affine_atol=R1_AFFINE_ABS_TOLERANCE,
    )

    gt_voxels = int(np.count_nonzero(ground_truth))
    pred_voxels = int(np.count_nonzero(prediction))
    denominator = gt_voxels + pred_voxels
    if denominator == 0:
        dice = 1.0
    else:
        intersection = int(np.count_nonzero(ground_truth & prediction))
        dice = float(2.0 * intersection / denominator)

    structure = np.ones((3, 3, 3), dtype=np.uint8)
    gt_labels_raw, gt_lesions_raw = ndimage.label(ground_truth, structure=structure)
    pred_labels_raw, pred_lesions_raw = ndimage.label(prediction, structure=structure)
    gt_lesions = int(gt_lesions_raw)
    pred_lesions = int(pred_lesions_raw)
    true_positive = _maximum_overlap_true_positives(
        gt_labels=cast(IntArray, gt_labels_raw),
        gt_count=gt_lesions,
        pred_labels=cast(IntArray, pred_labels_raw),
        pred_count=pred_lesions,
    )
    false_positive = pred_lesions - true_positive
    false_negative = gt_lesions - true_positive
    voxel_volume_mm3 = float(spacing[0] * spacing[1] * spacing[2])

    return R2NativeCaseMetrics(
        tumor_dice=dice,
        ground_truth_tumor_voxels=gt_voxels,
        predicted_tumor_voxels=pred_voxels,
        ground_truth_tumor_volume_mm3=float(gt_voxels * voxel_volume_mm3),
        predicted_tumor_volume_mm3=float(pred_voxels * voxel_volume_mm3),
        ground_truth_lesion_count=gt_lesions,
        predicted_lesion_count=pred_lesions,
        true_positive_lesion_count=true_positive,
        false_positive_lesion_count=false_positive,
        false_negative_lesion_count=false_negative,
    )


__all__ = ["R2NativeCaseMetrics", "compute_r2_native_case_metrics"]
