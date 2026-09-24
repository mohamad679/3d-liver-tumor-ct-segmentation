"""Research-v2 R1 pixel-to-metric audit contracts.

This module is deliberately separate from the immutable Phase 8 implementation. It contains
fail-closed access, label, geometry, transformation, and metric wrappers used by the R1 audit.
No function in this module trains a model or performs threshold selection.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final, cast

import nibabel as nib
import numpy as np
import numpy.typing as npt
from scipy import ndimage  # type: ignore[import-untyped]
from scipy.optimize import linear_sum_assignment  # type: ignore[import-untyped]

from protoem_ct.artifacts import DatasetManifest, DevelopmentSplitManifest
from protoem_ct.baselines.metrics import BaselineCaseMetrics, compute_baseline_case_metrics

R1_ALLOWED_PARTITIONS: Final[frozenset[str]] = frozenset({"train", "validation"})
R1_FORBIDDEN_PARTITIONS: Final[frozenset[str]] = frozenset({"internal_test", "external"})
R1_EXPECTED_TRAIN_CASES: Final[int] = 91
R1_EXPECTED_VALIDATION_CASES: Final[int] = 20
R1_EXPECTED_INTERNAL_TEST_CASES: Final[int] = 20
R1_ALLOWED_RAW_LABEL_VALUES: Final[tuple[int, ...]] = (0, 1, 2)
R1_TUMOR_RAW_LABEL_VALUE: Final[int] = 2
R1_AFFINE_ABS_TOLERANCE: Final[float] = 1e-6
R1_EXACT_METRIC_ABS_TOLERANCE: Final[float] = 1e-6
R1_SURFACE_METRIC_ABS_TOLERANCE: Final[float] = 1e-6
R1_NSD_TOLERANCE_MM: Final[float] = 1.0
R1_LESION_CONNECTIVITY: Final[int] = 26
R1_SURFACE_CONNECTIVITY: Final[int] = 6
R1_FIXTURE_PROBABILITY_THRESHOLD: Final[float] = 0.5

Array = npt.NDArray[np.generic]
BoolArray = npt.NDArray[np.bool_]
FloatArray = npt.NDArray[np.floating[Any]]


class R1AuditError(ValueError):
    """Base exception for Research-v2 R1 audit failures."""


class R1AccessBoundaryError(R1AuditError):
    """Raised when R1 code attempts to cross the train/validation array boundary."""


class R1LabelValidationError(R1AuditError):
    """Raised when a raw label array violates the LiTS R1 contract."""


class R1MetricInputError(R1AuditError):
    """Raised when metric input semantics are invalid or ambiguous."""


class R1GeometryError(R1AuditError):
    """Raised when image/prediction geometry cannot be interpreted safely."""


@dataclass(frozen=True, slots=True)
class R1AllowedCase:
    """One case authorized for array access during R1."""

    anonymous_patient_id: str
    anonymous_case_id: str
    partition: str
    relative_image_path: str
    relative_label_path: str
    image_sha256: str
    label_sha256: str


@dataclass(frozen=True, slots=True)
class R1ReferenceMetrics:
    """Independent reference result used to cross-check the project evaluator."""

    dice: float
    iou: float
    ground_truth_lesions: int
    predicted_lesions: int
    true_positive_lesions: int
    false_positive_lesions: int
    false_negative_lesions: int
    hd95_mm: float | None
    nsd: float


def authorized_r1_cases(
    manifest: DatasetManifest,
    split: DevelopmentSplitManifest,
) -> tuple[R1AllowedCase, ...]:
    """Return the 91 train + 20 validation cases without authorizing internal-test arrays."""

    if split.source_manifest_hash != manifest.manifest_hash:
        raise R1AccessBoundaryError("split source_manifest_hash does not match manifest_hash")
    cases_by_id = {case.anonymous_case_id: case for case in manifest.cases}
    assignments_by_id = {
        assignment.anonymous_case_id: assignment for assignment in split.assignments
    }
    if set(cases_by_id) != set(assignments_by_id):
        raise R1AccessBoundaryError("manifest and split case sets do not match exactly")

    allowed: list[R1AllowedCase] = []
    train_count = 0
    validation_count = 0
    internal_test_count = 0
    for case_id in sorted(cases_by_id):
        case = cases_by_id[case_id]
        assignment = assignments_by_id[case_id]
        if assignment.anonymous_patient_id != case.anonymous_patient_id:
            raise R1AccessBoundaryError("manifest/split anonymous patient identity mismatch")
        if assignment.partition == "internal_test":
            internal_test_count += 1
            continue
        require_r1_array_access(assignment.partition)
        if assignment.partition == "train":
            train_count += 1
        elif assignment.partition == "validation":
            validation_count += 1
        allowed.append(
            R1AllowedCase(
                anonymous_patient_id=case.anonymous_patient_id,
                anonymous_case_id=case.anonymous_case_id,
                partition=assignment.partition,
                relative_image_path=case.relative_image_path,
                relative_label_path=case.relative_label_path,
                image_sha256=case.image_sha256,
                label_sha256=case.label_sha256,
            )
        )

    if train_count != R1_EXPECTED_TRAIN_CASES:
        raise R1AccessBoundaryError(
            f"R1 requires exactly {R1_EXPECTED_TRAIN_CASES} train cases; observed {train_count}"
        )
    if validation_count != R1_EXPECTED_VALIDATION_CASES:
        raise R1AccessBoundaryError(
            "R1 requires exactly "
            f"{R1_EXPECTED_VALIDATION_CASES} validation cases; observed {validation_count}"
        )
    if internal_test_count != R1_EXPECTED_INTERNAL_TEST_CASES:
        raise R1AccessBoundaryError(
            "locked split must contain exactly "
            f"{R1_EXPECTED_INTERNAL_TEST_CASES} internal-test cases; observed {internal_test_count}"
        )
    if len(allowed) != R1_EXPECTED_TRAIN_CASES + R1_EXPECTED_VALIDATION_CASES:
        raise R1AccessBoundaryError("authorized R1 case count is not exactly 111")
    return tuple(allowed)


def require_r1_array_access(partition: str) -> None:
    """Fail closed unless an array belongs to train or validation."""

    if partition not in R1_ALLOWED_PARTITIONS:
        raise R1AccessBoundaryError(
            f"R1 array access is restricted to train/validation; rejected partition {partition!r}"
        )


def validate_raw_lits_label_array(label: Array) -> BoolArray:
    """Validate one raw LiTS-derived label and return tumor as raw label 2."""

    array = np.asarray(label)
    if array.ndim != 3:
        raise R1LabelValidationError("raw label array must be exactly 3D")
    if array.dtype.kind not in {"b", "i", "u", "f"}:
        raise R1LabelValidationError("raw label array must use a numeric dtype")
    if not bool(np.isfinite(array).all()):
        raise R1LabelValidationError("raw label array contains NaN or non-finite values")
    rounded = np.rint(array)
    if not bool(np.array_equal(array, rounded)):
        raise R1LabelValidationError("raw label array must be integer-valued")
    values = {int(value) for value in np.unique(rounded)}
    allowed = set(R1_ALLOWED_RAW_LABEL_VALUES)
    unexpected = values - allowed
    if unexpected:
        raise R1LabelValidationError(
            "raw label array contains values outside "
            f"{R1_ALLOWED_RAW_LABEL_VALUES}: {sorted(unexpected)}"
        )
    return cast(BoolArray, rounded == R1_TUMOR_RAW_LABEL_VALUE)


def validate_binary_metric_mask(mask: Array, *, field_name: str) -> BoolArray:
    """Require boolean/integer binary semantics; probability/logit arrays are rejected."""

    array = np.asarray(mask)
    if array.ndim != 3:
        raise R1MetricInputError(f"{field_name} must be exactly 3D")
    if array.dtype.kind == "f":
        raise R1MetricInputError(
            f"{field_name} uses floating-point values; threshold probability explicitly first"
        )
    if array.dtype.kind not in {"b", "i", "u"}:
        raise R1MetricInputError(f"{field_name} must use boolean or integer binary semantics")
    if not bool(np.isfinite(array).all()):
        raise R1MetricInputError(f"{field_name} contains NaN or non-finite values")
    values = {int(value) for value in np.unique(array)}
    if not values.issubset({0, 1}):
        raise R1MetricInputError(f"{field_name} must contain only binary values 0/1")
    return cast(BoolArray, array.astype(bool, copy=False))


def threshold_fixture_probability(probability: Array, *, threshold: float) -> BoolArray:
    """Threshold a synthetic fixture probability only at preregistered R1 value 0.5."""

    if threshold != R1_FIXTURE_PROBABILITY_THRESHOLD:
        raise R1MetricInputError(
            "R1 does not tune thresholds; synthetic fixture threshold must remain exactly 0.5"
        )
    array = np.asarray(probability)
    if array.ndim != 3 or array.dtype.kind != "f":
        raise R1MetricInputError("fixture probability must be a 3D floating-point array")
    float_array = cast(FloatArray, array)
    if not bool(np.isfinite(float_array).all()):
        raise R1MetricInputError("fixture probability contains NaN or non-finite values")
    if bool(np.any(float_array < 0.0)) or bool(np.any(float_array > 1.0)):
        raise R1MetricInputError(
            "fixture probability values must lie in [0, 1]; logits are rejected"
        )
    return float_array >= threshold


def affine_spacing_mm(affine: Array) -> tuple[float, float, float]:
    """Return voxel spacing from an affine after validating finite invertible geometry."""

    matrix = _validated_affine(affine, field_name="affine")
    spacing_array = np.sqrt(np.sum(matrix[:3, :3] ** 2, axis=0))
    if not bool(np.isfinite(spacing_array).all()) or bool(np.any(spacing_array <= 0.0)):
        raise R1GeometryError("affine implies non-finite or non-positive voxel spacing")
    return cast(tuple[float, float, float], tuple(float(v) for v in spacing_array))


def affine_orientation(affine: Array) -> tuple[str, str, str]:
    """Return nibabel axis codes after fail-closed affine validation."""

    matrix = _validated_affine(affine, field_name="affine")
    try:
        codes = nib.orientations.aff2axcodes(matrix)  # type: ignore[no-untyped-call]
    except (IndexError, ValueError) as exc:
        raise R1GeometryError("affine orientation could not be derived") from exc
    if len(codes) != 3 or any(code is None for code in codes):
        raise R1GeometryError("affine orientation is incomplete")
    return cast(tuple[str, str, str], tuple(str(code) for code in codes))


def validate_matching_geometry(
    *,
    ground_truth_shape: Sequence[int],
    prediction_shape: Sequence[int],
    ground_truth_affine: Array,
    prediction_affine: Array,
    affine_atol: float = R1_AFFINE_ABS_TOLERANCE,
) -> tuple[float, float, float]:
    """Require identical array grid geometry before metric computation."""

    gt_shape = _shape3(ground_truth_shape, field_name="ground_truth_shape")
    pred_shape = _shape3(prediction_shape, field_name="prediction_shape")
    if gt_shape != pred_shape:
        raise R1GeometryError("ground-truth and prediction shapes do not match")
    gt_affine = _validated_affine(ground_truth_affine, field_name="ground_truth_affine")
    pred_affine = _validated_affine(prediction_affine, field_name="prediction_affine")
    if not np.allclose(gt_affine, pred_affine, rtol=0.0, atol=affine_atol):
        raise R1GeometryError("ground-truth and prediction affines do not match")
    return affine_spacing_mm(gt_affine)


def compute_strict_case_metrics(
    *,
    case_identifier: str,
    ground_truth_mask: Array,
    prediction_mask: Array,
    ground_truth_affine: Array,
    prediction_affine: Array,
    nsd_tolerance_mm: float = R1_NSD_TOLERANCE_MM,
) -> BaselineCaseMetrics:
    """Fail-closed R1 wrapper around the existing project evaluator."""

    ground_truth = validate_binary_metric_mask(ground_truth_mask, field_name="ground_truth_mask")
    prediction = validate_binary_metric_mask(prediction_mask, field_name="prediction_mask")
    spacing = validate_matching_geometry(
        ground_truth_shape=ground_truth.shape,
        prediction_shape=prediction.shape,
        ground_truth_affine=ground_truth_affine,
        prediction_affine=prediction_affine,
    )
    if nsd_tolerance_mm != R1_NSD_TOLERANCE_MM:
        raise R1MetricInputError("R1 NSD tolerance is preregistered at exactly 1.0 mm")
    return compute_baseline_case_metrics(
        case_identifier=case_identifier,
        ground_truth_mask=ground_truth,
        prediction_mask=prediction,
        voxel_spacing_mm=spacing,
        nsd_tolerance_mm=nsd_tolerance_mm,
    )


def independent_reference_metrics(
    *,
    ground_truth_mask: Array,
    prediction_mask: Array,
    voxel_spacing_mm: Sequence[float],
    nsd_tolerance_mm: float = R1_NSD_TOLERANCE_MM,
) -> R1ReferenceMetrics:
    """Compute R1 fixture metrics without calling the project baseline evaluator."""

    ground_truth = validate_binary_metric_mask(ground_truth_mask, field_name="ground_truth_mask")
    prediction = validate_binary_metric_mask(prediction_mask, field_name="prediction_mask")
    if ground_truth.shape != prediction.shape:
        raise R1GeometryError("reference metric masks must have identical shapes")
    spacing = _spacing3(voxel_spacing_mm)
    if nsd_tolerance_mm != R1_NSD_TOLERANCE_MM:
        raise R1MetricInputError("R1 NSD tolerance is preregistered at exactly 1.0 mm")

    gt_count = int(np.count_nonzero(ground_truth))
    pred_count = int(np.count_nonzero(prediction))
    intersection = int(np.count_nonzero(ground_truth & prediction))
    union = int(np.count_nonzero(ground_truth | prediction))
    dice = 1.0 if gt_count + pred_count == 0 else 2.0 * intersection / (gt_count + pred_count)
    iou = 1.0 if union == 0 else intersection / union

    structure = np.ones((3, 3, 3), dtype=np.uint8)
    gt_labels, gt_lesions = ndimage.label(ground_truth, structure=structure)
    pred_labels, pred_lesions = ndimage.label(prediction, structure=structure)
    tp = _independent_overlap_matching(
        gt_labels=cast(npt.NDArray[np.integer[Any]], gt_labels),
        gt_count=int(gt_lesions),
        pred_labels=cast(npt.NDArray[np.integer[Any]], pred_labels),
        pred_count=int(pred_lesions),
    )
    fp = int(pred_lesions) - tp
    fn = int(gt_lesions) - tp
    hd95_mm, nsd = _independent_surface_metrics(
        ground_truth=ground_truth,
        prediction=prediction,
        spacing=spacing,
        tolerance_mm=nsd_tolerance_mm,
    )
    return R1ReferenceMetrics(
        dice=float(dice),
        iou=float(iou),
        ground_truth_lesions=int(gt_lesions),
        predicted_lesions=int(pred_lesions),
        true_positive_lesions=tp,
        false_positive_lesions=fp,
        false_negative_lesions=fn,
        hd95_mm=hd95_mm,
        nsd=nsd,
    )


def reorient_array_to_ras(
    array: Array,
    affine: Array,
) -> tuple[Array, npt.NDArray[np.float64]]:
    """Reorient a 3D array to RAS while preserving world coordinates."""

    data = np.asarray(array)
    if data.ndim != 3:
        raise R1GeometryError("reorientation requires an exactly 3D array")
    matrix = _validated_affine(affine, field_name="affine")
    source_orientation = nib.orientations.io_orientation(matrix)  # type: ignore[no-untyped-call]
    target_orientation = nib.orientations.axcodes2ornt(  # type: ignore[no-untyped-call]
        ("R", "A", "S")
    )
    transform = nib.orientations.ornt_transform(  # type: ignore[no-untyped-call]
        source_orientation, target_orientation
    )
    reoriented = nib.orientations.apply_orientation(  # type: ignore[no-untyped-call]
        data, transform
    )
    reoriented_affine = matrix @ nib.orientations.inv_ornt_aff(  # type: ignore[no-untyped-call]
        transform, data.shape
    )
    if affine_orientation(reoriented_affine) != ("R", "A", "S"):
        raise R1GeometryError("reorientation did not produce RAS axis codes")
    return np.asarray(reoriented), np.asarray(reoriented_affine, dtype=np.float64)


def grid_for_spacing(
    *,
    shape: Sequence[int],
    affine: Array,
    target_spacing_mm: Sequence[float],
) -> tuple[tuple[int, int, int], npt.NDArray[np.float64]]:
    """Construct a same-origin/same-direction grid at explicit spacing."""

    source_shape = _shape3(shape, field_name="shape")
    matrix = _validated_affine(affine, field_name="affine")
    source_spacing = affine_spacing_mm(matrix)
    target_spacing = _spacing3(target_spacing_mm)
    output_shape = cast(
        tuple[int, int, int],
        tuple(
            max(1, int(round((source_shape[i] - 1) * source_spacing[i] / target_spacing[i])) + 1)
            for i in range(3)
        ),
    )
    output_affine = matrix.copy()
    for axis in range(3):
        direction = matrix[:3, axis] / source_spacing[axis]
        output_affine[:3, axis] = direction * target_spacing[axis]
    return output_shape, output_affine


def resample_array_to_grid(
    array: Array,
    *,
    source_affine: Array,
    target_shape: Sequence[int],
    target_affine: Array,
    is_label_or_mask: bool,
) -> Array:
    """Affine-aware resampling with nearest for masks and linear for image/probability."""

    data = np.asarray(array)
    if data.ndim != 3:
        raise R1GeometryError("resampling requires an exactly 3D array")
    source_matrix = _validated_affine(source_affine, field_name="source_affine")
    output_matrix = _validated_affine(target_affine, field_name="target_affine")
    output_shape = _shape3(target_shape, field_name="target_shape")
    output_to_input = np.linalg.inv(source_matrix) @ output_matrix
    order = 0 if is_label_or_mask else 1
    result = ndimage.affine_transform(
        data,
        matrix=output_to_input[:3, :3],
        offset=output_to_input[:3, 3],
        output_shape=output_shape,
        order=order,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    if is_label_or_mask and data.dtype.kind in {"b", "i", "u"}:
        result = result.astype(data.dtype, copy=False)
    return np.asarray(result)


def restore_crop_to_full_grid(
    crop: Array,
    *,
    full_shape: Sequence[int],
    starts: Sequence[int],
) -> Array:
    """Restore one prediction crop into its full preprocessed grid without interpolation."""

    crop_array = np.asarray(crop)
    if crop_array.ndim != 3:
        raise R1GeometryError("prediction crop must be exactly 3D")
    target_shape = _shape3(full_shape, field_name="full_shape")
    start_tuple = _index3(starts, field_name="starts")
    stops = tuple(start_tuple[i] + int(crop_array.shape[i]) for i in range(3))
    if any(start_tuple[i] < 0 or stops[i] > target_shape[i] for i in range(3)):
        raise R1GeometryError("prediction crop lies outside the full preprocessed grid")
    restored = np.zeros(target_shape, dtype=crop_array.dtype)
    restored[
        start_tuple[0] : stops[0],
        start_tuple[1] : stops[1],
        start_tuple[2] : stops[2],
    ] = crop_array
    return np.asarray(restored)


def _validated_affine(affine: Array, *, field_name: str) -> npt.NDArray[np.float64]:
    matrix = np.asarray(affine, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise R1GeometryError(f"{field_name} must have shape (4, 4)")
    if not bool(np.isfinite(matrix).all()):
        raise R1GeometryError(f"{field_name} contains NaN or non-finite values")
    expected_row = np.array([0.0, 0.0, 0.0, 1.0])
    if not np.allclose(matrix[3], expected_row, rtol=0.0, atol=1e-9):
        raise R1GeometryError(f"{field_name} has an invalid homogeneous row")
    determinant = float(np.linalg.det(matrix[:3, :3]))
    if not math.isfinite(determinant) or abs(determinant) <= 1e-12:
        raise R1GeometryError(f"{field_name} is singular or degenerate")
    return matrix


def _shape3(shape: Sequence[int], *, field_name: str) -> tuple[int, int, int]:
    if len(shape) != 3:
        raise R1GeometryError(f"{field_name} must contain exactly three dimensions")
    values = tuple(int(value) for value in shape)
    if any(value <= 0 for value in values):
        raise R1GeometryError(f"{field_name} dimensions must be positive")
    return cast(tuple[int, int, int], values)


def _index3(values: Sequence[int], *, field_name: str) -> tuple[int, int, int]:
    if len(values) != 3:
        raise R1GeometryError(f"{field_name} must contain exactly three indices")
    return cast(tuple[int, int, int], tuple(int(value) for value in values))


def _spacing3(values: Sequence[float]) -> tuple[float, float, float]:
    if len(values) != 3:
        raise R1GeometryError("voxel spacing must contain exactly three values")
    spacing = tuple(float(value) for value in values)
    if any(not math.isfinite(value) or value <= 0.0 for value in spacing):
        raise R1GeometryError("voxel spacing values must be finite and positive")
    return cast(tuple[float, float, float], spacing)


def _independent_overlap_matching(
    *,
    gt_labels: npt.NDArray[np.integer[Any]],
    gt_count: int,
    pred_labels: npt.NDArray[np.integer[Any]],
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


def _independent_surface_metrics(
    *,
    ground_truth: BoolArray,
    prediction: BoolArray,
    spacing: tuple[float, float, float],
    tolerance_mm: float,
) -> tuple[float | None, float]:
    gt_empty = not bool(np.any(ground_truth))
    pred_empty = not bool(np.any(prediction))
    if gt_empty and pred_empty:
        return 0.0, 1.0
    if gt_empty or pred_empty:
        return None, 0.0

    surface_structure = ndimage.generate_binary_structure(rank=3, connectivity=1)
    gt_surface = ground_truth & ~ndimage.binary_erosion(
        ground_truth, structure=surface_structure, border_value=0
    )
    pred_surface = prediction & ~ndimage.binary_erosion(
        prediction, structure=surface_structure, border_value=0
    )
    gt_distance = ndimage.distance_transform_edt(~gt_surface, sampling=np.asarray(spacing))
    pred_distance = ndimage.distance_transform_edt(~pred_surface, sampling=np.asarray(spacing))
    gt_to_pred = np.asarray(pred_distance[gt_surface], dtype=np.float64)
    pred_to_gt = np.asarray(gt_distance[pred_surface], dtype=np.float64)
    distances = np.concatenate((gt_to_pred, pred_to_gt))
    if distances.size == 0 or not bool(np.isfinite(distances).all()):
        raise R1MetricInputError("surface-distance reference produced invalid distances")
    hd95 = float(np.percentile(distances, 95))
    nsd = float(np.count_nonzero(distances <= tolerance_mm) / distances.size)
    return hd95, nsd
