"""Deterministic binary calibration evaluation for Phase 7."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from protoem_ct.artifacts.hashing import sha256_json
from protoem_ct.uncertainty.artifacts import (
    PHASE7_CALIBRATION_BIN_SCHEMA_NAME,
    PHASE7_CALIBRATION_BIN_SCHEMA_VERSION,
    PHASE7_CALIBRATION_RESULT_SCHEMA_NAME,
    PHASE7_CALIBRATION_RESULT_SCHEMA_VERSION,
    Phase7CalibrationBin,
    Phase7CalibrationResult,
)
from protoem_ct.uncertainty.contracts import (
    BinaryPredictiveProbabilityMap,
    array_content_sha256,
)


class CalibrationEvaluationError(ValueError):
    """Raised when deterministic calibration evaluation cannot be computed."""


@dataclass(frozen=True, slots=True)
class CalibrationComputationResult:
    """Pure in-memory result for one binary calibration calculation."""

    calibration_result: Phase7CalibrationResult
    confidence_map: np.ndarray
    prediction_map: np.ndarray
    correctness_map: np.ndarray
    prediction_content_hash: str
    probability_content_hash: str
    reference_mask_content_hash: str


def compute_binary_calibration_ece(
    probability_map: BinaryPredictiveProbabilityMap,
    reference_mask: np.ndarray,
    *,
    bin_count: int = 10,
    common_grid_geometry_record_hash: str | None = None,
) -> CalibrationComputationResult:
    """Compute fixed-bin ECE on an already validated common grid.

    Confidence is ``max(p_fg, p_bg)``. The hard prediction is foreground only
    when ``p_fg > p_bg``; exact ties resolve to background. Fixed equal-width
    bins span ``[0, 1]``. Bins are half-open ``[lower, upper)`` except the final
    bin, which is closed on its upper edge, so a confidence exactly on an
    interior boundary belongs to the higher-index bin.
    """

    _revalidate_probability_map(probability_map)
    if not isinstance(bin_count, int) or bin_count <= 0:
        raise CalibrationEvaluationError("bin_count must be a positive integer.")

    resolved_common_grid_hash = _resolve_common_grid_hash(
        probability_map=probability_map,
        common_grid_geometry_record_hash=common_grid_geometry_record_hash,
    )
    reference = _require_binary_reference_mask(
        reference_mask,
        expected_shape=probability_map.probability_shape,
    )
    foreground = np.ascontiguousarray(probability_map.foreground_probability_map, dtype=np.float64)
    background = np.ascontiguousarray(probability_map.background_probability_map, dtype=np.float64)
    confidence = np.ascontiguousarray(np.maximum(foreground, background), dtype=np.float64)
    prediction = np.ascontiguousarray(foreground > background, dtype=np.bool_)
    correctness = np.ascontiguousarray(prediction == reference, dtype=np.bool_)

    valid_voxel_count = int(reference.size)
    bins = _build_calibration_bins(
        confidence=confidence.reshape(-1),
        correctness=correctness.reshape(-1),
        bin_count=bin_count,
        valid_voxel_count=valid_voxel_count,
    )
    ece = float(sum(bin_record.weighted_error for bin_record in bins))
    prediction_hash = array_content_sha256(prediction)
    probability_hash = _combined_probability_content_hash(probability_map)
    reference_hash = array_content_sha256(reference)
    calibration_result = _build_available_calibration_result(
        prediction_content_hash=prediction_hash,
        probability_content_hash=probability_hash,
        reference_mask_content_hash=reference_hash,
        common_grid_geometry_record_hash=resolved_common_grid_hash,
        bin_count=bin_count,
        valid_voxel_count=valid_voxel_count,
        ece=ece,
        bins=bins,
    )
    return CalibrationComputationResult(
        calibration_result=calibration_result,
        confidence_map=confidence,
        prediction_map=prediction,
        correctness_map=correctness,
        prediction_content_hash=prediction_hash,
        probability_content_hash=probability_hash,
        reference_mask_content_hash=reference_hash,
    )


def query_label_not_part_of_calibration_api() -> bool:
    """Return True because calibration accepts references only for evaluation."""

    return True


def _build_calibration_bins(
    *,
    confidence: np.ndarray,
    correctness: np.ndarray,
    bin_count: int,
    valid_voxel_count: int,
) -> tuple[Phase7CalibrationBin, ...]:
    edges = np.linspace(0.0, 1.0, bin_count + 1, dtype=np.float64)
    assigned_bins = np.minimum(
        np.floor(confidence * float(bin_count)).astype(np.int64),
        bin_count - 1,
    )
    records: list[Phase7CalibrationBin] = []
    for bin_index in range(bin_count):
        selected = assigned_bins == bin_index
        voxel_count = int(np.count_nonzero(selected))
        if voxel_count == 0:
            accuracy = None
            mean_confidence = None
            weighted_error = 0.0
        else:
            accuracy = float(np.mean(correctness[selected], dtype=np.float64))
            mean_confidence = float(np.mean(confidence[selected], dtype=np.float64))
            weighted_error = (voxel_count / valid_voxel_count) * abs(accuracy - mean_confidence)
        records.append(
            Phase7CalibrationBin(
                schema_name=PHASE7_CALIBRATION_BIN_SCHEMA_NAME,
                schema_version=PHASE7_CALIBRATION_BIN_SCHEMA_VERSION,
                bin_index=bin_index,
                bin_lower=float(edges[bin_index]),
                bin_upper=float(edges[bin_index + 1]),
                upper_inclusive=bin_index == bin_count - 1,
                voxel_count=voxel_count,
                accuracy=accuracy,
                mean_confidence=mean_confidence,
                weighted_error=float(weighted_error),
            )
        )
    return tuple(records)


def _build_available_calibration_result(
    *,
    prediction_content_hash: str,
    probability_content_hash: str,
    reference_mask_content_hash: str,
    common_grid_geometry_record_hash: str,
    bin_count: int,
    valid_voxel_count: int,
    ece: float,
    bins: tuple[Phase7CalibrationBin, ...],
) -> Phase7CalibrationResult:
    payload = {
        "schema_name": PHASE7_CALIBRATION_RESULT_SCHEMA_NAME,
        "schema_version": PHASE7_CALIBRATION_RESULT_SCHEMA_VERSION,
        "prediction_content_hash": prediction_content_hash,
        "probability_content_hash": probability_content_hash,
        "reference_mask_content_hash": reference_mask_content_hash,
        "common_grid_geometry_record_hash": common_grid_geometry_record_hash,
        "calibration_metric": "ece",
        "confidence_definition": "max_binary_probability",
        "binning_policy": "fixed_equal_width",
        "bin_count": bin_count,
        "valid_voxel_count": valid_voxel_count,
        "ece": ece,
        "bins": [
            {
                "schema_name": item.schema_name,
                "schema_version": item.schema_version,
                "bin_index": item.bin_index,
                "bin_lower": item.bin_lower,
                "bin_upper": item.bin_upper,
                "upper_inclusive": item.upper_inclusive,
                "voxel_count": item.voxel_count,
                "accuracy": item.accuracy,
                "mean_confidence": item.mean_confidence,
                "weighted_error": item.weighted_error,
            }
            for item in bins
        ],
        "availability_status": "available",
        "unavailable_reason": None,
    }
    return Phase7CalibrationResult(
        schema_name=PHASE7_CALIBRATION_RESULT_SCHEMA_NAME,
        schema_version=PHASE7_CALIBRATION_RESULT_SCHEMA_VERSION,
        calibration_result_hash=sha256_json(payload),
        prediction_content_hash=prediction_content_hash,
        probability_content_hash=probability_content_hash,
        reference_mask_content_hash=reference_mask_content_hash,
        common_grid_geometry_record_hash=common_grid_geometry_record_hash,
        calibration_metric="ece",
        confidence_definition="max_binary_probability",
        binning_policy="fixed_equal_width",
        bin_count=bin_count,
        valid_voxel_count=valid_voxel_count,
        ece=ece,
        bins=bins,
        availability_status="available",
        unavailable_reason=None,
    )


def _combined_probability_content_hash(probability_map: BinaryPredictiveProbabilityMap) -> str:
    return sha256_json(
        {
            "background_probability_content_hash": (
                probability_map.background_probability_content_hash
            ),
            "foreground_probability_content_hash": (
                probability_map.foreground_probability_content_hash
            ),
            "probability_shape": list(probability_map.probability_shape),
        }
    )


def _resolve_common_grid_hash(
    *,
    probability_map: BinaryPredictiveProbabilityMap,
    common_grid_geometry_record_hash: str | None,
) -> str:
    probability_grid_hash = probability_map.common_grid_identity_hash
    if probability_grid_hash is None and common_grid_geometry_record_hash is None:
        raise CalibrationEvaluationError(
            "common_grid_geometry_record_hash is required for calibration evaluation."
        )
    if (
        probability_grid_hash is not None
        and common_grid_geometry_record_hash is not None
        and probability_grid_hash != common_grid_geometry_record_hash
    ):
        raise CalibrationEvaluationError(
            "common_grid_geometry_record_hash must match probability_map common grid."
        )
    return common_grid_geometry_record_hash or probability_grid_hash or ""


def _require_binary_reference_mask(
    reference_mask: np.ndarray,
    *,
    expected_shape: tuple[int, int, int, int, int],
) -> np.ndarray:
    value = np.asarray(reference_mask)
    if value.shape != expected_shape:
        raise CalibrationEvaluationError("reference_mask must match probability shape [1,1,D,H,W].")
    if value.dtype.kind not in {"b", "u", "i", "f"}:
        raise CalibrationEvaluationError("reference_mask must be numeric or boolean.")
    contiguous = np.ascontiguousarray(value)
    if not np.all(np.isfinite(contiguous)):
        raise CalibrationEvaluationError("reference_mask must contain only finite values.")
    if not np.all(np.logical_or(contiguous == 0, contiguous == 1)):
        raise CalibrationEvaluationError("reference_mask must be binary with values 0 or 1.")
    return np.ascontiguousarray(contiguous.astype(np.bool_, copy=False))


def _revalidate_probability_map(probability_map: BinaryPredictiveProbabilityMap) -> None:
    BinaryPredictiveProbabilityMap(
        schema_name=probability_map.schema_name,
        schema_version=probability_map.schema_version,
        probability_map_identity_hash=probability_map.probability_map_identity_hash,
        foreground_probability_map=probability_map.foreground_probability_map,
        background_probability_map=probability_map.background_probability_map,
        foreground_probability_content_hash=probability_map.foreground_probability_content_hash,
        background_probability_content_hash=probability_map.background_probability_content_hash,
        probability_shape=probability_map.probability_shape,
        probability_sum_tolerance=probability_map.probability_sum_tolerance,
        source_prediction_identity_hash=probability_map.source_prediction_identity_hash,
        common_grid_identity_hash=probability_map.common_grid_identity_hash,
    )
