"""Deterministic voxel-level risk-coverage evaluation for Phase 7."""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from typing import Final

import numpy as np

from protoem_ct.artifacts.hashing import sha256_json
from protoem_ct.uncertainty.artifacts import (
    PHASE7_RISK_COVERAGE_POINT_SCHEMA_NAME,
    PHASE7_RISK_COVERAGE_POINT_SCHEMA_VERSION,
    PHASE7_RISK_COVERAGE_RESULT_SCHEMA_NAME,
    PHASE7_RISK_COVERAGE_RESULT_SCHEMA_VERSION,
    Phase7RiskCoveragePoint,
    Phase7RiskCoverageResult,
)
from protoem_ct.uncertainty.contracts import array_content_sha256

_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")
_VOLUME_SHAPE_RANK: Final[int] = 5


class RiskCoverageEvaluationError(ValueError):
    """Raised when deterministic risk-coverage evaluation cannot be completed."""


class RiskCoverageInputError(RiskCoverageEvaluationError):
    """Raised when risk-coverage inputs violate the evaluation contract."""


@dataclass(frozen=True, slots=True)
class RiskCoverageComputationResult:
    """Pure in-memory result for one deterministic risk-coverage curve."""

    risk_coverage_result: Phase7RiskCoverageResult
    retained_indices: tuple[int, ...]
    error_indicators: np.ndarray
    valid_voxel_mask: np.ndarray


def compute_risk_coverage(
    *,
    reference_mask: np.ndarray,
    uncertainty_map: np.ndarray,
    uncertainty_result_hash: str,
    common_grid_geometry_record_hash: str,
    prediction_map: np.ndarray | None = None,
    foreground_probability_map: np.ndarray | None = None,
    background_probability_map: np.ndarray | None = None,
    valid_voxel_mask: np.ndarray | None = None,
) -> RiskCoverageComputationResult:
    """Compute every-prefix risk coverage on an already aligned common grid.

    The public contract accepts either a supplied binary ``prediction_map`` or a
    foreground/background probability pair. Probability-derived hard prediction
    uses ``foreground_probability > background_probability``; exact ties resolve
    to background. Risk is the mean binary error
    ``prediction != reference_mask`` among the retained prefix after ordering
    valid voxels by ascending uncertainty, with C-order flattened index as the
    deterministic tie-break.
    """

    prediction = _resolve_prediction(
        prediction_map=prediction_map,
        foreground_probability_map=foreground_probability_map,
        background_probability_map=background_probability_map,
    )
    reference = _require_binary_volume(reference_mask, field_name="reference_mask")
    uncertainty = _require_uncertainty_volume(uncertainty_map)
    _require_same_shape(prediction, reference, uncertainty)
    mask = _resolve_valid_voxel_mask(valid_voxel_mask, shape=prediction.shape)
    _require_sha256(uncertainty_result_hash, field_name="uncertainty_result_hash")
    _require_sha256(
        common_grid_geometry_record_hash,
        field_name="common_grid_geometry_record_hash",
    )

    prediction_hash = array_content_sha256(prediction)
    reference_hash = array_content_sha256(reference)
    flat_valid = mask.reshape(-1, order="C")
    valid_voxel_count = int(np.count_nonzero(flat_valid))
    if valid_voxel_count == 0:
        result = _build_risk_coverage_result(
            uncertainty_result_hash=uncertainty_result_hash,
            prediction_content_hash=prediction_hash,
            reference_mask_content_hash=reference_hash,
            common_grid_geometry_record_hash=common_grid_geometry_record_hash,
            valid_voxel_count=0,
            points=(),
            availability_status="unavailable",
            unavailable_reason="no_valid_voxels",
        )
        return RiskCoverageComputationResult(
            risk_coverage_result=result,
            retained_indices=(),
            error_indicators=np.asarray([], dtype=np.uint8),
            valid_voxel_mask=mask,
        )

    flat_uncertainty = uncertainty.reshape(-1, order="C")
    valid_indices = np.flatnonzero(flat_valid)
    ordered_position = np.lexsort((valid_indices, flat_uncertainty[valid_indices]))
    retained_indices_array = valid_indices[ordered_position]
    flat_prediction = prediction.reshape(-1, order="C")
    flat_reference = reference.reshape(-1, order="C")
    errors = flat_prediction[retained_indices_array] != flat_reference[retained_indices_array]
    cumulative_errors = np.cumsum(errors.astype(np.float64))

    points = tuple(
        Phase7RiskCoveragePoint(
            schema_name=PHASE7_RISK_COVERAGE_POINT_SCHEMA_NAME,
            schema_version=PHASE7_RISK_COVERAGE_POINT_SCHEMA_VERSION,
            retained_voxel_count=index + 1,
            coverage=float((index + 1) / valid_voxel_count),
            risk=float(cumulative_errors[index] / (index + 1)),
        )
        for index in range(valid_voxel_count)
    )
    result = _build_risk_coverage_result(
        uncertainty_result_hash=uncertainty_result_hash,
        prediction_content_hash=prediction_hash,
        reference_mask_content_hash=reference_hash,
        common_grid_geometry_record_hash=common_grid_geometry_record_hash,
        valid_voxel_count=valid_voxel_count,
        points=points,
        availability_status="available",
        unavailable_reason=None,
    )
    return RiskCoverageComputationResult(
        risk_coverage_result=result,
        retained_indices=tuple(int(index) for index in retained_indices_array),
        error_indicators=np.ascontiguousarray(errors.astype(np.uint8)),
        valid_voxel_mask=mask,
    )


def derive_hard_prediction_from_binary_probabilities(
    *,
    foreground_probability_map: np.ndarray,
    background_probability_map: np.ndarray,
) -> np.ndarray:
    """Derive binary prediction from probabilities with ties resolving to background."""

    foreground = _require_probability_volume(
        foreground_probability_map,
        field_name="foreground_probability_map",
    )
    background = _require_probability_volume(
        background_probability_map,
        field_name="background_probability_map",
    )
    if foreground.shape != background.shape:
        raise RiskCoverageInputError("foreground/background probability maps must share shape.")
    return np.ascontiguousarray((foreground > background).astype(np.uint8))


def query_label_not_part_of_risk_coverage_prediction_api() -> bool:
    """Return True because only this evaluation module accepts a reference mask."""

    public_callables = (
        compute_risk_coverage,
        derive_hard_prediction_from_binary_probabilities,
    )
    forbidden = {"query_label", "query_labels", "label_map", "query_reference_mask"}
    return all(
        forbidden.isdisjoint(inspect.signature(callable_item).parameters)
        for callable_item in public_callables
    )


def _build_risk_coverage_result(
    *,
    uncertainty_result_hash: str,
    prediction_content_hash: str,
    reference_mask_content_hash: str,
    common_grid_geometry_record_hash: str,
    valid_voxel_count: int,
    points: tuple[Phase7RiskCoveragePoint, ...],
    availability_status: str,
    unavailable_reason: str | None,
) -> Phase7RiskCoverageResult:
    payload = {
        "schema_name": PHASE7_RISK_COVERAGE_RESULT_SCHEMA_NAME,
        "schema_version": PHASE7_RISK_COVERAGE_RESULT_SCHEMA_VERSION,
        "uncertainty_result_hash": uncertainty_result_hash,
        "prediction_content_hash": prediction_content_hash,
        "reference_mask_content_hash": reference_mask_content_hash,
        "common_grid_geometry_record_hash": common_grid_geometry_record_hash,
        "ordering": "low_uncertainty_first",
        "tie_break_policy": "flattened_index",
        "valid_voxel_count": valid_voxel_count,
        "points": [
            {
                "schema_name": point.schema_name,
                "schema_version": point.schema_version,
                "retained_voxel_count": point.retained_voxel_count,
                "coverage": point.coverage,
                "risk": point.risk,
            }
            for point in points
        ],
        "availability_status": availability_status,
        "unavailable_reason": unavailable_reason,
    }
    return Phase7RiskCoverageResult(
        schema_name=PHASE7_RISK_COVERAGE_RESULT_SCHEMA_NAME,
        schema_version=PHASE7_RISK_COVERAGE_RESULT_SCHEMA_VERSION,
        risk_coverage_result_hash=sha256_json(payload),
        uncertainty_result_hash=uncertainty_result_hash,
        prediction_content_hash=prediction_content_hash,
        reference_mask_content_hash=reference_mask_content_hash,
        common_grid_geometry_record_hash=common_grid_geometry_record_hash,
        ordering="low_uncertainty_first",
        tie_break_policy="flattened_index",
        valid_voxel_count=valid_voxel_count,
        points=points,
        availability_status=availability_status,
        unavailable_reason=unavailable_reason,
    )


def _resolve_prediction(
    *,
    prediction_map: np.ndarray | None,
    foreground_probability_map: np.ndarray | None,
    background_probability_map: np.ndarray | None,
) -> np.ndarray:
    has_prediction = prediction_map is not None
    has_probabilities = (
        foreground_probability_map is not None or background_probability_map is not None
    )
    if has_prediction == has_probabilities:
        raise RiskCoverageInputError(
            "provide exactly one prediction source: prediction_map or foreground/background "
            "probability maps."
        )
    if has_prediction:
        return _require_binary_volume(prediction_map, field_name="prediction_map")
    if foreground_probability_map is None or background_probability_map is None:
        raise RiskCoverageInputError(
            "foreground_probability_map and background_probability_map must be provided together."
        )
    return derive_hard_prediction_from_binary_probabilities(
        foreground_probability_map=foreground_probability_map,
        background_probability_map=background_probability_map,
    )


def _require_same_shape(*arrays: np.ndarray) -> None:
    shapes = {array.shape for array in arrays}
    if len(shapes) != 1:
        raise RiskCoverageInputError(
            "prediction, reference, uncertainty, and valid masks must share one [1,1,D,H,W] shape."
        )


def _resolve_valid_voxel_mask(
    valid_voxel_mask: np.ndarray | None,
    *,
    shape: tuple[int, ...],
) -> np.ndarray:
    if valid_voxel_mask is None:
        return np.ones(shape, dtype=bool)
    mask = np.asarray(valid_voxel_mask)
    if mask.shape != shape:
        raise RiskCoverageInputError("valid_voxel_mask must match prediction shape.")
    if mask.dtype != np.dtype(bool):
        raise RiskCoverageInputError("valid_voxel_mask must be boolean.")
    return np.ascontiguousarray(mask)


def _require_binary_volume(array: np.ndarray | None, *, field_name: str) -> np.ndarray:
    if array is None:
        raise RiskCoverageInputError(f"{field_name} is required.")
    value = np.asarray(array)
    _require_volume_shape(value, field_name=field_name)
    if not np.all(np.isfinite(value)):
        raise RiskCoverageInputError(f"{field_name} must contain only finite values.")
    if not np.all(np.isin(value, (0, 1, False, True))):
        raise RiskCoverageInputError(f"{field_name} must be binary with values 0/1.")
    return np.ascontiguousarray(value.astype(np.uint8, copy=False))


def _require_uncertainty_volume(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array)
    _require_volume_shape(value, field_name="uncertainty_map")
    if value.dtype.kind != "f":
        raise RiskCoverageInputError("uncertainty_map must be floating point.")
    uncertainty = np.ascontiguousarray(value.astype(np.float64, copy=False))
    if not np.all(np.isfinite(uncertainty)):
        raise RiskCoverageInputError("uncertainty_map must contain only finite values.")
    return uncertainty


def _require_probability_volume(array: np.ndarray, *, field_name: str) -> np.ndarray:
    value = np.asarray(array)
    _require_volume_shape(value, field_name=field_name)
    if value.dtype.kind != "f":
        raise RiskCoverageInputError(f"{field_name} must be floating point.")
    probability = np.ascontiguousarray(value.astype(np.float64, copy=False))
    if not np.all(np.isfinite(probability)):
        raise RiskCoverageInputError(f"{field_name} must contain only finite values.")
    if np.any(probability < 0.0) or np.any(probability > 1.0):
        raise RiskCoverageInputError(f"{field_name} probabilities must lie in [0,1].")
    return probability


def _require_volume_shape(array: np.ndarray, *, field_name: str) -> None:
    if array.ndim != _VOLUME_SHAPE_RANK or array.shape[0] != 1 or array.shape[1] != 1:
        raise RiskCoverageInputError(f"{field_name} must have shape [1,1,D,H,W].")
    if array.shape[2] <= 0 or array.shape[3] <= 0 or array.shape[4] <= 0:
        raise RiskCoverageInputError(f"{field_name} spatial dimensions must be positive.")


def _require_sha256(value: str, *, field_name: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise RiskCoverageInputError(f"{field_name} must be a lowercase SHA-256 hex digest.")
