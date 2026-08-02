"""Deterministic Phase 7 uncertainty-error and failure-detection evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np

from protoem_ct.artifacts.hashing import JsonValue, canonical_json_bytes, sha256_json
from protoem_ct.uncertainty.contracts import array_content_sha256

PHASE7_UNCERTAINTY_ERROR_CORRELATION_SCHEMA_NAME: Final[str] = (
    "phase7_uncertainty_error_correlation_result"
)
PHASE7_UNCERTAINTY_ERROR_CORRELATION_SCHEMA_VERSION: Final[str] = "v1"
PHASE7_FAILURE_DETECTION_AUROC_SCHEMA_NAME: Final[str] = "phase7_failure_detection_auroc_result"
PHASE7_FAILURE_DETECTION_AUROC_SCHEMA_VERSION: Final[str] = "v1"

AvailabilityStatus = Literal["available", "unavailable"]

_CORRELATION_METHOD: Final[str] = "spearman_average_rank"
_AUROC_METHOD: Final[str] = "mann_whitney_pairwise"
_AVAILABILITY_STATUSES: Final[frozenset[str]] = frozenset({"available", "unavailable"})
_UNAVAILABLE_REASONS: Final[frozenset[str]] = frozenset(
    {
        "insufficient_valid_voxels",
        "constant_uncertainty",
        "constant_error",
        "insufficient_cases",
        "one_class_failure_labels",
    }
)
_FLOAT_DTYPE: Final[np.dtype[np.float64]] = np.dtype(np.float64)
_BOOL_DTYPE: Final[np.dtype[np.bool_]] = np.dtype(np.bool_)


class Phase7FailureDetectionError(ValueError):
    """Base error for deterministic Phase 7 failure analysis."""


class FailureDetectionInputError(Phase7FailureDetectionError):
    """Raised when uncertainty-error or AUROC inputs are malformed."""


class FailureDetectionIdentityError(Phase7FailureDetectionError):
    """Raised when a self-identity hash does not match deterministic content."""


@dataclass(frozen=True, slots=True)
class Phase7UncertaintyErrorCorrelationResult:
    """Voxel-level Spearman correlation between uncertainty and binary errors."""

    schema_name: str
    schema_version: str
    correlation_result_hash: str
    uncertainty_content_hash: str
    error_indicator_content_hash: str
    valid_mask_content_hash: str | None
    valid_voxel_count: int
    correlation_method: str
    correlation_value: float | None
    availability_status: AvailabilityStatus
    unavailable_reason: str | None

    def __post_init__(self) -> None:
        if self.schema_name != PHASE7_UNCERTAINTY_ERROR_CORRELATION_SCHEMA_NAME:
            raise FailureDetectionInputError(
                f"schema_name must be {PHASE7_UNCERTAINTY_ERROR_CORRELATION_SCHEMA_NAME!r}."
            )
        if self.schema_version != PHASE7_UNCERTAINTY_ERROR_CORRELATION_SCHEMA_VERSION:
            raise FailureDetectionInputError(
                f"schema_version must be {PHASE7_UNCERTAINTY_ERROR_CORRELATION_SCHEMA_VERSION!r}."
            )
        _require_sha256(self.correlation_result_hash, field_name="correlation_result_hash")
        _require_sha256(self.uncertainty_content_hash, field_name="uncertainty_content_hash")
        _require_sha256(
            self.error_indicator_content_hash,
            field_name="error_indicator_content_hash",
        )
        if self.valid_mask_content_hash is not None:
            _require_sha256(self.valid_mask_content_hash, field_name="valid_mask_content_hash")
        _require_nonnegative_int(self.valid_voxel_count, field_name="valid_voxel_count")
        if self.correlation_method != _CORRELATION_METHOD:
            raise FailureDetectionInputError("correlation_method must be 'spearman_average_rank'.")
        _require_availability(
            self.availability_status,
            self.unavailable_reason,
            value=self.correlation_value,
            value_name="correlation_value",
        )
        if self.correlation_value is not None:
            _require_finite_float(self.correlation_value, field_name="correlation_value")
            if self.correlation_value < -1.0 or self.correlation_value > 1.0:
                raise FailureDetectionInputError("correlation_value must be in [-1, 1].")
        if self.correlation_result_hash != hash_uncertainty_error_correlation_result(self):
            raise FailureDetectionIdentityError(
                "correlation_result_hash does not match deterministic content."
            )


@dataclass(frozen=True, slots=True)
class Phase7FailureDetectionAUROCResult:
    """Case-level failure-detection AUROC with explicit eligibility."""

    schema_name: str
    schema_version: str
    auroc_result_hash: str
    case_uncertainty_score_content_hash: str
    failure_indicator_content_hash: str
    case_count: int
    positive_failure_case_count: int
    negative_nonfailure_case_count: int
    auroc_method: str
    auroc_value: float | None
    availability_status: AvailabilityStatus
    unavailable_reason: str | None

    def __post_init__(self) -> None:
        if self.schema_name != PHASE7_FAILURE_DETECTION_AUROC_SCHEMA_NAME:
            raise FailureDetectionInputError(
                f"schema_name must be {PHASE7_FAILURE_DETECTION_AUROC_SCHEMA_NAME!r}."
            )
        if self.schema_version != PHASE7_FAILURE_DETECTION_AUROC_SCHEMA_VERSION:
            raise FailureDetectionInputError(
                f"schema_version must be {PHASE7_FAILURE_DETECTION_AUROC_SCHEMA_VERSION!r}."
            )
        _require_sha256(self.auroc_result_hash, field_name="auroc_result_hash")
        _require_sha256(
            self.case_uncertainty_score_content_hash,
            field_name="case_uncertainty_score_content_hash",
        )
        _require_sha256(
            self.failure_indicator_content_hash,
            field_name="failure_indicator_content_hash",
        )
        _require_nonnegative_int(self.case_count, field_name="case_count")
        _require_nonnegative_int(
            self.positive_failure_case_count,
            field_name="positive_failure_case_count",
        )
        _require_nonnegative_int(
            self.negative_nonfailure_case_count,
            field_name="negative_nonfailure_case_count",
        )
        if (
            self.positive_failure_case_count + self.negative_nonfailure_case_count
            != self.case_count
        ):
            raise FailureDetectionInputError(
                "positive and negative case counts must sum to case_count."
            )
        if self.auroc_method != _AUROC_METHOD:
            raise FailureDetectionInputError("auroc_method must be 'mann_whitney_pairwise'.")
        _require_availability(
            self.availability_status,
            self.unavailable_reason,
            value=self.auroc_value,
            value_name="auroc_value",
        )
        if self.auroc_value is not None:
            _require_finite_float(self.auroc_value, field_name="auroc_value")
            if self.auroc_value < 0.0 or self.auroc_value > 1.0:
                raise FailureDetectionInputError("auroc_value must be in [0, 1].")
        if self.auroc_result_hash != hash_failure_detection_auroc_result(self):
            raise FailureDetectionIdentityError(
                "auroc_result_hash does not match deterministic content."
            )


def compute_uncertainty_error_correlation(
    *,
    uncertainty_values: np.ndarray,
    error_indicators: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> Phase7UncertaintyErrorCorrelationResult:
    """Compute deterministic voxel Spearman correlation for post-prediction evaluation.

    Values are converted to deterministic average ranks, including tied uncertainty
    values and tied binary errors, then Pearson correlation is computed over those
    ranks. Reference masks may be used by callers to construct `error_indicators`,
    but this API accepts no prediction, optimization, model, query-label, or
    reference-mask inputs.
    """

    uncertainty, errors, mask = _prepare_voxel_inputs(
        uncertainty_values=uncertainty_values,
        error_indicators=error_indicators,
        valid_mask=valid_mask,
    )
    selected_uncertainty = uncertainty[mask]
    selected_errors = errors[mask].astype(_FLOAT_DTYPE, copy=False)
    valid_count = int(selected_uncertainty.size)
    availability_status: AvailabilityStatus = "available"
    unavailable_reason: str | None = None
    correlation_value: float | None
    if valid_count < 2:
        availability_status = "unavailable"
        unavailable_reason = "insufficient_valid_voxels"
        correlation_value = None
    else:
        uncertainty_ranks = _average_ranks(selected_uncertainty)
        error_ranks = _average_ranks(selected_errors)
        uncertainty_delta = uncertainty_ranks - float(np.mean(uncertainty_ranks))
        error_delta = error_ranks - float(np.mean(error_ranks))
        uncertainty_sum_squares = float(np.dot(uncertainty_delta, uncertainty_delta))
        error_sum_squares = float(np.dot(error_delta, error_delta))
        if uncertainty_sum_squares == 0.0:
            availability_status = "unavailable"
            unavailable_reason = "constant_uncertainty"
            correlation_value = None
        elif error_sum_squares == 0.0:
            availability_status = "unavailable"
            unavailable_reason = "constant_error"
            correlation_value = None
        else:
            denominator = math.sqrt(uncertainty_sum_squares * error_sum_squares)
            correlation_value = float(np.dot(uncertainty_delta, error_delta) / denominator)
            correlation_value = _normalize_unit_interval_drift(
                correlation_value,
                lower=-1.0,
                upper=1.0,
                field_name="correlation_value",
            )
    return _build_correlation_result(
        uncertainty=selected_uncertainty,
        errors=errors[mask],
        valid_mask=mask if valid_mask is not None else None,
        valid_voxel_count=valid_count,
        correlation_value=correlation_value,
        availability_status=availability_status,
        unavailable_reason=unavailable_reason,
    )


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return deterministic one-based average ranks for finite one-dimensional values."""

    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape, dtype=_FLOAT_DTYPE)
    sorted_values = values[order]
    start = 0
    while start < int(sorted_values.size):
        stop = start + 1
        while stop < int(sorted_values.size) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        average_rank = (float(start + 1) + float(stop)) / 2.0
        ranks[order[start:stop]] = average_rank
        start = stop
    return ranks


def compute_failure_detection_auroc(
    *,
    case_uncertainty_scores: np.ndarray,
    failure_indicators: np.ndarray,
) -> Phase7FailureDetectionAUROCResult:
    """Compute deterministic case-level AUROC for failure detection.

    AUROC is the Mann-Whitney pairwise probability that a positive failure case
    receives a higher uncertainty score than a negative non-failure case, with
    ties receiving weight 0.5.
    """

    scores, failures = _prepare_case_inputs(
        case_uncertainty_scores=case_uncertainty_scores,
        failure_indicators=failure_indicators,
    )
    case_count = int(scores.size)
    positive_count = int(np.count_nonzero(failures))
    negative_count = case_count - positive_count
    availability_status: AvailabilityStatus = "available"
    unavailable_reason: str | None = None
    auroc_value: float | None
    if case_count < 2:
        availability_status = "unavailable"
        unavailable_reason = "insufficient_cases"
        auroc_value = None
    elif positive_count == 0 or negative_count == 0:
        availability_status = "unavailable"
        unavailable_reason = "one_class_failure_labels"
        auroc_value = None
    else:
        positive_scores = scores[failures]
        negative_scores = scores[np.logical_not(failures)]
        wins = 0.0
        for positive_score in positive_scores:
            greater = float(np.count_nonzero(positive_score > negative_scores))
            ties = float(np.count_nonzero(positive_score == negative_scores))
            wins += greater + (0.5 * ties)
        auroc_value = wins / float(positive_count * negative_count)
    return _build_auroc_result(
        scores=scores,
        failures=failures,
        case_count=case_count,
        positive_failure_case_count=positive_count,
        negative_nonfailure_case_count=negative_count,
        auroc_value=auroc_value,
        availability_status=availability_status,
        unavailable_reason=unavailable_reason,
    )


def uncertainty_error_correlation_result_identity_payload(
    result: Phase7UncertaintyErrorCorrelationResult,
) -> dict[str, JsonValue]:
    """Return the exact canonical identity payload for a correlation result."""

    return {
        "availability_status": result.availability_status,
        "correlation_method": result.correlation_method,
        "correlation_value": result.correlation_value,
        "error_indicator_content_hash": result.error_indicator_content_hash,
        "schema_name": result.schema_name,
        "schema_version": result.schema_version,
        "uncertainty_content_hash": result.uncertainty_content_hash,
        "unavailable_reason": result.unavailable_reason,
        "valid_mask_content_hash": result.valid_mask_content_hash,
        "valid_voxel_count": result.valid_voxel_count,
    }


def uncertainty_error_correlation_result_to_dict(
    result: Phase7UncertaintyErrorCorrelationResult,
) -> dict[str, JsonValue]:
    """Convert a correlation result to a deterministic mapping."""

    payload = uncertainty_error_correlation_result_identity_payload(result)
    payload["correlation_result_hash"] = result.correlation_result_hash
    return payload


def hash_uncertainty_error_correlation_result(
    result: Phase7UncertaintyErrorCorrelationResult,
) -> str:
    """Return the deterministic SHA-256 identity for a correlation result."""

    return sha256_json(uncertainty_error_correlation_result_identity_payload(result))


def uncertainty_error_correlation_result_to_json(
    result: Phase7UncertaintyErrorCorrelationResult,
) -> bytes:
    """Serialize a correlation result to canonical JSON bytes."""

    return canonical_json_bytes(uncertainty_error_correlation_result_to_dict(result)) + b"\n"


def failure_detection_auroc_result_identity_payload(
    result: Phase7FailureDetectionAUROCResult,
) -> dict[str, JsonValue]:
    """Return the exact canonical identity payload for an AUROC result."""

    return {
        "auroc_method": result.auroc_method,
        "auroc_value": result.auroc_value,
        "availability_status": result.availability_status,
        "case_count": result.case_count,
        "case_uncertainty_score_content_hash": result.case_uncertainty_score_content_hash,
        "failure_indicator_content_hash": result.failure_indicator_content_hash,
        "negative_nonfailure_case_count": result.negative_nonfailure_case_count,
        "positive_failure_case_count": result.positive_failure_case_count,
        "schema_name": result.schema_name,
        "schema_version": result.schema_version,
        "unavailable_reason": result.unavailable_reason,
    }


def failure_detection_auroc_result_to_dict(
    result: Phase7FailureDetectionAUROCResult,
) -> dict[str, JsonValue]:
    """Convert an AUROC result to a deterministic mapping."""

    payload = failure_detection_auroc_result_identity_payload(result)
    payload["auroc_result_hash"] = result.auroc_result_hash
    return payload


def hash_failure_detection_auroc_result(result: Phase7FailureDetectionAUROCResult) -> str:
    """Return the deterministic SHA-256 identity for an AUROC result."""

    return sha256_json(failure_detection_auroc_result_identity_payload(result))


def failure_detection_auroc_result_to_json(result: Phase7FailureDetectionAUROCResult) -> bytes:
    """Serialize an AUROC result to canonical JSON bytes."""

    return canonical_json_bytes(failure_detection_auroc_result_to_dict(result)) + b"\n"


def query_label_not_part_of_failure_detection_prediction_api() -> bool:
    """Return True because failure detection only consumes post-prediction evaluation inputs."""

    return True


def _build_correlation_result(
    *,
    uncertainty: np.ndarray,
    errors: np.ndarray,
    valid_mask: np.ndarray | None,
    valid_voxel_count: int,
    correlation_value: float | None,
    availability_status: AvailabilityStatus,
    unavailable_reason: str | None,
) -> Phase7UncertaintyErrorCorrelationResult:
    uncertainty_hash = array_content_sha256(np.ascontiguousarray(uncertainty))
    error_hash = array_content_sha256(np.ascontiguousarray(errors.astype(_BOOL_DTYPE, copy=False)))
    valid_mask_hash = (
        None if valid_mask is None else array_content_sha256(np.ascontiguousarray(valid_mask))
    )
    result_without_hash = _CorrelationResultWithoutHash(
        uncertainty_content_hash=uncertainty_hash,
        error_indicator_content_hash=error_hash,
        valid_mask_content_hash=valid_mask_hash,
        valid_voxel_count=valid_voxel_count,
        correlation_value=correlation_value,
        availability_status=availability_status,
        unavailable_reason=unavailable_reason,
    )
    payload = _correlation_identity_payload_from_parts(result_without_hash)
    return Phase7UncertaintyErrorCorrelationResult(
        schema_name=PHASE7_UNCERTAINTY_ERROR_CORRELATION_SCHEMA_NAME,
        schema_version=PHASE7_UNCERTAINTY_ERROR_CORRELATION_SCHEMA_VERSION,
        correlation_result_hash=sha256_json(payload),
        uncertainty_content_hash=uncertainty_hash,
        error_indicator_content_hash=error_hash,
        valid_mask_content_hash=valid_mask_hash,
        valid_voxel_count=valid_voxel_count,
        correlation_method=_CORRELATION_METHOD,
        correlation_value=correlation_value,
        availability_status=availability_status,
        unavailable_reason=unavailable_reason,
    )


def _build_auroc_result(
    *,
    scores: np.ndarray,
    failures: np.ndarray,
    case_count: int,
    positive_failure_case_count: int,
    negative_nonfailure_case_count: int,
    auroc_value: float | None,
    availability_status: AvailabilityStatus,
    unavailable_reason: str | None,
) -> Phase7FailureDetectionAUROCResult:
    score_hash = array_content_sha256(np.ascontiguousarray(scores))
    failure_hash = array_content_sha256(np.ascontiguousarray(failures.astype(_BOOL_DTYPE)))
    result_without_hash = _AUROCResultWithoutHash(
        case_uncertainty_score_content_hash=score_hash,
        failure_indicator_content_hash=failure_hash,
        case_count=case_count,
        positive_failure_case_count=positive_failure_case_count,
        negative_nonfailure_case_count=negative_nonfailure_case_count,
        auroc_value=auroc_value,
        availability_status=availability_status,
        unavailable_reason=unavailable_reason,
    )
    payload = _auroc_identity_payload_from_parts(result_without_hash)
    return Phase7FailureDetectionAUROCResult(
        schema_name=PHASE7_FAILURE_DETECTION_AUROC_SCHEMA_NAME,
        schema_version=PHASE7_FAILURE_DETECTION_AUROC_SCHEMA_VERSION,
        auroc_result_hash=sha256_json(payload),
        case_uncertainty_score_content_hash=score_hash,
        failure_indicator_content_hash=failure_hash,
        case_count=case_count,
        positive_failure_case_count=positive_failure_case_count,
        negative_nonfailure_case_count=negative_nonfailure_case_count,
        auroc_method=_AUROC_METHOD,
        auroc_value=auroc_value,
        availability_status=availability_status,
        unavailable_reason=unavailable_reason,
    )


@dataclass(frozen=True, slots=True)
class _CorrelationResultWithoutHash:
    uncertainty_content_hash: str
    error_indicator_content_hash: str
    valid_mask_content_hash: str | None
    valid_voxel_count: int
    correlation_value: float | None
    availability_status: AvailabilityStatus
    unavailable_reason: str | None


def _correlation_identity_payload_from_parts(
    result: _CorrelationResultWithoutHash,
) -> dict[str, JsonValue]:
    return {
        "availability_status": result.availability_status,
        "correlation_method": _CORRELATION_METHOD,
        "correlation_value": result.correlation_value,
        "error_indicator_content_hash": result.error_indicator_content_hash,
        "schema_name": PHASE7_UNCERTAINTY_ERROR_CORRELATION_SCHEMA_NAME,
        "schema_version": PHASE7_UNCERTAINTY_ERROR_CORRELATION_SCHEMA_VERSION,
        "uncertainty_content_hash": result.uncertainty_content_hash,
        "unavailable_reason": result.unavailable_reason,
        "valid_mask_content_hash": result.valid_mask_content_hash,
        "valid_voxel_count": result.valid_voxel_count,
    }


@dataclass(frozen=True, slots=True)
class _AUROCResultWithoutHash:
    case_uncertainty_score_content_hash: str
    failure_indicator_content_hash: str
    case_count: int
    positive_failure_case_count: int
    negative_nonfailure_case_count: int
    auroc_value: float | None
    availability_status: AvailabilityStatus
    unavailable_reason: str | None


def _auroc_identity_payload_from_parts(result: _AUROCResultWithoutHash) -> dict[str, JsonValue]:
    return {
        "auroc_method": _AUROC_METHOD,
        "auroc_value": result.auroc_value,
        "availability_status": result.availability_status,
        "case_count": result.case_count,
        "case_uncertainty_score_content_hash": result.case_uncertainty_score_content_hash,
        "failure_indicator_content_hash": result.failure_indicator_content_hash,
        "negative_nonfailure_case_count": result.negative_nonfailure_case_count,
        "positive_failure_case_count": result.positive_failure_case_count,
        "schema_name": PHASE7_FAILURE_DETECTION_AUROC_SCHEMA_NAME,
        "schema_version": PHASE7_FAILURE_DETECTION_AUROC_SCHEMA_VERSION,
        "unavailable_reason": result.unavailable_reason,
    }


def _prepare_voxel_inputs(
    *,
    uncertainty_values: np.ndarray,
    error_indicators: np.ndarray,
    valid_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    uncertainty = _require_finite_array(uncertainty_values, field_name="uncertainty_values")
    errors = _require_binary_array(error_indicators, field_name="error_indicators")
    if uncertainty.shape != errors.shape:
        raise FailureDetectionInputError(
            "uncertainty_values and error_indicators must have identical shapes."
        )
    if valid_mask is None:
        mask = np.ones(uncertainty.shape, dtype=_BOOL_DTYPE)
    else:
        mask = _require_bool_mask(valid_mask, field_name="valid_mask")
        if mask.shape != uncertainty.shape:
            raise FailureDetectionInputError("valid_mask must match uncertainty_values shape.")
    return uncertainty, errors, mask


def _prepare_case_inputs(
    *,
    case_uncertainty_scores: np.ndarray,
    failure_indicators: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    scores = _require_finite_array(
        case_uncertainty_scores,
        field_name="case_uncertainty_scores",
    )
    failures = _require_binary_array(failure_indicators, field_name="failure_indicators")
    if scores.ndim != 1:
        raise FailureDetectionInputError("case_uncertainty_scores must be one-dimensional.")
    if failures.ndim != 1:
        raise FailureDetectionInputError("failure_indicators must be one-dimensional.")
    if scores.shape != failures.shape:
        raise FailureDetectionInputError(
            "case_uncertainty_scores and failure_indicators must have identical shapes."
        )
    return scores, failures


def _require_finite_array(array: np.ndarray, *, field_name: str) -> np.ndarray:
    value = np.asarray(array)
    if value.dtype.kind not in {"f", "i", "u", "b"}:
        raise FailureDetectionInputError(f"{field_name} must be numeric.")
    contiguous = np.ascontiguousarray(value.astype(_FLOAT_DTYPE, copy=False))
    if not np.all(np.isfinite(contiguous)):
        raise FailureDetectionInputError(f"{field_name} must contain only finite values.")
    return contiguous


def _require_binary_array(array: np.ndarray, *, field_name: str) -> np.ndarray:
    value = np.asarray(array)
    if value.dtype.kind not in {"b", "i", "u", "f"}:
        raise FailureDetectionInputError(f"{field_name} must be binary.")
    contiguous = np.ascontiguousarray(value)
    if not np.all(np.isfinite(contiguous.astype(_FLOAT_DTYPE, copy=False))):
        raise FailureDetectionInputError(f"{field_name} must contain only finite values.")
    if not np.all(np.logical_or(contiguous == 0, contiguous == 1)):
        raise FailureDetectionInputError(f"{field_name} must contain only binary values 0/1.")
    return np.ascontiguousarray(contiguous.astype(_BOOL_DTYPE, copy=False))


def _require_bool_mask(array: np.ndarray, *, field_name: str) -> np.ndarray:
    value = np.asarray(array)
    if value.dtype.kind != "b":
        raise FailureDetectionInputError(f"{field_name} must be boolean.")
    return np.ascontiguousarray(value.astype(_BOOL_DTYPE, copy=False))


def _require_sha256(value: str, *, field_name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise FailureDetectionInputError(f"{field_name} must be a lowercase SHA-256 hex digest.")


def _require_nonnegative_int(value: int, *, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise FailureDetectionInputError(f"{field_name} must be a nonnegative integer.")


def _require_finite_float(value: float, *, field_name: str) -> None:
    if not math.isfinite(value):
        raise FailureDetectionInputError(f"{field_name} must be finite.")


def _require_availability(
    status: AvailabilityStatus,
    reason: str | None,
    *,
    value: float | None,
    value_name: str,
) -> None:
    if status not in _AVAILABILITY_STATUSES:
        raise FailureDetectionInputError(
            f"availability_status must be one of {sorted(_AVAILABILITY_STATUSES)!r}."
        )
    if reason is not None and reason not in _UNAVAILABLE_REASONS:
        raise FailureDetectionInputError(
            f"unavailable_reason must be one of {sorted(_UNAVAILABLE_REASONS)!r}."
        )
    if status == "available":
        if value is None:
            raise FailureDetectionInputError(f"available results require {value_name}.")
        if reason is not None:
            raise FailureDetectionInputError(
                "available results must not contain unavailable_reason."
            )
    else:
        if value is not None:
            raise FailureDetectionInputError(f"unavailable results must not report {value_name}.")
        if reason is None:
            raise FailureDetectionInputError("unavailable results require unavailable_reason.")


def _normalize_unit_interval_drift(
    value: float,
    *,
    lower: float,
    upper: float,
    field_name: str,
) -> float:
    if lower <= value <= upper:
        return value
    if math.isclose(value, lower, rel_tol=0.0, abs_tol=1e-12):
        return lower
    if math.isclose(value, upper, rel_tol=0.0, abs_tol=1e-12):
        return upper
    raise FailureDetectionInputError(f"{field_name} must be in [{lower}, {upper}].")


def failure_detection_public_prediction_api_has_no_labels() -> bool:
    """Return True because this module exposes no prediction, optimization, or model API."""

    return True
