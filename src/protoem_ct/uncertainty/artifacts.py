"""Deterministic Phase 7 uncertainty and evaluation artifact schemas."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, TypeAlias, cast

from protoem_ct.artifacts.hashing import JsonValue, canonical_json_bytes, sha256_json

PHASE7_UNCERTAINTY_RESULT_SCHEMA_NAME: Final[str] = "phase7_uncertainty_result"
PHASE7_UNCERTAINTY_RESULT_SCHEMA_VERSION: Final[str] = "v1"
PHASE7_CALIBRATION_BIN_SCHEMA_NAME: Final[str] = "phase7_calibration_bin"
PHASE7_CALIBRATION_BIN_SCHEMA_VERSION: Final[str] = "v1"
PHASE7_CALIBRATION_RESULT_SCHEMA_NAME: Final[str] = "phase7_calibration_result"
PHASE7_CALIBRATION_RESULT_SCHEMA_VERSION: Final[str] = "v1"
PHASE7_RISK_COVERAGE_POINT_SCHEMA_NAME: Final[str] = "phase7_risk_coverage_point"
PHASE7_RISK_COVERAGE_POINT_SCHEMA_VERSION: Final[str] = "v1"
PHASE7_RISK_COVERAGE_RESULT_SCHEMA_NAME: Final[str] = "phase7_risk_coverage_result"
PHASE7_RISK_COVERAGE_RESULT_SCHEMA_VERSION: Final[str] = "v1"
PHASE7_DEGRADATION_RESULT_SCHEMA_NAME: Final[str] = "phase7_degradation_result"
PHASE7_DEGRADATION_RESULT_SCHEMA_VERSION: Final[str] = "v1"
PHASE7_LESION_SUBGROUP_RECORD_SCHEMA_NAME: Final[str] = "phase7_lesion_subgroup_record"
PHASE7_LESION_SUBGROUP_RECORD_SCHEMA_VERSION: Final[str] = "v1"
PHASE7_LESION_SUBGROUP_RESULT_SCHEMA_NAME: Final[str] = "phase7_lesion_subgroup_result"
PHASE7_LESION_SUBGROUP_RESULT_SCHEMA_VERSION: Final[str] = "v1"

PHASE7_AVAILABILITY_STATUSES: Final[frozenset[str]] = frozenset({"available", "unavailable"})
PHASE7_UNCERTAINTY_TYPES: Final[frozenset[str]] = frozenset(
    {"predictive_entropy", "tta_variance", "ensemble_variance"}
)
PHASE7_LOG_BASES: Final[frozenset[str]] = frozenset({"natural"})
PHASE7_CALIBRATION_METRICS: Final[frozenset[str]] = frozenset({"ece"})
PHASE7_BINNING_POLICIES: Final[frozenset[str]] = frozenset({"fixed_equal_width"})
PHASE7_RISK_ORDERINGS: Final[frozenset[str]] = frozenset({"low_uncertainty_first"})
PHASE7_TIE_BREAK_POLICIES: Final[frozenset[str]] = frozenset({"flattened_index"})
PHASE7_METRIC_DIRECTIONS: Final[frozenset[str]] = frozenset({"higher_is_better", "lower_is_better"})
PHASE7_LESION_SUBGROUP_NAMES: Final[frozenset[str]] = frozenset(
    {"empty", "small", "medium", "large"}
)

MappingLike: TypeAlias = Mapping[str, object]

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[a-z0-9](?:[a-z0-9_.-]{0,126}[a-z0-9])?$")
_MESSAGE_RE = re.compile(r"^[^\r\n]+$")


class Phase7UncertaintyArtifactError(ValueError):
    """Base error for Phase 7 uncertainty artifact contracts."""


class Phase7UncertaintyArtifactValidationError(Phase7UncertaintyArtifactError):
    """Raised when an uncertainty artifact violates its schema contract."""


class Phase7UncertaintyArtifactSerializationError(Phase7UncertaintyArtifactError):
    """Raised when an uncertainty artifact mapping cannot be reconstructed safely."""


class Phase7UncertaintyArtifactHashError(Phase7UncertaintyArtifactError):
    """Raised when an embedded uncertainty artifact hash does not match content."""


@dataclass(frozen=True, slots=True)
class Phase7UncertaintyResult:
    """Immutable uncertainty-result identity without labels or evaluation metrics."""

    schema_name: str
    schema_version: str
    uncertainty_result_hash: str
    source_prediction_hash: str
    source_probability_hash: str
    uncertainty_type: str
    probability_space: str
    log_base: str
    sample_count: int
    uncertainty_map_content_hash: str | None
    variance_map_content_hash: str | None
    mean_uncertainty: float | None
    max_uncertainty: float | None
    availability_status: str
    unavailable_reason: str | None

    def __post_init__(self) -> None:
        _require_schema(self.schema_name, PHASE7_UNCERTAINTY_RESULT_SCHEMA_NAME)
        _require_version(self.schema_version, PHASE7_UNCERTAINTY_RESULT_SCHEMA_VERSION)
        _require_sha256(self.uncertainty_result_hash, field_name="uncertainty_result_hash")
        _require_sha256(self.source_prediction_hash, field_name="source_prediction_hash")
        _require_sha256(self.source_probability_hash, field_name="source_probability_hash")
        _require_allowed(
            self.uncertainty_type, PHASE7_UNCERTAINTY_TYPES, field_name="uncertainty_type"
        )
        if self.probability_space != "binary_foreground":
            raise Phase7UncertaintyArtifactValidationError(
                "probability_space must be binary_foreground."
            )
        _require_allowed(self.log_base, PHASE7_LOG_BASES, field_name="log_base")
        _require_positive_int(self.sample_count, field_name="sample_count")
        if self.uncertainty_type in {"tta_variance", "ensemble_variance"} and self.sample_count < 2:
            raise Phase7UncertaintyArtifactValidationError(
                "variance uncertainty results require sample_count >= 2."
            )
        _require_optional_sha256(
            self.uncertainty_map_content_hash,
            field_name="uncertainty_map_content_hash",
        )
        _require_optional_sha256(
            self.variance_map_content_hash,
            field_name="variance_map_content_hash",
        )
        _require_optional_nonnegative_float(self.mean_uncertainty, field_name="mean_uncertainty")
        _require_optional_nonnegative_float(self.max_uncertainty, field_name="max_uncertainty")
        _require_allowed(
            self.availability_status,
            PHASE7_AVAILABILITY_STATUSES,
            field_name="availability_status",
        )
        _require_optional_identifier(self.unavailable_reason, field_name="unavailable_reason")
        if self.availability_status == "available":
            if self.uncertainty_map_content_hash is None:
                raise Phase7UncertaintyArtifactValidationError(
                    "available uncertainty results require uncertainty_map_content_hash."
                )
            if self.mean_uncertainty is None or self.max_uncertainty is None:
                raise Phase7UncertaintyArtifactValidationError(
                    "available uncertainty results require summary values."
                )
            if self.unavailable_reason is not None:
                raise Phase7UncertaintyArtifactValidationError(
                    "available uncertainty results must not contain unavailable_reason."
                )
        elif self.unavailable_reason is None:
            raise Phase7UncertaintyArtifactValidationError(
                "unavailable uncertainty results require unavailable_reason."
            )
        expected_hash = hash_phase7_uncertainty_result(self)
        if self.uncertainty_result_hash != expected_hash:
            raise Phase7UncertaintyArtifactHashError(
                "uncertainty_result_hash does not match deterministic content."
            )


@dataclass(frozen=True, slots=True)
class Phase7CalibrationBin:
    """One deterministic fixed calibration bin."""

    schema_name: str
    schema_version: str
    bin_index: int
    bin_lower: float
    bin_upper: float
    upper_inclusive: bool
    voxel_count: int
    accuracy: float | None
    mean_confidence: float | None
    weighted_error: float

    def __post_init__(self) -> None:
        _require_schema(self.schema_name, PHASE7_CALIBRATION_BIN_SCHEMA_NAME)
        _require_version(self.schema_version, PHASE7_CALIBRATION_BIN_SCHEMA_VERSION)
        _require_nonnegative_int(self.bin_index, field_name="bin_index")
        _require_fraction_closed(self.bin_lower, field_name="bin_lower")
        _require_fraction_closed(self.bin_upper, field_name="bin_upper")
        if self.bin_upper <= self.bin_lower:
            raise Phase7UncertaintyArtifactValidationError("bin_upper must exceed bin_lower.")
        _require_nonnegative_int(self.voxel_count, field_name="voxel_count")
        _require_optional_fraction_closed(self.accuracy, field_name="accuracy")
        _require_optional_fraction_closed(self.mean_confidence, field_name="mean_confidence")
        _require_nonnegative_float(self.weighted_error, field_name="weighted_error")
        if self.voxel_count == 0:
            if self.accuracy is not None or self.mean_confidence is not None:
                raise Phase7UncertaintyArtifactValidationError(
                    "empty calibration bins must not report accuracy or confidence."
                )
            if self.weighted_error != 0.0:
                raise Phase7UncertaintyArtifactValidationError(
                    "empty calibration bins must have zero weighted_error."
                )


@dataclass(frozen=True, slots=True)
class Phase7CalibrationResult:
    """Immutable deterministic calibration-result artifact."""

    schema_name: str
    schema_version: str
    calibration_result_hash: str
    prediction_content_hash: str
    probability_content_hash: str
    reference_mask_content_hash: str
    common_grid_geometry_record_hash: str
    calibration_metric: str
    confidence_definition: str
    binning_policy: str
    bin_count: int
    valid_voxel_count: int
    ece: float | None
    bins: tuple[Phase7CalibrationBin, ...]
    availability_status: str
    unavailable_reason: str | None

    def __post_init__(self) -> None:
        _require_schema(self.schema_name, PHASE7_CALIBRATION_RESULT_SCHEMA_NAME)
        _require_version(self.schema_version, PHASE7_CALIBRATION_RESULT_SCHEMA_VERSION)
        _require_sha256(self.calibration_result_hash, field_name="calibration_result_hash")
        _require_sha256(self.prediction_content_hash, field_name="prediction_content_hash")
        _require_sha256(self.probability_content_hash, field_name="probability_content_hash")
        _require_sha256(self.reference_mask_content_hash, field_name="reference_mask_content_hash")
        _require_sha256(
            self.common_grid_geometry_record_hash,
            field_name="common_grid_geometry_record_hash",
        )
        _require_allowed(
            self.calibration_metric, PHASE7_CALIBRATION_METRICS, field_name="calibration_metric"
        )
        if self.confidence_definition != "max_binary_probability":
            raise Phase7UncertaintyArtifactValidationError(
                "confidence_definition must be max_binary_probability."
            )
        _require_allowed(self.binning_policy, PHASE7_BINNING_POLICIES, field_name="binning_policy")
        _require_positive_int(self.bin_count, field_name="bin_count")
        _require_nonnegative_int(self.valid_voxel_count, field_name="valid_voxel_count")
        _require_optional_fraction_closed(self.ece, field_name="ece")
        if len(self.bins) != self.bin_count:
            raise Phase7UncertaintyArtifactValidationError("bin_count must equal len(bins).")
        if [item.bin_index for item in self.bins] != list(range(self.bin_count)):
            raise Phase7UncertaintyArtifactValidationError("calibration bins must be contiguous.")
        _require_availability(self.availability_status, self.unavailable_reason)
        if self.availability_status == "available" and self.ece is None:
            raise Phase7UncertaintyArtifactValidationError("available calibration requires ece.")
        if self.availability_status == "unavailable" and self.ece is not None:
            raise Phase7UncertaintyArtifactValidationError(
                "unavailable calibration must not report ece."
            )
        expected_hash = hash_phase7_calibration_result(self)
        if self.calibration_result_hash != expected_hash:
            raise Phase7UncertaintyArtifactHashError(
                "calibration_result_hash does not match deterministic content."
            )


@dataclass(frozen=True, slots=True)
class Phase7RiskCoveragePoint:
    """One deterministic risk-coverage curve point."""

    schema_name: str
    schema_version: str
    retained_voxel_count: int
    coverage: float
    risk: float

    def __post_init__(self) -> None:
        _require_schema(self.schema_name, PHASE7_RISK_COVERAGE_POINT_SCHEMA_NAME)
        _require_version(self.schema_version, PHASE7_RISK_COVERAGE_POINT_SCHEMA_VERSION)
        _require_positive_int(self.retained_voxel_count, field_name="retained_voxel_count")
        _require_fraction_closed(self.coverage, field_name="coverage")
        _require_fraction_closed(self.risk, field_name="risk")


@dataclass(frozen=True, slots=True)
class Phase7RiskCoverageResult:
    """Immutable deterministic risk-coverage artifact."""

    schema_name: str
    schema_version: str
    risk_coverage_result_hash: str
    uncertainty_result_hash: str
    prediction_content_hash: str
    reference_mask_content_hash: str
    common_grid_geometry_record_hash: str
    ordering: str
    tie_break_policy: str
    valid_voxel_count: int
    points: tuple[Phase7RiskCoveragePoint, ...]
    availability_status: str
    unavailable_reason: str | None

    def __post_init__(self) -> None:
        _require_schema(self.schema_name, PHASE7_RISK_COVERAGE_RESULT_SCHEMA_NAME)
        _require_version(self.schema_version, PHASE7_RISK_COVERAGE_RESULT_SCHEMA_VERSION)
        _require_sha256(self.risk_coverage_result_hash, field_name="risk_coverage_result_hash")
        _require_sha256(self.uncertainty_result_hash, field_name="uncertainty_result_hash")
        _require_sha256(self.prediction_content_hash, field_name="prediction_content_hash")
        _require_sha256(self.reference_mask_content_hash, field_name="reference_mask_content_hash")
        _require_sha256(
            self.common_grid_geometry_record_hash,
            field_name="common_grid_geometry_record_hash",
        )
        _require_allowed(self.ordering, PHASE7_RISK_ORDERINGS, field_name="ordering")
        _require_allowed(
            self.tie_break_policy, PHASE7_TIE_BREAK_POLICIES, field_name="tie_break_policy"
        )
        _require_nonnegative_int(self.valid_voxel_count, field_name="valid_voxel_count")
        _require_availability(self.availability_status, self.unavailable_reason)
        if self.availability_status == "available" and not self.points:
            raise Phase7UncertaintyArtifactValidationError(
                "available risk-coverage results require points."
            )
        if self.availability_status == "unavailable" and self.points:
            raise Phase7UncertaintyArtifactValidationError(
                "unavailable risk-coverage results must not report points."
            )
        expected_hash = hash_phase7_risk_coverage_result(self)
        if self.risk_coverage_result_hash != expected_hash:
            raise Phase7UncertaintyArtifactHashError(
                "risk_coverage_result_hash does not match deterministic content."
            )


@dataclass(frozen=True, slots=True)
class Phase7DegradationResult:
    """Immutable deterministic metric-degradation artifact."""

    schema_name: str
    schema_version: str
    degradation_result_hash: str
    baseline_artifact_hash: str
    corrupted_artifact_hash: str
    metric_name: str
    metric_direction: str
    baseline_value: float | None
    corrupted_value: float | None
    absolute_degradation: float | None
    relative_degradation: float | None
    relative_epsilon: float
    availability_status: str
    unavailable_reason: str | None

    def __post_init__(self) -> None:
        _require_schema(self.schema_name, PHASE7_DEGRADATION_RESULT_SCHEMA_NAME)
        _require_version(self.schema_version, PHASE7_DEGRADATION_RESULT_SCHEMA_VERSION)
        _require_sha256(self.degradation_result_hash, field_name="degradation_result_hash")
        _require_sha256(self.baseline_artifact_hash, field_name="baseline_artifact_hash")
        _require_sha256(self.corrupted_artifact_hash, field_name="corrupted_artifact_hash")
        _require_identifier(self.metric_name, field_name="metric_name")
        _require_allowed(
            self.metric_direction, PHASE7_METRIC_DIRECTIONS, field_name="metric_direction"
        )
        _require_optional_finite_float(self.baseline_value, field_name="baseline_value")
        _require_optional_finite_float(self.corrupted_value, field_name="corrupted_value")
        _require_optional_finite_float(
            self.absolute_degradation,
            field_name="absolute_degradation",
        )
        _require_optional_finite_float(
            self.relative_degradation,
            field_name="relative_degradation",
        )
        _require_positive_float(self.relative_epsilon, field_name="relative_epsilon")
        _require_availability(self.availability_status, self.unavailable_reason)
        if self.availability_status == "available":
            if (
                self.baseline_value is None
                or self.corrupted_value is None
                or self.absolute_degradation is None
            ):
                raise Phase7UncertaintyArtifactValidationError(
                    "available degradation requires baseline, corrupted, and absolute values."
                )
            if self.metric_direction == "higher_is_better":
                expected_absolute = self.baseline_value - self.corrupted_value
            else:
                expected_absolute = self.corrupted_value - self.baseline_value
            if not math.isclose(
                self.absolute_degradation,
                expected_absolute,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise Phase7UncertaintyArtifactValidationError(
                    "absolute_degradation must match metric_direction."
                )
            if abs(self.baseline_value) >= self.relative_epsilon:
                expected_relative = expected_absolute / abs(self.baseline_value)
                if self.relative_degradation is None or not math.isclose(
                    self.relative_degradation,
                    expected_relative,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise Phase7UncertaintyArtifactValidationError(
                        "relative_degradation must match the documented formula when eligible."
                    )
            elif self.relative_degradation is not None:
                raise Phase7UncertaintyArtifactValidationError(
                    "relative_degradation must be unavailable near a zero baseline."
                )
        expected_hash = hash_phase7_degradation_result(self)
        if self.degradation_result_hash != expected_hash:
            raise Phase7UncertaintyArtifactHashError(
                "degradation_result_hash does not match deterministic content."
            )


@dataclass(frozen=True, slots=True)
class Phase7LesionSubgroupRecord:
    """One deterministic lesion-size subgroup summary record."""

    schema_name: str
    schema_version: str
    subgroup_name: str
    eligible_case_count: int
    empty_lesion_case_count: int
    skipped_case_count: int
    metric_availability_status: str
    mean_metric_value: float | None
    unavailable_reason: str | None

    def __post_init__(self) -> None:
        _require_schema(self.schema_name, PHASE7_LESION_SUBGROUP_RECORD_SCHEMA_NAME)
        _require_version(self.schema_version, PHASE7_LESION_SUBGROUP_RECORD_SCHEMA_VERSION)
        _require_allowed(
            self.subgroup_name, PHASE7_LESION_SUBGROUP_NAMES, field_name="subgroup_name"
        )
        _require_nonnegative_int(self.eligible_case_count, field_name="eligible_case_count")
        _require_nonnegative_int(
            self.empty_lesion_case_count,
            field_name="empty_lesion_case_count",
        )
        _require_nonnegative_int(self.skipped_case_count, field_name="skipped_case_count")
        _require_optional_fraction_closed(self.mean_metric_value, field_name="mean_metric_value")
        _require_availability(self.metric_availability_status, self.unavailable_reason)
        if self.metric_availability_status == "available" and self.mean_metric_value is None:
            raise Phase7UncertaintyArtifactValidationError(
                "available subgroup records require mean_metric_value."
            )
        if self.metric_availability_status == "unavailable" and self.mean_metric_value is not None:
            raise Phase7UncertaintyArtifactValidationError(
                "unavailable subgroup records must not report mean_metric_value."
            )


@dataclass(frozen=True, slots=True)
class Phase7LesionSubgroupResult:
    """Immutable deterministic lesion-subgroup artifact."""

    schema_name: str
    schema_version: str
    lesion_subgroup_result_hash: str
    reference_mask_content_hash: str
    prediction_content_hash: str
    metric_name: str
    subgroup_policy_name: str
    thresholds_voxels: tuple[int, ...]
    records: tuple[Phase7LesionSubgroupRecord, ...]

    def __post_init__(self) -> None:
        _require_schema(self.schema_name, PHASE7_LESION_SUBGROUP_RESULT_SCHEMA_NAME)
        _require_version(self.schema_version, PHASE7_LESION_SUBGROUP_RESULT_SCHEMA_VERSION)
        _require_sha256(
            self.lesion_subgroup_result_hash,
            field_name="lesion_subgroup_result_hash",
        )
        _require_sha256(self.reference_mask_content_hash, field_name="reference_mask_content_hash")
        _require_sha256(self.prediction_content_hash, field_name="prediction_content_hash")
        _require_identifier(self.metric_name, field_name="metric_name")
        _require_identifier(self.subgroup_policy_name, field_name="subgroup_policy_name")
        if not self.thresholds_voxels:
            raise Phase7UncertaintyArtifactValidationError("thresholds_voxels must be non-empty.")
        for threshold in self.thresholds_voxels:
            _require_nonnegative_int(threshold, field_name="thresholds_voxels")
        if tuple(sorted(self.thresholds_voxels)) != self.thresholds_voxels:
            raise Phase7UncertaintyArtifactValidationError("thresholds_voxels must be sorted.")
        if not self.records:
            raise Phase7UncertaintyArtifactValidationError("records must be non-empty.")
        if len({record.subgroup_name for record in self.records}) != len(self.records):
            raise Phase7UncertaintyArtifactValidationError(
                "lesion subgroup records must not contain duplicate subgroup names."
            )
        expected_hash = hash_phase7_lesion_subgroup_result(self)
        if self.lesion_subgroup_result_hash != expected_hash:
            raise Phase7UncertaintyArtifactHashError(
                "lesion_subgroup_result_hash does not match deterministic content."
            )


def phase7_uncertainty_result_identity_payload(
    result: Phase7UncertaintyResult,
) -> dict[str, JsonValue]:
    """Return the canonical identity payload for one uncertainty result."""

    return {
        "availability_status": result.availability_status,
        "log_base": result.log_base,
        "max_uncertainty": result.max_uncertainty,
        "mean_uncertainty": result.mean_uncertainty,
        "probability_space": result.probability_space,
        "sample_count": result.sample_count,
        "schema_name": result.schema_name,
        "schema_version": result.schema_version,
        "source_prediction_hash": result.source_prediction_hash,
        "source_probability_hash": result.source_probability_hash,
        "uncertainty_map_content_hash": result.uncertainty_map_content_hash,
        "uncertainty_type": result.uncertainty_type,
        "unavailable_reason": result.unavailable_reason,
        "variance_map_content_hash": result.variance_map_content_hash,
    }


def phase7_uncertainty_result_to_dict(result: Phase7UncertaintyResult) -> dict[str, JsonValue]:
    """Convert one uncertainty result to a canonical mapping."""

    payload = phase7_uncertainty_result_identity_payload(result)
    payload["uncertainty_result_hash"] = result.uncertainty_result_hash
    return payload


def hash_phase7_uncertainty_result(result: Phase7UncertaintyResult) -> str:
    """Return the canonical SHA-256 hash for one uncertainty result."""

    return sha256_json(phase7_uncertainty_result_identity_payload(result))


def phase7_calibration_bin_to_dict(bin_record: Phase7CalibrationBin) -> dict[str, JsonValue]:
    """Convert one calibration bin to a canonical mapping."""

    return {
        "accuracy": bin_record.accuracy,
        "bin_index": bin_record.bin_index,
        "bin_lower": bin_record.bin_lower,
        "bin_upper": bin_record.bin_upper,
        "mean_confidence": bin_record.mean_confidence,
        "schema_name": bin_record.schema_name,
        "schema_version": bin_record.schema_version,
        "upper_inclusive": bin_record.upper_inclusive,
        "voxel_count": bin_record.voxel_count,
        "weighted_error": bin_record.weighted_error,
    }


def phase7_calibration_result_identity_payload(
    result: Phase7CalibrationResult,
) -> dict[str, JsonValue]:
    """Return the canonical identity payload for one calibration result."""

    return {
        "availability_status": result.availability_status,
        "bin_count": result.bin_count,
        "binning_policy": result.binning_policy,
        "bins": [phase7_calibration_bin_to_dict(item) for item in result.bins],
        "calibration_metric": result.calibration_metric,
        "common_grid_geometry_record_hash": result.common_grid_geometry_record_hash,
        "confidence_definition": result.confidence_definition,
        "ece": result.ece,
        "prediction_content_hash": result.prediction_content_hash,
        "probability_content_hash": result.probability_content_hash,
        "reference_mask_content_hash": result.reference_mask_content_hash,
        "schema_name": result.schema_name,
        "schema_version": result.schema_version,
        "unavailable_reason": result.unavailable_reason,
        "valid_voxel_count": result.valid_voxel_count,
    }


def phase7_calibration_result_to_dict(result: Phase7CalibrationResult) -> dict[str, JsonValue]:
    """Convert one calibration result to a canonical mapping."""

    payload = phase7_calibration_result_identity_payload(result)
    payload["calibration_result_hash"] = result.calibration_result_hash
    return payload


def hash_phase7_calibration_result(result: Phase7CalibrationResult) -> str:
    """Return the canonical SHA-256 hash for one calibration result."""

    return sha256_json(phase7_calibration_result_identity_payload(result))


def phase7_risk_coverage_point_to_dict(point: Phase7RiskCoveragePoint) -> dict[str, JsonValue]:
    """Convert one risk-coverage point to a canonical mapping."""

    return {
        "coverage": point.coverage,
        "retained_voxel_count": point.retained_voxel_count,
        "risk": point.risk,
        "schema_name": point.schema_name,
        "schema_version": point.schema_version,
    }


def phase7_risk_coverage_result_identity_payload(
    result: Phase7RiskCoverageResult,
) -> dict[str, JsonValue]:
    """Return the canonical identity payload for one risk-coverage result."""

    return {
        "availability_status": result.availability_status,
        "common_grid_geometry_record_hash": result.common_grid_geometry_record_hash,
        "ordering": result.ordering,
        "points": [phase7_risk_coverage_point_to_dict(item) for item in result.points],
        "prediction_content_hash": result.prediction_content_hash,
        "reference_mask_content_hash": result.reference_mask_content_hash,
        "schema_name": result.schema_name,
        "schema_version": result.schema_version,
        "tie_break_policy": result.tie_break_policy,
        "uncertainty_result_hash": result.uncertainty_result_hash,
        "unavailable_reason": result.unavailable_reason,
        "valid_voxel_count": result.valid_voxel_count,
    }


def phase7_risk_coverage_result_to_dict(
    result: Phase7RiskCoverageResult,
) -> dict[str, JsonValue]:
    """Convert one risk-coverage result to a canonical mapping."""

    payload = phase7_risk_coverage_result_identity_payload(result)
    payload["risk_coverage_result_hash"] = result.risk_coverage_result_hash
    return payload


def hash_phase7_risk_coverage_result(result: Phase7RiskCoverageResult) -> str:
    """Return the canonical SHA-256 hash for one risk-coverage result."""

    return sha256_json(phase7_risk_coverage_result_identity_payload(result))


def phase7_degradation_result_identity_payload(
    result: Phase7DegradationResult,
) -> dict[str, JsonValue]:
    """Return the canonical identity payload for one degradation result."""

    return {
        "absolute_degradation": result.absolute_degradation,
        "availability_status": result.availability_status,
        "baseline_artifact_hash": result.baseline_artifact_hash,
        "baseline_value": result.baseline_value,
        "corrupted_artifact_hash": result.corrupted_artifact_hash,
        "corrupted_value": result.corrupted_value,
        "metric_direction": result.metric_direction,
        "metric_name": result.metric_name,
        "relative_degradation": result.relative_degradation,
        "relative_epsilon": result.relative_epsilon,
        "schema_name": result.schema_name,
        "schema_version": result.schema_version,
        "unavailable_reason": result.unavailable_reason,
    }


def phase7_degradation_result_to_dict(result: Phase7DegradationResult) -> dict[str, JsonValue]:
    """Convert one degradation result to a canonical mapping."""

    payload = phase7_degradation_result_identity_payload(result)
    payload["degradation_result_hash"] = result.degradation_result_hash
    return payload


def hash_phase7_degradation_result(result: Phase7DegradationResult) -> str:
    """Return the canonical SHA-256 hash for one degradation result."""

    return sha256_json(phase7_degradation_result_identity_payload(result))


def phase7_lesion_subgroup_record_to_dict(
    record: Phase7LesionSubgroupRecord,
) -> dict[str, JsonValue]:
    """Convert one lesion-subgroup record to a canonical mapping."""

    return {
        "eligible_case_count": record.eligible_case_count,
        "empty_lesion_case_count": record.empty_lesion_case_count,
        "mean_metric_value": record.mean_metric_value,
        "metric_availability_status": record.metric_availability_status,
        "schema_name": record.schema_name,
        "schema_version": record.schema_version,
        "skipped_case_count": record.skipped_case_count,
        "subgroup_name": record.subgroup_name,
        "unavailable_reason": record.unavailable_reason,
    }


def phase7_lesion_subgroup_result_identity_payload(
    result: Phase7LesionSubgroupResult,
) -> dict[str, JsonValue]:
    """Return the canonical identity payload for one lesion-subgroup result."""

    return {
        "metric_name": result.metric_name,
        "prediction_content_hash": result.prediction_content_hash,
        "records": [phase7_lesion_subgroup_record_to_dict(item) for item in result.records],
        "reference_mask_content_hash": result.reference_mask_content_hash,
        "schema_name": result.schema_name,
        "schema_version": result.schema_version,
        "subgroup_policy_name": result.subgroup_policy_name,
        "thresholds_voxels": list(result.thresholds_voxels),
    }


def phase7_lesion_subgroup_result_to_dict(
    result: Phase7LesionSubgroupResult,
) -> dict[str, JsonValue]:
    """Convert one lesion-subgroup result to a canonical mapping."""

    payload = phase7_lesion_subgroup_result_identity_payload(result)
    payload["lesion_subgroup_result_hash"] = result.lesion_subgroup_result_hash
    return payload


def hash_phase7_lesion_subgroup_result(result: Phase7LesionSubgroupResult) -> str:
    """Return the canonical SHA-256 hash for one lesion-subgroup result."""

    return sha256_json(phase7_lesion_subgroup_result_identity_payload(result))


def phase7_uncertainty_result_from_mapping(mapping: MappingLike) -> Phase7UncertaintyResult:
    """Reconstruct one uncertainty result from a strict mapping."""

    _require_exact_fields(
        mapping,
        required_fields=_UNCERTAINTY_RESULT_FIELDS,
        object_name="Phase7UncertaintyResult",
    )
    return Phase7UncertaintyResult(
        schema_name=_expect_string(mapping["schema_name"], field_name="schema_name"),
        schema_version=_expect_string(mapping["schema_version"], field_name="schema_version"),
        uncertainty_result_hash=_expect_string(
            mapping["uncertainty_result_hash"],
            field_name="uncertainty_result_hash",
        ),
        source_prediction_hash=_expect_string(
            mapping["source_prediction_hash"],
            field_name="source_prediction_hash",
        ),
        source_probability_hash=_expect_string(
            mapping["source_probability_hash"],
            field_name="source_probability_hash",
        ),
        uncertainty_type=_expect_string(mapping["uncertainty_type"], field_name="uncertainty_type"),
        probability_space=_expect_string(
            mapping["probability_space"], field_name="probability_space"
        ),
        log_base=_expect_string(mapping["log_base"], field_name="log_base"),
        sample_count=_expect_int(mapping["sample_count"], field_name="sample_count"),
        uncertainty_map_content_hash=_expect_optional_string(
            mapping["uncertainty_map_content_hash"],
            field_name="uncertainty_map_content_hash",
        ),
        variance_map_content_hash=_expect_optional_string(
            mapping["variance_map_content_hash"],
            field_name="variance_map_content_hash",
        ),
        mean_uncertainty=_expect_optional_float(
            mapping["mean_uncertainty"],
            field_name="mean_uncertainty",
        ),
        max_uncertainty=_expect_optional_float(
            mapping["max_uncertainty"],
            field_name="max_uncertainty",
        ),
        availability_status=_expect_string(
            mapping["availability_status"],
            field_name="availability_status",
        ),
        unavailable_reason=_expect_optional_string(
            mapping["unavailable_reason"],
            field_name="unavailable_reason",
        ),
    )


def phase7_calibration_bin_from_mapping(mapping: MappingLike) -> Phase7CalibrationBin:
    """Reconstruct one calibration bin from a strict mapping."""

    _require_exact_fields(
        mapping,
        required_fields=_CALIBRATION_BIN_FIELDS,
        object_name="Phase7CalibrationBin",
    )
    return Phase7CalibrationBin(
        schema_name=_expect_string(mapping["schema_name"], field_name="schema_name"),
        schema_version=_expect_string(mapping["schema_version"], field_name="schema_version"),
        bin_index=_expect_int(mapping["bin_index"], field_name="bin_index"),
        bin_lower=_expect_float(mapping["bin_lower"], field_name="bin_lower"),
        bin_upper=_expect_float(mapping["bin_upper"], field_name="bin_upper"),
        upper_inclusive=_expect_bool(mapping["upper_inclusive"], field_name="upper_inclusive"),
        voxel_count=_expect_int(mapping["voxel_count"], field_name="voxel_count"),
        accuracy=_expect_optional_float(mapping["accuracy"], field_name="accuracy"),
        mean_confidence=_expect_optional_float(
            mapping["mean_confidence"],
            field_name="mean_confidence",
        ),
        weighted_error=_expect_float(mapping["weighted_error"], field_name="weighted_error"),
    )


def phase7_calibration_result_from_mapping(mapping: MappingLike) -> Phase7CalibrationResult:
    """Reconstruct one calibration result from a strict mapping."""

    _require_exact_fields(
        mapping,
        required_fields=_CALIBRATION_RESULT_FIELDS,
        object_name="Phase7CalibrationResult",
    )
    bins_value = mapping["bins"]
    if not isinstance(bins_value, list):
        raise Phase7UncertaintyArtifactSerializationError("bins must be a list.")
    return Phase7CalibrationResult(
        schema_name=_expect_string(mapping["schema_name"], field_name="schema_name"),
        schema_version=_expect_string(mapping["schema_version"], field_name="schema_version"),
        calibration_result_hash=_expect_string(
            mapping["calibration_result_hash"],
            field_name="calibration_result_hash",
        ),
        prediction_content_hash=_expect_string(
            mapping["prediction_content_hash"],
            field_name="prediction_content_hash",
        ),
        probability_content_hash=_expect_string(
            mapping["probability_content_hash"],
            field_name="probability_content_hash",
        ),
        reference_mask_content_hash=_expect_string(
            mapping["reference_mask_content_hash"],
            field_name="reference_mask_content_hash",
        ),
        common_grid_geometry_record_hash=_expect_string(
            mapping["common_grid_geometry_record_hash"],
            field_name="common_grid_geometry_record_hash",
        ),
        calibration_metric=_expect_string(
            mapping["calibration_metric"],
            field_name="calibration_metric",
        ),
        confidence_definition=_expect_string(
            mapping["confidence_definition"],
            field_name="confidence_definition",
        ),
        binning_policy=_expect_string(mapping["binning_policy"], field_name="binning_policy"),
        bin_count=_expect_int(mapping["bin_count"], field_name="bin_count"),
        valid_voxel_count=_expect_int(mapping["valid_voxel_count"], field_name="valid_voxel_count"),
        ece=_expect_optional_float(mapping["ece"], field_name="ece"),
        bins=tuple(
            phase7_calibration_bin_from_mapping(_expect_mapping(item, field_name=f"bins[{index}]"))
            for index, item in enumerate(bins_value)
        ),
        availability_status=_expect_string(
            mapping["availability_status"],
            field_name="availability_status",
        ),
        unavailable_reason=_expect_optional_string(
            mapping["unavailable_reason"],
            field_name="unavailable_reason",
        ),
    )


def phase7_risk_coverage_point_from_mapping(mapping: MappingLike) -> Phase7RiskCoveragePoint:
    """Reconstruct one risk-coverage point from a strict mapping."""

    _require_exact_fields(
        mapping,
        required_fields=_RISK_COVERAGE_POINT_FIELDS,
        object_name="Phase7RiskCoveragePoint",
    )
    return Phase7RiskCoveragePoint(
        schema_name=_expect_string(mapping["schema_name"], field_name="schema_name"),
        schema_version=_expect_string(mapping["schema_version"], field_name="schema_version"),
        retained_voxel_count=_expect_int(
            mapping["retained_voxel_count"],
            field_name="retained_voxel_count",
        ),
        coverage=_expect_float(mapping["coverage"], field_name="coverage"),
        risk=_expect_float(mapping["risk"], field_name="risk"),
    )


def phase7_risk_coverage_result_from_mapping(mapping: MappingLike) -> Phase7RiskCoverageResult:
    """Reconstruct one risk-coverage result from a strict mapping."""

    _require_exact_fields(
        mapping,
        required_fields=_RISK_COVERAGE_RESULT_FIELDS,
        object_name="Phase7RiskCoverageResult",
    )
    points_value = mapping["points"]
    if not isinstance(points_value, list):
        raise Phase7UncertaintyArtifactSerializationError("points must be a list.")
    return Phase7RiskCoverageResult(
        schema_name=_expect_string(mapping["schema_name"], field_name="schema_name"),
        schema_version=_expect_string(mapping["schema_version"], field_name="schema_version"),
        risk_coverage_result_hash=_expect_string(
            mapping["risk_coverage_result_hash"],
            field_name="risk_coverage_result_hash",
        ),
        uncertainty_result_hash=_expect_string(
            mapping["uncertainty_result_hash"],
            field_name="uncertainty_result_hash",
        ),
        prediction_content_hash=_expect_string(
            mapping["prediction_content_hash"],
            field_name="prediction_content_hash",
        ),
        reference_mask_content_hash=_expect_string(
            mapping["reference_mask_content_hash"],
            field_name="reference_mask_content_hash",
        ),
        common_grid_geometry_record_hash=_expect_string(
            mapping["common_grid_geometry_record_hash"],
            field_name="common_grid_geometry_record_hash",
        ),
        ordering=_expect_string(mapping["ordering"], field_name="ordering"),
        tie_break_policy=_expect_string(mapping["tie_break_policy"], field_name="tie_break_policy"),
        valid_voxel_count=_expect_int(mapping["valid_voxel_count"], field_name="valid_voxel_count"),
        points=tuple(
            phase7_risk_coverage_point_from_mapping(
                _expect_mapping(item, field_name=f"points[{index}]")
            )
            for index, item in enumerate(points_value)
        ),
        availability_status=_expect_string(
            mapping["availability_status"],
            field_name="availability_status",
        ),
        unavailable_reason=_expect_optional_string(
            mapping["unavailable_reason"],
            field_name="unavailable_reason",
        ),
    )


def phase7_degradation_result_from_mapping(mapping: MappingLike) -> Phase7DegradationResult:
    """Reconstruct one degradation result from a strict mapping."""

    _require_exact_fields(
        mapping,
        required_fields=_DEGRADATION_RESULT_FIELDS,
        object_name="Phase7DegradationResult",
    )
    return Phase7DegradationResult(
        schema_name=_expect_string(mapping["schema_name"], field_name="schema_name"),
        schema_version=_expect_string(mapping["schema_version"], field_name="schema_version"),
        degradation_result_hash=_expect_string(
            mapping["degradation_result_hash"],
            field_name="degradation_result_hash",
        ),
        baseline_artifact_hash=_expect_string(
            mapping["baseline_artifact_hash"],
            field_name="baseline_artifact_hash",
        ),
        corrupted_artifact_hash=_expect_string(
            mapping["corrupted_artifact_hash"],
            field_name="corrupted_artifact_hash",
        ),
        metric_name=_expect_string(mapping["metric_name"], field_name="metric_name"),
        metric_direction=_expect_string(mapping["metric_direction"], field_name="metric_direction"),
        baseline_value=_expect_optional_float(
            mapping["baseline_value"], field_name="baseline_value"
        ),
        corrupted_value=_expect_optional_float(
            mapping["corrupted_value"],
            field_name="corrupted_value",
        ),
        absolute_degradation=_expect_optional_float(
            mapping["absolute_degradation"],
            field_name="absolute_degradation",
        ),
        relative_degradation=_expect_optional_float(
            mapping["relative_degradation"],
            field_name="relative_degradation",
        ),
        relative_epsilon=_expect_float(mapping["relative_epsilon"], field_name="relative_epsilon"),
        availability_status=_expect_string(
            mapping["availability_status"],
            field_name="availability_status",
        ),
        unavailable_reason=_expect_optional_string(
            mapping["unavailable_reason"],
            field_name="unavailable_reason",
        ),
    )


def phase7_lesion_subgroup_record_from_mapping(
    mapping: MappingLike,
) -> Phase7LesionSubgroupRecord:
    """Reconstruct one lesion subgroup record from a strict mapping."""

    _require_exact_fields(
        mapping,
        required_fields=_LESION_SUBGROUP_RECORD_FIELDS,
        object_name="Phase7LesionSubgroupRecord",
    )
    return Phase7LesionSubgroupRecord(
        schema_name=_expect_string(mapping["schema_name"], field_name="schema_name"),
        schema_version=_expect_string(mapping["schema_version"], field_name="schema_version"),
        subgroup_name=_expect_string(mapping["subgroup_name"], field_name="subgroup_name"),
        eligible_case_count=_expect_int(
            mapping["eligible_case_count"],
            field_name="eligible_case_count",
        ),
        empty_lesion_case_count=_expect_int(
            mapping["empty_lesion_case_count"],
            field_name="empty_lesion_case_count",
        ),
        skipped_case_count=_expect_int(
            mapping["skipped_case_count"],
            field_name="skipped_case_count",
        ),
        metric_availability_status=_expect_string(
            mapping["metric_availability_status"],
            field_name="metric_availability_status",
        ),
        mean_metric_value=_expect_optional_float(
            mapping["mean_metric_value"],
            field_name="mean_metric_value",
        ),
        unavailable_reason=_expect_optional_string(
            mapping["unavailable_reason"],
            field_name="unavailable_reason",
        ),
    )


def phase7_lesion_subgroup_result_from_mapping(
    mapping: MappingLike,
) -> Phase7LesionSubgroupResult:
    """Reconstruct one lesion subgroup result from a strict mapping."""

    _require_exact_fields(
        mapping,
        required_fields=_LESION_SUBGROUP_RESULT_FIELDS,
        object_name="Phase7LesionSubgroupResult",
    )
    records_value = mapping["records"]
    if not isinstance(records_value, list):
        raise Phase7UncertaintyArtifactSerializationError("records must be a list.")
    thresholds_value = mapping["thresholds_voxels"]
    if not isinstance(thresholds_value, list):
        raise Phase7UncertaintyArtifactSerializationError("thresholds_voxels must be a list.")
    return Phase7LesionSubgroupResult(
        schema_name=_expect_string(mapping["schema_name"], field_name="schema_name"),
        schema_version=_expect_string(mapping["schema_version"], field_name="schema_version"),
        lesion_subgroup_result_hash=_expect_string(
            mapping["lesion_subgroup_result_hash"],
            field_name="lesion_subgroup_result_hash",
        ),
        reference_mask_content_hash=_expect_string(
            mapping["reference_mask_content_hash"],
            field_name="reference_mask_content_hash",
        ),
        prediction_content_hash=_expect_string(
            mapping["prediction_content_hash"],
            field_name="prediction_content_hash",
        ),
        metric_name=_expect_string(mapping["metric_name"], field_name="metric_name"),
        subgroup_policy_name=_expect_string(
            mapping["subgroup_policy_name"],
            field_name="subgroup_policy_name",
        ),
        thresholds_voxels=tuple(
            _expect_int(item, field_name=f"thresholds_voxels[{index}]")
            for index, item in enumerate(thresholds_value)
        ),
        records=tuple(
            phase7_lesion_subgroup_record_from_mapping(
                _expect_mapping(item, field_name=f"records[{index}]")
            )
            for index, item in enumerate(records_value)
        ),
    )


def phase7_uncertainty_result_to_json(result: Phase7UncertaintyResult) -> bytes:
    """Serialize one uncertainty result to canonical JSON bytes."""

    return canonical_json_bytes(phase7_uncertainty_result_to_dict(result)) + b"\n"


def phase7_calibration_result_to_json(result: Phase7CalibrationResult) -> bytes:
    """Serialize one calibration result to canonical JSON bytes."""

    return canonical_json_bytes(phase7_calibration_result_to_dict(result)) + b"\n"


def phase7_risk_coverage_result_to_json(result: Phase7RiskCoverageResult) -> bytes:
    """Serialize one risk-coverage result to canonical JSON bytes."""

    return canonical_json_bytes(phase7_risk_coverage_result_to_dict(result)) + b"\n"


def phase7_degradation_result_to_json(result: Phase7DegradationResult) -> bytes:
    """Serialize one degradation result to canonical JSON bytes."""

    return canonical_json_bytes(phase7_degradation_result_to_dict(result)) + b"\n"


def phase7_lesion_subgroup_result_to_json(result: Phase7LesionSubgroupResult) -> bytes:
    """Serialize one lesion-subgroup result to canonical JSON bytes."""

    return canonical_json_bytes(phase7_lesion_subgroup_result_to_dict(result)) + b"\n"


def phase7_uncertainty_result_from_json(data: bytes | str) -> Phase7UncertaintyResult:
    """Deserialize one uncertainty result from canonical JSON-compatible bytes."""

    return phase7_uncertainty_result_from_mapping(_json_to_mapping(data))


def phase7_calibration_result_from_json(data: bytes | str) -> Phase7CalibrationResult:
    """Deserialize one calibration result from canonical JSON-compatible bytes."""

    return phase7_calibration_result_from_mapping(_json_to_mapping(data))


def phase7_risk_coverage_result_from_json(data: bytes | str) -> Phase7RiskCoverageResult:
    """Deserialize one risk-coverage result from canonical JSON-compatible bytes."""

    return phase7_risk_coverage_result_from_mapping(_json_to_mapping(data))


def phase7_degradation_result_from_json(data: bytes | str) -> Phase7DegradationResult:
    """Deserialize one degradation result from canonical JSON-compatible bytes."""

    return phase7_degradation_result_from_mapping(_json_to_mapping(data))


def phase7_lesion_subgroup_result_from_json(data: bytes | str) -> Phase7LesionSubgroupResult:
    """Deserialize one lesion-subgroup result from canonical JSON-compatible bytes."""

    return phase7_lesion_subgroup_result_from_mapping(_json_to_mapping(data))


_UNCERTAINTY_RESULT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_name",
        "schema_version",
        "uncertainty_result_hash",
        "source_prediction_hash",
        "source_probability_hash",
        "uncertainty_type",
        "probability_space",
        "log_base",
        "sample_count",
        "uncertainty_map_content_hash",
        "variance_map_content_hash",
        "mean_uncertainty",
        "max_uncertainty",
        "availability_status",
        "unavailable_reason",
    }
)
_CALIBRATION_BIN_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_name",
        "schema_version",
        "bin_index",
        "bin_lower",
        "bin_upper",
        "upper_inclusive",
        "voxel_count",
        "accuracy",
        "mean_confidence",
        "weighted_error",
    }
)
_CALIBRATION_RESULT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_name",
        "schema_version",
        "calibration_result_hash",
        "prediction_content_hash",
        "probability_content_hash",
        "reference_mask_content_hash",
        "common_grid_geometry_record_hash",
        "calibration_metric",
        "confidence_definition",
        "binning_policy",
        "bin_count",
        "valid_voxel_count",
        "ece",
        "bins",
        "availability_status",
        "unavailable_reason",
    }
)
_RISK_COVERAGE_POINT_FIELDS: Final[frozenset[str]] = frozenset(
    {"schema_name", "schema_version", "retained_voxel_count", "coverage", "risk"}
)
_RISK_COVERAGE_RESULT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_name",
        "schema_version",
        "risk_coverage_result_hash",
        "uncertainty_result_hash",
        "prediction_content_hash",
        "reference_mask_content_hash",
        "common_grid_geometry_record_hash",
        "ordering",
        "tie_break_policy",
        "valid_voxel_count",
        "points",
        "availability_status",
        "unavailable_reason",
    }
)
_DEGRADATION_RESULT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_name",
        "schema_version",
        "degradation_result_hash",
        "baseline_artifact_hash",
        "corrupted_artifact_hash",
        "metric_name",
        "metric_direction",
        "baseline_value",
        "corrupted_value",
        "absolute_degradation",
        "relative_degradation",
        "relative_epsilon",
        "availability_status",
        "unavailable_reason",
    }
)
_LESION_SUBGROUP_RECORD_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_name",
        "schema_version",
        "subgroup_name",
        "eligible_case_count",
        "empty_lesion_case_count",
        "skipped_case_count",
        "metric_availability_status",
        "mean_metric_value",
        "unavailable_reason",
    }
)
_LESION_SUBGROUP_RESULT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_name",
        "schema_version",
        "lesion_subgroup_result_hash",
        "reference_mask_content_hash",
        "prediction_content_hash",
        "metric_name",
        "subgroup_policy_name",
        "thresholds_voxels",
        "records",
    }
)


def _json_to_mapping(data: bytes | str) -> MappingLike:
    try:
        decoded = json.loads(data)
    except json.JSONDecodeError as exc:
        raise Phase7UncertaintyArtifactSerializationError("invalid JSON artifact payload.") from exc
    if not isinstance(decoded, Mapping):
        raise Phase7UncertaintyArtifactSerializationError("artifact JSON root must be an object.")
    return cast(MappingLike, decoded)


def _require_exact_fields(
    mapping: MappingLike,
    *,
    required_fields: frozenset[str],
    object_name: str,
) -> None:
    keys = set(mapping)
    missing = sorted(required_fields - keys)
    extra = sorted(keys - required_fields)
    if missing or extra:
        raise Phase7UncertaintyArtifactSerializationError(
            f"{object_name} fields mismatch; missing={missing!r}, extra={extra!r}."
        )


def _expect_mapping(value: object, *, field_name: str) -> MappingLike:
    if not isinstance(value, Mapping):
        raise Phase7UncertaintyArtifactSerializationError(f"{field_name} must be a mapping.")
    return cast(MappingLike, value)


def _expect_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise Phase7UncertaintyArtifactSerializationError(f"{field_name} must be a string.")
    return value


def _expect_optional_string(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _expect_string(value, field_name=field_name)


def _expect_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise Phase7UncertaintyArtifactSerializationError(f"{field_name} must be an integer.")
    return value


def _expect_float(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise Phase7UncertaintyArtifactSerializationError(f"{field_name} must be a finite number.")
    return float(value)


def _expect_optional_float(value: object, *, field_name: str) -> float | None:
    if value is None:
        return None
    return _expect_float(value, field_name=field_name)


def _expect_bool(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise Phase7UncertaintyArtifactSerializationError(f"{field_name} must be a boolean.")
    return value


def _require_schema(value: str, expected: str) -> None:
    if value != expected:
        raise Phase7UncertaintyArtifactValidationError(f"schema_name must be {expected!r}.")


def _require_version(value: str, expected: str) -> None:
    if value != expected:
        raise Phase7UncertaintyArtifactValidationError(f"schema_version must be {expected!r}.")


def _require_sha256(value: str, *, field_name: str) -> None:
    if not _SHA256_RE.fullmatch(value):
        raise Phase7UncertaintyArtifactValidationError(
            f"{field_name} must be a lowercase SHA-256 hex digest."
        )


def _require_optional_sha256(value: str | None, *, field_name: str) -> None:
    if value is not None:
        _require_sha256(value, field_name=field_name)


def _require_identifier(value: str, *, field_name: str) -> None:
    if not _IDENTIFIER_RE.fullmatch(value):
        raise Phase7UncertaintyArtifactValidationError(
            f"{field_name} must be a conservative identifier."
        )


def _require_optional_identifier(value: str | None, *, field_name: str) -> None:
    if value is not None:
        _require_identifier(value, field_name=field_name)


def _require_allowed(value: str, allowed: frozenset[str], *, field_name: str) -> None:
    if value not in allowed:
        raise Phase7UncertaintyArtifactValidationError(
            f"{field_name} must be one of {sorted(allowed)!r}."
        )


def _require_positive_int(value: int, *, field_name: str) -> None:
    if isinstance(value, bool) or value <= 0:
        raise Phase7UncertaintyArtifactValidationError(f"{field_name} must be a positive integer.")


def _require_nonnegative_int(value: int, *, field_name: str) -> None:
    if isinstance(value, bool) or value < 0:
        raise Phase7UncertaintyArtifactValidationError(
            f"{field_name} must be a nonnegative integer."
        )


def _require_finite_float(value: float, *, field_name: str) -> None:
    if not math.isfinite(value):
        raise Phase7UncertaintyArtifactValidationError(f"{field_name} must be finite.")


def _require_positive_float(value: float, *, field_name: str) -> None:
    _require_finite_float(value, field_name=field_name)
    if value <= 0.0:
        raise Phase7UncertaintyArtifactValidationError(f"{field_name} must be positive.")


def _require_nonnegative_float(value: float, *, field_name: str) -> None:
    _require_finite_float(value, field_name=field_name)
    if value < 0.0:
        raise Phase7UncertaintyArtifactValidationError(f"{field_name} must be nonnegative.")


def _require_optional_nonnegative_float(value: float | None, *, field_name: str) -> None:
    if value is not None:
        _require_nonnegative_float(value, field_name=field_name)


def _require_optional_finite_float(value: float | None, *, field_name: str) -> None:
    if value is not None:
        _require_finite_float(value, field_name=field_name)


def _require_fraction_closed(value: float, *, field_name: str) -> None:
    _require_finite_float(value, field_name=field_name)
    if value < 0.0 or value > 1.0:
        raise Phase7UncertaintyArtifactValidationError(f"{field_name} must be in [0, 1].")


def _require_optional_fraction_closed(value: float | None, *, field_name: str) -> None:
    if value is not None:
        _require_fraction_closed(value, field_name=field_name)


def _require_availability(status: str, reason: str | None) -> None:
    _require_allowed(status, PHASE7_AVAILABILITY_STATUSES, field_name="availability_status")
    _require_optional_identifier(reason, field_name="unavailable_reason")
    if status == "available" and reason is not None:
        raise Phase7UncertaintyArtifactValidationError(
            "available artifacts must not contain unavailable_reason."
        )
    if status == "unavailable" and reason is None:
        raise Phase7UncertaintyArtifactValidationError(
            "unavailable artifacts require unavailable_reason."
        )


def _require_optional_message(value: str | None, *, field_name: str) -> None:
    if value is not None and not _MESSAGE_RE.fullmatch(value):
        raise Phase7UncertaintyArtifactValidationError(
            f"{field_name} must be a single-line message."
        )


def _ensure_json_value(value: object, *, field_name: str) -> JsonValue:
    canonical_json_bytes(value)
    return cast(JsonValue, value)
