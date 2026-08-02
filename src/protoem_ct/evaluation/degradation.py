"""Deterministic Phase 7 metric degradation utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

from protoem_ct.artifacts.hashing import sha256_json
from protoem_ct.uncertainty.artifacts import (
    PHASE7_DEGRADATION_RESULT_SCHEMA_NAME,
    PHASE7_DEGRADATION_RESULT_SCHEMA_VERSION,
    PHASE7_METRIC_DIRECTIONS,
    Phase7DegradationResult,
)

DEFAULT_RELATIVE_DEGRADATION_EPSILON: Final[float] = 1e-8


class Phase7DegradationError(ValueError):
    """Raised when Phase 7 degradation analysis cannot be computed."""


class Phase7DegradationInputError(Phase7DegradationError):
    """Raised when degradation inputs are malformed."""


class Phase7DegradationArtifactError(Phase7DegradationError):
    """Raised when a degradation artifact cannot represent computed values."""


@dataclass(frozen=True, slots=True)
class MetricDegradationValues:
    """Direction-aware degradation values for one scalar metric comparison."""

    metric_name: str
    metric_direction: str
    baseline_value: float
    corrupted_value: float
    absolute_degradation: float
    relative_degradation: float | None
    relative_epsilon: float
    relative_available: bool


def compute_metric_degradation(
    *,
    metric_name: str,
    metric_direction: str,
    baseline_value: float,
    corrupted_value: float,
    relative_epsilon: float = DEFAULT_RELATIVE_DEGRADATION_EPSILON,
) -> MetricDegradationValues:
    """Compute absolute and relative degradation for one scalar metric.

    For ``higher_is_better`` metrics, degradation is ``baseline - corrupted``.
    For ``lower_is_better`` metrics, degradation is ``corrupted - baseline``.
    Relative degradation is unavailable when ``abs(baseline) < relative_epsilon``
    because the denominator would make the reported ratio misleading.
    """

    _require_metric_name(metric_name)
    _require_metric_direction(metric_direction)
    baseline = _require_finite_float(baseline_value, "baseline_value")
    corrupted = _require_finite_float(corrupted_value, "corrupted_value")
    epsilon = _require_positive_finite_float(relative_epsilon, "relative_epsilon")

    if metric_direction == "higher_is_better":
        absolute = baseline - corrupted
    else:
        absolute = corrupted - baseline

    relative_available = abs(baseline) >= epsilon
    relative = absolute / max(abs(baseline), epsilon) if relative_available else None
    return MetricDegradationValues(
        metric_name=metric_name,
        metric_direction=metric_direction,
        baseline_value=baseline,
        corrupted_value=corrupted,
        absolute_degradation=absolute,
        relative_degradation=relative,
        relative_epsilon=epsilon,
        relative_available=relative_available,
    )


def build_phase7_degradation_result(
    *,
    baseline_artifact_hash: str,
    corrupted_artifact_hash: str,
    metric_name: str,
    metric_direction: str,
    baseline_value: float,
    corrupted_value: float,
    relative_epsilon: float = DEFAULT_RELATIVE_DEGRADATION_EPSILON,
) -> Phase7DegradationResult:
    """Build a self-validating Phase 7 degradation artifact."""

    _require_sha256(baseline_artifact_hash, "baseline_artifact_hash")
    _require_sha256(corrupted_artifact_hash, "corrupted_artifact_hash")
    values = compute_metric_degradation(
        metric_name=metric_name,
        metric_direction=metric_direction,
        baseline_value=baseline_value,
        corrupted_value=corrupted_value,
        relative_epsilon=relative_epsilon,
    )
    payload = {
        "absolute_degradation": values.absolute_degradation,
        "availability_status": "available",
        "baseline_artifact_hash": baseline_artifact_hash,
        "baseline_value": values.baseline_value,
        "corrupted_artifact_hash": corrupted_artifact_hash,
        "corrupted_value": values.corrupted_value,
        "metric_direction": values.metric_direction,
        "metric_name": values.metric_name,
        "relative_degradation": values.relative_degradation,
        "relative_epsilon": values.relative_epsilon,
        "schema_name": PHASE7_DEGRADATION_RESULT_SCHEMA_NAME,
        "schema_version": PHASE7_DEGRADATION_RESULT_SCHEMA_VERSION,
        "unavailable_reason": None,
    }
    try:
        return Phase7DegradationResult(
            schema_name=PHASE7_DEGRADATION_RESULT_SCHEMA_NAME,
            schema_version=PHASE7_DEGRADATION_RESULT_SCHEMA_VERSION,
            degradation_result_hash=sha256_json(payload),
            baseline_artifact_hash=baseline_artifact_hash,
            corrupted_artifact_hash=corrupted_artifact_hash,
            metric_name=values.metric_name,
            metric_direction=values.metric_direction,
            baseline_value=values.baseline_value,
            corrupted_value=values.corrupted_value,
            absolute_degradation=values.absolute_degradation,
            relative_degradation=values.relative_degradation,
            relative_epsilon=values.relative_epsilon,
            availability_status="available",
            unavailable_reason=None,
        )
    except ValueError as exc:
        msg = "computed degradation values could not be represented by the Phase 7 artifact schema"
        raise Phase7DegradationArtifactError(msg) from exc


def query_label_not_part_of_phase7_degradation_api() -> bool:
    """Return True because degradation consumes only completed scalar metrics."""

    return True


def _require_metric_name(value: str) -> None:
    if not value or not value.replace("_", "").replace("-", "").isalnum():
        raise Phase7DegradationInputError("metric_name must be a conservative identifier.")


def _require_metric_direction(value: str) -> None:
    if value not in PHASE7_METRIC_DIRECTIONS:
        raise Phase7DegradationInputError(
            f"metric_direction must be one of {sorted(PHASE7_METRIC_DIRECTIONS)!r}."
        )


def _require_finite_float(value: float, field_name: str) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise Phase7DegradationInputError(f"{field_name} must be finite.")
    return converted


def _require_positive_finite_float(value: float, field_name: str) -> float:
    converted = _require_finite_float(value, field_name)
    if converted <= 0.0:
        raise Phase7DegradationInputError(f"{field_name} must be positive.")
    return converted


def _require_sha256(value: str, field_name: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise Phase7DegradationInputError(f"{field_name} must be a lowercase SHA-256 hex digest.")
