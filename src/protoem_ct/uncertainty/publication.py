"""JSON-first Phase 7 uncertainty publication helpers."""

from __future__ import annotations

import io
import json
from collections.abc import Mapping
from typing import Final, cast

from protoem_ct.artifacts.hashing import JsonValue, canonical_json_bytes
from protoem_ct.evaluation.failure_detection import (
    AvailabilityStatus,
    Phase7FailureDetectionAUROCResult,
    Phase7UncertaintyErrorCorrelationResult,
    failure_detection_auroc_result_to_dict,
    uncertainty_error_correlation_result_to_dict,
)
from protoem_ct.uncertainty.artifacts import (
    phase7_calibration_result_from_json,
    phase7_lesion_subgroup_result_from_json,
    phase7_risk_coverage_result_from_json,
    phase7_uncertainty_result_from_json,
)

PHASE7_FAILURE_DETECTION_COLLECTION_SCHEMA_NAME: Final[str] = "phase7_failure_detection_results"
PHASE7_FAILURE_DETECTION_COLLECTION_SCHEMA_VERSION: Final[str] = "v1"

_FAILURE_DETECTION_COLLECTION_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_name",
        "schema_version",
        "uncertainty_error_correlation",
        "failure_detection_auroc",
    }
)
_CORRELATION_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_name",
        "schema_version",
        "correlation_result_hash",
        "uncertainty_content_hash",
        "error_indicator_content_hash",
        "valid_mask_content_hash",
        "valid_voxel_count",
        "correlation_method",
        "correlation_value",
        "availability_status",
        "unavailable_reason",
    }
)
_AUROC_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_name",
        "schema_version",
        "auroc_result_hash",
        "case_uncertainty_score_content_hash",
        "failure_indicator_definition",
        "failure_indicator_content_hash",
        "failure_metric_name",
        "failure_metric_threshold",
        "case_count",
        "positive_failure_case_count",
        "negative_nonfailure_case_count",
        "auroc_method",
        "auroc_value",
        "availability_status",
        "unavailable_reason",
    }
)


class Phase7UncertaintyPublicationError(ValueError):
    """Raised when Phase 7 uncertainty publication inputs are invalid."""


def render_phase7_uncertainty_failure_table_markdown_from_json(
    *,
    uncertainty_result_json: bytes,
    failure_detection_json: bytes,
) -> bytes:
    """Render the uncertainty/failure table strictly from validated JSON bytes."""

    uncertainty = phase7_uncertainty_result_from_json(uncertainty_result_json)
    correlation, auroc = phase7_failure_detection_results_from_json(failure_detection_json)
    rows = [
        (
            "uncertainty",
            uncertainty.uncertainty_type,
            uncertainty.availability_status,
            _format_optional_float(uncertainty.mean_uncertainty),
            uncertainty.unavailable_reason or "none",
        ),
        (
            "uncertainty_error_correlation",
            correlation.correlation_method,
            correlation.availability_status,
            _format_optional_float(correlation.correlation_value),
            correlation.unavailable_reason or "none",
        ),
        (
            "failure_detection_auroc",
            auroc.auroc_method,
            auroc.availability_status,
            _format_optional_float(auroc.auroc_value),
            auroc.unavailable_reason or "none",
        ),
    ]
    lines = [
        "# Phase 7 Uncertainty and Failure Table",
        "",
        "| artifact | method | availability | value | unavailable_reason |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines.extend(
        f"| {name} | {method} | {status} | {value} | {reason} |"
        for name, method, status, value, reason in rows
    )
    return ("\n".join(lines) + "\n").encode("utf-8")


def render_phase7_subgroup_table_markdown_from_json(
    lesion_subgroup_result_json: bytes,
) -> bytes:
    """Render the lesion subgroup table strictly from validated JSON bytes."""

    result = phase7_lesion_subgroup_result_from_json(lesion_subgroup_result_json)
    lines = [
        "# Phase 7 Lesion Subgroup Table",
        "",
        f"- metric_name: `{result.metric_name}`",
        f"- subgroup_policy: `{result.subgroup_policy_name}`",
        "",
        (
            "| subgroup | eligible_cases | empty_lesion_cases | skipped_cases | "
            "availability | mean_metric | unavailable_reason |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for record in result.records:
        lines.append(
            "| "
            f"{record.subgroup_name} | "
            f"{record.eligible_case_count} | "
            f"{record.empty_lesion_case_count} | "
            f"{record.skipped_case_count} | "
            f"{record.metric_availability_status} | "
            f"{_format_optional_float(record.mean_metric_value)} | "
            f"{record.unavailable_reason or 'none'} |"
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def build_phase7_calibration_plot_png(calibration_result_json: bytes) -> bytes:
    """Render a deterministic calibration plot strictly from calibration JSON."""

    result = phase7_calibration_result_from_json(calibration_result_json)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - exercised only when matplotlib is unavailable.
        raise Phase7UncertaintyPublicationError("failed to import matplotlib.") from exc

    indices = [item.bin_index for item in result.bins]
    accuracy = [0.0 if item.accuracy is None else item.accuracy for item in result.bins]
    confidence = [
        0.0 if item.mean_confidence is None else item.mean_confidence for item in result.bins
    ]
    figure, axis = plt.subplots(figsize=(7.0, 4.0), dpi=120)
    axis.plot(indices, accuracy, label="accuracy", marker="o")
    axis.plot(indices, confidence, label="confidence", marker="s")
    axis.set_xlabel("calibration bin")
    axis.set_ylabel("value")
    axis.set_ylim(0.0, 1.0)
    axis.set_title("Phase 7 calibration")
    axis.grid(True, linewidth=0.5, alpha=0.35)
    axis.legend(loc="best", frameon=False)
    figure.tight_layout()
    output = io.BytesIO()
    figure.savefig(output, format="png", dpi=120, metadata={"Software": "protoem_ct"})
    plt.close(figure)
    return output.getvalue()


def build_phase7_risk_coverage_plot_png(risk_coverage_result_json: bytes) -> bytes:
    """Render a deterministic risk-coverage plot strictly from risk JSON."""

    result = phase7_risk_coverage_result_from_json(risk_coverage_result_json)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - exercised only when matplotlib is unavailable.
        raise Phase7UncertaintyPublicationError("failed to import matplotlib.") from exc

    coverage = [point.coverage for point in result.points]
    risk = [point.risk for point in result.points]
    figure, axis = plt.subplots(figsize=(7.0, 4.0), dpi=120)
    axis.plot(coverage, risk, label="risk", marker="o")
    axis.set_xlabel("coverage")
    axis.set_ylabel("risk")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_title("Phase 7 risk-coverage")
    axis.grid(True, linewidth=0.5, alpha=0.35)
    axis.legend(loc="best", frameon=False)
    figure.tight_layout()
    output = io.BytesIO()
    figure.savefig(output, format="png", dpi=120, metadata={"Software": "protoem_ct"})
    plt.close(figure)
    return output.getvalue()


def canonical_phase7_failure_detection_results_json(
    *,
    correlation_result: Phase7UncertaintyErrorCorrelationResult,
    auroc_result: Phase7FailureDetectionAUROCResult,
) -> bytes:
    """Serialize validated failure-detection results as one publication JSON object."""

    payload: dict[str, JsonValue] = {
        "failure_detection_auroc": failure_detection_auroc_result_to_dict(auroc_result),
        "schema_name": PHASE7_FAILURE_DETECTION_COLLECTION_SCHEMA_NAME,
        "schema_version": PHASE7_FAILURE_DETECTION_COLLECTION_SCHEMA_VERSION,
        "uncertainty_error_correlation": uncertainty_error_correlation_result_to_dict(
            correlation_result
        ),
    }
    return canonical_json_bytes(payload) + b"\n"


def phase7_failure_detection_results_from_json(
    data: bytes | str,
) -> tuple[Phase7UncertaintyErrorCorrelationResult, Phase7FailureDetectionAUROCResult]:
    """Validate and reconstruct the Phase 7 failure-detection publication JSON."""

    mapping = _parse_json_mapping(data)
    _require_exact_fields(
        mapping,
        required_fields=_FAILURE_DETECTION_COLLECTION_FIELDS,
        object_name="Phase7FailureDetectionResults",
    )
    if mapping["schema_name"] != PHASE7_FAILURE_DETECTION_COLLECTION_SCHEMA_NAME:
        raise Phase7UncertaintyPublicationError("invalid failure-detection schema_name.")
    if mapping["schema_version"] != PHASE7_FAILURE_DETECTION_COLLECTION_SCHEMA_VERSION:
        raise Phase7UncertaintyPublicationError("invalid failure-detection schema_version.")
    correlation = _correlation_result_from_mapping(
        _expect_mapping(
            mapping["uncertainty_error_correlation"],
            field_name="uncertainty_error_correlation",
        )
    )
    auroc = _auroc_result_from_mapping(
        _expect_mapping(mapping["failure_detection_auroc"], field_name="failure_detection_auroc")
    )
    return correlation, auroc


def _correlation_result_from_mapping(
    mapping: Mapping[str, object],
) -> Phase7UncertaintyErrorCorrelationResult:
    _require_exact_fields(
        mapping,
        required_fields=_CORRELATION_FIELDS,
        object_name="Phase7UncertaintyErrorCorrelationResult",
    )
    return Phase7UncertaintyErrorCorrelationResult(
        schema_name=_expect_string(mapping["schema_name"], field_name="schema_name"),
        schema_version=_expect_string(mapping["schema_version"], field_name="schema_version"),
        correlation_result_hash=_expect_string(
            mapping["correlation_result_hash"],
            field_name="correlation_result_hash",
        ),
        uncertainty_content_hash=_expect_string(
            mapping["uncertainty_content_hash"],
            field_name="uncertainty_content_hash",
        ),
        error_indicator_content_hash=_expect_string(
            mapping["error_indicator_content_hash"],
            field_name="error_indicator_content_hash",
        ),
        valid_mask_content_hash=_expect_optional_string(
            mapping["valid_mask_content_hash"],
            field_name="valid_mask_content_hash",
        ),
        valid_voxel_count=_expect_int(
            mapping["valid_voxel_count"],
            field_name="valid_voxel_count",
        ),
        correlation_method=_expect_string(
            mapping["correlation_method"],
            field_name="correlation_method",
        ),
        correlation_value=_expect_optional_float(
            mapping["correlation_value"],
            field_name="correlation_value",
        ),
        availability_status=_expect_availability_status(
            mapping["availability_status"],
            field_name="availability_status",
        ),
        unavailable_reason=_expect_optional_string(
            mapping["unavailable_reason"],
            field_name="unavailable_reason",
        ),
    )


def _auroc_result_from_mapping(mapping: Mapping[str, object]) -> Phase7FailureDetectionAUROCResult:
    _require_exact_fields(
        mapping,
        required_fields=_AUROC_FIELDS,
        object_name="Phase7FailureDetectionAUROCResult",
    )
    return Phase7FailureDetectionAUROCResult(
        schema_name=_expect_string(mapping["schema_name"], field_name="schema_name"),
        schema_version=_expect_string(mapping["schema_version"], field_name="schema_version"),
        auroc_result_hash=_expect_string(
            mapping["auroc_result_hash"], field_name="auroc_result_hash"
        ),
        case_uncertainty_score_content_hash=_expect_string(
            mapping["case_uncertainty_score_content_hash"],
            field_name="case_uncertainty_score_content_hash",
        ),
        failure_indicator_content_hash=_expect_string(
            mapping["failure_indicator_content_hash"],
            field_name="failure_indicator_content_hash",
        ),
        failure_metric_name=_expect_string(
            mapping["failure_metric_name"],
            field_name="failure_metric_name",
        ),
        failure_metric_threshold=_expect_float(
            mapping["failure_metric_threshold"],
            field_name="failure_metric_threshold",
        ),
        failure_indicator_definition=_expect_string(
            mapping["failure_indicator_definition"],
            field_name="failure_indicator_definition",
        ),
        case_count=_expect_int(mapping["case_count"], field_name="case_count"),
        positive_failure_case_count=_expect_int(
            mapping["positive_failure_case_count"],
            field_name="positive_failure_case_count",
        ),
        negative_nonfailure_case_count=_expect_int(
            mapping["negative_nonfailure_case_count"],
            field_name="negative_nonfailure_case_count",
        ),
        auroc_method=_expect_string(mapping["auroc_method"], field_name="auroc_method"),
        auroc_value=_expect_optional_float(mapping["auroc_value"], field_name="auroc_value"),
        availability_status=_expect_availability_status(
            mapping["availability_status"],
            field_name="availability_status",
        ),
        unavailable_reason=_expect_optional_string(
            mapping["unavailable_reason"],
            field_name="unavailable_reason",
        ),
    )


def _parse_json_mapping(data: bytes | str) -> Mapping[str, object]:
    try:
        loaded = json.loads(data)
    except json.JSONDecodeError as exc:
        raise Phase7UncertaintyPublicationError("publication input must be valid JSON.") from exc
    return _expect_mapping(loaded, field_name="json")


def _expect_mapping(value: object, *, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise Phase7UncertaintyPublicationError(f"{field_name} must be a JSON object.")
    return cast(Mapping[str, object], value)


def _require_exact_fields(
    mapping: Mapping[str, object],
    *,
    required_fields: frozenset[str],
    object_name: str,
) -> None:
    actual_fields = frozenset(mapping)
    if actual_fields != required_fields:
        missing = sorted(required_fields - actual_fields)
        extra = sorted(actual_fields - required_fields)
        raise Phase7UncertaintyPublicationError(
            f"{object_name} fields mismatch; missing={missing!r}, extra={extra!r}."
        )


def _expect_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise Phase7UncertaintyPublicationError(f"{field_name} must be a string.")
    return value


def _expect_optional_string(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _expect_string(value, field_name=field_name)


def _expect_int(value: object, *, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise Phase7UncertaintyPublicationError(f"{field_name} must be an integer.")
    return value


def _expect_availability_status(value: object, *, field_name: str) -> AvailabilityStatus:
    status = _expect_string(value, field_name=field_name)
    if status not in {"available", "unavailable"}:
        raise Phase7UncertaintyPublicationError(
            f"{field_name} must be 'available' or 'unavailable'."
        )
    return cast(AvailabilityStatus, status)


def _expect_optional_float(value: object, *, field_name: str) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise Phase7UncertaintyPublicationError(f"{field_name} must be numeric or null.")
    return float(value)


def _expect_float(value: object, *, field_name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise Phase7UncertaintyPublicationError(f"{field_name} must be numeric.")
    return float(value)


def _format_optional_float(value: float | None) -> str:
    return "null" if value is None else f"{value:.12g}"


__all__ = [
    "PHASE7_FAILURE_DETECTION_COLLECTION_SCHEMA_NAME",
    "PHASE7_FAILURE_DETECTION_COLLECTION_SCHEMA_VERSION",
    "Phase7UncertaintyPublicationError",
    "build_phase7_calibration_plot_png",
    "build_phase7_risk_coverage_plot_png",
    "canonical_phase7_failure_detection_results_json",
    "phase7_failure_detection_results_from_json",
    "render_phase7_subgroup_table_markdown_from_json",
    "render_phase7_uncertainty_failure_table_markdown_from_json",
]
