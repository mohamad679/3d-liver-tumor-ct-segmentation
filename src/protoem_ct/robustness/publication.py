"""Deterministic JSON-first Phase 7 robustness and uncertainty publication."""

from __future__ import annotations

import io
import json
import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

from protoem_ct.artifacts.hashing import JsonValue, canonical_json_bytes, sha256_json
from protoem_ct.data import InvalidDatasetRootError, validate_explicit_external_output_root
from protoem_ct.robustness.artifacts import (
    PHASE7_CORRUPTION_NAMES,
    Phase7CorruptionManifest,
    Phase7GeometryRecord,
    Phase7RunSummary,
    Phase7TransformResult,
    phase7_corruption_manifest_from_json,
    phase7_corruption_manifest_to_json,
    phase7_geometry_record_from_mapping,
    phase7_geometry_record_to_dict,
    phase7_run_summary_from_json,
    phase7_run_summary_to_json,
    phase7_transform_result_from_mapping,
    phase7_transform_result_to_dict,
)
from protoem_ct.uncertainty.artifacts import (
    phase7_calibration_result_from_json,
    phase7_calibration_result_to_json,
    phase7_degradation_result_from_json,
    phase7_degradation_result_to_json,
    phase7_lesion_subgroup_result_from_json,
    phase7_lesion_subgroup_result_to_json,
    phase7_risk_coverage_result_from_json,
    phase7_risk_coverage_result_to_json,
    phase7_uncertainty_result_from_json,
    phase7_uncertainty_result_to_json,
)
from protoem_ct.uncertainty.publication import (
    build_phase7_calibration_plot_png,
    build_phase7_risk_coverage_plot_png,
    phase7_failure_detection_results_from_json,
    render_phase7_subgroup_table_markdown_from_json,
    render_phase7_uncertainty_failure_table_markdown_from_json,
)

PHASE7_EFFECTIVE_CONFIG_NAME: Final[str] = "effective_config.json"
PHASE7_CORRUPTION_MANIFEST_NAME: Final[str] = "corruption_manifest.json"
PHASE7_TRANSFORM_RESULTS_NAME: Final[str] = "transform_results.json"
PHASE7_GEOMETRY_RECORDS_NAME: Final[str] = "geometry_records.json"
PHASE7_UNCERTAINTY_RESULT_NAME: Final[str] = "uncertainty_result.json"
PHASE7_CALIBRATION_RESULT_NAME: Final[str] = "calibration_result.json"
PHASE7_RISK_COVERAGE_RESULT_NAME: Final[str] = "risk_coverage_result.json"
PHASE7_FAILURE_DETECTION_NAME: Final[str] = "failure_detection.json"
PHASE7_DEGRADATION_RESULT_NAME: Final[str] = "degradation_result.json"
PHASE7_LESION_SUBGROUP_RESULT_NAME: Final[str] = "lesion_subgroup_result.json"
PHASE7_RUN_SUMMARY_NAME: Final[str] = "phase7_run_summary.json"
PHASE7_CORRUPTION_DEGRADATION_PLOT_NAME: Final[str] = "corruption_degradation_plot.png"
PHASE7_CALIBRATION_PLOT_NAME: Final[str] = "calibration_plot.png"
PHASE7_RISK_COVERAGE_PLOT_NAME: Final[str] = "risk_coverage_plot.png"
PHASE7_UNCERTAINTY_FAILURE_TABLE_NAME: Final[str] = "uncertainty_failure_table.md"
PHASE7_SUBGROUP_TABLE_NAME: Final[str] = "subgroup_table.md"
PHASE7_SUMMARY_MARKDOWN_NAME: Final[str] = "phase7_summary.md"

PHASE7_TRANSFORM_RESULTS_COLLECTION_SCHEMA_NAME: Final[str] = "phase7_transform_results"
PHASE7_GEOMETRY_RECORDS_COLLECTION_SCHEMA_NAME: Final[str] = "phase7_geometry_records"
PHASE7_PUBLICATION_COLLECTION_SCHEMA_VERSION: Final[str] = "v1"

PHASE7_PUBLICATION_FILENAMES: Final[tuple[str, ...]] = (
    PHASE7_EFFECTIVE_CONFIG_NAME,
    PHASE7_CORRUPTION_MANIFEST_NAME,
    PHASE7_TRANSFORM_RESULTS_NAME,
    PHASE7_GEOMETRY_RECORDS_NAME,
    PHASE7_UNCERTAINTY_RESULT_NAME,
    PHASE7_CALIBRATION_RESULT_NAME,
    PHASE7_RISK_COVERAGE_RESULT_NAME,
    PHASE7_FAILURE_DETECTION_NAME,
    PHASE7_DEGRADATION_RESULT_NAME,
    PHASE7_LESION_SUBGROUP_RESULT_NAME,
    PHASE7_RUN_SUMMARY_NAME,
    PHASE7_CORRUPTION_DEGRADATION_PLOT_NAME,
    PHASE7_CALIBRATION_PLOT_NAME,
    PHASE7_RISK_COVERAGE_PLOT_NAME,
    PHASE7_UNCERTAINTY_FAILURE_TABLE_NAME,
    PHASE7_SUBGROUP_TABLE_NAME,
    PHASE7_SUMMARY_MARKDOWN_NAME,
)

_COLLECTION_FIELDS: Final[frozenset[str]] = frozenset({"schema_name", "schema_version", "items"})


class Phase7PublicationError(ValueError):
    """Base error for Phase 7 publication."""


class Phase7PublicationValidationError(Phase7PublicationError):
    """Raised when Phase 7 publication inputs are malformed or inconsistent."""


class Phase7PublicationPathError(Phase7PublicationError):
    """Raised when the Phase 7 output root violates path-safety rules."""


class Phase7PublicationCollisionError(Phase7PublicationError):
    """Raised when Phase 7 publication would overwrite non-identical artifacts."""


class Phase7PublicationIOError(Phase7PublicationError):
    """Raised when staged Phase 7 filesystem publication fails."""


@dataclass(frozen=True, slots=True)
class Phase7PublicationInputs:
    """Validated JSON byte surfaces required for Phase 7 publication."""

    effective_config_json: bytes
    corruption_manifest_json: bytes
    transform_results_json: bytes
    geometry_records_json: bytes
    uncertainty_result_json: bytes
    calibration_result_json: bytes
    risk_coverage_result_json: bytes
    failure_detection_json: bytes
    degradation_result_json: bytes
    lesion_subgroup_result_json: bytes
    phase7_run_summary_json: bytes


@dataclass(frozen=True, slots=True)
class Phase7PublicationResult:
    """Deterministic paths for one completed Phase 7 publication."""

    output_root: Path
    reused_existing_output: bool
    published_filenames: tuple[str, ...]


def publish_phase7_artifacts(
    *,
    output_root: Path,
    inputs: Phase7PublicationInputs,
) -> Phase7PublicationResult:
    """Publish deterministic Phase 7 artifacts using atomic staged external writes."""

    artifact_bytes = build_phase7_publication_artifact_bytes(inputs)
    validated_output_root = _validate_phase7_output_root(output_root)
    reused_existing_output = _publish_directory_tree_if_absent_or_equal(
        output_root=validated_output_root,
        artifact_bytes=artifact_bytes,
    )
    return Phase7PublicationResult(
        output_root=validated_output_root,
        reused_existing_output=reused_existing_output,
        published_filenames=tuple(sorted(artifact_bytes)),
    )


def build_phase7_publication_artifact_bytes(
    inputs: Phase7PublicationInputs,
) -> dict[str, bytes]:
    """Validate all JSON bytes and derive every Markdown/plot artifact from them."""

    validated = _ValidatedPhase7PublicationInputs.from_inputs(inputs)
    uncertainty_failure_table = render_phase7_uncertainty_failure_table_markdown_from_json(
        uncertainty_result_json=validated.uncertainty_result_json,
        failure_detection_json=validated.failure_detection_json,
    )
    subgroup_table = render_phase7_subgroup_table_markdown_from_json(
        validated.lesion_subgroup_result_json
    )
    summary_markdown = render_phase7_summary_markdown_from_json_artifacts(validated)
    artifact_bytes = {
        PHASE7_EFFECTIVE_CONFIG_NAME: validated.effective_config_json,
        PHASE7_CORRUPTION_MANIFEST_NAME: validated.corruption_manifest_json,
        PHASE7_TRANSFORM_RESULTS_NAME: validated.transform_results_json,
        PHASE7_GEOMETRY_RECORDS_NAME: validated.geometry_records_json,
        PHASE7_UNCERTAINTY_RESULT_NAME: validated.uncertainty_result_json,
        PHASE7_CALIBRATION_RESULT_NAME: validated.calibration_result_json,
        PHASE7_RISK_COVERAGE_RESULT_NAME: validated.risk_coverage_result_json,
        PHASE7_FAILURE_DETECTION_NAME: validated.failure_detection_json,
        PHASE7_DEGRADATION_RESULT_NAME: validated.degradation_result_json,
        PHASE7_LESION_SUBGROUP_RESULT_NAME: validated.lesion_subgroup_result_json,
        PHASE7_RUN_SUMMARY_NAME: validated.phase7_run_summary_json,
        PHASE7_CORRUPTION_DEGRADATION_PLOT_NAME: build_phase7_corruption_degradation_plot_png(
            transform_results_json=validated.transform_results_json,
            degradation_result_json=validated.degradation_result_json,
        ),
        PHASE7_CALIBRATION_PLOT_NAME: build_phase7_calibration_plot_png(
            validated.calibration_result_json
        ),
        PHASE7_RISK_COVERAGE_PLOT_NAME: build_phase7_risk_coverage_plot_png(
            validated.risk_coverage_result_json
        ),
        PHASE7_UNCERTAINTY_FAILURE_TABLE_NAME: uncertainty_failure_table,
        PHASE7_SUBGROUP_TABLE_NAME: subgroup_table,
        PHASE7_SUMMARY_MARKDOWN_NAME: summary_markdown,
    }
    if tuple(sorted(artifact_bytes)) != tuple(sorted(PHASE7_PUBLICATION_FILENAMES)):
        raise Phase7PublicationValidationError("Phase 7 publication filenames are incomplete.")
    return artifact_bytes


def build_phase7_transform_results_collection_json(
    transform_results: Sequence[Phase7TransformResult],
) -> bytes:
    """Serialize an ordered transform-result collection for publication."""

    if not transform_results:
        raise Phase7PublicationValidationError("transform_results must be non-empty.")
    payload: dict[str, JsonValue] = {
        "items": [phase7_transform_result_to_dict(item) for item in transform_results],
        "schema_name": PHASE7_TRANSFORM_RESULTS_COLLECTION_SCHEMA_NAME,
        "schema_version": PHASE7_PUBLICATION_COLLECTION_SCHEMA_VERSION,
    }
    return canonical_json_bytes(payload) + b"\n"


def build_phase7_geometry_records_collection_json(
    geometry_records: Sequence[Phase7GeometryRecord],
) -> bytes:
    """Serialize an ordered geometry-record collection for publication."""

    if not geometry_records:
        raise Phase7PublicationValidationError("geometry_records must be non-empty.")
    payload: dict[str, JsonValue] = {
        "items": [phase7_geometry_record_to_dict(item) for item in geometry_records],
        "schema_name": PHASE7_GEOMETRY_RECORDS_COLLECTION_SCHEMA_NAME,
        "schema_version": PHASE7_PUBLICATION_COLLECTION_SCHEMA_VERSION,
    }
    return canonical_json_bytes(payload) + b"\n"


def build_phase7_corruption_degradation_plot_png(
    *,
    transform_results_json: bytes,
    degradation_result_json: bytes,
) -> bytes:
    """Render a deterministic degradation plot strictly from publication JSON."""

    transform_results = _transform_results_from_json(transform_results_json)
    degradation = phase7_degradation_result_from_json(degradation_result_json)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - exercised only when matplotlib is unavailable.
        raise Phase7PublicationIOError("failed to import matplotlib.") from exc

    labels = [f"transform_{index}" for index, _ in enumerate(transform_results)]
    values = [
        0.0 if degradation.absolute_degradation is None else degradation.absolute_degradation
        for _ in transform_results
    ]
    figure, axis = plt.subplots(figsize=(7.0, 4.0), dpi=120)
    axis.plot(labels, values, label=degradation.metric_name, marker="o")
    axis.set_xlabel("published transform")
    axis.set_ylabel("absolute degradation")
    axis.set_title("Phase 7 corruption degradation")
    axis.grid(True, linewidth=0.5, alpha=0.35)
    axis.legend(loc="best", frameon=False)
    figure.autofmt_xdate(rotation=30, ha="right")
    figure.tight_layout()
    output = io.BytesIO()
    figure.savefig(output, format="png", dpi=120, metadata={"Software": "protoem_ct"})
    plt.close(figure)
    return output.getvalue()


def render_phase7_summary_markdown_from_json_artifacts(
    validated: _ValidatedPhase7PublicationInputs,
) -> bytes:
    """Render Phase 7 summary Markdown strictly from validated JSON artifacts."""

    manifest = phase7_corruption_manifest_from_json(validated.corruption_manifest_json)
    transform_results = _transform_results_from_json(validated.transform_results_json)
    geometry_records = _geometry_records_from_json(validated.geometry_records_json)
    uncertainty = phase7_uncertainty_result_from_json(validated.uncertainty_result_json)
    calibration = phase7_calibration_result_from_json(validated.calibration_result_json)
    risk = phase7_risk_coverage_result_from_json(validated.risk_coverage_result_json)
    degradation = phase7_degradation_result_from_json(validated.degradation_result_json)
    subgroup = phase7_lesion_subgroup_result_from_json(validated.lesion_subgroup_result_json)
    summary = phase7_run_summary_from_json(validated.phase7_run_summary_json)
    lines = [
        "# Phase 7 Robustness and Uncertainty Summary",
        "",
        f"- config_hash: `{summary.config_hash}`",
        f"- corruption_manifest_hash: `{manifest.corruption_manifest_hash}`",
        f"- transform_result_count: `{len(transform_results)}`",
        f"- geometry_record_count: `{len(geometry_records)}`",
        f"- uncertainty_type: `{uncertainty.uncertainty_type}`",
        f"- uncertainty_availability: `{uncertainty.availability_status}`",
        f"- calibration_metric: `{calibration.calibration_metric}`",
        f"- calibration_ece: `{_format_optional_float(calibration.ece)}`",
        f"- risk_coverage_points: `{len(risk.points)}`",
        f"- degradation_metric: `{degradation.metric_name}`",
        f"- absolute_degradation: `{_format_optional_float(degradation.absolute_degradation)}`",
        f"- relative_degradation: `{_format_optional_float(degradation.relative_degradation)}`",
        f"- subgroup_metric: `{subgroup.metric_name}`",
        f"- execution_status: `{summary.execution_status}`",
        "",
        "Plots and tables are derived only from validated JSON artifacts.",
        "No real-data, external-validation, or theoretical robustness claim is made.",
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


@dataclass(frozen=True, slots=True)
class _ValidatedPhase7PublicationInputs:
    effective_config_json: bytes
    corruption_manifest_json: bytes
    transform_results_json: bytes
    geometry_records_json: bytes
    uncertainty_result_json: bytes
    calibration_result_json: bytes
    risk_coverage_result_json: bytes
    failure_detection_json: bytes
    degradation_result_json: bytes
    lesion_subgroup_result_json: bytes
    phase7_run_summary_json: bytes

    @classmethod
    def from_inputs(cls, inputs: Phase7PublicationInputs) -> _ValidatedPhase7PublicationInputs:
        effective_config_json = _canonical_json_mapping_bytes(
            inputs.effective_config_json,
            field_name="effective_config_json",
        )
        corruption_manifest = phase7_corruption_manifest_from_json(inputs.corruption_manifest_json)
        transform_results = _transform_results_from_json(inputs.transform_results_json)
        geometry_records = _geometry_records_from_json(inputs.geometry_records_json)
        uncertainty = phase7_uncertainty_result_from_json(inputs.uncertainty_result_json)
        calibration = phase7_calibration_result_from_json(inputs.calibration_result_json)
        risk = phase7_risk_coverage_result_from_json(inputs.risk_coverage_result_json)
        phase7_failure_detection_results_from_json(inputs.failure_detection_json)
        degradation = phase7_degradation_result_from_json(inputs.degradation_result_json)
        subgroup = phase7_lesion_subgroup_result_from_json(inputs.lesion_subgroup_result_json)
        summary = phase7_run_summary_from_json(inputs.phase7_run_summary_json)
        transform_results_hash = _hash_json_mapping_bytes(
            build_phase7_transform_results_collection_json(transform_results)
        )
        geometry_records_hash = _hash_json_mapping_bytes(
            build_phase7_geometry_records_collection_json(geometry_records)
        )
        failure_detection_hash = _hash_json_mapping_bytes(inputs.failure_detection_json)
        publication_payload_hash = _phase7_publication_payload_hash(
            effective_config_json=effective_config_json,
            corruption_manifest_hash=corruption_manifest.corruption_manifest_hash,
            transform_results_hash=transform_results_hash,
            geometry_records_hash=geometry_records_hash,
            uncertainty_result_hash=uncertainty.uncertainty_result_hash,
            calibration_result_hash=calibration.calibration_result_hash,
            risk_coverage_result_hash=risk.risk_coverage_result_hash,
            failure_detection_result_hash=failure_detection_hash,
            degradation_result_hash=degradation.degradation_result_hash,
            lesion_subgroup_result_hash=subgroup.lesion_subgroup_result_hash,
        )
        _validate_transform_manifest_geometry_consistency(
            manifest=corruption_manifest,
            transform_results=transform_results,
            geometry_records=geometry_records,
        )
        _validate_cross_artifact_hashes(
            config_hash=_hash_json_mapping_bytes(effective_config_json),
            uncertainty_result_hash=uncertainty.uncertainty_result_hash,
            calibration_result_hash=calibration.calibration_result_hash,
            risk_coverage_result_hash=risk.risk_coverage_result_hash,
            failure_detection_result_hash=failure_detection_hash,
            degradation_result_hash=degradation.degradation_result_hash,
            lesion_subgroup_result_hash=subgroup.lesion_subgroup_result_hash,
            manifest_hash=corruption_manifest.corruption_manifest_hash,
            transform_results_hash=transform_results_hash,
            geometry_records_hash=geometry_records_hash,
            publication_payload_hash=publication_payload_hash,
            summary=summary,
        )
        return cls(
            effective_config_json=effective_config_json,
            corruption_manifest_json=phase7_corruption_manifest_to_json(corruption_manifest),
            transform_results_json=build_phase7_transform_results_collection_json(
                transform_results
            ),
            geometry_records_json=build_phase7_geometry_records_collection_json(geometry_records),
            uncertainty_result_json=phase7_uncertainty_result_to_json(uncertainty),
            calibration_result_json=phase7_calibration_result_to_json(calibration),
            risk_coverage_result_json=phase7_risk_coverage_result_to_json(risk),
            failure_detection_json=canonical_json_bytes(
                _parse_json_mapping(inputs.failure_detection_json)
            )
            + b"\n",
            degradation_result_json=phase7_degradation_result_to_json(degradation),
            lesion_subgroup_result_json=phase7_lesion_subgroup_result_to_json(subgroup),
            phase7_run_summary_json=phase7_run_summary_to_json(summary),
        )


def _validate_cross_artifact_hashes(
    *,
    config_hash: str,
    uncertainty_result_hash: str,
    calibration_result_hash: str,
    risk_coverage_result_hash: str,
    failure_detection_result_hash: str,
    degradation_result_hash: str,
    lesion_subgroup_result_hash: str,
    manifest_hash: str,
    transform_results_hash: str,
    geometry_records_hash: str,
    publication_payload_hash: str,
    summary: Phase7RunSummary,
) -> None:
    if summary.config_hash != config_hash:
        raise Phase7PublicationValidationError("run summary config hash mismatch.")
    if summary.corruption_manifest_hash != manifest_hash:
        raise Phase7PublicationValidationError("run summary manifest hash mismatch.")
    expected = {
        "transform_results_hash": transform_results_hash,
        "geometry_records_hash": geometry_records_hash,
        "uncertainty_result_hash": uncertainty_result_hash,
        "calibration_result_hash": calibration_result_hash,
        "risk_coverage_result_hash": risk_coverage_result_hash,
        "failure_detection_result_hash": failure_detection_result_hash,
        "degradation_result_hash": degradation_result_hash,
        "lesion_subgroup_result_hash": lesion_subgroup_result_hash,
        "publication_payload_hash": publication_payload_hash,
    }
    for field_name, value in expected.items():
        if getattr(summary, field_name) != value:
            raise Phase7PublicationValidationError(f"run summary {field_name} mismatch.")


def _validate_transform_manifest_geometry_consistency(
    *,
    manifest: Phase7CorruptionManifest,
    transform_results: tuple[Phase7TransformResult, ...],
    geometry_records: tuple[Phase7GeometryRecord, ...],
) -> None:
    specification_names = tuple(item.corruption_name for item in manifest.specifications)
    if set(specification_names) != PHASE7_CORRUPTION_NAMES:
        raise Phase7PublicationValidationError("corruption manifest coverage is incomplete.")
    if len(set(specification_names)) != len(specification_names):
        raise Phase7PublicationValidationError("corruption manifest contains duplicate names.")
    if len(transform_results) != len(manifest.specifications):
        raise Phase7PublicationValidationError("transform result count must match manifest.")
    geometry_hashes = {item.geometry_record_hash for item in geometry_records}
    referenced_geometry_hashes: set[str] = set()
    for index, (specification, result) in enumerate(
        zip(manifest.specifications, transform_results, strict=True)
    ):
        if result.corruption_specification_hash != specification.corruption_specification_hash:
            raise Phase7PublicationValidationError(
                f"transform result {index} specification hash mismatch."
            )
        if specification.changes_geometry or specification.common_grid_restoration_required:
            if result.geometry_record_hash is None:
                raise Phase7PublicationValidationError(
                    f"transform result {index} requires a geometry record."
                )
            if not result.common_grid_restored:
                raise Phase7PublicationValidationError(
                    f"transform result {index} must restore the common grid."
                )
        if result.geometry_record_hash is not None:
            if result.geometry_record_hash not in geometry_hashes:
                raise Phase7PublicationValidationError(
                    f"transform result {index} geometry hash is missing."
                )
            referenced_geometry_hashes.add(result.geometry_record_hash)
    if referenced_geometry_hashes != geometry_hashes:
        raise Phase7PublicationValidationError(
            "geometry records must match referenced transform result hashes."
        )


def _phase7_publication_payload_hash(
    *,
    effective_config_json: bytes,
    corruption_manifest_hash: str,
    transform_results_hash: str,
    geometry_records_hash: str,
    uncertainty_result_hash: str,
    calibration_result_hash: str,
    risk_coverage_result_hash: str,
    failure_detection_result_hash: str,
    degradation_result_hash: str,
    lesion_subgroup_result_hash: str,
) -> str:
    return sha256_json(
        {
            "config_hash": _hash_json_mapping_bytes(effective_config_json),
            "corruption_manifest_hash": corruption_manifest_hash,
            "degradation_result_hash": degradation_result_hash,
            "failure_detection_result_hash": failure_detection_result_hash,
            "geometry_records_hash": geometry_records_hash,
            "lesion_subgroup_result_hash": lesion_subgroup_result_hash,
            "risk_coverage_result_hash": risk_coverage_result_hash,
            "calibration_result_hash": calibration_result_hash,
            "schema_name": "phase7_publication_payload",
            "schema_version": PHASE7_PUBLICATION_COLLECTION_SCHEMA_VERSION,
            "transform_results_hash": transform_results_hash,
            "uncertainty_result_hash": uncertainty_result_hash,
        }
    )


def _hash_json_mapping_bytes(data: bytes | str) -> str:
    return sha256_json(_parse_json_mapping(data))


def _transform_results_from_json(data: bytes | str) -> tuple[Phase7TransformResult, ...]:
    mapping = _parse_json_mapping(data)
    _require_collection_schema(
        mapping,
        expected_schema_name=PHASE7_TRANSFORM_RESULTS_COLLECTION_SCHEMA_NAME,
    )
    items = _expect_sequence(mapping["items"], field_name="items")
    if not items:
        raise Phase7PublicationValidationError("transform result collection must be non-empty.")
    return tuple(
        phase7_transform_result_from_mapping(_expect_mapping(item, field_name=f"items[{index}]"))
        for index, item in enumerate(items)
    )


def _geometry_records_from_json(data: bytes | str) -> tuple[Phase7GeometryRecord, ...]:
    mapping = _parse_json_mapping(data)
    _require_collection_schema(
        mapping,
        expected_schema_name=PHASE7_GEOMETRY_RECORDS_COLLECTION_SCHEMA_NAME,
    )
    items = _expect_sequence(mapping["items"], field_name="items")
    if not items:
        raise Phase7PublicationValidationError("geometry record collection must be non-empty.")
    return tuple(
        phase7_geometry_record_from_mapping(_expect_mapping(item, field_name=f"items[{index}]"))
        for index, item in enumerate(items)
    )


def _require_collection_schema(
    mapping: Mapping[str, object],
    *,
    expected_schema_name: str,
) -> None:
    _require_exact_fields(
        mapping,
        required_fields=_COLLECTION_FIELDS,
        object_name=expected_schema_name,
    )
    if mapping["schema_name"] != expected_schema_name:
        raise Phase7PublicationValidationError(f"{expected_schema_name} schema_name mismatch.")
    if mapping["schema_version"] != PHASE7_PUBLICATION_COLLECTION_SCHEMA_VERSION:
        raise Phase7PublicationValidationError(f"{expected_schema_name} schema_version mismatch.")


def _canonical_json_mapping_bytes(data: bytes | str, *, field_name: str) -> bytes:
    return canonical_json_bytes(_parse_json_mapping(data, field_name=field_name)) + b"\n"


def _parse_json_mapping(data: bytes | str, *, field_name: str = "json") -> Mapping[str, object]:
    try:
        loaded = json.loads(data)
    except json.JSONDecodeError as exc:
        raise Phase7PublicationValidationError(f"{field_name} must be valid JSON.") from exc
    return _expect_mapping(loaded, field_name=field_name)


def _expect_mapping(value: object, *, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise Phase7PublicationValidationError(f"{field_name} must be a JSON object.")
    return cast(Mapping[str, object], value)


def _expect_sequence(value: object, *, field_name: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise Phase7PublicationValidationError(f"{field_name} must be a JSON array.")
    return cast(Sequence[object], value)


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
        raise Phase7PublicationValidationError(
            f"{object_name} fields mismatch; missing={missing!r}, extra={extra!r}."
        )


def _validate_phase7_output_root(output_root: Path) -> Path:
    repository_root = Path(__file__).resolve().parents[3]
    try:
        resolved_output_root = validate_explicit_external_output_root(
            output_root,
            forbidden_roots=(repository_root,),
        )
    except InvalidDatasetRootError as exc:
        raise Phase7PublicationPathError(str(exc)) from exc
    _require_no_parent_traversal_components(output_root)
    _require_no_symlink_components(output_root, resolved_output_root=resolved_output_root)
    return resolved_output_root


def _publish_directory_tree_if_absent_or_equal(
    *,
    output_root: Path,
    artifact_bytes: Mapping[str, bytes],
) -> bool:
    expected_directory_paths = _expected_directory_paths(artifact_bytes)
    if output_root.exists():
        if not output_root.is_dir():
            raise Phase7PublicationPathError("output_root must be a directory when it exists.")
        _require_no_symlinks_in_tree(output_root)
        if any(output_root.iterdir()):
            if _existing_tree_matches(
                output_root=output_root,
                artifact_bytes=artifact_bytes,
                expected_directory_paths=expected_directory_paths,
            ):
                return True
            raise Phase7PublicationCollisionError(
                "output_root already contains non-identical Phase 7 artifacts."
            )

    staging_root = output_root.parent / f".{output_root.name}.phase7-publication.tmp"
    if staging_root.exists():
        raise Phase7PublicationCollisionError("staging output root already exists.")
    try:
        staging_root.mkdir(mode=0o700)
        for relative_directory in expected_directory_paths:
            (staging_root / relative_directory).mkdir(parents=True, exist_ok=True)
        for relative_path, payload in dict(sorted(artifact_bytes.items())).items():
            _write_bytes_in_staging_root(
                staging_root=staging_root,
                relative_path=relative_path,
                payload=payload,
            )
        if output_root.exists():
            if any(output_root.iterdir()):
                raise Phase7PublicationCollisionError(
                    "output_root became non-empty before staged publication."
                )
            output_root.rmdir()
        os.replace(staging_root, output_root)
    except Phase7PublicationError:
        _remove_tree_if_present(staging_root)
        raise
    except OSError as exc:
        _remove_tree_if_present(staging_root)
        raise Phase7PublicationIOError("failed to publish Phase 7 artifacts.") from exc
    return False


def _write_bytes_in_staging_root(
    *,
    staging_root: Path,
    relative_path: str,
    payload: bytes,
) -> None:
    normalized_relative_path = _normalize_relative_publication_path(relative_path)
    output_path = staging_root / normalized_relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(payload)


def _normalize_relative_publication_path(relative_path: str) -> str:
    from protoem_ct.data.phase2_paths import normalize_safe_relative_posix_path

    try:
        return normalize_safe_relative_posix_path(relative_path)
    except Exception as exc:
        raise Phase7PublicationPathError("relative publication path is unsafe.") from exc


def _expected_directory_paths(artifact_bytes: Mapping[str, bytes]) -> set[str]:
    directories: set[str] = set()
    for relative_path in artifact_bytes:
        parts = relative_path.split("/")[:-1]
        for index in range(1, len(parts) + 1):
            directories.add("/".join(parts[:index]))
    return directories


def _existing_tree_matches(
    *,
    output_root: Path,
    artifact_bytes: Mapping[str, bytes],
    expected_directory_paths: set[str],
) -> bool:
    existing_directory_paths: set[str] = set()
    existing_file_paths: dict[str, bytes] = {}
    for path in sorted(output_root.rglob("*")):
        relative_path = path.relative_to(output_root).as_posix()
        if path.is_symlink():
            raise Phase7PublicationPathError("output_root must not contain symlinked paths.")
        if path.is_dir():
            existing_directory_paths.add(relative_path)
        elif path.is_file():
            existing_file_paths[relative_path] = path.read_bytes()
        else:
            raise Phase7PublicationPathError("output_root must contain only directories or files.")
    if existing_directory_paths != expected_directory_paths:
        return False
    return existing_file_paths == artifact_bytes


def _require_no_parent_traversal_components(path: Path) -> None:
    if ".." in path.parts:
        raise Phase7PublicationPathError("output_root must not contain parent traversal.")


def _require_no_symlink_components(path: Path, *, resolved_output_root: Path) -> None:
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current = current / component
        if current.exists() and current.is_symlink():
            if _is_allowed_output_alias(current, resolved_output_root=resolved_output_root):
                continue
            raise Phase7PublicationPathError("output_root must not traverse symlinked paths.")
        if current.exists() and not current.is_dir() and current != path:
            raise Phase7PublicationPathError("output_root parent chain must contain directories.")


def _is_allowed_output_alias(symlink_path: Path, *, resolved_output_root: Path) -> bool:
    if symlink_path != Path("/tmp"):
        return False
    try:
        resolved_symlink = symlink_path.resolve(strict=True)
    except OSError:
        return False
    return _path_is_equal_or_nested(resolved_output_root, resolved_symlink)


def _path_is_equal_or_nested(path: Path, root: Path) -> bool:
    if path == root:
        return True
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _require_no_symlinks_in_tree(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_symlink():
            raise Phase7PublicationPathError("output_root must not contain symlinked paths.")


def _remove_tree_if_present(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _format_optional_float(value: float | None) -> str:
    return "null" if value is None else f"{value:.12g}"


__all__ = [
    "PHASE7_CALIBRATION_PLOT_NAME",
    "PHASE7_CALIBRATION_RESULT_NAME",
    "PHASE7_CORRUPTION_DEGRADATION_PLOT_NAME",
    "PHASE7_CORRUPTION_MANIFEST_NAME",
    "PHASE7_DEGRADATION_RESULT_NAME",
    "PHASE7_EFFECTIVE_CONFIG_NAME",
    "PHASE7_FAILURE_DETECTION_NAME",
    "PHASE7_GEOMETRY_RECORDS_COLLECTION_SCHEMA_NAME",
    "PHASE7_GEOMETRY_RECORDS_NAME",
    "PHASE7_LESION_SUBGROUP_RESULT_NAME",
    "PHASE7_PUBLICATION_COLLECTION_SCHEMA_VERSION",
    "PHASE7_PUBLICATION_FILENAMES",
    "PHASE7_RISK_COVERAGE_PLOT_NAME",
    "PHASE7_RISK_COVERAGE_RESULT_NAME",
    "PHASE7_RUN_SUMMARY_NAME",
    "PHASE7_SUBGROUP_TABLE_NAME",
    "PHASE7_SUMMARY_MARKDOWN_NAME",
    "PHASE7_TRANSFORM_RESULTS_COLLECTION_SCHEMA_NAME",
    "PHASE7_TRANSFORM_RESULTS_NAME",
    "PHASE7_UNCERTAINTY_FAILURE_TABLE_NAME",
    "PHASE7_UNCERTAINTY_RESULT_NAME",
    "Phase7PublicationCollisionError",
    "Phase7PublicationError",
    "Phase7PublicationIOError",
    "Phase7PublicationInputs",
    "Phase7PublicationPathError",
    "Phase7PublicationResult",
    "Phase7PublicationValidationError",
    "build_phase7_corruption_degradation_plot_png",
    "build_phase7_geometry_records_collection_json",
    "build_phase7_publication_artifact_bytes",
    "build_phase7_transform_results_collection_json",
    "publish_phase7_artifacts",
]
