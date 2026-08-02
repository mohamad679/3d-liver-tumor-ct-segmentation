"""Deterministic Phase 7 robustness artifact schemas and hashing contracts."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, TypeAlias, cast

from protoem_ct.artifacts.hashing import JsonValue, canonical_json_bytes, sha256_json

PHASE7_CORRUPTION_SPECIFICATION_SCHEMA_NAME: Final[str] = "phase7_corruption_specification"
PHASE7_CORRUPTION_SPECIFICATION_SCHEMA_VERSION: Final[str] = "v1"
PHASE7_CORRUPTION_MANIFEST_SCHEMA_NAME: Final[str] = "phase7_corruption_manifest"
PHASE7_CORRUPTION_MANIFEST_SCHEMA_VERSION: Final[str] = "v1"
PHASE7_TRANSFORM_RESULT_SCHEMA_NAME: Final[str] = "phase7_transform_result"
PHASE7_TRANSFORM_RESULT_SCHEMA_VERSION: Final[str] = "v1"
PHASE7_GEOMETRY_RECORD_SCHEMA_NAME: Final[str] = "phase7_geometry_record"
PHASE7_GEOMETRY_RECORD_SCHEMA_VERSION: Final[str] = "v1"
PHASE7_RUN_SUMMARY_SCHEMA_NAME: Final[str] = "phase7_run_summary"
PHASE7_RUN_SUMMARY_SCHEMA_VERSION: Final[str] = "v1"

PHASE7_SEVERITY_LEVELS: Final[frozenset[str]] = frozenset({"none", "low", "medium", "high"})
PHASE7_CORRUPTION_NAMES: Final[frozenset[str]] = frozenset(
    {
        "anisotropic_downsampling",
        "contrast_shift",
        "crop_fov",
        "gaussian_blur",
        "gaussian_noise",
        "hu_window_shift",
        "intensity_offset",
        "intensity_scale",
        "slice_thickness",
    }
)
PHASE7_GEOMETRY_CHANGE_TYPES: Final[frozenset[str]] = frozenset(
    {"none", "affine", "spacing", "orientation", "shape", "crop_fov", "resampling"}
)
PHASE7_EXECUTION_STATUSES: Final[frozenset[str]] = frozenset({"completed", "failed", "unsupported"})
PHASE7_MEMORY_AVAILABILITY_STATUSES: Final[frozenset[str]] = frozenset({"available", "unavailable"})

MappingLike: TypeAlias = Mapping[str, object]
Shape3D: TypeAlias = tuple[int, int, int]
Spacing3D: TypeAlias = tuple[float, float, float]

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[a-z0-9](?:[a-z0-9_.-]{0,126}[a-z0-9])?$")
_ORIENTATION_RE = re.compile(r"^[A-Z]{3}$")
_MESSAGE_RE = re.compile(r"^[^\r\n]+$")


class Phase7ArtifactError(ValueError):
    """Base error for Phase 7 artifact contracts."""


class Phase7ArtifactValidationError(Phase7ArtifactError):
    """Raised when a Phase 7 artifact violates its schema contract."""


class Phase7ArtifactSerializationError(Phase7ArtifactError):
    """Raised when a Phase 7 artifact mapping cannot be reconstructed safely."""


class Phase7ArtifactHashError(Phase7ArtifactError):
    """Raised when an embedded Phase 7 artifact hash does not match its content."""


@dataclass(frozen=True, slots=True)
class Phase7CorruptionSpecification:
    """Immutable identity for one deterministic robustness corruption specification."""

    schema_name: str
    schema_version: str
    corruption_specification_hash: str
    corruption_name: str
    severity: str
    parameters: Mapping[str, JsonValue]
    deterministic_seed: int | None
    changes_geometry: bool
    image_interpolation: str
    mask_interpolation: str
    common_grid_restoration_required: bool

    def __post_init__(self) -> None:
        _require_schema(self.schema_name, PHASE7_CORRUPTION_SPECIFICATION_SCHEMA_NAME)
        _require_version(self.schema_version, PHASE7_CORRUPTION_SPECIFICATION_SCHEMA_VERSION)
        _require_sha256(
            self.corruption_specification_hash,
            field_name="corruption_specification_hash",
        )
        _require_allowed(
            self.corruption_name, PHASE7_CORRUPTION_NAMES, field_name="corruption_name"
        )
        _require_allowed(self.severity, PHASE7_SEVERITY_LEVELS, field_name="severity")
        _require_json_mapping(self.parameters, field_name="parameters")
        _require_optional_seed(self.deterministic_seed, field_name="deterministic_seed")
        _require_identifier(self.image_interpolation, field_name="image_interpolation")
        if self.mask_interpolation != "nearest_neighbor":
            raise Phase7ArtifactValidationError(
                "mask_interpolation must be nearest_neighbor for Phase 7 mask safety."
            )
        if self.changes_geometry and not self.common_grid_restoration_required:
            raise Phase7ArtifactValidationError(
                "geometry-changing corruptions must require common-grid restoration."
            )
        expected_hash = hash_phase7_corruption_specification(self)
        if self.corruption_specification_hash != expected_hash:
            raise Phase7ArtifactHashError(
                "corruption_specification_hash does not match deterministic content."
            )


@dataclass(frozen=True, slots=True)
class Phase7CorruptionManifest:
    """Immutable ordered manifest of deterministic Phase 7 corruption specifications."""

    schema_name: str
    schema_version: str
    corruption_manifest_hash: str
    manifest_id: str
    config_hash: str
    input_image_content_hash: str
    input_mask_content_hash: str | None
    deterministic_seed: int
    specifications: tuple[Phase7CorruptionSpecification, ...]

    def __post_init__(self) -> None:
        _require_schema(self.schema_name, PHASE7_CORRUPTION_MANIFEST_SCHEMA_NAME)
        _require_version(self.schema_version, PHASE7_CORRUPTION_MANIFEST_SCHEMA_VERSION)
        _require_sha256(self.corruption_manifest_hash, field_name="corruption_manifest_hash")
        _require_identifier(self.manifest_id, field_name="manifest_id")
        _require_sha256(self.config_hash, field_name="config_hash")
        _require_sha256(self.input_image_content_hash, field_name="input_image_content_hash")
        _require_optional_sha256(self.input_mask_content_hash, field_name="input_mask_content_hash")
        _require_seed(self.deterministic_seed, field_name="deterministic_seed")
        if not self.specifications:
            raise Phase7ArtifactValidationError("specifications must be non-empty.")
        hashes = [item.corruption_specification_hash for item in self.specifications]
        if len(set(hashes)) != len(hashes):
            raise Phase7ArtifactValidationError("specifications must not contain duplicate hashes.")
        expected_hash = hash_phase7_corruption_manifest(self)
        if self.corruption_manifest_hash != expected_hash:
            raise Phase7ArtifactHashError(
                "corruption_manifest_hash does not match deterministic content."
            )


@dataclass(frozen=True, slots=True)
class Phase7GeometryRecord:
    """Immutable record of one explicit Phase 7 geometry state or geometry change."""

    schema_name: str
    schema_version: str
    geometry_record_hash: str
    transform_name: str
    geometry_change_type: str
    input_shape: Shape3D
    output_shape: Shape3D
    restored_shape: Shape3D | None
    input_spacing: Spacing3D
    output_spacing: Spacing3D
    restored_spacing: Spacing3D | None
    input_orientation: str
    output_orientation: str
    restored_orientation: str | None
    input_affine_hash: str
    output_affine_hash: str
    restored_affine_hash: str | None
    image_interpolation: str
    mask_interpolation: str
    binary_mask_preserved: bool
    common_grid_restored: bool

    def __post_init__(self) -> None:
        _require_schema(self.schema_name, PHASE7_GEOMETRY_RECORD_SCHEMA_NAME)
        _require_version(self.schema_version, PHASE7_GEOMETRY_RECORD_SCHEMA_VERSION)
        _require_sha256(self.geometry_record_hash, field_name="geometry_record_hash")
        _require_allowed(self.transform_name, PHASE7_CORRUPTION_NAMES, field_name="transform_name")
        _require_allowed(
            self.geometry_change_type,
            PHASE7_GEOMETRY_CHANGE_TYPES,
            field_name="geometry_change_type",
        )
        _require_shape(self.input_shape, field_name="input_shape")
        _require_shape(self.output_shape, field_name="output_shape")
        _require_optional_shape(self.restored_shape, field_name="restored_shape")
        _require_spacing(self.input_spacing, field_name="input_spacing")
        _require_spacing(self.output_spacing, field_name="output_spacing")
        _require_optional_spacing(self.restored_spacing, field_name="restored_spacing")
        _require_orientation(self.input_orientation, field_name="input_orientation")
        _require_orientation(self.output_orientation, field_name="output_orientation")
        _require_optional_orientation(self.restored_orientation, field_name="restored_orientation")
        _require_sha256(self.input_affine_hash, field_name="input_affine_hash")
        _require_sha256(self.output_affine_hash, field_name="output_affine_hash")
        _require_optional_sha256(self.restored_affine_hash, field_name="restored_affine_hash")
        _require_identifier(self.image_interpolation, field_name="image_interpolation")
        if self.mask_interpolation != "nearest_neighbor":
            raise Phase7ArtifactValidationError(
                "mask_interpolation must be nearest_neighbor for Phase 7 mask safety."
            )
        if not self.binary_mask_preserved:
            raise Phase7ArtifactValidationError("binary_mask_preserved must be true.")
        if self.geometry_change_type != "none" and not self.common_grid_restored:
            raise Phase7ArtifactValidationError(
                "geometry-changing records must have common_grid_restored=true."
            )
        if self.common_grid_restored and (
            self.restored_shape is None
            or self.restored_spacing is None
            or self.restored_orientation is None
            or self.restored_affine_hash is None
        ):
            raise Phase7ArtifactValidationError(
                "restored geometry fields are required when common_grid_restored is true."
            )
        expected_hash = hash_phase7_geometry_record(self)
        if self.geometry_record_hash != expected_hash:
            raise Phase7ArtifactHashError(
                "geometry_record_hash does not match deterministic content."
            )


@dataclass(frozen=True, slots=True)
class Phase7TransformResult:
    """Immutable result contract for one Phase 7 transform execution output."""

    schema_name: str
    schema_version: str
    transform_result_hash: str
    corruption_specification_hash: str
    input_image_content_hash: str
    input_mask_content_hash: str | None
    output_image_content_hash: str | None
    output_mask_content_hash: str | None
    geometry_record_hash: str | None
    finite_output: bool
    mask_binary_preserved: bool | None
    common_grid_restored: bool
    execution_status: str
    failure_code: str | None
    failure_message: str | None

    def __post_init__(self) -> None:
        _require_schema(self.schema_name, PHASE7_TRANSFORM_RESULT_SCHEMA_NAME)
        _require_version(self.schema_version, PHASE7_TRANSFORM_RESULT_SCHEMA_VERSION)
        _require_sha256(self.transform_result_hash, field_name="transform_result_hash")
        _require_sha256(
            self.corruption_specification_hash,
            field_name="corruption_specification_hash",
        )
        _require_sha256(self.input_image_content_hash, field_name="input_image_content_hash")
        _require_optional_sha256(self.input_mask_content_hash, field_name="input_mask_content_hash")
        _require_optional_sha256(
            self.output_image_content_hash,
            field_name="output_image_content_hash",
        )
        _require_optional_sha256(
            self.output_mask_content_hash, field_name="output_mask_content_hash"
        )
        _require_optional_sha256(self.geometry_record_hash, field_name="geometry_record_hash")
        _require_allowed(
            self.execution_status, PHASE7_EXECUTION_STATUSES, field_name="execution_status"
        )
        _require_optional_identifier(self.failure_code, field_name="failure_code")
        _require_optional_message(self.failure_message, field_name="failure_message")
        if self.execution_status == "completed":
            if not self.finite_output:
                raise Phase7ArtifactValidationError("completed transform results must be finite.")
            if self.output_image_content_hash is None:
                raise Phase7ArtifactValidationError(
                    "completed transform results require output_image_content_hash."
                )
            if self.input_mask_content_hash is not None and self.mask_binary_preserved is not True:
                raise Phase7ArtifactValidationError(
                    "completed mask transforms must preserve binary masks."
                )
            if self.failure_code is not None or self.failure_message is not None:
                raise Phase7ArtifactValidationError(
                    "completed transform results must not contain failure details."
                )
        else:
            if self.failure_code is None or self.failure_message is None:
                raise Phase7ArtifactValidationError(
                    "non-completed transform results require failure_code and failure_message."
                )
        expected_hash = hash_phase7_transform_result(self)
        if self.transform_result_hash != expected_hash:
            raise Phase7ArtifactHashError(
                "transform_result_hash does not match deterministic content."
            )


@dataclass(frozen=True, slots=True)
class Phase7RunSummary:
    """Immutable Phase 7 run summary without runtime or hardware in scientific identity."""

    schema_name: str
    schema_version: str
    phase7_run_summary_hash: str
    config_hash: str
    corruption_manifest_hash: str
    phase6_run_summary_hash: str
    uncertainty_result_hash: str | None
    calibration_result_hash: str | None
    risk_coverage_result_hash: str | None
    degradation_result_hash: str | None
    lesion_subgroup_result_hash: str | None
    execution_status: str
    failure_code: str | None
    failure_message: str | None
    duration_seconds: float | None
    memory_availability_status: str
    peak_host_memory_bytes: int | None

    def __post_init__(self) -> None:
        _require_schema(self.schema_name, PHASE7_RUN_SUMMARY_SCHEMA_NAME)
        _require_version(self.schema_version, PHASE7_RUN_SUMMARY_SCHEMA_VERSION)
        _require_sha256(self.phase7_run_summary_hash, field_name="phase7_run_summary_hash")
        _require_sha256(self.config_hash, field_name="config_hash")
        _require_sha256(self.corruption_manifest_hash, field_name="corruption_manifest_hash")
        _require_sha256(self.phase6_run_summary_hash, field_name="phase6_run_summary_hash")
        _require_optional_sha256(self.uncertainty_result_hash, field_name="uncertainty_result_hash")
        _require_optional_sha256(self.calibration_result_hash, field_name="calibration_result_hash")
        _require_optional_sha256(
            self.risk_coverage_result_hash,
            field_name="risk_coverage_result_hash",
        )
        _require_optional_sha256(self.degradation_result_hash, field_name="degradation_result_hash")
        _require_optional_sha256(
            self.lesion_subgroup_result_hash,
            field_name="lesion_subgroup_result_hash",
        )
        _require_allowed(
            self.execution_status, PHASE7_EXECUTION_STATUSES, field_name="execution_status"
        )
        _require_optional_identifier(self.failure_code, field_name="failure_code")
        _require_optional_message(self.failure_message, field_name="failure_message")
        _require_optional_nonnegative_float(self.duration_seconds, field_name="duration_seconds")
        _require_allowed(
            self.memory_availability_status,
            PHASE7_MEMORY_AVAILABILITY_STATUSES,
            field_name="memory_availability_status",
        )
        _require_optional_nonnegative_int(
            self.peak_host_memory_bytes,
            field_name="peak_host_memory_bytes",
        )
        if self.execution_status == "completed":
            if self.failure_code is not None or self.failure_message is not None:
                raise Phase7ArtifactValidationError(
                    "completed Phase 7 summaries must not contain failure details."
                )
            required_hashes = (
                self.uncertainty_result_hash,
                self.calibration_result_hash,
                self.risk_coverage_result_hash,
                self.degradation_result_hash,
                self.lesion_subgroup_result_hash,
            )
            if any(item is None for item in required_hashes):
                raise Phase7ArtifactValidationError(
                    "completed Phase 7 summaries require all result hashes."
                )
        else:
            if self.failure_code is None or self.failure_message is None:
                raise Phase7ArtifactValidationError(
                    "non-completed Phase 7 summaries require failure details."
                )
        expected_hash = hash_phase7_run_summary(self)
        if self.phase7_run_summary_hash != expected_hash:
            raise Phase7ArtifactHashError(
                "phase7_run_summary_hash does not match deterministic content."
            )


def phase7_corruption_specification_identity_payload(
    specification: Phase7CorruptionSpecification,
) -> dict[str, JsonValue]:
    """Return the canonical identity payload for one corruption specification."""

    return {
        "changes_geometry": specification.changes_geometry,
        "common_grid_restoration_required": specification.common_grid_restoration_required,
        "corruption_name": specification.corruption_name,
        "deterministic_seed": specification.deterministic_seed,
        "image_interpolation": specification.image_interpolation,
        "mask_interpolation": specification.mask_interpolation,
        "parameters": dict(specification.parameters),
        "schema_name": specification.schema_name,
        "schema_version": specification.schema_version,
        "severity": specification.severity,
    }


def phase7_corruption_specification_to_dict(
    specification: Phase7CorruptionSpecification,
) -> dict[str, JsonValue]:
    """Convert one corruption specification to a canonical mapping."""

    payload = phase7_corruption_specification_identity_payload(specification)
    payload["corruption_specification_hash"] = specification.corruption_specification_hash
    return payload


def hash_phase7_corruption_specification(specification: Phase7CorruptionSpecification) -> str:
    """Return the canonical SHA-256 hash for one corruption specification."""

    return sha256_json(phase7_corruption_specification_identity_payload(specification))


def phase7_corruption_manifest_identity_payload(
    manifest: Phase7CorruptionManifest,
) -> dict[str, JsonValue]:
    """Return the canonical identity payload for one corruption manifest."""

    return {
        "config_hash": manifest.config_hash,
        "deterministic_seed": manifest.deterministic_seed,
        "input_image_content_hash": manifest.input_image_content_hash,
        "input_mask_content_hash": manifest.input_mask_content_hash,
        "manifest_id": manifest.manifest_id,
        "schema_name": manifest.schema_name,
        "schema_version": manifest.schema_version,
        "specifications": [
            phase7_corruption_specification_to_dict(item) for item in manifest.specifications
        ],
    }


def phase7_corruption_manifest_to_dict(
    manifest: Phase7CorruptionManifest,
) -> dict[str, JsonValue]:
    """Convert one corruption manifest to a canonical mapping."""

    payload = phase7_corruption_manifest_identity_payload(manifest)
    payload["corruption_manifest_hash"] = manifest.corruption_manifest_hash
    return payload


def hash_phase7_corruption_manifest(manifest: Phase7CorruptionManifest) -> str:
    """Return the canonical SHA-256 hash for one corruption manifest."""

    return sha256_json(phase7_corruption_manifest_identity_payload(manifest))


def phase7_geometry_record_identity_payload(record: Phase7GeometryRecord) -> dict[str, JsonValue]:
    """Return the canonical identity payload for one geometry record."""

    return {
        "binary_mask_preserved": record.binary_mask_preserved,
        "common_grid_restored": record.common_grid_restored,
        "geometry_change_type": record.geometry_change_type,
        "image_interpolation": record.image_interpolation,
        "input_affine_hash": record.input_affine_hash,
        "input_orientation": record.input_orientation,
        "input_shape": list(record.input_shape),
        "input_spacing": list(record.input_spacing),
        "mask_interpolation": record.mask_interpolation,
        "output_affine_hash": record.output_affine_hash,
        "output_orientation": record.output_orientation,
        "output_shape": list(record.output_shape),
        "output_spacing": list(record.output_spacing),
        "restored_affine_hash": record.restored_affine_hash,
        "restored_orientation": record.restored_orientation,
        "restored_shape": None if record.restored_shape is None else list(record.restored_shape),
        "restored_spacing": None
        if record.restored_spacing is None
        else list(record.restored_spacing),
        "schema_name": record.schema_name,
        "schema_version": record.schema_version,
        "transform_name": record.transform_name,
    }


def phase7_geometry_record_to_dict(record: Phase7GeometryRecord) -> dict[str, JsonValue]:
    """Convert one geometry record to a canonical mapping."""

    payload = phase7_geometry_record_identity_payload(record)
    payload["geometry_record_hash"] = record.geometry_record_hash
    return payload


def hash_phase7_geometry_record(record: Phase7GeometryRecord) -> str:
    """Return the canonical SHA-256 hash for one geometry record."""

    return sha256_json(phase7_geometry_record_identity_payload(record))


def phase7_transform_result_identity_payload(
    result: Phase7TransformResult,
) -> dict[str, JsonValue]:
    """Return the canonical identity payload for one transform result."""

    return {
        "common_grid_restored": result.common_grid_restored,
        "corruption_specification_hash": result.corruption_specification_hash,
        "execution_status": result.execution_status,
        "failure_code": result.failure_code,
        "failure_message": result.failure_message,
        "finite_output": result.finite_output,
        "geometry_record_hash": result.geometry_record_hash,
        "input_image_content_hash": result.input_image_content_hash,
        "input_mask_content_hash": result.input_mask_content_hash,
        "mask_binary_preserved": result.mask_binary_preserved,
        "output_image_content_hash": result.output_image_content_hash,
        "output_mask_content_hash": result.output_mask_content_hash,
        "schema_name": result.schema_name,
        "schema_version": result.schema_version,
    }


def phase7_transform_result_to_dict(result: Phase7TransformResult) -> dict[str, JsonValue]:
    """Convert one transform result to a canonical mapping."""

    payload = phase7_transform_result_identity_payload(result)
    payload["transform_result_hash"] = result.transform_result_hash
    return payload


def hash_phase7_transform_result(result: Phase7TransformResult) -> str:
    """Return the canonical SHA-256 hash for one transform result."""

    return sha256_json(phase7_transform_result_identity_payload(result))


def phase7_run_summary_identity_payload(summary: Phase7RunSummary) -> dict[str, JsonValue]:
    """Return the canonical scientific identity payload for one Phase 7 run summary."""

    return {
        "calibration_result_hash": summary.calibration_result_hash,
        "config_hash": summary.config_hash,
        "corruption_manifest_hash": summary.corruption_manifest_hash,
        "degradation_result_hash": summary.degradation_result_hash,
        "execution_status": summary.execution_status,
        "failure_code": summary.failure_code,
        "failure_message": summary.failure_message,
        "lesion_subgroup_result_hash": summary.lesion_subgroup_result_hash,
        "phase6_run_summary_hash": summary.phase6_run_summary_hash,
        "risk_coverage_result_hash": summary.risk_coverage_result_hash,
        "schema_name": summary.schema_name,
        "schema_version": summary.schema_version,
        "uncertainty_result_hash": summary.uncertainty_result_hash,
    }


def phase7_run_summary_to_dict(summary: Phase7RunSummary) -> dict[str, JsonValue]:
    """Convert one Phase 7 run summary to a canonical mapping."""

    payload = phase7_run_summary_identity_payload(summary)
    payload["phase7_run_summary_hash"] = summary.phase7_run_summary_hash
    payload["duration_seconds"] = summary.duration_seconds
    payload["memory_availability_status"] = summary.memory_availability_status
    payload["peak_host_memory_bytes"] = summary.peak_host_memory_bytes
    return payload


def hash_phase7_run_summary(summary: Phase7RunSummary) -> str:
    """Return the canonical SHA-256 hash for one Phase 7 run summary."""

    return sha256_json(phase7_run_summary_identity_payload(summary))


def phase7_corruption_specification_from_mapping(
    mapping: MappingLike,
) -> Phase7CorruptionSpecification:
    """Reconstruct one corruption specification from a strict mapping."""

    _require_exact_fields(
        mapping,
        required_fields=_CORRUPTION_SPECIFICATION_FIELDS,
        object_name="Phase7CorruptionSpecification",
    )
    return Phase7CorruptionSpecification(
        schema_name=_expect_string(mapping["schema_name"], field_name="schema_name"),
        schema_version=_expect_string(mapping["schema_version"], field_name="schema_version"),
        corruption_specification_hash=_expect_string(
            mapping["corruption_specification_hash"],
            field_name="corruption_specification_hash",
        ),
        corruption_name=_expect_string(mapping["corruption_name"], field_name="corruption_name"),
        severity=_expect_string(mapping["severity"], field_name="severity"),
        parameters=_expect_json_mapping(mapping["parameters"], field_name="parameters"),
        deterministic_seed=_expect_optional_int(
            mapping["deterministic_seed"],
            field_name="deterministic_seed",
        ),
        changes_geometry=_expect_bool(mapping["changes_geometry"], field_name="changes_geometry"),
        image_interpolation=_expect_string(
            mapping["image_interpolation"],
            field_name="image_interpolation",
        ),
        mask_interpolation=_expect_string(
            mapping["mask_interpolation"],
            field_name="mask_interpolation",
        ),
        common_grid_restoration_required=_expect_bool(
            mapping["common_grid_restoration_required"],
            field_name="common_grid_restoration_required",
        ),
    )


def phase7_corruption_manifest_from_mapping(mapping: MappingLike) -> Phase7CorruptionManifest:
    """Reconstruct one corruption manifest from a strict mapping."""

    _require_exact_fields(
        mapping,
        required_fields=_CORRUPTION_MANIFEST_FIELDS,
        object_name="Phase7CorruptionManifest",
    )
    specs_value = mapping["specifications"]
    if not isinstance(specs_value, list):
        raise Phase7ArtifactSerializationError("specifications must be a list.")
    return Phase7CorruptionManifest(
        schema_name=_expect_string(mapping["schema_name"], field_name="schema_name"),
        schema_version=_expect_string(mapping["schema_version"], field_name="schema_version"),
        corruption_manifest_hash=_expect_string(
            mapping["corruption_manifest_hash"],
            field_name="corruption_manifest_hash",
        ),
        manifest_id=_expect_string(mapping["manifest_id"], field_name="manifest_id"),
        config_hash=_expect_string(mapping["config_hash"], field_name="config_hash"),
        input_image_content_hash=_expect_string(
            mapping["input_image_content_hash"],
            field_name="input_image_content_hash",
        ),
        input_mask_content_hash=_expect_optional_string(
            mapping["input_mask_content_hash"],
            field_name="input_mask_content_hash",
        ),
        deterministic_seed=_expect_int(
            mapping["deterministic_seed"], field_name="deterministic_seed"
        ),
        specifications=tuple(
            phase7_corruption_specification_from_mapping(
                _expect_mapping(item, field_name=f"specifications[{index}]")
            )
            for index, item in enumerate(specs_value)
        ),
    )


def phase7_geometry_record_from_mapping(mapping: MappingLike) -> Phase7GeometryRecord:
    """Reconstruct one geometry record from a strict mapping."""

    _require_exact_fields(
        mapping,
        required_fields=_GEOMETRY_RECORD_FIELDS,
        object_name="Phase7GeometryRecord",
    )
    return Phase7GeometryRecord(
        schema_name=_expect_string(mapping["schema_name"], field_name="schema_name"),
        schema_version=_expect_string(mapping["schema_version"], field_name="schema_version"),
        geometry_record_hash=_expect_string(
            mapping["geometry_record_hash"],
            field_name="geometry_record_hash",
        ),
        transform_name=_expect_string(mapping["transform_name"], field_name="transform_name"),
        geometry_change_type=_expect_string(
            mapping["geometry_change_type"],
            field_name="geometry_change_type",
        ),
        input_shape=_expect_shape(mapping["input_shape"], field_name="input_shape"),
        output_shape=_expect_shape(mapping["output_shape"], field_name="output_shape"),
        restored_shape=_expect_optional_shape(
            mapping["restored_shape"],
            field_name="restored_shape",
        ),
        input_spacing=_expect_spacing(mapping["input_spacing"], field_name="input_spacing"),
        output_spacing=_expect_spacing(mapping["output_spacing"], field_name="output_spacing"),
        restored_spacing=_expect_optional_spacing(
            mapping["restored_spacing"],
            field_name="restored_spacing",
        ),
        input_orientation=_expect_string(
            mapping["input_orientation"], field_name="input_orientation"
        ),
        output_orientation=_expect_string(
            mapping["output_orientation"],
            field_name="output_orientation",
        ),
        restored_orientation=_expect_optional_string(
            mapping["restored_orientation"],
            field_name="restored_orientation",
        ),
        input_affine_hash=_expect_string(
            mapping["input_affine_hash"], field_name="input_affine_hash"
        ),
        output_affine_hash=_expect_string(
            mapping["output_affine_hash"],
            field_name="output_affine_hash",
        ),
        restored_affine_hash=_expect_optional_string(
            mapping["restored_affine_hash"],
            field_name="restored_affine_hash",
        ),
        image_interpolation=_expect_string(
            mapping["image_interpolation"],
            field_name="image_interpolation",
        ),
        mask_interpolation=_expect_string(
            mapping["mask_interpolation"],
            field_name="mask_interpolation",
        ),
        binary_mask_preserved=_expect_bool(
            mapping["binary_mask_preserved"],
            field_name="binary_mask_preserved",
        ),
        common_grid_restored=_expect_bool(
            mapping["common_grid_restored"],
            field_name="common_grid_restored",
        ),
    )


def phase7_transform_result_from_mapping(mapping: MappingLike) -> Phase7TransformResult:
    """Reconstruct one transform result from a strict mapping."""

    _require_exact_fields(
        mapping,
        required_fields=_TRANSFORM_RESULT_FIELDS,
        object_name="Phase7TransformResult",
    )
    return Phase7TransformResult(
        schema_name=_expect_string(mapping["schema_name"], field_name="schema_name"),
        schema_version=_expect_string(mapping["schema_version"], field_name="schema_version"),
        transform_result_hash=_expect_string(
            mapping["transform_result_hash"],
            field_name="transform_result_hash",
        ),
        corruption_specification_hash=_expect_string(
            mapping["corruption_specification_hash"],
            field_name="corruption_specification_hash",
        ),
        input_image_content_hash=_expect_string(
            mapping["input_image_content_hash"],
            field_name="input_image_content_hash",
        ),
        input_mask_content_hash=_expect_optional_string(
            mapping["input_mask_content_hash"],
            field_name="input_mask_content_hash",
        ),
        output_image_content_hash=_expect_optional_string(
            mapping["output_image_content_hash"],
            field_name="output_image_content_hash",
        ),
        output_mask_content_hash=_expect_optional_string(
            mapping["output_mask_content_hash"],
            field_name="output_mask_content_hash",
        ),
        geometry_record_hash=_expect_optional_string(
            mapping["geometry_record_hash"],
            field_name="geometry_record_hash",
        ),
        finite_output=_expect_bool(mapping["finite_output"], field_name="finite_output"),
        mask_binary_preserved=_expect_optional_bool(
            mapping["mask_binary_preserved"],
            field_name="mask_binary_preserved",
        ),
        common_grid_restored=_expect_bool(
            mapping["common_grid_restored"],
            field_name="common_grid_restored",
        ),
        execution_status=_expect_string(mapping["execution_status"], field_name="execution_status"),
        failure_code=_expect_optional_string(mapping["failure_code"], field_name="failure_code"),
        failure_message=_expect_optional_string(
            mapping["failure_message"],
            field_name="failure_message",
        ),
    )


def phase7_run_summary_from_mapping(mapping: MappingLike) -> Phase7RunSummary:
    """Reconstruct one Phase 7 run summary from a strict mapping."""

    _require_exact_fields(
        mapping,
        required_fields=_RUN_SUMMARY_FIELDS,
        object_name="Phase7RunSummary",
    )
    return Phase7RunSummary(
        schema_name=_expect_string(mapping["schema_name"], field_name="schema_name"),
        schema_version=_expect_string(mapping["schema_version"], field_name="schema_version"),
        phase7_run_summary_hash=_expect_string(
            mapping["phase7_run_summary_hash"],
            field_name="phase7_run_summary_hash",
        ),
        config_hash=_expect_string(mapping["config_hash"], field_name="config_hash"),
        corruption_manifest_hash=_expect_string(
            mapping["corruption_manifest_hash"],
            field_name="corruption_manifest_hash",
        ),
        phase6_run_summary_hash=_expect_string(
            mapping["phase6_run_summary_hash"],
            field_name="phase6_run_summary_hash",
        ),
        uncertainty_result_hash=_expect_optional_string(
            mapping["uncertainty_result_hash"],
            field_name="uncertainty_result_hash",
        ),
        calibration_result_hash=_expect_optional_string(
            mapping["calibration_result_hash"],
            field_name="calibration_result_hash",
        ),
        risk_coverage_result_hash=_expect_optional_string(
            mapping["risk_coverage_result_hash"],
            field_name="risk_coverage_result_hash",
        ),
        degradation_result_hash=_expect_optional_string(
            mapping["degradation_result_hash"],
            field_name="degradation_result_hash",
        ),
        lesion_subgroup_result_hash=_expect_optional_string(
            mapping["lesion_subgroup_result_hash"],
            field_name="lesion_subgroup_result_hash",
        ),
        execution_status=_expect_string(mapping["execution_status"], field_name="execution_status"),
        failure_code=_expect_optional_string(mapping["failure_code"], field_name="failure_code"),
        failure_message=_expect_optional_string(
            mapping["failure_message"],
            field_name="failure_message",
        ),
        duration_seconds=_expect_optional_float(
            mapping["duration_seconds"],
            field_name="duration_seconds",
        ),
        memory_availability_status=_expect_string(
            mapping["memory_availability_status"],
            field_name="memory_availability_status",
        ),
        peak_host_memory_bytes=_expect_optional_int(
            mapping["peak_host_memory_bytes"],
            field_name="peak_host_memory_bytes",
        ),
    )


def phase7_corruption_specification_to_json(
    specification: Phase7CorruptionSpecification,
) -> bytes:
    """Serialize one corruption specification to canonical JSON bytes."""

    return canonical_json_bytes(phase7_corruption_specification_to_dict(specification)) + b"\n"


def phase7_corruption_manifest_to_json(manifest: Phase7CorruptionManifest) -> bytes:
    """Serialize one corruption manifest to canonical JSON bytes."""

    return canonical_json_bytes(phase7_corruption_manifest_to_dict(manifest)) + b"\n"


def phase7_geometry_record_to_json(record: Phase7GeometryRecord) -> bytes:
    """Serialize one geometry record to canonical JSON bytes."""

    return canonical_json_bytes(phase7_geometry_record_to_dict(record)) + b"\n"


def phase7_transform_result_to_json(result: Phase7TransformResult) -> bytes:
    """Serialize one transform result to canonical JSON bytes."""

    return canonical_json_bytes(phase7_transform_result_to_dict(result)) + b"\n"


def phase7_run_summary_to_json(summary: Phase7RunSummary) -> bytes:
    """Serialize one Phase 7 run summary to canonical JSON bytes."""

    return canonical_json_bytes(phase7_run_summary_to_dict(summary)) + b"\n"


def phase7_corruption_specification_from_json(data: bytes | str) -> Phase7CorruptionSpecification:
    """Deserialize one corruption specification from canonical JSON-compatible bytes."""

    return phase7_corruption_specification_from_mapping(_json_to_mapping(data))


def phase7_corruption_manifest_from_json(data: bytes | str) -> Phase7CorruptionManifest:
    """Deserialize one corruption manifest from canonical JSON-compatible bytes."""

    return phase7_corruption_manifest_from_mapping(_json_to_mapping(data))


def phase7_geometry_record_from_json(data: bytes | str) -> Phase7GeometryRecord:
    """Deserialize one geometry record from canonical JSON-compatible bytes."""

    return phase7_geometry_record_from_mapping(_json_to_mapping(data))


def phase7_transform_result_from_json(data: bytes | str) -> Phase7TransformResult:
    """Deserialize one transform result from canonical JSON-compatible bytes."""

    return phase7_transform_result_from_mapping(_json_to_mapping(data))


def phase7_run_summary_from_json(data: bytes | str) -> Phase7RunSummary:
    """Deserialize one Phase 7 run summary from canonical JSON-compatible bytes."""

    return phase7_run_summary_from_mapping(_json_to_mapping(data))


_CORRUPTION_SPECIFICATION_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_name",
        "schema_version",
        "corruption_specification_hash",
        "corruption_name",
        "severity",
        "parameters",
        "deterministic_seed",
        "changes_geometry",
        "image_interpolation",
        "mask_interpolation",
        "common_grid_restoration_required",
    }
)
_CORRUPTION_MANIFEST_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_name",
        "schema_version",
        "corruption_manifest_hash",
        "manifest_id",
        "config_hash",
        "input_image_content_hash",
        "input_mask_content_hash",
        "deterministic_seed",
        "specifications",
    }
)
_GEOMETRY_RECORD_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_name",
        "schema_version",
        "geometry_record_hash",
        "transform_name",
        "geometry_change_type",
        "input_shape",
        "output_shape",
        "restored_shape",
        "input_spacing",
        "output_spacing",
        "restored_spacing",
        "input_orientation",
        "output_orientation",
        "restored_orientation",
        "input_affine_hash",
        "output_affine_hash",
        "restored_affine_hash",
        "image_interpolation",
        "mask_interpolation",
        "binary_mask_preserved",
        "common_grid_restored",
    }
)
_TRANSFORM_RESULT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_name",
        "schema_version",
        "transform_result_hash",
        "corruption_specification_hash",
        "input_image_content_hash",
        "input_mask_content_hash",
        "output_image_content_hash",
        "output_mask_content_hash",
        "geometry_record_hash",
        "finite_output",
        "mask_binary_preserved",
        "common_grid_restored",
        "execution_status",
        "failure_code",
        "failure_message",
    }
)
_RUN_SUMMARY_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_name",
        "schema_version",
        "phase7_run_summary_hash",
        "config_hash",
        "corruption_manifest_hash",
        "phase6_run_summary_hash",
        "uncertainty_result_hash",
        "calibration_result_hash",
        "risk_coverage_result_hash",
        "degradation_result_hash",
        "lesion_subgroup_result_hash",
        "execution_status",
        "failure_code",
        "failure_message",
        "duration_seconds",
        "memory_availability_status",
        "peak_host_memory_bytes",
    }
)


def _json_to_mapping(data: bytes | str) -> MappingLike:
    try:
        decoded = json.loads(data)
    except json.JSONDecodeError as exc:
        raise Phase7ArtifactSerializationError("invalid JSON artifact payload.") from exc
    if not isinstance(decoded, Mapping):
        raise Phase7ArtifactSerializationError("artifact JSON root must be an object.")
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
        raise Phase7ArtifactSerializationError(
            f"{object_name} fields mismatch; missing={missing!r}, extra={extra!r}."
        )


def _expect_mapping(value: object, *, field_name: str) -> MappingLike:
    if not isinstance(value, Mapping):
        raise Phase7ArtifactSerializationError(f"{field_name} must be a mapping.")
    return cast(MappingLike, value)


def _expect_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise Phase7ArtifactSerializationError(f"{field_name} must be a string.")
    return value


def _expect_optional_string(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _expect_string(value, field_name=field_name)


def _expect_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise Phase7ArtifactSerializationError(f"{field_name} must be an integer.")
    return value


def _expect_optional_int(value: object, *, field_name: str) -> int | None:
    if value is None:
        return None
    return _expect_int(value, field_name=field_name)


def _expect_float(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise Phase7ArtifactSerializationError(f"{field_name} must be a finite number.")
    return float(value)


def _expect_optional_float(value: object, *, field_name: str) -> float | None:
    if value is None:
        return None
    return _expect_float(value, field_name=field_name)


def _expect_bool(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise Phase7ArtifactSerializationError(f"{field_name} must be a boolean.")
    return value


def _expect_optional_bool(value: object, *, field_name: str) -> bool | None:
    if value is None:
        return None
    return _expect_bool(value, field_name=field_name)


def _expect_json_mapping(value: object, *, field_name: str) -> dict[str, JsonValue]:
    mapping = _expect_mapping(value, field_name=field_name)
    return _require_json_mapping(mapping, field_name=field_name)


def _expect_shape(value: object, *, field_name: str) -> Shape3D:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise Phase7ArtifactSerializationError(f"{field_name} must be a 3-item list.")
    values = list(value)
    if len(values) != 3:
        raise Phase7ArtifactSerializationError(f"{field_name} must contain exactly 3 items.")
    return tuple(
        _expect_int(item, field_name=f"{field_name}[{index}]") for index, item in enumerate(values)
    )  # type: ignore[return-value]


def _expect_optional_shape(value: object, *, field_name: str) -> Shape3D | None:
    if value is None:
        return None
    return _expect_shape(value, field_name=field_name)


def _expect_spacing(value: object, *, field_name: str) -> Spacing3D:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise Phase7ArtifactSerializationError(f"{field_name} must be a 3-item list.")
    values = list(value)
    if len(values) != 3:
        raise Phase7ArtifactSerializationError(f"{field_name} must contain exactly 3 items.")
    return tuple(
        _expect_float(item, field_name=f"{field_name}[{index}]")
        for index, item in enumerate(values)
    )  # type: ignore[return-value]


def _expect_optional_spacing(value: object, *, field_name: str) -> Spacing3D | None:
    if value is None:
        return None
    return _expect_spacing(value, field_name=field_name)


def _require_schema(value: str, expected: str) -> None:
    if value != expected:
        raise Phase7ArtifactValidationError(f"schema_name must be {expected!r}.")


def _require_version(value: str, expected: str) -> None:
    if value != expected:
        raise Phase7ArtifactValidationError(f"schema_version must be {expected!r}.")


def _require_sha256(value: str, *, field_name: str) -> None:
    if not _SHA256_RE.fullmatch(value):
        raise Phase7ArtifactValidationError(f"{field_name} must be a lowercase SHA-256 hex digest.")


def _require_optional_sha256(value: str | None, *, field_name: str) -> None:
    if value is not None:
        _require_sha256(value, field_name=field_name)


def _require_identifier(value: str, *, field_name: str) -> None:
    if not _IDENTIFIER_RE.fullmatch(value):
        raise Phase7ArtifactValidationError(f"{field_name} must be a conservative identifier.")


def _require_optional_identifier(value: str | None, *, field_name: str) -> None:
    if value is not None:
        _require_identifier(value, field_name=field_name)


def _require_optional_message(value: str | None, *, field_name: str) -> None:
    if value is not None and not _MESSAGE_RE.fullmatch(value):
        raise Phase7ArtifactValidationError(f"{field_name} must be a single-line message.")


def _require_allowed(value: str, allowed: frozenset[str], *, field_name: str) -> None:
    if value not in allowed:
        raise Phase7ArtifactValidationError(f"{field_name} must be one of {sorted(allowed)!r}.")


def _require_seed(value: int, *, field_name: str) -> None:
    _require_nonnegative_int(value, field_name=field_name)


def _require_optional_seed(value: int | None, *, field_name: str) -> None:
    if value is not None:
        _require_seed(value, field_name=field_name)


def _require_nonnegative_int(value: int, *, field_name: str) -> None:
    if isinstance(value, bool) or value < 0:
        raise Phase7ArtifactValidationError(f"{field_name} must be a nonnegative integer.")


def _require_optional_nonnegative_int(value: int | None, *, field_name: str) -> None:
    if value is not None:
        _require_nonnegative_int(value, field_name=field_name)


def _require_finite_float(value: float, *, field_name: str) -> None:
    if not math.isfinite(value):
        raise Phase7ArtifactValidationError(f"{field_name} must be finite.")


def _require_optional_nonnegative_float(value: float | None, *, field_name: str) -> None:
    if value is None:
        return
    _require_finite_float(value, field_name=field_name)
    if value < 0.0:
        raise Phase7ArtifactValidationError(f"{field_name} must be nonnegative.")


def _require_json_mapping(value: Mapping[str, object], *, field_name: str) -> dict[str, JsonValue]:
    canonical_json_bytes(value)
    checked: dict[str, JsonValue] = {}
    for key, item in value.items():
        _require_identifier(key, field_name=f"{field_name} key")
        checked[key] = cast(JsonValue, item)
    return checked


def _require_shape(value: Shape3D, *, field_name: str) -> None:
    if len(value) != 3 or any(isinstance(item, bool) or item <= 0 for item in value):
        raise Phase7ArtifactValidationError(f"{field_name} must contain three positive integers.")


def _require_optional_shape(value: Shape3D | None, *, field_name: str) -> None:
    if value is not None:
        _require_shape(value, field_name=field_name)


def _require_spacing(value: Spacing3D, *, field_name: str) -> None:
    if len(value) != 3:
        raise Phase7ArtifactValidationError(f"{field_name} must contain three values.")
    for item in value:
        _require_finite_float(item, field_name=field_name)
        if item <= 0.0:
            raise Phase7ArtifactValidationError(f"{field_name} values must be positive.")


def _require_optional_spacing(value: Spacing3D | None, *, field_name: str) -> None:
    if value is not None:
        _require_spacing(value, field_name=field_name)


def _require_orientation(value: str, *, field_name: str) -> None:
    if not _ORIENTATION_RE.fullmatch(value):
        raise Phase7ArtifactValidationError(f"{field_name} must be a three-letter orientation.")


def _require_optional_orientation(value: str | None, *, field_name: str) -> None:
    if value is not None:
        _require_orientation(value, field_name=field_name)
