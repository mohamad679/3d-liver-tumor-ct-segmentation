"""Phase 8 aggregate-only domain-shift artifact contracts.

The generation surface in this module consumes only approved image-only Phase 8 artifacts. It does
not read raw images, labels, predictions, checkpoints, or local dataset paths.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, TypeAlias, cast

from protoem_ct.artifacts.hashing import JsonValue, canonical_json_bytes, sha256_json
from protoem_ct.external.manifest import Phase8ExternalImageManifest

PHASE8_DOMAIN_SHIFT_RECORD_SCHEMA_NAME: Final[str] = "phase8_domain_shift_record"
PHASE8_DOMAIN_SHIFT_RECORD_SCHEMA_VERSION: Final[str] = "v1"
PHASE8_DOMAIN_SHIFT_DIMENSION_SCHEMA_NAME: Final[str] = "phase8_domain_shift_dimension"
PHASE8_DOMAIN_SHIFT_DIMENSION_SCHEMA_VERSION: Final[str] = "v1"
PHASE8_FROZEN_AGGREGATE_REFERENCE_SCHEMA_NAME: Final[str] = "phase8_frozen_aggregate_reference"
PHASE8_FROZEN_AGGREGATE_REFERENCE_SCHEMA_VERSION: Final[str] = "v1"

PHASE8_DOMAIN_SHIFT_DIMENSIONS: Final[tuple[str, ...]] = (
    "source_identity",
    "acquisition_representation",
    "image_matrix",
    "slice_counts",
    "voxel_spacing",
    "anisotropy",
    "orientation",
    "hu_readiness",
    "modality",
    "image_qa_compatibility",
    "differences_vs_frozen_internal_preprocessing_contract",
)
PHASE8_DOMAIN_SHIFT_AVAILABILITY_STATUSES: Final[frozenset[str]] = frozenset(
    {"available", "unavailable"}
)
PHASE8_DOMAIN_SHIFT_COMPARISON_STATUSES: Final[frozenset[str]] = frozenset(
    {"not_compared", "compared_to_approved_aggregate_reference"}
)
PHASE8_FROZEN_AGGREGATE_REFERENCE_ROLES: Final[frozenset[str]] = frozenset(
    {"internal_preprocessing_contract"}
)
PHASE8_FROZEN_AGGREGATE_APPROVAL_STATES: Final[frozenset[str]] = frozenset({"approved"})

MappingLike: TypeAlias = Mapping[str, object]

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[a-z0-9](?:[a-z0-9_.-]{0,126}[a-z0-9])?$")
_FORBIDDEN_TEXT_RE = re.compile(
    r"(/Volumes|file://|PATIENT_DICOM|MASKS_DICOM|LABELLED_DICOM|MESHES_VTK|"
    r"ext-ircadb-[0-9]{3}|patient(?:id|name)?|accession|institution|physician|"
    r"birth|dob|mrn|scanner|vendor|manufacturer|site)",
    re.IGNORECASE,
)


class Phase8DomainShiftError(ValueError):
    """Base error for Phase 8 domain-shift contracts."""


class Phase8DomainShiftValidationError(Phase8DomainShiftError):
    """Raised when a domain-shift artifact violates its schema contract."""


class Phase8DomainShiftSerializationError(Phase8DomainShiftError):
    """Raised when domain-shift JSON cannot be reconstructed strictly."""


class Phase8DomainShiftHashError(Phase8DomainShiftError):
    """Raised when a domain-shift self-hash does not match its content."""


@dataclass(frozen=True, slots=True)
class Phase8FrozenAggregateReference:
    """Approved aggregate/config reference used for non-raw internal comparison."""

    schema_name: str
    schema_version: str
    reference_role: str
    referenced_schema_name: str
    referenced_schema_version: str
    artifact_hash: str
    approval_state: str

    def __post_init__(self) -> None:
        _require_exact(
            self.schema_name,
            PHASE8_FROZEN_AGGREGATE_REFERENCE_SCHEMA_NAME,
            "schema_name",
        )
        _require_exact(
            self.schema_version,
            PHASE8_FROZEN_AGGREGATE_REFERENCE_SCHEMA_VERSION,
            "schema_version",
        )
        _require_allowed(
            self.reference_role,
            PHASE8_FROZEN_AGGREGATE_REFERENCE_ROLES,
            "reference_role",
        )
        _require_identifier(self.referenced_schema_name, "referenced_schema_name")
        _require_identifier(self.referenced_schema_version, "referenced_schema_version")
        _require_sha256(self.artifact_hash, "artifact_hash")
        _require_allowed(
            self.approval_state,
            PHASE8_FROZEN_AGGREGATE_APPROVAL_STATES,
            "approval_state",
        )


@dataclass(frozen=True, slots=True)
class Phase8DomainShiftDimension:
    """One aggregate-only domain-shift dimension summary."""

    schema_name: str
    schema_version: str
    dimension_name: str
    availability_status: str
    unavailable_reason_codes: tuple[str, ...]
    external_aggregate_summary: Mapping[str, JsonValue] | None
    comparison_status: str
    approved_reference: Phase8FrozenAggregateReference | None
    difference_summary: Mapping[str, JsonValue] | None

    def __post_init__(self) -> None:
        _require_exact(self.schema_name, PHASE8_DOMAIN_SHIFT_DIMENSION_SCHEMA_NAME, "schema_name")
        _require_exact(
            self.schema_version,
            PHASE8_DOMAIN_SHIFT_DIMENSION_SCHEMA_VERSION,
            "schema_version",
        )
        if self.dimension_name not in PHASE8_DOMAIN_SHIFT_DIMENSIONS:
            raise Phase8DomainShiftValidationError(
                f"unsupported domain-shift dimension: {self.dimension_name!r}."
            )
        _require_allowed(
            self.availability_status,
            PHASE8_DOMAIN_SHIFT_AVAILABILITY_STATUSES,
            "availability_status",
        )
        object.__setattr__(
            self,
            "unavailable_reason_codes",
            tuple(sorted(self.unavailable_reason_codes)),
        )
        _require_reason_codes(self.unavailable_reason_codes)
        _require_allowed(
            self.comparison_status,
            PHASE8_DOMAIN_SHIFT_COMPARISON_STATUSES,
            "comparison_status",
        )
        if self.availability_status == "available":
            if self.external_aggregate_summary is None:
                raise Phase8DomainShiftValidationError(
                    "available dimensions require external_aggregate_summary."
                )
            if self.unavailable_reason_codes:
                raise Phase8DomainShiftValidationError(
                    "available dimensions must not include unavailable reasons."
                )
        elif self.external_aggregate_summary is not None or not self.unavailable_reason_codes:
            raise Phase8DomainShiftValidationError(
                "unavailable dimensions require reasons and no external summary."
            )
        if self.comparison_status == "compared_to_approved_aggregate_reference":
            if self.approved_reference is None or self.difference_summary is None:
                raise Phase8DomainShiftValidationError(
                    "compared dimensions require approved_reference and difference_summary."
                )
        elif self.approved_reference is not None or self.difference_summary is not None:
            raise Phase8DomainShiftValidationError(
                "not_compared dimensions must not include comparison references or differences."
            )
        _require_safe_json(self.external_aggregate_summary, "external_aggregate_summary")
        _require_safe_json(self.difference_summary, "difference_summary")

    @property
    def ordering_key(self) -> int:
        """Return the canonical dimension ordering key."""

        return PHASE8_DOMAIN_SHIFT_DIMENSIONS.index(self.dimension_name)


@dataclass(frozen=True, slots=True)
class Phase8DomainShiftRecord:
    """Aggregate-only Phase 8 external domain-shift summary."""

    schema_name: str
    schema_version: str
    domain_shift_record_hash: str
    external_manifest_hash: str
    external_manifest_schema_name: str
    external_manifest_schema_version: str
    external_cohort_identity: str
    aggregate_only: bool
    labels_accessed: bool
    raw_images_accessed_by_domain_shift_generation: bool
    scanner_vendor_site_recorded: bool
    dimensions: tuple[Phase8DomainShiftDimension, ...]

    def __post_init__(self) -> None:
        _require_exact(self.schema_name, PHASE8_DOMAIN_SHIFT_RECORD_SCHEMA_NAME, "schema_name")
        _require_exact(
            self.schema_version,
            PHASE8_DOMAIN_SHIFT_RECORD_SCHEMA_VERSION,
            "schema_version",
        )
        _require_sha256(self.domain_shift_record_hash, "domain_shift_record_hash")
        _require_sha256(self.external_manifest_hash, "external_manifest_hash")
        _require_identifier(self.external_manifest_schema_name, "external_manifest_schema_name")
        _require_identifier(
            self.external_manifest_schema_version,
            "external_manifest_schema_version",
        )
        _require_identifier(self.external_cohort_identity, "external_cohort_identity")
        if not self.aggregate_only:
            raise Phase8DomainShiftValidationError("domain-shift records must be aggregate-only.")
        if self.labels_accessed:
            raise Phase8DomainShiftValidationError(
                "domain-shift generation must not access labels."
            )
        if self.raw_images_accessed_by_domain_shift_generation:
            raise Phase8DomainShiftValidationError(
                "domain-shift generation must not access raw images."
            )
        if self.scanner_vendor_site_recorded:
            raise Phase8DomainShiftValidationError(
                "scanner/vendor/site fields must not be recorded."
            )
        ordered = tuple(sorted(self.dimensions, key=lambda item: item.ordering_key))
        object.__setattr__(self, "dimensions", ordered)
        dimension_names = tuple(item.dimension_name for item in ordered)
        if dimension_names != PHASE8_DOMAIN_SHIFT_DIMENSIONS:
            raise Phase8DomainShiftValidationError(
                "domain-shift records require each supported dimension exactly once."
            )
        expected_hash = hash_phase8_domain_shift_record(self)
        if self.domain_shift_record_hash != expected_hash:
            raise Phase8DomainShiftHashError(
                "domain_shift_record_hash does not match deterministic content."
            )


def phase8_frozen_aggregate_reference_to_dict(
    reference: Phase8FrozenAggregateReference,
) -> dict[str, JsonValue]:
    """Convert one approved aggregate reference to canonical JSON."""

    return {
        "approval_state": reference.approval_state,
        "artifact_hash": reference.artifact_hash,
        "reference_role": reference.reference_role,
        "referenced_schema_name": reference.referenced_schema_name,
        "referenced_schema_version": reference.referenced_schema_version,
        "schema_name": reference.schema_name,
        "schema_version": reference.schema_version,
    }


def phase8_domain_shift_dimension_to_dict(
    dimension: Phase8DomainShiftDimension,
) -> dict[str, JsonValue]:
    """Convert one domain-shift dimension to canonical JSON."""

    return {
        "approved_reference": None
        if dimension.approved_reference is None
        else phase8_frozen_aggregate_reference_to_dict(dimension.approved_reference),
        "availability_status": dimension.availability_status,
        "comparison_status": dimension.comparison_status,
        "difference_summary": (
            dict(dimension.difference_summary) if dimension.difference_summary is not None else None
        ),
        "dimension_name": dimension.dimension_name,
        "external_aggregate_summary": (
            dict(dimension.external_aggregate_summary)
            if dimension.external_aggregate_summary is not None
            else None
        ),
        "schema_name": dimension.schema_name,
        "schema_version": dimension.schema_version,
        "unavailable_reason_codes": list(dimension.unavailable_reason_codes),
    }


def phase8_domain_shift_record_identity_payload(
    record: Phase8DomainShiftRecord,
) -> dict[str, JsonValue]:
    """Return the domain-shift identity payload excluding the self-hash."""

    ordered = sorted(record.dimensions, key=lambda item: item.ordering_key)
    return {
        "aggregate_only": record.aggregate_only,
        "dimensions": [phase8_domain_shift_dimension_to_dict(item) for item in ordered],
        "external_cohort_identity": record.external_cohort_identity,
        "external_manifest_hash": record.external_manifest_hash,
        "external_manifest_schema_name": record.external_manifest_schema_name,
        "external_manifest_schema_version": record.external_manifest_schema_version,
        "labels_accessed": record.labels_accessed,
        "raw_images_accessed_by_domain_shift_generation": (
            record.raw_images_accessed_by_domain_shift_generation
        ),
        "scanner_vendor_site_recorded": record.scanner_vendor_site_recorded,
        "schema_name": record.schema_name,
        "schema_version": record.schema_version,
    }


def phase8_domain_shift_record_to_dict(record: Phase8DomainShiftRecord) -> dict[str, JsonValue]:
    """Convert one domain-shift record to canonical JSON."""

    payload = phase8_domain_shift_record_identity_payload(record)
    payload["domain_shift_record_hash"] = record.domain_shift_record_hash
    return payload


def hash_phase8_domain_shift_record(record: Phase8DomainShiftRecord) -> str:
    """Return the canonical SHA-256 identity hash for a domain-shift record."""

    return sha256_json(phase8_domain_shift_record_identity_payload(record))


def phase8_domain_shift_record_to_json(record: Phase8DomainShiftRecord) -> bytes:
    """Serialize one domain-shift record to canonical JSON bytes."""

    return canonical_json_bytes(phase8_domain_shift_record_to_dict(record)) + b"\n"


def phase8_frozen_aggregate_reference_from_mapping(
    mapping: MappingLike,
) -> Phase8FrozenAggregateReference:
    """Reconstruct one approved aggregate reference from a strict mapping."""

    _require_exact_fields(
        mapping,
        _FROZEN_AGGREGATE_REFERENCE_FIELDS,
        "Phase8FrozenAggregateReference",
    )
    return Phase8FrozenAggregateReference(
        schema_name=_expect_string(mapping["schema_name"], "schema_name"),
        schema_version=_expect_string(mapping["schema_version"], "schema_version"),
        reference_role=_expect_string(mapping["reference_role"], "reference_role"),
        referenced_schema_name=_expect_string(
            mapping["referenced_schema_name"],
            "referenced_schema_name",
        ),
        referenced_schema_version=_expect_string(
            mapping["referenced_schema_version"],
            "referenced_schema_version",
        ),
        artifact_hash=_expect_string(mapping["artifact_hash"], "artifact_hash"),
        approval_state=_expect_string(mapping["approval_state"], "approval_state"),
    )


def phase8_domain_shift_dimension_from_mapping(
    mapping: MappingLike,
) -> Phase8DomainShiftDimension:
    """Reconstruct one domain-shift dimension from a strict mapping."""

    _require_exact_fields(mapping, _DOMAIN_SHIFT_DIMENSION_FIELDS, "Phase8DomainShiftDimension")
    approved_reference_value = mapping["approved_reference"]
    return Phase8DomainShiftDimension(
        schema_name=_expect_string(mapping["schema_name"], "schema_name"),
        schema_version=_expect_string(mapping["schema_version"], "schema_version"),
        dimension_name=_expect_string(mapping["dimension_name"], "dimension_name"),
        availability_status=_expect_string(mapping["availability_status"], "availability_status"),
        unavailable_reason_codes=_expect_string_tuple(
            mapping["unavailable_reason_codes"],
            "unavailable_reason_codes",
        ),
        external_aggregate_summary=_expect_optional_json_mapping(
            mapping["external_aggregate_summary"],
            "external_aggregate_summary",
        ),
        comparison_status=_expect_string(mapping["comparison_status"], "comparison_status"),
        approved_reference=None
        if approved_reference_value is None
        else phase8_frozen_aggregate_reference_from_mapping(
            _expect_mapping(approved_reference_value, "approved_reference")
        ),
        difference_summary=_expect_optional_json_mapping(
            mapping["difference_summary"],
            "difference_summary",
        ),
    )


def phase8_domain_shift_record_from_mapping(mapping: MappingLike) -> Phase8DomainShiftRecord:
    """Reconstruct one domain-shift record from a strict mapping."""

    _require_exact_fields(mapping, _DOMAIN_SHIFT_RECORD_FIELDS, "Phase8DomainShiftRecord")
    dimensions_value = mapping["dimensions"]
    if not isinstance(dimensions_value, list):
        raise Phase8DomainShiftSerializationError("dimensions must be a list.")
    return Phase8DomainShiftRecord(
        schema_name=_expect_string(mapping["schema_name"], "schema_name"),
        schema_version=_expect_string(mapping["schema_version"], "schema_version"),
        domain_shift_record_hash=_expect_string(
            mapping["domain_shift_record_hash"],
            "domain_shift_record_hash",
        ),
        external_manifest_hash=_expect_string(
            mapping["external_manifest_hash"],
            "external_manifest_hash",
        ),
        external_manifest_schema_name=_expect_string(
            mapping["external_manifest_schema_name"],
            "external_manifest_schema_name",
        ),
        external_manifest_schema_version=_expect_string(
            mapping["external_manifest_schema_version"],
            "external_manifest_schema_version",
        ),
        external_cohort_identity=_expect_string(
            mapping["external_cohort_identity"],
            "external_cohort_identity",
        ),
        aggregate_only=_expect_bool(mapping["aggregate_only"], "aggregate_only"),
        labels_accessed=_expect_bool(mapping["labels_accessed"], "labels_accessed"),
        raw_images_accessed_by_domain_shift_generation=_expect_bool(
            mapping["raw_images_accessed_by_domain_shift_generation"],
            "raw_images_accessed_by_domain_shift_generation",
        ),
        scanner_vendor_site_recorded=_expect_bool(
            mapping["scanner_vendor_site_recorded"],
            "scanner_vendor_site_recorded",
        ),
        dimensions=tuple(
            phase8_domain_shift_dimension_from_mapping(
                _expect_mapping(item, f"dimensions[{index}]")
            )
            for index, item in enumerate(dimensions_value)
        ),
    )


def phase8_domain_shift_record_from_json(data: bytes | str) -> Phase8DomainShiftRecord:
    """Deserialize one domain-shift record from JSON bytes or text."""

    return phase8_domain_shift_record_from_mapping(_json_to_mapping(data))


def build_phase8_domain_shift_record(
    *,
    external_manifest: Phase8ExternalImageManifest,
    internal_preprocessing_reference: Phase8FrozenAggregateReference | None = None,
    internal_preprocessing_difference_summary: Mapping[str, JsonValue] | None = None,
) -> Phase8DomainShiftRecord:
    """Build an aggregate-only domain-shift record from approved image-only artifacts."""

    comparison_dimension = _internal_preprocessing_dimension(
        reference=internal_preprocessing_reference,
        difference_summary=internal_preprocessing_difference_summary,
    )
    dimensions = (
        _available_dimension("source_identity", _source_identity_summary(external_manifest)),
        _available_dimension(
            "acquisition_representation",
            _acquisition_representation_summary(external_manifest),
        ),
        _available_dimension("image_matrix", _image_matrix_summary(external_manifest)),
        _available_dimension("slice_counts", _slice_count_summary(external_manifest)),
        _available_dimension("voxel_spacing", _voxel_spacing_summary(external_manifest)),
        _available_dimension("anisotropy", _anisotropy_summary(external_manifest)),
        _available_dimension("orientation", _orientation_summary(external_manifest)),
        _available_dimension("hu_readiness", _hu_readiness_summary(external_manifest)),
        _available_dimension("modality", _modality_summary(external_manifest)),
        _available_dimension(
            "image_qa_compatibility",
            _image_qa_compatibility_summary(external_manifest),
        ),
        comparison_dimension,
    )
    payload: dict[str, JsonValue] = {
        "aggregate_only": True,
        "dimensions": [phase8_domain_shift_dimension_to_dict(item) for item in dimensions],
        "external_cohort_identity": external_manifest.cohort_identity,
        "external_manifest_hash": external_manifest.manifest_hash,
        "external_manifest_schema_name": external_manifest.schema_name,
        "external_manifest_schema_version": external_manifest.schema_version,
        "labels_accessed": False,
        "raw_images_accessed_by_domain_shift_generation": False,
        "scanner_vendor_site_recorded": False,
        "schema_name": PHASE8_DOMAIN_SHIFT_RECORD_SCHEMA_NAME,
        "schema_version": PHASE8_DOMAIN_SHIFT_RECORD_SCHEMA_VERSION,
    }
    return Phase8DomainShiftRecord(
        schema_name=PHASE8_DOMAIN_SHIFT_RECORD_SCHEMA_NAME,
        schema_version=PHASE8_DOMAIN_SHIFT_RECORD_SCHEMA_VERSION,
        domain_shift_record_hash=sha256_json(payload),
        external_manifest_hash=external_manifest.manifest_hash,
        external_manifest_schema_name=external_manifest.schema_name,
        external_manifest_schema_version=external_manifest.schema_version,
        external_cohort_identity=external_manifest.cohort_identity,
        aggregate_only=True,
        labels_accessed=False,
        raw_images_accessed_by_domain_shift_generation=False,
        scanner_vendor_site_recorded=False,
        dimensions=dimensions,
    )


def _available_dimension(
    dimension_name: str,
    summary: Mapping[str, JsonValue],
) -> Phase8DomainShiftDimension:
    return Phase8DomainShiftDimension(
        schema_name=PHASE8_DOMAIN_SHIFT_DIMENSION_SCHEMA_NAME,
        schema_version=PHASE8_DOMAIN_SHIFT_DIMENSION_SCHEMA_VERSION,
        dimension_name=dimension_name,
        availability_status="available",
        unavailable_reason_codes=(),
        external_aggregate_summary=summary,
        comparison_status="not_compared",
        approved_reference=None,
        difference_summary=None,
    )


def _internal_preprocessing_dimension(
    *,
    reference: Phase8FrozenAggregateReference | None,
    difference_summary: Mapping[str, JsonValue] | None,
) -> Phase8DomainShiftDimension:
    if reference is None and difference_summary is None:
        return Phase8DomainShiftDimension(
            schema_name=PHASE8_DOMAIN_SHIFT_DIMENSION_SCHEMA_NAME,
            schema_version=PHASE8_DOMAIN_SHIFT_DIMENSION_SCHEMA_VERSION,
            dimension_name="differences_vs_frozen_internal_preprocessing_contract",
            availability_status="unavailable",
            unavailable_reason_codes=("approved_internal_aggregate_reference_absent",),
            external_aggregate_summary=None,
            comparison_status="not_compared",
            approved_reference=None,
            difference_summary=None,
        )
    if reference is None or difference_summary is None:
        raise Phase8DomainShiftValidationError(
            "internal preprocessing comparison requires both approved reference and differences."
        )
    return Phase8DomainShiftDimension(
        schema_name=PHASE8_DOMAIN_SHIFT_DIMENSION_SCHEMA_NAME,
        schema_version=PHASE8_DOMAIN_SHIFT_DIMENSION_SCHEMA_VERSION,
        dimension_name="differences_vs_frozen_internal_preprocessing_contract",
        availability_status="available",
        unavailable_reason_codes=(),
        external_aggregate_summary={"comparison_basis": "approved_aggregate_or_config_reference"},
        comparison_status="compared_to_approved_aggregate_reference",
        approved_reference=reference,
        difference_summary=difference_summary,
    )


def _source_identity_summary(manifest: Phase8ExternalImageManifest) -> dict[str, JsonValue]:
    return {
        "case_count": manifest.case_count,
        "cohort_identity": manifest.cohort_identity,
        "source_artifact_schema_name": manifest.schema_name,
        "source_artifact_schema_version": manifest.schema_version,
    }


def _acquisition_representation_summary(
    manifest: Phase8ExternalImageManifest,
) -> dict[str, JsonValue]:
    return {
        "image_only_source_artifact": True,
        "label_archives_accessed": False,
        "raw_archive_member_paths_persisted": False,
        "representation": "dicom_ct_image_series_archive",
        "represented_case_count": manifest.case_count,
    }


def _image_matrix_summary(manifest: Phase8ExternalImageManifest) -> dict[str, JsonValue]:
    rows = tuple(case.image_rows for case in manifest.cases if case.image_rows is not None)
    columns = tuple(case.image_columns for case in manifest.cases if case.image_columns is not None)
    return {
        "available_case_count": len(rows),
        "columns_max": max(columns) if columns else None,
        "columns_min": min(columns) if columns else None,
        "rows_max": max(rows) if rows else None,
        "rows_min": min(rows) if rows else None,
        "unavailable_case_count": manifest.case_count - len(rows),
        "unique_matrix_count": len(
            {
                (case.image_rows, case.image_columns)
                for case in manifest.cases
                if case.image_rows is not None and case.image_columns is not None
            }
        ),
    }


def _slice_count_summary(manifest: Phase8ExternalImageManifest) -> dict[str, JsonValue]:
    counts = tuple(case.image_slice_count for case in manifest.cases if case.image_slice_count > 0)
    return {
        "available_case_count": len(counts),
        "slice_count_max": max(counts) if counts else None,
        "slice_count_mean": _mean(counts),
        "slice_count_min": min(counts) if counts else None,
        "unavailable_case_count": manifest.case_count - len(counts),
    }


def _voxel_spacing_summary(manifest: Phase8ExternalImageManifest) -> dict[str, JsonValue]:
    spacings = tuple(
        case.voxel_spacing_xyz_mm for case in manifest.cases if case.voxel_spacing_xyz_mm
    )
    return {
        "available_case_count": len(spacings),
        "spacing_x_max_mm": max((item[0] for item in spacings), default=None),
        "spacing_x_min_mm": min((item[0] for item in spacings), default=None),
        "spacing_y_max_mm": max((item[1] for item in spacings), default=None),
        "spacing_y_min_mm": min((item[1] for item in spacings), default=None),
        "spacing_z_max_mm": max((item[2] for item in spacings), default=None),
        "spacing_z_min_mm": min((item[2] for item in spacings), default=None),
        "unavailable_case_count": manifest.case_count - len(spacings),
        "unique_spacing_count": len(set(spacings)),
    }


def _anisotropy_summary(manifest: Phase8ExternalImageManifest) -> dict[str, JsonValue]:
    ratios = tuple(
        spacing[2] / min(spacing[0], spacing[1])
        for case in manifest.cases
        if (spacing := case.voxel_spacing_xyz_mm) is not None
    )
    return {
        "anisotropy_ratio_definition": "spacing_z_mm_divided_by_min_in_plane_spacing_mm",
        "anisotropy_ratio_max": max(ratios) if ratios else None,
        "anisotropy_ratio_mean": _mean(ratios),
        "anisotropy_ratio_min": min(ratios) if ratios else None,
        "available_case_count": len(ratios),
        "unavailable_case_count": manifest.case_count - len(ratios),
    }


def _orientation_summary(manifest: Phase8ExternalImageManifest) -> dict[str, JsonValue]:
    return {
        "orientation_value_status": "unavailable_not_persisted_in_image_manifest",
        "orientation_failed_case_count": sum(
            1 for case in manifest.cases if case.orientation_validation_state == "failed"
        ),
        "orientation_not_evaluated_case_count": sum(
            1 for case in manifest.cases if case.orientation_validation_state == "not_evaluated"
        ),
        "orientation_passed_case_count": sum(
            1 for case in manifest.cases if case.orientation_validation_state == "passed"
        ),
    }


def _hu_readiness_summary(manifest: Phase8ExternalImageManifest) -> dict[str, JsonValue]:
    failed_reason_codes = tuple(code for case in manifest.cases for code in case.qa_reason_codes)
    return {
        "hu_intensity_distribution_status": "unavailable_not_persisted_in_image_manifest",
        "hu_ready_case_count": sum(1 for case in manifest.cases if case.qa_status == "passed"),
        "missing_rescaling_failure_count": failed_reason_codes.count("missing_rescaling"),
        "non_finite_hu_failure_count": failed_reason_codes.count("non_finite_hu_pixels"),
        "pixel_decode_failure_count": failed_reason_codes.count("pixel_data_decode_failed"),
    }


def _modality_summary(manifest: Phase8ExternalImageManifest) -> dict[str, JsonValue]:
    failed_reason_codes = tuple(code for case in manifest.cases for code in case.qa_reason_codes)
    return {
        "ct_compatible_passed_case_count": sum(
            1 for case in manifest.cases if case.qa_status == "passed"
        ),
        "modality_value_status": "aggregate_ct_compatibility_only",
        "non_ct_modality_failure_count": failed_reason_codes.count("non_ct_modality"),
    }


def _image_qa_compatibility_summary(
    manifest: Phase8ExternalImageManifest,
) -> dict[str, JsonValue]:
    return {
        "failed_case_count": sum(1 for case in manifest.cases if case.qa_status == "failed"),
        "inclusion_eligible_for_inference_count": sum(
            1 for case in manifest.cases if case.inclusion_eligible_for_inference
        ),
        "passed_case_count": sum(1 for case in manifest.cases if case.qa_status == "passed"),
        "reason_code_counts": _reason_code_counts(manifest),
    }


def _reason_code_counts(manifest: Phase8ExternalImageManifest) -> dict[str, JsonValue]:
    counts: dict[str, int] = {}
    for case in manifest.cases:
        for code in case.qa_reason_codes:
            counts[code] = counts.get(code, 0) + 1
    return {key: counts[key] for key in sorted(counts)}


def _mean(values: Sequence[float | int]) -> float | None:
    if not values:
        return None
    result = float(sum(values)) / float(len(values))
    if not math.isfinite(result):
        raise Phase8DomainShiftValidationError("aggregate mean was non-finite.")
    return result


def _json_to_mapping(data: bytes | str) -> MappingLike:
    try:
        decoded = json.loads(data)
    except json.JSONDecodeError as exc:
        raise Phase8DomainShiftSerializationError("invalid domain-shift JSON.") from exc
    if not isinstance(decoded, Mapping):
        raise Phase8DomainShiftSerializationError("domain-shift JSON root must be an object.")
    return cast(MappingLike, decoded)


def _expect_mapping(value: object, field_name: str) -> MappingLike:
    if not isinstance(value, Mapping):
        raise Phase8DomainShiftSerializationError(f"{field_name} must be a mapping.")
    return cast(MappingLike, value)


def _expect_string(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise Phase8DomainShiftSerializationError(f"{field_name} must be a string.")
    return value


def _expect_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise Phase8DomainShiftSerializationError(f"{field_name} must be a boolean.")
    return value


def _expect_string_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise Phase8DomainShiftSerializationError(f"{field_name} must be a list.")
    return tuple(_expect_string(item, f"{field_name}[{index}]") for index, item in enumerate(value))


def _expect_optional_json_mapping(
    value: object,
    field_name: str,
) -> Mapping[str, JsonValue] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise Phase8DomainShiftSerializationError(f"{field_name} must be a mapping or null.")
    checked = _require_json_value(value, field_name)
    if not isinstance(checked, dict):
        raise Phase8DomainShiftSerializationError(f"{field_name} must be a mapping or null.")
    return checked


def _require_exact_fields(
    mapping: MappingLike,
    required_fields: frozenset[str],
    object_name: str,
) -> None:
    keys = set(mapping)
    missing = sorted(required_fields - keys)
    extra = sorted(keys - required_fields)
    if missing or extra:
        raise Phase8DomainShiftSerializationError(
            f"{object_name} fields mismatch; missing={missing!r}, extra={extra!r}."
        )


def _require_exact(value: str, expected: str, field_name: str) -> None:
    if value != expected:
        raise Phase8DomainShiftValidationError(f"{field_name} must be {expected!r}.")


def _require_allowed(value: str, allowed: frozenset[str], field_name: str) -> None:
    if value not in allowed:
        raise Phase8DomainShiftValidationError(f"{field_name} must be one of {sorted(allowed)!r}.")


def _require_sha256(value: str, field_name: str) -> None:
    if not _SHA256_RE.fullmatch(value):
        raise Phase8DomainShiftValidationError(
            f"{field_name} must be a lowercase 64-character SHA-256 hex digest."
        )


def _require_identifier(value: str, field_name: str) -> None:
    if not _IDENTIFIER_RE.fullmatch(value):
        raise Phase8DomainShiftValidationError(f"{field_name} must be a conservative identifier.")
    _require_safe_text(value, field_name)


def _require_reason_codes(values: tuple[str, ...]) -> None:
    for value in values:
        _require_identifier(value, "unavailable_reason_codes")


def _require_safe_json(value: object, location: str) -> None:
    _require_json_value(value, location)


def _require_json_value(value: object, location: str) -> JsonValue:
    if value is None:
        return None
    if isinstance(value, bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise Phase8DomainShiftValidationError(f"{location} must not be NaN or Infinity.")
        return value
    if isinstance(value, str):
        _require_safe_text(value, location)
        return value
    if isinstance(value, Mapping):
        checked: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise Phase8DomainShiftValidationError(f"{location} contains non-string keys.")
            _require_identifier(key, f"{location}.key")
            checked[key] = _require_json_value(item, f"{location}.{key}")
        return checked
    if isinstance(value, list):
        return [
            _require_json_value(item, f"{location}[{index}]") for index, item in enumerate(value)
        ]
    raise Phase8DomainShiftValidationError(f"{location} is not JSON-compatible.")


def _require_safe_text(value: str, field_name: str) -> None:
    if _FORBIDDEN_TEXT_RE.search(value):
        raise Phase8DomainShiftValidationError(
            f"{field_name} contains prohibited PHI, raw path, scanner/vendor/site, or case text."
        )


_FROZEN_AGGREGATE_REFERENCE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "approval_state",
        "artifact_hash",
        "reference_role",
        "referenced_schema_name",
        "referenced_schema_version",
        "schema_name",
        "schema_version",
    }
)
_DOMAIN_SHIFT_DIMENSION_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "approved_reference",
        "availability_status",
        "comparison_status",
        "difference_summary",
        "dimension_name",
        "external_aggregate_summary",
        "schema_name",
        "schema_version",
        "unavailable_reason_codes",
    }
)
_DOMAIN_SHIFT_RECORD_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "aggregate_only",
        "dimensions",
        "domain_shift_record_hash",
        "external_cohort_identity",
        "external_manifest_hash",
        "external_manifest_schema_name",
        "external_manifest_schema_version",
        "labels_accessed",
        "raw_images_accessed_by_domain_shift_generation",
        "scanner_vendor_site_recorded",
        "schema_name",
        "schema_version",
    }
)

__all__ = [
    "PHASE8_DOMAIN_SHIFT_AVAILABILITY_STATUSES",
    "PHASE8_DOMAIN_SHIFT_COMPARISON_STATUSES",
    "PHASE8_DOMAIN_SHIFT_DIMENSIONS",
    "PHASE8_DOMAIN_SHIFT_DIMENSION_SCHEMA_NAME",
    "PHASE8_DOMAIN_SHIFT_DIMENSION_SCHEMA_VERSION",
    "PHASE8_DOMAIN_SHIFT_RECORD_SCHEMA_NAME",
    "PHASE8_DOMAIN_SHIFT_RECORD_SCHEMA_VERSION",
    "PHASE8_FROZEN_AGGREGATE_APPROVAL_STATES",
    "PHASE8_FROZEN_AGGREGATE_REFERENCE_ROLES",
    "PHASE8_FROZEN_AGGREGATE_REFERENCE_SCHEMA_NAME",
    "PHASE8_FROZEN_AGGREGATE_REFERENCE_SCHEMA_VERSION",
    "Phase8DomainShiftDimension",
    "Phase8DomainShiftError",
    "Phase8DomainShiftHashError",
    "Phase8DomainShiftRecord",
    "Phase8DomainShiftSerializationError",
    "Phase8DomainShiftValidationError",
    "Phase8FrozenAggregateReference",
    "build_phase8_domain_shift_record",
    "hash_phase8_domain_shift_record",
    "phase8_domain_shift_dimension_from_mapping",
    "phase8_domain_shift_dimension_to_dict",
    "phase8_domain_shift_record_from_json",
    "phase8_domain_shift_record_from_mapping",
    "phase8_domain_shift_record_identity_payload",
    "phase8_domain_shift_record_to_dict",
    "phase8_domain_shift_record_to_json",
    "phase8_frozen_aggregate_reference_from_mapping",
    "phase8_frozen_aggregate_reference_to_dict",
]
