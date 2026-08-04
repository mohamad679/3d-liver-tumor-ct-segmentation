"""Phase 8 anonymous external image-manifest contract.

The contract in this module is image-only. It stores no label paths, no raw DICOM identifiers,
no absolute paths, and no timestamps. Case identifiers are deterministic from the public
3D-IRCADb-01 case ordinal only.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Final, TypeAlias, cast

from protoem_ct.artifacts.hashing import JsonValue, canonical_json_bytes, sha256_json
from protoem_ct.data._phase2_publication import (
    Phase2PublicationExistingOutputError,
    Phase2PublicationIOError,
    publish_text_no_overwrite,
)

PHASE8_EXTERNAL_IMAGE_MANIFEST_SCHEMA_NAME: Final[str] = "phase8_external_image_manifest"
PHASE8_EXTERNAL_IMAGE_MANIFEST_SCHEMA_VERSION: Final[str] = "v1"
PHASE8_EXTERNAL_IMAGE_CASE_SCHEMA_NAME: Final[str] = "phase8_external_image_case"
PHASE8_EXTERNAL_IMAGE_CASE_SCHEMA_VERSION: Final[str] = "v1"
PHASE8_EXTERNAL_COHORT_IDENTITY_3D_IRCADB_01: Final[str] = "3d_ircadb_01"
PHASE8_EXTERNAL_IMAGE_DICOM_ALLOWLIST_VERSION: Final[str] = (
    "phase8_image_manifest_dicom_header_allowlist_v1"
)
PHASE8_IRCADB_CASE_COUNT: Final[int] = 20

PHASE8_IMAGE_QA_STATUSES: Final[frozenset[str]] = frozenset({"passed", "failed"})
PHASE8_VALIDATION_STATES: Final[frozenset[str]] = frozenset({"passed", "failed", "not_evaluated"})
PHASE8_DICOM_SAFE_HEADER_ALLOWLIST: Final[frozenset[str]] = frozenset(
    {
        "Columns",
        "ImageOrientationPatient",
        "ImagePositionPatient",
        "Modality",
        "PixelSpacing",
        "RescaleIntercept",
        "RescaleSlope",
        "Rows",
        "SliceThickness",
        "SpacingBetweenSlices",
    }
)

MappingLike: TypeAlias = Mapping[str, object]

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ANONYMOUS_CASE_ID_RE = re.compile(r"^ext-ircadb-[0-9]{3}$")
_IMAGE_SERIES_ID_RE = re.compile(r"^image-series-[0-9a-f]{16,64}$")
_REASON_CODE_RE = re.compile(r"^[a-z0-9](?:[a-z0-9_]{0,126}[a-z0-9])?$")
_PHI_TOKEN_RE = re.compile(
    r"(patient(?:id|name)?|accession|institution|physician|referring|birth|dob|mrn)",
    re.IGNORECASE,
)


class Phase8ExternalManifestError(ValueError):
    """Base error for Phase 8 external image-manifest failures."""


class Phase8ExternalManifestValidationError(Phase8ExternalManifestError):
    """Raised when external image-manifest content violates the contract."""


class Phase8ExternalManifestSerializationError(Phase8ExternalManifestError):
    """Raised when external image-manifest JSON cannot be reconstructed strictly."""


class Phase8ExternalManifestHashError(Phase8ExternalManifestError):
    """Raised when the external image-manifest self-hash does not match."""


class Phase8ExternalManifestPublicationError(Phase8ExternalManifestError):
    """Raised when manifest publication cannot be completed without overwriting."""


@dataclass(frozen=True, slots=True)
class Phase8ExternalImageCase:
    """Anonymous image-only record for one 3D-IRCADb-01 case."""

    schema_name: str
    schema_version: str
    anonymous_case_id: str
    source_case_ordinal: int
    image_archive_relative_path: str
    image_archive_sha256: str
    image_archive_size_bytes: int
    image_member_count: int
    image_member_integrity_hash: str
    image_series_identity: str | None
    image_slice_count: int
    image_rows: int | None
    image_columns: int | None
    volume_shape_zyx: tuple[int, int, int] | None
    voxel_spacing_xyz_mm: tuple[float, float, float] | None
    orientation_validation_state: str
    slice_order_validation_state: str
    sop_instance_consistency_state: str
    series_consistency_state: str
    qa_status: str
    qa_reason_codes: tuple[str, ...]
    inclusion_eligible_for_inference: bool

    def __post_init__(self) -> None:
        _require_exact_schema(
            self.schema_name,
            PHASE8_EXTERNAL_IMAGE_CASE_SCHEMA_NAME,
            field_name="schema_name",
        )
        _require_exact_schema(
            self.schema_version,
            PHASE8_EXTERNAL_IMAGE_CASE_SCHEMA_VERSION,
            field_name="schema_version",
        )
        _require_case_ordinal(self.source_case_ordinal)
        expected_id = anonymous_ircadb_case_id(self.source_case_ordinal)
        if self.anonymous_case_id != expected_id:
            raise Phase8ExternalManifestValidationError(
                "anonymous_case_id must be derived only from source_case_ordinal."
            )
        _require_anonymous_case_id(self.anonymous_case_id, field_name="anonymous_case_id")
        _require_relative_posix_path(
            self.image_archive_relative_path,
            field_name="image_archive_relative_path",
        )
        if not self.image_archive_relative_path.endswith("/PATIENT_DICOM.zip"):
            raise Phase8ExternalManifestValidationError(
                "image_archive_relative_path must reference PATIENT_DICOM.zip only."
            )
        _require_sha256(self.image_archive_sha256, field_name="image_archive_sha256")
        _require_sha256(
            self.image_member_integrity_hash,
            field_name="image_member_integrity_hash",
        )
        _require_positive_int(self.image_archive_size_bytes, field_name="image_archive_size_bytes")
        _require_nonnegative_int(self.image_member_count, field_name="image_member_count")
        _require_nonnegative_int(self.image_slice_count, field_name="image_slice_count")
        if self.image_series_identity is not None:
            _require_image_series_identity(self.image_series_identity)
        if self.image_rows is not None:
            _require_positive_int(self.image_rows, field_name="image_rows")
        if self.image_columns is not None:
            _require_positive_int(self.image_columns, field_name="image_columns")
        if self.volume_shape_zyx is not None:
            _require_int_triplet(self.volume_shape_zyx, field_name="volume_shape_zyx")
        if self.voxel_spacing_xyz_mm is not None:
            _require_spacing_triplet(
                self.voxel_spacing_xyz_mm,
                field_name="voxel_spacing_xyz_mm",
            )
        for field_name in (
            "orientation_validation_state",
            "slice_order_validation_state",
            "sop_instance_consistency_state",
            "series_consistency_state",
        ):
            _require_allowed(
                getattr(self, field_name),
                PHASE8_VALIDATION_STATES,
                field_name=field_name,
            )
        _require_allowed(self.qa_status, PHASE8_IMAGE_QA_STATUSES, field_name="qa_status")
        object.__setattr__(
            self,
            "qa_reason_codes",
            tuple(sorted(self.qa_reason_codes)),
        )
        _require_reason_codes(self.qa_reason_codes)
        if self.qa_status == "passed" and self.qa_reason_codes:
            raise Phase8ExternalManifestValidationError(
                "passed image QA records must not include qa_reason_codes."
            )
        if self.qa_status == "failed" and not self.qa_reason_codes:
            raise Phase8ExternalManifestValidationError(
                "failed image QA records require qa_reason_codes."
            )
        if self.inclusion_eligible_for_inference and self.qa_status != "passed":
            raise Phase8ExternalManifestValidationError(
                "inclusion_eligible_for_inference requires qa_status='passed'."
            )
        if self.qa_status == "passed":
            _require_passed_case_geometry(self)

    @property
    def ordering_key(self) -> int:
        """Return the deterministic manifest ordering key."""

        return self.source_case_ordinal


@dataclass(frozen=True, slots=True)
class Phase8ExternalImageManifest:
    """Versioned anonymous Phase 8 image-only manifest for 3D-IRCADb-01."""

    schema_name: str
    schema_version: str
    manifest_hash: str
    cohort_identity: str
    dataset_archive_sha256: str
    dataset_archive_size_bytes: int
    adapter_name: str
    adapter_version: str
    dicom_header_allowlist_version: str
    case_count: int
    cases: tuple[Phase8ExternalImageCase, ...]

    def __post_init__(self) -> None:
        _require_exact_schema(
            self.schema_name,
            PHASE8_EXTERNAL_IMAGE_MANIFEST_SCHEMA_NAME,
            field_name="schema_name",
        )
        _require_exact_schema(
            self.schema_version,
            PHASE8_EXTERNAL_IMAGE_MANIFEST_SCHEMA_VERSION,
            field_name="schema_version",
        )
        _require_sha256(self.manifest_hash, field_name="manifest_hash")
        _require_exact_schema(
            self.cohort_identity,
            PHASE8_EXTERNAL_COHORT_IDENTITY_3D_IRCADB_01,
            field_name="cohort_identity",
        )
        _require_sha256(self.dataset_archive_sha256, field_name="dataset_archive_sha256")
        _require_positive_int(
            self.dataset_archive_size_bytes,
            field_name="dataset_archive_size_bytes",
        )
        _require_safe_token(self.adapter_name, field_name="adapter_name")
        _require_safe_token(self.adapter_version, field_name="adapter_version")
        _require_exact_schema(
            self.dicom_header_allowlist_version,
            PHASE8_EXTERNAL_IMAGE_DICOM_ALLOWLIST_VERSION,
            field_name="dicom_header_allowlist_version",
        )
        ordered = tuple(sorted(self.cases, key=lambda item: item.ordering_key))
        object.__setattr__(self, "cases", ordered)
        if self.case_count != len(self.cases):
            raise Phase8ExternalManifestValidationError("case_count must equal len(cases).")
        if self.case_count != PHASE8_IRCADB_CASE_COUNT:
            raise Phase8ExternalManifestValidationError(
                f"3D-IRCADb-01 image manifest requires exactly {PHASE8_IRCADB_CASE_COUNT} cases."
            )
        ordinals = [case.source_case_ordinal for case in self.cases]
        expected_ordinals = list(range(1, PHASE8_IRCADB_CASE_COUNT + 1))
        if ordinals != expected_ordinals:
            raise Phase8ExternalManifestValidationError(
                "cases must contain each public source_case_ordinal from 1 through 20 exactly once."
            )
        case_ids = [case.anonymous_case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise Phase8ExternalManifestValidationError("cases must not duplicate anonymous IDs.")
        expected_hash = hash_phase8_external_image_manifest(self)
        if self.manifest_hash != expected_hash:
            raise Phase8ExternalManifestHashError(
                "manifest_hash does not match deterministic image-manifest content."
            )


def anonymous_ircadb_case_id(source_case_ordinal: int) -> str:
    """Return the deterministic anonymous case ID for one public 3D-IRCADb-01 ordinal."""

    _require_case_ordinal(source_case_ordinal)
    return f"ext-ircadb-{source_case_ordinal:03d}"


def phase8_external_image_case_to_dict(case: Phase8ExternalImageCase) -> dict[str, JsonValue]:
    """Convert one image-only external case record to a canonical mapping."""

    return {
        "anonymous_case_id": case.anonymous_case_id,
        "image_archive_relative_path": case.image_archive_relative_path,
        "image_archive_sha256": case.image_archive_sha256,
        "image_archive_size_bytes": case.image_archive_size_bytes,
        "image_columns": case.image_columns,
        "image_member_count": case.image_member_count,
        "image_member_integrity_hash": case.image_member_integrity_hash,
        "image_rows": case.image_rows,
        "image_series_identity": case.image_series_identity,
        "image_slice_count": case.image_slice_count,
        "inclusion_eligible_for_inference": case.inclusion_eligible_for_inference,
        "orientation_validation_state": case.orientation_validation_state,
        "qa_reason_codes": list(case.qa_reason_codes),
        "qa_status": case.qa_status,
        "schema_name": case.schema_name,
        "schema_version": case.schema_version,
        "series_consistency_state": case.series_consistency_state,
        "slice_order_validation_state": case.slice_order_validation_state,
        "sop_instance_consistency_state": case.sop_instance_consistency_state,
        "source_case_ordinal": case.source_case_ordinal,
        "volume_shape_zyx": (
            list(case.volume_shape_zyx) if case.volume_shape_zyx is not None else None
        ),
        "voxel_spacing_xyz_mm": (
            list(case.voxel_spacing_xyz_mm) if case.voxel_spacing_xyz_mm is not None else None
        ),
    }


def phase8_external_image_manifest_identity_payload(
    manifest: Phase8ExternalImageManifest,
) -> dict[str, JsonValue]:
    """Return the canonical manifest identity payload, excluding ``manifest_hash``."""

    ordered = sorted(manifest.cases, key=lambda item: item.ordering_key)
    return {
        "adapter_name": manifest.adapter_name,
        "adapter_version": manifest.adapter_version,
        "case_count": manifest.case_count,
        "cases": [phase8_external_image_case_to_dict(case) for case in ordered],
        "cohort_identity": manifest.cohort_identity,
        "dataset_archive_sha256": manifest.dataset_archive_sha256,
        "dataset_archive_size_bytes": manifest.dataset_archive_size_bytes,
        "dicom_header_allowlist_version": manifest.dicom_header_allowlist_version,
        "schema_name": manifest.schema_name,
        "schema_version": manifest.schema_version,
    }


def phase8_external_image_manifest_to_dict(
    manifest: Phase8ExternalImageManifest,
) -> dict[str, JsonValue]:
    """Convert one external image manifest to a canonical mapping."""

    payload = phase8_external_image_manifest_identity_payload(manifest)
    payload["manifest_hash"] = manifest.manifest_hash
    return payload


def hash_phase8_external_image_manifest(manifest: Phase8ExternalImageManifest) -> str:
    """Return the lowercase SHA-256 identity hash for an external image manifest."""

    return sha256_json(phase8_external_image_manifest_identity_payload(manifest))


def phase8_external_image_case_from_mapping(mapping: MappingLike) -> Phase8ExternalImageCase:
    """Reconstruct one external image case from a strict mapping."""

    _require_exact_fields(
        mapping,
        required_fields=_PHASE8_EXTERNAL_IMAGE_CASE_FIELDS,
        object_name="Phase8ExternalImageCase",
    )
    return Phase8ExternalImageCase(
        schema_name=_expect_string(mapping["schema_name"], field_name="schema_name"),
        schema_version=_expect_string(mapping["schema_version"], field_name="schema_version"),
        anonymous_case_id=_expect_string(
            mapping["anonymous_case_id"],
            field_name="anonymous_case_id",
        ),
        source_case_ordinal=_expect_int(
            mapping["source_case_ordinal"],
            field_name="source_case_ordinal",
        ),
        image_archive_relative_path=_expect_string(
            mapping["image_archive_relative_path"],
            field_name="image_archive_relative_path",
        ),
        image_archive_sha256=_expect_string(
            mapping["image_archive_sha256"],
            field_name="image_archive_sha256",
        ),
        image_archive_size_bytes=_expect_int(
            mapping["image_archive_size_bytes"],
            field_name="image_archive_size_bytes",
        ),
        image_member_count=_expect_int(
            mapping["image_member_count"],
            field_name="image_member_count",
        ),
        image_member_integrity_hash=_expect_string(
            mapping["image_member_integrity_hash"],
            field_name="image_member_integrity_hash",
        ),
        image_series_identity=_expect_optional_string(
            mapping["image_series_identity"],
            field_name="image_series_identity",
        ),
        image_slice_count=_expect_int(
            mapping["image_slice_count"],
            field_name="image_slice_count",
        ),
        image_rows=_expect_optional_int(mapping["image_rows"], field_name="image_rows"),
        image_columns=_expect_optional_int(mapping["image_columns"], field_name="image_columns"),
        volume_shape_zyx=_expect_optional_int_triplet(
            mapping["volume_shape_zyx"],
            field_name="volume_shape_zyx",
        ),
        voxel_spacing_xyz_mm=_expect_optional_float_triplet(
            mapping["voxel_spacing_xyz_mm"],
            field_name="voxel_spacing_xyz_mm",
        ),
        orientation_validation_state=_expect_string(
            mapping["orientation_validation_state"],
            field_name="orientation_validation_state",
        ),
        slice_order_validation_state=_expect_string(
            mapping["slice_order_validation_state"],
            field_name="slice_order_validation_state",
        ),
        sop_instance_consistency_state=_expect_string(
            mapping["sop_instance_consistency_state"],
            field_name="sop_instance_consistency_state",
        ),
        series_consistency_state=_expect_string(
            mapping["series_consistency_state"],
            field_name="series_consistency_state",
        ),
        qa_status=_expect_string(mapping["qa_status"], field_name="qa_status"),
        qa_reason_codes=_expect_string_tuple(
            mapping["qa_reason_codes"],
            field_name="qa_reason_codes",
        ),
        inclusion_eligible_for_inference=_expect_bool(
            mapping["inclusion_eligible_for_inference"],
            field_name="inclusion_eligible_for_inference",
        ),
    )


def phase8_external_image_manifest_from_mapping(
    mapping: MappingLike,
) -> Phase8ExternalImageManifest:
    """Reconstruct one external image manifest from a strict mapping."""

    _require_exact_fields(
        mapping,
        required_fields=_PHASE8_EXTERNAL_IMAGE_MANIFEST_FIELDS,
        object_name="Phase8ExternalImageManifest",
    )
    cases_value = mapping["cases"]
    if not isinstance(cases_value, list):
        raise Phase8ExternalManifestSerializationError("cases must be a list.")
    return Phase8ExternalImageManifest(
        schema_name=_expect_string(mapping["schema_name"], field_name="schema_name"),
        schema_version=_expect_string(mapping["schema_version"], field_name="schema_version"),
        manifest_hash=_expect_string(mapping["manifest_hash"], field_name="manifest_hash"),
        cohort_identity=_expect_string(mapping["cohort_identity"], field_name="cohort_identity"),
        dataset_archive_sha256=_expect_string(
            mapping["dataset_archive_sha256"],
            field_name="dataset_archive_sha256",
        ),
        dataset_archive_size_bytes=_expect_int(
            mapping["dataset_archive_size_bytes"],
            field_name="dataset_archive_size_bytes",
        ),
        adapter_name=_expect_string(mapping["adapter_name"], field_name="adapter_name"),
        adapter_version=_expect_string(mapping["adapter_version"], field_name="adapter_version"),
        dicom_header_allowlist_version=_expect_string(
            mapping["dicom_header_allowlist_version"],
            field_name="dicom_header_allowlist_version",
        ),
        case_count=_expect_int(mapping["case_count"], field_name="case_count"),
        cases=tuple(
            phase8_external_image_case_from_mapping(
                _expect_mapping(case, field_name=f"cases[{index}]")
            )
            for index, case in enumerate(cases_value)
        ),
    )


def phase8_external_image_manifest_to_json(manifest: Phase8ExternalImageManifest) -> bytes:
    """Serialize one external image manifest to canonical JSON bytes."""

    return canonical_json_bytes(phase8_external_image_manifest_to_dict(manifest)) + b"\n"


def phase8_external_image_manifest_from_json(data: bytes | str) -> Phase8ExternalImageManifest:
    """Deserialize one external image manifest from JSON bytes or text."""

    return phase8_external_image_manifest_from_mapping(_json_to_mapping(data))


def build_phase8_external_image_manifest(
    *,
    dataset_archive_sha256: str,
    dataset_archive_size_bytes: int,
    adapter_name: str,
    adapter_version: str,
    cases: Sequence[Phase8ExternalImageCase],
) -> Phase8ExternalImageManifest:
    """Build a self-hashed anonymous external image manifest from validated case records."""

    ordered_cases = tuple(sorted(cases, key=lambda item: item.ordering_key))
    payload: dict[str, JsonValue] = {
        "adapter_name": adapter_name,
        "adapter_version": adapter_version,
        "case_count": len(ordered_cases),
        "cases": [phase8_external_image_case_to_dict(case) for case in ordered_cases],
        "cohort_identity": PHASE8_EXTERNAL_COHORT_IDENTITY_3D_IRCADB_01,
        "dataset_archive_sha256": dataset_archive_sha256,
        "dataset_archive_size_bytes": dataset_archive_size_bytes,
        "dicom_header_allowlist_version": PHASE8_EXTERNAL_IMAGE_DICOM_ALLOWLIST_VERSION,
        "schema_name": PHASE8_EXTERNAL_IMAGE_MANIFEST_SCHEMA_NAME,
        "schema_version": PHASE8_EXTERNAL_IMAGE_MANIFEST_SCHEMA_VERSION,
    }
    manifest_hash = sha256_json(payload)
    return Phase8ExternalImageManifest(
        schema_name=PHASE8_EXTERNAL_IMAGE_MANIFEST_SCHEMA_NAME,
        schema_version=PHASE8_EXTERNAL_IMAGE_MANIFEST_SCHEMA_VERSION,
        manifest_hash=manifest_hash,
        cohort_identity=PHASE8_EXTERNAL_COHORT_IDENTITY_3D_IRCADB_01,
        dataset_archive_sha256=dataset_archive_sha256,
        dataset_archive_size_bytes=dataset_archive_size_bytes,
        adapter_name=adapter_name,
        adapter_version=adapter_version,
        dicom_header_allowlist_version=PHASE8_EXTERNAL_IMAGE_DICOM_ALLOWLIST_VERSION,
        case_count=len(ordered_cases),
        cases=ordered_cases,
    )


def publish_phase8_external_image_manifest(
    manifest: Phase8ExternalImageManifest,
    output_path: Path,
) -> None:
    """Publish an external image manifest JSON file without overwriting an existing artifact."""

    if not output_path.is_absolute():
        raise Phase8ExternalManifestPublicationError("manifest output path must be absolute.")
    try:
        publish_text_no_overwrite(
            text=phase8_external_image_manifest_to_json(manifest).decode("utf-8"),
            output_path=output_path,
            temporary_exists_message="temporary Phase 8 manifest output already exists",
            final_exists_message="Phase 8 manifest output already exists",
        )
    except (Phase2PublicationExistingOutputError, Phase2PublicationIOError) as exc:
        raise Phase8ExternalManifestPublicationError(str(exc)) from exc


_PHASE8_EXTERNAL_IMAGE_CASE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_name",
        "schema_version",
        "anonymous_case_id",
        "source_case_ordinal",
        "image_archive_relative_path",
        "image_archive_sha256",
        "image_archive_size_bytes",
        "image_member_count",
        "image_member_integrity_hash",
        "image_series_identity",
        "image_slice_count",
        "image_rows",
        "image_columns",
        "volume_shape_zyx",
        "voxel_spacing_xyz_mm",
        "orientation_validation_state",
        "slice_order_validation_state",
        "sop_instance_consistency_state",
        "series_consistency_state",
        "qa_status",
        "qa_reason_codes",
        "inclusion_eligible_for_inference",
    }
)
_PHASE8_EXTERNAL_IMAGE_MANIFEST_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_name",
        "schema_version",
        "manifest_hash",
        "cohort_identity",
        "dataset_archive_sha256",
        "dataset_archive_size_bytes",
        "adapter_name",
        "adapter_version",
        "dicom_header_allowlist_version",
        "case_count",
        "cases",
    }
)


def _json_to_mapping(data: bytes | str) -> MappingLike:
    try:
        decoded = json.loads(data)
    except json.JSONDecodeError as exc:
        raise Phase8ExternalManifestSerializationError(
            "invalid external image-manifest JSON."
        ) from exc
    if not isinstance(decoded, Mapping):
        raise Phase8ExternalManifestSerializationError(
            "external image-manifest JSON root must be an object."
        )
    return cast(MappingLike, decoded)


def _expect_mapping(value: object, *, field_name: str) -> MappingLike:
    if not isinstance(value, Mapping):
        raise Phase8ExternalManifestSerializationError(f"{field_name} must be a mapping.")
    return cast(MappingLike, value)


def _expect_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise Phase8ExternalManifestSerializationError(f"{field_name} must be a string.")
    return value


def _expect_optional_string(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _expect_string(value, field_name=field_name)


def _expect_bool(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise Phase8ExternalManifestSerializationError(f"{field_name} must be a boolean.")
    return value


def _expect_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise Phase8ExternalManifestSerializationError(f"{field_name} must be an integer.")
    return value


def _expect_optional_int(value: object, *, field_name: str) -> int | None:
    if value is None:
        return None
    return _expect_int(value, field_name=field_name)


def _expect_int_triplet(value: object, *, field_name: str) -> tuple[int, int, int]:
    if not isinstance(value, list) or len(value) != 3:
        raise Phase8ExternalManifestSerializationError(f"{field_name} must be a 3-item array.")
    return tuple(
        _expect_int(item, field_name=f"{field_name}[{index}]") for index, item in enumerate(value)
    )  # type: ignore[return-value]


def _expect_optional_int_triplet(
    value: object,
    *,
    field_name: str,
) -> tuple[int, int, int] | None:
    if value is None:
        return None
    return _expect_int_triplet(value, field_name=field_name)


def _expect_float_triplet(value: object, *, field_name: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise Phase8ExternalManifestSerializationError(f"{field_name} must be a 3-item array.")
    return tuple(
        _expect_float(item, field_name=f"{field_name}[{index}]") for index, item in enumerate(value)
    )  # type: ignore[return-value]


def _expect_optional_float_triplet(
    value: object,
    *,
    field_name: str,
) -> tuple[float, float, float] | None:
    if value is None:
        return None
    return _expect_float_triplet(value, field_name=field_name)


def _expect_string_tuple(value: object, *, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise Phase8ExternalManifestSerializationError(f"{field_name} must be an array.")
    return tuple(
        _expect_string(item, field_name=f"{field_name}[{index}]")
        for index, item in enumerate(value)
    )


def _expect_float(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise Phase8ExternalManifestSerializationError(f"{field_name} must be a number.")
    result = float(value)
    if not math.isfinite(result):
        raise Phase8ExternalManifestSerializationError(f"{field_name} must be finite.")
    return result


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
        raise Phase8ExternalManifestSerializationError(
            f"{object_name} fields mismatch; missing={missing!r}, extra={extra!r}."
        )


def _require_exact_schema(value: str, expected: str, *, field_name: str) -> None:
    if value != expected:
        raise Phase8ExternalManifestValidationError(f"{field_name} must be {expected!r}.")


def _require_case_ordinal(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 999:
        raise Phase8ExternalManifestValidationError("source_case_ordinal must be in [1, 999].")


def _require_anonymous_case_id(value: str, *, field_name: str) -> None:
    if _ANONYMOUS_CASE_ID_RE.fullmatch(value) is None:
        raise Phase8ExternalManifestValidationError(
            f"{field_name} must use the ext-ircadb-NNN anonymous format."
        )
    _require_no_phi_token(value, field_name=field_name)


def _require_image_series_identity(value: str) -> None:
    if _IMAGE_SERIES_ID_RE.fullmatch(value) is None:
        raise Phase8ExternalManifestValidationError(
            "image_series_identity must be a non-PHI image-series hash identifier."
        )
    _require_no_phi_token(value, field_name="image_series_identity")


def _require_relative_posix_path(value: str, *, field_name: str) -> None:
    if not isinstance(value, str) or not value:
        raise Phase8ExternalManifestValidationError(
            f"{field_name} must be a nonempty relative POSIX path."
        )
    if value.startswith("~") or "\\" in value or "\x00" in value:
        raise Phase8ExternalManifestValidationError(
            f"{field_name} must be a normalized relative POSIX path."
        )
    posix_path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    if posix_path.is_absolute() or windows_path.is_absolute():
        raise Phase8ExternalManifestValidationError(f"{field_name} must not be absolute.")
    if str(posix_path) != value or value in {".", ".."}:
        raise Phase8ExternalManifestValidationError(f"{field_name} must be normalized.")
    if ".." in posix_path.parts or any(part == "" for part in posix_path.parts):
        raise Phase8ExternalManifestValidationError(
            f"{field_name} must not contain traversal or empty segments."
        )


def _require_sha256(value: str, *, field_name: str) -> None:
    if _SHA256_RE.fullmatch(value) is None:
        raise Phase8ExternalManifestValidationError(
            f"{field_name} must be a lowercase 64-character SHA-256 hex digest."
        )


def _require_safe_token(value: str, *, field_name: str) -> None:
    if _REASON_CODE_RE.fullmatch(value) is None:
        raise Phase8ExternalManifestValidationError(f"{field_name} must be a safe token.")
    _require_no_phi_token(value, field_name=field_name)


def _require_reason_codes(values: tuple[str, ...]) -> None:
    if len(values) != len(set(values)):
        raise Phase8ExternalManifestValidationError("qa_reason_codes must not contain duplicates.")
    for value in values:
        if _REASON_CODE_RE.fullmatch(value) is None:
            raise Phase8ExternalManifestValidationError(
                "qa_reason_codes must use conservative reason-code tokens."
            )
        _require_no_phi_token(value, field_name="qa_reason_codes")


def _require_allowed(value: str, allowed: frozenset[str], *, field_name: str) -> None:
    if value not in allowed:
        raise Phase8ExternalManifestValidationError(
            f"{field_name} must be one of {sorted(allowed)!r}."
        )


def _require_positive_int(value: int, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Phase8ExternalManifestValidationError(f"{field_name} must be a positive integer.")


def _require_nonnegative_int(value: int, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Phase8ExternalManifestValidationError(f"{field_name} must be a nonnegative integer.")


def _require_int_triplet(value: tuple[int, int, int], *, field_name: str) -> None:
    if len(value) != 3:
        raise Phase8ExternalManifestValidationError(f"{field_name} must contain three integers.")
    for item in value:
        _require_positive_int(item, field_name=field_name)


def _require_spacing_triplet(value: tuple[float, float, float], *, field_name: str) -> None:
    if len(value) != 3:
        raise Phase8ExternalManifestValidationError(f"{field_name} must contain three numbers.")
    for item in value:
        if isinstance(item, bool) or not isinstance(item, float) or not math.isfinite(item):
            raise Phase8ExternalManifestValidationError(
                f"{field_name} values must be finite floats."
            )
        if item <= 0.0:
            raise Phase8ExternalManifestValidationError(f"{field_name} values must be positive.")


def _require_passed_case_geometry(case: Phase8ExternalImageCase) -> None:
    if (
        case.image_series_identity is None
        or case.image_slice_count <= 0
        or case.image_rows is None
        or case.image_columns is None
        or case.volume_shape_zyx is None
        or case.voxel_spacing_xyz_mm is None
    ):
        raise Phase8ExternalManifestValidationError(
            "passed image QA records require complete image geometry."
        )
    if case.volume_shape_zyx != (
        case.image_slice_count,
        case.image_rows,
        case.image_columns,
    ):
        raise Phase8ExternalManifestValidationError(
            "volume_shape_zyx must match slice count, rows, and columns."
        )


def _require_no_phi_token(value: str, *, field_name: str) -> None:
    if _PHI_TOKEN_RE.search(value) is not None:
        raise Phase8ExternalManifestValidationError(
            f"{field_name} must not contain PHI-bearing identifier tokens."
        )


__all__ = [
    "PHASE8_DICOM_SAFE_HEADER_ALLOWLIST",
    "PHASE8_EXTERNAL_COHORT_IDENTITY_3D_IRCADB_01",
    "PHASE8_EXTERNAL_IMAGE_CASE_SCHEMA_NAME",
    "PHASE8_EXTERNAL_IMAGE_CASE_SCHEMA_VERSION",
    "PHASE8_EXTERNAL_IMAGE_DICOM_ALLOWLIST_VERSION",
    "PHASE8_EXTERNAL_IMAGE_MANIFEST_SCHEMA_NAME",
    "PHASE8_EXTERNAL_IMAGE_MANIFEST_SCHEMA_VERSION",
    "PHASE8_IMAGE_QA_STATUSES",
    "PHASE8_IRCADB_CASE_COUNT",
    "PHASE8_VALIDATION_STATES",
    "Phase8ExternalImageCase",
    "Phase8ExternalImageManifest",
    "Phase8ExternalManifestError",
    "Phase8ExternalManifestHashError",
    "Phase8ExternalManifestPublicationError",
    "Phase8ExternalManifestSerializationError",
    "Phase8ExternalManifestValidationError",
    "anonymous_ircadb_case_id",
    "build_phase8_external_image_manifest",
    "hash_phase8_external_image_manifest",
    "phase8_external_image_case_from_mapping",
    "phase8_external_image_case_to_dict",
    "phase8_external_image_manifest_from_json",
    "phase8_external_image_manifest_from_mapping",
    "phase8_external_image_manifest_identity_payload",
    "phase8_external_image_manifest_to_dict",
    "phase8_external_image_manifest_to_json",
    "publish_phase8_external_image_manifest",
]
