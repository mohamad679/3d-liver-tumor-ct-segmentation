"""Phase 8 final closure / non-scientific reproduction-provenance record.

This module builds and (de)serializes one small, self-hashed, non-medical artifact that records
whether the documented Phase 8 provenance chain — development provenance, definitive configuration,
selected checkpoint, freeze, preregistration, prediction lock, external evaluation summary,
corrected internal/external comparison, final report — resolves consistently.

This module never opens a medical image, DICOM archive, NIfTI volume, prediction array, or label.
It never trains, infers, or recomputes any metric. It only records identity strings (SHA-256 hashes,
Git commit hashes, schema names, file paths, and booleans) that were already computed by earlier
Phase 8 packages or by this closure's own read-only hash verification of existing artifacts.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypeAlias, cast

from protoem_ct.artifacts.hashing import JsonValue, canonical_json_bytes, sha256_json
from protoem_ct.data._phase2_publication import (
    Phase2PublicationExistingOutputError,
    Phase2PublicationIOError,
    publish_text_no_overwrite,
)

PHASE8_CLOSURE_REPRODUCTION_RECORD_SCHEMA_NAME: Final[str] = (
    "phase8_final_closure_reproduction_record"
)
PHASE8_CLOSURE_REPRODUCTION_RECORD_SCHEMA_VERSION: Final[str] = "v1"

PHASE8_CLOSURE_STATUSES: Final[frozenset[str]] = frozenset(
    {"closed_negative_external_validation_result", "blocked"}
)

MappingLike: TypeAlias = Mapping[str, object]

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class Phase8ClosureError(ValueError):
    """Base error for the Phase 8 closure/reproduction record contract."""


class Phase8ClosureValidationError(Phase8ClosureError):
    """Raised when a closure record violates its schema contract."""


class Phase8ClosureSerializationError(Phase8ClosureError):
    """Raised when a closure record mapping cannot be reconstructed safely."""


class Phase8ClosureHashError(Phase8ClosureError):
    """Raised when a closure record self-hash does not match its content."""


class Phase8ClosurePublicationError(Phase8ClosureError):
    """Raised when publishing a closure record fails or would overwrite an existing artifact."""


@dataclass(frozen=True, slots=True)
class Phase8ClosureReproductionRecord:
    """Non-medical, self-hashed record of the verified Phase 8 provenance chain at closure."""

    schema_name: str
    schema_version: str
    closure_record_hash: str
    closure_status: str
    freeze_artifact_sha256: str
    checkpoint_sha256: str
    preregistration_hash: str
    prediction_lock_hash: str
    external_metric_report_hash: str
    domain_shift_record_hash: str
    original_comparison_artifact_sha256: str
    corrected_comparison_artifact_relative_path: str
    provenance_fix_commit: str
    final_report_relative_path: str
    external_drive_accessible_this_session: bool
    scientific_computation_rerun: bool

    def __post_init__(self) -> None:
        _require_exact(
            self.schema_name,
            PHASE8_CLOSURE_REPRODUCTION_RECORD_SCHEMA_NAME,
            "schema_name",
        )
        _require_exact(
            self.schema_version,
            PHASE8_CLOSURE_REPRODUCTION_RECORD_SCHEMA_VERSION,
            "schema_version",
        )
        _require_sha256(self.closure_record_hash, "closure_record_hash")
        _require_allowed(self.closure_status, PHASE8_CLOSURE_STATUSES, "closure_status")
        _require_sha256(self.freeze_artifact_sha256, "freeze_artifact_sha256")
        _require_sha256(self.checkpoint_sha256, "checkpoint_sha256")
        _require_sha256(self.preregistration_hash, "preregistration_hash")
        _require_sha256(self.prediction_lock_hash, "prediction_lock_hash")
        _require_sha256(self.external_metric_report_hash, "external_metric_report_hash")
        _require_sha256(self.domain_shift_record_hash, "domain_shift_record_hash")
        _require_sha256(
            self.original_comparison_artifact_sha256,
            "original_comparison_artifact_sha256",
        )
        _require_nonempty(
            self.corrected_comparison_artifact_relative_path,
            "corrected_comparison_artifact_relative_path",
        )
        _require_git_commit(self.provenance_fix_commit, "provenance_fix_commit")
        _require_nonempty(self.final_report_relative_path, "final_report_relative_path")
        if self.scientific_computation_rerun:
            raise Phase8ClosureValidationError(
                "closure records must never claim that scientific computation was rerun."
            )
        expected_hash = hash_phase8_closure_reproduction_record(self)
        if self.closure_record_hash != expected_hash:
            raise Phase8ClosureHashError(
                "closure_record_hash does not match deterministic content."
            )


def phase8_closure_reproduction_record_identity_payload(
    record: Phase8ClosureReproductionRecord,
) -> dict[str, JsonValue]:
    """Return the closure-record identity payload, excluding the self-hash field."""

    return {
        "checkpoint_sha256": record.checkpoint_sha256,
        "closure_status": record.closure_status,
        "corrected_comparison_artifact_relative_path": (
            record.corrected_comparison_artifact_relative_path
        ),
        "domain_shift_record_hash": record.domain_shift_record_hash,
        "external_drive_accessible_this_session": record.external_drive_accessible_this_session,
        "external_metric_report_hash": record.external_metric_report_hash,
        "final_report_relative_path": record.final_report_relative_path,
        "freeze_artifact_sha256": record.freeze_artifact_sha256,
        "original_comparison_artifact_sha256": record.original_comparison_artifact_sha256,
        "prediction_lock_hash": record.prediction_lock_hash,
        "preregistration_hash": record.preregistration_hash,
        "provenance_fix_commit": record.provenance_fix_commit,
        "schema_name": record.schema_name,
        "schema_version": record.schema_version,
        "scientific_computation_rerun": record.scientific_computation_rerun,
    }


def phase8_closure_reproduction_record_to_dict(
    record: Phase8ClosureReproductionRecord,
) -> dict[str, JsonValue]:
    """Convert one closure record to canonical JSON, including the self-hash field."""

    payload = phase8_closure_reproduction_record_identity_payload(record)
    payload["closure_record_hash"] = record.closure_record_hash
    return payload


def hash_phase8_closure_reproduction_record(record: Phase8ClosureReproductionRecord) -> str:
    """Return the canonical SHA-256 identity hash for a closure record."""

    return sha256_json(phase8_closure_reproduction_record_identity_payload(record))


def build_phase8_closure_reproduction_record(
    *,
    closure_status: str,
    freeze_artifact_sha256: str,
    checkpoint_sha256: str,
    preregistration_hash: str,
    prediction_lock_hash: str,
    external_metric_report_hash: str,
    domain_shift_record_hash: str,
    original_comparison_artifact_sha256: str,
    corrected_comparison_artifact_relative_path: str,
    provenance_fix_commit: str,
    final_report_relative_path: str,
    external_drive_accessible_this_session: bool,
) -> Phase8ClosureReproductionRecord:
    """Build one self-hashed Phase 8 closure/reproduction record from verified identities."""

    payload: dict[str, JsonValue] = {
        "checkpoint_sha256": checkpoint_sha256,
        "closure_status": closure_status,
        "corrected_comparison_artifact_relative_path": (
            corrected_comparison_artifact_relative_path
        ),
        "domain_shift_record_hash": domain_shift_record_hash,
        "external_drive_accessible_this_session": external_drive_accessible_this_session,
        "external_metric_report_hash": external_metric_report_hash,
        "final_report_relative_path": final_report_relative_path,
        "freeze_artifact_sha256": freeze_artifact_sha256,
        "original_comparison_artifact_sha256": original_comparison_artifact_sha256,
        "prediction_lock_hash": prediction_lock_hash,
        "preregistration_hash": preregistration_hash,
        "provenance_fix_commit": provenance_fix_commit,
        "schema_name": PHASE8_CLOSURE_REPRODUCTION_RECORD_SCHEMA_NAME,
        "schema_version": PHASE8_CLOSURE_REPRODUCTION_RECORD_SCHEMA_VERSION,
        "scientific_computation_rerun": False,
    }
    return Phase8ClosureReproductionRecord(
        schema_name=PHASE8_CLOSURE_REPRODUCTION_RECORD_SCHEMA_NAME,
        schema_version=PHASE8_CLOSURE_REPRODUCTION_RECORD_SCHEMA_VERSION,
        closure_record_hash=sha256_json(payload),
        closure_status=closure_status,
        freeze_artifact_sha256=freeze_artifact_sha256,
        checkpoint_sha256=checkpoint_sha256,
        preregistration_hash=preregistration_hash,
        prediction_lock_hash=prediction_lock_hash,
        external_metric_report_hash=external_metric_report_hash,
        domain_shift_record_hash=domain_shift_record_hash,
        original_comparison_artifact_sha256=original_comparison_artifact_sha256,
        corrected_comparison_artifact_relative_path=(corrected_comparison_artifact_relative_path),
        provenance_fix_commit=provenance_fix_commit,
        final_report_relative_path=final_report_relative_path,
        external_drive_accessible_this_session=external_drive_accessible_this_session,
        scientific_computation_rerun=False,
    )


def phase8_closure_reproduction_record_from_json(payload: bytes) -> Phase8ClosureReproductionRecord:
    """Reconstruct one closure record from canonical JSON bytes, re-validating its self-hash."""

    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise Phase8ClosureSerializationError("closure record payload is not valid JSON.") from exc
    mapping = _expect_mapping(decoded, "closure record")
    return Phase8ClosureReproductionRecord(
        schema_name=_expect_string(mapping.get("schema_name"), "schema_name"),
        schema_version=_expect_string(mapping.get("schema_version"), "schema_version"),
        closure_record_hash=_expect_string(
            mapping.get("closure_record_hash"), "closure_record_hash"
        ),
        closure_status=_expect_string(mapping.get("closure_status"), "closure_status"),
        freeze_artifact_sha256=_expect_string(
            mapping.get("freeze_artifact_sha256"), "freeze_artifact_sha256"
        ),
        checkpoint_sha256=_expect_string(mapping.get("checkpoint_sha256"), "checkpoint_sha256"),
        preregistration_hash=_expect_string(
            mapping.get("preregistration_hash"), "preregistration_hash"
        ),
        prediction_lock_hash=_expect_string(
            mapping.get("prediction_lock_hash"), "prediction_lock_hash"
        ),
        external_metric_report_hash=_expect_string(
            mapping.get("external_metric_report_hash"), "external_metric_report_hash"
        ),
        domain_shift_record_hash=_expect_string(
            mapping.get("domain_shift_record_hash"), "domain_shift_record_hash"
        ),
        original_comparison_artifact_sha256=_expect_string(
            mapping.get("original_comparison_artifact_sha256"),
            "original_comparison_artifact_sha256",
        ),
        corrected_comparison_artifact_relative_path=_expect_string(
            mapping.get("corrected_comparison_artifact_relative_path"),
            "corrected_comparison_artifact_relative_path",
        ),
        provenance_fix_commit=_expect_string(
            mapping.get("provenance_fix_commit"), "provenance_fix_commit"
        ),
        final_report_relative_path=_expect_string(
            mapping.get("final_report_relative_path"), "final_report_relative_path"
        ),
        external_drive_accessible_this_session=_expect_bool(
            mapping.get("external_drive_accessible_this_session"),
            "external_drive_accessible_this_session",
        ),
        scientific_computation_rerun=_expect_bool(
            mapping.get("scientific_computation_rerun"), "scientific_computation_rerun"
        ),
    )


def publish_phase8_closure_reproduction_record(
    record: Phase8ClosureReproductionRecord, output_path: Path
) -> None:
    """Publish one closure record as canonical JSON, refusing to overwrite an existing artifact."""

    payload = phase8_closure_reproduction_record_to_dict(record)
    try:
        publish_text_no_overwrite(
            text=(canonical_json_bytes(payload) + b"\n").decode("utf-8"),
            output_path=output_path,
            temporary_exists_message="temporary Phase 8 closure record already exists.",
            final_exists_message="Phase 8 closure record already exists.",
        )
    except (Phase2PublicationExistingOutputError, Phase2PublicationIOError) as exc:
        raise Phase8ClosurePublicationError(str(exc)) from exc


def _expect_mapping(value: object, field_name: str) -> MappingLike:
    if not isinstance(value, Mapping):
        raise Phase8ClosureSerializationError(f"{field_name} must be a mapping.")
    return cast(MappingLike, value)


def _expect_string(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise Phase8ClosureSerializationError(f"{field_name} must be a string.")
    return value


def _expect_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise Phase8ClosureSerializationError(f"{field_name} must be a boolean.")
    return value


def _require_exact(value: str, expected: str, field_name: str) -> None:
    if value != expected:
        raise Phase8ClosureValidationError(f"{field_name} must equal {expected!r}, got {value!r}.")


def _require_allowed(value: str, allowed: frozenset[str], field_name: str) -> None:
    if value not in allowed:
        raise Phase8ClosureValidationError(f"{field_name} must be one of {sorted(allowed)}.")


def _require_sha256(value: str, field_name: str) -> None:
    if not _SHA256_RE.match(value):
        raise Phase8ClosureValidationError(f"{field_name} must be a 64-hex-character SHA-256.")


def _require_git_commit(value: str, field_name: str) -> None:
    if not _GIT_COMMIT_RE.match(value):
        raise Phase8ClosureValidationError(f"{field_name} must be a 40-hex-character Git commit.")


def _require_nonempty(value: str, field_name: str) -> None:
    if not value:
        raise Phase8ClosureValidationError(f"{field_name} must be a non-empty string.")


__all__ = [
    "PHASE8_CLOSURE_REPRODUCTION_RECORD_SCHEMA_NAME",
    "PHASE8_CLOSURE_REPRODUCTION_RECORD_SCHEMA_VERSION",
    "PHASE8_CLOSURE_STATUSES",
    "Phase8ClosureError",
    "Phase8ClosureHashError",
    "Phase8ClosurePublicationError",
    "Phase8ClosureReproductionRecord",
    "Phase8ClosureSerializationError",
    "Phase8ClosureValidationError",
    "build_phase8_closure_reproduction_record",
    "hash_phase8_closure_reproduction_record",
    "phase8_closure_reproduction_record_from_json",
    "phase8_closure_reproduction_record_identity_payload",
    "phase8_closure_reproduction_record_to_dict",
    "publish_phase8_closure_reproduction_record",
]
