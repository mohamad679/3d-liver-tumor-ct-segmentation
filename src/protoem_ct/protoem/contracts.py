"""Typed Phase 6 ProtoEM initialization boundary contracts."""

from __future__ import annotations

import inspect
import math
import re
from dataclasses import dataclass
from typing import Final

from protoem_ct.models.interfaces import FeatureEncoding3D
from protoem_ct.retrieval.inference import PrototypeInferenceResult
from protoem_ct.retrieval.prototypes import SupportPrototypeMemory

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[a-z0-9](?:[a-z0-9_.-]{0,126}[a-z0-9])?$")

PROTOEM_PHASE5_ARTIFACT_REFERENCE_SCHEMA_NAME: Final[str] = "protoem_phase5_artifact_reference"
PROTOEM_PHASE5_ARTIFACT_REFERENCE_SCHEMA_VERSION: Final[str] = "v1"
PROTOEM_SUPPORT_PROVENANCE_SCHEMA_NAME: Final[str] = "protoem_support_provenance"
PROTOEM_SUPPORT_PROVENANCE_SCHEMA_VERSION: Final[str] = "v1"
PROTOEM_QUERY_FEATURE_INPUT_SCHEMA_NAME: Final[str] = "protoem_query_feature_input"
PROTOEM_QUERY_FEATURE_INPUT_SCHEMA_VERSION: Final[str] = "v1"
PROTOEM_PROTOTYPE_INPUT_SCHEMA_NAME: Final[str] = "protoem_prototype_input"
PROTOEM_PROTOTYPE_INPUT_SCHEMA_VERSION: Final[str] = "v1"
PROTOEM_INFERENCE_INPUT_SCHEMA_NAME: Final[str] = "protoem_inference_input"
PROTOEM_INFERENCE_INPUT_SCHEMA_VERSION: Final[str] = "v1"
PROTOEM_PHASE5_REFERENCE_BUNDLE_SCHEMA_NAME: Final[str] = "protoem_phase5_reference_bundle"
PROTOEM_PHASE5_REFERENCE_BUNDLE_SCHEMA_VERSION: Final[str] = "v1"


class ProtoEMContractError(ValueError):
    """Base error for Phase 6 ProtoEM initialization contracts."""


class ProtoEMContractValidationError(ProtoEMContractError):
    """Raised when one ProtoEM initialization contract is malformed."""


@dataclass(frozen=True, slots=True)
class ProtoEMPhase5ArtifactReference:
    """One exact Phase 5 artifact reference required by ProtoEM initialization."""

    schema_name: str
    schema_version: str
    artifact_schema_name: str
    artifact_hash: str

    def __post_init__(self) -> None:
        _require_schema_name(
            self.schema_name,
            expected=PROTOEM_PHASE5_ARTIFACT_REFERENCE_SCHEMA_NAME,
            field_name="schema_name",
        )
        _require_schema_version(
            self.schema_version,
            expected=PROTOEM_PHASE5_ARTIFACT_REFERENCE_SCHEMA_VERSION,
            field_name="schema_version",
        )
        _require_identifier(self.artifact_schema_name, field_name="artifact_schema_name")
        _require_sha256(self.artifact_hash, field_name="artifact_hash")


@dataclass(frozen=True, slots=True)
class ProtoEMSupportProvenance:
    """One explicit support provenance record carried from Phase 5 into Phase 6."""

    schema_name: str
    schema_version: str
    support_identifier: str
    support_patient_id: str
    support_case_id: str
    source_input_identity: str
    support_manifest_hash: str
    support_assignment_index: int
    dataset_manifest_hash: str | None
    encoder_identity: str
    preprocessing_hash: str
    checkpoint_hash: str
    feature_stage: str
    normalization_name: str

    def __post_init__(self) -> None:
        _require_schema_name(
            self.schema_name,
            expected=PROTOEM_SUPPORT_PROVENANCE_SCHEMA_NAME,
            field_name="schema_name",
        )
        _require_schema_version(
            self.schema_version,
            expected=PROTOEM_SUPPORT_PROVENANCE_SCHEMA_VERSION,
            field_name="schema_version",
        )
        _require_identifier(self.support_identifier, field_name="support_identifier")
        _require_identifier(self.support_patient_id, field_name="support_patient_id")
        _require_identifier(self.support_case_id, field_name="support_case_id")
        _require_identifier(self.source_input_identity, field_name="source_input_identity")
        _require_sha256(self.support_manifest_hash, field_name="support_manifest_hash")
        _require_nonnegative_int(
            self.support_assignment_index,
            field_name="support_assignment_index",
        )
        _require_optional_sha256(self.dataset_manifest_hash, field_name="dataset_manifest_hash")
        _require_identifier(self.encoder_identity, field_name="encoder_identity")
        _require_sha256(self.preprocessing_hash, field_name="preprocessing_hash")
        _require_sha256(self.checkpoint_hash, field_name="checkpoint_hash")
        _require_identifier(self.feature_stage, field_name="feature_stage")
        _require_identifier(self.normalization_name, field_name="normalization_name")

    @property
    def ordering_key(self) -> tuple[str, str, str, int, str]:
        """Return the deterministic canonical ordering key."""

        return (
            self.support_patient_id,
            self.support_case_id,
            self.source_input_identity,
            self.support_assignment_index,
            self.support_identifier,
        )


@dataclass(frozen=True, slots=True)
class ProtoEMQueryFeatureInput:
    """Phase 5 query feature contract accepted by ProtoEM initialization."""

    schema_name: str
    schema_version: str
    query_feature_encoding: FeatureEncoding3D
    query_identity: str
    query_patient_id: str
    query_case_id: str
    dataset_manifest_hash: str | None

    def __post_init__(self) -> None:
        _require_schema_name(
            self.schema_name,
            expected=PROTOEM_QUERY_FEATURE_INPUT_SCHEMA_NAME,
            field_name="schema_name",
        )
        _require_schema_version(
            self.schema_version,
            expected=PROTOEM_QUERY_FEATURE_INPUT_SCHEMA_VERSION,
            field_name="schema_version",
        )
        _require_identifier(self.query_identity, field_name="query_identity")
        _require_identifier(self.query_patient_id, field_name="query_patient_id")
        _require_identifier(self.query_case_id, field_name="query_case_id")
        _require_optional_sha256(self.dataset_manifest_hash, field_name="dataset_manifest_hash")
        if self.query_feature_encoding.metadata.input_identity != self.query_identity:
            raise ProtoEMContractValidationError(
                "query_identity must match query_feature_encoding.metadata.input_identity."
            )


@dataclass(frozen=True, slots=True)
class ProtoEMPrototypeInput:
    """Phase 5 prototype-memory and support-provenance contract for ProtoEM initialization."""

    schema_name: str
    schema_version: str
    prototype_memory: SupportPrototypeMemory
    support_provenance: tuple[ProtoEMSupportProvenance, ...]

    def __post_init__(self) -> None:
        _require_schema_name(
            self.schema_name,
            expected=PROTOEM_PROTOTYPE_INPUT_SCHEMA_NAME,
            field_name="schema_name",
        )
        _require_schema_version(
            self.schema_version,
            expected=PROTOEM_PROTOTYPE_INPUT_SCHEMA_VERSION,
            field_name="schema_version",
        )
        if not self.support_provenance:
            raise ProtoEMContractValidationError(
                "support_provenance must contain at least one support record."
            )


@dataclass(frozen=True, slots=True)
class ProtoEMInferenceInput:
    """Phase 5 prototype-only inference contract accepted by ProtoEM initialization."""

    schema_name: str
    schema_version: str
    inference_result: PrototypeInferenceResult

    def __post_init__(self) -> None:
        _require_schema_name(
            self.schema_name,
            expected=PROTOEM_INFERENCE_INPUT_SCHEMA_NAME,
            field_name="schema_name",
        )
        _require_schema_version(
            self.schema_version,
            expected=PROTOEM_INFERENCE_INPUT_SCHEMA_VERSION,
            field_name="schema_version",
        )


@dataclass(frozen=True, slots=True)
class ProtoEMPhase5ReferenceBundle:
    """Exact Phase 5 reference bundle required by one ProtoEM initialization call."""

    schema_name: str
    schema_version: str
    initialization_reference: ProtoEMPhase5ArtifactReference
    prototype_reference: ProtoEMPhase5ArtifactReference
    inference_reference: ProtoEMPhase5ArtifactReference

    def __post_init__(self) -> None:
        _require_schema_name(
            self.schema_name,
            expected=PROTOEM_PHASE5_REFERENCE_BUNDLE_SCHEMA_NAME,
            field_name="schema_name",
        )
        _require_schema_version(
            self.schema_version,
            expected=PROTOEM_PHASE5_REFERENCE_BUNDLE_SCHEMA_VERSION,
            field_name="schema_version",
        )


def protoem_support_provenance_to_dict(
    provenance: ProtoEMSupportProvenance,
) -> dict[str, str | int | None]:
    """Convert one support provenance record to a canonical mapping."""

    return {
        "schema_name": provenance.schema_name,
        "schema_version": provenance.schema_version,
        "support_identifier": provenance.support_identifier,
        "support_patient_id": provenance.support_patient_id,
        "support_case_id": provenance.support_case_id,
        "source_input_identity": provenance.source_input_identity,
        "support_manifest_hash": provenance.support_manifest_hash,
        "support_assignment_index": provenance.support_assignment_index,
        "dataset_manifest_hash": provenance.dataset_manifest_hash,
        "encoder_identity": provenance.encoder_identity,
        "preprocessing_hash": provenance.preprocessing_hash,
        "checkpoint_hash": provenance.checkpoint_hash,
        "feature_stage": provenance.feature_stage,
        "normalization_name": provenance.normalization_name,
    }


def protoem_phase5_artifact_reference_to_dict(
    reference: ProtoEMPhase5ArtifactReference,
) -> dict[str, str]:
    """Convert one Phase 5 artifact reference to a canonical mapping."""

    return {
        "schema_name": reference.schema_name,
        "schema_version": reference.schema_version,
        "artifact_schema_name": reference.artifact_schema_name,
        "artifact_hash": reference.artifact_hash,
    }


def query_label_not_part_of_initialization_api() -> bool:
    """Return True when the public ProtoEM initialization API excludes labels and masks."""

    return all(
        "label" not in parameter.name and "mask" not in parameter.name
        for parameter in inspect.signature(_initialization_api_signature_target).parameters.values()
    )


def _initialization_api_signature_target(
    *,
    config: object,
    query_input: object,
    prototype_input: object,
    inference_input: object,
    phase5_references: object,
) -> object:
    return config, query_input, prototype_input, inference_input, phase5_references


def _require_schema_name(value: str, *, expected: str, field_name: str) -> None:
    if value != expected:
        raise ProtoEMContractValidationError(f"{field_name} must equal {expected!r}.")


def _require_schema_version(value: str, *, expected: str, field_name: str) -> None:
    if value != expected:
        raise ProtoEMContractValidationError(f"{field_name} must equal {expected!r}.")


def _require_sha256(value: str, *, field_name: str) -> None:
    if not _SHA256_RE.fullmatch(value):
        raise ProtoEMContractValidationError(
            f"{field_name} must be a lowercase 64-character SHA-256 hex digest."
        )


def _require_optional_sha256(value: str | None, *, field_name: str) -> None:
    if value is None:
        return
    _require_sha256(value, field_name=field_name)


def _require_identifier(value: str, *, field_name: str) -> None:
    if not _IDENTIFIER_RE.fullmatch(value):
        raise ProtoEMContractValidationError(
            f"{field_name} must match the conservative identifier pattern."
        )


def _require_nonnegative_int(value: int, *, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ProtoEMContractValidationError(f"{field_name} must be a nonnegative integer.")


def _require_finite_float(value: float, *, field_name: str) -> None:
    if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(value):
        raise ProtoEMContractValidationError(f"{field_name} must be a finite number.")


__all__ = [
    "PROTOEM_INFERENCE_INPUT_SCHEMA_NAME",
    "PROTOEM_INFERENCE_INPUT_SCHEMA_VERSION",
    "PROTOEM_PHASE5_ARTIFACT_REFERENCE_SCHEMA_NAME",
    "PROTOEM_PHASE5_ARTIFACT_REFERENCE_SCHEMA_VERSION",
    "PROTOEM_PHASE5_REFERENCE_BUNDLE_SCHEMA_NAME",
    "PROTOEM_PHASE5_REFERENCE_BUNDLE_SCHEMA_VERSION",
    "PROTOEM_PROTOTYPE_INPUT_SCHEMA_NAME",
    "PROTOEM_PROTOTYPE_INPUT_SCHEMA_VERSION",
    "PROTOEM_QUERY_FEATURE_INPUT_SCHEMA_NAME",
    "PROTOEM_QUERY_FEATURE_INPUT_SCHEMA_VERSION",
    "PROTOEM_SUPPORT_PROVENANCE_SCHEMA_NAME",
    "PROTOEM_SUPPORT_PROVENANCE_SCHEMA_VERSION",
    "ProtoEMContractError",
    "ProtoEMContractValidationError",
    "ProtoEMInferenceInput",
    "ProtoEMPhase5ArtifactReference",
    "ProtoEMPhase5ReferenceBundle",
    "ProtoEMPrototypeInput",
    "ProtoEMQueryFeatureInput",
    "ProtoEMSupportProvenance",
    "protoem_phase5_artifact_reference_to_dict",
    "protoem_support_provenance_to_dict",
    "query_label_not_part_of_initialization_api",
]
