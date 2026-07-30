"""Deterministic Phase 4 few-shot artifact schemas and hashing contracts."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Final, TypeVar

from protoem_ct.artifacts.hashing import canonical_json_bytes, sha256_json

FEWSHOT_SUPPORT_MANIFEST_VERSION: Final[str] = "fewshot_support_manifest_v1"
FEWSHOT_ADAPTATION_CONFIG_VERSION: Final[str] = "fewshot_adaptation_config_v1"
FEWSHOT_PROTOCOL_TABLE_VERSION: Final[str] = "fewshot_protocol_table_v1"
FEWSHOT_RUN_SUMMARY_VERSION: Final[str] = "fewshot_run_summary_v1"
SUPPORTED_FEWSHOT_SCHEMA_VERSIONS: Final[frozenset[str]] = frozenset(
    {
        FEWSHOT_SUPPORT_MANIFEST_VERSION,
        FEWSHOT_ADAPTATION_CONFIG_VERSION,
        FEWSHOT_PROTOCOL_TABLE_VERSION,
        FEWSHOT_RUN_SUMMARY_VERSION,
    }
)
SUPPORTED_FEWSHOT_K_VALUES: Final[frozenset[int]] = frozenset({1, 2, 5, 10, 20})
SUPPORTED_ADAPTATION_MODES: Final[frozenset[str]] = frozenset(
    {"head_only", "decoder_only", "full_finetune"}
)
SUPPORTED_BASELINE_FAMILIES: Final[frozenset[str]] = frozenset({"monai_segresnet", "nnunet_v2"})
SUPPORTED_STRATIFICATION_STATUSES: Final[frozenset[str]] = frozenset(
    {"not_requested", "stratified", "fallback_unstratified"}
)
SUPPORTED_INITIALIZATION_REFERENCE_TYPES: Final[frozenset[str]] = frozenset(
    {"baseline_provenance", "baseline_checkpoint", "model_init"}
)
SUPPORTED_PROTOCOL_RUN_STATUSES: Final[frozenset[str]] = frozenset(
    {"planned", "running", "completed", "failed"}
)
SUPPORTED_MEMORY_AVAILABILITY_STATUSES: Final[frozenset[str]] = frozenset(
    {"available", "unavailable", "not_recorded"}
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_IDENTIFIER_RE = re.compile(r"^[a-z0-9](?:[a-z0-9_.-]{0,62}[a-z0-9])?$")
_SAFE_MESSAGE_RE = re.compile(r"^[^\r\n]+$")
_ALLOWED_SUPPORT_ASSIGNMENT_FIELDS: Final[frozenset[str]] = frozenset(
    {"anonymous_case_id", "anonymous_patient_id"}
)
_ALLOWED_INITIALIZATION_REFERENCE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "artifact_sha256",
        "checkpoint_sha256",
        "reference_identifier",
        "reference_type",
    }
)
_ALLOWED_SUPPORT_MANIFEST_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "artifact_hash",
        "assignments",
        "immutable_test_cohort_hash",
        "manifest_id",
        "replicate_id",
        "requested_k",
        "schema_version",
        "selection_policy_version",
        "selection_seed",
        "source_development_manifest_hash",
        "source_development_split_hash",
        "source_lesion_summary_hash",
        "stratification_reason",
        "stratification_status",
    }
)
_ALLOWED_ADAPTATION_CONFIG_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "adaptation_mode",
        "amp_enabled",
        "artifact_hash",
        "baseline_family",
        "gradient_accumulation_steps",
        "initialization_reference",
        "learning_rate",
        "max_adaptation_steps",
        "optimizer_name",
        "schema_version",
        "seed",
        "support_batch_size",
        "weight_decay",
    }
)
_ALLOWED_PROTOCOL_ROW_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "adaptation_config_hash",
        "adaptation_mode",
        "initialization_reference",
        "planned_output_identifier",
        "replicate_id",
        "requested_k",
        "run_status",
        "support_manifest_hash",
    }
)
_ALLOWED_PROTOCOL_TABLE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "artifact_hash",
        "immutable_test_cohort_hash",
        "rows",
        "schema_version",
        "source_development_manifest_hash",
        "source_development_split_hash",
        "source_lesion_summary_hash",
    }
)
_ALLOWED_METRIC_REFERENCE_FIELDS: Final[frozenset[str]] = frozenset(
    {"artifact_role", "artifact_sha256"}
)
_ALLOWED_RUN_SUMMARY_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "adaptation_config_hash",
        "artifact_hash",
        "duration_seconds",
        "failure_code",
        "failure_message",
        "initialization_reference",
        "memory_availability_status",
        "metric_artifact_references",
        "peak_allocated_memory_bytes",
        "peak_host_memory_bytes",
        "peak_reserved_memory_bytes",
        "run_status",
        "schema_version",
        "support_manifest_hash",
        "trainable_parameter_count",
    }
)
_ZERO_SHA256 = "0" * 64
_ArtifactType = TypeVar(
    "_ArtifactType",
    "FewshotSupportManifest",
    "FewshotAdaptationConfig",
    "FewshotProtocolTable",
    "FewshotRunSummary",
)


class FewshotArtifactError(ValueError):
    """Base error for Phase 4 few-shot artifact contracts."""


class FewshotArtifactValidationError(FewshotArtifactError):
    """Raised when few-shot artifact content violates the contract."""


class FewshotArtifactSerializationError(FewshotArtifactError):
    """Raised when few-shot artifact JSON cannot be parsed safely."""


class FewshotArtifactVersionError(FewshotArtifactError):
    """Raised when a few-shot artifact uses an unsupported schema version."""


class FewshotArtifactHashError(FewshotArtifactError):
    """Raised when an artifact hash does not match the deterministic content."""


@dataclass(frozen=True, slots=True)
class FewshotSupportAssignment:
    """One ordered anonymous support patient/case assignment."""

    anonymous_patient_id: str
    anonymous_case_id: str

    def __post_init__(self) -> None:
        _require_safe_identifier(self.anonymous_patient_id, field_name="anonymous_patient_id")
        _require_safe_identifier(self.anonymous_case_id, field_name="anonymous_case_id")
        if self.anonymous_patient_id == self.anonymous_case_id:
            raise FewshotArtifactValidationError(
                "anonymous_patient_id and anonymous_case_id must differ."
            )


@dataclass(frozen=True, slots=True)
class FewshotInitializationReference:
    """Deterministic initialization or checkpoint provenance reference."""

    reference_type: str
    reference_identifier: str
    artifact_sha256: str | None = None
    checkpoint_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.reference_type not in SUPPORTED_INITIALIZATION_REFERENCE_TYPES:
            raise FewshotArtifactValidationError(
                f"Unsupported initialization reference_type: {self.reference_type!r}."
            )
        _require_safe_identifier(self.reference_identifier, field_name="reference_identifier")
        _require_optional_sha256(self.artifact_sha256, field_name="artifact_sha256")
        _require_optional_sha256(self.checkpoint_sha256, field_name="checkpoint_sha256")
        if self.reference_type == "baseline_checkpoint" and self.checkpoint_sha256 is None:
            raise FewshotArtifactValidationError(
                "baseline_checkpoint references require checkpoint_sha256."
            )
        if self.reference_type == "baseline_provenance" and self.artifact_sha256 is None:
            raise FewshotArtifactValidationError(
                "baseline_provenance references require artifact_sha256."
            )


@dataclass(frozen=True, slots=True)
class FewshotSupportManifest:
    """Deterministic Phase 4 support-manifest artifact."""

    artifact_hash: str
    schema_version: str
    manifest_id: str
    requested_k: int
    replicate_id: str
    selection_seed: int
    selection_policy_version: str
    stratification_status: str
    stratification_reason: str | None
    source_development_manifest_hash: str
    source_development_split_hash: str
    source_lesion_summary_hash: str | None
    immutable_test_cohort_hash: str
    assignments: tuple[FewshotSupportAssignment, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version, expected=FEWSHOT_SUPPORT_MANIFEST_VERSION)
        _require_sha256(self.artifact_hash, field_name="artifact_hash")
        _require_safe_identifier(self.manifest_id, field_name="manifest_id")
        _require_k(self.requested_k)
        _require_safe_identifier(self.replicate_id, field_name="replicate_id")
        _require_seed(self.selection_seed, field_name="selection_seed")
        _require_safe_identifier(
            self.selection_policy_version,
            field_name="selection_policy_version",
        )
        if self.stratification_status not in SUPPORTED_STRATIFICATION_STATUSES:
            raise FewshotArtifactValidationError(
                f"Unsupported stratification_status: {self.stratification_status!r}."
            )
        _require_optional_reason(self.stratification_reason, field_name="stratification_reason")
        _require_stratification_reason_constraints(
            status=self.stratification_status,
            reason=self.stratification_reason,
        )
        _require_sha256(
            self.source_development_manifest_hash,
            field_name="source_development_manifest_hash",
        )
        _require_sha256(
            self.source_development_split_hash,
            field_name="source_development_split_hash",
        )
        _require_optional_sha256(
            self.source_lesion_summary_hash,
            field_name="source_lesion_summary_hash",
        )
        _require_sha256(self.immutable_test_cohort_hash, field_name="immutable_test_cohort_hash")
        normalized_assignments = _normalize_support_assignments(
            self.assignments,
            requested_k=self.requested_k,
        )
        object.__setattr__(self, "assignments", normalized_assignments)
        expected_hash = hash_fewshot_support_manifest(self)
        if self.artifact_hash != expected_hash:
            raise FewshotArtifactHashError(
                "artifact_hash does not match the deterministic support-manifest content."
            )


@dataclass(frozen=True, slots=True)
class FewshotAdaptationConfig:
    """Deterministic Phase 4 adaptation-configuration artifact."""

    artifact_hash: str
    schema_version: str
    adaptation_mode: str
    seed: int
    baseline_family: str
    initialization_reference: FewshotInitializationReference
    optimizer_name: str
    learning_rate: float
    weight_decay: float
    max_adaptation_steps: int
    support_batch_size: int
    gradient_accumulation_steps: int
    amp_enabled: bool

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version, expected=FEWSHOT_ADAPTATION_CONFIG_VERSION)
        _require_sha256(self.artifact_hash, field_name="artifact_hash")
        if self.adaptation_mode not in SUPPORTED_ADAPTATION_MODES:
            raise FewshotArtifactValidationError(
                f"Unsupported adaptation_mode: {self.adaptation_mode!r}."
            )
        _require_seed(self.seed, field_name="seed")
        if self.baseline_family not in SUPPORTED_BASELINE_FAMILIES:
            raise FewshotArtifactValidationError(
                f"Unsupported baseline_family: {self.baseline_family!r}."
            )
        _require_safe_identifier(self.optimizer_name, field_name="optimizer_name")
        _require_positive_finite_float(self.learning_rate, field_name="learning_rate")
        _require_nonnegative_finite_float(self.weight_decay, field_name="weight_decay")
        _require_positive_int(self.max_adaptation_steps, field_name="max_adaptation_steps")
        _require_positive_int(self.support_batch_size, field_name="support_batch_size")
        _require_positive_int(
            self.gradient_accumulation_steps,
            field_name="gradient_accumulation_steps",
        )
        expected_hash = hash_fewshot_adaptation_config(self)
        if self.artifact_hash != expected_hash:
            raise FewshotArtifactHashError(
                "artifact_hash does not match the deterministic adaptation-config content."
            )


@dataclass(frozen=True, slots=True)
class FewshotProtocolTableRow:
    """One deterministic protocol-table row."""

    requested_k: int
    replicate_id: str
    support_manifest_hash: str
    adaptation_mode: str
    adaptation_config_hash: str
    initialization_reference: FewshotInitializationReference
    planned_output_identifier: str
    run_status: str

    def __post_init__(self) -> None:
        _require_k(self.requested_k)
        _require_safe_identifier(self.replicate_id, field_name="replicate_id")
        _require_sha256(self.support_manifest_hash, field_name="support_manifest_hash")
        if self.adaptation_mode not in SUPPORTED_ADAPTATION_MODES:
            raise FewshotArtifactValidationError(
                f"Unsupported adaptation_mode: {self.adaptation_mode!r}."
            )
        _require_sha256(self.adaptation_config_hash, field_name="adaptation_config_hash")
        _require_safe_identifier(
            self.planned_output_identifier,
            field_name="planned_output_identifier",
        )
        if self.run_status not in SUPPORTED_PROTOCOL_RUN_STATUSES:
            raise FewshotArtifactValidationError(f"Unsupported run_status: {self.run_status!r}.")


@dataclass(frozen=True, slots=True)
class FewshotProtocolTable:
    """Deterministic Phase 4 protocol-table artifact."""

    artifact_hash: str
    schema_version: str
    source_development_manifest_hash: str
    source_development_split_hash: str
    source_lesion_summary_hash: str | None
    immutable_test_cohort_hash: str
    rows: tuple[FewshotProtocolTableRow, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version, expected=FEWSHOT_PROTOCOL_TABLE_VERSION)
        _require_sha256(self.artifact_hash, field_name="artifact_hash")
        _require_sha256(
            self.source_development_manifest_hash,
            field_name="source_development_manifest_hash",
        )
        _require_sha256(
            self.source_development_split_hash,
            field_name="source_development_split_hash",
        )
        _require_optional_sha256(
            self.source_lesion_summary_hash,
            field_name="source_lesion_summary_hash",
        )
        _require_sha256(self.immutable_test_cohort_hash, field_name="immutable_test_cohort_hash")
        normalized_rows = _normalize_protocol_rows(self.rows)
        object.__setattr__(self, "rows", normalized_rows)
        expected_hash = hash_fewshot_protocol_table(self)
        if self.artifact_hash != expected_hash:
            raise FewshotArtifactHashError(
                "artifact_hash does not match the deterministic protocol-table content."
            )


@dataclass(frozen=True, slots=True)
class FewshotMetricArtifactReference:
    """Hash-only link to one persisted metric-related artifact."""

    artifact_role: str
    artifact_sha256: str

    def __post_init__(self) -> None:
        _require_safe_identifier(self.artifact_role, field_name="artifact_role")
        _require_sha256(self.artifact_sha256, field_name="artifact_sha256")


@dataclass(frozen=True, slots=True)
class FewshotRunSummary:
    """Deterministic Phase 4 run-summary artifact."""

    artifact_hash: str
    schema_version: str
    support_manifest_hash: str
    adaptation_config_hash: str
    initialization_reference: FewshotInitializationReference
    run_status: str
    trainable_parameter_count: int
    duration_seconds: float | None
    memory_availability_status: str
    peak_allocated_memory_bytes: int | None
    peak_reserved_memory_bytes: int | None
    peak_host_memory_bytes: int | None
    metric_artifact_references: tuple[FewshotMetricArtifactReference, ...] = field(
        default_factory=tuple
    )
    failure_code: str | None = None
    failure_message: str | None = None

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version, expected=FEWSHOT_RUN_SUMMARY_VERSION)
        _require_sha256(self.artifact_hash, field_name="artifact_hash")
        _require_sha256(self.support_manifest_hash, field_name="support_manifest_hash")
        _require_sha256(self.adaptation_config_hash, field_name="adaptation_config_hash")
        if self.run_status not in SUPPORTED_PROTOCOL_RUN_STATUSES:
            raise FewshotArtifactValidationError(f"Unsupported run_status: {self.run_status!r}.")
        _require_nonnegative_int(
            self.trainable_parameter_count,
            field_name="trainable_parameter_count",
        )
        _require_optional_nonnegative_finite_float(
            self.duration_seconds,
            field_name="duration_seconds",
        )
        if self.memory_availability_status not in SUPPORTED_MEMORY_AVAILABILITY_STATUSES:
            raise FewshotArtifactValidationError(
                "memory_availability_status must be one of "
                f"{sorted(SUPPORTED_MEMORY_AVAILABILITY_STATUSES)!r}."
            )
        _require_optional_nonnegative_int(
            self.peak_allocated_memory_bytes,
            field_name="peak_allocated_memory_bytes",
        )
        _require_optional_nonnegative_int(
            self.peak_reserved_memory_bytes,
            field_name="peak_reserved_memory_bytes",
        )
        _require_optional_nonnegative_int(
            self.peak_host_memory_bytes,
            field_name="peak_host_memory_bytes",
        )
        _require_memory_constraints(self)
        normalized_metric_references = _normalize_metric_artifact_references(
            self.metric_artifact_references
        )
        object.__setattr__(self, "metric_artifact_references", normalized_metric_references)
        _require_optional_safe_identifier(self.failure_code, field_name="failure_code")
        _require_optional_reason(self.failure_message, field_name="failure_message")
        _require_failure_constraints(
            run_status=self.run_status,
            failure_code=self.failure_code,
            failure_message=self.failure_message,
        )
        expected_hash = hash_fewshot_run_summary(self)
        if self.artifact_hash != expected_hash:
            raise FewshotArtifactHashError(
                "artifact_hash does not match the deterministic run-summary content."
            )


FewshotArtifact = (
    FewshotSupportManifest | FewshotAdaptationConfig | FewshotProtocolTable | FewshotRunSummary
)


def hash_fewshot_support_manifest(artifact: FewshotSupportManifest) -> str:
    """Return the deterministic SHA-256 hash for one support-manifest artifact."""

    return sha256_json(fewshot_support_manifest_hash_payload(artifact))


def hash_fewshot_adaptation_config(artifact: FewshotAdaptationConfig) -> str:
    """Return the deterministic SHA-256 hash for one adaptation-config artifact."""

    return sha256_json(fewshot_adaptation_config_hash_payload(artifact))


def hash_fewshot_protocol_table(artifact: FewshotProtocolTable) -> str:
    """Return the deterministic SHA-256 hash for one protocol-table artifact."""

    return sha256_json(fewshot_protocol_table_hash_payload(artifact))


def hash_fewshot_run_summary(artifact: FewshotRunSummary) -> str:
    """Return the deterministic SHA-256 hash for one run-summary artifact."""

    return sha256_json(fewshot_run_summary_hash_payload(artifact))


def fewshot_support_manifest_hash_payload(
    artifact: FewshotSupportManifest,
) -> dict[str, object]:
    """Return the support-manifest hash payload excluding ``artifact_hash``."""

    payload = fewshot_support_manifest_to_dict(artifact)
    payload.pop("artifact_hash")
    return payload


def fewshot_adaptation_config_hash_payload(
    artifact: FewshotAdaptationConfig,
) -> dict[str, object]:
    """Return the adaptation-config hash payload excluding ``artifact_hash``."""

    payload = fewshot_adaptation_config_to_dict(artifact)
    payload.pop("artifact_hash")
    return payload


def fewshot_protocol_table_hash_payload(artifact: FewshotProtocolTable) -> dict[str, object]:
    """Return the protocol-table hash payload excluding ``artifact_hash``."""

    payload = fewshot_protocol_table_to_dict(artifact)
    payload.pop("artifact_hash")
    return payload


def fewshot_run_summary_hash_payload(artifact: FewshotRunSummary) -> dict[str, object]:
    """Return the run-summary hash payload excluding ``artifact_hash``."""

    payload = fewshot_run_summary_to_dict(artifact)
    payload.pop("artifact_hash")
    return payload


def fewshot_support_manifest_to_dict(artifact: FewshotSupportManifest) -> dict[str, object]:
    """Convert one support-manifest artifact to a JSON-compatible mapping."""

    return {
        "artifact_hash": artifact.artifact_hash,
        "schema_version": artifact.schema_version,
        "manifest_id": artifact.manifest_id,
        "requested_k": artifact.requested_k,
        "replicate_id": artifact.replicate_id,
        "selection_seed": artifact.selection_seed,
        "selection_policy_version": artifact.selection_policy_version,
        "stratification_status": artifact.stratification_status,
        "stratification_reason": artifact.stratification_reason,
        "source_development_manifest_hash": artifact.source_development_manifest_hash,
        "source_development_split_hash": artifact.source_development_split_hash,
        "source_lesion_summary_hash": artifact.source_lesion_summary_hash,
        "immutable_test_cohort_hash": artifact.immutable_test_cohort_hash,
        "assignments": [
            {
                "anonymous_patient_id": assignment.anonymous_patient_id,
                "anonymous_case_id": assignment.anonymous_case_id,
            }
            for assignment in artifact.assignments
        ],
    }


def fewshot_adaptation_config_to_dict(artifact: FewshotAdaptationConfig) -> dict[str, object]:
    """Convert one adaptation-config artifact to a JSON-compatible mapping."""

    return {
        "artifact_hash": artifact.artifact_hash,
        "schema_version": artifact.schema_version,
        "adaptation_mode": artifact.adaptation_mode,
        "seed": artifact.seed,
        "baseline_family": artifact.baseline_family,
        "initialization_reference": _initialization_reference_to_dict(
            artifact.initialization_reference
        ),
        "optimizer_name": artifact.optimizer_name,
        "learning_rate": artifact.learning_rate,
        "weight_decay": artifact.weight_decay,
        "max_adaptation_steps": artifact.max_adaptation_steps,
        "support_batch_size": artifact.support_batch_size,
        "gradient_accumulation_steps": artifact.gradient_accumulation_steps,
        "amp_enabled": artifact.amp_enabled,
    }


def fewshot_protocol_table_to_dict(artifact: FewshotProtocolTable) -> dict[str, object]:
    """Convert one protocol-table artifact to a JSON-compatible mapping."""

    return {
        "artifact_hash": artifact.artifact_hash,
        "schema_version": artifact.schema_version,
        "source_development_manifest_hash": artifact.source_development_manifest_hash,
        "source_development_split_hash": artifact.source_development_split_hash,
        "source_lesion_summary_hash": artifact.source_lesion_summary_hash,
        "immutable_test_cohort_hash": artifact.immutable_test_cohort_hash,
        "rows": [
            {
                "requested_k": row.requested_k,
                "replicate_id": row.replicate_id,
                "support_manifest_hash": row.support_manifest_hash,
                "adaptation_mode": row.adaptation_mode,
                "adaptation_config_hash": row.adaptation_config_hash,
                "initialization_reference": _initialization_reference_to_dict(
                    row.initialization_reference
                ),
                "planned_output_identifier": row.planned_output_identifier,
                "run_status": row.run_status,
            }
            for row in artifact.rows
        ],
    }


def fewshot_run_summary_to_dict(artifact: FewshotRunSummary) -> dict[str, object]:
    """Convert one run-summary artifact to a JSON-compatible mapping."""

    return {
        "artifact_hash": artifact.artifact_hash,
        "schema_version": artifact.schema_version,
        "support_manifest_hash": artifact.support_manifest_hash,
        "adaptation_config_hash": artifact.adaptation_config_hash,
        "initialization_reference": _initialization_reference_to_dict(
            artifact.initialization_reference
        ),
        "run_status": artifact.run_status,
        "trainable_parameter_count": artifact.trainable_parameter_count,
        "duration_seconds": artifact.duration_seconds,
        "memory_availability_status": artifact.memory_availability_status,
        "peak_allocated_memory_bytes": artifact.peak_allocated_memory_bytes,
        "peak_reserved_memory_bytes": artifact.peak_reserved_memory_bytes,
        "peak_host_memory_bytes": artifact.peak_host_memory_bytes,
        "metric_artifact_references": [
            {
                "artifact_role": item.artifact_role,
                "artifact_sha256": item.artifact_sha256,
            }
            for item in artifact.metric_artifact_references
        ],
        "failure_code": artifact.failure_code,
        "failure_message": artifact.failure_message,
    }


def fewshot_support_manifest_to_json(artifact: FewshotSupportManifest) -> bytes:
    """Serialize one support-manifest artifact deterministically."""

    return canonical_json_bytes(fewshot_support_manifest_to_dict(artifact)) + b"\n"


def fewshot_adaptation_config_to_json(artifact: FewshotAdaptationConfig) -> bytes:
    """Serialize one adaptation-config artifact deterministically."""

    return canonical_json_bytes(fewshot_adaptation_config_to_dict(artifact)) + b"\n"


def fewshot_protocol_table_to_json(artifact: FewshotProtocolTable) -> bytes:
    """Serialize one protocol-table artifact deterministically."""

    return canonical_json_bytes(fewshot_protocol_table_to_dict(artifact)) + b"\n"


def fewshot_run_summary_to_json(artifact: FewshotRunSummary) -> bytes:
    """Serialize one run-summary artifact deterministically."""

    return canonical_json_bytes(fewshot_run_summary_to_dict(artifact)) + b"\n"


def fewshot_artifact_to_dict(artifact: FewshotArtifact) -> dict[str, object]:
    """Convert one few-shot artifact to a JSON-compatible mapping."""

    if isinstance(artifact, FewshotSupportManifest):
        return fewshot_support_manifest_to_dict(artifact)
    if isinstance(artifact, FewshotAdaptationConfig):
        return fewshot_adaptation_config_to_dict(artifact)
    if isinstance(artifact, FewshotProtocolTable):
        return fewshot_protocol_table_to_dict(artifact)
    if isinstance(artifact, FewshotRunSummary):
        return fewshot_run_summary_to_dict(artifact)
    raise FewshotArtifactSerializationError(
        f"Unsupported few-shot artifact type: {type(artifact).__name__}."
    )


def fewshot_artifact_to_json(artifact: FewshotArtifact) -> bytes:
    """Serialize one few-shot artifact deterministically."""

    return canonical_json_bytes(fewshot_artifact_to_dict(artifact)) + b"\n"


def fewshot_support_manifest_from_json(payload: bytes | str) -> FewshotSupportManifest:
    """Parse and validate one support-manifest artifact."""

    parsed = _parse_json_object(payload, context="few-shot support manifest")
    _require_exact_fields(
        parsed,
        expected=_ALLOWED_SUPPORT_MANIFEST_FIELDS,
        context="few-shot support manifest",
    )
    return FewshotSupportManifest(
        artifact_hash=_expect_string(parsed["artifact_hash"], field_name="artifact_hash"),
        schema_version=_expect_string(parsed["schema_version"], field_name="schema_version"),
        manifest_id=_expect_string(parsed["manifest_id"], field_name="manifest_id"),
        requested_k=_expect_int(parsed["requested_k"], field_name="requested_k"),
        replicate_id=_expect_string(parsed["replicate_id"], field_name="replicate_id"),
        selection_seed=_expect_int(parsed["selection_seed"], field_name="selection_seed"),
        selection_policy_version=_expect_string(
            parsed["selection_policy_version"],
            field_name="selection_policy_version",
        ),
        stratification_status=_expect_string(
            parsed["stratification_status"],
            field_name="stratification_status",
        ),
        stratification_reason=_expect_optional_string(
            parsed["stratification_reason"],
            field_name="stratification_reason",
        ),
        source_development_manifest_hash=_expect_string(
            parsed["source_development_manifest_hash"],
            field_name="source_development_manifest_hash",
        ),
        source_development_split_hash=_expect_string(
            parsed["source_development_split_hash"],
            field_name="source_development_split_hash",
        ),
        source_lesion_summary_hash=_expect_optional_string(
            parsed["source_lesion_summary_hash"],
            field_name="source_lesion_summary_hash",
        ),
        immutable_test_cohort_hash=_expect_string(
            parsed["immutable_test_cohort_hash"],
            field_name="immutable_test_cohort_hash",
        ),
        assignments=_parse_support_assignments(parsed["assignments"]),
    )


def fewshot_adaptation_config_from_json(payload: bytes | str) -> FewshotAdaptationConfig:
    """Parse and validate one adaptation-config artifact."""

    parsed = _parse_json_object(payload, context="few-shot adaptation config")
    _require_exact_fields(
        parsed,
        expected=_ALLOWED_ADAPTATION_CONFIG_FIELDS,
        context="few-shot adaptation config",
    )
    return FewshotAdaptationConfig(
        artifact_hash=_expect_string(parsed["artifact_hash"], field_name="artifact_hash"),
        schema_version=_expect_string(parsed["schema_version"], field_name="schema_version"),
        adaptation_mode=_expect_string(
            parsed["adaptation_mode"],
            field_name="adaptation_mode",
        ),
        seed=_expect_int(parsed["seed"], field_name="seed"),
        baseline_family=_expect_string(parsed["baseline_family"], field_name="baseline_family"),
        initialization_reference=_parse_initialization_reference(
            parsed["initialization_reference"],
            context="few-shot adaptation config.initialization_reference",
        ),
        optimizer_name=_expect_string(parsed["optimizer_name"], field_name="optimizer_name"),
        learning_rate=_expect_float(parsed["learning_rate"], field_name="learning_rate"),
        weight_decay=_expect_float(parsed["weight_decay"], field_name="weight_decay"),
        max_adaptation_steps=_expect_int(
            parsed["max_adaptation_steps"],
            field_name="max_adaptation_steps",
        ),
        support_batch_size=_expect_int(
            parsed["support_batch_size"],
            field_name="support_batch_size",
        ),
        gradient_accumulation_steps=_expect_int(
            parsed["gradient_accumulation_steps"],
            field_name="gradient_accumulation_steps",
        ),
        amp_enabled=_expect_bool(parsed["amp_enabled"], field_name="amp_enabled"),
    )


def fewshot_protocol_table_from_json(payload: bytes | str) -> FewshotProtocolTable:
    """Parse and validate one protocol-table artifact."""

    parsed = _parse_json_object(payload, context="few-shot protocol table")
    _require_exact_fields(
        parsed,
        expected=_ALLOWED_PROTOCOL_TABLE_FIELDS,
        context="few-shot protocol table",
    )
    return FewshotProtocolTable(
        artifact_hash=_expect_string(parsed["artifact_hash"], field_name="artifact_hash"),
        schema_version=_expect_string(parsed["schema_version"], field_name="schema_version"),
        source_development_manifest_hash=_expect_string(
            parsed["source_development_manifest_hash"],
            field_name="source_development_manifest_hash",
        ),
        source_development_split_hash=_expect_string(
            parsed["source_development_split_hash"],
            field_name="source_development_split_hash",
        ),
        source_lesion_summary_hash=_expect_optional_string(
            parsed["source_lesion_summary_hash"],
            field_name="source_lesion_summary_hash",
        ),
        immutable_test_cohort_hash=_expect_string(
            parsed["immutable_test_cohort_hash"],
            field_name="immutable_test_cohort_hash",
        ),
        rows=_parse_protocol_rows(parsed["rows"]),
    )


def fewshot_run_summary_from_json(payload: bytes | str) -> FewshotRunSummary:
    """Parse and validate one run-summary artifact."""

    parsed = _parse_json_object(payload, context="few-shot run summary")
    _require_exact_fields(
        parsed,
        expected=_ALLOWED_RUN_SUMMARY_FIELDS,
        context="few-shot run summary",
    )
    return FewshotRunSummary(
        artifact_hash=_expect_string(parsed["artifact_hash"], field_name="artifact_hash"),
        schema_version=_expect_string(parsed["schema_version"], field_name="schema_version"),
        support_manifest_hash=_expect_string(
            parsed["support_manifest_hash"],
            field_name="support_manifest_hash",
        ),
        adaptation_config_hash=_expect_string(
            parsed["adaptation_config_hash"],
            field_name="adaptation_config_hash",
        ),
        initialization_reference=_parse_initialization_reference(
            parsed["initialization_reference"],
            context="few-shot run summary.initialization_reference",
        ),
        run_status=_expect_string(parsed["run_status"], field_name="run_status"),
        trainable_parameter_count=_expect_int(
            parsed["trainable_parameter_count"],
            field_name="trainable_parameter_count",
        ),
        duration_seconds=_expect_optional_float(
            parsed["duration_seconds"],
            field_name="duration_seconds",
        ),
        memory_availability_status=_expect_string(
            parsed["memory_availability_status"],
            field_name="memory_availability_status",
        ),
        peak_allocated_memory_bytes=_expect_optional_int(
            parsed["peak_allocated_memory_bytes"],
            field_name="peak_allocated_memory_bytes",
        ),
        peak_reserved_memory_bytes=_expect_optional_int(
            parsed["peak_reserved_memory_bytes"],
            field_name="peak_reserved_memory_bytes",
        ),
        peak_host_memory_bytes=_expect_optional_int(
            parsed["peak_host_memory_bytes"],
            field_name="peak_host_memory_bytes",
        ),
        metric_artifact_references=_parse_metric_artifact_references(
            parsed["metric_artifact_references"]
        ),
        failure_code=_expect_optional_string(parsed["failure_code"], field_name="failure_code"),
        failure_message=_expect_optional_string(
            parsed["failure_message"],
            field_name="failure_message",
        ),
    )


def fewshot_artifact_from_json(
    payload: bytes | str,
    artifact_type: type[_ArtifactType] | None = None,
) -> FewshotArtifact | _ArtifactType:
    """Parse and validate one few-shot artifact with optional type assertion."""

    parsed = _parse_json_object(payload, context="few-shot artifact")
    schema_version = _expect_string(parsed.get("schema_version"), field_name="schema_version")
    artifact: FewshotArtifact
    if schema_version == FEWSHOT_SUPPORT_MANIFEST_VERSION:
        artifact = fewshot_support_manifest_from_json(payload)
    elif schema_version == FEWSHOT_ADAPTATION_CONFIG_VERSION:
        artifact = fewshot_adaptation_config_from_json(payload)
    elif schema_version == FEWSHOT_PROTOCOL_TABLE_VERSION:
        artifact = fewshot_protocol_table_from_json(payload)
    elif schema_version == FEWSHOT_RUN_SUMMARY_VERSION:
        artifact = fewshot_run_summary_from_json(payload)
    else:
        raise FewshotArtifactVersionError(
            f"Unsupported few-shot schema_version: {schema_version!r}."
        )
    if artifact_type is not None and not isinstance(artifact, artifact_type):
        raise FewshotArtifactSerializationError(
            f"Expected {artifact_type.__name__}, got {type(artifact).__name__}."
        )
    return artifact


def _initialization_reference_to_dict(
    reference: FewshotInitializationReference,
) -> dict[str, object]:
    return {
        "reference_type": reference.reference_type,
        "reference_identifier": reference.reference_identifier,
        "artifact_sha256": reference.artifact_sha256,
        "checkpoint_sha256": reference.checkpoint_sha256,
    }


def _normalize_support_assignments(
    assignments: tuple[FewshotSupportAssignment, ...],
    *,
    requested_k: int,
) -> tuple[FewshotSupportAssignment, ...]:
    if len(assignments) != requested_k:
        raise FewshotArtifactValidationError("Support assignment count must equal requested_k.")
    patient_ids: set[str] = set()
    case_ids: set[str] = set()
    for assignment in assignments:
        if assignment.anonymous_patient_id in patient_ids:
            raise FewshotArtifactValidationError(
                "Support assignments must use unique anonymous_patient_id values."
            )
        if assignment.anonymous_case_id in case_ids:
            raise FewshotArtifactValidationError(
                "Support assignments must use unique anonymous_case_id values."
            )
        patient_ids.add(assignment.anonymous_patient_id)
        case_ids.add(assignment.anonymous_case_id)
    return assignments


def _normalize_protocol_rows(
    rows: tuple[FewshotProtocolTableRow, ...],
) -> tuple[FewshotProtocolTableRow, ...]:
    sorted_rows = tuple(sorted(rows, key=_protocol_row_sort_key))
    seen_keys: set[tuple[object, ...]] = set()
    for row in sorted_rows:
        key = _protocol_row_sort_key(row)
        if key in seen_keys:
            raise FewshotArtifactValidationError("Protocol table rows must be unique.")
        seen_keys.add(key)
    return sorted_rows


def _protocol_row_sort_key(row: FewshotProtocolTableRow) -> tuple[object, ...]:
    return (
        row.requested_k,
        row.replicate_id,
        row.adaptation_mode,
        row.planned_output_identifier,
        row.support_manifest_hash,
        row.adaptation_config_hash,
        row.initialization_reference.reference_type,
        row.initialization_reference.reference_identifier,
        row.run_status,
    )


def _normalize_metric_artifact_references(
    references: tuple[FewshotMetricArtifactReference, ...],
) -> tuple[FewshotMetricArtifactReference, ...]:
    seen_roles: set[str] = set()
    normalized = []
    for reference in sorted(
        references, key=lambda item: (item.artifact_role, item.artifact_sha256)
    ):
        if reference.artifact_role in seen_roles:
            raise FewshotArtifactValidationError(
                "metric_artifact_references must use unique artifact_role values."
            )
        seen_roles.add(reference.artifact_role)
        normalized.append(reference)
    return tuple(normalized)


def _require_schema_version(value: str, *, expected: str) -> None:
    if value != expected:
        raise FewshotArtifactVersionError(
            f"Unsupported few-shot schema_version: {value!r}; expected {expected!r}."
        )


def _require_k(value: int) -> None:
    if value not in SUPPORTED_FEWSHOT_K_VALUES:
        raise FewshotArtifactValidationError(
            f"requested_k must be one of {sorted(SUPPORTED_FEWSHOT_K_VALUES)!r}."
        )


def _require_seed(value: int, *, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise FewshotArtifactValidationError(f"{field_name} must be a nonnegative integer.")


def _require_positive_int(value: int, *, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise FewshotArtifactValidationError(f"{field_name} must be a positive integer.")


def _require_nonnegative_int(value: int, *, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise FewshotArtifactValidationError(f"{field_name} must be a nonnegative integer.")


def _require_optional_nonnegative_int(value: int | None, *, field_name: str) -> None:
    if value is not None:
        _require_nonnegative_int(value, field_name=field_name)


def _require_positive_finite_float(value: float, *, field_name: str) -> None:
    if not isinstance(value, float | int) or isinstance(value, bool) or not math.isfinite(value):
        raise FewshotArtifactValidationError(f"{field_name} must be a finite number.")
    if float(value) <= 0.0:
        raise FewshotArtifactValidationError(f"{field_name} must be strictly positive.")


def _require_nonnegative_finite_float(value: float, *, field_name: str) -> None:
    if not isinstance(value, float | int) or isinstance(value, bool) or not math.isfinite(value):
        raise FewshotArtifactValidationError(f"{field_name} must be a finite number.")
    if float(value) < 0.0:
        raise FewshotArtifactValidationError(f"{field_name} must be nonnegative.")


def _require_optional_nonnegative_finite_float(
    value: float | None,
    *,
    field_name: str,
) -> None:
    if value is not None:
        _require_nonnegative_finite_float(value, field_name=field_name)


def _require_sha256(value: str, *, field_name: str) -> None:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise FewshotArtifactValidationError(f"{field_name} must be a lowercase SHA-256 string.")


def _require_optional_sha256(value: str | None, *, field_name: str) -> None:
    if value is not None:
        _require_sha256(value, field_name=field_name)


def _require_safe_identifier(value: str, *, field_name: str) -> None:
    if not isinstance(value, str) or not _SAFE_IDENTIFIER_RE.fullmatch(value):
        raise FewshotArtifactValidationError(
            f"{field_name} must match the conservative identifier grammar."
        )


def _require_optional_safe_identifier(value: str | None, *, field_name: str) -> None:
    if value is not None:
        _require_safe_identifier(value, field_name=field_name)


def _require_optional_reason(value: str | None, *, field_name: str) -> None:
    if value is None:
        return
    if not isinstance(value, str) or not value.strip():
        raise FewshotArtifactValidationError(f"{field_name} must be null or a non-empty string.")
    if not _SAFE_MESSAGE_RE.fullmatch(value):
        raise FewshotArtifactValidationError(f"{field_name} must not contain newlines.")


def _require_stratification_reason_constraints(*, status: str, reason: str | None) -> None:
    if status == "fallback_unstratified" and reason is None:
        raise FewshotArtifactValidationError(
            "fallback_unstratified stratification_status requires stratification_reason."
        )


def _require_failure_constraints(
    *,
    run_status: str,
    failure_code: str | None,
    failure_message: str | None,
) -> None:
    if run_status == "failed":
        if failure_code is None and failure_message is None:
            raise FewshotArtifactValidationError(
                "failed run_status requires failure_code or failure_message."
            )
    elif failure_code is not None or failure_message is not None:
        raise FewshotArtifactValidationError(
            "failure_code and failure_message are allowed only when run_status is 'failed'."
        )


def _require_memory_constraints(summary: FewshotRunSummary) -> None:
    memory_values = (
        summary.peak_allocated_memory_bytes,
        summary.peak_reserved_memory_bytes,
        summary.peak_host_memory_bytes,
    )
    if summary.memory_availability_status == "available":
        if all(value is None for value in memory_values):
            raise FewshotArtifactValidationError(
                "available memory_availability_status requires at least one memory field."
            )
    else:
        if any(value is not None for value in memory_values):
            raise FewshotArtifactValidationError(
                "Unavailable or not_recorded memory status requires null memory fields."
            )


def _parse_json_object(payload: bytes | str, *, context: str) -> dict[str, Any]:
    text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
    try:
        parsed = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_bad_json_constant,
        )
    except Exception as error:
        raise FewshotArtifactSerializationError(
            f"Failed to parse {context} JSON: {error}"
        ) from error
    if not isinstance(parsed, dict):
        raise FewshotArtifactSerializationError(f"{context} JSON must encode an object.")
    return parsed


def _require_exact_fields(
    parsed: dict[str, Any],
    *,
    expected: frozenset[str],
    context: str,
) -> None:
    unknown_fields = set(parsed) - expected
    if unknown_fields:
        names = ", ".join(sorted(unknown_fields))
        raise FewshotArtifactSerializationError(f"Unknown {context} field(s): {names}.")
    missing_fields = expected - set(parsed)
    if missing_fields:
        names = ", ".join(sorted(missing_fields))
        raise FewshotArtifactSerializationError(f"Missing {context} field(s): {names}.")


def _parse_support_assignments(value: Any) -> tuple[FewshotSupportAssignment, ...]:
    items = _expect_list(value, field_name="assignments")
    assignments: list[FewshotSupportAssignment] = []
    for index, item in enumerate(items):
        parsed = _expect_object(item, field_name=f"assignments[{index}]")
        _require_exact_fields(
            parsed,
            expected=_ALLOWED_SUPPORT_ASSIGNMENT_FIELDS,
            context=f"assignments[{index}]",
        )
        assignments.append(
            FewshotSupportAssignment(
                anonymous_patient_id=_expect_string(
                    parsed["anonymous_patient_id"],
                    field_name=f"assignments[{index}].anonymous_patient_id",
                ),
                anonymous_case_id=_expect_string(
                    parsed["anonymous_case_id"],
                    field_name=f"assignments[{index}].anonymous_case_id",
                ),
            )
        )
    return tuple(assignments)


def _parse_initialization_reference(
    value: Any,
    *,
    context: str,
) -> FewshotInitializationReference:
    parsed = _expect_object(value, field_name=context)
    _require_exact_fields(
        parsed,
        expected=_ALLOWED_INITIALIZATION_REFERENCE_FIELDS,
        context=context,
    )
    return FewshotInitializationReference(
        reference_type=_expect_string(
            parsed["reference_type"],
            field_name=f"{context}.reference_type",
        ),
        reference_identifier=_expect_string(
            parsed["reference_identifier"],
            field_name=f"{context}.reference_identifier",
        ),
        artifact_sha256=_expect_optional_string(
            parsed["artifact_sha256"],
            field_name=f"{context}.artifact_sha256",
        ),
        checkpoint_sha256=_expect_optional_string(
            parsed["checkpoint_sha256"],
            field_name=f"{context}.checkpoint_sha256",
        ),
    )


def _parse_protocol_rows(value: Any) -> tuple[FewshotProtocolTableRow, ...]:
    items = _expect_list(value, field_name="rows")
    rows: list[FewshotProtocolTableRow] = []
    for index, item in enumerate(items):
        parsed = _expect_object(item, field_name=f"rows[{index}]")
        _require_exact_fields(
            parsed, expected=_ALLOWED_PROTOCOL_ROW_FIELDS, context=f"rows[{index}]"
        )
        rows.append(
            FewshotProtocolTableRow(
                requested_k=_expect_int(
                    parsed["requested_k"], field_name=f"rows[{index}].requested_k"
                ),
                replicate_id=_expect_string(
                    parsed["replicate_id"], field_name=f"rows[{index}].replicate_id"
                ),
                support_manifest_hash=_expect_string(
                    parsed["support_manifest_hash"],
                    field_name=f"rows[{index}].support_manifest_hash",
                ),
                adaptation_mode=_expect_string(
                    parsed["adaptation_mode"],
                    field_name=f"rows[{index}].adaptation_mode",
                ),
                adaptation_config_hash=_expect_string(
                    parsed["adaptation_config_hash"],
                    field_name=f"rows[{index}].adaptation_config_hash",
                ),
                initialization_reference=_parse_initialization_reference(
                    parsed["initialization_reference"],
                    context=f"rows[{index}].initialization_reference",
                ),
                planned_output_identifier=_expect_string(
                    parsed["planned_output_identifier"],
                    field_name=f"rows[{index}].planned_output_identifier",
                ),
                run_status=_expect_string(
                    parsed["run_status"], field_name=f"rows[{index}].run_status"
                ),
            )
        )
    return tuple(rows)


def _parse_metric_artifact_references(
    value: Any,
) -> tuple[FewshotMetricArtifactReference, ...]:
    items = _expect_list(value, field_name="metric_artifact_references")
    references: list[FewshotMetricArtifactReference] = []
    for index, item in enumerate(items):
        parsed = _expect_object(item, field_name=f"metric_artifact_references[{index}]")
        _require_exact_fields(
            parsed,
            expected=_ALLOWED_METRIC_REFERENCE_FIELDS,
            context=f"metric_artifact_references[{index}]",
        )
        references.append(
            FewshotMetricArtifactReference(
                artifact_role=_expect_string(
                    parsed["artifact_role"],
                    field_name=f"metric_artifact_references[{index}].artifact_role",
                ),
                artifact_sha256=_expect_string(
                    parsed["artifact_sha256"],
                    field_name=f"metric_artifact_references[{index}].artifact_sha256",
                ),
            )
        )
    return tuple(references)


def _expect_object(value: Any, *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise FewshotArtifactSerializationError(f"{field_name} must be a JSON object.")
    return value


def _expect_list(value: Any, *, field_name: str) -> list[Any]:
    if not isinstance(value, list):
        raise FewshotArtifactSerializationError(f"{field_name} must be a JSON array.")
    return value


def _expect_string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise FewshotArtifactSerializationError(f"{field_name} must be a string.")
    return value


def _expect_optional_string(value: Any, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _expect_string(value, field_name=field_name)


def _expect_int(value: Any, *, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise FewshotArtifactSerializationError(f"{field_name} must be an integer.")
    return value


def _expect_optional_int(value: Any, *, field_name: str) -> int | None:
    if value is None:
        return None
    return _expect_int(value, field_name=field_name)


def _expect_bool(value: Any, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise FewshotArtifactSerializationError(f"{field_name} must be a boolean.")
    return value


def _expect_float(value: Any, *, field_name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(value):
        raise FewshotArtifactSerializationError(f"{field_name} must be a finite number.")
    return float(value)


def _expect_optional_float(value: Any, *, field_name: str) -> float | None:
    if value is None:
        return None
    return _expect_float(value, field_name=field_name)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FewshotArtifactSerializationError(f"Duplicate JSON key: {key}.")
        result[key] = value
    return result


def _bad_json_constant(constant: str) -> Any:
    raise FewshotArtifactSerializationError(
        f"Invalid JSON constant in few-shot artifact: {constant!r}."
    )


__all__ = [
    "FEWSHOT_ADAPTATION_CONFIG_VERSION",
    "FEWSHOT_PROTOCOL_TABLE_VERSION",
    "FEWSHOT_RUN_SUMMARY_VERSION",
    "FEWSHOT_SUPPORT_MANIFEST_VERSION",
    "FewshotAdaptationConfig",
    "FewshotArtifact",
    "FewshotArtifactError",
    "FewshotArtifactHashError",
    "FewshotArtifactSerializationError",
    "FewshotArtifactValidationError",
    "FewshotArtifactVersionError",
    "FewshotInitializationReference",
    "FewshotMetricArtifactReference",
    "FewshotProtocolTable",
    "FewshotProtocolTableRow",
    "FewshotRunSummary",
    "FewshotSupportAssignment",
    "FewshotSupportManifest",
    "SUPPORTED_ADAPTATION_MODES",
    "SUPPORTED_BASELINE_FAMILIES",
    "SUPPORTED_FEWSHOT_K_VALUES",
    "SUPPORTED_FEWSHOT_SCHEMA_VERSIONS",
    "SUPPORTED_INITIALIZATION_REFERENCE_TYPES",
    "SUPPORTED_MEMORY_AVAILABILITY_STATUSES",
    "SUPPORTED_PROTOCOL_RUN_STATUSES",
    "SUPPORTED_STRATIFICATION_STATUSES",
    "fewshot_adaptation_config_from_json",
    "fewshot_adaptation_config_hash_payload",
    "fewshot_adaptation_config_to_dict",
    "fewshot_adaptation_config_to_json",
    "fewshot_artifact_from_json",
    "fewshot_artifact_to_dict",
    "fewshot_artifact_to_json",
    "fewshot_protocol_table_from_json",
    "fewshot_protocol_table_hash_payload",
    "fewshot_protocol_table_to_dict",
    "fewshot_protocol_table_to_json",
    "fewshot_run_summary_from_json",
    "fewshot_run_summary_hash_payload",
    "fewshot_run_summary_to_dict",
    "fewshot_run_summary_to_json",
    "fewshot_support_manifest_from_json",
    "fewshot_support_manifest_hash_payload",
    "fewshot_support_manifest_to_dict",
    "fewshot_support_manifest_to_json",
    "hash_fewshot_adaptation_config",
    "hash_fewshot_protocol_table",
    "hash_fewshot_run_summary",
    "hash_fewshot_support_manifest",
]
