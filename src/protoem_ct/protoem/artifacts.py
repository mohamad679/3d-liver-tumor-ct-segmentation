"""Deterministic Phase 6 ProtoEM-CT artifact schemas and hashing contracts."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, TypeAlias, TypeVar, cast

from protoem_ct.artifacts.hashing import JsonValue, canonical_json_bytes, sha256_json

PROTOEM_CONFIG_SCHEMA_NAME: Final[str] = "protoem_config"
PROTOEM_CONFIG_SCHEMA_VERSION: Final[str] = "v1"
PROTOEM_OBJECTIVE_WEIGHTS_SCHEMA_NAME: Final[str] = "protoem_objective_weights"
PROTOEM_OBJECTIVE_WEIGHTS_SCHEMA_VERSION: Final[str] = "v1"
PROTOEM_ITERATION_RECORD_SCHEMA_NAME: Final[str] = "protoem_iteration_record"
PROTOEM_ITERATION_RECORD_SCHEMA_VERSION: Final[str] = "v1"
PROTOEM_OBJECTIVE_TRACE_SCHEMA_NAME: Final[str] = "protoem_objective_trace"
PROTOEM_OBJECTIVE_TRACE_SCHEMA_VERSION: Final[str] = "v1"
PROTOEM_STOPPING_RECORD_SCHEMA_NAME: Final[str] = "protoem_stopping_record"
PROTOEM_STOPPING_RECORD_SCHEMA_VERSION: Final[str] = "v1"
PROTOEM_COLLAPSE_RECORD_SCHEMA_NAME: Final[str] = "protoem_collapse_record"
PROTOEM_COLLAPSE_RECORD_SCHEMA_VERSION: Final[str] = "v1"
PROTOEM_ABLATION_DEFINITION_SCHEMA_NAME: Final[str] = "protoem_ablation_definition"
PROTOEM_ABLATION_DEFINITION_SCHEMA_VERSION: Final[str] = "v1"
PROTOEM_ABLATION_OVERRIDE_SCHEMA_NAME: Final[str] = "protoem_ablation_override"
PROTOEM_ABLATION_OVERRIDE_SCHEMA_VERSION: Final[str] = "v1"
PROTOEM_RUN_SUMMARY_SCHEMA_NAME: Final[str] = "protoem_run_summary"
PROTOEM_RUN_SUMMARY_SCHEMA_VERSION: Final[str] = "v1"

PROTOEM_ITERATION_INDEX_BASE: Final[str] = "zero_based"
SUPPORTED_PROTOEM_UPDATE_SCHEDULES: Final[frozenset[str]] = frozenset(
    {"fixed_em_like", "learned_positive_step"}
)
SUPPORTED_PROTOEM_PROTOTYPE_MODES: Final[frozenset[str]] = frozenset(
    {"single_prototype", "multiple_prototypes"}
)
SUPPORTED_PROTOEM_STOP_REASONS: Final[frozenset[str]] = frozenset(
    {
        "max_iterations",
        "tolerance_reached",
        "foreground_collapse",
        "background_collapse",
        "no_confident_voxels",
        "non_finite_objective",
        "numerical_failure",
    }
)
SUPPORTED_PROTOEM_COLLAPSE_TYPES: Final[frozenset[str]] = frozenset(
    {"foreground_collapse", "background_collapse"}
)
SUPPORTED_PROTOEM_ABLATION_VARIANTS: Final[frozenset[str]] = frozenset(
    {
        "full_protoem",
        "no_retrieval",
        "no_transduction",
        "no_class_balance",
        "no_proximal",
        "single_prototype",
        "multiple_prototypes",
        "fixed_update_schedule",
        "learned_update_schedule",
        "head_only",
        "decoder_only",
        "full_finetune",
    }
)
SUPPORTED_PROTOEM_EXECUTION_STATUSES: Final[frozenset[str]] = frozenset({"completed", "failed"})
SUPPORTED_PROTOEM_MEMORY_AVAILABILITY_STATUSES: Final[frozenset[str]] = frozenset(
    {"available", "unavailable"}
)
SUPPORTED_PROTOEM_PHASE5_SCHEMA_REFERENCES: Final[frozenset[str]] = frozenset(
    {
        "phase5_comparison_table",
        "phase5_comparison_record",
        "phase5_method_definition",
        "phase5_run_summary",
        "prototype_inference_summary",
        "prototype_memory",
        "retrieval_comparison_table",
        "retrieval_embedding_artifact",
        "retrieval_result",
    }
)
COMPATIBILITY_CRITICAL_CONFIG_OVERRIDE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "confidence_threshold",
        "convergence_tolerance",
        "explicit_seed",
        "foreground_prior",
        "max_iterations",
        "minimum_iterations",
        "objective_weights.class_balance",
        "objective_weights.consistency",
        "objective_weights.proximal",
        "objective_weights.query_entropy",
        "objective_weights.support",
        "phase5_inference_artifact_hash",
        "phase5_inference_schema_name",
        "phase5_initialization_artifact_hash",
        "phase5_initialization_schema_name",
        "phase5_prototype_artifact_hash",
        "phase5_prototype_schema_name",
        "prototype_mode",
        "temperature",
        "update_schedule",
    }
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[a-z0-9](?:[a-z0-9_.-]{0,126}[a-z0-9])?$")
_MESSAGE_RE = re.compile(r"^[^\r\n]+$")


class ProtoEMArtifactError(ValueError):
    """Base error for Phase 6 ProtoEM artifact contracts."""


class ProtoEMArtifactValidationError(ProtoEMArtifactError):
    """Raised when one ProtoEM artifact violates the schema contract."""


class ProtoEMArtifactSerializationError(ProtoEMArtifactError):
    """Raised when one ProtoEM artifact mapping cannot be reconstructed safely."""


class ProtoEMArtifactHashError(ProtoEMArtifactError):
    """Raised when one ProtoEM artifact self-hash does not match its content."""


@dataclass(frozen=True, slots=True)
class ProtoEMObjectiveWeights:
    """Deterministic nonnegative objective weights for the ProtoEM-CT objective."""

    schema_name: str
    schema_version: str
    support: float
    query_entropy: float
    class_balance: float
    consistency: float
    proximal: float

    def __post_init__(self) -> None:
        _require_schema_name(
            self.schema_name,
            expected=PROTOEM_OBJECTIVE_WEIGHTS_SCHEMA_NAME,
            field_name="schema_name",
        )
        _require_schema_version(
            self.schema_version,
            expected=PROTOEM_OBJECTIVE_WEIGHTS_SCHEMA_VERSION,
            field_name="schema_version",
        )
        _require_nonnegative_finite_float(self.support, field_name="support")
        _require_nonnegative_finite_float(self.query_entropy, field_name="query_entropy")
        _require_nonnegative_finite_float(self.class_balance, field_name="class_balance")
        _require_nonnegative_finite_float(self.consistency, field_name="consistency")
        _require_nonnegative_finite_float(self.proximal, field_name="proximal")


@dataclass(frozen=True, slots=True)
class ProtoEMConfig:
    """Deterministic configuration identity for one ProtoEM-CT transductive adaptation run."""

    schema_name: str
    schema_version: str
    config_hash: str
    explicit_seed: int
    max_iterations: int
    minimum_iterations: int
    convergence_tolerance: float
    temperature: float
    confidence_threshold: float
    foreground_prior: float
    update_schedule: str
    prototype_mode: str
    objective_weights: ProtoEMObjectiveWeights
    phase5_initialization_schema_name: str
    phase5_initialization_artifact_hash: str
    phase5_prototype_schema_name: str
    phase5_prototype_artifact_hash: str
    phase5_inference_schema_name: str
    phase5_inference_artifact_hash: str

    def __post_init__(self) -> None:
        _require_schema_name(
            self.schema_name,
            expected=PROTOEM_CONFIG_SCHEMA_NAME,
            field_name="schema_name",
        )
        _require_schema_version(
            self.schema_version,
            expected=PROTOEM_CONFIG_SCHEMA_VERSION,
            field_name="schema_version",
        )
        _require_sha256(self.config_hash, field_name="config_hash")
        _require_seed(self.explicit_seed, field_name="explicit_seed")
        _require_positive_int(self.max_iterations, field_name="max_iterations")
        _require_nonnegative_int(self.minimum_iterations, field_name="minimum_iterations")
        if self.minimum_iterations > self.max_iterations:
            raise ProtoEMArtifactValidationError(
                "minimum_iterations must be less than or equal to max_iterations."
            )
        _require_nonnegative_finite_float(
            self.convergence_tolerance,
            field_name="convergence_tolerance",
        )
        _require_positive_finite_float(self.temperature, field_name="temperature")
        _require_fraction_closed(self.confidence_threshold, field_name="confidence_threshold")
        _require_fraction_open(self.foreground_prior, field_name="foreground_prior")
        if self.update_schedule not in SUPPORTED_PROTOEM_UPDATE_SCHEDULES:
            raise ProtoEMArtifactValidationError(
                f"update_schedule must be one of {sorted(SUPPORTED_PROTOEM_UPDATE_SCHEDULES)!r}."
            )
        if self.prototype_mode not in SUPPORTED_PROTOEM_PROTOTYPE_MODES:
            raise ProtoEMArtifactValidationError(
                f"prototype_mode must be one of {sorted(SUPPORTED_PROTOEM_PROTOTYPE_MODES)!r}."
            )
        _require_phase5_schema_reference(
            self.phase5_initialization_schema_name,
            field_name="phase5_initialization_schema_name",
        )
        _require_sha256(
            self.phase5_initialization_artifact_hash,
            field_name="phase5_initialization_artifact_hash",
        )
        _require_phase5_schema_reference(
            self.phase5_prototype_schema_name,
            field_name="phase5_prototype_schema_name",
        )
        _require_sha256(
            self.phase5_prototype_artifact_hash,
            field_name="phase5_prototype_artifact_hash",
        )
        _require_phase5_schema_reference(
            self.phase5_inference_schema_name,
            field_name="phase5_inference_schema_name",
        )
        _require_sha256(
            self.phase5_inference_artifact_hash,
            field_name="phase5_inference_artifact_hash",
        )
        expected_hash = hash_protoem_config(self)
        if self.config_hash != expected_hash:
            raise ProtoEMArtifactHashError(
                "config_hash does not match the deterministic ProtoEM config content."
            )


@dataclass(frozen=True, slots=True)
class ProtoEMIterationRecord:
    """One zero-based deterministic ProtoEM-CT iteration record."""

    schema_name: str
    schema_version: str
    iteration_index_base: str
    iteration_index: int
    support_objective: float
    query_entropy_objective: float
    class_balance_objective: float
    consistency_objective: float
    proximal_objective: float
    total_objective: float
    confident_voxel_count: int
    foreground_assignment_count: int
    background_assignment_count: int
    foreground_fraction: float
    state_identity_hash_before: str
    state_identity_hash_after: str
    prototype_identity_hash_before: str
    prototype_identity_hash_after: str
    convergence_delta: float
    finite_status_ok: bool
    collapse_status_detected: bool

    def __post_init__(self) -> None:
        _require_schema_name(
            self.schema_name,
            expected=PROTOEM_ITERATION_RECORD_SCHEMA_NAME,
            field_name="schema_name",
        )
        _require_schema_version(
            self.schema_version,
            expected=PROTOEM_ITERATION_RECORD_SCHEMA_VERSION,
            field_name="schema_version",
        )
        if self.iteration_index_base != PROTOEM_ITERATION_INDEX_BASE:
            raise ProtoEMArtifactValidationError(
                f"iteration_index_base must equal {PROTOEM_ITERATION_INDEX_BASE!r}."
            )
        _require_nonnegative_int(self.iteration_index, field_name="iteration_index")
        _require_finite_float(self.support_objective, field_name="support_objective")
        _require_finite_float(
            self.query_entropy_objective,
            field_name="query_entropy_objective",
        )
        _require_finite_float(
            self.class_balance_objective,
            field_name="class_balance_objective",
        )
        _require_finite_float(self.consistency_objective, field_name="consistency_objective")
        _require_finite_float(self.proximal_objective, field_name="proximal_objective")
        _require_finite_float(self.total_objective, field_name="total_objective")
        _require_nonnegative_int(self.confident_voxel_count, field_name="confident_voxel_count")
        _require_nonnegative_int(
            self.foreground_assignment_count,
            field_name="foreground_assignment_count",
        )
        _require_nonnegative_int(
            self.background_assignment_count,
            field_name="background_assignment_count",
        )
        if (
            self.foreground_assignment_count + self.background_assignment_count
            != self.confident_voxel_count
        ):
            raise ProtoEMArtifactValidationError(
                "foreground_assignment_count plus background_assignment_count must equal "
                "confident_voxel_count."
            )
        _require_fraction_closed(self.foreground_fraction, field_name="foreground_fraction")
        _require_sha256(self.state_identity_hash_before, field_name="state_identity_hash_before")
        _require_sha256(self.state_identity_hash_after, field_name="state_identity_hash_after")
        _require_sha256(
            self.prototype_identity_hash_before,
            field_name="prototype_identity_hash_before",
        )
        _require_sha256(
            self.prototype_identity_hash_after,
            field_name="prototype_identity_hash_after",
        )
        _require_nonnegative_finite_float(self.convergence_delta, field_name="convergence_delta")
        if not self.finite_status_ok:
            raise ProtoEMArtifactValidationError(
                "Iteration records must have finite_status_ok=true."
            )


@dataclass(frozen=True, slots=True)
class ProtoEMObjectiveTrace:
    """Deterministic ordered trace of zero-based ProtoEM-CT iteration records."""

    schema_name: str
    schema_version: str
    objective_trace_hash: str
    iteration_index_base: str
    iterations: tuple[ProtoEMIterationRecord, ...]

    def __post_init__(self) -> None:
        _require_schema_name(
            self.schema_name,
            expected=PROTOEM_OBJECTIVE_TRACE_SCHEMA_NAME,
            field_name="schema_name",
        )
        _require_schema_version(
            self.schema_version,
            expected=PROTOEM_OBJECTIVE_TRACE_SCHEMA_VERSION,
            field_name="schema_version",
        )
        _require_sha256(self.objective_trace_hash, field_name="objective_trace_hash")
        if self.iteration_index_base != PROTOEM_ITERATION_INDEX_BASE:
            raise ProtoEMArtifactValidationError(
                f"iteration_index_base must equal {PROTOEM_ITERATION_INDEX_BASE!r}."
            )
        if not self.iterations:
            raise ProtoEMArtifactValidationError("ProtoEMObjectiveTrace must not be empty.")
        expected_indices = tuple(range(len(self.iterations)))
        actual_indices = tuple(record.iteration_index for record in self.iterations)
        if actual_indices != expected_indices:
            raise ProtoEMArtifactValidationError(
                "ProtoEMObjectiveTrace iteration indices must be strictly increasing contiguous "
                "zero-based integers with no duplicates."
            )
        expected_hash = hash_protoem_objective_trace(self)
        if self.objective_trace_hash != expected_hash:
            raise ProtoEMArtifactHashError(
                "objective_trace_hash does not match the deterministic ProtoEM trace content."
            )


@dataclass(frozen=True, slots=True)
class ProtoEMStoppingRecord:
    """Deterministic stopping outcome for one ProtoEM-CT adaptation run."""

    schema_name: str
    schema_version: str
    stopping_record_hash: str
    stop_reason: str
    final_iteration_index: int
    completed_iteration_count: int
    converged: bool
    failed: bool
    failure_code: str | None
    failure_message: str | None

    def __post_init__(self) -> None:
        _require_schema_name(
            self.schema_name,
            expected=PROTOEM_STOPPING_RECORD_SCHEMA_NAME,
            field_name="schema_name",
        )
        _require_schema_version(
            self.schema_version,
            expected=PROTOEM_STOPPING_RECORD_SCHEMA_VERSION,
            field_name="schema_version",
        )
        _require_sha256(self.stopping_record_hash, field_name="stopping_record_hash")
        if self.stop_reason not in SUPPORTED_PROTOEM_STOP_REASONS:
            raise ProtoEMArtifactValidationError(
                f"stop_reason must be one of {sorted(SUPPORTED_PROTOEM_STOP_REASONS)!r}."
            )
        _require_nonnegative_int(self.final_iteration_index, field_name="final_iteration_index")
        _require_positive_int(
            self.completed_iteration_count,
            field_name="completed_iteration_count",
        )
        if self.completed_iteration_count != self.final_iteration_index + 1:
            raise ProtoEMArtifactValidationError(
                "completed_iteration_count must equal final_iteration_index + 1."
            )
        _require_failure_consistency(
            failed=self.failed,
            failure_code=self.failure_code,
            failure_message=self.failure_message,
            field_prefix="ProtoEMStoppingRecord",
        )
        if self.converged and self.failed:
            raise ProtoEMArtifactValidationError("converged and failed cannot both be true.")
        if self.converged and self.stop_reason != "tolerance_reached":
            raise ProtoEMArtifactValidationError(
                "converged=true requires stop_reason='tolerance_reached'."
            )
        if self.stop_reason == "tolerance_reached" and not self.converged:
            raise ProtoEMArtifactValidationError(
                "stop_reason='tolerance_reached' requires converged=true."
            )
        if self.stop_reason == "max_iterations" and self.failed:
            raise ProtoEMArtifactValidationError(
                "stop_reason='max_iterations' must not mark the run as failed."
            )
        if (
            self.stop_reason
            in {
                "foreground_collapse",
                "background_collapse",
                "no_confident_voxels",
                "non_finite_objective",
                "numerical_failure",
            }
            and not self.failed
        ):
            raise ProtoEMArtifactValidationError(
                "failure-like stop reasons must mark the run as failed."
            )
        expected_hash = hash_protoem_stopping_record(self)
        if self.stopping_record_hash != expected_hash:
            raise ProtoEMArtifactHashError(
                "stopping_record_hash does not match the deterministic stopping content."
            )


@dataclass(frozen=True, slots=True)
class ProtoEMCollapseRecord:
    """Deterministic record of detected foreground or background collapse."""

    schema_name: str
    schema_version: str
    collapse_record_hash: str
    collapse_type: str
    iteration_index: int
    foreground_count: int
    background_count: int
    confident_voxel_count: int
    foreground_fraction: float
    configured_lower_safeguard: float
    configured_upper_safeguard: float
    detected: bool

    def __post_init__(self) -> None:
        _require_schema_name(
            self.schema_name,
            expected=PROTOEM_COLLAPSE_RECORD_SCHEMA_NAME,
            field_name="schema_name",
        )
        _require_schema_version(
            self.schema_version,
            expected=PROTOEM_COLLAPSE_RECORD_SCHEMA_VERSION,
            field_name="schema_version",
        )
        _require_sha256(self.collapse_record_hash, field_name="collapse_record_hash")
        if self.collapse_type not in SUPPORTED_PROTOEM_COLLAPSE_TYPES:
            raise ProtoEMArtifactValidationError(
                f"collapse_type must be one of {sorted(SUPPORTED_PROTOEM_COLLAPSE_TYPES)!r}."
            )
        _require_nonnegative_int(self.iteration_index, field_name="iteration_index")
        _require_nonnegative_int(self.foreground_count, field_name="foreground_count")
        _require_nonnegative_int(self.background_count, field_name="background_count")
        _require_nonnegative_int(self.confident_voxel_count, field_name="confident_voxel_count")
        if self.foreground_count + self.background_count != self.confident_voxel_count:
            raise ProtoEMArtifactValidationError(
                "foreground_count plus background_count must equal confident_voxel_count."
            )
        _require_fraction_closed(self.foreground_fraction, field_name="foreground_fraction")
        _require_fraction_closed(
            self.configured_lower_safeguard,
            field_name="configured_lower_safeguard",
        )
        _require_fraction_closed(
            self.configured_upper_safeguard,
            field_name="configured_upper_safeguard",
        )
        if self.configured_lower_safeguard >= self.configured_upper_safeguard:
            raise ProtoEMArtifactValidationError(
                "configured_lower_safeguard must be less than configured_upper_safeguard."
            )
        expected_hash = hash_protoem_collapse_record(self)
        if self.collapse_record_hash != expected_hash:
            raise ProtoEMArtifactHashError(
                "collapse_record_hash does not match the deterministic collapse content."
            )


@dataclass(frozen=True, slots=True)
class ProtoEMAblationOverride:
    """One explicit immutable config override used by a named ablation variant."""

    schema_name: str
    schema_version: str
    field_name: str
    override_value: JsonValue

    def __post_init__(self) -> None:
        _require_schema_name(
            self.schema_name,
            expected=PROTOEM_ABLATION_OVERRIDE_SCHEMA_NAME,
            field_name="schema_name",
        )
        _require_schema_version(
            self.schema_version,
            expected=PROTOEM_ABLATION_OVERRIDE_SCHEMA_VERSION,
            field_name="schema_version",
        )
        if self.field_name not in COMPATIBILITY_CRITICAL_CONFIG_OVERRIDE_FIELDS:
            raise ProtoEMArtifactValidationError(
                "field_name must be one documented compatibility-critical ProtoEM config field."
            )
        canonical_json_bytes(self.override_value)


@dataclass(frozen=True, slots=True)
class ProtoEMAblationDefinition:
    """Immutable named ablation identity with explicit documented config overrides only."""

    schema_name: str
    schema_version: str
    ablation_definition_hash: str
    variant_name: str
    overrides: tuple[ProtoEMAblationOverride, ...]

    def __post_init__(self) -> None:
        _require_schema_name(
            self.schema_name,
            expected=PROTOEM_ABLATION_DEFINITION_SCHEMA_NAME,
            field_name="schema_name",
        )
        _require_schema_version(
            self.schema_version,
            expected=PROTOEM_ABLATION_DEFINITION_SCHEMA_VERSION,
            field_name="schema_version",
        )
        _require_sha256(self.ablation_definition_hash, field_name="ablation_definition_hash")
        if self.variant_name not in SUPPORTED_PROTOEM_ABLATION_VARIANTS:
            raise ProtoEMArtifactValidationError(
                f"variant_name must be one of {sorted(SUPPORTED_PROTOEM_ABLATION_VARIANTS)!r}."
            )
        ordered_overrides = tuple(sorted(self.overrides, key=lambda item: item.field_name))
        object.__setattr__(self, "overrides", ordered_overrides)
        field_names = [item.field_name for item in self.overrides]
        if len(set(field_names)) != len(field_names):
            raise ProtoEMArtifactValidationError(
                "ProtoEMAblationDefinition overrides must not repeat the same field_name."
            )
        expected_hash = hash_protoem_ablation_definition(self)
        if self.ablation_definition_hash != expected_hash:
            raise ProtoEMArtifactHashError(
                "ablation_definition_hash does not match deterministic ablation content."
            )


@dataclass(frozen=True, slots=True)
class ProtoEMRunSummary:
    """Deterministic ProtoEM-CT run summary without fabricated metrics."""

    schema_name: str
    schema_version: str
    run_summary_hash: str
    config_hash: str
    phase5_initialization_artifact_hash: str
    phase5_prototype_artifact_hash: str
    phase5_inference_artifact_hash: str
    objective_trace_hash: str
    stopping_record_hash: str
    collapse_record_hash: str | None
    ablation_definition_hash: str
    final_output_artifact_hash: str | None
    final_inference_artifact_hash: str | None
    execution_status: str
    failure_code: str | None
    failure_message: str | None
    duration_seconds: float | None
    memory_availability_status: str
    peak_allocated_memory_bytes: int | None
    peak_reserved_memory_bytes: int | None
    peak_host_memory_bytes: int | None

    def __post_init__(self) -> None:
        _require_schema_name(
            self.schema_name,
            expected=PROTOEM_RUN_SUMMARY_SCHEMA_NAME,
            field_name="schema_name",
        )
        _require_schema_version(
            self.schema_version,
            expected=PROTOEM_RUN_SUMMARY_SCHEMA_VERSION,
            field_name="schema_version",
        )
        _require_sha256(self.run_summary_hash, field_name="run_summary_hash")
        _require_sha256(self.config_hash, field_name="config_hash")
        _require_sha256(
            self.phase5_initialization_artifact_hash,
            field_name="phase5_initialization_artifact_hash",
        )
        _require_sha256(
            self.phase5_prototype_artifact_hash,
            field_name="phase5_prototype_artifact_hash",
        )
        _require_sha256(
            self.phase5_inference_artifact_hash,
            field_name="phase5_inference_artifact_hash",
        )
        _require_sha256(self.objective_trace_hash, field_name="objective_trace_hash")
        _require_sha256(self.stopping_record_hash, field_name="stopping_record_hash")
        _require_optional_sha256(self.collapse_record_hash, field_name="collapse_record_hash")
        _require_sha256(
            self.ablation_definition_hash,
            field_name="ablation_definition_hash",
        )
        _require_optional_sha256(
            self.final_output_artifact_hash,
            field_name="final_output_artifact_hash",
        )
        _require_optional_sha256(
            self.final_inference_artifact_hash,
            field_name="final_inference_artifact_hash",
        )
        if self.execution_status not in SUPPORTED_PROTOEM_EXECUTION_STATUSES:
            raise ProtoEMArtifactValidationError(
                f"execution_status must be one of {sorted(SUPPORTED_PROTOEM_EXECUTION_STATUSES)!r}."
            )
        _require_failure_consistency(
            failed=self.execution_status == "failed",
            failure_code=self.failure_code,
            failure_message=self.failure_message,
            field_prefix="ProtoEMRunSummary",
        )
        if self.execution_status == "completed" and (
            self.final_output_artifact_hash is None or self.final_inference_artifact_hash is None
        ):
            raise ProtoEMArtifactValidationError(
                "completed ProtoEMRunSummary requires final output and inference artifact hashes."
            )
        if self.execution_status == "failed" and (
            self.final_output_artifact_hash is not None
            or self.final_inference_artifact_hash is not None
        ):
            raise ProtoEMArtifactValidationError(
                "failed ProtoEMRunSummary must not claim final output or inference hashes."
            )
        _require_optional_nonnegative_finite_float(
            self.duration_seconds,
            field_name="duration_seconds",
        )
        if self.memory_availability_status not in SUPPORTED_PROTOEM_MEMORY_AVAILABILITY_STATUSES:
            raise ProtoEMArtifactValidationError(
                "memory_availability_status must be 'available' or 'unavailable'."
            )
        if self.memory_availability_status == "available":
            for name, value in (
                ("peak_allocated_memory_bytes", self.peak_allocated_memory_bytes),
                ("peak_reserved_memory_bytes", self.peak_reserved_memory_bytes),
                ("peak_host_memory_bytes", self.peak_host_memory_bytes),
            ):
                if value is None:
                    raise ProtoEMArtifactValidationError(
                        f"{name} must be present when memory_availability_status='available'."
                    )
                _require_nonnegative_int(value, field_name=name)
        else:
            if any(
                value is not None
                for value in (
                    self.peak_allocated_memory_bytes,
                    self.peak_reserved_memory_bytes,
                    self.peak_host_memory_bytes,
                )
            ):
                raise ProtoEMArtifactValidationError(
                    "memory fields must be null when memory_availability_status='unavailable'."
                )
        expected_hash = hash_protoem_run_summary(self)
        if self.run_summary_hash != expected_hash:
            raise ProtoEMArtifactHashError(
                "run_summary_hash does not match deterministic ProtoEM run-summary content."
            )


ProtoEMArtifact: TypeAlias = (
    ProtoEMConfig
    | ProtoEMObjectiveTrace
    | ProtoEMStoppingRecord
    | ProtoEMCollapseRecord
    | ProtoEMAblationDefinition
    | ProtoEMRunSummary
)
_ArtifactType = TypeVar("_ArtifactType", bound=ProtoEMArtifact)
MappingLike = Mapping[str, object]

_OBJECTIVE_WEIGHTS_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_name",
        "schema_version",
        "support",
        "query_entropy",
        "class_balance",
        "consistency",
        "proximal",
    }
)
_CONFIG_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_name",
        "schema_version",
        "config_hash",
        "explicit_seed",
        "max_iterations",
        "minimum_iterations",
        "convergence_tolerance",
        "temperature",
        "confidence_threshold",
        "foreground_prior",
        "update_schedule",
        "prototype_mode",
        "objective_weights",
        "phase5_initialization_schema_name",
        "phase5_initialization_artifact_hash",
        "phase5_prototype_schema_name",
        "phase5_prototype_artifact_hash",
        "phase5_inference_schema_name",
        "phase5_inference_artifact_hash",
    }
)
_ITERATION_RECORD_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_name",
        "schema_version",
        "iteration_index_base",
        "iteration_index",
        "support_objective",
        "query_entropy_objective",
        "class_balance_objective",
        "consistency_objective",
        "proximal_objective",
        "total_objective",
        "confident_voxel_count",
        "foreground_assignment_count",
        "background_assignment_count",
        "foreground_fraction",
        "state_identity_hash_before",
        "state_identity_hash_after",
        "prototype_identity_hash_before",
        "prototype_identity_hash_after",
        "convergence_delta",
        "finite_status_ok",
        "collapse_status_detected",
    }
)
_OBJECTIVE_TRACE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_name",
        "schema_version",
        "objective_trace_hash",
        "iteration_index_base",
        "iterations",
    }
)
_STOPPING_RECORD_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_name",
        "schema_version",
        "stopping_record_hash",
        "stop_reason",
        "final_iteration_index",
        "completed_iteration_count",
        "converged",
        "failed",
        "failure_code",
        "failure_message",
    }
)
_COLLAPSE_RECORD_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_name",
        "schema_version",
        "collapse_record_hash",
        "collapse_type",
        "iteration_index",
        "foreground_count",
        "background_count",
        "confident_voxel_count",
        "foreground_fraction",
        "configured_lower_safeguard",
        "configured_upper_safeguard",
        "detected",
    }
)
_ABLATION_OVERRIDE_FIELDS: Final[frozenset[str]] = frozenset(
    {"schema_name", "schema_version", "field_name", "override_value"}
)
_ABLATION_DEFINITION_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_name",
        "schema_version",
        "ablation_definition_hash",
        "variant_name",
        "overrides",
    }
)
_RUN_SUMMARY_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_name",
        "schema_version",
        "run_summary_hash",
        "config_hash",
        "phase5_initialization_artifact_hash",
        "phase5_prototype_artifact_hash",
        "phase5_inference_artifact_hash",
        "objective_trace_hash",
        "stopping_record_hash",
        "collapse_record_hash",
        "ablation_definition_hash",
        "final_output_artifact_hash",
        "final_inference_artifact_hash",
        "execution_status",
        "failure_code",
        "failure_message",
        "duration_seconds",
        "memory_availability_status",
        "peak_allocated_memory_bytes",
        "peak_reserved_memory_bytes",
        "peak_host_memory_bytes",
    }
)


def protoem_objective_weights_to_dict(weights: ProtoEMObjectiveWeights) -> dict[str, JsonValue]:
    """Convert ProtoEM objective weights to a canonical mapping."""

    return {
        "schema_name": weights.schema_name,
        "schema_version": weights.schema_version,
        "support": weights.support,
        "query_entropy": weights.query_entropy,
        "class_balance": weights.class_balance,
        "consistency": weights.consistency,
        "proximal": weights.proximal,
    }


def protoem_config_identity_payload(config: ProtoEMConfig) -> dict[str, JsonValue]:
    """Return the exact canonical identity payload for one ProtoEM config."""

    return {
        "confidence_threshold": config.confidence_threshold,
        "convergence_tolerance": config.convergence_tolerance,
        "explicit_seed": config.explicit_seed,
        "foreground_prior": config.foreground_prior,
        "max_iterations": config.max_iterations,
        "minimum_iterations": config.minimum_iterations,
        "objective_weights": protoem_objective_weights_to_dict(config.objective_weights),
        "phase5_inference_artifact_hash": config.phase5_inference_artifact_hash,
        "phase5_inference_schema_name": config.phase5_inference_schema_name,
        "phase5_initialization_artifact_hash": config.phase5_initialization_artifact_hash,
        "phase5_initialization_schema_name": config.phase5_initialization_schema_name,
        "phase5_prototype_artifact_hash": config.phase5_prototype_artifact_hash,
        "phase5_prototype_schema_name": config.phase5_prototype_schema_name,
        "prototype_mode": config.prototype_mode,
        "schema_name": config.schema_name,
        "schema_version": config.schema_version,
        "temperature": config.temperature,
        "update_schedule": config.update_schedule,
    }


def protoem_config_to_dict(config: ProtoEMConfig) -> dict[str, JsonValue]:
    """Convert one ProtoEM config to a canonical mapping."""

    payload = protoem_config_identity_payload(config)
    payload["config_hash"] = config.config_hash
    return payload


def hash_protoem_config(config: ProtoEMConfig) -> str:
    """Return the canonical SHA-256 hash for one ProtoEM config."""

    return sha256_json(protoem_config_identity_payload(config))


def protoem_iteration_record_to_dict(record: ProtoEMIterationRecord) -> dict[str, JsonValue]:
    """Convert one ProtoEM iteration record to a canonical mapping."""

    return {
        "schema_name": record.schema_name,
        "schema_version": record.schema_version,
        "iteration_index_base": record.iteration_index_base,
        "iteration_index": record.iteration_index,
        "support_objective": record.support_objective,
        "query_entropy_objective": record.query_entropy_objective,
        "class_balance_objective": record.class_balance_objective,
        "consistency_objective": record.consistency_objective,
        "proximal_objective": record.proximal_objective,
        "total_objective": record.total_objective,
        "confident_voxel_count": record.confident_voxel_count,
        "foreground_assignment_count": record.foreground_assignment_count,
        "background_assignment_count": record.background_assignment_count,
        "foreground_fraction": record.foreground_fraction,
        "state_identity_hash_before": record.state_identity_hash_before,
        "state_identity_hash_after": record.state_identity_hash_after,
        "prototype_identity_hash_before": record.prototype_identity_hash_before,
        "prototype_identity_hash_after": record.prototype_identity_hash_after,
        "convergence_delta": record.convergence_delta,
        "finite_status_ok": record.finite_status_ok,
        "collapse_status_detected": record.collapse_status_detected,
    }


def protoem_objective_trace_identity_payload(
    trace: ProtoEMObjectiveTrace,
) -> dict[str, JsonValue]:
    """Return the exact canonical identity payload for one ProtoEM objective trace."""

    return {
        "iteration_index_base": trace.iteration_index_base,
        "iterations": [protoem_iteration_record_to_dict(item) for item in trace.iterations],
        "schema_name": trace.schema_name,
        "schema_version": trace.schema_version,
    }


def protoem_objective_trace_to_dict(trace: ProtoEMObjectiveTrace) -> dict[str, JsonValue]:
    """Convert one ProtoEM objective trace to a canonical mapping."""

    payload = protoem_objective_trace_identity_payload(trace)
    payload["objective_trace_hash"] = trace.objective_trace_hash
    return payload


def hash_protoem_objective_trace(trace: ProtoEMObjectiveTrace) -> str:
    """Return the canonical SHA-256 hash for one ProtoEM objective trace."""

    return sha256_json(protoem_objective_trace_identity_payload(trace))


def protoem_stopping_record_identity_payload(
    record: ProtoEMStoppingRecord,
) -> dict[str, JsonValue]:
    """Return the exact canonical identity payload for one stopping record."""

    return {
        "completed_iteration_count": record.completed_iteration_count,
        "converged": record.converged,
        "failed": record.failed,
        "failure_code": record.failure_code,
        "failure_message": record.failure_message,
        "final_iteration_index": record.final_iteration_index,
        "schema_name": record.schema_name,
        "schema_version": record.schema_version,
        "stop_reason": record.stop_reason,
    }


def protoem_stopping_record_to_dict(record: ProtoEMStoppingRecord) -> dict[str, JsonValue]:
    """Convert one ProtoEM stopping record to a canonical mapping."""

    payload = protoem_stopping_record_identity_payload(record)
    payload["stopping_record_hash"] = record.stopping_record_hash
    return payload


def hash_protoem_stopping_record(record: ProtoEMStoppingRecord) -> str:
    """Return the canonical SHA-256 hash for one stopping record."""

    return sha256_json(protoem_stopping_record_identity_payload(record))


def protoem_collapse_record_identity_payload(record: ProtoEMCollapseRecord) -> dict[str, JsonValue]:
    """Return the exact canonical identity payload for one collapse record."""

    return {
        "background_count": record.background_count,
        "collapse_type": record.collapse_type,
        "confident_voxel_count": record.confident_voxel_count,
        "configured_lower_safeguard": record.configured_lower_safeguard,
        "configured_upper_safeguard": record.configured_upper_safeguard,
        "detected": record.detected,
        "foreground_count": record.foreground_count,
        "foreground_fraction": record.foreground_fraction,
        "iteration_index": record.iteration_index,
        "schema_name": record.schema_name,
        "schema_version": record.schema_version,
    }


def protoem_collapse_record_to_dict(record: ProtoEMCollapseRecord) -> dict[str, JsonValue]:
    """Convert one ProtoEM collapse record to a canonical mapping."""

    payload = protoem_collapse_record_identity_payload(record)
    payload["collapse_record_hash"] = record.collapse_record_hash
    return payload


def hash_protoem_collapse_record(record: ProtoEMCollapseRecord) -> str:
    """Return the canonical SHA-256 hash for one collapse record."""

    return sha256_json(protoem_collapse_record_identity_payload(record))


def protoem_ablation_override_to_dict(
    override: ProtoEMAblationOverride,
) -> dict[str, JsonValue]:
    """Convert one ProtoEM ablation override to a canonical mapping."""

    return {
        "schema_name": override.schema_name,
        "schema_version": override.schema_version,
        "field_name": override.field_name,
        "override_value": override.override_value,
    }


def protoem_ablation_definition_identity_payload(
    definition: ProtoEMAblationDefinition,
) -> dict[str, JsonValue]:
    """Return the exact canonical identity payload for one ablation definition."""

    return {
        "overrides": [protoem_ablation_override_to_dict(item) for item in definition.overrides],
        "schema_name": definition.schema_name,
        "schema_version": definition.schema_version,
        "variant_name": definition.variant_name,
    }


def protoem_ablation_definition_to_dict(
    definition: ProtoEMAblationDefinition,
) -> dict[str, JsonValue]:
    """Convert one ProtoEM ablation definition to a canonical mapping."""

    payload = protoem_ablation_definition_identity_payload(definition)
    payload["ablation_definition_hash"] = definition.ablation_definition_hash
    return payload


def hash_protoem_ablation_definition(definition: ProtoEMAblationDefinition) -> str:
    """Return the canonical SHA-256 hash for one ablation definition."""

    return sha256_json(protoem_ablation_definition_identity_payload(definition))


def protoem_run_summary_identity_payload(summary: ProtoEMRunSummary) -> dict[str, JsonValue]:
    """Return the exact canonical identity payload for one ProtoEM run summary."""

    return {
        "ablation_definition_hash": summary.ablation_definition_hash,
        "collapse_record_hash": summary.collapse_record_hash,
        "config_hash": summary.config_hash,
        "execution_status": summary.execution_status,
        "failure_code": summary.failure_code,
        "failure_message": summary.failure_message,
        "final_inference_artifact_hash": summary.final_inference_artifact_hash,
        "final_output_artifact_hash": summary.final_output_artifact_hash,
        "objective_trace_hash": summary.objective_trace_hash,
        "phase5_inference_artifact_hash": summary.phase5_inference_artifact_hash,
        "phase5_initialization_artifact_hash": summary.phase5_initialization_artifact_hash,
        "phase5_prototype_artifact_hash": summary.phase5_prototype_artifact_hash,
        "schema_name": summary.schema_name,
        "schema_version": summary.schema_version,
        "stopping_record_hash": summary.stopping_record_hash,
    }


def protoem_run_summary_to_dict(summary: ProtoEMRunSummary) -> dict[str, JsonValue]:
    """Convert one ProtoEM run summary to a canonical mapping."""

    payload = protoem_run_summary_identity_payload(summary)
    payload["run_summary_hash"] = summary.run_summary_hash
    payload["duration_seconds"] = summary.duration_seconds
    payload["memory_availability_status"] = summary.memory_availability_status
    payload["peak_allocated_memory_bytes"] = summary.peak_allocated_memory_bytes
    payload["peak_reserved_memory_bytes"] = summary.peak_reserved_memory_bytes
    payload["peak_host_memory_bytes"] = summary.peak_host_memory_bytes
    return payload


def hash_protoem_run_summary(summary: ProtoEMRunSummary) -> str:
    """Return the canonical SHA-256 hash for one ProtoEM run summary."""

    return sha256_json(protoem_run_summary_identity_payload(summary))


def protoem_objective_weights_from_mapping(mapping: MappingLike) -> ProtoEMObjectiveWeights:
    """Reconstruct objective weights from one strict mapping."""

    _require_exact_fields(
        mapping,
        required_fields=_OBJECTIVE_WEIGHTS_FIELDS,
        object_name="ProtoEMObjectiveWeights",
    )
    return ProtoEMObjectiveWeights(
        schema_name=_expect_string(mapping["schema_name"], field_name="schema_name"),
        schema_version=_expect_string(mapping["schema_version"], field_name="schema_version"),
        support=_expect_float(mapping["support"], field_name="support"),
        query_entropy=_expect_float(mapping["query_entropy"], field_name="query_entropy"),
        class_balance=_expect_float(mapping["class_balance"], field_name="class_balance"),
        consistency=_expect_float(mapping["consistency"], field_name="consistency"),
        proximal=_expect_float(mapping["proximal"], field_name="proximal"),
    )


def protoem_config_from_mapping(mapping: MappingLike) -> ProtoEMConfig:
    """Reconstruct a ProtoEM config from one strict mapping."""

    _require_exact_fields(mapping, required_fields=_CONFIG_FIELDS, object_name="ProtoEMConfig")
    return ProtoEMConfig(
        schema_name=_expect_string(mapping["schema_name"], field_name="schema_name"),
        schema_version=_expect_string(mapping["schema_version"], field_name="schema_version"),
        config_hash=_expect_string(mapping["config_hash"], field_name="config_hash"),
        explicit_seed=_expect_int(mapping["explicit_seed"], field_name="explicit_seed"),
        max_iterations=_expect_int(mapping["max_iterations"], field_name="max_iterations"),
        minimum_iterations=_expect_int(
            mapping["minimum_iterations"], field_name="minimum_iterations"
        ),
        convergence_tolerance=_expect_float(
            mapping["convergence_tolerance"],
            field_name="convergence_tolerance",
        ),
        temperature=_expect_float(mapping["temperature"], field_name="temperature"),
        confidence_threshold=_expect_float(
            mapping["confidence_threshold"],
            field_name="confidence_threshold",
        ),
        foreground_prior=_expect_float(mapping["foreground_prior"], field_name="foreground_prior"),
        update_schedule=_expect_string(mapping["update_schedule"], field_name="update_schedule"),
        prototype_mode=_expect_string(mapping["prototype_mode"], field_name="prototype_mode"),
        objective_weights=protoem_objective_weights_from_mapping(
            _expect_mapping(mapping["objective_weights"], field_name="objective_weights")
        ),
        phase5_initialization_schema_name=_expect_string(
            mapping["phase5_initialization_schema_name"],
            field_name="phase5_initialization_schema_name",
        ),
        phase5_initialization_artifact_hash=_expect_string(
            mapping["phase5_initialization_artifact_hash"],
            field_name="phase5_initialization_artifact_hash",
        ),
        phase5_prototype_schema_name=_expect_string(
            mapping["phase5_prototype_schema_name"],
            field_name="phase5_prototype_schema_name",
        ),
        phase5_prototype_artifact_hash=_expect_string(
            mapping["phase5_prototype_artifact_hash"],
            field_name="phase5_prototype_artifact_hash",
        ),
        phase5_inference_schema_name=_expect_string(
            mapping["phase5_inference_schema_name"],
            field_name="phase5_inference_schema_name",
        ),
        phase5_inference_artifact_hash=_expect_string(
            mapping["phase5_inference_artifact_hash"],
            field_name="phase5_inference_artifact_hash",
        ),
    )


def protoem_iteration_record_from_mapping(mapping: MappingLike) -> ProtoEMIterationRecord:
    """Reconstruct one ProtoEM iteration record from a strict mapping."""

    _require_exact_fields(
        mapping,
        required_fields=_ITERATION_RECORD_FIELDS,
        object_name="ProtoEMIterationRecord",
    )
    return ProtoEMIterationRecord(
        schema_name=_expect_string(mapping["schema_name"], field_name="schema_name"),
        schema_version=_expect_string(mapping["schema_version"], field_name="schema_version"),
        iteration_index_base=_expect_string(
            mapping["iteration_index_base"],
            field_name="iteration_index_base",
        ),
        iteration_index=_expect_int(mapping["iteration_index"], field_name="iteration_index"),
        support_objective=_expect_float(
            mapping["support_objective"],
            field_name="support_objective",
        ),
        query_entropy_objective=_expect_float(
            mapping["query_entropy_objective"],
            field_name="query_entropy_objective",
        ),
        class_balance_objective=_expect_float(
            mapping["class_balance_objective"],
            field_name="class_balance_objective",
        ),
        consistency_objective=_expect_float(
            mapping["consistency_objective"],
            field_name="consistency_objective",
        ),
        proximal_objective=_expect_float(
            mapping["proximal_objective"],
            field_name="proximal_objective",
        ),
        total_objective=_expect_float(mapping["total_objective"], field_name="total_objective"),
        confident_voxel_count=_expect_int(
            mapping["confident_voxel_count"],
            field_name="confident_voxel_count",
        ),
        foreground_assignment_count=_expect_int(
            mapping["foreground_assignment_count"],
            field_name="foreground_assignment_count",
        ),
        background_assignment_count=_expect_int(
            mapping["background_assignment_count"],
            field_name="background_assignment_count",
        ),
        foreground_fraction=_expect_float(
            mapping["foreground_fraction"],
            field_name="foreground_fraction",
        ),
        state_identity_hash_before=_expect_string(
            mapping["state_identity_hash_before"],
            field_name="state_identity_hash_before",
        ),
        state_identity_hash_after=_expect_string(
            mapping["state_identity_hash_after"],
            field_name="state_identity_hash_after",
        ),
        prototype_identity_hash_before=_expect_string(
            mapping["prototype_identity_hash_before"],
            field_name="prototype_identity_hash_before",
        ),
        prototype_identity_hash_after=_expect_string(
            mapping["prototype_identity_hash_after"],
            field_name="prototype_identity_hash_after",
        ),
        convergence_delta=_expect_float(
            mapping["convergence_delta"],
            field_name="convergence_delta",
        ),
        finite_status_ok=_expect_bool(mapping["finite_status_ok"], field_name="finite_status_ok"),
        collapse_status_detected=_expect_bool(
            mapping["collapse_status_detected"],
            field_name="collapse_status_detected",
        ),
    )


def protoem_objective_trace_from_mapping(mapping: MappingLike) -> ProtoEMObjectiveTrace:
    """Reconstruct one ProtoEM objective trace from a strict mapping."""

    _require_exact_fields(
        mapping,
        required_fields=_OBJECTIVE_TRACE_FIELDS,
        object_name="ProtoEMObjectiveTrace",
    )
    iterations_value = mapping["iterations"]
    if not isinstance(iterations_value, list):
        raise ProtoEMArtifactSerializationError("ProtoEMObjectiveTrace.iterations must be a list.")
    return ProtoEMObjectiveTrace(
        schema_name=_expect_string(mapping["schema_name"], field_name="schema_name"),
        schema_version=_expect_string(mapping["schema_version"], field_name="schema_version"),
        objective_trace_hash=_expect_string(
            mapping["objective_trace_hash"],
            field_name="objective_trace_hash",
        ),
        iteration_index_base=_expect_string(
            mapping["iteration_index_base"],
            field_name="iteration_index_base",
        ),
        iterations=tuple(
            protoem_iteration_record_from_mapping(
                _expect_mapping(item, field_name=f"iterations[{index}]")
            )
            for index, item in enumerate(iterations_value)
        ),
    )


def protoem_stopping_record_from_mapping(mapping: MappingLike) -> ProtoEMStoppingRecord:
    """Reconstruct one ProtoEM stopping record from a strict mapping."""

    _require_exact_fields(
        mapping,
        required_fields=_STOPPING_RECORD_FIELDS,
        object_name="ProtoEMStoppingRecord",
    )
    return ProtoEMStoppingRecord(
        schema_name=_expect_string(mapping["schema_name"], field_name="schema_name"),
        schema_version=_expect_string(mapping["schema_version"], field_name="schema_version"),
        stopping_record_hash=_expect_string(
            mapping["stopping_record_hash"],
            field_name="stopping_record_hash",
        ),
        stop_reason=_expect_string(mapping["stop_reason"], field_name="stop_reason"),
        final_iteration_index=_expect_int(
            mapping["final_iteration_index"],
            field_name="final_iteration_index",
        ),
        completed_iteration_count=_expect_int(
            mapping["completed_iteration_count"],
            field_name="completed_iteration_count",
        ),
        converged=_expect_bool(mapping["converged"], field_name="converged"),
        failed=_expect_bool(mapping["failed"], field_name="failed"),
        failure_code=_expect_optional_string(mapping["failure_code"], field_name="failure_code"),
        failure_message=_expect_optional_string(
            mapping["failure_message"],
            field_name="failure_message",
        ),
    )


def protoem_collapse_record_from_mapping(mapping: MappingLike) -> ProtoEMCollapseRecord:
    """Reconstruct one ProtoEM collapse record from a strict mapping."""

    _require_exact_fields(
        mapping,
        required_fields=_COLLAPSE_RECORD_FIELDS,
        object_name="ProtoEMCollapseRecord",
    )
    return ProtoEMCollapseRecord(
        schema_name=_expect_string(mapping["schema_name"], field_name="schema_name"),
        schema_version=_expect_string(mapping["schema_version"], field_name="schema_version"),
        collapse_record_hash=_expect_string(
            mapping["collapse_record_hash"],
            field_name="collapse_record_hash",
        ),
        collapse_type=_expect_string(mapping["collapse_type"], field_name="collapse_type"),
        iteration_index=_expect_int(mapping["iteration_index"], field_name="iteration_index"),
        foreground_count=_expect_int(mapping["foreground_count"], field_name="foreground_count"),
        background_count=_expect_int(mapping["background_count"], field_name="background_count"),
        confident_voxel_count=_expect_int(
            mapping["confident_voxel_count"],
            field_name="confident_voxel_count",
        ),
        foreground_fraction=_expect_float(
            mapping["foreground_fraction"],
            field_name="foreground_fraction",
        ),
        configured_lower_safeguard=_expect_float(
            mapping["configured_lower_safeguard"],
            field_name="configured_lower_safeguard",
        ),
        configured_upper_safeguard=_expect_float(
            mapping["configured_upper_safeguard"],
            field_name="configured_upper_safeguard",
        ),
        detected=_expect_bool(mapping["detected"], field_name="detected"),
    )


def protoem_ablation_override_from_mapping(mapping: MappingLike) -> ProtoEMAblationOverride:
    """Reconstruct one ProtoEM ablation override from a strict mapping."""

    _require_exact_fields(
        mapping,
        required_fields=_ABLATION_OVERRIDE_FIELDS,
        object_name="ProtoEMAblationOverride",
    )
    override_value = mapping["override_value"]
    canonical_json_bytes(override_value)
    return ProtoEMAblationOverride(
        schema_name=_expect_string(mapping["schema_name"], field_name="schema_name"),
        schema_version=_expect_string(mapping["schema_version"], field_name="schema_version"),
        field_name=_expect_string(mapping["field_name"], field_name="field_name"),
        override_value=cast(JsonValue, override_value),
    )


def protoem_ablation_definition_from_mapping(mapping: MappingLike) -> ProtoEMAblationDefinition:
    """Reconstruct one ProtoEM ablation definition from a strict mapping."""

    _require_exact_fields(
        mapping,
        required_fields=_ABLATION_DEFINITION_FIELDS,
        object_name="ProtoEMAblationDefinition",
    )
    overrides_value = mapping["overrides"]
    if not isinstance(overrides_value, list):
        raise ProtoEMArtifactSerializationError(
            "ProtoEMAblationDefinition.overrides must be a list."
        )
    return ProtoEMAblationDefinition(
        schema_name=_expect_string(mapping["schema_name"], field_name="schema_name"),
        schema_version=_expect_string(mapping["schema_version"], field_name="schema_version"),
        ablation_definition_hash=_expect_string(
            mapping["ablation_definition_hash"],
            field_name="ablation_definition_hash",
        ),
        variant_name=_expect_string(mapping["variant_name"], field_name="variant_name"),
        overrides=tuple(
            protoem_ablation_override_from_mapping(
                _expect_mapping(item, field_name=f"overrides[{index}]")
            )
            for index, item in enumerate(overrides_value)
        ),
    )


def protoem_run_summary_from_mapping(mapping: MappingLike) -> ProtoEMRunSummary:
    """Reconstruct one ProtoEM run summary from a strict mapping."""

    _require_exact_fields(
        mapping,
        required_fields=_RUN_SUMMARY_FIELDS,
        object_name="ProtoEMRunSummary",
    )
    return ProtoEMRunSummary(
        schema_name=_expect_string(mapping["schema_name"], field_name="schema_name"),
        schema_version=_expect_string(mapping["schema_version"], field_name="schema_version"),
        run_summary_hash=_expect_string(mapping["run_summary_hash"], field_name="run_summary_hash"),
        config_hash=_expect_string(mapping["config_hash"], field_name="config_hash"),
        phase5_initialization_artifact_hash=_expect_string(
            mapping["phase5_initialization_artifact_hash"],
            field_name="phase5_initialization_artifact_hash",
        ),
        phase5_prototype_artifact_hash=_expect_string(
            mapping["phase5_prototype_artifact_hash"],
            field_name="phase5_prototype_artifact_hash",
        ),
        phase5_inference_artifact_hash=_expect_string(
            mapping["phase5_inference_artifact_hash"],
            field_name="phase5_inference_artifact_hash",
        ),
        objective_trace_hash=_expect_string(
            mapping["objective_trace_hash"],
            field_name="objective_trace_hash",
        ),
        stopping_record_hash=_expect_string(
            mapping["stopping_record_hash"],
            field_name="stopping_record_hash",
        ),
        collapse_record_hash=_expect_optional_string(
            mapping["collapse_record_hash"],
            field_name="collapse_record_hash",
        ),
        ablation_definition_hash=_expect_string(
            mapping["ablation_definition_hash"],
            field_name="ablation_definition_hash",
        ),
        final_output_artifact_hash=_expect_optional_string(
            mapping["final_output_artifact_hash"],
            field_name="final_output_artifact_hash",
        ),
        final_inference_artifact_hash=_expect_optional_string(
            mapping["final_inference_artifact_hash"],
            field_name="final_inference_artifact_hash",
        ),
        execution_status=_expect_string(
            mapping["execution_status"],
            field_name="execution_status",
        ),
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
        peak_allocated_memory_bytes=_expect_optional_int(
            mapping["peak_allocated_memory_bytes"],
            field_name="peak_allocated_memory_bytes",
        ),
        peak_reserved_memory_bytes=_expect_optional_int(
            mapping["peak_reserved_memory_bytes"],
            field_name="peak_reserved_memory_bytes",
        ),
        peak_host_memory_bytes=_expect_optional_int(
            mapping["peak_host_memory_bytes"],
            field_name="peak_host_memory_bytes",
        ),
    )


def protoem_config_to_json(config: ProtoEMConfig) -> bytes:
    """Serialize one ProtoEM config to canonical JSON bytes."""

    return canonical_json_bytes(protoem_config_to_dict(config)) + b"\n"


def protoem_objective_trace_to_json(trace: ProtoEMObjectiveTrace) -> bytes:
    """Serialize one ProtoEM objective trace to canonical JSON bytes."""

    return canonical_json_bytes(protoem_objective_trace_to_dict(trace)) + b"\n"


def protoem_stopping_record_to_json(record: ProtoEMStoppingRecord) -> bytes:
    """Serialize one ProtoEM stopping record to canonical JSON bytes."""

    return canonical_json_bytes(protoem_stopping_record_to_dict(record)) + b"\n"


def protoem_collapse_record_to_json(record: ProtoEMCollapseRecord) -> bytes:
    """Serialize one ProtoEM collapse record to canonical JSON bytes."""

    return canonical_json_bytes(protoem_collapse_record_to_dict(record)) + b"\n"


def protoem_ablation_definition_to_json(definition: ProtoEMAblationDefinition) -> bytes:
    """Serialize one ProtoEM ablation definition to canonical JSON bytes."""

    return canonical_json_bytes(protoem_ablation_definition_to_dict(definition)) + b"\n"


def protoem_run_summary_to_json(summary: ProtoEMRunSummary) -> bytes:
    """Serialize one ProtoEM run summary to canonical JSON bytes."""

    return canonical_json_bytes(protoem_run_summary_to_dict(summary)) + b"\n"


def protoem_config_from_json(payload: bytes | str) -> ProtoEMConfig:
    """Parse one ProtoEM config from canonical JSON bytes or text."""

    return protoem_config_from_mapping(_parse_json_mapping(payload, object_name="ProtoEMConfig"))


def protoem_objective_trace_from_json(payload: bytes | str) -> ProtoEMObjectiveTrace:
    """Parse one ProtoEM objective trace from canonical JSON bytes or text."""

    return protoem_objective_trace_from_mapping(
        _parse_json_mapping(payload, object_name="ProtoEMObjectiveTrace")
    )


def protoem_stopping_record_from_json(payload: bytes | str) -> ProtoEMStoppingRecord:
    """Parse one ProtoEM stopping record from canonical JSON bytes or text."""

    return protoem_stopping_record_from_mapping(
        _parse_json_mapping(payload, object_name="ProtoEMStoppingRecord")
    )


def protoem_collapse_record_from_json(payload: bytes | str) -> ProtoEMCollapseRecord:
    """Parse one ProtoEM collapse record from canonical JSON bytes or text."""

    return protoem_collapse_record_from_mapping(
        _parse_json_mapping(payload, object_name="ProtoEMCollapseRecord")
    )


def protoem_ablation_definition_from_json(payload: bytes | str) -> ProtoEMAblationDefinition:
    """Parse one ProtoEM ablation definition from canonical JSON bytes or text."""

    return protoem_ablation_definition_from_mapping(
        _parse_json_mapping(payload, object_name="ProtoEMAblationDefinition")
    )


def protoem_run_summary_from_json(payload: bytes | str) -> ProtoEMRunSummary:
    """Parse one ProtoEM run summary from canonical JSON bytes or text."""

    return protoem_run_summary_from_mapping(
        _parse_json_mapping(payload, object_name="ProtoEMRunSummary")
    )


def protoem_artifact_to_dict(artifact: ProtoEMArtifact) -> dict[str, JsonValue]:
    """Convert a ProtoEM artifact to a canonical mapping."""

    if isinstance(artifact, ProtoEMConfig):
        return protoem_config_to_dict(artifact)
    if isinstance(artifact, ProtoEMObjectiveTrace):
        return protoem_objective_trace_to_dict(artifact)
    if isinstance(artifact, ProtoEMStoppingRecord):
        return protoem_stopping_record_to_dict(artifact)
    if isinstance(artifact, ProtoEMCollapseRecord):
        return protoem_collapse_record_to_dict(artifact)
    if isinstance(artifact, ProtoEMAblationDefinition):
        return protoem_ablation_definition_to_dict(artifact)
    if isinstance(artifact, ProtoEMRunSummary):
        return protoem_run_summary_to_dict(artifact)
    raise TypeError(f"Unsupported ProtoEM artifact type: {type(artifact).__name__}")


def protoem_artifact_to_json(artifact: ProtoEMArtifact) -> bytes:
    """Serialize one ProtoEM artifact to canonical JSON bytes."""

    return canonical_json_bytes(protoem_artifact_to_dict(artifact)) + b"\n"


def protoem_artifact_from_json(
    payload: bytes | str, artifact_type: type[_ArtifactType]
) -> _ArtifactType:
    """Parse one ProtoEM artifact of the requested type from canonical JSON."""

    if artifact_type is ProtoEMConfig:
        return protoem_config_from_json(payload)  # type: ignore[return-value]
    if artifact_type is ProtoEMObjectiveTrace:
        return protoem_objective_trace_from_json(payload)  # type: ignore[return-value]
    if artifact_type is ProtoEMStoppingRecord:
        return protoem_stopping_record_from_json(payload)  # type: ignore[return-value]
    if artifact_type is ProtoEMCollapseRecord:
        return protoem_collapse_record_from_json(payload)  # type: ignore[return-value]
    if artifact_type is ProtoEMAblationDefinition:
        return protoem_ablation_definition_from_json(payload)  # type: ignore[return-value]
    if artifact_type is ProtoEMRunSummary:
        return protoem_run_summary_from_json(payload)  # type: ignore[return-value]
    raise TypeError(f"Unsupported ProtoEM artifact type: {artifact_type!r}")


def _require_schema_name(value: str, *, expected: str, field_name: str) -> None:
    if value != expected:
        raise ProtoEMArtifactValidationError(f"{field_name} must equal {expected!r}.")


def _require_schema_version(value: str, *, expected: str, field_name: str) -> None:
    if value != expected:
        raise ProtoEMArtifactValidationError(f"{field_name} must equal {expected!r}.")


def _require_sha256(value: str, *, field_name: str) -> None:
    if not _SHA256_RE.fullmatch(value):
        raise ProtoEMArtifactValidationError(
            f"{field_name} must be a lowercase 64-character SHA-256 hex digest."
        )


def _require_optional_sha256(value: str | None, *, field_name: str) -> None:
    if value is None:
        return
    _require_sha256(value, field_name=field_name)


def _require_identifier(value: str, *, field_name: str) -> None:
    if not _IDENTIFIER_RE.fullmatch(value):
        raise ProtoEMArtifactValidationError(
            f"{field_name} must match the conservative identifier pattern."
        )


def _require_phase5_schema_reference(value: str, *, field_name: str) -> None:
    _require_identifier(value, field_name=field_name)
    if value not in SUPPORTED_PROTOEM_PHASE5_SCHEMA_REFERENCES:
        raise ProtoEMArtifactValidationError(
            f"{field_name} must be one of {sorted(SUPPORTED_PROTOEM_PHASE5_SCHEMA_REFERENCES)!r}."
        )


def _require_seed(value: int, *, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ProtoEMArtifactValidationError(f"{field_name} must be a nonnegative integer seed.")


def _require_positive_int(value: int, *, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ProtoEMArtifactValidationError(f"{field_name} must be a positive integer.")


def _require_nonnegative_int(value: int, *, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ProtoEMArtifactValidationError(f"{field_name} must be a nonnegative integer.")


def _require_finite_float(value: float, *, field_name: str) -> None:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ProtoEMArtifactValidationError(f"{field_name} must be a finite number.")
    if not math.isfinite(float(value)):
        raise ProtoEMArtifactValidationError(f"{field_name} must be finite.")


def _require_nonnegative_finite_float(value: float, *, field_name: str) -> None:
    _require_finite_float(value, field_name=field_name)
    if float(value) < 0.0:
        raise ProtoEMArtifactValidationError(f"{field_name} must be nonnegative.")


def _require_positive_finite_float(value: float, *, field_name: str) -> None:
    _require_finite_float(value, field_name=field_name)
    if float(value) <= 0.0:
        raise ProtoEMArtifactValidationError(f"{field_name} must be positive.")


def _require_fraction_closed(value: float, *, field_name: str) -> None:
    _require_finite_float(value, field_name=field_name)
    if not 0.0 <= float(value) <= 1.0:
        raise ProtoEMArtifactValidationError(
            f"{field_name} must lie in the closed interval [0, 1]."
        )


def _require_fraction_open(value: float, *, field_name: str) -> None:
    _require_finite_float(value, field_name=field_name)
    if not 0.0 < float(value) < 1.0:
        raise ProtoEMArtifactValidationError(f"{field_name} must lie in the open interval (0, 1).")


def _require_optional_nonnegative_finite_float(value: float | None, *, field_name: str) -> None:
    if value is None:
        return
    _require_nonnegative_finite_float(value, field_name=field_name)


def _require_failure_consistency(
    *,
    failed: bool,
    failure_code: str | None,
    failure_message: str | None,
    field_prefix: str,
) -> None:
    if failed:
        if failure_code is None or failure_message is None:
            raise ProtoEMArtifactValidationError(
                f"{field_prefix} failed state requires both failure_code and failure_message."
            )
        _require_identifier(failure_code, field_name="failure_code")
        if not _MESSAGE_RE.fullmatch(failure_message):
            raise ProtoEMArtifactValidationError(
                "failure_message must be a single-line nonempty message."
            )
        return
    if failure_code is not None or failure_message is not None:
        raise ProtoEMArtifactValidationError(
            f"{field_prefix} non-failed state must not include failure_code or failure_message."
        )


def _expect_mapping(value: object, *, field_name: str) -> MappingLike:
    if not isinstance(value, dict):
        raise ProtoEMArtifactSerializationError(f"{field_name} must be a mapping object.")
    if not all(isinstance(key, str) for key in value):
        raise ProtoEMArtifactSerializationError(f"{field_name} must use only string keys.")
    return value


def _expect_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ProtoEMArtifactSerializationError(f"{field_name} must be a string.")
    return value


def _expect_optional_string(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _expect_string(value, field_name=field_name)


def _expect_bool(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ProtoEMArtifactSerializationError(f"{field_name} must be a boolean.")
    return value


def _expect_int(value: object, *, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ProtoEMArtifactSerializationError(f"{field_name} must be an integer.")
    return value


def _expect_optional_int(value: object, *, field_name: str) -> int | None:
    if value is None:
        return None
    return _expect_int(value, field_name=field_name)


def _expect_float(value: object, *, field_name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ProtoEMArtifactSerializationError(f"{field_name} must be a finite number.")
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ProtoEMArtifactSerializationError(f"{field_name} must be finite.")
    return numeric_value


def _expect_optional_float(value: object, *, field_name: str) -> float | None:
    if value is None:
        return None
    return _expect_float(value, field_name=field_name)


def _require_exact_fields(
    mapping: MappingLike,
    *,
    required_fields: frozenset[str],
    object_name: str,
) -> None:
    unknown_fields = set(mapping) - required_fields
    if unknown_fields:
        names = ", ".join(sorted(unknown_fields))
        raise ProtoEMArtifactSerializationError(f"Unknown {object_name} field(s): {names}.")
    missing_fields = required_fields - set(mapping)
    if missing_fields:
        names = ", ".join(sorted(missing_fields))
        raise ProtoEMArtifactSerializationError(f"Missing {object_name} field(s): {names}.")


def _parse_json_mapping(payload: bytes | str, *, object_name: str) -> MappingLike:
    try:
        decoded = payload.decode("utf-8") if isinstance(payload, bytes) else payload
        value = json.loads(decoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtoEMArtifactSerializationError(f"Invalid {object_name} JSON payload.") from exc
    return _expect_mapping(value, field_name=object_name)


__all__ = [
    "COMPATIBILITY_CRITICAL_CONFIG_OVERRIDE_FIELDS",
    "PROTOEM_ABLATION_DEFINITION_SCHEMA_NAME",
    "PROTOEM_ABLATION_DEFINITION_SCHEMA_VERSION",
    "PROTOEM_ABLATION_OVERRIDE_SCHEMA_NAME",
    "PROTOEM_ABLATION_OVERRIDE_SCHEMA_VERSION",
    "PROTOEM_COLLAPSE_RECORD_SCHEMA_NAME",
    "PROTOEM_COLLAPSE_RECORD_SCHEMA_VERSION",
    "PROTOEM_CONFIG_SCHEMA_NAME",
    "PROTOEM_CONFIG_SCHEMA_VERSION",
    "PROTOEM_ITERATION_INDEX_BASE",
    "PROTOEM_ITERATION_RECORD_SCHEMA_NAME",
    "PROTOEM_ITERATION_RECORD_SCHEMA_VERSION",
    "PROTOEM_OBJECTIVE_TRACE_SCHEMA_NAME",
    "PROTOEM_OBJECTIVE_TRACE_SCHEMA_VERSION",
    "PROTOEM_OBJECTIVE_WEIGHTS_SCHEMA_NAME",
    "PROTOEM_OBJECTIVE_WEIGHTS_SCHEMA_VERSION",
    "PROTOEM_RUN_SUMMARY_SCHEMA_NAME",
    "PROTOEM_RUN_SUMMARY_SCHEMA_VERSION",
    "PROTOEM_STOPPING_RECORD_SCHEMA_NAME",
    "PROTOEM_STOPPING_RECORD_SCHEMA_VERSION",
    "SUPPORTED_PROTOEM_ABLATION_VARIANTS",
    "SUPPORTED_PROTOEM_COLLAPSE_TYPES",
    "SUPPORTED_PROTOEM_EXECUTION_STATUSES",
    "SUPPORTED_PROTOEM_MEMORY_AVAILABILITY_STATUSES",
    "SUPPORTED_PROTOEM_PHASE5_SCHEMA_REFERENCES",
    "SUPPORTED_PROTOEM_PROTOTYPE_MODES",
    "SUPPORTED_PROTOEM_STOP_REASONS",
    "SUPPORTED_PROTOEM_UPDATE_SCHEDULES",
    "ProtoEMAblationDefinition",
    "ProtoEMAblationOverride",
    "ProtoEMArtifact",
    "ProtoEMArtifactError",
    "ProtoEMArtifactHashError",
    "ProtoEMArtifactSerializationError",
    "ProtoEMArtifactValidationError",
    "ProtoEMCollapseRecord",
    "ProtoEMConfig",
    "ProtoEMIterationRecord",
    "ProtoEMObjectiveTrace",
    "ProtoEMObjectiveWeights",
    "ProtoEMRunSummary",
    "ProtoEMStoppingRecord",
    "hash_protoem_ablation_definition",
    "hash_protoem_collapse_record",
    "hash_protoem_config",
    "hash_protoem_objective_trace",
    "hash_protoem_run_summary",
    "hash_protoem_stopping_record",
    "protoem_ablation_definition_from_json",
    "protoem_ablation_definition_from_mapping",
    "protoem_ablation_definition_identity_payload",
    "protoem_ablation_definition_to_dict",
    "protoem_ablation_definition_to_json",
    "protoem_artifact_from_json",
    "protoem_artifact_to_dict",
    "protoem_artifact_to_json",
    "protoem_collapse_record_from_json",
    "protoem_collapse_record_from_mapping",
    "protoem_collapse_record_identity_payload",
    "protoem_collapse_record_to_dict",
    "protoem_collapse_record_to_json",
    "protoem_config_from_json",
    "protoem_config_from_mapping",
    "protoem_config_identity_payload",
    "protoem_config_to_dict",
    "protoem_config_to_json",
    "protoem_objective_trace_from_json",
    "protoem_objective_trace_from_mapping",
    "protoem_objective_trace_identity_payload",
    "protoem_objective_trace_to_dict",
    "protoem_objective_trace_to_json",
    "protoem_objective_weights_from_mapping",
    "protoem_objective_weights_to_dict",
    "protoem_run_summary_from_json",
    "protoem_run_summary_from_mapping",
    "protoem_run_summary_identity_payload",
    "protoem_run_summary_to_dict",
    "protoem_run_summary_to_json",
    "protoem_stopping_record_from_json",
    "protoem_stopping_record_from_mapping",
    "protoem_stopping_record_identity_payload",
    "protoem_stopping_record_to_dict",
    "protoem_stopping_record_to_json",
]
