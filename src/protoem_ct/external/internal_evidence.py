"""Phase 8 real-development internal evidence contracts.

The contracts in this module are metadata-only readiness inputs for Phase 8
Wave 4 remediation.  They perform no filesystem I/O, dataset access, model
execution, or scientific selection.  Checkpoint file verification is exposed as
a pure input check over already supplied digest and byte-size values.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, TypeAlias, cast

from protoem_ct.artifacts.hashing import JsonValue, canonical_json_bytes, sha256_json

PHASE8_PREPROCESSING_DECISION_SCHEMA_NAME: Final[str] = "phase8_preprocessing_decision"
PHASE8_FIXED_CANDIDATE_INVENTORY_SCHEMA_NAME: Final[str] = "phase8_fixed_candidate_inventory"
PHASE8_CHECKPOINT_METADATA_SCHEMA_NAME: Final[str] = "phase8_checkpoint_metadata"
PHASE8_VALIDATION_EVIDENCE_REFERENCE_SCHEMA_NAME: Final[str] = (
    "phase8_validation_evidence_reference"
)
PHASE8_MODEL_SELECTION_DECISION_SCHEMA_NAME: Final[str] = "phase8_model_selection_decision"
PHASE8_THRESHOLD_DECISION_SCHEMA_NAME: Final[str] = "phase8_threshold_decision"
PHASE8_SUPPORT_POLICY_SCHEMA_NAME: Final[str] = "phase8_support_policy"
PHASE8_INTERNAL_EVIDENCE_PACKAGE_SCHEMA_NAME: Final[str] = "phase8_internal_evidence_package"
PHASE8_INTERNAL_EVIDENCE_SCHEMA_VERSION: Final[str] = "v1"

MappingLike: TypeAlias = Mapping[str, object]

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[a-z0-9](?:[a-z0-9_.-]{0,126}[a-z0-9])?$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{7,64}$")
_ZERO_SHA256: Final[str] = "0" * 64
_UNRESOLVED_TOKENS: Final[frozenset[str]] = frozenset(
    {"pending", "placeholder", "tbd", "todo", "unknown", "unresolved"}
)
_FIT_SCOPES: Final[frozenset[str]] = frozenset({"train_only", "no_data_dependent_fit"})
_EVIDENCE_STATUSES: Final[frozenset[str]] = frozenset({"unresolved", "resolved", "freeze_ready"})
_EXECUTABLE_STATES: Final[frozenset[str]] = frozenset(
    {"executable", "not_executable", "failed", "planned"}
)
_FAILURE_POLICIES: Final[frozenset[str]] = frozenset(
    {"record_and_exclude", "block_if_all_fail", "single_candidate_no_comparison"}
)
_TRAINING_MODES: Final[frozenset[str]] = frozenset(
    {
        "baseline_training",
        "head_only",
        "decoder_only",
        "full_finetune",
        "prototype_only",
        "protoem_fixed_em_like",
    }
)
_CHECKPOINT_STATUSES: Final[frozenset[str]] = frozenset(
    {"completed", "failed", "synthetic_smoke", "unresolved"}
)
_COMPLETION_STATUSES: Final[frozenset[str]] = frozenset({"completed", "failed", "unresolved"})
_SELECTION_STATUSES: Final[frozenset[str]] = frozenset(
    {"unresolved", "selected", "freeze_ready", "single_candidate_preregistered"}
)
_THRESHOLD_POLICIES: Final[frozenset[str]] = frozenset(
    {"fixed_constant", "development_validation_selected"}
)
_SUPPORT_POLICIES: Final[frozenset[str]] = frozenset({"no_support", "internal_only_support"})
_FREEZE_STATUSES: Final[frozenset[str]] = frozenset({"unresolved", "resolved", "freeze_ready"})
_READINESS_STATES: Final[frozenset[str]] = frozenset({"BLOCKED", "READY"})


class Phase8InternalEvidenceError(ValueError):
    """Base error for Phase 8 internal-evidence contracts."""


class Phase8InternalEvidenceValidationError(Phase8InternalEvidenceError):
    """Raised when internal evidence violates the contract."""


class Phase8InternalEvidenceSerializationError(Phase8InternalEvidenceError):
    """Raised when an evidence mapping cannot be reconstructed strictly."""


class Phase8InternalEvidenceHashError(Phase8InternalEvidenceError):
    """Raised when an evidence self-hash or linked hash is invalid."""


@dataclass(frozen=True, slots=True)
class ArtifactReference:
    """Stable artifact reference that excludes local filesystem identity."""

    schema_name: str
    schema_version: str
    artifact_hash: str
    artifact_role: str

    def __post_init__(self) -> None:
        _require_identifier(self.schema_name, field_name="schema_name")
        _require_schema_version(self.schema_version, field_name="schema_version")
        _require_sha256(self.artifact_hash, field_name="artifact_hash")
        _require_identifier(self.artifact_role, field_name="artifact_role")


@dataclass(frozen=True, slots=True)
class Phase8PreprocessingDecision:
    """Immutable preprocessing evidence for a real-development candidate."""

    schema_name: str
    schema_version: str
    candidate_id: str
    development_manifest_hash: str
    development_split_hash: str
    preprocessing_config_schema_name: str
    preprocessing_config_schema_version: str
    preprocessing_config_hash: str
    fit_scope: str
    fit_artifact_reference: ArtifactReference | None
    image_interpolation_policy: str
    label_interpolation_policy: str
    orientation_policy_reference: ArtifactReference
    spacing_policy_reference: ArtifactReference
    intensity_policy_reference: ArtifactReference
    crop_roi_policy_reference: ArtifactReference
    normalization_policy_reference: ArtifactReference
    originating_git_commit: str
    no_external_data: bool
    evidence_status: str
    preprocessing_decision_hash: str

    def __post_init__(self) -> None:
        _require_schema(self.schema_name, PHASE8_PREPROCESSING_DECISION_SCHEMA_NAME)
        _require_schema_version_exact(self.schema_version)
        _require_identifier(self.candidate_id, field_name="candidate_id")
        _require_sha256(self.development_manifest_hash, field_name="development_manifest_hash")
        _require_sha256(self.development_split_hash, field_name="development_split_hash")
        _require_identifier(
            self.preprocessing_config_schema_name,
            field_name="preprocessing_config_schema_name",
        )
        _require_schema_version(
            self.preprocessing_config_schema_version,
            field_name="preprocessing_config_schema_version",
        )
        _require_sha256(self.preprocessing_config_hash, field_name="preprocessing_config_hash")
        _require_allowed(self.fit_scope, _FIT_SCOPES, field_name="fit_scope")
        _require_identifier(
            self.image_interpolation_policy,
            field_name="image_interpolation_policy",
            allow_unresolved=True,
        )
        _require_identifier(
            self.label_interpolation_policy,
            field_name="label_interpolation_policy",
            allow_unresolved=True,
        )
        _require_git_commit(self.originating_git_commit, field_name="originating_git_commit")
        _require_allowed(self.evidence_status, _EVIDENCE_STATUSES, field_name="evidence_status")
        _require_true(self.no_external_data, field_name="no_external_data")
        if self.fit_scope == "no_data_dependent_fit" and self.fit_artifact_reference is not None:
            raise Phase8InternalEvidenceValidationError(
                "no_data_dependent_fit preprocessing must not include a fit artifact."
            )
        if self.fit_scope == "train_only" and self.fit_artifact_reference is None:
            raise Phase8InternalEvidenceValidationError(
                "train_only preprocessing requires a fit artifact reference."
            )
        if self.evidence_status == "freeze_ready":
            _require_resolved_references(
                self.orientation_policy_reference,
                self.spacing_policy_reference,
                self.intensity_policy_reference,
                self.crop_roi_policy_reference,
                self.normalization_policy_reference,
            )
        _require_self_hash(
            self.preprocessing_decision_hash,
            hash_phase8_preprocessing_decision(self),
            field_name="preprocessing_decision_hash",
        )


@dataclass(frozen=True, slots=True)
class CandidateDefinition:
    """One immutable candidate in the fixed Phase 8 development inventory."""

    candidate_id: str
    model_family: str
    method_identity: str
    training_adaptation_mode: str
    training_config_reference: ArtifactReference
    preprocessing_evidence_hash: str
    fixed_seeds: tuple[int, ...]
    executable_readiness_state: str
    failure_handling_policy: str
    uses_external_artifacts: bool

    def __post_init__(self) -> None:
        _require_identifier(self.candidate_id, field_name="candidate_id")
        _require_identifier(self.model_family, field_name="model_family")
        _require_identifier(self.method_identity, field_name="method_identity")
        _require_allowed(
            self.training_adaptation_mode,
            _TRAINING_MODES,
            field_name="training_adaptation_mode",
        )
        _require_sha256(self.preprocessing_evidence_hash, field_name="preprocessing_evidence_hash")
        object.__setattr__(self, "fixed_seeds", tuple(self.fixed_seeds))
        if not self.fixed_seeds:
            raise Phase8InternalEvidenceValidationError("fixed_seeds must not be empty.")
        for index, seed in enumerate(self.fixed_seeds):
            _require_nonnegative_int(seed, field_name=f"fixed_seeds[{index}]")
        _require_allowed(
            self.executable_readiness_state,
            _EXECUTABLE_STATES,
            field_name="executable_readiness_state",
        )
        _require_allowed(
            self.failure_handling_policy,
            _FAILURE_POLICIES,
            field_name="failure_handling_policy",
        )
        _require_false(self.uses_external_artifacts, field_name="uses_external_artifacts")

    @property
    def ordering_key(self) -> str:
        """Return the immutable deterministic candidate ordering key."""

        return self.candidate_id


@dataclass(frozen=True, slots=True)
class Phase8FixedCandidateInventory:
    """Immutable inventory of candidates fixed before selection starts."""

    schema_name: str
    schema_version: str
    inventory_state: str
    candidate_set_locked: bool
    selection_started: bool
    supports_single_candidate_without_comparison: bool
    candidates: tuple[CandidateDefinition, ...]
    inventory_hash: str

    def __post_init__(self) -> None:
        _require_schema(self.schema_name, PHASE8_FIXED_CANDIDATE_INVENTORY_SCHEMA_NAME)
        _require_schema_version_exact(self.schema_version)
        _require_allowed(self.inventory_state, _FREEZE_STATUSES, field_name="inventory_state")
        object.__setattr__(
            self,
            "candidates",
            tuple(sorted(self.candidates, key=lambda candidate: candidate.ordering_key)),
        )
        if not self.candidates:
            raise Phase8InternalEvidenceValidationError("candidate inventory must not be empty.")
        ids = [candidate.candidate_id for candidate in self.candidates]
        if len(ids) != len(set(ids)):
            raise Phase8InternalEvidenceValidationError(
                "candidate inventory contains duplicate candidate IDs."
            )
        if self.selection_started and not self.candidate_set_locked:
            raise Phase8InternalEvidenceValidationError(
                "candidate set cannot be unlocked after selection starts."
            )
        if len(self.candidates) == 1 and not self.supports_single_candidate_without_comparison:
            raise Phase8InternalEvidenceValidationError(
                "single-candidate inventories must explicitly avoid comparison claims."
            )
        if self.inventory_state == "freeze_ready":
            _require_true(self.candidate_set_locked, field_name="candidate_set_locked")
            for candidate in self.candidates:
                if candidate.executable_readiness_state != "executable":
                    raise Phase8InternalEvidenceValidationError(
                        "freeze_ready inventory requires every candidate to be executable."
                    )
        _require_self_hash(self.inventory_hash, hash_phase8_fixed_candidate_inventory(self))


@dataclass(frozen=True, slots=True)
class Phase8CheckpointMetadata:
    """Immutable metadata identity for one checkpoint without local paths."""

    schema_name: str
    schema_version: str
    checkpoint_sha256: str
    checkpoint_byte_size: int
    serialization_format: str
    model_family: str
    candidate_id: str
    seed: int
    training_adaptation_mode: str
    originating_git_commit: str
    package_environment_reference: ArtifactReference
    training_config_hash: str
    preprocessing_evidence_hash: str
    development_manifest_hash: str
    development_split_hash: str
    validation_metric_artifact_reference: ArtifactReference | None
    device_type: str
    amp_state: str
    completion_status: str
    failure_codes: tuple[str, ...]
    no_external_data: bool
    freeze_eligible: bool
    checkpoint_metadata_hash: str

    def __post_init__(self) -> None:
        _require_schema(self.schema_name, PHASE8_CHECKPOINT_METADATA_SCHEMA_NAME)
        _require_schema_version_exact(self.schema_version)
        _require_sha256(self.checkpoint_sha256, field_name="checkpoint_sha256")
        _require_positive_int(self.checkpoint_byte_size, field_name="checkpoint_byte_size")
        _require_identifier(self.serialization_format, field_name="serialization_format")
        _require_identifier(self.model_family, field_name="model_family")
        _require_identifier(self.candidate_id, field_name="candidate_id")
        _require_nonnegative_int(self.seed, field_name="seed")
        _require_allowed(
            self.training_adaptation_mode,
            _TRAINING_MODES,
            field_name="training_adaptation_mode",
        )
        _require_git_commit(self.originating_git_commit, field_name="originating_git_commit")
        _require_sha256(self.training_config_hash, field_name="training_config_hash")
        _require_sha256(self.preprocessing_evidence_hash, field_name="preprocessing_evidence_hash")
        _require_sha256(self.development_manifest_hash, field_name="development_manifest_hash")
        _require_sha256(self.development_split_hash, field_name="development_split_hash")
        _require_identifier(self.device_type, field_name="device_type", allow_unresolved=True)
        _require_identifier(self.amp_state, field_name="amp_state", allow_unresolved=True)
        _require_allowed(
            self.completion_status,
            _CHECKPOINT_STATUSES,
            field_name="completion_status",
        )
        object.__setattr__(self, "failure_codes", tuple(sorted(self.failure_codes)))
        for index, code in enumerate(self.failure_codes):
            _require_identifier(code, field_name=f"failure_codes[{index}]", allow_unresolved=True)
        _require_true(self.no_external_data, field_name="no_external_data")
        if self.completion_status == "failed" and not self.failure_codes:
            raise Phase8InternalEvidenceValidationError(
                "failed checkpoint metadata requires failure_codes."
            )
        if self.completion_status != "failed" and self.failure_codes:
            raise Phase8InternalEvidenceValidationError(
                "non-failed checkpoint metadata must not include failure_codes."
            )
        if self.freeze_eligible:
            if self.completion_status != "completed":
                raise Phase8InternalEvidenceValidationError(
                    "freeze_eligible checkpoint metadata requires completed status."
                )
            if self.validation_metric_artifact_reference is None:
                raise Phase8InternalEvidenceValidationError(
                    "freeze_eligible checkpoint metadata requires validation provenance."
                )
        _require_self_hash(
            self.checkpoint_metadata_hash,
            hash_phase8_checkpoint_metadata(self),
            field_name="checkpoint_metadata_hash",
        )


@dataclass(frozen=True, slots=True)
class Phase8ValidationEvidenceReference:
    """Development-validation evidence link without metric values."""

    schema_name: str
    schema_version: str
    candidate_id: str
    checkpoint_hash: str
    validation_cohort_split_hash: str
    metric_artifact_schema_name: str
    metric_artifact_schema_version: str
    metric_artifact_hash: str
    primary_metric_name: str
    valid_case_count: int
    empty_target_accounting_reference: ArtifactReference
    development_validation_only: bool
    internal_test_not_used: bool
    external_data_not_used: bool
    validation_evidence_hash: str

    def __post_init__(self) -> None:
        _require_schema(self.schema_name, PHASE8_VALIDATION_EVIDENCE_REFERENCE_SCHEMA_NAME)
        _require_schema_version_exact(self.schema_version)
        _require_identifier(self.candidate_id, field_name="candidate_id")
        _require_sha256(self.checkpoint_hash, field_name="checkpoint_hash")
        _require_sha256(
            self.validation_cohort_split_hash,
            field_name="validation_cohort_split_hash",
        )
        _require_identifier(
            self.metric_artifact_schema_name, field_name="metric_artifact_schema_name"
        )
        _require_schema_version(
            self.metric_artifact_schema_version,
            field_name="metric_artifact_schema_version",
        )
        _require_sha256(self.metric_artifact_hash, field_name="metric_artifact_hash")
        _require_identifier(self.primary_metric_name, field_name="primary_metric_name")
        _require_positive_int(self.valid_case_count, field_name="valid_case_count")
        _require_true(self.development_validation_only, field_name="development_validation_only")
        _require_true(self.internal_test_not_used, field_name="internal_test_not_used")
        _require_true(self.external_data_not_used, field_name="external_data_not_used")
        _require_self_hash(
            self.validation_evidence_hash,
            hash_phase8_validation_evidence_reference(self),
            field_name="validation_evidence_hash",
        )


@dataclass(frozen=True, slots=True)
class FailedCandidateAccounting:
    """Accounting for one failed or excluded candidate during selection."""

    candidate_id: str
    failure_code: str
    checkpoint_hash: str | None
    validation_evidence_hash: str | None

    def __post_init__(self) -> None:
        _require_identifier(self.candidate_id, field_name="candidate_id")
        _require_identifier(self.failure_code, field_name="failure_code")
        _require_optional_sha256(self.checkpoint_hash, field_name="checkpoint_hash")
        _require_optional_sha256(
            self.validation_evidence_hash,
            field_name="validation_evidence_hash",
        )


@dataclass(frozen=True, slots=True)
class Phase8ModelSelectionDecision:
    """Immutable development-only model-selection decision."""

    schema_name: str
    schema_version: str
    fixed_candidate_inventory_hash: str
    selection_rule_version: str
    primary_metric: str
    deterministic_tie_break_order: tuple[str, ...]
    selected_candidate_id: str | None
    selected_checkpoint_hash: str | None
    selected_validation_evidence_hash: str | None
    failed_candidate_accounting: tuple[FailedCandidateAccounting, ...]
    selection_status: str
    internal_test_not_used: bool
    external_data_not_used: bool
    model_selection_decision_hash: str

    def __post_init__(self) -> None:
        _require_schema(self.schema_name, PHASE8_MODEL_SELECTION_DECISION_SCHEMA_NAME)
        _require_schema_version_exact(self.schema_version)
        _require_sha256(
            self.fixed_candidate_inventory_hash,
            field_name="fixed_candidate_inventory_hash",
        )
        _require_identifier(self.selection_rule_version, field_name="selection_rule_version")
        _require_identifier(self.primary_metric, field_name="primary_metric")
        object.__setattr__(
            self,
            "deterministic_tie_break_order",
            tuple(self.deterministic_tie_break_order),
        )
        if not self.deterministic_tie_break_order:
            raise Phase8InternalEvidenceValidationError(
                "deterministic_tie_break_order must not be empty."
            )
        for index, rule in enumerate(self.deterministic_tie_break_order):
            _require_identifier(rule, field_name=f"deterministic_tie_break_order[{index}]")
        _require_optional_identifier(
            self.selected_candidate_id,
            field_name="selected_candidate_id",
        )
        _require_optional_sha256(
            self.selected_checkpoint_hash,
            field_name="selected_checkpoint_hash",
        )
        _require_optional_sha256(
            self.selected_validation_evidence_hash,
            field_name="selected_validation_evidence_hash",
        )
        object.__setattr__(
            self,
            "failed_candidate_accounting",
            tuple(
                sorted(
                    self.failed_candidate_accounting,
                    key=lambda item: (item.candidate_id, item.failure_code),
                )
            ),
        )
        _require_allowed(self.selection_status, _SELECTION_STATUSES, field_name="selection_status")
        _require_true(self.internal_test_not_used, field_name="internal_test_not_used")
        _require_true(self.external_data_not_used, field_name="external_data_not_used")
        if self.selection_status == "unresolved":
            if (
                self.selected_candidate_id is not None
                or self.selected_checkpoint_hash is not None
                or self.selected_validation_evidence_hash is not None
            ):
                raise Phase8InternalEvidenceValidationError(
                    "unresolved selection must not include selected candidate evidence."
                )
        else:
            if (
                self.selected_candidate_id is None
                or self.selected_checkpoint_hash is None
                or self.selected_validation_evidence_hash is None
            ):
                raise Phase8InternalEvidenceValidationError(
                    "resolved selection requires selected candidate, checkpoint, "
                    "and validation evidence."
                )
        _require_self_hash(
            self.model_selection_decision_hash,
            hash_phase8_model_selection_decision(self),
            field_name="model_selection_decision_hash",
        )


@dataclass(frozen=True, slots=True)
class Phase8ThresholdDecision:
    """Immutable threshold policy without selecting a threshold prematurely."""

    schema_name: str
    schema_version: str
    policy_type: str
    threshold_value: float | None
    candidate_id: str | None
    checkpoint_hash: str | None
    selection_objective: str | None
    candidate_threshold_grid_reference: ArtifactReference | None
    development_metric_reference: ArtifactReference | None
    fixed_constant_rationale: str | None
    development_only_provenance: bool
    internal_test_not_used: bool
    external_data_not_used: bool
    freeze_status: str
    threshold_decision_hash: str

    def __post_init__(self) -> None:
        _require_schema(self.schema_name, PHASE8_THRESHOLD_DECISION_SCHEMA_NAME)
        _require_schema_version_exact(self.schema_version)
        _require_allowed(self.policy_type, _THRESHOLD_POLICIES, field_name="policy_type")
        _require_optional_probability(self.threshold_value, field_name="threshold_value")
        _require_optional_identifier(self.candidate_id, field_name="candidate_id")
        _require_optional_sha256(self.checkpoint_hash, field_name="checkpoint_hash")
        _require_optional_identifier(self.selection_objective, field_name="selection_objective")
        _require_optional_identifier(
            self.fixed_constant_rationale,
            field_name="fixed_constant_rationale",
        )
        _require_true(self.development_only_provenance, field_name="development_only_provenance")
        _require_true(self.internal_test_not_used, field_name="internal_test_not_used")
        _require_true(self.external_data_not_used, field_name="external_data_not_used")
        _require_allowed(self.freeze_status, _FREEZE_STATUSES, field_name="freeze_status")
        if self.policy_type == "fixed_constant":
            if self.threshold_value is None or self.fixed_constant_rationale is None:
                raise Phase8InternalEvidenceValidationError(
                    "fixed_constant threshold requires threshold_value and "
                    "fixed_constant_rationale."
                )
            if (
                self.selection_objective is not None
                or self.candidate_threshold_grid_reference is not None
                or self.development_metric_reference is not None
            ):
                raise Phase8InternalEvidenceValidationError(
                    "fixed_constant threshold must not include validation-selection fields."
                )
        if self.policy_type == "development_validation_selected" and (
            self.selection_objective is None
            or self.candidate_threshold_grid_reference is None
            or self.development_metric_reference is None
        ):
            raise Phase8InternalEvidenceValidationError(
                "development_validation_selected threshold requires objective, grid, "
                "and metric references."
            )
        if self.freeze_status == "freeze_ready" and (
            self.threshold_value is None
            or self.candidate_id is None
            or self.checkpoint_hash is None
        ):
            raise Phase8InternalEvidenceValidationError(
                "freeze_ready threshold requires resolved candidate, checkpoint, and threshold."
            )
        _require_self_hash(
            self.threshold_decision_hash,
            hash_phase8_threshold_decision(self),
            field_name="threshold_decision_hash",
        )


@dataclass(frozen=True, slots=True)
class Phase8SupportPolicy:
    """Immutable no-support or internal-only support policy."""

    schema_name: str
    schema_version: str
    policy_type: str
    selected_candidate_id: str | None
    selected_checkpoint_hash: str | None
    internal_support_manifest_reference: ArtifactReference | None
    support_k: int | None
    support_seed_reference: ArtifactReference | None
    external_support_labels_forbidden: bool
    external_prototype_construction_forbidden: bool
    external_support_state: str
    freeze_status: str
    support_policy_hash: str

    def __post_init__(self) -> None:
        _require_schema(self.schema_name, PHASE8_SUPPORT_POLICY_SCHEMA_NAME)
        _require_schema_version_exact(self.schema_version)
        _require_allowed(self.policy_type, _SUPPORT_POLICIES, field_name="policy_type")
        _require_optional_identifier(self.selected_candidate_id, field_name="selected_candidate_id")
        _require_optional_sha256(
            self.selected_checkpoint_hash,
            field_name="selected_checkpoint_hash",
        )
        if self.support_k is not None:
            _require_positive_int(self.support_k, field_name="support_k")
        _require_true(
            self.external_support_labels_forbidden,
            field_name="external_support_labels_forbidden",
        )
        _require_true(
            self.external_prototype_construction_forbidden,
            field_name="external_prototype_construction_forbidden",
        )
        if self.external_support_state != "forbidden":
            raise Phase8InternalEvidenceValidationError("external support state must be forbidden.")
        _require_allowed(self.freeze_status, _FREEZE_STATUSES, field_name="freeze_status")
        if self.policy_type == "no_support" and (
            self.internal_support_manifest_reference is not None
            or self.support_k is not None
            or self.support_seed_reference is not None
        ):
            raise Phase8InternalEvidenceValidationError(
                "no_support policy rejects support manifest, K, and seed fields."
            )
        if self.policy_type == "internal_only_support" and (
            self.internal_support_manifest_reference is None
            or self.support_k is None
            or self.support_seed_reference is None
        ):
            raise Phase8InternalEvidenceValidationError(
                "internal_only_support requires support manifest, K, and seed references."
            )
        if self.freeze_status == "freeze_ready" and (
            self.selected_candidate_id is None or self.selected_checkpoint_hash is None
        ):
            raise Phase8InternalEvidenceValidationError(
                "freeze_ready support policy requires selected workflow reference."
            )
        _require_self_hash(
            self.support_policy_hash,
            hash_phase8_support_policy(self),
            field_name="support_policy_hash",
        )


@dataclass(frozen=True, slots=True)
class Phase8InternalEvidencePackage:
    """Aggregate internal evidence package for Phase 8 readiness validation."""

    schema_name: str
    schema_version: str
    preprocessing_decision: Phase8PreprocessingDecision
    candidate_inventory: Phase8FixedCandidateInventory
    checkpoint_metadata: Phase8CheckpointMetadata
    validation_evidence: Phase8ValidationEvidenceReference
    model_selection_decision: Phase8ModelSelectionDecision
    threshold_decision: Phase8ThresholdDecision
    support_policy: Phase8SupportPolicy
    package_readiness_state: str
    blocker_reason_codes: tuple[str, ...]
    package_hash: str

    def __post_init__(self) -> None:
        _require_schema(self.schema_name, PHASE8_INTERNAL_EVIDENCE_PACKAGE_SCHEMA_NAME)
        _require_schema_version_exact(self.schema_version)
        _require_allowed(
            self.package_readiness_state,
            _READINESS_STATES,
            field_name="package_readiness_state",
        )
        object.__setattr__(self, "blocker_reason_codes", tuple(sorted(self.blocker_reason_codes)))
        for index, code in enumerate(self.blocker_reason_codes):
            _require_identifier(code, field_name=f"blocker_reason_codes[{index}]")
        blockers = _package_blockers(
            preprocessing_decision=self.preprocessing_decision,
            candidate_inventory=self.candidate_inventory,
            checkpoint_metadata=self.checkpoint_metadata,
            validation_evidence=self.validation_evidence,
            model_selection_decision=self.model_selection_decision,
            threshold_decision=self.threshold_decision,
            support_policy=self.support_policy,
        )
        if self.package_readiness_state == "READY" and blockers:
            raise Phase8InternalEvidenceValidationError(
                f"READY package has unresolved blockers: {blockers!r}."
            )
        if self.package_readiness_state == "BLOCKED" and not self.blocker_reason_codes:
            raise Phase8InternalEvidenceValidationError(
                "BLOCKED package requires blocker_reason_codes."
            )
        if sorted(blockers) != list(self.blocker_reason_codes):
            raise Phase8InternalEvidenceValidationError(
                "blocker_reason_codes must exactly match cross-contract validation blockers."
            )
        _require_self_hash(
            self.package_hash,
            hash_phase8_internal_evidence_package(self),
            field_name="package_hash",
        )


def artifact_reference_to_dict(reference: ArtifactReference) -> dict[str, JsonValue]:
    """Convert an artifact reference to a canonical mapping."""

    return {
        "artifact_hash": reference.artifact_hash,
        "artifact_role": reference.artifact_role,
        "schema_name": reference.schema_name,
        "schema_version": reference.schema_version,
    }


def phase8_preprocessing_decision_identity_payload(
    decision: Phase8PreprocessingDecision,
) -> dict[str, JsonValue]:
    """Return the canonical identity payload for preprocessing evidence."""

    return {
        "candidate_id": decision.candidate_id,
        "crop_roi_policy_reference": artifact_reference_to_dict(decision.crop_roi_policy_reference),
        "development_manifest_hash": decision.development_manifest_hash,
        "development_split_hash": decision.development_split_hash,
        "evidence_status": decision.evidence_status,
        "fit_artifact_reference": _optional_reference_to_dict(decision.fit_artifact_reference),
        "fit_scope": decision.fit_scope,
        "image_interpolation_policy": decision.image_interpolation_policy,
        "intensity_policy_reference": artifact_reference_to_dict(
            decision.intensity_policy_reference
        ),
        "label_interpolation_policy": decision.label_interpolation_policy,
        "no_external_data": decision.no_external_data,
        "normalization_policy_reference": artifact_reference_to_dict(
            decision.normalization_policy_reference
        ),
        "orientation_policy_reference": artifact_reference_to_dict(
            decision.orientation_policy_reference
        ),
        "originating_git_commit": decision.originating_git_commit,
        "preprocessing_config_hash": decision.preprocessing_config_hash,
        "preprocessing_config_schema_name": decision.preprocessing_config_schema_name,
        "preprocessing_config_schema_version": decision.preprocessing_config_schema_version,
        "schema_name": decision.schema_name,
        "schema_version": decision.schema_version,
        "spacing_policy_reference": artifact_reference_to_dict(decision.spacing_policy_reference),
    }


def phase8_preprocessing_decision_to_dict(
    decision: Phase8PreprocessingDecision,
) -> dict[str, JsonValue]:
    """Convert preprocessing evidence to a canonical mapping."""

    payload = phase8_preprocessing_decision_identity_payload(decision)
    payload["preprocessing_decision_hash"] = decision.preprocessing_decision_hash
    return payload


def hash_phase8_preprocessing_decision(decision: Phase8PreprocessingDecision) -> str:
    """Return the canonical self-hash for preprocessing evidence."""

    return sha256_json(phase8_preprocessing_decision_identity_payload(decision))


def candidate_definition_to_dict(candidate: CandidateDefinition) -> dict[str, JsonValue]:
    """Convert one candidate definition to a canonical mapping."""

    return {
        "candidate_id": candidate.candidate_id,
        "executable_readiness_state": candidate.executable_readiness_state,
        "failure_handling_policy": candidate.failure_handling_policy,
        "fixed_seeds": list(candidate.fixed_seeds),
        "method_identity": candidate.method_identity,
        "model_family": candidate.model_family,
        "preprocessing_evidence_hash": candidate.preprocessing_evidence_hash,
        "training_adaptation_mode": candidate.training_adaptation_mode,
        "training_config_reference": artifact_reference_to_dict(
            candidate.training_config_reference
        ),
        "uses_external_artifacts": candidate.uses_external_artifacts,
    }


def phase8_fixed_candidate_inventory_identity_payload(
    inventory: Phase8FixedCandidateInventory,
) -> dict[str, JsonValue]:
    """Return the canonical identity payload for a candidate inventory."""

    return {
        "candidate_set_locked": inventory.candidate_set_locked,
        "candidates": [candidate_definition_to_dict(item) for item in inventory.candidates],
        "inventory_state": inventory.inventory_state,
        "schema_name": inventory.schema_name,
        "schema_version": inventory.schema_version,
        "selection_started": inventory.selection_started,
        "supports_single_candidate_without_comparison": (
            inventory.supports_single_candidate_without_comparison
        ),
    }


def phase8_fixed_candidate_inventory_to_dict(
    inventory: Phase8FixedCandidateInventory,
) -> dict[str, JsonValue]:
    """Convert a candidate inventory to a canonical mapping."""

    payload = phase8_fixed_candidate_inventory_identity_payload(inventory)
    payload["inventory_hash"] = inventory.inventory_hash
    return payload


def hash_phase8_fixed_candidate_inventory(inventory: Phase8FixedCandidateInventory) -> str:
    """Return the canonical self-hash for a candidate inventory."""

    return sha256_json(phase8_fixed_candidate_inventory_identity_payload(inventory))


def phase8_checkpoint_metadata_identity_payload(
    metadata: Phase8CheckpointMetadata,
) -> dict[str, JsonValue]:
    """Return the canonical identity payload for checkpoint metadata."""

    return {
        "candidate_id": metadata.candidate_id,
        "checkpoint_byte_size": metadata.checkpoint_byte_size,
        "checkpoint_sha256": metadata.checkpoint_sha256,
        "completion_status": metadata.completion_status,
        "development_manifest_hash": metadata.development_manifest_hash,
        "development_split_hash": metadata.development_split_hash,
        "failure_codes": list(metadata.failure_codes),
        "freeze_eligible": metadata.freeze_eligible,
        "model_family": metadata.model_family,
        "no_external_data": metadata.no_external_data,
        "originating_git_commit": metadata.originating_git_commit,
        "package_environment_reference": artifact_reference_to_dict(
            metadata.package_environment_reference
        ),
        "preprocessing_evidence_hash": metadata.preprocessing_evidence_hash,
        "schema_name": metadata.schema_name,
        "schema_version": metadata.schema_version,
        "seed": metadata.seed,
        "serialization_format": metadata.serialization_format,
        "training_adaptation_mode": metadata.training_adaptation_mode,
        "training_config_hash": metadata.training_config_hash,
        "validation_metric_artifact_reference": _optional_reference_to_dict(
            metadata.validation_metric_artifact_reference
        ),
    }


def phase8_checkpoint_metadata_to_dict(
    metadata: Phase8CheckpointMetadata,
) -> dict[str, JsonValue]:
    """Convert checkpoint metadata to a canonical mapping."""

    payload = phase8_checkpoint_metadata_identity_payload(metadata)
    payload["amp_state"] = metadata.amp_state
    payload["checkpoint_metadata_hash"] = metadata.checkpoint_metadata_hash
    payload["device_type"] = metadata.device_type
    return payload


def hash_phase8_checkpoint_metadata(metadata: Phase8CheckpointMetadata) -> str:
    """Return the canonical self-hash for checkpoint metadata."""

    return sha256_json(phase8_checkpoint_metadata_identity_payload(metadata))


def phase8_validation_evidence_reference_identity_payload(
    evidence: Phase8ValidationEvidenceReference,
) -> dict[str, JsonValue]:
    """Return the canonical identity payload for validation evidence."""

    return {
        "candidate_id": evidence.candidate_id,
        "checkpoint_hash": evidence.checkpoint_hash,
        "development_validation_only": evidence.development_validation_only,
        "empty_target_accounting_reference": artifact_reference_to_dict(
            evidence.empty_target_accounting_reference
        ),
        "external_data_not_used": evidence.external_data_not_used,
        "internal_test_not_used": evidence.internal_test_not_used,
        "metric_artifact_hash": evidence.metric_artifact_hash,
        "metric_artifact_schema_name": evidence.metric_artifact_schema_name,
        "metric_artifact_schema_version": evidence.metric_artifact_schema_version,
        "primary_metric_name": evidence.primary_metric_name,
        "schema_name": evidence.schema_name,
        "schema_version": evidence.schema_version,
        "valid_case_count": evidence.valid_case_count,
        "validation_cohort_split_hash": evidence.validation_cohort_split_hash,
    }


def phase8_validation_evidence_reference_to_dict(
    evidence: Phase8ValidationEvidenceReference,
) -> dict[str, JsonValue]:
    """Convert validation evidence to a canonical mapping."""

    payload = phase8_validation_evidence_reference_identity_payload(evidence)
    payload["validation_evidence_hash"] = evidence.validation_evidence_hash
    return payload


def hash_phase8_validation_evidence_reference(
    evidence: Phase8ValidationEvidenceReference,
) -> str:
    """Return the canonical self-hash for validation evidence."""

    return sha256_json(phase8_validation_evidence_reference_identity_payload(evidence))


def failed_candidate_accounting_to_dict(
    accounting: FailedCandidateAccounting,
) -> dict[str, JsonValue]:
    """Convert failed-candidate accounting to a canonical mapping."""

    return {
        "candidate_id": accounting.candidate_id,
        "checkpoint_hash": accounting.checkpoint_hash,
        "failure_code": accounting.failure_code,
        "validation_evidence_hash": accounting.validation_evidence_hash,
    }


def phase8_model_selection_decision_identity_payload(
    decision: Phase8ModelSelectionDecision,
) -> dict[str, JsonValue]:
    """Return the canonical identity payload for model selection."""

    return {
        "deterministic_tie_break_order": list(decision.deterministic_tie_break_order),
        "external_data_not_used": decision.external_data_not_used,
        "failed_candidate_accounting": [
            failed_candidate_accounting_to_dict(item)
            for item in decision.failed_candidate_accounting
        ],
        "fixed_candidate_inventory_hash": decision.fixed_candidate_inventory_hash,
        "internal_test_not_used": decision.internal_test_not_used,
        "primary_metric": decision.primary_metric,
        "schema_name": decision.schema_name,
        "schema_version": decision.schema_version,
        "selected_candidate_id": decision.selected_candidate_id,
        "selected_checkpoint_hash": decision.selected_checkpoint_hash,
        "selected_validation_evidence_hash": decision.selected_validation_evidence_hash,
        "selection_rule_version": decision.selection_rule_version,
        "selection_status": decision.selection_status,
    }


def phase8_model_selection_decision_to_dict(
    decision: Phase8ModelSelectionDecision,
) -> dict[str, JsonValue]:
    """Convert model-selection evidence to a canonical mapping."""

    payload = phase8_model_selection_decision_identity_payload(decision)
    payload["model_selection_decision_hash"] = decision.model_selection_decision_hash
    return payload


def hash_phase8_model_selection_decision(decision: Phase8ModelSelectionDecision) -> str:
    """Return the canonical self-hash for model selection."""

    return sha256_json(phase8_model_selection_decision_identity_payload(decision))


def phase8_threshold_decision_identity_payload(
    decision: Phase8ThresholdDecision,
) -> dict[str, JsonValue]:
    """Return the canonical identity payload for threshold evidence."""

    return {
        "candidate_id": decision.candidate_id,
        "candidate_threshold_grid_reference": _optional_reference_to_dict(
            decision.candidate_threshold_grid_reference
        ),
        "checkpoint_hash": decision.checkpoint_hash,
        "development_metric_reference": _optional_reference_to_dict(
            decision.development_metric_reference
        ),
        "development_only_provenance": decision.development_only_provenance,
        "external_data_not_used": decision.external_data_not_used,
        "fixed_constant_rationale": decision.fixed_constant_rationale,
        "freeze_status": decision.freeze_status,
        "internal_test_not_used": decision.internal_test_not_used,
        "policy_type": decision.policy_type,
        "schema_name": decision.schema_name,
        "schema_version": decision.schema_version,
        "selection_objective": decision.selection_objective,
        "threshold_value": decision.threshold_value,
    }


def phase8_threshold_decision_to_dict(decision: Phase8ThresholdDecision) -> dict[str, JsonValue]:
    """Convert threshold evidence to a canonical mapping."""

    payload = phase8_threshold_decision_identity_payload(decision)
    payload["threshold_decision_hash"] = decision.threshold_decision_hash
    return payload


def hash_phase8_threshold_decision(decision: Phase8ThresholdDecision) -> str:
    """Return the canonical self-hash for threshold evidence."""

    return sha256_json(phase8_threshold_decision_identity_payload(decision))


def phase8_support_policy_identity_payload(policy: Phase8SupportPolicy) -> dict[str, JsonValue]:
    """Return the canonical identity payload for support policy evidence."""

    return {
        "external_prototype_construction_forbidden": (
            policy.external_prototype_construction_forbidden
        ),
        "external_support_labels_forbidden": policy.external_support_labels_forbidden,
        "external_support_state": policy.external_support_state,
        "freeze_status": policy.freeze_status,
        "internal_support_manifest_reference": _optional_reference_to_dict(
            policy.internal_support_manifest_reference
        ),
        "policy_type": policy.policy_type,
        "schema_name": policy.schema_name,
        "schema_version": policy.schema_version,
        "selected_candidate_id": policy.selected_candidate_id,
        "selected_checkpoint_hash": policy.selected_checkpoint_hash,
        "support_k": policy.support_k,
        "support_seed_reference": _optional_reference_to_dict(policy.support_seed_reference),
    }


def phase8_support_policy_to_dict(policy: Phase8SupportPolicy) -> dict[str, JsonValue]:
    """Convert support policy evidence to a canonical mapping."""

    payload = phase8_support_policy_identity_payload(policy)
    payload["support_policy_hash"] = policy.support_policy_hash
    return payload


def hash_phase8_support_policy(policy: Phase8SupportPolicy) -> str:
    """Return the canonical self-hash for support policy evidence."""

    return sha256_json(phase8_support_policy_identity_payload(policy))


def phase8_internal_evidence_package_identity_payload(
    package: Phase8InternalEvidencePackage,
) -> dict[str, JsonValue]:
    """Return the canonical identity payload for an aggregate package."""

    return {
        "blocker_reason_codes": list(package.blocker_reason_codes),
        "candidate_inventory": phase8_fixed_candidate_inventory_to_dict(
            package.candidate_inventory
        ),
        "checkpoint_metadata": phase8_checkpoint_metadata_to_dict(package.checkpoint_metadata),
        "model_selection_decision": phase8_model_selection_decision_to_dict(
            package.model_selection_decision
        ),
        "package_readiness_state": package.package_readiness_state,
        "preprocessing_decision": phase8_preprocessing_decision_to_dict(
            package.preprocessing_decision
        ),
        "schema_name": package.schema_name,
        "schema_version": package.schema_version,
        "support_policy": phase8_support_policy_to_dict(package.support_policy),
        "threshold_decision": phase8_threshold_decision_to_dict(package.threshold_decision),
        "validation_evidence": phase8_validation_evidence_reference_to_dict(
            package.validation_evidence
        ),
    }


def phase8_internal_evidence_package_to_dict(
    package: Phase8InternalEvidencePackage,
) -> dict[str, JsonValue]:
    """Convert an aggregate package to a canonical mapping."""

    payload = phase8_internal_evidence_package_identity_payload(package)
    payload["package_hash"] = package.package_hash
    return payload


def hash_phase8_internal_evidence_package(package: Phase8InternalEvidencePackage) -> str:
    """Return the canonical self-hash for an aggregate package."""

    return sha256_json(phase8_internal_evidence_package_identity_payload(package))


def phase8_internal_evidence_package_to_json(package: Phase8InternalEvidencePackage) -> bytes:
    """Serialize an aggregate package to canonical JSON bytes."""

    return canonical_json_bytes(phase8_internal_evidence_package_to_dict(package)) + b"\n"


def verify_checkpoint_file_identity_values(
    *,
    expected_checkpoint_sha256: str,
    expected_checkpoint_byte_size: int,
    observed_checkpoint_sha256: str,
    observed_checkpoint_byte_size: int,
) -> None:
    """Verify checkpoint digest and byte size from already supplied observations."""

    _require_sha256(expected_checkpoint_sha256, field_name="expected_checkpoint_sha256")
    _require_sha256(observed_checkpoint_sha256, field_name="observed_checkpoint_sha256")
    _require_positive_int(
        expected_checkpoint_byte_size,
        field_name="expected_checkpoint_byte_size",
    )
    _require_positive_int(
        observed_checkpoint_byte_size,
        field_name="observed_checkpoint_byte_size",
    )
    if expected_checkpoint_sha256 != observed_checkpoint_sha256:
        raise Phase8InternalEvidenceHashError("observed checkpoint SHA-256 does not match.")
    if expected_checkpoint_byte_size != observed_checkpoint_byte_size:
        raise Phase8InternalEvidenceValidationError("observed checkpoint byte size does not match.")


def artifact_reference_from_mapping(mapping: MappingLike) -> ArtifactReference:
    """Reconstruct an artifact reference from a strict mapping."""

    _require_exact_fields(mapping, _ARTIFACT_REFERENCE_FIELDS, object_name="ArtifactReference")
    return ArtifactReference(
        schema_name=_expect_string(mapping["schema_name"], field_name="schema_name"),
        schema_version=_expect_string(mapping["schema_version"], field_name="schema_version"),
        artifact_hash=_expect_string(mapping["artifact_hash"], field_name="artifact_hash"),
        artifact_role=_expect_string(mapping["artifact_role"], field_name="artifact_role"),
    )


def phase8_preprocessing_decision_from_mapping(
    mapping: MappingLike,
) -> Phase8PreprocessingDecision:
    """Reconstruct preprocessing evidence from a strict mapping."""

    _require_exact_fields(
        mapping,
        _PREPROCESSING_DECISION_FIELDS,
        object_name="Phase8PreprocessingDecision",
    )
    fit_reference = mapping["fit_artifact_reference"]
    return Phase8PreprocessingDecision(
        schema_name=_expect_string(mapping["schema_name"], field_name="schema_name"),
        schema_version=_expect_string(mapping["schema_version"], field_name="schema_version"),
        candidate_id=_expect_string(mapping["candidate_id"], field_name="candidate_id"),
        development_manifest_hash=_expect_string(
            mapping["development_manifest_hash"],
            field_name="development_manifest_hash",
        ),
        development_split_hash=_expect_string(
            mapping["development_split_hash"],
            field_name="development_split_hash",
        ),
        preprocessing_config_schema_name=_expect_string(
            mapping["preprocessing_config_schema_name"],
            field_name="preprocessing_config_schema_name",
        ),
        preprocessing_config_schema_version=_expect_string(
            mapping["preprocessing_config_schema_version"],
            field_name="preprocessing_config_schema_version",
        ),
        preprocessing_config_hash=_expect_string(
            mapping["preprocessing_config_hash"],
            field_name="preprocessing_config_hash",
        ),
        fit_scope=_expect_string(mapping["fit_scope"], field_name="fit_scope"),
        fit_artifact_reference=None
        if fit_reference is None
        else artifact_reference_from_mapping(
            _expect_mapping(fit_reference, field_name="fit_artifact_reference")
        ),
        image_interpolation_policy=_expect_string(
            mapping["image_interpolation_policy"],
            field_name="image_interpolation_policy",
        ),
        label_interpolation_policy=_expect_string(
            mapping["label_interpolation_policy"],
            field_name="label_interpolation_policy",
        ),
        orientation_policy_reference=artifact_reference_from_mapping(
            _expect_mapping(
                mapping["orientation_policy_reference"],
                field_name="orientation_policy_reference",
            )
        ),
        spacing_policy_reference=artifact_reference_from_mapping(
            _expect_mapping(
                mapping["spacing_policy_reference"],
                field_name="spacing_policy_reference",
            )
        ),
        intensity_policy_reference=artifact_reference_from_mapping(
            _expect_mapping(
                mapping["intensity_policy_reference"],
                field_name="intensity_policy_reference",
            )
        ),
        crop_roi_policy_reference=artifact_reference_from_mapping(
            _expect_mapping(
                mapping["crop_roi_policy_reference"],
                field_name="crop_roi_policy_reference",
            )
        ),
        normalization_policy_reference=artifact_reference_from_mapping(
            _expect_mapping(
                mapping["normalization_policy_reference"],
                field_name="normalization_policy_reference",
            )
        ),
        originating_git_commit=_expect_string(
            mapping["originating_git_commit"],
            field_name="originating_git_commit",
        ),
        no_external_data=_expect_bool(mapping["no_external_data"], field_name="no_external_data"),
        evidence_status=_expect_string(mapping["evidence_status"], field_name="evidence_status"),
        preprocessing_decision_hash=_expect_string(
            mapping["preprocessing_decision_hash"],
            field_name="preprocessing_decision_hash",
        ),
    )


def phase8_fixed_candidate_inventory_from_mapping(
    mapping: MappingLike,
) -> Phase8FixedCandidateInventory:
    """Reconstruct a candidate inventory from a strict mapping."""

    _require_exact_fields(
        mapping,
        _FIXED_CANDIDATE_INVENTORY_FIELDS,
        object_name="Phase8FixedCandidateInventory",
    )
    return Phase8FixedCandidateInventory(
        schema_name=_expect_string(mapping["schema_name"], field_name="schema_name"),
        schema_version=_expect_string(mapping["schema_version"], field_name="schema_version"),
        inventory_state=_expect_string(mapping["inventory_state"], field_name="inventory_state"),
        candidate_set_locked=_expect_bool(
            mapping["candidate_set_locked"],
            field_name="candidate_set_locked",
        ),
        selection_started=_expect_bool(
            mapping["selection_started"], field_name="selection_started"
        ),
        supports_single_candidate_without_comparison=_expect_bool(
            mapping["supports_single_candidate_without_comparison"],
            field_name="supports_single_candidate_without_comparison",
        ),
        candidates=_candidate_definitions_from_sequence(mapping["candidates"]),
        inventory_hash=_expect_string(mapping["inventory_hash"], field_name="inventory_hash"),
    )


def phase8_checkpoint_metadata_from_mapping(mapping: MappingLike) -> Phase8CheckpointMetadata:
    """Reconstruct checkpoint metadata from a strict mapping."""

    _require_exact_fields(
        mapping,
        _CHECKPOINT_METADATA_FIELDS,
        object_name="Phase8CheckpointMetadata",
    )
    metric_reference = mapping["validation_metric_artifact_reference"]
    return Phase8CheckpointMetadata(
        schema_name=_expect_string(mapping["schema_name"], field_name="schema_name"),
        schema_version=_expect_string(mapping["schema_version"], field_name="schema_version"),
        checkpoint_sha256=_expect_string(
            mapping["checkpoint_sha256"],
            field_name="checkpoint_sha256",
        ),
        checkpoint_byte_size=_expect_int(
            mapping["checkpoint_byte_size"],
            field_name="checkpoint_byte_size",
        ),
        serialization_format=_expect_string(
            mapping["serialization_format"],
            field_name="serialization_format",
        ),
        model_family=_expect_string(mapping["model_family"], field_name="model_family"),
        candidate_id=_expect_string(mapping["candidate_id"], field_name="candidate_id"),
        seed=_expect_int(mapping["seed"], field_name="seed"),
        training_adaptation_mode=_expect_string(
            mapping["training_adaptation_mode"],
            field_name="training_adaptation_mode",
        ),
        originating_git_commit=_expect_string(
            mapping["originating_git_commit"],
            field_name="originating_git_commit",
        ),
        package_environment_reference=artifact_reference_from_mapping(
            _expect_mapping(
                mapping["package_environment_reference"],
                field_name="package_environment_reference",
            )
        ),
        training_config_hash=_expect_string(
            mapping["training_config_hash"],
            field_name="training_config_hash",
        ),
        preprocessing_evidence_hash=_expect_string(
            mapping["preprocessing_evidence_hash"],
            field_name="preprocessing_evidence_hash",
        ),
        development_manifest_hash=_expect_string(
            mapping["development_manifest_hash"],
            field_name="development_manifest_hash",
        ),
        development_split_hash=_expect_string(
            mapping["development_split_hash"],
            field_name="development_split_hash",
        ),
        validation_metric_artifact_reference=None
        if metric_reference is None
        else artifact_reference_from_mapping(
            _expect_mapping(
                metric_reference,
                field_name="validation_metric_artifact_reference",
            )
        ),
        device_type=_expect_string(mapping["device_type"], field_name="device_type"),
        amp_state=_expect_string(mapping["amp_state"], field_name="amp_state"),
        completion_status=_expect_string(
            mapping["completion_status"],
            field_name="completion_status",
        ),
        failure_codes=_expect_string_tuple(mapping["failure_codes"], field_name="failure_codes"),
        no_external_data=_expect_bool(mapping["no_external_data"], field_name="no_external_data"),
        freeze_eligible=_expect_bool(mapping["freeze_eligible"], field_name="freeze_eligible"),
        checkpoint_metadata_hash=_expect_string(
            mapping["checkpoint_metadata_hash"],
            field_name="checkpoint_metadata_hash",
        ),
    )


def phase8_validation_evidence_reference_from_mapping(
    mapping: MappingLike,
) -> Phase8ValidationEvidenceReference:
    """Reconstruct validation evidence from a strict mapping."""

    _require_exact_fields(
        mapping,
        _VALIDATION_EVIDENCE_REFERENCE_FIELDS,
        object_name="Phase8ValidationEvidenceReference",
    )
    return Phase8ValidationEvidenceReference(
        schema_name=_expect_string(mapping["schema_name"], field_name="schema_name"),
        schema_version=_expect_string(mapping["schema_version"], field_name="schema_version"),
        candidate_id=_expect_string(mapping["candidate_id"], field_name="candidate_id"),
        checkpoint_hash=_expect_string(mapping["checkpoint_hash"], field_name="checkpoint_hash"),
        validation_cohort_split_hash=_expect_string(
            mapping["validation_cohort_split_hash"],
            field_name="validation_cohort_split_hash",
        ),
        metric_artifact_schema_name=_expect_string(
            mapping["metric_artifact_schema_name"],
            field_name="metric_artifact_schema_name",
        ),
        metric_artifact_schema_version=_expect_string(
            mapping["metric_artifact_schema_version"],
            field_name="metric_artifact_schema_version",
        ),
        metric_artifact_hash=_expect_string(
            mapping["metric_artifact_hash"],
            field_name="metric_artifact_hash",
        ),
        primary_metric_name=_expect_string(
            mapping["primary_metric_name"],
            field_name="primary_metric_name",
        ),
        valid_case_count=_expect_int(mapping["valid_case_count"], field_name="valid_case_count"),
        empty_target_accounting_reference=artifact_reference_from_mapping(
            _expect_mapping(
                mapping["empty_target_accounting_reference"],
                field_name="empty_target_accounting_reference",
            )
        ),
        development_validation_only=_expect_bool(
            mapping["development_validation_only"],
            field_name="development_validation_only",
        ),
        internal_test_not_used=_expect_bool(
            mapping["internal_test_not_used"],
            field_name="internal_test_not_used",
        ),
        external_data_not_used=_expect_bool(
            mapping["external_data_not_used"],
            field_name="external_data_not_used",
        ),
        validation_evidence_hash=_expect_string(
            mapping["validation_evidence_hash"],
            field_name="validation_evidence_hash",
        ),
    )


def phase8_model_selection_decision_from_mapping(
    mapping: MappingLike,
) -> Phase8ModelSelectionDecision:
    """Reconstruct model-selection evidence from a strict mapping."""

    _require_exact_fields(
        mapping,
        _MODEL_SELECTION_DECISION_FIELDS,
        object_name="Phase8ModelSelectionDecision",
    )
    return Phase8ModelSelectionDecision(
        schema_name=_expect_string(mapping["schema_name"], field_name="schema_name"),
        schema_version=_expect_string(mapping["schema_version"], field_name="schema_version"),
        fixed_candidate_inventory_hash=_expect_string(
            mapping["fixed_candidate_inventory_hash"],
            field_name="fixed_candidate_inventory_hash",
        ),
        selection_rule_version=_expect_string(
            mapping["selection_rule_version"],
            field_name="selection_rule_version",
        ),
        primary_metric=_expect_string(mapping["primary_metric"], field_name="primary_metric"),
        deterministic_tie_break_order=_expect_string_tuple(
            mapping["deterministic_tie_break_order"],
            field_name="deterministic_tie_break_order",
        ),
        selected_candidate_id=_expect_optional_string(
            mapping["selected_candidate_id"],
            field_name="selected_candidate_id",
        ),
        selected_checkpoint_hash=_expect_optional_string(
            mapping["selected_checkpoint_hash"],
            field_name="selected_checkpoint_hash",
        ),
        selected_validation_evidence_hash=_expect_optional_string(
            mapping["selected_validation_evidence_hash"],
            field_name="selected_validation_evidence_hash",
        ),
        failed_candidate_accounting=_failed_accounting_from_sequence(
            mapping["failed_candidate_accounting"]
        ),
        selection_status=_expect_string(mapping["selection_status"], field_name="selection_status"),
        internal_test_not_used=_expect_bool(
            mapping["internal_test_not_used"],
            field_name="internal_test_not_used",
        ),
        external_data_not_used=_expect_bool(
            mapping["external_data_not_used"],
            field_name="external_data_not_used",
        ),
        model_selection_decision_hash=_expect_string(
            mapping["model_selection_decision_hash"],
            field_name="model_selection_decision_hash",
        ),
    )


def phase8_threshold_decision_from_mapping(mapping: MappingLike) -> Phase8ThresholdDecision:
    """Reconstruct threshold evidence from a strict mapping."""

    _require_exact_fields(
        mapping,
        _THRESHOLD_DECISION_FIELDS,
        object_name="Phase8ThresholdDecision",
    )
    grid_reference = mapping["candidate_threshold_grid_reference"]
    metric_reference = mapping["development_metric_reference"]
    return Phase8ThresholdDecision(
        schema_name=_expect_string(mapping["schema_name"], field_name="schema_name"),
        schema_version=_expect_string(mapping["schema_version"], field_name="schema_version"),
        policy_type=_expect_string(mapping["policy_type"], field_name="policy_type"),
        threshold_value=_expect_optional_float(
            mapping["threshold_value"],
            field_name="threshold_value",
        ),
        candidate_id=_expect_optional_string(mapping["candidate_id"], field_name="candidate_id"),
        checkpoint_hash=_expect_optional_string(
            mapping["checkpoint_hash"], field_name="checkpoint_hash"
        ),
        selection_objective=_expect_optional_string(
            mapping["selection_objective"],
            field_name="selection_objective",
        ),
        candidate_threshold_grid_reference=None
        if grid_reference is None
        else artifact_reference_from_mapping(
            _expect_mapping(
                grid_reference,
                field_name="candidate_threshold_grid_reference",
            )
        ),
        development_metric_reference=None
        if metric_reference is None
        else artifact_reference_from_mapping(
            _expect_mapping(metric_reference, field_name="development_metric_reference")
        ),
        fixed_constant_rationale=_expect_optional_string(
            mapping["fixed_constant_rationale"],
            field_name="fixed_constant_rationale",
        ),
        development_only_provenance=_expect_bool(
            mapping["development_only_provenance"],
            field_name="development_only_provenance",
        ),
        internal_test_not_used=_expect_bool(
            mapping["internal_test_not_used"],
            field_name="internal_test_not_used",
        ),
        external_data_not_used=_expect_bool(
            mapping["external_data_not_used"],
            field_name="external_data_not_used",
        ),
        freeze_status=_expect_string(mapping["freeze_status"], field_name="freeze_status"),
        threshold_decision_hash=_expect_string(
            mapping["threshold_decision_hash"],
            field_name="threshold_decision_hash",
        ),
    )


def phase8_support_policy_from_mapping(mapping: MappingLike) -> Phase8SupportPolicy:
    """Reconstruct support policy evidence from a strict mapping."""

    _require_exact_fields(mapping, _SUPPORT_POLICY_FIELDS, object_name="Phase8SupportPolicy")
    manifest_reference = mapping["internal_support_manifest_reference"]
    seed_reference = mapping["support_seed_reference"]
    return Phase8SupportPolicy(
        schema_name=_expect_string(mapping["schema_name"], field_name="schema_name"),
        schema_version=_expect_string(mapping["schema_version"], field_name="schema_version"),
        policy_type=_expect_string(mapping["policy_type"], field_name="policy_type"),
        selected_candidate_id=_expect_optional_string(
            mapping["selected_candidate_id"],
            field_name="selected_candidate_id",
        ),
        selected_checkpoint_hash=_expect_optional_string(
            mapping["selected_checkpoint_hash"],
            field_name="selected_checkpoint_hash",
        ),
        internal_support_manifest_reference=None
        if manifest_reference is None
        else artifact_reference_from_mapping(
            _expect_mapping(
                manifest_reference,
                field_name="internal_support_manifest_reference",
            )
        ),
        support_k=_expect_optional_int(mapping["support_k"], field_name="support_k"),
        support_seed_reference=None
        if seed_reference is None
        else artifact_reference_from_mapping(
            _expect_mapping(seed_reference, field_name="support_seed_reference")
        ),
        external_support_labels_forbidden=_expect_bool(
            mapping["external_support_labels_forbidden"],
            field_name="external_support_labels_forbidden",
        ),
        external_prototype_construction_forbidden=_expect_bool(
            mapping["external_prototype_construction_forbidden"],
            field_name="external_prototype_construction_forbidden",
        ),
        external_support_state=_expect_string(
            mapping["external_support_state"],
            field_name="external_support_state",
        ),
        freeze_status=_expect_string(mapping["freeze_status"], field_name="freeze_status"),
        support_policy_hash=_expect_string(
            mapping["support_policy_hash"],
            field_name="support_policy_hash",
        ),
    )


def phase8_internal_evidence_package_from_mapping(
    mapping: MappingLike,
) -> Phase8InternalEvidencePackage:
    """Reconstruct an aggregate package from a strict mapping."""

    _require_exact_fields(
        mapping,
        _INTERNAL_EVIDENCE_PACKAGE_FIELDS,
        object_name="Phase8InternalEvidencePackage",
    )
    return Phase8InternalEvidencePackage(
        schema_name=_expect_string(mapping["schema_name"], field_name="schema_name"),
        schema_version=_expect_string(mapping["schema_version"], field_name="schema_version"),
        preprocessing_decision=phase8_preprocessing_decision_from_mapping(
            _expect_mapping(mapping["preprocessing_decision"], field_name="preprocessing_decision")
        ),
        candidate_inventory=phase8_fixed_candidate_inventory_from_mapping(
            _expect_mapping(mapping["candidate_inventory"], field_name="candidate_inventory")
        ),
        checkpoint_metadata=phase8_checkpoint_metadata_from_mapping(
            _expect_mapping(mapping["checkpoint_metadata"], field_name="checkpoint_metadata")
        ),
        validation_evidence=phase8_validation_evidence_reference_from_mapping(
            _expect_mapping(mapping["validation_evidence"], field_name="validation_evidence")
        ),
        model_selection_decision=phase8_model_selection_decision_from_mapping(
            _expect_mapping(
                mapping["model_selection_decision"],
                field_name="model_selection_decision",
            )
        ),
        threshold_decision=phase8_threshold_decision_from_mapping(
            _expect_mapping(mapping["threshold_decision"], field_name="threshold_decision")
        ),
        support_policy=phase8_support_policy_from_mapping(
            _expect_mapping(mapping["support_policy"], field_name="support_policy")
        ),
        package_readiness_state=_expect_string(
            mapping["package_readiness_state"],
            field_name="package_readiness_state",
        ),
        blocker_reason_codes=_expect_string_tuple(
            mapping["blocker_reason_codes"],
            field_name="blocker_reason_codes",
        ),
        package_hash=_expect_string(mapping["package_hash"], field_name="package_hash"),
    )


def phase8_internal_evidence_package_from_json(
    data: bytes | str,
) -> Phase8InternalEvidencePackage:
    """Deserialize an aggregate package from JSON bytes or text."""

    return phase8_internal_evidence_package_from_mapping(_json_to_mapping(data))


_ARTIFACT_REFERENCE_FIELDS: Final[frozenset[str]] = frozenset(
    {"schema_name", "schema_version", "artifact_hash", "artifact_role"}
)
_PREPROCESSING_DECISION_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_name",
        "schema_version",
        "candidate_id",
        "development_manifest_hash",
        "development_split_hash",
        "preprocessing_config_schema_name",
        "preprocessing_config_schema_version",
        "preprocessing_config_hash",
        "fit_scope",
        "fit_artifact_reference",
        "image_interpolation_policy",
        "label_interpolation_policy",
        "orientation_policy_reference",
        "spacing_policy_reference",
        "intensity_policy_reference",
        "crop_roi_policy_reference",
        "normalization_policy_reference",
        "originating_git_commit",
        "no_external_data",
        "evidence_status",
        "preprocessing_decision_hash",
    }
)
_CANDIDATE_DEFINITION_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "candidate_id",
        "model_family",
        "method_identity",
        "training_adaptation_mode",
        "training_config_reference",
        "preprocessing_evidence_hash",
        "fixed_seeds",
        "executable_readiness_state",
        "failure_handling_policy",
        "uses_external_artifacts",
    }
)
_FIXED_CANDIDATE_INVENTORY_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_name",
        "schema_version",
        "inventory_state",
        "candidate_set_locked",
        "selection_started",
        "supports_single_candidate_without_comparison",
        "candidates",
        "inventory_hash",
    }
)
_CHECKPOINT_METADATA_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_name",
        "schema_version",
        "checkpoint_sha256",
        "checkpoint_byte_size",
        "serialization_format",
        "model_family",
        "candidate_id",
        "seed",
        "training_adaptation_mode",
        "originating_git_commit",
        "package_environment_reference",
        "training_config_hash",
        "preprocessing_evidence_hash",
        "development_manifest_hash",
        "development_split_hash",
        "validation_metric_artifact_reference",
        "device_type",
        "amp_state",
        "completion_status",
        "failure_codes",
        "no_external_data",
        "freeze_eligible",
        "checkpoint_metadata_hash",
    }
)
_VALIDATION_EVIDENCE_REFERENCE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_name",
        "schema_version",
        "candidate_id",
        "checkpoint_hash",
        "validation_cohort_split_hash",
        "metric_artifact_schema_name",
        "metric_artifact_schema_version",
        "metric_artifact_hash",
        "primary_metric_name",
        "valid_case_count",
        "empty_target_accounting_reference",
        "development_validation_only",
        "internal_test_not_used",
        "external_data_not_used",
        "validation_evidence_hash",
    }
)
_FAILED_CANDIDATE_ACCOUNTING_FIELDS: Final[frozenset[str]] = frozenset(
    {"candidate_id", "failure_code", "checkpoint_hash", "validation_evidence_hash"}
)
_MODEL_SELECTION_DECISION_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_name",
        "schema_version",
        "fixed_candidate_inventory_hash",
        "selection_rule_version",
        "primary_metric",
        "deterministic_tie_break_order",
        "selected_candidate_id",
        "selected_checkpoint_hash",
        "selected_validation_evidence_hash",
        "failed_candidate_accounting",
        "selection_status",
        "internal_test_not_used",
        "external_data_not_used",
        "model_selection_decision_hash",
    }
)
_THRESHOLD_DECISION_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_name",
        "schema_version",
        "policy_type",
        "threshold_value",
        "candidate_id",
        "checkpoint_hash",
        "selection_objective",
        "candidate_threshold_grid_reference",
        "development_metric_reference",
        "fixed_constant_rationale",
        "development_only_provenance",
        "internal_test_not_used",
        "external_data_not_used",
        "freeze_status",
        "threshold_decision_hash",
    }
)
_SUPPORT_POLICY_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_name",
        "schema_version",
        "policy_type",
        "selected_candidate_id",
        "selected_checkpoint_hash",
        "internal_support_manifest_reference",
        "support_k",
        "support_seed_reference",
        "external_support_labels_forbidden",
        "external_prototype_construction_forbidden",
        "external_support_state",
        "freeze_status",
        "support_policy_hash",
    }
)
_INTERNAL_EVIDENCE_PACKAGE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_name",
        "schema_version",
        "preprocessing_decision",
        "candidate_inventory",
        "checkpoint_metadata",
        "validation_evidence",
        "model_selection_decision",
        "threshold_decision",
        "support_policy",
        "package_readiness_state",
        "blocker_reason_codes",
        "package_hash",
    }
)


def _optional_reference_to_dict(reference: ArtifactReference | None) -> dict[str, JsonValue] | None:
    return None if reference is None else artifact_reference_to_dict(reference)


def _candidate_definitions_from_sequence(value: object) -> tuple[CandidateDefinition, ...]:
    sequence = _expect_list(value, field_name="candidates")
    candidates: list[CandidateDefinition] = []
    for index, item in enumerate(sequence):
        mapping = _expect_mapping(item, field_name=f"candidates[{index}]")
        _require_exact_fields(
            mapping,
            _CANDIDATE_DEFINITION_FIELDS,
            object_name=f"CandidateDefinition[{index}]",
        )
        candidates.append(
            CandidateDefinition(
                candidate_id=_expect_string(mapping["candidate_id"], field_name="candidate_id"),
                model_family=_expect_string(mapping["model_family"], field_name="model_family"),
                method_identity=_expect_string(
                    mapping["method_identity"],
                    field_name="method_identity",
                ),
                training_adaptation_mode=_expect_string(
                    mapping["training_adaptation_mode"],
                    field_name="training_adaptation_mode",
                ),
                training_config_reference=artifact_reference_from_mapping(
                    _expect_mapping(
                        mapping["training_config_reference"],
                        field_name="training_config_reference",
                    )
                ),
                preprocessing_evidence_hash=_expect_string(
                    mapping["preprocessing_evidence_hash"],
                    field_name="preprocessing_evidence_hash",
                ),
                fixed_seeds=tuple(
                    _expect_int_list(mapping["fixed_seeds"], field_name="fixed_seeds")
                ),
                executable_readiness_state=_expect_string(
                    mapping["executable_readiness_state"],
                    field_name="executable_readiness_state",
                ),
                failure_handling_policy=_expect_string(
                    mapping["failure_handling_policy"],
                    field_name="failure_handling_policy",
                ),
                uses_external_artifacts=_expect_bool(
                    mapping["uses_external_artifacts"],
                    field_name="uses_external_artifacts",
                ),
            )
        )
    return tuple(candidates)


def _failed_accounting_from_sequence(value: object) -> tuple[FailedCandidateAccounting, ...]:
    sequence = _expect_list(value, field_name="failed_candidate_accounting")
    records: list[FailedCandidateAccounting] = []
    for index, item in enumerate(sequence):
        mapping = _expect_mapping(item, field_name=f"failed_candidate_accounting[{index}]")
        _require_exact_fields(
            mapping,
            _FAILED_CANDIDATE_ACCOUNTING_FIELDS,
            object_name=f"FailedCandidateAccounting[{index}]",
        )
        records.append(
            FailedCandidateAccounting(
                candidate_id=_expect_string(mapping["candidate_id"], field_name="candidate_id"),
                failure_code=_expect_string(mapping["failure_code"], field_name="failure_code"),
                checkpoint_hash=_expect_optional_string(
                    mapping["checkpoint_hash"],
                    field_name="checkpoint_hash",
                ),
                validation_evidence_hash=_expect_optional_string(
                    mapping["validation_evidence_hash"],
                    field_name="validation_evidence_hash",
                ),
            )
        )
    return tuple(records)


def _package_blockers(
    *,
    preprocessing_decision: Phase8PreprocessingDecision,
    candidate_inventory: Phase8FixedCandidateInventory,
    checkpoint_metadata: Phase8CheckpointMetadata,
    validation_evidence: Phase8ValidationEvidenceReference,
    model_selection_decision: Phase8ModelSelectionDecision,
    threshold_decision: Phase8ThresholdDecision,
    support_policy: Phase8SupportPolicy,
) -> tuple[str, ...]:
    blockers: set[str] = set()
    candidate_by_id = {
        candidate.candidate_id: candidate for candidate in candidate_inventory.candidates
    }
    candidate_ids = set(candidate_by_id)
    selected_candidate_id = model_selection_decision.selected_candidate_id
    selected_checkpoint_hash = model_selection_decision.selected_checkpoint_hash
    selected_validation_hash = model_selection_decision.selected_validation_evidence_hash
    checkpoint_candidate = candidate_by_id.get(checkpoint_metadata.candidate_id)
    selected_candidate = (
        candidate_by_id.get(selected_candidate_id) if selected_candidate_id is not None else None
    )

    if preprocessing_decision.candidate_id not in candidate_ids:
        blockers.add("preprocessing_candidate_not_in_inventory")
    if checkpoint_metadata.candidate_id not in candidate_ids:
        blockers.add("checkpoint_candidate_not_in_inventory")
    if validation_evidence.candidate_id not in candidate_ids:
        blockers.add("validation_candidate_not_in_inventory")
    if selected_candidate_id not in candidate_ids:
        blockers.add("selected_candidate_not_in_inventory")
    if (
        model_selection_decision.fixed_candidate_inventory_hash
        != candidate_inventory.inventory_hash
    ):
        blockers.add("inventory_hash_mismatch")
    if checkpoint_metadata.candidate_id != validation_evidence.candidate_id:
        blockers.add("checkpoint_validation_candidate_mismatch")
    if checkpoint_metadata.checkpoint_sha256 != validation_evidence.checkpoint_hash:
        blockers.add("checkpoint_validation_hash_mismatch")
    if selected_candidate_id != checkpoint_metadata.candidate_id:
        blockers.add("selected_checkpoint_candidate_mismatch")
    if selected_checkpoint_hash != checkpoint_metadata.checkpoint_sha256:
        blockers.add("selected_checkpoint_hash_mismatch")
    if selected_validation_hash != validation_evidence.validation_evidence_hash:
        blockers.add("selected_validation_evidence_hash_mismatch")
    if model_selection_decision.primary_metric != validation_evidence.primary_metric_name:
        blockers.add("selection_primary_metric_mismatch")
    if checkpoint_metadata.validation_metric_artifact_reference is None:
        blockers.add("checkpoint_validation_metric_reference_missing")
    elif (
        checkpoint_metadata.validation_metric_artifact_reference.artifact_hash
        != validation_evidence.metric_artifact_hash
    ):
        blockers.add("checkpoint_validation_metric_hash_mismatch")
    if preprocessing_decision.preprocessing_decision_hash != (
        checkpoint_metadata.preprocessing_evidence_hash
    ):
        blockers.add("preprocessing_hash_mismatch")
    if checkpoint_candidate is None:
        blockers.add("candidate_preprocessing_reference_mismatch")
        blockers.add("training_config_hash_mismatch")
        blockers.add("checkpoint_model_family_mismatch")
        blockers.add("checkpoint_training_mode_mismatch")
        blockers.add("checkpoint_seed_not_in_candidate_fixed_seeds")
    else:
        if checkpoint_metadata.preprocessing_evidence_hash != (
            checkpoint_candidate.preprocessing_evidence_hash
        ):
            blockers.add("candidate_preprocessing_reference_mismatch")
        if checkpoint_metadata.training_config_hash != (
            checkpoint_candidate.training_config_reference.artifact_hash
        ):
            blockers.add("training_config_hash_mismatch")
        if checkpoint_metadata.model_family != checkpoint_candidate.model_family:
            blockers.add("checkpoint_model_family_mismatch")
        if checkpoint_metadata.training_adaptation_mode != (
            checkpoint_candidate.training_adaptation_mode
        ):
            blockers.add("checkpoint_training_mode_mismatch")
        if checkpoint_metadata.seed not in checkpoint_candidate.fixed_seeds:
            blockers.add("checkpoint_seed_not_in_candidate_fixed_seeds")
    if selected_candidate is not None and selected_candidate != checkpoint_candidate:
        blockers.add("selected_candidate_checkpoint_metadata_mismatch")
    if (
        checkpoint_metadata.development_manifest_hash
        != preprocessing_decision.development_manifest_hash
    ):
        blockers.add("development_manifest_hash_mismatch")
    if checkpoint_metadata.development_split_hash != preprocessing_decision.development_split_hash:
        blockers.add("development_split_hash_mismatch")
    if (
        validation_evidence.validation_cohort_split_hash
        != checkpoint_metadata.development_split_hash
    ):
        blockers.add("validation_split_hash_mismatch")
    if threshold_decision.candidate_id != selected_candidate_id:
        blockers.add("threshold_candidate_mismatch")
    if threshold_decision.checkpoint_hash != selected_checkpoint_hash:
        blockers.add("threshold_checkpoint_mismatch")
    if support_policy.selected_candidate_id != selected_candidate_id:
        blockers.add("support_candidate_mismatch")
    if support_policy.selected_checkpoint_hash != selected_checkpoint_hash:
        blockers.add("support_checkpoint_mismatch")
    if not all(
        (
            preprocessing_decision.no_external_data,
            not any(
                candidate.uses_external_artifacts for candidate in candidate_inventory.candidates
            ),
            checkpoint_metadata.no_external_data,
            validation_evidence.external_data_not_used,
            model_selection_decision.external_data_not_used,
            threshold_decision.external_data_not_used,
            support_policy.external_support_labels_forbidden,
            support_policy.external_prototype_construction_forbidden,
        )
    ):
        blockers.add("external_evidence_used")
    if not all(
        (
            validation_evidence.internal_test_not_used,
            model_selection_decision.internal_test_not_used,
            threshold_decision.internal_test_not_used,
        )
    ):
        blockers.add("internal_test_used_for_selection")
    if preprocessing_decision.evidence_status != "freeze_ready":
        blockers.add("preprocessing_not_freeze_ready")
    if candidate_inventory.inventory_state != "freeze_ready":
        blockers.add("inventory_not_freeze_ready")
    if not checkpoint_metadata.freeze_eligible:
        blockers.add("checkpoint_not_freeze_eligible")
    if model_selection_decision.selection_status not in {
        "freeze_ready",
        "single_candidate_preregistered",
    }:
        blockers.add("model_selection_not_freeze_ready")
    if (
        model_selection_decision.selection_status == "single_candidate_preregistered"
        and len(candidate_inventory.candidates) != 1
    ):
        blockers.add("single_candidate_preregistered_inventory_cardinality_mismatch")
    if threshold_decision.freeze_status != "freeze_ready":
        blockers.add("threshold_not_freeze_ready")
    if support_policy.freeze_status != "freeze_ready":
        blockers.add("support_policy_not_freeze_ready")
    if checkpoint_metadata.completion_status != "completed":
        blockers.add("checkpoint_not_completed")
    return tuple(sorted(blockers))


def _json_to_mapping(data: bytes | str) -> MappingLike:
    try:
        decoded = json.loads(data)
    except json.JSONDecodeError as exc:
        raise Phase8InternalEvidenceSerializationError("invalid internal evidence JSON.") from exc
    if not isinstance(decoded, Mapping):
        raise Phase8InternalEvidenceSerializationError(
            "internal evidence JSON root must be an object."
        )
    return cast(MappingLike, decoded)


def _expect_mapping(value: object, *, field_name: str) -> MappingLike:
    if not isinstance(value, Mapping):
        raise Phase8InternalEvidenceSerializationError(f"{field_name} must be a mapping.")
    return cast(MappingLike, value)


def _expect_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise Phase8InternalEvidenceSerializationError(f"{field_name} must be a string.")
    return value


def _expect_optional_string(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _expect_string(value, field_name=field_name)


def _expect_bool(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise Phase8InternalEvidenceSerializationError(f"{field_name} must be a boolean.")
    return value


def _expect_int(value: object, *, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise Phase8InternalEvidenceSerializationError(f"{field_name} must be an integer.")
    return value


def _expect_optional_int(value: object, *, field_name: str) -> int | None:
    if value is None:
        return None
    return _expect_int(value, field_name=field_name)


def _expect_optional_float(value: object, *, field_name: str) -> float | None:
    if value is None:
        return None
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise Phase8InternalEvidenceSerializationError(f"{field_name} must be a number.")
    return float(value)


def _expect_list(value: object, *, field_name: str) -> list[object]:
    if not isinstance(value, list):
        raise Phase8InternalEvidenceSerializationError(f"{field_name} must be a list.")
    return value


def _expect_int_list(value: object, *, field_name: str) -> tuple[int, ...]:
    return tuple(
        _expect_int(item, field_name=f"{field_name}[{index}]")
        for index, item in enumerate(_expect_list(value, field_name=field_name))
    )


def _expect_string_tuple(value: object, *, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise Phase8InternalEvidenceSerializationError(f"{field_name} must be a list of strings.")
    if not isinstance(value, list):
        raise Phase8InternalEvidenceSerializationError(f"{field_name} must be a list of strings.")
    return tuple(
        _expect_string(item, field_name=f"{field_name}[{index}]")
        for index, item in enumerate(value)
    )


def _require_exact_fields(
    mapping: MappingLike,
    required_fields: frozenset[str],
    *,
    object_name: str,
) -> None:
    keys = set(mapping)
    missing = sorted(required_fields - keys)
    extra = sorted(keys - required_fields)
    if missing or extra:
        raise Phase8InternalEvidenceSerializationError(
            f"{object_name} fields mismatch; missing={missing!r}, extra={extra!r}."
        )


def _require_schema(value: str, expected: str) -> None:
    if value != expected:
        raise Phase8InternalEvidenceValidationError(
            f"schema_name must equal {expected!r}, got {value!r}."
        )


def _require_schema_version_exact(value: str) -> None:
    if value != PHASE8_INTERNAL_EVIDENCE_SCHEMA_VERSION:
        raise Phase8InternalEvidenceValidationError(
            f"schema_version must equal {PHASE8_INTERNAL_EVIDENCE_SCHEMA_VERSION!r}."
        )


def _require_schema_version(value: str, *, field_name: str) -> None:
    if not re.fullmatch(r"^v[1-9][0-9]*$", value):
        raise Phase8InternalEvidenceValidationError(f"{field_name} must be a vN schema version.")


def _require_allowed(value: str, allowed: frozenset[str], *, field_name: str) -> None:
    if value not in allowed:
        raise Phase8InternalEvidenceValidationError(
            f"{field_name} must be one of {sorted(allowed)!r}, got {value!r}."
        )


def _require_identifier(
    value: str,
    *,
    field_name: str,
    allow_unresolved: bool = False,
) -> None:
    if not _IDENTIFIER_RE.fullmatch(value):
        raise Phase8InternalEvidenceValidationError(
            f"{field_name} must be a lowercase safe identifier."
        )
    if not allow_unresolved and value in _UNRESOLVED_TOKENS:
        raise Phase8InternalEvidenceValidationError(f"{field_name} must not be unresolved.")
    if _contains_unresolved_token(value):
        raise Phase8InternalEvidenceValidationError(f"{field_name} must not contain placeholders.")
    _reject_pathlike_identity(value, field_name=field_name)


def _require_optional_identifier(value: str | None, *, field_name: str) -> None:
    if value is not None:
        _require_identifier(value, field_name=field_name)


def _require_sha256(value: str, *, field_name: str = "self_hash") -> None:
    if not _SHA256_RE.fullmatch(value):
        raise Phase8InternalEvidenceHashError(f"{field_name} must be a lowercase SHA-256 digest.")
    if value == _ZERO_SHA256:
        raise Phase8InternalEvidenceHashError(f"{field_name} must not be the zero SHA-256 digest.")


def _require_optional_sha256(value: str | None, *, field_name: str) -> None:
    if value is not None:
        _require_sha256(value, field_name=field_name)


def _require_git_commit(value: str, *, field_name: str) -> None:
    if not _GIT_COMMIT_RE.fullmatch(value) or value == _ZERO_SHA256:
        raise Phase8InternalEvidenceValidationError(
            f"{field_name} must be a lowercase Git commit hash."
        )


def _require_nonnegative_int(value: int, *, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise Phase8InternalEvidenceValidationError(f"{field_name} must be a nonnegative integer.")


def _require_positive_int(value: int, *, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise Phase8InternalEvidenceValidationError(f"{field_name} must be a positive integer.")


def _require_optional_probability(value: float | None, *, field_name: str) -> None:
    if value is None:
        return
    if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(value):
        raise Phase8InternalEvidenceValidationError(f"{field_name} must be finite.")
    if value < 0.0 or value > 1.0:
        raise Phase8InternalEvidenceValidationError(
            f"{field_name} must be within the probability interval [0, 1]."
        )


def _require_true(value: bool, *, field_name: str) -> None:
    if value is not True:
        raise Phase8InternalEvidenceValidationError(f"{field_name} must be true.")


def _require_false(value: bool, *, field_name: str) -> None:
    if value is not False:
        raise Phase8InternalEvidenceValidationError(f"{field_name} must be false.")


def _require_resolved_references(*references: ArtifactReference | None) -> None:
    for reference in references:
        if reference is None:
            raise Phase8InternalEvidenceValidationError(
                "freeze-ready evidence requires resolved references."
            )


def _require_self_hash(value: str, expected: str, *, field_name: str = "self_hash") -> None:
    _require_sha256(value, field_name=field_name)
    if value != expected:
        raise Phase8InternalEvidenceHashError(f"{field_name} does not match canonical identity.")


def _reject_pathlike_identity(value: str, *, field_name: str) -> None:
    if value.startswith("/") or value.startswith("~") or "\\" in value:
        raise Phase8InternalEvidenceValidationError(
            f"{field_name} must not contain local path identity."
        )
    if "://" in value or ":" in value:
        raise Phase8InternalEvidenceValidationError(
            f"{field_name} must not contain URI or absolute path identity."
        )


def _contains_unresolved_token(value: str) -> bool:
    return any(part in _UNRESOLVED_TOKENS for part in re.split(r"[_.@:/-]+", value))


__all__ = [
    "PHASE8_CHECKPOINT_METADATA_SCHEMA_NAME",
    "PHASE8_FIXED_CANDIDATE_INVENTORY_SCHEMA_NAME",
    "PHASE8_INTERNAL_EVIDENCE_PACKAGE_SCHEMA_NAME",
    "PHASE8_INTERNAL_EVIDENCE_SCHEMA_VERSION",
    "PHASE8_MODEL_SELECTION_DECISION_SCHEMA_NAME",
    "PHASE8_PREPROCESSING_DECISION_SCHEMA_NAME",
    "PHASE8_SUPPORT_POLICY_SCHEMA_NAME",
    "PHASE8_THRESHOLD_DECISION_SCHEMA_NAME",
    "PHASE8_VALIDATION_EVIDENCE_REFERENCE_SCHEMA_NAME",
    "ArtifactReference",
    "CandidateDefinition",
    "FailedCandidateAccounting",
    "Phase8CheckpointMetadata",
    "Phase8FixedCandidateInventory",
    "Phase8InternalEvidenceError",
    "Phase8InternalEvidenceHashError",
    "Phase8InternalEvidencePackage",
    "Phase8InternalEvidenceSerializationError",
    "Phase8InternalEvidenceValidationError",
    "Phase8ModelSelectionDecision",
    "Phase8PreprocessingDecision",
    "Phase8SupportPolicy",
    "Phase8ThresholdDecision",
    "Phase8ValidationEvidenceReference",
    "artifact_reference_from_mapping",
    "artifact_reference_to_dict",
    "candidate_definition_to_dict",
    "failed_candidate_accounting_to_dict",
    "hash_phase8_checkpoint_metadata",
    "hash_phase8_fixed_candidate_inventory",
    "hash_phase8_internal_evidence_package",
    "hash_phase8_model_selection_decision",
    "hash_phase8_preprocessing_decision",
    "hash_phase8_support_policy",
    "hash_phase8_threshold_decision",
    "hash_phase8_validation_evidence_reference",
    "phase8_checkpoint_metadata_from_mapping",
    "phase8_checkpoint_metadata_identity_payload",
    "phase8_checkpoint_metadata_to_dict",
    "phase8_fixed_candidate_inventory_from_mapping",
    "phase8_fixed_candidate_inventory_identity_payload",
    "phase8_fixed_candidate_inventory_to_dict",
    "phase8_internal_evidence_package_from_json",
    "phase8_internal_evidence_package_from_mapping",
    "phase8_internal_evidence_package_identity_payload",
    "phase8_internal_evidence_package_to_dict",
    "phase8_internal_evidence_package_to_json",
    "phase8_model_selection_decision_from_mapping",
    "phase8_model_selection_decision_identity_payload",
    "phase8_model_selection_decision_to_dict",
    "phase8_preprocessing_decision_from_mapping",
    "phase8_preprocessing_decision_identity_payload",
    "phase8_preprocessing_decision_to_dict",
    "phase8_support_policy_from_mapping",
    "phase8_support_policy_identity_payload",
    "phase8_support_policy_to_dict",
    "phase8_threshold_decision_from_mapping",
    "phase8_threshold_decision_identity_payload",
    "phase8_threshold_decision_to_dict",
    "phase8_validation_evidence_reference_from_mapping",
    "phase8_validation_evidence_reference_identity_payload",
    "phase8_validation_evidence_reference_to_dict",
    "verify_checkpoint_file_identity_values",
]
