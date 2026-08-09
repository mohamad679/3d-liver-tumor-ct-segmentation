"""Phase 8 definitive-development training contracts and plumbing (Substage 4A).

This module is metadata-only, contract-and-boundary plumbing for a *future*
definitive MONAI SegResNet development-training run and validation-only model
selection. It performs no filesystem access to image, label, prediction, or
checkpoint bytes; it never constructs a model, loads training data, runs
inference, computes a real metric, or trains anything.

Two guarding invariants hold for every code path reachable from this module's
public API in this session:

* :func:`build_phase8_definitive_training_config` can only ever produce a
  :class:`Phase8DefinitiveTrainingConfig` whose ``execution_release_state``
  equals ``"awaiting_explicit_user_approval"``.
* :func:`build_unreleased_definitive_execution_release` can only ever produce
  a :class:`Phase8DefinitiveExecutionRelease` whose ``release_state`` equals
  ``"not_released"``. No function reachable from this module's public API
  (and no CLI command) can construct a ``released`` release. A ``released``
  release can only be exercised by directly constructing the frozen
  dataclass, which this module does not do anywhere in its own source.

Every execution and publication entry point in this module
(:func:`execute_phase8_definitive_training` and the ``publish_phase8_definitive_*``
functions) validates that a supplied release is bound to the supplied config
by hash and that its ``release_state`` equals ``"released"`` *before* opening
any file path, reading any manifest or split, or importing any heavy runtime
dependency. Since no ``released`` release can be produced from this module's
own public API, every real invocation of these entry points in this
environment fails closed with :class:`Phase8DefinitiveTrainingNotReleasedError`
(or a hash-mismatch :class:`Phase8DefinitiveTrainingValidationError`) -- that
is intentional and is exactly what the boundary tests in this remediation
substage prove.

Threshold policy is fixed to ``fixed_constant`` at ``0.5`` and support policy
is fixed to ``no_support`` for the single preregistered candidate
``monai_segresnet_baseline``; both are embedded directly in
:class:`Phase8DefinitiveTrainingConfig` rather than referenced indirectly,
because the config must be fully self-describing and hashable without
depending on a separately mutable freeze-state artifact.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol, TypeAlias, cast

from protoem_ct.artifacts.hashing import JsonValue, canonical_json_bytes, sha256_file, sha256_json
from protoem_ct.data._phase2_publication import (
    Phase2PublicationExistingOutputError,
    Phase2PublicationIOError,
    publish_text_no_overwrite,
)
from protoem_ct.data.phase2_paths import (
    InvalidDatasetRootError,
    validate_explicit_external_output_root,
)
from protoem_ct.external.internal_evidence import (
    ArtifactReference,
    Phase8CheckpointMetadata,
    Phase8FixedCandidateInventory,
    Phase8ModelSelectionDecision,
    Phase8PreprocessingDecision,
    Phase8SupportPolicy,
    Phase8ThresholdDecision,
    Phase8ValidationEvidenceReference,
    artifact_reference_from_mapping,
    artifact_reference_to_dict,
    phase8_checkpoint_metadata_to_dict,
    phase8_model_selection_decision_to_dict,
    phase8_preprocessing_decision_to_dict,
    phase8_support_policy_to_dict,
    phase8_threshold_decision_to_dict,
    phase8_validation_evidence_reference_to_dict,
)

MappingLike: TypeAlias = Mapping[str, object]

# ---------------------------------------------------------------------------
# Schema identity
# ---------------------------------------------------------------------------

PHASE8_DEFINITIVE_TRAINING_SCHEMA_VERSION: Final[str] = "v1"
PHASE8_DEFINITIVE_TRAINING_CONFIG_SCHEMA_NAME: Final[str] = "phase8_definitive_training_config"
PHASE8_DEFINITIVE_EXECUTION_RELEASE_SCHEMA_NAME: Final[str] = "phase8_definitive_execution_release"

PHASE8_DEFINITIVE_TRAINING_CONFIG_FILENAME: Final[str] = "phase8_definitive_training_config.json"
PHASE8_DEFINITIVE_EXECUTION_RELEASE_FILENAME: Final[str] = (
    "phase8_definitive_execution_release.json"
)
PHASE8_DEFINITIVE_PREPROCESSING_EVIDENCE_FILENAME: Final[str] = (
    "phase8_definitive_preprocessing_evidence.json"
)
PHASE8_DEFINITIVE_CHECKPOINT_METADATA_FILENAME: Final[str] = (
    "phase8_definitive_checkpoint_metadata.json"
)
PHASE8_DEFINITIVE_VALIDATION_EVIDENCE_FILENAME: Final[str] = (
    "phase8_definitive_validation_evidence.json"
)
PHASE8_DEFINITIVE_MODEL_SELECTION_FILENAME: Final[str] = "phase8_definitive_model_selection.json"
PHASE8_DEFINITIVE_THRESHOLD_DECISION_FILENAME: Final[str] = (
    "phase8_definitive_threshold_decision.json"
)
PHASE8_DEFINITIVE_SUPPORT_POLICY_FILENAME: Final[str] = "phase8_definitive_support_policy.json"

REQUIRED_CANDIDATE_ID: Final[str] = "monai_segresnet_baseline"
REQUIRED_DEVICE_TYPE: Final[str] = "cpu"
REQUIRED_THRESHOLD_POLICY_TYPE: Final[str] = "fixed_constant"
REQUIRED_THRESHOLD_VALUE: Final[float] = 0.5
REQUIRED_SUPPORT_POLICY_TYPE: Final[str] = "no_support"
AWAITING_EXPLICIT_USER_APPROVAL: Final[str] = "awaiting_explicit_user_approval"
RELEASE_STATE_NOT_RELEASED: Final[str] = "not_released"
RELEASE_STATE_RELEASED: Final[str] = "released"
RELEASE_AUTHORIZED_BY_EXPLICIT_USER_APPROVAL: Final[str] = "explicit_user_approval"
SELECTION_STATUS_SINGLE_CANDIDATE_PREREGISTERED: Final[str] = "single_candidate_preregistered"
PRIMARY_METRIC_TUMOR_DICE: Final[str] = "tumor_dice"

# Deterministic tie-break rule identifiers for this module's single-candidate
# selection publisher. This order intentionally differs from the general
# multi-candidate order recorded in
# docs/phase8/INTERNAL_EVIDENCE_REMEDIATION_PLAN.md (which starts with
# validation tumor Dice descending); this module's fixed tuple is exactly:
# lesion F1 descending, then HD95 ascending when defined, then false-positive
# lesions per scan ascending, then trainable parameter count ascending, then
# candidate ID lexicographic.
DEFINITIVE_TIE_BREAK_ORDER: Final[tuple[str, ...]] = (
    "lesion_f1_desc",
    "hd95_asc_when_defined",
    "fp_lesions_per_scan_asc",
    "trainable_param_count_asc",
    "candidate_id_lexicographic",
)

_ALLOWED_RELEASE_STATES: Final[frozenset[str]] = frozenset(
    {RELEASE_STATE_NOT_RELEASED, RELEASE_STATE_RELEASED}
)
_ARTIFACT_ROLE_DENYLIST_SUBSTRINGS: Final[tuple[str, ...]] = ("internal_test", "external")

_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z0-9](?:[a-z0-9_.-]{0,126}[a-z0-9])?$")
_ZERO_SHA256: Final[str] = "0" * 64
_UNRESOLVED_TOKENS: Final[frozenset[str]] = frozenset(
    {"pending", "placeholder", "tbd", "todo", "unknown", "unresolved"}
)


class Phase8DefinitiveTrainingError(ValueError):
    """Base error for the Phase 8 definitive-training contracts and plumbing."""


class Phase8DefinitiveTrainingValidationError(Phase8DefinitiveTrainingError):
    """Raised when a definitive-training contract violates its invariants."""


class Phase8DefinitiveTrainingSerializationError(Phase8DefinitiveTrainingError):
    """Raised when a definitive-training mapping cannot be reconstructed strictly."""


class Phase8DefinitiveTrainingHashError(Phase8DefinitiveTrainingError):
    """Raised when a definitive-training self-hash or bound hash does not match."""


class Phase8DefinitiveTrainingPublicationError(Phase8DefinitiveTrainingError):
    """Raised when guarded definitive-training publication fails safely."""


class Phase8DefinitiveTrainingNotReleasedError(Phase8DefinitiveTrainingError):
    """Raised when definitive execution or publication is attempted without release.

    Raised before any file path is opened, before any manifest or split file
    is read, and before any heavy runtime dependency is imported.
    """


# ---------------------------------------------------------------------------
# A. Definitive training configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Phase8DefinitiveTrainingConfig:
    """Versioned, self-describing definitive-training configuration contract."""

    schema_name: str
    schema_version: str
    candidate_id: str
    input_binding_hash: str
    fixed_candidate_inventory_hash: str
    preprocessing_decision_reference: ArtifactReference
    training_config_reference: ArtifactReference
    fixed_seeds: tuple[int, ...]
    threshold_policy_type: str
    threshold_value: float
    support_policy_type: str
    device_type: str
    amp_enabled: bool
    no_external_data: bool
    internal_test_excluded: bool
    execution_release_state: str
    config_hash: str

    def __post_init__(self) -> None:
        _require_schema(self.schema_name, PHASE8_DEFINITIVE_TRAINING_CONFIG_SCHEMA_NAME)
        _require_schema_version_exact(self.schema_version)
        if self.candidate_id != REQUIRED_CANDIDATE_ID:
            raise Phase8DefinitiveTrainingValidationError(
                f"candidate_id must equal {REQUIRED_CANDIDATE_ID!r}."
            )
        _require_sha256(self.input_binding_hash, field_name="input_binding_hash")
        _require_sha256(
            self.fixed_candidate_inventory_hash,
            field_name="fixed_candidate_inventory_hash",
        )
        object.__setattr__(self, "fixed_seeds", tuple(self.fixed_seeds))
        if not self.fixed_seeds:
            raise Phase8DefinitiveTrainingValidationError("fixed_seeds must not be empty.")
        for index, seed in enumerate(self.fixed_seeds):
            _require_nonnegative_int(seed, field_name=f"fixed_seeds[{index}]")
        if self.threshold_policy_type != REQUIRED_THRESHOLD_POLICY_TYPE:
            raise Phase8DefinitiveTrainingValidationError(
                f"threshold_policy_type must equal {REQUIRED_THRESHOLD_POLICY_TYPE!r}."
            )
        if self.threshold_value != REQUIRED_THRESHOLD_VALUE:
            raise Phase8DefinitiveTrainingValidationError(
                f"threshold_value must equal {REQUIRED_THRESHOLD_VALUE!r}."
            )
        if self.support_policy_type != REQUIRED_SUPPORT_POLICY_TYPE:
            raise Phase8DefinitiveTrainingValidationError(
                f"support_policy_type must equal {REQUIRED_SUPPORT_POLICY_TYPE!r}."
            )
        if self.device_type != REQUIRED_DEVICE_TYPE:
            raise Phase8DefinitiveTrainingValidationError(
                f"device_type must equal {REQUIRED_DEVICE_TYPE!r}."
            )
        _require_false(self.amp_enabled, field_name="amp_enabled")
        _require_true(self.no_external_data, field_name="no_external_data")
        _require_true(self.internal_test_excluded, field_name="internal_test_excluded")
        if self.execution_release_state != AWAITING_EXPLICIT_USER_APPROVAL:
            raise Phase8DefinitiveTrainingValidationError(
                f"execution_release_state must equal {AWAITING_EXPLICIT_USER_APPROVAL!r}."
            )
        _require_self_hash(
            self.config_hash,
            _config_identity_payload(
                schema_name=self.schema_name,
                schema_version=self.schema_version,
                candidate_id=self.candidate_id,
                input_binding_hash=self.input_binding_hash,
                fixed_candidate_inventory_hash=self.fixed_candidate_inventory_hash,
                preprocessing_decision_reference=self.preprocessing_decision_reference,
                training_config_reference=self.training_config_reference,
                fixed_seeds=self.fixed_seeds,
                threshold_policy_type=self.threshold_policy_type,
                threshold_value=self.threshold_value,
                support_policy_type=self.support_policy_type,
                device_type=self.device_type,
                amp_enabled=self.amp_enabled,
                no_external_data=self.no_external_data,
                internal_test_excluded=self.internal_test_excluded,
                execution_release_state=self.execution_release_state,
            ),
            field_name="config_hash",
        )


def _config_identity_payload(
    *,
    schema_name: str,
    schema_version: str,
    candidate_id: str,
    input_binding_hash: str,
    fixed_candidate_inventory_hash: str,
    preprocessing_decision_reference: ArtifactReference,
    training_config_reference: ArtifactReference,
    fixed_seeds: tuple[int, ...],
    threshold_policy_type: str,
    threshold_value: float,
    support_policy_type: str,
    device_type: str,
    amp_enabled: bool,
    no_external_data: bool,
    internal_test_excluded: bool,
    execution_release_state: str,
) -> dict[str, JsonValue]:
    return {
        "amp_enabled": amp_enabled,
        "candidate_id": candidate_id,
        "device_type": device_type,
        "execution_release_state": execution_release_state,
        "fixed_candidate_inventory_hash": fixed_candidate_inventory_hash,
        "fixed_seeds": list(fixed_seeds),
        "input_binding_hash": input_binding_hash,
        "internal_test_excluded": internal_test_excluded,
        "no_external_data": no_external_data,
        "preprocessing_decision_reference": artifact_reference_to_dict(
            preprocessing_decision_reference
        ),
        "schema_name": schema_name,
        "schema_version": schema_version,
        "support_policy_type": support_policy_type,
        "threshold_policy_type": threshold_policy_type,
        "threshold_value": threshold_value,
        "training_config_reference": artifact_reference_to_dict(training_config_reference),
    }


def phase8_definitive_training_config_identity_payload(
    config: Phase8DefinitiveTrainingConfig,
) -> dict[str, JsonValue]:
    """Return the canonical identity payload for a definitive-training config."""

    return _config_identity_payload(
        schema_name=config.schema_name,
        schema_version=config.schema_version,
        candidate_id=config.candidate_id,
        input_binding_hash=config.input_binding_hash,
        fixed_candidate_inventory_hash=config.fixed_candidate_inventory_hash,
        preprocessing_decision_reference=config.preprocessing_decision_reference,
        training_config_reference=config.training_config_reference,
        fixed_seeds=config.fixed_seeds,
        threshold_policy_type=config.threshold_policy_type,
        threshold_value=config.threshold_value,
        support_policy_type=config.support_policy_type,
        device_type=config.device_type,
        amp_enabled=config.amp_enabled,
        no_external_data=config.no_external_data,
        internal_test_excluded=config.internal_test_excluded,
        execution_release_state=config.execution_release_state,
    )


def hash_phase8_definitive_training_config(config: Phase8DefinitiveTrainingConfig) -> str:
    """Return the canonical self-hash for a definitive-training config."""

    return sha256_json(phase8_definitive_training_config_identity_payload(config))


def phase8_definitive_training_config_to_dict(
    config: Phase8DefinitiveTrainingConfig,
) -> dict[str, JsonValue]:
    """Convert a definitive-training config to a canonical mapping."""

    payload = phase8_definitive_training_config_identity_payload(config)
    payload["config_hash"] = config.config_hash
    return payload


def phase8_definitive_training_config_to_json(config: Phase8DefinitiveTrainingConfig) -> bytes:
    """Serialize a definitive-training config to canonical JSON bytes."""

    return canonical_json_bytes(phase8_definitive_training_config_to_dict(config)) + b"\n"


_CONFIG_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_name",
        "schema_version",
        "candidate_id",
        "input_binding_hash",
        "fixed_candidate_inventory_hash",
        "preprocessing_decision_reference",
        "training_config_reference",
        "fixed_seeds",
        "threshold_policy_type",
        "threshold_value",
        "support_policy_type",
        "device_type",
        "amp_enabled",
        "no_external_data",
        "internal_test_excluded",
        "execution_release_state",
        "config_hash",
    }
)


def phase8_definitive_training_config_from_mapping(
    mapping: MappingLike,
) -> Phase8DefinitiveTrainingConfig:
    """Reconstruct a definitive-training config from a strict mapping."""

    _require_exact_fields(mapping, _CONFIG_FIELDS, object_name="Phase8DefinitiveTrainingConfig")
    return Phase8DefinitiveTrainingConfig(
        schema_name=_expect_string(mapping["schema_name"], field_name="schema_name"),
        schema_version=_expect_string(mapping["schema_version"], field_name="schema_version"),
        candidate_id=_expect_string(mapping["candidate_id"], field_name="candidate_id"),
        input_binding_hash=_expect_string(
            mapping["input_binding_hash"], field_name="input_binding_hash"
        ),
        fixed_candidate_inventory_hash=_expect_string(
            mapping["fixed_candidate_inventory_hash"],
            field_name="fixed_candidate_inventory_hash",
        ),
        preprocessing_decision_reference=artifact_reference_from_mapping(
            _expect_mapping(
                mapping["preprocessing_decision_reference"],
                field_name="preprocessing_decision_reference",
            )
        ),
        training_config_reference=artifact_reference_from_mapping(
            _expect_mapping(
                mapping["training_config_reference"], field_name="training_config_reference"
            )
        ),
        fixed_seeds=_expect_int_tuple(mapping["fixed_seeds"], field_name="fixed_seeds"),
        threshold_policy_type=_expect_string(
            mapping["threshold_policy_type"], field_name="threshold_policy_type"
        ),
        threshold_value=_expect_float(mapping["threshold_value"], field_name="threshold_value"),
        support_policy_type=_expect_string(
            mapping["support_policy_type"], field_name="support_policy_type"
        ),
        device_type=_expect_string(mapping["device_type"], field_name="device_type"),
        amp_enabled=_expect_bool(mapping["amp_enabled"], field_name="amp_enabled"),
        no_external_data=_expect_bool(mapping["no_external_data"], field_name="no_external_data"),
        internal_test_excluded=_expect_bool(
            mapping["internal_test_excluded"], field_name="internal_test_excluded"
        ),
        execution_release_state=_expect_string(
            mapping["execution_release_state"], field_name="execution_release_state"
        ),
        config_hash=_expect_string(mapping["config_hash"], field_name="config_hash"),
    )


def build_phase8_definitive_training_config(
    *,
    input_binding_hash: str,
    fixed_candidate_inventory_hash: str,
    preprocessing_decision_reference: ArtifactReference,
    training_config_reference: ArtifactReference,
    fixed_seeds: tuple[int, ...],
) -> Phase8DefinitiveTrainingConfig:
    """Build a definitive-training config awaiting explicit user approval.

    There is no parameter capable of setting ``candidate_id`` to anything
    other than ``monai_segresnet_baseline``, ``device_type`` to anything
    other than ``"cpu"``, ``amp_enabled`` to ``True``, the threshold policy to
    anything other than the fixed 0.5 constant, the support policy to
    anything other than ``no_support``, or ``execution_release_state`` to
    anything other than ``"awaiting_explicit_user_approval"``.
    """

    fixed_seeds_tuple = tuple(fixed_seeds)
    payload = _config_identity_payload(
        schema_name=PHASE8_DEFINITIVE_TRAINING_CONFIG_SCHEMA_NAME,
        schema_version=PHASE8_DEFINITIVE_TRAINING_SCHEMA_VERSION,
        candidate_id=REQUIRED_CANDIDATE_ID,
        input_binding_hash=input_binding_hash,
        fixed_candidate_inventory_hash=fixed_candidate_inventory_hash,
        preprocessing_decision_reference=preprocessing_decision_reference,
        training_config_reference=training_config_reference,
        fixed_seeds=fixed_seeds_tuple,
        threshold_policy_type=REQUIRED_THRESHOLD_POLICY_TYPE,
        threshold_value=REQUIRED_THRESHOLD_VALUE,
        support_policy_type=REQUIRED_SUPPORT_POLICY_TYPE,
        device_type=REQUIRED_DEVICE_TYPE,
        amp_enabled=False,
        no_external_data=True,
        internal_test_excluded=True,
        execution_release_state=AWAITING_EXPLICIT_USER_APPROVAL,
    )
    return Phase8DefinitiveTrainingConfig(
        schema_name=PHASE8_DEFINITIVE_TRAINING_CONFIG_SCHEMA_NAME,
        schema_version=PHASE8_DEFINITIVE_TRAINING_SCHEMA_VERSION,
        candidate_id=REQUIRED_CANDIDATE_ID,
        input_binding_hash=input_binding_hash,
        fixed_candidate_inventory_hash=fixed_candidate_inventory_hash,
        preprocessing_decision_reference=preprocessing_decision_reference,
        training_config_reference=training_config_reference,
        fixed_seeds=fixed_seeds_tuple,
        threshold_policy_type=REQUIRED_THRESHOLD_POLICY_TYPE,
        threshold_value=REQUIRED_THRESHOLD_VALUE,
        support_policy_type=REQUIRED_SUPPORT_POLICY_TYPE,
        device_type=REQUIRED_DEVICE_TYPE,
        amp_enabled=False,
        no_external_data=True,
        internal_test_excluded=True,
        execution_release_state=AWAITING_EXPLICIT_USER_APPROVAL,
        config_hash=sha256_json(payload),
    )


# ---------------------------------------------------------------------------
# B. Definitive execution release
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Phase8DefinitiveExecutionRelease:
    """Explicit, separately versioned authorization to run one definitive config.

    No function reachable from this module's public API (and no CLI command)
    can construct an instance with ``release_state == "released"``. Only
    direct dataclass construction -- exercised solely inside this
    remediation substage's own tests -- can produce one, to prove the
    contract's shape without ever emitting a real-release code path.
    """

    schema_name: str
    schema_version: str
    release_state: str
    bound_config_hash: str
    release_scope_description: str
    authorized_max_training_steps: int
    authorized_by: str | None
    release_hash: str

    def __post_init__(self) -> None:
        _require_schema(self.schema_name, PHASE8_DEFINITIVE_EXECUTION_RELEASE_SCHEMA_NAME)
        _require_schema_version_exact(self.schema_version)
        _require_allowed(self.release_state, _ALLOWED_RELEASE_STATES, field_name="release_state")
        _require_sha256(self.bound_config_hash, field_name="bound_config_hash")
        _require_identifier(self.release_scope_description, field_name="release_scope_description")
        _require_positive_int(
            self.authorized_max_training_steps, field_name="authorized_max_training_steps"
        )
        if self.release_state == RELEASE_STATE_RELEASED:
            if self.authorized_by != RELEASE_AUTHORIZED_BY_EXPLICIT_USER_APPROVAL:
                raise Phase8DefinitiveTrainingValidationError(
                    "released release requires "
                    f"authorized_by={RELEASE_AUTHORIZED_BY_EXPLICIT_USER_APPROVAL!r}."
                )
        elif self.authorized_by is not None:
            raise Phase8DefinitiveTrainingValidationError(
                "not_released release must not include authorized_by."
            )
        _require_self_hash(
            self.release_hash,
            _release_identity_payload(
                schema_name=self.schema_name,
                schema_version=self.schema_version,
                release_state=self.release_state,
                bound_config_hash=self.bound_config_hash,
                release_scope_description=self.release_scope_description,
                authorized_max_training_steps=self.authorized_max_training_steps,
                authorized_by=self.authorized_by,
            ),
            field_name="release_hash",
        )


def _release_identity_payload(
    *,
    schema_name: str,
    schema_version: str,
    release_state: str,
    bound_config_hash: str,
    release_scope_description: str,
    authorized_max_training_steps: int,
    authorized_by: str | None,
) -> dict[str, JsonValue]:
    return {
        "authorized_by": authorized_by,
        "authorized_max_training_steps": authorized_max_training_steps,
        "bound_config_hash": bound_config_hash,
        "release_scope_description": release_scope_description,
        "release_state": release_state,
        "schema_name": schema_name,
        "schema_version": schema_version,
    }


def phase8_definitive_execution_release_identity_payload(
    release: Phase8DefinitiveExecutionRelease,
) -> dict[str, JsonValue]:
    """Return the canonical identity payload for an execution release."""

    return _release_identity_payload(
        schema_name=release.schema_name,
        schema_version=release.schema_version,
        release_state=release.release_state,
        bound_config_hash=release.bound_config_hash,
        release_scope_description=release.release_scope_description,
        authorized_max_training_steps=release.authorized_max_training_steps,
        authorized_by=release.authorized_by,
    )


def hash_phase8_definitive_execution_release(release: Phase8DefinitiveExecutionRelease) -> str:
    """Return the canonical self-hash for an execution release."""

    return sha256_json(phase8_definitive_execution_release_identity_payload(release))


def phase8_definitive_execution_release_to_dict(
    release: Phase8DefinitiveExecutionRelease,
) -> dict[str, JsonValue]:
    """Convert an execution release to a canonical mapping."""

    payload = phase8_definitive_execution_release_identity_payload(release)
    payload["release_hash"] = release.release_hash
    return payload


def phase8_definitive_execution_release_to_json(
    release: Phase8DefinitiveExecutionRelease,
) -> bytes:
    """Serialize an execution release to canonical JSON bytes."""

    return canonical_json_bytes(phase8_definitive_execution_release_to_dict(release)) + b"\n"


_RELEASE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_name",
        "schema_version",
        "release_state",
        "bound_config_hash",
        "release_scope_description",
        "authorized_max_training_steps",
        "authorized_by",
        "release_hash",
    }
)


def phase8_definitive_execution_release_from_mapping(
    mapping: MappingLike,
) -> Phase8DefinitiveExecutionRelease:
    """Reconstruct an execution release from a strict mapping."""

    _require_exact_fields(mapping, _RELEASE_FIELDS, object_name="Phase8DefinitiveExecutionRelease")
    return Phase8DefinitiveExecutionRelease(
        schema_name=_expect_string(mapping["schema_name"], field_name="schema_name"),
        schema_version=_expect_string(mapping["schema_version"], field_name="schema_version"),
        release_state=_expect_string(mapping["release_state"], field_name="release_state"),
        bound_config_hash=_expect_string(
            mapping["bound_config_hash"], field_name="bound_config_hash"
        ),
        release_scope_description=_expect_string(
            mapping["release_scope_description"], field_name="release_scope_description"
        ),
        authorized_max_training_steps=_expect_int(
            mapping["authorized_max_training_steps"],
            field_name="authorized_max_training_steps",
        ),
        authorized_by=_expect_optional_string(mapping["authorized_by"], field_name="authorized_by"),
        release_hash=_expect_string(mapping["release_hash"], field_name="release_hash"),
    )


def build_unreleased_definitive_execution_release(
    *,
    bound_config_hash: str,
    release_scope_description: str,
    authorized_max_training_steps: int,
) -> Phase8DefinitiveExecutionRelease:
    """Build a ``not_released`` execution release. Never produces ``released``.

    There is no parameter capable of setting ``release_state`` to
    ``"released"`` or ``authorized_by`` to any value.
    """

    payload = _release_identity_payload(
        schema_name=PHASE8_DEFINITIVE_EXECUTION_RELEASE_SCHEMA_NAME,
        schema_version=PHASE8_DEFINITIVE_TRAINING_SCHEMA_VERSION,
        release_state=RELEASE_STATE_NOT_RELEASED,
        bound_config_hash=bound_config_hash,
        release_scope_description=release_scope_description,
        authorized_max_training_steps=authorized_max_training_steps,
        authorized_by=None,
    )
    return Phase8DefinitiveExecutionRelease(
        schema_name=PHASE8_DEFINITIVE_EXECUTION_RELEASE_SCHEMA_NAME,
        schema_version=PHASE8_DEFINITIVE_TRAINING_SCHEMA_VERSION,
        release_state=RELEASE_STATE_NOT_RELEASED,
        bound_config_hash=bound_config_hash,
        release_scope_description=release_scope_description,
        authorized_max_training_steps=authorized_max_training_steps,
        authorized_by=None,
        release_hash=sha256_json(payload),
    )


# ---------------------------------------------------------------------------
# C. Fail-closed release boundary
# ---------------------------------------------------------------------------


def _gate_release(
    config: Phase8DefinitiveTrainingConfig,
    release: Phase8DefinitiveExecutionRelease,
) -> None:
    """Validate release binding and release state before any I/O or heavy import.

    Checks the hash binding first (independent of release state), then the
    release state. Both checks run before this module opens any file path,
    reads any manifest or split, or imports torch/monai/nibabel.
    """

    if release.bound_config_hash != config.config_hash:
        raise Phase8DefinitiveTrainingValidationError(
            "release bound_config_hash does not match the supplied config_hash."
        )
    if release.release_state != RELEASE_STATE_RELEASED:
        raise Phase8DefinitiveTrainingNotReleasedError(
            "definitive execution/publication is not released: explicit user "
            "approval is required and no code path in this session's tooling "
            "can produce a released execution release."
        )


class Phase8DefinitiveTrainingExecutor(Protocol):
    """Method surface a future definitive-training executor must implement.

    Every method here is a typed stub only. This module never calls, imports
    an implementation of, or otherwise exercises any of these methods.
    """

    def construct_model(self, config: Phase8DefinitiveTrainingConfig) -> object:
        """Construct the model architecture described by the config."""
        raise NotImplementedError

    def load_training_data(self, config: Phase8DefinitiveTrainingConfig) -> object:
        """Load the train-partition data referenced by the bound input binding."""
        raise NotImplementedError

    def run_validation_inference(self, config: Phase8DefinitiveTrainingConfig) -> object:
        """Run validation-partition inference and return raw predictions."""
        raise NotImplementedError

    def persist_checkpoint(self, config: Phase8DefinitiveTrainingConfig, model: object) -> object:
        """Persist a trained checkpoint and return its identity metadata."""
        raise NotImplementedError

    def publish_validation_metrics(
        self, config: Phase8DefinitiveTrainingConfig, predictions: object
    ) -> object:
        """Compute and publish development-validation metrics evidence."""
        raise NotImplementedError

    def publish_preprocessing_evidence(self, config: Phase8DefinitiveTrainingConfig) -> object:
        """Publish the preprocessing evidence actually applied during execution."""
        raise NotImplementedError

    def publish_checkpoint_metadata(
        self, config: Phase8DefinitiveTrainingConfig, checkpoint: object
    ) -> object:
        """Publish checkpoint metadata linking the checkpoint to this config."""
        raise NotImplementedError


def execute_phase8_definitive_training(
    config: Phase8DefinitiveTrainingConfig,
    release: Phase8DefinitiveExecutionRelease,
    executor: Phase8DefinitiveTrainingExecutor | None = None,
    **kwargs: object,
) -> None:
    """Fail closed unless bound to an explicitly released config.

    Validates the release binding and release state before opening any file
    path, reading any manifest or split, or importing torch/monai/nibabel.
    Since no code path in this session's own tooling can produce a released
    release, every real invocation of this function in this environment
    raises :class:`Phase8DefinitiveTrainingNotReleasedError` (or, for a
    mismatched hash, :class:`Phase8DefinitiveTrainingValidationError`) --
    that is intentional.
    """

    _gate_release(config, release)
    if executor is None:
        raise Phase8DefinitiveTrainingValidationError(
            "no Phase8DefinitiveTrainingExecutor implementation was supplied."
        )
    executor.construct_model(config)


# ---------------------------------------------------------------------------
# D. Output-root safety
# ---------------------------------------------------------------------------


def validate_phase8_definitive_output_root(*, output_root: Path, repository_root: Path) -> Path:
    """Validate an explicit, non-symlinked, outside-repository output root."""

    if output_root.exists() and output_root.is_symlink():
        raise Phase8DefinitiveTrainingPublicationError("output_root must not be a symlink.")
    try:
        return validate_explicit_external_output_root(
            output_root, forbidden_roots=(repository_root,)
        )
    except InvalidDatasetRootError as exc:
        raise Phase8DefinitiveTrainingPublicationError(
            "invalid definitive-training output root."
        ) from exc


def _require_output_root_empty_or_absent(output_root: Path) -> None:
    if output_root.exists() and any(output_root.iterdir()):
        raise Phase8DefinitiveTrainingPublicationError(
            "definitive-training output root already contains files."
        )


def _publish_json_bytes(path: Path, data: bytes) -> None:
    try:
        publish_text_no_overwrite(
            text=data.decode("utf-8"),
            output_path=path,
            temporary_exists_message=(
                "temporary Phase 8 definitive-training output already exists"
            ),
            final_exists_message="Phase 8 definitive-training output already exists",
        )
    except (Phase2PublicationExistingOutputError, Phase2PublicationIOError) as exc:
        raise Phase8DefinitiveTrainingPublicationError(
            "failed to publish definitive-training artifact."
        ) from exc


# ---------------------------------------------------------------------------
# E. Plan-only publication (config + unreleased release)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Phase8DefinitiveTrainingPlanPublicationResult:
    """Result of one guarded definitive-training plan-only publication."""

    output_root: Path
    config_hash: str
    release_hash: str
    artifact_hashes: dict[str, str]


def publish_phase8_definitive_training_plan(
    *,
    config: Phase8DefinitiveTrainingConfig,
    release: Phase8DefinitiveExecutionRelease,
    output_root: Path,
    repository_root: Path,
) -> Phase8DefinitiveTrainingPlanPublicationResult:
    """Publish a definitive-training config and its unreleased execution release.

    This is planning-only: it never opens an image, label, prediction, or
    checkpoint file, and it never trains, infers, or computes a real metric.
    The published release's ``release_state`` is always ``"not_released"``
    because :func:`build_unreleased_definitive_execution_release` is the
    only release-builder reachable from this function.
    """

    if release.bound_config_hash != config.config_hash:
        raise Phase8DefinitiveTrainingPublicationError(
            "release does not reference the supplied config."
        )
    canonical_repository_root = repository_root.resolve(strict=True)
    canonical_output_root = validate_phase8_definitive_output_root(
        output_root=output_root, repository_root=canonical_repository_root
    )
    _require_output_root_empty_or_absent(canonical_output_root)

    _publish_json_bytes(
        canonical_output_root / PHASE8_DEFINITIVE_TRAINING_CONFIG_FILENAME,
        phase8_definitive_training_config_to_json(config),
    )
    _publish_json_bytes(
        canonical_output_root / PHASE8_DEFINITIVE_EXECUTION_RELEASE_FILENAME,
        phase8_definitive_execution_release_to_json(release),
    )

    artifact_hashes = {
        path.name: sha256_file(path)
        for path in sorted(canonical_output_root.iterdir(), key=lambda item: item.name)
        if path.is_file()
    }
    return Phase8DefinitiveTrainingPlanPublicationResult(
        output_root=canonical_output_root,
        config_hash=config.config_hash,
        release_hash=release.release_hash,
        artifact_hashes=artifact_hashes,
    )


# ---------------------------------------------------------------------------
# F. Future publishers -- gated behind an explicit release; dead in practice
# ---------------------------------------------------------------------------


def _reject_denylisted_artifact_role(reference: ArtifactReference, *, field_name: str) -> None:
    role = reference.artifact_role
    for substring in _ARTIFACT_ROLE_DENYLIST_SUBSTRINGS:
        if substring in role:
            raise Phase8DefinitiveTrainingValidationError(
                f"{field_name}.artifact_role must not reference {substring!r} evidence."
            )


def publish_phase8_definitive_preprocessing_evidence(
    *,
    config: Phase8DefinitiveTrainingConfig,
    release: Phase8DefinitiveExecutionRelease,
    preprocessing_decision: Phase8PreprocessingDecision,
    output_root: Path,
    repository_root: Path,
) -> str:
    """Publish preprocessing evidence bound to a released definitive config.

    Reuses :class:`Phase8PreprocessingDecision` from Substage 1 unchanged.
    Raises :class:`Phase8DefinitiveTrainingNotReleasedError` (or a hash
    mismatch :class:`Phase8DefinitiveTrainingValidationError`) before any
    file path is opened when ``release`` is absent, ``not_released``, or
    bound to a different config.
    """

    _gate_release(config, release)
    if preprocessing_decision.candidate_id != config.candidate_id:
        raise Phase8DefinitiveTrainingValidationError(
            "preprocessing_decision.candidate_id does not match config.candidate_id."
        )
    if (
        preprocessing_decision.preprocessing_decision_hash
        != config.preprocessing_decision_reference.artifact_hash
    ):
        raise Phase8DefinitiveTrainingValidationError(
            "preprocessing_decision does not match config.preprocessing_decision_reference."
        )
    canonical_repository_root = repository_root.resolve(strict=True)
    canonical_output_root = validate_phase8_definitive_output_root(
        output_root=output_root, repository_root=canonical_repository_root
    )
    path = canonical_output_root / PHASE8_DEFINITIVE_PREPROCESSING_EVIDENCE_FILENAME
    _publish_json_bytes(
        path,
        canonical_json_bytes(phase8_preprocessing_decision_to_dict(preprocessing_decision)) + b"\n",
    )
    return sha256_file(path)


def publish_phase8_definitive_checkpoint_metadata(
    *,
    config: Phase8DefinitiveTrainingConfig,
    release: Phase8DefinitiveExecutionRelease,
    checkpoint_metadata: Phase8CheckpointMetadata,
    output_root: Path,
    repository_root: Path,
) -> str:
    """Publish checkpoint metadata bound to a released definitive config.

    Reuses :class:`Phase8CheckpointMetadata` from Substage 1 unchanged. A
    checkpoint is refused as freeze-eligible unless
    ``completion_status == "completed"`` -- this is already enforced by
    :class:`Phase8CheckpointMetadata` itself; this function adds a redundant
    explicit guard and never silently swallows that error.
    """

    _gate_release(config, release)
    if checkpoint_metadata.candidate_id != config.candidate_id:
        raise Phase8DefinitiveTrainingValidationError(
            "checkpoint_metadata.candidate_id does not match config.candidate_id."
        )
    if checkpoint_metadata.training_config_hash != config.training_config_reference.artifact_hash:
        raise Phase8DefinitiveTrainingValidationError(
            "checkpoint_metadata.training_config_hash does not match "
            "config.training_config_reference."
        )
    if checkpoint_metadata.freeze_eligible and checkpoint_metadata.completion_status != "completed":
        raise Phase8DefinitiveTrainingValidationError(
            "freeze_eligible checkpoint metadata requires completion_status='completed'."
        )
    canonical_repository_root = repository_root.resolve(strict=True)
    canonical_output_root = validate_phase8_definitive_output_root(
        output_root=output_root, repository_root=canonical_repository_root
    )
    path = canonical_output_root / PHASE8_DEFINITIVE_CHECKPOINT_METADATA_FILENAME
    _publish_json_bytes(
        path,
        canonical_json_bytes(phase8_checkpoint_metadata_to_dict(checkpoint_metadata)) + b"\n",
    )
    return sha256_file(path)


def publish_phase8_definitive_validation_evidence(
    *,
    config: Phase8DefinitiveTrainingConfig,
    release: Phase8DefinitiveExecutionRelease,
    validation_evidence: Phase8ValidationEvidenceReference,
    output_root: Path,
    repository_root: Path,
) -> str:
    """Publish validation-only evidence bound to a released definitive config.

    Reuses :class:`Phase8ValidationEvidenceReference` from Substage 1
    unchanged. Accepts already-computed metric artifact references/hashes as
    opaque inputs; never computes a metric itself. Rejects any artifact
    reference whose ``artifact_role`` suggests internal-test or external
    origin.
    """

    _gate_release(config, release)
    if validation_evidence.candidate_id != config.candidate_id:
        raise Phase8DefinitiveTrainingValidationError(
            "validation_evidence.candidate_id does not match config.candidate_id."
        )
    _reject_denylisted_artifact_role(
        validation_evidence.empty_target_accounting_reference,
        field_name="empty_target_accounting_reference",
    )
    canonical_repository_root = repository_root.resolve(strict=True)
    canonical_output_root = validate_phase8_definitive_output_root(
        output_root=output_root, repository_root=canonical_repository_root
    )
    path = canonical_output_root / PHASE8_DEFINITIVE_VALIDATION_EVIDENCE_FILENAME
    _publish_json_bytes(
        path,
        canonical_json_bytes(phase8_validation_evidence_reference_to_dict(validation_evidence))
        + b"\n",
    )
    return sha256_file(path)


def _model_selection_decision_payload(
    *,
    fixed_candidate_inventory_hash: str,
    selection_rule_version: str,
    primary_metric: str,
    deterministic_tie_break_order: tuple[str, ...],
    selected_candidate_id: str,
    selected_checkpoint_hash: str,
    selected_validation_evidence_hash: str,
) -> dict[str, JsonValue]:
    return {
        "deterministic_tie_break_order": list(deterministic_tie_break_order),
        "external_data_not_used": True,
        "failed_candidate_accounting": [],
        "fixed_candidate_inventory_hash": fixed_candidate_inventory_hash,
        "internal_test_not_used": True,
        "primary_metric": primary_metric,
        "schema_name": "phase8_model_selection_decision",
        "schema_version": "v1",
        "selected_candidate_id": selected_candidate_id,
        "selected_checkpoint_hash": selected_checkpoint_hash,
        "selected_validation_evidence_hash": selected_validation_evidence_hash,
        "selection_rule_version": selection_rule_version,
        "selection_status": SELECTION_STATUS_SINGLE_CANDIDATE_PREREGISTERED,
    }


def publish_phase8_definitive_model_selection(
    *,
    config: Phase8DefinitiveTrainingConfig,
    release: Phase8DefinitiveExecutionRelease,
    candidate_inventory: Phase8FixedCandidateInventory,
    selected_checkpoint_metadata: Phase8CheckpointMetadata,
    selected_validation_evidence: Phase8ValidationEvidenceReference,
    output_root: Path,
    repository_root: Path,
) -> str:
    """Publish single-candidate model selection bound to a released config.

    Reuses :class:`Phase8ModelSelectionDecision` from Substage 1 unchanged.
    Hard-coded to accept only a single-candidate inventory and always
    produces ``selection_status="single_candidate_preregistered"`` with this
    module's fixed :data:`DEFINITIVE_TIE_BREAK_ORDER`.
    """

    _gate_release(config, release)
    if len(candidate_inventory.candidates) != 1:
        raise Phase8DefinitiveTrainingValidationError(
            "definitive model selection requires a single-candidate inventory."
        )
    candidate = candidate_inventory.candidates[0]
    if candidate.candidate_id != config.candidate_id:
        raise Phase8DefinitiveTrainingValidationError(
            "candidate_inventory candidate_id does not match config.candidate_id."
        )
    if selected_checkpoint_metadata.candidate_id != candidate.candidate_id:
        raise Phase8DefinitiveTrainingValidationError(
            "selected_checkpoint_metadata.candidate_id does not match the bound candidate."
        )
    if selected_validation_evidence.candidate_id != candidate.candidate_id:
        raise Phase8DefinitiveTrainingValidationError(
            "selected_validation_evidence.candidate_id does not match the bound candidate."
        )

    selection_rule_version = "phase8_definitive_substage4a_v1"
    payload = _model_selection_decision_payload(
        fixed_candidate_inventory_hash=candidate_inventory.inventory_hash,
        selection_rule_version=selection_rule_version,
        primary_metric=PRIMARY_METRIC_TUMOR_DICE,
        deterministic_tie_break_order=DEFINITIVE_TIE_BREAK_ORDER,
        selected_candidate_id=candidate.candidate_id,
        selected_checkpoint_hash=selected_checkpoint_metadata.checkpoint_sha256,
        selected_validation_evidence_hash=(selected_validation_evidence.validation_evidence_hash),
    )
    decision = Phase8ModelSelectionDecision(
        schema_name="phase8_model_selection_decision",
        schema_version="v1",
        fixed_candidate_inventory_hash=candidate_inventory.inventory_hash,
        selection_rule_version=selection_rule_version,
        primary_metric=PRIMARY_METRIC_TUMOR_DICE,
        deterministic_tie_break_order=DEFINITIVE_TIE_BREAK_ORDER,
        selected_candidate_id=candidate.candidate_id,
        selected_checkpoint_hash=selected_checkpoint_metadata.checkpoint_sha256,
        selected_validation_evidence_hash=(selected_validation_evidence.validation_evidence_hash),
        failed_candidate_accounting=(),
        selection_status=SELECTION_STATUS_SINGLE_CANDIDATE_PREREGISTERED,
        internal_test_not_used=True,
        external_data_not_used=True,
        model_selection_decision_hash=sha256_json(payload),
    )
    canonical_repository_root = repository_root.resolve(strict=True)
    canonical_output_root = validate_phase8_definitive_output_root(
        output_root=output_root, repository_root=canonical_repository_root
    )
    path = canonical_output_root / PHASE8_DEFINITIVE_MODEL_SELECTION_FILENAME
    _publish_json_bytes(
        path,
        canonical_json_bytes(phase8_model_selection_decision_to_dict(decision)) + b"\n",
    )
    return sha256_file(path)


def _threshold_decision_payload(*, candidate_id: str, checkpoint_hash: str) -> dict[str, JsonValue]:
    return {
        "candidate_id": candidate_id,
        "candidate_threshold_grid_reference": None,
        "checkpoint_hash": checkpoint_hash,
        "development_metric_reference": None,
        "development_only_provenance": True,
        "external_data_not_used": True,
        "fixed_constant_rationale": "phase8_definitive_fixed_threshold_policy",
        "freeze_status": "freeze_ready",
        "internal_test_not_used": True,
        "policy_type": REQUIRED_THRESHOLD_POLICY_TYPE,
        "schema_name": "phase8_threshold_decision",
        "schema_version": "v1",
        "selection_objective": None,
        "threshold_value": REQUIRED_THRESHOLD_VALUE,
    }


def publish_phase8_definitive_threshold_decision(
    *,
    config: Phase8DefinitiveTrainingConfig,
    release: Phase8DefinitiveExecutionRelease,
    checkpoint_hash: str,
    output_root: Path,
    repository_root: Path,
) -> str:
    """Publish the fixed 0.5 threshold decision bound to a released config.

    Reuses :class:`Phase8ThresholdDecision` from Substage 1 unchanged. There
    is no ``threshold_value`` parameter and no threshold-grid/search
    parameter: the policy is always ``fixed_constant`` at ``0.5``.
    """

    _gate_release(config, release)
    _require_sha256(checkpoint_hash, field_name="checkpoint_hash")
    payload = _threshold_decision_payload(
        candidate_id=config.candidate_id, checkpoint_hash=checkpoint_hash
    )
    decision = Phase8ThresholdDecision(
        schema_name="phase8_threshold_decision",
        schema_version="v1",
        policy_type=REQUIRED_THRESHOLD_POLICY_TYPE,
        threshold_value=REQUIRED_THRESHOLD_VALUE,
        candidate_id=config.candidate_id,
        checkpoint_hash=checkpoint_hash,
        selection_objective=None,
        candidate_threshold_grid_reference=None,
        development_metric_reference=None,
        fixed_constant_rationale="phase8_definitive_fixed_threshold_policy",
        development_only_provenance=True,
        internal_test_not_used=True,
        external_data_not_used=True,
        freeze_status="freeze_ready",
        threshold_decision_hash=sha256_json(payload),
    )
    canonical_repository_root = repository_root.resolve(strict=True)
    canonical_output_root = validate_phase8_definitive_output_root(
        output_root=output_root, repository_root=canonical_repository_root
    )
    path = canonical_output_root / PHASE8_DEFINITIVE_THRESHOLD_DECISION_FILENAME
    _publish_json_bytes(
        path,
        canonical_json_bytes(phase8_threshold_decision_to_dict(decision)) + b"\n",
    )
    return sha256_file(path)


def _support_policy_payload(*, candidate_id: str, checkpoint_hash: str) -> dict[str, JsonValue]:
    return {
        "external_prototype_construction_forbidden": True,
        "external_support_labels_forbidden": True,
        "external_support_state": "forbidden",
        "freeze_status": "freeze_ready",
        "internal_support_manifest_reference": None,
        "policy_type": REQUIRED_SUPPORT_POLICY_TYPE,
        "schema_name": "phase8_support_policy",
        "schema_version": "v1",
        "selected_candidate_id": candidate_id,
        "selected_checkpoint_hash": checkpoint_hash,
        "support_k": None,
        "support_seed_reference": None,
    }


def publish_phase8_definitive_support_policy(
    *,
    config: Phase8DefinitiveTrainingConfig,
    release: Phase8DefinitiveExecutionRelease,
    checkpoint_hash: str,
    output_root: Path,
    repository_root: Path,
) -> str:
    """Publish the fixed no-support policy bound to a released config.

    Reuses :class:`Phase8SupportPolicy` from Substage 1 unchanged. There is
    no support-manifest-reference parameter: the policy is always
    ``no_support``.
    """

    _gate_release(config, release)
    _require_sha256(checkpoint_hash, field_name="checkpoint_hash")
    payload = _support_policy_payload(
        candidate_id=config.candidate_id, checkpoint_hash=checkpoint_hash
    )
    policy = Phase8SupportPolicy(
        schema_name="phase8_support_policy",
        schema_version="v1",
        policy_type=REQUIRED_SUPPORT_POLICY_TYPE,
        selected_candidate_id=config.candidate_id,
        selected_checkpoint_hash=checkpoint_hash,
        internal_support_manifest_reference=None,
        support_k=None,
        support_seed_reference=None,
        external_support_labels_forbidden=True,
        external_prototype_construction_forbidden=True,
        external_support_state="forbidden",
        freeze_status="freeze_ready",
        support_policy_hash=sha256_json(payload),
    )
    canonical_repository_root = repository_root.resolve(strict=True)
    canonical_output_root = validate_phase8_definitive_output_root(
        output_root=output_root, repository_root=canonical_repository_root
    )
    path = canonical_output_root / PHASE8_DEFINITIVE_SUPPORT_POLICY_FILENAME
    _publish_json_bytes(
        path,
        canonical_json_bytes(phase8_support_policy_to_dict(policy)) + b"\n",
    )
    return sha256_file(path)


# ---------------------------------------------------------------------------
# Shared validation and serialization helpers
# ---------------------------------------------------------------------------


def _require_schema(value: str, expected: str) -> None:
    if value != expected:
        raise Phase8DefinitiveTrainingValidationError(
            f"schema_name must equal {expected!r}, got {value!r}."
        )


def _require_schema_version_exact(value: str) -> None:
    if value != PHASE8_DEFINITIVE_TRAINING_SCHEMA_VERSION:
        raise Phase8DefinitiveTrainingValidationError(
            f"schema_version must equal {PHASE8_DEFINITIVE_TRAINING_SCHEMA_VERSION!r}."
        )


def _require_allowed(value: str, allowed: frozenset[str], *, field_name: str) -> None:
    if value not in allowed:
        raise Phase8DefinitiveTrainingValidationError(
            f"{field_name} must be one of {sorted(allowed)!r}, got {value!r}."
        )


def _require_identifier(value: str, *, field_name: str) -> None:
    if not _IDENTIFIER_RE.fullmatch(value):
        raise Phase8DefinitiveTrainingValidationError(
            f"{field_name} must be a lowercase safe identifier."
        )
    if value in _UNRESOLVED_TOKENS or _contains_unresolved_token(value):
        raise Phase8DefinitiveTrainingValidationError(f"{field_name} must not be unresolved.")
    if value.startswith("/") or value.startswith("~") or "\\" in value or "://" in value:
        raise Phase8DefinitiveTrainingValidationError(
            f"{field_name} must not contain local path or URI identity."
        )


def _contains_unresolved_token(value: str) -> bool:
    return any(part in _UNRESOLVED_TOKENS for part in re.split(r"[_.@:/-]+", value))


def _require_sha256(value: str, *, field_name: str) -> None:
    if not _SHA256_RE.fullmatch(value):
        raise Phase8DefinitiveTrainingHashError(f"{field_name} must be a lowercase SHA-256 digest.")
    if value == _ZERO_SHA256:
        raise Phase8DefinitiveTrainingHashError(
            f"{field_name} must not be the zero SHA-256 digest."
        )


def _require_nonnegative_int(value: int, *, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise Phase8DefinitiveTrainingValidationError(
            f"{field_name} must be a nonnegative integer."
        )


def _require_positive_int(value: int, *, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise Phase8DefinitiveTrainingValidationError(f"{field_name} must be a positive integer.")


def _require_true(value: bool, *, field_name: str) -> None:
    if value is not True:
        raise Phase8DefinitiveTrainingValidationError(f"{field_name} must be true.")


def _require_false(value: bool, *, field_name: str) -> None:
    if value is not False:
        raise Phase8DefinitiveTrainingValidationError(f"{field_name} must be false.")


def _require_self_hash(value: str, payload: dict[str, JsonValue], *, field_name: str) -> None:
    _require_sha256(value, field_name=field_name)
    expected = sha256_json(payload)
    if value != expected:
        raise Phase8DefinitiveTrainingHashError(f"{field_name} does not match canonical identity.")


def _require_exact_fields(
    mapping: MappingLike, required_fields: frozenset[str], *, object_name: str
) -> None:
    keys = set(mapping)
    missing = sorted(required_fields - keys)
    extra = sorted(keys - required_fields)
    if missing or extra:
        raise Phase8DefinitiveTrainingSerializationError(
            f"{object_name} fields mismatch; missing={missing!r}, extra={extra!r}."
        )


def _expect_mapping(value: object, *, field_name: str) -> MappingLike:
    if not isinstance(value, Mapping):
        raise Phase8DefinitiveTrainingSerializationError(f"{field_name} must be a mapping.")
    return cast(MappingLike, value)


def _expect_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise Phase8DefinitiveTrainingSerializationError(f"{field_name} must be a string.")
    return value


def _expect_optional_string(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _expect_string(value, field_name=field_name)


def _expect_bool(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise Phase8DefinitiveTrainingSerializationError(f"{field_name} must be a boolean.")
    return value


def _expect_int(value: object, *, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise Phase8DefinitiveTrainingSerializationError(f"{field_name} must be an integer.")
    return value


def _expect_float(value: object, *, field_name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise Phase8DefinitiveTrainingSerializationError(f"{field_name} must be a number.")
    return float(value)


def _expect_list(value: object, *, field_name: str) -> list[object]:
    if not isinstance(value, list):
        raise Phase8DefinitiveTrainingSerializationError(f"{field_name} must be a list.")
    return value


def _expect_int_tuple(value: object, *, field_name: str) -> tuple[int, ...]:
    sequence = _expect_list(value, field_name=field_name)
    return tuple(
        _expect_int(item, field_name=f"{field_name}[{index}]")
        for index, item in enumerate(sequence)
    )


__all__ = [
    "AWAITING_EXPLICIT_USER_APPROVAL",
    "DEFINITIVE_TIE_BREAK_ORDER",
    "PHASE8_DEFINITIVE_CHECKPOINT_METADATA_FILENAME",
    "PHASE8_DEFINITIVE_EXECUTION_RELEASE_FILENAME",
    "PHASE8_DEFINITIVE_EXECUTION_RELEASE_SCHEMA_NAME",
    "PHASE8_DEFINITIVE_MODEL_SELECTION_FILENAME",
    "PHASE8_DEFINITIVE_PREPROCESSING_EVIDENCE_FILENAME",
    "PHASE8_DEFINITIVE_SUPPORT_POLICY_FILENAME",
    "PHASE8_DEFINITIVE_THRESHOLD_DECISION_FILENAME",
    "PHASE8_DEFINITIVE_TRAINING_CONFIG_FILENAME",
    "PHASE8_DEFINITIVE_TRAINING_CONFIG_SCHEMA_NAME",
    "PHASE8_DEFINITIVE_TRAINING_SCHEMA_VERSION",
    "PHASE8_DEFINITIVE_VALIDATION_EVIDENCE_FILENAME",
    "PRIMARY_METRIC_TUMOR_DICE",
    "RELEASE_AUTHORIZED_BY_EXPLICIT_USER_APPROVAL",
    "RELEASE_STATE_NOT_RELEASED",
    "RELEASE_STATE_RELEASED",
    "REQUIRED_CANDIDATE_ID",
    "REQUIRED_DEVICE_TYPE",
    "REQUIRED_SUPPORT_POLICY_TYPE",
    "REQUIRED_THRESHOLD_POLICY_TYPE",
    "REQUIRED_THRESHOLD_VALUE",
    "SELECTION_STATUS_SINGLE_CANDIDATE_PREREGISTERED",
    "Phase8DefinitiveExecutionRelease",
    "Phase8DefinitiveTrainingConfig",
    "Phase8DefinitiveTrainingError",
    "Phase8DefinitiveTrainingExecutor",
    "Phase8DefinitiveTrainingHashError",
    "Phase8DefinitiveTrainingNotReleasedError",
    "Phase8DefinitiveTrainingPlanPublicationResult",
    "Phase8DefinitiveTrainingPublicationError",
    "Phase8DefinitiveTrainingSerializationError",
    "Phase8DefinitiveTrainingValidationError",
    "build_phase8_definitive_training_config",
    "build_unreleased_definitive_execution_release",
    "execute_phase8_definitive_training",
    "hash_phase8_definitive_execution_release",
    "hash_phase8_definitive_training_config",
    "phase8_definitive_execution_release_from_mapping",
    "phase8_definitive_execution_release_identity_payload",
    "phase8_definitive_execution_release_to_dict",
    "phase8_definitive_execution_release_to_json",
    "phase8_definitive_training_config_from_mapping",
    "phase8_definitive_training_config_identity_payload",
    "phase8_definitive_training_config_to_dict",
    "phase8_definitive_training_config_to_json",
    "publish_phase8_definitive_checkpoint_metadata",
    "publish_phase8_definitive_model_selection",
    "publish_phase8_definitive_preprocessing_evidence",
    "publish_phase8_definitive_support_policy",
    "publish_phase8_definitive_threshold_decision",
    "publish_phase8_definitive_training_plan",
    "publish_phase8_definitive_validation_evidence",
    "validate_phase8_definitive_output_root",
]
