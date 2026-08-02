"""Deterministic ProtoEM-CT transductive state contracts and initialization."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Final, cast

import numpy as np

from protoem_ct.artifacts.hashing import JsonValue, canonical_json_bytes, sha256_json
from protoem_ct.protoem.artifacts import SUPPORTED_PROTOEM_PROTOTYPE_MODES, ProtoEMConfig
from protoem_ct.protoem.initialize import ProtoEMInitializationBundle
from protoem_ct.protoem.objective import (
    PROTOEM_E_STEP_RESULT_SCHEMA_NAME,
    PROTOEM_M_STEP_RESULT_SCHEMA_NAME,
    PROTOEM_OBJECTIVE_TERMS_SCHEMA_NAME,
    ProtoEMEStepResult,
    ProtoEMMStepResult,
    ProtoEMObjectiveTerms,
)

PROTOEM_PROTOTYPE_STATE_SCHEMA_NAME: Final[str] = "protoem_prototype_state"
PROTOEM_PROTOTYPE_STATE_SCHEMA_VERSION: Final[str] = "v1"
PROTOEM_POSTERIOR_STATE_SCHEMA_NAME: Final[str] = "protoem_posterior_state"
PROTOEM_POSTERIOR_STATE_SCHEMA_VERSION: Final[str] = "v1"
PROTOEM_ASSIGNMENT_STATE_SCHEMA_NAME: Final[str] = "protoem_assignment_state"
PROTOEM_ASSIGNMENT_STATE_SCHEMA_VERSION: Final[str] = "v1"
PROTOEM_TRANSDUCTIVE_STATE_SCHEMA_NAME: Final[str] = "protoem_transductive_state"
PROTOEM_TRANSDUCTIVE_STATE_SCHEMA_VERSION: Final[str] = "v1"
PROTOEM_STATE_TRANSITION_SCHEMA_NAME: Final[str] = "protoem_state_transition"
PROTOEM_STATE_TRANSITION_SCHEMA_VERSION: Final[str] = "v1"

_FLOAT_DTYPE: Final[np.dtype[np.float64]] = np.dtype(np.float64)
_BOOL_DTYPE: Final[np.dtype[np.bool_]] = np.dtype(np.bool_)
_PREDICTION_DTYPE: Final[np.dtype[np.uint8]] = np.dtype(np.uint8)
_FLOATING_KINDS: Final[frozenset[str]] = frozenset({"f"})
_NUMERIC_KINDS: Final[frozenset[str]] = frozenset({"f", "i", "u"})
_SHA256_HEX: Final[frozenset[str]] = frozenset("0123456789abcdef")
_OBJECTIVE_TOLERANCE: Final[float] = 1e-12


class ProtoEMStateError(ValueError):
    """Base error for deterministic ProtoEM transductive state handling."""


class InvalidStateError(ProtoEMStateError):
    """Raised when one state object is malformed or internally inconsistent."""


class IncompatibleStateTransitionError(ProtoEMStateError):
    """Raised when one E-step/M-step result is incompatible with one source state."""


class StateIdentityMismatchError(ProtoEMStateError):
    """Raised when one state self-hash or transition self-hash does not validate."""


class PosteriorConsistencyFailureError(ProtoEMStateError):
    """Raised when one posterior state violates ProtoEM posterior consistency rules."""


class AssignmentConsistencyFailureError(ProtoEMStateError):
    """Raised when one assignment state violates ProtoEM assignment consistency rules."""


class PrototypeStateFailureError(ProtoEMStateError):
    """Raised when one prototype state violates ProtoEM prototype-state rules."""


@dataclass(frozen=True, slots=True)
class ProtoEMPrototypeState:
    """Deterministic foreground/background prototype state."""

    schema_name: str
    schema_version: str
    foreground_prototypes: np.ndarray
    background_prototypes: np.ndarray
    prototype_mode: str
    channel_count: int
    foreground_content_hash: str
    background_content_hash: str
    prototype_state_identity_hash: str

    def __post_init__(self) -> None:
        _require_schema_name(
            self.schema_name,
            expected=PROTOEM_PROTOTYPE_STATE_SCHEMA_NAME,
            field_name="schema_name",
        )
        _require_schema_version(
            self.schema_version,
            expected=PROTOEM_PROTOTYPE_STATE_SCHEMA_VERSION,
            field_name="schema_version",
        )
        if self.prototype_mode not in SUPPORTED_PROTOEM_PROTOTYPE_MODES:
            raise PrototypeStateFailureError(
                f"prototype_mode must be one of {sorted(SUPPORTED_PROTOEM_PROTOTYPE_MODES)!r}."
            )
        if self.channel_count <= 0:
            raise PrototypeStateFailureError("channel_count must be positive.")
        foreground_bank = _require_prototype_array(
            self.foreground_prototypes,
            field_name="foreground_prototypes",
            channel_count=self.channel_count,
            prototype_mode=self.prototype_mode,
        )
        background_bank = _require_prototype_array(
            self.background_prototypes,
            field_name="background_prototypes",
            channel_count=self.channel_count,
            prototype_mode=self.prototype_mode,
        )
        _require_sha256(self.foreground_content_hash, field_name="foreground_content_hash")
        _require_sha256(self.background_content_hash, field_name="background_content_hash")
        _require_sha256(
            self.prototype_state_identity_hash,
            field_name="prototype_state_identity_hash",
        )
        if self.foreground_content_hash != _prototype_content_sha256(foreground_bank):
            raise PrototypeStateFailureError(
                "foreground_content_hash must match foreground_prototypes content."
            )
        if self.background_content_hash != _prototype_content_sha256(background_bank):
            raise PrototypeStateFailureError(
                "background_content_hash must match background_prototypes content."
            )
        if self.prototype_state_identity_hash != hash_protoem_prototype_state(self):
            raise StateIdentityMismatchError(
                "prototype_state_identity_hash does not match deterministic content."
            )


@dataclass(frozen=True, slots=True)
class ProtoEMPosteriorState:
    """Deterministic posterior and confidence state."""

    schema_name: str
    schema_version: str
    foreground_posterior_map: np.ndarray
    background_posterior_map: np.ndarray
    confidence_map: np.ndarray
    confident_voxel_mask: np.ndarray
    zero_norm_query_voxel_mask: np.ndarray
    posterior_state_identity_hash: str

    def __post_init__(self) -> None:
        _require_schema_name(
            self.schema_name,
            expected=PROTOEM_POSTERIOR_STATE_SCHEMA_NAME,
            field_name="schema_name",
        )
        _require_schema_version(
            self.schema_version,
            expected=PROTOEM_POSTERIOR_STATE_SCHEMA_VERSION,
            field_name="schema_version",
        )
        fg = _require_probability_volume(
            self.foreground_posterior_map,
            field_name="foreground_posterior_map",
            error_type=PosteriorConsistencyFailureError,
        )
        bg = _require_probability_volume(
            self.background_posterior_map,
            field_name="background_posterior_map",
            error_type=PosteriorConsistencyFailureError,
        )
        confidence = _require_float_volume(
            self.confidence_map,
            field_name="confidence_map",
            error_type=PosteriorConsistencyFailureError,
        )
        confident_mask = _require_bool_volume(
            self.confident_voxel_mask,
            field_name="confident_voxel_mask",
            error_type=PosteriorConsistencyFailureError,
        )
        zero_norm_mask = _require_bool_volume(
            self.zero_norm_query_voxel_mask,
            field_name="zero_norm_query_voxel_mask",
            error_type=PosteriorConsistencyFailureError,
        )
        _require_sha256(
            self.posterior_state_identity_hash,
            field_name="posterior_state_identity_hash",
        )
        if (
            fg.shape != bg.shape
            or fg.shape != confidence.shape
            or fg.shape != confident_mask.shape
            or fg.shape != zero_norm_mask.shape
        ):
            raise PosteriorConsistencyFailureError(
                "all posterior-state arrays must share the same [1,1,D,H,W] shape."
            )
        if not np.allclose(fg + bg, 1.0, rtol=0.0, atol=_OBJECTIVE_TOLERANCE):
            raise PosteriorConsistencyFailureError(
                "foreground/background posteriors must sum to one voxelwise."
            )
        expected_confidence = np.maximum(fg, bg)
        if not np.allclose(
            confidence.astype(_FLOAT_DTYPE, copy=False),
            expected_confidence,
            rtol=0.0,
            atol=_OBJECTIVE_TOLERANCE,
        ):
            raise PosteriorConsistencyFailureError(
                "confidence_map must equal max(foreground_posterior_map, background_posterior_map)."
            )
        if self.posterior_state_identity_hash != hash_protoem_posterior_state(self):
            raise StateIdentityMismatchError(
                "posterior_state_identity_hash does not match deterministic content."
            )


@dataclass(frozen=True, slots=True)
class ProtoEMAssignmentState:
    """Deterministic hard-assignment and confident-count state."""

    schema_name: str
    schema_version: str
    hard_assignment_map: np.ndarray
    confident_voxel_count: int
    foreground_assignment_count: int
    background_assignment_count: int
    foreground_fraction: float
    assignment_state_identity_hash: str

    def __post_init__(self) -> None:
        _require_schema_name(
            self.schema_name,
            expected=PROTOEM_ASSIGNMENT_STATE_SCHEMA_NAME,
            field_name="schema_name",
        )
        _require_schema_version(
            self.schema_version,
            expected=PROTOEM_ASSIGNMENT_STATE_SCHEMA_VERSION,
            field_name="schema_version",
        )
        hard_assignment = _require_assignment_volume(
            self.hard_assignment_map,
            field_name="hard_assignment_map",
            error_type=AssignmentConsistencyFailureError,
        )
        _require_sha256(
            self.assignment_state_identity_hash,
            field_name="assignment_state_identity_hash",
        )
        for field_name in (
            "confident_voxel_count",
            "foreground_assignment_count",
            "background_assignment_count",
        ):
            value = cast(int, getattr(self, field_name))
            if value < 0:
                raise AssignmentConsistencyFailureError(f"{field_name} must be nonnegative.")
        _require_fraction_closed(
            self.foreground_fraction,
            field_name="foreground_fraction",
            error_type=AssignmentConsistencyFailureError,
        )
        _ = hard_assignment
        if self.assignment_state_identity_hash != hash_protoem_assignment_state(self):
            raise StateIdentityMismatchError(
                "assignment_state_identity_hash does not match deterministic content."
            )


@dataclass(frozen=True, slots=True)
class ProtoEMTransductiveState:
    """Deterministic ProtoEM transductive state at one zero-based iteration."""

    schema_name: str
    schema_version: str
    iteration_index: int
    initialization_identity_hash: str
    query_feature_content_hash: str
    prototype_state: ProtoEMPrototypeState
    posterior_state: ProtoEMPosteriorState
    assignment_state: ProtoEMAssignmentState
    previous_objective_total: float | None
    current_objective_terms: ProtoEMObjectiveTerms | None
    convergence_delta: float | None
    state_identity_hash: str

    def __post_init__(self) -> None:
        _require_schema_name(
            self.schema_name,
            expected=PROTOEM_TRANSDUCTIVE_STATE_SCHEMA_NAME,
            field_name="schema_name",
        )
        _require_schema_version(
            self.schema_version,
            expected=PROTOEM_TRANSDUCTIVE_STATE_SCHEMA_VERSION,
            field_name="schema_version",
        )
        if self.iteration_index < 0:
            raise InvalidStateError("iteration_index must be zero-based and nonnegative.")
        _require_sha256(
            self.initialization_identity_hash,
            field_name="initialization_identity_hash",
        )
        _require_sha256(
            self.query_feature_content_hash,
            field_name="query_feature_content_hash",
        )
        _require_sha256(self.state_identity_hash, field_name="state_identity_hash")
        shape = self.posterior_state.foreground_posterior_map.shape
        if self.assignment_state.hard_assignment_map.shape != shape:
            raise InvalidStateError(
                "assignment_state.hard_assignment_map must match posterior-state shape."
            )
        if self.prototype_state.channel_count <= 0:
            raise InvalidStateError("prototype_state.channel_count must be positive.")
        if self.previous_objective_total is not None and not math.isfinite(
            self.previous_objective_total
        ):
            raise InvalidStateError("previous_objective_total must be finite when present.")
        if self.convergence_delta is not None and (
            not math.isfinite(self.convergence_delta) or self.convergence_delta < 0.0
        ):
            raise InvalidStateError(
                "convergence_delta must be finite and nonnegative when present."
            )
        if self.current_objective_terms is None and self.convergence_delta is not None:
            raise InvalidStateError(
                "convergence_delta requires current_objective_terms to be present."
            )
        if self.current_objective_terms is None and self.previous_objective_total is not None:
            raise InvalidStateError(
                "previous_objective_total requires current_objective_terms tracking."
            )
        _validate_assignment_vs_posterior(
            posterior_state=self.posterior_state,
            assignment_state=self.assignment_state,
        )
        if self.current_objective_terms is not None and self.previous_objective_total is not None:
            expected_delta = abs(self.current_objective_terms.total - self.previous_objective_total)
            if self.convergence_delta is None or not math.isclose(
                self.convergence_delta,
                expected_delta,
                rel_tol=0.0,
                abs_tol=_OBJECTIVE_TOLERANCE,
            ):
                raise InvalidStateError(
                    "convergence_delta must equal "
                    "|current_objective_total - previous_objective_total|."
                )
        if (
            self.current_objective_terms is not None
            and self.previous_objective_total is None
            and self.convergence_delta is not None
        ):
            raise InvalidStateError(
                "convergence_delta must be null when no previous objective total exists."
            )
        if self.state_identity_hash != hash_protoem_transductive_state(self):
            raise StateIdentityMismatchError(
                "state_identity_hash does not match deterministic state content."
            )


@dataclass(frozen=True, slots=True)
class ProtoEMStateTransition:
    """Deterministic one-step transition between two ProtoEM transductive states."""

    schema_name: str
    schema_version: str
    source_state_identity: str
    target_state_identity: str
    iteration_index_before: int
    iteration_index_after: int
    e_step_result_identity: str
    m_step_result_identity: str
    objective_terms_identity: str
    transition_identity_hash: str

    def __post_init__(self) -> None:
        _require_schema_name(
            self.schema_name,
            expected=PROTOEM_STATE_TRANSITION_SCHEMA_NAME,
            field_name="schema_name",
        )
        _require_schema_version(
            self.schema_version,
            expected=PROTOEM_STATE_TRANSITION_SCHEMA_VERSION,
            field_name="schema_version",
        )
        for field_name in (
            "source_state_identity",
            "target_state_identity",
            "e_step_result_identity",
            "m_step_result_identity",
            "objective_terms_identity",
            "transition_identity_hash",
        ):
            _require_sha256(cast(str, getattr(self, field_name)), field_name=field_name)
        if self.iteration_index_before < 0 or self.iteration_index_after < 0:
            raise IncompatibleStateTransitionError(
                "transition iteration indices must be nonnegative."
            )
        if self.iteration_index_after != self.iteration_index_before + 1:
            raise IncompatibleStateTransitionError(
                "iteration_index_after must equal iteration_index_before + 1."
            )
        if self.transition_identity_hash != hash_protoem_state_transition(self):
            raise StateIdentityMismatchError(
                "transition_identity_hash does not match deterministic transition content."
            )


def build_initial_protoem_transductive_state(
    *,
    initialization_bundle: ProtoEMInitializationBundle,
    config: ProtoEMConfig,
) -> ProtoEMTransductiveState:
    """Construct the deterministic initial transductive state from Phase 5 initialization."""

    _validate_config_initialization_match(
        config=config, initialization_bundle=initialization_bundle
    )
    if config.prototype_mode != "single_prototype":
        raise PrototypeStateFailureError(
            "Phase 6 substage 4 initialization supports single_prototype only."
        )
    query_feature_data = np.asarray(initialization_bundle.query_feature_encoding.feature_data)
    if (
        _array_content_sha256(query_feature_data)
        != initialization_bundle.query_feature_content_sha256
    ):
        raise InvalidStateError(
            "initialization_bundle query_feature_content_sha256 must match query feature content."
        )
    query_features = np.ascontiguousarray(query_feature_data.astype(_FLOAT_DTYPE, copy=False))
    zero_norm_mask = _zero_norm_query_mask(query_features)
    foreground_posterior, background_posterior = _posterior_from_phase5_scores(
        foreground_scores=initialization_bundle.initial_foreground_score_map,
        background_scores=initialization_bundle.initial_background_score_map,
        zero_norm_mask=zero_norm_mask,
        temperature=config.temperature,
        foreground_prior=config.foreground_prior,
    )
    confidence_map = np.maximum(foreground_posterior, background_posterior)
    confident_mask = confidence_map >= config.confidence_threshold

    phase5_prediction = np.ascontiguousarray(
        initialization_bundle.initial_prediction_mask.astype(_PREDICTION_DTYPE, copy=False)
    )
    expected_assignment = (foreground_posterior > background_posterior).astype(
        _PREDICTION_DTYPE,
        copy=False,
    )
    if not np.array_equal(phase5_prediction, expected_assignment):
        raise AssignmentConsistencyFailureError(
            "Phase 5 hard prediction must agree with posterior argmax "
            "under background tie resolution."
        )

    prototype_state = build_protoem_prototype_state(
        foreground_prototypes=np.asarray(
            initialization_bundle.foreground_prototype.prototype_vector,
            dtype=np.float64,
        ),
        background_prototypes=np.asarray(
            initialization_bundle.background_prototype.prototype_vector,
            dtype=np.float64,
        ),
        prototype_mode=config.prototype_mode,
        channel_count=int(initialization_bundle.foreground_prototype.feature_channels),
    )
    posterior_state = build_protoem_posterior_state(
        foreground_posterior_map=foreground_posterior,
        background_posterior_map=background_posterior,
        confidence_map=confidence_map,
        confident_voxel_mask=confident_mask,
        zero_norm_query_voxel_mask=zero_norm_mask,
    )
    assignment_state = build_protoem_assignment_state(
        hard_assignment_map=phase5_prediction,
        confident_voxel_mask=confident_mask,
    )
    return _build_transductive_state(
        iteration_index=0,
        initialization_identity_hash=initialization_bundle.initialization_identity_hash,
        query_feature_content_hash=_array_content_sha256(query_features),
        prototype_state=prototype_state,
        posterior_state=posterior_state,
        assignment_state=assignment_state,
        previous_objective_total=None,
        current_objective_terms=None,
        convergence_delta=None,
    )


def build_next_protoem_transductive_state(
    *,
    source_state: ProtoEMTransductiveState,
    e_step_result: ProtoEMEStepResult,
    m_step_result: ProtoEMMStepResult,
    objective_terms: ProtoEMObjectiveTerms,
) -> tuple[ProtoEMTransductiveState, ProtoEMStateTransition]:
    """Construct one deterministic next state and its transition record."""

    if e_step_result.schema_name != PROTOEM_E_STEP_RESULT_SCHEMA_NAME:
        raise IncompatibleStateTransitionError("unexpected E-step schema_name.")
    if m_step_result.schema_name != PROTOEM_M_STEP_RESULT_SCHEMA_NAME:
        raise IncompatibleStateTransitionError("unexpected M-step schema_name.")
    if objective_terms.schema_name != PROTOEM_OBJECTIVE_TERMS_SCHEMA_NAME:
        raise IncompatibleStateTransitionError("unexpected objective_terms schema_name.")
    if e_step_result.query_feature_content_sha256 != source_state.query_feature_content_hash:
        raise IncompatibleStateTransitionError(
            "E-step query feature hash must match the source state."
        )
    if (
        e_step_result.foreground_bank_content_sha256
        != source_state.prototype_state.foreground_content_hash
        or e_step_result.background_bank_content_sha256
        != source_state.prototype_state.background_content_hash
    ):
        raise IncompatibleStateTransitionError(
            "E-step prototype hashes must match the source prototype state."
        )
    if m_step_result.query_feature_content_sha256 != source_state.query_feature_content_hash:
        raise IncompatibleStateTransitionError(
            "M-step query feature hash must match the source state."
        )
    if (
        m_step_result.previous_foreground_bank_content_sha256
        != source_state.prototype_state.foreground_content_hash
        or m_step_result.previous_background_bank_content_sha256
        != source_state.prototype_state.background_content_hash
    ):
        raise IncompatibleStateTransitionError(
            "M-step previous prototype hashes must match the source prototype state."
        )
    if m_step_result.objective_terms != objective_terms:
        raise IncompatibleStateTransitionError(
            "objective_terms must match m_step_result.objective_terms exactly."
        )
    if source_state.prototype_state.prototype_mode != "single_prototype":
        raise IncompatibleStateTransitionError(
            "Phase 6 substage 4 state transitions support single_prototype only."
        )

    target_prototype_state = build_protoem_prototype_state(
        foreground_prototypes=m_step_result.updated_foreground_prototype,
        background_prototypes=m_step_result.updated_background_prototype,
        prototype_mode=source_state.prototype_state.prototype_mode,
        channel_count=source_state.prototype_state.channel_count,
    )
    target_posterior_state = build_protoem_posterior_state(
        foreground_posterior_map=e_step_result.foreground_posterior_map,
        background_posterior_map=e_step_result.background_posterior_map,
        confidence_map=e_step_result.confidence_mask_result.confidence_map,
        confident_voxel_mask=e_step_result.confidence_mask_result.confident_mask,
        zero_norm_query_voxel_mask=e_step_result.zero_norm_query_voxel_mask,
    )
    target_assignment_state = build_protoem_assignment_state(
        hard_assignment_map=e_step_result.hard_assignment_map,
        confident_voxel_mask=e_step_result.confidence_mask_result.confident_mask,
    )

    previous_total = (
        source_state.current_objective_terms.total
        if source_state.current_objective_terms is not None
        else None
    )
    convergence_delta = (
        abs(objective_terms.total - previous_total) if previous_total is not None else None
    )
    target_state = _build_transductive_state(
        iteration_index=source_state.iteration_index + 1,
        initialization_identity_hash=source_state.initialization_identity_hash,
        query_feature_content_hash=source_state.query_feature_content_hash,
        prototype_state=target_prototype_state,
        posterior_state=target_posterior_state,
        assignment_state=target_assignment_state,
        previous_objective_total=previous_total,
        current_objective_terms=objective_terms,
        convergence_delta=convergence_delta,
    )
    objective_terms_identity = hash_protoem_objective_terms_state(objective_terms)
    transition_identity_hash = sha256_json(
        protoem_state_transition_identity_payload_from_parts(
            source_state_identity=source_state.state_identity_hash,
            target_state_identity=target_state.state_identity_hash,
            iteration_index_before=source_state.iteration_index,
            iteration_index_after=target_state.iteration_index,
            e_step_result_identity=e_step_result.result_identity_hash,
            m_step_result_identity=m_step_result.result_identity_hash,
            objective_terms_identity=objective_terms_identity,
        )
    )
    return target_state, ProtoEMStateTransition(
        schema_name=PROTOEM_STATE_TRANSITION_SCHEMA_NAME,
        schema_version=PROTOEM_STATE_TRANSITION_SCHEMA_VERSION,
        source_state_identity=source_state.state_identity_hash,
        target_state_identity=target_state.state_identity_hash,
        iteration_index_before=source_state.iteration_index,
        iteration_index_after=target_state.iteration_index,
        e_step_result_identity=e_step_result.result_identity_hash,
        m_step_result_identity=m_step_result.result_identity_hash,
        objective_terms_identity=objective_terms_identity,
        transition_identity_hash=transition_identity_hash,
    )


def build_protoem_prototype_state(
    *,
    foreground_prototypes: np.ndarray,
    background_prototypes: np.ndarray,
    prototype_mode: str,
    channel_count: int,
) -> ProtoEMPrototypeState:
    """Build one deterministic prototype state."""

    foreground_array = np.ascontiguousarray(foreground_prototypes.astype(_FLOAT_DTYPE, copy=False))
    background_array = np.ascontiguousarray(background_prototypes.astype(_FLOAT_DTYPE, copy=False))
    foreground_content_hash = _prototype_content_sha256(foreground_array)
    background_content_hash = _prototype_content_sha256(background_array)
    prototype_state_identity_hash = sha256_json(
        protoem_prototype_state_identity_payload_from_parts(
            prototype_mode=prototype_mode,
            channel_count=channel_count,
            foreground_content_hash=foreground_content_hash,
            background_content_hash=background_content_hash,
            foreground_prototype_count=int(_prototype_bank_view(foreground_array).shape[0]),
            background_prototype_count=int(_prototype_bank_view(background_array).shape[0]),
        )
    )
    return ProtoEMPrototypeState(
        schema_name=PROTOEM_PROTOTYPE_STATE_SCHEMA_NAME,
        schema_version=PROTOEM_PROTOTYPE_STATE_SCHEMA_VERSION,
        foreground_prototypes=foreground_array,
        background_prototypes=background_array,
        prototype_mode=prototype_mode,
        channel_count=channel_count,
        foreground_content_hash=foreground_content_hash,
        background_content_hash=background_content_hash,
        prototype_state_identity_hash=prototype_state_identity_hash,
    )


def build_protoem_posterior_state(
    *,
    foreground_posterior_map: np.ndarray,
    background_posterior_map: np.ndarray,
    confidence_map: np.ndarray,
    confident_voxel_mask: np.ndarray,
    zero_norm_query_voxel_mask: np.ndarray,
) -> ProtoEMPosteriorState:
    """Build one deterministic posterior state."""

    foreground_array = np.ascontiguousarray(
        foreground_posterior_map.astype(_FLOAT_DTYPE, copy=False)
    )
    background_array = np.ascontiguousarray(
        background_posterior_map.astype(_FLOAT_DTYPE, copy=False)
    )
    confidence_array = np.ascontiguousarray(confidence_map.astype(_FLOAT_DTYPE, copy=False))
    confident_mask_array = np.ascontiguousarray(
        confident_voxel_mask.astype(_BOOL_DTYPE, copy=False)
    )
    zero_norm_array = np.ascontiguousarray(
        zero_norm_query_voxel_mask.astype(_BOOL_DTYPE, copy=False)
    )
    posterior_state_identity_hash = sha256_json(
        protoem_posterior_state_identity_payload_from_parts(
            foreground_posterior_content_hash=_array_content_sha256(foreground_array),
            background_posterior_content_hash=_array_content_sha256(background_array),
            confidence_content_hash=_array_content_sha256(confidence_array),
            confident_mask_content_hash=_array_content_sha256(confident_mask_array),
            zero_norm_query_voxel_content_hash=_array_content_sha256(zero_norm_array),
        )
    )
    return ProtoEMPosteriorState(
        schema_name=PROTOEM_POSTERIOR_STATE_SCHEMA_NAME,
        schema_version=PROTOEM_POSTERIOR_STATE_SCHEMA_VERSION,
        foreground_posterior_map=foreground_array,
        background_posterior_map=background_array,
        confidence_map=confidence_array,
        confident_voxel_mask=confident_mask_array,
        zero_norm_query_voxel_mask=zero_norm_array,
        posterior_state_identity_hash=posterior_state_identity_hash,
    )


def build_protoem_assignment_state(
    *,
    hard_assignment_map: np.ndarray,
    confident_voxel_mask: np.ndarray,
) -> ProtoEMAssignmentState:
    """Build one deterministic assignment state from one assignment map and confident mask."""

    assignment = np.ascontiguousarray(hard_assignment_map.astype(_PREDICTION_DTYPE, copy=False))
    confident_mask = np.ascontiguousarray(confident_voxel_mask.astype(_BOOL_DTYPE, copy=False))
    if assignment.shape != confident_mask.shape:
        raise AssignmentConsistencyFailureError(
            "hard_assignment_map and confident_voxel_mask must share the same shape."
        )
    confident_voxel_count = int(np.count_nonzero(confident_mask))
    foreground_assignment_count = int(np.count_nonzero(assignment[confident_mask]))
    background_assignment_count = confident_voxel_count - foreground_assignment_count
    foreground_fraction = (
        foreground_assignment_count / confident_voxel_count if confident_voxel_count > 0 else 0.0
    )
    assignment_state_identity_hash = sha256_json(
        protoem_assignment_state_identity_payload_from_parts(
            hard_assignment_content_hash=_array_content_sha256(assignment),
            confident_voxel_count=confident_voxel_count,
            foreground_assignment_count=foreground_assignment_count,
            background_assignment_count=background_assignment_count,
            foreground_fraction=float(foreground_fraction),
        )
    )
    return ProtoEMAssignmentState(
        schema_name=PROTOEM_ASSIGNMENT_STATE_SCHEMA_NAME,
        schema_version=PROTOEM_ASSIGNMENT_STATE_SCHEMA_VERSION,
        hard_assignment_map=assignment,
        confident_voxel_count=confident_voxel_count,
        foreground_assignment_count=foreground_assignment_count,
        background_assignment_count=background_assignment_count,
        foreground_fraction=float(foreground_fraction),
        assignment_state_identity_hash=assignment_state_identity_hash,
    )


def hash_protoem_objective_terms_state(objective_terms: ProtoEMObjectiveTerms) -> str:
    """Return the deterministic objective-terms identity hash used by transductive state."""

    return sha256_json(protoem_objective_terms_identity_payload(objective_terms))


def hash_protoem_prototype_state(state: ProtoEMPrototypeState) -> str:
    """Return the deterministic prototype-state identity hash."""

    return sha256_json(protoem_prototype_state_identity_payload(state))


def hash_protoem_posterior_state(state: ProtoEMPosteriorState) -> str:
    """Return the deterministic posterior-state identity hash."""

    return sha256_json(protoem_posterior_state_identity_payload(state))


def hash_protoem_assignment_state(state: ProtoEMAssignmentState) -> str:
    """Return the deterministic assignment-state identity hash."""

    return sha256_json(protoem_assignment_state_identity_payload(state))


def hash_protoem_transductive_state(state: ProtoEMTransductiveState) -> str:
    """Return the deterministic transductive-state identity hash."""

    return sha256_json(protoem_transductive_state_identity_payload(state))


def hash_protoem_state_transition(transition: ProtoEMStateTransition) -> str:
    """Return the deterministic state-transition identity hash."""

    return sha256_json(protoem_state_transition_identity_payload(transition))


def protoem_objective_terms_identity_payload(
    objective_terms: ProtoEMObjectiveTerms,
) -> dict[str, JsonValue]:
    """Return the canonical objective-terms identity payload used by state objects."""

    return {
        "schema_name": objective_terms.schema_name,
        "schema_version": objective_terms.schema_version,
        "support": objective_terms.support,
        "query_entropy": objective_terms.query_entropy,
        "class_balance": objective_terms.class_balance,
        "consistency": objective_terms.consistency,
        "proximal": objective_terms.proximal,
        "total": objective_terms.total,
        "weight_support": objective_terms.weights.support,
        "weight_query_entropy": objective_terms.weights.query_entropy,
        "weight_class_balance": objective_terms.weights.class_balance,
        "weight_consistency": objective_terms.weights.consistency,
        "weight_proximal": objective_terms.weights.proximal,
    }


def protoem_prototype_state_identity_payload(
    state: ProtoEMPrototypeState,
) -> dict[str, JsonValue]:
    """Return the canonical prototype-state identity payload."""

    foreground_bank = _prototype_bank_view(state.foreground_prototypes)
    background_bank = _prototype_bank_view(state.background_prototypes)
    return protoem_prototype_state_identity_payload_from_parts(
        prototype_mode=state.prototype_mode,
        channel_count=state.channel_count,
        foreground_content_hash=state.foreground_content_hash,
        background_content_hash=state.background_content_hash,
        foreground_prototype_count=int(foreground_bank.shape[0]),
        background_prototype_count=int(background_bank.shape[0]),
    )


def protoem_posterior_state_identity_payload(
    state: ProtoEMPosteriorState,
) -> dict[str, JsonValue]:
    """Return the canonical posterior-state identity payload."""

    return protoem_posterior_state_identity_payload_from_parts(
        foreground_posterior_content_hash=_array_content_sha256(state.foreground_posterior_map),
        background_posterior_content_hash=_array_content_sha256(state.background_posterior_map),
        confidence_content_hash=_array_content_sha256(state.confidence_map),
        confident_mask_content_hash=_array_content_sha256(state.confident_voxel_mask),
        zero_norm_query_voxel_content_hash=_array_content_sha256(state.zero_norm_query_voxel_mask),
    )


def protoem_assignment_state_identity_payload(
    state: ProtoEMAssignmentState,
) -> dict[str, JsonValue]:
    """Return the canonical assignment-state identity payload."""

    return protoem_assignment_state_identity_payload_from_parts(
        hard_assignment_content_hash=_array_content_sha256(state.hard_assignment_map),
        confident_voxel_count=state.confident_voxel_count,
        foreground_assignment_count=state.foreground_assignment_count,
        background_assignment_count=state.background_assignment_count,
        foreground_fraction=state.foreground_fraction,
    )


def protoem_transductive_state_identity_payload(
    state: ProtoEMTransductiveState,
) -> dict[str, JsonValue]:
    """Return the canonical transductive-state identity payload."""

    return protoem_transductive_state_identity_payload_from_parts(
        iteration_index=state.iteration_index,
        initialization_identity_hash=state.initialization_identity_hash,
        query_feature_content_hash=state.query_feature_content_hash,
        prototype_state_identity_hash=state.prototype_state.prototype_state_identity_hash,
        posterior_state_identity_hash=state.posterior_state.posterior_state_identity_hash,
        assignment_state_identity_hash=state.assignment_state.assignment_state_identity_hash,
        previous_objective_total=state.previous_objective_total,
        current_objective_terms_identity=(
            hash_protoem_objective_terms_state(state.current_objective_terms)
            if state.current_objective_terms is not None
            else None
        ),
        convergence_delta=state.convergence_delta,
    )


def protoem_state_transition_identity_payload(
    transition: ProtoEMStateTransition,
) -> dict[str, JsonValue]:
    """Return the canonical transition identity payload."""

    return protoem_state_transition_identity_payload_from_parts(
        source_state_identity=transition.source_state_identity,
        target_state_identity=transition.target_state_identity,
        iteration_index_before=transition.iteration_index_before,
        iteration_index_after=transition.iteration_index_after,
        e_step_result_identity=transition.e_step_result_identity,
        m_step_result_identity=transition.m_step_result_identity,
        objective_terms_identity=transition.objective_terms_identity,
    )


def protoem_prototype_state_to_dict(state: ProtoEMPrototypeState) -> dict[str, JsonValue]:
    """Convert one prototype state to a canonical summary mapping."""

    foreground_bank = _prototype_bank_view(state.foreground_prototypes)
    background_bank = _prototype_bank_view(state.background_prototypes)
    return {
        **protoem_prototype_state_identity_payload(state),
        "prototype_state_identity_hash": state.prototype_state_identity_hash,
        "foreground_shape": cast(JsonValue, list(foreground_bank.shape)),
        "background_shape": cast(JsonValue, list(background_bank.shape)),
        "foreground_dtype": str(np.asarray(state.foreground_prototypes).dtype),
        "background_dtype": str(np.asarray(state.background_prototypes).dtype),
    }


def protoem_posterior_state_to_dict(state: ProtoEMPosteriorState) -> dict[str, JsonValue]:
    """Convert one posterior state to a canonical summary mapping."""

    return {
        **protoem_posterior_state_identity_payload(state),
        "posterior_state_identity_hash": state.posterior_state_identity_hash,
        "shape": cast(JsonValue, list(state.foreground_posterior_map.shape)),
        "dtype": str(np.asarray(state.foreground_posterior_map).dtype),
    }


def protoem_assignment_state_to_dict(state: ProtoEMAssignmentState) -> dict[str, JsonValue]:
    """Convert one assignment state to a canonical summary mapping."""

    return {
        **protoem_assignment_state_identity_payload(state),
        "assignment_state_identity_hash": state.assignment_state_identity_hash,
        "shape": cast(JsonValue, list(state.hard_assignment_map.shape)),
        "dtype": str(np.asarray(state.hard_assignment_map).dtype),
    }


def protoem_transductive_state_to_dict(
    state: ProtoEMTransductiveState,
) -> dict[str, JsonValue]:
    """Convert one transductive state to a canonical summary mapping."""

    return {
        **protoem_transductive_state_identity_payload(state),
        "state_identity_hash": state.state_identity_hash,
        "prototype_state": cast(JsonValue, protoem_prototype_state_to_dict(state.prototype_state)),
        "posterior_state": cast(JsonValue, protoem_posterior_state_to_dict(state.posterior_state)),
        "assignment_state": cast(
            JsonValue,
            protoem_assignment_state_to_dict(state.assignment_state),
        ),
    }


def protoem_state_transition_to_dict(
    transition: ProtoEMStateTransition,
) -> dict[str, JsonValue]:
    """Convert one state transition to a canonical summary mapping."""

    return {
        **protoem_state_transition_identity_payload(transition),
        "transition_identity_hash": transition.transition_identity_hash,
    }


def protoem_transductive_state_to_json(state: ProtoEMTransductiveState) -> bytes:
    """Serialize one transductive state summary to canonical JSON bytes."""

    return canonical_json_bytes(protoem_transductive_state_to_dict(state)) + b"\n"


def protoem_state_transition_to_json(transition: ProtoEMStateTransition) -> bytes:
    """Serialize one state transition summary to canonical JSON bytes."""

    return canonical_json_bytes(protoem_state_transition_to_dict(transition)) + b"\n"


def _build_transductive_state(
    *,
    iteration_index: int,
    initialization_identity_hash: str,
    query_feature_content_hash: str,
    prototype_state: ProtoEMPrototypeState,
    posterior_state: ProtoEMPosteriorState,
    assignment_state: ProtoEMAssignmentState,
    previous_objective_total: float | None,
    current_objective_terms: ProtoEMObjectiveTerms | None,
    convergence_delta: float | None,
) -> ProtoEMTransductiveState:
    current_objective_terms_identity = (
        hash_protoem_objective_terms_state(current_objective_terms)
        if current_objective_terms is not None
        else None
    )
    state_identity_hash = sha256_json(
        protoem_transductive_state_identity_payload_from_parts(
            iteration_index=iteration_index,
            initialization_identity_hash=initialization_identity_hash,
            query_feature_content_hash=query_feature_content_hash,
            prototype_state_identity_hash=prototype_state.prototype_state_identity_hash,
            posterior_state_identity_hash=posterior_state.posterior_state_identity_hash,
            assignment_state_identity_hash=assignment_state.assignment_state_identity_hash,
            previous_objective_total=previous_objective_total,
            current_objective_terms_identity=current_objective_terms_identity,
            convergence_delta=convergence_delta,
        )
    )
    return ProtoEMTransductiveState(
        schema_name=PROTOEM_TRANSDUCTIVE_STATE_SCHEMA_NAME,
        schema_version=PROTOEM_TRANSDUCTIVE_STATE_SCHEMA_VERSION,
        iteration_index=iteration_index,
        initialization_identity_hash=initialization_identity_hash,
        query_feature_content_hash=query_feature_content_hash,
        prototype_state=prototype_state,
        posterior_state=posterior_state,
        assignment_state=assignment_state,
        previous_objective_total=previous_objective_total,
        current_objective_terms=current_objective_terms,
        convergence_delta=convergence_delta,
        state_identity_hash=state_identity_hash,
    )


def protoem_prototype_state_identity_payload_from_parts(
    *,
    prototype_mode: str,
    channel_count: int,
    foreground_content_hash: str,
    background_content_hash: str,
    foreground_prototype_count: int,
    background_prototype_count: int,
) -> dict[str, JsonValue]:
    return {
        "schema_name": PROTOEM_PROTOTYPE_STATE_SCHEMA_NAME,
        "schema_version": PROTOEM_PROTOTYPE_STATE_SCHEMA_VERSION,
        "prototype_mode": prototype_mode,
        "channel_count": channel_count,
        "foreground_content_hash": foreground_content_hash,
        "background_content_hash": background_content_hash,
        "foreground_prototype_count": foreground_prototype_count,
        "background_prototype_count": background_prototype_count,
    }


def protoem_posterior_state_identity_payload_from_parts(
    *,
    foreground_posterior_content_hash: str,
    background_posterior_content_hash: str,
    confidence_content_hash: str,
    confident_mask_content_hash: str,
    zero_norm_query_voxel_content_hash: str,
) -> dict[str, JsonValue]:
    return {
        "schema_name": PROTOEM_POSTERIOR_STATE_SCHEMA_NAME,
        "schema_version": PROTOEM_POSTERIOR_STATE_SCHEMA_VERSION,
        "foreground_posterior_content_hash": foreground_posterior_content_hash,
        "background_posterior_content_hash": background_posterior_content_hash,
        "confidence_content_hash": confidence_content_hash,
        "confident_mask_content_hash": confident_mask_content_hash,
        "zero_norm_query_voxel_content_hash": zero_norm_query_voxel_content_hash,
    }


def protoem_assignment_state_identity_payload_from_parts(
    *,
    hard_assignment_content_hash: str,
    confident_voxel_count: int,
    foreground_assignment_count: int,
    background_assignment_count: int,
    foreground_fraction: float,
) -> dict[str, JsonValue]:
    return {
        "schema_name": PROTOEM_ASSIGNMENT_STATE_SCHEMA_NAME,
        "schema_version": PROTOEM_ASSIGNMENT_STATE_SCHEMA_VERSION,
        "hard_assignment_content_hash": hard_assignment_content_hash,
        "confident_voxel_count": confident_voxel_count,
        "foreground_assignment_count": foreground_assignment_count,
        "background_assignment_count": background_assignment_count,
        "foreground_fraction": foreground_fraction,
    }


def protoem_transductive_state_identity_payload_from_parts(
    *,
    iteration_index: int,
    initialization_identity_hash: str,
    query_feature_content_hash: str,
    prototype_state_identity_hash: str,
    posterior_state_identity_hash: str,
    assignment_state_identity_hash: str,
    previous_objective_total: float | None,
    current_objective_terms_identity: str | None,
    convergence_delta: float | None,
) -> dict[str, JsonValue]:
    return {
        "schema_name": PROTOEM_TRANSDUCTIVE_STATE_SCHEMA_NAME,
        "schema_version": PROTOEM_TRANSDUCTIVE_STATE_SCHEMA_VERSION,
        "iteration_index": iteration_index,
        "initialization_identity_hash": initialization_identity_hash,
        "query_feature_content_hash": query_feature_content_hash,
        "prototype_state_identity_hash": prototype_state_identity_hash,
        "posterior_state_identity_hash": posterior_state_identity_hash,
        "assignment_state_identity_hash": assignment_state_identity_hash,
        "previous_objective_total": previous_objective_total,
        "current_objective_terms_identity": current_objective_terms_identity,
        "convergence_delta": convergence_delta,
    }


def protoem_state_transition_identity_payload_from_parts(
    *,
    source_state_identity: str,
    target_state_identity: str,
    iteration_index_before: int,
    iteration_index_after: int,
    e_step_result_identity: str,
    m_step_result_identity: str,
    objective_terms_identity: str,
) -> dict[str, JsonValue]:
    return {
        "schema_name": PROTOEM_STATE_TRANSITION_SCHEMA_NAME,
        "schema_version": PROTOEM_STATE_TRANSITION_SCHEMA_VERSION,
        "source_state_identity": source_state_identity,
        "target_state_identity": target_state_identity,
        "iteration_index_before": iteration_index_before,
        "iteration_index_after": iteration_index_after,
        "e_step_result_identity": e_step_result_identity,
        "m_step_result_identity": m_step_result_identity,
        "objective_terms_identity": objective_terms_identity,
    }


def _validate_config_initialization_match(
    *,
    config: ProtoEMConfig,
    initialization_bundle: ProtoEMInitializationBundle,
) -> None:
    if (
        config.phase5_initialization_schema_name
        != initialization_bundle.phase5_initialization_reference.artifact_schema_name
        or config.phase5_initialization_artifact_hash
        != initialization_bundle.phase5_initialization_reference.artifact_hash
    ):
        raise IncompatibleStateTransitionError(
            "config Phase 5 initialization reference must match initialization bundle."
        )
    if (
        config.phase5_prototype_schema_name
        != initialization_bundle.phase5_prototype_reference.artifact_schema_name
        or config.phase5_prototype_artifact_hash
        != initialization_bundle.phase5_prototype_reference.artifact_hash
    ):
        raise IncompatibleStateTransitionError(
            "config Phase 5 prototype reference must match initialization bundle."
        )
    if (
        config.phase5_inference_schema_name
        != initialization_bundle.phase5_inference_reference.artifact_schema_name
        or config.phase5_inference_artifact_hash
        != initialization_bundle.phase5_inference_reference.artifact_hash
    ):
        raise IncompatibleStateTransitionError(
            "config Phase 5 inference reference must match initialization bundle."
        )


def _posterior_from_phase5_scores(
    *,
    foreground_scores: np.ndarray,
    background_scores: np.ndarray,
    zero_norm_mask: np.ndarray,
    temperature: float,
    foreground_prior: float,
) -> tuple[np.ndarray, np.ndarray]:
    _require_positive_float(temperature, field_name="temperature")
    _require_fraction_open(
        foreground_prior,
        field_name="foreground_prior",
        error_type=InvalidStateError,
    )
    fg = _require_float_volume(
        foreground_scores,
        field_name="foreground_scores",
        error_type=InvalidStateError,
    )
    bg = _require_float_volume(
        background_scores,
        field_name="background_scores",
        error_type=InvalidStateError,
    )
    zero_mask = _require_bool_volume(
        zero_norm_mask,
        field_name="zero_norm_mask",
        error_type=InvalidStateError,
    )
    if fg.shape != bg.shape or fg.shape != zero_mask.shape:
        raise InvalidStateError(
            "foreground/background scores and zero-norm mask must share the same shape."
        )
    foreground_logits = fg.astype(_FLOAT_DTYPE, copy=False) / temperature + math.log(
        foreground_prior
    )
    background_logits = bg.astype(_FLOAT_DTYPE, copy=False) / temperature + math.log(
        1.0 - foreground_prior
    )
    zero_mask_bool = zero_mask.astype(_BOOL_DTYPE, copy=False)
    foreground_logits[zero_mask_bool] = math.log(foreground_prior)
    background_logits[zero_mask_bool] = math.log(1.0 - foreground_prior)

    stacked = np.stack((background_logits, foreground_logits), axis=-1)
    max_logits = np.max(stacked, axis=-1, keepdims=True)
    stable = stacked - max_logits
    exp_logits = np.exp(stable, dtype=_FLOAT_DTYPE)
    sums = np.sum(exp_logits, axis=-1, keepdims=True)
    posterior = exp_logits / sums
    if not np.isfinite(posterior).all():
        raise InvalidStateError(
            "posterior probabilities derived from Phase 5 scores must be finite."
        )
    background_posterior = np.ascontiguousarray(posterior[..., 0].astype(_FLOAT_DTYPE, copy=False))
    foreground_posterior = np.ascontiguousarray(posterior[..., 1].astype(_FLOAT_DTYPE, copy=False))
    return foreground_posterior, background_posterior


def _zero_norm_query_mask(query_features: np.ndarray) -> np.ndarray:
    if query_features.ndim != 5 or int(query_features.shape[0]) != 1:
        raise InvalidStateError("query feature tensor must have shape [1,C,D,H,W].")
    norms = np.linalg.norm(np.moveaxis(query_features, 1, -1), axis=-1)
    if not np.isfinite(norms).all():
        raise InvalidStateError("query feature norms must be finite.")
    return np.ascontiguousarray((norms <= 0.0)[:, np.newaxis, :, :, :].astype(_BOOL_DTYPE))


def _validate_assignment_vs_posterior(
    *,
    posterior_state: ProtoEMPosteriorState,
    assignment_state: ProtoEMAssignmentState,
) -> None:
    expected_assignment = (
        posterior_state.foreground_posterior_map > posterior_state.background_posterior_map
    ).astype(_PREDICTION_DTYPE, copy=False)
    if not np.array_equal(expected_assignment, assignment_state.hard_assignment_map):
        raise AssignmentConsistencyFailureError(
            "hard_assignment_map must agree with posterior argmax under background tie resolution."
        )
    expected_confident_mask = posterior_state.confident_voxel_mask.astype(_BOOL_DTYPE, copy=False)
    confident_count = int(np.count_nonzero(expected_confident_mask))
    foreground_count = int(
        np.count_nonzero(assignment_state.hard_assignment_map[expected_confident_mask])
    )
    background_count = confident_count - foreground_count
    expected_fraction = foreground_count / confident_count if confident_count > 0 else 0.0
    if assignment_state.confident_voxel_count != confident_count:
        raise AssignmentConsistencyFailureError(
            "assignment_state.confident_voxel_count must match the confident voxel mask."
        )
    if assignment_state.foreground_assignment_count != foreground_count:
        raise AssignmentConsistencyFailureError(
            "assignment_state.foreground_assignment_count must match confident assignments."
        )
    if assignment_state.background_assignment_count != background_count:
        raise AssignmentConsistencyFailureError(
            "assignment_state.background_assignment_count must match confident assignments."
        )
    if not math.isclose(
        assignment_state.foreground_fraction,
        expected_fraction,
        rel_tol=0.0,
        abs_tol=_OBJECTIVE_TOLERANCE,
    ):
        raise AssignmentConsistencyFailureError(
            "assignment_state.foreground_fraction must match the confident assignment fraction."
        )


def _require_prototype_array(
    array: np.ndarray,
    *,
    field_name: str,
    channel_count: int,
    prototype_mode: str,
) -> np.ndarray:
    if not isinstance(array, np.ndarray):
        raise PrototypeStateFailureError(f"{field_name} must be a NumPy array.")
    if array.ndim == 1:
        bank = array[np.newaxis, :]
    elif array.ndim == 2:
        bank = array
    else:
        raise PrototypeStateFailureError(f"{field_name} must have shape [C] or [K,C].")
    if int(bank.shape[1]) != channel_count:
        raise PrototypeStateFailureError(
            f"{field_name} channel dimension must equal channel_count."
        )
    if bank.shape[0] <= 0:
        raise PrototypeStateFailureError(f"{field_name} must contain at least one prototype.")
    if prototype_mode == "single_prototype" and int(bank.shape[0]) != 1:
        raise PrototypeStateFailureError(
            f"{field_name} must contain exactly one prototype in single_prototype mode."
        )
    if bank.dtype.kind not in _NUMERIC_KINDS:
        raise PrototypeStateFailureError(f"{field_name} must use a numeric dtype.")
    bank64 = np.ascontiguousarray(bank.astype(_FLOAT_DTYPE, copy=False))
    if not np.isfinite(bank64).all():
        raise PrototypeStateFailureError(f"{field_name} must contain only finite values.")
    norms = np.linalg.norm(bank64, axis=1)
    if np.any(norms <= 0.0):
        raise PrototypeStateFailureError(f"{field_name} must not contain zero-norm prototypes.")
    return bank64


def _prototype_bank_view(array: np.ndarray) -> np.ndarray:
    if array.ndim == 1:
        return np.ascontiguousarray(array[np.newaxis, :].astype(_FLOAT_DTYPE, copy=False))
    return np.ascontiguousarray(array.astype(_FLOAT_DTYPE, copy=False))


def _prototype_content_sha256(array: np.ndarray) -> str:
    bank = _prototype_bank_view(array)
    return _array_content_sha256(_canonicalize_bank(bank))


def _canonicalize_bank(bank: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(bank.astype(_FLOAT_DTYPE, copy=False))
    keys = [row.tobytes(order="C") for row in contiguous]
    order = np.argsort(np.asarray(keys, dtype=np.bytes_))
    return np.ascontiguousarray(contiguous[order])


def _require_float_volume(
    array: np.ndarray,
    *,
    field_name: str,
    error_type: type[Exception],
) -> np.ndarray:
    if not isinstance(array, np.ndarray):
        raise error_type(f"{field_name} must be a NumPy array.")
    if array.ndim != 5 or tuple(int(item) for item in array.shape[:2]) != (1, 1):
        raise error_type(f"{field_name} must have shape [1,1,D,H,W].")
    if array.dtype.kind not in _FLOATING_KINDS:
        raise error_type(f"{field_name} must use a floating dtype.")
    array64 = np.ascontiguousarray(array.astype(_FLOAT_DTYPE, copy=False))
    if not np.isfinite(array64).all():
        raise error_type(f"{field_name} must contain only finite values.")
    return array64


def _require_probability_volume(
    array: np.ndarray,
    *,
    field_name: str,
    error_type: type[Exception],
) -> np.ndarray:
    volume = _require_float_volume(array, field_name=field_name, error_type=error_type)
    if np.any(volume < 0.0) or np.any(volume > 1.0):
        raise error_type(f"{field_name} must lie in [0,1].")
    return volume


def _require_bool_volume(
    array: np.ndarray,
    *,
    field_name: str,
    error_type: type[Exception],
) -> np.ndarray:
    if not isinstance(array, np.ndarray):
        raise error_type(f"{field_name} must be a NumPy array.")
    if array.ndim != 5 or tuple(int(item) for item in array.shape[:2]) != (1, 1):
        raise error_type(f"{field_name} must have shape [1,1,D,H,W].")
    if array.dtype.kind not in frozenset({"b", "i", "u"}):
        raise error_type(f"{field_name} must use a boolean-compatible dtype.")
    values = set(int(item) for item in np.unique(array).tolist())
    if values - {0, 1}:
        raise error_type(f"{field_name} must be binary.")
    return np.ascontiguousarray(array.astype(_BOOL_DTYPE, copy=False))


def _require_assignment_volume(
    array: np.ndarray,
    *,
    field_name: str,
    error_type: type[Exception],
) -> np.ndarray:
    if not isinstance(array, np.ndarray):
        raise error_type(f"{field_name} must be a NumPy array.")
    if array.ndim != 5 or tuple(int(item) for item in array.shape[:2]) != (1, 1):
        raise error_type(f"{field_name} must have shape [1,1,D,H,W].")
    if array.dtype.kind not in frozenset({"b", "i", "u"}):
        raise error_type(f"{field_name} must use a binary-compatible dtype.")
    values = set(int(item) for item in np.unique(array).tolist())
    if values - {0, 1}:
        raise error_type(f"{field_name} must be binary.")
    return np.ascontiguousarray(array.astype(_PREDICTION_DTYPE, copy=False))


def _array_content_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def _require_schema_name(value: str, *, expected: str, field_name: str) -> None:
    if value != expected:
        raise InvalidStateError(f"{field_name} must equal {expected!r}.")


def _require_schema_version(value: str, *, expected: str, field_name: str) -> None:
    if value != expected:
        raise InvalidStateError(f"{field_name} must equal {expected!r}.")


def _require_sha256(value: str, *, field_name: str) -> None:
    if len(value) != 64 or any(char not in _SHA256_HEX for char in value):
        raise InvalidStateError(
            f"{field_name} must be a lowercase 64-character SHA-256 hex digest."
        )


def _require_positive_float(value: float, *, field_name: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise InvalidStateError(f"{field_name} must be positive and finite.")


def _require_fraction_open(
    value: float,
    *,
    field_name: str,
    error_type: type[Exception],
) -> None:
    if not math.isfinite(value) or value <= 0.0 or value >= 1.0:
        raise error_type(f"{field_name} must lie in (0,1).")


def _require_fraction_closed(
    value: float,
    *,
    field_name: str,
    error_type: type[Exception],
) -> None:
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise error_type(f"{field_name} must lie in [0,1].")


__all__ = [
    "PROTOEM_ASSIGNMENT_STATE_SCHEMA_NAME",
    "PROTOEM_ASSIGNMENT_STATE_SCHEMA_VERSION",
    "PROTOEM_POSTERIOR_STATE_SCHEMA_NAME",
    "PROTOEM_POSTERIOR_STATE_SCHEMA_VERSION",
    "PROTOEM_PROTOTYPE_STATE_SCHEMA_NAME",
    "PROTOEM_PROTOTYPE_STATE_SCHEMA_VERSION",
    "PROTOEM_STATE_TRANSITION_SCHEMA_NAME",
    "PROTOEM_STATE_TRANSITION_SCHEMA_VERSION",
    "PROTOEM_TRANSDUCTIVE_STATE_SCHEMA_NAME",
    "PROTOEM_TRANSDUCTIVE_STATE_SCHEMA_VERSION",
    "AssignmentConsistencyFailureError",
    "IncompatibleStateTransitionError",
    "InvalidStateError",
    "PosteriorConsistencyFailureError",
    "ProtoEMAssignmentState",
    "ProtoEMPosteriorState",
    "ProtoEMPrototypeState",
    "ProtoEMStateError",
    "ProtoEMStateTransition",
    "ProtoEMTransductiveState",
    "PrototypeStateFailureError",
    "StateIdentityMismatchError",
    "build_initial_protoem_transductive_state",
    "build_next_protoem_transductive_state",
    "build_protoem_assignment_state",
    "build_protoem_posterior_state",
    "build_protoem_prototype_state",
    "hash_protoem_assignment_state",
    "hash_protoem_objective_terms_state",
    "hash_protoem_posterior_state",
    "hash_protoem_prototype_state",
    "hash_protoem_state_transition",
    "hash_protoem_transductive_state",
    "protoem_assignment_state_identity_payload",
    "protoem_assignment_state_to_dict",
    "protoem_objective_terms_identity_payload",
    "protoem_posterior_state_identity_payload",
    "protoem_posterior_state_to_dict",
    "protoem_prototype_state_identity_payload",
    "protoem_prototype_state_to_dict",
    "protoem_state_transition_identity_payload",
    "protoem_state_transition_to_dict",
    "protoem_state_transition_to_json",
    "protoem_transductive_state_identity_payload",
    "protoem_transductive_state_to_dict",
    "protoem_transductive_state_to_json",
]
