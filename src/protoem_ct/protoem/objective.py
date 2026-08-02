"""Deterministic ProtoEM-CT objective, E-step, and M-step contracts."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Final, cast

import numpy as np

from protoem_ct.artifacts.hashing import JsonValue, sha256_json
from protoem_ct.protoem.artifacts import ProtoEMObjectiveWeights

PROTOEM_CONFIDENCE_MASK_RESULT_SCHEMA_NAME: Final[str] = "protoem_confidence_mask_result"
PROTOEM_CONFIDENCE_MASK_RESULT_SCHEMA_VERSION: Final[str] = "v1"
PROTOEM_OBJECTIVE_TERMS_SCHEMA_NAME: Final[str] = "protoem_objective_terms"
PROTOEM_OBJECTIVE_TERMS_SCHEMA_VERSION: Final[str] = "v1"
PROTOEM_E_STEP_RESULT_SCHEMA_NAME: Final[str] = "protoem_e_step_result"
PROTOEM_E_STEP_RESULT_SCHEMA_VERSION: Final[str] = "v1"
PROTOEM_M_STEP_RESULT_SCHEMA_NAME: Final[str] = "protoem_m_step_result"
PROTOEM_M_STEP_RESULT_SCHEMA_VERSION: Final[str] = "v1"

PROTOEM_MULTI_PROTOTYPE_AGGREGATION_RULE: Final[str] = "max_class_similarity"
PROTOEM_POSTERIOR_TIE_POLICY: Final[str] = "background_on_exact_tie"
_FLOAT_DTYPE: Final[np.dtype[np.float64]] = np.dtype(np.float64)
_MASK_DTYPE: Final[np.dtype[np.bool_]] = np.dtype(np.bool_)
_PREDICTION_DTYPE: Final[np.dtype[np.uint8]] = np.dtype(np.uint8)
_FLOATING_KINDS: Final[frozenset[str]] = frozenset({"f"})
_NUMERIC_KINDS: Final[frozenset[str]] = frozenset({"f", "i", "u"})
_SHA256_HEX: Final[frozenset[str]] = frozenset("0123456789abcdef")
_OBJECTIVE_TOLERANCE: Final[float] = 1e-12
_COSINE_DRIFT_TOLERANCE: Final[float] = 1e-12


class ProtoEMObjectiveError(ValueError):
    """Base error for deterministic ProtoEM objective evaluation."""


class InvalidEStepInputError(ProtoEMObjectiveError):
    """Raised when one E-step input is malformed or non-finite."""


class InvalidMStepInputError(ProtoEMObjectiveError):
    """Raised when one M-step input is malformed or inconsistent."""


class IncompatiblePrototypeBankError(ProtoEMObjectiveError):
    """Raised when one prototype bank is malformed or incompatible with query features."""


class ZeroNormPrototypeError(ProtoEMObjectiveError):
    """Raised when one prototype bank contains a zero-norm prototype."""


class NoConfidentVoxelsError(ProtoEMObjectiveError):
    """Raised when the confident-voxel mask is empty for the M-step."""


class EmptyEffectiveForegroundUpdateError(ProtoEMObjectiveError):
    """Raised when no confident effective foreground weight exists for an M-step update."""


class EmptyEffectiveBackgroundUpdateError(ProtoEMObjectiveError):
    """Raised when no confident effective background weight exists for an M-step update."""


class NonFiniteObjectiveError(ProtoEMObjectiveError):
    """Raised when one objective term or result is non-finite."""


class ObjectiveConsistencyFailureError(ProtoEMObjectiveError):
    """Raised when the total objective does not equal the exact weighted sum."""


@dataclass(frozen=True, slots=True)
class ProtoEMConfidenceMaskResult:
    """Deterministic confidence-mask summary for one E-step."""

    schema_name: str
    schema_version: str
    confidence_map: np.ndarray
    confident_mask: np.ndarray
    confident_voxel_count: int
    foreground_assignment_count: int
    background_assignment_count: int
    foreground_fraction: float

    def __post_init__(self) -> None:
        _require_schema_name(
            self.schema_name,
            expected=PROTOEM_CONFIDENCE_MASK_RESULT_SCHEMA_NAME,
            field_name="schema_name",
        )
        _require_schema_version(
            self.schema_version,
            expected=PROTOEM_CONFIDENCE_MASK_RESULT_SCHEMA_VERSION,
            field_name="schema_version",
        )
        confidence_map = _require_volume(
            self.confidence_map,
            field_name="confidence_map",
            expected_dtype_kinds=_FLOATING_KINDS,
        )
        confident_mask = _require_volume(
            self.confident_mask,
            field_name="confident_mask",
            expected_dtype_kinds=frozenset({"b", "i", "u"}),
        )
        if confidence_map.shape != confident_mask.shape:
            raise InvalidEStepInputError(
                "confidence_map and confident_mask must have identical shapes."
            )
        if self.confident_voxel_count < 0:
            raise InvalidEStepInputError("confident_voxel_count must be nonnegative.")
        if self.foreground_assignment_count < 0:
            raise InvalidEStepInputError("foreground_assignment_count must be nonnegative.")
        if self.background_assignment_count < 0:
            raise InvalidEStepInputError("background_assignment_count must be nonnegative.")
        mask_count = int(np.count_nonzero(confident_mask))
        if self.confident_voxel_count != mask_count:
            raise InvalidEStepInputError(
                "confident_voxel_count must match the number of true voxels in confident_mask."
            )
        if self.foreground_assignment_count + self.background_assignment_count != mask_count:
            raise InvalidEStepInputError(
                "foreground/background assignment counts must sum to confident_voxel_count."
            )
        if mask_count == 0:
            if self.foreground_fraction != 0.0:
                raise InvalidEStepInputError(
                    "foreground_fraction must equal 0.0 when no confident voxels exist."
                )
        else:
            expected_fraction = self.foreground_assignment_count / mask_count
            if not math.isclose(
                self.foreground_fraction,
                expected_fraction,
                rel_tol=0.0,
                abs_tol=_OBJECTIVE_TOLERANCE,
            ):
                raise InvalidEStepInputError(
                    "foreground_fraction must equal the exact confident foreground fraction."
                )
        _require_finite_fraction_closed(self.foreground_fraction, field_name="foreground_fraction")


@dataclass(frozen=True, slots=True)
class ProtoEMObjectiveTerms:
    """Deterministic ProtoEM-CT objective terms and exact weighted total."""

    schema_name: str
    schema_version: str
    support: float
    query_entropy: float
    class_balance: float
    consistency: float
    proximal: float
    total: float
    weights: ProtoEMObjectiveWeights

    def __post_init__(self) -> None:
        _require_schema_name(
            self.schema_name,
            expected=PROTOEM_OBJECTIVE_TERMS_SCHEMA_NAME,
            field_name="schema_name",
        )
        _require_schema_version(
            self.schema_version,
            expected=PROTOEM_OBJECTIVE_TERMS_SCHEMA_VERSION,
            field_name="schema_version",
        )
        for field_name in (
            "support",
            "query_entropy",
            "class_balance",
            "consistency",
            "proximal",
            "total",
        ):
            value = cast(float, getattr(self, field_name))
            if not math.isfinite(value):
                raise NonFiniteObjectiveError(f"{field_name} must be finite.")
        for field_name in (
            "support",
            "query_entropy",
            "class_balance",
            "consistency",
            "proximal",
        ):
            value = cast(float, getattr(self, field_name))
            if value < 0.0:
                raise NonFiniteObjectiveError(f"{field_name} must be nonnegative.")
        expected_total = (
            self.weights.support * self.support
            + self.weights.query_entropy * self.query_entropy
            + self.weights.class_balance * self.class_balance
            + self.weights.consistency * self.consistency
            + self.weights.proximal * self.proximal
        )
        if not math.isclose(
            self.total,
            expected_total,
            rel_tol=0.0,
            abs_tol=_OBJECTIVE_TOLERANCE,
        ):
            raise ObjectiveConsistencyFailureError(
                "total must equal the exact weighted sum of ProtoEM objective terms."
            )


@dataclass(frozen=True, slots=True)
class ProtoEMEStepResult:
    """Deterministic ProtoEM-CT E-step result."""

    schema_name: str
    schema_version: str
    result_identity_hash: str
    query_feature_content_sha256: str
    foreground_bank_content_sha256: str
    background_bank_content_sha256: str
    temperature: float
    confidence_threshold: float
    foreground_prior: float
    prototype_aggregation_rule: str
    posterior_tie_policy: str
    foreground_posterior_map: np.ndarray
    background_posterior_map: np.ndarray
    hard_assignment_map: np.ndarray
    zero_norm_query_voxel_mask: np.ndarray
    confidence_mask_result: ProtoEMConfidenceMaskResult

    def __post_init__(self) -> None:
        _require_schema_name(
            self.schema_name,
            expected=PROTOEM_E_STEP_RESULT_SCHEMA_NAME,
            field_name="schema_name",
        )
        _require_schema_version(
            self.schema_version,
            expected=PROTOEM_E_STEP_RESULT_SCHEMA_VERSION,
            field_name="schema_version",
        )
        _require_sha256(self.result_identity_hash, field_name="result_identity_hash")
        _require_sha256(
            self.query_feature_content_sha256,
            field_name="query_feature_content_sha256",
        )
        _require_sha256(
            self.foreground_bank_content_sha256,
            field_name="foreground_bank_content_sha256",
        )
        _require_sha256(
            self.background_bank_content_sha256,
            field_name="background_bank_content_sha256",
        )
        _require_positive_finite_float(self.temperature, field_name="temperature")
        _require_finite_fraction_closed(
            self.confidence_threshold,
            field_name="confidence_threshold",
        )
        _require_finite_fraction_open(self.foreground_prior, field_name="foreground_prior")
        if self.prototype_aggregation_rule != PROTOEM_MULTI_PROTOTYPE_AGGREGATION_RULE:
            raise InvalidEStepInputError(
                "prototype_aggregation_rule must equal the deterministic Phase 6 rule."
            )
        if self.posterior_tie_policy != PROTOEM_POSTERIOR_TIE_POLICY:
            raise InvalidEStepInputError("posterior_tie_policy must equal background tie-break.")
        foreground_posterior = _require_probability_volume(
            self.foreground_posterior_map,
            field_name="foreground_posterior_map",
        )
        background_posterior = _require_probability_volume(
            self.background_posterior_map,
            field_name="background_posterior_map",
        )
        hard_assignment = _require_binary_volume(
            self.hard_assignment_map,
            field_name="hard_assignment_map",
        )
        zero_norm_mask = _require_boolean_volume(
            self.zero_norm_query_voxel_mask,
            field_name="zero_norm_query_voxel_mask",
        )
        if (
            foreground_posterior.shape != background_posterior.shape
            or foreground_posterior.shape != hard_assignment.shape
            or foreground_posterior.shape != zero_norm_mask.shape
            or foreground_posterior.shape != self.confidence_mask_result.confidence_map.shape
        ):
            raise InvalidEStepInputError("all E-step output volumes must have identical shapes.")
        posterior_sum = foreground_posterior + background_posterior
        if not np.allclose(posterior_sum, 1.0, rtol=0.0, atol=_OBJECTIVE_TOLERANCE):
            raise InvalidEStepInputError(
                "foreground/background posteriors must sum to one at every voxel."
            )
        expected_assignment = (foreground_posterior > background_posterior).astype(
            _PREDICTION_DTYPE, copy=False
        )
        if not np.array_equal(expected_assignment, hard_assignment):
            raise InvalidEStepInputError(
                "hard_assignment_map must apply argmax with background on exact ties."
            )
        expected_confidence = np.maximum(foreground_posterior, background_posterior)
        if not np.allclose(
            expected_confidence,
            self.confidence_mask_result.confidence_map.astype(_FLOAT_DTYPE, copy=False),
            rtol=0.0,
            atol=_OBJECTIVE_TOLERANCE,
        ):
            raise InvalidEStepInputError(
                "confidence_map must equal max(foreground_posterior, background_posterior)."
            )
        expected_confident_mask = expected_confidence >= self.confidence_threshold
        if not np.array_equal(
            expected_confident_mask,
            self.confidence_mask_result.confident_mask.astype(_MASK_DTYPE, copy=False),
        ):
            raise InvalidEStepInputError(
                "confident_mask must equal confidence >= confidence_threshold."
            )
        if self.result_identity_hash != hash_protoem_e_step_result(self):
            raise ObjectiveConsistencyFailureError(
                "result_identity_hash does not match deterministic E-step content."
            )


@dataclass(frozen=True, slots=True)
class ProtoEMMStepResult:
    """Deterministic ProtoEM-CT M-step result plus exact objective terms."""

    schema_name: str
    schema_version: str
    result_identity_hash: str
    query_feature_content_sha256: str
    previous_foreground_bank_content_sha256: str
    previous_background_bank_content_sha256: str
    initial_foreground_bank_content_sha256: str
    initial_background_bank_content_sha256: str
    updated_foreground_prototype: np.ndarray
    updated_background_prototype: np.ndarray
    updated_foreground_content_sha256: str
    updated_background_content_sha256: str
    effective_foreground_weight: float
    effective_background_weight: float
    confident_voxel_count: int
    objective_terms: ProtoEMObjectiveTerms

    def __post_init__(self) -> None:
        _require_schema_name(
            self.schema_name,
            expected=PROTOEM_M_STEP_RESULT_SCHEMA_NAME,
            field_name="schema_name",
        )
        _require_schema_version(
            self.schema_version,
            expected=PROTOEM_M_STEP_RESULT_SCHEMA_VERSION,
            field_name="schema_version",
        )
        for field_name in (
            "result_identity_hash",
            "query_feature_content_sha256",
            "previous_foreground_bank_content_sha256",
            "previous_background_bank_content_sha256",
            "initial_foreground_bank_content_sha256",
            "initial_background_bank_content_sha256",
            "updated_foreground_content_sha256",
            "updated_background_content_sha256",
        ):
            _require_sha256(cast(str, getattr(self, field_name)), field_name=field_name)
        updated_foreground = _require_prototype_vector(
            self.updated_foreground_prototype,
            field_name="updated_foreground_prototype",
        )
        updated_background = _require_prototype_vector(
            self.updated_background_prototype,
            field_name="updated_background_prototype",
        )
        if updated_foreground.shape != updated_background.shape:
            raise InvalidMStepInputError(
                "updated foreground/background prototypes must share the same channel shape."
            )
        _require_positive_finite_float(
            self.effective_foreground_weight,
            field_name="effective_foreground_weight",
        )
        _require_positive_finite_float(
            self.effective_background_weight,
            field_name="effective_background_weight",
        )
        if self.confident_voxel_count <= 0:
            raise InvalidMStepInputError("confident_voxel_count must be positive.")
        if self.updated_foreground_content_sha256 != _array_content_sha256(updated_foreground):
            raise InvalidMStepInputError(
                "updated_foreground_content_sha256 must match updated_foreground_prototype."
            )
        if self.updated_background_content_sha256 != _array_content_sha256(updated_background):
            raise InvalidMStepInputError(
                "updated_background_content_sha256 must match updated_background_prototype."
            )
        if self.result_identity_hash != hash_protoem_m_step_result(self):
            raise ObjectiveConsistencyFailureError(
                "result_identity_hash does not match deterministic M-step content."
            )


def run_protoem_e_step(
    *,
    query_feature_tensor: np.ndarray,
    foreground_prototypes: np.ndarray,
    background_prototypes: np.ndarray,
    temperature: float,
    confidence_threshold: float,
    foreground_prior: float,
) -> ProtoEMEStepResult:
    """Run the deterministic ProtoEM-CT E-step on one query feature tensor."""

    _require_positive_finite_float(temperature, field_name="temperature")
    _require_finite_fraction_closed(
        confidence_threshold,
        field_name="confidence_threshold",
    )
    _require_finite_fraction_open(foreground_prior, field_name="foreground_prior")
    query_features = _require_query_feature_tensor(
        query_feature_tensor,
        field_name="query_feature_tensor",
    )
    channel_count = int(query_features.shape[1])
    foreground_bank = _require_prototype_bank(
        foreground_prototypes,
        field_name="foreground_prototypes",
        channel_count=channel_count,
    )
    background_bank = _require_prototype_bank(
        background_prototypes,
        field_name="background_prototypes",
        channel_count=channel_count,
    )

    query_vectors = np.moveaxis(query_features, 1, -1)
    query_norms = np.linalg.norm(query_vectors, axis=-1)
    if not np.isfinite(query_norms).all():
        raise InvalidEStepInputError("query voxel norms must be finite.")
    zero_norm_mask_spatial = query_norms <= 0.0
    zero_norm_mask = zero_norm_mask_spatial[:, np.newaxis, :, :, :]

    foreground_similarity = _class_similarity(query_vectors, query_norms, foreground_bank)
    background_similarity = _class_similarity(query_vectors, query_norms, background_bank)

    foreground_logits = foreground_similarity / temperature + math.log(foreground_prior)
    background_logits = background_similarity / temperature + math.log(1.0 - foreground_prior)
    if np.any(zero_norm_mask_spatial):
        foreground_logits[zero_norm_mask_spatial] = math.log(foreground_prior)
        background_logits[zero_norm_mask_spatial] = math.log(1.0 - foreground_prior)

    stacked_logits = np.stack((background_logits, foreground_logits), axis=-1)
    max_logits = np.max(stacked_logits, axis=-1, keepdims=True)
    stable_logits = stacked_logits - max_logits
    exp_logits = np.exp(stable_logits, dtype=_FLOAT_DTYPE)
    exp_sums = np.sum(exp_logits, axis=-1, keepdims=True)
    posterior = exp_logits / exp_sums
    if not np.isfinite(posterior).all():
        raise InvalidEStepInputError("posterior probabilities must be finite.")

    background_posterior = posterior[..., 0][:, np.newaxis, :, :, :]
    foreground_posterior = posterior[..., 1][:, np.newaxis, :, :, :]
    hard_assignment = (foreground_posterior > background_posterior).astype(
        _PREDICTION_DTYPE, copy=False
    )
    confidence_map = np.maximum(foreground_posterior, background_posterior)
    confident_mask = confidence_map >= confidence_threshold

    confident_voxel_count = int(np.count_nonzero(confident_mask))
    foreground_assignment_count = int(np.count_nonzero(hard_assignment[confident_mask]))
    background_assignment_count = confident_voxel_count - foreground_assignment_count
    foreground_fraction = (
        foreground_assignment_count / confident_voxel_count if confident_voxel_count > 0 else 0.0
    )
    confidence_mask_result = ProtoEMConfidenceMaskResult(
        schema_name=PROTOEM_CONFIDENCE_MASK_RESULT_SCHEMA_NAME,
        schema_version=PROTOEM_CONFIDENCE_MASK_RESULT_SCHEMA_VERSION,
        confidence_map=np.ascontiguousarray(confidence_map.astype(_FLOAT_DTYPE, copy=False)),
        confident_mask=np.ascontiguousarray(confident_mask.astype(_MASK_DTYPE, copy=False)),
        confident_voxel_count=confident_voxel_count,
        foreground_assignment_count=foreground_assignment_count,
        background_assignment_count=background_assignment_count,
        foreground_fraction=float(foreground_fraction),
    )

    query_hash = _array_content_sha256(query_features)
    foreground_hash = _prototype_bank_content_sha256(foreground_bank)
    background_hash = _prototype_bank_content_sha256(background_bank)
    result_identity_hash = sha256_json(
        protoem_e_step_identity_payload_from_parts(
            query_feature_content_sha256=query_hash,
            foreground_bank_content_sha256=foreground_hash,
            background_bank_content_sha256=background_hash,
            temperature=temperature,
            confidence_threshold=confidence_threshold,
            foreground_prior=foreground_prior,
            foreground_posterior_content_sha256=_array_content_sha256(foreground_posterior),
            background_posterior_content_sha256=_array_content_sha256(background_posterior),
            hard_assignment_content_sha256=_array_content_sha256(hard_assignment),
            confidence_content_sha256=_array_content_sha256(confidence_mask_result.confidence_map),
            confident_mask_content_sha256=_array_content_sha256(
                confidence_mask_result.confident_mask
            ),
            zero_norm_query_voxel_content_sha256=_array_content_sha256(zero_norm_mask),
            confident_voxel_count=confident_voxel_count,
            foreground_assignment_count=foreground_assignment_count,
            background_assignment_count=background_assignment_count,
            foreground_fraction=foreground_fraction,
        )
    )
    return ProtoEMEStepResult(
        schema_name=PROTOEM_E_STEP_RESULT_SCHEMA_NAME,
        schema_version=PROTOEM_E_STEP_RESULT_SCHEMA_VERSION,
        result_identity_hash=result_identity_hash,
        query_feature_content_sha256=query_hash,
        foreground_bank_content_sha256=foreground_hash,
        background_bank_content_sha256=background_hash,
        temperature=temperature,
        confidence_threshold=confidence_threshold,
        foreground_prior=foreground_prior,
        prototype_aggregation_rule=PROTOEM_MULTI_PROTOTYPE_AGGREGATION_RULE,
        posterior_tie_policy=PROTOEM_POSTERIOR_TIE_POLICY,
        foreground_posterior_map=np.ascontiguousarray(
            foreground_posterior.astype(_FLOAT_DTYPE, copy=False)
        ),
        background_posterior_map=np.ascontiguousarray(
            background_posterior.astype(_FLOAT_DTYPE, copy=False)
        ),
        hard_assignment_map=np.ascontiguousarray(
            hard_assignment.astype(_PREDICTION_DTYPE, copy=False)
        ),
        zero_norm_query_voxel_mask=np.ascontiguousarray(
            zero_norm_mask.astype(_MASK_DTYPE, copy=False)
        ),
        confidence_mask_result=confidence_mask_result,
    )


def compute_protoem_objective_terms(
    *,
    objective_weights: ProtoEMObjectiveWeights,
    foreground_posterior_map: np.ndarray,
    background_posterior_map: np.ndarray,
    previous_foreground_posterior_map: np.ndarray,
    previous_background_posterior_map: np.ndarray,
    updated_foreground_prototype: np.ndarray,
    updated_background_prototype: np.ndarray,
    support_foreground_reference: np.ndarray,
    support_background_reference: np.ndarray,
    proximal_foreground_reference: np.ndarray,
    proximal_background_reference: np.ndarray,
    foreground_prior: float,
) -> ProtoEMObjectiveTerms:
    """Compute the exact ProtoEM-CT objective terms from one E-step and one M-step state."""

    _require_finite_fraction_open(foreground_prior, field_name="foreground_prior")
    foreground_posterior = _require_probability_volume(
        foreground_posterior_map,
        field_name="foreground_posterior_map",
    )
    background_posterior = _require_probability_volume(
        background_posterior_map,
        field_name="background_posterior_map",
    )
    previous_foreground_posterior = _require_probability_volume(
        previous_foreground_posterior_map,
        field_name="previous_foreground_posterior_map",
    )
    previous_background_posterior = _require_probability_volume(
        previous_background_posterior_map,
        field_name="previous_background_posterior_map",
    )
    if (
        foreground_posterior.shape != background_posterior.shape
        or foreground_posterior.shape != previous_foreground_posterior.shape
        or foreground_posterior.shape != previous_background_posterior.shape
    ):
        raise InvalidMStepInputError(
            "current and previous posterior maps must share the same shape."
        )
    _require_posterior_pair(
        foreground_posterior,
        background_posterior,
        field_name="current posterior maps",
    )
    _require_posterior_pair(
        previous_foreground_posterior,
        previous_background_posterior,
        field_name="previous posterior maps",
    )

    updated_foreground = _require_prototype_vector(
        updated_foreground_prototype,
        field_name="updated_foreground_prototype",
    )
    updated_background = _require_prototype_vector(
        updated_background_prototype,
        field_name="updated_background_prototype",
    )
    support_foreground = _reference_vector_from_bank(
        support_foreground_reference,
        field_name="support_foreground_reference",
        channel_count=int(updated_foreground.shape[0]),
    )
    support_background = _reference_vector_from_bank(
        support_background_reference,
        field_name="support_background_reference",
        channel_count=int(updated_background.shape[0]),
    )
    proximal_foreground = _reference_vector_from_bank(
        proximal_foreground_reference,
        field_name="proximal_foreground_reference",
        channel_count=int(updated_foreground.shape[0]),
    )
    proximal_background = _reference_vector_from_bank(
        proximal_background_reference,
        field_name="proximal_background_reference",
        channel_count=int(updated_background.shape[0]),
    )

    support_term = _paired_mean_squared_distance(
        first_current=updated_foreground,
        first_reference=support_foreground,
        second_current=updated_background,
        second_reference=support_background,
    )
    query_entropy_term = _mean_binary_entropy(foreground_posterior)
    class_balance_term = float((float(np.mean(foreground_posterior)) - foreground_prior) ** 2)
    consistency_term = 0.5 * (
        float(np.mean((foreground_posterior - previous_foreground_posterior) ** 2))
        + float(np.mean((background_posterior - previous_background_posterior) ** 2))
    )
    proximal_term = _paired_mean_squared_distance(
        first_current=updated_foreground,
        first_reference=proximal_foreground,
        second_current=updated_background,
        second_reference=proximal_background,
    )
    total = (
        objective_weights.support * support_term
        + objective_weights.query_entropy * query_entropy_term
        + objective_weights.class_balance * class_balance_term
        + objective_weights.consistency * consistency_term
        + objective_weights.proximal * proximal_term
    )
    return ProtoEMObjectiveTerms(
        schema_name=PROTOEM_OBJECTIVE_TERMS_SCHEMA_NAME,
        schema_version=PROTOEM_OBJECTIVE_TERMS_SCHEMA_VERSION,
        support=support_term,
        query_entropy=query_entropy_term,
        class_balance=class_balance_term,
        consistency=consistency_term,
        proximal=proximal_term,
        total=total,
        weights=objective_weights,
    )


def run_protoem_m_step(
    *,
    query_feature_tensor: np.ndarray,
    foreground_posterior_map: np.ndarray,
    background_posterior_map: np.ndarray,
    confidence_mask: np.ndarray,
    initial_foreground_prototype: np.ndarray,
    initial_background_prototype: np.ndarray,
    previous_foreground_prototype: np.ndarray,
    previous_background_prototype: np.ndarray,
    objective_weights: ProtoEMObjectiveWeights,
    foreground_prior: float,
    proximal_foreground_reference: np.ndarray,
    proximal_background_reference: np.ndarray,
    previous_foreground_posterior_map: np.ndarray,
    previous_background_posterior_map: np.ndarray,
    support_foreground_reference: np.ndarray | None = None,
    support_background_reference: np.ndarray | None = None,
) -> ProtoEMMStepResult:
    """Run the deterministic ProtoEM-CT M-step and exact objective evaluation."""

    _require_finite_fraction_open(foreground_prior, field_name="foreground_prior")
    query_features = _require_query_feature_tensor(
        query_feature_tensor,
        field_name="query_feature_tensor",
    )
    channel_count = int(query_features.shape[1])
    foreground_posterior = _require_probability_volume(
        foreground_posterior_map,
        field_name="foreground_posterior_map",
    )
    background_posterior = _require_probability_volume(
        background_posterior_map,
        field_name="background_posterior_map",
    )
    confident_mask = _require_boolean_volume(confidence_mask, field_name="confidence_mask")
    if (
        foreground_posterior.shape != background_posterior.shape
        or foreground_posterior.shape != confident_mask.shape
    ):
        raise InvalidMStepInputError(
            "posterior maps and confidence_mask must share the same shape."
        )
    if foreground_posterior.shape[2:] != query_features.shape[2:]:
        raise InvalidMStepInputError("posterior maps must match the query feature spatial shape.")
    _require_posterior_pair(
        foreground_posterior,
        background_posterior,
        field_name="current posterior maps",
    )
    if not np.any(confident_mask):
        raise NoConfidentVoxelsError("M-step requires at least one confident voxel.")

    initial_foreground_bank = _require_prototype_bank(
        initial_foreground_prototype,
        field_name="initial_foreground_prototype",
        channel_count=channel_count,
    )
    initial_background_bank = _require_prototype_bank(
        initial_background_prototype,
        field_name="initial_background_prototype",
        channel_count=channel_count,
    )
    previous_foreground_bank = _require_prototype_bank(
        previous_foreground_prototype,
        field_name="previous_foreground_prototype",
        channel_count=channel_count,
    )
    previous_background_bank = _require_prototype_bank(
        previous_background_prototype,
        field_name="previous_background_prototype",
        channel_count=channel_count,
    )

    feature_vectors = np.moveaxis(query_features, 1, -1)
    masked_features = feature_vectors[confident_mask[:, 0, :, :, :]]
    masked_foreground_weights = foreground_posterior[confident_mask].astype(
        _FLOAT_DTYPE, copy=False
    )
    masked_background_weights = background_posterior[confident_mask].astype(
        _FLOAT_DTYPE, copy=False
    )

    effective_foreground_weight = float(np.sum(masked_foreground_weights))
    effective_background_weight = float(np.sum(masked_background_weights))
    if not math.isfinite(effective_foreground_weight) or not math.isfinite(
        effective_background_weight
    ):
        raise InvalidMStepInputError("effective class weights must be finite.")
    if effective_foreground_weight <= 0.0:
        raise EmptyEffectiveForegroundUpdateError(
            "M-step foreground update has zero effective confident weight."
        )
    if effective_background_weight <= 0.0:
        raise EmptyEffectiveBackgroundUpdateError(
            "M-step background update has zero effective confident weight."
        )

    updated_foreground = (
        np.sum(
            masked_features * masked_foreground_weights[:, np.newaxis],
            axis=0,
            dtype=_FLOAT_DTYPE,
        )
        / effective_foreground_weight
    )
    updated_background = (
        np.sum(
            masked_features * masked_background_weights[:, np.newaxis],
            axis=0,
            dtype=_FLOAT_DTYPE,
        )
        / effective_background_weight
    )
    updated_foreground = np.ascontiguousarray(updated_foreground.astype(_FLOAT_DTYPE, copy=False))
    updated_background = np.ascontiguousarray(updated_background.astype(_FLOAT_DTYPE, copy=False))
    _require_prototype_vector(updated_foreground, field_name="updated_foreground")
    _require_prototype_vector(updated_background, field_name="updated_background")

    if support_foreground_reference is None or support_background_reference is None:
        raise InvalidMStepInputError(
            "support prototype references are required for deterministic ProtoEM objective terms."
        )

    objective_terms = compute_protoem_objective_terms(
        objective_weights=objective_weights,
        foreground_posterior_map=foreground_posterior,
        background_posterior_map=background_posterior,
        previous_foreground_posterior_map=previous_foreground_posterior_map,
        previous_background_posterior_map=previous_background_posterior_map,
        updated_foreground_prototype=updated_foreground,
        updated_background_prototype=updated_background,
        support_foreground_reference=support_foreground_reference,
        support_background_reference=support_background_reference,
        proximal_foreground_reference=proximal_foreground_reference,
        proximal_background_reference=proximal_background_reference,
        foreground_prior=foreground_prior,
    )

    query_hash = _array_content_sha256(query_features)
    previous_foreground_hash = _prototype_bank_content_sha256(previous_foreground_bank)
    previous_background_hash = _prototype_bank_content_sha256(previous_background_bank)
    initial_foreground_hash = _prototype_bank_content_sha256(initial_foreground_bank)
    initial_background_hash = _prototype_bank_content_sha256(initial_background_bank)
    updated_foreground_hash = _array_content_sha256(updated_foreground)
    updated_background_hash = _array_content_sha256(updated_background)
    result_identity_hash = sha256_json(
        protoem_m_step_identity_payload_from_parts(
            query_feature_content_sha256=query_hash,
            previous_foreground_bank_content_sha256=previous_foreground_hash,
            previous_background_bank_content_sha256=previous_background_hash,
            initial_foreground_bank_content_sha256=initial_foreground_hash,
            initial_background_bank_content_sha256=initial_background_hash,
            updated_foreground_content_sha256=updated_foreground_hash,
            updated_background_content_sha256=updated_background_hash,
            effective_foreground_weight=effective_foreground_weight,
            effective_background_weight=effective_background_weight,
            confident_voxel_count=int(np.count_nonzero(confident_mask)),
            objective_terms=objective_terms,
        )
    )
    return ProtoEMMStepResult(
        schema_name=PROTOEM_M_STEP_RESULT_SCHEMA_NAME,
        schema_version=PROTOEM_M_STEP_RESULT_SCHEMA_VERSION,
        result_identity_hash=result_identity_hash,
        query_feature_content_sha256=query_hash,
        previous_foreground_bank_content_sha256=previous_foreground_hash,
        previous_background_bank_content_sha256=previous_background_hash,
        initial_foreground_bank_content_sha256=initial_foreground_hash,
        initial_background_bank_content_sha256=initial_background_hash,
        updated_foreground_prototype=updated_foreground,
        updated_background_prototype=updated_background,
        updated_foreground_content_sha256=updated_foreground_hash,
        updated_background_content_sha256=updated_background_hash,
        effective_foreground_weight=effective_foreground_weight,
        effective_background_weight=effective_background_weight,
        confident_voxel_count=int(np.count_nonzero(confident_mask)),
        objective_terms=objective_terms,
    )


def protoem_e_step_identity_payload(result: ProtoEMEStepResult) -> dict[str, JsonValue]:
    """Return the canonical E-step identity payload."""

    return protoem_e_step_identity_payload_from_parts(
        query_feature_content_sha256=result.query_feature_content_sha256,
        foreground_bank_content_sha256=result.foreground_bank_content_sha256,
        background_bank_content_sha256=result.background_bank_content_sha256,
        temperature=result.temperature,
        confidence_threshold=result.confidence_threshold,
        foreground_prior=result.foreground_prior,
        foreground_posterior_content_sha256=_array_content_sha256(result.foreground_posterior_map),
        background_posterior_content_sha256=_array_content_sha256(result.background_posterior_map),
        hard_assignment_content_sha256=_array_content_sha256(result.hard_assignment_map),
        confidence_content_sha256=_array_content_sha256(
            result.confidence_mask_result.confidence_map
        ),
        confident_mask_content_sha256=_array_content_sha256(
            result.confidence_mask_result.confident_mask
        ),
        zero_norm_query_voxel_content_sha256=_array_content_sha256(
            result.zero_norm_query_voxel_mask
        ),
        confident_voxel_count=result.confidence_mask_result.confident_voxel_count,
        foreground_assignment_count=result.confidence_mask_result.foreground_assignment_count,
        background_assignment_count=result.confidence_mask_result.background_assignment_count,
        foreground_fraction=result.confidence_mask_result.foreground_fraction,
    )


def hash_protoem_e_step_result(result: ProtoEMEStepResult) -> str:
    """Return the deterministic E-step identity hash."""

    return sha256_json(protoem_e_step_identity_payload(result))


def protoem_m_step_identity_payload(result: ProtoEMMStepResult) -> dict[str, JsonValue]:
    """Return the canonical M-step identity payload."""

    return protoem_m_step_identity_payload_from_parts(
        query_feature_content_sha256=result.query_feature_content_sha256,
        previous_foreground_bank_content_sha256=result.previous_foreground_bank_content_sha256,
        previous_background_bank_content_sha256=result.previous_background_bank_content_sha256,
        initial_foreground_bank_content_sha256=result.initial_foreground_bank_content_sha256,
        initial_background_bank_content_sha256=result.initial_background_bank_content_sha256,
        updated_foreground_content_sha256=result.updated_foreground_content_sha256,
        updated_background_content_sha256=result.updated_background_content_sha256,
        effective_foreground_weight=result.effective_foreground_weight,
        effective_background_weight=result.effective_background_weight,
        confident_voxel_count=result.confident_voxel_count,
        objective_terms=result.objective_terms,
    )


def hash_protoem_m_step_result(result: ProtoEMMStepResult) -> str:
    """Return the deterministic M-step identity hash."""

    return sha256_json(protoem_m_step_identity_payload(result))


def protoem_e_step_identity_payload_from_parts(
    *,
    query_feature_content_sha256: str,
    foreground_bank_content_sha256: str,
    background_bank_content_sha256: str,
    temperature: float,
    confidence_threshold: float,
    foreground_prior: float,
    foreground_posterior_content_sha256: str,
    background_posterior_content_sha256: str,
    hard_assignment_content_sha256: str,
    confidence_content_sha256: str,
    confident_mask_content_sha256: str,
    zero_norm_query_voxel_content_sha256: str,
    confident_voxel_count: int,
    foreground_assignment_count: int,
    background_assignment_count: int,
    foreground_fraction: float,
) -> dict[str, JsonValue]:
    return {
        "schema_name": PROTOEM_E_STEP_RESULT_SCHEMA_NAME,
        "schema_version": PROTOEM_E_STEP_RESULT_SCHEMA_VERSION,
        "query_feature_content_sha256": query_feature_content_sha256,
        "foreground_bank_content_sha256": foreground_bank_content_sha256,
        "background_bank_content_sha256": background_bank_content_sha256,
        "temperature": temperature,
        "confidence_threshold": confidence_threshold,
        "foreground_prior": foreground_prior,
        "prototype_aggregation_rule": PROTOEM_MULTI_PROTOTYPE_AGGREGATION_RULE,
        "posterior_tie_policy": PROTOEM_POSTERIOR_TIE_POLICY,
        "foreground_posterior_content_sha256": foreground_posterior_content_sha256,
        "background_posterior_content_sha256": background_posterior_content_sha256,
        "hard_assignment_content_sha256": hard_assignment_content_sha256,
        "confidence_content_sha256": confidence_content_sha256,
        "confident_mask_content_sha256": confident_mask_content_sha256,
        "zero_norm_query_voxel_content_sha256": zero_norm_query_voxel_content_sha256,
        "confident_voxel_count": confident_voxel_count,
        "foreground_assignment_count": foreground_assignment_count,
        "background_assignment_count": background_assignment_count,
        "foreground_fraction": foreground_fraction,
    }


def protoem_m_step_identity_payload_from_parts(
    *,
    query_feature_content_sha256: str,
    previous_foreground_bank_content_sha256: str,
    previous_background_bank_content_sha256: str,
    initial_foreground_bank_content_sha256: str,
    initial_background_bank_content_sha256: str,
    updated_foreground_content_sha256: str,
    updated_background_content_sha256: str,
    effective_foreground_weight: float,
    effective_background_weight: float,
    confident_voxel_count: int,
    objective_terms: ProtoEMObjectiveTerms,
) -> dict[str, JsonValue]:
    return {
        "schema_name": PROTOEM_M_STEP_RESULT_SCHEMA_NAME,
        "schema_version": PROTOEM_M_STEP_RESULT_SCHEMA_VERSION,
        "query_feature_content_sha256": query_feature_content_sha256,
        "previous_foreground_bank_content_sha256": previous_foreground_bank_content_sha256,
        "previous_background_bank_content_sha256": previous_background_bank_content_sha256,
        "initial_foreground_bank_content_sha256": initial_foreground_bank_content_sha256,
        "initial_background_bank_content_sha256": initial_background_bank_content_sha256,
        "updated_foreground_content_sha256": updated_foreground_content_sha256,
        "updated_background_content_sha256": updated_background_content_sha256,
        "effective_foreground_weight": effective_foreground_weight,
        "effective_background_weight": effective_background_weight,
        "confident_voxel_count": confident_voxel_count,
        "objective_support": objective_terms.support,
        "objective_query_entropy": objective_terms.query_entropy,
        "objective_class_balance": objective_terms.class_balance,
        "objective_consistency": objective_terms.consistency,
        "objective_proximal": objective_terms.proximal,
        "objective_total": objective_terms.total,
    }


def _require_query_feature_tensor(array: np.ndarray, *, field_name: str) -> np.ndarray:
    query_features = _require_volume(
        array,
        field_name=field_name,
        expected_dtype_kinds=_FLOATING_KINDS,
    )
    if int(query_features.shape[0]) != 1:
        raise InvalidEStepInputError(f"{field_name} must have batch dimension 1.")
    return np.ascontiguousarray(query_features.astype(_FLOAT_DTYPE, copy=False))


def _require_volume(
    array: np.ndarray,
    *,
    field_name: str,
    expected_dtype_kinds: frozenset[str],
) -> np.ndarray:
    if not isinstance(array, np.ndarray):
        raise InvalidEStepInputError(f"{field_name} must be a NumPy array.")
    if array.ndim != 5 or tuple(int(item) for item in array.shape[:2])[:1] != (1,):
        raise InvalidEStepInputError(f"{field_name} must have shape [1,*,D,H,W].")
    if array.dtype.kind not in expected_dtype_kinds:
        raise InvalidEStepInputError(f"{field_name} has an invalid dtype.")
    if not np.isfinite(array).all():
        raise InvalidEStepInputError(f"{field_name} must contain only finite values.")
    return np.ascontiguousarray(array)


def _require_probability_volume(array: np.ndarray, *, field_name: str) -> np.ndarray:
    volume = _require_volume(
        array,
        field_name=field_name,
        expected_dtype_kinds=_FLOATING_KINDS,
    ).astype(_FLOAT_DTYPE, copy=False)
    if tuple(int(item) for item in volume.shape[:2]) != (1, 1):
        raise InvalidMStepInputError(f"{field_name} must have shape [1,1,D,H,W].")
    if np.any(volume < 0.0) or np.any(volume > 1.0):
        raise InvalidMStepInputError(f"{field_name} must lie in [0,1].")
    return np.ascontiguousarray(volume)


def _require_boolean_volume(array: np.ndarray, *, field_name: str) -> np.ndarray:
    if not isinstance(array, np.ndarray):
        raise InvalidMStepInputError(f"{field_name} must be a NumPy array.")
    if array.ndim != 5 or tuple(int(item) for item in array.shape[:2]) != (1, 1):
        raise InvalidMStepInputError(f"{field_name} must have shape [1,1,D,H,W].")
    if array.dtype.kind not in frozenset({"b", "i", "u"}):
        raise InvalidMStepInputError(f"{field_name} must use a boolean-compatible dtype.")
    values = set(int(item) for item in np.unique(array).tolist())
    if values - {0, 1}:
        raise InvalidMStepInputError(f"{field_name} must be binary.")
    return np.ascontiguousarray(array.astype(_MASK_DTYPE, copy=False))


def _require_binary_volume(array: np.ndarray, *, field_name: str) -> np.ndarray:
    if not isinstance(array, np.ndarray):
        raise InvalidEStepInputError(f"{field_name} must be a NumPy array.")
    if array.ndim != 5 or tuple(int(item) for item in array.shape[:2]) != (1, 1):
        raise InvalidEStepInputError(f"{field_name} must have shape [1,1,D,H,W].")
    if array.dtype.kind not in frozenset({"b", "i", "u"}):
        raise InvalidEStepInputError(f"{field_name} must use a binary-compatible dtype.")
    values = set(int(item) for item in np.unique(array).tolist())
    if values - {0, 1}:
        raise InvalidEStepInputError(f"{field_name} must be binary.")
    return np.ascontiguousarray(array.astype(_PREDICTION_DTYPE, copy=False))


def _require_prototype_bank(
    prototypes: np.ndarray,
    *,
    field_name: str,
    channel_count: int,
) -> np.ndarray:
    if not isinstance(prototypes, np.ndarray):
        raise IncompatiblePrototypeBankError(f"{field_name} must be a NumPy array.")
    if prototypes.ndim == 1:
        bank = prototypes[np.newaxis, :]
    elif prototypes.ndim == 2:
        bank = prototypes
    else:
        raise IncompatiblePrototypeBankError(f"{field_name} must have shape [C] or [K,C].")
    if bank.shape[0] <= 0:
        raise IncompatiblePrototypeBankError(f"{field_name} must contain at least one prototype.")
    if int(bank.shape[1]) != channel_count:
        raise IncompatiblePrototypeBankError(
            f"{field_name} channel dimension must equal the query feature channel count."
        )
    if bank.dtype.kind not in _NUMERIC_KINDS:
        raise IncompatiblePrototypeBankError(f"{field_name} must use a numeric dtype.")
    bank64 = np.ascontiguousarray(bank.astype(_FLOAT_DTYPE, copy=False))
    if not np.isfinite(bank64).all():
        raise IncompatiblePrototypeBankError(f"{field_name} must contain only finite values.")
    norms = np.linalg.norm(bank64, axis=1)
    if np.any(~np.isfinite(norms)):
        raise IncompatiblePrototypeBankError(f"{field_name} norms must be finite.")
    if np.any(norms <= 0.0):
        raise ZeroNormPrototypeError(f"{field_name} contains one zero-norm prototype.")
    return _canonicalize_prototype_bank(bank64)


def _reference_vector_from_bank(
    prototypes: np.ndarray,
    *,
    field_name: str,
    channel_count: int,
) -> np.ndarray:
    bank = _require_prototype_bank(
        prototypes,
        field_name=field_name,
        channel_count=channel_count,
    )
    vector = np.mean(bank, axis=0, dtype=_FLOAT_DTYPE)
    if not np.isfinite(vector).all():
        raise InvalidMStepInputError(f"{field_name} reference vector must be finite.")
    if math.isclose(float(np.linalg.norm(vector)), 0.0, rel_tol=0.0, abs_tol=0.0):
        raise ZeroNormPrototypeError(f"{field_name} reference mean must not have zero norm.")
    return np.ascontiguousarray(vector.astype(_FLOAT_DTYPE, copy=False))


def _require_prototype_vector(array: np.ndarray, *, field_name: str) -> np.ndarray:
    if not isinstance(array, np.ndarray):
        raise InvalidMStepInputError(f"{field_name} must be a NumPy array.")
    if array.ndim != 1:
        raise InvalidMStepInputError(f"{field_name} must have shape [C].")
    if array.dtype.kind not in _NUMERIC_KINDS:
        raise InvalidMStepInputError(f"{field_name} must use a numeric dtype.")
    vector = np.ascontiguousarray(array.astype(_FLOAT_DTYPE, copy=False))
    if not np.isfinite(vector).all():
        raise InvalidMStepInputError(f"{field_name} must contain only finite values.")
    if math.isclose(float(np.linalg.norm(vector)), 0.0, rel_tol=0.0, abs_tol=0.0):
        raise ZeroNormPrototypeError(f"{field_name} must not have zero norm.")
    return vector


def _class_similarity(
    query_vectors: np.ndarray,
    query_norms: np.ndarray,
    prototype_bank: np.ndarray,
) -> np.ndarray:
    safe_query_norms = np.where(query_norms > 0.0, query_norms, 1.0)
    normalized_queries = query_vectors / safe_query_norms[..., np.newaxis]
    prototype_norms = np.linalg.norm(prototype_bank, axis=1)
    normalized_bank = prototype_bank / prototype_norms[:, np.newaxis]
    class_scores = np.tensordot(normalized_queries, normalized_bank.T, axes=([-1], [0]))
    class_scores = np.clip(
        class_scores,
        -1.0 - _COSINE_DRIFT_TOLERANCE,
        1.0 + _COSINE_DRIFT_TOLERANCE,
    )
    if np.any(class_scores < -1.0 - _COSINE_DRIFT_TOLERANCE) or np.any(
        class_scores > 1.0 + _COSINE_DRIFT_TOLERANCE
    ):
        raise InvalidEStepInputError("cosine similarities exceed the tolerated numeric drift.")
    class_scores = np.clip(class_scores, -1.0, 1.0)
    aggregated = np.max(class_scores, axis=-1)
    if not np.isfinite(aggregated).all():
        raise InvalidEStepInputError("aggregated class similarities must be finite.")
    return cast(np.ndarray, aggregated.astype(_FLOAT_DTYPE, copy=False))


def _mean_binary_entropy(foreground_posterior: np.ndarray) -> float:
    posterior = foreground_posterior.astype(_FLOAT_DTYPE, copy=False)
    log_p = np.zeros_like(posterior)
    positive_mask = posterior > 0.0
    log_p[positive_mask] = np.log(posterior[positive_mask])
    complement = 1.0 - posterior
    log_complement = np.zeros_like(complement)
    complement_positive_mask = complement > 0.0
    log_complement[complement_positive_mask] = np.log(complement[complement_positive_mask])
    entropy = -(posterior * log_p + complement * log_complement)
    mean_entropy = float(np.mean(entropy))
    if not math.isfinite(mean_entropy) or mean_entropy < 0.0:
        raise NonFiniteObjectiveError("query entropy term must be finite and nonnegative.")
    return mean_entropy


def _paired_mean_squared_distance(
    *,
    first_current: np.ndarray,
    first_reference: np.ndarray,
    second_current: np.ndarray,
    second_reference: np.ndarray,
) -> float:
    first = float(np.mean((first_current - first_reference) ** 2))
    second = float(np.mean((second_current - second_reference) ** 2))
    value = 0.5 * (first + second)
    if not math.isfinite(value) or value < 0.0:
        raise NonFiniteObjectiveError(
            "paired mean-squared distance must be finite and nonnegative."
        )
    return value


def _require_posterior_pair(
    foreground_posterior: np.ndarray,
    background_posterior: np.ndarray,
    *,
    field_name: str,
) -> None:
    sums = foreground_posterior + background_posterior
    if not np.allclose(sums, 1.0, rtol=0.0, atol=_OBJECTIVE_TOLERANCE):
        raise InvalidMStepInputError(f"{field_name} must sum to one voxelwise.")


def _canonicalize_prototype_bank(bank: np.ndarray) -> np.ndarray:
    keys = [row.tobytes(order="C") for row in np.ascontiguousarray(bank)]
    order = np.argsort(np.asarray(keys, dtype=np.bytes_))
    return np.ascontiguousarray(bank[order])


def _prototype_bank_content_sha256(bank: np.ndarray) -> str:
    return _array_content_sha256(_canonicalize_prototype_bank(bank))


def _array_content_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def _require_schema_name(value: str, *, expected: str, field_name: str) -> None:
    if value != expected:
        raise ProtoEMObjectiveError(f"{field_name} must equal {expected!r}.")


def _require_schema_version(value: str, *, expected: str, field_name: str) -> None:
    if value != expected:
        raise ProtoEMObjectiveError(f"{field_name} must equal {expected!r}.")


def _require_sha256(value: str, *, field_name: str) -> None:
    if len(value) != 64 or any(char not in _SHA256_HEX for char in value):
        raise ProtoEMObjectiveError(
            f"{field_name} must be a lowercase 64-character SHA-256 hex digest."
        )


def _require_positive_finite_float(value: float, *, field_name: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ProtoEMObjectiveError(f"{field_name} must be positive and finite.")


def _require_finite_fraction_closed(value: float, *, field_name: str) -> None:
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ProtoEMObjectiveError(f"{field_name} must lie in [0,1].")


def _require_finite_fraction_open(value: float, *, field_name: str) -> None:
    if not math.isfinite(value) or value <= 0.0 or value >= 1.0:
        raise ProtoEMObjectiveError(f"{field_name} must lie in (0,1).")


__all__ = [
    "PROTOEM_CONFIDENCE_MASK_RESULT_SCHEMA_NAME",
    "PROTOEM_CONFIDENCE_MASK_RESULT_SCHEMA_VERSION",
    "PROTOEM_E_STEP_RESULT_SCHEMA_NAME",
    "PROTOEM_E_STEP_RESULT_SCHEMA_VERSION",
    "PROTOEM_M_STEP_RESULT_SCHEMA_NAME",
    "PROTOEM_M_STEP_RESULT_SCHEMA_VERSION",
    "PROTOEM_MULTI_PROTOTYPE_AGGREGATION_RULE",
    "PROTOEM_OBJECTIVE_TERMS_SCHEMA_NAME",
    "PROTOEM_OBJECTIVE_TERMS_SCHEMA_VERSION",
    "PROTOEM_POSTERIOR_TIE_POLICY",
    "EmptyEffectiveBackgroundUpdateError",
    "EmptyEffectiveForegroundUpdateError",
    "IncompatiblePrototypeBankError",
    "InvalidEStepInputError",
    "InvalidMStepInputError",
    "NoConfidentVoxelsError",
    "NonFiniteObjectiveError",
    "ObjectiveConsistencyFailureError",
    "ProtoEMConfidenceMaskResult",
    "ProtoEMEStepResult",
    "ProtoEMMStepResult",
    "ProtoEMObjectiveError",
    "ProtoEMObjectiveTerms",
    "ZeroNormPrototypeError",
    "compute_protoem_objective_terms",
    "hash_protoem_e_step_result",
    "hash_protoem_m_step_result",
    "protoem_e_step_identity_payload",
    "protoem_m_step_identity_payload",
    "run_protoem_e_step",
    "run_protoem_m_step",
]
