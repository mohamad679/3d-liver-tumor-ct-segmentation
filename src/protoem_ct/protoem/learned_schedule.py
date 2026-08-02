"""Deterministic positive-step schedule contracts for ProtoEM-CT."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

from protoem_ct.artifacts.hashing import JsonValue, sha256_json
from protoem_ct.protoem.objective import (
    PROTOEM_M_STEP_RESULT_SCHEMA_NAME,
    PROTOEM_M_STEP_RESULT_SCHEMA_VERSION,
    ProtoEMMStepResult,
    compute_protoem_objective_terms,
    protoem_m_step_identity_payload_from_parts,
)

PROTOEM_POSITIVE_STEP_SCHEDULE_SCHEMA_NAME: Final[str] = "protoem_positive_step_schedule"
PROTOEM_POSITIVE_STEP_SCHEDULE_SCHEMA_VERSION: Final[str] = "v1"
PROTOEM_POSITIVE_STEP_RECORD_SCHEMA_NAME: Final[str] = "protoem_positive_step_record"
PROTOEM_POSITIVE_STEP_RECORD_SCHEMA_VERSION: Final[str] = "v1"
PROTOEM_LEARNED_SCHEDULE_RESULT_SCHEMA_NAME: Final[str] = "protoem_learned_schedule_result"
PROTOEM_LEARNED_SCHEDULE_RESULT_SCHEMA_VERSION: Final[str] = "v1"
PROTOEM_PARAMETERIZED_POSITIVE_STEP_SCHEDULE_NAME: Final[str] = "learned_positive_step"

_FLOAT_DTYPE: Final[np.dtype[np.float64]] = np.dtype(np.float64)
_SHA256_HEX: Final[frozenset[str]] = frozenset("0123456789abcdef")
_TOLERANCE: Final[float] = 1e-12


class ProtoEMLearnedScheduleError(ValueError):
    """Base error for deterministic positive-step schedule handling."""


class InvalidPositiveStepScheduleError(ProtoEMLearnedScheduleError):
    """Raised when one positive-step schedule contract is malformed."""


class NonPositiveEffectiveStepError(ProtoEMLearnedScheduleError):
    """Raised when one effective positive step is non-positive."""


class ScheduleLengthMismatchError(ProtoEMLearnedScheduleError):
    """Raised when schedule length is incompatible with max-iteration expectations."""


class NonFiniteStepParameterError(ProtoEMLearnedScheduleError):
    """Raised when one raw or effective step parameter is non-finite."""


class LearnedScheduleOrchestrationFailure(ProtoEMLearnedScheduleError):
    """Raised when one positive-step update cannot be applied safely."""


class ScheduleIdentityMismatchError(ProtoEMLearnedScheduleError):
    """Raised when one schedule or schedule-result self-hash is invalid."""


@dataclass(frozen=True, slots=True)
class ProtoEMPositiveStepSchedule:
    """Deterministic shared-scalar positive-step schedule over max_iterations."""

    schema_name: str
    schema_version: str
    schedule_identity_hash: str
    update_schedule_name: str
    raw_step_parameters: tuple[float, ...]
    effective_positive_step_parameters: tuple[float, ...]
    epsilon: float
    max_iteration_compatibility: int

    def __post_init__(self) -> None:
        _require_schema_name(
            self.schema_name,
            expected=PROTOEM_POSITIVE_STEP_SCHEDULE_SCHEMA_NAME,
            field_name="schema_name",
            error_type=InvalidPositiveStepScheduleError,
        )
        _require_schema_version(
            self.schema_version,
            expected=PROTOEM_POSITIVE_STEP_SCHEDULE_SCHEMA_VERSION,
            field_name="schema_version",
            error_type=InvalidPositiveStepScheduleError,
        )
        _require_sha256(
            self.schedule_identity_hash,
            field_name="schedule_identity_hash",
            error_type=InvalidPositiveStepScheduleError,
        )
        if self.update_schedule_name != PROTOEM_PARAMETERIZED_POSITIVE_STEP_SCHEDULE_NAME:
            raise InvalidPositiveStepScheduleError(
                "update_schedule_name must equal 'learned_positive_step'."
            )
        _require_positive_finite_float(
            self.epsilon,
            field_name="epsilon",
            error_type=InvalidPositiveStepScheduleError,
        )
        if self.max_iteration_compatibility <= 0:
            raise InvalidPositiveStepScheduleError("max_iteration_compatibility must be positive.")
        if len(self.raw_step_parameters) != self.max_iteration_compatibility:
            raise ScheduleLengthMismatchError(
                "raw_step_parameters length must equal max_iteration_compatibility."
            )
        if len(self.effective_positive_step_parameters) != self.max_iteration_compatibility:
            raise ScheduleLengthMismatchError(
                "effective_positive_step_parameters length must equal max_iteration_compatibility."
            )
        for index, raw_value in enumerate(self.raw_step_parameters):
            if not math.isfinite(raw_value):
                raise NonFiniteStepParameterError(f"raw_step_parameters[{index}] must be finite.")
        for index, effective_value in enumerate(self.effective_positive_step_parameters):
            if not math.isfinite(effective_value):
                raise NonFiniteStepParameterError(
                    f"effective_positive_step_parameters[{index}] must be finite."
                )
            if effective_value <= 0.0:
                raise NonPositiveEffectiveStepError(
                    f"effective_positive_step_parameters[{index}] must be strictly positive."
                )
            expected_effective = effective_positive_step(
                raw_value=self.raw_step_parameters[index],
                epsilon=self.epsilon,
            )
            if not math.isclose(
                effective_value,
                expected_effective,
                rel_tol=0.0,
                abs_tol=_TOLERANCE,
            ):
                raise InvalidPositiveStepScheduleError(
                    "effective_positive_step_parameters must equal "
                    "softplus(raw_step_parameters) + epsilon elementwise."
                )
        if self.schedule_identity_hash != hash_protoem_positive_step_schedule(self):
            raise ScheduleIdentityMismatchError(
                "schedule_identity_hash does not match deterministic schedule content."
            )


@dataclass(frozen=True, slots=True)
class ProtoEMPositiveStepRecord:
    """One deterministic learned-schedule step actually applied at one iteration."""

    schema_name: str
    schema_version: str
    record_identity_hash: str
    iteration_index: int
    alpha_raw: float
    alpha_effective: float
    source_prototype_state_identity_hash: str
    m_step_target_result_identity_hash: str
    updated_m_step_result_identity_hash: str

    def __post_init__(self) -> None:
        _require_schema_name(
            self.schema_name,
            expected=PROTOEM_POSITIVE_STEP_RECORD_SCHEMA_NAME,
            field_name="schema_name",
            error_type=LearnedScheduleOrchestrationFailure,
        )
        _require_schema_version(
            self.schema_version,
            expected=PROTOEM_POSITIVE_STEP_RECORD_SCHEMA_VERSION,
            field_name="schema_version",
            error_type=LearnedScheduleOrchestrationFailure,
        )
        _require_sha256(
            self.record_identity_hash,
            field_name="record_identity_hash",
            error_type=LearnedScheduleOrchestrationFailure,
        )
        if self.iteration_index < 0:
            raise LearnedScheduleOrchestrationFailure("iteration_index must be nonnegative.")
        if not math.isfinite(self.alpha_raw):
            raise NonFiniteStepParameterError("alpha_raw must be finite.")
        _require_positive_finite_float(
            self.alpha_effective,
            field_name="alpha_effective",
            error_type=NonPositiveEffectiveStepError,
        )
        for field_name in (
            "source_prototype_state_identity_hash",
            "m_step_target_result_identity_hash",
            "updated_m_step_result_identity_hash",
        ):
            _require_sha256(
                getattr(self, field_name),
                field_name=field_name,
                error_type=LearnedScheduleOrchestrationFailure,
            )
        if self.record_identity_hash != hash_protoem_positive_step_record(self):
            raise ScheduleIdentityMismatchError(
                "record_identity_hash does not match deterministic positive-step content."
            )


@dataclass(frozen=True, slots=True)
class ProtoEMLearnedScheduleResult:
    """Deterministic record of the positive-step schedule actually used in one run."""

    schema_name: str
    schema_version: str
    learned_schedule_result_identity_hash: str
    schedule_identity_hash: str
    update_schedule_name: str
    completed_iteration_count: int
    records: tuple[ProtoEMPositiveStepRecord, ...]

    def __post_init__(self) -> None:
        _require_schema_name(
            self.schema_name,
            expected=PROTOEM_LEARNED_SCHEDULE_RESULT_SCHEMA_NAME,
            field_name="schema_name",
            error_type=LearnedScheduleOrchestrationFailure,
        )
        _require_schema_version(
            self.schema_version,
            expected=PROTOEM_LEARNED_SCHEDULE_RESULT_SCHEMA_VERSION,
            field_name="schema_version",
            error_type=LearnedScheduleOrchestrationFailure,
        )
        _require_sha256(
            self.learned_schedule_result_identity_hash,
            field_name="learned_schedule_result_identity_hash",
            error_type=LearnedScheduleOrchestrationFailure,
        )
        _require_sha256(
            self.schedule_identity_hash,
            field_name="schedule_identity_hash",
            error_type=LearnedScheduleOrchestrationFailure,
        )
        if self.update_schedule_name != PROTOEM_PARAMETERIZED_POSITIVE_STEP_SCHEDULE_NAME:
            raise LearnedScheduleOrchestrationFailure(
                "update_schedule_name must equal 'learned_positive_step'."
            )
        if self.completed_iteration_count < 0:
            raise LearnedScheduleOrchestrationFailure(
                "completed_iteration_count must be nonnegative."
            )
        if len(self.records) != self.completed_iteration_count:
            raise LearnedScheduleOrchestrationFailure(
                "completed_iteration_count must equal len(records)."
            )
        indices = tuple(record.iteration_index for record in self.records)
        if indices != tuple(range(len(self.records))):
            raise LearnedScheduleOrchestrationFailure(
                "learned-schedule records must use contiguous zero-based iteration indices."
            )
        if self.learned_schedule_result_identity_hash != hash_protoem_learned_schedule_result(self):
            raise ScheduleIdentityMismatchError(
                "learned_schedule_result_identity_hash does not match deterministic "
                "schedule-result content."
            )


def effective_positive_step(*, raw_value: float, epsilon: float) -> float:
    """Convert one raw step parameter to one strictly positive effective step."""

    _require_positive_finite_float(
        epsilon,
        field_name="epsilon",
        error_type=InvalidPositiveStepScheduleError,
    )
    if not math.isfinite(raw_value):
        raise NonFiniteStepParameterError("raw_value must be finite.")
    effective = float(
        np.logaddexp(np.float64(raw_value), np.float64(0.0)).astype(_FLOAT_DTYPE) + epsilon
    )
    if not math.isfinite(effective):
        raise NonFiniteStepParameterError("effective positive step must be finite.")
    if effective <= 0.0:
        raise NonPositiveEffectiveStepError("effective positive step must be strictly positive.")
    return effective


def build_protoem_positive_step_schedule(
    *,
    raw_step_parameters: Sequence[float],
    epsilon: float,
    max_iteration_compatibility: int,
) -> ProtoEMPositiveStepSchedule:
    """Build one deterministic shared-scalar positive-step schedule."""

    if max_iteration_compatibility <= 0:
        raise InvalidPositiveStepScheduleError("max_iteration_compatibility must be positive.")
    raw_parameters = tuple(float(value) for value in raw_step_parameters)
    if len(raw_parameters) != max_iteration_compatibility:
        raise ScheduleLengthMismatchError(
            "raw_step_parameters length must equal max_iteration_compatibility."
        )
    effective_parameters = tuple(
        effective_positive_step(raw_value=value, epsilon=epsilon) for value in raw_parameters
    )
    payload = {
        "schema_name": PROTOEM_POSITIVE_STEP_SCHEDULE_SCHEMA_NAME,
        "schema_version": PROTOEM_POSITIVE_STEP_SCHEDULE_SCHEMA_VERSION,
        "update_schedule_name": PROTOEM_PARAMETERIZED_POSITIVE_STEP_SCHEDULE_NAME,
        "raw_step_parameters": list(raw_parameters),
        "effective_positive_step_parameters": list(effective_parameters),
        "epsilon": epsilon,
        "max_iteration_compatibility": max_iteration_compatibility,
    }
    return ProtoEMPositiveStepSchedule(
        schema_name=PROTOEM_POSITIVE_STEP_SCHEDULE_SCHEMA_NAME,
        schema_version=PROTOEM_POSITIVE_STEP_SCHEDULE_SCHEMA_VERSION,
        schedule_identity_hash=sha256_json(payload),
        update_schedule_name=PROTOEM_PARAMETERIZED_POSITIVE_STEP_SCHEDULE_NAME,
        raw_step_parameters=raw_parameters,
        effective_positive_step_parameters=effective_parameters,
        epsilon=epsilon,
        max_iteration_compatibility=max_iteration_compatibility,
    )


def build_protoem_learned_schedule_result(
    *,
    schedule: ProtoEMPositiveStepSchedule,
    records: Sequence[ProtoEMPositiveStepRecord],
) -> ProtoEMLearnedScheduleResult:
    """Build one deterministic learned-schedule execution result."""

    ordered_records = tuple(records)
    payload = {
        "schema_name": PROTOEM_LEARNED_SCHEDULE_RESULT_SCHEMA_NAME,
        "schema_version": PROTOEM_LEARNED_SCHEDULE_RESULT_SCHEMA_VERSION,
        "schedule_identity_hash": schedule.schedule_identity_hash,
        "update_schedule_name": schedule.update_schedule_name,
        "completed_iteration_count": len(ordered_records),
        "records": [
            protoem_positive_step_record_identity_payload(item) for item in ordered_records
        ],
    }
    return ProtoEMLearnedScheduleResult(
        schema_name=PROTOEM_LEARNED_SCHEDULE_RESULT_SCHEMA_NAME,
        schema_version=PROTOEM_LEARNED_SCHEDULE_RESULT_SCHEMA_VERSION,
        learned_schedule_result_identity_hash=sha256_json(payload),
        schedule_identity_hash=schedule.schedule_identity_hash,
        update_schedule_name=schedule.update_schedule_name,
        completed_iteration_count=len(ordered_records),
        records=ordered_records,
    )


def apply_positive_step_to_m_step_result(
    *,
    iteration_index: int,
    schedule: ProtoEMPositiveStepSchedule,
    source_prototype_state_identity_hash: str,
    current_foreground_prototype: np.ndarray,
    current_background_prototype: np.ndarray,
    target_m_step_result: ProtoEMMStepResult,
    foreground_posterior_map: np.ndarray,
    background_posterior_map: np.ndarray,
    previous_foreground_posterior_map: np.ndarray,
    previous_background_posterior_map: np.ndarray,
    support_foreground_reference: np.ndarray,
    support_background_reference: np.ndarray,
    proximal_foreground_reference: np.ndarray,
    proximal_background_reference: np.ndarray,
    foreground_prior: float,
) -> tuple[ProtoEMMStepResult, ProtoEMPositiveStepRecord]:
    """Apply one shared positive interpolation step toward one standard M-step target."""

    if target_m_step_result.schema_name != PROTOEM_M_STEP_RESULT_SCHEMA_NAME:
        raise LearnedScheduleOrchestrationFailure("unexpected target M-step schema_name.")
    if target_m_step_result.schema_version != PROTOEM_M_STEP_RESULT_SCHEMA_VERSION:
        raise LearnedScheduleOrchestrationFailure("unexpected target M-step schema_version.")
    if iteration_index < 0:
        raise LearnedScheduleOrchestrationFailure("iteration_index must be nonnegative.")
    if iteration_index >= schedule.max_iteration_compatibility:
        raise ScheduleLengthMismatchError(
            "iteration_index exceeds schedule.max_iteration_compatibility."
        )

    alpha_raw = schedule.raw_step_parameters[iteration_index]
    alpha_effective = schedule.effective_positive_step_parameters[iteration_index]
    current_foreground = _require_vector(
        current_foreground_prototype,
        field_name="current_foreground_prototype",
    )
    current_background = _require_vector(
        current_background_prototype,
        field_name="current_background_prototype",
    )
    target_foreground = _require_vector(
        target_m_step_result.updated_foreground_prototype,
        field_name="target_m_step_result.updated_foreground_prototype",
    )
    target_background = _require_vector(
        target_m_step_result.updated_background_prototype,
        field_name="target_m_step_result.updated_background_prototype",
    )
    if current_foreground.shape != target_foreground.shape:
        raise LearnedScheduleOrchestrationFailure(
            "foreground current/target prototypes must share the same shape."
        )
    if current_background.shape != target_background.shape:
        raise LearnedScheduleOrchestrationFailure(
            "background current/target prototypes must share the same shape."
        )

    updated_foreground = np.ascontiguousarray(
        current_foreground + alpha_effective * (target_foreground - current_foreground),
        dtype=_FLOAT_DTYPE,
    )
    updated_background = np.ascontiguousarray(
        current_background + alpha_effective * (target_background - current_background),
        dtype=_FLOAT_DTYPE,
    )
    if not np.isfinite(updated_foreground).all() or not np.isfinite(updated_background).all():
        raise LearnedScheduleOrchestrationFailure(
            "positive-step prototype updates must remain finite."
        )

    objective_terms = compute_protoem_objective_terms(
        objective_weights=target_m_step_result.objective_terms.weights,
        foreground_posterior_map=foreground_posterior_map,
        background_posterior_map=background_posterior_map,
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
    updated_foreground_hash = _array_content_sha256(updated_foreground)
    updated_background_hash = _array_content_sha256(updated_background)
    m_step_payload = protoem_m_step_identity_payload_from_parts(
        query_feature_content_sha256=target_m_step_result.query_feature_content_sha256,
        previous_foreground_bank_content_sha256=(
            target_m_step_result.previous_foreground_bank_content_sha256
        ),
        previous_background_bank_content_sha256=(
            target_m_step_result.previous_background_bank_content_sha256
        ),
        initial_foreground_bank_content_sha256=(
            target_m_step_result.initial_foreground_bank_content_sha256
        ),
        initial_background_bank_content_sha256=(
            target_m_step_result.initial_background_bank_content_sha256
        ),
        updated_foreground_content_sha256=updated_foreground_hash,
        updated_background_content_sha256=updated_background_hash,
        effective_foreground_weight=target_m_step_result.effective_foreground_weight,
        effective_background_weight=target_m_step_result.effective_background_weight,
        confident_voxel_count=target_m_step_result.confident_voxel_count,
        objective_terms=objective_terms,
    )
    updated_m_step_result = ProtoEMMStepResult(
        schema_name=PROTOEM_M_STEP_RESULT_SCHEMA_NAME,
        schema_version=PROTOEM_M_STEP_RESULT_SCHEMA_VERSION,
        result_identity_hash=sha256_json(m_step_payload),
        query_feature_content_sha256=target_m_step_result.query_feature_content_sha256,
        previous_foreground_bank_content_sha256=(
            target_m_step_result.previous_foreground_bank_content_sha256
        ),
        previous_background_bank_content_sha256=(
            target_m_step_result.previous_background_bank_content_sha256
        ),
        initial_foreground_bank_content_sha256=(
            target_m_step_result.initial_foreground_bank_content_sha256
        ),
        initial_background_bank_content_sha256=(
            target_m_step_result.initial_background_bank_content_sha256
        ),
        updated_foreground_prototype=updated_foreground,
        updated_background_prototype=updated_background,
        updated_foreground_content_sha256=updated_foreground_hash,
        updated_background_content_sha256=updated_background_hash,
        effective_foreground_weight=target_m_step_result.effective_foreground_weight,
        effective_background_weight=target_m_step_result.effective_background_weight,
        confident_voxel_count=target_m_step_result.confident_voxel_count,
        objective_terms=objective_terms,
    )
    record_payload = {
        "schema_name": PROTOEM_POSITIVE_STEP_RECORD_SCHEMA_NAME,
        "schema_version": PROTOEM_POSITIVE_STEP_RECORD_SCHEMA_VERSION,
        "iteration_index": iteration_index,
        "alpha_raw": alpha_raw,
        "alpha_effective": alpha_effective,
        "source_prototype_state_identity_hash": source_prototype_state_identity_hash,
        "m_step_target_result_identity_hash": target_m_step_result.result_identity_hash,
        "updated_m_step_result_identity_hash": updated_m_step_result.result_identity_hash,
    }
    record = ProtoEMPositiveStepRecord(
        schema_name=PROTOEM_POSITIVE_STEP_RECORD_SCHEMA_NAME,
        schema_version=PROTOEM_POSITIVE_STEP_RECORD_SCHEMA_VERSION,
        record_identity_hash=sha256_json(record_payload),
        iteration_index=iteration_index,
        alpha_raw=alpha_raw,
        alpha_effective=alpha_effective,
        source_prototype_state_identity_hash=source_prototype_state_identity_hash,
        m_step_target_result_identity_hash=target_m_step_result.result_identity_hash,
        updated_m_step_result_identity_hash=updated_m_step_result.result_identity_hash,
    )
    return updated_m_step_result, record


def protoem_positive_step_schedule_identity_payload(
    schedule: ProtoEMPositiveStepSchedule,
) -> dict[str, JsonValue]:
    """Return the exact identity payload for one positive-step schedule."""

    return {
        "schema_name": schedule.schema_name,
        "schema_version": schedule.schema_version,
        "update_schedule_name": schedule.update_schedule_name,
        "raw_step_parameters": list(schedule.raw_step_parameters),
        "effective_positive_step_parameters": list(schedule.effective_positive_step_parameters),
        "epsilon": schedule.epsilon,
        "max_iteration_compatibility": schedule.max_iteration_compatibility,
    }


def protoem_positive_step_record_identity_payload(
    record: ProtoEMPositiveStepRecord,
) -> dict[str, JsonValue]:
    """Return the exact identity payload for one applied positive-step record."""

    return {
        "schema_name": record.schema_name,
        "schema_version": record.schema_version,
        "iteration_index": record.iteration_index,
        "alpha_raw": record.alpha_raw,
        "alpha_effective": record.alpha_effective,
        "source_prototype_state_identity_hash": record.source_prototype_state_identity_hash,
        "m_step_target_result_identity_hash": record.m_step_target_result_identity_hash,
        "updated_m_step_result_identity_hash": record.updated_m_step_result_identity_hash,
    }


def protoem_learned_schedule_result_identity_payload(
    result: ProtoEMLearnedScheduleResult,
) -> dict[str, JsonValue]:
    """Return the exact identity payload for one learned-schedule result."""

    return {
        "schema_name": result.schema_name,
        "schema_version": result.schema_version,
        "schedule_identity_hash": result.schedule_identity_hash,
        "update_schedule_name": result.update_schedule_name,
        "completed_iteration_count": result.completed_iteration_count,
        "records": [protoem_positive_step_record_identity_payload(item) for item in result.records],
    }


def hash_protoem_positive_step_schedule(schedule: ProtoEMPositiveStepSchedule) -> str:
    """Return the deterministic positive-step schedule hash."""

    return sha256_json(protoem_positive_step_schedule_identity_payload(schedule))


def hash_protoem_positive_step_record(record: ProtoEMPositiveStepRecord) -> str:
    """Return the deterministic applied positive-step record hash."""

    return sha256_json(protoem_positive_step_record_identity_payload(record))


def hash_protoem_learned_schedule_result(result: ProtoEMLearnedScheduleResult) -> str:
    """Return the deterministic learned-schedule result hash."""

    return sha256_json(protoem_learned_schedule_result_identity_payload(result))


def query_label_not_part_of_learned_schedule_api() -> bool:
    """Return whether query labels are absent from the public schedule API."""

    return True


def _array_content_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _require_vector(value: np.ndarray, *, field_name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1:
        raise LearnedScheduleOrchestrationFailure(f"{field_name} must be one-dimensional.")
    if array.dtype.kind not in {"f", "i", "u"}:
        raise LearnedScheduleOrchestrationFailure(f"{field_name} must be numeric.")
    copied = np.ascontiguousarray(array.astype(_FLOAT_DTYPE, copy=False))
    if not np.isfinite(copied).all():
        raise LearnedScheduleOrchestrationFailure(f"{field_name} must be finite.")
    return copied


def _require_positive_finite_float(
    value: float,
    *,
    field_name: str,
    error_type: type[ProtoEMLearnedScheduleError],
) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise error_type(f"{field_name} must be finite and strictly positive.")


def _require_schema_name(
    value: str,
    *,
    expected: str,
    field_name: str,
    error_type: type[ProtoEMLearnedScheduleError],
) -> None:
    if value != expected:
        raise error_type(f"{field_name} must equal {expected!r}.")


def _require_schema_version(
    value: str,
    *,
    expected: str,
    field_name: str,
    error_type: type[ProtoEMLearnedScheduleError],
) -> None:
    if value != expected:
        raise error_type(f"{field_name} must equal {expected!r}.")


def _require_sha256(
    value: str,
    *,
    field_name: str,
    error_type: type[ProtoEMLearnedScheduleError],
) -> None:
    if len(value) != 64 or any(character not in _SHA256_HEX for character in value):
        raise error_type(f"{field_name} must be one lowercase hexadecimal SHA-256 digest.")


__all__ = [
    "InvalidPositiveStepScheduleError",
    "LearnedScheduleOrchestrationFailure",
    "NonFiniteStepParameterError",
    "NonPositiveEffectiveStepError",
    "PROTOEM_LEARNED_SCHEDULE_RESULT_SCHEMA_NAME",
    "PROTOEM_LEARNED_SCHEDULE_RESULT_SCHEMA_VERSION",
    "PROTOEM_PARAMETERIZED_POSITIVE_STEP_SCHEDULE_NAME",
    "PROTOEM_POSITIVE_STEP_RECORD_SCHEMA_NAME",
    "PROTOEM_POSITIVE_STEP_RECORD_SCHEMA_VERSION",
    "PROTOEM_POSITIVE_STEP_SCHEDULE_SCHEMA_NAME",
    "PROTOEM_POSITIVE_STEP_SCHEDULE_SCHEMA_VERSION",
    "ProtoEMLearnedScheduleError",
    "ProtoEMLearnedScheduleResult",
    "ProtoEMPositiveStepRecord",
    "ProtoEMPositiveStepSchedule",
    "ScheduleIdentityMismatchError",
    "ScheduleLengthMismatchError",
    "apply_positive_step_to_m_step_result",
    "build_protoem_learned_schedule_result",
    "build_protoem_positive_step_schedule",
    "effective_positive_step",
    "hash_protoem_learned_schedule_result",
    "hash_protoem_positive_step_record",
    "hash_protoem_positive_step_schedule",
    "protoem_learned_schedule_result_identity_payload",
    "protoem_positive_step_record_identity_payload",
    "protoem_positive_step_schedule_identity_payload",
    "query_label_not_part_of_learned_schedule_api",
]
