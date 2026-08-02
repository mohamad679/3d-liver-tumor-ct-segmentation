"""Deterministic ProtoEM-CT stopping and collapse decision contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

from protoem_ct.artifacts.hashing import JsonValue, sha256_json
from protoem_ct.protoem.artifacts import (
    PROTOEM_COLLAPSE_RECORD_SCHEMA_NAME,
    PROTOEM_COLLAPSE_RECORD_SCHEMA_VERSION,
    PROTOEM_STOPPING_RECORD_SCHEMA_NAME,
    PROTOEM_STOPPING_RECORD_SCHEMA_VERSION,
    SUPPORTED_PROTOEM_COLLAPSE_TYPES,
    SUPPORTED_PROTOEM_STOP_REASONS,
    ProtoEMCollapseRecord,
    ProtoEMConfig,
    ProtoEMIterationRecord,
    ProtoEMStoppingRecord,
    hash_protoem_collapse_record,
    hash_protoem_stopping_record,
)
from protoem_ct.protoem.objective import ProtoEMConfidenceMaskResult
from protoem_ct.protoem.state import ProtoEMTransductiveState

PROTOEM_STOPPING_DECISION_SCHEMA_NAME: Final[str] = "protoem_stopping_decision"
PROTOEM_STOPPING_DECISION_SCHEMA_VERSION: Final[str] = "v1"
PROTOEM_COLLAPSE_DECISION_SCHEMA_NAME: Final[str] = "protoem_collapse_decision"
PROTOEM_COLLAPSE_DECISION_SCHEMA_VERSION: Final[str] = "v1"

_SHA256_HEX: Final[frozenset[str]] = frozenset("0123456789abcdef")
_OBJECTIVE_TOLERANCE: Final[float] = 1e-12
_NO_SAFEGUARD_LOWER: Final[float] = 0.0
_NO_SAFEGUARD_UPPER: Final[float] = 1.0


class ProtoEMOrchestrationError(ValueError):
    """Base error for deterministic ProtoEM orchestration and stopping logic."""


class OrchestrationInputError(ProtoEMOrchestrationError):
    """Raised when one orchestration input contract is unsupported or malformed."""


class StoppingConsistencyError(ProtoEMOrchestrationError):
    """Raised when one stopping decision is inconsistent with artifacts or state."""


class CollapseConsistencyError(ProtoEMOrchestrationError):
    """Raised when one collapse decision is inconsistent with observed assignments."""


class NumericalOrchestrationFailure(ProtoEMOrchestrationError):
    """Raised when one numerical failure must abort orchestration explicitly."""


class IterationConsistencyError(ProtoEMOrchestrationError):
    """Raised when one deterministic iteration sequence violates orchestration rules."""


@dataclass(frozen=True, slots=True)
class ProtoEMCollapseDecision:
    """Deterministic hard-stop decision from one E-step confidence summary."""

    schema_name: str
    schema_version: str
    decision_identity_hash: str
    iteration_index: int
    hard_stop: bool
    stop_reason: str | None
    collapse_type: str | None
    confident_voxel_count: int
    foreground_assignment_count: int
    background_assignment_count: int
    foreground_fraction: float
    configured_lower_safeguard: float
    configured_upper_safeguard: float
    detected: bool
    collapse_record: ProtoEMCollapseRecord | None

    def __post_init__(self) -> None:
        _require_schema_name(
            self.schema_name,
            expected=PROTOEM_COLLAPSE_DECISION_SCHEMA_NAME,
            field_name="schema_name",
            error_type=CollapseConsistencyError,
        )
        _require_schema_version(
            self.schema_version,
            expected=PROTOEM_COLLAPSE_DECISION_SCHEMA_VERSION,
            field_name="schema_version",
            error_type=CollapseConsistencyError,
        )
        _require_sha256(
            self.decision_identity_hash,
            field_name="decision_identity_hash",
            error_type=CollapseConsistencyError,
        )
        _require_nonnegative_int(
            self.iteration_index,
            field_name="iteration_index",
            error_type=CollapseConsistencyError,
        )
        _require_nonnegative_int(
            self.confident_voxel_count,
            field_name="confident_voxel_count",
            error_type=CollapseConsistencyError,
        )
        _require_nonnegative_int(
            self.foreground_assignment_count,
            field_name="foreground_assignment_count",
            error_type=CollapseConsistencyError,
        )
        _require_nonnegative_int(
            self.background_assignment_count,
            field_name="background_assignment_count",
            error_type=CollapseConsistencyError,
        )
        if (
            self.foreground_assignment_count + self.background_assignment_count
            != self.confident_voxel_count
        ):
            raise CollapseConsistencyError(
                "foreground/background assignment counts must sum to confident_voxel_count."
            )
        _require_fraction_closed(
            self.foreground_fraction,
            field_name="foreground_fraction",
            error_type=CollapseConsistencyError,
        )
        _require_fraction_closed(
            self.configured_lower_safeguard,
            field_name="configured_lower_safeguard",
            error_type=CollapseConsistencyError,
        )
        _require_fraction_closed(
            self.configured_upper_safeguard,
            field_name="configured_upper_safeguard",
            error_type=CollapseConsistencyError,
        )
        if self.configured_lower_safeguard >= self.configured_upper_safeguard:
            raise CollapseConsistencyError(
                "configured_lower_safeguard must be less than configured_upper_safeguard."
            )
        if self.stop_reason is not None and self.stop_reason not in SUPPORTED_PROTOEM_STOP_REASONS:
            raise CollapseConsistencyError(
                f"stop_reason must be one of {sorted(SUPPORTED_PROTOEM_STOP_REASONS)!r}."
            )
        if (
            self.collapse_type is not None
            and self.collapse_type not in SUPPORTED_PROTOEM_COLLAPSE_TYPES
        ):
            raise CollapseConsistencyError(
                f"collapse_type must be one of {sorted(SUPPORTED_PROTOEM_COLLAPSE_TYPES)!r}."
            )
        expected_fraction = (
            self.foreground_assignment_count / self.confident_voxel_count
            if self.confident_voxel_count > 0
            else 0.0
        )
        if not math.isclose(
            self.foreground_fraction,
            expected_fraction,
            rel_tol=0.0,
            abs_tol=_OBJECTIVE_TOLERANCE,
        ):
            raise CollapseConsistencyError(
                "foreground_fraction must match the confident foreground assignment fraction."
            )
        if self.hard_stop and self.stop_reason is None:
            raise CollapseConsistencyError("hard_stop requires one explicit stop_reason.")
        if not self.hard_stop and self.stop_reason is not None:
            raise CollapseConsistencyError("non-stop collapse decisions must not set stop_reason.")
        if self.detected:
            if self.collapse_type is None:
                raise CollapseConsistencyError("detected collapse requires collapse_type.")
            if self.stop_reason != self.collapse_type:
                raise CollapseConsistencyError(
                    "detected collapse stop_reason must equal collapse_type."
                )
            if self.collapse_record is None:
                raise CollapseConsistencyError("detected collapse requires collapse_record.")
        else:
            if self.collapse_type is not None:
                raise CollapseConsistencyError(
                    "undetected collapse decisions must not set collapse_type."
                )
            if self.collapse_record is not None:
                raise CollapseConsistencyError(
                    "undetected collapse decisions must not carry collapse_record."
                )
        if self.stop_reason == "no_confident_voxels":
            if self.detected:
                raise CollapseConsistencyError(
                    "no_confident_voxels is not a foreground/background collapse record."
                )
            if self.confident_voxel_count != 0:
                raise CollapseConsistencyError(
                    "no_confident_voxels requires confident_voxel_count == 0."
                )
        if self.collapse_record is not None:
            if self.collapse_record.iteration_index != self.iteration_index:
                raise CollapseConsistencyError(
                    "collapse_record.iteration_index must match decision iteration_index."
                )
            if self.collapse_record.foreground_count != self.foreground_assignment_count:
                raise CollapseConsistencyError(
                    "collapse_record.foreground_count must match the decision counts."
                )
            if self.collapse_record.background_count != self.background_assignment_count:
                raise CollapseConsistencyError(
                    "collapse_record.background_count must match the decision counts."
                )
            if self.collapse_record.confident_voxel_count != self.confident_voxel_count:
                raise CollapseConsistencyError(
                    "collapse_record.confident_voxel_count must match the decision count."
                )
            if not math.isclose(
                self.collapse_record.foreground_fraction,
                self.foreground_fraction,
                rel_tol=0.0,
                abs_tol=_OBJECTIVE_TOLERANCE,
            ):
                raise CollapseConsistencyError(
                    "collapse_record.foreground_fraction must match the decision fraction."
                )
            if (
                self.collapse_record.configured_lower_safeguard != self.configured_lower_safeguard
                or self.collapse_record.configured_upper_safeguard
                != self.configured_upper_safeguard
            ):
                raise CollapseConsistencyError(
                    "collapse_record safeguards must match the decision safeguards."
                )
        if self.decision_identity_hash != hash_protoem_collapse_decision(self):
            raise CollapseConsistencyError(
                "decision_identity_hash does not match deterministic collapse-decision content."
            )


@dataclass(frozen=True, slots=True)
class ProtoEMStoppingDecision:
    """Deterministic stopping outcome derived from completed iterations and hard failures."""

    schema_name: str
    schema_version: str
    decision_identity_hash: str
    should_stop: bool
    stop_reason: str | None
    completed_iteration_count: int
    final_iteration_index: int | None
    converged: bool
    failed: bool
    failure_code: str | None
    failure_message: str | None
    stopping_record: ProtoEMStoppingRecord | None

    def __post_init__(self) -> None:
        _require_schema_name(
            self.schema_name,
            expected=PROTOEM_STOPPING_DECISION_SCHEMA_NAME,
            field_name="schema_name",
            error_type=StoppingConsistencyError,
        )
        _require_schema_version(
            self.schema_version,
            expected=PROTOEM_STOPPING_DECISION_SCHEMA_VERSION,
            field_name="schema_version",
            error_type=StoppingConsistencyError,
        )
        _require_sha256(
            self.decision_identity_hash,
            field_name="decision_identity_hash",
            error_type=StoppingConsistencyError,
        )
        _require_nonnegative_int(
            self.completed_iteration_count,
            field_name="completed_iteration_count",
            error_type=StoppingConsistencyError,
        )
        if self.final_iteration_index is None:
            if self.completed_iteration_count != 0:
                raise StoppingConsistencyError(
                    "final_iteration_index may be null only when no iterations completed."
                )
        else:
            _require_nonnegative_int(
                self.final_iteration_index,
                field_name="final_iteration_index",
                error_type=StoppingConsistencyError,
            )
            if self.final_iteration_index + 1 != self.completed_iteration_count:
                raise StoppingConsistencyError(
                    "final_iteration_index must equal completed_iteration_count - 1."
                )
        if self.stop_reason is not None and self.stop_reason not in SUPPORTED_PROTOEM_STOP_REASONS:
            raise StoppingConsistencyError(
                f"stop_reason must be one of {sorted(SUPPORTED_PROTOEM_STOP_REASONS)!r}."
            )
        if self.should_stop and self.stop_reason is None:
            raise StoppingConsistencyError("should_stop requires one explicit stop_reason.")
        if not self.should_stop and self.stop_reason is not None:
            raise StoppingConsistencyError("non-stop decisions must not set stop_reason.")
        _require_failure_consistency(
            failed=self.failed,
            failure_code=self.failure_code,
            failure_message=self.failure_message,
            field_prefix="ProtoEMStoppingDecision",
        )
        if self.converged and self.failed:
            raise StoppingConsistencyError("converged and failed cannot both be true.")
        if self.converged and self.stop_reason != "tolerance_reached":
            raise StoppingConsistencyError(
                "converged=true requires stop_reason='tolerance_reached'."
            )
        if self.stop_reason == "tolerance_reached" and not self.converged:
            raise StoppingConsistencyError(
                "stop_reason='tolerance_reached' requires converged=true."
            )
        if self.stop_reason == "max_iterations" and self.failed:
            raise StoppingConsistencyError(
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
            and self.should_stop
            and not self.failed
        ):
            raise StoppingConsistencyError("failure-like stop reasons must set failed=true.")
        if self.stopping_record is None:
            if self.completed_iteration_count != 0 and self.should_stop:
                raise StoppingConsistencyError(
                    "completed stopping decisions require a ProtoEMStoppingRecord."
                )
        else:
            if self.completed_iteration_count == 0:
                raise StoppingConsistencyError(
                    "ProtoEMStoppingRecord cannot represent zero completed iterations."
                )
            if self.stop_reason != self.stopping_record.stop_reason:
                raise StoppingConsistencyError(
                    "stopping_record.stop_reason must match the stopping decision."
                )
            if self.converged != self.stopping_record.converged:
                raise StoppingConsistencyError(
                    "stopping_record.converged must match the stopping decision."
                )
            if self.failed != self.stopping_record.failed:
                raise StoppingConsistencyError(
                    "stopping_record.failed must match the stopping decision."
                )
            if self.failure_code != self.stopping_record.failure_code:
                raise StoppingConsistencyError(
                    "stopping_record.failure_code must match the stopping decision."
                )
            if self.failure_message != self.stopping_record.failure_message:
                raise StoppingConsistencyError(
                    "stopping_record.failure_message must match the stopping decision."
                )
        if self.decision_identity_hash != hash_protoem_stopping_decision(self):
            raise StoppingConsistencyError(
                "decision_identity_hash does not match deterministic stopping-decision content."
            )


def evaluate_protoem_collapse(
    *,
    iteration_index: int,
    confidence_mask_result: ProtoEMConfidenceMaskResult,
) -> ProtoEMCollapseDecision:
    """Evaluate hard pre-M-step collapse and no-confident-voxel conditions."""

    confident_voxel_count = confidence_mask_result.confident_voxel_count
    foreground_count = confidence_mask_result.foreground_assignment_count
    background_count = confidence_mask_result.background_assignment_count
    foreground_fraction = confidence_mask_result.foreground_fraction
    lower = _NO_SAFEGUARD_LOWER
    upper = _NO_SAFEGUARD_UPPER

    if confident_voxel_count == 0:
        return _build_collapse_decision(
            iteration_index=iteration_index,
            hard_stop=True,
            stop_reason="no_confident_voxels",
            collapse_type=None,
            confident_voxel_count=0,
            foreground_assignment_count=0,
            background_assignment_count=0,
            foreground_fraction=0.0,
            configured_lower_safeguard=lower,
            configured_upper_safeguard=upper,
            detected=False,
            collapse_record=None,
        )
    if foreground_count == 0:
        collapse_record = _build_collapse_record(
            collapse_type="foreground_collapse",
            iteration_index=iteration_index,
            foreground_count=foreground_count,
            background_count=background_count,
            confident_voxel_count=confident_voxel_count,
            foreground_fraction=foreground_fraction,
            configured_lower_safeguard=lower,
            configured_upper_safeguard=upper,
        )
        return _build_collapse_decision(
            iteration_index=iteration_index,
            hard_stop=True,
            stop_reason="foreground_collapse",
            collapse_type="foreground_collapse",
            confident_voxel_count=confident_voxel_count,
            foreground_assignment_count=foreground_count,
            background_assignment_count=background_count,
            foreground_fraction=foreground_fraction,
            configured_lower_safeguard=lower,
            configured_upper_safeguard=upper,
            detected=True,
            collapse_record=collapse_record,
        )
    if background_count == 0:
        collapse_record = _build_collapse_record(
            collapse_type="background_collapse",
            iteration_index=iteration_index,
            foreground_count=foreground_count,
            background_count=background_count,
            confident_voxel_count=confident_voxel_count,
            foreground_fraction=foreground_fraction,
            configured_lower_safeguard=lower,
            configured_upper_safeguard=upper,
        )
        return _build_collapse_decision(
            iteration_index=iteration_index,
            hard_stop=True,
            stop_reason="background_collapse",
            collapse_type="background_collapse",
            confident_voxel_count=confident_voxel_count,
            foreground_assignment_count=foreground_count,
            background_assignment_count=background_count,
            foreground_fraction=foreground_fraction,
            configured_lower_safeguard=lower,
            configured_upper_safeguard=upper,
            detected=True,
            collapse_record=collapse_record,
        )
    return _build_collapse_decision(
        iteration_index=iteration_index,
        hard_stop=False,
        stop_reason=None,
        collapse_type=None,
        confident_voxel_count=confident_voxel_count,
        foreground_assignment_count=foreground_count,
        background_assignment_count=background_count,
        foreground_fraction=foreground_fraction,
        configured_lower_safeguard=lower,
        configured_upper_safeguard=upper,
        detected=False,
        collapse_record=None,
    )


def evaluate_protoem_stopping_after_iteration(
    *,
    config: ProtoEMConfig,
    completed_records: tuple[ProtoEMIterationRecord, ...],
    current_state: ProtoEMTransductiveState,
) -> ProtoEMStoppingDecision:
    """Evaluate deterministic post-transition tolerance and bounded-completion stopping."""

    completed_iteration_count = len(completed_records)
    if completed_iteration_count == 0:
        raise StoppingConsistencyError(
            "evaluate_protoem_stopping_after_iteration requires at least one completed iteration."
        )
    if current_state.iteration_index != completed_iteration_count:
        raise StoppingConsistencyError(
            "current_state.iteration_index must equal the completed transition count."
        )
    last_index = completed_iteration_count - 1
    if completed_iteration_count >= config.minimum_iterations:
        delta = current_state.convergence_delta
        if delta is not None and delta <= config.convergence_tolerance:
            return _build_stopping_decision(
                should_stop=True,
                stop_reason="tolerance_reached",
                completed_iteration_count=completed_iteration_count,
                final_iteration_index=last_index,
                converged=True,
                failed=False,
                failure_code=None,
                failure_message=None,
            )
    if completed_iteration_count >= config.max_iterations:
        return _build_stopping_decision(
            should_stop=True,
            stop_reason="max_iterations",
            completed_iteration_count=completed_iteration_count,
            final_iteration_index=last_index,
            converged=False,
            failed=False,
            failure_code=None,
            failure_message=None,
        )
    return _build_stopping_decision(
        should_stop=False,
        stop_reason=None,
        completed_iteration_count=completed_iteration_count,
        final_iteration_index=last_index,
        converged=False,
        failed=False,
        failure_code=None,
        failure_message=None,
    )


def build_failure_stopping_decision(
    *,
    stop_reason: str,
    completed_records: tuple[ProtoEMIterationRecord, ...],
    failure_code: str,
    failure_message: str,
) -> ProtoEMStoppingDecision:
    """Build one explicit failed stopping decision with an artifact when possible."""

    completed_iteration_count = len(completed_records)
    final_iteration_index = completed_iteration_count - 1 if completed_iteration_count > 0 else None
    return _build_stopping_decision(
        should_stop=True,
        stop_reason=stop_reason,
        completed_iteration_count=completed_iteration_count,
        final_iteration_index=final_iteration_index,
        converged=False,
        failed=True,
        failure_code=failure_code,
        failure_message=failure_message,
    )


def hash_protoem_collapse_decision(decision: ProtoEMCollapseDecision) -> str:
    """Return the canonical hash for one collapse decision."""

    return sha256_json(protoem_collapse_decision_identity_payload(decision))


def hash_protoem_stopping_decision(decision: ProtoEMStoppingDecision) -> str:
    """Return the canonical hash for one stopping decision."""

    return sha256_json(protoem_stopping_decision_identity_payload(decision))


def protoem_collapse_decision_identity_payload(
    decision: ProtoEMCollapseDecision,
) -> dict[str, JsonValue]:
    """Return the canonical identity payload for one collapse decision."""

    return {
        "schema_name": decision.schema_name,
        "schema_version": decision.schema_version,
        "iteration_index": decision.iteration_index,
        "hard_stop": decision.hard_stop,
        "stop_reason": decision.stop_reason,
        "collapse_type": decision.collapse_type,
        "confident_voxel_count": decision.confident_voxel_count,
        "foreground_assignment_count": decision.foreground_assignment_count,
        "background_assignment_count": decision.background_assignment_count,
        "foreground_fraction": decision.foreground_fraction,
        "configured_lower_safeguard": decision.configured_lower_safeguard,
        "configured_upper_safeguard": decision.configured_upper_safeguard,
        "detected": decision.detected,
        "collapse_record_hash": (
            decision.collapse_record.collapse_record_hash
            if decision.collapse_record is not None
            else None
        ),
    }


def protoem_stopping_decision_identity_payload(
    decision: ProtoEMStoppingDecision,
) -> dict[str, JsonValue]:
    """Return the canonical identity payload for one stopping decision."""

    return {
        "schema_name": decision.schema_name,
        "schema_version": decision.schema_version,
        "should_stop": decision.should_stop,
        "stop_reason": decision.stop_reason,
        "completed_iteration_count": decision.completed_iteration_count,
        "final_iteration_index": decision.final_iteration_index,
        "converged": decision.converged,
        "failed": decision.failed,
        "failure_code": decision.failure_code,
        "failure_message": decision.failure_message,
        "stopping_record_hash": (
            decision.stopping_record.stopping_record_hash
            if decision.stopping_record is not None
            else None
        ),
    }


def _build_collapse_record(
    *,
    collapse_type: str,
    iteration_index: int,
    foreground_count: int,
    background_count: int,
    confident_voxel_count: int,
    foreground_fraction: float,
    configured_lower_safeguard: float,
    configured_upper_safeguard: float,
) -> ProtoEMCollapseRecord:
    payload = {
        "schema_name": PROTOEM_COLLAPSE_RECORD_SCHEMA_NAME,
        "schema_version": PROTOEM_COLLAPSE_RECORD_SCHEMA_VERSION,
        "collapse_type": collapse_type,
        "iteration_index": iteration_index,
        "foreground_count": foreground_count,
        "background_count": background_count,
        "confident_voxel_count": confident_voxel_count,
        "foreground_fraction": foreground_fraction,
        "configured_lower_safeguard": configured_lower_safeguard,
        "configured_upper_safeguard": configured_upper_safeguard,
        "detected": True,
    }
    collapse_record_hash = sha256_json(payload)
    record = ProtoEMCollapseRecord(
        schema_name=PROTOEM_COLLAPSE_RECORD_SCHEMA_NAME,
        schema_version=PROTOEM_COLLAPSE_RECORD_SCHEMA_VERSION,
        collapse_record_hash=collapse_record_hash,
        collapse_type=collapse_type,
        iteration_index=iteration_index,
        foreground_count=foreground_count,
        background_count=background_count,
        confident_voxel_count=confident_voxel_count,
        foreground_fraction=foreground_fraction,
        configured_lower_safeguard=configured_lower_safeguard,
        configured_upper_safeguard=configured_upper_safeguard,
        detected=True,
    )
    if record.collapse_record_hash != hash_protoem_collapse_record(record):
        raise CollapseConsistencyError(
            "collapse_record_hash does not match deterministic collapse content."
        )
    return record


def _build_collapse_decision(
    *,
    iteration_index: int,
    hard_stop: bool,
    stop_reason: str | None,
    collapse_type: str | None,
    confident_voxel_count: int,
    foreground_assignment_count: int,
    background_assignment_count: int,
    foreground_fraction: float,
    configured_lower_safeguard: float,
    configured_upper_safeguard: float,
    detected: bool,
    collapse_record: ProtoEMCollapseRecord | None,
) -> ProtoEMCollapseDecision:
    payload = {
        "schema_name": PROTOEM_COLLAPSE_DECISION_SCHEMA_NAME,
        "schema_version": PROTOEM_COLLAPSE_DECISION_SCHEMA_VERSION,
        "iteration_index": iteration_index,
        "hard_stop": hard_stop,
        "stop_reason": stop_reason,
        "collapse_type": collapse_type,
        "confident_voxel_count": confident_voxel_count,
        "foreground_assignment_count": foreground_assignment_count,
        "background_assignment_count": background_assignment_count,
        "foreground_fraction": foreground_fraction,
        "configured_lower_safeguard": configured_lower_safeguard,
        "configured_upper_safeguard": configured_upper_safeguard,
        "detected": detected,
        "collapse_record_hash": (
            collapse_record.collapse_record_hash if collapse_record is not None else None
        ),
    }
    return ProtoEMCollapseDecision(
        schema_name=PROTOEM_COLLAPSE_DECISION_SCHEMA_NAME,
        schema_version=PROTOEM_COLLAPSE_DECISION_SCHEMA_VERSION,
        decision_identity_hash=sha256_json(payload),
        iteration_index=iteration_index,
        hard_stop=hard_stop,
        stop_reason=stop_reason,
        collapse_type=collapse_type,
        confident_voxel_count=confident_voxel_count,
        foreground_assignment_count=foreground_assignment_count,
        background_assignment_count=background_assignment_count,
        foreground_fraction=foreground_fraction,
        configured_lower_safeguard=configured_lower_safeguard,
        configured_upper_safeguard=configured_upper_safeguard,
        detected=detected,
        collapse_record=collapse_record,
    )


def _build_stopping_decision(
    *,
    should_stop: bool,
    stop_reason: str | None,
    completed_iteration_count: int,
    final_iteration_index: int | None,
    converged: bool,
    failed: bool,
    failure_code: str | None,
    failure_message: str | None,
) -> ProtoEMStoppingDecision:
    stopping_record = None
    if should_stop and completed_iteration_count > 0 and stop_reason is not None:
        stopping_record = _build_stopping_record(
            stop_reason=stop_reason,
            final_iteration_index=final_iteration_index,
            completed_iteration_count=completed_iteration_count,
            converged=converged,
            failed=failed,
            failure_code=failure_code,
            failure_message=failure_message,
        )
    payload = {
        "schema_name": PROTOEM_STOPPING_DECISION_SCHEMA_NAME,
        "schema_version": PROTOEM_STOPPING_DECISION_SCHEMA_VERSION,
        "should_stop": should_stop,
        "stop_reason": stop_reason,
        "completed_iteration_count": completed_iteration_count,
        "final_iteration_index": final_iteration_index,
        "converged": converged,
        "failed": failed,
        "failure_code": failure_code,
        "failure_message": failure_message,
        "stopping_record_hash": (
            stopping_record.stopping_record_hash if stopping_record is not None else None
        ),
    }
    return ProtoEMStoppingDecision(
        schema_name=PROTOEM_STOPPING_DECISION_SCHEMA_NAME,
        schema_version=PROTOEM_STOPPING_DECISION_SCHEMA_VERSION,
        decision_identity_hash=sha256_json(payload),
        should_stop=should_stop,
        stop_reason=stop_reason,
        completed_iteration_count=completed_iteration_count,
        final_iteration_index=final_iteration_index,
        converged=converged,
        failed=failed,
        failure_code=failure_code,
        failure_message=failure_message,
        stopping_record=stopping_record,
    )


def _build_stopping_record(
    *,
    stop_reason: str,
    final_iteration_index: int | None,
    completed_iteration_count: int,
    converged: bool,
    failed: bool,
    failure_code: str | None,
    failure_message: str | None,
) -> ProtoEMStoppingRecord:
    if final_iteration_index is None:
        raise StoppingConsistencyError(
            "ProtoEMStoppingRecord requires final_iteration_index when iterations completed."
        )
    payload = {
        "schema_name": PROTOEM_STOPPING_RECORD_SCHEMA_NAME,
        "schema_version": PROTOEM_STOPPING_RECORD_SCHEMA_VERSION,
        "stop_reason": stop_reason,
        "final_iteration_index": final_iteration_index,
        "completed_iteration_count": completed_iteration_count,
        "converged": converged,
        "failed": failed,
        "failure_code": failure_code,
        "failure_message": failure_message,
    }
    stopping_record_hash = sha256_json(payload)
    record = ProtoEMStoppingRecord(
        schema_name=PROTOEM_STOPPING_RECORD_SCHEMA_NAME,
        schema_version=PROTOEM_STOPPING_RECORD_SCHEMA_VERSION,
        stopping_record_hash=stopping_record_hash,
        stop_reason=stop_reason,
        final_iteration_index=final_iteration_index,
        completed_iteration_count=completed_iteration_count,
        converged=converged,
        failed=failed,
        failure_code=failure_code,
        failure_message=failure_message,
    )
    if record.stopping_record_hash != hash_protoem_stopping_record(record):
        raise StoppingConsistencyError(
            "stopping_record_hash does not match deterministic stopping content."
        )
    return record


def _require_schema_name(
    value: str,
    *,
    expected: str,
    field_name: str,
    error_type: type[Exception],
) -> None:
    if value != expected:
        raise error_type(f"{field_name} must equal {expected!r}.")


def _require_schema_version(
    value: str,
    *,
    expected: str,
    field_name: str,
    error_type: type[Exception],
) -> None:
    if value != expected:
        raise error_type(f"{field_name} must equal {expected!r}.")


def _require_sha256(value: str, *, field_name: str, error_type: type[Exception]) -> None:
    if len(value) != 64 or any(character not in _SHA256_HEX for character in value):
        raise error_type(f"{field_name} must be a lowercase SHA-256 hex digest.")


def _require_nonnegative_int(value: int, *, field_name: str, error_type: type[Exception]) -> None:
    if value < 0:
        raise error_type(f"{field_name} must be nonnegative.")


def _require_fraction_closed(
    value: float,
    *,
    field_name: str,
    error_type: type[Exception],
) -> None:
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise error_type(f"{field_name} must lie in the closed interval [0,1].")


def _require_failure_consistency(
    *,
    failed: bool,
    failure_code: str | None,
    failure_message: str | None,
    field_prefix: str,
) -> None:
    if failed:
        if failure_code is None or failure_message is None:
            raise StoppingConsistencyError(
                f"{field_prefix} requires failure_code and failure_message when failed=true."
            )
        if not failure_code:
            raise StoppingConsistencyError(f"{field_prefix} failure_code must be nonempty.")
        if "\n" in failure_message or "\r" in failure_message or not failure_message:
            raise StoppingConsistencyError(
                f"{field_prefix} failure_message must be one nonempty line."
            )
    elif failure_code is not None or failure_message is not None:
        raise StoppingConsistencyError(
            f"{field_prefix} must not set failure_code/failure_message when failed=false."
        )


__all__ = [
    "PROTOEM_COLLAPSE_DECISION_SCHEMA_NAME",
    "PROTOEM_COLLAPSE_DECISION_SCHEMA_VERSION",
    "PROTOEM_STOPPING_DECISION_SCHEMA_NAME",
    "PROTOEM_STOPPING_DECISION_SCHEMA_VERSION",
    "CollapseConsistencyError",
    "IterationConsistencyError",
    "NumericalOrchestrationFailure",
    "OrchestrationInputError",
    "ProtoEMCollapseDecision",
    "ProtoEMOrchestrationError",
    "ProtoEMStoppingDecision",
    "StoppingConsistencyError",
    "build_failure_stopping_decision",
    "evaluate_protoem_collapse",
    "evaluate_protoem_stopping_after_iteration",
    "hash_protoem_collapse_decision",
    "hash_protoem_stopping_decision",
    "protoem_collapse_decision_identity_payload",
    "protoem_stopping_decision_identity_payload",
]
