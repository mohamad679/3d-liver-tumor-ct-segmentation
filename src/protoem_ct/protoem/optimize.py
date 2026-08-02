"""Deterministic ProtoEM-CT iterative orchestration over existing pure contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np

from protoem_ct.artifacts.hashing import JsonValue, sha256_json
from protoem_ct.protoem.artifacts import (
    PROTOEM_ITERATION_INDEX_BASE,
    PROTOEM_ITERATION_RECORD_SCHEMA_NAME,
    PROTOEM_ITERATION_RECORD_SCHEMA_VERSION,
    PROTOEM_OBJECTIVE_TRACE_SCHEMA_NAME,
    PROTOEM_OBJECTIVE_TRACE_SCHEMA_VERSION,
    ProtoEMCollapseRecord,
    ProtoEMConfig,
    ProtoEMIterationRecord,
    ProtoEMObjectiveTrace,
    ProtoEMStoppingRecord,
    hash_protoem_objective_trace,
    protoem_iteration_record_to_dict,
)
from protoem_ct.protoem.initialize import ProtoEMInitializationBundle
from protoem_ct.protoem.learned_schedule import (
    LearnedScheduleOrchestrationFailure,
    ProtoEMLearnedScheduleResult,
    ProtoEMPositiveStepSchedule,
    ScheduleLengthMismatchError,
    apply_positive_step_to_m_step_result,
    build_protoem_learned_schedule_result,
)
from protoem_ct.protoem.objective import (
    EmptyEffectiveBackgroundUpdateError,
    EmptyEffectiveForegroundUpdateError,
    IncompatiblePrototypeBankError,
    InvalidEStepInputError,
    InvalidMStepInputError,
    NoConfidentVoxelsError,
    NonFiniteObjectiveError,
    ObjectiveConsistencyFailureError,
    ProtoEMEStepResult,
    ProtoEMMStepResult,
    ProtoEMObjectiveTerms,
    ZeroNormPrototypeError,
    run_protoem_e_step,
    run_protoem_m_step,
)
from protoem_ct.protoem.state import (
    AssignmentConsistencyFailureError,
    IncompatibleStateTransitionError,
    InvalidStateError,
    PosteriorConsistencyFailureError,
    ProtoEMStateTransition,
    ProtoEMTransductiveState,
    PrototypeStateFailureError,
    StateIdentityMismatchError,
    build_initial_protoem_transductive_state,
    build_next_protoem_transductive_state,
    hash_protoem_objective_terms_state,
)
from protoem_ct.protoem.stopping import (
    IterationConsistencyError,
    NumericalOrchestrationFailure,
    OrchestrationInputError,
    ProtoEMCollapseDecision,
    ProtoEMStoppingDecision,
    build_failure_stopping_decision,
    evaluate_protoem_collapse,
    evaluate_protoem_stopping_after_iteration,
)

PROTOEM_ITERATION_EXECUTION_SCHEMA_NAME: Final[str] = "protoem_iteration_execution"
PROTOEM_ITERATION_EXECUTION_SCHEMA_VERSION: Final[str] = "v1"
PROTOEM_OPTIMIZATION_RESULT_SCHEMA_NAME: Final[str] = "protoem_optimization_result"
PROTOEM_OPTIMIZATION_RESULT_SCHEMA_VERSION: Final[str] = "v1"

_FLOAT_DTYPE: Final[np.dtype[np.float64]] = np.dtype(np.float64)
_SHA256_HEX: Final[frozenset[str]] = frozenset("0123456789abcdef")
_OBJECTIVE_TOLERANCE: Final[float] = 1e-12


@dataclass(frozen=True, slots=True)
class ProtoEMIterationExecution:
    """One completed deterministic ProtoEM transition and its iteration artifact."""

    schema_name: str
    schema_version: str
    execution_identity_hash: str
    iteration_index: int
    source_state: ProtoEMTransductiveState
    e_step_result: ProtoEMEStepResult
    m_step_result: ProtoEMMStepResult
    objective_terms: ProtoEMObjectiveTerms
    target_state: ProtoEMTransductiveState
    state_transition: ProtoEMStateTransition
    iteration_record: ProtoEMIterationRecord

    def __post_init__(self) -> None:
        _require_schema_name(
            self.schema_name,
            expected=PROTOEM_ITERATION_EXECUTION_SCHEMA_NAME,
            field_name="schema_name",
            error_type=IterationConsistencyError,
        )
        _require_schema_version(
            self.schema_version,
            expected=PROTOEM_ITERATION_EXECUTION_SCHEMA_VERSION,
            field_name="schema_version",
            error_type=IterationConsistencyError,
        )
        _require_sha256(
            self.execution_identity_hash,
            field_name="execution_identity_hash",
            error_type=IterationConsistencyError,
        )
        if self.iteration_index != self.source_state.iteration_index:
            raise IterationConsistencyError(
                "iteration_index must equal source_state.iteration_index."
            )
        if self.target_state.iteration_index != self.source_state.iteration_index + 1:
            raise IterationConsistencyError(
                "target_state.iteration_index must equal source_state.iteration_index + 1."
            )
        if self.iteration_record.iteration_index != self.iteration_index:
            raise IterationConsistencyError(
                "iteration_record.iteration_index must equal execution iteration_index."
            )
        if self.objective_terms != self.m_step_result.objective_terms:
            raise IterationConsistencyError(
                "objective_terms must equal m_step_result.objective_terms."
            )
        if self.state_transition.source_state_identity != self.source_state.state_identity_hash:
            raise IterationConsistencyError(
                "state_transition.source_state_identity must match source_state."
            )
        if self.state_transition.target_state_identity != self.target_state.state_identity_hash:
            raise IterationConsistencyError(
                "state_transition.target_state_identity must match target_state."
            )
        if (
            self.iteration_record.state_identity_hash_before
            != self.source_state.state_identity_hash
        ):
            raise IterationConsistencyError(
                "iteration_record.state_identity_hash_before must match source_state."
            )
        if self.iteration_record.state_identity_hash_after != self.target_state.state_identity_hash:
            raise IterationConsistencyError(
                "iteration_record.state_identity_hash_after must match target_state."
            )
        if (
            self.iteration_record.prototype_identity_hash_before
            != self.source_state.prototype_state.prototype_state_identity_hash
        ):
            raise IterationConsistencyError(
                "iteration_record.prototype_identity_hash_before must match source prototype state."
            )
        if (
            self.iteration_record.prototype_identity_hash_after
            != self.target_state.prototype_state.prototype_state_identity_hash
        ):
            raise IterationConsistencyError(
                "iteration_record.prototype_identity_hash_after must match target prototype state."
            )
        if self.execution_identity_hash != hash_protoem_iteration_execution(self):
            raise IterationConsistencyError(
                "execution_identity_hash does not match deterministic iteration content."
            )


@dataclass(frozen=True, slots=True)
class ProtoEMOptimizationResult:
    """Deterministic Phase 6 ProtoEM-CT orchestration result."""

    schema_name: str
    schema_version: str
    result_identity_hash: str
    config_hash: str
    initialization_identity_hash: str
    initial_state: ProtoEMTransductiveState
    final_state: ProtoEMTransductiveState
    iteration_executions: tuple[ProtoEMIterationExecution, ...]
    objective_trace: ProtoEMObjectiveTrace | None
    stopping_decision: ProtoEMStoppingDecision
    stopping_record: ProtoEMStoppingRecord | None
    collapse_decision: ProtoEMCollapseDecision | None
    collapse_record: ProtoEMCollapseRecord | None
    learned_schedule_result: ProtoEMLearnedScheduleResult | None
    execution_status: str
    failure_code: str | None
    failure_message: str | None

    def __post_init__(self) -> None:
        _require_schema_name(
            self.schema_name,
            expected=PROTOEM_OPTIMIZATION_RESULT_SCHEMA_NAME,
            field_name="schema_name",
            error_type=IterationConsistencyError,
        )
        _require_schema_version(
            self.schema_version,
            expected=PROTOEM_OPTIMIZATION_RESULT_SCHEMA_VERSION,
            field_name="schema_version",
            error_type=IterationConsistencyError,
        )
        _require_sha256(
            self.result_identity_hash,
            field_name="result_identity_hash",
            error_type=IterationConsistencyError,
        )
        _require_sha256(
            self.config_hash, field_name="config_hash", error_type=IterationConsistencyError
        )
        _require_sha256(
            self.initialization_identity_hash,
            field_name="initialization_identity_hash",
            error_type=IterationConsistencyError,
        )
        if self.initial_state.initialization_identity_hash != self.initialization_identity_hash:
            raise IterationConsistencyError(
                "initial_state.initialization_identity_hash must match the result."
            )
        if self.final_state.initialization_identity_hash != self.initialization_identity_hash:
            raise IterationConsistencyError(
                "final_state.initialization_identity_hash must match the result."
            )
        if self.execution_status not in {"completed", "failed"}:
            raise IterationConsistencyError("execution_status must be 'completed' or 'failed'.")
        if self.execution_status == "completed":
            if self.failure_code is not None or self.failure_message is not None:
                raise IterationConsistencyError(
                    "completed optimization results must not set failure_code or failure_message."
                )
        else:
            if self.failure_code is None or self.failure_message is None:
                raise IterationConsistencyError(
                    "failed optimization results require failure_code and failure_message."
                )
        if self.learned_schedule_result is not None and (
            self.learned_schedule_result.completed_iteration_count != len(self.iteration_executions)
        ):
            raise IterationConsistencyError(
                "learned_schedule_result.completed_iteration_count must match the "
                "completed iteration count."
            )
        if self.stopping_record is not self.stopping_decision.stopping_record:
            raise IterationConsistencyError(
                "stopping_record must be the exact artifact carried by stopping_decision."
            )
        if self.collapse_decision is None:
            if self.collapse_record is not None:
                raise IterationConsistencyError(
                    "collapse_record requires one explicit collapse_decision."
                )
        else:
            if self.collapse_record is not self.collapse_decision.collapse_record:
                raise IterationConsistencyError(
                    "collapse_record must be the exact artifact carried by collapse_decision."
                )
        if self.iteration_executions:
            expected_indices = tuple(range(len(self.iteration_executions)))
            actual_indices = tuple(item.iteration_index for item in self.iteration_executions)
            if actual_indices != expected_indices:
                raise IterationConsistencyError(
                    "iteration_executions must use contiguous zero-based iteration indices."
                )
            if self.final_state is not self.iteration_executions[-1].target_state:
                raise IterationConsistencyError(
                    "final_state must equal the target state of the final iteration execution."
                )
            if self.objective_trace is None:
                raise IterationConsistencyError(
                    "completed iteration executions require one ProtoEMObjectiveTrace."
                )
            if len(self.objective_trace.iterations) != len(self.iteration_executions):
                raise IterationConsistencyError(
                    "objective_trace length must equal the completed iteration count."
                )
            if self.stopping_record is None:
                raise IterationConsistencyError(
                    "completed iteration executions require one ProtoEMStoppingRecord."
                )
        else:
            if self.final_state is not self.initial_state:
                raise IterationConsistencyError(
                    "zero-iteration optimization results must keep final_state == initial_state."
                )
            if self.objective_trace is not None:
                raise IterationConsistencyError(
                    "zero-iteration optimization results must not fabricate an objective trace."
                )
            if self.stopping_record is not None:
                raise IterationConsistencyError(
                    "zero-iteration optimization results must not fabricate a stopping record."
                )
        if self.result_identity_hash != hash_protoem_optimization_result(self):
            raise IterationConsistencyError(
                "result_identity_hash does not match deterministic optimization content."
            )


def run_protoem_optimization(
    *,
    config: ProtoEMConfig,
    initialization_bundle: ProtoEMInitializationBundle,
    positive_step_schedule: ProtoEMPositiveStepSchedule | None = None,
) -> ProtoEMOptimizationResult:
    """Run deterministic Phase 6 ProtoEM-CT orchestration over existing pure contracts."""

    if config.update_schedule == "fixed_em_like":
        if positive_step_schedule is not None:
            raise OrchestrationInputError(
                "fixed_em_like does not accept one explicit positive_step_schedule."
            )
        return _run_fixed_em_like_optimization(
            config=config,
            initialization_bundle=initialization_bundle,
        )
    if config.update_schedule == "learned_positive_step":
        if positive_step_schedule is None:
            raise OrchestrationInputError(
                "learned_positive_step requires one explicit positive_step_schedule."
            )
        return _run_learned_positive_step_optimization(
            config=config,
            initialization_bundle=initialization_bundle,
            positive_step_schedule=positive_step_schedule,
        )
    raise OrchestrationInputError(
        "unsupported ProtoEM update_schedule; expected 'fixed_em_like' or 'learned_positive_step'."
    )


def _run_fixed_em_like_optimization(
    *,
    config: ProtoEMConfig,
    initialization_bundle: ProtoEMInitializationBundle,
) -> ProtoEMOptimizationResult:
    """Run the existing fixed EM-like orchestration path unchanged."""

    if config.prototype_mode != "single_prototype":
        raise OrchestrationInputError(
            "Phase 6 substage 7 supports prototype_mode='single_prototype' only."
        )

    query_features = np.ascontiguousarray(
        np.asarray(initialization_bundle.query_feature_encoding.feature_data).astype(
            _FLOAT_DTYPE, copy=False
        )
    )
    support_foreground_reference = np.ascontiguousarray(
        np.asarray(initialization_bundle.foreground_prototype.prototype_vector).astype(
            _FLOAT_DTYPE, copy=False
        )
    )
    support_background_reference = np.ascontiguousarray(
        np.asarray(initialization_bundle.background_prototype.prototype_vector).astype(
            _FLOAT_DTYPE, copy=False
        )
    )
    try:
        initial_state = build_initial_protoem_transductive_state(
            initialization_bundle=initialization_bundle,
            config=config,
        )
    except (
        InvalidStateError,
        StateIdentityMismatchError,
        PosteriorConsistencyFailureError,
        AssignmentConsistencyFailureError,
        PrototypeStateFailureError,
        IncompatibleStateTransitionError,
    ) as error:
        raise OrchestrationInputError(str(error)) from error

    current_state = initial_state
    executions: list[ProtoEMIterationExecution] = []
    iteration_records: list[ProtoEMIterationRecord] = []
    final_collapse_decision: ProtoEMCollapseDecision | None = None
    stopping_decision: ProtoEMStoppingDecision | None = None
    execution_status = "completed"
    failure_code: str | None = None
    failure_message: str | None = None

    for transition_index in range(config.max_iterations):
        try:
            e_step_result = run_protoem_e_step(
                query_feature_tensor=query_features,
                foreground_prototypes=current_state.prototype_state.foreground_prototypes,
                background_prototypes=current_state.prototype_state.background_prototypes,
                temperature=config.temperature,
                confidence_threshold=config.confidence_threshold,
                foreground_prior=config.foreground_prior,
            )
            _ensure_finite_e_step_result(e_step_result)
        except (
            InvalidEStepInputError,
            IncompatiblePrototypeBankError,
            ZeroNormPrototypeError,
            NumericalOrchestrationFailure,
        ) as error:
            execution_status = "failed"
            failure_code = "numerical_failure"
            failure_message = f"E-step failure: {error}"
            stopping_decision = build_failure_stopping_decision(
                stop_reason="numerical_failure",
                completed_records=tuple(iteration_records),
                failure_code=failure_code,
                failure_message=failure_message,
            )
            break

        collapse_decision = evaluate_protoem_collapse(
            iteration_index=transition_index,
            confidence_mask_result=e_step_result.confidence_mask_result,
        )
        if collapse_decision.hard_stop:
            execution_status = "failed"
            final_collapse_decision = collapse_decision
            if collapse_decision.stop_reason is None:
                raise IterationConsistencyError(
                    "hard-stop collapse decisions must define one stop_reason."
                )
            failure_code = collapse_decision.stop_reason
            if collapse_decision.stop_reason == "no_confident_voxels":
                failure_message = "No confident voxels remained after the E-step."
            else:
                failure_message = f"Detected {collapse_decision.stop_reason} before the M-step."
            stopping_decision = build_failure_stopping_decision(
                stop_reason=collapse_decision.stop_reason,
                completed_records=tuple(iteration_records),
                failure_code=failure_code,
                failure_message=failure_message,
            )
            break

        try:
            m_step_result = run_protoem_m_step(
                query_feature_tensor=query_features,
                foreground_posterior_map=e_step_result.foreground_posterior_map,
                background_posterior_map=e_step_result.background_posterior_map,
                confidence_mask=e_step_result.confidence_mask_result.confident_mask,
                initial_foreground_prototype=support_foreground_reference,
                initial_background_prototype=support_background_reference,
                previous_foreground_prototype=current_state.prototype_state.foreground_prototypes,
                previous_background_prototype=current_state.prototype_state.background_prototypes,
                objective_weights=config.objective_weights,
                foreground_prior=config.foreground_prior,
                proximal_foreground_reference=support_foreground_reference,
                proximal_background_reference=support_background_reference,
                previous_foreground_posterior_map=current_state.posterior_state.foreground_posterior_map,
                previous_background_posterior_map=current_state.posterior_state.background_posterior_map,
                support_foreground_reference=support_foreground_reference,
                support_background_reference=support_background_reference,
            )
            _ensure_finite_m_step_result(m_step_result)
        except NonFiniteObjectiveError as error:
            execution_status = "failed"
            failure_code = "non_finite_objective"
            failure_message = f"Non-finite objective: {error}"
            stopping_decision = build_failure_stopping_decision(
                stop_reason="non_finite_objective",
                completed_records=tuple(iteration_records),
                failure_code=failure_code,
                failure_message=failure_message,
            )
            break
        except (
            InvalidMStepInputError,
            NoConfidentVoxelsError,
            EmptyEffectiveForegroundUpdateError,
            EmptyEffectiveBackgroundUpdateError,
            ObjectiveConsistencyFailureError,
            NumericalOrchestrationFailure,
        ) as error:
            execution_status = "failed"
            failure_code = "numerical_failure"
            failure_message = f"M-step failure: {error}"
            stopping_decision = build_failure_stopping_decision(
                stop_reason="numerical_failure",
                completed_records=tuple(iteration_records),
                failure_code=failure_code,
                failure_message=failure_message,
            )
            break

        try:
            next_state, state_transition = build_next_protoem_transductive_state(
                source_state=current_state,
                e_step_result=e_step_result,
                m_step_result=m_step_result,
                objective_terms=m_step_result.objective_terms,
            )
        except (
            InvalidStateError,
            StateIdentityMismatchError,
            PosteriorConsistencyFailureError,
            AssignmentConsistencyFailureError,
            PrototypeStateFailureError,
            IncompatibleStateTransitionError,
        ) as error:
            execution_status = "failed"
            failure_code = "numerical_failure"
            failure_message = f"State transition failure: {error}"
            stopping_decision = build_failure_stopping_decision(
                stop_reason="numerical_failure",
                completed_records=tuple(iteration_records),
                failure_code=failure_code,
                failure_message=failure_message,
            )
            break

        iteration_record = _build_iteration_record(
            iteration_index=transition_index,
            source_state=current_state,
            target_state=next_state,
            objective_terms=m_step_result.objective_terms,
            collapse_status_detected=False,
        )
        iteration_records.append(iteration_record)
        execution = _build_iteration_execution(
            iteration_index=transition_index,
            source_state=current_state,
            e_step_result=e_step_result,
            m_step_result=m_step_result,
            objective_terms=m_step_result.objective_terms,
            target_state=next_state,
            state_transition=state_transition,
            iteration_record=iteration_record,
        )
        executions.append(execution)
        current_state = next_state

        stopping_decision = evaluate_protoem_stopping_after_iteration(
            config=config,
            completed_records=tuple(iteration_records),
            current_state=current_state,
        )
        if stopping_decision.should_stop:
            break
    else:
        raise IterationConsistencyError(
            "ProtoEM optimization loop exceeded config.max_iterations unexpectedly."
        )

    if stopping_decision is None:
        raise IterationConsistencyError("ProtoEM optimization must end with one stopping decision.")

    objective_trace = (
        _build_objective_trace(tuple(iteration_records)) if iteration_records else None
    )
    if execution_status == "completed":
        failure_code = None
        failure_message = None
    return _build_optimization_result(
        config=config,
        initialization_identity_hash=initialization_bundle.initialization_identity_hash,
        initial_state=initial_state,
        final_state=current_state,
        iteration_executions=tuple(executions),
        objective_trace=objective_trace,
        stopping_decision=stopping_decision,
        collapse_decision=final_collapse_decision,
        learned_schedule_result=None,
        execution_status=execution_status,
        failure_code=failure_code,
        failure_message=failure_message,
    )


def _run_learned_positive_step_optimization(
    *,
    config: ProtoEMConfig,
    initialization_bundle: ProtoEMInitializationBundle,
    positive_step_schedule: ProtoEMPositiveStepSchedule,
) -> ProtoEMOptimizationResult:
    """Run the parameterized positive-step schedule path over the fixed ProtoEM target M-step."""

    if config.prototype_mode != "single_prototype":
        raise OrchestrationInputError(
            "Phase 6 substage 7 supports prototype_mode='single_prototype' only."
        )
    if positive_step_schedule.update_schedule_name != config.update_schedule:
        raise OrchestrationInputError(
            "positive_step_schedule.update_schedule_name must match config.update_schedule."
        )
    if positive_step_schedule.max_iteration_compatibility != config.max_iterations:
        raise OrchestrationInputError(
            "positive_step_schedule.max_iteration_compatibility must equal config.max_iterations."
        )

    query_features = np.ascontiguousarray(
        np.asarray(initialization_bundle.query_feature_encoding.feature_data).astype(
            _FLOAT_DTYPE, copy=False
        )
    )
    support_foreground_reference = np.ascontiguousarray(
        np.asarray(initialization_bundle.foreground_prototype.prototype_vector).astype(
            _FLOAT_DTYPE, copy=False
        )
    )
    support_background_reference = np.ascontiguousarray(
        np.asarray(initialization_bundle.background_prototype.prototype_vector).astype(
            _FLOAT_DTYPE, copy=False
        )
    )
    try:
        initial_state = build_initial_protoem_transductive_state(
            initialization_bundle=initialization_bundle,
            config=config,
        )
    except (
        InvalidStateError,
        StateIdentityMismatchError,
        PosteriorConsistencyFailureError,
        AssignmentConsistencyFailureError,
        PrototypeStateFailureError,
        IncompatibleStateTransitionError,
    ) as error:
        raise OrchestrationInputError(str(error)) from error

    current_state = initial_state
    executions: list[ProtoEMIterationExecution] = []
    iteration_records: list[ProtoEMIterationRecord] = []
    schedule_records = []
    final_collapse_decision: ProtoEMCollapseDecision | None = None
    stopping_decision: ProtoEMStoppingDecision | None = None
    execution_status = "completed"
    failure_code: str | None = None
    failure_message: str | None = None

    for transition_index in range(config.max_iterations):
        try:
            e_step_result = run_protoem_e_step(
                query_feature_tensor=query_features,
                foreground_prototypes=current_state.prototype_state.foreground_prototypes,
                background_prototypes=current_state.prototype_state.background_prototypes,
                temperature=config.temperature,
                confidence_threshold=config.confidence_threshold,
                foreground_prior=config.foreground_prior,
            )
            _ensure_finite_e_step_result(e_step_result)
        except (
            InvalidEStepInputError,
            IncompatiblePrototypeBankError,
            ZeroNormPrototypeError,
            NumericalOrchestrationFailure,
        ) as error:
            execution_status = "failed"
            failure_code = "numerical_failure"
            failure_message = f"E-step failure: {error}"
            stopping_decision = build_failure_stopping_decision(
                stop_reason="numerical_failure",
                completed_records=tuple(iteration_records),
                failure_code=failure_code,
                failure_message=failure_message,
            )
            break

        collapse_decision = evaluate_protoem_collapse(
            iteration_index=transition_index,
            confidence_mask_result=e_step_result.confidence_mask_result,
        )
        if collapse_decision.hard_stop:
            execution_status = "failed"
            final_collapse_decision = collapse_decision
            if collapse_decision.stop_reason is None:
                raise IterationConsistencyError(
                    "hard-stop collapse decisions must define one stop_reason."
                )
            failure_code = collapse_decision.stop_reason
            if collapse_decision.stop_reason == "no_confident_voxels":
                failure_message = "No confident voxels remained after the E-step."
            else:
                failure_message = f"Detected {collapse_decision.stop_reason} before the M-step."
            stopping_decision = build_failure_stopping_decision(
                stop_reason=collapse_decision.stop_reason,
                completed_records=tuple(iteration_records),
                failure_code=failure_code,
                failure_message=failure_message,
            )
            break

        try:
            m_step_target = run_protoem_m_step(
                query_feature_tensor=query_features,
                foreground_posterior_map=e_step_result.foreground_posterior_map,
                background_posterior_map=e_step_result.background_posterior_map,
                confidence_mask=e_step_result.confidence_mask_result.confident_mask,
                initial_foreground_prototype=support_foreground_reference,
                initial_background_prototype=support_background_reference,
                previous_foreground_prototype=current_state.prototype_state.foreground_prototypes,
                previous_background_prototype=current_state.prototype_state.background_prototypes,
                objective_weights=config.objective_weights,
                foreground_prior=config.foreground_prior,
                proximal_foreground_reference=support_foreground_reference,
                proximal_background_reference=support_background_reference,
                previous_foreground_posterior_map=current_state.posterior_state.foreground_posterior_map,
                previous_background_posterior_map=current_state.posterior_state.background_posterior_map,
                support_foreground_reference=support_foreground_reference,
                support_background_reference=support_background_reference,
            )
            _ensure_finite_m_step_result(m_step_target)
        except NonFiniteObjectiveError as error:
            execution_status = "failed"
            failure_code = "non_finite_objective"
            failure_message = f"Non-finite objective: {error}"
            stopping_decision = build_failure_stopping_decision(
                stop_reason="non_finite_objective",
                completed_records=tuple(iteration_records),
                failure_code=failure_code,
                failure_message=failure_message,
            )
            break
        except (
            InvalidMStepInputError,
            NoConfidentVoxelsError,
            EmptyEffectiveForegroundUpdateError,
            EmptyEffectiveBackgroundUpdateError,
            ObjectiveConsistencyFailureError,
            NumericalOrchestrationFailure,
        ) as error:
            execution_status = "failed"
            failure_code = "numerical_failure"
            failure_message = f"M-step failure: {error}"
            stopping_decision = build_failure_stopping_decision(
                stop_reason="numerical_failure",
                completed_records=tuple(iteration_records),
                failure_code=failure_code,
                failure_message=failure_message,
            )
            break

        try:
            adjusted_m_step_result, schedule_record = apply_positive_step_to_m_step_result(
                iteration_index=transition_index,
                schedule=positive_step_schedule,
                source_prototype_state_identity_hash=(
                    current_state.prototype_state.prototype_state_identity_hash
                ),
                current_foreground_prototype=current_state.prototype_state.foreground_prototypes,
                current_background_prototype=current_state.prototype_state.background_prototypes,
                target_m_step_result=m_step_target,
                foreground_posterior_map=e_step_result.foreground_posterior_map,
                background_posterior_map=e_step_result.background_posterior_map,
                previous_foreground_posterior_map=(
                    current_state.posterior_state.foreground_posterior_map
                ),
                previous_background_posterior_map=(
                    current_state.posterior_state.background_posterior_map
                ),
                support_foreground_reference=support_foreground_reference,
                support_background_reference=support_background_reference,
                proximal_foreground_reference=support_foreground_reference,
                proximal_background_reference=support_background_reference,
                foreground_prior=config.foreground_prior,
            )
            _ensure_finite_m_step_result(adjusted_m_step_result)
        except (
            LearnedScheduleOrchestrationFailure,
            ScheduleLengthMismatchError,
            ObjectiveConsistencyFailureError,
            NonFiniteObjectiveError,
        ) as error:
            execution_status = "failed"
            failure_code = "numerical_failure"
            failure_message = f"Learned schedule failure: {error}"
            stopping_decision = build_failure_stopping_decision(
                stop_reason="numerical_failure",
                completed_records=tuple(iteration_records),
                failure_code=failure_code,
                failure_message=failure_message,
            )
            break

        try:
            next_state, state_transition = build_next_protoem_transductive_state(
                source_state=current_state,
                e_step_result=e_step_result,
                m_step_result=adjusted_m_step_result,
                objective_terms=adjusted_m_step_result.objective_terms,
            )
        except (
            InvalidStateError,
            StateIdentityMismatchError,
            PosteriorConsistencyFailureError,
            AssignmentConsistencyFailureError,
            PrototypeStateFailureError,
            IncompatibleStateTransitionError,
        ) as error:
            execution_status = "failed"
            failure_code = "numerical_failure"
            failure_message = f"State transition failure: {error}"
            stopping_decision = build_failure_stopping_decision(
                stop_reason="numerical_failure",
                completed_records=tuple(iteration_records),
                failure_code=failure_code,
                failure_message=failure_message,
            )
            break

        iteration_record = _build_iteration_record(
            iteration_index=transition_index,
            source_state=current_state,
            target_state=next_state,
            objective_terms=adjusted_m_step_result.objective_terms,
            collapse_status_detected=False,
        )
        iteration_records.append(iteration_record)
        schedule_records.append(schedule_record)
        execution = _build_iteration_execution(
            iteration_index=transition_index,
            source_state=current_state,
            e_step_result=e_step_result,
            m_step_result=adjusted_m_step_result,
            objective_terms=adjusted_m_step_result.objective_terms,
            target_state=next_state,
            state_transition=state_transition,
            iteration_record=iteration_record,
        )
        executions.append(execution)
        current_state = next_state

        stopping_decision = evaluate_protoem_stopping_after_iteration(
            config=config,
            completed_records=tuple(iteration_records),
            current_state=current_state,
        )
        if stopping_decision.should_stop:
            break
    else:
        raise IterationConsistencyError(
            "ProtoEM optimization loop exceeded config.max_iterations unexpectedly."
        )

    if stopping_decision is None:
        raise IterationConsistencyError("ProtoEM optimization must end with one stopping decision.")

    objective_trace = (
        _build_objective_trace(tuple(iteration_records)) if iteration_records else None
    )
    learned_schedule_result = build_protoem_learned_schedule_result(
        schedule=positive_step_schedule,
        records=tuple(schedule_records),
    )
    if execution_status == "completed":
        failure_code = None
        failure_message = None
    return _build_optimization_result(
        config=config,
        initialization_identity_hash=initialization_bundle.initialization_identity_hash,
        initial_state=initial_state,
        final_state=current_state,
        iteration_executions=tuple(executions),
        objective_trace=objective_trace,
        stopping_decision=stopping_decision,
        collapse_decision=final_collapse_decision,
        learned_schedule_result=learned_schedule_result,
        execution_status=execution_status,
        failure_code=failure_code,
        failure_message=failure_message,
    )


def hash_protoem_iteration_execution(execution: ProtoEMIterationExecution) -> str:
    """Return the canonical hash for one iteration execution."""

    return sha256_json(protoem_iteration_execution_identity_payload(execution))


def hash_protoem_optimization_result(result: ProtoEMOptimizationResult) -> str:
    """Return the canonical hash for one optimization result."""

    return sha256_json(protoem_optimization_result_identity_payload(result))


def protoem_iteration_execution_identity_payload(
    execution: ProtoEMIterationExecution,
) -> dict[str, JsonValue]:
    """Return the canonical identity payload for one iteration execution."""

    return {
        "schema_name": execution.schema_name,
        "schema_version": execution.schema_version,
        "iteration_index": execution.iteration_index,
        "source_state_identity_hash": execution.source_state.state_identity_hash,
        "e_step_result_identity_hash": execution.e_step_result.result_identity_hash,
        "m_step_result_identity_hash": execution.m_step_result.result_identity_hash,
        "objective_terms_identity_hash": hash_protoem_objective_terms_state(
            execution.objective_terms
        ),
        "target_state_identity_hash": execution.target_state.state_identity_hash,
        "state_transition_identity_hash": execution.state_transition.transition_identity_hash,
        "iteration_record": protoem_iteration_record_to_dict(execution.iteration_record),
    }


def protoem_optimization_result_identity_payload(
    result: ProtoEMOptimizationResult,
) -> dict[str, JsonValue]:
    """Return the canonical identity payload for one optimization result."""

    return {
        "schema_name": result.schema_name,
        "schema_version": result.schema_version,
        "config_hash": result.config_hash,
        "initialization_identity_hash": result.initialization_identity_hash,
        "initial_state_identity_hash": result.initial_state.state_identity_hash,
        "final_state_identity_hash": result.final_state.state_identity_hash,
        "iteration_execution_hashes": [
            item.execution_identity_hash for item in result.iteration_executions
        ],
        "objective_trace_hash": (
            result.objective_trace.objective_trace_hash
            if result.objective_trace is not None
            else None
        ),
        "stopping_decision_hash": result.stopping_decision.decision_identity_hash,
        "stopping_record_hash": (
            result.stopping_record.stopping_record_hash
            if result.stopping_record is not None
            else None
        ),
        "collapse_decision_hash": (
            result.collapse_decision.decision_identity_hash
            if result.collapse_decision is not None
            else None
        ),
        "collapse_record_hash": (
            result.collapse_record.collapse_record_hash
            if result.collapse_record is not None
            else None
        ),
        **(
            {
                "learned_schedule_result_hash": (
                    result.learned_schedule_result.learned_schedule_result_identity_hash
                )
            }
            if result.learned_schedule_result is not None
            else {}
        ),
        "execution_status": result.execution_status,
        "failure_code": result.failure_code,
        "failure_message": result.failure_message,
    }


def _build_iteration_record(
    *,
    iteration_index: int,
    source_state: ProtoEMTransductiveState,
    target_state: ProtoEMTransductiveState,
    objective_terms: ProtoEMObjectiveTerms,
    collapse_status_detected: bool,
) -> ProtoEMIterationRecord:
    convergence_delta = target_state.convergence_delta
    if convergence_delta is None:
        convergence_delta = 0.0
    return ProtoEMIterationRecord(
        schema_name=PROTOEM_ITERATION_RECORD_SCHEMA_NAME,
        schema_version=PROTOEM_ITERATION_RECORD_SCHEMA_VERSION,
        iteration_index_base=PROTOEM_ITERATION_INDEX_BASE,
        iteration_index=iteration_index,
        support_objective=objective_terms.support,
        query_entropy_objective=objective_terms.query_entropy,
        class_balance_objective=objective_terms.class_balance,
        consistency_objective=objective_terms.consistency,
        proximal_objective=objective_terms.proximal,
        total_objective=objective_terms.total,
        confident_voxel_count=target_state.assignment_state.confident_voxel_count,
        foreground_assignment_count=target_state.assignment_state.foreground_assignment_count,
        background_assignment_count=target_state.assignment_state.background_assignment_count,
        foreground_fraction=target_state.assignment_state.foreground_fraction,
        state_identity_hash_before=source_state.state_identity_hash,
        state_identity_hash_after=target_state.state_identity_hash,
        prototype_identity_hash_before=source_state.prototype_state.prototype_state_identity_hash,
        prototype_identity_hash_after=target_state.prototype_state.prototype_state_identity_hash,
        convergence_delta=convergence_delta,
        finite_status_ok=True,
        collapse_status_detected=collapse_status_detected,
    )


def _build_iteration_execution(
    *,
    iteration_index: int,
    source_state: ProtoEMTransductiveState,
    e_step_result: ProtoEMEStepResult,
    m_step_result: ProtoEMMStepResult,
    objective_terms: ProtoEMObjectiveTerms,
    target_state: ProtoEMTransductiveState,
    state_transition: ProtoEMStateTransition,
    iteration_record: ProtoEMIterationRecord,
) -> ProtoEMIterationExecution:
    payload = {
        "schema_name": PROTOEM_ITERATION_EXECUTION_SCHEMA_NAME,
        "schema_version": PROTOEM_ITERATION_EXECUTION_SCHEMA_VERSION,
        "iteration_index": iteration_index,
        "source_state_identity_hash": source_state.state_identity_hash,
        "e_step_result_identity_hash": e_step_result.result_identity_hash,
        "m_step_result_identity_hash": m_step_result.result_identity_hash,
        "objective_terms_identity_hash": hash_protoem_objective_terms_state(objective_terms),
        "target_state_identity_hash": target_state.state_identity_hash,
        "state_transition_identity_hash": state_transition.transition_identity_hash,
        "iteration_record": protoem_iteration_record_to_dict(iteration_record),
    }
    return ProtoEMIterationExecution(
        schema_name=PROTOEM_ITERATION_EXECUTION_SCHEMA_NAME,
        schema_version=PROTOEM_ITERATION_EXECUTION_SCHEMA_VERSION,
        execution_identity_hash=sha256_json(payload),
        iteration_index=iteration_index,
        source_state=source_state,
        e_step_result=e_step_result,
        m_step_result=m_step_result,
        objective_terms=objective_terms,
        target_state=target_state,
        state_transition=state_transition,
        iteration_record=iteration_record,
    )


def _build_objective_trace(records: tuple[ProtoEMIterationRecord, ...]) -> ProtoEMObjectiveTrace:
    payload = {
        "schema_name": PROTOEM_OBJECTIVE_TRACE_SCHEMA_NAME,
        "schema_version": PROTOEM_OBJECTIVE_TRACE_SCHEMA_VERSION,
        "iteration_index_base": PROTOEM_ITERATION_INDEX_BASE,
        "iterations": [protoem_iteration_record_to_dict(item) for item in records],
    }
    trace = ProtoEMObjectiveTrace(
        schema_name=PROTOEM_OBJECTIVE_TRACE_SCHEMA_NAME,
        schema_version=PROTOEM_OBJECTIVE_TRACE_SCHEMA_VERSION,
        objective_trace_hash=sha256_json(payload),
        iteration_index_base=PROTOEM_ITERATION_INDEX_BASE,
        iterations=records,
    )
    if trace.objective_trace_hash != hash_protoem_objective_trace(trace):
        raise IterationConsistencyError(
            "objective_trace_hash does not match deterministic optimization trace content."
        )
    return trace


def _build_optimization_result(
    *,
    config: ProtoEMConfig,
    initialization_identity_hash: str,
    initial_state: ProtoEMTransductiveState,
    final_state: ProtoEMTransductiveState,
    iteration_executions: tuple[ProtoEMIterationExecution, ...],
    objective_trace: ProtoEMObjectiveTrace | None,
    stopping_decision: ProtoEMStoppingDecision,
    collapse_decision: ProtoEMCollapseDecision | None,
    learned_schedule_result: ProtoEMLearnedScheduleResult | None,
    execution_status: str,
    failure_code: str | None,
    failure_message: str | None,
) -> ProtoEMOptimizationResult:
    payload = {
        "schema_name": PROTOEM_OPTIMIZATION_RESULT_SCHEMA_NAME,
        "schema_version": PROTOEM_OPTIMIZATION_RESULT_SCHEMA_VERSION,
        "config_hash": config.config_hash,
        "initialization_identity_hash": initialization_identity_hash,
        "initial_state_identity_hash": initial_state.state_identity_hash,
        "final_state_identity_hash": final_state.state_identity_hash,
        "iteration_execution_hashes": [
            item.execution_identity_hash for item in iteration_executions
        ],
        "objective_trace_hash": (
            objective_trace.objective_trace_hash if objective_trace is not None else None
        ),
        "stopping_decision_hash": stopping_decision.decision_identity_hash,
        "stopping_record_hash": (
            stopping_decision.stopping_record.stopping_record_hash
            if stopping_decision.stopping_record is not None
            else None
        ),
        "collapse_decision_hash": (
            collapse_decision.decision_identity_hash if collapse_decision is not None else None
        ),
        "collapse_record_hash": (
            collapse_decision.collapse_record.collapse_record_hash
            if collapse_decision is not None and collapse_decision.collapse_record is not None
            else None
        ),
        **(
            {
                "learned_schedule_result_hash": (
                    learned_schedule_result.learned_schedule_result_identity_hash
                )
            }
            if learned_schedule_result is not None
            else {}
        ),
        "execution_status": execution_status,
        "failure_code": failure_code,
        "failure_message": failure_message,
    }
    return ProtoEMOptimizationResult(
        schema_name=PROTOEM_OPTIMIZATION_RESULT_SCHEMA_NAME,
        schema_version=PROTOEM_OPTIMIZATION_RESULT_SCHEMA_VERSION,
        result_identity_hash=sha256_json(payload),
        config_hash=config.config_hash,
        initialization_identity_hash=initialization_identity_hash,
        initial_state=initial_state,
        final_state=final_state,
        iteration_executions=iteration_executions,
        objective_trace=objective_trace,
        stopping_decision=stopping_decision,
        stopping_record=stopping_decision.stopping_record,
        collapse_decision=collapse_decision,
        collapse_record=(
            collapse_decision.collapse_record if collapse_decision is not None else None
        ),
        learned_schedule_result=learned_schedule_result,
        execution_status=execution_status,
        failure_code=failure_code,
        failure_message=failure_message,
    )


def _ensure_finite_e_step_result(result: ProtoEMEStepResult) -> None:
    for array in (
        result.foreground_posterior_map,
        result.background_posterior_map,
        result.hard_assignment_map,
        result.zero_norm_query_voxel_mask,
        result.confidence_mask_result.confidence_map,
    ):
        if not np.isfinite(np.asarray(array, dtype=np.float64)).all():
            raise NumericalOrchestrationFailure("E-step output must remain finite.")


def _ensure_finite_m_step_result(result: ProtoEMMStepResult) -> None:
    for array in (
        result.updated_foreground_prototype,
        result.updated_background_prototype,
    ):
        if not np.isfinite(np.asarray(array, dtype=np.float64)).all():
            raise NumericalOrchestrationFailure("M-step output must remain finite.")
    for value in (
        result.effective_foreground_weight,
        result.effective_background_weight,
        result.objective_terms.support,
        result.objective_terms.query_entropy,
        result.objective_terms.class_balance,
        result.objective_terms.consistency,
        result.objective_terms.proximal,
        result.objective_terms.total,
    ):
        if not math.isfinite(value):
            raise NumericalOrchestrationFailure("M-step objective values must remain finite.")


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


__all__ = [
    "PROTOEM_ITERATION_EXECUTION_SCHEMA_NAME",
    "PROTOEM_ITERATION_EXECUTION_SCHEMA_VERSION",
    "PROTOEM_OPTIMIZATION_RESULT_SCHEMA_NAME",
    "PROTOEM_OPTIMIZATION_RESULT_SCHEMA_VERSION",
    "ProtoEMIterationExecution",
    "ProtoEMOptimizationResult",
    "hash_protoem_iteration_execution",
    "hash_protoem_optimization_result",
    "protoem_iteration_execution_identity_payload",
    "protoem_optimization_result_identity_payload",
    "run_protoem_optimization",
]
