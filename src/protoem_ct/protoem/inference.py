"""Deterministic ProtoEM-CT final inference and in-memory execution packaging."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Final

import numpy as np

from protoem_ct.artifacts.hashing import JsonValue, sha256_json
from protoem_ct.protoem.artifacts import (
    PROTOEM_RUN_SUMMARY_SCHEMA_NAME,
    PROTOEM_RUN_SUMMARY_SCHEMA_VERSION,
    ProtoEMAblationDefinition,
    ProtoEMCollapseRecord,
    ProtoEMConfig,
    ProtoEMObjectiveTrace,
    ProtoEMRunSummary,
    ProtoEMStoppingRecord,
    hash_protoem_run_summary,
)
from protoem_ct.protoem.initialize import ProtoEMInitializationBundle
from protoem_ct.protoem.optimize import ProtoEMOptimizationResult
from protoem_ct.protoem.stopping import ProtoEMCollapseDecision, ProtoEMStoppingDecision

PROTOEM_FINAL_INFERENCE_RESULT_SCHEMA_NAME: Final[str] = "protoem_final_inference_result"
PROTOEM_FINAL_INFERENCE_RESULT_SCHEMA_VERSION: Final[str] = "v1"
PROTOEM_EXECUTION_PACKAGE_SCHEMA_NAME: Final[str] = "protoem_execution_package"
PROTOEM_EXECUTION_PACKAGE_SCHEMA_VERSION: Final[str] = "v1"

_FLOAT_DTYPE: Final[np.dtype[np.float64]] = np.dtype(np.float64)
_BOOL_DTYPE: Final[np.dtype[np.bool_]] = np.dtype(np.bool_)
_PREDICTION_DTYPE: Final[np.dtype[np.uint8]] = np.dtype(np.uint8)
_SHA256_HEX: Final[frozenset[str]] = frozenset("0123456789abcdef")
_OBJECTIVE_TOLERANCE: Final[float] = 1e-12


class ProtoEMInferenceError(ValueError):
    """Base error for deterministic ProtoEM final inference and packaging."""


class FinalInferenceConstructionError(ProtoEMInferenceError):
    """Raised when one final inference artifact is malformed or unsupported."""


class ExecutionPackageConsistencyError(ProtoEMInferenceError):
    """Raised when one execution package is internally inconsistent."""


class RunSummaryAssemblyError(ProtoEMInferenceError):
    """Raised when one ProtoEM run summary cannot be assembled honestly."""


class ProtoEMInferenceIdentityMismatchError(ProtoEMInferenceError):
    """Raised when one final inference or execution package self-hash mismatches."""


@dataclass(frozen=True, slots=True)
class ProtoEMFinalInferenceResult:
    """Deterministic final in-memory inference result from one completed optimization run."""

    schema_name: str
    schema_version: str
    inference_identity_hash: str
    config_hash: str
    initialization_identity_hash: str
    optimization_result_identity_hash: str
    final_state_identity_hash: str
    foreground_posterior_map: np.ndarray
    background_posterior_map: np.ndarray
    confidence_map: np.ndarray
    prediction_map: np.ndarray
    final_foreground_prototypes: np.ndarray
    final_background_prototypes: np.ndarray
    foreground_posterior_content_hash: str
    background_posterior_content_hash: str
    confidence_content_hash: str
    prediction_content_hash: str
    final_foreground_prototype_content_hash: str
    final_background_prototype_content_hash: str
    stopping_reason: str
    completed_iteration_count: int

    def __post_init__(self) -> None:
        _require_schema_name(
            self.schema_name,
            expected=PROTOEM_FINAL_INFERENCE_RESULT_SCHEMA_NAME,
            field_name="schema_name",
            error_type=FinalInferenceConstructionError,
        )
        _require_schema_version(
            self.schema_version,
            expected=PROTOEM_FINAL_INFERENCE_RESULT_SCHEMA_VERSION,
            field_name="schema_version",
            error_type=FinalInferenceConstructionError,
        )
        for field_name in (
            "inference_identity_hash",
            "config_hash",
            "initialization_identity_hash",
            "optimization_result_identity_hash",
            "final_state_identity_hash",
            "foreground_posterior_content_hash",
            "background_posterior_content_hash",
            "confidence_content_hash",
            "prediction_content_hash",
            "final_foreground_prototype_content_hash",
            "final_background_prototype_content_hash",
        ):
            _require_sha256(
                getattr(self, field_name),
                field_name=field_name,
                error_type=FinalInferenceConstructionError,
            )
        foreground = _require_probability_volume(
            self.foreground_posterior_map,
            field_name="foreground_posterior_map",
            error_type=FinalInferenceConstructionError,
        )
        background = _require_probability_volume(
            self.background_posterior_map,
            field_name="background_posterior_map",
            error_type=FinalInferenceConstructionError,
        )
        confidence = _require_float_volume(
            self.confidence_map,
            field_name="confidence_map",
            error_type=FinalInferenceConstructionError,
        )
        prediction = _require_prediction_volume(
            self.prediction_map,
            field_name="prediction_map",
            error_type=FinalInferenceConstructionError,
        )
        if (
            foreground.shape != background.shape
            or foreground.shape != confidence.shape
            or foreground.shape != prediction.shape
        ):
            raise FinalInferenceConstructionError(
                "final inference arrays must share the same [1,1,D,H,W] shape."
            )
        if not np.allclose(foreground + background, 1.0, rtol=0.0, atol=_OBJECTIVE_TOLERANCE):
            raise FinalInferenceConstructionError(
                "foreground/background posterior maps must sum to one voxelwise."
            )
        expected_confidence = np.maximum(foreground, background)
        if not np.allclose(confidence, expected_confidence, rtol=0.0, atol=_OBJECTIVE_TOLERANCE):
            raise FinalInferenceConstructionError(
                "confidence_map must equal max(foreground_posterior_map, background_posterior_map)."
            )
        expected_prediction = np.greater(foreground, background).astype(_PREDICTION_DTYPE)
        if not np.array_equal(prediction, expected_prediction):
            raise FinalInferenceConstructionError(
                "prediction_map must mark foreground only where foreground posterior "
                "exceeds background posterior."
            )
        foreground_prototypes = _require_prototype_array(
            self.final_foreground_prototypes,
            field_name="final_foreground_prototypes",
            error_type=FinalInferenceConstructionError,
        )
        background_prototypes = _require_prototype_array(
            self.final_background_prototypes,
            field_name="final_background_prototypes",
            error_type=FinalInferenceConstructionError,
        )
        if foreground_prototypes.shape != background_prototypes.shape:
            raise FinalInferenceConstructionError(
                "final foreground/background prototype states must share identical shapes."
            )
        if self.foreground_posterior_content_hash != _array_content_sha256(foreground):
            raise FinalInferenceConstructionError(
                "foreground_posterior_content_hash must match foreground_posterior_map content."
            )
        if self.background_posterior_content_hash != _array_content_sha256(background):
            raise FinalInferenceConstructionError(
                "background_posterior_content_hash must match background_posterior_map content."
            )
        if self.confidence_content_hash != _array_content_sha256(confidence):
            raise FinalInferenceConstructionError(
                "confidence_content_hash must match confidence_map content."
            )
        if self.prediction_content_hash != _array_content_sha256(prediction):
            raise FinalInferenceConstructionError(
                "prediction_content_hash must match prediction_map content."
            )
        if self.final_foreground_prototype_content_hash != _array_content_sha256(
            foreground_prototypes
        ):
            raise FinalInferenceConstructionError(
                "final_foreground_prototype_content_hash must match final foreground "
                "prototype content."
            )
        if self.final_background_prototype_content_hash != _array_content_sha256(
            background_prototypes
        ):
            raise FinalInferenceConstructionError(
                "final_background_prototype_content_hash must match final background "
                "prototype content."
            )
        if not self.stopping_reason:
            raise FinalInferenceConstructionError("stopping_reason must be non-empty.")
        if self.completed_iteration_count <= 0:
            raise FinalInferenceConstructionError(
                "completed_iteration_count must be positive for one final inference result."
            )
        if self.inference_identity_hash != hash_protoem_final_inference_result(self):
            raise ProtoEMInferenceIdentityMismatchError(
                "inference_identity_hash does not match deterministic final inference content."
            )


@dataclass(frozen=True, slots=True)
class ProtoEMExecutionPackage:
    """Publication-ready deterministic in-memory packaging for one executed ProtoEM run."""

    schema_name: str
    schema_version: str
    package_identity_hash: str
    config: ProtoEMConfig
    ablation_definition: ProtoEMAblationDefinition
    initialization_bundle: ProtoEMInitializationBundle
    optimization_result: ProtoEMOptimizationResult
    final_inference_result: ProtoEMFinalInferenceResult | None
    objective_trace: ProtoEMObjectiveTrace | None
    stopping_decision: ProtoEMStoppingDecision
    stopping_record: ProtoEMStoppingRecord | None
    collapse_decision: ProtoEMCollapseDecision | None
    collapse_record: ProtoEMCollapseRecord | None
    run_summary: ProtoEMRunSummary | None

    def __post_init__(self) -> None:
        _require_schema_name(
            self.schema_name,
            expected=PROTOEM_EXECUTION_PACKAGE_SCHEMA_NAME,
            field_name="schema_name",
            error_type=ExecutionPackageConsistencyError,
        )
        _require_schema_version(
            self.schema_version,
            expected=PROTOEM_EXECUTION_PACKAGE_SCHEMA_VERSION,
            field_name="schema_version",
            error_type=ExecutionPackageConsistencyError,
        )
        _require_sha256(
            self.package_identity_hash,
            field_name="package_identity_hash",
            error_type=ExecutionPackageConsistencyError,
        )
        if self.optimization_result.config_hash != self.config.config_hash:
            raise ExecutionPackageConsistencyError(
                "optimization_result.config_hash must match config.config_hash."
            )
        if (
            self.optimization_result.initialization_identity_hash
            != self.initialization_bundle.initialization_identity_hash
        ):
            raise ExecutionPackageConsistencyError(
                "optimization_result.initialization_identity_hash must match initialization_bundle."
            )
        if self.objective_trace is not self.optimization_result.objective_trace:
            raise ExecutionPackageConsistencyError(
                "objective_trace must be the exact artifact carried by optimization_result."
            )
        if self.stopping_decision is not self.optimization_result.stopping_decision:
            raise ExecutionPackageConsistencyError(
                "stopping_decision must be the exact decision carried by optimization_result."
            )
        if self.stopping_record is not self.optimization_result.stopping_record:
            raise ExecutionPackageConsistencyError(
                "stopping_record must be the exact artifact carried by optimization_result."
            )
        if self.collapse_decision is not self.optimization_result.collapse_decision:
            raise ExecutionPackageConsistencyError(
                "collapse_decision must be the exact decision carried by optimization_result."
            )
        if self.collapse_record is not self.optimization_result.collapse_record:
            raise ExecutionPackageConsistencyError(
                "collapse_record must be the exact artifact carried by optimization_result."
            )
        if self.optimization_result.execution_status == "completed":
            if self.final_inference_result is None:
                raise ExecutionPackageConsistencyError(
                    "completed optimization results require one final inference result."
                )
            if self.run_summary is None:
                raise ExecutionPackageConsistencyError(
                    "completed optimization results require one ProtoEMRunSummary."
                )
        else:
            if self.final_inference_result is not None:
                raise ExecutionPackageConsistencyError(
                    "failed optimization results must not fabricate one final inference result."
                )
            if (
                self.run_summary is None
                and self.optimization_result.iteration_executions
                and self.stopping_record is not None
                and self.objective_trace is not None
            ):
                raise ExecutionPackageConsistencyError(
                    "failed optimization results with completed iterations require "
                    "one ProtoEMRunSummary."
                )
        if self.final_inference_result is not None:
            if self.final_inference_result.config_hash != self.config.config_hash:
                raise ExecutionPackageConsistencyError(
                    "final_inference_result.config_hash must match config.config_hash."
                )
            if (
                self.final_inference_result.initialization_identity_hash
                != self.initialization_bundle.initialization_identity_hash
            ):
                raise ExecutionPackageConsistencyError(
                    "final_inference_result.initialization_identity_hash must match "
                    "initialization_bundle."
                )
            if (
                self.final_inference_result.optimization_result_identity_hash
                != self.optimization_result.result_identity_hash
            ):
                raise ExecutionPackageConsistencyError(
                    "final_inference_result.optimization_result_identity_hash must "
                    "match optimization_result."
                )
            if (
                self.final_inference_result.final_state_identity_hash
                != self.optimization_result.final_state.state_identity_hash
            ):
                raise ExecutionPackageConsistencyError(
                    "final_inference_result.final_state_identity_hash must match "
                    "optimization_result.final_state."
                )
        if self.run_summary is not None:
            _validate_run_summary_consistency(
                summary=self.run_summary,
                config=self.config,
                ablation_definition=self.ablation_definition,
                initialization_bundle=self.initialization_bundle,
                optimization_result=self.optimization_result,
                final_inference_result=self.final_inference_result,
            )
        if self.package_identity_hash != hash_protoem_execution_package(self):
            raise ProtoEMInferenceIdentityMismatchError(
                "package_identity_hash does not match deterministic execution-package content."
            )


def build_protoem_final_inference_result(
    *,
    optimization_result: ProtoEMOptimizationResult,
) -> ProtoEMFinalInferenceResult:
    """Build one deterministic final inference result from one completed optimization result."""

    if optimization_result.execution_status != "completed":
        raise FinalInferenceConstructionError(
            "final inference requires one completed optimization result."
        )
    if optimization_result.stopping_decision.stop_reason is None:
        raise FinalInferenceConstructionError(
            "completed optimization results must expose one stopping reason."
        )
    completed_iteration_count = len(optimization_result.iteration_executions)
    if completed_iteration_count == 0:
        raise FinalInferenceConstructionError(
            "final inference cannot be constructed from one zero-iteration optimization result."
        )
    final_state = optimization_result.final_state
    foreground_posterior = _copy_float_volume(final_state.posterior_state.foreground_posterior_map)
    background_posterior = _copy_float_volume(final_state.posterior_state.background_posterior_map)
    confidence = _copy_float_volume(final_state.posterior_state.confidence_map)
    prediction = np.greater(foreground_posterior, background_posterior).astype(
        _PREDICTION_DTYPE, copy=False
    )
    foreground_prototypes = _copy_float_array(final_state.prototype_state.foreground_prototypes)
    background_prototypes = _copy_float_array(final_state.prototype_state.background_prototypes)
    payload = _protoem_final_inference_identity_payload_from_parts(
        config_hash=optimization_result.config_hash,
        initialization_identity_hash=optimization_result.initialization_identity_hash,
        optimization_result_identity_hash=optimization_result.result_identity_hash,
        final_state_identity_hash=final_state.state_identity_hash,
        foreground_posterior_content_hash=_array_content_sha256(foreground_posterior),
        background_posterior_content_hash=_array_content_sha256(background_posterior),
        confidence_content_hash=_array_content_sha256(confidence),
        prediction_content_hash=_array_content_sha256(prediction),
        final_foreground_prototype_content_hash=_array_content_sha256(foreground_prototypes),
        final_background_prototype_content_hash=_array_content_sha256(background_prototypes),
        stopping_reason=optimization_result.stopping_decision.stop_reason,
        completed_iteration_count=completed_iteration_count,
    )
    return ProtoEMFinalInferenceResult(
        schema_name=PROTOEM_FINAL_INFERENCE_RESULT_SCHEMA_NAME,
        schema_version=PROTOEM_FINAL_INFERENCE_RESULT_SCHEMA_VERSION,
        inference_identity_hash=sha256_json(payload),
        config_hash=optimization_result.config_hash,
        initialization_identity_hash=optimization_result.initialization_identity_hash,
        optimization_result_identity_hash=optimization_result.result_identity_hash,
        final_state_identity_hash=final_state.state_identity_hash,
        foreground_posterior_map=foreground_posterior,
        background_posterior_map=background_posterior,
        confidence_map=confidence,
        prediction_map=np.ascontiguousarray(prediction),
        final_foreground_prototypes=foreground_prototypes,
        final_background_prototypes=background_prototypes,
        foreground_posterior_content_hash=_array_content_sha256(foreground_posterior),
        background_posterior_content_hash=_array_content_sha256(background_posterior),
        confidence_content_hash=_array_content_sha256(confidence),
        prediction_content_hash=_array_content_sha256(prediction),
        final_foreground_prototype_content_hash=_array_content_sha256(foreground_prototypes),
        final_background_prototype_content_hash=_array_content_sha256(background_prototypes),
        stopping_reason=optimization_result.stopping_decision.stop_reason,
        completed_iteration_count=completed_iteration_count,
    )


def build_protoem_run_summary(
    *,
    config: ProtoEMConfig,
    ablation_definition: ProtoEMAblationDefinition,
    initialization_bundle: ProtoEMInitializationBundle,
    optimization_result: ProtoEMOptimizationResult,
    final_inference_result: ProtoEMFinalInferenceResult | None,
) -> ProtoEMRunSummary:
    """Build one deterministic ProtoEM run summary when executed artifacts permit it."""

    if optimization_result.objective_trace is None or optimization_result.stopping_record is None:
        raise RunSummaryAssemblyError(
            "ProtoEMRunSummary requires one objective trace and one stopping record."
        )
    if optimization_result.execution_status == "completed":
        if final_inference_result is None:
            raise RunSummaryAssemblyError(
                "completed ProtoEM runs require one final inference result."
            )
        execution_status = "completed"
        final_output_artifact_hash = optimization_result.final_state.state_identity_hash
        final_inference_artifact_hash = final_inference_result.inference_identity_hash
        failure_code = None
        failure_message = None
    elif optimization_result.execution_status == "failed":
        if final_inference_result is not None:
            raise RunSummaryAssemblyError(
                "failed ProtoEM runs must not claim one final inference artifact."
            )
        execution_status = "failed"
        final_output_artifact_hash = None
        final_inference_artifact_hash = None
        failure_code = optimization_result.failure_code
        failure_message = optimization_result.failure_message
    else:
        raise RunSummaryAssemblyError("unsupported optimization_result.execution_status.")
    if failure_code is None and execution_status == "failed":
        raise RunSummaryAssemblyError("failed run summaries require failure_code.")
    payload = {
        "schema_name": PROTOEM_RUN_SUMMARY_SCHEMA_NAME,
        "schema_version": PROTOEM_RUN_SUMMARY_SCHEMA_VERSION,
        "config_hash": config.config_hash,
        "phase5_initialization_artifact_hash": (
            initialization_bundle.phase5_initialization_reference.artifact_hash
        ),
        "phase5_prototype_artifact_hash": (
            initialization_bundle.phase5_prototype_reference.artifact_hash
        ),
        "phase5_inference_artifact_hash": (
            initialization_bundle.phase5_inference_reference.artifact_hash
        ),
        "objective_trace_hash": optimization_result.objective_trace.objective_trace_hash,
        "stopping_record_hash": optimization_result.stopping_record.stopping_record_hash,
        "collapse_record_hash": (
            optimization_result.collapse_record.collapse_record_hash
            if optimization_result.collapse_record is not None
            else None
        ),
        "ablation_definition_hash": ablation_definition.ablation_definition_hash,
        "final_output_artifact_hash": final_output_artifact_hash,
        "final_inference_artifact_hash": final_inference_artifact_hash,
        "execution_status": execution_status,
        "failure_code": failure_code,
        "failure_message": failure_message,
    }
    return ProtoEMRunSummary(
        schema_name=PROTOEM_RUN_SUMMARY_SCHEMA_NAME,
        schema_version=PROTOEM_RUN_SUMMARY_SCHEMA_VERSION,
        run_summary_hash=sha256_json(payload),
        config_hash=config.config_hash,
        phase5_initialization_artifact_hash=(
            initialization_bundle.phase5_initialization_reference.artifact_hash
        ),
        phase5_prototype_artifact_hash=(
            initialization_bundle.phase5_prototype_reference.artifact_hash
        ),
        phase5_inference_artifact_hash=(
            initialization_bundle.phase5_inference_reference.artifact_hash
        ),
        objective_trace_hash=optimization_result.objective_trace.objective_trace_hash,
        stopping_record_hash=optimization_result.stopping_record.stopping_record_hash,
        collapse_record_hash=(
            optimization_result.collapse_record.collapse_record_hash
            if optimization_result.collapse_record is not None
            else None
        ),
        ablation_definition_hash=ablation_definition.ablation_definition_hash,
        final_output_artifact_hash=final_output_artifact_hash,
        final_inference_artifact_hash=final_inference_artifact_hash,
        execution_status=execution_status,
        failure_code=failure_code,
        failure_message=failure_message,
        duration_seconds=None,
        memory_availability_status="unavailable",
        peak_allocated_memory_bytes=None,
        peak_reserved_memory_bytes=None,
        peak_host_memory_bytes=None,
    )


def build_protoem_execution_package(
    *,
    config: ProtoEMConfig,
    ablation_definition: ProtoEMAblationDefinition,
    initialization_bundle: ProtoEMInitializationBundle,
    optimization_result: ProtoEMOptimizationResult,
    final_inference_result: ProtoEMFinalInferenceResult | None = None,
) -> ProtoEMExecutionPackage:
    """Build one deterministic in-memory ProtoEM execution package."""

    if config.config_hash != optimization_result.config_hash:
        raise ExecutionPackageConsistencyError(
            "config.config_hash must match optimization_result.config_hash."
        )
    if (
        initialization_bundle.initialization_identity_hash
        != optimization_result.initialization_identity_hash
    ):
        raise ExecutionPackageConsistencyError(
            "initialization bundle must match optimization_result.initialization_identity_hash."
        )
    if (
        config.phase5_initialization_artifact_hash
        != initialization_bundle.phase5_initialization_reference.artifact_hash
        or config.phase5_prototype_artifact_hash
        != initialization_bundle.phase5_prototype_reference.artifact_hash
        or config.phase5_inference_artifact_hash
        != initialization_bundle.phase5_inference_reference.artifact_hash
    ):
        raise ExecutionPackageConsistencyError(
            "config Phase 5 reference hashes must match initialization-bundle references."
        )
    resolved_final_inference: ProtoEMFinalInferenceResult | None
    if optimization_result.execution_status == "completed":
        resolved_final_inference = final_inference_result or build_protoem_final_inference_result(
            optimization_result=optimization_result
        )
        run_summary = build_protoem_run_summary(
            config=config,
            ablation_definition=ablation_definition,
            initialization_bundle=initialization_bundle,
            optimization_result=optimization_result,
            final_inference_result=resolved_final_inference,
        )
    else:
        if final_inference_result is not None:
            raise ExecutionPackageConsistencyError(
                "failed optimization results must not supply one final inference result."
            )
        resolved_final_inference = None
        if (
            optimization_result.objective_trace is not None
            and optimization_result.stopping_record is not None
        ):
            run_summary = build_protoem_run_summary(
                config=config,
                ablation_definition=ablation_definition,
                initialization_bundle=initialization_bundle,
                optimization_result=optimization_result,
                final_inference_result=None,
            )
        else:
            run_summary = None
    payload = {
        "schema_name": PROTOEM_EXECUTION_PACKAGE_SCHEMA_NAME,
        "schema_version": PROTOEM_EXECUTION_PACKAGE_SCHEMA_VERSION,
        "config_hash": config.config_hash,
        "ablation_definition_hash": ablation_definition.ablation_definition_hash,
        "initialization_identity_hash": initialization_bundle.initialization_identity_hash,
        "optimization_result_identity_hash": optimization_result.result_identity_hash,
        "final_inference_identity_hash": (
            resolved_final_inference.inference_identity_hash
            if resolved_final_inference is not None
            else None
        ),
        "objective_trace_hash": (
            optimization_result.objective_trace.objective_trace_hash
            if optimization_result.objective_trace is not None
            else None
        ),
        "stopping_decision_hash": optimization_result.stopping_decision.decision_identity_hash,
        "stopping_record_hash": (
            optimization_result.stopping_record.stopping_record_hash
            if optimization_result.stopping_record is not None
            else None
        ),
        "collapse_decision_hash": (
            optimization_result.collapse_decision.decision_identity_hash
            if optimization_result.collapse_decision is not None
            else None
        ),
        "collapse_record_hash": (
            optimization_result.collapse_record.collapse_record_hash
            if optimization_result.collapse_record is not None
            else None
        ),
        "run_summary_hash": run_summary.run_summary_hash if run_summary is not None else None,
    }
    return ProtoEMExecutionPackage(
        schema_name=PROTOEM_EXECUTION_PACKAGE_SCHEMA_NAME,
        schema_version=PROTOEM_EXECUTION_PACKAGE_SCHEMA_VERSION,
        package_identity_hash=sha256_json(payload),
        config=config,
        ablation_definition=ablation_definition,
        initialization_bundle=initialization_bundle,
        optimization_result=optimization_result,
        final_inference_result=resolved_final_inference,
        objective_trace=optimization_result.objective_trace,
        stopping_decision=optimization_result.stopping_decision,
        stopping_record=optimization_result.stopping_record,
        collapse_decision=optimization_result.collapse_decision,
        collapse_record=optimization_result.collapse_record,
        run_summary=run_summary,
    )


def protoem_final_inference_identity_payload(
    result: ProtoEMFinalInferenceResult,
) -> dict[str, JsonValue]:
    """Return the exact identity payload for one final ProtoEM inference result."""

    return _protoem_final_inference_identity_payload_from_parts(
        config_hash=result.config_hash,
        initialization_identity_hash=result.initialization_identity_hash,
        optimization_result_identity_hash=result.optimization_result_identity_hash,
        final_state_identity_hash=result.final_state_identity_hash,
        foreground_posterior_content_hash=result.foreground_posterior_content_hash,
        background_posterior_content_hash=result.background_posterior_content_hash,
        confidence_content_hash=result.confidence_content_hash,
        prediction_content_hash=result.prediction_content_hash,
        final_foreground_prototype_content_hash=result.final_foreground_prototype_content_hash,
        final_background_prototype_content_hash=result.final_background_prototype_content_hash,
        stopping_reason=result.stopping_reason,
        completed_iteration_count=result.completed_iteration_count,
    )


def protoem_execution_package_identity_payload(
    package: ProtoEMExecutionPackage,
) -> dict[str, JsonValue]:
    """Return the exact identity payload for one ProtoEM execution package."""

    return {
        "schema_name": package.schema_name,
        "schema_version": package.schema_version,
        "config_hash": package.config.config_hash,
        "ablation_definition_hash": package.ablation_definition.ablation_definition_hash,
        "initialization_identity_hash": package.initialization_bundle.initialization_identity_hash,
        "optimization_result_identity_hash": package.optimization_result.result_identity_hash,
        "final_inference_identity_hash": (
            package.final_inference_result.inference_identity_hash
            if package.final_inference_result is not None
            else None
        ),
        "objective_trace_hash": (
            package.objective_trace.objective_trace_hash
            if package.objective_trace is not None
            else None
        ),
        "stopping_decision_hash": package.stopping_decision.decision_identity_hash,
        "stopping_record_hash": (
            package.stopping_record.stopping_record_hash
            if package.stopping_record is not None
            else None
        ),
        "collapse_decision_hash": (
            package.collapse_decision.decision_identity_hash
            if package.collapse_decision is not None
            else None
        ),
        "collapse_record_hash": (
            package.collapse_record.collapse_record_hash
            if package.collapse_record is not None
            else None
        ),
        "run_summary_hash": package.run_summary.run_summary_hash if package.run_summary else None,
    }


def hash_protoem_final_inference_result(result: ProtoEMFinalInferenceResult) -> str:
    """Return the deterministic final inference hash."""

    return sha256_json(protoem_final_inference_identity_payload(result))


def hash_protoem_execution_package(package: ProtoEMExecutionPackage) -> str:
    """Return the deterministic execution package hash."""

    return sha256_json(protoem_execution_package_identity_payload(package))


def protoem_final_inference_to_dict(result: ProtoEMFinalInferenceResult) -> dict[str, JsonValue]:
    """Convert one final inference result to a canonical mapping."""

    return {
        "schema_name": result.schema_name,
        "schema_version": result.schema_version,
        "inference_identity_hash": result.inference_identity_hash,
        "config_hash": result.config_hash,
        "initialization_identity_hash": result.initialization_identity_hash,
        "optimization_result_identity_hash": result.optimization_result_identity_hash,
        "final_state_identity_hash": result.final_state_identity_hash,
        "foreground_posterior_shape": list(result.foreground_posterior_map.shape),
        "background_posterior_shape": list(result.background_posterior_map.shape),
        "confidence_shape": list(result.confidence_map.shape),
        "prediction_shape": list(result.prediction_map.shape),
        "foreground_posterior_content_hash": result.foreground_posterior_content_hash,
        "background_posterior_content_hash": result.background_posterior_content_hash,
        "confidence_content_hash": result.confidence_content_hash,
        "prediction_content_hash": result.prediction_content_hash,
        "final_foreground_prototype_content_hash": result.final_foreground_prototype_content_hash,
        "final_background_prototype_content_hash": result.final_background_prototype_content_hash,
        "stopping_reason": result.stopping_reason,
        "completed_iteration_count": result.completed_iteration_count,
    }


def protoem_execution_package_to_dict(package: ProtoEMExecutionPackage) -> dict[str, JsonValue]:
    """Convert one execution package to a canonical mapping."""

    return {
        "schema_name": package.schema_name,
        "schema_version": package.schema_version,
        "package_identity_hash": package.package_identity_hash,
        "config_hash": package.config.config_hash,
        "ablation_definition_hash": package.ablation_definition.ablation_definition_hash,
        "initialization_identity_hash": package.initialization_bundle.initialization_identity_hash,
        "optimization_result_identity_hash": package.optimization_result.result_identity_hash,
        "final_inference_identity_hash": (
            package.final_inference_result.inference_identity_hash
            if package.final_inference_result is not None
            else None
        ),
        "objective_trace_hash": (
            package.objective_trace.objective_trace_hash
            if package.objective_trace is not None
            else None
        ),
        "stopping_decision_hash": package.stopping_decision.decision_identity_hash,
        "stopping_record_hash": (
            package.stopping_record.stopping_record_hash
            if package.stopping_record is not None
            else None
        ),
        "collapse_decision_hash": (
            package.collapse_decision.decision_identity_hash
            if package.collapse_decision is not None
            else None
        ),
        "collapse_record_hash": (
            package.collapse_record.collapse_record_hash
            if package.collapse_record is not None
            else None
        ),
        "run_summary_hash": package.run_summary.run_summary_hash if package.run_summary else None,
    }


def query_label_not_part_of_final_inference_api() -> bool:
    """Return whether query labels are absent from the public final inference API."""

    return True


def _protoem_final_inference_identity_payload_from_parts(
    *,
    config_hash: str,
    initialization_identity_hash: str,
    optimization_result_identity_hash: str,
    final_state_identity_hash: str,
    foreground_posterior_content_hash: str,
    background_posterior_content_hash: str,
    confidence_content_hash: str,
    prediction_content_hash: str,
    final_foreground_prototype_content_hash: str,
    final_background_prototype_content_hash: str,
    stopping_reason: str,
    completed_iteration_count: int,
) -> dict[str, JsonValue]:
    return {
        "schema_name": PROTOEM_FINAL_INFERENCE_RESULT_SCHEMA_NAME,
        "schema_version": PROTOEM_FINAL_INFERENCE_RESULT_SCHEMA_VERSION,
        "config_hash": config_hash,
        "initialization_identity_hash": initialization_identity_hash,
        "optimization_result_identity_hash": optimization_result_identity_hash,
        "final_state_identity_hash": final_state_identity_hash,
        "foreground_posterior_content_hash": foreground_posterior_content_hash,
        "background_posterior_content_hash": background_posterior_content_hash,
        "confidence_content_hash": confidence_content_hash,
        "prediction_content_hash": prediction_content_hash,
        "final_foreground_prototype_content_hash": final_foreground_prototype_content_hash,
        "final_background_prototype_content_hash": final_background_prototype_content_hash,
        "stopping_reason": stopping_reason,
        "completed_iteration_count": completed_iteration_count,
    }


def _validate_run_summary_consistency(
    *,
    summary: ProtoEMRunSummary,
    config: ProtoEMConfig,
    ablation_definition: ProtoEMAblationDefinition,
    initialization_bundle: ProtoEMInitializationBundle,
    optimization_result: ProtoEMOptimizationResult,
    final_inference_result: ProtoEMFinalInferenceResult | None,
) -> None:
    if summary.config_hash != config.config_hash:
        raise ExecutionPackageConsistencyError("run_summary.config_hash must match config.")
    if (
        summary.phase5_initialization_artifact_hash
        != initialization_bundle.phase5_initialization_reference.artifact_hash
    ):
        raise ExecutionPackageConsistencyError(
            "run_summary.phase5_initialization_artifact_hash must match initialization bundle."
        )
    if (
        summary.phase5_prototype_artifact_hash
        != initialization_bundle.phase5_prototype_reference.artifact_hash
    ):
        raise ExecutionPackageConsistencyError(
            "run_summary.phase5_prototype_artifact_hash must match initialization bundle."
        )
    if (
        summary.phase5_inference_artifact_hash
        != initialization_bundle.phase5_inference_reference.artifact_hash
    ):
        raise ExecutionPackageConsistencyError(
            "run_summary.phase5_inference_artifact_hash must match initialization bundle."
        )
    if summary.ablation_definition_hash != ablation_definition.ablation_definition_hash:
        raise ExecutionPackageConsistencyError(
            "run_summary.ablation_definition_hash must match ablation_definition."
        )
    if optimization_result.objective_trace is None or optimization_result.stopping_record is None:
        raise ExecutionPackageConsistencyError(
            "run_summary requires executed optimization_result artifacts."
        )
    if summary.objective_trace_hash != optimization_result.objective_trace.objective_trace_hash:
        raise ExecutionPackageConsistencyError(
            "run_summary.objective_trace_hash must match optimization_result."
        )
    if summary.stopping_record_hash != optimization_result.stopping_record.stopping_record_hash:
        raise ExecutionPackageConsistencyError(
            "run_summary.stopping_record_hash must match optimization_result."
        )
    expected_collapse_hash = (
        optimization_result.collapse_record.collapse_record_hash
        if optimization_result.collapse_record is not None
        else None
    )
    if summary.collapse_record_hash != expected_collapse_hash:
        raise ExecutionPackageConsistencyError(
            "run_summary.collapse_record_hash must match optimization_result."
        )
    if optimization_result.execution_status == "completed":
        if final_inference_result is None:
            raise ExecutionPackageConsistencyError(
                "completed run_summary validation requires final_inference_result."
            )
        if summary.execution_status != "completed":
            raise ExecutionPackageConsistencyError(
                "completed optimization results require run_summary.execution_status='completed'."
            )
        if (
            summary.final_output_artifact_hash
            != optimization_result.final_state.state_identity_hash
        ):
            raise ExecutionPackageConsistencyError(
                "run_summary.final_output_artifact_hash must equal final state identity."
            )
        if summary.final_inference_artifact_hash != final_inference_result.inference_identity_hash:
            raise ExecutionPackageConsistencyError(
                "run_summary.final_inference_artifact_hash must equal final inference identity."
            )
    else:
        if summary.execution_status != "failed":
            raise ExecutionPackageConsistencyError(
                "failed optimization results require run_summary.execution_status='failed'."
            )
        if summary.final_output_artifact_hash is not None:
            raise ExecutionPackageConsistencyError(
                "failed run_summary must not claim one final output artifact hash."
            )
        if summary.final_inference_artifact_hash is not None:
            raise ExecutionPackageConsistencyError(
                "failed run_summary must not claim one final inference artifact hash."
            )
    if summary.run_summary_hash != hash_protoem_run_summary(summary):
        raise ExecutionPackageConsistencyError("run_summary hash must validate before packaging.")


def _copy_float_volume(value: np.ndarray) -> np.ndarray:
    array = _require_float_volume(
        value,
        field_name="value",
        error_type=FinalInferenceConstructionError,
    )
    return np.ascontiguousarray(array, dtype=_FLOAT_DTYPE)


def _copy_float_array(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim not in {1, 2}:
        raise FinalInferenceConstructionError("prototype arrays must be one- or two-dimensional.")
    if array.dtype.kind not in {"f", "i", "u"}:
        raise FinalInferenceConstructionError("prototype arrays must be numeric.")
    copied = np.ascontiguousarray(array.astype(_FLOAT_DTYPE, copy=False))
    if not np.isfinite(copied).all():
        raise FinalInferenceConstructionError("prototype arrays must be finite.")
    return copied


def _require_probability_volume(
    value: np.ndarray,
    *,
    field_name: str,
    error_type: type[FinalInferenceConstructionError],
) -> np.ndarray:
    array = _require_float_volume(value, field_name=field_name, error_type=error_type)
    if np.any(array < 0.0) or np.any(array > 1.0):
        raise error_type(f"{field_name} must lie in [0, 1].")
    return array


def _require_prediction_volume(
    value: np.ndarray,
    *,
    field_name: str,
    error_type: type[FinalInferenceConstructionError],
) -> np.ndarray:
    array = np.asarray(value)
    if array.shape[:2] != (1, 1) or array.ndim != 5:
        raise error_type(f"{field_name} must have shape [1,1,D,H,W].")
    if array.dtype.kind not in {"b", "i", "u"}:
        raise error_type(f"{field_name} must be binary.")
    copied = np.ascontiguousarray(array.astype(_PREDICTION_DTYPE, copy=False))
    if not np.isin(copied, np.asarray([0, 1], dtype=_PREDICTION_DTYPE)).all():
        raise error_type(f"{field_name} must contain only 0/1 values.")
    return copied


def _require_float_volume(
    value: np.ndarray,
    *,
    field_name: str,
    error_type: type[FinalInferenceConstructionError],
) -> np.ndarray:
    array = np.asarray(value)
    if array.shape[:2] != (1, 1) or array.ndim != 5:
        raise error_type(f"{field_name} must have shape [1,1,D,H,W].")
    if array.dtype.kind not in {"f", "i", "u"}:
        raise error_type(f"{field_name} must be numeric.")
    copied = np.ascontiguousarray(array.astype(_FLOAT_DTYPE, copy=False))
    if not np.isfinite(copied).all():
        raise error_type(f"{field_name} must be finite.")
    return copied


def _require_prototype_array(
    value: np.ndarray,
    *,
    field_name: str,
    error_type: type[FinalInferenceConstructionError],
) -> np.ndarray:
    copied = _copy_float_array(value)
    if copied.ndim == 2 and copied.shape[0] <= 0:
        raise error_type(f"{field_name} must not be empty.")
    if copied.ndim == 1 and copied.shape[0] <= 0:
        raise error_type(f"{field_name} must not be empty.")
    return copied


def _array_content_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _require_schema_name(
    value: str,
    *,
    expected: str,
    field_name: str,
    error_type: type[ProtoEMInferenceError],
) -> None:
    if value != expected:
        raise error_type(f"{field_name} must equal {expected!r}.")


def _require_schema_version(
    value: str,
    *,
    expected: str,
    field_name: str,
    error_type: type[ProtoEMInferenceError],
) -> None:
    if value != expected:
        raise error_type(f"{field_name} must equal {expected!r}.")


def _require_sha256(
    value: str,
    *,
    field_name: str,
    error_type: type[ProtoEMInferenceError],
) -> None:
    if len(value) != 64 or any(character not in _SHA256_HEX for character in value):
        raise error_type(f"{field_name} must be one lowercase hexadecimal SHA-256 digest.")


__all__ = [
    "ExecutionPackageConsistencyError",
    "FinalInferenceConstructionError",
    "PROTOEM_EXECUTION_PACKAGE_SCHEMA_NAME",
    "PROTOEM_EXECUTION_PACKAGE_SCHEMA_VERSION",
    "PROTOEM_FINAL_INFERENCE_RESULT_SCHEMA_NAME",
    "PROTOEM_FINAL_INFERENCE_RESULT_SCHEMA_VERSION",
    "ProtoEMExecutionPackage",
    "ProtoEMFinalInferenceResult",
    "ProtoEMInferenceError",
    "ProtoEMInferenceIdentityMismatchError",
    "RunSummaryAssemblyError",
    "build_protoem_execution_package",
    "build_protoem_final_inference_result",
    "build_protoem_run_summary",
    "hash_protoem_execution_package",
    "hash_protoem_final_inference_result",
    "protoem_execution_package_identity_payload",
    "protoem_execution_package_to_dict",
    "protoem_final_inference_identity_payload",
    "protoem_final_inference_to_dict",
    "query_label_not_part_of_final_inference_api",
]
