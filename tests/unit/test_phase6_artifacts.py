"""Unit tests for deterministic Phase 6 ProtoEM artifact contracts."""

from __future__ import annotations

import json
import math
from collections.abc import Callable

import pytest

from protoem_ct.artifacts.hashing import canonical_json_bytes, sha256_json
from protoem_ct.protoem import (
    COMPATIBILITY_CRITICAL_CONFIG_OVERRIDE_FIELDS,
    PROTOEM_ABLATION_DEFINITION_SCHEMA_NAME,
    PROTOEM_ABLATION_DEFINITION_SCHEMA_VERSION,
    PROTOEM_ABLATION_OVERRIDE_SCHEMA_NAME,
    PROTOEM_ABLATION_OVERRIDE_SCHEMA_VERSION,
    PROTOEM_COLLAPSE_RECORD_SCHEMA_NAME,
    PROTOEM_COLLAPSE_RECORD_SCHEMA_VERSION,
    PROTOEM_CONFIG_SCHEMA_NAME,
    PROTOEM_CONFIG_SCHEMA_VERSION,
    PROTOEM_ITERATION_INDEX_BASE,
    PROTOEM_ITERATION_RECORD_SCHEMA_NAME,
    PROTOEM_ITERATION_RECORD_SCHEMA_VERSION,
    PROTOEM_OBJECTIVE_TRACE_SCHEMA_NAME,
    PROTOEM_OBJECTIVE_TRACE_SCHEMA_VERSION,
    PROTOEM_OBJECTIVE_WEIGHTS_SCHEMA_NAME,
    PROTOEM_OBJECTIVE_WEIGHTS_SCHEMA_VERSION,
    PROTOEM_RUN_SUMMARY_SCHEMA_NAME,
    PROTOEM_RUN_SUMMARY_SCHEMA_VERSION,
    PROTOEM_STOPPING_RECORD_SCHEMA_NAME,
    PROTOEM_STOPPING_RECORD_SCHEMA_VERSION,
    ProtoEMAblationDefinition,
    ProtoEMAblationOverride,
    ProtoEMArtifactHashError,
    ProtoEMArtifactSerializationError,
    ProtoEMArtifactValidationError,
    ProtoEMCollapseRecord,
    ProtoEMConfig,
    ProtoEMIterationRecord,
    ProtoEMObjectiveTrace,
    ProtoEMObjectiveWeights,
    ProtoEMRunSummary,
    ProtoEMStoppingRecord,
    hash_protoem_config,
    hash_protoem_objective_trace,
    protoem_ablation_definition_from_json,
    protoem_ablation_definition_to_json,
    protoem_artifact_from_json,
    protoem_config_from_json,
    protoem_config_identity_payload,
    protoem_config_to_json,
    protoem_objective_trace_from_json,
    protoem_objective_trace_identity_payload,
    protoem_objective_trace_to_json,
    protoem_run_summary_from_json,
    protoem_run_summary_to_json,
)


def _weights() -> ProtoEMObjectiveWeights:
    return ProtoEMObjectiveWeights(
        schema_name=PROTOEM_OBJECTIVE_WEIGHTS_SCHEMA_NAME,
        schema_version=PROTOEM_OBJECTIVE_WEIGHTS_SCHEMA_VERSION,
        support=1.0,
        query_entropy=0.25,
        class_balance=0.1,
        consistency=0.5,
        proximal=0.2,
    )


def _config(
    *,
    explicit_seed: int = 1729,
    max_iterations: int = 6,
    minimum_iterations: int = 1,
    convergence_tolerance: float = 0.001,
    temperature: float = 1.25,
    confidence_threshold: float = 0.9,
    foreground_prior: float = 0.2,
    update_schedule: str = "fixed_em_like",
    prototype_mode: str = "single_prototype",
    weights: ProtoEMObjectiveWeights | None = None,
    phase5_initialization_schema_name: str = "prototype_inference_summary",
    phase5_initialization_artifact_hash: str = "1" * 64,
    phase5_prototype_schema_name: str = "prototype_memory",
    phase5_prototype_artifact_hash: str = "2" * 64,
    phase5_inference_schema_name: str = "phase5_run_summary",
    phase5_inference_artifact_hash: str = "3" * 64,
) -> ProtoEMConfig:
    actual_weights = weights or _weights()
    config_hash = sha256_json(
        {
            "schema_name": PROTOEM_CONFIG_SCHEMA_NAME,
            "schema_version": PROTOEM_CONFIG_SCHEMA_VERSION,
            "explicit_seed": explicit_seed,
            "max_iterations": max_iterations,
            "minimum_iterations": minimum_iterations,
            "convergence_tolerance": convergence_tolerance,
            "temperature": temperature,
            "confidence_threshold": confidence_threshold,
            "foreground_prior": foreground_prior,
            "update_schedule": update_schedule,
            "prototype_mode": prototype_mode,
            "objective_weights": {
                "schema_name": actual_weights.schema_name,
                "schema_version": actual_weights.schema_version,
                "support": actual_weights.support,
                "query_entropy": actual_weights.query_entropy,
                "class_balance": actual_weights.class_balance,
                "consistency": actual_weights.consistency,
                "proximal": actual_weights.proximal,
            },
            "phase5_initialization_schema_name": phase5_initialization_schema_name,
            "phase5_initialization_artifact_hash": phase5_initialization_artifact_hash,
            "phase5_prototype_schema_name": phase5_prototype_schema_name,
            "phase5_prototype_artifact_hash": phase5_prototype_artifact_hash,
            "phase5_inference_schema_name": phase5_inference_schema_name,
            "phase5_inference_artifact_hash": phase5_inference_artifact_hash,
        }
    )
    return ProtoEMConfig(
        schema_name=PROTOEM_CONFIG_SCHEMA_NAME,
        schema_version=PROTOEM_CONFIG_SCHEMA_VERSION,
        config_hash=config_hash,
        explicit_seed=explicit_seed,
        max_iterations=max_iterations,
        minimum_iterations=minimum_iterations,
        convergence_tolerance=convergence_tolerance,
        temperature=temperature,
        confidence_threshold=confidence_threshold,
        foreground_prior=foreground_prior,
        update_schedule=update_schedule,
        prototype_mode=prototype_mode,
        objective_weights=actual_weights,
        phase5_initialization_schema_name=phase5_initialization_schema_name,
        phase5_initialization_artifact_hash=phase5_initialization_artifact_hash,
        phase5_prototype_schema_name=phase5_prototype_schema_name,
        phase5_prototype_artifact_hash=phase5_prototype_artifact_hash,
        phase5_inference_schema_name=phase5_inference_schema_name,
        phase5_inference_artifact_hash=phase5_inference_artifact_hash,
    )


def _iteration(
    *,
    iteration_index: int,
    collapse_status_detected: bool = False,
) -> ProtoEMIterationRecord:
    return ProtoEMIterationRecord(
        schema_name=PROTOEM_ITERATION_RECORD_SCHEMA_NAME,
        schema_version=PROTOEM_ITERATION_RECORD_SCHEMA_VERSION,
        iteration_index_base=PROTOEM_ITERATION_INDEX_BASE,
        iteration_index=iteration_index,
        support_objective=0.5 + iteration_index,
        query_entropy_objective=0.1,
        class_balance_objective=0.2,
        consistency_objective=0.3,
        proximal_objective=0.4,
        total_objective=1.5 + iteration_index,
        confident_voxel_count=10,
        foreground_assignment_count=4,
        background_assignment_count=6,
        foreground_fraction=0.4,
        state_identity_hash_before="a" * 64,
        state_identity_hash_after="b" * 64,
        prototype_identity_hash_before="c" * 64,
        prototype_identity_hash_after="d" * 64,
        convergence_delta=0.01,
        finite_status_ok=True,
        collapse_status_detected=collapse_status_detected,
    )


def _trace(
    *,
    iterations: tuple[ProtoEMIterationRecord, ...] | None = None,
) -> ProtoEMObjectiveTrace:
    actual_iterations = iterations or (_iteration(iteration_index=0), _iteration(iteration_index=1))
    trace_hash = sha256_json(
        {
            "schema_name": PROTOEM_OBJECTIVE_TRACE_SCHEMA_NAME,
            "schema_version": PROTOEM_OBJECTIVE_TRACE_SCHEMA_VERSION,
            "iteration_index_base": PROTOEM_ITERATION_INDEX_BASE,
            "iterations": [
                {
                    "schema_name": item.schema_name,
                    "schema_version": item.schema_version,
                    "iteration_index_base": item.iteration_index_base,
                    "iteration_index": item.iteration_index,
                    "support_objective": item.support_objective,
                    "query_entropy_objective": item.query_entropy_objective,
                    "class_balance_objective": item.class_balance_objective,
                    "consistency_objective": item.consistency_objective,
                    "proximal_objective": item.proximal_objective,
                    "total_objective": item.total_objective,
                    "confident_voxel_count": item.confident_voxel_count,
                    "foreground_assignment_count": item.foreground_assignment_count,
                    "background_assignment_count": item.background_assignment_count,
                    "foreground_fraction": item.foreground_fraction,
                    "state_identity_hash_before": item.state_identity_hash_before,
                    "state_identity_hash_after": item.state_identity_hash_after,
                    "prototype_identity_hash_before": item.prototype_identity_hash_before,
                    "prototype_identity_hash_after": item.prototype_identity_hash_after,
                    "convergence_delta": item.convergence_delta,
                    "finite_status_ok": item.finite_status_ok,
                    "collapse_status_detected": item.collapse_status_detected,
                }
                for item in actual_iterations
            ],
        }
    )
    return ProtoEMObjectiveTrace(
        schema_name=PROTOEM_OBJECTIVE_TRACE_SCHEMA_NAME,
        schema_version=PROTOEM_OBJECTIVE_TRACE_SCHEMA_VERSION,
        objective_trace_hash=trace_hash,
        iteration_index_base=PROTOEM_ITERATION_INDEX_BASE,
        iterations=actual_iterations,
    )


def _stopping(
    *,
    stop_reason: str = "tolerance_reached",
    converged: bool = True,
    failed: bool = False,
    failure_code: str | None = None,
    failure_message: str | None = None,
) -> ProtoEMStoppingRecord:
    stopping_hash = sha256_json(
        {
            "schema_name": PROTOEM_STOPPING_RECORD_SCHEMA_NAME,
            "schema_version": PROTOEM_STOPPING_RECORD_SCHEMA_VERSION,
            "stop_reason": stop_reason,
            "final_iteration_index": 1,
            "completed_iteration_count": 2,
            "converged": converged,
            "failed": failed,
            "failure_code": failure_code,
            "failure_message": failure_message,
        }
    )
    return ProtoEMStoppingRecord(
        schema_name=PROTOEM_STOPPING_RECORD_SCHEMA_NAME,
        schema_version=PROTOEM_STOPPING_RECORD_SCHEMA_VERSION,
        stopping_record_hash=stopping_hash,
        stop_reason=stop_reason,
        final_iteration_index=1,
        completed_iteration_count=2,
        converged=converged,
        failed=failed,
        failure_code=failure_code,
        failure_message=failure_message,
    )


def _collapse() -> ProtoEMCollapseRecord:
    collapse_hash = sha256_json(
        {
            "schema_name": PROTOEM_COLLAPSE_RECORD_SCHEMA_NAME,
            "schema_version": PROTOEM_COLLAPSE_RECORD_SCHEMA_VERSION,
            "collapse_type": "foreground_collapse",
            "iteration_index": 1,
            "foreground_count": 0,
            "background_count": 10,
            "confident_voxel_count": 10,
            "foreground_fraction": 0.0,
            "configured_lower_safeguard": 0.05,
            "configured_upper_safeguard": 0.95,
            "detected": True,
        }
    )
    return ProtoEMCollapseRecord(
        schema_name=PROTOEM_COLLAPSE_RECORD_SCHEMA_NAME,
        schema_version=PROTOEM_COLLAPSE_RECORD_SCHEMA_VERSION,
        collapse_record_hash=collapse_hash,
        collapse_type="foreground_collapse",
        iteration_index=1,
        foreground_count=0,
        background_count=10,
        confident_voxel_count=10,
        foreground_fraction=0.0,
        configured_lower_safeguard=0.05,
        configured_upper_safeguard=0.95,
        detected=True,
    )


def _ablation() -> ProtoEMAblationDefinition:
    override = ProtoEMAblationOverride(
        schema_name=PROTOEM_ABLATION_OVERRIDE_SCHEMA_NAME,
        schema_version=PROTOEM_ABLATION_OVERRIDE_SCHEMA_VERSION,
        field_name="objective_weights.class_balance",
        override_value=0.0,
    )
    ablation_hash = sha256_json(
        {
            "schema_name": PROTOEM_ABLATION_DEFINITION_SCHEMA_NAME,
            "schema_version": PROTOEM_ABLATION_DEFINITION_SCHEMA_VERSION,
            "variant_name": "no_class_balance",
            "overrides": [
                {
                    "schema_name": override.schema_name,
                    "schema_version": override.schema_version,
                    "field_name": override.field_name,
                    "override_value": override.override_value,
                }
            ],
        }
    )
    return ProtoEMAblationDefinition(
        schema_name=PROTOEM_ABLATION_DEFINITION_SCHEMA_NAME,
        schema_version=PROTOEM_ABLATION_DEFINITION_SCHEMA_VERSION,
        ablation_definition_hash=ablation_hash,
        variant_name="no_class_balance",
        overrides=(override,),
    )


def _summary(
    *,
    execution_status: str = "completed",
    failure_code: str | None = None,
    failure_message: str | None = None,
    final_output_artifact_hash: str | None = "9" * 64,
    final_inference_artifact_hash: str | None = "a" * 64,
    duration_seconds: float | None = None,
    memory_availability_status: str = "unavailable",
    peak_allocated_memory_bytes: int | None = None,
    peak_reserved_memory_bytes: int | None = None,
    peak_host_memory_bytes: int | None = None,
    collapse_record_hash: str | None = None,
) -> ProtoEMRunSummary:
    config = _config()
    trace = _trace()
    stopping = _stopping(
        stop_reason="tolerance_reached" if execution_status == "completed" else "numerical_failure",
        converged=execution_status == "completed",
        failed=execution_status == "failed",
        failure_code=failure_code,
        failure_message=failure_message,
    )
    ablation = _ablation()
    run_summary_hash = sha256_json(
        {
            "schema_name": PROTOEM_RUN_SUMMARY_SCHEMA_NAME,
            "schema_version": PROTOEM_RUN_SUMMARY_SCHEMA_VERSION,
            "config_hash": config.config_hash,
            "phase5_initialization_artifact_hash": config.phase5_initialization_artifact_hash,
            "phase5_prototype_artifact_hash": config.phase5_prototype_artifact_hash,
            "phase5_inference_artifact_hash": config.phase5_inference_artifact_hash,
            "objective_trace_hash": trace.objective_trace_hash,
            "stopping_record_hash": stopping.stopping_record_hash,
            "collapse_record_hash": collapse_record_hash,
            "ablation_definition_hash": ablation.ablation_definition_hash,
            "final_output_artifact_hash": final_output_artifact_hash,
            "final_inference_artifact_hash": final_inference_artifact_hash,
            "execution_status": execution_status,
            "failure_code": failure_code,
            "failure_message": failure_message,
        }
    )
    return ProtoEMRunSummary(
        schema_name=PROTOEM_RUN_SUMMARY_SCHEMA_NAME,
        schema_version=PROTOEM_RUN_SUMMARY_SCHEMA_VERSION,
        run_summary_hash=run_summary_hash,
        config_hash=config.config_hash,
        phase5_initialization_artifact_hash=config.phase5_initialization_artifact_hash,
        phase5_prototype_artifact_hash=config.phase5_prototype_artifact_hash,
        phase5_inference_artifact_hash=config.phase5_inference_artifact_hash,
        objective_trace_hash=trace.objective_trace_hash,
        stopping_record_hash=stopping.stopping_record_hash,
        collapse_record_hash=collapse_record_hash,
        ablation_definition_hash=ablation.ablation_definition_hash,
        final_output_artifact_hash=final_output_artifact_hash,
        final_inference_artifact_hash=final_inference_artifact_hash,
        execution_status=execution_status,
        failure_code=failure_code,
        failure_message=failure_message,
        duration_seconds=duration_seconds,
        memory_availability_status=memory_availability_status,
        peak_allocated_memory_bytes=peak_allocated_memory_bytes,
        peak_reserved_memory_bytes=peak_reserved_memory_bytes,
        peak_host_memory_bytes=peak_host_memory_bytes,
    )


def test_config_round_trip_and_hash_stability() -> None:
    config = _config()
    encoded = protoem_config_to_json(config)
    parsed = protoem_config_from_json(encoded)

    assert parsed == config
    assert hash_protoem_config(parsed) == config.config_hash
    assert protoem_artifact_from_json(encoded, ProtoEMConfig) == config
    assert encoded.endswith(b"\n")


@pytest.mark.parametrize(
    ("field_name", "builder"),
    [
        ("explicit_seed", lambda: _config(explicit_seed=1730)),
        ("max_iterations", lambda: _config(max_iterations=7)),
        ("minimum_iterations", lambda: _config(minimum_iterations=2)),
        ("convergence_tolerance", lambda: _config(convergence_tolerance=0.002)),
        ("temperature", lambda: _config(temperature=1.5)),
        ("confidence_threshold", lambda: _config(confidence_threshold=0.85)),
        ("foreground_prior", lambda: _config(foreground_prior=0.3)),
        ("update_schedule", lambda: _config(update_schedule="learned_positive_step")),
        ("prototype_mode", lambda: _config(prototype_mode="multiple_prototypes")),
        (
            "objective_weights.support",
            lambda: _config(
                weights=ProtoEMObjectiveWeights(
                    schema_name=PROTOEM_OBJECTIVE_WEIGHTS_SCHEMA_NAME,
                    schema_version=PROTOEM_OBJECTIVE_WEIGHTS_SCHEMA_VERSION,
                    support=1.1,
                    query_entropy=0.25,
                    class_balance=0.1,
                    consistency=0.5,
                    proximal=0.2,
                )
            ),
        ),
        (
            "objective_weights.query_entropy",
            lambda: _config(
                weights=ProtoEMObjectiveWeights(
                    schema_name=PROTOEM_OBJECTIVE_WEIGHTS_SCHEMA_NAME,
                    schema_version=PROTOEM_OBJECTIVE_WEIGHTS_SCHEMA_VERSION,
                    support=1.0,
                    query_entropy=0.35,
                    class_balance=0.1,
                    consistency=0.5,
                    proximal=0.2,
                )
            ),
        ),
        (
            "objective_weights.class_balance",
            lambda: _config(
                weights=ProtoEMObjectiveWeights(
                    schema_name=PROTOEM_OBJECTIVE_WEIGHTS_SCHEMA_NAME,
                    schema_version=PROTOEM_OBJECTIVE_WEIGHTS_SCHEMA_VERSION,
                    support=1.0,
                    query_entropy=0.25,
                    class_balance=0.2,
                    consistency=0.5,
                    proximal=0.2,
                )
            ),
        ),
        (
            "objective_weights.consistency",
            lambda: _config(
                weights=ProtoEMObjectiveWeights(
                    schema_name=PROTOEM_OBJECTIVE_WEIGHTS_SCHEMA_NAME,
                    schema_version=PROTOEM_OBJECTIVE_WEIGHTS_SCHEMA_VERSION,
                    support=1.0,
                    query_entropy=0.25,
                    class_balance=0.1,
                    consistency=0.6,
                    proximal=0.2,
                )
            ),
        ),
        (
            "objective_weights.proximal",
            lambda: _config(
                weights=ProtoEMObjectiveWeights(
                    schema_name=PROTOEM_OBJECTIVE_WEIGHTS_SCHEMA_NAME,
                    schema_version=PROTOEM_OBJECTIVE_WEIGHTS_SCHEMA_VERSION,
                    support=1.0,
                    query_entropy=0.25,
                    class_balance=0.1,
                    consistency=0.5,
                    proximal=0.3,
                )
            ),
        ),
        (
            "phase5_initialization_schema_name",
            lambda: _config(phase5_initialization_schema_name="phase5_run_summary"),
        ),
        (
            "phase5_initialization_artifact_hash",
            lambda: _config(phase5_initialization_artifact_hash="4" * 64),
        ),
        (
            "phase5_prototype_schema_name",
            lambda: _config(phase5_prototype_schema_name="retrieval_result"),
        ),
        (
            "phase5_prototype_artifact_hash",
            lambda: _config(phase5_prototype_artifact_hash="5" * 64),
        ),
        (
            "phase5_inference_schema_name",
            lambda: _config(phase5_inference_schema_name="prototype_memory"),
        ),
        (
            "phase5_inference_artifact_hash",
            lambda: _config(phase5_inference_artifact_hash="6" * 64),
        ),
    ],
)
def test_changing_compatibility_critical_field_changes_config_hash(
    field_name: str,
    builder: Callable[[], ProtoEMConfig],
) -> None:
    baseline = _config()
    changed = builder()

    assert field_name in COMPATIBILITY_CRITICAL_CONFIG_OVERRIDE_FIELDS
    assert changed.config_hash != baseline.config_hash


def test_invalid_objective_weights_rejected() -> None:
    with pytest.raises(ProtoEMArtifactValidationError):
        ProtoEMObjectiveWeights(
            schema_name=PROTOEM_OBJECTIVE_WEIGHTS_SCHEMA_NAME,
            schema_version=PROTOEM_OBJECTIVE_WEIGHTS_SCHEMA_VERSION,
            support=-0.1,
            query_entropy=0.0,
            class_balance=0.0,
            consistency=0.0,
            proximal=0.0,
        )


@pytest.mark.parametrize(
    "builder",
    [
        lambda: _config(max_iterations=0),
        lambda: _config(minimum_iterations=7),
        lambda: _config(temperature=0.0),
        lambda: _config(confidence_threshold=1.1),
        lambda: _config(foreground_prior=1.0),
        lambda: _config(convergence_tolerance=-0.01),
    ],
)
def test_invalid_thresholds_and_iteration_bounds_rejected(
    builder: Callable[[], ProtoEMConfig],
) -> None:
    with pytest.raises(ProtoEMArtifactValidationError):
        builder()


def test_iteration_trace_ordering_and_duplicate_rejection() -> None:
    with pytest.raises(ProtoEMArtifactValidationError):
        _trace(iterations=(_iteration(iteration_index=0), _iteration(iteration_index=0)))

    with pytest.raises(ProtoEMArtifactValidationError):
        _trace(iterations=(_iteration(iteration_index=0), _iteration(iteration_index=2)))


def test_non_finite_objective_rejection() -> None:
    with pytest.raises(ProtoEMArtifactValidationError):
        ProtoEMIterationRecord(
            schema_name=PROTOEM_ITERATION_RECORD_SCHEMA_NAME,
            schema_version=PROTOEM_ITERATION_RECORD_SCHEMA_VERSION,
            iteration_index_base=PROTOEM_ITERATION_INDEX_BASE,
            iteration_index=0,
            support_objective=math.nan,
            query_entropy_objective=0.1,
            class_balance_objective=0.2,
            consistency_objective=0.3,
            proximal_objective=0.4,
            total_objective=1.0,
            confident_voxel_count=10,
            foreground_assignment_count=4,
            background_assignment_count=6,
            foreground_fraction=0.4,
            state_identity_hash_before="a" * 64,
            state_identity_hash_after="b" * 64,
            prototype_identity_hash_before="c" * 64,
            prototype_identity_hash_after="d" * 64,
            convergence_delta=0.01,
            finite_status_ok=True,
            collapse_status_detected=False,
        )


def test_stop_reason_consistency() -> None:
    with pytest.raises(ProtoEMArtifactValidationError):
        _stopping(stop_reason="tolerance_reached", converged=False, failed=False)

    with pytest.raises(ProtoEMArtifactValidationError):
        _stopping(
            stop_reason="numerical_failure",
            converged=False,
            failed=False,
        )


def test_collapse_record_validation() -> None:
    collapse = _collapse()
    assert collapse.detected is True

    with pytest.raises(ProtoEMArtifactValidationError):
        ProtoEMCollapseRecord(
            schema_name=PROTOEM_COLLAPSE_RECORD_SCHEMA_NAME,
            schema_version=PROTOEM_COLLAPSE_RECORD_SCHEMA_VERSION,
            collapse_record_hash="0" * 64,
            collapse_type="foreground_collapse",
            iteration_index=1,
            foreground_count=0,
            background_count=10,
            confident_voxel_count=9,
            foreground_fraction=0.0,
            configured_lower_safeguard=0.95,
            configured_upper_safeguard=0.05,
            detected=True,
        )


def test_ablation_variant_validation() -> None:
    assert _ablation().variant_name == "no_class_balance"

    with pytest.raises(ProtoEMArtifactValidationError):
        ProtoEMAblationDefinition(
            schema_name=PROTOEM_ABLATION_DEFINITION_SCHEMA_NAME,
            schema_version=PROTOEM_ABLATION_DEFINITION_SCHEMA_VERSION,
            ablation_definition_hash="0" * 64,
            variant_name="invalid_variant",
            overrides=(),
        )


def test_run_summary_success_failure_consistency() -> None:
    completed = _summary()
    failed = _summary(
        execution_status="failed",
        failure_code="numerical_failure",
        failure_message="objective became non-finite",
        final_output_artifact_hash=None,
        final_inference_artifact_hash=None,
    )

    assert completed.execution_status == "completed"
    assert failed.execution_status == "failed"

    with pytest.raises(ProtoEMArtifactValidationError):
        _summary(
            execution_status="failed",
            failure_code="numerical_failure",
            failure_message="objective became non-finite",
            final_output_artifact_hash="9" * 64,
            final_inference_artifact_hash=None,
        )


def test_unknown_field_rejection() -> None:
    config = _config()
    payload = json.loads(protoem_config_to_json(config))
    payload["unexpected"] = True

    with pytest.raises(ProtoEMArtifactSerializationError):
        protoem_config_from_json(json.dumps(payload).encode("utf-8"))


def test_self_hash_mismatch_rejection() -> None:
    config = _config()
    payload = json.loads(protoem_config_to_json(config))
    payload["config_hash"] = "f" * 64

    with pytest.raises(ProtoEMArtifactHashError):
        protoem_config_from_json(json.dumps(payload).encode("utf-8"))


def test_canonical_byte_identical_serialization() -> None:
    config = _config()
    encoded = protoem_config_to_json(config)

    assert encoded == protoem_config_to_json(config)
    assert encoded == canonical_json_bytes(json.loads(encoded)) + b"\n"


def test_objective_trace_round_trip_and_identity_payload() -> None:
    trace = _trace()
    encoded = protoem_objective_trace_to_json(trace)
    parsed = protoem_objective_trace_from_json(encoded)

    assert parsed == trace
    assert hash_protoem_objective_trace(parsed) == trace.objective_trace_hash
    assert protoem_objective_trace_identity_payload(trace)["iteration_index_base"] == "zero_based"


def test_ablation_round_trip() -> None:
    ablation = _ablation()

    assert (
        protoem_ablation_definition_from_json(protoem_ablation_definition_to_json(ablation))
        == ablation
    )


def test_run_summary_round_trip_and_runtime_fields_not_in_identity() -> None:
    first = _summary(duration_seconds=1.0)
    second = _summary(duration_seconds=9.0)

    assert first.run_summary_hash == second.run_summary_hash
    assert protoem_run_summary_from_json(protoem_run_summary_to_json(first)) == first


def test_available_memory_requires_all_fields() -> None:
    with pytest.raises(ProtoEMArtifactValidationError):
        _summary(memory_availability_status="available", peak_allocated_memory_bytes=1)


def test_config_identity_payload_fields_are_exact() -> None:
    payload = protoem_config_identity_payload(_config())

    assert set(payload) == {
        "confidence_threshold",
        "convergence_tolerance",
        "explicit_seed",
        "foreground_prior",
        "max_iterations",
        "minimum_iterations",
        "objective_weights",
        "phase5_inference_artifact_hash",
        "phase5_inference_schema_name",
        "phase5_initialization_artifact_hash",
        "phase5_initialization_schema_name",
        "phase5_prototype_artifact_hash",
        "phase5_prototype_schema_name",
        "prototype_mode",
        "schema_name",
        "schema_version",
        "temperature",
        "update_schedule",
    }
