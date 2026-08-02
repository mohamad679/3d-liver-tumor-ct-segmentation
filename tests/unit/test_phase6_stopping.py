"""Unit tests for deterministic ProtoEM stopping and collapse decisions."""

from __future__ import annotations

import inspect

import pytest

from protoem_ct.artifacts.hashing import sha256_json
from protoem_ct.protoem import (
    PROTOEM_CONFIG_SCHEMA_NAME,
    PROTOEM_CONFIG_SCHEMA_VERSION,
    PROTOEM_ITERATION_INDEX_BASE,
    PROTOEM_ITERATION_RECORD_SCHEMA_NAME,
    PROTOEM_ITERATION_RECORD_SCHEMA_VERSION,
    PROTOEM_OBJECTIVE_WEIGHTS_SCHEMA_NAME,
    PROTOEM_OBJECTIVE_WEIGHTS_SCHEMA_VERSION,
    CollapseConsistencyError,
    ProtoEMCollapseDecision,
    ProtoEMConfidenceMaskResult,
    ProtoEMConfig,
    ProtoEMIterationRecord,
    ProtoEMObjectiveWeights,
    ProtoEMStoppingDecision,
    build_failure_stopping_decision,
    evaluate_protoem_collapse,
    evaluate_protoem_stopping_after_iteration,
)


def _weights() -> ProtoEMObjectiveWeights:
    return ProtoEMObjectiveWeights(
        schema_name=PROTOEM_OBJECTIVE_WEIGHTS_SCHEMA_NAME,
        schema_version=PROTOEM_OBJECTIVE_WEIGHTS_SCHEMA_VERSION,
        support=1.0,
        query_entropy=1.0,
        class_balance=1.0,
        consistency=1.0,
        proximal=1.0,
    )


def _config(*, max_iterations: int = 4, minimum_iterations: int = 2) -> ProtoEMConfig:
    weights = _weights()
    payload = {
        "schema_name": PROTOEM_CONFIG_SCHEMA_NAME,
        "schema_version": PROTOEM_CONFIG_SCHEMA_VERSION,
        "explicit_seed": 1729,
        "max_iterations": max_iterations,
        "minimum_iterations": minimum_iterations,
        "convergence_tolerance": 0.0,
        "temperature": 1.0,
        "confidence_threshold": 0.0,
        "foreground_prior": 0.5,
        "update_schedule": "fixed_em_like",
        "prototype_mode": "single_prototype",
        "objective_weights": {
            "schema_name": weights.schema_name,
            "schema_version": weights.schema_version,
            "support": weights.support,
            "query_entropy": weights.query_entropy,
            "class_balance": weights.class_balance,
            "consistency": weights.consistency,
            "proximal": weights.proximal,
        },
        "phase5_initialization_schema_name": "phase5_run_summary",
        "phase5_initialization_artifact_hash": "1" * 64,
        "phase5_prototype_schema_name": "prototype_memory",
        "phase5_prototype_artifact_hash": "2" * 64,
        "phase5_inference_schema_name": "prototype_only_inference_result",
        "phase5_inference_artifact_hash": "3" * 64,
    }
    return ProtoEMConfig(
        schema_name=PROTOEM_CONFIG_SCHEMA_NAME,
        schema_version=PROTOEM_CONFIG_SCHEMA_VERSION,
        config_hash=sha256_json(payload),
        explicit_seed=1729,
        max_iterations=max_iterations,
        minimum_iterations=minimum_iterations,
        convergence_tolerance=0.0,
        temperature=1.0,
        confidence_threshold=0.0,
        foreground_prior=0.5,
        update_schedule="fixed_em_like",
        prototype_mode="single_prototype",
        objective_weights=weights,
        phase5_initialization_schema_name="phase5_run_summary",
        phase5_initialization_artifact_hash="1" * 64,
        phase5_prototype_schema_name="prototype_memory",
        phase5_prototype_artifact_hash="2" * 64,
        phase5_inference_schema_name="prototype_only_inference_result",
        phase5_inference_artifact_hash="3" * 64,
    )


def _confidence_mask_result(
    *,
    confident_voxel_count: int,
    foreground_assignment_count: int,
    background_assignment_count: int,
    foreground_fraction: float,
) -> ProtoEMConfidenceMaskResult:
    if confident_voxel_count == 0:
        confidence_map = [[[[[0.0]]]]]
        confident_mask = [[[[[0]]]]]
    else:
        confidence_map = [[[[[1.0]]]]]
        confident_mask = [[[[[1]]]]]
    return ProtoEMConfidenceMaskResult(
        schema_name="protoem_confidence_mask_result",
        schema_version="v1",
        confidence_map=__import__("numpy").asarray(confidence_map, dtype=float),
        confident_mask=__import__("numpy").asarray(confident_mask, dtype=bool),
        confident_voxel_count=confident_voxel_count,
        foreground_assignment_count=foreground_assignment_count,
        background_assignment_count=background_assignment_count,
        foreground_fraction=foreground_fraction,
    )


def _iteration(*, iteration_index: int, convergence_delta: float) -> ProtoEMIterationRecord:
    return ProtoEMIterationRecord(
        schema_name=PROTOEM_ITERATION_RECORD_SCHEMA_NAME,
        schema_version=PROTOEM_ITERATION_RECORD_SCHEMA_VERSION,
        iteration_index_base=PROTOEM_ITERATION_INDEX_BASE,
        iteration_index=iteration_index,
        support_objective=0.1,
        query_entropy_objective=0.2,
        class_balance_objective=0.3,
        consistency_objective=0.4,
        proximal_objective=0.5,
        total_objective=1.5,
        confident_voxel_count=10,
        foreground_assignment_count=4,
        background_assignment_count=6,
        foreground_fraction=0.4,
        state_identity_hash_before="a" * 64,
        state_identity_hash_after="b" * 64,
        prototype_identity_hash_before="c" * 64,
        prototype_identity_hash_after="d" * 64,
        convergence_delta=convergence_delta,
        finite_status_ok=True,
        collapse_status_detected=False,
    )


class _State:
    def __init__(self, iteration_index: int, convergence_delta: float | None) -> None:
        self.iteration_index = iteration_index
        self.convergence_delta = convergence_delta


def test_collapse_record_consistency() -> None:
    decision = evaluate_protoem_collapse(
        iteration_index=0,
        confidence_mask_result=_confidence_mask_result(
            confident_voxel_count=1,
            foreground_assignment_count=1,
            background_assignment_count=0,
            foreground_fraction=1.0,
        ),
    )

    assert isinstance(decision, ProtoEMCollapseDecision)
    assert decision.detected is True
    assert decision.stop_reason == "background_collapse"
    assert decision.collapse_record is not None
    assert decision.collapse_record.background_count == 0


def test_no_confident_voxel_decision_has_no_collapse_record() -> None:
    decision = evaluate_protoem_collapse(
        iteration_index=0,
        confidence_mask_result=_confidence_mask_result(
            confident_voxel_count=0,
            foreground_assignment_count=0,
            background_assignment_count=0,
            foreground_fraction=0.0,
        ),
    )

    assert decision.hard_stop is True
    assert decision.stop_reason == "no_confident_voxels"
    assert decision.detected is False
    assert decision.collapse_record is None


def test_stopping_record_consistency_for_tolerance() -> None:
    decision = evaluate_protoem_stopping_after_iteration(
        config=_config(max_iterations=4, minimum_iterations=2),
        completed_records=(
            _iteration(iteration_index=0, convergence_delta=0.0),
            _iteration(iteration_index=1, convergence_delta=0.0),
        ),
        current_state=_State(iteration_index=2, convergence_delta=0.0),  # type: ignore[arg-type]
    )

    assert isinstance(decision, ProtoEMStoppingDecision)
    assert decision.should_stop is True
    assert decision.stop_reason == "tolerance_reached"
    assert decision.stopping_record is not None
    assert decision.stopping_record.completed_iteration_count == 2


def test_failure_stopping_decision_omits_record_for_zero_completed_iterations() -> None:
    decision = build_failure_stopping_decision(
        stop_reason="numerical_failure",
        completed_records=(),
        failure_code="numerical_failure",
        failure_message="Synthetic failure.",
    )

    assert decision.should_stop is True
    assert decision.failed is True
    assert decision.stopping_record is None


def test_collapse_decision_validation_rejects_inconsistent_counts() -> None:
    with pytest.raises(CollapseConsistencyError):
        ProtoEMCollapseDecision(
            schema_name="protoem_collapse_decision",
            schema_version="v1",
            decision_identity_hash="0" * 64,
            iteration_index=0,
            hard_stop=False,
            stop_reason=None,
            collapse_type=None,
            confident_voxel_count=1,
            foreground_assignment_count=1,
            background_assignment_count=1,
            foreground_fraction=1.0,
            configured_lower_safeguard=0.0,
            configured_upper_safeguard=1.0,
            detected=False,
            collapse_record=None,
        )


def test_query_labels_reference_masks_absent_from_public_apis() -> None:
    for func in (
        build_failure_stopping_decision,
        evaluate_protoem_collapse,
        evaluate_protoem_stopping_after_iteration,
    ):
        signature_text = str(inspect.signature(func)).lower()
        assert "query_label" not in signature_text
        assert "reference_mask" not in signature_text
    assert "query_label" not in ProtoEMStoppingDecision.__dataclass_fields__
    assert "reference_mask" not in ProtoEMStoppingDecision.__dataclass_fields__
    assert "query_label" not in ProtoEMCollapseDecision.__dataclass_fields__
    assert "reference_mask" not in ProtoEMCollapseDecision.__dataclass_fields__
