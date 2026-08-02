"""Unit tests for deterministic ProtoEM ablation planning and result wiring."""

from __future__ import annotations

import inspect

import pytest

from protoem_ct.artifacts.hashing import sha256_json
from protoem_ct.protoem import (
    PROTOEM_ABLATION_VARIANT_ORDER,
    AblationPlanError,
    ProtoEMAblationExecutionPlan,
    ProtoEMConfig,
    ProtoEMObjectiveWeights,
    apply_protoem_ablation_to_config,
    build_protoem_ablation_definition,
    build_protoem_ablation_execution_plan,
    build_protoem_ablation_execution_result,
    query_label_not_part_of_ablation_api,
)


def _weights() -> ProtoEMObjectiveWeights:
    return ProtoEMObjectiveWeights(
        schema_name="protoem_objective_weights",
        schema_version="v1",
        support=1.0,
        query_entropy=0.5,
        class_balance=0.25,
        consistency=0.75,
        proximal=0.125,
    )


def _config() -> ProtoEMConfig:
    weights = _weights()
    config_hash = sha256_json(
        {
            "schema_name": "protoem_config",
            "schema_version": "v1",
            "explicit_seed": 11,
            "max_iterations": 4,
            "minimum_iterations": 1,
            "convergence_tolerance": 0.0,
            "temperature": 1.0,
            "confidence_threshold": 0.6,
            "foreground_prior": 0.4,
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
            "phase5_initialization_schema_name": "prototype_inference_summary",
            "phase5_initialization_artifact_hash": "1" * 64,
            "phase5_prototype_schema_name": "prototype_memory",
            "phase5_prototype_artifact_hash": "2" * 64,
            "phase5_inference_schema_name": "phase5_run_summary",
            "phase5_inference_artifact_hash": "3" * 64,
        }
    )
    return ProtoEMConfig(
        schema_name="protoem_config",
        schema_version="v1",
        config_hash=config_hash,
        explicit_seed=11,
        max_iterations=4,
        minimum_iterations=1,
        convergence_tolerance=0.0,
        temperature=1.0,
        confidence_threshold=0.6,
        foreground_prior=0.4,
        update_schedule="fixed_em_like",
        prototype_mode="single_prototype",
        objective_weights=weights,
        phase5_initialization_schema_name="prototype_inference_summary",
        phase5_initialization_artifact_hash="1" * 64,
        phase5_prototype_schema_name="prototype_memory",
        phase5_prototype_artifact_hash="2" * 64,
        phase5_inference_schema_name="phase5_run_summary",
        phase5_inference_artifact_hash="3" * 64,
    )


def test_exact_12_variant_ablation_order() -> None:
    plan = build_protoem_ablation_execution_plan()

    assert isinstance(plan, ProtoEMAblationExecutionPlan)
    assert tuple(entry.ablation_definition.variant_name for entry in plan.entries) == (
        PROTOEM_ABLATION_VARIANT_ORDER
    )


def test_exact_override_mapping_for_every_variant() -> None:
    expected = {
        "full_protoem": (),
        "no_retrieval": (),
        "no_transduction": (),
        "no_class_balance": (("objective_weights.class_balance", 0.0),),
        "no_proximal": (("objective_weights.proximal", 0.0),),
        "single_prototype": (("prototype_mode", "single_prototype"),),
        "multiple_prototypes": (("prototype_mode", "multiple_prototypes"),),
        "fixed_update_schedule": (("update_schedule", "fixed_em_like"),),
        "learned_update_schedule": (("update_schedule", "learned_positive_step"),),
        "head_only": (),
        "decoder_only": (),
        "full_finetune": (),
    }
    for variant_name, expected_overrides in expected.items():
        definition = build_protoem_ablation_definition(variant_name=variant_name)
        observed = tuple((item.field_name, item.override_value) for item in definition.overrides)
        assert observed == expected_overrides


def test_no_class_balance_sets_only_class_balance_weight_to_zero() -> None:
    config = _config()
    definition = build_protoem_ablation_definition(variant_name="no_class_balance")

    updated = apply_protoem_ablation_to_config(config=config, definition=definition)

    assert updated.objective_weights.class_balance == 0.0
    assert updated.objective_weights.proximal == config.objective_weights.proximal
    assert updated.objective_weights.support == config.objective_weights.support
    assert updated.update_schedule == config.update_schedule
    assert updated.prototype_mode == config.prototype_mode


def test_no_proximal_sets_only_proximal_weight_to_zero() -> None:
    config = _config()
    definition = build_protoem_ablation_definition(variant_name="no_proximal")

    updated = apply_protoem_ablation_to_config(config=config, definition=definition)

    assert updated.objective_weights.proximal == 0.0
    assert updated.objective_weights.class_balance == config.objective_weights.class_balance
    assert updated.objective_weights.query_entropy == config.objective_weights.query_entropy


def test_no_transduction_links_to_phase5_inference_without_running_protoem() -> None:
    config = _config()
    plan = build_protoem_ablation_execution_plan()
    entry = next(
        item for item in plan.entries if item.ablation_definition.variant_name == "no_transduction"
    )

    result = build_protoem_ablation_execution_result(
        plan_entry=entry,
        base_config=config,
        phase5_reference_schema_name="phase5_run_summary",
        phase5_reference_artifact_hash="a" * 64,
    )

    assert result.result_mode == "phase5_baseline"
    assert result.execution_package is None
    assert result.plan_entry.phase5_baseline_kind == "prototype_only_inference"
    assert result.phase5_reference_schema_name == "phase5_run_summary"


def test_no_retrieval_links_to_phase5_no_retrieval_artifact() -> None:
    config = _config()
    plan = build_protoem_ablation_execution_plan()
    entry = next(
        item for item in plan.entries if item.ablation_definition.variant_name == "no_retrieval"
    )

    result = build_protoem_ablation_execution_result(
        plan_entry=entry,
        base_config=config,
        phase5_reference_schema_name="phase5_comparison_record",
        phase5_reference_artifact_hash="b" * 64,
    )

    assert result.result_mode == "phase5_baseline"
    assert result.plan_entry.phase5_method_name == "no_retrieval"
    assert result.phase5_reference_schema_name == "phase5_comparison_record"


def test_learned_update_schedule_becomes_protoem_executable_in_plan() -> None:
    plan = build_protoem_ablation_execution_plan()
    entry = next(
        item
        for item in plan.entries
        if item.ablation_definition.variant_name == "learned_update_schedule"
    )
    assert entry.execution_mode == "protoem_execution"
    assert entry.unsupported_code is None
    assert entry.unsupported_message is None


def test_multiple_prototypes_remains_unsupported_if_current_path_cannot_execute() -> None:
    config = _config()
    plan = build_protoem_ablation_execution_plan()
    entry = next(
        item
        for item in plan.entries
        if item.ablation_definition.variant_name == "multiple_prototypes"
    )

    result = build_protoem_ablation_execution_result(plan_entry=entry, base_config=config)

    assert result.result_mode == "unsupported"
    assert result.effective_config is not None
    assert result.effective_config.prototype_mode == "multiple_prototypes"
    assert result.unsupported_code == "multiple_prototypes_unsupported"


def test_head_decoder_and_full_finetune_remain_provenance_only() -> None:
    config = _config()
    plan = build_protoem_ablation_execution_plan()
    for variant_name in ("head_only", "decoder_only", "full_finetune"):
        entry = next(
            item for item in plan.entries if item.ablation_definition.variant_name == variant_name
        )
        result = build_protoem_ablation_execution_result(plan_entry=entry, base_config=config)
        assert result.result_mode == "unsupported"
        assert result.unsupported_code == f"{variant_name}_provenance_only"


def test_duplicate_variant_rejection() -> None:
    plan = build_protoem_ablation_execution_plan()
    duplicate_entries = (plan.entries[0], plan.entries[0], *plan.entries[2:])
    payload = {
        "schema_name": "protoem_ablation_execution_plan",
        "schema_version": "v1",
        "entries": [],
    }
    with pytest.raises(AblationPlanError):
        ProtoEMAblationExecutionPlan(
            schema_name="protoem_ablation_execution_plan",
            schema_version="v1",
            plan_identity_hash=sha256_json(payload),
            entries=duplicate_entries,
        )


def test_unknown_override_rejection() -> None:
    with pytest.raises(AblationPlanError):
        build_protoem_ablation_definition(variant_name="unknown_variant")


def test_deterministic_plan_identity() -> None:
    first = build_protoem_ablation_execution_plan()
    second = build_protoem_ablation_execution_plan()

    assert first.plan_identity_hash == second.plan_identity_hash


def test_query_labels_reference_masks_absent_from_public_apis() -> None:
    assert query_label_not_part_of_ablation_api() is True
    for callable_object in (
        build_protoem_ablation_definition,
        build_protoem_ablation_execution_plan,
        build_protoem_ablation_execution_result,
    ):
        signature_text = str(inspect.signature(callable_object))
        assert "query_label" not in signature_text
        assert "reference_mask" not in signature_text
