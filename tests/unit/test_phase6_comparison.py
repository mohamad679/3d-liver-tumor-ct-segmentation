"""Unit tests for deterministic Phase 6 ablation comparison artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from protoem_ct.protoem import (
    PHASE6_ABLATION_COMPARISON_JSON_NAME,
    PROTOEM_ABLATION_COMPARISON_RECORD_SCHEMA_NAME,
    PROTOEM_ABLATION_COMPARISON_TABLE_SCHEMA_NAME,
    PROTOEM_ABLATION_RUN_INVENTORY_SCHEMA_NAME,
    PROTOEM_ABLATION_VARIANT_ORDER,
    Phase6PositiveStepScheduleSettings,
    Phase6PublicationSettings,
    build_protoem_ablation_comparison_table,
    build_protoem_ablation_run_inventory,
    load_phase6_protoem_settings,
    protoem_ablation_comparison_table_to_json,
    render_protoem_ablation_comparison_markdown_from_json,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _default_settings() -> Phase6PublicationSettings:
    return load_phase6_protoem_settings(REPO_ROOT / "configs" / "phase6_protoem_ct.yaml")


def _positive_step_settings() -> Phase6PublicationSettings:
    base = _default_settings()
    return Phase6PublicationSettings(
        schema_version=base.schema_version,
        synthetic_mode_only=base.synthetic_mode_only,
        explicit_seed=base.explicit_seed,
        max_iterations=base.max_iterations,
        minimum_iterations=base.minimum_iterations,
        convergence_tolerance=base.convergence_tolerance,
        temperature=base.temperature,
        confidence_threshold=base.confidence_threshold,
        foreground_prior=base.foreground_prior,
        update_schedule="learned_positive_step",
        prototype_mode=base.prototype_mode,
        objective_weights=base.objective_weights,
        phase5_initialization_schema_name=base.phase5_initialization_schema_name,
        phase5_prototype_schema_name=base.phase5_prototype_schema_name,
        phase5_inference_schema_name=base.phase5_inference_schema_name,
        positive_step_schedule=Phase6PositiveStepScheduleSettings(
            epsilon=1.0e-6,
            raw_step_parameters=(0.0,) * base.max_iterations,
        ),
    )


def test_exact_12_row_canonical_order() -> None:
    table = build_protoem_ablation_comparison_table(settings=_default_settings())

    assert table.schema_name == PROTOEM_ABLATION_COMPARISON_TABLE_SCHEMA_NAME
    assert tuple(record.variant_name for record in table.records) == PROTOEM_ABLATION_VARIANT_ORDER


def test_full_protoem_executes() -> None:
    table = build_protoem_ablation_comparison_table(settings=_default_settings())
    record = next(item for item in table.records if item.variant_name == "full_protoem")

    assert record.execution_status == "executed"
    assert record.config_hash is not None
    assert record.initialization_identity_hash is not None
    assert record.objective_trace_hash is not None
    assert record.final_inference_identity_hash is not None
    assert record.metrics_availability_status == "available"


def test_no_retrieval_links_phase5_artifact() -> None:
    table = build_protoem_ablation_comparison_table(settings=_default_settings())
    record = next(item for item in table.records if item.variant_name == "no_retrieval")

    assert record.execution_status == "phase5_baseline_link"
    assert record.baseline_artifact_hash is not None
    assert record.final_inference_identity_hash is not None
    assert record.config_hash is None


def test_no_transduction_links_phase5_inference() -> None:
    table = build_protoem_ablation_comparison_table(settings=_default_settings())
    record = next(item for item in table.records if item.variant_name == "no_transduction")

    assert record.execution_status == "phase5_baseline_link"
    assert record.baseline_artifact_hash == record.final_inference_identity_hash


def test_no_class_balance_changes_only_its_intended_weight() -> None:
    table = build_protoem_ablation_comparison_table(settings=_default_settings())
    full_record = next(item for item in table.records if item.variant_name == "full_protoem")
    no_balance_record = next(
        item for item in table.records if item.variant_name == "no_class_balance"
    )

    assert no_balance_record.execution_status == "executed"
    assert no_balance_record.config_hash != full_record.config_hash
    assert no_balance_record.final_inference_identity_hash is not None


def test_no_proximal_changes_only_its_intended_weight() -> None:
    table = build_protoem_ablation_comparison_table(settings=_default_settings())
    full_record = next(item for item in table.records if item.variant_name == "full_protoem")
    no_proximal_record = next(item for item in table.records if item.variant_name == "no_proximal")

    assert no_proximal_record.execution_status == "executed"
    assert no_proximal_record.config_hash != full_record.config_hash


def test_fixed_update_executes() -> None:
    table = build_protoem_ablation_comparison_table(settings=_default_settings())
    record = next(item for item in table.records if item.variant_name == "fixed_update_schedule")

    assert record.execution_status == "executed"
    assert record.reason_code is None


def test_explicit_positive_step_schedule_executes() -> None:
    table = build_protoem_ablation_comparison_table(settings=_positive_step_settings())
    record = next(item for item in table.records if item.variant_name == "learned_update_schedule")

    assert record.execution_status == "executed"
    assert record.config_hash is not None
    assert record.reason_code is None


def test_missing_positive_step_schedule_is_explicitly_unsupported() -> None:
    table = build_protoem_ablation_comparison_table(settings=_default_settings())
    record = next(item for item in table.records if item.variant_name == "learned_update_schedule")

    assert record.execution_status == "unsupported"
    assert record.reason_code == "missing_positive_step_schedule"


def test_unsupported_multiple_prototype_status_remains_explicit() -> None:
    table = build_protoem_ablation_comparison_table(settings=_default_settings())
    record = next(item for item in table.records if item.variant_name == "multiple_prototypes")

    assert record.execution_status == "unsupported"
    assert record.reason_code == "multiple_prototypes_unsupported"


def test_adaptation_modes_remain_provenance_only() -> None:
    table = build_protoem_ablation_comparison_table(settings=_default_settings())

    for variant_name in ("head_only", "decoder_only", "full_finetune"):
        record = next(item for item in table.records if item.variant_name == variant_name)
        assert record.execution_status == "provenance_only"
        assert record.reason_code == f"{variant_name}_provenance_only"


def test_failed_variant_is_retained(monkeypatch: pytest.MonkeyPatch) -> None:
    from protoem_ct.protoem import comparison as comparison_module

    original = comparison_module._execute_variant_package

    def failing_execute_variant_package(
        settings: Phase6PublicationSettings,
    ) -> object:
        if settings.objective_weights.class_balance == 0.0:
            raise RuntimeError("forced comparison failure")
        return original(settings)

    monkeypatch.setattr(
        comparison_module,
        "_execute_variant_package",
        failing_execute_variant_package,
    )

    table = build_protoem_ablation_comparison_table(settings=_default_settings())
    record = next(item for item in table.records if item.variant_name == "no_class_balance")

    assert record.execution_status == "failed"
    assert record.reason_code == "execution_failure"
    assert "forced comparison failure" in (record.reason_message or "")


def test_metrics_unavailable_when_prediction_absent() -> None:
    table = build_protoem_ablation_comparison_table(settings=_default_settings())
    record = next(item for item in table.records if item.variant_name == "multiple_prototypes")

    assert record.metrics_availability_status == "unavailable"
    assert record.dice is None
    assert record.iou is None


def test_reference_mask_changes_only_metric_fields() -> None:
    base = build_protoem_ablation_comparison_table(settings=_default_settings())
    altered = build_protoem_ablation_comparison_table(
        settings=_default_settings(),
        reference_mask=np.asarray([[[[[0, 0, 0]]]]], dtype=np.uint8),
    )
    base_record = next(item for item in base.records if item.variant_name == "full_protoem")
    altered_record = next(item for item in altered.records if item.variant_name == "full_protoem")

    assert base_record.final_inference_identity_hash == altered_record.final_inference_identity_hash
    assert base_record.final_prediction_content_hash == altered_record.final_prediction_content_hash
    assert base_record.objective_trace_hash == altered_record.objective_trace_hash
    assert (base_record.dice, base_record.iou) != (altered_record.dice, altered_record.iou)


def test_deterministic_order_and_identity() -> None:
    first = build_protoem_ablation_comparison_table(settings=_default_settings())
    second = build_protoem_ablation_comparison_table(settings=_default_settings())

    assert first.comparison_table_hash == second.comparison_table_hash
    assert protoem_ablation_comparison_table_to_json(
        first
    ) == protoem_ablation_comparison_table_to_json(second)


def test_markdown_is_derived_from_json() -> None:
    table = build_protoem_ablation_comparison_table(settings=_default_settings())
    payload = protoem_ablation_comparison_table_to_json(table)
    markdown = render_protoem_ablation_comparison_markdown_from_json(payload).decode("utf-8")

    assert PHASE6_ABLATION_COMPARISON_JSON_NAME.endswith(".json")
    assert "full_protoem" in markdown
    assert table.records[0].execution_status in markdown


def test_inventory_reuses_canonical_records() -> None:
    table = build_protoem_ablation_comparison_table(settings=_default_settings())
    inventory = build_protoem_ablation_run_inventory(comparison_table=table)

    assert inventory.schema_name == PROTOEM_ABLATION_RUN_INVENTORY_SCHEMA_NAME
    assert (
        tuple(record.variant_name for record in inventory.records) == PROTOEM_ABLATION_VARIANT_ORDER
    )


def test_record_schema_name_constant_is_stable() -> None:
    assert PROTOEM_ABLATION_COMPARISON_RECORD_SCHEMA_NAME == "protoem_ablation_comparison_record"
