"""Integration tests for the synthetic Phase 6 transductive flow."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pytest

from protoem_ct.protoem import (
    PHASE6_EFFECTIVE_CONFIG_NAME,
    PHASE6_FINAL_INFERENCE_NAME,
    PHASE6_SUMMARY_MARKDOWN_NAME,
    InvalidEStepInputError,
    Phase6PublicationSettings,
    ProtoEMExecutionPackage,
    ProtoEMOptimizationResult,
    build_phase6_synthetic_execution_package,
    load_phase6_protoem_settings,
    run_and_publish_phase6_protoem,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _default_settings() -> Phase6PublicationSettings:
    return load_phase6_protoem_settings(REPO_ROOT / "configs" / "phase6_protoem_ct.yaml")


def _two_iteration_failure_settings() -> Phase6PublicationSettings:
    base = _default_settings()
    return Phase6PublicationSettings(
        schema_version=base.schema_version,
        synthetic_mode_only=base.synthetic_mode_only,
        explicit_seed=base.explicit_seed,
        max_iterations=2,
        minimum_iterations=base.minimum_iterations,
        convergence_tolerance=base.convergence_tolerance,
        temperature=base.temperature,
        confidence_threshold=base.confidence_threshold,
        foreground_prior=base.foreground_prior,
        update_schedule=base.update_schedule,
        prototype_mode=base.prototype_mode,
        objective_weights=base.objective_weights,
        phase5_initialization_schema_name=base.phase5_initialization_schema_name,
        phase5_prototype_schema_name=base.phase5_prototype_schema_name,
        phase5_inference_schema_name=base.phase5_inference_schema_name,
        positive_step_schedule=base.positive_step_schedule,
    )


def test_objective_trace_contains_all_terms_and_contiguous_order() -> None:
    package = build_phase6_synthetic_execution_package(_default_settings())

    assert package.objective_trace is not None
    indices = [record.iteration_index for record in package.objective_trace.iterations]
    assert indices == list(range(len(indices)))
    for record in package.objective_trace.iterations:
        assert record.support_objective >= 0.0
        assert record.query_entropy_objective >= 0.0
        assert record.class_balance_objective >= 0.0
        assert record.consistency_objective >= 0.0
        assert record.proximal_objective >= 0.0


def test_final_inference_only_present_for_completed_run() -> None:
    package = build_phase6_synthetic_execution_package(_default_settings())

    assert package.optimization_result.execution_status == "completed"
    assert package.final_inference_result is not None


def test_failure_path_does_not_fabricate_inference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from protoem_ct.protoem.objective import run_protoem_e_step as original_e_step

    call_count = {"count": 0}

    def failing_second_e_step(**kwargs: Any) -> Any:
        call_count["count"] += 1
        if call_count["count"] == 2:
            raise InvalidEStepInputError("forced second-iteration failure")
        return original_e_step(**kwargs)

    monkeypatch.setattr(
        "protoem_ct.protoem.optimize.run_protoem_e_step",
        failing_second_e_step,
    )
    result = run_and_publish_phase6_protoem(
        output_root=tmp_path / "published",
        settings=_two_iteration_failure_settings(),
    )

    assert result.execution_status == "failed"
    assert result.final_inference_path is None
    assert not (tmp_path / "published" / PHASE6_FINAL_INFERENCE_NAME).exists()
    assert (tmp_path / "published" / PHASE6_EFFECTIVE_CONFIG_NAME).exists()
    assert (tmp_path / "published" / PHASE6_SUMMARY_MARKDOWN_NAME).exists()


def test_query_label_and_reference_mask_are_isolated_from_public_apis() -> None:
    for callable_object in (
        build_phase6_synthetic_execution_package,
        run_and_publish_phase6_protoem,
    ):
        signature_text = str(inspect.signature(callable_object)).lower()
        assert "query_label" not in signature_text
        assert "reference_mask" not in signature_text
    assert "query_label" not in ProtoEMExecutionPackage.__dataclass_fields__
    assert "reference_mask" not in ProtoEMOptimizationResult.__dataclass_fields__
