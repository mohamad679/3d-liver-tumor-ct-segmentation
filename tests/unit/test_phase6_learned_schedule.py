"""Unit tests for deterministic positive-step schedule handling."""

from __future__ import annotations

import inspect
import math
from importlib import import_module
from typing import Any

import numpy as np
import pytest

from protoem_ct.protoem import (
    InvalidPositiveStepScheduleError,
    NonFiniteStepParameterError,
    OrchestrationInputError,
    ProtoEMPositiveStepSchedule,
    ScheduleLengthMismatchError,
    apply_positive_step_to_m_step_result,
    build_initial_protoem_transductive_state,
    build_protoem_positive_step_schedule,
    effective_positive_step,
    run_protoem_e_step,
    run_protoem_m_step,
    run_protoem_optimization,
)


def _optimize_helpers() -> Any:
    for module_name in ("tests.unit.test_phase6_optimize", "test_phase6_optimize"):
        try:
            return import_module(module_name)
        except ModuleNotFoundError:
            continue
    raise ModuleNotFoundError("Unable to import test_phase6_optimize helpers.")


def _config_override(*args: Any, **kwargs: Any) -> Any:
    return _optimize_helpers()._config_override(*args, **kwargs)


def _config_with_threshold(*args: Any, **kwargs: Any) -> Any:
    return _optimize_helpers()._config_with_threshold(*args, **kwargs)


def _initialization_bundle(*args: Any, **kwargs: Any) -> Any:
    return _optimize_helpers()._initialization_bundle(*args, **kwargs)


def _schedule_for_config(*args: Any, **kwargs: Any) -> Any:
    return _optimize_helpers()._schedule_for_config(*args, **kwargs)


def _single_iteration_fixed_result() -> tuple[Any, Any, Any]:
    bundle, _ = _initialization_bundle()
    config = _config_override(
        _config_with_threshold(0.0),
        max_iterations=1,
        minimum_iterations=1,
    )
    return bundle, config, run_protoem_optimization(config=config, initialization_bundle=bundle)


def _single_iteration_target_inputs() -> tuple[Any, Any, Any, Any, Any]:
    bundle, _ = _initialization_bundle()
    config = _config_override(
        _config_with_threshold(0.0),
        max_iterations=1,
        minimum_iterations=1,
    )
    initial_state = build_initial_protoem_transductive_state(
        initialization_bundle=bundle,
        config=config,
    )
    query_features = np.asarray(
        bundle.query_feature_encoding.feature_data,
        dtype=np.float64,
    )
    support_foreground = np.asarray(bundle.foreground_prototype.prototype_vector, dtype=np.float64)
    support_background = np.asarray(bundle.background_prototype.prototype_vector, dtype=np.float64)
    e_step_result = run_protoem_e_step(
        query_feature_tensor=query_features,
        foreground_prototypes=initial_state.prototype_state.foreground_prototypes,
        background_prototypes=initial_state.prototype_state.background_prototypes,
        temperature=config.temperature,
        confidence_threshold=config.confidence_threshold,
        foreground_prior=config.foreground_prior,
    )
    m_step_result = run_protoem_m_step(
        query_feature_tensor=query_features,
        foreground_posterior_map=e_step_result.foreground_posterior_map,
        background_posterior_map=e_step_result.background_posterior_map,
        confidence_mask=e_step_result.confidence_mask_result.confident_mask,
        initial_foreground_prototype=support_foreground,
        initial_background_prototype=support_background,
        previous_foreground_prototype=initial_state.prototype_state.foreground_prototypes,
        previous_background_prototype=initial_state.prototype_state.background_prototypes,
        objective_weights=config.objective_weights,
        foreground_prior=config.foreground_prior,
        proximal_foreground_reference=support_foreground,
        proximal_background_reference=support_background,
        previous_foreground_posterior_map=initial_state.posterior_state.foreground_posterior_map,
        previous_background_posterior_map=initial_state.posterior_state.background_posterior_map,
        support_foreground_reference=support_foreground,
        support_background_reference=support_background,
    )
    return bundle, config, initial_state, e_step_result, m_step_result


def _raw_for_effective_alpha(alpha: float, *, epsilon: float) -> float:
    if alpha <= epsilon:
        raise ValueError("alpha must exceed epsilon to invert softplus.")
    return float(np.log(np.expm1(alpha - epsilon)))


def test_softplus_transform_gives_strictly_positive_steps() -> None:
    epsilon = 1e-6

    assert effective_positive_step(raw_value=-1000.0, epsilon=epsilon) > 0.0
    assert effective_positive_step(raw_value=0.0, epsilon=epsilon) > 0.0
    assert effective_positive_step(raw_value=4.0, epsilon=epsilon) > 0.0


def test_exact_known_raw_to_effective_step_values() -> None:
    epsilon = 1e-6

    observed_zero = effective_positive_step(raw_value=0.0, epsilon=epsilon)
    observed_one = effective_positive_step(raw_value=math.log(math.e - 1.0), epsilon=epsilon)

    assert observed_zero == pytest.approx(math.log(2.0) + epsilon, abs=1e-12)
    assert observed_one == pytest.approx(1.0 + epsilon, abs=1e-12)


def test_epsilon_handling() -> None:
    low = effective_positive_step(raw_value=-3.0, epsilon=1e-6)
    high = effective_positive_step(raw_value=-3.0, epsilon=1e-3)

    assert high - low == pytest.approx(1e-3 - 1e-6, abs=1e-12)


def test_non_finite_raw_parameter_rejection() -> None:
    with pytest.raises(NonFiniteStepParameterError):
        effective_positive_step(raw_value=float("nan"), epsilon=1e-6)

    with pytest.raises(NonFiniteStepParameterError):
        build_protoem_positive_step_schedule(
            raw_step_parameters=(0.0, float("inf")),
            epsilon=1e-6,
            max_iteration_compatibility=2,
        )


def test_schedule_length_mismatch_rejection() -> None:
    with pytest.raises(ScheduleLengthMismatchError):
        build_protoem_positive_step_schedule(
            raw_step_parameters=(0.0,),
            epsilon=1e-6,
            max_iteration_compatibility=2,
        )


def test_one_step_interpolation_formula() -> None:
    _, config, initial_state, e_step_result, m_step_result = _single_iteration_target_inputs()
    alpha = 0.5
    schedule = build_protoem_positive_step_schedule(
        raw_step_parameters=(_raw_for_effective_alpha(alpha, epsilon=1e-6),),
        epsilon=1e-6,
        max_iteration_compatibility=config.max_iterations,
    )

    adjusted, _ = apply_positive_step_to_m_step_result(
        iteration_index=0,
        schedule=schedule,
        source_prototype_state_identity_hash=(
            initial_state.prototype_state.prototype_state_identity_hash
        ),
        current_foreground_prototype=initial_state.prototype_state.foreground_prototypes,
        current_background_prototype=initial_state.prototype_state.background_prototypes,
        target_m_step_result=m_step_result,
        foreground_posterior_map=e_step_result.foreground_posterior_map,
        background_posterior_map=e_step_result.background_posterior_map,
        previous_foreground_posterior_map=initial_state.posterior_state.foreground_posterior_map,
        previous_background_posterior_map=initial_state.posterior_state.background_posterior_map,
        support_foreground_reference=np.asarray(
            initial_state.prototype_state.foreground_prototypes,
            dtype=np.float64,
        ),
        support_background_reference=np.asarray(
            initial_state.prototype_state.background_prototypes,
            dtype=np.float64,
        ),
        proximal_foreground_reference=np.asarray(
            initial_state.prototype_state.foreground_prototypes,
            dtype=np.float64,
        ),
        proximal_background_reference=np.asarray(
            initial_state.prototype_state.background_prototypes,
            dtype=np.float64,
        ),
        foreground_prior=config.foreground_prior,
    )

    expected_foreground = initial_state.prototype_state.foreground_prototypes + alpha * (
        m_step_result.updated_foreground_prototype
        - initial_state.prototype_state.foreground_prototypes
    )
    expected_background = initial_state.prototype_state.background_prototypes + alpha * (
        m_step_result.updated_background_prototype
        - initial_state.prototype_state.background_prototypes
    )

    assert np.allclose(adjusted.updated_foreground_prototype, expected_foreground, atol=1e-12)
    assert np.allclose(adjusted.updated_background_prototype, expected_background, atol=1e-12)


def test_alpha_equal_one_reproduces_standard_m_step_target_within_tolerance() -> None:
    bundle, fixed_config, fixed_result = _single_iteration_fixed_result()
    learned_config = _config_override(
        fixed_config,
        update_schedule="learned_positive_step",
    )
    alpha = 1.0
    schedule = build_protoem_positive_step_schedule(
        raw_step_parameters=(_raw_for_effective_alpha(alpha, epsilon=1e-6),),
        epsilon=1e-6,
        max_iteration_compatibility=learned_config.max_iterations,
    )

    learned_result = run_protoem_optimization(
        config=learned_config,
        initialization_bundle=bundle,
        positive_step_schedule=schedule,
    )

    assert np.allclose(
        learned_result.final_state.prototype_state.foreground_prototypes,
        fixed_result.final_state.prototype_state.foreground_prototypes,
        atol=1e-6,
    )
    assert np.allclose(
        learned_result.final_state.prototype_state.background_prototypes,
        fixed_result.final_state.prototype_state.background_prototypes,
        atol=1e-6,
    )


def test_alpha_less_than_one_gives_interpolation() -> None:
    bundle, config, fixed_result = _single_iteration_fixed_result()
    learned_config = _config_override(config, update_schedule="learned_positive_step")
    initial_state = build_initial_protoem_transductive_state(
        initialization_bundle=bundle,
        config=learned_config,
    )
    alpha = 0.25
    learned_result = run_protoem_optimization(
        config=learned_config,
        initialization_bundle=bundle,
        positive_step_schedule=build_protoem_positive_step_schedule(
            raw_step_parameters=(_raw_for_effective_alpha(alpha, epsilon=1e-6),),
            epsilon=1e-6,
            max_iteration_compatibility=learned_config.max_iterations,
        ),
    )

    expected_foreground = initial_state.prototype_state.foreground_prototypes + alpha * (
        fixed_result.final_state.prototype_state.foreground_prototypes
        - initial_state.prototype_state.foreground_prototypes
    )
    expected_background = initial_state.prototype_state.background_prototypes + alpha * (
        fixed_result.final_state.prototype_state.background_prototypes
        - initial_state.prototype_state.background_prototypes
    )
    assert np.allclose(
        learned_result.final_state.prototype_state.foreground_prototypes,
        expected_foreground,
        atol=1e-6,
    )
    assert np.allclose(
        learned_result.final_state.prototype_state.background_prototypes,
        expected_background,
        atol=1e-6,
    )


def test_alpha_greater_than_one_permits_extrapolation() -> None:
    bundle, config, fixed_result = _single_iteration_fixed_result()
    learned_config = _config_override(config, update_schedule="learned_positive_step")
    initial_state = build_initial_protoem_transductive_state(
        initialization_bundle=bundle,
        config=learned_config,
    )
    alpha = 1.5
    learned_result = run_protoem_optimization(
        config=learned_config,
        initialization_bundle=bundle,
        positive_step_schedule=build_protoem_positive_step_schedule(
            raw_step_parameters=(_raw_for_effective_alpha(alpha, epsilon=1e-6),),
            epsilon=1e-6,
            max_iteration_compatibility=learned_config.max_iterations,
        ),
    )

    expected_foreground = initial_state.prototype_state.foreground_prototypes + alpha * (
        fixed_result.final_state.prototype_state.foreground_prototypes
        - initial_state.prototype_state.foreground_prototypes
    )
    expected_background = initial_state.prototype_state.background_prototypes + alpha * (
        fixed_result.final_state.prototype_state.background_prototypes
        - initial_state.prototype_state.background_prototypes
    )
    assert np.allclose(
        learned_result.final_state.prototype_state.foreground_prototypes,
        expected_foreground,
        atol=1e-6,
    )
    assert np.allclose(
        learned_result.final_state.prototype_state.background_prototypes,
        expected_background,
        atol=1e-6,
    )


def test_fixed_em_like_results_remain_unchanged() -> None:
    bundle, config, first = _single_iteration_fixed_result()
    second = run_protoem_optimization(config=config, initialization_bundle=bundle)

    assert first.result_identity_hash == second.result_identity_hash
    assert first.learned_schedule_result is None
    with pytest.raises(OrchestrationInputError):
        run_protoem_optimization(
            config=config,
            initialization_bundle=bundle,
            positive_step_schedule=_schedule_for_config(config),
        )


def test_learned_positive_step_dispatch_works_only_with_explicit_schedule() -> None:
    bundle, _ = _initialization_bundle()
    learned_config = _config_override(
        _config_with_threshold(0.0),
        update_schedule="learned_positive_step",
        max_iterations=1,
        minimum_iterations=1,
    )

    with pytest.raises(OrchestrationInputError):
        run_protoem_optimization(config=learned_config, initialization_bundle=bundle)

    result = run_protoem_optimization(
        config=learned_config,
        initialization_bundle=bundle,
        positive_step_schedule=_schedule_for_config(learned_config),
    )
    assert result.learned_schedule_result is not None


def test_missing_learned_schedule_is_rejected() -> None:
    bundle, _ = _initialization_bundle()
    learned_config = _config_override(
        _config_with_threshold(0.0),
        update_schedule="learned_positive_step",
        max_iterations=1,
        minimum_iterations=1,
    )

    with pytest.raises(
        OrchestrationInputError, match="requires one explicit positive_step_schedule"
    ):
        run_protoem_optimization(config=learned_config, initialization_bundle=bundle)


def test_objective_trace_and_stopping_behavior_remain_valid() -> None:
    bundle, _ = _initialization_bundle()
    learned_config = _config_override(
        _config_with_threshold(0.0),
        update_schedule="learned_positive_step",
        max_iterations=1,
        minimum_iterations=1,
    )
    result = run_protoem_optimization(
        config=learned_config,
        initialization_bundle=bundle,
        positive_step_schedule=_schedule_for_config(learned_config),
    )

    assert result.objective_trace is not None
    assert result.stopping_record is not None
    assert result.learned_schedule_result is not None
    assert result.learned_schedule_result.completed_iteration_count == 1
    assert result.stopping_record.completed_iteration_count == 1


def test_schedule_parameter_changes_alter_result_identity() -> None:
    bundle, _ = _initialization_bundle()
    learned_config = _config_override(
        _config_with_threshold(0.0),
        update_schedule="learned_positive_step",
        max_iterations=1,
        minimum_iterations=1,
    )
    first = run_protoem_optimization(
        config=learned_config,
        initialization_bundle=bundle,
        positive_step_schedule=build_protoem_positive_step_schedule(
            raw_step_parameters=(0.0,),
            epsilon=1e-6,
            max_iteration_compatibility=1,
        ),
    )
    second = run_protoem_optimization(
        config=learned_config,
        initialization_bundle=bundle,
        positive_step_schedule=build_protoem_positive_step_schedule(
            raw_step_parameters=(0.5,),
            epsilon=1e-6,
            max_iteration_compatibility=1,
        ),
    )

    assert first.result_identity_hash != second.result_identity_hash


def test_repeated_execution_deterministic() -> None:
    bundle, _ = _initialization_bundle()
    learned_config = _config_override(
        _config_with_threshold(0.0),
        update_schedule="learned_positive_step",
        max_iterations=1,
        minimum_iterations=1,
    )
    schedule = _schedule_for_config(learned_config, raw_step_parameters=(0.25,))

    first = run_protoem_optimization(
        config=learned_config,
        initialization_bundle=bundle,
        positive_step_schedule=schedule,
    )
    second = run_protoem_optimization(
        config=learned_config,
        initialization_bundle=bundle,
        positive_step_schedule=schedule,
    )

    assert first.result_identity_hash == second.result_identity_hash
    assert first.learned_schedule_result is not None
    assert second.learned_schedule_result is not None
    assert (
        first.learned_schedule_result.learned_schedule_result_identity_hash
        == second.learned_schedule_result.learned_schedule_result_identity_hash
    )


def test_input_arrays_remain_unchanged() -> None:
    features = np.asarray([[[[[1.0, 0.0]]], [[[0.0, 1.0]]]]], dtype=np.float32)
    before = features.copy()
    bundle, _ = _initialization_bundle(feature_values=features)
    learned_config = _config_override(
        _config_with_threshold(0.0),
        update_schedule="learned_positive_step",
        max_iterations=1,
        minimum_iterations=1,
    )

    _ = run_protoem_optimization(
        config=learned_config,
        initialization_bundle=bundle,
        positive_step_schedule=_schedule_for_config(learned_config),
    )

    assert np.array_equal(features, before)


def test_no_optimizer_training_query_label_or_reference_mask_in_public_apis() -> None:
    for callable_object in (
        build_protoem_positive_step_schedule,
        apply_positive_step_to_m_step_result,
        run_protoem_optimization,
    ):
        signature_text = str(inspect.signature(callable_object)).lower()
        assert "optimizer" not in signature_text
        assert "training" not in signature_text
        assert "query_label" not in signature_text
        assert "reference_mask" not in signature_text

    assert "optimizer" not in ProtoEMPositiveStepSchedule.__dataclass_fields__
    assert "training" not in ProtoEMPositiveStepSchedule.__dataclass_fields__
    assert "query_label" not in ProtoEMPositiveStepSchedule.__dataclass_fields__
    assert "reference_mask" not in ProtoEMPositiveStepSchedule.__dataclass_fields__


def test_invalid_epsilon_rejection() -> None:
    with pytest.raises(InvalidPositiveStepScheduleError):
        build_protoem_positive_step_schedule(
            raw_step_parameters=(0.0,),
            epsilon=0.0,
            max_iteration_compatibility=1,
        )
