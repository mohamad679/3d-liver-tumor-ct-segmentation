"""Unit tests for deterministic ProtoEM-CT objective, E-step, and M-step contracts."""

from __future__ import annotations

import inspect
import math

import numpy as np
import pytest

from protoem_ct.protoem import (
    EmptyEffectiveBackgroundUpdateError,
    EmptyEffectiveForegroundUpdateError,
    IncompatiblePrototypeBankError,
    InvalidEStepInputError,
    NoConfidentVoxelsError,
    ProtoEMObjectiveTerms,
    ProtoEMObjectiveWeights,
    ZeroNormPrototypeError,
    compute_protoem_objective_terms,
    hash_protoem_e_step_result,
    run_protoem_e_step,
    run_protoem_m_step,
)


def _weights(
    *,
    support: float = 1.0,
    query_entropy: float = 1.0,
    class_balance: float = 1.0,
    consistency: float = 1.0,
    proximal: float = 1.0,
) -> ProtoEMObjectiveWeights:
    return ProtoEMObjectiveWeights(
        schema_name="protoem_objective_weights",
        schema_version="v1",
        support=support,
        query_entropy=query_entropy,
        class_balance=class_balance,
        consistency=consistency,
        proximal=proximal,
    )


def _query_features_two_voxels() -> np.ndarray:
    return np.asarray([[[[[1.0, 0.0]]], [[[0.0, 1.0]]]]], dtype=np.float64)


def _single_voxel_query(vector: tuple[float, float]) -> np.ndarray:
    return np.asarray([[[[[vector[0]]]], [[[vector[1]]]]]], dtype=np.float64)


def _posterior_map(values: tuple[float, ...]) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(1, 1, 1, 1, len(values))


def _mask(values: tuple[int, ...]) -> np.ndarray:
    return np.asarray(values, dtype=bool).reshape(1, 1, 1, 1, len(values))


def test_exact_known_binary_posterior_computation() -> None:
    result = run_protoem_e_step(
        query_feature_tensor=_single_voxel_query((1.0, 0.0)),
        foreground_prototypes=np.asarray([1.0, 0.0], dtype=np.float64),
        background_prototypes=np.asarray([0.0, 1.0], dtype=np.float64),
        temperature=1.0,
        confidence_threshold=0.0,
        foreground_prior=0.5,
    )

    expected = math.exp(1.0) / (math.exp(1.0) + math.exp(0.0))
    assert np.allclose(result.foreground_posterior_map, expected, rtol=0.0, atol=1e-12)
    assert np.allclose(result.background_posterior_map, 1.0 - expected, rtol=0.0, atol=1e-12)


def test_temperature_effect() -> None:
    colder = run_protoem_e_step(
        query_feature_tensor=_single_voxel_query((1.0, 0.0)),
        foreground_prototypes=np.asarray([1.0, 0.0], dtype=np.float64),
        background_prototypes=np.asarray([0.0, 1.0], dtype=np.float64),
        temperature=0.5,
        confidence_threshold=0.0,
        foreground_prior=0.5,
    )
    hotter = run_protoem_e_step(
        query_feature_tensor=_single_voxel_query((1.0, 0.0)),
        foreground_prototypes=np.asarray([1.0, 0.0], dtype=np.float64),
        background_prototypes=np.asarray([0.0, 1.0], dtype=np.float64),
        temperature=2.0,
        confidence_threshold=0.0,
        foreground_prior=0.5,
    )

    assert float(colder.foreground_posterior_map[0, 0, 0, 0, 0]) > float(
        hotter.foreground_posterior_map[0, 0, 0, 0, 0]
    )


def test_foreground_prior_effect() -> None:
    low_prior = run_protoem_e_step(
        query_feature_tensor=_single_voxel_query((0.0, 0.0)),
        foreground_prototypes=np.asarray([1.0, 0.0], dtype=np.float64),
        background_prototypes=np.asarray([0.0, 1.0], dtype=np.float64),
        temperature=1.0,
        confidence_threshold=0.0,
        foreground_prior=0.2,
    )
    high_prior = run_protoem_e_step(
        query_feature_tensor=_single_voxel_query((0.0, 0.0)),
        foreground_prototypes=np.asarray([1.0, 0.0], dtype=np.float64),
        background_prototypes=np.asarray([0.0, 1.0], dtype=np.float64),
        temperature=1.0,
        confidence_threshold=0.0,
        foreground_prior=0.8,
    )

    assert np.allclose(low_prior.foreground_posterior_map, 0.2, rtol=0.0, atol=1e-12)
    assert np.allclose(high_prior.foreground_posterior_map, 0.8, rtol=0.0, atol=1e-12)


def test_exact_tie_resolves_to_background() -> None:
    result = run_protoem_e_step(
        query_feature_tensor=_single_voxel_query((1.0, 0.0)),
        foreground_prototypes=np.asarray([1.0, 0.0], dtype=np.float64),
        background_prototypes=np.asarray([1.0, 0.0], dtype=np.float64),
        temperature=1.0,
        confidence_threshold=0.0,
        foreground_prior=0.5,
    )

    assert int(result.hard_assignment_map[0, 0, 0, 0, 0]) == 0


def test_confidence_threshold_boundary_behavior() -> None:
    expected = math.exp(1.0) / (math.exp(1.0) + math.exp(0.0))
    result = run_protoem_e_step(
        query_feature_tensor=_single_voxel_query((1.0, 0.0)),
        foreground_prototypes=np.asarray([1.0, 0.0], dtype=np.float64),
        background_prototypes=np.asarray([0.0, 1.0], dtype=np.float64),
        temperature=1.0,
        confidence_threshold=expected,
        foreground_prior=0.5,
    )

    assert bool(result.confidence_mask_result.confident_mask[0, 0, 0, 0, 0]) is True


def test_zero_norm_query_voxel_uses_prior_only_posterior() -> None:
    result = run_protoem_e_step(
        query_feature_tensor=_single_voxel_query((0.0, 0.0)),
        foreground_prototypes=np.asarray([1.0, 0.0], dtype=np.float64),
        background_prototypes=np.asarray([0.0, 1.0], dtype=np.float64),
        temperature=1.0,
        confidence_threshold=0.5,
        foreground_prior=0.8,
    )

    assert np.allclose(result.foreground_posterior_map, 0.8, rtol=0.0, atol=1e-12)
    assert np.allclose(result.background_posterior_map, 0.2, rtol=0.0, atol=1e-12)
    assert bool(result.zero_norm_query_voxel_mask[0, 0, 0, 0, 0]) is True


def test_zero_norm_prototype_rejection() -> None:
    with pytest.raises(ZeroNormPrototypeError):
        run_protoem_e_step(
            query_feature_tensor=_single_voxel_query((1.0, 0.0)),
            foreground_prototypes=np.asarray([0.0, 0.0], dtype=np.float64),
            background_prototypes=np.asarray([0.0, 1.0], dtype=np.float64),
            temperature=1.0,
            confidence_threshold=0.0,
            foreground_prior=0.5,
        )


def test_multiple_prototype_deterministic_aggregation() -> None:
    query = _single_voxel_query((1.0, 0.0))
    foreground_a = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    foreground_b = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    background_a = np.asarray([[-1.0, 0.0], [0.0, -1.0]], dtype=np.float64)
    background_b = np.asarray([[0.0, -1.0], [-1.0, 0.0]], dtype=np.float64)

    first = run_protoem_e_step(
        query_feature_tensor=query,
        foreground_prototypes=foreground_a,
        background_prototypes=background_a,
        temperature=1.0,
        confidence_threshold=0.0,
        foreground_prior=0.5,
    )
    second = run_protoem_e_step(
        query_feature_tensor=query,
        foreground_prototypes=foreground_b,
        background_prototypes=background_b,
        temperature=1.0,
        confidence_threshold=0.0,
        foreground_prior=0.5,
    )

    assert np.array_equal(first.hard_assignment_map, second.hard_assignment_map)
    assert hash_protoem_e_step_result(first) == hash_protoem_e_step_result(second)


def test_exact_posterior_weighted_foreground_update() -> None:
    result = run_protoem_m_step(
        query_feature_tensor=_query_features_two_voxels(),
        foreground_posterior_map=_posterior_map((0.75, 0.25)),
        background_posterior_map=_posterior_map((0.25, 0.75)),
        confidence_mask=_mask((1, 1)),
        initial_foreground_prototype=np.asarray([1.0, 0.0], dtype=np.float64),
        initial_background_prototype=np.asarray([0.0, 1.0], dtype=np.float64),
        previous_foreground_prototype=np.asarray([1.0, 0.0], dtype=np.float64),
        previous_background_prototype=np.asarray([0.0, 1.0], dtype=np.float64),
        objective_weights=_weights(),
        foreground_prior=0.5,
        proximal_foreground_reference=np.asarray([1.0, 0.0], dtype=np.float64),
        proximal_background_reference=np.asarray([0.0, 1.0], dtype=np.float64),
        previous_foreground_posterior_map=_posterior_map((0.5, 0.5)),
        previous_background_posterior_map=_posterior_map((0.5, 0.5)),
        support_foreground_reference=np.asarray([1.0, 0.0], dtype=np.float64),
        support_background_reference=np.asarray([0.0, 1.0], dtype=np.float64),
    )

    assert np.allclose(result.updated_foreground_prototype, np.asarray([0.75, 0.25]))


def test_exact_posterior_weighted_background_update() -> None:
    result = run_protoem_m_step(
        query_feature_tensor=_query_features_two_voxels(),
        foreground_posterior_map=_posterior_map((0.75, 0.25)),
        background_posterior_map=_posterior_map((0.25, 0.75)),
        confidence_mask=_mask((1, 1)),
        initial_foreground_prototype=np.asarray([1.0, 0.0], dtype=np.float64),
        initial_background_prototype=np.asarray([0.0, 1.0], dtype=np.float64),
        previous_foreground_prototype=np.asarray([1.0, 0.0], dtype=np.float64),
        previous_background_prototype=np.asarray([0.0, 1.0], dtype=np.float64),
        objective_weights=_weights(),
        foreground_prior=0.5,
        proximal_foreground_reference=np.asarray([1.0, 0.0], dtype=np.float64),
        proximal_background_reference=np.asarray([0.0, 1.0], dtype=np.float64),
        previous_foreground_posterior_map=_posterior_map((0.5, 0.5)),
        previous_background_posterior_map=_posterior_map((0.5, 0.5)),
        support_foreground_reference=np.asarray([1.0, 0.0], dtype=np.float64),
        support_background_reference=np.asarray([0.0, 1.0], dtype=np.float64),
    )

    assert np.allclose(result.updated_background_prototype, np.asarray([0.25, 0.75]))


def test_no_confident_voxel_failure() -> None:
    with pytest.raises(NoConfidentVoxelsError):
        run_protoem_m_step(
            query_feature_tensor=_query_features_two_voxels(),
            foreground_posterior_map=_posterior_map((0.75, 0.25)),
            background_posterior_map=_posterior_map((0.25, 0.75)),
            confidence_mask=_mask((0, 0)),
            initial_foreground_prototype=np.asarray([1.0, 0.0], dtype=np.float64),
            initial_background_prototype=np.asarray([0.0, 1.0], dtype=np.float64),
            previous_foreground_prototype=np.asarray([1.0, 0.0], dtype=np.float64),
            previous_background_prototype=np.asarray([0.0, 1.0], dtype=np.float64),
            objective_weights=_weights(),
            foreground_prior=0.5,
            proximal_foreground_reference=np.asarray([1.0, 0.0], dtype=np.float64),
            proximal_background_reference=np.asarray([0.0, 1.0], dtype=np.float64),
            previous_foreground_posterior_map=_posterior_map((0.5, 0.5)),
            previous_background_posterior_map=_posterior_map((0.5, 0.5)),
            support_foreground_reference=np.asarray([1.0, 0.0], dtype=np.float64),
            support_background_reference=np.asarray([0.0, 1.0], dtype=np.float64),
        )


def test_empty_foreground_effective_weight_failure() -> None:
    with pytest.raises(EmptyEffectiveForegroundUpdateError):
        run_protoem_m_step(
            query_feature_tensor=_query_features_two_voxels(),
            foreground_posterior_map=_posterior_map((0.0, 0.0)),
            background_posterior_map=_posterior_map((1.0, 1.0)),
            confidence_mask=_mask((1, 1)),
            initial_foreground_prototype=np.asarray([1.0, 0.0], dtype=np.float64),
            initial_background_prototype=np.asarray([0.0, 1.0], dtype=np.float64),
            previous_foreground_prototype=np.asarray([1.0, 0.0], dtype=np.float64),
            previous_background_prototype=np.asarray([0.0, 1.0], dtype=np.float64),
            objective_weights=_weights(),
            foreground_prior=0.5,
            proximal_foreground_reference=np.asarray([1.0, 0.0], dtype=np.float64),
            proximal_background_reference=np.asarray([0.0, 1.0], dtype=np.float64),
            previous_foreground_posterior_map=_posterior_map((0.5, 0.5)),
            previous_background_posterior_map=_posterior_map((0.5, 0.5)),
            support_foreground_reference=np.asarray([1.0, 0.0], dtype=np.float64),
            support_background_reference=np.asarray([0.0, 1.0], dtype=np.float64),
        )


def test_empty_background_effective_weight_failure() -> None:
    with pytest.raises(EmptyEffectiveBackgroundUpdateError):
        run_protoem_m_step(
            query_feature_tensor=_query_features_two_voxels(),
            foreground_posterior_map=_posterior_map((1.0, 1.0)),
            background_posterior_map=_posterior_map((0.0, 0.0)),
            confidence_mask=_mask((1, 1)),
            initial_foreground_prototype=np.asarray([1.0, 0.0], dtype=np.float64),
            initial_background_prototype=np.asarray([0.0, 1.0], dtype=np.float64),
            previous_foreground_prototype=np.asarray([1.0, 0.0], dtype=np.float64),
            previous_background_prototype=np.asarray([0.0, 1.0], dtype=np.float64),
            objective_weights=_weights(),
            foreground_prior=0.5,
            proximal_foreground_reference=np.asarray([1.0, 0.0], dtype=np.float64),
            proximal_background_reference=np.asarray([0.0, 1.0], dtype=np.float64),
            previous_foreground_posterior_map=_posterior_map((0.5, 0.5)),
            previous_background_posterior_map=_posterior_map((0.5, 0.5)),
            support_foreground_reference=np.asarray([1.0, 0.0], dtype=np.float64),
            support_background_reference=np.asarray([0.0, 1.0], dtype=np.float64),
        )


def test_exact_formulas_for_all_five_objective_terms() -> None:
    terms = compute_protoem_objective_terms(
        objective_weights=_weights(),
        foreground_posterior_map=_posterior_map((0.75, 0.25)),
        background_posterior_map=_posterior_map((0.25, 0.75)),
        previous_foreground_posterior_map=_posterior_map((0.5, 0.5)),
        previous_background_posterior_map=_posterior_map((0.5, 0.5)),
        updated_foreground_prototype=np.asarray([0.75, 0.25], dtype=np.float64),
        updated_background_prototype=np.asarray([0.25, 0.75], dtype=np.float64),
        support_foreground_reference=np.asarray([1.0, 0.0], dtype=np.float64),
        support_background_reference=np.asarray([0.0, 1.0], dtype=np.float64),
        proximal_foreground_reference=np.asarray([1.0, 0.0], dtype=np.float64),
        proximal_background_reference=np.asarray([0.0, 1.0], dtype=np.float64),
        foreground_prior=0.4,
    )

    expected_support = 0.0625
    entropy_single = -(0.75 * math.log(0.75) + 0.25 * math.log(0.25))
    expected_entropy = entropy_single
    expected_balance = (0.5 - 0.4) ** 2
    expected_consistency = 0.0625
    expected_proximal = 0.0625

    assert math.isclose(terms.support, expected_support, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(terms.query_entropy, expected_entropy, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(terms.class_balance, expected_balance, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(terms.consistency, expected_consistency, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(terms.proximal, expected_proximal, rel_tol=0.0, abs_tol=1e-12)


def test_exact_weighted_total() -> None:
    weights = _weights(
        support=1.0,
        query_entropy=2.0,
        class_balance=3.0,
        consistency=4.0,
        proximal=5.0,
    )
    terms = compute_protoem_objective_terms(
        objective_weights=weights,
        foreground_posterior_map=_posterior_map((0.75, 0.25)),
        background_posterior_map=_posterior_map((0.25, 0.75)),
        previous_foreground_posterior_map=_posterior_map((0.5, 0.5)),
        previous_background_posterior_map=_posterior_map((0.5, 0.5)),
        updated_foreground_prototype=np.asarray([0.75, 0.25], dtype=np.float64),
        updated_background_prototype=np.asarray([0.25, 0.75], dtype=np.float64),
        support_foreground_reference=np.asarray([1.0, 0.0], dtype=np.float64),
        support_background_reference=np.asarray([0.0, 1.0], dtype=np.float64),
        proximal_foreground_reference=np.asarray([1.0, 0.0], dtype=np.float64),
        proximal_background_reference=np.asarray([0.0, 1.0], dtype=np.float64),
        foreground_prior=0.4,
    )

    expected = (
        weights.support * terms.support
        + weights.query_entropy * terms.query_entropy
        + weights.class_balance * terms.class_balance
        + weights.consistency * terms.consistency
        + weights.proximal * terms.proximal
    )
    assert math.isclose(terms.total, expected, rel_tol=0.0, abs_tol=1e-12)


def test_zero_objective_weight_behavior() -> None:
    terms = compute_protoem_objective_terms(
        objective_weights=_weights(
            support=0.0,
            query_entropy=1.0,
            class_balance=0.0,
            consistency=0.0,
            proximal=0.0,
        ),
        foreground_posterior_map=_posterior_map((0.75, 0.25)),
        background_posterior_map=_posterior_map((0.25, 0.75)),
        previous_foreground_posterior_map=_posterior_map((0.5, 0.5)),
        previous_background_posterior_map=_posterior_map((0.5, 0.5)),
        updated_foreground_prototype=np.asarray([0.75, 0.25], dtype=np.float64),
        updated_background_prototype=np.asarray([0.25, 0.75], dtype=np.float64),
        support_foreground_reference=np.asarray([1.0, 0.0], dtype=np.float64),
        support_background_reference=np.asarray([0.0, 1.0], dtype=np.float64),
        proximal_foreground_reference=np.asarray([1.0, 0.0], dtype=np.float64),
        proximal_background_reference=np.asarray([0.0, 1.0], dtype=np.float64),
        foreground_prior=0.4,
    )

    assert terms.support > 0.0
    assert terms.class_balance > 0.0
    assert terms.consistency > 0.0
    assert terms.proximal > 0.0
    assert math.isclose(terms.total, terms.query_entropy, rel_tol=0.0, abs_tol=1e-12)


def test_non_finite_input_rejection() -> None:
    with pytest.raises(InvalidEStepInputError):
        run_protoem_e_step(
            query_feature_tensor=np.asarray([[[[[np.nan]]], [[[0.0]]]]], dtype=np.float64),
            foreground_prototypes=np.asarray([1.0, 0.0], dtype=np.float64),
            background_prototypes=np.asarray([0.0, 1.0], dtype=np.float64),
            temperature=1.0,
            confidence_threshold=0.0,
            foreground_prior=0.5,
        )


def test_shape_channel_incompatibility_rejection() -> None:
    with pytest.raises(IncompatiblePrototypeBankError):
        run_protoem_e_step(
            query_feature_tensor=_single_voxel_query((1.0, 0.0)),
            foreground_prototypes=np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
            background_prototypes=np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
            temperature=1.0,
            confidence_threshold=0.0,
            foreground_prior=0.5,
        )


def test_repeated_execution_deterministic() -> None:
    query_feature_tensor = _query_features_two_voxels()
    foreground_prototypes = np.asarray([1.0, 0.0], dtype=np.float64)
    background_prototypes = np.asarray([0.0, 1.0], dtype=np.float64)
    first = run_protoem_e_step(
        query_feature_tensor=query_feature_tensor,
        foreground_prototypes=foreground_prototypes,
        background_prototypes=background_prototypes,
        temperature=1.0,
        confidence_threshold=0.5,
        foreground_prior=0.5,
    )
    second = run_protoem_e_step(
        query_feature_tensor=query_feature_tensor,
        foreground_prototypes=foreground_prototypes,
        background_prototypes=background_prototypes,
        temperature=1.0,
        confidence_threshold=0.5,
        foreground_prior=0.5,
    )

    assert np.array_equal(first.foreground_posterior_map, second.foreground_posterior_map)
    assert hash_protoem_e_step_result(first) == hash_protoem_e_step_result(second)


def test_equivalent_non_contiguous_numpy_input_gives_identical_result() -> None:
    base = np.asarray([[[[[1.0, 0.0]]], [[[0.0, 1.0]]]]], dtype=np.float64)
    non_contiguous = base[..., ::-1][..., ::-1]
    first = run_protoem_e_step(
        query_feature_tensor=base,
        foreground_prototypes=np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64),
        background_prototypes=np.asarray([[0.0, -1.0], [-1.0, 0.0]], dtype=np.float64),
        temperature=1.0,
        confidence_threshold=0.5,
        foreground_prior=0.5,
    )
    second = run_protoem_e_step(
        query_feature_tensor=non_contiguous,
        foreground_prototypes=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)[::-1],
        background_prototypes=np.asarray([[-1.0, 0.0], [0.0, -1.0]], dtype=np.float64)[:, ::-1][
            :, ::-1
        ],
        temperature=1.0,
        confidence_threshold=0.5,
        foreground_prior=0.5,
    )

    assert np.array_equal(first.foreground_posterior_map, second.foreground_posterior_map)
    assert hash_protoem_e_step_result(first) == hash_protoem_e_step_result(second)


def test_query_labels_reference_masks_absent_from_all_public_apis() -> None:
    for func in (run_protoem_e_step, run_protoem_m_step, compute_protoem_objective_terms):
        signature_text = str(inspect.signature(func)).lower()
        assert "query_label" not in signature_text
        assert "reference_mask" not in signature_text
    assert "query_label" not in ProtoEMObjectiveTerms.__dataclass_fields__
    assert "reference_mask" not in ProtoEMObjectiveTerms.__dataclass_fields__
