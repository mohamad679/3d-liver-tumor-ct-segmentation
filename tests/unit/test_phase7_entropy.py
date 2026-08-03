from __future__ import annotations

import hashlib
import inspect
import math
from dataclasses import fields

import numpy as np
import pytest

from protoem_ct.uncertainty.artifacts import (
    Phase7UncertaintyArtifactHashError,
    phase7_uncertainty_result_from_json,
    phase7_uncertainty_result_to_json,
)
from protoem_ct.uncertainty.contracts import (
    BinaryPredictiveProbabilityMap,
    ProbabilityMapValidationError,
    build_binary_predictive_probability_map,
    build_predictive_entropy_input,
)
from protoem_ct.uncertainty.entropy import (
    PredictiveEntropyComputationResult,
    PredictiveEntropyError,
    compute_predictive_entropy,
    query_label_not_part_of_predictive_entropy_api,
)


def test_predictive_entropy_exact_values() -> None:
    probability_map = _probability_map(np.array([[[[[0.0, 1.0], [0.5, 0.25]]]]], dtype=np.float64))

    result = compute_predictive_entropy(probability_map)

    expected_asymmetric = -(0.25 * math.log(0.25) + 0.75 * math.log(0.75))
    expected = np.array([[[[[0.0, 0.0], [math.log(2.0), expected_asymmetric]]]]])
    assert np.allclose(result.entropy_map, expected, rtol=0.0, atol=1e-15)
    assert result.entropy_input.log_base == "natural"
    assert result.entropy_input.zero_probability_policy == "zero_log_zero_is_zero"


def test_entropy_output_and_content_hash_are_finite_and_deterministic() -> None:
    probability_map = _probability_map(_foreground_probabilities())

    first = compute_predictive_entropy(probability_map)
    second = compute_predictive_entropy(probability_map)

    assert np.all(np.isfinite(first.entropy_map))
    assert np.array_equal(first.entropy_map, second.entropy_map)
    assert first.entropy_map_content_hash == second.entropy_map_content_hash
    assert first.uncertainty_result.uncertainty_result_hash == (
        second.uncertainty_result.uncertainty_result_hash
    )
    assert first.uncertainty_result.uncertainty_type == "predictive_entropy"
    assert first.uncertainty_result.sample_count == 1
    assert first.uncertainty_result.variance_map_content_hash is None


def test_non_contiguous_probability_arrays_produce_identical_entropy_identity() -> None:
    foreground = _foreground_probabilities()
    background = 1.0 - foreground
    foreground_container = np.empty((1, 1, 1, 2, 4), dtype=np.float64)
    background_container = np.empty((1, 1, 1, 2, 4), dtype=np.float64)
    foreground_container[..., ::2] = foreground
    background_container[..., ::2] = background

    contiguous = _probability_map(foreground)
    non_contiguous = build_binary_predictive_probability_map(
        foreground_probability_map=foreground_container[..., ::2],
        background_probability_map=background_container[..., ::2],
        source_prediction_identity_hash=_sha("phase6-prediction"),
        common_grid_identity_hash=_sha("common-grid"),
    )

    first = compute_predictive_entropy(contiguous)
    second = compute_predictive_entropy(non_contiguous)

    assert not foreground_container[..., ::2].flags.c_contiguous
    assert np.array_equal(first.entropy_map, second.entropy_map)
    assert first.entropy_map_content_hash == second.entropy_map_content_hash
    assert first.uncertainty_result.uncertainty_result_hash == (
        second.uncertainty_result.uncertainty_result_hash
    )


def test_invalid_probability_contract_is_rejected() -> None:
    probability_map = _probability_map(_foreground_probabilities())
    probability_map.foreground_probability_map[0, 0, 0, 0, 0] = np.nan

    with pytest.raises(ProbabilityMapValidationError, match="finite"):
        compute_predictive_entropy(probability_map)


def test_entropy_input_must_match_probability_map() -> None:
    probability_map = _probability_map(_foreground_probabilities())
    other_probability_map = _probability_map(
        np.array([[[[[0.2, 0.8], [0.7, 0.3]]]]], dtype=np.float64)
    )
    mismatched_input = build_predictive_entropy_input(other_probability_map)

    with pytest.raises(ProbabilityMapValidationError, match="probability_map_identity_hash"):
        compute_predictive_entropy(probability_map, entropy_input=mismatched_input)


def test_source_prediction_hash_is_required_for_uncertainty_artifact() -> None:
    foreground = _foreground_probabilities()
    probability_map = build_binary_predictive_probability_map(
        foreground_probability_map=foreground,
        background_probability_map=1.0 - foreground,
        common_grid_identity_hash=_sha("common-grid"),
    )

    with pytest.raises(PredictiveEntropyError, match="source_prediction_identity_hash"):
        compute_predictive_entropy(probability_map)


def test_uncertainty_artifact_self_hash_validates() -> None:
    result = compute_predictive_entropy(_probability_map(_foreground_probabilities()))
    artifact = result.uncertainty_result

    round_tripped = phase7_uncertainty_result_from_json(phase7_uncertainty_result_to_json(artifact))
    assert round_tripped == artifact

    payload = (
        phase7_uncertainty_result_to_json(artifact)
        .decode("utf-8")
        .replace(
            artifact.uncertainty_result_hash,
            "0" * 64,
            1,
        )
    )
    with pytest.raises(Phase7UncertaintyArtifactHashError, match="hash"):
        phase7_uncertainty_result_from_json(payload)


def test_repeated_execution_is_byte_identical_at_canonical_artifact_level() -> None:
    probability_map = _probability_map(_foreground_probabilities())

    first = compute_predictive_entropy(probability_map)
    second = compute_predictive_entropy(probability_map)

    assert phase7_uncertainty_result_to_json(first.uncertainty_result) == (
        phase7_uncertainty_result_to_json(second.uncertainty_result)
    )
    assert first.entropy_result_metadata.entropy_result_metadata_hash == (
        second.entropy_result_metadata.entropy_result_metadata_hash
    )


def test_query_label_reference_mask_absent_from_public_api() -> None:
    prohibited = {
        "query_label",
        "query_labels",
        "label_map",
        "reference_mask",
        "query_reference_mask",
    }
    public_names = set(inspect.signature(compute_predictive_entropy).parameters)
    public_names.update(field.name for field in fields(PredictiveEntropyComputationResult))

    assert prohibited.isdisjoint(public_names)
    assert query_label_not_part_of_predictive_entropy_api()


def _foreground_probabilities() -> np.ndarray:
    return np.array([[[[[0.1, 0.9], [0.4, 0.6]]]]], dtype=np.float64)


def _probability_map(foreground: np.ndarray) -> BinaryPredictiveProbabilityMap:
    return build_binary_predictive_probability_map(
        foreground_probability_map=foreground,
        background_probability_map=1.0 - foreground,
        source_prediction_identity_hash=_sha("phase6-prediction"),
        common_grid_identity_hash=_sha("common-grid"),
    )


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
