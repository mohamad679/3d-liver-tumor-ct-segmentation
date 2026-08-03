from __future__ import annotations

import hashlib
import inspect
from dataclasses import fields

import numpy as np
import pytest

from protoem_ct.uncertainty.artifacts import (
    phase7_uncertainty_result_from_json,
    phase7_uncertainty_result_to_json,
)
from protoem_ct.uncertainty.contracts import (
    BinaryPredictiveProbabilityMap,
    TTAContractValidationError,
    TTAProbabilitySamples,
    TTASampleRecord,
    build_binary_predictive_probability_map,
    build_tta_probability_samples,
    build_tta_sample_manifest,
    build_tta_sample_record,
)
from protoem_ct.uncertainty.tta import (
    TTAVarianceResult,
    compute_ensemble_probability_variance,
    compute_tta_probability_variance,
    query_label_not_part_of_tta_api,
)


def test_exact_variance_for_known_two_sample_probabilities() -> None:
    samples = _probability_samples(
        (
            np.array([0.2, 0.6], dtype=np.float64),
            np.array([0.4, 0.2], dtype=np.float64),
        )
    )

    result = compute_tta_probability_variance(samples)

    expected = np.array([[[[[0.01, 0.04]]]]], dtype=np.float64)
    assert np.allclose(result.foreground_variance_map, expected, rtol=0.0, atol=1e-15)
    assert np.allclose(result.background_variance_map, expected, rtol=0.0, atol=1e-15)
    assert result.uncertainty_result.uncertainty_type == "tta_variance"
    assert result.uncertainty_result.sample_count == 2
    assert result.uncertainty_result.mean_uncertainty == pytest.approx(0.025)
    assert result.uncertainty_result.max_uncertainty == pytest.approx(0.04)


def test_exact_variance_for_known_three_sample_probabilities() -> None:
    samples = _probability_samples(
        (
            np.array([0.0, 0.5], dtype=np.float64),
            np.array([0.3, 0.5], dtype=np.float64),
            np.array([0.6, 0.2], dtype=np.float64),
        )
    )

    result = compute_tta_probability_variance(samples)

    expected = np.var(
        np.array(
            [
                [[[[[0.0, 0.5]]]]],
                [[[[[0.3, 0.5]]]]],
                [[[[[0.6, 0.2]]]]],
            ],
            dtype=np.float64,
        ),
        axis=0,
    )
    assert np.allclose(result.foreground_variance_map, expected, rtol=0.0, atol=1e-15)
    assert np.allclose(result.background_variance_map, expected, rtol=0.0, atol=1e-15)


def test_sample_count_requires_at_least_two_samples() -> None:
    probability_map = _probability_map("sample-0", np.array([0.2, 0.8], dtype=np.float64))
    manifest = build_tta_sample_manifest((_sample_record(0, probability_map),))

    with pytest.raises(TTAContractValidationError, match="at least two samples"):
        build_tta_probability_samples(
            sample_manifest=manifest,
            probability_maps=(probability_map,),
        )


def test_mismatched_shapes_are_rejected_by_sample_contract() -> None:
    first = _probability_map("sample-0", np.array([0.2, 0.8], dtype=np.float64))
    second = build_binary_predictive_probability_map(
        foreground_probability_map=np.array([[[[[0.3, 0.6], [0.5, 0.4]]]]], dtype=np.float64),
        background_probability_map=np.array([[[[[0.7, 0.4], [0.5, 0.6]]]]], dtype=np.float64),
        source_prediction_identity_hash=_sha("prediction-sample-1"),
        common_grid_identity_hash=_sha("common-grid"),
    )
    manifest = build_tta_sample_manifest((_sample_record(0, first), _sample_record(1, second)))

    with pytest.raises(TTAContractValidationError, match="share one"):
        build_tta_probability_samples(
            sample_manifest=manifest,
            probability_maps=(first, second),
        )


def test_mismatched_common_grid_is_rejected_by_manifest_contract() -> None:
    first = _probability_map("sample-0", np.array([0.2, 0.8], dtype=np.float64))
    second = _probability_map(
        "sample-1",
        np.array([0.3, 0.6], dtype=np.float64),
        common_grid_identity_hash=_sha("other-grid"),
    )

    with pytest.raises(TTAContractValidationError, match="same common grid"):
        build_tta_sample_manifest(
            (
                _sample_record(0, first),
                _sample_record(1, second, common_grid_identity_hash=_sha("other-grid")),
            )
        )


def test_mismatched_map_and_manifest_grid_is_rejected_by_sample_contract() -> None:
    first = _probability_map("sample-0", np.array([0.2, 0.8], dtype=np.float64))
    second = _probability_map("sample-1", np.array([0.3, 0.6], dtype=np.float64))
    malformed_map = build_binary_predictive_probability_map(
        foreground_probability_map=second.foreground_probability_map,
        background_probability_map=second.background_probability_map,
        source_prediction_identity_hash=second.source_prediction_identity_hash,
        common_grid_identity_hash=None,
    )
    manifest = build_tta_sample_manifest(
        (
            _sample_record(0, first),
            _sample_record(1, malformed_map),
        )
    )

    with pytest.raises(TTAContractValidationError, match="same common_grid_identity_hash"):
        build_tta_probability_samples(
            sample_manifest=manifest,
            probability_maps=(first, malformed_map),
        )


def test_deterministic_result_and_artifact_self_hash() -> None:
    samples = _probability_samples(
        (
            np.array([0.2, 0.6], dtype=np.float64),
            np.array([0.4, 0.2], dtype=np.float64),
        )
    )

    first = compute_tta_probability_variance(samples)
    second = compute_tta_probability_variance(samples)

    assert np.array_equal(first.foreground_variance_map, second.foreground_variance_map)
    assert first.uncertainty_result == second.uncertainty_result
    reconstructed = phase7_uncertainty_result_from_json(
        phase7_uncertainty_result_to_json(first.uncertainty_result)
    )
    assert reconstructed == first.uncertainty_result


def test_repeated_execution_is_byte_identical_at_canonical_artifact_level() -> None:
    samples = _probability_samples(
        (
            np.array([0.2, 0.6], dtype=np.float64),
            np.array([0.4, 0.2], dtype=np.float64),
        )
    )

    first_json = phase7_uncertainty_result_to_json(
        compute_tta_probability_variance(samples).uncertainty_result
    )
    second_json = phase7_uncertainty_result_to_json(
        compute_tta_probability_variance(samples).uncertainty_result
    )

    assert first_json == second_json


def test_noncontiguous_probability_arrays_produce_identical_identity_and_output() -> None:
    contiguous = _probability_samples(
        (
            np.array([0.2, 0.6, 0.7, 0.1], dtype=np.float64),
            np.array([0.4, 0.2, 0.1, 0.3], dtype=np.float64),
        ),
        shape=(1, 1, 1, 2, 2),
    )
    base_a = np.array([0.9, 0.2, 0.8, 0.6, 0.5, 0.7, 0.4, 0.1], dtype=np.float64)
    base_b = np.array([0.9, 0.4, 0.8, 0.2, 0.5, 0.1, 0.7, 0.3], dtype=np.float64)
    noncontiguous = _probability_samples(
        (
            base_a[1::2],
            base_b[1::2],
        ),
        shape=(1, 1, 1, 2, 2),
    )

    contiguous_result = compute_tta_probability_variance(contiguous)
    noncontiguous_result = compute_tta_probability_variance(noncontiguous)

    assert not base_a[1::2].flags.c_contiguous
    assert contiguous.tta_probability_samples_identity_hash == (
        noncontiguous.tta_probability_samples_identity_hash
    )
    assert np.array_equal(
        contiguous_result.foreground_variance_map,
        noncontiguous_result.foreground_variance_map,
    )
    assert contiguous_result.uncertainty_result == noncontiguous_result.uncertainty_result


def test_ensemble_interface_uses_explicit_probability_sample_aggregation() -> None:
    samples = _probability_samples(
        (
            np.array([0.2, 0.6], dtype=np.float64),
            np.array([0.4, 0.2], dtype=np.float64),
        )
    )

    result = compute_ensemble_probability_variance(samples)

    assert result.uncertainty_result.uncertainty_type == "ensemble_variance"
    assert result.uncertainty_result.source_probability_hash == (
        samples.tta_probability_samples_identity_hash
    )


def test_query_labels_reference_masks_and_model_callables_are_absent_from_public_api() -> None:
    public_contracts = (TTAVarianceResult,)
    prohibited_fields = {
        "query_label",
        "query_labels",
        "label_map",
        "reference_mask",
        "query_reference_mask",
        "model",
        "model_callable",
        "predictor",
    }

    for contract in public_contracts:
        field_names = {field.name for field in fields(contract)}
        assert field_names.isdisjoint(prohibited_fields)

    for function in (compute_tta_probability_variance, compute_ensemble_probability_variance):
        parameter_names = set(inspect.signature(function).parameters)
        assert parameter_names.isdisjoint(prohibited_fields)

    assert query_label_not_part_of_tta_api()


def _probability_samples(
    foreground_values: tuple[np.ndarray, ...],
    *,
    shape: tuple[int, int, int, int, int] = (1, 1, 1, 1, 2),
) -> TTAProbabilitySamples:
    probability_maps = tuple(
        _probability_map(f"sample-{index}", values, shape=shape)
        for index, values in enumerate(foreground_values)
    )
    manifest = build_tta_sample_manifest(
        tuple(
            _sample_record(index, probability_map)
            for index, probability_map in enumerate(probability_maps)
        )
    )
    return build_tta_probability_samples(
        sample_manifest=manifest,
        probability_maps=probability_maps,
    )


def _probability_map(
    sample_id: str,
    foreground_values: np.ndarray,
    *,
    shape: tuple[int, int, int, int, int] = (1, 1, 1, 1, 2),
    common_grid_identity_hash: str = "",
) -> BinaryPredictiveProbabilityMap:
    foreground = np.asarray(foreground_values, dtype=np.float64).reshape(shape)
    background = 1.0 - foreground
    return build_binary_predictive_probability_map(
        foreground_probability_map=foreground,
        background_probability_map=background,
        source_prediction_identity_hash=_sha(f"prediction-{sample_id}"),
        common_grid_identity_hash=common_grid_identity_hash or _sha("common-grid"),
    )


def _sample_record(
    sample_index: int,
    probability_map: BinaryPredictiveProbabilityMap,
    *,
    common_grid_identity_hash: str = "",
) -> TTASampleRecord:
    return build_tta_sample_record(
        sample_index=sample_index,
        sample_id=f"sample-{sample_index}",
        probability_map_identity_hash=probability_map.probability_map_identity_hash,
        common_grid_identity_hash=common_grid_identity_hash or _sha("common-grid"),
        transform_manifest_hash=_sha(f"transform-{sample_index}"),
    )


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
