from __future__ import annotations

import hashlib
import inspect
from dataclasses import fields

import numpy as np
import pytest

from protoem_ct.evaluation.calibration import (
    CalibrationComputationResult,
    CalibrationEvaluationError,
    compute_binary_calibration_ece,
    query_label_not_part_of_calibration_api,
)
from protoem_ct.uncertainty.artifacts import (
    Phase7UncertaintyArtifactHashError,
    phase7_calibration_result_from_json,
    phase7_calibration_result_to_json,
)
from protoem_ct.uncertainty.contracts import (
    BinaryPredictiveProbabilityMap,
    build_binary_predictive_probability_map,
)


def test_exact_ece_for_known_probabilities_and_reference() -> None:
    probabilities = _probability_map(np.array([[[[[0.9, 0.8, 0.6, 0.4]]]]], dtype=np.float64))
    reference = np.array([[[[[1, 1, 0, 0]]]]], dtype=np.uint8)

    result = compute_binary_calibration_ece(probabilities, reference, bin_count=5)

    assert result.calibration_result.valid_voxel_count == 4
    assert result.calibration_result.ece == pytest.approx(0.125)
    assert result.calibration_result.bins[3].voxel_count == 2
    assert result.calibration_result.bins[3].accuracy == pytest.approx(0.5)
    assert result.calibration_result.bins[3].mean_confidence == pytest.approx(0.6)
    assert result.calibration_result.bins[4].voxel_count == 2
    assert result.calibration_result.bins[4].accuracy == pytest.approx(1.0)
    assert result.calibration_result.bins[4].mean_confidence == pytest.approx(0.85)


def test_fixed_bin_boundary_policy_and_prediction_tie_handling() -> None:
    probabilities = _probability_map(np.array([[[[[0.5, 0.75, 1.0]]]]], dtype=np.float64))
    reference = np.array([[[[[0, 1, 1]]]]], dtype=np.uint8)

    result = compute_binary_calibration_ece(probabilities, reference, bin_count=4)

    assert result.prediction_map.tolist() == [[[[[False, True, True]]]]]
    assert result.calibration_result.bins[2].voxel_count == 1
    assert result.calibration_result.bins[2].bin_lower == 0.5
    assert result.calibration_result.bins[2].bin_upper == 0.75
    assert result.calibration_result.bins[3].voxel_count == 2
    assert result.calibration_result.bins[3].upper_inclusive is True


def test_empty_bins_are_retained_with_unavailable_bin_fields() -> None:
    probabilities = _probability_map(np.array([[[[[0.9, 0.95]]]]], dtype=np.float64))
    reference = np.array([[[[[1, 1]]]]], dtype=np.uint8)

    result = compute_binary_calibration_ece(probabilities, reference, bin_count=10)

    empty_bins = [item for item in result.calibration_result.bins if item.voxel_count == 0]
    assert empty_bins
    assert all(item.accuracy is None for item in empty_bins)
    assert all(item.mean_confidence is None for item in empty_bins)
    assert all(item.weighted_error == 0.0 for item in empty_bins)


def test_invalid_probability_or_reference_shape_is_rejected() -> None:
    probabilities = _probability_map(np.array([[[[[0.2, 0.8]]]]], dtype=np.float64))
    probabilities.foreground_probability_map[0, 0, 0, 0, 0] = np.nan

    with pytest.raises(CalibrationEvaluationError, match="reference_mask"):
        compute_binary_calibration_ece(
            _probability_map(np.array([[[[[0.2, 0.8]]]]], dtype=np.float64)),
            np.array([[[0, 1]]], dtype=np.uint8),
        )
    with pytest.raises(Exception, match="finite"):
        compute_binary_calibration_ece(
            probabilities,
            np.array([[[[[0, 1]]]]], dtype=np.uint8),
        )


def test_nonbinary_reference_is_rejected() -> None:
    probabilities = _probability_map(np.array([[[[[0.2, 0.8]]]]], dtype=np.float64))

    with pytest.raises(CalibrationEvaluationError, match="binary"):
        compute_binary_calibration_ece(
            probabilities,
            np.array([[[[[0, 2]]]]], dtype=np.uint8),
        )


def test_common_grid_hash_is_required_and_propagated() -> None:
    foreground = np.array([[[[[0.2, 0.8]]]]], dtype=np.float64)
    without_grid = build_binary_predictive_probability_map(
        foreground_probability_map=foreground,
        background_probability_map=1.0 - foreground,
        source_prediction_identity_hash=_sha("prediction"),
    )

    with pytest.raises(CalibrationEvaluationError, match="common_grid"):
        compute_binary_calibration_ece(
            without_grid,
            np.array([[[[[0, 1]]]]], dtype=np.uint8),
        )

    result = compute_binary_calibration_ece(
        without_grid,
        np.array([[[[[0, 1]]]]], dtype=np.uint8),
        common_grid_geometry_record_hash=_sha("common-grid"),
    )
    assert result.calibration_result.common_grid_geometry_record_hash == _sha("common-grid")


def test_common_grid_hash_mismatch_is_rejected() -> None:
    probabilities = _probability_map(np.array([[[[[0.2, 0.8]]]]], dtype=np.float64))

    with pytest.raises(CalibrationEvaluationError, match="must match"):
        compute_binary_calibration_ece(
            probabilities,
            np.array([[[[[0, 1]]]]], dtype=np.uint8),
            common_grid_geometry_record_hash=_sha("other-grid"),
        )


def test_artifact_self_hash_validates_and_is_deterministic() -> None:
    result = compute_binary_calibration_ece(
        _probability_map(np.array([[[[[0.2, 0.8]]]]], dtype=np.float64)),
        np.array([[[[[0, 1]]]]], dtype=np.uint8),
    )

    reconstructed = phase7_calibration_result_from_json(
        phase7_calibration_result_to_json(result.calibration_result)
    )
    assert reconstructed == result.calibration_result

    payload = (
        phase7_calibration_result_to_json(result.calibration_result)
        .decode("utf-8")
        .replace(result.calibration_result.calibration_result_hash, "0" * 64, 1)
    )
    with pytest.raises(Phase7UncertaintyArtifactHashError, match="hash"):
        phase7_calibration_result_from_json(payload)


def test_repeated_execution_is_byte_identical_at_canonical_artifact_level() -> None:
    probabilities = _probability_map(np.array([[[[[0.2, 0.8, 0.55, 0.45]]]]], dtype=np.float64))
    reference = np.array([[[[[0, 1, 1, 0]]]]], dtype=np.uint8)

    first = compute_binary_calibration_ece(probabilities, reference, bin_count=6)
    second = compute_binary_calibration_ece(probabilities, reference, bin_count=6)

    assert phase7_calibration_result_to_json(first.calibration_result) == (
        phase7_calibration_result_to_json(second.calibration_result)
    )
    assert np.array_equal(first.confidence_map, second.confidence_map)
    assert np.array_equal(first.prediction_map, second.prediction_map)


def test_query_labels_absent_from_non_evaluation_api() -> None:
    prohibited = {
        "query_label",
        "query_labels",
        "label_map",
        "query_reference_mask",
        "optimizer",
        "model",
        "predictor",
    }
    parameter_names = set(inspect.signature(compute_binary_calibration_ece).parameters)
    result_fields = {field.name for field in fields(CalibrationComputationResult)}

    assert prohibited.isdisjoint(parameter_names)
    assert prohibited.isdisjoint(result_fields)
    assert "reference_mask" in parameter_names
    assert query_label_not_part_of_calibration_api()


def _probability_map(foreground: np.ndarray) -> BinaryPredictiveProbabilityMap:
    return build_binary_predictive_probability_map(
        foreground_probability_map=foreground,
        background_probability_map=1.0 - foreground,
        source_prediction_identity_hash=_sha("prediction"),
        common_grid_identity_hash=_sha("common-grid"),
    )


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
