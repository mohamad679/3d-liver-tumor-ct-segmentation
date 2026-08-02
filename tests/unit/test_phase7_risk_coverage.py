from __future__ import annotations

import inspect

import numpy as np
import pytest

from protoem_ct.evaluation.risk_coverage import (
    RiskCoverageInputError,
    compute_risk_coverage,
    derive_hard_prediction_from_binary_probabilities,
    query_label_not_part_of_risk_coverage_prediction_api,
)
from protoem_ct.uncertainty.artifacts import (
    Phase7RiskCoverageResult,
    phase7_risk_coverage_result_from_json,
    phase7_risk_coverage_result_to_json,
)

HEX_1 = "1" * 64
HEX_2 = "2" * 64


def _volume(values: list[int]) -> np.ndarray:
    return np.asarray(values, dtype=np.uint8).reshape(1, 1, 1, 1, len(values))


def _float_volume(values: list[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(1, 1, 1, 1, len(values))


def test_exact_risk_coverage_points_for_known_arrays() -> None:
    prediction = _volume([0, 1, 1, 0])
    reference = _volume([0, 0, 1, 1])
    uncertainty = _float_volume([0.2, 0.1, 0.4, 0.3])

    result = compute_risk_coverage(
        prediction_map=prediction,
        reference_mask=reference,
        uncertainty_map=uncertainty,
        uncertainty_result_hash=HEX_1,
        common_grid_geometry_record_hash=HEX_2,
    )

    assert result.retained_indices == (1, 0, 3, 2)
    assert [point.retained_voxel_count for point in result.risk_coverage_result.points] == [
        1,
        2,
        3,
        4,
    ]
    np.testing.assert_allclose(
        [point.coverage for point in result.risk_coverage_result.points],
        [0.25, 0.5, 0.75, 1.0],
    )
    np.testing.assert_allclose(
        [point.risk for point in result.risk_coverage_result.points],
        [1.0, 0.5, 2.0 / 3.0, 0.5],
    )
    assert result.risk_coverage_result.availability_status == "available"


def test_tie_breaks_by_flattened_index() -> None:
    prediction = _volume([1, 1, 0, 0])
    reference = _volume([0, 1, 0, 1])
    uncertainty = _float_volume([0.5, 0.2, 0.5, 0.2])

    result = compute_risk_coverage(
        prediction_map=prediction,
        reference_mask=reference,
        uncertainty_map=uncertainty,
        uncertainty_result_hash=HEX_1,
        common_grid_geometry_record_hash=HEX_2,
    )

    assert result.retained_indices == (1, 3, 0, 2)
    np.testing.assert_allclose(
        [point.risk for point in result.risk_coverage_result.points],
        [0.0, 0.5, 2.0 / 3.0, 0.5],
    )


def test_prediction_can_be_derived_from_probabilities_with_ties_background() -> None:
    foreground = _float_volume([0.7, 0.5, 0.2, 1.0])
    background = _float_volume([0.3, 0.5, 0.8, 0.0])

    prediction = derive_hard_prediction_from_binary_probabilities(
        foreground_probability_map=foreground,
        background_probability_map=background,
    )

    np.testing.assert_array_equal(prediction, _volume([1, 0, 0, 1]))
    result = compute_risk_coverage(
        foreground_probability_map=foreground,
        background_probability_map=background,
        reference_mask=_volume([1, 0, 1, 1]),
        uncertainty_map=_float_volume([0.1, 0.2, 0.3, 0.4]),
        uncertainty_result_hash=HEX_1,
        common_grid_geometry_record_hash=HEX_2,
    )
    assert [point.risk for point in result.risk_coverage_result.points][-1] == 0.25


def test_invalid_shape_rejected() -> None:
    with pytest.raises(RiskCoverageInputError, match=r"\[1,1,D,H,W\]"):
        compute_risk_coverage(
            prediction_map=np.zeros((1, 2, 2), dtype=np.uint8),
            reference_mask=_volume([0, 1, 0, 1]),
            uncertainty_map=_float_volume([0.1, 0.2, 0.3, 0.4]),
            uncertainty_result_hash=HEX_1,
            common_grid_geometry_record_hash=HEX_2,
        )


@pytest.mark.parametrize(
    ("field_name", "prediction", "reference"),
    [
        ("prediction_map", _volume([0, 2, 0, 1]), _volume([0, 1, 0, 1])),
        ("reference_mask", _volume([0, 1, 0, 1]), _volume([0, 1, 3, 1])),
    ],
)
def test_nonbinary_prediction_or_reference_rejected(
    field_name: str,
    prediction: np.ndarray,
    reference: np.ndarray,
) -> None:
    with pytest.raises(RiskCoverageInputError, match=field_name):
        compute_risk_coverage(
            prediction_map=prediction,
            reference_mask=reference,
            uncertainty_map=_float_volume([0.1, 0.2, 0.3, 0.4]),
            uncertainty_result_hash=HEX_1,
            common_grid_geometry_record_hash=HEX_2,
        )


def test_no_valid_voxel_returns_unavailable_artifact() -> None:
    result = compute_risk_coverage(
        prediction_map=_volume([0, 1, 0, 1]),
        reference_mask=_volume([0, 1, 1, 1]),
        uncertainty_map=_float_volume([0.1, 0.2, 0.3, 0.4]),
        uncertainty_result_hash=HEX_1,
        common_grid_geometry_record_hash=HEX_2,
        valid_voxel_mask=np.zeros((1, 1, 1, 1, 4), dtype=bool),
    )

    artifact = result.risk_coverage_result
    assert artifact.availability_status == "unavailable"
    assert artifact.unavailable_reason == "no_valid_voxels"
    assert artifact.valid_voxel_count == 0
    assert artifact.points == ()


def test_artifact_self_hash_valid_and_deterministic() -> None:
    first = compute_risk_coverage(
        prediction_map=_volume([0, 1, 1, 0]),
        reference_mask=_volume([0, 0, 1, 1]),
        uncertainty_map=_float_volume([0.2, 0.1, 0.4, 0.3]),
        uncertainty_result_hash=HEX_1,
        common_grid_geometry_record_hash=HEX_2,
    ).risk_coverage_result
    second = compute_risk_coverage(
        prediction_map=np.asfortranarray(_volume([0, 1, 1, 0])),
        reference_mask=np.asfortranarray(_volume([0, 0, 1, 1])),
        uncertainty_map=np.asfortranarray(_float_volume([0.2, 0.1, 0.4, 0.3])),
        uncertainty_result_hash=HEX_1,
        common_grid_geometry_record_hash=HEX_2,
    ).risk_coverage_result

    assert isinstance(first, Phase7RiskCoverageResult)
    assert first.risk_coverage_result_hash == second.risk_coverage_result_hash
    assert (
        phase7_risk_coverage_result_from_json(phase7_risk_coverage_result_to_json(first)) == first
    )


def test_repeated_execution_byte_identical_at_canonical_artifact_level() -> None:
    prediction = _volume([1, 0, 1, 0])
    reference = _volume([1, 1, 1, 0])
    uncertainty = _float_volume([0.3, 0.2, 0.1, 0.4])

    first = compute_risk_coverage(
        prediction_map=prediction,
        reference_mask=reference,
        uncertainty_map=uncertainty,
        uncertainty_result_hash=HEX_1,
        common_grid_geometry_record_hash=HEX_2,
    ).risk_coverage_result
    second = compute_risk_coverage(
        prediction_map=prediction,
        reference_mask=reference,
        uncertainty_map=uncertainty,
        uncertainty_result_hash=HEX_1,
        common_grid_geometry_record_hash=HEX_2,
    ).risk_coverage_result

    assert phase7_risk_coverage_result_to_json(first) == phase7_risk_coverage_result_to_json(second)


def test_query_labels_absent_from_public_prediction_apis() -> None:
    assert query_label_not_part_of_risk_coverage_prediction_api()
    forbidden = {"query_label", "query_labels", "label_map", "query_reference_mask"}
    for public_callable in (
        compute_risk_coverage,
        derive_hard_prediction_from_binary_probabilities,
    ):
        assert forbidden.isdisjoint(inspect.signature(public_callable).parameters)
