"""Unit tests for Phase 7 degradation utilities."""

from __future__ import annotations

import inspect
import math

import pytest

from protoem_ct.evaluation import degradation
from protoem_ct.evaluation.degradation import (
    Phase7DegradationInputError,
    build_phase7_degradation_result,
    compute_metric_degradation,
    query_label_not_part_of_phase7_degradation_api,
)
from protoem_ct.uncertainty.artifacts import (
    phase7_degradation_result_from_json,
    phase7_degradation_result_to_json,
)

HEX_1 = "1" * 64
HEX_2 = "2" * 64


def test_higher_is_better_degradation_formula() -> None:
    result = compute_metric_degradation(
        metric_name="dice",
        metric_direction="higher_is_better",
        baseline_value=0.8,
        corrupted_value=0.6,
    )

    assert result.absolute_degradation == pytest.approx(0.2)
    assert result.relative_degradation == pytest.approx(0.25)
    assert result.relative_available is True


def test_lower_is_better_degradation_formula() -> None:
    result = compute_metric_degradation(
        metric_name="risk",
        metric_direction="lower_is_better",
        baseline_value=0.2,
        corrupted_value=0.5,
    )

    assert result.absolute_degradation == pytest.approx(0.3)
    assert result.relative_degradation == pytest.approx(1.5)
    assert result.relative_available is True


def test_relative_degradation_near_zero_baseline_unavailable() -> None:
    result = compute_metric_degradation(
        metric_name="dice",
        metric_direction="higher_is_better",
        baseline_value=1e-10,
        corrupted_value=0.0,
        relative_epsilon=1e-8,
    )

    assert result.absolute_degradation == pytest.approx(1e-10)
    assert result.relative_degradation is None
    assert result.relative_available is False


def test_degradation_artifact_self_hash_and_deterministic_serialization() -> None:
    result = build_phase7_degradation_result(
        baseline_artifact_hash=HEX_1,
        corrupted_artifact_hash=HEX_2,
        metric_name="dice",
        metric_direction="higher_is_better",
        baseline_value=0.8,
        corrupted_value=0.6,
    )

    round_trip = phase7_degradation_result_from_json(phase7_degradation_result_to_json(result))
    assert round_trip == result
    assert phase7_degradation_result_to_json(round_trip) == phase7_degradation_result_to_json(
        result
    )


def test_lower_is_better_artifact_is_available() -> None:
    result = build_phase7_degradation_result(
        baseline_artifact_hash=HEX_1,
        corrupted_artifact_hash=HEX_2,
        metric_name="risk",
        metric_direction="lower_is_better",
        baseline_value=0.2,
        corrupted_value=0.5,
    )

    assert result.absolute_degradation == pytest.approx(0.3)
    assert result.relative_degradation == pytest.approx(1.5)
    assert result.availability_status == "available"
    assert result.unavailable_reason is None
    assert phase7_degradation_result_from_json(phase7_degradation_result_to_json(result)) == result


def test_invalid_degradation_inputs_rejected() -> None:
    with pytest.raises(Phase7DegradationInputError):
        compute_metric_degradation(
            metric_name="dice",
            metric_direction="higher_is_better",
            baseline_value=math.nan,
            corrupted_value=0.5,
        )
    with pytest.raises(Phase7DegradationInputError):
        compute_metric_degradation(
            metric_name="dice",
            metric_direction="sideways",
            baseline_value=0.5,
            corrupted_value=0.4,
        )
    with pytest.raises(Phase7DegradationInputError):
        build_phase7_degradation_result(
            baseline_artifact_hash="bad",
            corrupted_artifact_hash=HEX_2,
            metric_name="dice",
            metric_direction="higher_is_better",
            baseline_value=0.5,
            corrupted_value=0.4,
        )


def test_query_labels_absent_from_degradation_api() -> None:
    assert query_label_not_part_of_phase7_degradation_api()
    forbidden = {"query_label", "query_labels", "reference_mask", "prediction_mask"}
    for name, func in inspect.getmembers(degradation, inspect.isfunction):
        if name.startswith("_"):
            continue
        assert forbidden.isdisjoint(inspect.signature(func).parameters)
