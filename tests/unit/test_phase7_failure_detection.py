"""Focused tests for deterministic Phase 7 failure analysis."""

from __future__ import annotations

import inspect
import math

import numpy as np
import pytest

from protoem_ct.evaluation.failure_detection import (
    FailureDetectionInputError,
    compute_failure_detection_auroc,
    compute_uncertainty_error_correlation,
    failure_detection_auroc_result_to_json,
    failure_detection_public_prediction_api_has_no_labels,
    query_label_not_part_of_failure_detection_prediction_api,
    uncertainty_error_correlation_result_to_json,
)


def test_exact_pearson_correlation_known_case() -> None:
    result = compute_uncertainty_error_correlation(
        uncertainty_values=np.array([0.0, 1.0, 2.0, 3.0]),
        error_indicators=np.array([0, 0, 1, 1]),
    )

    assert result.availability_status == "available"
    assert result.unavailable_reason is None
    assert result.correlation_method == "pearson"
    assert result.valid_voxel_count == 4
    assert result.correlation_value == pytest.approx(2.0 / math.sqrt(5.0))


def test_correlation_uses_valid_mask() -> None:
    result = compute_uncertainty_error_correlation(
        uncertainty_values=np.array([0.0, 1.0, 2.0, 3.0]),
        error_indicators=np.array([0, 0, 1, 1]),
        valid_mask=np.array([False, True, True, True]),
    )

    assert result.availability_status == "available"
    assert result.valid_voxel_count == 3
    assert result.valid_mask_content_hash is not None


def test_zero_variance_uncertainty_unavailable() -> None:
    result = compute_uncertainty_error_correlation(
        uncertainty_values=np.array([1.0, 1.0, 1.0, 1.0]),
        error_indicators=np.array([0, 1, 0, 1]),
    )

    assert result.availability_status == "unavailable"
    assert result.correlation_value is None
    assert result.unavailable_reason == "constant_uncertainty"


def test_zero_variance_error_unavailable() -> None:
    result = compute_uncertainty_error_correlation(
        uncertainty_values=np.array([0.1, 0.2, 0.3]),
        error_indicators=np.array([1, 1, 1]),
    )

    assert result.availability_status == "unavailable"
    assert result.correlation_value is None
    assert result.unavailable_reason == "constant_error"


def test_exact_auroc_known_case() -> None:
    result = compute_failure_detection_auroc(
        case_uncertainty_scores=np.array([0.1, 0.4, 0.8, 0.9]),
        failure_indicators=np.array([0, 0, 1, 1]),
    )

    assert result.availability_status == "available"
    assert result.auroc_method == "mann_whitney_pairwise"
    assert result.case_count == 4
    assert result.positive_failure_case_count == 2
    assert result.negative_nonfailure_case_count == 2
    assert result.auroc_value == pytest.approx(1.0)


def test_auroc_tie_handling() -> None:
    result = compute_failure_detection_auroc(
        case_uncertainty_scores=np.array([0.5, 0.5, 0.2, 0.8]),
        failure_indicators=np.array([1, 0, 0, 1]),
    )

    assert result.availability_status == "available"
    assert result.auroc_value == pytest.approx(0.875)


@pytest.mark.parametrize(
    ("scores", "failures", "reason"),
    [
        (np.array([0.2]), np.array([1]), "insufficient_cases"),
        (np.array([0.2, 0.3]), np.array([1, 1]), "one_class_failure_labels"),
        (np.array([0.2, 0.3]), np.array([0, 0]), "one_class_failure_labels"),
    ],
)
def test_auroc_ineligible_inputs_return_unavailable(
    scores: np.ndarray,
    failures: np.ndarray,
    reason: str,
) -> None:
    result = compute_failure_detection_auroc(
        case_uncertainty_scores=scores,
        failure_indicators=failures,
    )

    assert result.availability_status == "unavailable"
    assert result.auroc_value is None
    assert result.unavailable_reason == reason


@pytest.mark.parametrize(
    ("uncertainty", "errors"),
    [
        (np.array([0.1, np.nan]), np.array([0, 1])),
        (np.array([0.1, np.inf]), np.array([0, 1])),
        (np.array([0.1, 0.2]), np.array([0, 2])),
    ],
)
def test_invalid_correlation_inputs_rejected(
    uncertainty: np.ndarray,
    errors: np.ndarray,
) -> None:
    with pytest.raises(FailureDetectionInputError):
        compute_uncertainty_error_correlation(
            uncertainty_values=uncertainty,
            error_indicators=errors,
        )


@pytest.mark.parametrize(
    ("scores", "failures"),
    [
        (np.array([0.1, np.nan]), np.array([0, 1])),
        (np.array([0.1, np.inf]), np.array([0, 1])),
        (np.array([0.1, 0.2]), np.array([0, 2])),
    ],
)
def test_invalid_auroc_inputs_rejected(scores: np.ndarray, failures: np.ndarray) -> None:
    with pytest.raises(FailureDetectionInputError):
        compute_failure_detection_auroc(
            case_uncertainty_scores=scores,
            failure_indicators=failures,
        )


def test_result_identities_are_deterministic_and_json_is_byte_identical() -> None:
    first = compute_uncertainty_error_correlation(
        uncertainty_values=np.array([[0.0, 1.0], [2.0, 3.0]]),
        error_indicators=np.array([[0, 0], [1, 1]]),
    )
    second = compute_uncertainty_error_correlation(
        uncertainty_values=np.asfortranarray(np.array([[0.0, 1.0], [2.0, 3.0]])),
        error_indicators=np.asfortranarray(np.array([[0, 0], [1, 1]])),
    )
    assert first.correlation_result_hash == second.correlation_result_hash
    assert uncertainty_error_correlation_result_to_json(
        first
    ) == uncertainty_error_correlation_result_to_json(second)

    first_auc = compute_failure_detection_auroc(
        case_uncertainty_scores=np.array([0.1, 0.4, 0.8, 0.9]),
        failure_indicators=np.array([0, 0, 1, 1]),
    )
    second_auc = compute_failure_detection_auroc(
        case_uncertainty_scores=np.asfortranarray(np.array([0.1, 0.4, 0.8, 0.9])),
        failure_indicators=np.asfortranarray(np.array([0, 0, 1, 1])),
    )
    assert first_auc.auroc_result_hash == second_auc.auroc_result_hash
    assert failure_detection_auroc_result_to_json(
        first_auc
    ) == failure_detection_auroc_result_to_json(second_auc)


def test_no_query_labels_or_reference_masks_in_prediction_apis() -> None:
    assert query_label_not_part_of_failure_detection_prediction_api()
    assert failure_detection_public_prediction_api_has_no_labels()
    for function in (
        compute_uncertainty_error_correlation,
        compute_failure_detection_auroc,
    ):
        parameter_names = set(inspect.signature(function).parameters)
        assert "query_label" not in parameter_names
        assert "query_labels" not in parameter_names
        assert "reference_mask" not in parameter_names
        assert "query_reference_mask" not in parameter_names
