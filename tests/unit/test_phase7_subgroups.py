"""Unit tests for Phase 7 lesion-size subgroup analysis."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from protoem_ct.evaluation import subgroups
from protoem_ct.evaluation.subgroups import (
    LESION_SUBGROUP_THRESHOLDS_VOXELS,
    LesionSubgroupCase,
    Phase7SubgroupInputError,
    build_phase7_lesion_subgroup_result,
    classify_lesion_size,
    query_label_not_part_of_phase7_subgroup_prediction_api,
)
from protoem_ct.uncertainty.artifacts import (
    phase7_lesion_subgroup_result_from_json,
    phase7_lesion_subgroup_result_to_json,
)


def _mask_with_count(count: int) -> np.ndarray:
    mask = np.zeros((5, 5, 5), dtype=np.uint8)
    mask.reshape(-1)[:count] = 1
    return mask


def test_subgroup_classification_thresholds() -> None:
    assert LESION_SUBGROUP_THRESHOLDS_VOXELS == (0, 10, 100)
    assert classify_lesion_size(0) == "empty"
    assert classify_lesion_size(1) == "small"
    assert classify_lesion_size(10) == "small"
    assert classify_lesion_size(11) == "medium"
    assert classify_lesion_size(100) == "medium"
    assert classify_lesion_size(101) == "large"


def test_exact_subgroup_dice_on_known_masks() -> None:
    reference = np.zeros((2, 2, 2), dtype=np.uint8)
    prediction = np.zeros((2, 2, 2), dtype=np.uint8)
    reference.reshape(-1)[:4] = 1
    prediction.reshape(-1)[2:6] = 1

    analysis = build_phase7_lesion_subgroup_result(
        (LesionSubgroupCase("case_small", reference, prediction),),
        metric_name="dice",
    )

    small = next(record for record in analysis.result.records if record.subgroup_name == "small")
    assert small.metric_availability_status == "available"
    assert small.eligible_case_count == 1
    assert small.mean_metric_value == pytest.approx(0.5)


def test_exact_subgroup_iou_on_known_masks() -> None:
    reference = np.zeros((2, 2, 2), dtype=np.uint8)
    prediction = np.zeros((2, 2, 2), dtype=np.uint8)
    reference.reshape(-1)[:4] = 1
    prediction.reshape(-1)[2:6] = 1

    analysis = build_phase7_lesion_subgroup_result(
        (LesionSubgroupCase("case_small", reference, prediction),),
        metric_name="iou",
    )

    small = next(record for record in analysis.result.records if record.subgroup_name == "small")
    assert small.mean_metric_value == pytest.approx(2.0 / 6.0)


def test_empty_lesion_handling_and_counts_persisted() -> None:
    empty = np.zeros((2, 2, 2), dtype=np.uint8)
    analysis = build_phase7_lesion_subgroup_result(
        (
            LesionSubgroupCase("empty_case", empty, empty),
            LesionSubgroupCase("small_case", _mask_with_count(2), _mask_with_count(2)),
        ),
        metric_name="dice",
    )

    empty_record = next(
        record for record in analysis.result.records if record.subgroup_name == "empty"
    )
    large_record = next(
        record for record in analysis.result.records if record.subgroup_name == "large"
    )
    assert empty_record.eligible_case_count == 1
    assert empty_record.empty_lesion_case_count == 1
    assert empty_record.mean_metric_value == pytest.approx(1.0)
    assert large_record.metric_availability_status == "unavailable"
    assert large_record.unavailable_reason == "no_eligible_cases"


def test_nonbinary_and_shape_rejection() -> None:
    reference = np.zeros((2, 2, 2), dtype=np.uint8)
    prediction = np.zeros((2, 2, 2), dtype=np.uint8)
    nonbinary = reference.copy()
    nonbinary[0, 0, 0] = 2
    with pytest.raises(Phase7SubgroupInputError):
        build_phase7_lesion_subgroup_result(
            (LesionSubgroupCase("case_a", nonbinary, prediction),),
            metric_name="dice",
        )
    with pytest.raises(Phase7SubgroupInputError):
        build_phase7_lesion_subgroup_result(
            (LesionSubgroupCase("case_a", reference, prediction[:, :, :1]),),
            metric_name="dice",
        )
    with pytest.raises(Phase7SubgroupInputError):
        build_phase7_lesion_subgroup_result(
            (LesionSubgroupCase("case_a", reference[0], prediction[0]),),
            metric_name="dice",
        )


def test_subgroup_artifact_deterministic_and_self_hashing() -> None:
    cases = (
        LesionSubgroupCase("case_empty", _mask_with_count(0), _mask_with_count(0)),
        LesionSubgroupCase("case_small", _mask_with_count(3), _mask_with_count(2)),
        LesionSubgroupCase("case_medium", _mask_with_count(20), _mask_with_count(18)),
    )

    first = build_phase7_lesion_subgroup_result(cases, metric_name="dice")
    second = build_phase7_lesion_subgroup_result(cases, metric_name="dice")

    assert second.result == first.result
    assert phase7_lesion_subgroup_result_to_json(second.result) == (
        phase7_lesion_subgroup_result_to_json(first.result)
    )
    assert (
        phase7_lesion_subgroup_result_from_json(phase7_lesion_subgroup_result_to_json(first.result))
        == first.result
    )


def test_non_contiguous_masks_produce_same_subgroup_artifact() -> None:
    reference = np.zeros((3, 4, 5), dtype=np.uint8)
    prediction = np.zeros((3, 4, 5), dtype=np.uint8)
    reference[:, :2, :2] = 1
    prediction[:, :2, :2] = 1

    contiguous = build_phase7_lesion_subgroup_result(
        (LesionSubgroupCase("case_a", reference, prediction),),
        metric_name="dice",
    )
    non_contiguous = build_phase7_lesion_subgroup_result(
        (LesionSubgroupCase("case_a", reference[:, :, ::-1][:, :, ::-1], prediction.T.T),),
        metric_name="dice",
    )

    assert non_contiguous.result == contiguous.result


def test_query_labels_absent_from_non_prediction_subgroup_api() -> None:
    assert query_label_not_part_of_phase7_subgroup_prediction_api()
    forbidden = {"query_label", "query_labels", "label_map", "query_reference_mask"}
    for name, func in inspect.getmembers(subgroups, inspect.isfunction):
        if name.startswith("_"):
            continue
        assert forbidden.isdisjoint(inspect.signature(func).parameters)
