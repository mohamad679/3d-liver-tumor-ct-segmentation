from __future__ import annotations

import pytest

from protoem_ct.research_v2.r2_selection import (
    R2_EXPECTED_EMPTY_CASE_ID,
    R2_EXPECTED_POSITIVE_CASE_IDS,
    assert_closed_r1_expected_selection,
    select_r2_train_cases,
)


def _record(
    index: int,
    *,
    partition: str = "train",
    tumor_voxels: int,
    case_id: str | None = None,
) -> dict[str, object]:
    return {
        "partition": partition,
        "anonymous_case_id": case_id or f"case_synthetic_{index:03d}",
        "anonymous_patient_id": f"pat_synthetic_{index:03d}_{partition}",
        "tumor_voxel_count": tumor_voxels,
        "tumor_volume_mm3": float(tumor_voxels) if tumor_voxels else 0.0,
        "tumor_lesion_count_26c": 1 if tumor_voxels else 0,
    }


def _closed_r1_shape_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    expected_by_positive_rank = {
        17: R2_EXPECTED_POSITIVE_CASE_IDS[0],
        42: R2_EXPECTED_POSITIVE_CASE_IDS[1],
        68: R2_EXPECTED_POSITIVE_CASE_IDS[2],
    }
    for positive_rank in range(86):
        records.append(
            _record(
                positive_rank,
                tumor_voxels=positive_rank + 1,
                case_id=expected_by_positive_rank.get(positive_rank),
            )
        )
    records.append(_record(200, tumor_voxels=0, case_id=R2_EXPECTED_EMPTY_CASE_ID))
    for index in range(4):
        records.append(_record(201 + index, tumor_voxels=0, case_id=f"case_z_empty_{index}"))
    return records


def test_select_r2_train_cases_uses_preregistered_quantile_ranks_and_empty_rule() -> None:
    selected = select_r2_train_cases(_closed_r1_shape_records())
    assert [case.role for case in selected] == [
        "positive_q20",
        "positive_q50",
        "positive_q80",
        "empty_lexicographic_first",
    ]
    assert tuple(case.anonymous_case_id for case in selected[:3]) == R2_EXPECTED_POSITIVE_CASE_IDS
    assert selected[3].anonymous_case_id == R2_EXPECTED_EMPTY_CASE_ID
    assert_closed_r1_expected_selection(selected)


def test_validation_metadata_cannot_change_train_selection() -> None:
    records = _closed_r1_shape_records()
    baseline = select_r2_train_cases(records)
    records.extend(
        [
            _record(
                500 + index,
                partition="validation",
                tumor_voxels=10_000_000 + index,
                case_id=f"case_validation_{index}",
            )
            for index in range(20)
        ]
    )
    assert select_r2_train_cases(records) == baseline


def test_train_count_drift_fails_closed() -> None:
    records = _closed_r1_shape_records()[:-1]
    with pytest.raises(ValueError, match="exactly 91 train metadata records"):
        select_r2_train_cases(records)


def test_duplicate_train_case_id_fails_closed() -> None:
    records = _closed_r1_shape_records()
    records[-1]["anonymous_case_id"] = records[-2]["anonymous_case_id"]
    with pytest.raises(ValueError, match="duplicate anonymous_case_id"):
        select_r2_train_cases(records)


def test_inconsistent_empty_tumor_metadata_fails_closed() -> None:
    records = _closed_r1_shape_records()
    records[-1]["tumor_volume_mm3"] = 1.0
    with pytest.raises(ValueError, match="zero tumor voxels and zero tumor volume"):
        select_r2_train_cases(records)
