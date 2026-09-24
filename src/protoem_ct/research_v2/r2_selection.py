"""Deterministic train-only case selection for Research-v2 R2."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

R2_EXPECTED_TRAIN_CASES: Final[int] = 91
R2_POSITIVE_QUANTILES: Final[tuple[float, float, float]] = (0.20, 0.50, 0.80)
R2_EXPECTED_POSITIVE_CASE_IDS: Final[tuple[str, str, str]] = (
    "case_65917a1f3a47339a558042dac093bf1a",
    "case_f232be7b649d2166e5158aa636b38766",
    "case_6e4782e999691662f69fdf25f0e2e423",
)
R2_EXPECTED_EMPTY_CASE_ID: Final[str] = "case_25928d2451a7d948a79e60084e3c5d32"


@dataclass(frozen=True, slots=True)
class R2SelectedCase:
    """Sanitized metadata for one deterministic R2 train case."""

    role: str
    anonymous_case_id: str
    anonymous_patient_id: str
    tumor_voxel_count: int
    tumor_volume_mm3: float
    tumor_lesion_count_26c: int


def _require_record(record: Mapping[str, Any]) -> None:
    required = {
        "partition",
        "anonymous_case_id",
        "anonymous_patient_id",
        "tumor_voxel_count",
        "tumor_volume_mm3",
        "tumor_lesion_count_26c",
    }
    missing = required - set(record)
    if missing:
        raise ValueError(f"R2 selection record is missing fields: {sorted(missing)}")
    if record["partition"] not in {"train", "validation"}:
        raise ValueError("R2 selection metadata contains an unexpected partition")
    if not isinstance(record["anonymous_case_id"], str) or not record["anonymous_case_id"]:
        raise ValueError("anonymous_case_id must be a non-empty string")
    if not isinstance(record["anonymous_patient_id"], str) or not record["anonymous_patient_id"]:
        raise ValueError("anonymous_patient_id must be a non-empty string")
    tumor_voxels = record["tumor_voxel_count"]
    if not isinstance(tumor_voxels, int) or isinstance(tumor_voxels, bool) or tumor_voxels < 0:
        raise ValueError("tumor_voxel_count must be a non-negative integer")
    tumor_volume = float(record["tumor_volume_mm3"])
    if not math.isfinite(tumor_volume) or tumor_volume < 0.0:
        raise ValueError("tumor_volume_mm3 must be finite and non-negative")
    lesion_count = record["tumor_lesion_count_26c"]
    if not isinstance(lesion_count, int) or isinstance(lesion_count, bool) or lesion_count < 0:
        raise ValueError("tumor_lesion_count_26c must be a non-negative integer")
    if (tumor_voxels == 0) != (tumor_volume == 0.0):
        raise ValueError("zero tumor voxels and zero tumor volume must agree")


def _selected_case(record: Mapping[str, Any], *, role: str) -> R2SelectedCase:
    return R2SelectedCase(
        role=role,
        anonymous_case_id=str(record["anonymous_case_id"]),
        anonymous_patient_id=str(record["anonymous_patient_id"]),
        tumor_voxel_count=int(record["tumor_voxel_count"]),
        tumor_volume_mm3=float(record["tumor_volume_mm3"]),
        tumor_lesion_count_26c=int(record["tumor_lesion_count_26c"]),
    )


def select_r2_train_cases(records: Sequence[Mapping[str, Any]]) -> tuple[R2SelectedCase, ...]:
    """Select three positive train cases plus one train-empty case when available.

    The function may receive sanitized train/validation metadata from R1, but it filters to train
    before ranking and never requires or authorizes medical-array access.
    """

    for record in records:
        _require_record(record)
    train = [record for record in records if record["partition"] == "train"]
    if len(train) != R2_EXPECTED_TRAIN_CASES:
        raise ValueError(
            f"R2 requires exactly {R2_EXPECTED_TRAIN_CASES} train metadata records; got {len(train)}"
        )
    case_ids = [str(record["anonymous_case_id"]) for record in train]
    patient_ids = [str(record["anonymous_patient_id"]) for record in train]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("train metadata contains duplicate anonymous_case_id values")
    if len(patient_ids) != len(set(patient_ids)):
        raise ValueError("train metadata contains duplicate anonymous_patient_id values")

    positives = sorted(
        (record for record in train if int(record["tumor_voxel_count"]) > 0),
        key=lambda record: (int(record["tumor_voxel_count"]), str(record["anonymous_case_id"])),
    )
    if len(positives) < len(R2_POSITIVE_QUANTILES):
        raise ValueError("R2 requires at least three positive train cases")

    selected: list[R2SelectedCase] = []
    for quantile in R2_POSITIVE_QUANTILES:
        index = math.floor((len(positives) - 1) * quantile)
        role = f"positive_q{int(round(quantile * 100)):02d}"
        selected.append(_selected_case(positives[index], role=role))

    empties = sorted(
        (record for record in train if int(record["tumor_voxel_count"]) == 0),
        key=lambda record: str(record["anonymous_case_id"]),
    )
    if empties:
        selected.append(_selected_case(empties[0], role="empty_lexicographic_first"))
    return tuple(selected)


def assert_closed_r1_expected_selection(selected: Sequence[R2SelectedCase]) -> None:
    """Fail closed if the deterministic selection differs from the preregistered R1 evidence."""

    positive_ids = tuple(case.anonymous_case_id for case in selected if case.tumor_voxel_count > 0)
    if positive_ids != R2_EXPECTED_POSITIVE_CASE_IDS:
        raise ValueError(
            "deterministic positive selection differs from the preregistered closed-R1 selection"
        )
    empty_ids = tuple(case.anonymous_case_id for case in selected if case.tumor_voxel_count == 0)
    if empty_ids != (R2_EXPECTED_EMPTY_CASE_ID,):
        raise ValueError(
            "deterministic empty-case selection differs from the preregistered closed-R1 selection"
        )


__all__ = [
    "R2_EXPECTED_EMPTY_CASE_ID",
    "R2_EXPECTED_POSITIVE_CASE_IDS",
    "R2_EXPECTED_TRAIN_CASES",
    "R2_POSITIVE_QUANTILES",
    "R2SelectedCase",
    "assert_closed_r1_expected_selection",
    "select_r2_train_cases",
]
