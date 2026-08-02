"""Deterministic Phase 7 lesion-size subgroup analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, cast

import numpy as np

from protoem_ct.artifacts.hashing import sha256_json
from protoem_ct.evaluation.metrics import (
    compute_binary_confusion_counts,
    dice_from_counts,
    iou_from_counts,
)
from protoem_ct.uncertainty.artifacts import (
    PHASE7_LESION_SUBGROUP_RECORD_SCHEMA_NAME,
    PHASE7_LESION_SUBGROUP_RECORD_SCHEMA_VERSION,
    PHASE7_LESION_SUBGROUP_RESULT_SCHEMA_NAME,
    PHASE7_LESION_SUBGROUP_RESULT_SCHEMA_VERSION,
    Phase7LesionSubgroupRecord,
    Phase7LesionSubgroupResult,
    phase7_lesion_subgroup_record_to_dict,
)
from protoem_ct.uncertainty.contracts import array_content_sha256

MaskMetricName = Literal["dice", "iou"]
LesionSubgroupName = Literal["empty", "small", "medium", "large"]

LESION_SUBGROUP_POLICY_NAME: Final[str] = "voxel_count_v1"
LESION_SUBGROUP_THRESHOLDS_VOXELS: Final[tuple[int, int, int]] = (0, 10, 100)
LESION_SUBGROUP_ORDER: Final[tuple[LesionSubgroupName, ...]] = (
    "empty",
    "small",
    "medium",
    "large",
)


class Phase7SubgroupError(ValueError):
    """Raised when Phase 7 lesion subgroup analysis cannot be computed."""


class Phase7SubgroupInputError(Phase7SubgroupError):
    """Raised when subgroup mask inputs are invalid."""


@dataclass(frozen=True, slots=True)
class LesionSubgroupCase:
    """One common-grid binary reference/prediction pair for evaluation only."""

    case_id: str
    reference_mask: np.ndarray
    prediction_mask: np.ndarray


@dataclass(frozen=True, slots=True)
class LesionSubgroupCaseMetric:
    """Per-case metric retained for deterministic subgroup aggregation."""

    case_id: str
    subgroup_name: LesionSubgroupName
    reference_foreground_voxels: int
    prediction_foreground_voxels: int
    metric_value: float


@dataclass(frozen=True, slots=True)
class LesionSubgroupAnalysis:
    """In-memory lesion subgroup analysis plus its self-validating artifact."""

    metric_name: MaskMetricName
    case_metrics: tuple[LesionSubgroupCaseMetric, ...]
    result: Phase7LesionSubgroupResult


def classify_lesion_size(foreground_voxel_count: int) -> LesionSubgroupName:
    """Classify a lesion by foreground voxel count.

    The policy is explicit and deterministic: empty equals 0 voxels, small is
    1-10 voxels, medium is 11-100 voxels, and large is greater than 100 voxels.
    """

    if foreground_voxel_count < 0:
        raise Phase7SubgroupInputError("foreground_voxel_count must be nonnegative.")
    if foreground_voxel_count == 0:
        return "empty"
    if foreground_voxel_count <= 10:
        return "small"
    if foreground_voxel_count <= 100:
        return "medium"
    return "large"


def build_phase7_lesion_subgroup_result(
    cases: tuple[LesionSubgroupCase, ...],
    *,
    metric_name: MaskMetricName,
) -> LesionSubgroupAnalysis:
    """Build deterministic lesion-size subgroup metrics from common-grid masks.

    Reference masks are accepted only by this evaluation API. Callers are
    responsible for restoring predictions and references to a validated common
    grid before calling this function.
    """

    if metric_name not in {"dice", "iou"}:
        raise Phase7SubgroupInputError("metric_name must be 'dice' or 'iou'.")
    if not cases:
        raise Phase7SubgroupInputError("at least one subgroup case is required.")

    seen_case_ids: set[str] = set()
    case_metrics: list[LesionSubgroupCaseMetric] = []
    reference_hash_records: list[dict[str, str]] = []
    prediction_hash_records: list[dict[str, str]] = []
    for case in cases:
        _require_case_id(case.case_id)
        if case.case_id in seen_case_ids:
            raise Phase7SubgroupInputError("case_id values must be unique.")
        seen_case_ids.add(case.case_id)
        reference = _require_binary_mask(case.reference_mask, field_name="reference_mask")
        prediction = _require_binary_mask(case.prediction_mask, field_name="prediction_mask")
        if reference.shape != prediction.shape:
            raise Phase7SubgroupInputError(
                "reference_mask and prediction_mask must share shape on the common grid."
            )
        counts = compute_binary_confusion_counts(reference, prediction)
        metric_value = (
            dice_from_counts(counts) if metric_name == "dice" else iou_from_counts(counts)
        )
        foreground_count = counts.ground_truth_foreground_voxels
        case_metrics.append(
            LesionSubgroupCaseMetric(
                case_id=case.case_id,
                subgroup_name=classify_lesion_size(foreground_count),
                reference_foreground_voxels=foreground_count,
                prediction_foreground_voxels=counts.predicted_foreground_voxels,
                metric_value=metric_value,
            )
        )
        reference_hash_records.append(
            {"case_id": case.case_id, "content_hash": array_content_sha256(reference)}
        )
        prediction_hash_records.append(
            {"case_id": case.case_id, "content_hash": array_content_sha256(prediction)}
        )

    records = tuple(
        _build_subgroup_record(
            subgroup_name=subgroup_name,
            case_metrics=tuple(case_metrics),
        )
        for subgroup_name in LESION_SUBGROUP_ORDER
    )
    payload = {
        "metric_name": metric_name,
        "prediction_content_hash": sha256_json({"cases": prediction_hash_records}),
        "records": [phase7_lesion_subgroup_record_to_dict(record) for record in records],
        "reference_mask_content_hash": sha256_json({"cases": reference_hash_records}),
        "schema_name": PHASE7_LESION_SUBGROUP_RESULT_SCHEMA_NAME,
        "schema_version": PHASE7_LESION_SUBGROUP_RESULT_SCHEMA_VERSION,
        "subgroup_policy_name": LESION_SUBGROUP_POLICY_NAME,
        "thresholds_voxels": list(LESION_SUBGROUP_THRESHOLDS_VOXELS),
    }
    result = Phase7LesionSubgroupResult(
        schema_name=PHASE7_LESION_SUBGROUP_RESULT_SCHEMA_NAME,
        schema_version=PHASE7_LESION_SUBGROUP_RESULT_SCHEMA_VERSION,
        lesion_subgroup_result_hash=sha256_json(payload),
        reference_mask_content_hash=cast(str, payload["reference_mask_content_hash"]),
        prediction_content_hash=cast(str, payload["prediction_content_hash"]),
        metric_name=metric_name,
        subgroup_policy_name=LESION_SUBGROUP_POLICY_NAME,
        thresholds_voxels=LESION_SUBGROUP_THRESHOLDS_VOXELS,
        records=records,
    )
    return LesionSubgroupAnalysis(
        metric_name=metric_name,
        case_metrics=tuple(case_metrics),
        result=result,
    )


def query_label_not_part_of_phase7_subgroup_prediction_api() -> bool:
    """Return True because subgroup labels are accepted only after prediction."""

    return True


def _build_subgroup_record(
    *,
    subgroup_name: LesionSubgroupName,
    case_metrics: tuple[LesionSubgroupCaseMetric, ...],
) -> Phase7LesionSubgroupRecord:
    matching_values = [
        case_metric.metric_value
        for case_metric in case_metrics
        if case_metric.subgroup_name == subgroup_name
    ]
    eligible_count = len(matching_values)
    empty_count = (
        sum(1 for case_metric in case_metrics if case_metric.subgroup_name == "empty")
        if subgroup_name == "empty"
        else 0
    )
    if matching_values:
        status = "available"
        mean_metric_value = float(np.mean(np.asarray(matching_values, dtype=np.float64)))
        unavailable_reason = None
    else:
        status = "unavailable"
        mean_metric_value = None
        unavailable_reason = "no_eligible_cases"
    return Phase7LesionSubgroupRecord(
        schema_name=PHASE7_LESION_SUBGROUP_RECORD_SCHEMA_NAME,
        schema_version=PHASE7_LESION_SUBGROUP_RECORD_SCHEMA_VERSION,
        subgroup_name=subgroup_name,
        eligible_case_count=eligible_count,
        empty_lesion_case_count=empty_count,
        skipped_case_count=0,
        metric_availability_status=status,
        mean_metric_value=mean_metric_value,
        unavailable_reason=unavailable_reason,
    )


def _require_case_id(value: str) -> None:
    if not value or not value.replace("_", "").replace("-", "").isalnum():
        raise Phase7SubgroupInputError("case_id must be a conservative identifier.")


def _require_binary_mask(mask: np.ndarray, *, field_name: str) -> np.ndarray:
    value = np.asarray(mask)
    if value.ndim != 3:
        raise Phase7SubgroupInputError(f"{field_name} must be a 3D common-grid mask.")
    if value.size == 0 or any(axis <= 0 for axis in value.shape):
        raise Phase7SubgroupInputError(f"{field_name} shape must be non-empty.")
    if value.dtype.kind in {"f", "c"} and not np.all(np.isfinite(value)):
        raise Phase7SubgroupInputError(f"{field_name} must contain finite values.")
    if not np.all((value == 0) | (value == 1)):
        raise Phase7SubgroupInputError(f"{field_name} must be binary with values 0 and 1.")
    return np.ascontiguousarray(value == 1, dtype=np.bool_)
