"""Deterministic binary-tumor segmentation metrics and artifact contracts."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
from scipy import ndimage  # type: ignore[import-untyped]
from scipy.optimize import linear_sum_assignment  # type: ignore[import-untyped]

from protoem_ct.artifacts.hashing import canonical_json_bytes, sha256_json

BASELINE_CASE_METRICS_VERSION = "baseline_case_metrics_v1"
BASELINE_METRIC_REPORT_VERSION = "baseline_metric_report_v1"
BASELINE_SURFACE_CONNECTIVITY = 6
BASELINE_LESION_CONNECTIVITY = 26
SUPPORTED_BASELINE_METRIC_FAMILIES: tuple[str, ...] = ("nnunet_v2", "monai_segresnet")

_SAFE_IDENTIFIER_RE = re.compile(r"^[a-z0-9](?:[a-z0-9_-]{0,62}[a-z0-9])?$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_WINDOWS_ABSOLUTE_PATH_RE = re.compile(r"^[A-Za-z]:[\\/]")
_CASE_FIELDS = {
    "absolute_volume_error_ml",
    "artifact_hash",
    "case_identifier",
    "contract_version",
    "false_negative_lesion_count",
    "false_positive_lesion_count",
    "false_positive_lesions_per_scan",
    "ground_truth_foreground_voxels",
    "ground_truth_lesion_count",
    "ground_truth_volume_ml",
    "hd95_mm",
    "intersection_voxels",
    "lesion_f1",
    "lesion_precision",
    "lesion_recall",
    "normalized_surface_dice",
    "nsd_tolerance_mm",
    "predicted_foreground_voxels",
    "predicted_lesion_count",
    "predicted_volume_ml",
    "relative_volume_error",
    "signed_volume_error_ml",
    "true_positive_lesion_count",
    "tumor_dice",
    "tumor_iou",
    "union_voxels",
    "voxel_spacing_mm",
}
_REPORT_FIELDS = {
    "aggregate",
    "artifact_hash",
    "baseline_family",
    "case_records",
    "contract_version",
    "dataset_manifest_sha256",
    "development_split_sha256",
    "lesion_connectivity",
    "metric_config_sha256",
    "nsd_tolerance_mm",
    "prediction_manifest_sha256",
    "run_identifier",
}
_AGGREGATE_FIELDS = {
    "case_count",
    "defined_hd95_case_count",
    "defined_lesion_precision_case_count",
    "defined_lesion_recall_case_count",
    "defined_relative_volume_error_case_count",
    "macro_dice",
    "macro_hd95_mm",
    "macro_iou",
    "macro_lesion_f1",
    "macro_lesion_precision",
    "macro_lesion_recall",
    "macro_normalized_surface_dice",
    "macro_relative_volume_error",
    "mean_absolute_volume_error_ml",
    "mean_false_positive_lesions_per_scan",
    "mean_signed_volume_error_ml",
    "micro_lesion_f1",
    "micro_lesion_precision",
    "micro_lesion_recall",
    "pooled_dice",
    "pooled_iou",
    "total_false_negative_lesions",
    "total_false_positive_lesions",
    "total_ground_truth_lesions",
    "total_ground_truth_voxels",
    "total_intersection_voxels",
    "total_predicted_lesions",
    "total_predicted_voxels",
    "total_true_positive_lesions",
    "total_union_voxels",
}


class BaselineMetricError(ValueError):
    """Base error for deterministic baseline metric contracts."""


class BaselineMetricValidationError(BaselineMetricError):
    """Raised when metric inputs or persisted values violate the contract."""


class BaselineMetricSerializationError(BaselineMetricError):
    """Raised when metric artifacts cannot be serialized or parsed safely."""


class BaselineMetricVersionError(BaselineMetricError):
    """Raised when an unsupported metric contract version is encountered."""


class BaselineMetricHashError(BaselineMetricError):
    """Raised when a persisted artifact hash does not match its content."""


@dataclass(frozen=True, slots=True)
class BaselineCaseMetrics:
    """Immutable deterministic metrics for one anonymous 3D binary case."""

    artifact_hash: str
    contract_version: str
    case_identifier: str
    voxel_spacing_mm: tuple[float, float, float]
    nsd_tolerance_mm: float
    ground_truth_foreground_voxels: int
    predicted_foreground_voxels: int
    intersection_voxels: int
    union_voxels: int
    tumor_dice: float
    tumor_iou: float
    hd95_mm: float | None
    normalized_surface_dice: float
    ground_truth_lesion_count: int
    predicted_lesion_count: int
    true_positive_lesion_count: int
    false_positive_lesion_count: int
    false_negative_lesion_count: int
    lesion_recall: float | None
    lesion_precision: float | None
    lesion_f1: float
    false_positive_lesions_per_scan: int
    ground_truth_volume_ml: float
    predicted_volume_ml: float
    signed_volume_error_ml: float
    absolute_volume_error_ml: float
    relative_volume_error: float | None

    def __post_init__(self) -> None:
        _require_contract_version(self.contract_version, BASELINE_CASE_METRICS_VERSION)
        _require_sha256(self.artifact_hash, field_name="artifact_hash")
        _require_case_identifier(self.case_identifier)
        spacing = _normalize_spacing(self.voxel_spacing_mm)
        object.__setattr__(self, "voxel_spacing_mm", spacing)
        tolerance = _normalize_nonnegative_float(
            self.nsd_tolerance_mm,
            field_name="nsd_tolerance_mm",
        )
        object.__setattr__(self, "nsd_tolerance_mm", tolerance)
        _require_nonnegative_int(
            self.ground_truth_foreground_voxels,
            field_name="ground_truth_foreground_voxels",
        )
        _require_nonnegative_int(
            self.predicted_foreground_voxels,
            field_name="predicted_foreground_voxels",
        )
        _require_nonnegative_int(self.intersection_voxels, field_name="intersection_voxels")
        _require_nonnegative_int(self.union_voxels, field_name="union_voxels")
        _require_unit_interval(self.tumor_dice, field_name="tumor_dice")
        _require_unit_interval(self.tumor_iou, field_name="tumor_iou")
        _require_optional_nonnegative_float(self.hd95_mm, field_name="hd95_mm")
        _require_unit_interval(
            self.normalized_surface_dice,
            field_name="normalized_surface_dice",
        )
        _require_nonnegative_int(
            self.ground_truth_lesion_count,
            field_name="ground_truth_lesion_count",
        )
        _require_nonnegative_int(
            self.predicted_lesion_count,
            field_name="predicted_lesion_count",
        )
        _require_nonnegative_int(
            self.true_positive_lesion_count,
            field_name="true_positive_lesion_count",
        )
        _require_nonnegative_int(
            self.false_positive_lesion_count,
            field_name="false_positive_lesion_count",
        )
        _require_nonnegative_int(
            self.false_negative_lesion_count,
            field_name="false_negative_lesion_count",
        )
        _require_optional_unit_interval(self.lesion_recall, field_name="lesion_recall")
        _require_optional_unit_interval(self.lesion_precision, field_name="lesion_precision")
        _require_unit_interval(self.lesion_f1, field_name="lesion_f1")
        _require_nonnegative_int(
            self.false_positive_lesions_per_scan,
            field_name="false_positive_lesions_per_scan",
        )
        _require_nonnegative_float(
            self.ground_truth_volume_ml,
            field_name="ground_truth_volume_ml",
        )
        _require_nonnegative_float(self.predicted_volume_ml, field_name="predicted_volume_ml")
        _require_finite_float(self.signed_volume_error_ml, field_name="signed_volume_error_ml")
        _require_nonnegative_float(
            self.absolute_volume_error_ml,
            field_name="absolute_volume_error_ml",
        )
        _require_optional_finite_float(
            self.relative_volume_error,
            field_name="relative_volume_error",
        )
        _validate_case_consistency(self)
        _require_no_absolute_paths(_case_metrics_payload(self))
        expected_hash = sha256_json(_case_metrics_payload_without_hash(self))
        if self.artifact_hash != expected_hash:
            raise BaselineMetricHashError(
                "artifact_hash does not match the deterministic case-metrics content."
            )


@dataclass(frozen=True, slots=True)
class BaselineMetricAggregate:
    """Deterministic aggregate metrics across an ordered case tuple."""

    case_count: int
    total_ground_truth_voxels: int
    total_predicted_voxels: int
    total_intersection_voxels: int
    total_union_voxels: int
    pooled_dice: float
    pooled_iou: float
    macro_dice: float
    macro_iou: float
    macro_normalized_surface_dice: float
    macro_hd95_mm: float | None
    defined_hd95_case_count: int
    total_ground_truth_lesions: int
    total_predicted_lesions: int
    total_true_positive_lesions: int
    total_false_positive_lesions: int
    total_false_negative_lesions: int
    micro_lesion_recall: float | None
    micro_lesion_precision: float | None
    micro_lesion_f1: float
    macro_lesion_recall: float | None
    macro_lesion_precision: float | None
    macro_lesion_f1: float
    defined_lesion_recall_case_count: int
    defined_lesion_precision_case_count: int
    mean_false_positive_lesions_per_scan: float
    mean_signed_volume_error_ml: float
    mean_absolute_volume_error_ml: float
    macro_relative_volume_error: float | None
    defined_relative_volume_error_case_count: int

    def __post_init__(self) -> None:
        _require_nonnegative_int(self.case_count, field_name="case_count")
        _require_nonnegative_int(
            self.total_ground_truth_voxels,
            field_name="total_ground_truth_voxels",
        )
        _require_nonnegative_int(
            self.total_predicted_voxels,
            field_name="total_predicted_voxels",
        )
        _require_nonnegative_int(
            self.total_intersection_voxels,
            field_name="total_intersection_voxels",
        )
        _require_nonnegative_int(self.total_union_voxels, field_name="total_union_voxels")
        _require_unit_interval(self.pooled_dice, field_name="pooled_dice")
        _require_unit_interval(self.pooled_iou, field_name="pooled_iou")
        _require_unit_interval(self.macro_dice, field_name="macro_dice")
        _require_unit_interval(self.macro_iou, field_name="macro_iou")
        _require_unit_interval(
            self.macro_normalized_surface_dice,
            field_name="macro_normalized_surface_dice",
        )
        _require_optional_nonnegative_float(self.macro_hd95_mm, field_name="macro_hd95_mm")
        _require_nonnegative_int(
            self.defined_hd95_case_count,
            field_name="defined_hd95_case_count",
        )
        _require_nonnegative_int(
            self.total_ground_truth_lesions,
            field_name="total_ground_truth_lesions",
        )
        _require_nonnegative_int(
            self.total_predicted_lesions,
            field_name="total_predicted_lesions",
        )
        _require_nonnegative_int(
            self.total_true_positive_lesions,
            field_name="total_true_positive_lesions",
        )
        _require_nonnegative_int(
            self.total_false_positive_lesions,
            field_name="total_false_positive_lesions",
        )
        _require_nonnegative_int(
            self.total_false_negative_lesions,
            field_name="total_false_negative_lesions",
        )
        _require_optional_unit_interval(
            self.micro_lesion_recall,
            field_name="micro_lesion_recall",
        )
        _require_optional_unit_interval(
            self.micro_lesion_precision,
            field_name="micro_lesion_precision",
        )
        _require_unit_interval(self.micro_lesion_f1, field_name="micro_lesion_f1")
        _require_optional_unit_interval(
            self.macro_lesion_recall,
            field_name="macro_lesion_recall",
        )
        _require_optional_unit_interval(
            self.macro_lesion_precision,
            field_name="macro_lesion_precision",
        )
        _require_unit_interval(self.macro_lesion_f1, field_name="macro_lesion_f1")
        _require_nonnegative_int(
            self.defined_lesion_recall_case_count,
            field_name="defined_lesion_recall_case_count",
        )
        _require_nonnegative_int(
            self.defined_lesion_precision_case_count,
            field_name="defined_lesion_precision_case_count",
        )
        _require_nonnegative_float(
            self.mean_false_positive_lesions_per_scan,
            field_name="mean_false_positive_lesions_per_scan",
        )
        _require_finite_float(
            self.mean_signed_volume_error_ml,
            field_name="mean_signed_volume_error_ml",
        )
        _require_nonnegative_float(
            self.mean_absolute_volume_error_ml,
            field_name="mean_absolute_volume_error_ml",
        )
        _require_optional_finite_float(
            self.macro_relative_volume_error,
            field_name="macro_relative_volume_error",
        )
        _require_nonnegative_int(
            self.defined_relative_volume_error_case_count,
            field_name="defined_relative_volume_error_case_count",
        )
        _require_no_absolute_paths(_aggregate_payload(self))


@dataclass(frozen=True, slots=True)
class BaselineMetricReport:
    """Immutable versioned report for Phase 3 baseline metrics."""

    artifact_hash: str
    contract_version: str
    baseline_family: str
    run_identifier: str
    dataset_manifest_sha256: str
    development_split_sha256: str
    prediction_manifest_sha256: str
    metric_config_sha256: str
    nsd_tolerance_mm: float
    lesion_connectivity: int
    case_records: tuple[BaselineCaseMetrics, ...] = field(default_factory=tuple)
    aggregate: BaselineMetricAggregate | None = None

    def __post_init__(self) -> None:
        _require_contract_version(self.contract_version, BASELINE_METRIC_REPORT_VERSION)
        _require_sha256(self.artifact_hash, field_name="artifact_hash")
        if self.baseline_family not in SUPPORTED_BASELINE_METRIC_FAMILIES:
            raise BaselineMetricValidationError(
                f"Unsupported baseline family: {self.baseline_family!r}."
            )
        _require_case_identifier(self.run_identifier)
        _require_sha256(self.dataset_manifest_sha256, field_name="dataset_manifest_sha256")
        _require_sha256(self.development_split_sha256, field_name="development_split_sha256")
        _require_sha256(self.prediction_manifest_sha256, field_name="prediction_manifest_sha256")
        _require_sha256(self.metric_config_sha256, field_name="metric_config_sha256")
        tolerance = _normalize_nonnegative_float(
            self.nsd_tolerance_mm,
            field_name="nsd_tolerance_mm",
        )
        object.__setattr__(self, "nsd_tolerance_mm", tolerance)
        if self.lesion_connectivity != BASELINE_LESION_CONNECTIVITY:
            raise BaselineMetricValidationError(
                f"lesion_connectivity must be {BASELINE_LESION_CONNECTIVITY}."
            )
        _require_case_tuple(self.case_records)
        sorted_cases = tuple(sorted(self.case_records, key=lambda item: item.case_identifier))
        if self.case_records != sorted_cases:
            raise BaselineMetricValidationError(
                "case_records must be sorted by ascending anonymous case identifier."
            )
        _require_unique_case_identifiers(self.case_records)
        for case_record in self.case_records:
            if case_record.nsd_tolerance_mm != self.nsd_tolerance_mm:
                raise BaselineMetricValidationError(
                    "All case_records must use the report nsd_tolerance_mm."
                )
        expected_aggregate = _build_aggregate(self.case_records)
        if self.aggregate is None:
            object.__setattr__(self, "aggregate", expected_aggregate)
        elif self.aggregate != expected_aggregate:
            raise BaselineMetricValidationError(
                "aggregate must equal the deterministic aggregate over case_records."
            )
        _require_no_absolute_paths(_metric_report_payload(self))
        expected_hash = sha256_json(_metric_report_payload_without_hash(self))
        if self.artifact_hash != expected_hash:
            raise BaselineMetricHashError(
                "artifact_hash does not match the deterministic metric-report content."
            )


def compute_baseline_case_metrics(
    *,
    case_identifier: str,
    ground_truth_mask: np.ndarray[Any, Any],
    prediction_mask: np.ndarray[Any, Any],
    voxel_spacing_mm: Sequence[float],
    nsd_tolerance_mm: float,
) -> BaselineCaseMetrics:
    """Compute deterministic binary-tumor metrics for one 3D case."""

    _require_case_identifier(case_identifier)
    spacing = _normalize_spacing(voxel_spacing_mm)
    tolerance = _normalize_nonnegative_float(nsd_tolerance_mm, field_name="nsd_tolerance_mm")
    ground_truth = _validate_mask_array(ground_truth_mask, field_name="ground_truth_mask")
    prediction = _validate_mask_array(prediction_mask, field_name="prediction_mask")
    if ground_truth.shape != prediction.shape:
        raise BaselineMetricValidationError(
            "ground_truth_mask and prediction_mask must have exactly matching shapes."
        )

    gt_voxels = int(np.count_nonzero(ground_truth))
    pred_voxels = int(np.count_nonzero(prediction))
    intersection_voxels = int(np.count_nonzero(ground_truth & prediction))
    union_voxels = int(np.count_nonzero(ground_truth | prediction))

    both_empty = gt_voxels == 0 and pred_voxels == 0
    gt_empty = gt_voxels == 0
    pred_empty = pred_voxels == 0

    tumor_dice = _safe_dice(gt_voxels, pred_voxels, intersection_voxels)
    tumor_iou = _safe_iou(union_voxels, intersection_voxels)
    hd95_mm: float | None
    normalized_surface_dice: float
    if both_empty:
        hd95_mm = 0.0
        normalized_surface_dice = 1.0
    elif gt_empty or pred_empty:
        hd95_mm = None
        normalized_surface_dice = 0.0
    else:
        hd95_mm, normalized_surface_dice = _compute_surface_metrics(
            ground_truth=ground_truth,
            prediction=prediction,
            voxel_spacing_mm=spacing,
            nsd_tolerance_mm=tolerance,
        )

    gt_labels, gt_lesion_count = _label_lesions(ground_truth)
    pred_labels, pred_lesion_count = _label_lesions(prediction)
    true_positive_lesion_count = _count_true_positive_lesions(
        gt_labels=gt_labels,
        gt_lesion_count=gt_lesion_count,
        pred_labels=pred_labels,
        pred_lesion_count=pred_lesion_count,
    )
    false_positive_lesion_count = pred_lesion_count - true_positive_lesion_count
    false_negative_lesion_count = gt_lesion_count - true_positive_lesion_count
    lesion_recall = _safe_optional_ratio(
        numerator=true_positive_lesion_count,
        denominator=gt_lesion_count,
    )
    lesion_precision = _safe_optional_ratio(
        numerator=true_positive_lesion_count,
        denominator=pred_lesion_count,
    )
    lesion_f1 = _safe_lesion_f1(
        ground_truth_lesion_count=gt_lesion_count,
        predicted_lesion_count=pred_lesion_count,
        recall=lesion_recall,
        precision=lesion_precision,
    )

    voxel_volume_ml = np.float64(spacing[0]) * np.float64(spacing[1]) * np.float64(spacing[2])
    voxel_volume_ml /= np.float64(1000.0)
    ground_truth_volume_ml = float(np.float64(gt_voxels) * voxel_volume_ml)
    predicted_volume_ml = float(np.float64(pred_voxels) * voxel_volume_ml)
    signed_volume_error_ml = float(
        np.float64(predicted_volume_ml) - np.float64(ground_truth_volume_ml)
    )
    absolute_volume_error_ml = float(abs(np.float64(signed_volume_error_ml)))
    relative_volume_error = (
        None
        if gt_voxels == 0
        else float(np.float64(signed_volume_error_ml) / np.float64(ground_truth_volume_ml))
    )

    payload_without_hash = {
        "absolute_volume_error_ml": absolute_volume_error_ml,
        "case_identifier": case_identifier,
        "contract_version": BASELINE_CASE_METRICS_VERSION,
        "false_negative_lesion_count": false_negative_lesion_count,
        "false_positive_lesion_count": false_positive_lesion_count,
        "false_positive_lesions_per_scan": false_positive_lesion_count,
        "ground_truth_foreground_voxels": gt_voxels,
        "ground_truth_lesion_count": gt_lesion_count,
        "ground_truth_volume_ml": ground_truth_volume_ml,
        "hd95_mm": hd95_mm,
        "intersection_voxels": intersection_voxels,
        "lesion_f1": lesion_f1,
        "lesion_precision": lesion_precision,
        "lesion_recall": lesion_recall,
        "normalized_surface_dice": normalized_surface_dice,
        "nsd_tolerance_mm": tolerance,
        "predicted_foreground_voxels": pred_voxels,
        "predicted_lesion_count": pred_lesion_count,
        "predicted_volume_ml": predicted_volume_ml,
        "relative_volume_error": relative_volume_error,
        "signed_volume_error_ml": signed_volume_error_ml,
        "true_positive_lesion_count": true_positive_lesion_count,
        "tumor_dice": tumor_dice,
        "tumor_iou": tumor_iou,
        "union_voxels": union_voxels,
        "voxel_spacing_mm": list(spacing),
    }
    return BaselineCaseMetrics(
        artifact_hash=sha256_json(payload_without_hash),
        contract_version=BASELINE_CASE_METRICS_VERSION,
        case_identifier=case_identifier,
        voxel_spacing_mm=spacing,
        nsd_tolerance_mm=tolerance,
        ground_truth_foreground_voxels=gt_voxels,
        predicted_foreground_voxels=pred_voxels,
        intersection_voxels=intersection_voxels,
        union_voxels=union_voxels,
        tumor_dice=tumor_dice,
        tumor_iou=tumor_iou,
        hd95_mm=hd95_mm,
        normalized_surface_dice=normalized_surface_dice,
        ground_truth_lesion_count=gt_lesion_count,
        predicted_lesion_count=pred_lesion_count,
        true_positive_lesion_count=true_positive_lesion_count,
        false_positive_lesion_count=false_positive_lesion_count,
        false_negative_lesion_count=false_negative_lesion_count,
        lesion_recall=lesion_recall,
        lesion_precision=lesion_precision,
        lesion_f1=lesion_f1,
        false_positive_lesions_per_scan=false_positive_lesion_count,
        ground_truth_volume_ml=ground_truth_volume_ml,
        predicted_volume_ml=predicted_volume_ml,
        signed_volume_error_ml=signed_volume_error_ml,
        absolute_volume_error_ml=absolute_volume_error_ml,
        relative_volume_error=relative_volume_error,
    )


def build_baseline_metric_report(
    *,
    baseline_family: str,
    run_identifier: str,
    dataset_manifest_sha256: str,
    development_split_sha256: str,
    prediction_manifest_sha256: str,
    metric_config_sha256: str,
    nsd_tolerance_mm: float,
    case_records: Sequence[BaselineCaseMetrics],
) -> BaselineMetricReport:
    """Build the deterministic Phase 3 metric report."""

    cases = tuple(sorted(case_records, key=lambda item: item.case_identifier))
    aggregate = _build_aggregate(cases)
    payload_without_hash = {
        "aggregate": _aggregate_payload(aggregate),
        "baseline_family": baseline_family,
        "case_records": [_case_metrics_payload(case_record) for case_record in cases],
        "contract_version": BASELINE_METRIC_REPORT_VERSION,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "development_split_sha256": development_split_sha256,
        "lesion_connectivity": BASELINE_LESION_CONNECTIVITY,
        "metric_config_sha256": metric_config_sha256,
        "nsd_tolerance_mm": nsd_tolerance_mm,
        "prediction_manifest_sha256": prediction_manifest_sha256,
        "run_identifier": run_identifier,
    }
    return BaselineMetricReport(
        artifact_hash=sha256_json(payload_without_hash),
        contract_version=BASELINE_METRIC_REPORT_VERSION,
        baseline_family=baseline_family,
        run_identifier=run_identifier,
        dataset_manifest_sha256=dataset_manifest_sha256,
        development_split_sha256=development_split_sha256,
        prediction_manifest_sha256=prediction_manifest_sha256,
        metric_config_sha256=metric_config_sha256,
        nsd_tolerance_mm=nsd_tolerance_mm,
        lesion_connectivity=BASELINE_LESION_CONNECTIVITY,
        case_records=cases,
        aggregate=aggregate,
    )


def baseline_case_metrics_to_json(case_metrics: BaselineCaseMetrics) -> bytes:
    """Serialize one case-metrics artifact deterministically."""

    return canonical_json_bytes(_case_metrics_payload(case_metrics)) + b"\n"


def baseline_case_metrics_from_json(payload: bytes | str) -> BaselineCaseMetrics:
    """Parse and validate one case-metrics artifact."""

    parsed = _parse_json_object(payload, context="baseline case metrics")
    _require_exact_fields(parsed, expected=_CASE_FIELDS, context="baseline case metrics")
    contract_version = _expect_string(parsed["contract_version"], field_name="contract_version")
    _require_contract_version(contract_version, BASELINE_CASE_METRICS_VERSION)
    case_metrics = BaselineCaseMetrics(
        artifact_hash=_expect_string(parsed["artifact_hash"], field_name="artifact_hash"),
        contract_version=contract_version,
        case_identifier=_expect_string(parsed["case_identifier"], field_name="case_identifier"),
        voxel_spacing_mm=_expect_float_tuple3(
            parsed["voxel_spacing_mm"],
            field_name="voxel_spacing_mm",
        ),
        nsd_tolerance_mm=_expect_number(parsed["nsd_tolerance_mm"], field_name="nsd_tolerance_mm"),
        ground_truth_foreground_voxels=_expect_int(
            parsed["ground_truth_foreground_voxels"],
            field_name="ground_truth_foreground_voxels",
        ),
        predicted_foreground_voxels=_expect_int(
            parsed["predicted_foreground_voxels"],
            field_name="predicted_foreground_voxels",
        ),
        intersection_voxels=_expect_int(
            parsed["intersection_voxels"],
            field_name="intersection_voxels",
        ),
        union_voxels=_expect_int(parsed["union_voxels"], field_name="union_voxels"),
        tumor_dice=_expect_number(parsed["tumor_dice"], field_name="tumor_dice"),
        tumor_iou=_expect_number(parsed["tumor_iou"], field_name="tumor_iou"),
        hd95_mm=_expect_optional_number(parsed["hd95_mm"], field_name="hd95_mm"),
        normalized_surface_dice=_expect_number(
            parsed["normalized_surface_dice"],
            field_name="normalized_surface_dice",
        ),
        ground_truth_lesion_count=_expect_int(
            parsed["ground_truth_lesion_count"],
            field_name="ground_truth_lesion_count",
        ),
        predicted_lesion_count=_expect_int(
            parsed["predicted_lesion_count"],
            field_name="predicted_lesion_count",
        ),
        true_positive_lesion_count=_expect_int(
            parsed["true_positive_lesion_count"],
            field_name="true_positive_lesion_count",
        ),
        false_positive_lesion_count=_expect_int(
            parsed["false_positive_lesion_count"],
            field_name="false_positive_lesion_count",
        ),
        false_negative_lesion_count=_expect_int(
            parsed["false_negative_lesion_count"],
            field_name="false_negative_lesion_count",
        ),
        lesion_recall=_expect_optional_number(
            parsed["lesion_recall"],
            field_name="lesion_recall",
        ),
        lesion_precision=_expect_optional_number(
            parsed["lesion_precision"],
            field_name="lesion_precision",
        ),
        lesion_f1=_expect_number(parsed["lesion_f1"], field_name="lesion_f1"),
        false_positive_lesions_per_scan=_expect_int(
            parsed["false_positive_lesions_per_scan"],
            field_name="false_positive_lesions_per_scan",
        ),
        ground_truth_volume_ml=_expect_number(
            parsed["ground_truth_volume_ml"],
            field_name="ground_truth_volume_ml",
        ),
        predicted_volume_ml=_expect_number(
            parsed["predicted_volume_ml"],
            field_name="predicted_volume_ml",
        ),
        signed_volume_error_ml=_expect_number(
            parsed["signed_volume_error_ml"],
            field_name="signed_volume_error_ml",
        ),
        absolute_volume_error_ml=_expect_number(
            parsed["absolute_volume_error_ml"],
            field_name="absolute_volume_error_ml",
        ),
        relative_volume_error=_expect_optional_number(
            parsed["relative_volume_error"],
            field_name="relative_volume_error",
        ),
    )
    expected_hash = sha256_json(_case_metrics_payload_without_hash(case_metrics))
    if case_metrics.artifact_hash != expected_hash:
        raise BaselineMetricHashError(
            "artifact_hash does not match the parsed case-metrics content."
        )
    return case_metrics


def baseline_metric_report_to_json(report: BaselineMetricReport) -> bytes:
    """Serialize one baseline metric report deterministically."""

    return canonical_json_bytes(_metric_report_payload(report)) + b"\n"


def baseline_metric_report_from_json(payload: bytes | str) -> BaselineMetricReport:
    """Parse and validate one baseline metric report."""

    parsed = _parse_json_object(payload, context="baseline metric report")
    _require_exact_fields(parsed, expected=_REPORT_FIELDS, context="baseline metric report")
    contract_version = _expect_string(parsed["contract_version"], field_name="contract_version")
    _require_contract_version(contract_version, BASELINE_METRIC_REPORT_VERSION)
    case_payloads = parsed["case_records"]
    if not isinstance(case_payloads, list):
        raise BaselineMetricSerializationError("case_records must be a JSON array.")
    case_records = tuple(
        baseline_case_metrics_from_json(json.dumps(item, separators=(",", ":")))
        for item in case_payloads
    )
    aggregate = _baseline_metric_aggregate_from_json_value(parsed["aggregate"])
    report = BaselineMetricReport(
        artifact_hash=_expect_string(parsed["artifact_hash"], field_name="artifact_hash"),
        contract_version=contract_version,
        baseline_family=_expect_string(parsed["baseline_family"], field_name="baseline_family"),
        run_identifier=_expect_string(parsed["run_identifier"], field_name="run_identifier"),
        dataset_manifest_sha256=_expect_string(
            parsed["dataset_manifest_sha256"],
            field_name="dataset_manifest_sha256",
        ),
        development_split_sha256=_expect_string(
            parsed["development_split_sha256"],
            field_name="development_split_sha256",
        ),
        prediction_manifest_sha256=_expect_string(
            parsed["prediction_manifest_sha256"],
            field_name="prediction_manifest_sha256",
        ),
        metric_config_sha256=_expect_string(
            parsed["metric_config_sha256"],
            field_name="metric_config_sha256",
        ),
        nsd_tolerance_mm=_expect_number(parsed["nsd_tolerance_mm"], field_name="nsd_tolerance_mm"),
        lesion_connectivity=_expect_int(
            parsed["lesion_connectivity"],
            field_name="lesion_connectivity",
        ),
        case_records=case_records,
        aggregate=aggregate,
    )
    expected_hash = sha256_json(_metric_report_payload_without_hash(report))
    if report.artifact_hash != expected_hash:
        raise BaselineMetricHashError(
            "artifact_hash does not match the parsed metric-report content."
        )
    return report


def _validate_mask_array(mask: np.ndarray[Any, Any], *, field_name: str) -> np.ndarray[Any, Any]:
    array = np.asarray(mask)
    if array.dtype == object:
        raise BaselineMetricValidationError(f"{field_name} must not be an object array.")
    if array.ndim != 3:
        raise BaselineMetricValidationError(f"{field_name} must be exactly three-dimensional.")
    if any(dimension == 0 for dimension in array.shape):
        raise BaselineMetricValidationError(f"{field_name} dimensions must all be nonzero.")
    if np.issubdtype(array.dtype, np.bool_):
        return array.astype(np.bool_, copy=False)
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.complexfloating):
        raise BaselineMetricValidationError(
            f"{field_name} must contain boolean values or numeric 0/1 values."
        )
    if not np.isfinite(array).all():
        raise BaselineMetricValidationError(f"{field_name} must not contain NaN or Inf.")
    invalid_mask = (array != 0) & (array != 1)
    if np.any(invalid_mask):
        raise BaselineMetricValidationError(
            f"{field_name} must contain only boolean values or numeric 0/1 values."
        )
    if np.any(array < 0):
        raise BaselineMetricValidationError(f"{field_name} must not contain negative values.")
    return array.astype(np.bool_, copy=False)


def _normalize_spacing(voxel_spacing_mm: Sequence[float]) -> tuple[float, float, float]:
    if len(voxel_spacing_mm) != 3:
        raise BaselineMetricValidationError(
            "voxel_spacing_mm must contain exactly three finite positive values."
        )
    normalized: list[float] = []
    for index, value in enumerate(voxel_spacing_mm):
        if isinstance(value, bool):
            raise BaselineMetricValidationError(
                "voxel_spacing_mm must contain exactly three finite positive values."
            )
        try:
            scalar = float(value)
        except (TypeError, ValueError) as error:
            raise BaselineMetricValidationError(
                "voxel_spacing_mm must contain exactly three finite positive values."
            ) from error
        if not math.isfinite(scalar) or scalar <= 0.0:
            raise BaselineMetricValidationError(
                "voxel_spacing_mm must contain exactly three finite positive values."
            )
        normalized.append(scalar)
        if index > 2:
            break
    return (normalized[0], normalized[1], normalized[2])


def _normalize_nonnegative_float(value: object, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise BaselineMetricValidationError(f"{field_name} must be a finite nonnegative float.")
    try:
        scalar = float(cast(float, value))
    except (TypeError, ValueError) as error:
        raise BaselineMetricValidationError(
            f"{field_name} must be a finite nonnegative float."
        ) from error
    if not math.isfinite(scalar) or scalar < 0.0:
        raise BaselineMetricValidationError(f"{field_name} must be a finite nonnegative float.")
    return scalar


def _compute_surface_metrics(
    *,
    ground_truth: np.ndarray[Any, Any],
    prediction: np.ndarray[Any, Any],
    voxel_spacing_mm: tuple[float, float, float],
    nsd_tolerance_mm: float,
) -> tuple[float, float]:
    surface_structure = ndimage.generate_binary_structure(3, 1)
    gt_surface = _extract_surface(ground_truth, structure=surface_structure)
    pred_surface = _extract_surface(prediction, structure=surface_structure)

    pred_distance_transform = ndimage.distance_transform_edt(
        ~pred_surface,
        sampling=np.asarray(voxel_spacing_mm, dtype=np.float64),
    )
    gt_distance_transform = ndimage.distance_transform_edt(
        ~gt_surface,
        sampling=np.asarray(voxel_spacing_mm, dtype=np.float64),
    )
    gt_to_pred = np.asarray(pred_distance_transform[gt_surface], dtype=np.float64)
    pred_to_gt = np.asarray(gt_distance_transform[pred_surface], dtype=np.float64)
    bidirectional_distances = np.concatenate((gt_to_pred, pred_to_gt), dtype=np.float64)
    hd95_mm = float(np.percentile(bidirectional_distances, 95))
    nsd = float(
        np.count_nonzero(bidirectional_distances <= np.float64(nsd_tolerance_mm))
        / bidirectional_distances.size
    )
    _reject_nonfinite_result(hd95_mm, field_name="hd95_mm")
    _reject_nonfinite_result(nsd, field_name="normalized_surface_dice")
    return hd95_mm, nsd


def _extract_surface(
    mask: np.ndarray[Any, Any],
    *,
    structure: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    eroded = ndimage.binary_erosion(mask, structure=structure, border_value=0)
    return cast(np.ndarray[Any, Any], mask & ~eroded)


def _label_lesions(mask: np.ndarray[Any, Any]) -> tuple[np.ndarray[Any, Any], int]:
    structure = ndimage.generate_binary_structure(3, 3)
    labeled, count = ndimage.label(mask, structure=structure)
    return np.asarray(labeled, dtype=np.int32), int(count)


def _count_true_positive_lesions(
    *,
    gt_labels: np.ndarray[Any, Any],
    gt_lesion_count: int,
    pred_labels: np.ndarray[Any, Any],
    pred_lesion_count: int,
) -> int:
    if gt_lesion_count == 0 or pred_lesion_count == 0:
        return 0
    overlaps = _overlap_matrix(
        gt_labels=gt_labels,
        gt_lesion_count=gt_lesion_count,
        pred_labels=pred_labels,
        pred_lesion_count=pred_lesion_count,
    )
    return _deterministic_maximum_overlap_match_count(overlaps)


def _overlap_matrix(
    *,
    gt_labels: np.ndarray[Any, Any],
    gt_lesion_count: int,
    pred_labels: np.ndarray[Any, Any],
    pred_lesion_count: int,
) -> np.ndarray[Any, Any]:
    overlaps = np.zeros((gt_lesion_count, pred_lesion_count), dtype=np.int64)
    valid = (gt_labels > 0) & (pred_labels > 0)
    if not np.any(valid):
        return overlaps
    gt_nonzero = gt_labels[valid].astype(np.int64, copy=False) - 1
    pred_nonzero = pred_labels[valid].astype(np.int64, copy=False) - 1
    encoded = gt_nonzero * np.int64(pred_lesion_count) + pred_nonzero
    counts = np.bincount(encoded, minlength=gt_lesion_count * pred_lesion_count)
    return counts.reshape((gt_lesion_count, pred_lesion_count))


def _deterministic_maximum_overlap_match_count(overlaps: np.ndarray[Any, Any]) -> int:
    gt_count, pred_count = overlaps.shape
    size = max(gt_count, pred_count)
    if size == 0:
        return 0
    max_overlap = int(np.max(overlaps)) if overlaps.size > 0 else 0
    big = size * size + 1
    cost = np.full((size, size), max_overlap * big, dtype=np.int64)
    for row_index in range(gt_count):
        for col_index in range(pred_count):
            overlap = int(overlaps[row_index, col_index])
            tie_break = row_index * size + col_index
            cost[row_index, col_index] = (max_overlap - overlap) * big + tie_break
    row_indices, col_indices = linear_sum_assignment(cost)
    match_count = 0
    for row_index, col_index in zip(row_indices, col_indices, strict=True):
        if row_index >= gt_count or col_index >= pred_count:
            continue
        if overlaps[row_index, col_index] > 0:
            match_count += 1
    return match_count


def _safe_dice(
    ground_truth_voxels: int,
    predicted_voxels: int,
    intersection_voxels: int,
) -> float:
    denominator = ground_truth_voxels + predicted_voxels
    if denominator == 0:
        return 1.0
    return float((2.0 * np.float64(intersection_voxels)) / np.float64(denominator))


def _safe_iou(union_voxels: int, intersection_voxels: int) -> float:
    if union_voxels == 0:
        return 1.0
    return float(np.float64(intersection_voxels) / np.float64(union_voxels))


def _safe_optional_ratio(*, numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return float(np.float64(numerator) / np.float64(denominator))


def _safe_lesion_f1(
    *,
    ground_truth_lesion_count: int,
    predicted_lesion_count: int,
    recall: float | None,
    precision: float | None,
) -> float:
    if ground_truth_lesion_count == 0 and predicted_lesion_count == 0:
        return 1.0
    if ground_truth_lesion_count == 0 or predicted_lesion_count == 0:
        return 0.0
    if recall is None or precision is None:
        raise BaselineMetricValidationError(
            "Defined lesion counts require defined recall and precision."
        )
    if recall + precision == 0.0:
        return 0.0
    return float(
        (2.0 * np.float64(recall) * np.float64(precision)) / np.float64(recall + precision)
    )


def _build_aggregate(case_records: Sequence[BaselineCaseMetrics]) -> BaselineMetricAggregate:
    case_count = len(case_records)
    total_ground_truth_voxels = sum(case.ground_truth_foreground_voxels for case in case_records)
    total_predicted_voxels = sum(case.predicted_foreground_voxels for case in case_records)
    total_intersection_voxels = sum(case.intersection_voxels for case in case_records)
    total_union_voxels = sum(case.union_voxels for case in case_records)
    pooled_dice = _safe_dice(
        total_ground_truth_voxels,
        total_predicted_voxels,
        total_intersection_voxels,
    )
    pooled_iou = _safe_iou(total_union_voxels, total_intersection_voxels)
    macro_dice = _mean_defined([case.tumor_dice for case in case_records], default=1.0)
    macro_iou = _mean_defined([case.tumor_iou for case in case_records], default=1.0)
    macro_nsd = _mean_defined(
        [case.normalized_surface_dice for case in case_records],
        default=1.0,
    )
    hd95_values = [case.hd95_mm for case in case_records if case.hd95_mm is not None]
    macro_hd95 = _mean_optional(hd95_values)
    total_ground_truth_lesions = sum(case.ground_truth_lesion_count for case in case_records)
    total_predicted_lesions = sum(case.predicted_lesion_count for case in case_records)
    total_true_positive_lesions = sum(case.true_positive_lesion_count for case in case_records)
    total_false_positive_lesions = sum(case.false_positive_lesion_count for case in case_records)
    total_false_negative_lesions = sum(case.false_negative_lesion_count for case in case_records)
    micro_lesion_recall = _safe_optional_ratio(
        numerator=total_true_positive_lesions,
        denominator=total_ground_truth_lesions,
    )
    micro_lesion_precision = _safe_optional_ratio(
        numerator=total_true_positive_lesions,
        denominator=total_predicted_lesions,
    )
    micro_lesion_f1 = _safe_lesion_f1(
        ground_truth_lesion_count=total_ground_truth_lesions,
        predicted_lesion_count=total_predicted_lesions,
        recall=micro_lesion_recall,
        precision=micro_lesion_precision,
    )
    recall_values = [case.lesion_recall for case in case_records if case.lesion_recall is not None]
    precision_values = [
        case.lesion_precision for case in case_records if case.lesion_precision is not None
    ]
    macro_lesion_recall = _mean_optional(recall_values)
    macro_lesion_precision = _mean_optional(precision_values)
    macro_lesion_f1 = _mean_defined([case.lesion_f1 for case in case_records], default=1.0)
    mean_false_positive_lesions_per_scan = _mean_defined(
        [float(case.false_positive_lesions_per_scan) for case in case_records],
        default=0.0,
    )
    mean_signed_volume_error_ml = _mean_defined(
        [case.signed_volume_error_ml for case in case_records],
        default=0.0,
    )
    mean_absolute_volume_error_ml = _mean_defined(
        [case.absolute_volume_error_ml for case in case_records],
        default=0.0,
    )
    relative_volume_error_values = [
        case.relative_volume_error
        for case in case_records
        if case.relative_volume_error is not None
    ]
    macro_relative_volume_error = _mean_optional(relative_volume_error_values)
    return BaselineMetricAggregate(
        case_count=case_count,
        total_ground_truth_voxels=total_ground_truth_voxels,
        total_predicted_voxels=total_predicted_voxels,
        total_intersection_voxels=total_intersection_voxels,
        total_union_voxels=total_union_voxels,
        pooled_dice=pooled_dice,
        pooled_iou=pooled_iou,
        macro_dice=macro_dice,
        macro_iou=macro_iou,
        macro_normalized_surface_dice=macro_nsd,
        macro_hd95_mm=macro_hd95,
        defined_hd95_case_count=len(hd95_values),
        total_ground_truth_lesions=total_ground_truth_lesions,
        total_predicted_lesions=total_predicted_lesions,
        total_true_positive_lesions=total_true_positive_lesions,
        total_false_positive_lesions=total_false_positive_lesions,
        total_false_negative_lesions=total_false_negative_lesions,
        micro_lesion_recall=micro_lesion_recall,
        micro_lesion_precision=micro_lesion_precision,
        micro_lesion_f1=micro_lesion_f1,
        macro_lesion_recall=macro_lesion_recall,
        macro_lesion_precision=macro_lesion_precision,
        macro_lesion_f1=macro_lesion_f1,
        defined_lesion_recall_case_count=len(recall_values),
        defined_lesion_precision_case_count=len(precision_values),
        mean_false_positive_lesions_per_scan=mean_false_positive_lesions_per_scan,
        mean_signed_volume_error_ml=mean_signed_volume_error_ml,
        mean_absolute_volume_error_ml=mean_absolute_volume_error_ml,
        macro_relative_volume_error=macro_relative_volume_error,
        defined_relative_volume_error_case_count=len(relative_volume_error_values),
    )


def _mean_defined(values: Sequence[float], *, default: float) -> float:
    if not values:
        return default
    result = float(np.sum(np.asarray(values, dtype=np.float64)) / np.float64(len(values)))
    _reject_nonfinite_result(result, field_name="mean")
    return result


def _mean_optional(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return _mean_defined(values, default=0.0)


def _case_metrics_payload(case_metrics: BaselineCaseMetrics) -> dict[str, Any]:
    return {
        "absolute_volume_error_ml": case_metrics.absolute_volume_error_ml,
        "artifact_hash": case_metrics.artifact_hash,
        "case_identifier": case_metrics.case_identifier,
        "contract_version": case_metrics.contract_version,
        "false_negative_lesion_count": case_metrics.false_negative_lesion_count,
        "false_positive_lesion_count": case_metrics.false_positive_lesion_count,
        "false_positive_lesions_per_scan": case_metrics.false_positive_lesions_per_scan,
        "ground_truth_foreground_voxels": case_metrics.ground_truth_foreground_voxels,
        "ground_truth_lesion_count": case_metrics.ground_truth_lesion_count,
        "ground_truth_volume_ml": case_metrics.ground_truth_volume_ml,
        "hd95_mm": case_metrics.hd95_mm,
        "intersection_voxels": case_metrics.intersection_voxels,
        "lesion_f1": case_metrics.lesion_f1,
        "lesion_precision": case_metrics.lesion_precision,
        "lesion_recall": case_metrics.lesion_recall,
        "normalized_surface_dice": case_metrics.normalized_surface_dice,
        "nsd_tolerance_mm": case_metrics.nsd_tolerance_mm,
        "predicted_foreground_voxels": case_metrics.predicted_foreground_voxels,
        "predicted_lesion_count": case_metrics.predicted_lesion_count,
        "predicted_volume_ml": case_metrics.predicted_volume_ml,
        "relative_volume_error": case_metrics.relative_volume_error,
        "signed_volume_error_ml": case_metrics.signed_volume_error_ml,
        "true_positive_lesion_count": case_metrics.true_positive_lesion_count,
        "tumor_dice": case_metrics.tumor_dice,
        "tumor_iou": case_metrics.tumor_iou,
        "union_voxels": case_metrics.union_voxels,
        "voxel_spacing_mm": list(case_metrics.voxel_spacing_mm),
    }


def _case_metrics_payload_without_hash(case_metrics: BaselineCaseMetrics) -> dict[str, Any]:
    payload = _case_metrics_payload(case_metrics)
    del payload["artifact_hash"]
    return payload


def _aggregate_payload(aggregate: BaselineMetricAggregate) -> dict[str, Any]:
    return {
        "case_count": aggregate.case_count,
        "defined_hd95_case_count": aggregate.defined_hd95_case_count,
        "defined_lesion_precision_case_count": aggregate.defined_lesion_precision_case_count,
        "defined_lesion_recall_case_count": aggregate.defined_lesion_recall_case_count,
        "defined_relative_volume_error_case_count": (
            aggregate.defined_relative_volume_error_case_count
        ),
        "macro_dice": aggregate.macro_dice,
        "macro_hd95_mm": aggregate.macro_hd95_mm,
        "macro_iou": aggregate.macro_iou,
        "macro_lesion_f1": aggregate.macro_lesion_f1,
        "macro_lesion_precision": aggregate.macro_lesion_precision,
        "macro_lesion_recall": aggregate.macro_lesion_recall,
        "macro_normalized_surface_dice": aggregate.macro_normalized_surface_dice,
        "macro_relative_volume_error": aggregate.macro_relative_volume_error,
        "mean_absolute_volume_error_ml": aggregate.mean_absolute_volume_error_ml,
        "mean_false_positive_lesions_per_scan": aggregate.mean_false_positive_lesions_per_scan,
        "mean_signed_volume_error_ml": aggregate.mean_signed_volume_error_ml,
        "micro_lesion_f1": aggregate.micro_lesion_f1,
        "micro_lesion_precision": aggregate.micro_lesion_precision,
        "micro_lesion_recall": aggregate.micro_lesion_recall,
        "pooled_dice": aggregate.pooled_dice,
        "pooled_iou": aggregate.pooled_iou,
        "total_false_negative_lesions": aggregate.total_false_negative_lesions,
        "total_false_positive_lesions": aggregate.total_false_positive_lesions,
        "total_ground_truth_lesions": aggregate.total_ground_truth_lesions,
        "total_ground_truth_voxels": aggregate.total_ground_truth_voxels,
        "total_intersection_voxels": aggregate.total_intersection_voxels,
        "total_predicted_lesions": aggregate.total_predicted_lesions,
        "total_predicted_voxels": aggregate.total_predicted_voxels,
        "total_true_positive_lesions": aggregate.total_true_positive_lesions,
        "total_union_voxels": aggregate.total_union_voxels,
    }


def _metric_report_payload(report: BaselineMetricReport) -> dict[str, Any]:
    if report.aggregate is None:
        raise BaselineMetricValidationError("aggregate must be present before serialization.")
    return {
        "aggregate": _aggregate_payload(report.aggregate),
        "artifact_hash": report.artifact_hash,
        "baseline_family": report.baseline_family,
        "case_records": [_case_metrics_payload(case_record) for case_record in report.case_records],
        "contract_version": report.contract_version,
        "dataset_manifest_sha256": report.dataset_manifest_sha256,
        "development_split_sha256": report.development_split_sha256,
        "lesion_connectivity": report.lesion_connectivity,
        "metric_config_sha256": report.metric_config_sha256,
        "nsd_tolerance_mm": report.nsd_tolerance_mm,
        "prediction_manifest_sha256": report.prediction_manifest_sha256,
        "run_identifier": report.run_identifier,
    }


def _metric_report_payload_without_hash(report: BaselineMetricReport) -> dict[str, Any]:
    payload = _metric_report_payload(report)
    del payload["artifact_hash"]
    return payload


def _baseline_metric_aggregate_from_json_value(value: Any) -> BaselineMetricAggregate:
    if not isinstance(value, dict):
        raise BaselineMetricSerializationError("aggregate must be a JSON object.")
    _require_exact_fields(value, expected=_AGGREGATE_FIELDS, context="baseline metric aggregate")
    return BaselineMetricAggregate(
        case_count=_expect_int(value["case_count"], field_name="case_count"),
        total_ground_truth_voxels=_expect_int(
            value["total_ground_truth_voxels"],
            field_name="total_ground_truth_voxels",
        ),
        total_predicted_voxels=_expect_int(
            value["total_predicted_voxels"],
            field_name="total_predicted_voxels",
        ),
        total_intersection_voxels=_expect_int(
            value["total_intersection_voxels"],
            field_name="total_intersection_voxels",
        ),
        total_union_voxels=_expect_int(
            value["total_union_voxels"],
            field_name="total_union_voxels",
        ),
        pooled_dice=_expect_number(value["pooled_dice"], field_name="pooled_dice"),
        pooled_iou=_expect_number(value["pooled_iou"], field_name="pooled_iou"),
        macro_dice=_expect_number(value["macro_dice"], field_name="macro_dice"),
        macro_iou=_expect_number(value["macro_iou"], field_name="macro_iou"),
        macro_normalized_surface_dice=_expect_number(
            value["macro_normalized_surface_dice"],
            field_name="macro_normalized_surface_dice",
        ),
        macro_hd95_mm=_expect_optional_number(value["macro_hd95_mm"], field_name="macro_hd95_mm"),
        defined_hd95_case_count=_expect_int(
            value["defined_hd95_case_count"],
            field_name="defined_hd95_case_count",
        ),
        total_ground_truth_lesions=_expect_int(
            value["total_ground_truth_lesions"],
            field_name="total_ground_truth_lesions",
        ),
        total_predicted_lesions=_expect_int(
            value["total_predicted_lesions"],
            field_name="total_predicted_lesions",
        ),
        total_true_positive_lesions=_expect_int(
            value["total_true_positive_lesions"],
            field_name="total_true_positive_lesions",
        ),
        total_false_positive_lesions=_expect_int(
            value["total_false_positive_lesions"],
            field_name="total_false_positive_lesions",
        ),
        total_false_negative_lesions=_expect_int(
            value["total_false_negative_lesions"],
            field_name="total_false_negative_lesions",
        ),
        micro_lesion_recall=_expect_optional_number(
            value["micro_lesion_recall"],
            field_name="micro_lesion_recall",
        ),
        micro_lesion_precision=_expect_optional_number(
            value["micro_lesion_precision"],
            field_name="micro_lesion_precision",
        ),
        micro_lesion_f1=_expect_number(value["micro_lesion_f1"], field_name="micro_lesion_f1"),
        macro_lesion_recall=_expect_optional_number(
            value["macro_lesion_recall"],
            field_name="macro_lesion_recall",
        ),
        macro_lesion_precision=_expect_optional_number(
            value["macro_lesion_precision"],
            field_name="macro_lesion_precision",
        ),
        macro_lesion_f1=_expect_number(value["macro_lesion_f1"], field_name="macro_lesion_f1"),
        defined_lesion_recall_case_count=_expect_int(
            value["defined_lesion_recall_case_count"],
            field_name="defined_lesion_recall_case_count",
        ),
        defined_lesion_precision_case_count=_expect_int(
            value["defined_lesion_precision_case_count"],
            field_name="defined_lesion_precision_case_count",
        ),
        mean_false_positive_lesions_per_scan=_expect_number(
            value["mean_false_positive_lesions_per_scan"],
            field_name="mean_false_positive_lesions_per_scan",
        ),
        mean_signed_volume_error_ml=_expect_number(
            value["mean_signed_volume_error_ml"],
            field_name="mean_signed_volume_error_ml",
        ),
        mean_absolute_volume_error_ml=_expect_number(
            value["mean_absolute_volume_error_ml"],
            field_name="mean_absolute_volume_error_ml",
        ),
        macro_relative_volume_error=_expect_optional_number(
            value["macro_relative_volume_error"],
            field_name="macro_relative_volume_error",
        ),
        defined_relative_volume_error_case_count=_expect_int(
            value["defined_relative_volume_error_case_count"],
            field_name="defined_relative_volume_error_case_count",
        ),
    )


def _validate_case_consistency(case_metrics: BaselineCaseMetrics) -> None:
    if case_metrics.intersection_voxels > case_metrics.ground_truth_foreground_voxels:
        raise BaselineMetricValidationError(
            "intersection_voxels must not exceed ground_truth_foreground_voxels."
        )
    if case_metrics.intersection_voxels > case_metrics.predicted_foreground_voxels:
        raise BaselineMetricValidationError(
            "intersection_voxels must not exceed predicted_foreground_voxels."
        )
    if case_metrics.union_voxels < case_metrics.intersection_voxels:
        raise BaselineMetricValidationError("union_voxels must be at least intersection_voxels.")
    if case_metrics.true_positive_lesion_count > case_metrics.ground_truth_lesion_count:
        raise BaselineMetricValidationError(
            "true_positive_lesion_count must not exceed ground_truth_lesion_count."
        )
    if case_metrics.true_positive_lesion_count > case_metrics.predicted_lesion_count:
        raise BaselineMetricValidationError(
            "true_positive_lesion_count must not exceed predicted_lesion_count."
        )
    if case_metrics.false_positive_lesions_per_scan != case_metrics.false_positive_lesion_count:
        raise BaselineMetricValidationError(
            "false_positive_lesions_per_scan must equal false_positive_lesion_count."
        )


def _require_case_tuple(case_records: Sequence[BaselineCaseMetrics]) -> None:
    for case_record in case_records:
        if not isinstance(case_record, BaselineCaseMetrics):
            raise BaselineMetricValidationError(
                "case_records must contain only BaselineCaseMetrics values."
            )


def _require_unique_case_identifiers(case_records: Sequence[BaselineCaseMetrics]) -> None:
    identifiers = [case_record.case_identifier for case_record in case_records]
    if len(set(identifiers)) != len(identifiers):
        raise BaselineMetricValidationError(
            "case_records must not contain duplicate case identifiers."
        )


def _require_case_identifier(value: str) -> None:
    if not _SAFE_IDENTIFIER_RE.fullmatch(value):
        raise BaselineMetricValidationError(
            "anonymous case identifiers must match the conservative ASCII identifier policy."
        )
    _require_no_absolute_path_string(value, field_name="case_identifier")


def _require_contract_version(value: str, expected: str) -> None:
    if value != expected:
        raise BaselineMetricVersionError(f"Unsupported metric contract version: {value!r}.")


def _require_sha256(value: str, *, field_name: str) -> None:
    if not _SHA256_RE.fullmatch(value):
        raise BaselineMetricValidationError(
            f"{field_name} must be exactly 64 lowercase hexadecimal characters."
        )


def _require_nonnegative_int(value: int, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BaselineMetricValidationError(f"{field_name} must be a nonnegative integer.")


def _require_finite_float(value: float, *, field_name: str) -> None:
    if not isinstance(value, float) or not math.isfinite(value):
        raise BaselineMetricValidationError(f"{field_name} must be a finite float.")


def _require_optional_finite_float(value: float | None, *, field_name: str) -> None:
    if value is None:
        return
    _require_finite_float(value, field_name=field_name)


def _require_nonnegative_float(value: float, *, field_name: str) -> None:
    _require_finite_float(value, field_name=field_name)
    if value < 0.0:
        raise BaselineMetricValidationError(f"{field_name} must be nonnegative.")


def _require_optional_nonnegative_float(value: float | None, *, field_name: str) -> None:
    if value is None:
        return
    _require_nonnegative_float(value, field_name=field_name)


def _require_unit_interval(value: float, *, field_name: str) -> None:
    _require_finite_float(value, field_name=field_name)
    if value < 0.0 or value > 1.0:
        raise BaselineMetricValidationError(f"{field_name} must be within [0.0, 1.0].")


def _require_optional_unit_interval(value: float | None, *, field_name: str) -> None:
    if value is None:
        return
    _require_unit_interval(value, field_name=field_name)


def _reject_nonfinite_result(value: float, *, field_name: str) -> None:
    if not math.isfinite(value):
        raise BaselineMetricValidationError(f"Computed {field_name} must be finite.")


def _parse_json_object(payload: bytes | str, *, context: str) -> dict[str, Any]:
    text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
    try:
        parsed = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_bad_json_constant,
        )
    except Exception as error:
        raise BaselineMetricSerializationError(
            f"Failed to parse {context} JSON: {error}"
        ) from error
    if not isinstance(parsed, dict):
        raise BaselineMetricSerializationError(f"{context} JSON must encode an object.")
    return parsed


def _require_exact_fields(
    payload: Mapping[str, Any],
    *,
    expected: set[str],
    context: str,
) -> None:
    unknown_fields = set(payload) - expected
    if unknown_fields:
        raise BaselineMetricSerializationError(
            f"Unknown {context} field(s): {', '.join(sorted(unknown_fields))}."
        )
    missing_fields = expected - set(payload)
    if missing_fields:
        raise BaselineMetricSerializationError(
            f"Missing {context} field(s): {', '.join(sorted(missing_fields))}."
        )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BaselineMetricSerializationError(f"Duplicate JSON object key: {key!r}.")
        result[key] = value
    return result


def _bad_json_constant(constant: str) -> Any:
    raise BaselineMetricSerializationError(f"Invalid JSON constant: {constant!r}.")


def _expect_string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise BaselineMetricSerializationError(f"{field_name} must be a string.")
    return value


def _expect_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BaselineMetricSerializationError(f"{field_name} must be an integer.")
    return value


def _expect_number(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BaselineMetricSerializationError(f"{field_name} must be a number.")
    scalar = float(value)
    if not math.isfinite(scalar):
        raise BaselineMetricSerializationError(f"{field_name} must be finite.")
    return scalar


def _expect_optional_number(value: Any, *, field_name: str) -> float | None:
    if value is None:
        return None
    return _expect_number(value, field_name=field_name)


def _expect_float_tuple3(value: Any, *, field_name: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise BaselineMetricSerializationError(f"{field_name} must be a length-3 array.")
    return tuple(_expect_number(item, field_name=field_name) for item in value)  # type: ignore[return-value]


def _require_no_absolute_paths(value: Any) -> None:
    if isinstance(value, str):
        _require_no_absolute_path_string(value, field_name="persisted string")
        return
    if isinstance(value, Mapping):
        for item in value.values():
            _require_no_absolute_paths(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _require_no_absolute_paths(item)


def _require_no_absolute_path_string(value: str, *, field_name: str) -> None:
    if value.startswith("/") or _WINDOWS_ABSOLUTE_PATH_RE.match(value):
        raise BaselineMetricValidationError(f"{field_name} must not contain an absolute path.")
