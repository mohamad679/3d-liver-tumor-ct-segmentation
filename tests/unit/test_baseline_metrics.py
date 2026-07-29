from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest

from protoem_ct.baselines.metrics import (
    BASELINE_CASE_METRICS_VERSION,
    BASELINE_LESION_CONNECTIVITY,
    BASELINE_METRIC_REPORT_VERSION,
    BaselineCaseMetrics,
    BaselineMetricHashError,
    BaselineMetricReport,
    BaselineMetricSerializationError,
    BaselineMetricValidationError,
    BaselineMetricVersionError,
    baseline_case_metrics_from_json,
    baseline_case_metrics_to_json,
    baseline_metric_report_from_json,
    baseline_metric_report_to_json,
    build_baseline_metric_report,
    compute_baseline_case_metrics,
)


def _compute_case(
    *,
    case_identifier: str = "case_001",
    ground_truth_mask: np.ndarray[Any, Any] | None = None,
    prediction_mask: np.ndarray[Any, Any] | None = None,
    voxel_spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
    nsd_tolerance_mm: float = 1.0,
) -> BaselineCaseMetrics:
    ground_truth = (
        np.zeros((3, 3, 3), dtype=np.uint8) if ground_truth_mask is None else ground_truth_mask
    )
    prediction = np.zeros((3, 3, 3), dtype=np.uint8) if prediction_mask is None else prediction_mask
    return compute_baseline_case_metrics(
        case_identifier=case_identifier,
        ground_truth_mask=ground_truth,
        prediction_mask=prediction,
        voxel_spacing_mm=voxel_spacing_mm,
        nsd_tolerance_mm=nsd_tolerance_mm,
    )


def _report(case_records: tuple[BaselineCaseMetrics, ...]) -> BaselineMetricReport:
    return build_baseline_metric_report(
        baseline_family="nnunet_v2",
        run_identifier="run_001",
        dataset_manifest_sha256="1" * 64,
        development_split_sha256="2" * 64,
        prediction_manifest_sha256="3" * 64,
        metric_config_sha256="4" * 64,
        nsd_tolerance_mm=1.0,
        case_records=case_records,
    )


def test_perfect_overlap() -> None:
    ground_truth = np.zeros((3, 3, 3), dtype=np.uint8)
    ground_truth[1, 1, 1] = 1

    case_metrics = _compute_case(ground_truth_mask=ground_truth, prediction_mask=ground_truth)

    assert case_metrics.tumor_dice == 1.0
    assert case_metrics.tumor_iou == 1.0
    assert case_metrics.hd95_mm == 0.0
    assert case_metrics.normalized_surface_dice == 1.0
    assert case_metrics.true_positive_lesion_count == 1
    assert case_metrics.false_positive_lesion_count == 0
    assert case_metrics.false_negative_lesion_count == 0


def test_partial_overlap() -> None:
    ground_truth = np.zeros((3, 3, 3), dtype=np.uint8)
    prediction = np.zeros((3, 3, 3), dtype=np.uint8)
    ground_truth[1, 1, 1] = 1
    ground_truth[1, 1, 2] = 1
    prediction[1, 1, 1] = 1
    prediction[1, 2, 1] = 1

    case_metrics = _compute_case(ground_truth_mask=ground_truth, prediction_mask=prediction)

    assert case_metrics.intersection_voxels == 1
    assert case_metrics.union_voxels == 3
    assert case_metrics.tumor_dice == pytest.approx(0.5)
    assert case_metrics.tumor_iou == pytest.approx(1.0 / 3.0)


def test_disjoint_nonempty_masks() -> None:
    ground_truth = np.zeros((3, 3, 3), dtype=np.uint8)
    prediction = np.zeros((3, 3, 3), dtype=np.uint8)
    ground_truth[0, 0, 0] = 1
    prediction[2, 2, 2] = 1

    case_metrics = _compute_case(ground_truth_mask=ground_truth, prediction_mask=prediction)

    assert case_metrics.tumor_dice == 0.0
    assert case_metrics.tumor_iou == 0.0
    assert case_metrics.normalized_surface_dice == 0.0
    assert case_metrics.true_positive_lesion_count == 0
    assert case_metrics.false_positive_lesion_count == 1
    assert case_metrics.false_negative_lesion_count == 1


def test_both_empty_case() -> None:
    case_metrics = _compute_case()

    assert case_metrics.tumor_dice == 1.0
    assert case_metrics.tumor_iou == 1.0
    assert case_metrics.hd95_mm == 0.0
    assert case_metrics.normalized_surface_dice == 1.0
    assert case_metrics.lesion_recall is None
    assert case_metrics.lesion_precision is None
    assert case_metrics.lesion_f1 == 1.0
    assert case_metrics.relative_volume_error is None


def test_empty_prediction() -> None:
    ground_truth = np.zeros((3, 3, 3), dtype=np.uint8)
    ground_truth[1, 1, 1] = 1

    case_metrics = _compute_case(ground_truth_mask=ground_truth)

    assert case_metrics.tumor_dice == 0.0
    assert case_metrics.tumor_iou == 0.0
    assert case_metrics.hd95_mm is None
    assert case_metrics.normalized_surface_dice == 0.0
    assert case_metrics.lesion_recall == 0.0
    assert case_metrics.lesion_precision is None
    assert case_metrics.lesion_f1 == 0.0


def test_empty_ground_truth() -> None:
    prediction = np.zeros((3, 3, 3), dtype=np.uint8)
    prediction[1, 1, 1] = 1

    case_metrics = _compute_case(prediction_mask=prediction)

    assert case_metrics.tumor_dice == 0.0
    assert case_metrics.tumor_iou == 0.0
    assert case_metrics.hd95_mm is None
    assert case_metrics.normalized_surface_dice == 0.0
    assert case_metrics.lesion_recall is None
    assert case_metrics.lesion_precision == 0.0
    assert case_metrics.lesion_f1 == 0.0


def test_anisotropic_spacing_and_volume_errors() -> None:
    ground_truth = np.zeros((3, 3, 3), dtype=np.uint8)
    prediction = np.zeros((3, 3, 3), dtype=np.uint8)
    ground_truth[1, 1, 1] = 1
    prediction[1, 1, 1] = 1
    prediction[1, 1, 2] = 1

    case_metrics = _compute_case(
        ground_truth_mask=ground_truth,
        prediction_mask=prediction,
        voxel_spacing_mm=(1.0, 2.0, 3.0),
    )

    assert case_metrics.ground_truth_volume_ml == pytest.approx(0.006)
    assert case_metrics.predicted_volume_ml == pytest.approx(0.012)
    assert case_metrics.signed_volume_error_ml == pytest.approx(0.006)
    assert case_metrics.absolute_volume_error_ml == pytest.approx(0.006)
    assert case_metrics.relative_volume_error == pytest.approx(1.0)


def test_physical_hd95_one_voxel_displacement() -> None:
    ground_truth = np.zeros((3, 3, 3), dtype=np.uint8)
    prediction = np.zeros((3, 3, 3), dtype=np.uint8)
    ground_truth[1, 1, 1] = 1
    prediction[2, 1, 1] = 1

    case_metrics = _compute_case(
        ground_truth_mask=ground_truth,
        prediction_mask=prediction,
        voxel_spacing_mm=(2.0, 1.0, 1.0),
    )

    assert case_metrics.hd95_mm == pytest.approx(2.0)


def test_nsd_at_zero_tolerance() -> None:
    ground_truth = np.zeros((3, 3, 3), dtype=np.uint8)
    prediction = np.zeros((3, 3, 3), dtype=np.uint8)
    ground_truth[1, 1, 1] = 1
    prediction[2, 1, 1] = 1

    case_metrics = _compute_case(
        ground_truth_mask=ground_truth,
        prediction_mask=prediction,
        nsd_tolerance_mm=0.0,
    )

    assert case_metrics.normalized_surface_dice == 0.0


def test_nsd_at_inclusive_tolerance_boundary() -> None:
    ground_truth = np.zeros((3, 3, 3), dtype=np.uint8)
    prediction = np.zeros((3, 3, 3), dtype=np.uint8)
    ground_truth[1, 1, 1] = 1
    prediction[2, 1, 1] = 1

    case_metrics = _compute_case(
        ground_truth_mask=ground_truth,
        prediction_mask=prediction,
        nsd_tolerance_mm=1.0,
    )

    assert case_metrics.normalized_surface_dice == 1.0


def test_multiple_correctly_matched_lesions() -> None:
    ground_truth = np.zeros((5, 5, 5), dtype=np.uint8)
    prediction = np.zeros((5, 5, 5), dtype=np.uint8)
    ground_truth[0, 0, 0] = 1
    ground_truth[4, 4, 4] = 1
    prediction[0, 0, 0] = 1
    prediction[4, 4, 4] = 1

    case_metrics = _compute_case(ground_truth_mask=ground_truth, prediction_mask=prediction)

    assert case_metrics.ground_truth_lesion_count == 2
    assert case_metrics.predicted_lesion_count == 2
    assert case_metrics.true_positive_lesion_count == 2
    assert case_metrics.false_positive_lesion_count == 0
    assert case_metrics.false_negative_lesion_count == 0


def test_missed_lesion() -> None:
    ground_truth = np.zeros((5, 5, 5), dtype=np.uint8)
    prediction = np.zeros((5, 5, 5), dtype=np.uint8)
    ground_truth[0, 0, 0] = 1
    ground_truth[4, 4, 4] = 1
    prediction[0, 0, 0] = 1

    case_metrics = _compute_case(ground_truth_mask=ground_truth, prediction_mask=prediction)

    assert case_metrics.true_positive_lesion_count == 1
    assert case_metrics.false_negative_lesion_count == 1
    assert case_metrics.lesion_recall == pytest.approx(0.5)


def test_false_positive_lesion() -> None:
    ground_truth = np.zeros((5, 5, 5), dtype=np.uint8)
    prediction = np.zeros((5, 5, 5), dtype=np.uint8)
    ground_truth[0, 0, 0] = 1
    prediction[0, 0, 0] = 1
    prediction[4, 4, 4] = 1

    case_metrics = _compute_case(ground_truth_mask=ground_truth, prediction_mask=prediction)

    assert case_metrics.true_positive_lesion_count == 1
    assert case_metrics.false_positive_lesion_count == 1
    assert case_metrics.lesion_precision == pytest.approx(0.5)


def test_split_prediction_lesion_case() -> None:
    ground_truth = np.zeros((5, 5, 5), dtype=np.uint8)
    prediction = np.zeros((5, 5, 5), dtype=np.uint8)
    ground_truth[2, 2, 1:4] = 1
    prediction[2, 2, 1] = 1
    prediction[2, 2, 3] = 1

    case_metrics = _compute_case(ground_truth_mask=ground_truth, prediction_mask=prediction)

    assert case_metrics.ground_truth_lesion_count == 1
    assert case_metrics.predicted_lesion_count == 2
    assert case_metrics.true_positive_lesion_count == 1
    assert case_metrics.false_positive_lesion_count == 1


def test_merged_prediction_lesion_case() -> None:
    ground_truth = np.zeros((5, 5, 5), dtype=np.uint8)
    prediction = np.zeros((5, 5, 5), dtype=np.uint8)
    ground_truth[2, 2, 1] = 1
    ground_truth[2, 2, 3] = 1
    prediction[2, 2, 1:4] = 1

    case_metrics = _compute_case(ground_truth_mask=ground_truth, prediction_mask=prediction)

    assert case_metrics.ground_truth_lesion_count == 2
    assert case_metrics.predicted_lesion_count == 1
    assert case_metrics.true_positive_lesion_count == 1
    assert case_metrics.false_negative_lesion_count == 1


def test_deterministic_one_to_one_matching() -> None:
    ground_truth = np.zeros((6, 6, 6), dtype=np.uint8)
    prediction = np.zeros((6, 6, 6), dtype=np.uint8)
    ground_truth[2, 2, 2] = 1
    ground_truth[2, 2, 3] = 1
    prediction[2, 2, 2] = 1
    prediction[2, 2, 3] = 1
    prediction[2, 2, 4] = 1

    case_metrics = _compute_case(ground_truth_mask=ground_truth, prediction_mask=prediction)

    assert case_metrics.true_positive_lesion_count == 1
    assert case_metrics.false_positive_lesion_count == 0
    assert case_metrics.false_negative_lesion_count == 0


def test_relative_volume_error_null_behavior() -> None:
    prediction = np.zeros((3, 3, 3), dtype=np.uint8)
    prediction[1, 1, 1] = 1

    case_metrics = _compute_case(prediction_mask=prediction)

    assert case_metrics.relative_volume_error is None


def test_shape_mismatch_rejected() -> None:
    with pytest.raises(BaselineMetricValidationError):
        _compute_case(
            ground_truth_mask=np.zeros((3, 3, 3), dtype=np.uint8),
            prediction_mask=np.zeros((3, 3, 2), dtype=np.uint8),
        )


def test_invalid_dimensionality_rejected() -> None:
    with pytest.raises(BaselineMetricValidationError):
        _compute_case(
            ground_truth_mask=np.zeros((3, 3), dtype=np.uint8),
            prediction_mask=np.zeros((3, 3), dtype=np.uint8),
        )


@pytest.mark.parametrize(
    "invalid_value",
    [0.5, -1.0, 2.0],
)
def test_invalid_mask_values_rejected(invalid_value: float) -> None:
    invalid = np.zeros((3, 3, 3), dtype=np.float64)
    invalid[1, 1, 1] = invalid_value

    with pytest.raises(BaselineMetricValidationError):
        _compute_case(ground_truth_mask=invalid, prediction_mask=invalid)


def test_nan_and_inf_rejection() -> None:
    nan_mask = np.zeros((3, 3, 3), dtype=np.float64)
    nan_mask[1, 1, 1] = np.nan
    with pytest.raises(BaselineMetricValidationError):
        _compute_case(ground_truth_mask=nan_mask, prediction_mask=np.zeros((3, 3, 3)))

    inf_mask = np.zeros((3, 3, 3), dtype=np.float64)
    inf_mask[1, 1, 1] = np.inf
    with pytest.raises(BaselineMetricValidationError):
        _compute_case(ground_truth_mask=inf_mask, prediction_mask=np.zeros((3, 3, 3)))


def test_invalid_spacing_rejected() -> None:
    with pytest.raises(BaselineMetricValidationError):
        _compute_case(voxel_spacing_mm=(1.0, 1.0, 0.0))


def test_invalid_tolerance_rejected() -> None:
    with pytest.raises(BaselineMetricValidationError):
        _compute_case(nsd_tolerance_mm=-1.0)


def test_duplicate_case_identifiers_rejected() -> None:
    case_metrics = _compute_case(case_identifier="case_001")
    duplicate = _compute_case(case_identifier="case_001")

    with pytest.raises(BaselineMetricValidationError):
        _report((case_metrics, duplicate))


def test_deterministic_aggregate_ordering() -> None:
    case_b = _compute_case(case_identifier="case_b")
    case_a = _compute_case(case_identifier="case_a")

    report = _report((case_b, case_a))

    assert tuple(case.case_identifier for case in report.case_records) == ("case_a", "case_b")


def test_pooled_versus_macro_metric_distinction() -> None:
    case_perfect = _compute_case(
        case_identifier="case_a",
        ground_truth_mask=np.pad(np.ones((1, 1, 1), dtype=np.uint8), 1),
        prediction_mask=np.pad(np.ones((1, 1, 1), dtype=np.uint8), 1),
    )
    ground_truth = np.zeros((3, 3, 3), dtype=np.uint8)
    prediction = np.zeros((3, 3, 3), dtype=np.uint8)
    ground_truth[1, 1, 1] = 1
    ground_truth[1, 1, 2] = 1
    prediction[1, 1, 1] = 1
    case_partial = _compute_case(
        case_identifier="case_b",
        ground_truth_mask=ground_truth,
        prediction_mask=prediction,
    )

    report = _report((case_perfect, case_partial))

    assert report.aggregate is not None
    assert report.aggregate.pooled_dice != report.aggregate.macro_dice


def test_undefined_aggregate_hd95_behavior() -> None:
    ground_truth = np.zeros((3, 3, 3), dtype=np.uint8)
    ground_truth[1, 1, 1] = 1
    case_a = _compute_case(case_identifier="case_a", ground_truth_mask=ground_truth)
    case_b = _compute_case(
        case_identifier="case_b",
        prediction_mask=np.pad(np.ones((1, 1, 1), dtype=np.uint8), 1),
    )

    report = _report((case_a, case_b))

    assert report.aggregate is not None
    assert report.aggregate.defined_hd95_case_count == 0
    assert report.aggregate.macro_hd95_mm is None


def test_deterministic_byte_identical_json() -> None:
    case_metrics = _compute_case()
    report = _report((case_metrics,))

    assert baseline_case_metrics_to_json(case_metrics) == baseline_case_metrics_to_json(
        case_metrics
    )
    assert baseline_metric_report_to_json(report) == baseline_metric_report_to_json(report)


def test_hash_recomputation() -> None:
    case_metrics = _compute_case()
    parsed_case = baseline_case_metrics_from_json(baseline_case_metrics_to_json(case_metrics))
    report = _report((case_metrics,))
    parsed_report = baseline_metric_report_from_json(baseline_metric_report_to_json(report))

    assert parsed_case.artifact_hash == case_metrics.artifact_hash
    assert parsed_report.artifact_hash == report.artifact_hash


def test_tamper_detection() -> None:
    case_payload = json.loads(baseline_case_metrics_to_json(_compute_case()))
    case_payload["tumor_dice"] = 0.123
    with pytest.raises(BaselineMetricHashError):
        baseline_case_metrics_from_json(json.dumps(case_payload).encode("utf-8"))

    report_payload = json.loads(baseline_metric_report_to_json(_report((_compute_case(),))))
    report_payload["run_identifier"] = "tampered_run"
    with pytest.raises(BaselineMetricHashError):
        baseline_metric_report_from_json(json.dumps(report_payload).encode("utf-8"))


def test_unknown_field_rejection() -> None:
    case_payload = json.loads(baseline_case_metrics_to_json(_compute_case()))
    case_payload["unknown_field"] = "value"
    with pytest.raises(BaselineMetricSerializationError):
        baseline_case_metrics_from_json(json.dumps(case_payload).encode("utf-8"))

    report_payload = json.loads(baseline_metric_report_to_json(_report((_compute_case(),))))
    report_payload["unknown_field"] = "value"
    with pytest.raises(BaselineMetricSerializationError):
        baseline_metric_report_from_json(json.dumps(report_payload).encode("utf-8"))


def test_unknown_version_rejection() -> None:
    case_payload = json.loads(baseline_case_metrics_to_json(_compute_case()))
    case_payload["contract_version"] = "baseline_case_metrics_v999"
    with pytest.raises(BaselineMetricVersionError):
        baseline_case_metrics_from_json(json.dumps(case_payload).encode("utf-8"))

    report_payload = json.loads(baseline_metric_report_to_json(_report((_compute_case(),))))
    report_payload["contract_version"] = "baseline_metric_report_v999"
    with pytest.raises(BaselineMetricVersionError):
        baseline_metric_report_from_json(json.dumps(report_payload).encode("utf-8"))


def test_absolute_path_rejection() -> None:
    case_metrics = _compute_case()
    with pytest.raises(BaselineMetricValidationError):
        cast(Any, replace)(case_metrics, case_identifier="/tmp/bad")

    report = _report((case_metrics,))
    with pytest.raises(BaselineMetricValidationError):
        cast(Any, replace)(report, run_identifier="/tmp/bad")


def test_case_and_report_contract_versions_and_fields() -> None:
    case_metrics = _compute_case()
    report = _report((case_metrics,))

    assert case_metrics.contract_version == BASELINE_CASE_METRICS_VERSION
    assert report.contract_version == BASELINE_METRIC_REPORT_VERSION
    assert report.lesion_connectivity == BASELINE_LESION_CONNECTIVITY


def test_importing_baselines_and_metrics_does_not_import_heavy_dependencies(
    run_import_guard: Callable[[tuple[str, ...], tuple[str, ...]], None],
) -> None:
    run_import_guard(
        ("protoem_ct.baselines", "protoem_ct.baselines.metrics"),
        ("torch", "torchvision", "monai", "nnunetv2", "SimpleITK", "mlflow"),
    )
