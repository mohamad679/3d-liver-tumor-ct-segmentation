"""Integration tests for deterministic Phase 7 publication helpers."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from protoem_ct.artifacts.hashing import canonical_json_bytes, sha256_json
from protoem_ct.evaluation.failure_detection import (
    compute_failure_detection_auroc,
    compute_uncertainty_error_correlation,
)
from protoem_ct.robustness.artifacts import (
    PHASE7_CORRUPTION_MANIFEST_SCHEMA_NAME,
    PHASE7_CORRUPTION_MANIFEST_SCHEMA_VERSION,
    PHASE7_CORRUPTION_SPECIFICATION_SCHEMA_NAME,
    PHASE7_CORRUPTION_SPECIFICATION_SCHEMA_VERSION,
    PHASE7_GEOMETRY_RECORD_SCHEMA_NAME,
    PHASE7_GEOMETRY_RECORD_SCHEMA_VERSION,
    PHASE7_RUN_SUMMARY_SCHEMA_NAME,
    PHASE7_RUN_SUMMARY_SCHEMA_VERSION,
    PHASE7_TRANSFORM_RESULT_SCHEMA_NAME,
    PHASE7_TRANSFORM_RESULT_SCHEMA_VERSION,
    Phase7CorruptionManifest,
    Phase7CorruptionSpecification,
    Phase7GeometryRecord,
    Phase7RunSummary,
    Phase7TransformResult,
    phase7_corruption_manifest_from_json,
    phase7_corruption_manifest_to_json,
    phase7_corruption_specification_from_json,
    phase7_corruption_specification_to_dict,
    phase7_geometry_record_from_json,
    phase7_run_summary_from_json,
    phase7_run_summary_to_json,
    phase7_transform_result_from_json,
)
from protoem_ct.robustness.publication import (
    PHASE7_CALIBRATION_PLOT_NAME,
    PHASE7_CORRUPTION_DEGRADATION_PLOT_NAME,
    PHASE7_PUBLICATION_FILENAMES,
    PHASE7_RISK_COVERAGE_PLOT_NAME,
    PHASE7_RUN_SUMMARY_NAME,
    PHASE7_SUBGROUP_TABLE_NAME,
    PHASE7_SUMMARY_MARKDOWN_NAME,
    PHASE7_UNCERTAINTY_FAILURE_TABLE_NAME,
    Phase7PublicationCollisionError,
    Phase7PublicationInputs,
    Phase7PublicationPathError,
    Phase7PublicationValidationError,
    build_phase7_corruption_degradation_plot_png,
    build_phase7_geometry_records_collection_json,
    build_phase7_publication_artifact_bytes,
    build_phase7_transform_results_collection_json,
    publish_phase7_artifacts,
)
from protoem_ct.uncertainty.artifacts import (
    PHASE7_CALIBRATION_BIN_SCHEMA_NAME,
    PHASE7_CALIBRATION_BIN_SCHEMA_VERSION,
    PHASE7_CALIBRATION_RESULT_SCHEMA_NAME,
    PHASE7_CALIBRATION_RESULT_SCHEMA_VERSION,
    PHASE7_DEGRADATION_RESULT_SCHEMA_NAME,
    PHASE7_DEGRADATION_RESULT_SCHEMA_VERSION,
    PHASE7_LESION_SUBGROUP_RECORD_SCHEMA_NAME,
    PHASE7_LESION_SUBGROUP_RECORD_SCHEMA_VERSION,
    PHASE7_LESION_SUBGROUP_RESULT_SCHEMA_NAME,
    PHASE7_LESION_SUBGROUP_RESULT_SCHEMA_VERSION,
    PHASE7_RISK_COVERAGE_POINT_SCHEMA_NAME,
    PHASE7_RISK_COVERAGE_POINT_SCHEMA_VERSION,
    PHASE7_RISK_COVERAGE_RESULT_SCHEMA_NAME,
    PHASE7_RISK_COVERAGE_RESULT_SCHEMA_VERSION,
    PHASE7_UNCERTAINTY_RESULT_SCHEMA_NAME,
    PHASE7_UNCERTAINTY_RESULT_SCHEMA_VERSION,
    Phase7CalibrationBin,
    Phase7CalibrationResult,
    Phase7DegradationResult,
    Phase7LesionSubgroupRecord,
    Phase7LesionSubgroupResult,
    Phase7RiskCoveragePoint,
    Phase7RiskCoverageResult,
    Phase7UncertaintyArtifactSerializationError,
    Phase7UncertaintyResult,
    phase7_calibration_bin_to_dict,
    phase7_calibration_result_from_json,
    phase7_calibration_result_to_json,
    phase7_degradation_result_from_json,
    phase7_degradation_result_to_json,
    phase7_lesion_subgroup_record_to_dict,
    phase7_lesion_subgroup_result_from_json,
    phase7_lesion_subgroup_result_to_json,
    phase7_risk_coverage_point_to_dict,
    phase7_risk_coverage_result_from_json,
    phase7_risk_coverage_result_to_json,
    phase7_uncertainty_result_from_json,
    phase7_uncertainty_result_to_json,
)
from protoem_ct.uncertainty.publication import (
    canonical_phase7_failure_detection_results_json,
    render_phase7_subgroup_table_markdown_from_json,
    render_phase7_uncertainty_failure_table_markdown_from_json,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
HEX_1 = "1" * 64
HEX_2 = "2" * 64
HEX_3 = "3" * 64
HEX_4 = "4" * 64
HEX_5 = "5" * 64
HEX_6 = "6" * 64
HEX_7 = "7" * 64
HEX_8 = "8" * 64


def test_valid_publication_with_synthetic_validated_json_bytes(tmp_path: Path) -> None:
    inputs = _publication_inputs()
    output_root = tmp_path / "phase7-published"

    result = publish_phase7_artifacts(output_root=output_root, inputs=inputs)

    assert result.reused_existing_output is False
    assert set(result.published_filenames) == set(PHASE7_PUBLICATION_FILENAMES)
    assert {
        path.relative_to(output_root).as_posix()
        for path in output_root.rglob("*")
        if path.is_file()
    } == set(PHASE7_PUBLICATION_FILENAMES)


def test_markdown_is_derived_from_json_and_malformed_json_fails(tmp_path: Path) -> None:
    inputs = _publication_inputs()
    output_root = tmp_path / "phase7-published"

    publish_phase7_artifacts(output_root=output_root, inputs=inputs)

    run_summary = json.loads((output_root / PHASE7_RUN_SUMMARY_NAME).read_text("utf-8"))
    summary_markdown = (output_root / PHASE7_SUMMARY_MARKDOWN_NAME).read_text("utf-8")
    uncertainty_table = (output_root / PHASE7_UNCERTAINTY_FAILURE_TABLE_NAME).read_text("utf-8")
    subgroup_table = (output_root / PHASE7_SUBGROUP_TABLE_NAME).read_text("utf-8")
    assert run_summary["config_hash"] in summary_markdown
    assert "predictive_entropy" in uncertainty_table
    assert "small" in subgroup_table

    broken = replace(inputs, uncertainty_result_json=b"{not-json")
    with pytest.raises(Phase7UncertaintyArtifactSerializationError):
        build_phase7_publication_artifact_bytes(broken)

    with pytest.raises(Phase7UncertaintyArtifactSerializationError):
        render_phase7_uncertainty_failure_table_markdown_from_json(
            uncertainty_result_json=b"{not-json",
            failure_detection_json=inputs.failure_detection_json,
        )
    with pytest.raises(Phase7UncertaintyArtifactSerializationError):
        render_phase7_subgroup_table_markdown_from_json(b"{not-json")


def test_plots_are_derived_from_json_inputs(tmp_path: Path) -> None:
    inputs = _publication_inputs()
    output_root = tmp_path / "phase7-published"

    publish_phase7_artifacts(output_root=output_root, inputs=inputs)

    regenerated = build_phase7_corruption_degradation_plot_png(
        transform_results_json=inputs.transform_results_json,
        degradation_result_json=inputs.degradation_result_json,
    )
    assert regenerated == (output_root / PHASE7_CORRUPTION_DEGRADATION_PLOT_NAME).read_bytes()
    assert (output_root / PHASE7_CALIBRATION_PLOT_NAME).stat().st_size > 0
    assert (output_root / PHASE7_RISK_COVERAGE_PLOT_NAME).stat().st_size > 0


def test_safe_idempotent_regeneration(tmp_path: Path) -> None:
    inputs = _publication_inputs()
    output_root = tmp_path / "phase7-published"

    first = publish_phase7_artifacts(output_root=output_root, inputs=inputs)
    second = publish_phase7_artifacts(output_root=output_root, inputs=inputs)

    assert first.reused_existing_output is False
    assert second.reused_existing_output is True


def test_incompatible_overwrite_rejection(tmp_path: Path) -> None:
    inputs = _publication_inputs()
    output_root = tmp_path / "phase7-published"
    output_root.mkdir()
    (output_root / PHASE7_RUN_SUMMARY_NAME).write_text("not canonical\n", encoding="utf-8")

    with pytest.raises(Phase7PublicationCollisionError):
        publish_phase7_artifacts(output_root=output_root, inputs=inputs)


def test_repository_internal_root_rejection() -> None:
    with pytest.raises(Phase7PublicationPathError):
        publish_phase7_artifacts(
            output_root=REPO_ROOT / "phase7-forbidden-output",
            inputs=_publication_inputs(),
        )


def test_parent_traversal_output_root_rejection(tmp_path: Path) -> None:
    with pytest.raises(Phase7PublicationPathError):
        publish_phase7_artifacts(
            output_root=tmp_path / ".." / "escape",
            inputs=_publication_inputs(),
        )


def test_symlink_escape_rejection(tmp_path: Path) -> None:
    external_target = tmp_path / "external"
    external_target.mkdir()
    symlink_root = tmp_path / "symlink-root"
    symlink_root.symlink_to(external_target, target_is_directory=True)

    with pytest.raises(Phase7PublicationPathError):
        publish_phase7_artifacts(
            output_root=symlink_root / "nested",
            inputs=_publication_inputs(),
        )


def test_no_generated_artifacts_under_repository(tmp_path: Path) -> None:
    publish_phase7_artifacts(
        output_root=tmp_path / "phase7-published", inputs=_publication_inputs()
    )

    assert not (REPO_ROOT / PHASE7_SUMMARY_MARKDOWN_NAME).exists()
    assert not (REPO_ROOT / PHASE7_CORRUPTION_DEGRADATION_PLOT_NAME).exists()


def test_cross_artifact_hash_mismatch_rejected() -> None:
    inputs = _publication_inputs()
    mapping = json.loads(inputs.phase7_run_summary_json)
    mapping["uncertainty_result_hash"] = HEX_8
    broken = replace(inputs, phase7_run_summary_json=_run_summary_json_from_mapping(mapping))

    with pytest.raises(Phase7PublicationValidationError, match="uncertainty_result_hash"):
        build_phase7_publication_artifact_bytes(broken)


def _publication_inputs() -> Phase7PublicationInputs:
    manifest = _manifest()
    transform = _transform_result()
    geometry = _geometry_record()
    uncertainty = _uncertainty_result()
    calibration = _calibration_result()
    risk = _risk_coverage_result(uncertainty_hash=uncertainty.uncertainty_result_hash)
    degradation = _degradation_result()
    subgroup = _lesion_subgroup_result()
    correlation = compute_uncertainty_error_correlation(
        uncertainty_values=np.asarray([0.1, 0.9, 0.4], dtype=np.float64),
        error_indicators=np.asarray([0, 1, 0], dtype=np.uint8),
    )
    auroc = compute_failure_detection_auroc(
        case_uncertainty_scores=np.asarray([0.1, 0.9], dtype=np.float64),
        failure_indicators=np.asarray([0, 1], dtype=np.uint8),
    )
    summary = _run_summary(
        manifest_hash=manifest.corruption_manifest_hash,
        uncertainty_hash=uncertainty.uncertainty_result_hash,
        calibration_hash=calibration.calibration_result_hash,
        risk_hash=risk.risk_coverage_result_hash,
        degradation_hash=degradation.degradation_result_hash,
        subgroup_hash=subgroup.lesion_subgroup_result_hash,
    )
    return Phase7PublicationInputs(
        effective_config_json=canonical_json_bytes(
            {
                "phase7_robustness_uncertainty": {
                    "schema_version": "v1",
                    "synthetic_mode_only": True,
                }
            }
        )
        + b"\n",
        corruption_manifest_json=phase7_corruption_manifest_to_json(manifest),
        transform_results_json=build_phase7_transform_results_collection_json((transform,)),
        geometry_records_json=build_phase7_geometry_records_collection_json((geometry,)),
        uncertainty_result_json=phase7_uncertainty_result_to_json(uncertainty),
        calibration_result_json=phase7_calibration_result_to_json(calibration),
        risk_coverage_result_json=phase7_risk_coverage_result_to_json(risk),
        failure_detection_json=canonical_phase7_failure_detection_results_json(
            correlation_result=correlation,
            auroc_result=auroc,
        ),
        degradation_result_json=phase7_degradation_result_to_json(degradation),
        lesion_subgroup_result_json=phase7_lesion_subgroup_result_to_json(subgroup),
        phase7_run_summary_json=phase7_run_summary_to_json(summary),
    )


def _corruption_specification() -> Phase7CorruptionSpecification:
    payload = {
        "changes_geometry": False,
        "common_grid_restoration_required": False,
        "corruption_name": "gaussian_noise",
        "deterministic_seed": 1729,
        "image_interpolation": "linear",
        "mask_interpolation": "nearest_neighbor",
        "parameters": {"sigma_hu": 5.0},
        "schema_name": PHASE7_CORRUPTION_SPECIFICATION_SCHEMA_NAME,
        "schema_version": PHASE7_CORRUPTION_SPECIFICATION_SCHEMA_VERSION,
        "severity": "low",
    }
    return phase7_corruption_specification_from_json(
        canonical_json_bytes({"corruption_specification_hash": sha256_json(payload), **payload})
    )


def _manifest() -> Phase7CorruptionManifest:
    specification = _corruption_specification()
    payload = {
        "config_hash": HEX_1,
        "deterministic_seed": 1729,
        "input_image_content_hash": HEX_2,
        "input_mask_content_hash": HEX_3,
        "manifest_id": "phase7_synthetic_manifest",
        "schema_name": PHASE7_CORRUPTION_MANIFEST_SCHEMA_NAME,
        "schema_version": PHASE7_CORRUPTION_MANIFEST_SCHEMA_VERSION,
        "specifications": [phase7_corruption_specification_to_dict(specification)],
    }
    return phase7_corruption_manifest_from_json(
        canonical_json_bytes({"corruption_manifest_hash": sha256_json(payload), **payload})
    )


def _geometry_record() -> Phase7GeometryRecord:
    payload = {
        "binary_mask_preserved": True,
        "common_grid_restored": True,
        "geometry_change_type": "resampling",
        "image_interpolation": "linear",
        "input_affine_hash": HEX_1,
        "input_orientation": "RAS",
        "input_shape": (4, 5, 6),
        "input_spacing": (1.0, 1.0, 1.0),
        "mask_interpolation": "nearest_neighbor",
        "output_affine_hash": HEX_2,
        "output_orientation": "RAS",
        "output_shape": (4, 5, 3),
        "output_spacing": (1.0, 1.0, 2.0),
        "restored_affine_hash": HEX_3,
        "restored_orientation": "RAS",
        "restored_shape": (4, 5, 6),
        "restored_spacing": (1.0, 1.0, 1.0),
        "schema_name": PHASE7_GEOMETRY_RECORD_SCHEMA_NAME,
        "schema_version": PHASE7_GEOMETRY_RECORD_SCHEMA_VERSION,
        "transform_name": "slice_thickness",
    }
    json_payload = _json_shape(payload)
    return phase7_geometry_record_from_json(
        canonical_json_bytes({"geometry_record_hash": sha256_json(json_payload), **json_payload})
    )


def _transform_result() -> Phase7TransformResult:
    payload = {
        "common_grid_restored": True,
        "corruption_specification_hash": HEX_1,
        "execution_status": "completed",
        "failure_code": None,
        "failure_message": None,
        "finite_output": True,
        "geometry_record_hash": HEX_6,
        "input_image_content_hash": HEX_2,
        "input_mask_content_hash": HEX_3,
        "mask_binary_preserved": True,
        "output_image_content_hash": HEX_4,
        "output_mask_content_hash": HEX_5,
        "schema_name": PHASE7_TRANSFORM_RESULT_SCHEMA_NAME,
        "schema_version": PHASE7_TRANSFORM_RESULT_SCHEMA_VERSION,
    }
    return phase7_transform_result_from_json(
        canonical_json_bytes({"transform_result_hash": sha256_json(payload), **payload})
    )


def _uncertainty_result() -> Phase7UncertaintyResult:
    payload = {
        "availability_status": "available",
        "log_base": "natural",
        "max_uncertainty": 0.5,
        "mean_uncertainty": 0.25,
        "probability_space": "binary_foreground",
        "sample_count": 1,
        "schema_name": PHASE7_UNCERTAINTY_RESULT_SCHEMA_NAME,
        "schema_version": PHASE7_UNCERTAINTY_RESULT_SCHEMA_VERSION,
        "source_prediction_hash": HEX_1,
        "source_probability_hash": HEX_2,
        "uncertainty_map_content_hash": HEX_3,
        "uncertainty_type": "predictive_entropy",
        "unavailable_reason": None,
        "variance_map_content_hash": None,
    }
    return phase7_uncertainty_result_from_json(
        canonical_json_bytes({"uncertainty_result_hash": sha256_json(payload), **payload})
    )


def _calibration_result() -> Phase7CalibrationResult:
    bins = (
        Phase7CalibrationBin(
            schema_name=PHASE7_CALIBRATION_BIN_SCHEMA_NAME,
            schema_version=PHASE7_CALIBRATION_BIN_SCHEMA_VERSION,
            bin_index=0,
            bin_lower=0.0,
            bin_upper=0.5,
            upper_inclusive=False,
            voxel_count=0,
            accuracy=None,
            mean_confidence=None,
            weighted_error=0.0,
        ),
        Phase7CalibrationBin(
            schema_name=PHASE7_CALIBRATION_BIN_SCHEMA_NAME,
            schema_version=PHASE7_CALIBRATION_BIN_SCHEMA_VERSION,
            bin_index=1,
            bin_lower=0.5,
            bin_upper=1.0,
            upper_inclusive=True,
            voxel_count=4,
            accuracy=0.75,
            mean_confidence=0.8,
            weighted_error=0.05,
        ),
    )
    payload = {
        "availability_status": "available",
        "bin_count": 2,
        "binning_policy": "fixed_equal_width",
        "bins": [phase7_calibration_bin_to_dict(item) for item in bins],
        "calibration_metric": "ece",
        "common_grid_geometry_record_hash": HEX_4,
        "confidence_definition": "max_binary_probability",
        "ece": 0.05,
        "prediction_content_hash": HEX_1,
        "probability_content_hash": HEX_2,
        "reference_mask_content_hash": HEX_3,
        "schema_name": PHASE7_CALIBRATION_RESULT_SCHEMA_NAME,
        "schema_version": PHASE7_CALIBRATION_RESULT_SCHEMA_VERSION,
        "unavailable_reason": None,
        "valid_voxel_count": 4,
    }
    return phase7_calibration_result_from_json(
        canonical_json_bytes({"calibration_result_hash": sha256_json(payload), **payload})
    )


def _risk_coverage_result(*, uncertainty_hash: str) -> Phase7RiskCoverageResult:
    points = (
        Phase7RiskCoveragePoint(
            schema_name=PHASE7_RISK_COVERAGE_POINT_SCHEMA_NAME,
            schema_version=PHASE7_RISK_COVERAGE_POINT_SCHEMA_VERSION,
            retained_voxel_count=1,
            coverage=0.25,
            risk=0.0,
        ),
        Phase7RiskCoveragePoint(
            schema_name=PHASE7_RISK_COVERAGE_POINT_SCHEMA_NAME,
            schema_version=PHASE7_RISK_COVERAGE_POINT_SCHEMA_VERSION,
            retained_voxel_count=4,
            coverage=1.0,
            risk=0.25,
        ),
    )
    payload = {
        "availability_status": "available",
        "common_grid_geometry_record_hash": HEX_4,
        "ordering": "low_uncertainty_first",
        "points": [phase7_risk_coverage_point_to_dict(item) for item in points],
        "prediction_content_hash": HEX_2,
        "reference_mask_content_hash": HEX_3,
        "schema_name": PHASE7_RISK_COVERAGE_RESULT_SCHEMA_NAME,
        "schema_version": PHASE7_RISK_COVERAGE_RESULT_SCHEMA_VERSION,
        "tie_break_policy": "flattened_index",
        "uncertainty_result_hash": uncertainty_hash,
        "unavailable_reason": None,
        "valid_voxel_count": 4,
    }
    return phase7_risk_coverage_result_from_json(
        canonical_json_bytes({"risk_coverage_result_hash": sha256_json(payload), **payload})
    )


def _degradation_result() -> Phase7DegradationResult:
    payload = {
        "absolute_degradation": 0.2,
        "availability_status": "available",
        "baseline_artifact_hash": HEX_1,
        "baseline_value": 0.8,
        "corrupted_artifact_hash": HEX_2,
        "corrupted_value": 0.6,
        "metric_direction": "higher_is_better",
        "metric_name": "dice",
        "relative_degradation": 0.25,
        "relative_epsilon": 1e-8,
        "schema_name": PHASE7_DEGRADATION_RESULT_SCHEMA_NAME,
        "schema_version": PHASE7_DEGRADATION_RESULT_SCHEMA_VERSION,
        "unavailable_reason": None,
    }
    return phase7_degradation_result_from_json(
        canonical_json_bytes({"degradation_result_hash": sha256_json(payload), **payload})
    )


def _lesion_subgroup_result() -> Phase7LesionSubgroupResult:
    record = Phase7LesionSubgroupRecord(
        schema_name=PHASE7_LESION_SUBGROUP_RECORD_SCHEMA_NAME,
        schema_version=PHASE7_LESION_SUBGROUP_RECORD_SCHEMA_VERSION,
        subgroup_name="small",
        eligible_case_count=2,
        empty_lesion_case_count=0,
        skipped_case_count=0,
        metric_availability_status="available",
        mean_metric_value=0.7,
        unavailable_reason=None,
    )
    payload = {
        "metric_name": "dice",
        "prediction_content_hash": HEX_2,
        "records": [phase7_lesion_subgroup_record_to_dict(record)],
        "reference_mask_content_hash": HEX_1,
        "schema_name": PHASE7_LESION_SUBGROUP_RESULT_SCHEMA_NAME,
        "schema_version": PHASE7_LESION_SUBGROUP_RESULT_SCHEMA_VERSION,
        "subgroup_policy_name": "voxel_count_v1",
        "thresholds_voxels": [0, 10, 100],
    }
    return phase7_lesion_subgroup_result_from_json(
        canonical_json_bytes({"lesion_subgroup_result_hash": sha256_json(payload), **payload})
    )


def _run_summary(
    *,
    manifest_hash: str,
    uncertainty_hash: str,
    calibration_hash: str,
    risk_hash: str,
    degradation_hash: str,
    subgroup_hash: str,
) -> Phase7RunSummary:
    payload = {
        "calibration_result_hash": calibration_hash,
        "config_hash": HEX_1,
        "corruption_manifest_hash": manifest_hash,
        "degradation_result_hash": degradation_hash,
        "execution_status": "completed",
        "failure_code": None,
        "failure_message": None,
        "lesion_subgroup_result_hash": subgroup_hash,
        "phase6_run_summary_hash": HEX_3,
        "risk_coverage_result_hash": risk_hash,
        "schema_name": PHASE7_RUN_SUMMARY_SCHEMA_NAME,
        "schema_version": PHASE7_RUN_SUMMARY_SCHEMA_VERSION,
        "uncertainty_result_hash": uncertainty_hash,
    }
    return phase7_run_summary_from_json(
        canonical_json_bytes(
            {
                "duration_seconds": None,
                "memory_availability_status": "unavailable",
                "peak_host_memory_bytes": None,
                "phase7_run_summary_hash": sha256_json(payload),
                **payload,
            }
        )
    )


def _run_summary_json_from_mapping(mapping: dict[str, object]) -> bytes:
    identity_payload = {
        key: value
        for key, value in mapping.items()
        if key
        not in {
            "phase7_run_summary_hash",
            "duration_seconds",
            "memory_availability_status",
            "peak_host_memory_bytes",
        }
    }
    mapping["phase7_run_summary_hash"] = sha256_json(identity_payload)
    return canonical_json_bytes(mapping) + b"\n"


def _json_shape(mapping: dict[str, object]) -> dict[str, object]:
    shaped = dict(mapping)
    for field in ("input_shape", "output_shape", "restored_shape"):
        value = shaped[field]
        shaped[field] = list(value) if isinstance(value, tuple) else value
    for field in ("input_spacing", "output_spacing", "restored_spacing"):
        value = shaped[field]
        shaped[field] = list(value) if isinstance(value, tuple) else value
    return shaped
