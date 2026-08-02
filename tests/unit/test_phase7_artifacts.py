"""Unit tests for Phase 7 artifact schemas and canonical hashes."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from protoem_ct.artifacts.hashing import HashInputError, canonical_json_bytes, sha256_json
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
    Phase7ArtifactHashError,
    Phase7ArtifactSerializationError,
    Phase7ArtifactValidationError,
    Phase7CorruptionManifest,
    Phase7CorruptionSpecification,
    Phase7GeometryRecord,
    Phase7RunSummary,
    Phase7TransformResult,
    phase7_corruption_manifest_from_json,
    phase7_corruption_manifest_identity_payload,
    phase7_corruption_manifest_to_dict,
    phase7_corruption_manifest_to_json,
    phase7_corruption_specification_from_json,
    phase7_corruption_specification_to_dict,
    phase7_corruption_specification_to_json,
    phase7_geometry_record_from_json,
    phase7_geometry_record_identity_payload,
    phase7_geometry_record_to_dict,
    phase7_geometry_record_to_json,
    phase7_run_summary_from_json,
    phase7_run_summary_identity_payload,
    phase7_run_summary_to_dict,
    phase7_run_summary_to_json,
    phase7_transform_result_from_json,
    phase7_transform_result_identity_payload,
    phase7_transform_result_to_dict,
    phase7_transform_result_to_json,
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
    Phase7UncertaintyArtifactHashError,
    Phase7UncertaintyArtifactSerializationError,
    Phase7UncertaintyArtifactValidationError,
    Phase7UncertaintyResult,
    phase7_calibration_result_from_json,
    phase7_calibration_result_identity_payload,
    phase7_calibration_result_to_dict,
    phase7_calibration_result_to_json,
    phase7_degradation_result_from_json,
    phase7_degradation_result_identity_payload,
    phase7_degradation_result_to_dict,
    phase7_degradation_result_to_json,
    phase7_lesion_subgroup_result_from_json,
    phase7_lesion_subgroup_result_identity_payload,
    phase7_lesion_subgroup_result_to_dict,
    phase7_lesion_subgroup_result_to_json,
    phase7_risk_coverage_result_from_json,
    phase7_risk_coverage_result_identity_payload,
    phase7_risk_coverage_result_to_dict,
    phase7_risk_coverage_result_to_json,
    phase7_uncertainty_result_from_json,
    phase7_uncertainty_result_identity_payload,
    phase7_uncertainty_result_to_dict,
    phase7_uncertainty_result_to_json,
)

HEX_1 = "1" * 64
HEX_2 = "2" * 64
HEX_3 = "3" * 64
HEX_4 = "4" * 64
HEX_5 = "5" * 64
HEX_6 = "6" * 64
HEX_7 = "7" * 64
HEX_8 = "8" * 64
HEX_9 = "9" * 64
HEX_A = "a" * 64
HEX_B = "b" * 64
HEX_C = "c" * 64
HEX_D = "d" * 64


def _corruption_specification_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_name": PHASE7_CORRUPTION_SPECIFICATION_SCHEMA_NAME,
        "schema_version": PHASE7_CORRUPTION_SPECIFICATION_SCHEMA_VERSION,
        "corruption_name": "gaussian_noise",
        "severity": "low",
        "parameters": {"sigma_hu": 5.0},
        "deterministic_seed": 1729,
        "changes_geometry": False,
        "image_interpolation": "linear",
        "mask_interpolation": "nearest_neighbor",
        "common_grid_restoration_required": False,
    }
    payload.update(overrides)
    return payload


def _corruption_specification(**overrides: object) -> Phase7CorruptionSpecification:
    payload = _corruption_specification_payload(**overrides)
    return phase7_corruption_specification_from_json(
        canonical_json_bytes({"corruption_specification_hash": sha256_json(payload), **payload})
    )


def _manifest_payload(**overrides: object) -> dict[str, object]:
    spec = _corruption_specification()
    payload: dict[str, object] = {
        "schema_name": PHASE7_CORRUPTION_MANIFEST_SCHEMA_NAME,
        "schema_version": PHASE7_CORRUPTION_MANIFEST_SCHEMA_VERSION,
        "manifest_id": "phase7_synthetic_manifest",
        "config_hash": HEX_1,
        "input_image_content_hash": HEX_2,
        "input_mask_content_hash": HEX_3,
        "deterministic_seed": 1729,
        "specifications": [phase7_corruption_specification_to_dict(spec)],
    }
    payload.update(overrides)
    return payload


def _manifest(**overrides: object) -> Phase7CorruptionManifest:
    payload = _manifest_payload(**overrides)
    return phase7_corruption_manifest_from_json(
        canonical_json_bytes({"corruption_manifest_hash": sha256_json(payload), **payload})
    )


def _geometry_record_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_name": PHASE7_GEOMETRY_RECORD_SCHEMA_NAME,
        "schema_version": PHASE7_GEOMETRY_RECORD_SCHEMA_VERSION,
        "transform_name": "slice_thickness",
        "geometry_change_type": "resampling",
        "input_shape": [4, 5, 6],
        "output_shape": [4, 5, 3],
        "restored_shape": [4, 5, 6],
        "input_spacing": [1.0, 1.0, 1.0],
        "output_spacing": [1.0, 1.0, 2.0],
        "restored_spacing": [1.0, 1.0, 1.0],
        "input_orientation": "RAS",
        "output_orientation": "RAS",
        "restored_orientation": "RAS",
        "input_affine_hash": HEX_1,
        "output_affine_hash": HEX_2,
        "restored_affine_hash": HEX_3,
        "image_interpolation": "linear",
        "mask_interpolation": "nearest_neighbor",
        "binary_mask_preserved": True,
        "common_grid_restored": True,
    }
    payload.update(overrides)
    return payload


def _geometry_record(**overrides: object) -> Phase7GeometryRecord:
    payload = _geometry_record_payload(**overrides)
    return phase7_geometry_record_from_json(
        canonical_json_bytes({"geometry_record_hash": sha256_json(payload), **payload})
    )


def _transform_result_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_name": PHASE7_TRANSFORM_RESULT_SCHEMA_NAME,
        "schema_version": PHASE7_TRANSFORM_RESULT_SCHEMA_VERSION,
        "corruption_specification_hash": HEX_1,
        "input_image_content_hash": HEX_2,
        "input_mask_content_hash": HEX_3,
        "output_image_content_hash": HEX_4,
        "output_mask_content_hash": HEX_5,
        "geometry_record_hash": HEX_6,
        "finite_output": True,
        "mask_binary_preserved": True,
        "common_grid_restored": True,
        "execution_status": "completed",
        "failure_code": None,
        "failure_message": None,
    }
    payload.update(overrides)
    return payload


def _transform_result(**overrides: object) -> Phase7TransformResult:
    payload = _transform_result_payload(**overrides)
    return phase7_transform_result_from_json(
        canonical_json_bytes({"transform_result_hash": sha256_json(payload), **payload})
    )


def _uncertainty_result_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_name": PHASE7_UNCERTAINTY_RESULT_SCHEMA_NAME,
        "schema_version": PHASE7_UNCERTAINTY_RESULT_SCHEMA_VERSION,
        "source_prediction_hash": HEX_1,
        "source_probability_hash": HEX_2,
        "uncertainty_type": "predictive_entropy",
        "probability_space": "binary_foreground",
        "log_base": "natural",
        "sample_count": 1,
        "uncertainty_map_content_hash": HEX_3,
        "variance_map_content_hash": None,
        "mean_uncertainty": 0.25,
        "max_uncertainty": 0.5,
        "availability_status": "available",
        "unavailable_reason": None,
    }
    payload.update(overrides)
    return payload


def _uncertainty_result(**overrides: object) -> Phase7UncertaintyResult:
    payload = _uncertainty_result_payload(**overrides)
    return phase7_uncertainty_result_from_json(
        canonical_json_bytes({"uncertainty_result_hash": sha256_json(payload), **payload})
    )


def _calibration_bins() -> tuple[Phase7CalibrationBin, ...]:
    return (
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


def _calibration_result_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_name": PHASE7_CALIBRATION_RESULT_SCHEMA_NAME,
        "schema_version": PHASE7_CALIBRATION_RESULT_SCHEMA_VERSION,
        "prediction_content_hash": HEX_1,
        "probability_content_hash": HEX_2,
        "reference_mask_content_hash": HEX_3,
        "common_grid_geometry_record_hash": HEX_4,
        "calibration_metric": "ece",
        "confidence_definition": "max_binary_probability",
        "binning_policy": "fixed_equal_width",
        "bin_count": 2,
        "valid_voxel_count": 4,
        "ece": 0.05,
        "bins": [phase7_calibration_bin_to_dict(item) for item in _calibration_bins()],
        "availability_status": "available",
        "unavailable_reason": None,
    }
    payload.update(overrides)
    return payload


def _calibration_result(**overrides: object) -> Phase7CalibrationResult:
    payload = _calibration_result_payload(**overrides)
    return phase7_calibration_result_from_json(
        canonical_json_bytes({"calibration_result_hash": sha256_json(payload), **payload})
    )


def phase7_calibration_bin_to_dict(bin_record: Phase7CalibrationBin) -> dict[str, object]:
    return {
        "schema_name": bin_record.schema_name,
        "schema_version": bin_record.schema_version,
        "bin_index": bin_record.bin_index,
        "bin_lower": bin_record.bin_lower,
        "bin_upper": bin_record.bin_upper,
        "upper_inclusive": bin_record.upper_inclusive,
        "voxel_count": bin_record.voxel_count,
        "accuracy": bin_record.accuracy,
        "mean_confidence": bin_record.mean_confidence,
        "weighted_error": bin_record.weighted_error,
    }


def _risk_coverage_point() -> Phase7RiskCoveragePoint:
    return Phase7RiskCoveragePoint(
        schema_name=PHASE7_RISK_COVERAGE_POINT_SCHEMA_NAME,
        schema_version=PHASE7_RISK_COVERAGE_POINT_SCHEMA_VERSION,
        retained_voxel_count=4,
        coverage=1.0,
        risk=0.25,
    )


def _risk_coverage_result_payload(**overrides: object) -> dict[str, object]:
    point = _risk_coverage_point()
    payload: dict[str, object] = {
        "schema_name": PHASE7_RISK_COVERAGE_RESULT_SCHEMA_NAME,
        "schema_version": PHASE7_RISK_COVERAGE_RESULT_SCHEMA_VERSION,
        "uncertainty_result_hash": HEX_1,
        "prediction_content_hash": HEX_2,
        "reference_mask_content_hash": HEX_3,
        "common_grid_geometry_record_hash": HEX_4,
        "ordering": "low_uncertainty_first",
        "tie_break_policy": "flattened_index",
        "valid_voxel_count": 4,
        "points": [
            {
                "schema_name": point.schema_name,
                "schema_version": point.schema_version,
                "retained_voxel_count": point.retained_voxel_count,
                "coverage": point.coverage,
                "risk": point.risk,
            }
        ],
        "availability_status": "available",
        "unavailable_reason": None,
    }
    payload.update(overrides)
    return payload


def _risk_coverage_result(**overrides: object) -> Phase7RiskCoverageResult:
    payload = _risk_coverage_result_payload(**overrides)
    return phase7_risk_coverage_result_from_json(
        canonical_json_bytes({"risk_coverage_result_hash": sha256_json(payload), **payload})
    )


def _degradation_result_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_name": PHASE7_DEGRADATION_RESULT_SCHEMA_NAME,
        "schema_version": PHASE7_DEGRADATION_RESULT_SCHEMA_VERSION,
        "baseline_artifact_hash": HEX_1,
        "corrupted_artifact_hash": HEX_2,
        "metric_name": "dice",
        "metric_direction": "higher_is_better",
        "baseline_value": 0.8,
        "corrupted_value": 0.6,
        "absolute_degradation": 0.2,
        "relative_degradation": 0.25,
        "relative_epsilon": 1e-8,
        "availability_status": "available",
        "unavailable_reason": None,
    }
    payload.update(overrides)
    return payload


def _degradation_result(**overrides: object) -> Phase7DegradationResult:
    payload = _degradation_result_payload(**overrides)
    return phase7_degradation_result_from_json(
        canonical_json_bytes({"degradation_result_hash": sha256_json(payload), **payload})
    )


def _lesion_subgroup_record(name: str = "small") -> Phase7LesionSubgroupRecord:
    return Phase7LesionSubgroupRecord(
        schema_name=PHASE7_LESION_SUBGROUP_RECORD_SCHEMA_NAME,
        schema_version=PHASE7_LESION_SUBGROUP_RECORD_SCHEMA_VERSION,
        subgroup_name=name,
        eligible_case_count=2,
        empty_lesion_case_count=0,
        skipped_case_count=0,
        metric_availability_status="available",
        mean_metric_value=0.7,
        unavailable_reason=None,
    )


def _lesion_subgroup_result_payload(**overrides: object) -> dict[str, object]:
    record = _lesion_subgroup_record()
    payload: dict[str, object] = {
        "schema_name": PHASE7_LESION_SUBGROUP_RESULT_SCHEMA_NAME,
        "schema_version": PHASE7_LESION_SUBGROUP_RESULT_SCHEMA_VERSION,
        "reference_mask_content_hash": HEX_1,
        "prediction_content_hash": HEX_2,
        "metric_name": "dice",
        "subgroup_policy_name": "voxel_count_v1",
        "thresholds_voxels": [0, 10, 100],
        "records": [
            {
                "schema_name": record.schema_name,
                "schema_version": record.schema_version,
                "subgroup_name": record.subgroup_name,
                "eligible_case_count": record.eligible_case_count,
                "empty_lesion_case_count": record.empty_lesion_case_count,
                "skipped_case_count": record.skipped_case_count,
                "metric_availability_status": record.metric_availability_status,
                "mean_metric_value": record.mean_metric_value,
                "unavailable_reason": record.unavailable_reason,
            }
        ],
    }
    payload.update(overrides)
    return payload


def _lesion_subgroup_result(**overrides: object) -> Phase7LesionSubgroupResult:
    payload = _lesion_subgroup_result_payload(**overrides)
    return phase7_lesion_subgroup_result_from_json(
        canonical_json_bytes({"lesion_subgroup_result_hash": sha256_json(payload), **payload})
    )


def _run_summary_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_name": PHASE7_RUN_SUMMARY_SCHEMA_NAME,
        "schema_version": PHASE7_RUN_SUMMARY_SCHEMA_VERSION,
        "config_hash": HEX_1,
        "corruption_manifest_hash": HEX_2,
        "phase6_run_summary_hash": HEX_3,
        "uncertainty_result_hash": HEX_4,
        "calibration_result_hash": HEX_5,
        "risk_coverage_result_hash": HEX_6,
        "degradation_result_hash": HEX_7,
        "lesion_subgroup_result_hash": HEX_8,
        "execution_status": "completed",
        "failure_code": None,
        "failure_message": None,
    }
    payload.update(overrides)
    return payload


def _run_summary(**overrides: object) -> Phase7RunSummary:
    payload = _run_summary_payload(**overrides)
    return phase7_run_summary_from_json(
        canonical_json_bytes(
            {
                "phase7_run_summary_hash": sha256_json(payload),
                **payload,
                "duration_seconds": None,
                "memory_availability_status": "unavailable",
                "peak_host_memory_bytes": None,
            }
        )
    )


RoundTripCase = tuple[
    object,
    Callable[[object], dict[str, object]],
    Callable[[object], bytes],
    Callable[[bytes | str], object],
]


@pytest.mark.parametrize(
    ("artifact", "to_dict", "to_json", "from_json"),
    [
        (
            _corruption_specification(),
            phase7_corruption_specification_to_dict,
            phase7_corruption_specification_to_json,
            phase7_corruption_specification_from_json,
        ),
        (
            _manifest(),
            phase7_corruption_manifest_to_dict,
            phase7_corruption_manifest_to_json,
            phase7_corruption_manifest_from_json,
        ),
        (
            _geometry_record(),
            phase7_geometry_record_to_dict,
            phase7_geometry_record_to_json,
            phase7_geometry_record_from_json,
        ),
        (
            _transform_result(),
            phase7_transform_result_to_dict,
            phase7_transform_result_to_json,
            phase7_transform_result_from_json,
        ),
        (
            _uncertainty_result(),
            phase7_uncertainty_result_to_dict,
            phase7_uncertainty_result_to_json,
            phase7_uncertainty_result_from_json,
        ),
        (
            _calibration_result(),
            phase7_calibration_result_to_dict,
            phase7_calibration_result_to_json,
            phase7_calibration_result_from_json,
        ),
        (
            _risk_coverage_result(),
            phase7_risk_coverage_result_to_dict,
            phase7_risk_coverage_result_to_json,
            phase7_risk_coverage_result_from_json,
        ),
        (
            _degradation_result(),
            phase7_degradation_result_to_dict,
            phase7_degradation_result_to_json,
            phase7_degradation_result_from_json,
        ),
        (
            _lesion_subgroup_result(),
            phase7_lesion_subgroup_result_to_dict,
            phase7_lesion_subgroup_result_to_json,
            phase7_lesion_subgroup_result_from_json,
        ),
        (
            _run_summary(),
            phase7_run_summary_to_dict,
            phase7_run_summary_to_json,
            phase7_run_summary_from_json,
        ),
    ],
)
def test_round_trip_and_byte_identical_canonical_serialization(
    artifact: object,
    to_dict: Callable[[object], dict[str, object]],
    to_json: Callable[[object], bytes],
    from_json: Callable[[bytes | str], object],
) -> None:
    encoded = to_json(artifact)
    reconstructed = from_json(encoded)

    assert reconstructed == artifact
    assert to_json(reconstructed) == encoded
    assert encoded == canonical_json_bytes(to_dict(artifact)) + b"\n"


@pytest.mark.parametrize(
    ("payload", "hash_field", "from_json", "error_type"),
    [
        (
            _corruption_specification_payload(),
            "corruption_specification_hash",
            phase7_corruption_specification_from_json,
            Phase7ArtifactSerializationError,
        ),
        (
            _uncertainty_result_payload(),
            "uncertainty_result_hash",
            phase7_uncertainty_result_from_json,
            Phase7UncertaintyArtifactSerializationError,
        ),
    ],
)
def test_unknown_field_rejected(
    payload: dict[str, object],
    hash_field: str,
    from_json: Callable[[bytes], object],
    error_type: type[Exception],
) -> None:
    mapping = {hash_field: sha256_json(payload), **payload, "unexpected": True}

    with pytest.raises(error_type):
        from_json(canonical_json_bytes(mapping))


def test_invalid_schema_and_version_rejected() -> None:
    with pytest.raises(Phase7ArtifactValidationError):
        _corruption_specification(schema_name="wrong_schema")

    with pytest.raises(Phase7UncertaintyArtifactValidationError):
        _uncertainty_result(schema_version="v999")


def test_self_hash_mismatch_rejected() -> None:
    payload = _transform_result_payload()
    with pytest.raises(Phase7ArtifactHashError):
        phase7_transform_result_from_json(
            canonical_json_bytes({"transform_result_hash": HEX_A, **payload})
        )

    payload_u = _degradation_result_payload()
    with pytest.raises(Phase7UncertaintyArtifactHashError):
        phase7_degradation_result_from_json(
            canonical_json_bytes({"degradation_result_hash": HEX_A, **payload_u})
        )


def test_nan_and_infinity_rejected() -> None:
    with pytest.raises((Phase7ArtifactValidationError, HashInputError)):
        _corruption_specification(parameters={"sigma_hu": float("nan")})

    with pytest.raises((Phase7UncertaintyArtifactValidationError, HashInputError)):
        _uncertainty_result(mean_uncertainty=float("inf"))


def test_sha_format_validation() -> None:
    with pytest.raises(Phase7ArtifactValidationError):
        _manifest(config_hash="ABC")

    with pytest.raises(Phase7UncertaintyArtifactValidationError):
        _calibration_result(prediction_content_hash="not-a-sha")


def test_identity_hash_changes_when_critical_fields_change() -> None:
    base_spec = _corruption_specification()
    shifted_spec = _corruption_specification(parameters={"sigma_hu": 8.0})
    assert base_spec.corruption_specification_hash != shifted_spec.corruption_specification_hash

    base_uncertainty = _uncertainty_result()
    changed_uncertainty = _uncertainty_result(log_base="natural", sample_count=2)
    assert base_uncertainty.uncertainty_result_hash != changed_uncertainty.uncertainty_result_hash

    base_degradation = _degradation_result()
    changed_degradation = _degradation_result(
        corrupted_value=0.5, absolute_degradation=0.3, relative_degradation=0.37499999999999994
    )
    assert base_degradation.degradation_result_hash != changed_degradation.degradation_result_hash


def test_mask_geometry_safety_validation() -> None:
    with pytest.raises(Phase7ArtifactValidationError):
        _geometry_record(mask_interpolation="linear")

    with pytest.raises(Phase7ArtifactValidationError):
        _geometry_record(binary_mask_preserved=False)

    with pytest.raises(Phase7ArtifactValidationError):
        _geometry_record(geometry_change_type="resampling", common_grid_restored=False)


def test_availability_and_degenerate_case_consistency() -> None:
    unavailable_uncertainty = _uncertainty_result(
        uncertainty_map_content_hash=None,
        mean_uncertainty=None,
        max_uncertainty=None,
        availability_status="unavailable",
        unavailable_reason="insufficient_samples",
    )
    assert unavailable_uncertainty.availability_status == "unavailable"

    with pytest.raises(Phase7UncertaintyArtifactValidationError):
        _risk_coverage_result(
            points=[],
            availability_status="available",
        )

    near_zero = _degradation_result(
        baseline_value=0.0,
        corrupted_value=0.0,
        absolute_degradation=0.0,
        relative_degradation=None,
    )
    assert near_zero.relative_degradation is None


def test_identity_payloads_exclude_runtime_and_local_environment_fields() -> None:
    payloads = [
        phase7_corruption_manifest_identity_payload(_manifest()),
        phase7_geometry_record_identity_payload(_geometry_record()),
        phase7_transform_result_identity_payload(_transform_result()),
        phase7_uncertainty_result_identity_payload(_uncertainty_result()),
        phase7_calibration_result_identity_payload(_calibration_result()),
        phase7_risk_coverage_result_identity_payload(_risk_coverage_result()),
        phase7_degradation_result_identity_payload(_degradation_result()),
        phase7_lesion_subgroup_result_identity_payload(_lesion_subgroup_result()),
        phase7_run_summary_identity_payload(_run_summary()),
    ]
    forbidden = {"timestamp", "hostname", "hardware", "runtime", "duration_seconds", "path"}

    for payload in payloads:
        assert forbidden.isdisjoint(payload)
