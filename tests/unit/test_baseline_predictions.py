from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from typing import Any, cast

import pytest

from protoem_ct.baselines.predictions import (
    BASELINE_PREDICTION_FOREGROUND_LABEL,
    BASELINE_PREDICTION_MANIFEST_VERSION,
    BASELINE_PREDICTION_RECORD_VERSION,
    BaselinePredictionHashError,
    BaselinePredictionManifest,
    BaselinePredictionRecord,
    BaselinePredictionSerializationError,
    BaselinePredictionValidationError,
    BaselinePredictionVersionError,
    baseline_prediction_manifest_from_json,
    baseline_prediction_manifest_to_json,
    build_baseline_prediction_manifest,
)


def _record(
    *,
    case_identifier: str = "case_001",
    prediction_path: str = "case_001.nii.gz",
) -> BaselinePredictionRecord:
    return BaselinePredictionRecord(
        contract_version=BASELINE_PREDICTION_RECORD_VERSION,
        case_identifier=case_identifier,
        prediction_path=prediction_path,
        prediction_sha256="a" * 64,
        byte_size=123,
        shape=(24, 24, 16),
        voxel_spacing_mm=(1.0, 1.0, 2.5),
        affine=(
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 2.5, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        foreground_label=BASELINE_PREDICTION_FOREGROUND_LABEL,
    )


def _manifest(*, baseline_family: str = "nnunet_v2") -> BaselinePredictionManifest:
    return build_baseline_prediction_manifest(
        baseline_family=baseline_family,
        run_identifier="run_001",
        dataset_manifest_sha256="1" * 64,
        development_split_sha256="2" * 64,
        checkpoint_sha256="3" * 64,
        prediction_records=(_record(case_identifier="case_a", prediction_path="case_a.nii.gz"),),
    )


def test_valid_manifests_for_both_baseline_families() -> None:
    assert _manifest(baseline_family="nnunet_v2").baseline_family == "nnunet_v2"
    assert _manifest(baseline_family="monai_segresnet").baseline_family == "monai_segresnet"


def test_deterministic_ordering_and_json_bytes() -> None:
    manifest = build_baseline_prediction_manifest(
        baseline_family="nnunet_v2",
        run_identifier="run_001",
        dataset_manifest_sha256="1" * 64,
        development_split_sha256="2" * 64,
        checkpoint_sha256="3" * 64,
        prediction_records=(
            _record(case_identifier="case_b", prediction_path="case_b.nii.gz"),
            _record(case_identifier="case_a", prediction_path="case_a.nii.gz"),
        ),
    )

    assert tuple(record.case_identifier for record in manifest.prediction_records) == (
        "case_a",
        "case_b",
    )
    assert baseline_prediction_manifest_to_json(manifest) == baseline_prediction_manifest_to_json(
        manifest
    )


def test_hash_verification_and_tamper_rejection() -> None:
    manifest = _manifest()
    parsed = baseline_prediction_manifest_from_json(baseline_prediction_manifest_to_json(manifest))

    assert parsed.artifact_hash == manifest.artifact_hash

    payload = json.loads(baseline_prediction_manifest_to_json(manifest))
    payload["checkpoint_sha256"] = "4" * 64
    with pytest.raises(BaselinePredictionHashError):
        baseline_prediction_manifest_from_json(json.dumps(payload).encode("utf-8"))


def test_invalid_hash_shape_spacing_affine_path_extension_duplicate_and_unsorted_records() -> None:
    with pytest.raises(BaselinePredictionValidationError):
        BaselinePredictionRecord(
            contract_version=BASELINE_PREDICTION_RECORD_VERSION,
            case_identifier="case_001",
            prediction_path="case_001.nii.gz",
            prediction_sha256="bad",
            byte_size=1,
            shape=(24, 24, 16),
            voxel_spacing_mm=(1.0, 1.0, 2.5),
            affine=((1.0, 0.0, 0.0, 0.0),) * 4,
        )
    with pytest.raises(BaselinePredictionValidationError):
        cast(Any, replace)(_record(), shape=(24, 24, 0))
    with pytest.raises(BaselinePredictionValidationError):
        cast(Any, replace)(_record(), voxel_spacing_mm=(1.0, 0.0, 2.5))
    with pytest.raises(BaselinePredictionValidationError):
        cast(Any, replace)(
            _record(),
            affine=((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0)),
        )
    with pytest.raises(BaselinePredictionValidationError):
        cast(Any, replace)(_record(), prediction_path="case_001.nii")
    with pytest.raises(BaselinePredictionValidationError):
        build_baseline_prediction_manifest(
            baseline_family="nnunet_v2",
            run_identifier="run_001",
            dataset_manifest_sha256="1" * 64,
            development_split_sha256="2" * 64,
            checkpoint_sha256="3" * 64,
            prediction_records=(
                _record(case_identifier="case_b", prediction_path="case_b.nii.gz"),
                _record(case_identifier="case_b", prediction_path="case_b_copy.nii.gz"),
            ),
        )
    with pytest.raises(BaselinePredictionValidationError):
        cast(
            Any,
            replace,
        )(
            _manifest(),
            prediction_records=(
                _record(case_identifier="case_b", prediction_path="case_b.nii.gz"),
                _record(case_identifier="case_a", prediction_path="case_a.nii.gz"),
            ),
        )


def test_absolute_path_rejection() -> None:
    with pytest.raises(BaselinePredictionValidationError):
        cast(Any, replace)(_record(), prediction_path="/tmp/case_001.nii.gz")

    with pytest.raises(BaselinePredictionValidationError):
        build_baseline_prediction_manifest(
            baseline_family="nnunet_v2",
            run_identifier="/tmp/run",
            dataset_manifest_sha256="1" * 64,
            development_split_sha256="2" * 64,
            checkpoint_sha256="3" * 64,
            prediction_records=(_record(),),
        )


def test_unknown_fields_and_versions_rejected() -> None:
    manifest = _manifest()
    payload = json.loads(baseline_prediction_manifest_to_json(manifest))
    payload["unknown_field"] = "value"
    with pytest.raises(BaselinePredictionSerializationError):
        baseline_prediction_manifest_from_json(json.dumps(payload).encode("utf-8"))

    payload = json.loads(baseline_prediction_manifest_to_json(manifest))
    payload["contract_version"] = "baseline_prediction_manifest_v999"
    with pytest.raises(BaselinePredictionVersionError):
        baseline_prediction_manifest_from_json(json.dumps(payload).encode("utf-8"))


def test_contract_versions_and_foreground_label_are_fixed() -> None:
    record = _record()
    manifest = _manifest()

    assert record.contract_version == BASELINE_PREDICTION_RECORD_VERSION
    assert manifest.contract_version == BASELINE_PREDICTION_MANIFEST_VERSION
    assert record.foreground_label == 1


def test_importing_predictions_module_does_not_import_heavy_dependencies(
    run_import_guard: Callable[[tuple[str, ...], tuple[str, ...]], None],
) -> None:
    run_import_guard(
        ("protoem_ct.baselines.predictions",),
        ("torch", "torchvision", "monai", "nnunetv2", "SimpleITK", "mlflow"),
    )
