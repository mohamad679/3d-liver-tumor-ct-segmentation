"""Unit tests for Phase 1 artifact schema contracts."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, cast

import pytest

from protoem_ct.artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    ArtifactSchemaVersionError,
    ArtifactStageError,
    ArtifactValidationError,
    EvaluationArtifact,
    EvaluationCaseMetrics,
    InferenceArtifact,
    PreprocessArtifact,
    ReportArtifact,
    RunMetadata,
    SyntheticManifest,
    ValidationArtifact,
    artifact_from_json,
    artifact_to_dict,
    artifact_to_json,
)

HASH = "0" * 64


def _case_metric(case_id: str = "case-001", **overrides: object) -> EvaluationCaseMetrics:
    metric = EvaluationCaseMetrics(
        case_id=case_id,
        true_positives=2,
        false_positives=1,
        false_negatives=1,
        true_negatives=4,
        ground_truth_foreground_voxels=3,
        predicted_foreground_voxels=3,
        dice=2 / 3,
        iou=0.5,
    )
    return cast(EvaluationCaseMetrics, cast(Any, replace)(metric, **overrides))


def _manifest(**overrides: object) -> SyntheticManifest:
    manifest = SyntheticManifest(
        schema_version=ARTIFACT_SCHEMA_VERSION,
        stage="synthetic-manifest",
        created_at_utc="2026-07-25T00:00:00Z",
        git_commit="abc1234",
        config_hash=HASH,
        manifest_hash=HASH,
        dataset_id="phase1-synthetic",
        seed=1729,
        case_ids=("case-001", "case-002"),
        image_paths=(
            "generated/phase1/data/images/case-001.nii.gz",
            "generated/phase1/data/images/case-002.nii.gz",
        ),
        label_paths=(
            "generated/phase1/data/labels/case-001.nii.gz",
            "generated/phase1/data/labels/case-002.nii.gz",
        ),
        shape=(8, 8, 6),
        spacing=(1.5, 1.5, 2.0),
    )
    return cast(SyntheticManifest, cast(Any, replace)(manifest, **overrides))


def test_artifact_to_json_is_deterministic_and_writes_utf8_newline(tmp_path: Path) -> None:
    manifest = _manifest()
    output_path = tmp_path / "manifest.json"

    first = artifact_to_json(manifest, output_path)
    second = artifact_to_json(manifest)

    assert first == second
    assert first.endswith("\n")
    assert output_path.read_bytes().endswith(b"\n")
    assert list(json.loads(first)) == sorted(json.loads(first))


def test_artifact_to_dict_uses_json_compatible_lists() -> None:
    value = artifact_to_dict(_manifest())

    assert value["case_ids"] == ["case-001", "case-002"]
    assert value["shape"] == [8, 8, 6]
    assert value["spacing"] == [1.5, 1.5, 2.0]


def test_artifact_json_round_trip_resolves_stage() -> None:
    manifest = _manifest()
    loaded: object = artifact_from_json(artifact_to_json(manifest))

    assert loaded == manifest


def test_artifact_json_round_trip_with_explicit_type() -> None:
    manifest = _manifest()
    loaded = artifact_from_json(artifact_to_json(manifest), SyntheticManifest)

    assert loaded == manifest


def test_incorrect_schema_version_is_rejected() -> None:
    manifest_json = artifact_to_json(_manifest()).replace(
        '"schema_version":"1"',
        '"schema_version":"2"',
    )

    with pytest.raises(ArtifactSchemaVersionError):
        artifact_from_json(manifest_json, SyntheticManifest)


def test_incorrect_stage_is_rejected() -> None:
    manifest_json = artifact_to_json(_manifest()).replace(
        '"stage":"synthetic-manifest"',
        '"stage":"validate"',
    )

    with pytest.raises(ArtifactStageError):
        artifact_from_json(manifest_json, SyntheticManifest)


def test_duplicate_case_ids_are_rejected() -> None:
    with pytest.raises(ArtifactValidationError):
        _manifest(case_ids=("case-001", "case-001"))


def test_unequal_manifest_list_lengths_are_rejected() -> None:
    with pytest.raises(ArtifactValidationError):
        _manifest(label_paths=("generated/phase1/data/labels/case-001.nii.gz",))


def test_absolute_paths_are_rejected() -> None:
    bad_path = str(PurePosixPath("/") / "generated" / "phase1" / "data" / "case-001.nii.gz")

    with pytest.raises(ArtifactValidationError):
        _manifest(image_paths=(bad_path, "generated/phase1/data/images/case-002.nii.gz"))


@pytest.mark.parametrize(
    "bad_path",
    [
        "../data/images/case-001.nii.gz",
        "generated/../data/images/case-001.nii.gz",
        "generated\\..\\data\\images\\case-001.nii.gz",
    ],
)
def test_parent_traversal_is_rejected(bad_path: str) -> None:
    with pytest.raises(ArtifactValidationError):
        _manifest(image_paths=(bad_path, "generated/phase1/data/images/case-002.nii.gz"))


@pytest.mark.parametrize("bad_shape", [(8, 8), (8, 8, 0), (8, -1, 6), (8.0, 8, 6)])
def test_invalid_shape_is_rejected(bad_shape: tuple[object, ...]) -> None:
    with pytest.raises(ArtifactValidationError):
        _manifest(shape=bad_shape)


@pytest.mark.parametrize("bad_spacing", [(1.5, 1.5), (1.5, 0.0, 2.0), (1.5, float("inf"), 2.0)])
def test_invalid_spacing_is_rejected(bad_spacing: tuple[float, ...]) -> None:
    with pytest.raises(ArtifactValidationError):
        _manifest(spacing=bad_spacing)


def test_valid_relative_synthetic_manifest_is_accepted() -> None:
    manifest = _manifest()

    assert manifest.dataset_id == "phase1-synthetic"
    assert manifest.case_ids == ("case-001", "case-002")


def test_all_stage_specific_artifacts_are_constructible() -> None:
    common = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "created_at_utc": "2026-07-25T00:00:00Z",
        "git_commit": "abc1234",
        "config_hash": HASH,
        "manifest_hash": HASH,
    }

    artifacts = [
        RunMetadata(
            stage="run-metadata",
            run_id="phase1-local",
            phase="1",
            command=("uv", "run"),
            **common,
        ),
        ValidationArtifact(
            stage="validate",
            valid_case_count=2,
            invalid_case_count=0,
            validated_case_ids=("case-001", "case-002"),
            **common,
        ),
        PreprocessArtifact(
            stage="preprocess",
            output_case_ids=("case-001", "case-002"),
            output_image_paths=(
                "images/case-001.nii",
                "images/case-002.nii",
            ),
            output_label_paths=(
                "labels/case-001.nii",
                "labels/case-002.nii",
            ),
            target_spacing=(1.5, 1.5, 2.0),
            preprocessing_parameters=MappingProxyType({"clip_min": -1000.0, "clip_max": 1000.0}),
            **common,
        ),
        InferenceArtifact(
            stage="infer-dummy",
            prediction_case_ids=("case-001", "case-002"),
            prediction_paths=(
                "predictions/case-001.nii",
                "predictions/case-002.nii",
            ),
            prediction_hashes=(HASH, "1" * 64),
            method="dummy",
            deterministic_seed=2718,
            threshold=0.5,
            prediction_dtype="uint8",
            **common,
        ),
        EvaluationArtifact(
            stage="evaluate",
            method="dummy",
            case_ids=("case-001", "case-002"),
            case_metrics=(
                EvaluationCaseMetrics(
                    case_id="case-001",
                    true_positives=2,
                    false_positives=1,
                    false_negatives=1,
                    true_negatives=4,
                    ground_truth_foreground_voxels=3,
                    predicted_foreground_voxels=3,
                    dice=2 / 3,
                    iou=0.5,
                ),
                EvaluationCaseMetrics(
                    case_id="case-002",
                    true_positives=0,
                    false_positives=0,
                    false_negatives=0,
                    true_negatives=8,
                    ground_truth_foreground_voxels=0,
                    predicted_foreground_voxels=0,
                    dice=1.0,
                    iou=1.0,
                ),
            ),
            macro_mean_dice=((2 / 3) + 1.0) / 2,
            macro_mean_iou=(0.5 + 1.0) / 2,
            micro_dice=2 / 3,
            micro_iou=0.5,
            total_true_positives=2,
            total_false_positives=1,
            total_false_negatives=1,
            total_true_negatives=12,
            evaluated_case_count=2,
            **common,
        ),
        ReportArtifact(
            stage="reporting",
            source_artifact_paths=("generated/phase1/artifacts/evaluation.json",),
            report_path="generated/phase1/reports/synthetic_report.md",
            report_format="markdown",
            report_sha256=HASH,
            evaluated_case_count=2,
            method="dummy",
            macro_mean_dice=0.75,
            macro_mean_iou=0.5,
            micro_dice=0.8,
            micro_iou=2 / 3,
            reported_metric_names=("dice", "iou"),
            **common,
        ),
    ]

    assert [artifact.stage for artifact in artifacts] == [
        "run-metadata",
        "validate",
        "preprocess",
        "infer-dummy",
        "evaluate",
        "reporting",
    ]


def test_preprocess_artifact_requires_relative_output_paths() -> None:
    common: dict[str, Any] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "stage": "preprocess",
        "created_at_utc": "2026-07-25T00:00:00Z",
        "git_commit": "abc1234",
        "config_hash": HASH,
        "manifest_hash": HASH,
        "output_case_ids": ("case-001",),
        "target_spacing": (1.5, 1.5, 2.0),
        "preprocessing_parameters": MappingProxyType({"clip_min": -1000.0, "clip_max": 1000.0}),
    }

    with pytest.raises(ArtifactValidationError):
        PreprocessArtifact(
            output_image_paths=("/absolute/case-001.nii",),
            output_label_paths=("labels/case-001.nii",),
            **common,
        )


def test_preprocess_artifact_requires_equal_output_lengths() -> None:
    common: dict[str, Any] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "stage": "preprocess",
        "created_at_utc": "2026-07-25T00:00:00Z",
        "git_commit": "abc1234",
        "config_hash": HASH,
        "manifest_hash": HASH,
        "output_case_ids": ("case-001", "case-002"),
        "target_spacing": (1.5, 1.5, 2.0),
        "preprocessing_parameters": MappingProxyType({"clip_min": -1000.0, "clip_max": 1000.0}),
    }

    with pytest.raises(ArtifactValidationError):
        PreprocessArtifact(
            output_image_paths=("images/case-001.nii",),
            output_label_paths=("labels/case-001.nii", "labels/case-002.nii"),
            **common,
        )


def test_inference_artifact_requires_relative_prediction_paths() -> None:
    common: dict[str, Any] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "stage": "infer-dummy",
        "created_at_utc": "2026-07-25T00:00:00Z",
        "git_commit": "abc1234",
        "config_hash": HASH,
        "manifest_hash": HASH,
        "prediction_case_ids": ("case-001",),
        "prediction_hashes": (HASH,),
        "method": "dummy",
        "deterministic_seed": 2718,
        "threshold": 0.5,
        "prediction_dtype": "uint8",
    }

    with pytest.raises(ArtifactValidationError):
        InferenceArtifact(prediction_paths=("/absolute/case-001.nii",), **common)


def test_inference_artifact_requires_prediction_hashes() -> None:
    common: dict[str, Any] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "stage": "infer-dummy",
        "created_at_utc": "2026-07-25T00:00:00Z",
        "git_commit": "abc1234",
        "config_hash": HASH,
        "manifest_hash": HASH,
        "prediction_case_ids": ("case-001",),
        "prediction_paths": ("predictions/case-001.nii",),
        "method": "dummy",
        "deterministic_seed": 2718,
        "threshold": 0.5,
        "prediction_dtype": "uint8",
    }

    with pytest.raises(ArtifactValidationError):
        InferenceArtifact(prediction_hashes=("not-a-sha256",), **common)


def test_inference_artifact_requires_equal_prediction_lengths() -> None:
    common: dict[str, Any] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "stage": "infer-dummy",
        "created_at_utc": "2026-07-25T00:00:00Z",
        "git_commit": "abc1234",
        "config_hash": HASH,
        "manifest_hash": HASH,
        "prediction_case_ids": ("case-001", "case-002"),
        "prediction_paths": ("predictions/case-001.nii",),
        "method": "dummy",
        "deterministic_seed": 2718,
        "threshold": 0.5,
        "prediction_dtype": "uint8",
    }

    with pytest.raises(ArtifactValidationError):
        InferenceArtifact(prediction_hashes=(HASH, HASH), **common)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [("threshold", 1.1), ("prediction_dtype", "float32")],
)
def test_inference_artifact_rejects_invalid_output_settings(
    field_name: str,
    value: object,
) -> None:
    common: dict[str, Any] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "stage": "infer-dummy",
        "created_at_utc": "2026-07-25T00:00:00Z",
        "git_commit": "abc1234",
        "config_hash": HASH,
        "manifest_hash": HASH,
        "prediction_case_ids": ("case-001",),
        "prediction_paths": ("predictions/case-001.nii",),
        "prediction_hashes": (HASH,),
        "method": "dummy",
        "deterministic_seed": 2718,
        "threshold": 0.5,
        "prediction_dtype": "uint8",
    }
    common[field_name] = value

    with pytest.raises(ArtifactValidationError):
        InferenceArtifact(**common)


def test_evaluation_artifact_round_trips_full_machine_readable_metrics() -> None:
    case_metrics = (
        _case_metric("case-001"),
        EvaluationCaseMetrics(
            case_id="case-002",
            true_positives=0,
            false_positives=0,
            false_negatives=2,
            true_negatives=6,
            ground_truth_foreground_voxels=2,
            predicted_foreground_voxels=0,
            dice=0.0,
            iou=0.0,
        ),
    )
    artifact = EvaluationArtifact(
        schema_version=ARTIFACT_SCHEMA_VERSION,
        stage="evaluate",
        created_at_utc="2026-07-25T00:00:00Z",
        git_commit="abc1234",
        config_hash=HASH,
        manifest_hash=HASH,
        method="dummy",
        case_ids=("case-001", "case-002"),
        case_metrics=case_metrics,
        macro_mean_dice=1 / 3,
        macro_mean_iou=0.25,
        micro_dice=0.5,
        micro_iou=1 / 3,
        total_true_positives=2,
        total_false_positives=1,
        total_false_negatives=3,
        total_true_negatives=10,
        evaluated_case_count=2,
    )

    loaded = artifact_from_json(artifact_to_json(artifact), EvaluationArtifact)

    assert loaded == artifact
    assert loaded.case_metrics[0].true_positives == 2
    assert loaded.case_metrics[1].dice == 0.0


def test_evaluation_case_metrics_reject_inconsistent_foreground_counts() -> None:
    with pytest.raises(ArtifactValidationError, match="ground_truth_foreground"):
        _case_metric(ground_truth_foreground_voxels=2)

    with pytest.raises(ArtifactValidationError, match="predicted_foreground"):
        _case_metric(predicted_foreground_voxels=2)


def test_evaluation_case_metrics_reject_inconsistent_overlap_metrics() -> None:
    with pytest.raises(ArtifactValidationError, match="dice"):
        _case_metric(dice=0.25)

    with pytest.raises(ArtifactValidationError, match="iou"):
        _case_metric(iou=0.25)


def test_evaluation_artifact_rejects_non_dummy_method() -> None:
    with pytest.raises(ArtifactValidationError, match="method"):
        EvaluationArtifact(
            schema_version=ARTIFACT_SCHEMA_VERSION,
            stage="evaluate",
            created_at_utc="2026-07-25T00:00:00Z",
            git_commit="abc1234",
            config_hash=HASH,
            manifest_hash=HASH,
            method="other",
            case_ids=("case-001",),
            case_metrics=(_case_metric("case-001"),),
            macro_mean_dice=2 / 3,
            macro_mean_iou=0.5,
            micro_dice=2 / 3,
            micro_iou=0.5,
            total_true_positives=2,
            total_false_positives=1,
            total_false_negatives=1,
            total_true_negatives=4,
            evaluated_case_count=1,
        )


def test_evaluation_artifact_rejects_inconsistent_aggregate_counts() -> None:
    with pytest.raises(ArtifactValidationError, match="total_true_positives"):
        EvaluationArtifact(
            schema_version=ARTIFACT_SCHEMA_VERSION,
            stage="evaluate",
            created_at_utc="2026-07-25T00:00:00Z",
            git_commit="abc1234",
            config_hash=HASH,
            manifest_hash=HASH,
            method="dummy",
            case_ids=("case-001",),
            case_metrics=(_case_metric("case-001"),),
            macro_mean_dice=2 / 3,
            macro_mean_iou=0.5,
            micro_dice=2 / 3,
            micro_iou=0.5,
            total_true_positives=3,
            total_false_positives=1,
            total_false_negatives=1,
            total_true_negatives=4,
            evaluated_case_count=1,
        )


def test_evaluation_artifact_rejects_inconsistent_case_order() -> None:
    with pytest.raises(ArtifactValidationError, match="case_ids"):
        EvaluationArtifact(
            schema_version=ARTIFACT_SCHEMA_VERSION,
            stage="evaluate",
            created_at_utc="2026-07-25T00:00:00Z",
            git_commit="abc1234",
            config_hash=HASH,
            manifest_hash=HASH,
            method="dummy",
            case_ids=("case-002",),
            case_metrics=(_case_metric("case-001"),),
            macro_mean_dice=2 / 3,
            macro_mean_iou=0.5,
            micro_dice=2 / 3,
            micro_iou=0.5,
            total_true_positives=2,
            total_false_positives=1,
            total_false_negatives=1,
            total_true_negatives=4,
            evaluated_case_count=1,
        )


def test_report_artifact_round_trips_full_machine_readable_summary() -> None:
    artifact = ReportArtifact(
        schema_version=ARTIFACT_SCHEMA_VERSION,
        stage="reporting",
        created_at_utc="2026-07-25T00:00:00Z",
        git_commit="abc1234",
        config_hash=HASH,
        manifest_hash=HASH,
        source_artifact_paths=(
            "artifacts/manifest.json",
            "artifacts/validation.json",
            "artifacts/preprocess.json",
            "artifacts/inference.json",
            "artifacts/evaluation.json",
        ),
        report_path="reports/synthetic_report.md",
        report_format="markdown",
        report_sha256=HASH,
        evaluated_case_count=2,
        method="dummy",
        macro_mean_dice=0.75,
        macro_mean_iou=0.5,
        micro_dice=0.8,
        micro_iou=2 / 3,
        reported_metric_names=("dice", "iou"),
    )

    loaded = artifact_from_json(artifact_to_json(artifact), ReportArtifact)

    assert loaded == artifact
    assert loaded.report_format == "markdown"
    assert loaded.report_sha256 == HASH


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("report_format", "html"),
        ("report_sha256", "not-a-sha256"),
        ("evaluated_case_count", 0),
        ("method", "model"),
        ("macro_mean_dice", 1.1),
        ("reported_metric_names", ("dice",)),
    ],
)
def test_report_artifact_rejects_invalid_summary_fields(
    field_name: str,
    value: object,
) -> None:
    common: dict[str, Any] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "stage": "reporting",
        "created_at_utc": "2026-07-25T00:00:00Z",
        "git_commit": "abc1234",
        "config_hash": HASH,
        "manifest_hash": HASH,
        "source_artifact_paths": ("artifacts/evaluation.json",),
        "report_path": "reports/synthetic_report.md",
        "report_format": "markdown",
        "report_sha256": HASH,
        "evaluated_case_count": 2,
        "method": "dummy",
        "macro_mean_dice": 0.75,
        "macro_mean_iou": 0.5,
        "micro_dice": 0.8,
        "micro_iou": 2 / 3,
        "reported_metric_names": ("dice", "iou"),
    }
    common[field_name] = value

    with pytest.raises(ArtifactValidationError):
        ReportArtifact(**common)
