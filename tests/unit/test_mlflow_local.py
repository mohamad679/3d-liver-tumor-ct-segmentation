"""Unit tests for Phase 1 local MLflow tracking."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import mlflow
import nibabel as nib
import pytest
from mlflow.entities import Run
from mlflow.tracking import MlflowClient

from protoem_ct.artifacts import (
    EvaluationArtifact,
    InferenceArtifact,
    ReportArtifact,
    artifact_to_dict,
    sha256_file,
)
from protoem_ct.data.manifest_validation import validate_synthetic_manifest
from protoem_ct.data.preprocessing import preprocess_synthetic_dataset
from protoem_ct.data.synthetic import create_synthetic_dataset, load_synthetic_config
from protoem_ct.evaluation.dummy_inference import run_dummy_inference
from protoem_ct.evaluation.metrics import evaluate_predictions
from protoem_ct.reporting import generate_synthetic_report
from protoem_ct.tracking import (
    CLINICAL_RESULT_TAG,
    LOCAL_MLFLOW_ARTIFACT_NAMES,
    LOCAL_MLFLOW_METRIC_NAMES,
    SCIENTIFIC_RESULT_TAG,
    LocalMlflowArtifactError,
    LocalMlflowInputArtifactError,
    LocalMlflowPathError,
    LocalMlflowTrackingFailureError,
    track_synthetic_run,
)

GIT_COMMIT = "2e1a3f7"
CREATED_AT_UTC = "2026-07-25T01:00:00Z"
VALIDATION_CREATED_AT_UTC = "2026-07-25T01:05:00Z"
PREPROCESS_CREATED_AT_UTC = "2026-07-25T01:10:00Z"
INFERENCE_CREATED_AT_UTC = "2026-07-25T01:15:00Z"
EVALUATION_CREATED_AT_UTC = "2026-07-25T01:20:00Z"
REPORT_CREATED_AT_UTC = "2026-07-25T01:25:00Z"


@dataclass(frozen=True, slots=True)
class _PreparedPipeline:
    manifest_path: Path
    validation_path: Path
    preprocess_path: Path
    inference_path: Path
    evaluation_path: Path
    report_path: Path
    report_artifact_path: Path
    evaluation: EvaluationArtifact
    inference: InferenceArtifact
    report_artifact: ReportArtifact


@pytest.fixture(autouse=True)
def _close_active_mlflow_run() -> None:
    mlflow.end_run()


def _prepared_pipeline(root: Path) -> _PreparedPipeline:
    root.mkdir(parents=True, exist_ok=True)
    config = load_synthetic_config(Path("configs/data/synthetic.yaml"))
    result = create_synthetic_dataset(
        config,
        output_root=root / "data-root",
        git_commit=GIT_COMMIT,
        created_at_utc=CREATED_AT_UTC,
    )
    validation_path = root / "artifacts" / "validation.json"
    validate_synthetic_manifest(
        result.manifest_path,
        data_root=root / "data-root",
        output_path=validation_path,
        git_commit=GIT_COMMIT,
        created_at_utc=VALIDATION_CREATED_AT_UTC,
    )
    preprocess_path = root / "artifacts" / "preprocess.json"
    preprocess_synthetic_dataset(
        result.manifest_path,
        validation_path,
        data_root=root / "data-root",
        output_root=root / "preprocessed",
        artifact_output_path=preprocess_path,
        config_path=Path("configs/experiment/phase1.yaml"),
        git_commit=GIT_COMMIT,
        created_at_utc=PREPROCESS_CREATED_AT_UTC,
    )
    inference_path = root / "artifacts" / "inference.json"
    inference = run_dummy_inference(
        preprocess_path,
        preprocessed_root=root / "preprocessed",
        output_root=root / "predictions",
        artifact_output_path=inference_path,
        config_path=Path("configs/experiment/phase1.yaml"),
        git_commit=GIT_COMMIT,
        created_at_utc=INFERENCE_CREATED_AT_UTC,
    )
    evaluation_path = root / "artifacts" / "evaluation.json"
    evaluation = evaluate_predictions(
        preprocess_path,
        inference_path,
        preprocessed_root=root / "preprocessed",
        prediction_root=root / "predictions",
        artifact_output_path=evaluation_path,
        git_commit=GIT_COMMIT,
        created_at_utc=EVALUATION_CREATED_AT_UTC,
    )
    report_path = root / "reports" / "synthetic_report.md"
    report_artifact_path = root / "artifacts" / "report.json"
    report_artifact = generate_synthetic_report(
        result.manifest_path,
        validation_path,
        preprocess_path,
        inference_path,
        evaluation_path,
        report_output_path=report_path,
        artifact_output_path=report_artifact_path,
        git_commit=GIT_COMMIT,
        created_at_utc=REPORT_CREATED_AT_UTC,
    )
    return _PreparedPipeline(
        manifest_path=result.manifest_path,
        validation_path=validation_path,
        preprocess_path=preprocess_path,
        inference_path=inference_path,
        evaluation_path=evaluation_path,
        report_path=report_path,
        report_artifact_path=report_artifact_path,
        evaluation=evaluation,
        inference=inference,
        report_artifact=report_artifact,
    )


def _track(
    prepared: _PreparedPipeline,
    *,
    tracking_root: Path,
    experiment_name: str = "phase1-unit",
    run_name: str = "unit-run",
) -> tuple[MlflowClient, str]:
    result = track_synthetic_run(
        prepared.manifest_path,
        prepared.validation_path,
        prepared.preprocess_path,
        prepared.inference_path,
        prepared.evaluation_path,
        prepared.report_path,
        prepared.report_artifact_path,
        tracking_root=tracking_root,
        experiment_name=experiment_name,
        run_name=run_name,
    )
    return MlflowClient(tracking_uri=tracking_root.resolve(strict=True).as_uri()), result.run_id


def _artifact_paths(client: MlflowClient, run_id: str, path: str = "") -> tuple[str, ...]:
    paths: list[str] = []
    for file_info in client.list_artifacts(run_id, path):
        if file_info.is_dir:
            paths.extend(_artifact_paths(client, run_id, file_info.path))
        else:
            paths.append(file_info.path)
    return tuple(sorted(paths))


def _write_artifact_payload(path: Path, artifact: object, **overrides: object) -> None:
    payload = cast(dict[str, Any], artifact_to_dict(cast(Any, artifact)))
    payload.update(overrides)
    path.write_text(
        json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _get_run(client: MlflowClient, run_id: str) -> Run:
    return client.get_run(run_id)


def test_valid_linked_artifacts_create_one_local_mlflow_run(tmp_path: Path) -> None:
    prepared = _prepared_pipeline(tmp_path / "pipeline")
    tracking_root = tmp_path / "tracking"

    result = track_synthetic_run(
        prepared.manifest_path,
        prepared.validation_path,
        prepared.preprocess_path,
        prepared.inference_path,
        prepared.evaluation_path,
        prepared.report_path,
        prepared.report_artifact_path,
        tracking_root=tracking_root,
        experiment_name="phase1-unit",
        run_name="unit-run",
    )

    client = MlflowClient(tracking_uri=tracking_root.resolve(strict=True).as_uri())
    experiment = client.get_experiment_by_name("phase1-unit")
    assert experiment is not None
    assert result.experiment_id == experiment.experiment_id
    assert result.run_name == "unit-run"
    assert result.run_status == "FINISHED"
    assert result.tracking_root == tracking_root.resolve(strict=True)
    assert result.config_hash == prepared.evaluation.config_hash
    assert result.manifest_hash == prepared.evaluation.manifest_hash
    assert result.logged_metric_names == LOCAL_MLFLOW_METRIC_NAMES
    assert result.logged_artifact_names == LOCAL_MLFLOW_ARTIFACT_NAMES
    assert len(client.search_runs([experiment.experiment_id])) == 1


def test_named_experiment_is_created_and_reused_with_separate_runs(tmp_path: Path) -> None:
    prepared = _prepared_pipeline(tmp_path / "pipeline")
    tracking_root = tmp_path / "tracking"

    first_client, first_run_id = _track(
        prepared,
        tracking_root=tracking_root,
        experiment_name="phase1-reuse",
        run_name="first-run",
    )
    second_client, second_run_id = _track(
        prepared,
        tracking_root=tracking_root,
        experiment_name="phase1-reuse",
        run_name="second-run",
    )

    experiment = first_client.get_experiment_by_name("phase1-reuse")
    assert experiment is not None
    assert first_run_id != second_run_id
    runs = second_client.search_runs([experiment.experiment_id])
    assert len(runs) == 2
    assert {run.data.tags["mlflow.runName"] for run in runs} == {"first-run", "second-run"}


def test_required_tags_and_false_result_flags_are_recorded(tmp_path: Path) -> None:
    prepared = _prepared_pipeline(tmp_path / "pipeline")
    client, run_id = _track(prepared, tracking_root=tmp_path / "tracking")
    tags = _get_run(client, run_id).data.tags

    assert tags["project"] == "protoem-ct"
    assert tags["phase"] == "1"
    assert tags["data_kind"] == "synthetic"
    assert tags["inference_method"] == "dummy"
    assert tags["purpose"] == "pipeline_verification"
    assert tags["scientific_result"] == SCIENTIFIC_RESULT_TAG
    assert tags["clinical_result"] == CLINICAL_RESULT_TAG
    assert tags["mlflow.runName"] == "unit-run"


def test_required_parameters_are_recorded_without_absolute_paths(tmp_path: Path) -> None:
    prepared = _prepared_pipeline(tmp_path / "pipeline")
    client, run_id = _track(prepared, tracking_root=tmp_path / "tracking")
    params = _get_run(client, run_id).data.params

    assert params["schema_version"] == "1"
    assert params["dataset_id"] == "phase1-synthetic"
    assert params["case_count"] == str(prepared.evaluation.evaluated_case_count)
    assert params["image_shape"] == "8x8x6"
    assert params["spacing"] == "1.500000,1.500000,2.000000"
    assert params["preprocessing_clip_range"] == "-1000.000000,1000.000000"
    assert params["preprocessing_output_range"] == "0.000000,1.000000"
    assert params["preprocessing_image_dtype"] == "float32"
    assert params["preprocessing_label_dtype"] == "uint8"
    assert params["preprocessing_resampling"] == "false"
    assert params["inference_seed"] == str(prepared.inference.deterministic_seed)
    assert params["inference_threshold"] == "0.500000"
    assert params["inference_dtype"] == "uint8"
    assert params["config_hash"] == prepared.evaluation.config_hash
    assert params["manifest_hash"] == prepared.evaluation.manifest_hash
    for key in (
        "git_commit_manifest",
        "git_commit_validation",
        "git_commit_preprocess",
        "git_commit_inference",
        "git_commit_evaluation",
        "git_commit_report",
    ):
        assert params[key] == GIT_COMMIT
    assert all(str(tmp_path) not in value for value in params.values())


def test_persisted_evaluation_metrics_are_logged_exactly(tmp_path: Path) -> None:
    prepared = _prepared_pipeline(tmp_path / "pipeline")
    client, run_id = _track(prepared, tracking_root=tmp_path / "tracking")
    metrics = _get_run(client, run_id).data.metrics

    assert metrics["macro_dice"] == prepared.evaluation.macro_mean_dice
    assert metrics["macro_iou"] == prepared.evaluation.macro_mean_iou
    assert metrics["micro_dice"] == prepared.evaluation.micro_dice
    assert metrics["micro_iou"] == prepared.evaluation.micro_iou
    assert metrics["aggregate_tp"] == float(prepared.evaluation.total_true_positives)
    assert metrics["aggregate_fp"] == float(prepared.evaluation.total_false_positives)
    assert metrics["aggregate_fn"] == float(prepared.evaluation.total_false_negatives)
    assert metrics["aggregate_tn"] == float(prepared.evaluation.total_true_negatives)
    assert metrics["evaluated_case_count"] == float(prepared.evaluation.evaluated_case_count)


def test_metrics_are_not_recomputed_and_nibabel_is_never_used(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared_pipeline(tmp_path / "pipeline")

    def fail_load(*_args: object, **_kwargs: object) -> object:
        msg = "nibabel load must not be called during tracking"
        raise AssertionError(msg)

    def fail_metric_recompute(*_args: object, **_kwargs: object) -> object:
        msg = "metrics must not be recomputed during tracking"
        raise AssertionError(msg)

    monkeypatch.setattr(nib, "load", fail_load)
    monkeypatch.setattr(
        "protoem_ct.evaluation.metrics.compute_case_metrics",
        fail_metric_recompute,
    )

    _track(prepared, tracking_root=tmp_path / "tracking")


def test_expected_json_and_markdown_artifacts_are_logged_without_nifti(
    tmp_path: Path,
) -> None:
    prepared = _prepared_pipeline(tmp_path / "pipeline")
    client, run_id = _track(prepared, tracking_root=tmp_path / "tracking")
    artifact_paths = _artifact_paths(client, run_id)

    assert artifact_paths == tuple(sorted(LOCAL_MLFLOW_ARTIFACT_NAMES))
    assert all(path.endswith((".json", ".md")) for path in artifact_paths)
    assert not any(path.endswith((".nii", ".nii.gz")) for path in artifact_paths)


def test_report_sha256_is_verified(tmp_path: Path) -> None:
    prepared = _prepared_pipeline(tmp_path / "pipeline")
    prepared.report_path.write_text("altered\n", encoding="utf-8")

    with pytest.raises(LocalMlflowArtifactError, match="SHA-256"):
        _track(prepared, tracking_root=tmp_path / "tracking")


@pytest.mark.parametrize(
    ("path_name", "artifact", "overrides", "match"),
    [
        ("evaluation_path", "evaluation", {"config_hash": "1" * 64}, "config hashes"),
        ("inference_path", "inference", {"manifest_hash": "1" * 64}, "manifest hashes"),
        (
            "inference_path",
            "inference",
            {
                "prediction_case_ids": [
                    "synthetic-0002",
                    "synthetic-0001",
                    "synthetic-0000",
                ]
            },
            "case IDs",
        ),
        ("evaluation_path", "evaluation", {"stage": "reporting"}, "invalid evaluation artifact"),
        ("inference_path", "inference", {"method": "model"}, "invalid inference artifact"),
    ],
)
def test_inconsistent_artifacts_are_rejected(
    tmp_path: Path,
    path_name: str,
    artifact: str,
    overrides: dict[str, object],
    match: str,
) -> None:
    prepared = _prepared_pipeline(tmp_path / "pipeline")
    artifact_path = cast(Path, getattr(prepared, path_name))
    artifact_value = getattr(prepared, artifact)
    _write_artifact_payload(artifact_path, artifact_value, **overrides)

    with pytest.raises(LocalMlflowInputArtifactError, match=match):
        _track(prepared, tracking_root=tmp_path / "tracking")


def test_malformed_and_missing_artifacts_are_rejected(tmp_path: Path) -> None:
    malformed = _prepared_pipeline(tmp_path / "malformed")
    malformed.evaluation_path.write_text("{\n", encoding="utf-8")
    with pytest.raises(LocalMlflowInputArtifactError, match="invalid evaluation artifact"):
        _track(malformed, tracking_root=tmp_path / "tracking-malformed")

    missing = _prepared_pipeline(tmp_path / "missing")
    missing.evaluation_path.unlink()
    with pytest.raises(LocalMlflowInputArtifactError, match="evaluation artifact is missing"):
        _track(missing, tracking_root=tmp_path / "tracking-missing")


def test_missing_report_is_rejected(tmp_path: Path) -> None:
    prepared = _prepared_pipeline(tmp_path / "pipeline")
    prepared.report_path.unlink()

    with pytest.raises(LocalMlflowArtifactError, match="Markdown report is missing"):
        _track(prepared, tracking_root=tmp_path / "tracking")


def test_empty_experiment_name_and_run_name_are_rejected(tmp_path: Path) -> None:
    prepared = _prepared_pipeline(tmp_path / "pipeline")
    with pytest.raises(LocalMlflowInputArtifactError, match="experiment_name"):
        _track(prepared, tracking_root=tmp_path / "tracking-exp", experiment_name=" ")
    with pytest.raises(LocalMlflowInputArtifactError, match="run_name"):
        _track(prepared, tracking_root=tmp_path / "tracking-run", run_name="")


def test_invalid_tracking_roots_are_rejected(tmp_path: Path) -> None:
    prepared = _prepared_pipeline(tmp_path / "pipeline")
    regular_file = tmp_path / "tracking-file"
    regular_file.write_text("not a directory\n", encoding="utf-8")

    with pytest.raises(LocalMlflowPathError, match="not a directory"):
        _track(prepared, tracking_root=regular_file)
    with pytest.raises(LocalMlflowPathError, match="not a URI"):
        _track(prepared, tracking_root=Path("https://example.com/mlflow"))
    with pytest.raises(LocalMlflowPathError, match="parent traversal"):
        _track(prepared, tracking_root=tmp_path / ".." / "tracking")


def test_source_files_remain_unchanged(tmp_path: Path) -> None:
    prepared = _prepared_pipeline(tmp_path / "pipeline")
    source_paths = (
        prepared.manifest_path,
        prepared.validation_path,
        prepared.preprocess_path,
        prepared.inference_path,
        prepared.evaluation_path,
        prepared.report_path,
        prepared.report_artifact_path,
    )
    before = {path: sha256_file(path) for path in source_paths}

    _track(prepared, tracking_root=tmp_path / "tracking")

    assert {path: sha256_file(path) for path in source_paths} == before


def test_failed_call_does_not_remove_earlier_successful_run(tmp_path: Path) -> None:
    prepared = _prepared_pipeline(tmp_path / "pipeline")
    tracking_root = tmp_path / "tracking"
    client, first_run_id = _track(
        prepared,
        tracking_root=tracking_root,
        experiment_name="phase1-failure",
        run_name="successful",
    )
    _write_artifact_payload(prepared.inference_path, prepared.inference, manifest_hash="1" * 64)

    with pytest.raises(LocalMlflowInputArtifactError):
        _track(
            prepared,
            tracking_root=tracking_root,
            experiment_name="phase1-failure",
            run_name="failed-before-run",
        )

    assert client.get_run(first_run_id).info.status == "FINISHED"


def test_mlflow_logging_failure_marks_run_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared_pipeline(tmp_path / "pipeline")
    tracking_root = tmp_path / "tracking"
    original_log_metric = mlflow.log_metric

    def fail_log_metric(*_args: object, **_kwargs: object) -> object:
        msg = "forced metric logging failure"
        raise RuntimeError(msg)

    monkeypatch.setattr(mlflow, "log_metric", fail_log_metric)

    with pytest.raises(LocalMlflowTrackingFailureError, match="local MLflow tracking failed"):
        _track(
            prepared,
            tracking_root=tracking_root,
            experiment_name="phase1-failed-run",
            run_name="failed-run",
        )

    monkeypatch.setattr(mlflow, "log_metric", original_log_metric)
    client = MlflowClient(tracking_uri=tracking_root.resolve(strict=True).as_uri())
    experiment = client.get_experiment_by_name("phase1-failed-run")
    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    assert runs[0].info.status == "FAILED"
