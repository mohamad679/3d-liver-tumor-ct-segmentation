"""Integration tests for the local MLflow tracking CLI command."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from mlflow.tracking import MlflowClient
from typer.testing import CliRunner

from protoem_ct.artifacts import (
    EvaluationArtifact,
    InferenceArtifact,
    ReportArtifact,
    artifact_to_dict,
)
from protoem_ct.cli.main import app
from protoem_ct.data.manifest_validation import validate_synthetic_manifest
from protoem_ct.data.preprocessing import preprocess_synthetic_dataset
from protoem_ct.data.synthetic import create_synthetic_dataset, load_synthetic_config
from protoem_ct.evaluation.dummy_inference import run_dummy_inference
from protoem_ct.evaluation.metrics import evaluate_predictions
from protoem_ct.reporting import generate_synthetic_report
from protoem_ct.tracking import LOCAL_MLFLOW_ARTIFACT_NAMES, LOCAL_MLFLOW_METRIC_NAMES

GIT_COMMIT = "2e1a3f7"
CREATED_AT_UTC = "2026-07-25T01:00:00Z"
VALIDATION_CREATED_AT_UTC = "2026-07-25T01:05:00Z"
PREPROCESS_CREATED_AT_UTC = "2026-07-25T01:10:00Z"
INFERENCE_CREATED_AT_UTC = "2026-07-25T01:15:00Z"
EVALUATION_CREATED_AT_UTC = "2026-07-25T01:20:00Z"
REPORT_CREATED_AT_UTC = "2026-07-25T01:25:00Z"


def _invoke_track_run(*args: str) -> Any:
    runner = CliRunner()
    return runner.invoke(app, ["track-run", *args])


def _build_pipeline(root: Path) -> tuple[EvaluationArtifact, InferenceArtifact, ReportArtifact]:
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
    report_artifact = generate_synthetic_report(
        result.manifest_path,
        validation_path,
        preprocess_path,
        inference_path,
        evaluation_path,
        report_output_path=root / "reports" / "synthetic_report.md",
        artifact_output_path=root / "artifacts" / "report.json",
        git_commit=GIT_COMMIT,
        created_at_utc=REPORT_CREATED_AT_UTC,
    )
    return evaluation, inference, report_artifact


def _track_args(
    tracking_root: Path,
    *,
    experiment_name: str = "phase1-cli",
    run_name: str = "cli-run",
) -> list[str]:
    return [
        "--manifest",
        "data-root/generated/phase1/data/synthetic_manifest.json",
        "--validation-artifact",
        "artifacts/validation.json",
        "--preprocess-artifact",
        "artifacts/preprocess.json",
        "--inference-artifact",
        "artifacts/inference.json",
        "--evaluation-artifact",
        "artifacts/evaluation.json",
        "--report",
        "reports/synthetic_report.md",
        "--report-artifact",
        "artifacts/report.json",
        "--tracking-root",
        str(tracking_root),
        "--experiment-name",
        experiment_name,
        "--run-name",
        run_name,
    ]


def _artifact_paths(client: MlflowClient, run_id: str, path: str = "") -> tuple[str, ...]:
    paths: list[str] = []
    for file_info in client.list_artifacts(run_id, path):
        if file_info.is_dir:
            paths.extend(_artifact_paths(client, run_id, file_info.path))
        else:
            paths.append(file_info.path)
    return tuple(sorted(paths))


def _error_output(result: Any) -> str:
    return cast(str, getattr(result, "stderr", "") or result.output)


def _assert_nonzero_without_traceback(result: Any, tmp_path: Path) -> str:
    error_output = _error_output(result)
    assert result.exit_code != 0
    assert "Traceback" not in result.output
    assert "Traceback" not in error_output
    assert str(tmp_path) not in result.output
    assert str(tmp_path) not in error_output
    return error_output


def _write_inference_payload(
    path: Path,
    artifact: InferenceArtifact,
    **overrides: object,
) -> None:
    payload = cast(dict[str, Any], artifact_to_dict(artifact))
    payload.update(overrides)
    path.write_text(
        json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def test_track_run_cli_success_creates_local_store_and_records_expected_data(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    evaluation, _inference, report_artifact = _build_pipeline(tmp_path)
    tracking_root = tmp_path / "tracking-store"
    monkeypatch.chdir(tmp_path)

    result = _invoke_track_run(*_track_args(tracking_root))

    assert result.exit_code == 0
    output_lines = result.output.splitlines()
    assert output_lines[0] == "local MLflow tracking success"
    assert output_lines[1] == "experiment name: phase1-cli"
    assert output_lines[2] == "run name: cli-run"
    assert output_lines[3].startswith("run ID: ")
    assert output_lines[4] == f"logged metric count: {len(LOCAL_MLFLOW_METRIC_NAMES)}"
    assert output_lines[5] == f"logged artifact count: {len(LOCAL_MLFLOW_ARTIFACT_NAMES)}"
    assert output_lines[6] == f"config hash: {evaluation.config_hash}"
    assert output_lines[7] == f"manifest hash: {evaluation.manifest_hash}"
    assert output_lines[8] == "synthetic pipeline verification only"
    assert len(output_lines) == 9
    assert str(tmp_path) not in result.output
    assert not (tmp_path / "mlruns").exists()

    run_id = output_lines[3].split(": ", maxsplit=1)[1]
    client = MlflowClient(tracking_uri=tracking_root.resolve(strict=True).as_uri())
    experiment = client.get_experiment_by_name("phase1-cli")
    assert experiment is not None
    run = client.get_run(run_id)
    assert run.info.experiment_id == experiment.experiment_id
    assert run.data.tags["project"] == "protoem-ct"
    assert run.data.tags["purpose"] == "pipeline_verification"
    assert run.data.tags["scientific_result"] == "false"
    assert run.data.tags["clinical_result"] == "false"
    assert run.data.params["dataset_id"] == "phase1-synthetic"
    assert run.data.params["config_hash"] == evaluation.config_hash
    assert run.data.params["manifest_hash"] == evaluation.manifest_hash
    assert run.data.metrics["macro_dice"] == evaluation.macro_mean_dice
    assert run.data.metrics["macro_iou"] == evaluation.macro_mean_iou
    assert run.data.metrics["micro_dice"] == evaluation.micro_dice
    assert run.data.metrics["micro_iou"] == evaluation.micro_iou
    assert run.data.metrics["evaluated_case_count"] == float(evaluation.evaluated_case_count)
    assert _artifact_paths(client, run_id) == tuple(sorted(LOCAL_MLFLOW_ARTIFACT_NAMES))
    assert report_artifact.report_sha256


def test_track_run_cli_inconsistent_artifact_exits_nonzero_without_traceback(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    _evaluation, inference, _report_artifact = _build_pipeline(tmp_path)
    _write_inference_payload(
        tmp_path / "artifacts" / "inference.json",
        inference,
        manifest_hash="1" * 64,
    )
    monkeypatch.chdir(tmp_path)

    result = _invoke_track_run(*_track_args(tmp_path / "tracking-store"))
    error_output = _assert_nonzero_without_traceback(result, tmp_path)

    assert "manifest hashes" in error_output


def test_track_run_cli_invalid_tracking_root_exits_nonzero_without_traceback(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    _build_pipeline(tmp_path)
    invalid_tracking_root = tmp_path / "tracking-file"
    invalid_tracking_root.write_text("not a directory\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    result = _invoke_track_run(*_track_args(invalid_tracking_root))
    error_output = _assert_nonzero_without_traceback(result, tmp_path)

    assert "not a directory" in error_output


def test_track_run_cli_missing_report_exits_nonzero_without_traceback(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    _build_pipeline(tmp_path)
    (tmp_path / "reports" / "synthetic_report.md").unlink()
    monkeypatch.chdir(tmp_path)

    result = _invoke_track_run(*_track_args(tmp_path / "tracking-store"))
    error_output = _assert_nonzero_without_traceback(result, tmp_path)

    assert "Markdown report is missing" in error_output
