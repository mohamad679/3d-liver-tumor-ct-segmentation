"""Integration tests for the report CLI command."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from typer.testing import CliRunner

from protoem_ct.artifacts import (
    EvaluationArtifact,
    InferenceArtifact,
    ReportArtifact,
    artifact_from_json,
    artifact_to_dict,
    sha256_file,
)
from protoem_ct.cli.main import app
from protoem_ct.data.manifest_validation import validate_synthetic_manifest
from protoem_ct.data.preprocessing import preprocess_synthetic_dataset
from protoem_ct.data.synthetic import create_synthetic_dataset, load_synthetic_config
from protoem_ct.evaluation.dummy_inference import run_dummy_inference
from protoem_ct.evaluation.metrics import evaluate_predictions

GIT_COMMIT = "c3cf8ae"
CREATED_AT_UTC = "2026-07-25T01:00:00Z"
VALIDATION_CREATED_AT_UTC = "2026-07-25T01:05:00Z"
PREPROCESS_CREATED_AT_UTC = "2026-07-25T01:10:00Z"
INFERENCE_CREATED_AT_UTC = "2026-07-25T01:15:00Z"
EVALUATION_CREATED_AT_UTC = "2026-07-25T01:20:00Z"
REPORT_CREATED_AT_UTC = "2026-07-25T01:25:00Z"


def _invoke_report(*args: str) -> Any:
    """Invoke the public report CLI command."""
    runner = CliRunner()
    return runner.invoke(app, ["report", *args])


def _create_validate_preprocess_infer_evaluate(
    root: Path,
) -> tuple[EvaluationArtifact, InferenceArtifact]:
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
    evaluation = evaluate_predictions(
        preprocess_path,
        inference_path,
        preprocessed_root=root / "preprocessed",
        prediction_root=root / "predictions",
        artifact_output_path=root / "artifacts" / "evaluation.json",
        git_commit=GIT_COMMIT,
        created_at_utc=EVALUATION_CREATED_AT_UTC,
    )
    return evaluation, inference


def _report_args(
    *,
    report_output: str = "reports/synthetic_report.md",
    artifact_output: str = "artifacts/report.json",
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
        "--report-output",
        report_output,
        "--artifact-output",
        artifact_output,
        "--git-commit",
        GIT_COMMIT,
        "--created-at-utc",
        REPORT_CREATED_AT_UTC,
    ]


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


def test_report_cli_success_writes_markdown_and_readable_artifact(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    evaluation, _inference = _create_validate_preprocess_infer_evaluate(tmp_path)
    monkeypatch.chdir(tmp_path)

    result = _invoke_report(*_report_args())

    assert result.exit_code == 0
    report_path = tmp_path / "reports" / "synthetic_report.md"
    report_artifact_path = tmp_path / "artifacts" / "report.json"
    assert report_path.is_file()
    assert report_artifact_path.is_file()
    artifact = artifact_from_json(report_artifact_path.read_text(encoding="utf-8"), ReportArtifact)
    assert artifact.report_sha256 == sha256_file(report_path)
    assert result.output == (
        "report generation success\n"
        "report path: reports/synthetic_report.md\n"
        "report artifact path: artifacts/report.json\n"
        f"evaluated case count: {evaluation.evaluated_case_count}\n"
        f"macro Dice: {artifact.macro_mean_dice}\n"
        f"macro IoU: {artifact.macro_mean_iou}\n"
        f"report SHA-256: {artifact.report_sha256}\n"
        f"config hash: {artifact.config_hash}\n"
        f"manifest hash: {artifact.manifest_hash}\n"
        "method: dummy\n"
    )
    assert artifact.macro_mean_dice == evaluation.macro_mean_dice
    assert artifact.macro_mean_iou == evaluation.macro_mean_iou
    assert artifact.micro_dice == evaluation.micro_dice
    assert artifact.micro_iou == evaluation.micro_iou
    assert "synthetic Phase 1 pipeline report" in report_path.read_text(encoding="utf-8")


def test_report_cli_repeated_independent_runs_produce_byte_identical_markdown(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    _create_validate_preprocess_infer_evaluate(first_root)
    _create_validate_preprocess_infer_evaluate(second_root)

    monkeypatch.chdir(first_root)
    first = _invoke_report(*_report_args())
    monkeypatch.chdir(second_root)
    second = _invoke_report(*_report_args())

    assert first.exit_code == 0
    assert second.exit_code == 0
    assert (first_root / "reports" / "synthetic_report.md").read_bytes() == (
        second_root / "reports" / "synthetic_report.md"
    ).read_bytes()


def test_report_cli_inconsistent_input_artifact_exits_nonzero_without_traceback(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    _evaluation, inference = _create_validate_preprocess_infer_evaluate(tmp_path)
    _write_inference_payload(
        tmp_path / "artifacts" / "inference.json",
        inference,
        manifest_hash="1" * 64,
    )
    monkeypatch.chdir(tmp_path)

    result = _invoke_report(*_report_args())
    error_output = _assert_nonzero_without_traceback(result, tmp_path)

    assert "manifest hashes" in error_output
    assert not (tmp_path / "reports" / "synthetic_report.md").exists()
    assert not (tmp_path / "artifacts" / "report.json").exists()


def test_report_cli_existing_output_exits_nonzero_without_traceback(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    _evaluation, _inference = _create_validate_preprocess_infer_evaluate(tmp_path)
    report_path = tmp_path / "reports" / "synthetic_report.md"
    report_path.parent.mkdir()
    report_path.write_text("existing\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    result = _invoke_report(*_report_args())
    error_output = _assert_nonzero_without_traceback(result, tmp_path)

    assert "overwrite" in error_output
    assert not (tmp_path / "artifacts" / "report.json").exists()
