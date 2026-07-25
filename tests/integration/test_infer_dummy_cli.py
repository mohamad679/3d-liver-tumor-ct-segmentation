"""Integration tests for the infer-dummy CLI command."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from typer.testing import CliRunner

from protoem_ct.artifacts import InferenceArtifact, PreprocessArtifact, artifact_from_json
from protoem_ct.cli.main import app
from protoem_ct.data.manifest_validation import validate_synthetic_manifest
from protoem_ct.data.preprocessing import preprocess_synthetic_dataset
from protoem_ct.data.synthetic import create_synthetic_dataset, load_synthetic_config

GIT_COMMIT = "8ce4f1e"
CREATED_AT_UTC = "2026-07-25T01:00:00Z"
VALIDATION_CREATED_AT_UTC = "2026-07-25T01:05:00Z"
PREPROCESS_CREATED_AT_UTC = "2026-07-25T01:10:00Z"
INFERENCE_CREATED_AT_UTC = "2026-07-25T01:15:00Z"


def _invoke_infer_dummy(*args: str) -> Any:
    """Invoke the public infer-dummy CLI command."""
    runner = CliRunner()
    return runner.invoke(app, ["infer-dummy", *args])


def _create_validate_preprocess(tmp_path: Path) -> PreprocessArtifact:
    config = load_synthetic_config(Path("configs/data/synthetic.yaml"))
    result = create_synthetic_dataset(
        config,
        output_root=tmp_path / "data-root",
        git_commit=GIT_COMMIT,
        created_at_utc=CREATED_AT_UTC,
    )
    validation_path = tmp_path / "artifacts" / "validation.json"
    validate_synthetic_manifest(
        result.manifest_path,
        data_root=tmp_path / "data-root",
        output_path=validation_path,
        git_commit=GIT_COMMIT,
        created_at_utc=VALIDATION_CREATED_AT_UTC,
    )
    return preprocess_synthetic_dataset(
        result.manifest_path,
        validation_path,
        data_root=tmp_path / "data-root",
        output_root=tmp_path / "preprocessed",
        artifact_output_path=tmp_path / "artifacts" / "preprocess.json",
        config_path=Path("configs/experiment/phase1.yaml"),
        git_commit=GIT_COMMIT,
        created_at_utc=PREPROCESS_CREATED_AT_UTC,
    )


def _infer_args(
    config_path: Path,
    *,
    output_root: str = "predictions",
    artifact_output: str = "artifacts/inference.json",
) -> list[str]:
    return [
        "--preprocess-artifact",
        "artifacts/preprocess.json",
        "--preprocessed-root",
        "preprocessed",
        "--output-root",
        output_root,
        "--artifact-output",
        artifact_output,
        "--config",
        str(config_path),
        "--git-commit",
        GIT_COMMIT,
        "--created-at-utc",
        INFERENCE_CREATED_AT_UTC,
    ]


def _error_output(result: Any) -> str:
    return cast(str, getattr(result, "stderr", "") or result.output)


def _assert_nonzero_without_traceback(result: Any, tmp_path: Path) -> str:
    error_output = _error_output(result)
    assert result.exit_code != 0
    assert "Traceback" not in result.output
    assert "Traceback" not in error_output
    assert str(tmp_path) not in error_output
    return error_output


def test_infer_dummy_cli_success_writes_predictions_and_readable_artifact(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    preprocess_artifact = _create_validate_preprocess(tmp_path)
    config_path = Path("configs/experiment/phase1.yaml").resolve()
    monkeypatch.chdir(tmp_path)

    result = _invoke_infer_dummy(*_infer_args(config_path))

    assert result.exit_code == 0
    assert result.output == (
        "dummy inference success\n"
        "prediction count: 3\n"
        "inference artifact path: artifacts/inference.json\n"
        f"config hash: {preprocess_artifact.config_hash}\n"
        f"manifest hash: {preprocess_artifact.manifest_hash}\n"
        "method: dummy\n"
    )
    artifact_path = tmp_path / "artifacts" / "inference.json"
    artifact = artifact_from_json(artifact_path.read_text(encoding="utf-8"), InferenceArtifact)
    assert artifact.prediction_case_ids == preprocess_artifact.output_case_ids
    assert artifact.config_hash == preprocess_artifact.config_hash
    assert artifact.manifest_hash == preprocess_artifact.manifest_hash
    assert artifact.method == "dummy"
    for relative_path in artifact.prediction_paths:
        assert (tmp_path / "predictions" / relative_path).is_file()


def test_infer_dummy_cli_repeated_runs_have_identical_prediction_hashes(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    _preprocess_artifact = _create_validate_preprocess(tmp_path)
    config_path = Path("configs/experiment/phase1.yaml").resolve()
    monkeypatch.chdir(tmp_path)

    first = _invoke_infer_dummy(
        *_infer_args(
            config_path,
            output_root="first-predictions",
            artifact_output="first-artifacts/inference.json",
        )
    )
    second = _invoke_infer_dummy(
        *_infer_args(
            config_path,
            output_root="second-predictions",
            artifact_output="second-artifacts/inference.json",
        )
    )

    assert first.exit_code == 0
    assert second.exit_code == 0
    first_artifact = artifact_from_json(
        (tmp_path / "first-artifacts" / "inference.json").read_text(encoding="utf-8"),
        InferenceArtifact,
    )
    second_artifact = artifact_from_json(
        (tmp_path / "second-artifacts" / "inference.json").read_text(encoding="utf-8"),
        InferenceArtifact,
    )
    assert first_artifact.prediction_hashes == second_artifact.prediction_hashes


def test_infer_dummy_cli_invalid_preprocessing_artifact_exits_nonzero_without_traceback(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    _preprocess_artifact = _create_validate_preprocess(tmp_path)
    (tmp_path / "artifacts" / "preprocess.json").write_text("{}\n", encoding="utf-8")
    config_path = Path("configs/experiment/phase1.yaml").resolve()
    monkeypatch.chdir(tmp_path)

    result = _invoke_infer_dummy(*_infer_args(config_path))
    error_output = _assert_nonzero_without_traceback(result, tmp_path)

    assert "invalid preprocessing artifact" in error_output
    assert not (tmp_path / "predictions").exists()
    assert not (tmp_path / "artifacts" / "inference.json").exists()


def test_infer_dummy_cli_missing_preprocessed_image_exits_nonzero_without_traceback(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    preprocess_artifact = _create_validate_preprocess(tmp_path)
    (tmp_path / "preprocessed" / preprocess_artifact.output_image_paths[0]).unlink()
    config_path = Path("configs/experiment/phase1.yaml").resolve()
    monkeypatch.chdir(tmp_path)

    result = _invoke_infer_dummy(*_infer_args(config_path))
    error_output = _assert_nonzero_without_traceback(result, tmp_path)

    assert "missing" in error_output
    assert not (tmp_path / "predictions").exists()
    assert not (tmp_path / "artifacts" / "inference.json").exists()


def test_infer_dummy_cli_nonempty_output_root_exits_nonzero_without_traceback(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    _preprocess_artifact = _create_validate_preprocess(tmp_path)
    output_root = tmp_path / "predictions"
    output_root.mkdir()
    (output_root / "existing.txt").write_text("occupied\n", encoding="utf-8")
    config_path = Path("configs/experiment/phase1.yaml").resolve()
    monkeypatch.chdir(tmp_path)

    result = _invoke_infer_dummy(*_infer_args(config_path))
    error_output = _assert_nonzero_without_traceback(result, tmp_path)

    assert "output root must be empty or nonexistent" in error_output
    assert not (tmp_path / "artifacts" / "inference.json").exists()


def test_infer_dummy_cli_existing_artifact_output_exits_nonzero_without_traceback(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    _preprocess_artifact = _create_validate_preprocess(tmp_path)
    (tmp_path / "artifacts" / "inference.json").write_text("{}\n", encoding="utf-8")
    config_path = Path("configs/experiment/phase1.yaml").resolve()
    monkeypatch.chdir(tmp_path)

    result = _invoke_infer_dummy(*_infer_args(config_path))
    error_output = _assert_nonzero_without_traceback(result, tmp_path)

    assert "overwrite" in error_output
    assert not (tmp_path / "predictions").exists()


def test_infer_dummy_cli_malformed_artifact_path_values_exit_without_traceback(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    _preprocess_artifact = _create_validate_preprocess(tmp_path)
    preprocess_artifact_path = tmp_path / "artifacts" / "preprocess.json"
    payload = json.loads(preprocess_artifact_path.read_text(encoding="utf-8"))
    payload["output_image_paths"][0] = "../escape.nii"
    preprocess_artifact_path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    config_path = Path("configs/experiment/phase1.yaml").resolve()
    monkeypatch.chdir(tmp_path)

    result = _invoke_infer_dummy(*_infer_args(config_path))
    error_output = _assert_nonzero_without_traceback(result, tmp_path)

    assert "parent traversal" in error_output
    assert not (tmp_path / "predictions").exists()
    assert not (tmp_path / "artifacts" / "inference.json").exists()
