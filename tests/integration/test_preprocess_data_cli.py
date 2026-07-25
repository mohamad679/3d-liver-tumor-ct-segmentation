"""Integration tests for the preprocess-data CLI command."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import nibabel as nib
import pytest
from nibabel.filebasedimages import ImageFileError
from typer.testing import CliRunner

from protoem_ct.artifacts import PreprocessArtifact, SyntheticManifest, artifact_from_json
from protoem_ct.artifacts.hashing import hash_config
from protoem_ct.cli.main import app
from protoem_ct.data.manifest_validation import validate_synthetic_manifest
from protoem_ct.data.preprocessing import (
    load_preprocessing_config,
    preprocessing_config_hash_payload,
)
from protoem_ct.data.synthetic import create_synthetic_dataset, load_synthetic_config

GIT_COMMIT = "728d38b"
CREATED_AT_UTC = "2026-07-25T01:00:00Z"
VALIDATION_CREATED_AT_UTC = "2026-07-25T01:05:00Z"
PREPROCESS_CREATED_AT_UTC = "2026-07-25T01:10:00Z"
NIFTI1_HEADER_AND_EXTENDER_SIZE = 352


def _invoke_preprocess_data(*args: str) -> Any:
    """Invoke the public preprocess-data CLI command."""
    runner = CliRunner()
    return runner.invoke(app, ["preprocess-data", *args])


def _create_and_validate(tmp_path: Path) -> tuple[SyntheticManifest, Path, Path]:
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
    return result.manifest, result.manifest_path, validation_path


def _preprocess_args(config_path: Path) -> list[str]:
    return [
        "--manifest",
        "data-root/generated/phase1/data/synthetic_manifest.json",
        "--validation-artifact",
        "artifacts/validation.json",
        "--data-root",
        "data-root",
        "--output-root",
        "preprocessed",
        "--artifact-output",
        "artifacts/preprocess.json",
        "--config",
        str(config_path),
        "--git-commit",
        GIT_COMMIT,
        "--created-at-utc",
        PREPROCESS_CREATED_AT_UTC,
    ]


def _error_output(result: Any) -> str:
    return cast(str, getattr(result, "stderr", "") or result.output)


def _assert_nonzero_without_traceback(result: Any) -> str:
    error_output = _error_output(result)
    assert result.exit_code != 0
    assert "Traceback" not in result.output
    assert "Traceback" not in error_output
    return error_output


def _corrupt_nifti_header(path: Path) -> None:
    """Replace the NIfTI-1 header without truncating the test input."""
    payload = path.read_bytes()
    if len(payload) < NIFTI1_HEADER_AND_EXTENDER_SIZE:
        msg = f"test NIfTI is unexpectedly short: {path}"
        raise AssertionError(msg)
    path.write_bytes(
        bytes(NIFTI1_HEADER_AND_EXTENDER_SIZE) + payload[NIFTI1_HEADER_AND_EXTENDER_SIZE:]
    )


def test_preprocess_data_cli_success_writes_outputs_and_readable_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, _manifest_path, _validation_path = _create_and_validate(tmp_path)
    config_path = Path("configs/experiment/phase1.yaml").resolve()
    config = load_preprocessing_config(config_path)
    config_hash = hash_config(preprocessing_config_hash_payload(config))
    monkeypatch.chdir(tmp_path)

    result = _invoke_preprocess_data(*_preprocess_args(config_path))

    assert result.exit_code == 0
    assert result.output == (
        "preprocessing success\n"
        "processed case count: 3\n"
        "preprocessing artifact path: artifacts/preprocess.json\n"
        f"config hash: {config_hash}\n"
        f"manifest hash: {manifest.manifest_hash}\n"
    )
    artifact_path = tmp_path / "artifacts" / "preprocess.json"
    artifact = artifact_from_json(artifact_path.read_text(encoding="utf-8"), PreprocessArtifact)
    assert artifact.output_case_ids == manifest.case_ids
    for relative_path in [*artifact.output_image_paths, *artifact.output_label_paths]:
        assert (tmp_path / "preprocessed" / relative_path).is_file()


def test_preprocess_data_cli_corrupted_image_exits_nonzero_without_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, _manifest_path, _validation_path = _create_and_validate(tmp_path)
    corrupted_image_path = tmp_path / "data-root" / manifest.image_paths[0]
    _corrupt_nifti_header(corrupted_image_path)
    corrupted_image_bytes = corrupted_image_path.read_bytes()
    with pytest.raises(ImageFileError):
        nib.load(str(corrupted_image_path))
    config_path = Path("configs/experiment/phase1.yaml").resolve()
    monkeypatch.chdir(tmp_path)

    result = _invoke_preprocess_data(*_preprocess_args(config_path))
    error_output = _assert_nonzero_without_traceback(result)

    assert "failed input NIfTI validation" in error_output
    assert not (tmp_path / "preprocessed").exists()
    assert not (tmp_path / "artifacts" / "preprocess.json").exists()
    assert corrupted_image_path.read_bytes() == corrupted_image_bytes


def test_preprocess_data_cli_inconsistent_validation_artifact_exits_nonzero_without_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _manifest, _manifest_path, validation_path = _create_and_validate(tmp_path)
    payload = json.loads(validation_path.read_text(encoding="utf-8"))
    payload["manifest_hash"] = "1" * 64
    validation_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    config_path = Path("configs/experiment/phase1.yaml").resolve()
    monkeypatch.chdir(tmp_path)

    result = _invoke_preprocess_data(*_preprocess_args(config_path))
    error_output = _assert_nonzero_without_traceback(result)

    assert "manifest hash" in error_output


def test_preprocess_data_cli_nonempty_output_root_exits_nonzero_without_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _manifest, _manifest_path, _validation_path = _create_and_validate(tmp_path)
    output_root = tmp_path / "preprocessed"
    output_root.mkdir()
    (output_root / "existing.txt").write_text("occupied\n", encoding="utf-8")
    config_path = Path("configs/experiment/phase1.yaml").resolve()
    monkeypatch.chdir(tmp_path)

    result = _invoke_preprocess_data(*_preprocess_args(config_path))
    error_output = _assert_nonzero_without_traceback(result)

    assert "output root must be empty or nonexistent" in error_output


def test_preprocess_data_cli_existing_artifact_output_exits_nonzero_without_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _manifest, _manifest_path, _validation_path = _create_and_validate(tmp_path)
    artifact_output_path = tmp_path / "artifacts" / "preprocess.json"
    artifact_output_path.write_text("{}\n", encoding="utf-8")
    config_path = Path("configs/experiment/phase1.yaml").resolve()
    monkeypatch.chdir(tmp_path)

    result = _invoke_preprocess_data(*_preprocess_args(config_path))
    error_output = _assert_nonzero_without_traceback(result)

    assert "overwrite" in error_output
