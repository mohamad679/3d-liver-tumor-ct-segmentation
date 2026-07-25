"""Integration tests for the validate-data CLI command."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import nibabel as nib
import numpy as np
from typer.testing import CliRunner

from protoem_ct.artifacts import SyntheticManifest, ValidationArtifact, artifact_from_json
from protoem_ct.cli.main import app
from protoem_ct.data.synthetic import create_synthetic_dataset, load_synthetic_config

GIT_COMMIT = "f0b34b3"
CREATED_AT_UTC = "2026-07-25T01:00:00Z"
VALIDATION_CREATED_AT_UTC = "2026-07-25T01:05:00Z"


def _invoke_validate_data(*args: str) -> Any:
    """Invoke the public validate-data CLI command."""
    runner = CliRunner()
    return runner.invoke(app, ["validate-data", *args])


def _create_dataset(tmp_path: Path) -> tuple[Path, Path, SyntheticManifest]:
    config = load_synthetic_config(Path("configs/data/synthetic.yaml"))
    result = create_synthetic_dataset(
        config,
        output_root=tmp_path / "data-root",
        git_commit=GIT_COMMIT,
        created_at_utc=CREATED_AT_UTC,
    )
    return result.manifest_path, tmp_path / "data-root", result.manifest


def _validate_args(manifest_path: Path, data_root: Path, output_path: Path) -> list[str]:
    return [
        "--manifest",
        str(manifest_path),
        "--data-root",
        str(data_root),
        "--output",
        str(output_path),
        "--git-commit",
        GIT_COMMIT,
        "--created-at-utc",
        VALIDATION_CREATED_AT_UTC,
    ]


def _error_output(result: Any) -> str:
    return cast(str, getattr(result, "stderr", "") or result.output)


def test_validate_data_cli_success_writes_readable_artifact(tmp_path: Path) -> None:
    manifest_path, data_root, manifest = _create_dataset(tmp_path)
    output_path = tmp_path / "artifacts" / "validation.json"

    result = _invoke_validate_data(*_validate_args(manifest_path, data_root, output_path))

    assert result.exit_code == 0
    assert result.output == (
        "validation success\n"
        "valid case count: 3\n"
        f"validation artifact path: {output_path}\n"
        f"config hash: {manifest.config_hash}\n"
        f"manifest hash: {manifest.manifest_hash}\n"
    )
    assert output_path.is_file()
    artifact = artifact_from_json(output_path.read_text(encoding="utf-8"), ValidationArtifact)
    assert artifact.valid_case_count == 3
    assert artifact.invalid_case_count == 0
    assert artifact.validated_case_ids == manifest.case_ids


def test_validate_data_cli_corrupted_label_exits_nonzero_without_traceback(
    tmp_path: Path,
) -> None:
    manifest_path, data_root, manifest = _create_dataset(tmp_path)
    label_path = data_root / manifest.label_paths[0]
    label = cast(Any, nib.load(str(label_path)))
    image = nib.Nifti1Image(  # type: ignore[no-untyped-call]
        np.full(manifest.shape, 2, dtype=np.uint8),
        label.affine,
    )
    nib.save(image, str(label_path))

    result = _invoke_validate_data(
        *_validate_args(manifest_path, data_root, tmp_path / "artifacts" / "validation.json")
    )
    error_output = _error_output(result)

    assert result.exit_code != 0
    assert "failed NIfTI validation" in error_output
    assert "Traceback" not in result.output
    assert "Traceback" not in error_output


def test_validate_data_cli_altered_manifest_exits_nonzero_without_traceback(
    tmp_path: Path,
) -> None:
    manifest_path, data_root, _manifest = _create_dataset(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["dataset_id"] = "altered"
    manifest_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    result = _invoke_validate_data(
        *_validate_args(manifest_path, data_root, tmp_path / "artifacts" / "validation.json")
    )
    error_output = _error_output(result)

    assert result.exit_code != 0
    assert "hash mismatch" in error_output
    assert "Traceback" not in result.output
    assert "Traceback" not in error_output


def test_validate_data_cli_existing_output_exits_nonzero_without_traceback(
    tmp_path: Path,
) -> None:
    manifest_path, data_root, _manifest = _create_dataset(tmp_path)
    output_path = tmp_path / "artifacts" / "validation.json"
    output_path.parent.mkdir()
    output_path.write_text("{}\n", encoding="utf-8")

    result = _invoke_validate_data(*_validate_args(manifest_path, data_root, output_path))
    error_output = _error_output(result)

    assert result.exit_code != 0
    assert "overwrite" in error_output
    assert "Traceback" not in result.output
    assert "Traceback" not in error_output
