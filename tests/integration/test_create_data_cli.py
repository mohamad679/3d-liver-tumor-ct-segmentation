"""Integration tests for the create-data CLI command."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from protoem_ct.cli.main import app

GIT_COMMIT = "f0b9066"
CREATED_AT_UTC = "2026-07-25T00:00:00Z"


def _invoke_create_data(*args: str) -> Any:
    """Invoke the public create-data CLI command."""
    runner = CliRunner()
    return runner.invoke(app, ["create-data", *args])


def test_create_data_cli_success_generates_expected_files(tmp_path: Path) -> None:
    output_root = tmp_path / "output"

    result = _invoke_create_data(
        "--config",
        "configs/data/synthetic.yaml",
        "--output-root",
        str(output_root),
        "--git-commit",
        GIT_COMMIT,
        "--created-at-utc",
        CREATED_AT_UTC,
    )

    assert result.exit_code == 0
    assert "create-data success" in result.output
    assert "case count: 3" in result.output
    assert "manifest path: generated/phase1/data/synthetic_manifest.json" in result.output
    assert "config hash: " in result.output
    assert "manifest hash: " in result.output

    manifest_path = output_root / "generated/phase1/data/synthetic_manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_files = [*manifest["image_paths"], *manifest["label_paths"]]
    assert len(expected_files) == 6
    for relative_path in expected_files:
        assert (output_root / relative_path).is_file()


def test_create_data_cli_invalid_config_exits_nonzero_without_traceback(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(
        json.dumps(
            {
                "dataset_id": "",
                "seed": 1729,
                "case_count": 3,
                "shape": [8, 8, 6],
                "spacing": [1.5, 1.5, 2.0],
                "generated_data_root": "generated/phase1/data",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    result = _invoke_create_data(
        "--config",
        str(config_path),
        "--output-root",
        str(tmp_path / "output"),
        "--git-commit",
        GIT_COMMIT,
        "--created-at-utc",
        CREATED_AT_UTC,
    )
    error_output = getattr(result, "stderr", "") or result.output

    assert result.exit_code != 0
    assert "dataset_id" in error_output
    assert "Traceback" not in result.output
    assert "Traceback" not in error_output


def test_create_data_cli_nonempty_output_root_exits_nonzero_without_traceback(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "output"
    output_root.mkdir()
    (output_root / "existing.txt").write_text("occupied\n", encoding="utf-8")

    result = _invoke_create_data(
        "--config",
        "configs/data/synthetic.yaml",
        "--output-root",
        str(output_root),
        "--git-commit",
        GIT_COMMIT,
        "--created-at-utc",
        CREATED_AT_UTC,
    )
    error_output = getattr(result, "stderr", "") or result.output

    assert result.exit_code != 0
    assert "output root must be empty or nonexistent" in error_output
    assert "Traceback" not in result.output
    assert "Traceback" not in error_output
