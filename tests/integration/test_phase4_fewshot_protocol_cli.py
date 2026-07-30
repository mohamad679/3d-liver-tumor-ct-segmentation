"""Integration tests for the Phase 4 few-shot protocol CLI."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from protoem_ct.artifacts import phase2_artifact_to_json
from protoem_ct.cli.main import app
from protoem_ct.fewshot import fewshot_protocol_table_from_json


def _load_helpers() -> Any:
    helper_path = Path(__file__).parents[1] / "unit" / "test_fewshot_protocol.py"
    spec = importlib.util.spec_from_file_location("_fewshot_protocol_helpers", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("few-shot protocol helpers could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_HELPERS = _load_helpers()
REPO_ROOT = Path(__file__).resolve().parents[2]


def _invoke(*args: str) -> Any:
    runner = CliRunner()
    return runner.invoke(app, [*args])


def _write_phase2_fixture(root: Path) -> tuple[Path, Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    manifest = _HELPERS._manifest()
    split = _HELPERS._split(manifest)
    lesion = _HELPERS._lesion_artifact(manifest, split)
    manifest_path = root / "manifest.json"
    split_path = root / "split.json"
    lesion_path = root / "lesion.json"
    manifest_path.write_text(phase2_artifact_to_json(manifest), encoding="utf-8")
    split_path.write_text(phase2_artifact_to_json(split), encoding="utf-8")
    lesion_path.write_text(phase2_artifact_to_json(lesion), encoding="utf-8")
    return manifest_path, split_path, lesion_path


def _command_args(
    manifest_path: Path,
    split_path: Path,
    lesion_path: Path,
    output_root: Path,
) -> list[str]:
    return [
        "generate-phase4-fewshot-protocol",
        "--manifest",
        str(manifest_path),
        "--split",
        str(split_path),
        "--lesion-artifact",
        str(lesion_path),
        "--output-root",
        str(output_root),
        "--initialization-reference-type",
        "baseline_provenance",
        "--initialization-reference-identifier",
        "baseline_init_001",
        "--initialization-artifact-sha256",
        "1" * 64,
        "--base-seed",
        "1729",
        "--config",
        str(REPO_ROOT / "configs" / "phase4_fewshot_protocol.yaml"),
    ]


def _markdown_data_row_count(markdown_path: Path) -> int:
    return (
        sum(
            1
            for line in markdown_path.read_text(encoding="utf-8").splitlines()
            if line.startswith("| ")
        )
        - 2
    )


def test_cli_help_exposes_phase4_protocol_command() -> None:
    result = _invoke("--help")

    assert result.exit_code == 0
    assert "generate-phase4-fewshot-protocol" in result.output


def test_cli_generates_deterministic_protocol_artifacts_across_two_output_roots(
    tmp_path: Path,
) -> None:
    manifest_path, split_path, lesion_path = _write_phase2_fixture(tmp_path / "inputs")
    first_root = tmp_path / "first-output"
    second_root = tmp_path / "second-output"
    tracked_before = {
        path.relative_to(REPO_ROOT).as_posix(): path.read_bytes()
        for directory_name in ("data", "reports", "src")
        for path in (REPO_ROOT / directory_name).rglob("*")
        if path.is_file() and path.suffix in {".json", ".md", ".py", ".yaml"}
    }

    first = _invoke(*_command_args(manifest_path, split_path, lesion_path, first_root))
    second = _invoke(*_command_args(manifest_path, split_path, lesion_path, second_root))

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert "Phase 4 few-shot protocol generation success" in first.output
    assert "support manifest count: 15" in first.output
    assert "adaptation config count: 45" in first.output
    assert "protocol row count: 45" in first.output
    assert "leakage check passed: true" in first.output

    first_support_files = sorted(first_root.glob("support_manifests/k*/**/*.json"))
    second_support_files = sorted(second_root.glob("support_manifests/k*/**/*.json"))
    assert len(first_support_files) == 15
    assert len(second_support_files) == 15

    first_config_files = sorted((first_root / "adaptation_configs").glob("*.json"))
    second_config_files = sorted((second_root / "adaptation_configs").glob("*.json"))
    assert len(first_config_files) == 45
    assert len(second_config_files) == 45

    first_protocol = fewshot_protocol_table_from_json(
        (first_root / "protocol" / "fewshot_protocol_table.json").read_bytes()
    )
    second_protocol = fewshot_protocol_table_from_json(
        (second_root / "protocol" / "fewshot_protocol_table.json").read_bytes()
    )
    assert len(first_protocol.rows) == 45
    assert len(second_protocol.rows) == 45
    assert _markdown_data_row_count(first_root / "protocol" / "fewshot_protocol_table.md") == 45
    assert _markdown_data_row_count(second_root / "protocol" / "fewshot_protocol_table.md") == 45

    first_summary = json.loads((first_root / "generation_summary.json").read_text(encoding="utf-8"))
    second_summary = json.loads(
        (second_root / "generation_summary.json").read_text(encoding="utf-8")
    )
    assert first_summary["support_manifest_count"] == 15
    assert first_summary["adaptation_config_count"] == 45
    assert first_summary["protocol_row_count"] == 45
    assert first_summary["leakage_check_passed"] is True
    assert first_summary["internal_test_patient_overlap_count"] == 0
    assert first_summary["internal_test_case_overlap_count"] == 0
    assert first_summary == second_summary

    first_bytes = {
        path.relative_to(first_root).as_posix(): path.read_bytes()
        for path in first_root.rglob("*")
        if path.is_file()
    }
    second_bytes = {
        path.relative_to(second_root).as_posix(): path.read_bytes()
        for path in second_root.rglob("*")
        if path.is_file()
    }
    assert first_bytes == second_bytes

    tracked_after = {
        path.relative_to(REPO_ROOT).as_posix(): path.read_bytes()
        for directory_name in ("data", "reports", "src")
        for path in (REPO_ROOT / directory_name).rglob("*")
        if path.is_file() and path.suffix in {".json", ".md", ".py", ".yaml"}
    }
    assert tracked_before == tracked_after
