"""Integration tests for the Phase 5 retrieval CLI."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner, Result

from protoem_ct.cli.main import app
from protoem_ct.retrieval import (
    PHASE5_COMPARISON_JSON_NAME,
    PHASE5_COMPARISON_MARKDOWN_NAME,
    PHASE5_EFFECTIVE_CONFIG_JSON_NAME,
    PHASE5_RUN_SUMMARY_JSON_NAME,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _invoke(*args: str) -> Result:
    runner = CliRunner()
    return runner.invoke(app, [*args])


def test_cli_help_exposes_phase5_command() -> None:
    result = _invoke("--help")

    assert result.exit_code == 0
    assert "run-phase5-retrieval" in result.output


def test_successful_synthetic_cli_run(tmp_path: Path) -> None:
    output_root = tmp_path / "published"
    result = _invoke(
        "run-phase5-retrieval",
        "--config",
        str(REPO_ROOT / "configs" / "phase5_foundation_retrieval.yaml"),
        "--output-root",
        str(output_root),
    )

    assert result.exit_code == 0, result.output
    assert "Phase 5 retrieval comparison success" in result.output
    assert "mode: synthetic_only" in result.output
    assert (output_root / PHASE5_COMPARISON_JSON_NAME).exists()
    assert (output_root / PHASE5_COMPARISON_MARKDOWN_NAME).exists()
    assert (output_root / PHASE5_RUN_SUMMARY_JSON_NAME).exists()
    assert (output_root / PHASE5_EFFECTIVE_CONFIG_JSON_NAME).exists()


def test_cli_failure_returns_nonzero_for_unsafe_output_root(tmp_path: Path) -> None:
    unsafe_root = REPO_ROOT / "src" / "phase5-forbidden-output"
    result = _invoke(
        "run-phase5-retrieval",
        "--config",
        str(REPO_ROOT / "configs" / "phase5_foundation_retrieval.yaml"),
        "--output-root",
        str(unsafe_root),
    )

    assert result.exit_code != 0
    assert "Phase 5 retrieval error:" in result.output
