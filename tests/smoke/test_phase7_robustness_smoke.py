"""Smoke test for the bounded CPU Phase 7 synthetic workflow."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from protoem_ct.cli.main import app

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_phase7_synthetic_smoke_completes(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "run-phase7-robustness-uncertainty",
            "--config",
            str(REPO_ROOT / "configs" / "phase7_robustness_uncertainty.yaml"),
            "--output-root",
            str(tmp_path / "phase7-smoke"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Phase 7 robustness/uncertainty success" in result.output
    assert (tmp_path / "phase7-smoke" / "phase7_run_summary.json").exists()
