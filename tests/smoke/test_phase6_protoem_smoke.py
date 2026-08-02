"""Bounded CPU smoke test for the synthetic Phase 6 ProtoEM path."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from protoem_ct.cli.main import app
from protoem_ct.protoem import (
    PHASE6_ABLATION_COMPARISON_JSON_NAME,
    PHASE6_ABLATION_COMPARISON_MARKDOWN_NAME,
    PHASE6_ABLATION_PLAN_NAME,
    PHASE6_ABLATION_RUN_INVENTORY_JSON_NAME,
    PHASE6_CONVERGENCE_PLOT_NAME,
    PHASE6_EFFECTIVE_CONFIG_NAME,
    PHASE6_FINAL_INFERENCE_NAME,
    PHASE6_INITIALIZATION_SUMMARY_NAME,
    PHASE6_OBJECTIVE_TRACE_NAME,
    PHASE6_RUN_SUMMARY_NAME,
    PHASE6_STOPPING_RECORD_NAME,
    PHASE6_SUMMARY_MARKDOWN_NAME,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_phase6_protoem_smoke(tmp_path: Path) -> None:
    runner = CliRunner()
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    args = [
        "run-phase6-protoem",
        "--config",
        str(REPO_ROOT / "configs" / "phase6_protoem_ct.yaml"),
    ]

    first = runner.invoke(app, [*args, "--output-root", str(first_root)])
    second = runner.invoke(app, [*args, "--output-root", str(second_root)])

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    for filename in (
        PHASE6_EFFECTIVE_CONFIG_NAME,
        PHASE6_INITIALIZATION_SUMMARY_NAME,
        PHASE6_OBJECTIVE_TRACE_NAME,
        PHASE6_STOPPING_RECORD_NAME,
        PHASE6_RUN_SUMMARY_NAME,
        PHASE6_ABLATION_PLAN_NAME,
        PHASE6_ABLATION_COMPARISON_JSON_NAME,
        PHASE6_ABLATION_COMPARISON_MARKDOWN_NAME,
        PHASE6_ABLATION_RUN_INVENTORY_JSON_NAME,
        PHASE6_SUMMARY_MARKDOWN_NAME,
        PHASE6_FINAL_INFERENCE_NAME,
    ):
        assert (first_root / filename).read_bytes() == (second_root / filename).read_bytes()
    assert (first_root / PHASE6_CONVERGENCE_PLOT_NAME).exists()
