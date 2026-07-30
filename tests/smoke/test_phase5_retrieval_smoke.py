"""Bounded CPU smoke test for the synthetic Phase 5 retrieval path."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from protoem_ct.cli.main import app
from protoem_ct.retrieval import (
    PHASE5_COMPARISON_JSON_NAME,
    PHASE5_COMPARISON_MARKDOWN_NAME,
    PHASE5_EFFECTIVE_CONFIG_JSON_NAME,
    PHASE5_RUN_SUMMARY_JSON_NAME,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_phase5_retrieval_smoke(tmp_path: Path) -> None:
    runner = CliRunner()
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    args = [
        "run-phase5-retrieval",
        "--config",
        str(REPO_ROOT / "configs" / "phase5_foundation_retrieval.yaml"),
    ]

    first = runner.invoke(app, [*args, "--output-root", str(first_root)])
    second = runner.invoke(app, [*args, "--output-root", str(second_root)])

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    for filename in (
        PHASE5_COMPARISON_JSON_NAME,
        PHASE5_COMPARISON_MARKDOWN_NAME,
        PHASE5_RUN_SUMMARY_JSON_NAME,
        PHASE5_EFFECTIVE_CONFIG_JSON_NAME,
    ):
        assert (first_root / filename).read_bytes() == (second_root / filename).read_bytes()
