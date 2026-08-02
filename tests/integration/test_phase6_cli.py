"""Integration tests for the Phase 6 ProtoEM-CT CLI."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner, Result

from protoem_ct.cli.main import app
from protoem_ct.protoem import (
    PHASE6_ABLATION_PLAN_NAME,
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


def _invoke(*args: str) -> Result:
    return CliRunner().invoke(app, [*args])


def _write_positive_step_config(path: Path) -> None:
    path.write_text(
        "\n".join(
            (
                "phase6_protoem_ct:",
                "  schema_version: v1",
                "  synthetic_mode_only: true",
                "  explicit_seed: 1729",
                "  max_iterations: 1",
                "  minimum_iterations: 1",
                "  convergence_tolerance: 1.0e-6",
                "  temperature: 1.0",
                "  confidence_threshold: 0.55",
                "  foreground_prior: 0.5",
                "  update_schedule: learned_positive_step",
                "  prototype_mode: single_prototype",
                "  objective_weights:",
                "    support: 1.0",
                "    query_entropy: 1.0",
                "    class_balance: 1.0",
                "    consistency: 1.0",
                "    proximal: 1.0",
                "  phase5_initialization_schema_name: phase5_run_summary",
                "  phase5_prototype_schema_name: prototype_memory",
                "  phase5_inference_schema_name: prototype_only_inference_result",
                "  positive_step_schedule:",
                "    epsilon: 1.0e-6",
                "    raw_step_parameters: [0.0]",
                "",
            )
        ),
        encoding="utf-8",
    )


def test_cli_help_exposes_phase6_command() -> None:
    result = _invoke("--help")

    assert result.exit_code == 0
    assert "run-phase6-protoem" in result.output


def test_successful_fixed_em_like_synthetic_cli_run(tmp_path: Path) -> None:
    output_root = tmp_path / "published"
    result = _invoke(
        "run-phase6-protoem",
        "--config",
        str(REPO_ROOT / "configs" / "phase6_protoem_ct.yaml"),
        "--output-root",
        str(output_root),
    )

    assert result.exit_code == 0, result.output
    assert "Phase 6 ProtoEM success" in result.output
    assert "mode: synthetic_only" in result.output
    for filename in (
        PHASE6_EFFECTIVE_CONFIG_NAME,
        PHASE6_INITIALIZATION_SUMMARY_NAME,
        PHASE6_OBJECTIVE_TRACE_NAME,
        PHASE6_STOPPING_RECORD_NAME,
        PHASE6_RUN_SUMMARY_NAME,
        PHASE6_ABLATION_PLAN_NAME,
        PHASE6_CONVERGENCE_PLOT_NAME,
        PHASE6_SUMMARY_MARKDOWN_NAME,
        PHASE6_FINAL_INFERENCE_NAME,
    ):
        assert (output_root / filename).exists()


def test_successful_explicit_positive_step_cli_run(tmp_path: Path) -> None:
    config_path = tmp_path / "phase6_positive.yaml"
    _write_positive_step_config(config_path)
    output_root = tmp_path / "published"

    result = _invoke(
        "run-phase6-protoem",
        "--config",
        str(config_path),
        "--output-root",
        str(output_root),
    )

    assert result.exit_code == 0, result.output
    assert "execution status: completed" in result.output
    assert (output_root / PHASE6_FINAL_INFERENCE_NAME).exists()


def test_cli_failure_returns_nonzero_for_unsafe_output_root() -> None:
    unsafe_root = REPO_ROOT / "src" / "phase6-forbidden-output"
    result = _invoke(
        "run-phase6-protoem",
        "--config",
        str(REPO_ROOT / "configs" / "phase6_protoem_ct.yaml"),
        "--output-root",
        str(unsafe_root),
    )

    assert result.exit_code != 0
    assert "Phase 6 ProtoEM error:" in result.output
