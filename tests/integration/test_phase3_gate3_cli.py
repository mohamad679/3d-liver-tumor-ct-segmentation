from __future__ import annotations

import importlib
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

import pytest
from typer.testing import CliRunner

requires_baseline_environment = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None or importlib.util.find_spec("monai") is None,
    reason="Phase 3 Gate 3 integration test requires the isolated baseline environment.",
)


def _run_cli_module(*arguments: str) -> subprocess.CompletedProcess[str]:
    """Run the CLI through the current Python interpreter's module entry point."""
    return subprocess.run(
        [sys.executable, "-m", "protoem_ct.cli.main", *arguments],
        capture_output=True,
        text=True,
        check=False,
    )


def _run_packaged_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    """Run the packaged CLI script installed beside the current Python interpreter."""
    executable = Path(sys.executable).with_name("protoem-ct")
    return subprocess.run(
        [str(executable), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )


def test_importing_cli_main_does_not_import_baseline_runtime_dependencies() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import protoem_ct.cli.main; "
                "forbidden = ('torch', 'monai', 'nnunetv2', 'mlflow'); "
                "assert all(name not in sys.modules for name in forbidden)"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert result.stderr == ""


def test_cli_module_and_packaged_entrypoint_expose_gate3_command() -> None:
    module_result = _run_cli_module("--help")
    packaged_result = _run_packaged_cli("--help")

    assert module_result.returncode == 0, module_result.stderr
    assert packaged_result.returncode == 0, packaged_result.stderr
    assert module_result.stdout.strip()
    assert packaged_result.stdout.strip()
    assert "run-phase3-gate3-synthetic" in module_result.stdout
    assert "run-phase3-gate3-synthetic" in packaged_result.stdout


def test_cli_module_invalid_command_returns_nonzero() -> None:
    result = _run_cli_module("not-a-protoem-ct-command")

    assert result.returncode != 0
    assert result.stdout or result.stderr


def test_importing_gate3_does_not_import_torch() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import importlib, sys; "
                "importlib.import_module('protoem_ct.baselines.gate3'); "
                "forbidden = ('torch', 'mlflow'); "
                "assert all(name not in sys.modules for name in forbidden)"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_gate3_cli_calls_and_catches_concrete_gate3_definitions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from protoem_ct.baselines import gate3 as gate3_module
    from protoem_ct.cli import main as cli_main

    calls: list[dict[str, object]] = []

    def fail_with_gate3_error(**kwargs: object) -> NoReturn:
        calls.append(kwargs)
        raise gate3_module.Phase3Gate3Error("expected test failure")

    monkeypatch.setattr(gate3_module, "run_phase3_gate3_synthetic", fail_with_gate3_error)
    result = CliRunner().invoke(
        cli_main.app,
        [
            "run-phase3-gate3-synthetic",
            "--output-root",
            str(tmp_path / "phase3-gate3-output"),
            "--git-commit",
            "8701699171e140bcd80041810417390c889ca484",
            "--start-timestamp",
            "2026-07-29T00:00:00Z",
            "--end-timestamp",
            "2026-07-29T01:00:00Z",
        ],
    )

    assert result.exit_code == 1
    assert "Phase 3 Gate 3 error: Phase3Gate3Error" in result.output
    assert len(calls) == 1
    assert calls[0]["run_identifier"] == "phase3_gate3_synthetic"


@requires_baseline_environment
def test_gate3_lazy_torch_loader_resolves_installed_torch() -> None:
    from protoem_ct.baselines.gate3 import _import_torch

    torch = _import_torch()

    assert torch.__name__ == "torch"


@requires_baseline_environment
def test_phase3_gate3_cli_smoke(tmp_path: Path) -> None:
    output_root = tmp_path / "phase3-gate3-output"
    result = _run_cli_module(
        "run-phase3-gate3-synthetic",
        "--output-root",
        str(output_root),
        "--git-commit",
        "8701699171e140bcd80041810417390c889ca484",
        "--start-timestamp",
        "2026-07-29T00:00:00Z",
        "--end-timestamp",
        "2026-07-29T01:00:00Z",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout
    assert "report version: phase3_gate3_report_v1" in result.stdout
    assert output_root.exists()
    shutil.rmtree(output_root)
