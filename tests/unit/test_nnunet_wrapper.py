from __future__ import annotations

import importlib
import os
import stat
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from protoem_ct.baselines import _log_paths
from protoem_ct.baselines.nnunet import (
    NNUNET_ALLOWED_INHERITED_ENVIRONMENT_KEYS,
    NNUNET_V2_DEFAULT_CHECKPOINT_NAME,
    NNUNET_V2_DEFAULT_PLANS_IDENTIFIER,
    NNUNET_V2_DEFAULT_TRAINER,
    NNUNET_V2_EXECUTABLE_PLAN_AND_PREPROCESS,
    NNUNET_V2_EXECUTABLE_PREDICT,
    NNUNET_V2_EXECUTABLE_TRAIN,
    NnUNetExecutionError,
    NnUNetWrapperValidationError,
    _redact_log_content,
    build_nnunet_v2_plan_and_preprocess_command,
    build_nnunet_v2_predict_command,
    build_nnunet_v2_run_config,
    build_nnunet_v2_runtime_environment,
    build_nnunet_v2_train_command,
    execute_nnunet_command,
)
from protoem_ct.baselines.paths import validate_baseline_run_root


def _runtime_paths(tmp_path: Path) -> tuple[Path, Any]:
    raw_root = tmp_path / "raw-root"
    dataset_root = raw_root / "Dataset901_ProtoEMCTSynthetic" / "imagesTs"
    dataset_root.mkdir(parents=True)
    run_parent = tmp_path / "runs"
    run_parent.mkdir()
    run_paths = validate_baseline_run_root(run_parent / "run-001", baseline_family="nnunet_v2")
    return raw_root, run_paths


def _make_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _runtime_environment_with_path(bin_dir: Path) -> dict[str, str]:
    return {"PATH": f"{bin_dir}:{os.environ['PATH']}"}


def test_exact_locally_verified_command_tuples_and_determinism(tmp_path: Path) -> None:
    run_config = build_nnunet_v2_run_config(seed=17)
    raw_root, run_paths = _runtime_paths(tmp_path)

    plan_command = build_nnunet_v2_plan_and_preprocess_command(run_config)
    train_command = build_nnunet_v2_train_command(run_config)
    predict_command = build_nnunet_v2_predict_command(
        run_config,
        input_images_directory=raw_root / "Dataset901_ProtoEMCTSynthetic" / "imagesTs",
        output_predictions_directory=run_paths.predictions_dir / "nnunet_predictions",
    )

    assert plan_command == (
        NNUNET_V2_EXECUTABLE_PLAN_AND_PREPROCESS,
        "-d",
        "901",
        "--verify_dataset_integrity",
        "-pl",
        "ExperimentPlanner",
        "-overwrite_plans_name",
        NNUNET_V2_DEFAULT_PLANS_IDENTIFIER,
        "-c",
        "3d_fullres",
    )
    assert train_command == (
        NNUNET_V2_EXECUTABLE_TRAIN,
        "901",
        "3d_fullres",
        "0",
        "-tr",
        NNUNET_V2_DEFAULT_TRAINER,
        "-p",
        NNUNET_V2_DEFAULT_PLANS_IDENTIFIER,
        "-device",
        "cpu",
    )
    assert predict_command == (
        NNUNET_V2_EXECUTABLE_PREDICT,
        "-i",
        str(raw_root / "Dataset901_ProtoEMCTSynthetic" / "imagesTs"),
        "-o",
        str(run_paths.predictions_dir / "nnunet_predictions"),
        "-d",
        "901",
        "-p",
        NNUNET_V2_DEFAULT_PLANS_IDENTIFIER,
        "-tr",
        NNUNET_V2_DEFAULT_TRAINER,
        "-c",
        "3d_fullres",
        "-f",
        "0",
        "-chk",
        NNUNET_V2_DEFAULT_CHECKPOINT_NAME,
        "-device",
        "cpu",
    )
    assert build_nnunet_v2_train_command(run_config) == train_command


def test_fixed_executable_names_and_invalid_identifier_rejection() -> None:
    run_config = build_nnunet_v2_run_config()
    assert (
        build_nnunet_v2_plan_and_preprocess_command(run_config)[0]
        == NNUNET_V2_EXECUTABLE_PLAN_AND_PREPROCESS
    )
    assert build_nnunet_v2_train_command(run_config)[0] == NNUNET_V2_EXECUTABLE_TRAIN

    with pytest.raises(NnUNetWrapperValidationError):
        build_nnunet_v2_run_config(trainer_class="bad\ntrainer")


def test_sanitized_environment_allowlist_and_credential_exclusion(tmp_path: Path) -> None:
    raw_root, run_paths = _runtime_paths(tmp_path)
    inherited = {
        "PATH": "/usr/bin:/bin",
        "HOME": "/Users/example",
        "LANG": "en_US.UTF-8",
        "TMPDIR": "/tmp",
        "AWS_SECRET_ACCESS_KEY": "secret",
        "OPENAI_API_KEY": "secret",
    }
    runtime = build_nnunet_v2_runtime_environment(
        raw_root=raw_root,
        run_paths=run_paths,
        inherited_environment=inherited,
    )

    assert set(runtime.environment).issuperset(
        {
            "PATH",
            "HOME",
            "LANG",
            "nnUNet_raw",
            "nnUNet_preprocessed",
            "nnUNet_results",
        }
    )
    assert "AWS_SECRET_ACCESS_KEY" not in runtime.environment
    assert "OPENAI_API_KEY" not in runtime.environment
    assert set(runtime.environment).issubset(
        set(NNUNET_ALLOWED_INHERITED_ENVIRONMENT_KEYS)
        | {"nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"}
    )


def test_path_constraints_reject_repository_contained_raw_root(tmp_path: Path) -> None:
    run_parent = tmp_path / "runs"
    run_parent.mkdir()
    run_paths = validate_baseline_run_root(run_parent / "run-001", baseline_family="nnunet_v2")

    with pytest.raises(NnUNetWrapperValidationError):
        build_nnunet_v2_runtime_environment(
            raw_root=Path(__file__).resolve().parents[2],
            run_paths=run_paths,
        )


def test_executor_timeout_nonzero_missing_output_and_log_no_overwrite(tmp_path: Path) -> None:
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    success_script = bin_dir / NNUNET_V2_EXECUTABLE_TRAIN
    _make_executable(
        success_script,
        (
            "#!/usr/bin/env python3\n"
            "from pathlib import Path\n"
            "import sys\n"
            "Path(sys.argv[1]).write_text('ok\\n', encoding='utf-8')\n"
        ),
    )
    output_path = tmp_path / "output.txt"
    result = execute_nnunet_command(
        (NNUNET_V2_EXECUTABLE_TRAIN, str(output_path)),
        runtime_environment=_runtime_environment_with_path(bin_dir),
        working_directory=workdir,
        logs_directory=logs_dir,
        timeout_seconds=5.0,
        stage_name="custom-success",
        required_output_paths=(output_path,),
    )
    assert result.return_code == 0
    assert result.stage_status == "completed"
    with pytest.raises(NnUNetExecutionError) as exc_info:
        execute_nnunet_command(
            (NNUNET_V2_EXECUTABLE_TRAIN, str(output_path)),
            runtime_environment=_runtime_environment_with_path(bin_dir),
            working_directory=workdir,
            logs_directory=logs_dir,
            timeout_seconds=5.0,
            stage_name="custom-success",
            required_output_paths=(),
        )
    assert exc_info.value.failure_code == "nnunet_output_collision"

    missing_script = bin_dir / NNUNET_V2_EXECUTABLE_PLAN_AND_PREPROCESS
    _make_executable(missing_script, "#!/usr/bin/env python3\nprint('ok')\n")
    missing_logs = tmp_path / "logs-missing"
    missing_logs.mkdir()
    with pytest.raises(NnUNetExecutionError) as exc_info:
        execute_nnunet_command(
            (NNUNET_V2_EXECUTABLE_PLAN_AND_PREPROCESS,),
            runtime_environment=_runtime_environment_with_path(bin_dir),
            working_directory=workdir,
            logs_directory=missing_logs,
            timeout_seconds=5.0,
            stage_name="custom-missing",
            required_output_paths=(tmp_path / "never-created.txt",),
        )
    assert exc_info.value.failure_code == "nnunet_output_missing"

    nonzero_script = bin_dir / NNUNET_V2_EXECUTABLE_PREDICT
    _make_executable(nonzero_script, "#!/usr/bin/env python3\nimport sys\nsys.exit(3)\n")
    nonzero_logs = tmp_path / "logs-nonzero"
    nonzero_logs.mkdir()
    with pytest.raises(NnUNetExecutionError) as exc_info:
        execute_nnunet_command(
            (NNUNET_V2_EXECUTABLE_PREDICT,),
            runtime_environment=_runtime_environment_with_path(bin_dir),
            working_directory=workdir,
            logs_directory=nonzero_logs,
            timeout_seconds=5.0,
            stage_name="custom-nonzero",
        )
    assert exc_info.value.failure_code == "nnunet_nonzero_exit"
    assert exc_info.value.return_code == 3

    timeout_script = bin_dir / NNUNET_V2_EXECUTABLE_PLAN_AND_PREPROCESS
    _make_executable(
        timeout_script,
        "#!/usr/bin/env python3\nimport time\ntime.sleep(2)\n",
    )
    timeout_logs = tmp_path / "logs-timeout"
    timeout_logs.mkdir()
    with pytest.raises(NnUNetExecutionError) as exc_info:
        execute_nnunet_command(
            (NNUNET_V2_EXECUTABLE_PLAN_AND_PREPROCESS,),
            runtime_environment=_runtime_environment_with_path(bin_dir),
            working_directory=workdir,
            logs_directory=timeout_logs,
            timeout_seconds=0.1,
            stage_name="custom-timeout",
        )
    assert exc_info.value.failure_code == "nnunet_timeout"


def test_executor_uses_subprocess_run_without_shell(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    nnunet_module = importlib.import_module("protoem_ct.baselines.nnunet")
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls: dict[str, Any] = {}

    class Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(*args: Any, **kwargs: Any) -> Completed:
        calls["args"] = args
        calls["kwargs"] = kwargs
        return Completed()

    monkeypatch.setattr(nnunet_module.subprocess, "run", fake_run)
    script = bin_dir / NNUNET_V2_EXECUTABLE_TRAIN
    _make_executable(script, "#!/usr/bin/env python3\n")

    execute_nnunet_command(
        (NNUNET_V2_EXECUTABLE_TRAIN,),
        runtime_environment=_runtime_environment_with_path(bin_dir),
        working_directory=workdir,
        logs_directory=logs_dir,
        timeout_seconds=5.0,
        stage_name="custom-shell-check",
    )

    assert calls["kwargs"]["shell"] is False


def test_log_redaction_replaces_known_roots_longest_first_and_deterministically(
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / "runtime-root"
    logs_dir = runtime_root / "logs"
    logs_dir.mkdir(parents=True)
    working_directory = runtime_root / "work"
    working_directory.mkdir()
    raw_root = runtime_root / "nnUNet_raw"
    preprocessed_root = raw_root / "preprocessed"
    results_root = runtime_root / "nnUNet_results"
    runtime_environment = {
        "nnUNet_raw": str(raw_root),
        "nnUNet_preprocessed": str(preprocessed_root),
        "nnUNet_results": str(results_root),
        "TMPDIR": str(runtime_root / "tmp"),
    }
    content = "\n".join(
        (
            str(runtime_root),
            str(raw_root),
            str(preprocessed_root),
            str(results_root),
            str(Path.home()),
            str(Path(sys.prefix).resolve()),
            str(Path(__file__).resolve().parents[2]),
            str(working_directory),
        )
    )

    first = _redact_log_content(
        content,
        runtime_environment=runtime_environment,
        working_directory=working_directory,
        logs_directory=logs_dir,
        log_redaction_roots=None,
    )
    second = _redact_log_content(
        content,
        runtime_environment=runtime_environment,
        working_directory=working_directory,
        logs_directory=logs_dir,
        log_redaction_roots=None,
    )

    assert first == second
    assert "<RUNTIME_ROOT>" in first
    assert "<NNUNET_RAW>" in first
    assert "<NNUNET_PREPROCESSED>" in first
    assert "<NNUNET_RESULTS>" in first
    assert "<HOME>" in first
    assert "<BASELINE_ENV>" in first
    assert "<REPOSITORY_ROOT>" in first
    assert "<TEMP_ROOT>" in first
    assert "<NNUNET_RAW>/preprocessed" not in first
    assert str(runtime_root) not in first
    assert str(Path.home()) not in first
    assert str(Path(sys.prefix).resolve()) not in first
    assert not _log_paths._contains_absolute_filesystem_path(first)
    assert first.endswith("\n")
    assert not first.endswith("\n\n")


def test_log_redaction_handles_residual_paths_tracebacks_spaces_and_uris(
    tmp_path: Path,
) -> None:
    logs_dir = tmp_path / "runtime" / "logs"
    logs_dir.mkdir(parents=True)
    working_directory = tmp_path / "runtime" / "work"
    working_directory.mkdir()
    unknown_path = "/opt/package/build/source.py"
    content = (
        f"unknown={unknown_path},\n"
        'File "/opt/package path/module.py", line 17, in execute\n'
        "documentation=https://github.com/MIC-DKFZ/nnUNet/blob/master/"
        "documentation/resenc_presets.md\n"
        "fallback=http://example.org/project/reference.html\n"
        "artifact=mlflow-artifacts:/phase3/run\n"
        "flag=--configuration=3d_fullres\n"
        "number=1/2\n\n\n"
    )

    first = _redact_log_content(
        content,
        runtime_environment={},
        working_directory=working_directory,
        logs_directory=logs_dir,
        log_redaction_roots=None,
    )
    second = _redact_log_content(
        content,
        runtime_environment={},
        working_directory=working_directory,
        logs_directory=logs_dir,
        log_redaction_roots=None,
    )

    assert first == second
    assert "unknown=<ABSOLUTE_PATH>," in first
    assert 'File "<ABSOLUTE_PATH>", line 17' in first
    assert "https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/resenc_presets.md" in first
    assert "http://example.org/project/reference.html" in first
    assert "mlflow-artifacts:/phase3/run" in first
    assert "--configuration=3d_fullres" in first
    assert "number=1/2" in first
    assert unknown_path not in first
    assert not _log_paths._contains_absolute_filesystem_path(first)
    assert first.endswith("\n")
    assert not first.endswith("\n\n")


@pytest.mark.parametrize(
    "value",
    [
        "https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/resenc_presets.md",
        "http://example.org/reference/path",
        "mlflow-artifacts:/phase3/run",
        "<RUNTIME_ROOT>/relative/output",
        "--output-root",
        "1/2",
    ],
)
def test_canonical_log_detector_accepts_non_file_values(value: str) -> None:
    assert not _log_paths._contains_absolute_filesystem_path(value)


@pytest.mark.parametrize(
    "value",
    [
        "/private/runtime/output.log",
        'File "/private/runtime path/module.py", line 17',
    ],
)
def test_canonical_log_detector_rejects_absolute_posix_paths(value: str) -> None:
    assert _log_paths._contains_absolute_filesystem_path(value)


def test_executor_raises_typed_failure_when_redaction_cannot_remove_windows_path(
    tmp_path: Path,
) -> None:
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    script = bin_dir / NNUNET_V2_EXECUTABLE_TRAIN
    _make_executable(
        script,
        "#!/usr/bin/env python3\nprint('C:/sensitive/path')\n",
    )

    with pytest.raises(NnUNetExecutionError) as exc_info:
        execute_nnunet_command(
            (NNUNET_V2_EXECUTABLE_TRAIN,),
            runtime_environment=_runtime_environment_with_path(bin_dir),
            working_directory=workdir,
            logs_directory=logs_dir,
            timeout_seconds=5.0,
            stage_name="custom-redaction-failure",
        )

    assert exc_info.value.failure_code == "nnunet_log_redaction_failed"
    assert "C:/sensitive/path" not in str(exc_info.value)
    assert list(logs_dir.glob("*")) == []


def test_import_guard_for_baselines_and_nnunet_module(
    run_import_guard: Callable[[tuple[str, ...], tuple[str, ...]], None],
) -> None:
    run_import_guard(
        ("protoem_ct.baselines", "protoem_ct.baselines.nnunet"),
        ("torch", "torchvision", "monai", "nnunetv2", "SimpleITK", "mlflow"),
    )
