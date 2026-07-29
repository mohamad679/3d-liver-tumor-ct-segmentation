"""Pure nnU-Net v2 wrapper contracts for Phase 3 without runtime execution side effects."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any

from protoem_ct.artifacts.hashing import canonical_json_bytes, sha256_json
from protoem_ct.baselines import _log_paths
from protoem_ct.baselines.paths import ValidatedBaselineRunPaths

NNUNET_V2_RUN_CONFIG_VERSION = "nnunet_v2_run_config_v1"
NNUNET_V2_EXECUTABLE_PLAN_AND_PREPROCESS = "nnUNetv2_plan_and_preprocess"
NNUNET_V2_EXECUTABLE_TRAIN = "nnUNetv2_train"
NNUNET_V2_EXECUTABLE_PREDICT = "nnUNetv2_predict"
NNUNET_V2_SUPPORTED_DATASET_ID = 901
NNUNET_V2_SUPPORTED_CONFIGURATION = "3d_fullres"
NNUNET_V2_SUPPORTED_FOLD = 0
NNUNET_V2_SUPPORTED_DEVICE = "cpu"
NNUNET_V2_DEFAULT_TRAINER = "nnUNetTrainer"
NNUNET_V2_DEFAULT_PLANS_IDENTIFIER = "nnUNetPlans"
NNUNET_V2_DEFAULT_CHECKPOINT_NAME = "checkpoint_final.pth"
NNUNET_V2_DEFAULT_PLANNER = "ExperimentPlanner"
NNUNET_ALLOWED_INHERITED_ENVIRONMENT_KEYS: tuple[str, ...] = (
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_COLLATE",
    "LC_CTYPE",
    "LC_MESSAGES",
    "LC_MONETARY",
    "LC_NUMERIC",
    "LC_TIME",
    "PATH",
    "REQUESTS_CA_BUNDLE",
    "SSL_CERT_DIR",
    "SSL_CERT_FILE",
    "TEMP",
    "TMP",
    "TMPDIR",
)

NNUNET_EXECUTION_FAILURE_CODES: tuple[str, ...] = (
    "nnunet_executable_missing",
    "nnunet_log_redaction_failed",
    "nnunet_nonzero_exit",
    "nnunet_output_collision",
    "nnunet_output_missing",
    "nnunet_prediction_import_failed",
    "nnunet_timeout",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,126}[A-Za-z0-9])?$")
_SAFE_COMMAND_PART_RE = re.compile(r"^[^\s`$|;&<>\\\n\r\t]+$")
_RUN_CONFIG_FIELDS = {
    "artifact_hash",
    "checkpoint_name",
    "configuration",
    "continue_training",
    "contract_version",
    "dataset_id",
    "device",
    "enable_amp",
    "fold",
    "plans_identifier",
    "save_probabilities",
    "seed",
    "trainer_class",
    "verify_dataset_integrity",
}


class NnUNetWrapperError(ValueError):
    """Base error for Phase 3 nnU-Net wrapper contracts."""


class NnUNetWrapperValidationError(NnUNetWrapperError):
    """Raised when nnU-Net wrapper inputs violate the contract."""


class NnUNetWrapperSerializationError(NnUNetWrapperError):
    """Raised when nnU-Net wrapper JSON cannot be parsed safely."""


class NnUNetWrapperVersionError(NnUNetWrapperError):
    """Raised when an unsupported nnU-Net wrapper version is encountered."""


class NnUNetWrapperHashError(NnUNetWrapperError):
    """Raised when a persisted nnU-Net wrapper artifact hash does not match its content."""


@dataclass(frozen=True, slots=True)
class NnUNetExecutionError(NnUNetWrapperError):
    """Typed subprocess execution failure with a machine-readable code."""

    failure_code: str
    stage_name: str
    return_code: int | None = None

    def __post_init__(self) -> None:
        if self.failure_code not in NNUNET_EXECUTION_FAILURE_CODES:
            raise NnUNetWrapperValidationError(
                f"Unsupported nnU-Net failure code: {self.failure_code!r}."
            )

    def __str__(self) -> str:
        return self.failure_code


@dataclass(frozen=True, slots=True)
class NnUNetV2RunConfig:
    """Immutable deterministic nnU-Net v2 run configuration artifact."""

    artifact_hash: str
    contract_version: str
    dataset_id: int
    configuration: str
    fold: int
    trainer_class: str
    plans_identifier: str
    checkpoint_name: str
    seed: int
    device: str
    enable_amp: bool
    verify_dataset_integrity: bool
    save_probabilities: bool
    continue_training: bool

    def __post_init__(self) -> None:
        if self.contract_version != NNUNET_V2_RUN_CONFIG_VERSION:
            raise NnUNetWrapperVersionError(
                f"Unsupported nnU-Net run-config version: {self.contract_version!r}."
            )
        _require_sha256(self.artifact_hash, field_name="artifact_hash")
        if self.dataset_id != NNUNET_V2_SUPPORTED_DATASET_ID:
            raise NnUNetWrapperValidationError(
                f"dataset_id must equal {NNUNET_V2_SUPPORTED_DATASET_ID}."
            )
        if self.configuration != NNUNET_V2_SUPPORTED_CONFIGURATION:
            raise NnUNetWrapperValidationError(
                f"configuration must equal {NNUNET_V2_SUPPORTED_CONFIGURATION!r}."
            )
        if self.fold != NNUNET_V2_SUPPORTED_FOLD:
            raise NnUNetWrapperValidationError(f"fold must equal {NNUNET_V2_SUPPORTED_FOLD}.")
        _require_safe_identifier(self.trainer_class, field_name="trainer_class")
        _require_safe_identifier(self.plans_identifier, field_name="plans_identifier")
        _require_safe_identifier(self.checkpoint_name, field_name="checkpoint_name")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise NnUNetWrapperValidationError("seed must be a nonnegative integer.")
        if self.device != NNUNET_V2_SUPPORTED_DEVICE:
            raise NnUNetWrapperValidationError(f"device must equal {NNUNET_V2_SUPPORTED_DEVICE!r}.")
        if self.enable_amp:
            raise NnUNetWrapperValidationError("enable_amp must be false for the initial CPU path.")
        if not self.verify_dataset_integrity:
            raise NnUNetWrapperValidationError("verify_dataset_integrity must be true.")
        if self.save_probabilities:
            raise NnUNetWrapperValidationError("save_probabilities must be false.")
        if self.continue_training:
            raise NnUNetWrapperValidationError("continue_training must be false.")
        _require_no_absolute_paths(_run_config_payload(self))
        expected_hash = sha256_json(_run_config_payload_without_hash(self))
        if self.artifact_hash != expected_hash:
            raise NnUNetWrapperHashError(
                "artifact_hash does not match the deterministic nnU-Net run-config content."
            )


@dataclass(frozen=True, slots=True)
class NnUNetRuntimeEnvironment:
    """Validated runtime-only nnU-Net path and environment bundle."""

    raw_root: Path
    preprocessed_root: Path
    results_root: Path
    environment: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class NnUNetExecutionResult:
    """Captured outcome for one nnU-Net subprocess execution."""

    stage_name: str
    command: tuple[str, ...]
    return_code: int
    stdout_log_path: Path
    stderr_log_path: Path
    stage_status: str


def nnunet_v2_run_config_to_json(run_config: NnUNetV2RunConfig) -> bytes:
    """Serialize a deterministic nnU-Net run configuration artifact."""

    return canonical_json_bytes(_run_config_payload(run_config)) + b"\n"


def nnunet_v2_run_config_from_json(payload: bytes | str) -> NnUNetV2RunConfig:
    """Parse and validate a deterministic nnU-Net run configuration artifact."""

    parsed = _parse_json_object(payload, context="nnU-Net run config")
    _require_exact_fields(parsed, expected=_RUN_CONFIG_FIELDS, context="nnU-Net run config")
    run_config = NnUNetV2RunConfig(
        artifact_hash=_expect_string(parsed["artifact_hash"], field_name="artifact_hash"),
        contract_version=_expect_string(
            parsed["contract_version"],
            field_name="contract_version",
        ),
        dataset_id=_expect_int(parsed["dataset_id"], field_name="dataset_id"),
        configuration=_expect_string(parsed["configuration"], field_name="configuration"),
        fold=_expect_int(parsed["fold"], field_name="fold"),
        trainer_class=_expect_string(parsed["trainer_class"], field_name="trainer_class"),
        plans_identifier=_expect_string(
            parsed["plans_identifier"],
            field_name="plans_identifier",
        ),
        checkpoint_name=_expect_string(
            parsed["checkpoint_name"],
            field_name="checkpoint_name",
        ),
        seed=_expect_int(parsed["seed"], field_name="seed"),
        device=_expect_string(parsed["device"], field_name="device"),
        enable_amp=_expect_bool(parsed["enable_amp"], field_name="enable_amp"),
        verify_dataset_integrity=_expect_bool(
            parsed["verify_dataset_integrity"],
            field_name="verify_dataset_integrity",
        ),
        save_probabilities=_expect_bool(
            parsed["save_probabilities"],
            field_name="save_probabilities",
        ),
        continue_training=_expect_bool(
            parsed["continue_training"],
            field_name="continue_training",
        ),
    )
    expected_hash = sha256_json(_run_config_payload_without_hash(run_config))
    if run_config.artifact_hash != expected_hash:
        raise NnUNetWrapperHashError(
            "artifact_hash does not match the parsed nnU-Net run-config content."
        )
    return run_config


def build_nnunet_v2_run_config(
    *,
    dataset_id: int = NNUNET_V2_SUPPORTED_DATASET_ID,
    configuration: str = NNUNET_V2_SUPPORTED_CONFIGURATION,
    fold: int = NNUNET_V2_SUPPORTED_FOLD,
    trainer_class: str = NNUNET_V2_DEFAULT_TRAINER,
    plans_identifier: str = NNUNET_V2_DEFAULT_PLANS_IDENTIFIER,
    checkpoint_name: str = NNUNET_V2_DEFAULT_CHECKPOINT_NAME,
    seed: int = 0,
    device: str = NNUNET_V2_SUPPORTED_DEVICE,
    enable_amp: bool = False,
    verify_dataset_integrity: bool = True,
    save_probabilities: bool = False,
    continue_training: bool = False,
) -> NnUNetV2RunConfig:
    """Build the deterministic initial nnU-Net v2 run configuration artifact."""

    payload_without_hash = {
        "checkpoint_name": checkpoint_name,
        "configuration": configuration,
        "continue_training": continue_training,
        "contract_version": NNUNET_V2_RUN_CONFIG_VERSION,
        "dataset_id": dataset_id,
        "device": device,
        "enable_amp": enable_amp,
        "fold": fold,
        "plans_identifier": plans_identifier,
        "save_probabilities": save_probabilities,
        "seed": seed,
        "trainer_class": trainer_class,
        "verify_dataset_integrity": verify_dataset_integrity,
    }
    return NnUNetV2RunConfig(
        artifact_hash=sha256_json(payload_without_hash),
        contract_version=NNUNET_V2_RUN_CONFIG_VERSION,
        dataset_id=dataset_id,
        configuration=configuration,
        fold=fold,
        trainer_class=trainer_class,
        plans_identifier=plans_identifier,
        checkpoint_name=checkpoint_name,
        seed=seed,
        device=device,
        enable_amp=enable_amp,
        verify_dataset_integrity=verify_dataset_integrity,
        save_probabilities=save_probabilities,
        continue_training=continue_training,
    )


def build_nnunet_v2_plan_and_preprocess_command(
    run_config: NnUNetV2RunConfig,
) -> tuple[str, ...]:
    """Build the deterministic plan-and-preprocess command tuple."""

    return (
        NNUNET_V2_EXECUTABLE_PLAN_AND_PREPROCESS,
        "-d",
        str(run_config.dataset_id),
        "--verify_dataset_integrity",
        "-pl",
        NNUNET_V2_DEFAULT_PLANNER,
        "-overwrite_plans_name",
        run_config.plans_identifier,
        "-c",
        run_config.configuration,
    )


def build_nnunet_v2_train_command(
    run_config: NnUNetV2RunConfig,
) -> tuple[str, ...]:
    """Build the deterministic training command tuple."""

    command = [
        NNUNET_V2_EXECUTABLE_TRAIN,
        str(run_config.dataset_id),
        run_config.configuration,
        str(run_config.fold),
        "-tr",
        run_config.trainer_class,
        "-p",
        run_config.plans_identifier,
        "-device",
        run_config.device,
    ]
    if run_config.save_probabilities:
        command.append("--npz")
    if run_config.continue_training:
        command.append("--c")
    return tuple(command)


def build_nnunet_v2_predict_command(
    run_config: NnUNetV2RunConfig,
    *,
    input_images_directory: Path,
    output_predictions_directory: Path,
) -> tuple[str, ...]:
    """Build the deterministic prediction command tuple."""

    input_root = _validate_existing_external_directory(
        input_images_directory,
        field_name="input_images_directory",
    )
    output_root = _validate_external_runtime_path(
        output_predictions_directory,
        field_name="output_predictions_directory",
    )
    return (
        NNUNET_V2_EXECUTABLE_PREDICT,
        "-i",
        str(input_root),
        "-o",
        str(output_root),
        "-d",
        str(run_config.dataset_id),
        "-p",
        run_config.plans_identifier,
        "-tr",
        run_config.trainer_class,
        "-c",
        run_config.configuration,
        "-f",
        str(run_config.fold),
        "-chk",
        run_config.checkpoint_name,
        "-device",
        run_config.device,
    )


def build_nnunet_v2_runtime_environment(
    *,
    raw_root: Path,
    run_paths: ValidatedBaselineRunPaths,
    inherited_environment: Mapping[str, str] | None = None,
) -> NnUNetRuntimeEnvironment:
    """Construct the sanitized nnU-Net runtime environment without persisting paths."""

    if run_paths.baseline_family != "nnunet_v2":
        raise NnUNetWrapperValidationError(
            "run_paths must belong to the nnunet_v2 baseline family."
        )
    raw_dataset_root = _validate_existing_external_directory(raw_root, field_name="raw_root")
    preprocessed_root = _validate_path_beneath(
        run_paths.temporary_dir / "nnunet_preprocessed",
        parent=run_paths.temporary_dir,
        field_name="preprocessed_root",
    )
    results_root = _validate_path_beneath(
        run_paths.checkpoints_dir / "nnunet_results",
        parent=run_paths.checkpoints_dir,
        field_name="results_root",
    )
    environment = _build_sanitized_environment(
        raw_root=raw_dataset_root,
        preprocessed_root=preprocessed_root,
        results_root=results_root,
        inherited_environment=inherited_environment or os.environ,
    )
    return NnUNetRuntimeEnvironment(
        raw_root=raw_dataset_root,
        preprocessed_root=preprocessed_root,
        results_root=results_root,
        environment=environment,
    )


def execute_nnunet_command(
    command: tuple[str, ...],
    *,
    runtime_environment: Mapping[str, str],
    working_directory: Path,
    logs_directory: Path,
    timeout_seconds: float,
    stage_name: str,
    required_output_paths: Sequence[Path] = (),
    log_redaction_roots: Mapping[str, Path] | None = None,
) -> NnUNetExecutionResult:
    """Execute a validated nnU-Net command tuple without a shell."""

    _validate_nnunet_command(command)
    working_directory_path = _validate_existing_external_directory(
        working_directory,
        field_name="working_directory",
    )
    logs_directory_path = _validate_existing_external_directory(
        logs_directory,
        field_name="logs_directory",
    )
    if timeout_seconds <= 0.0 or not math.isfinite(timeout_seconds):
        raise NnUNetWrapperValidationError("timeout_seconds must be a finite positive number.")
    for output_path in required_output_paths:
        if Path(output_path).exists():
            raise NnUNetExecutionError("nnunet_output_collision", stage_name=stage_name)
    stdout_log_path = logs_directory_path / f"{stage_name}.stdout.log"
    stderr_log_path = logs_directory_path / f"{stage_name}.stderr.log"
    if stdout_log_path.exists() or stderr_log_path.exists():
        raise NnUNetExecutionError("nnunet_output_collision", stage_name=stage_name)
    try:
        completed = subprocess.run(
            command,
            cwd=working_directory_path,
            env=dict(runtime_environment),
            shell=False,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except FileNotFoundError as error:
        raise NnUNetExecutionError(
            "nnunet_executable_missing",
            stage_name=stage_name,
        ) from error
    except subprocess.TimeoutExpired as error:
        timeout_stdout = (
            error.stdout.decode("utf-8")
            if isinstance(error.stdout, bytes)
            else (error.stdout or "")
        )
        timeout_stderr = (
            error.stderr.decode("utf-8")
            if isinstance(error.stderr, bytes)
            else (error.stderr or "")
        )
        _write_log(
            stdout_log_path,
            timeout_stdout,
            stage_name=stage_name,
            log_redaction_roots=log_redaction_roots,
            runtime_environment=runtime_environment,
            working_directory=working_directory_path,
            logs_directory=logs_directory_path,
        )
        _write_log(
            stderr_log_path,
            timeout_stderr,
            stage_name=stage_name,
            log_redaction_roots=log_redaction_roots,
            runtime_environment=runtime_environment,
            working_directory=working_directory_path,
            logs_directory=logs_directory_path,
        )
        raise NnUNetExecutionError("nnunet_timeout", stage_name=stage_name) from error
    _write_log(
        stdout_log_path,
        completed.stdout,
        stage_name=stage_name,
        log_redaction_roots=log_redaction_roots,
        runtime_environment=runtime_environment,
        working_directory=working_directory_path,
        logs_directory=logs_directory_path,
    )
    _write_log(
        stderr_log_path,
        completed.stderr,
        stage_name=stage_name,
        log_redaction_roots=log_redaction_roots,
        runtime_environment=runtime_environment,
        working_directory=working_directory_path,
        logs_directory=logs_directory_path,
    )
    if completed.returncode != 0:
        raise NnUNetExecutionError(
            "nnunet_nonzero_exit",
            stage_name=stage_name,
            return_code=completed.returncode,
        )
    for output_path in required_output_paths:
        if not Path(output_path).exists():
            raise NnUNetExecutionError("nnunet_output_missing", stage_name=stage_name)
    return NnUNetExecutionResult(
        stage_name=stage_name,
        command=command,
        return_code=completed.returncode,
        stdout_log_path=stdout_log_path,
        stderr_log_path=stderr_log_path,
        stage_status="completed",
    )


def _build_sanitized_environment(
    *,
    raw_root: Path,
    preprocessed_root: Path,
    results_root: Path,
    inherited_environment: Mapping[str, str],
) -> dict[str, str]:
    environment: dict[str, str] = {}
    for key in NNUNET_ALLOWED_INHERITED_ENVIRONMENT_KEYS:
        value = inherited_environment.get(key)
        if value is not None:
            environment[key] = value
    environment["nnUNet_raw"] = str(raw_root)
    environment["nnUNet_preprocessed"] = str(preprocessed_root)
    environment["nnUNet_results"] = str(results_root)
    return environment


def _validate_nnunet_command(command: tuple[str, ...]) -> None:
    if not command:
        raise NnUNetWrapperValidationError("command must not be empty.")
    if any(not isinstance(part, str) or not part for part in command):
        raise NnUNetWrapperValidationError("command must contain only nonempty string arguments.")
    executable = command[0]
    if executable not in {
        NNUNET_V2_EXECUTABLE_PLAN_AND_PREPROCESS,
        NNUNET_V2_EXECUTABLE_TRAIN,
        NNUNET_V2_EXECUTABLE_PREDICT,
    }:
        raise NnUNetWrapperValidationError("Unsupported nnU-Net executable.")
    for part in command:
        if any(ord(character) < 32 or ord(character) == 127 for character in part):
            raise NnUNetWrapperValidationError(
                "command arguments must not contain control characters."
            )
        if executable == part:
            continue
        if part.startswith("-"):
            continue
        if not _SAFE_COMMAND_PART_RE.fullmatch(part):
            if Path(part).is_absolute():
                continue
            raise NnUNetWrapperValidationError(
                "command contains unsafe shell-like argument content."
            )


def _write_log(
    path: Path,
    content: str,
    *,
    stage_name: str,
    log_redaction_roots: Mapping[str, Path] | None,
    runtime_environment: Mapping[str, str],
    working_directory: Path,
    logs_directory: Path,
) -> None:
    if path.exists():
        raise NnUNetExecutionError("nnunet_output_collision", stage_name=path.stem)
    redacted = _redact_log_content(
        content,
        log_redaction_roots=log_redaction_roots,
        runtime_environment=runtime_environment,
        working_directory=working_directory,
        logs_directory=logs_directory,
    )
    if _log_paths._contains_absolute_filesystem_path(redacted):
        raise NnUNetExecutionError("nnunet_log_redaction_failed", stage_name=stage_name)
    path.write_text(redacted, encoding="utf-8")


def _redact_log_content(
    content: str,
    *,
    log_redaction_roots: Mapping[str, Path] | None,
    runtime_environment: Mapping[str, str],
    working_directory: Path,
    logs_directory: Path,
) -> str:
    replacements = _build_log_redaction_replacements(
        log_redaction_roots=log_redaction_roots,
        runtime_environment=runtime_environment,
        working_directory=working_directory,
        logs_directory=logs_directory,
    )
    redacted = content
    for source, token in replacements:
        redacted = redacted.replace(source, token)
    redacted = _log_paths._redact_residual_absolute_paths(redacted)
    return redacted.rstrip("\r\n") + "\n"


def _build_log_redaction_replacements(
    *,
    log_redaction_roots: Mapping[str, Path] | None,
    runtime_environment: Mapping[str, str],
    working_directory: Path,
    logs_directory: Path,
) -> tuple[tuple[str, str], ...]:
    runtime_root = logs_directory.parent
    replacements: dict[str, str] = {
        os.path.normpath(os.fspath(runtime_root)): "<RUNTIME_ROOT>",
        os.path.normpath(os.fspath(Path(__file__).resolve().parents[3])): "<REPOSITORY_ROOT>",
        os.path.normpath(os.fspath(Path.home())): "<HOME>",
        os.path.normpath(os.fspath(Path(sys.prefix).resolve())): "<BASELINE_ENV>",
        os.path.normpath(os.fspath(working_directory)): "<TEMP_ROOT>",
    }
    for environment_key, token in (
        ("nnUNet_raw", "<NNUNET_RAW>"),
        ("nnUNet_preprocessed", "<NNUNET_PREPROCESSED>"),
        ("nnUNet_results", "<NNUNET_RESULTS>"),
    ):
        value = runtime_environment.get(environment_key)
        if value:
            replacements[os.path.normpath(os.fspath(Path(value)))] = token
    for environment_key in ("TEMP", "TMP", "TMPDIR"):
        value = runtime_environment.get(environment_key)
        if value:
            replacements[os.path.normpath(value)] = "<TEMP_ROOT>"
    if log_redaction_roots is not None:
        for token, path in log_redaction_roots.items():
            replacements[os.path.normpath(os.fspath(path))] = token
    sorted_items = sorted(replacements.items(), key=lambda item: (-len(item[0]), item[0]))
    return tuple((source, token) for source, token in sorted_items if source)


def _run_config_payload(run_config: NnUNetV2RunConfig) -> dict[str, Any]:
    return {
        "artifact_hash": run_config.artifact_hash,
        "checkpoint_name": run_config.checkpoint_name,
        "configuration": run_config.configuration,
        "continue_training": run_config.continue_training,
        "contract_version": run_config.contract_version,
        "dataset_id": run_config.dataset_id,
        "device": run_config.device,
        "enable_amp": run_config.enable_amp,
        "fold": run_config.fold,
        "plans_identifier": run_config.plans_identifier,
        "save_probabilities": run_config.save_probabilities,
        "seed": run_config.seed,
        "trainer_class": run_config.trainer_class,
        "verify_dataset_integrity": run_config.verify_dataset_integrity,
    }


def _run_config_payload_without_hash(run_config: NnUNetV2RunConfig) -> dict[str, Any]:
    payload = _run_config_payload(run_config)
    del payload["artifact_hash"]
    return payload


def _require_safe_identifier(value: str, *, field_name: str) -> None:
    if not _SAFE_IDENTIFIER_RE.fullmatch(value):
        raise NnUNetWrapperValidationError(
            f"{field_name} must match the conservative ASCII identifier grammar."
        )
    if any(
        character in value for character in ("`", "$", "|", ";", "&", "<", ">", "\n", "\r", "\t")
    ):
        raise NnUNetWrapperValidationError(
            f"{field_name} must not contain shell metacharacters or control characters."
        )
    _require_no_absolute_path_string(value, field_name=field_name)


def _validate_existing_external_directory(path: Path, *, field_name: str) -> Path:
    candidate = _validate_external_runtime_path(path, field_name=field_name)
    if not candidate.exists():
        raise NnUNetWrapperValidationError(f"{field_name} must exist.")
    if not candidate.is_dir():
        raise NnUNetWrapperValidationError(f"{field_name} must be a directory.")
    return candidate


def _validate_path_beneath(path: Path, *, parent: Path, field_name: str) -> Path:
    candidate = _validate_external_runtime_path(path, field_name=field_name)
    parent_resolved = parent.resolve(strict=False)
    if candidate == parent_resolved or not _is_relative_to(candidate, parent_resolved):
        raise NnUNetWrapperValidationError(
            f"{field_name} must be strictly beneath the approved parent directory."
        )
    return candidate


def _validate_external_runtime_path(path: Path, *, field_name: str) -> Path:
    repository_root = Path(__file__).resolve().parents[3]
    candidate = Path(path)
    if not candidate.is_absolute():
        raise NnUNetWrapperValidationError(f"{field_name} must be an absolute path.")
    normalized = os.path.normpath(os.fspath(candidate))
    if ".." in Path(normalized).parts:
        raise NnUNetWrapperValidationError(
            f"{field_name} must not contain parent-directory traversal."
        )
    for component in candidate.parts[1:]:
        if component == "":
            raise NnUNetWrapperValidationError(
                f"{field_name} must not contain empty path components."
            )
        if any(ord(character) < 32 or ord(character) == 127 for character in component):
            raise NnUNetWrapperValidationError(f"{field_name} must not contain control characters.")
    _require_no_symlink_components(candidate.parent)
    resolved = candidate.resolve(strict=False)
    if resolved == repository_root or _is_relative_to(resolved, repository_root):
        raise NnUNetWrapperValidationError(
            f"{field_name} must resolve outside the repository root."
        )
    return resolved


def _require_no_symlink_components(path: Path) -> None:
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current = current / component
        if current.is_symlink():
            raise NnUNetWrapperValidationError(
                "Runtime paths must not traverse symlinked path components."
            )


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _require_sha256(value: str, *, field_name: str) -> None:
    if not _SHA256_RE.fullmatch(value):
        raise NnUNetWrapperValidationError(
            f"{field_name} must be exactly 64 lowercase hexadecimal characters."
        )


def _require_no_absolute_paths(value: Any) -> None:
    if isinstance(value, str):
        _require_no_absolute_path_string(value, field_name="persisted string")
        return
    if isinstance(value, Mapping):
        for item in value.values():
            _require_no_absolute_paths(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _require_no_absolute_paths(item)


def _require_no_absolute_path_string(value: str, *, field_name: str) -> None:
    if value.startswith("/") or PureWindowsPath(value).is_absolute():
        raise NnUNetWrapperValidationError(f"{field_name} must not contain an absolute path.")


def _parse_json_object(payload: bytes | str, *, context: str) -> dict[str, Any]:
    text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
    try:
        parsed = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_bad_json_constant,
        )
    except Exception as error:
        raise NnUNetWrapperSerializationError(f"Failed to parse {context} JSON: {error}") from error
    if not isinstance(parsed, dict):
        raise NnUNetWrapperSerializationError(f"{context} JSON must encode an object.")
    return parsed


def _require_exact_fields(
    payload: Mapping[str, Any],
    *,
    expected: set[str],
    context: str,
) -> None:
    unknown_fields = set(payload) - expected
    if unknown_fields:
        raise NnUNetWrapperSerializationError(
            f"Unknown {context} field(s): {', '.join(sorted(unknown_fields))}."
        )
    missing_fields = expected - set(payload)
    if missing_fields:
        raise NnUNetWrapperSerializationError(
            f"Missing {context} field(s): {', '.join(sorted(missing_fields))}."
        )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise NnUNetWrapperSerializationError(f"Duplicate JSON object key: {key!r}.")
        result[key] = value
    return result


def _bad_json_constant(constant: str) -> Any:
    raise NnUNetWrapperSerializationError(f"Invalid JSON constant: {constant!r}.")


def _expect_string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise NnUNetWrapperSerializationError(f"{field_name} must be a string.")
    return value


def _expect_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise NnUNetWrapperSerializationError(f"{field_name} must be an integer.")
    return value


def _expect_bool(value: Any, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise NnUNetWrapperSerializationError(f"{field_name} must be a boolean.")
    return value
