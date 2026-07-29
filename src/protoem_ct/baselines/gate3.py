"""Phase 3 Gate 3 synthetic orchestration and report contracts."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import re
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, cast

from protoem_ct.artifacts.hashing import canonical_json_bytes, sha256_file, sha256_json
from protoem_ct.baselines import _log_paths
from protoem_ct.baselines._mlflow_local import (
    Phase3MlflowMetadataError,
    Phase3MlflowVerificationResult,
    audit_phase3_mlflow_tracking_directory,
)
from protoem_ct.baselines._torch_runtime import configure_phase3_cpu_torch_runtime
from protoem_ct.baselines.metrics import baseline_metric_report_from_json
from protoem_ct.baselines.monai_segresnet import (
    MONAI_SEGRESNET_DEFAULT_SEED,
    MonaiSegResNetExecutionResult,
    MonaiSegResNetRunConfig,
    build_monai_segresnet_run_config,
    run_monai_segresnet_baseline,
)
from protoem_ct.baselines.nnunet import NnUNetV2RunConfig, build_nnunet_v2_run_config
from protoem_ct.baselines.nnunet_tiny import (
    NNUNET_TINY_STRATEGY_IDENTIFIER,
    NnUNetTinyExecutionResult,
    run_nnunet_tiny_baseline,
)
from protoem_ct.baselines.paths import (
    ValidatedBaselineRunPaths,
    create_validated_run_directory_tree,
    validate_baseline_run_root,
)
from protoem_ct.baselines.predictions import baseline_prediction_manifest_from_json
from protoem_ct.baselines.provenance import (
    BaselineProvenanceHashError,
    BaselineProvenanceSerializationError,
    BaselineRunProvenance,
    baseline_run_provenance_from_json,
)
from protoem_ct.baselines.synthetic import generate_baseline_synthetic_fixture

PHASE3_GATE3_REPORT_VERSION = "phase3_gate3_report_v1"
PHASE3_GATE3_FAILURE_CODES: frozenset[str] = frozenset(
    {
        "phase3_absolute_path_persisted",
        "phase3_log_redaction_failed",
        "phase3_mlflow_metadata_invalid",
        "phase3_provenance_hash_mismatch",
        "phase3_provenance_link_mismatch",
        "phase3_provenance_missing",
    }
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SAFE_IDENTIFIER_RE = re.compile(r"^[a-z0-9](?:[a-z0-9_-]{0,62}[a-z0-9])?$")


class Phase3Gate3Error(ValueError):
    """Base error for the Phase 3 Gate 3 synthetic orchestration."""


class Phase3Gate3ValidationError(Phase3Gate3Error):
    """Raised when Gate 3 synthetic contracts are invalid."""


class Phase3Gate3SerializationError(Phase3Gate3Error):
    """Raised when Gate 3 artifacts cannot be serialized or parsed safely."""


class Phase3Gate3VersionError(Phase3Gate3Error):
    """Raised when an unsupported Gate 3 report version is encountered."""


class Phase3Gate3HashError(Phase3Gate3Error):
    """Raised when a Gate 3 report hash does not match its content."""


@dataclass(frozen=True, slots=True)
class Phase3Gate3ExecutionError(Phase3Gate3Error):
    """Typed Gate 3 execution failure carrying a machine-readable failure code."""

    failure_code: str

    def __post_init__(self) -> None:
        if self.failure_code not in PHASE3_GATE3_FAILURE_CODES:
            raise Phase3Gate3ValidationError(
                f"Unsupported Gate 3 failure code: {self.failure_code!r}."
            )

    def __str__(self) -> str:
        return self.failure_code


@dataclass(frozen=True, slots=True)
class Phase3Gate3Report:
    """Immutable non-path evidence artifact for the synthetic Gate 3 path."""

    artifact_hash: str
    contract_version: str
    run_identifier: str
    git_commit: str
    baseline_environment_lock_sha256: str
    synthetic_fixture_artifact_sha256: str
    selected_device: str
    amp_enabled: bool
    nnunet_execution_strategy_identifier: str
    nnunet_run_config_sha256: str
    monai_run_config_sha256: str
    both_checkpoint_sha256: Mapping[str, str]
    both_prediction_manifest_sha256: Mapping[str, str]
    both_metric_report_sha256: Mapping[str, str]
    both_provenance_sha256: Mapping[str, str]
    cpu_shape_smoke_status_by_baseline: Mapping[str, bool]
    overfit_status_by_baseline: Mapping[str, bool]
    initial_loss_by_baseline: Mapping[str, float]
    final_loss_by_baseline: Mapping[str, float]
    best_loss_by_baseline: Mapping[str, float]
    training_step_count_by_baseline: Mapping[str, int]
    deterministic_metric_json_verified_by_baseline: Mapping[str, bool]
    mlflow_metadata_verified_by_baseline: Mapping[str, bool]
    overall_status: str
    failure_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.contract_version != PHASE3_GATE3_REPORT_VERSION:
            raise Phase3Gate3VersionError(
                f"Unsupported Gate 3 report version: {self.contract_version!r}."
            )
        _require_sha256(self.artifact_hash, field_name="artifact_hash")
        _require_sha256(
            self.baseline_environment_lock_sha256,
            field_name="baseline_environment_lock_sha256",
        )
        _require_sha256(
            self.synthetic_fixture_artifact_sha256,
            field_name="synthetic_fixture_artifact_sha256",
        )
        if not _SAFE_IDENTIFIER_RE.fullmatch(self.run_identifier):
            raise Phase3Gate3ValidationError(
                "run_identifier must match the conservative ASCII identifier grammar."
            )
        if not _GIT_COMMIT_RE.fullmatch(self.git_commit):
            raise Phase3Gate3ValidationError(
                "git_commit must be 40 lowercase hexadecimal characters."
            )
        if self.selected_device != "cpu":
            raise Phase3Gate3ValidationError("selected_device must equal 'cpu'.")
        if self.amp_enabled:
            raise Phase3Gate3ValidationError("amp_enabled must be false.")
        if self.nnunet_execution_strategy_identifier != NNUNET_TINY_STRATEGY_IDENTIFIER:
            raise Phase3Gate3ValidationError("Unexpected nnU-Net strategy identifier.")
        if self.artifact_hash != sha256_json(_report_payload_without_hash(self)):
            raise Phase3Gate3HashError(
                "artifact_hash does not match the deterministic Gate 3 report content."
            )


@dataclass(frozen=True, slots=True)
class Phase3Gate3ExecutionResult:
    """Bundle of published synthetic Gate 3 execution artifacts."""

    fixture_artifact_sha256: str
    monai_result: MonaiSegResNetExecutionResult
    nnunet_result: NnUNetTinyExecutionResult
    report: Phase3Gate3Report
    report_path: Path


def _verify_persisted_baseline_provenance(
    *,
    provenance_path: Path,
    expected_provenance: BaselineRunProvenance,
    checkpoint_path: Path,
    prediction_manifest_path: Path,
    metric_report_path: Path,
) -> BaselineRunProvenance:
    if not provenance_path.is_file():
        raise Phase3Gate3ExecutionError("phase3_provenance_missing")
    try:
        provenance = baseline_run_provenance_from_json(provenance_path.read_bytes())
    except (BaselineProvenanceSerializationError, BaselineProvenanceHashError) as exc:
        raise Phase3Gate3ExecutionError("phase3_provenance_hash_mismatch") from exc
    if provenance.status != "completed":
        raise Phase3Gate3ExecutionError("phase3_provenance_link_mismatch")
    if provenance.artifact_hash != expected_provenance.artifact_hash:
        raise Phase3Gate3ExecutionError("phase3_provenance_hash_mismatch")
    if provenance.checkpoint_sha256 != sha256_file(checkpoint_path):
        raise Phase3Gate3ExecutionError("phase3_provenance_link_mismatch")
    prediction_manifest = baseline_prediction_manifest_from_json(
        prediction_manifest_path.read_bytes()
    )
    if provenance.prediction_manifest_sha256 != prediction_manifest.artifact_hash:
        raise Phase3Gate3ExecutionError("phase3_provenance_link_mismatch")
    metric_report = baseline_metric_report_from_json(metric_report_path.read_bytes())
    if provenance.metrics_artifact_sha256 != metric_report.artifact_hash:
        raise Phase3Gate3ExecutionError("phase3_provenance_link_mismatch")
    return provenance


def _audit_project_json_artifacts(*, output_root: Path) -> None:
    for path in sorted(output_root.rglob("*.json")):
        _reject_persisted_path_text(path.read_text(encoding="utf-8"))


def _audit_nnunet_logs(*, run_paths: ValidatedBaselineRunPaths) -> None:
    for path in sorted(run_paths.logs_dir.glob("*.log")):
        persisted_log = path.read_text(encoding="utf-8")
        if _log_paths._contains_absolute_filesystem_path(persisted_log):
            raise Phase3Gate3ExecutionError("phase3_log_redaction_failed")


def _require_mlflow_verification(
    verification: Phase3MlflowVerificationResult,
    *,
    baseline_family: str,
    run_identifier: str,
) -> None:
    if (
        not verification.verified
        or verification.database_filename != "tracking.db"
        or verification.database_byte_size <= 0
        or verification.baseline_family != baseline_family
        or verification.run_identifier != run_identifier
    ):
        raise Phase3Gate3ExecutionError("phase3_mlflow_metadata_invalid")


def _reject_persisted_path_text(value: str) -> None:
    if _log_paths._contains_absolute_filesystem_path(value):
        raise Phase3Gate3ExecutionError("phase3_absolute_path_persisted")
    if "MLFLOW_ALLOW_FILE_STORE" in value:
        raise Phase3Gate3ExecutionError("phase3_mlflow_metadata_invalid")


def build_phase3_gate3_report(
    *,
    run_identifier: str,
    git_commit: str,
    baseline_environment_lock_sha256: str,
    synthetic_fixture_artifact_sha256: str,
    nnunet_run_config: NnUNetV2RunConfig,
    monai_run_config: MonaiSegResNetRunConfig,
    nnunet_result: NnUNetTinyExecutionResult,
    monai_result: MonaiSegResNetExecutionResult,
) -> Phase3Gate3Report:
    """Build the immutable non-path Gate 3 report artifact."""

    payload_without_hash = {
        "amp_enabled": False,
        "baseline_environment_lock_sha256": baseline_environment_lock_sha256,
        "both_checkpoint_sha256": {
            "monai_segresnet": monai_result.checkpoint_sha256,
            "nnunet_v2": nnunet_result.checkpoint_sha256,
        },
        "both_metric_report_sha256": {
            "monai_segresnet": monai_result.prediction_import_result.metric_report.artifact_hash,
            "nnunet_v2": nnunet_result.prediction_import_result.metric_report.artifact_hash,
        },
        "both_prediction_manifest_sha256": {
            "monai_segresnet": (
                monai_result.prediction_import_result.prediction_manifest.artifact_hash
            ),
            "nnunet_v2": nnunet_result.prediction_import_result.prediction_manifest.artifact_hash,
        },
        "both_provenance_sha256": {
            "monai_segresnet": monai_result.provenance.artifact_hash,
            "nnunet_v2": nnunet_result.provenance.artifact_hash,
        },
        "contract_version": PHASE3_GATE3_REPORT_VERSION,
        "cpu_shape_smoke_status_by_baseline": {
            "monai_segresnet": monai_result.shape_smoke_passed,
            "nnunet_v2": nnunet_result.shape_smoke_passed,
        },
        "deterministic_metric_json_verified_by_baseline": {
            "monai_segresnet": monai_result.deterministic_metric_json_verified,
            "nnunet_v2": nnunet_result.deterministic_metric_json_verified,
        },
        "failure_codes": [],
        "git_commit": git_commit,
        "initial_loss_by_baseline": {
            "monai_segresnet": monai_result.initial_loss,
            "nnunet_v2": nnunet_result.initial_loss,
        },
        "final_loss_by_baseline": {
            "monai_segresnet": monai_result.final_loss,
            "nnunet_v2": nnunet_result.final_loss,
        },
        "best_loss_by_baseline": {
            "monai_segresnet": monai_result.best_loss,
            "nnunet_v2": nnunet_result.best_loss,
        },
        "mlflow_metadata_verified_by_baseline": {
            "monai_segresnet": monai_result.mlflow_metadata_verified,
            "nnunet_v2": nnunet_result.mlflow_metadata_verified,
        },
        "monai_run_config_sha256": monai_run_config.artifact_hash,
        "nnunet_execution_strategy_identifier": NNUNET_TINY_STRATEGY_IDENTIFIER,
        "nnunet_run_config_sha256": nnunet_run_config.artifact_hash,
        "overall_status": "completed",
        "overfit_status_by_baseline": {
            "monai_segresnet": monai_result.overfit_passed,
            "nnunet_v2": nnunet_result.overfit_passed,
        },
        "run_identifier": run_identifier,
        "selected_device": "cpu",
        "synthetic_fixture_artifact_sha256": synthetic_fixture_artifact_sha256,
        "training_step_count_by_baseline": {
            "monai_segresnet": monai_result.training_step_count,
            "nnunet_v2": nnunet_result.training_step_count,
        },
    }
    return Phase3Gate3Report(
        artifact_hash=sha256_json(payload_without_hash),
        contract_version=PHASE3_GATE3_REPORT_VERSION,
        run_identifier=run_identifier,
        git_commit=git_commit,
        baseline_environment_lock_sha256=baseline_environment_lock_sha256,
        synthetic_fixture_artifact_sha256=synthetic_fixture_artifact_sha256,
        selected_device="cpu",
        amp_enabled=False,
        nnunet_execution_strategy_identifier=NNUNET_TINY_STRATEGY_IDENTIFIER,
        nnunet_run_config_sha256=nnunet_run_config.artifact_hash,
        monai_run_config_sha256=monai_run_config.artifact_hash,
        both_checkpoint_sha256=cast(
            Mapping[str, str], payload_without_hash["both_checkpoint_sha256"]
        ),
        both_prediction_manifest_sha256=cast(
            Mapping[str, str], payload_without_hash["both_prediction_manifest_sha256"]
        ),
        both_metric_report_sha256=cast(
            Mapping[str, str], payload_without_hash["both_metric_report_sha256"]
        ),
        both_provenance_sha256=cast(
            Mapping[str, str], payload_without_hash["both_provenance_sha256"]
        ),
        cpu_shape_smoke_status_by_baseline=cast(
            Mapping[str, bool], payload_without_hash["cpu_shape_smoke_status_by_baseline"]
        ),
        overfit_status_by_baseline=cast(
            Mapping[str, bool], payload_without_hash["overfit_status_by_baseline"]
        ),
        initial_loss_by_baseline=cast(
            Mapping[str, float], payload_without_hash["initial_loss_by_baseline"]
        ),
        final_loss_by_baseline=cast(
            Mapping[str, float], payload_without_hash["final_loss_by_baseline"]
        ),
        best_loss_by_baseline=cast(
            Mapping[str, float], payload_without_hash["best_loss_by_baseline"]
        ),
        training_step_count_by_baseline=cast(
            Mapping[str, int], payload_without_hash["training_step_count_by_baseline"]
        ),
        deterministic_metric_json_verified_by_baseline=cast(
            Mapping[str, bool],
            payload_without_hash["deterministic_metric_json_verified_by_baseline"],
        ),
        mlflow_metadata_verified_by_baseline=cast(
            Mapping[str, bool], payload_without_hash["mlflow_metadata_verified_by_baseline"]
        ),
        overall_status="completed",
        failure_codes=(),
    )


def run_phase3_gate3_synthetic(
    *,
    output_root: Path,
    git_commit: str,
    run_identifier: str,
    start_timestamp: str,
    end_timestamp: str,
    baseline_environment_lock_path: Path,
) -> Phase3Gate3ExecutionResult:
    """Run the reproducible synthetic Gate 3 orchestration outside the repository."""

    _validate_output_root(output_root)
    output_root.mkdir(mode=0o700)
    fixture_root = output_root / "synthetic_fixture"
    fixture_result = generate_baseline_synthetic_fixture(fixture_root)
    fixture_artifact_path = fixture_root / "baseline_synthetic_fixture.json"
    package_versions = {
        name: importlib.metadata.version(name)
        for name in ("mlflow", "monai", "nnunetv2", "numpy", "scipy", "torch", "torchvision")
    }
    metric_config_sha256 = sha256_json(
        {
            "metric_contract_version": "baseline_metric_report_v1",
            "nsd_tolerance_mm": 1.0,
            "surface_connectivity": 6,
            "lesion_connectivity": 26,
        }
    )
    torch = _import_torch()
    configure_phase3_cpu_torch_runtime(torch=torch)
    nnunet_run_config = build_nnunet_v2_run_config(seed=MONAI_SEGRESNET_DEFAULT_SEED)
    monai_run_config = build_monai_segresnet_run_config()
    nnunet_paths = create_validated_run_directory_tree(
        validate_baseline_run_root(output_root / "nnunet_v2_run", baseline_family="nnunet_v2")
    )
    monai_paths = create_validated_run_directory_tree(
        validate_baseline_run_root(
            output_root / "monai_segresnet_run",
            baseline_family="monai_segresnet",
        )
    )
    try:
        nnunet_result = run_nnunet_tiny_baseline(
            run_config=nnunet_run_config,
            run_paths=nnunet_paths,
            fixture_root=fixture_root,
            fixture_artifact_path=fixture_artifact_path,
            run_identifier=f"{run_identifier}_nnunet_v2",
            git_commit=git_commit,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            dataset_manifest_sha256=fixture_result.artifact.artifact_hash,
            development_split_sha256=fixture_result.artifact.artifact_hash,
            metric_config_sha256=metric_config_sha256,
            nsd_tolerance_mm=1.0,
            package_versions=package_versions,
            python_version=platform.python_version(),
            platform_system=platform.system(),
            platform_machine=platform.machine(),
        )
        monai_result = run_monai_segresnet_baseline(
            run_config=monai_run_config,
            run_paths=monai_paths,
            fixture_root=fixture_root,
            fixture_artifact_path=fixture_artifact_path,
            run_identifier=f"{run_identifier}_monai_segresnet",
            git_commit=git_commit,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            dataset_manifest_sha256=fixture_result.artifact.artifact_hash,
            development_split_sha256=fixture_result.artifact.artifact_hash,
            metric_config_sha256=metric_config_sha256,
            nsd_tolerance_mm=1.0,
            package_versions=package_versions,
            python_version=platform.python_version(),
            platform_system=platform.system(),
            platform_machine=platform.machine(),
        )
    except Phase3MlflowMetadataError as exc:
        raise Phase3Gate3ExecutionError("phase3_mlflow_metadata_invalid") from exc
    _verify_persisted_baseline_provenance(
        provenance_path=nnunet_result.provenance_path,
        expected_provenance=nnunet_result.provenance,
        checkpoint_path=nnunet_result.checkpoint_path,
        prediction_manifest_path=nnunet_paths.metrics_dir / "prediction_manifest.json",
        metric_report_path=nnunet_paths.metrics_dir / "metric_report.json",
    )
    _verify_persisted_baseline_provenance(
        provenance_path=monai_result.provenance_path,
        expected_provenance=monai_result.provenance,
        checkpoint_path=monai_result.checkpoint_path,
        prediction_manifest_path=monai_paths.metrics_dir / "prediction_manifest.json",
        metric_report_path=monai_paths.metrics_dir / "metric_report.json",
    )
    try:
        audit_phase3_mlflow_tracking_directory(tracking_directory=nnunet_paths.mlflow_dir)
        audit_phase3_mlflow_tracking_directory(tracking_directory=monai_paths.mlflow_dir)
    except Phase3MlflowMetadataError as exc:
        raise Phase3Gate3ExecutionError("phase3_mlflow_metadata_invalid") from exc
    _require_mlflow_verification(
        nnunet_result.mlflow_verification,
        baseline_family="nnunet_v2",
        run_identifier=f"{run_identifier}_nnunet_v2",
    )
    _require_mlflow_verification(
        monai_result.mlflow_verification,
        baseline_family="monai_segresnet",
        run_identifier=f"{run_identifier}_monai_segresnet",
    )
    _audit_nnunet_logs(run_paths=nnunet_paths)
    _audit_project_json_artifacts(output_root=output_root)
    report = build_phase3_gate3_report(
        run_identifier=run_identifier,
        git_commit=git_commit,
        baseline_environment_lock_sha256=sha256_file(baseline_environment_lock_path),
        synthetic_fixture_artifact_sha256=fixture_result.artifact.artifact_hash,
        nnunet_run_config=nnunet_run_config,
        monai_run_config=monai_run_config,
        nnunet_result=nnunet_result,
        monai_result=monai_result,
    )
    report_path = output_root / "phase3_gate3_report.json"
    report_path.write_bytes(phase3_gate3_report_to_json(report))
    return Phase3Gate3ExecutionResult(
        fixture_artifact_sha256=fixture_result.artifact.artifact_hash,
        monai_result=monai_result,
        nnunet_result=nnunet_result,
        report=report,
        report_path=report_path,
    )


def _import_torch() -> Any:
    return import_module("torch")


def phase3_gate3_report_to_json(report: Phase3Gate3Report) -> bytes:
    """Serialize a deterministic Gate 3 report artifact."""

    return canonical_json_bytes(_report_payload(report)) + b"\n"


def phase3_gate3_report_from_json(payload: bytes | str) -> Phase3Gate3Report:
    """Parse and validate a deterministic Gate 3 report artifact."""

    text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
    try:
        parsed = json.loads(text)
    except Exception as error:
        raise Phase3Gate3SerializationError(
            f"Failed to parse Gate 3 report JSON: {error}"
        ) from error
    return Phase3Gate3Report(
        artifact_hash=parsed["artifact_hash"],
        contract_version=parsed["contract_version"],
        run_identifier=parsed["run_identifier"],
        git_commit=parsed["git_commit"],
        baseline_environment_lock_sha256=parsed["baseline_environment_lock_sha256"],
        synthetic_fixture_artifact_sha256=parsed["synthetic_fixture_artifact_sha256"],
        selected_device=parsed["selected_device"],
        amp_enabled=parsed["amp_enabled"],
        nnunet_execution_strategy_identifier=parsed["nnunet_execution_strategy_identifier"],
        nnunet_run_config_sha256=parsed["nnunet_run_config_sha256"],
        monai_run_config_sha256=parsed["monai_run_config_sha256"],
        both_checkpoint_sha256=parsed["both_checkpoint_sha256"],
        both_prediction_manifest_sha256=parsed["both_prediction_manifest_sha256"],
        both_metric_report_sha256=parsed["both_metric_report_sha256"],
        both_provenance_sha256=parsed["both_provenance_sha256"],
        cpu_shape_smoke_status_by_baseline=parsed["cpu_shape_smoke_status_by_baseline"],
        overfit_status_by_baseline=parsed["overfit_status_by_baseline"],
        initial_loss_by_baseline=parsed["initial_loss_by_baseline"],
        final_loss_by_baseline=parsed["final_loss_by_baseline"],
        best_loss_by_baseline=parsed["best_loss_by_baseline"],
        training_step_count_by_baseline=parsed["training_step_count_by_baseline"],
        deterministic_metric_json_verified_by_baseline=parsed[
            "deterministic_metric_json_verified_by_baseline"
        ],
        mlflow_metadata_verified_by_baseline=parsed["mlflow_metadata_verified_by_baseline"],
        overall_status=parsed["overall_status"],
        failure_codes=tuple(parsed["failure_codes"]),
    )


def _report_payload(report: Phase3Gate3Report) -> dict[str, Any]:
    return {
        "amp_enabled": report.amp_enabled,
        "artifact_hash": report.artifact_hash,
        "baseline_environment_lock_sha256": report.baseline_environment_lock_sha256,
        "both_checkpoint_sha256": dict(report.both_checkpoint_sha256),
        "both_metric_report_sha256": dict(report.both_metric_report_sha256),
        "both_prediction_manifest_sha256": dict(report.both_prediction_manifest_sha256),
        "both_provenance_sha256": dict(report.both_provenance_sha256),
        "contract_version": report.contract_version,
        "cpu_shape_smoke_status_by_baseline": dict(report.cpu_shape_smoke_status_by_baseline),
        "deterministic_metric_json_verified_by_baseline": dict(
            report.deterministic_metric_json_verified_by_baseline
        ),
        "failure_codes": list(report.failure_codes),
        "git_commit": report.git_commit,
        "initial_loss_by_baseline": dict(report.initial_loss_by_baseline),
        "final_loss_by_baseline": dict(report.final_loss_by_baseline),
        "best_loss_by_baseline": dict(report.best_loss_by_baseline),
        "mlflow_metadata_verified_by_baseline": dict(report.mlflow_metadata_verified_by_baseline),
        "monai_run_config_sha256": report.monai_run_config_sha256,
        "nnunet_execution_strategy_identifier": report.nnunet_execution_strategy_identifier,
        "nnunet_run_config_sha256": report.nnunet_run_config_sha256,
        "overall_status": report.overall_status,
        "overfit_status_by_baseline": dict(report.overfit_status_by_baseline),
        "run_identifier": report.run_identifier,
        "selected_device": report.selected_device,
        "synthetic_fixture_artifact_sha256": report.synthetic_fixture_artifact_sha256,
        "training_step_count_by_baseline": dict(report.training_step_count_by_baseline),
    }


def _report_payload_without_hash(report: Phase3Gate3Report) -> dict[str, Any]:
    payload = _report_payload(report)
    del payload["artifact_hash"]
    return payload


def _validate_output_root(path: Path) -> None:
    if not path.is_absolute():
        raise Phase3Gate3ValidationError("output_root must be an absolute path.")
    if path.exists():
        raise Phase3Gate3ValidationError("output_root must not already exist.")
    if not path.parent.exists():
        raise Phase3Gate3ValidationError("output_root parent must already exist.")
    repo_root = Path(__file__).resolve().parents[3]
    if path.resolve(strict=False) == repo_root or repo_root in path.resolve(strict=False).parents:
        raise Phase3Gate3ValidationError("output_root must be outside the repository.")


def _require_sha256(value: str, *, field_name: str) -> None:
    if not _SHA256_RE.fullmatch(value):
        raise Phase3Gate3ValidationError(f"{field_name} must be a lowercase SHA-256 hex.")
