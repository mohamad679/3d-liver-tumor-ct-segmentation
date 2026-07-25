"""Local-only MLflow tracking for Phase 1 synthetic pipeline verification."""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import TypeVar
from urllib.parse import urlparse

import mlflow
from mlflow.entities import Experiment
from mlflow.tracking import MlflowClient

from protoem_ct.artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    ArtifactError,
    EvaluationArtifact,
    InferenceArtifact,
    PreprocessArtifact,
    ReportArtifact,
    SyntheticManifest,
    ValidationArtifact,
    artifact_from_json,
    hash_manifest,
    sha256_file,
)
from protoem_ct.artifacts.hashing import HashFileError, HashingError
from protoem_ct.data.synthetic import synthetic_manifest_hash_payload

PROJECT_TAG = "protoem-ct"
PHASE_TAG = "1"
DATA_KIND_TAG = "synthetic"
INFERENCE_METHOD_TAG = "dummy"
PURPOSE_TAG = "pipeline_verification"
SCIENTIFIC_RESULT_TAG = "false"
CLINICAL_RESULT_TAG = "false"

LOCAL_MLFLOW_METRIC_NAMES = (
    "macro_dice",
    "macro_iou",
    "micro_dice",
    "micro_iou",
    "aggregate_tp",
    "aggregate_fp",
    "aggregate_fn",
    "aggregate_tn",
    "evaluated_case_count",
)
LOCAL_MLFLOW_ARTIFACT_NAMES = (
    "json/synthetic_manifest.json",
    "json/validation_artifact.json",
    "json/preprocess_artifact.json",
    "json/inference_artifact.json",
    "json/evaluation_artifact.json",
    "reports/synthetic_report.md",
    "json/report_artifact.json",
)

_JSON_ARTIFACT_PATHS = (
    ("json", "synthetic_manifest.json"),
    ("json", "validation_artifact.json"),
    ("json", "preprocess_artifact.json"),
    ("json", "inference_artifact.json"),
    ("json", "evaluation_artifact.json"),
    ("json", "report_artifact.json"),
)

ArtifactT = TypeVar(
    "ArtifactT",
    SyntheticManifest,
    ValidationArtifact,
    PreprocessArtifact,
    InferenceArtifact,
    EvaluationArtifact,
    ReportArtifact,
)


class LocalMlflowTrackingError(ValueError):
    """Base error for Phase 1 local MLflow tracking failures."""


class LocalMlflowInputArtifactError(LocalMlflowTrackingError):
    """Raised when persisted tracking inputs are missing, invalid, or inconsistent."""


class LocalMlflowPathError(LocalMlflowTrackingError):
    """Raised when the local tracking path or supplied file paths are invalid."""


class LocalMlflowArtifactError(LocalMlflowTrackingError):
    """Raised when the Markdown report or report artifact is missing or invalid."""


class LocalMlflowTrackingFailureError(LocalMlflowTrackingError):
    """Raised when MLflow tracking cannot complete."""


@dataclass(frozen=True, slots=True)
class LocalMlflowRunResult:
    """Immutable summary of one locally tracked MLflow run."""

    experiment_name: str
    experiment_id: str
    run_name: str
    run_id: str
    run_status: str
    tracking_root: Path
    config_hash: str
    manifest_hash: str
    logged_metric_names: tuple[str, ...]
    logged_artifact_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _TrackingInputPaths:
    manifest: Path
    validation_artifact: Path
    preprocess_artifact: Path
    inference_artifact: Path
    evaluation_artifact: Path
    report: Path
    report_artifact: Path


@dataclass(frozen=True, slots=True)
class _TrackingInputs:
    manifest: SyntheticManifest
    validation: ValidationArtifact
    preprocess: PreprocessArtifact
    inference: InferenceArtifact
    evaluation: EvaluationArtifact
    report: ReportArtifact


def track_synthetic_run(
    manifest_path: Path,
    validation_artifact_path: Path,
    preprocess_artifact_path: Path,
    inference_artifact_path: Path,
    evaluation_artifact_path: Path,
    report_path: Path,
    report_artifact_path: Path,
    *,
    tracking_root: Path,
    experiment_name: str,
    run_name: str,
) -> LocalMlflowRunResult:
    """Track a Phase 1 synthetic pipeline verification run in a local MLflow store."""
    _require_nonempty_text(experiment_name, "experiment_name")
    _require_nonempty_text(run_name, "run_name")
    tracking_root_resolved = _prepare_tracking_root(tracking_root)
    input_paths = _TrackingInputPaths(
        manifest=manifest_path,
        validation_artifact=validation_artifact_path,
        preprocess_artifact=preprocess_artifact_path,
        inference_artifact=inference_artifact_path,
        evaluation_artifact=evaluation_artifact_path,
        report=report_path,
        report_artifact=report_artifact_path,
    )
    _require_input_files(input_paths)
    inputs = _load_and_verify_inputs(input_paths)
    tracking_uri = tracking_root_resolved.as_uri()

    try:
        _allow_mlflow_file_store()
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient(tracking_uri=tracking_uri)
        experiment_id = _get_or_create_experiment_id(
            client,
            experiment_name=experiment_name,
        )
        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name=run_name,
            tags=_tracking_tags(),
            log_system_metrics=False,
        ) as active_run:
            run_id = active_run.info.run_id
            _log_params(inputs)
            _log_metrics(inputs.evaluation)
            _log_artifacts(
                input_paths,
                tracking_root=tracking_root_resolved,
            )
        run = client.get_run(run_id)
    except LocalMlflowTrackingError:
        raise
    except Exception as exc:
        msg = "local MLflow tracking failed"
        raise LocalMlflowTrackingFailureError(msg) from exc

    return LocalMlflowRunResult(
        experiment_name=experiment_name,
        experiment_id=experiment_id,
        run_name=run_name,
        run_id=run_id,
        run_status=run.info.status,
        tracking_root=tracking_root_resolved,
        config_hash=inputs.evaluation.config_hash,
        manifest_hash=inputs.evaluation.manifest_hash,
        logged_metric_names=LOCAL_MLFLOW_METRIC_NAMES,
        logged_artifact_names=LOCAL_MLFLOW_ARTIFACT_NAMES,
    )


def _prepare_tracking_root(tracking_root: Path) -> Path:
    _require_no_uri_scheme(tracking_root)
    _require_no_parent_traversal(tracking_root, "tracking_root")
    if not tracking_root.is_absolute():
        msg = "tracking_root must be an explicit absolute local filesystem path"
        raise LocalMlflowPathError(msg)
    if tracking_root.exists() and not tracking_root.is_dir():
        msg = "tracking_root exists and is not a directory"
        raise LocalMlflowPathError(msg)
    try:
        tracking_root.mkdir(parents=True, exist_ok=True)
        resolved = tracking_root.resolve(strict=True)
    except OSError as exc:
        msg = "tracking_root could not be created or resolved"
        raise LocalMlflowPathError(msg) from exc
    if not resolved.is_dir():
        msg = "tracking_root is not a directory"
        raise LocalMlflowPathError(msg)
    return resolved


def _allow_mlflow_file_store() -> None:
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"


def _require_input_files(input_paths: _TrackingInputPaths) -> None:
    for path, role in (
        (input_paths.manifest, "synthetic manifest"),
        (input_paths.validation_artifact, "validation artifact"),
        (input_paths.preprocess_artifact, "preprocessing artifact"),
        (input_paths.inference_artifact, "inference artifact"),
        (input_paths.evaluation_artifact, "evaluation artifact"),
        (input_paths.report, "Markdown report"),
        (input_paths.report_artifact, "report artifact"),
    ):
        _require_no_parent_traversal(path, role)
        if not path.exists() or not path.is_file():
            if role == "Markdown report":
                msg = "Markdown report is missing or is not a regular file"
                raise LocalMlflowArtifactError(msg)
            if role == "report artifact":
                msg = "report artifact is missing or is not a regular file"
                raise LocalMlflowArtifactError(msg)
            msg = f"{role} is missing or is not a regular file"
            raise LocalMlflowInputArtifactError(msg)


def _load_and_verify_inputs(input_paths: _TrackingInputPaths) -> _TrackingInputs:
    manifest = _load_artifact(input_paths.manifest, SyntheticManifest, "synthetic manifest")
    validation = _load_artifact(
        input_paths.validation_artifact,
        ValidationArtifact,
        "validation",
    )
    preprocess = _load_artifact(
        input_paths.preprocess_artifact,
        PreprocessArtifact,
        "preprocessing",
    )
    inference = _load_artifact(
        input_paths.inference_artifact,
        InferenceArtifact,
        "inference",
    )
    evaluation = _load_artifact(
        input_paths.evaluation_artifact,
        EvaluationArtifact,
        "evaluation",
    )
    report = _load_artifact(input_paths.report_artifact, ReportArtifact, "report")
    inputs = _TrackingInputs(
        manifest=manifest,
        validation=validation,
        preprocess=preprocess,
        inference=inference,
        evaluation=evaluation,
        report=report,
    )
    _verify_linkage(inputs)
    _verify_report_hash(input_paths.report, report)
    return inputs


def _load_artifact(path: Path, artifact_type: type[ArtifactT], role: str) -> ArtifactT:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        msg = f"failed to read {role} artifact"
        raise LocalMlflowInputArtifactError(msg) from exc
    try:
        return artifact_from_json(text, artifact_type)
    except ArtifactError as exc:
        msg = f"invalid {role} artifact: {exc}"
        raise LocalMlflowInputArtifactError(msg) from exc


def _verify_linkage(inputs: _TrackingInputs) -> None:
    schema_versions = {
        inputs.manifest.schema_version,
        inputs.validation.schema_version,
        inputs.preprocess.schema_version,
        inputs.inference.schema_version,
        inputs.evaluation.schema_version,
        inputs.report.schema_version,
    }
    if schema_versions != {ARTIFACT_SCHEMA_VERSION}:
        msg = "input artifact schema versions are incompatible"
        raise LocalMlflowInputArtifactError(msg)
    _verify_manifest_hash(inputs.manifest)

    if inputs.validation.config_hash != inputs.manifest.config_hash:
        msg = "validation config hash does not match synthetic manifest config hash"
        raise LocalMlflowInputArtifactError(msg)
    if inputs.preprocess.config_hash != inputs.inference.config_hash:
        msg = "preprocessing and inference config hashes do not match"
        raise LocalMlflowInputArtifactError(msg)
    if inputs.preprocess.config_hash != inputs.evaluation.config_hash:
        msg = "preprocessing and evaluation config hashes do not match"
        raise LocalMlflowInputArtifactError(msg)
    if inputs.evaluation.config_hash != inputs.report.config_hash:
        msg = "evaluation and report config hashes do not match"
        raise LocalMlflowInputArtifactError(msg)

    manifest_hashes = {
        inputs.manifest.manifest_hash,
        inputs.validation.manifest_hash,
        inputs.preprocess.manifest_hash,
        inputs.inference.manifest_hash,
        inputs.evaluation.manifest_hash,
        inputs.report.manifest_hash,
    }
    if len(manifest_hashes) != 1:
        msg = "manifest hashes do not match across tracking inputs"
        raise LocalMlflowInputArtifactError(msg)

    expected_case_ids = inputs.manifest.case_ids
    if inputs.validation.invalid_case_count != 0:
        msg = "validation invalid_case_count must equal zero"
        raise LocalMlflowInputArtifactError(msg)
    if inputs.validation.valid_case_count != len(expected_case_ids):
        msg = "validation valid_case_count must match manifest case count"
        raise LocalMlflowInputArtifactError(msg)
    if inputs.validation.validated_case_ids != expected_case_ids:
        msg = "validation case IDs must match manifest case IDs in order"
        raise LocalMlflowInputArtifactError(msg)
    if inputs.preprocess.output_case_ids != expected_case_ids:
        msg = "preprocessing case IDs must match manifest case IDs in order"
        raise LocalMlflowInputArtifactError(msg)
    if inputs.inference.prediction_case_ids != expected_case_ids:
        msg = "inference case IDs must match preprocessing case IDs in order"
        raise LocalMlflowInputArtifactError(msg)
    if inputs.evaluation.case_ids != expected_case_ids:
        msg = "evaluation case IDs must match inference case IDs in order"
        raise LocalMlflowInputArtifactError(msg)
    if tuple(metric.case_id for metric in inputs.evaluation.case_metrics) != expected_case_ids:
        msg = "evaluation per-case metrics must match case IDs in order"
        raise LocalMlflowInputArtifactError(msg)

    case_count = len(expected_case_ids)
    if len(inputs.preprocess.output_case_ids) != case_count:
        msg = "preprocessing case count must match manifest case count"
        raise LocalMlflowInputArtifactError(msg)
    if len(inputs.inference.prediction_case_ids) != case_count:
        msg = "inference case count must match preprocessing case count"
        raise LocalMlflowInputArtifactError(msg)
    if inputs.evaluation.evaluated_case_count != case_count:
        msg = "evaluation case count must match inference case count"
        raise LocalMlflowInputArtifactError(msg)
    if inputs.report.evaluated_case_count != inputs.evaluation.evaluated_case_count:
        msg = "report evaluated case count must match evaluation case count"
        raise LocalMlflowInputArtifactError(msg)
    if inputs.report.macro_mean_dice != inputs.evaluation.macro_mean_dice:
        msg = "report macro Dice must match evaluation artifact"
        raise LocalMlflowInputArtifactError(msg)
    if inputs.report.macro_mean_iou != inputs.evaluation.macro_mean_iou:
        msg = "report macro IoU must match evaluation artifact"
        raise LocalMlflowInputArtifactError(msg)
    if inputs.report.micro_dice != inputs.evaluation.micro_dice:
        msg = "report micro Dice must match evaluation artifact"
        raise LocalMlflowInputArtifactError(msg)
    if inputs.report.micro_iou != inputs.evaluation.micro_iou:
        msg = "report micro IoU must match evaluation artifact"
        raise LocalMlflowInputArtifactError(msg)
    if inputs.inference.method != INFERENCE_METHOD_TAG:
        msg = "inference method must be fixed to 'dummy'"
        raise LocalMlflowInputArtifactError(msg)
    if inputs.evaluation.method != INFERENCE_METHOD_TAG:
        msg = "evaluation method must be fixed to 'dummy'"
        raise LocalMlflowInputArtifactError(msg)
    if inputs.report.method != INFERENCE_METHOD_TAG:
        msg = "report method must be fixed to 'dummy'"
        raise LocalMlflowInputArtifactError(msg)
    _require_preprocessing_parameters(inputs.preprocess)


def _verify_manifest_hash(manifest: SyntheticManifest) -> None:
    try:
        recomputed_hash = hash_manifest(synthetic_manifest_hash_payload(manifest))
    except (ArtifactError, HashingError) as exc:
        msg = "failed to recompute synthetic manifest hash"
        raise LocalMlflowInputArtifactError(msg) from exc
    if recomputed_hash != manifest.manifest_hash:
        msg = "synthetic manifest hash mismatch"
        raise LocalMlflowInputArtifactError(msg)


def _verify_report_hash(report_path: Path, report_artifact: ReportArtifact) -> None:
    try:
        report_sha256 = sha256_file(report_path)
    except HashFileError as exc:
        msg = "failed to hash Markdown report"
        raise LocalMlflowArtifactError(msg) from exc
    if report_sha256 != report_artifact.report_sha256:
        msg = "Markdown report SHA-256 does not match report artifact"
        raise LocalMlflowArtifactError(msg)


def _require_preprocessing_parameters(preprocess: PreprocessArtifact) -> None:
    expected_keys = {
        "clip_max",
        "clip_min",
        "image_dtype",
        "label_dtype",
        "output_max",
        "output_min",
        "resampling",
    }
    if set(preprocess.preprocessing_parameters) != expected_keys:
        msg = "preprocessing parameters do not contain the expected Phase 1 fields"
        raise LocalMlflowInputArtifactError(msg)


def _get_or_create_experiment_id(client: MlflowClient, *, experiment_name: str) -> str:
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return client.create_experiment(experiment_name)
    return _experiment_id(experiment)


def _experiment_id(experiment: Experiment) -> str:
    return str(experiment.experiment_id)


def _tracking_tags() -> dict[str, str]:
    return {
        "project": PROJECT_TAG,
        "phase": PHASE_TAG,
        "data_kind": DATA_KIND_TAG,
        "inference_method": INFERENCE_METHOD_TAG,
        "purpose": PURPOSE_TAG,
        "scientific_result": SCIENTIFIC_RESULT_TAG,
        "clinical_result": CLINICAL_RESULT_TAG,
    }


def _log_params(inputs: _TrackingInputs) -> None:
    parameters = inputs.preprocess.preprocessing_parameters
    params = {
        "schema_version": inputs.evaluation.schema_version,
        "dataset_id": inputs.manifest.dataset_id,
        "case_count": str(len(inputs.manifest.case_ids)),
        "image_shape": _format_int_tuple(inputs.manifest.shape),
        "spacing": _format_float_tuple(inputs.manifest.spacing),
        "preprocessing_clip_range": (
            f"{_format_param(parameters['clip_min'])},{_format_param(parameters['clip_max'])}"
        ),
        "preprocessing_output_range": (
            f"{_format_param(parameters['output_min'])},{_format_param(parameters['output_max'])}"
        ),
        "preprocessing_image_dtype": _format_param(parameters["image_dtype"]),
        "preprocessing_label_dtype": _format_param(parameters["label_dtype"]),
        "preprocessing_resampling": _format_param(parameters["resampling"]),
        "inference_seed": str(inputs.inference.deterministic_seed),
        "inference_threshold": _format_param(inputs.inference.threshold),
        "inference_dtype": inputs.inference.prediction_dtype,
        "config_hash": inputs.evaluation.config_hash,
        "manifest_hash": inputs.evaluation.manifest_hash,
        "git_commit_manifest": inputs.manifest.git_commit,
        "git_commit_validation": inputs.validation.git_commit,
        "git_commit_preprocess": inputs.preprocess.git_commit,
        "git_commit_inference": inputs.inference.git_commit,
        "git_commit_evaluation": inputs.evaluation.git_commit,
        "git_commit_report": inputs.report.git_commit,
    }
    for key in sorted(params):
        mlflow.log_param(key, params[key])


def _log_metrics(evaluation: EvaluationArtifact) -> None:
    values = {
        "macro_dice": evaluation.macro_mean_dice,
        "macro_iou": evaluation.macro_mean_iou,
        "micro_dice": evaluation.micro_dice,
        "micro_iou": evaluation.micro_iou,
        "aggregate_tp": float(evaluation.total_true_positives),
        "aggregate_fp": float(evaluation.total_false_positives),
        "aggregate_fn": float(evaluation.total_false_negatives),
        "aggregate_tn": float(evaluation.total_true_negatives),
        "evaluated_case_count": float(evaluation.evaluated_case_count),
    }
    for name in LOCAL_MLFLOW_METRIC_NAMES:
        mlflow.log_metric(name, values[name])


def _log_artifacts(input_paths: _TrackingInputPaths, *, tracking_root: Path) -> None:
    source_paths = (
        input_paths.manifest,
        input_paths.validation_artifact,
        input_paths.preprocess_artifact,
        input_paths.inference_artifact,
        input_paths.evaluation_artifact,
        input_paths.report_artifact,
    )
    with tempfile.TemporaryDirectory(
        prefix=".protoem_ct_mlflow_staging_",
        dir=tracking_root,
    ) as tmp:
        staging_root = Path(tmp)
        for source_path, (artifact_subdir, artifact_filename) in zip(
            source_paths,
            _JSON_ARTIFACT_PATHS,
            strict=True,
        ):
            staged_path = staging_root / artifact_filename
            shutil.copyfile(source_path, staged_path)
            mlflow.log_artifact(str(staged_path), artifact_path=artifact_subdir)

        staged_report_path = staging_root / "synthetic_report.md"
        shutil.copyfile(input_paths.report, staged_report_path)
        mlflow.log_artifact(str(staged_report_path), artifact_path="reports")


def _format_int_tuple(values: tuple[int, int, int]) -> str:
    return "x".join(str(value) for value in values)


def _format_float_tuple(values: tuple[float, float, float]) -> str:
    return ",".join(_format_param(value) for value in values)


def _format_param(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _require_nonempty_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        msg = f"{field_name} must be a nonempty string"
        raise LocalMlflowInputArtifactError(msg)


def _require_no_uri_scheme(path: Path) -> None:
    parsed = urlparse(str(path))
    if parsed.scheme:
        msg = "tracking_root must be a local filesystem path, not a URI"
        raise LocalMlflowPathError(msg)


def _require_no_parent_traversal(path: Path, role: str) -> None:
    value = str(path)
    if ".." in PurePosixPath(value).parts or ".." in PureWindowsPath(value).parts:
        msg = f"{role} must not contain parent traversal"
        raise LocalMlflowPathError(msg)
