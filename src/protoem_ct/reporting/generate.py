"""Deterministic Phase 1 synthetic Markdown report generation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import TypeVar, cast

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
    artifact_to_json,
    hash_manifest,
)
from protoem_ct.artifacts.hashing import HashingError, JsonValue
from protoem_ct.data.synthetic import synthetic_manifest_hash_payload
from protoem_ct.evaluation.dummy_inference import DUMMY_INFERENCE_METHOD, DUMMY_INFERENCE_STAGE
from protoem_ct.evaluation.metrics import EVALUATION_STAGE

REPORT_STAGE = "reporting"
REPORT_FORMAT = "markdown"
REPORT_NUMERIC_PRECISION = 6
REPORTED_METRIC_NAMES = ("dice", "iou")

ArtifactT = TypeVar(
    "ArtifactT",
    SyntheticManifest,
    ValidationArtifact,
    PreprocessArtifact,
    InferenceArtifact,
    EvaluationArtifact,
)


class ReportGenerationError(ValueError):
    """Base error for Phase 1 synthetic report generation failures."""


class ReportInputArtifactError(ReportGenerationError):
    """Raised when report input artifacts are missing, invalid, or inconsistent."""


class ReportPathError(ReportGenerationError):
    """Raised when report input or output paths are unsafe."""


class ReportOutputCollisionError(ReportGenerationError):
    """Raised when report generation would overwrite an existing output."""


class ReportRenderingError(ReportGenerationError):
    """Raised when the Markdown report or report artifact cannot be rendered."""


@dataclass(frozen=True, slots=True)
class _ReportInputs:
    manifest: SyntheticManifest
    validation: ValidationArtifact
    preprocess: PreprocessArtifact
    inference: InferenceArtifact
    evaluation: EvaluationArtifact


@dataclass(frozen=True, slots=True)
class _ReportInputPaths:
    manifest: Path
    validation_artifact: Path
    preprocess_artifact: Path
    inference_artifact: Path
    evaluation_artifact: Path


@dataclass(frozen=True, slots=True)
class _PreparedOutputPaths:
    report_output: Path
    artifact_output: Path


def generate_synthetic_report(
    manifest_path: Path,
    validation_artifact_path: Path,
    preprocess_artifact_path: Path,
    inference_artifact_path: Path,
    evaluation_artifact_path: Path,
    *,
    report_output_path: Path,
    artifact_output_path: Path,
    git_commit: str,
    created_at_utc: str,
) -> ReportArtifact:
    """Generate a deterministic Markdown report from persisted Phase 1 JSON artifacts."""
    _require_nonempty_text(git_commit, "git_commit")
    _require_nonempty_text(created_at_utc, "created_at_utc")

    input_paths = _ReportInputPaths(
        manifest=manifest_path,
        validation_artifact=validation_artifact_path,
        preprocess_artifact=preprocess_artifact_path,
        inference_artifact=inference_artifact_path,
        evaluation_artifact=evaluation_artifact_path,
    )

    report_temp_path: Path | None = None
    artifact_temp_path: Path | None = None
    published_report = False
    published_artifact = False
    prepared_outputs: _PreparedOutputPaths | None = None
    try:
        inputs = _load_and_verify_inputs(input_paths)
        report_text = _render_markdown_report(
            inputs,
            input_paths=input_paths,
            report_output_path=report_output_path,
            artifact_output_path=artifact_output_path,
            git_commit=git_commit,
            created_at_utc=created_at_utc,
        )
        report_sha256 = _sha256_text(report_text)
        report_artifact = _build_report_artifact(
            inputs,
            input_paths=input_paths,
            report_output_path=report_output_path,
            report_sha256=report_sha256,
            git_commit=git_commit,
            created_at_utc=created_at_utc,
        )
        prepared_outputs = _prepare_output_paths(report_output_path, artifact_output_path)
        report_temp_path = _write_report_to_temp(report_text, prepared_outputs.report_output)
        artifact_temp_path = _write_artifact_to_temp(
            report_artifact,
            prepared_outputs.artifact_output,
        )
        _publish_file(report_temp_path, prepared_outputs.report_output, "report")
        published_report = True
        _publish_file(artifact_temp_path, prepared_outputs.artifact_output, "report artifact")
        published_artifact = True
        return report_artifact
    except ReportGenerationError:
        _cleanup_after_failure(
            report_temp_path=report_temp_path,
            artifact_temp_path=artifact_temp_path,
            prepared_outputs=prepared_outputs,
            published_report=published_report,
            published_artifact=published_artifact,
        )
        raise
    except Exception as exc:
        _cleanup_after_failure(
            report_temp_path=report_temp_path,
            artifact_temp_path=artifact_temp_path,
            prepared_outputs=prepared_outputs,
            published_report=published_report,
            published_artifact=published_artifact,
        )
        msg = f"report rendering failed: {exc}"
        raise ReportRenderingError(msg) from exc


def _load_and_verify_inputs(input_paths: _ReportInputPaths) -> _ReportInputs:
    for path in (
        input_paths.manifest,
        input_paths.validation_artifact,
        input_paths.preprocess_artifact,
        input_paths.inference_artifact,
        input_paths.evaluation_artifact,
    ):
        _require_no_parent_traversal(path, "input artifact path")

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
    inputs = _ReportInputs(
        manifest=manifest,
        validation=validation,
        preprocess=preprocess,
        inference=inference,
        evaluation=evaluation,
    )
    _verify_artifact_linkage(inputs)
    return inputs


def _load_artifact(path: Path, artifact_type: type[ArtifactT], role: str) -> ArtifactT:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        msg = f"failed to read {role} artifact"
        raise ReportInputArtifactError(msg) from exc

    try:
        return artifact_from_json(text, artifact_type)
    except ArtifactError as exc:
        msg = f"invalid {role} artifact: {exc}"
        raise ReportInputArtifactError(msg) from exc


def _verify_artifact_linkage(inputs: _ReportInputs) -> None:
    schema_versions = {
        inputs.manifest.schema_version,
        inputs.validation.schema_version,
        inputs.preprocess.schema_version,
        inputs.inference.schema_version,
        inputs.evaluation.schema_version,
    }
    if schema_versions != {ARTIFACT_SCHEMA_VERSION}:
        msg = "input artifact schema versions are incompatible"
        raise ReportInputArtifactError(msg)
    _verify_manifest_hash(inputs.manifest)

    if inputs.validation.config_hash != inputs.manifest.config_hash:
        msg = "validation config hash does not match synthetic manifest config hash"
        raise ReportInputArtifactError(msg)
    if inputs.preprocess.config_hash != inputs.inference.config_hash:
        msg = "preprocessing and inference config hashes do not match"
        raise ReportInputArtifactError(msg)
    if inputs.preprocess.config_hash != inputs.evaluation.config_hash:
        msg = "preprocessing and evaluation config hashes do not match"
        raise ReportInputArtifactError(msg)

    manifest_hashes = {
        inputs.manifest.manifest_hash,
        inputs.validation.manifest_hash,
        inputs.preprocess.manifest_hash,
        inputs.inference.manifest_hash,
        inputs.evaluation.manifest_hash,
    }
    if len(manifest_hashes) != 1:
        msg = "manifest hashes do not match across report input artifacts"
        raise ReportInputArtifactError(msg)

    if inputs.validation.invalid_case_count != 0:
        msg = "validation invalid_case_count must equal zero"
        raise ReportInputArtifactError(msg)
    if inputs.validation.valid_case_count != len(inputs.manifest.case_ids):
        msg = "validation valid_case_count must match manifest case count"
        raise ReportInputArtifactError(msg)

    expected_case_ids = inputs.manifest.case_ids
    if inputs.validation.validated_case_ids != expected_case_ids:
        msg = "validation case IDs must match manifest case IDs in order"
        raise ReportInputArtifactError(msg)
    if inputs.preprocess.output_case_ids != expected_case_ids:
        msg = "preprocessing case IDs must match manifest case IDs in order"
        raise ReportInputArtifactError(msg)
    if inputs.inference.prediction_case_ids != expected_case_ids:
        msg = "inference case IDs must match preprocessing case IDs in order"
        raise ReportInputArtifactError(msg)
    if inputs.evaluation.case_ids != expected_case_ids:
        msg = "evaluation case IDs must match inference case IDs in order"
        raise ReportInputArtifactError(msg)
    if tuple(metric.case_id for metric in inputs.evaluation.case_metrics) != expected_case_ids:
        msg = "evaluation per-case metrics must match case IDs in order"
        raise ReportInputArtifactError(msg)

    case_count = len(expected_case_ids)
    if len(inputs.preprocess.output_case_ids) != case_count:
        msg = "preprocessing case count must match manifest case count"
        raise ReportInputArtifactError(msg)
    if len(inputs.inference.prediction_case_ids) != case_count:
        msg = "inference case count must match preprocessing case count"
        raise ReportInputArtifactError(msg)
    if inputs.evaluation.evaluated_case_count != case_count:
        msg = "evaluation case count must match inference case count"
        raise ReportInputArtifactError(msg)
    if inputs.inference.method != DUMMY_INFERENCE_METHOD:
        msg = "inference method must be fixed to 'dummy'"
        raise ReportInputArtifactError(msg)
    if inputs.evaluation.method != DUMMY_INFERENCE_METHOD:
        msg = "evaluation method must be fixed to 'dummy'"
        raise ReportInputArtifactError(msg)
    if inputs.inference.stage != DUMMY_INFERENCE_STAGE:
        msg = "inference stage must be 'infer-dummy'"
        raise ReportInputArtifactError(msg)
    if inputs.evaluation.stage != EVALUATION_STAGE:
        msg = "evaluation stage must be 'evaluate'"
        raise ReportInputArtifactError(msg)


def _verify_manifest_hash(manifest: SyntheticManifest) -> None:
    try:
        recomputed_hash = hash_manifest(synthetic_manifest_hash_payload(manifest))
    except (ArtifactError, HashingError) as exc:
        msg = "failed to recompute synthetic manifest hash"
        raise ReportInputArtifactError(msg) from exc
    if recomputed_hash != manifest.manifest_hash:
        msg = "synthetic manifest hash mismatch"
        raise ReportInputArtifactError(msg)


def _render_markdown_report(
    inputs: _ReportInputs,
    *,
    input_paths: _ReportInputPaths,
    report_output_path: Path,
    artifact_output_path: Path,
    git_commit: str,
    created_at_utc: str,
) -> str:
    lines = [
        "# ProtoEM-CT Phase 1 Synthetic Pipeline Report",
        "",
        "> Warning: This is a synthetic Phase 1 pipeline report for pipeline verification only.",
        (
            "> Warning: Metrics are from dummy inference on synthetic data and are not "
            "scientific model results."
        ),
        "",
        "## Run Metadata",
        "",
        f"- Schema version: {ARTIFACT_SCHEMA_VERSION}",
        f"- Git commit: {git_commit}",
        f"- Created at UTC: {created_at_utc}",
        f"- Report format: {REPORT_FORMAT}",
        f"- Numeric precision: {REPORT_NUMERIC_PRECISION} decimal places",
        "",
        "## Dataset Summary",
        "",
        f"- Dataset ID: {inputs.manifest.dataset_id}",
        f"- Case count: {len(inputs.manifest.case_ids)}",
        f"- Shape: {_format_int_tuple(inputs.manifest.shape)}",
        f"- Spacing: {_format_float_tuple(inputs.manifest.spacing)}",
        f"- Synthetic config hash: {inputs.manifest.config_hash}",
        f"- Manifest hash: {inputs.manifest.manifest_hash}",
        "",
        "## Validation Summary",
        "",
        f"- Valid case count: {inputs.validation.valid_case_count}",
        f"- Invalid case count: {inputs.validation.invalid_case_count}",
        "",
        "## Preprocessing Summary",
        "",
        f"- Target spacing: {_format_float_tuple(inputs.preprocess.target_spacing)}",
        "- Preprocessing parameters:",
        *_format_preprocessing_parameters(inputs.preprocess.preprocessing_parameters),
        "",
        "## Dummy-Inference Summary",
        "",
        f"- Method: {inputs.inference.method}",
        f"- Deterministic seed: {inputs.inference.deterministic_seed}",
        f"- Threshold: {_format_float(inputs.inference.threshold)}",
        f"- Prediction dtype: {inputs.inference.prediction_dtype}",
        "",
        "## Evaluation Summary",
        "",
        f"- Evaluated case count: {inputs.evaluation.evaluated_case_count}",
        f"- Macro Dice: {_format_float(inputs.evaluation.macro_mean_dice)}",
        f"- Macro IoU: {_format_float(inputs.evaluation.macro_mean_iou)}",
        f"- Micro Dice: {_format_float(inputs.evaluation.micro_dice)}",
        f"- Micro IoU: {_format_float(inputs.evaluation.micro_iou)}",
        f"- Total TP: {inputs.evaluation.total_true_positives}",
        f"- Total FP: {inputs.evaluation.total_false_positives}",
        f"- Total FN: {inputs.evaluation.total_false_negatives}",
        f"- Total TN: {inputs.evaluation.total_true_negatives}",
        "",
        "## Ordered Per-Case Metric Table",
        "",
        "| Case ID | TP | FP | FN | TN | GT foreground | Pred foreground | Dice | IoU |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        *_format_case_metric_rows(inputs.evaluation),
        "",
        "## Artifact Linkage",
        "",
        f"- Synthetic manifest: {_safe_path_label(input_paths.manifest)}",
        f"- Validation artifact: {_safe_path_label(input_paths.validation_artifact)}",
        f"- Preprocessing artifact: {_safe_path_label(input_paths.preprocess_artifact)}",
        f"- Inference artifact: {_safe_path_label(input_paths.inference_artifact)}",
        f"- Evaluation artifact: {_safe_path_label(input_paths.evaluation_artifact)}",
        f"- Report output: {_safe_path_label(report_output_path)}",
        f"- Report artifact output: {_safe_path_label(artifact_output_path)}",
        f"- Pipeline config hash: {inputs.evaluation.config_hash}",
        f"- Manifest hash: {inputs.evaluation.manifest_hash}",
        "",
        "## Reproducibility",
        "",
        "- Inputs are persisted JSON artifacts only.",
        "- No image, label, or prediction files are loaded during report generation.",
        "- No MLflow state, Git state, environment metadata, or current time is read.",
        "- Re-running with identical explicit inputs produces byte-identical Markdown.",
    ]
    report_text = "\n".join(lines) + "\n"
    _validate_report_text(report_text)
    return report_text


def _build_report_artifact(
    inputs: _ReportInputs,
    *,
    input_paths: _ReportInputPaths,
    report_output_path: Path,
    report_sha256: str,
    git_commit: str,
    created_at_utc: str,
) -> ReportArtifact:
    try:
        return ReportArtifact(
            schema_version=ARTIFACT_SCHEMA_VERSION,
            stage=REPORT_STAGE,
            created_at_utc=created_at_utc,
            git_commit=git_commit,
            config_hash=inputs.evaluation.config_hash,
            manifest_hash=inputs.evaluation.manifest_hash,
            source_artifact_paths=(
                _safe_path_label(input_paths.manifest),
                _safe_path_label(input_paths.validation_artifact),
                _safe_path_label(input_paths.preprocess_artifact),
                _safe_path_label(input_paths.inference_artifact),
                _safe_path_label(input_paths.evaluation_artifact),
            ),
            report_path=_safe_path_label(report_output_path),
            report_format=REPORT_FORMAT,
            report_sha256=report_sha256,
            evaluated_case_count=inputs.evaluation.evaluated_case_count,
            method=inputs.evaluation.method,
            macro_mean_dice=inputs.evaluation.macro_mean_dice,
            macro_mean_iou=inputs.evaluation.macro_mean_iou,
            micro_dice=inputs.evaluation.micro_dice,
            micro_iou=inputs.evaluation.micro_iou,
            reported_metric_names=REPORTED_METRIC_NAMES,
        )
    except ArtifactError as exc:
        msg = f"failed to construct report artifact: {exc}"
        raise ReportRenderingError(msg) from exc


def _prepare_output_paths(
    report_output_path: Path,
    artifact_output_path: Path,
) -> _PreparedOutputPaths:
    _require_output_candidate(report_output_path, "report_output_path")
    _require_output_candidate(artifact_output_path, "artifact_output_path")

    report_output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        report_parent = report_output_path.parent.resolve(strict=True)
        artifact_parent = artifact_output_path.parent.resolve(strict=True)
    except OSError as exc:
        msg = "failed to resolve report output parent"
        raise ReportPathError(msg) from exc
    if not report_parent.is_dir() or not artifact_parent.is_dir():
        msg = "report output parent must be a directory"
        raise ReportPathError(msg)

    report_output = report_parent / report_output_path.name
    artifact_output = artifact_parent / artifact_output_path.name
    if report_output == artifact_output:
        msg = "report_output_path and artifact_output_path must be different files"
        raise ReportPathError(msg)
    if report_output.exists() or report_output.is_symlink():
        msg = "refusing to overwrite existing report output"
        raise ReportOutputCollisionError(msg)
    if artifact_output.exists() or artifact_output.is_symlink():
        msg = "refusing to overwrite existing report artifact"
        raise ReportOutputCollisionError(msg)

    return _PreparedOutputPaths(
        report_output=report_output,
        artifact_output=artifact_output,
    )


def _require_output_candidate(path: Path, role: str) -> None:
    if not path.name:
        msg = f"{role} must name an output file"
        raise ReportPathError(msg)
    _require_no_parent_traversal(path, role)
    if path.exists() or path.is_symlink():
        msg = f"refusing to overwrite existing {role}"
        raise ReportOutputCollisionError(msg)


def _write_report_to_temp(report_text: str, report_output_path: Path) -> Path:
    descriptor, temp_path_text = tempfile.mkstemp(
        prefix=f".{report_output_path.name}.",
        suffix=".tmp",
        dir=report_output_path.parent,
    )
    os.close(descriptor)
    temp_path = Path(temp_path_text)
    try:
        temp_path.write_text(report_text, encoding="utf-8", newline="\n")
    except OSError as exc:
        temp_path.unlink(missing_ok=True)
        msg = "failed to write staged Markdown report"
        raise ReportRenderingError(msg) from exc
    return temp_path


def _write_artifact_to_temp(artifact: ReportArtifact, artifact_output_path: Path) -> Path:
    descriptor, temp_path_text = tempfile.mkstemp(
        prefix=f".{artifact_output_path.name}.",
        suffix=".tmp",
        dir=artifact_output_path.parent,
    )
    os.close(descriptor)
    temp_path = Path(temp_path_text)
    try:
        artifact_to_json(artifact, temp_path)
    except ArtifactError as exc:
        temp_path.unlink(missing_ok=True)
        msg = "failed to write staged report artifact"
        raise ReportRenderingError(msg) from exc
    return temp_path


def _publish_file(temp_path: Path, output_path: Path, role: str) -> None:
    if output_path.exists() or output_path.is_symlink():
        msg = f"refusing to overwrite existing {role}"
        raise ReportOutputCollisionError(msg)
    temp_path.replace(output_path)


def _cleanup_after_failure(
    *,
    report_temp_path: Path | None,
    artifact_temp_path: Path | None,
    prepared_outputs: _PreparedOutputPaths | None,
    published_report: bool,
    published_artifact: bool,
) -> None:
    if report_temp_path is not None and report_temp_path.exists():
        report_temp_path.unlink(missing_ok=True)
    if artifact_temp_path is not None and artifact_temp_path.exists():
        artifact_temp_path.unlink(missing_ok=True)
    if prepared_outputs is None:
        return
    if published_report and prepared_outputs.report_output.exists():
        prepared_outputs.report_output.unlink(missing_ok=True)
    if published_artifact and prepared_outputs.artifact_output.exists():
        prepared_outputs.artifact_output.unlink(missing_ok=True)


def _format_preprocessing_parameters(parameters: object) -> list[str]:
    if not isinstance(parameters, Mapping):
        msg = "preprocessing parameters must be a mapping"
        raise ReportRenderingError(msg)
    items: dict[str, JsonValue] = {}
    for key, value in parameters.items():
        if not isinstance(key, str):
            msg = "preprocessing parameter keys must be strings"
            raise ReportRenderingError(msg)
        items[key] = cast(JsonValue, value)
    return [f"  - {key}: {_format_json_value(items[key])}" for key in sorted(items)]


def _format_case_metric_rows(evaluation: EvaluationArtifact) -> list[str]:
    rows: list[str] = []
    for metric in evaluation.case_metrics:
        rows.append(
            "| "
            f"{metric.case_id} | "
            f"{metric.true_positives} | "
            f"{metric.false_positives} | "
            f"{metric.false_negatives} | "
            f"{metric.true_negatives} | "
            f"{metric.ground_truth_foreground_voxels} | "
            f"{metric.predicted_foreground_voxels} | "
            f"{_format_float(metric.dice)} | "
            f"{_format_float(metric.iou)} |"
        )
    return rows


def _format_int_tuple(values: tuple[int, int, int]) -> str:
    return " x ".join(str(value) for value in values)


def _format_float_tuple(values: tuple[float, float, float]) -> str:
    return " x ".join(_format_float(value) for value in values)


def _format_json_value(value: JsonValue) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _format_float(value)
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)
    except (TypeError, ValueError) as exc:
        msg = "failed to format preprocessing parameter value"
        raise ReportRenderingError(msg) from exc


def _format_float(value: float) -> str:
    if not isinstance(value, float) or not math.isfinite(value):
        msg = "report numeric value must be finite"
        raise ReportRenderingError(msg)
    return f"{value:.{REPORT_NUMERIC_PRECISION}f}"


def _validate_report_text(report_text: str) -> None:
    if not report_text.endswith("\n") or report_text.endswith("\n\n"):
        msg = "report must contain exactly one trailing newline"
        raise ReportRenderingError(msg)
    forbidden_terms = (
        "model performance",
        "clinical validity",
        "state of the art",
        "generalization",
        "NaN",
        "Infinity",
    )
    lowered = report_text.lower()
    for term in forbidden_terms:
        if term.lower() in lowered:
            msg = f"report contains forbidden term: {term}"
            raise ReportRenderingError(msg)


def _safe_path_label(path: Path) -> str:
    _require_no_parent_traversal(path, "artifact path")
    if path.is_absolute():
        if not path.name:
            msg = "absolute artifact path must name a file"
            raise ReportPathError(msg)
        return path.name
    label = PurePosixPath(path.as_posix())
    if label.is_absolute() or ".." in label.parts or not label.parts:
        msg = "artifact path representation must be relative"
        raise ReportPathError(msg)
    return label.as_posix()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _require_no_parent_traversal(path: Path, role: str) -> None:
    value = str(path)
    if ".." in PurePosixPath(value).parts or ".." in PureWindowsPath(value).parts:
        msg = f"{role} must not contain parent traversal"
        raise ReportPathError(msg)


def _require_nonempty_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        msg = f"{field_name} must be a nonempty string"
        raise ReportInputArtifactError(msg)
