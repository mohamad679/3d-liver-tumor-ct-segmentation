"""Command-line interface for ProtoEM-CT."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from protoem_ct.data.manifest_validation import (
    ManifestValidationError,
    validate_synthetic_manifest,
)
from protoem_ct.data.preprocessing import PreprocessingError, preprocess_synthetic_dataset
from protoem_ct.data.synthetic import (
    SyntheticDataError,
    create_synthetic_dataset,
    load_synthetic_config,
)
from protoem_ct.data.validation import NiftiValidationError, validate_nifti_pair
from protoem_ct.evaluation.dummy_inference import (
    DummyInferenceError,
    run_dummy_inference,
)
from protoem_ct.evaluation.metrics import EvaluationError, evaluate_predictions
from protoem_ct.reporting import ReportGenerationError, generate_synthetic_report
from protoem_ct.tracking import LocalMlflowTrackingError, track_synthetic_run

app = typer.Typer(help="ProtoEM-CT command-line tools.")


@app.callback()
def main() -> None:
    """Run ProtoEM-CT command-line tools."""


@app.command("create-data")
def create_data(
    config_path: Annotated[
        Path,
        typer.Option(
            "--config",
            help="Path to the synthetic data OmegaConf YAML file.",
        ),
    ] = Path("configs/data/synthetic.yaml"),
    output_root: Annotated[
        Path,
        typer.Option(
            "--output-root",
            help="Empty or nonexistent root under which generated data will be written.",
        ),
    ] = ...,  # type: ignore[assignment]
    git_commit: Annotated[
        str,
        typer.Option(
            "--git-commit",
            help="Explicit Git commit recorded in the manifest.",
        ),
    ] = ...,  # type: ignore[assignment]
    created_at_utc: Annotated[
        str,
        typer.Option(
            "--created-at-utc",
            help="Explicit UTC creation timestamp recorded in the manifest.",
        ),
    ] = ...,  # type: ignore[assignment]
) -> None:
    """Create deterministic Phase 1 synthetic NIfTI inputs and manifest."""
    try:
        config = load_synthetic_config(config_path)
        result = create_synthetic_dataset(
            config,
            output_root=output_root,
            git_commit=git_commit,
            created_at_utc=created_at_utc,
        )
    except SyntheticDataError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from None

    typer.echo("create-data success")
    typer.echo(f"case count: {len(result.manifest.case_ids)}")
    typer.echo(f"manifest path: {result.manifest_relative_path.as_posix()}")
    typer.echo(f"config hash: {result.config_hash}")
    typer.echo(f"manifest hash: {result.manifest_hash}")


@app.command("validate-data")
def validate_data(
    manifest_path: Annotated[
        Path,
        typer.Option(
            "--manifest",
            help="Path to the synthetic manifest JSON file.",
        ),
    ] = ...,  # type: ignore[assignment]
    data_root: Annotated[
        Path,
        typer.Option(
            "--data-root",
            help="Root directory beneath which manifest image and label paths are resolved.",
        ),
    ] = ...,  # type: ignore[assignment]
    output_path: Annotated[
        Path,
        typer.Option(
            "--output",
            help="Path to the validation artifact JSON file to create.",
        ),
    ] = ...,  # type: ignore[assignment]
    git_commit: Annotated[
        str,
        typer.Option(
            "--git-commit",
            help="Explicit Git commit recorded in the validation artifact.",
        ),
    ] = ...,  # type: ignore[assignment]
    created_at_utc: Annotated[
        str,
        typer.Option(
            "--created-at-utc",
            help="Explicit UTC creation timestamp recorded in the validation artifact.",
        ),
    ] = ...,  # type: ignore[assignment]
    affine_tolerance: Annotated[
        float,
        typer.Option(
            "--affine-tolerance",
            help="Absolute tolerance for affine and spacing comparisons.",
        ),
    ] = 1e-5,
) -> None:
    """Validate a Phase 1 synthetic manifest and write a validation artifact."""
    try:
        artifact = validate_synthetic_manifest(
            manifest_path,
            data_root=data_root,
            output_path=output_path,
            git_commit=git_commit,
            created_at_utc=created_at_utc,
            affine_tolerance=affine_tolerance,
        )
    except ManifestValidationError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from None

    typer.echo("validation success")
    typer.echo(f"valid case count: {artifact.valid_case_count}")
    typer.echo(f"validation artifact path: {output_path}")
    typer.echo(f"config hash: {artifact.config_hash}")
    typer.echo(f"manifest hash: {artifact.manifest_hash}")


@app.command("preprocess-data")
def preprocess_data(
    manifest_path: Annotated[
        Path,
        typer.Option(
            "--manifest",
            help="Path to the synthetic manifest JSON file.",
        ),
    ] = ...,  # type: ignore[assignment]
    validation_artifact_path: Annotated[
        Path,
        typer.Option(
            "--validation-artifact",
            help="Path to the validation artifact JSON file.",
        ),
    ] = ...,  # type: ignore[assignment]
    data_root: Annotated[
        Path,
        typer.Option(
            "--data-root",
            help="Root directory beneath which manifest image and label paths are resolved.",
        ),
    ] = ...,  # type: ignore[assignment]
    output_root: Annotated[
        Path,
        typer.Option(
            "--output-root",
            help="Empty or nonexistent root where preprocessed outputs will be written.",
        ),
    ] = ...,  # type: ignore[assignment]
    artifact_output_path: Annotated[
        Path,
        typer.Option(
            "--artifact-output",
            help="Path to the preprocessing artifact JSON file to create.",
        ),
    ] = ...,  # type: ignore[assignment]
    config_path: Annotated[
        Path,
        typer.Option(
            "--config",
            help="Path to the Phase 1 OmegaConf YAML file.",
        ),
    ] = ...,  # type: ignore[assignment]
    git_commit: Annotated[
        str,
        typer.Option(
            "--git-commit",
            help="Explicit Git commit recorded in the preprocessing artifact.",
        ),
    ] = ...,  # type: ignore[assignment]
    created_at_utc: Annotated[
        str,
        typer.Option(
            "--created-at-utc",
            help="Explicit UTC creation timestamp recorded in the preprocessing artifact.",
        ),
    ] = ...,  # type: ignore[assignment]
) -> None:
    """Preprocess a validated Phase 1 synthetic manifest."""
    try:
        artifact = preprocess_synthetic_dataset(
            manifest_path,
            validation_artifact_path,
            data_root=data_root,
            output_root=output_root,
            artifact_output_path=artifact_output_path,
            config_path=config_path,
            git_commit=git_commit,
            created_at_utc=created_at_utc,
        )
    except PreprocessingError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from None

    typer.echo("preprocessing success")
    typer.echo(f"processed case count: {len(artifact.output_case_ids)}")
    typer.echo(f"preprocessing artifact path: {artifact_output_path}")
    typer.echo(f"config hash: {artifact.config_hash}")
    typer.echo(f"manifest hash: {artifact.manifest_hash}")


@app.command("infer-dummy")
def infer_dummy(
    preprocess_artifact_path: Annotated[
        Path,
        typer.Option(
            "--preprocess-artifact",
            help="Path to the preprocessing artifact JSON file.",
        ),
    ] = ...,  # type: ignore[assignment]
    preprocessed_root: Annotated[
        Path,
        typer.Option(
            "--preprocessed-root",
            help="Root directory beneath which preprocessed image paths are resolved.",
        ),
    ] = ...,  # type: ignore[assignment]
    output_root: Annotated[
        Path,
        typer.Option(
            "--output-root",
            help="Empty or nonexistent root where dummy predictions will be written.",
        ),
    ] = ...,  # type: ignore[assignment]
    artifact_output_path: Annotated[
        Path,
        typer.Option(
            "--artifact-output",
            help="Path to the inference artifact JSON file to create.",
        ),
    ] = ...,  # type: ignore[assignment]
    config_path: Annotated[
        Path,
        typer.Option(
            "--config",
            help="Path to the Phase 1 OmegaConf YAML file.",
        ),
    ] = ...,  # type: ignore[assignment]
    git_commit: Annotated[
        str,
        typer.Option(
            "--git-commit",
            help="Explicit Git commit recorded in the inference artifact.",
        ),
    ] = ...,  # type: ignore[assignment]
    created_at_utc: Annotated[
        str,
        typer.Option(
            "--created-at-utc",
            help="Explicit UTC creation timestamp recorded in the inference artifact.",
        ),
    ] = ...,  # type: ignore[assignment]
) -> None:
    """Run deterministic Phase 1 dummy inference."""
    try:
        artifact = run_dummy_inference(
            preprocess_artifact_path,
            preprocessed_root=preprocessed_root,
            output_root=output_root,
            artifact_output_path=artifact_output_path,
            config_path=config_path,
            git_commit=git_commit,
            created_at_utc=created_at_utc,
        )
    except DummyInferenceError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from None

    typer.echo("dummy inference success")
    typer.echo(f"prediction count: {len(artifact.prediction_case_ids)}")
    typer.echo(f"inference artifact path: {artifact_output_path}")
    typer.echo(f"config hash: {artifact.config_hash}")
    typer.echo(f"manifest hash: {artifact.manifest_hash}")
    typer.echo(f"method: {artifact.method}")


@app.command("evaluate")
def evaluate(
    preprocess_artifact_path: Annotated[
        Path,
        typer.Option(
            "--preprocess-artifact",
            help="Path to the preprocessing artifact JSON file.",
        ),
    ] = ...,  # type: ignore[assignment]
    inference_artifact_path: Annotated[
        Path,
        typer.Option(
            "--inference-artifact",
            help="Path to the inference artifact JSON file.",
        ),
    ] = ...,  # type: ignore[assignment]
    preprocessed_root: Annotated[
        Path,
        typer.Option(
            "--preprocessed-root",
            help="Root directory beneath which preprocessed label paths are resolved.",
        ),
    ] = ...,  # type: ignore[assignment]
    prediction_root: Annotated[
        Path,
        typer.Option(
            "--prediction-root",
            help="Root directory beneath which prediction paths are resolved.",
        ),
    ] = ...,  # type: ignore[assignment]
    artifact_output_path: Annotated[
        Path,
        typer.Option(
            "--artifact-output",
            help="Path to the evaluation artifact JSON file to create.",
        ),
    ] = ...,  # type: ignore[assignment]
    git_commit: Annotated[
        str,
        typer.Option(
            "--git-commit",
            help="Explicit Git commit recorded in the evaluation artifact.",
        ),
    ] = ...,  # type: ignore[assignment]
    created_at_utc: Annotated[
        str,
        typer.Option(
            "--created-at-utc",
            help="Explicit UTC creation timestamp recorded in the evaluation artifact.",
        ),
    ] = ...,  # type: ignore[assignment]
) -> None:
    """Evaluate deterministic Phase 1 dummy predictions."""
    try:
        artifact = evaluate_predictions(
            preprocess_artifact_path,
            inference_artifact_path,
            preprocessed_root=preprocessed_root,
            prediction_root=prediction_root,
            artifact_output_path=artifact_output_path,
            git_commit=git_commit,
            created_at_utc=created_at_utc,
        )
    except EvaluationError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from None

    typer.echo("evaluation success")
    typer.echo(f"evaluated case count: {artifact.evaluated_case_count}")
    typer.echo(f"evaluation artifact path: {artifact_output_path}")
    typer.echo(f"macro Dice: {artifact.macro_mean_dice}")
    typer.echo(f"macro IoU: {artifact.macro_mean_iou}")
    typer.echo(f"config hash: {artifact.config_hash}")
    typer.echo(f"manifest hash: {artifact.manifest_hash}")
    typer.echo(f"method: {artifact.method}")


@app.command("report")
def report(
    manifest_path: Annotated[
        Path,
        typer.Option(
            "--manifest",
            help="Path to the synthetic manifest JSON file.",
        ),
    ] = ...,  # type: ignore[assignment]
    validation_artifact_path: Annotated[
        Path,
        typer.Option(
            "--validation-artifact",
            help="Path to the validation artifact JSON file.",
        ),
    ] = ...,  # type: ignore[assignment]
    preprocess_artifact_path: Annotated[
        Path,
        typer.Option(
            "--preprocess-artifact",
            help="Path to the preprocessing artifact JSON file.",
        ),
    ] = ...,  # type: ignore[assignment]
    inference_artifact_path: Annotated[
        Path,
        typer.Option(
            "--inference-artifact",
            help="Path to the inference artifact JSON file.",
        ),
    ] = ...,  # type: ignore[assignment]
    evaluation_artifact_path: Annotated[
        Path,
        typer.Option(
            "--evaluation-artifact",
            help="Path to the evaluation artifact JSON file.",
        ),
    ] = ...,  # type: ignore[assignment]
    report_output_path: Annotated[
        Path,
        typer.Option(
            "--report-output",
            help="Path to the Markdown report file to create.",
        ),
    ] = ...,  # type: ignore[assignment]
    artifact_output_path: Annotated[
        Path,
        typer.Option(
            "--artifact-output",
            help="Path to the report artifact JSON file to create.",
        ),
    ] = ...,  # type: ignore[assignment]
    git_commit: Annotated[
        str,
        typer.Option(
            "--git-commit",
            help="Explicit Git commit recorded in the report artifact.",
        ),
    ] = ...,  # type: ignore[assignment]
    created_at_utc: Annotated[
        str,
        typer.Option(
            "--created-at-utc",
            help="Explicit UTC creation timestamp recorded in the report artifact.",
        ),
    ] = ...,  # type: ignore[assignment]
) -> None:
    """Generate the deterministic Phase 1 synthetic report."""
    try:
        artifact = generate_synthetic_report(
            manifest_path,
            validation_artifact_path,
            preprocess_artifact_path,
            inference_artifact_path,
            evaluation_artifact_path,
            report_output_path=report_output_path,
            artifact_output_path=artifact_output_path,
            git_commit=git_commit,
            created_at_utc=created_at_utc,
        )
    except ReportGenerationError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from None

    typer.echo("report generation success")
    typer.echo(f"report path: {report_output_path}")
    typer.echo(f"report artifact path: {artifact_output_path}")
    typer.echo(f"evaluated case count: {artifact.evaluated_case_count}")
    typer.echo(f"macro Dice: {artifact.macro_mean_dice}")
    typer.echo(f"macro IoU: {artifact.macro_mean_iou}")
    typer.echo(f"report SHA-256: {artifact.report_sha256}")
    typer.echo(f"config hash: {artifact.config_hash}")
    typer.echo(f"manifest hash: {artifact.manifest_hash}")
    typer.echo(f"method: {artifact.method}")


@app.command("validate-pair")
def validate_pair(
    image_path: Annotated[Path, typer.Argument(help="Path to the NIfTI image file.")],
    label_path: Annotated[Path, typer.Argument(help="Path to the NIfTI label file.")],
    affine_tolerance: Annotated[
        float,
        typer.Option(
            "--affine-tolerance",
            help="Absolute tolerance for affine matrix comparison.",
        ),
    ] = 1e-5,
) -> None:
    """Validate a NIfTI image and binary label pair."""
    try:
        result = validate_nifti_pair(
            image_path,
            label_path,
            affine_tolerance=affine_tolerance,
        )
    except NiftiValidationError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from None

    typer.echo("validation success")
    typer.echo(f"shape: {result.shape}")
    typer.echo(f"label values: {list(result.label_values)}")


@app.command("track-run")
def track_run(
    manifest_path: Annotated[
        Path,
        typer.Option(
            "--manifest",
            help="Path to the synthetic manifest JSON file.",
        ),
    ] = ...,  # type: ignore[assignment]
    validation_artifact_path: Annotated[
        Path,
        typer.Option(
            "--validation-artifact",
            help="Path to the validation artifact JSON file.",
        ),
    ] = ...,  # type: ignore[assignment]
    preprocess_artifact_path: Annotated[
        Path,
        typer.Option(
            "--preprocess-artifact",
            help="Path to the preprocessing artifact JSON file.",
        ),
    ] = ...,  # type: ignore[assignment]
    inference_artifact_path: Annotated[
        Path,
        typer.Option(
            "--inference-artifact",
            help="Path to the inference artifact JSON file.",
        ),
    ] = ...,  # type: ignore[assignment]
    evaluation_artifact_path: Annotated[
        Path,
        typer.Option(
            "--evaluation-artifact",
            help="Path to the evaluation artifact JSON file.",
        ),
    ] = ...,  # type: ignore[assignment]
    report_path: Annotated[
        Path,
        typer.Option(
            "--report",
            help="Path to the Markdown report file.",
        ),
    ] = ...,  # type: ignore[assignment]
    report_artifact_path: Annotated[
        Path,
        typer.Option(
            "--report-artifact",
            help="Path to the report artifact JSON file.",
        ),
    ] = ...,  # type: ignore[assignment]
    tracking_root: Annotated[
        Path,
        typer.Option(
            "--tracking-root",
            help="Explicit absolute local filesystem root for MLflow tracking.",
        ),
    ] = ...,  # type: ignore[assignment]
    experiment_name: Annotated[
        str,
        typer.Option(
            "--experiment-name",
            help="Explicit MLflow experiment name.",
        ),
    ] = ...,  # type: ignore[assignment]
    run_name: Annotated[
        str,
        typer.Option(
            "--run-name",
            help="Explicit MLflow run name.",
        ),
    ] = ...,  # type: ignore[assignment]
) -> None:
    """Track a Phase 1 synthetic pipeline verification run in local MLflow."""
    try:
        result = track_synthetic_run(
            manifest_path,
            validation_artifact_path,
            preprocess_artifact_path,
            inference_artifact_path,
            evaluation_artifact_path,
            report_path,
            report_artifact_path,
            tracking_root=tracking_root,
            experiment_name=experiment_name,
            run_name=run_name,
        )
    except LocalMlflowTrackingError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from None

    typer.echo("local MLflow tracking success")
    typer.echo(f"experiment name: {result.experiment_name}")
    typer.echo(f"run name: {result.run_name}")
    typer.echo(f"run ID: {result.run_id}")
    typer.echo(f"logged metric count: {len(result.logged_metric_names)}")
    typer.echo(f"logged artifact count: {len(result.logged_artifact_names)}")
    typer.echo(f"config hash: {result.config_hash}")
    typer.echo(f"manifest hash: {result.manifest_hash}")
    typer.echo("synthetic pipeline verification only")
