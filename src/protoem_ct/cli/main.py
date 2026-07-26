"""Command-line interface for ProtoEM-CT."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from protoem_ct.artifacts import Phase2ArtifactError
from protoem_ct.data import (
    SUPPORTED_LITS_SUFFIXES,
    AdapterLayoutSpec,
    AnonymousIdConfig,
    DevelopmentSplitError,
    DevelopmentSplitPolicy,
    GeometryLabelQaConfig,
    GeometryLabelQaError,
    LiTSFilenameConvention,
    ManifestBuilderError,
    Phase2PathError,
    build_development_split,
    build_lits_development_manifest,
    dry_run_lits_inventory,
    read_id_key_file,
    run_geometry_label_qa,
)
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


def _raise_phase2_lits_cli_error(exc: Exception) -> None:
    """Exit with a concise Phase 2 LiTS error that does not reveal local source details."""
    typer.secho(f"Phase 2 LiTS error: {type(exc).__name__}", err=True, fg=typer.colors.RED)
    raise typer.Exit(code=1) from None


def _raise_phase2_split_cli_error(exc: Exception) -> None:
    """Exit with a concise Phase 2 split error that does not reveal manifest details."""
    typer.secho(f"Phase 2 split error: {type(exc).__name__}", err=True, fg=typer.colors.RED)
    raise typer.Exit(code=1) from None


def _raise_phase2_geometry_qa_cli_error(exc: Exception) -> None:
    """Exit with a concise Phase 2 QA error that does not reveal source details."""
    typer.secho(
        f"Phase 2 geometry-label QA error: {type(exc).__name__}",
        err=True,
        fg=typer.colors.RED,
    )
    raise typer.Exit(code=1) from None


@app.callback()
def main() -> None:
    """Run ProtoEM-CT command-line tools."""


@app.command("inventory-lits")
def inventory_lits(
    dataset_root: Annotated[
        Path,
        typer.Option(
            "--dataset-root",
            help="Explicit absolute LiTS development-cohort dataset root.",
        ),
    ] = ...,  # type: ignore[assignment]
    image_pattern: Annotated[
        str,
        typer.Option(
            "--image-pattern",
            help="Conservative relative POSIX glob for image files.",
        ),
    ] = ...,  # type: ignore[assignment]
    label_pattern: Annotated[
        str,
        typer.Option(
            "--label-pattern",
            help="Conservative relative POSIX glob for label files.",
        ),
    ] = ...,  # type: ignore[assignment]
    recursive: Annotated[
        bool,
        typer.Option(
            "--recursive/--no-recursive",
            help="Whether to traverse recursively beneath the explicit root.",
        ),
    ] = False,
    image_prefix: Annotated[
        str,
        typer.Option(
            "--image-prefix",
            help="Explicit LiTS-style image filename prefix.",
        ),
    ] = ...,  # type: ignore[assignment]
    label_prefix: Annotated[
        str,
        typer.Option(
            "--label-prefix",
            help="Explicit LiTS-style label filename prefix.",
        ),
    ] = ...,  # type: ignore[assignment]
) -> None:
    """Dry-run LiTS-style development-cohort discovery without opening or hashing files."""
    try:
        result = dry_run_lits_inventory(
            dataset_root,
            layout=AdapterLayoutSpec(
                image_pattern=image_pattern,
                label_pattern=label_pattern,
                recursive=recursive,
            ),
            convention=LiTSFilenameConvention(
                image_prefix=image_prefix,
                label_prefix=label_prefix,
                allowed_suffixes=SUPPORTED_LITS_SUFFIXES,
            ),
        )
    except (ManifestBuilderError, Phase2PathError) as exc:
        _raise_phase2_lits_cli_error(exc)

    typer.echo("inventory success")
    typer.echo(f"adapter: {result.adapter_name}@{result.adapter_version}")
    typer.echo(f"case count: {result.case_count}")
    typer.echo(f"image count: {result.image_count}")
    typer.echo(f"label count: {result.label_count}")
    typer.echo("no files were opened or hashed")


@app.command("build-lits-manifest")
def build_lits_manifest(
    dataset_root: Annotated[
        Path,
        typer.Option(
            "--dataset-root",
            help="Explicit absolute LiTS development-cohort dataset root.",
        ),
    ] = ...,  # type: ignore[assignment]
    image_pattern: Annotated[
        str,
        typer.Option(
            "--image-pattern",
            help="Conservative relative POSIX glob for image files.",
        ),
    ] = ...,  # type: ignore[assignment]
    label_pattern: Annotated[
        str,
        typer.Option(
            "--label-pattern",
            help="Conservative relative POSIX glob for label files.",
        ),
    ] = ...,  # type: ignore[assignment]
    recursive: Annotated[
        bool,
        typer.Option(
            "--recursive/--no-recursive",
            help="Whether to traverse recursively beneath the explicit root.",
        ),
    ] = False,
    image_prefix: Annotated[
        str,
        typer.Option(
            "--image-prefix",
            help="Explicit LiTS-style image filename prefix.",
        ),
    ] = ...,  # type: ignore[assignment]
    label_prefix: Annotated[
        str,
        typer.Option(
            "--label-prefix",
            help="Explicit LiTS-style label filename prefix.",
        ),
    ] = ...,  # type: ignore[assignment]
    dataset_id: Annotated[
        str,
        typer.Option(
            "--dataset-id",
            help="Explicit development dataset ID to store in the manifest.",
        ),
    ] = ...,  # type: ignore[assignment]
    project_namespace: Annotated[
        str,
        typer.Option(
            "--project-namespace",
            help="Nonsecret namespace for deterministic anonymous IDs.",
        ),
    ] = ...,  # type: ignore[assignment]
    patient_id_prefix: Annotated[
        str,
        typer.Option(
            "--patient-id-prefix",
            help="Safe anonymous patient ID prefix.",
        ),
    ] = ...,  # type: ignore[assignment]
    case_id_prefix: Annotated[
        str,
        typer.Option(
            "--case-id-prefix",
            help="Safe anonymous case ID prefix.",
        ),
    ] = ...,  # type: ignore[assignment]
    digest_length: Annotated[
        int,
        typer.Option(
            "--digest-length",
            help="Number of lowercase HMAC-SHA256 hex characters to retain.",
        ),
    ] = ...,  # type: ignore[assignment]
    id_key_file: Annotated[
        Path,
        typer.Option(
            "--id-key-file",
            help="Explicit absolute regular file containing HMAC key material.",
        ),
    ] = ...,  # type: ignore[assignment]
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            help="Explicit absolute manifest JSON output path outside the dataset root.",
        ),
    ] = ...,  # type: ignore[assignment]
    git_commit: Annotated[
        str,
        typer.Option(
            "--git-commit",
            help="Explicit Git commit recorded in the manifest.",
        ),
    ] = ...,  # type: ignore[assignment]
    generated_at_utc: Annotated[
        str,
        typer.Option(
            "--generated-at-utc",
            help="Explicit UTC generation timestamp recorded in the manifest.",
        ),
    ] = ...,  # type: ignore[assignment]
) -> None:
    """Build an anonymous LiTS development manifest from explicit inputs."""
    try:
        id_key = read_id_key_file(id_key_file)
        result = build_lits_development_manifest(
            dataset_root,
            layout=AdapterLayoutSpec(
                image_pattern=image_pattern,
                label_pattern=label_pattern,
                recursive=recursive,
            ),
            convention=LiTSFilenameConvention(
                image_prefix=image_prefix,
                label_prefix=label_prefix,
                allowed_suffixes=SUPPORTED_LITS_SUFFIXES,
            ),
            dataset_id=dataset_id,
            generated_at_utc=generated_at_utc,
            git_commit=git_commit,
            anonymous_id_config=AnonymousIdConfig(
                project_namespace=project_namespace,
                patient_id_prefix=patient_id_prefix,
                case_id_prefix=case_id_prefix,
                digest_length=digest_length,
            ),
            id_key=id_key,
            output_path=output,
        )
    except (ManifestBuilderError, Phase2ArtifactError, Phase2PathError) as exc:
        _raise_phase2_lits_cli_error(exc)

    typer.echo("manifest generation success")
    typer.echo(f"dataset ID: {result.dataset_id}")
    typer.echo(f"case count: {result.case_count}")
    typer.echo(f"manifest output path: {output}")
    typer.echo(f"dataset-root fingerprint: {result.dataset_root_fingerprint}")
    typer.echo(f"manifest hash: {result.manifest_hash}")
    typer.echo("LiTS development cohort")
    typer.echo("LiTS and MSD Task03 Liver are not independent cohorts")


@app.command("build-development-split")
def build_development_split_cli(
    manifest_path: Annotated[
        Path,
        typer.Option(
            "--manifest",
            help="Explicit absolute anonymous Phase 2 dataset manifest JSON path.",
        ),
    ] = ...,  # type: ignore[assignment]
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            help="Explicit absolute development split JSON output path.",
        ),
    ] = ...,  # type: ignore[assignment]
    policy_version: Annotated[
        str,
        typer.Option(
            "--policy-version",
            help="Explicit split policy version.",
        ),
    ] = ...,  # type: ignore[assignment]
    split_seed: Annotated[
        int,
        typer.Option(
            "--split-seed",
            help="Explicit nonnegative split seed.",
        ),
    ] = ...,  # type: ignore[assignment]
    train_patient_count: Annotated[
        int,
        typer.Option(
            "--train-patient-count",
            help="Explicit train patient count.",
        ),
    ] = ...,  # type: ignore[assignment]
    validation_patient_count: Annotated[
        int,
        typer.Option(
            "--validation-patient-count",
            help="Explicit validation patient count.",
        ),
    ] = ...,  # type: ignore[assignment]
    internal_test_patient_count: Annotated[
        int,
        typer.Option(
            "--internal-test-patient-count",
            help="Explicit immutable internal-test patient count.",
        ),
    ] = ...,  # type: ignore[assignment]
    git_commit: Annotated[
        str,
        typer.Option(
            "--git-commit",
            help="Explicit Git commit recorded in the split artifact.",
        ),
    ] = ...,  # type: ignore[assignment]
    generated_at_utc: Annotated[
        str,
        typer.Option(
            "--generated-at-utc",
            help="Explicit UTC generation timestamp recorded in the split artifact.",
        ),
    ] = ...,  # type: ignore[assignment]
) -> None:
    """Build a deterministic patient-level development split from an anonymous manifest."""
    try:
        result = build_development_split(
            manifest_path,
            policy=DevelopmentSplitPolicy(
                policy_version=policy_version,
                split_seed=split_seed,
                train_patient_count=train_patient_count,
                validation_patient_count=validation_patient_count,
                internal_test_patient_count=internal_test_patient_count,
            ),
            output_path=output,
            git_commit=git_commit,
            generated_at_utc=generated_at_utc,
        )
    except (DevelopmentSplitError, Phase2ArtifactError) as exc:
        _raise_phase2_split_cli_error(exc)

    typer.echo("split generation success")
    typer.echo(f"policy version: {result.policy_version}")
    typer.echo(f"split seed: {result.split_seed}")
    typer.echo(f"train patient count: {result.train_patient_count}")
    typer.echo(f"validation patient count: {result.validation_patient_count}")
    typer.echo(f"internal-test patient count: {result.internal_test_patient_count}")
    typer.echo(f"total case count: {result.total_case_count}")
    typer.echo(f"source manifest hash: {result.source_manifest_hash}")
    typer.echo(f"split hash: {result.split_hash}")
    typer.echo(f"output path: {output}")


@app.command("qa-geometry-labels")
def qa_geometry_labels(
    manifest_path: Annotated[
        Path,
        typer.Option(
            "--manifest",
            help="Explicit absolute anonymous Phase 2 dataset manifest JSON path.",
        ),
    ] = ...,  # type: ignore[assignment]
    split_path: Annotated[
        Path,
        typer.Option(
            "--split",
            help="Explicit absolute patient-level development split JSON path.",
        ),
    ] = ...,  # type: ignore[assignment]
    dataset_root: Annotated[
        Path,
        typer.Option(
            "--dataset-root",
            help="Explicit absolute dataset root containing manifest-relative source files.",
        ),
    ] = ...,  # type: ignore[assignment]
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            help="Explicit absolute geometry-label QA JSON output path.",
        ),
    ] = ...,  # type: ignore[assignment]
    allowed_label_values: Annotated[
        list[int],
        typer.Option(
            "--allowed-label-value",
            help="Allowed integer label value; repeat this option for every allowed value.",
        ),
    ] = ...,  # type: ignore[assignment]
    tumor_label_value: Annotated[
        int,
        typer.Option(
            "--tumor-label-value",
            help="Explicit integer tumor label value.",
        ),
    ] = ...,  # type: ignore[assignment]
    affine_tolerance: Annotated[
        float,
        typer.Option(
            "--affine-tolerance",
            help="Explicit positive absolute tolerance for affine and spacing comparisons.",
        ),
    ] = ...,  # type: ignore[assignment]
    git_commit: Annotated[
        str,
        typer.Option(
            "--git-commit",
            help="Explicit Git commit recorded in the QA artifact.",
        ),
    ] = ...,  # type: ignore[assignment]
    created_at_utc: Annotated[
        str,
        typer.Option(
            "--created-at-utc",
            help="Explicit UTC creation timestamp recorded in the QA artifact.",
        ),
    ] = ...,  # type: ignore[assignment]
) -> None:
    """Run Phase 2 geometry and label QA from anonymous manifest and split artifacts."""
    try:
        artifact = run_geometry_label_qa(
            manifest_path,
            split_path,
            dataset_root=dataset_root,
            output_path=output,
            config=GeometryLabelQaConfig(
                allowed_label_values=tuple(allowed_label_values),
                tumor_label_value=tumor_label_value,
                affine_tolerance=affine_tolerance,
            ),
            git_commit=git_commit,
            created_at_utc=created_at_utc,
        )
    except (GeometryLabelQaError, Phase2ArtifactError, Phase2PathError) as exc:
        _raise_phase2_geometry_qa_cli_error(exc)

    typer.echo("geometry and label QA completed")
    typer.echo(f"case count: {artifact.case_count}")
    typer.echo(f"passed case count: {artifact.passed_case_count}")
    typer.echo(f"failed case count: {artifact.failed_case_count}")
    typer.echo(f"config hash: {artifact.config_hash}")
    typer.echo(f"manifest hash: {artifact.manifest_hash}")
    typer.echo(f"split hash: {artifact.split_hash}")
    typer.echo(f"QA artifact hash: {artifact.qa_artifact_hash}")
    typer.echo(f"output path: {output}")


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
