"""Command-line interface for ProtoEM-CT."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from protoem_ct.artifacts import Phase2ArtifactError
from protoem_ct.baselines import (
    BASELINE_PREDICTION_MANIFEST_VERSION,
    BASELINE_SYNTHETIC_DATASET_NAME,
    BASELINE_SYNTHETIC_FIXTURE_VERSION,
    BASELINE_SYNTHETIC_TEST_CASE_IDENTIFIERS,
    BASELINE_SYNTHETIC_TRAINING_CASE_IDENTIFIERS,
    NNUNET_V2_RUN_CONFIG_VERSION,
    BaselineEvaluationError,
    BaselineSyntheticFixtureError,
    NnUNetWrapperError,
    build_nnunet_v2_plan_and_preprocess_command,
    build_nnunet_v2_predict_command,
    build_nnunet_v2_runtime_environment,
    build_nnunet_v2_train_command,
    generate_baseline_synthetic_fixture,
    import_saved_baseline_predictions,
    load_reference_label_mapping_artifact,
    nnunet_v2_run_config_from_json,
    validate_baseline_run_root,
)
from protoem_ct.data import (
    SUPPORTED_LITS_SUFFIXES,
    AdapterLayoutSpec,
    AnonymousIdConfig,
    DevelopmentLeakageAuditConfig,
    DevelopmentLeakageAuditError,
    DevelopmentQaReportConfig,
    DevelopmentQaReportError,
    DevelopmentSplitError,
    DevelopmentSplitPolicy,
    DevelopmentSummaryConfig,
    DevelopmentSummaryError,
    GeometryLabelQaConfig,
    GeometryLabelQaError,
    LesionComponentsConfig,
    LesionComponentsError,
    LiTSFilenameConvention,
    ManifestBuilderError,
    Phase2PathError,
    assemble_development_qa_report,
    build_development_split,
    build_lits_development_manifest,
    dry_run_lits_inventory,
    read_id_key_file,
    run_development_data_summary,
    run_development_leakage_audit,
    run_geometry_label_qa,
    run_lesion_component_analysis,
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


def _raise_phase2_lesion_cli_error(exc: Exception) -> None:
    """Exit with a concise Phase 2 lesion error that does not reveal source details."""
    typer.secho(
        f"Phase 2 lesion-components error: {type(exc).__name__}",
        err=True,
        fg=typer.colors.RED,
    )
    raise typer.Exit(code=1) from None


def _raise_phase2_development_summary_cli_error(exc: Exception) -> None:
    """Exit with a concise Phase 2 summary error that does not reveal source details."""
    typer.secho(
        f"Phase 2 development-summary error: {type(exc).__name__}",
        err=True,
        fg=typer.colors.RED,
    )
    raise typer.Exit(code=1) from None


def _raise_phase2_development_qa_report_cli_error(exc: Exception) -> None:
    """Exit with a concise Phase 2 final QA report error without artifact-content leakage."""
    typer.secho(
        f"Phase 2 development-QA report error: {type(exc).__name__}",
        err=True,
        fg=typer.colors.RED,
    )
    raise typer.Exit(code=1) from None


def _raise_phase2_development_leakage_audit_cli_error(exc: Exception) -> None:
    """Exit with a concise Phase 2 leakage-audit error without artifact-content leakage."""
    typer.secho(
        f"Phase 2 development leakage-audit error: {type(exc).__name__}",
        err=True,
        fg=typer.colors.RED,
    )
    raise typer.Exit(code=1) from None


def _raise_baseline_synthetic_fixture_cli_error(exc: Exception) -> None:
    """Exit with a concise Phase 3 synthetic-fixture error without path leakage."""
    typer.secho(
        f"Phase 3 baseline synthetic fixture error: {type(exc).__name__}",
        err=True,
        fg=typer.colors.RED,
    )
    raise typer.Exit(code=1) from None


def _raise_baseline_prediction_import_cli_error(exc: Exception) -> None:
    """Exit with a concise Phase 3 prediction-import error without path leakage."""
    typer.secho(
        f"Phase 3 baseline prediction import error: {type(exc).__name__}",
        err=True,
        fg=typer.colors.RED,
    )
    raise typer.Exit(code=1) from None


def _raise_nnunet_wrapper_cli_error(exc: Exception) -> None:
    """Exit with a concise Phase 3 nnU-Net wrapper error without path leakage."""
    typer.secho(
        f"Phase 3 nnU-Net wrapper error: {type(exc).__name__}",
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


@app.command("summarize-lesions")
def summarize_lesions(
    manifest_path: Annotated[
        Path,
        typer.Option(
            "--manifest",
            help="Explicit absolute anonymous Phase 2 dataset manifest JSON path.",
        ),
    ] = ...,  # type: ignore[assignment]
    geometry_qa_artifact_path: Annotated[
        Path,
        typer.Option(
            "--geometry-qa-artifact",
            help="Explicit absolute geometry-label QA artifact JSON path.",
        ),
    ] = ...,  # type: ignore[assignment]
    dataset_root: Annotated[
        Path,
        typer.Option(
            "--dataset-root",
            help="Explicit absolute dataset root containing manifest-relative label files.",
        ),
    ] = ...,  # type: ignore[assignment]
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            help="Explicit absolute lesion-components JSON output path.",
        ),
    ] = ...,  # type: ignore[assignment]
    connectivity: Annotated[
        int,
        typer.Option(
            "--connectivity",
            help="Explicit 3D connectivity: 6, 18, or 26.",
        ),
    ] = ...,  # type: ignore[assignment]
    tumor_label_value: Annotated[
        int,
        typer.Option(
            "--tumor-label-value",
            help="Explicit integer tumor label value.",
        ),
    ] = ...,  # type: ignore[assignment]
    git_commit: Annotated[
        str,
        typer.Option(
            "--git-commit",
            help="Explicit Git commit recorded in the lesion artifact.",
        ),
    ] = ...,  # type: ignore[assignment]
    created_at_utc: Annotated[
        str,
        typer.Option(
            "--created-at-utc",
            help="Explicit UTC creation timestamp recorded in the lesion artifact.",
        ),
    ] = ...,  # type: ignore[assignment]
) -> None:
    """Summarize deterministic 3D lesion components from geometry-QA-passing labels."""
    try:
        artifact = run_lesion_component_analysis(
            manifest_path,
            geometry_qa_artifact_path,
            dataset_root=dataset_root,
            output_path=output,
            config=LesionComponentsConfig(
                connectivity=connectivity,
                tumor_label_value=tumor_label_value,
            ),
            git_commit=git_commit,
            created_at_utc=created_at_utc,
        )
    except (LesionComponentsError, Phase2ArtifactError, Phase2PathError) as exc:
        _raise_phase2_lesion_cli_error(exc)

    total_lesion_count = sum(
        record.lesion_count or 0 for record in artifact.case_records if record.analysis_performed
    )
    typer.echo("lesion summary completed")
    typer.echo(f"case count: {artifact.case_count}")
    typer.echo(f"analyzed case count: {artifact.analyzed_case_count}")
    typer.echo(f"skipped case count: {artifact.skipped_case_count}")
    typer.echo(f"total lesion count: {total_lesion_count}")
    typer.echo(f"connectivity: {artifact.connectivity}")
    typer.echo(f"config hash: {artifact.config_hash}")
    typer.echo(f"manifest hash: {artifact.manifest_hash}")
    typer.echo(f"split hash: {artifact.split_hash}")
    typer.echo(f"lesion artifact hash: {artifact.lesion_artifact_hash}")
    typer.echo(f"output path: {output}")


@app.command("summarize-development-data")
def summarize_development_data(
    manifest_path: Annotated[
        Path,
        typer.Option(
            "--manifest",
            help="Explicit absolute anonymous Phase 2 dataset manifest JSON path.",
        ),
    ] = ...,  # type: ignore[assignment]
    geometry_qa_artifact_path: Annotated[
        Path,
        typer.Option(
            "--geometry-qa-artifact",
            help="Explicit absolute geometry-label QA artifact JSON path.",
        ),
    ] = ...,  # type: ignore[assignment]
    lesion_artifact_path: Annotated[
        Path,
        typer.Option(
            "--lesion-artifact",
            help="Explicit absolute lesion-components artifact JSON path.",
        ),
    ] = ...,  # type: ignore[assignment]
    dataset_root: Annotated[
        Path,
        typer.Option(
            "--dataset-root",
            help="Explicit absolute dataset root containing manifest-relative image files.",
        ),
    ] = ...,  # type: ignore[assignment]
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            help="Explicit absolute development-summary JSON output path.",
        ),
    ] = ...,  # type: ignore[assignment]
    histogram_min: Annotated[
        float,
        typer.Option(
            "--histogram-min",
            help="Explicit fixed histogram lower edge.",
        ),
    ] = ...,  # type: ignore[assignment]
    histogram_max: Annotated[
        float,
        typer.Option(
            "--histogram-max",
            help="Explicit fixed histogram upper edge.",
        ),
    ] = ...,  # type: ignore[assignment]
    histogram_bin_count: Annotated[
        int,
        typer.Option(
            "--histogram-bin-count",
            help="Explicit fixed histogram bin count.",
        ),
    ] = ...,  # type: ignore[assignment]
    git_commit: Annotated[
        str,
        typer.Option(
            "--git-commit",
            help="Explicit Git commit recorded in the summary artifact.",
        ),
    ] = ...,  # type: ignore[assignment]
    created_at_utc: Annotated[
        str,
        typer.Option(
            "--created-at-utc",
            help="Explicit UTC creation timestamp recorded in the summary artifact.",
        ),
    ] = ...,  # type: ignore[assignment]
) -> None:
    """Summarize fixed CT histograms and deterministic development data statistics."""
    try:
        artifact = run_development_data_summary(
            manifest_path,
            geometry_qa_artifact_path,
            lesion_artifact_path,
            dataset_root=dataset_root,
            output_path=output,
            config=DevelopmentSummaryConfig(
                histogram_min=histogram_min,
                histogram_max=histogram_max,
                histogram_bin_count=histogram_bin_count,
            ),
            git_commit=git_commit,
            created_at_utc=created_at_utc,
        )
    except (DevelopmentSummaryError, Phase2ArtifactError, Phase2PathError) as exc:
        _raise_phase2_development_summary_cli_error(exc)

    typer.echo("development data summary completed")
    typer.echo(f"case count: {artifact.case_count}")
    typer.echo(f"analyzed case count: {artifact.analyzed_case_count}")
    typer.echo(f"skipped case count: {artifact.skipped_case_count}")
    typer.echo(f"aggregate image voxel count: {artifact.aggregate_image_voxel_count}")
    typer.echo(f"total lesion count: {artifact.total_lesion_count}")
    typer.echo(f"histogram bin count: {artifact.histogram_bin_count}")
    typer.echo(f"config hash: {artifact.config_hash}")
    typer.echo(f"manifest hash: {artifact.manifest_hash}")
    typer.echo(f"split hash: {artifact.split_hash}")
    typer.echo(f"summary artifact hash: {artifact.summary_artifact_hash}")
    typer.echo(f"output path: {output}")


@app.command("build-development-qa-report")
def build_development_qa_report(
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
            help="Explicit absolute Phase 2 development split JSON path.",
        ),
    ] = ...,  # type: ignore[assignment]
    geometry_qa_artifact_path: Annotated[
        Path,
        typer.Option(
            "--geometry-qa-artifact",
            help="Explicit absolute geometry-label QA artifact JSON path.",
        ),
    ] = ...,  # type: ignore[assignment]
    lesion_artifact_path: Annotated[
        Path,
        typer.Option(
            "--lesion-artifact",
            help="Explicit absolute lesion-components artifact JSON path.",
        ),
    ] = ...,  # type: ignore[assignment]
    development_summary_artifact_path: Annotated[
        Path,
        typer.Option(
            "--development-summary-artifact",
            help="Explicit absolute development-summary artifact JSON path.",
        ),
    ] = ...,  # type: ignore[assignment]
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            help="Explicit absolute final development QA JSON report output path.",
        ),
    ] = ...,  # type: ignore[assignment]
    report_contract_version: Annotated[
        str,
        typer.Option(
            "--report-contract-version",
            help="Explicit final QA JSON report contract version.",
        ),
    ] = ...,  # type: ignore[assignment]
    git_commit: Annotated[
        str,
        typer.Option(
            "--git-commit",
            help="Explicit Git commit recorded in the final QA artifact.",
        ),
    ] = ...,  # type: ignore[assignment]
    created_at_utc: Annotated[
        str,
        typer.Option(
            "--created-at-utc",
            help="Explicit UTC creation timestamp recorded in the final QA artifact.",
        ),
    ] = ...,  # type: ignore[assignment]
) -> None:
    """Assemble the final Phase 2 DevelopmentQaArtifact JSON-only report."""
    try:
        artifact = assemble_development_qa_report(
            manifest_path,
            split_path,
            geometry_qa_artifact_path,
            lesion_artifact_path,
            development_summary_artifact_path,
            output_path=output,
            config=DevelopmentQaReportConfig(
                report_contract_version=report_contract_version,
            ),
            git_commit=git_commit,
            created_at_utc=created_at_utc,
        )
    except (DevelopmentQaReportError, Phase2ArtifactError) as exc:
        _raise_phase2_development_qa_report_cli_error(exc)

    typer.echo("development QA JSON report completed")
    typer.echo(f"case count: {artifact.case_count}")
    typer.echo(f"passed case count: {artifact.passed_case_count}")
    typer.echo(f"failed case count: {artifact.failed_case_count}")
    typer.echo(f"report config hash: {artifact.config_hash}")
    typer.echo(f"manifest hash: {artifact.manifest_hash}")
    typer.echo(f"split hash: {artifact.split_hash}")
    typer.echo(f"geometry QA artifact hash: {artifact.geometry_qa_artifact_hash}")
    typer.echo(f"lesion artifact hash: {artifact.lesion_artifact_hash}")
    typer.echo(f"development summary artifact hash: {artifact.development_summary_artifact_hash}")
    typer.echo(f"final QA artifact hash: {artifact.qa_artifact_hash}")
    typer.echo(f"output path: {output}")


@app.command("audit-development-leakage")
def audit_development_leakage(
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
            help="Explicit absolute Phase 2 development split JSON path.",
        ),
    ] = ...,  # type: ignore[assignment]
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            help="Explicit absolute development leakage-audit JSON output path.",
        ),
    ] = ...,  # type: ignore[assignment]
    audit_contract_version: Annotated[
        str,
        typer.Option(
            "--audit-contract-version",
            help="Explicit development leakage-audit contract version.",
        ),
    ] = ...,  # type: ignore[assignment]
    git_commit: Annotated[
        str,
        typer.Option(
            "--git-commit",
            help="Explicit Git commit recorded in the leakage-audit artifact.",
        ),
    ] = ...,  # type: ignore[assignment]
    created_at_utc: Annotated[
        str,
        typer.Option(
            "--created-at-utc",
            help="Explicit UTC creation timestamp recorded in the leakage-audit artifact.",
        ),
    ] = ...,  # type: ignore[assignment]
) -> None:
    """Generate a deterministic Phase 2 development leakage-audit JSON artifact."""
    try:
        artifact = run_development_leakage_audit(
            manifest_path,
            split_path,
            output_path=output,
            config=DevelopmentLeakageAuditConfig(
                audit_contract_version=audit_contract_version,
            ),
            git_commit=git_commit,
            created_at_utc=created_at_utc,
        )
    except (DevelopmentLeakageAuditError, Phase2ArtifactError) as exc:
        _raise_phase2_development_leakage_audit_cli_error(exc)

    patient_overlap_total = sum(artifact.pairwise_patient_overlap_counts.values())
    case_overlap_total = sum(artifact.pairwise_case_overlap_counts.values())
    typer.echo("development leakage audit completed")
    typer.echo(f"audit passed: {str(artifact.audit_passed).lower()}")
    typer.echo(f"manifest case count: {artifact.manifest_case_count}")
    typer.echo(f"manifest patient count: {artifact.manifest_patient_count}")
    typer.echo(f"train patient count: {artifact.patient_counts_by_partition['train']}")
    typer.echo(f"validation patient count: {artifact.patient_counts_by_partition['validation']}")
    typer.echo(
        f"internal-test patient count: {artifact.patient_counts_by_partition['internal_test']}"
    )
    typer.echo(f"total finding count: {artifact.critical_finding_count}")
    typer.echo(f"patient-overlap total: {patient_overlap_total}")
    typer.echo(f"case-overlap total: {case_overlap_total}")
    typer.echo(
        "image-hash cross-partition overlap count: "
        f"{artifact.image_hash_cross_partition_overlap_count}"
    )
    typer.echo(
        "label-hash cross-partition overlap count: "
        f"{artifact.label_hash_cross_partition_overlap_count}"
    )
    typer.echo(
        "image-label-pair cross-partition overlap count: "
        f"{artifact.image_label_pair_cross_partition_overlap_count}"
    )
    typer.echo(f"audit config hash: {artifact.config_hash}")
    typer.echo(f"manifest hash: {artifact.manifest_hash}")
    typer.echo(f"split hash: {artifact.split_hash}")
    typer.echo(f"leakage-audit hash: {artifact.audit_hash}")
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


@app.command("generate-baseline-synthetic-fixture")
def generate_baseline_synthetic_fixture_command(
    output_root: Annotated[
        Path,
        typer.Option(
            "--output-root",
            help=(
                "Absolute non-existing external output root for the fixed "
                "baseline synthetic fixture."
            ),
        ),
    ] = ...,  # type: ignore[assignment]
) -> None:
    """Generate the fixed Phase 3 synthetic fixture outside the repository."""
    try:
        result = generate_baseline_synthetic_fixture(output_root)
    except BaselineSyntheticFixtureError as exc:
        _raise_baseline_synthetic_fixture_cli_error(exc)

    typer.echo("baseline synthetic fixture generation success")
    typer.echo(f"contract version: {BASELINE_SYNTHETIC_FIXTURE_VERSION}")
    typer.echo(f"dataset name: {BASELINE_SYNTHETIC_DATASET_NAME}")
    typer.echo(f"training case count: {len(BASELINE_SYNTHETIC_TRAINING_CASE_IDENTIFIERS)}")
    typer.echo(f"test case count: {len(BASELINE_SYNTHETIC_TEST_CASE_IDENTIFIERS)}")
    typer.echo(f"total generated file count: {result.total_generated_file_count}")
    typer.echo(f"artifact hash: {result.artifact.artifact_hash}")


@app.command("inspect-nnunet-v2-baseline-plan")
def inspect_nnunet_v2_baseline_plan(
    run_config_path: Annotated[
        Path,
        typer.Option(
            "--run-config",
            help="Absolute path to the serialized nnU-Net v2 run-config JSON artifact.",
        ),
    ] = ...,  # type: ignore[assignment]
    runtime_raw_root: Annotated[
        Path,
        typer.Option(
            "--runtime-raw-root",
            help=(
                "Absolute external nnU-Net raw root that contains the synthetic dataset directory."
            ),
        ),
    ] = ...,  # type: ignore[assignment]
    runtime_run_root: Annotated[
        Path,
        typer.Option(
            "--runtime-run-root",
            help="Absolute external nnU-Net baseline run root to validate without creating.",
        ),
    ] = ...,  # type: ignore[assignment]
) -> None:
    """Validate nnU-Net baseline planning inputs and print aggregate command metadata only."""

    try:
        run_config = nnunet_v2_run_config_from_json(run_config_path.read_bytes())
        run_paths = validate_baseline_run_root(
            runtime_run_root,
            baseline_family="nnunet_v2",
        )
        _runtime_environment = build_nnunet_v2_runtime_environment(
            raw_root=runtime_raw_root,
            run_paths=run_paths,
        )
        plan_command = build_nnunet_v2_plan_and_preprocess_command(run_config)
        train_command = build_nnunet_v2_train_command(run_config)
        predict_command = build_nnunet_v2_predict_command(
            run_config,
            input_images_directory=runtime_raw_root / BASELINE_SYNTHETIC_DATASET_NAME / "imagesTs",
            output_predictions_directory=run_paths.predictions_dir / "nnunet_predictions",
        )
    except (OSError, NnUNetWrapperError) as exc:
        _raise_nnunet_wrapper_cli_error(exc)

    typer.echo("nnU-Net v2 baseline plan inspection success")
    typer.echo(f"contract version: {NNUNET_V2_RUN_CONFIG_VERSION}")
    typer.echo(f"dataset ID: {run_config.dataset_id}")
    typer.echo(f"configuration: {run_config.configuration}")
    typer.echo(f"fold: {run_config.fold}")
    typer.echo(f"trainer class: {run_config.trainer_class}")
    typer.echo(f"plans identifier: {run_config.plans_identifier}")
    typer.echo(f"planning executable: {plan_command[0]}")
    typer.echo(f"training executable: {train_command[0]}")
    typer.echo(f"prediction executable: {predict_command[0]}")
    typer.echo("command count: 3")


@app.command("import-baseline-predictions")
def import_baseline_predictions_command(
    baseline_family: Annotated[
        str,
        typer.Option(
            "--baseline-family",
            help="Baseline family name: nnunet_v2 or monai_segresnet.",
        ),
    ] = ...,  # type: ignore[assignment]
    run_identifier: Annotated[
        str,
        typer.Option(
            "--run-identifier",
            help="Conservative anonymous run identifier.",
        ),
    ] = ...,  # type: ignore[assignment]
    prediction_directory: Annotated[
        Path,
        typer.Option(
            "--prediction-directory",
            help="Absolute external directory containing one <case>.nii.gz prediction per case.",
        ),
    ] = ...,  # type: ignore[assignment]
    reference_mapping_artifact: Annotated[
        Path,
        typer.Option(
            "--reference-mapping-artifact",
            help=(
                "Absolute external JSON artifact that maps anonymous case "
                "identifiers to label paths."
            ),
        ),
    ] = ...,  # type: ignore[assignment]
    dataset_manifest_sha256: Annotated[
        str,
        typer.Option(
            "--dataset-manifest-sha256",
            help="Dataset-manifest SHA-256 linked to the imported predictions.",
        ),
    ] = ...,  # type: ignore[assignment]
    development_split_sha256: Annotated[
        str,
        typer.Option(
            "--development-split-sha256",
            help="Development-split SHA-256 linked to the imported predictions.",
        ),
    ] = ...,  # type: ignore[assignment]
    checkpoint_sha256: Annotated[
        str,
        typer.Option(
            "--checkpoint-sha256",
            help="Checkpoint SHA-256 linked to the imported predictions.",
        ),
    ] = ...,  # type: ignore[assignment]
    metric_config_sha256: Annotated[
        str,
        typer.Option(
            "--metric-config-sha256",
            help="Metric-config SHA-256 for the saved-prediction evaluation.",
        ),
    ] = ...,  # type: ignore[assignment]
    nsd_tolerance_mm: Annotated[
        float,
        typer.Option(
            "--nsd-tolerance-mm",
            help="Explicit normalized-surface-Dice tolerance in millimetres.",
        ),
    ] = ...,  # type: ignore[assignment]
    affine_tolerance_mm: Annotated[
        float,
        typer.Option(
            "--affine-tolerance-mm",
            help="Explicit absolute affine and spacing tolerance in millimetres.",
        ),
    ] = ...,  # type: ignore[assignment]
    prediction_manifest_output: Annotated[
        Path,
        typer.Option(
            "--prediction-manifest-output",
            help="Absolute external output path for the saved-prediction manifest JSON artifact.",
        ),
    ] = ...,  # type: ignore[assignment]
    metric_report_output: Annotated[
        Path,
        typer.Option(
            "--metric-report-output",
            help="Absolute external output path for the deterministic metric-report JSON artifact.",
        ),
    ] = ...,  # type: ignore[assignment]
) -> None:
    """Import saved baseline predictions and evaluate them with project-owned metrics only."""

    try:
        label_paths_by_case = load_reference_label_mapping_artifact(reference_mapping_artifact)
        result = import_saved_baseline_predictions(
            baseline_family=baseline_family,
            run_identifier=run_identifier,
            prediction_directory=prediction_directory,
            label_paths_by_case=label_paths_by_case,
            dataset_manifest_sha256=dataset_manifest_sha256,
            development_split_sha256=development_split_sha256,
            checkpoint_sha256=checkpoint_sha256,
            metric_config_sha256=metric_config_sha256,
            nsd_tolerance_mm=nsd_tolerance_mm,
            affine_tolerance_mm=affine_tolerance_mm,
            prediction_manifest_output_path=prediction_manifest_output,
            metric_report_output_path=metric_report_output,
        )
    except (BaselineEvaluationError, OSError) as exc:
        _raise_baseline_prediction_import_cli_error(exc)

    typer.echo("baseline prediction import success")
    typer.echo(f"manifest version: {BASELINE_PREDICTION_MANIFEST_VERSION}")
    typer.echo(f"baseline family: {baseline_family}")
    typer.echo(f"case count: {len(result.prediction_manifest.prediction_records)}")
    typer.echo(f"prediction manifest hash: {result.prediction_manifest.artifact_hash}")
    typer.echo(f"metric report hash: {result.metric_report.artifact_hash}")
