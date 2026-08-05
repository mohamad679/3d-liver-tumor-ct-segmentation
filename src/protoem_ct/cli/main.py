"""Command-line interface for ProtoEM-CT."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Final, cast

import numpy as np
import typer
from omegaconf import DictConfig, OmegaConf

from protoem_ct.artifacts import DatasetManifest, Phase2ArtifactError, phase2_artifact_from_json
from protoem_ct.artifacts.hashing import JsonValue, canonical_json_bytes, sha256_json
from protoem_ct.baselines import (
    BASELINE_PREDICTION_MANIFEST_VERSION,
    BASELINE_SYNTHETIC_DATASET_NAME,
    BASELINE_SYNTHETIC_FIXTURE_VERSION,
    BASELINE_SYNTHETIC_TEST_CASE_IDENTIFIERS,
    BASELINE_SYNTHETIC_TRAINING_CASE_IDENTIFIERS,
    NNUNET_V2_RUN_CONFIG_VERSION,
    PHASE3_GATE3_REPORT_VERSION,
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
from protoem_ct.evaluation.calibration import compute_binary_calibration_ece
from protoem_ct.evaluation.degradation import build_phase7_degradation_result
from protoem_ct.evaluation.dummy_inference import (
    DummyInferenceError,
    run_dummy_inference,
)
from protoem_ct.evaluation.failure_detection import (
    compute_failure_detection_auroc,
    compute_uncertainty_error_correlation,
)
from protoem_ct.evaluation.metrics import (
    EvaluationError,
    compute_binary_confusion_counts,
    dice_from_counts,
    evaluate_predictions,
)
from protoem_ct.evaluation.risk_coverage import compute_risk_coverage
from protoem_ct.evaluation.subgroups import (
    LesionSubgroupCase,
    build_phase7_lesion_subgroup_result,
)
from protoem_ct.external import (
    Phase8FreezeError,
    Phase8Wave2PublicationError,
    Phase8Wave3PublicationError,
    run_phase8_wave2_image_inventory,
    run_phase8_wave3_policy_publication,
    run_phase8_wave4_readiness_publication,
)
from protoem_ct.external.definitive_training import (
    Phase8DefinitiveTrainingError,
    build_phase8_definitive_training_config,
    build_unreleased_definitive_execution_release,
    execute_phase8_definitive_training,
    phase8_definitive_execution_release_from_mapping,
    phase8_definitive_training_config_from_mapping,
    publish_phase8_definitive_training_plan,
)
from protoem_ct.external.definitive_training_pilot import (
    DEFAULT_WALL_CLOCK_LIMIT_SECONDS,
    Phase8BoundedPilotError,
    run_phase8_bounded_pilot_with_watchdog,
)
from protoem_ct.external.internal_evidence import (
    PHASE8_CHECKPOINT_METADATA_SCHEMA_NAME,
    PHASE8_INTERNAL_EVIDENCE_SCHEMA_VERSION,
    PHASE8_VALIDATION_EVIDENCE_REFERENCE_SCHEMA_NAME,
    ArtifactReference,
    phase8_fixed_candidate_inventory_from_mapping,
    phase8_preprocessing_decision_from_mapping,
)
from protoem_ct.external.real_development_runner import (
    Phase8RealDevelopmentRunnerError,
    build_phase8_real_development_case_bindings,
    build_phase8_real_development_input_binding,
    build_phase8_real_development_run_plan,
    run_phase8_real_development_plan_publication,
)
from protoem_ct.external.tiny_real_verification import (
    Phase8TinyRealVerificationError,
    run_phase8_tiny_real_development_verification,
)
from protoem_ct.fewshot import (
    FewshotArtifactValidationError,
    FewshotInitializationReference,
    FewshotProtocolError,
    FewshotPublicationCollisionError,
    FewshotPublicationConfigError,
    FewshotPublicationError,
    FewshotPublicationIOError,
    FewshotPublicationPathError,
    generate_and_publish_phase4_fewshot_protocol,
    load_phase4_fewshot_protocol_settings,
)
from protoem_ct.protoem import (
    Phase6PublicationCollisionError,
    Phase6PublicationConfigError,
    Phase6PublicationError,
    Phase6PublicationIOError,
    Phase6PublicationPathError,
    load_phase6_protoem_settings,
    run_and_publish_phase6_protoem,
)
from protoem_ct.reporting import ReportGenerationError, generate_synthetic_report
from protoem_ct.retrieval import (
    Phase5ComparisonCollisionError,
    Phase5ComparisonConfigError,
    Phase5ComparisonError,
    Phase5ComparisonIOError,
    Phase5ComparisonLeakageError,
    Phase5ComparisonPathError,
    load_phase5_foundation_retrieval_settings,
    run_and_publish_phase5_retrieval,
)
from protoem_ct.robustness.artifacts import (
    PHASE7_CORRUPTION_MANIFEST_SCHEMA_NAME,
    PHASE7_CORRUPTION_MANIFEST_SCHEMA_VERSION,
    PHASE7_CORRUPTION_SPECIFICATION_SCHEMA_NAME,
    PHASE7_CORRUPTION_SPECIFICATION_SCHEMA_VERSION,
    PHASE7_RUN_SUMMARY_SCHEMA_NAME,
    PHASE7_RUN_SUMMARY_SCHEMA_VERSION,
    Phase7CorruptionManifest,
    Phase7CorruptionSpecification,
    Phase7GeometryRecord,
    Phase7RunSummary,
    Phase7TransformResult,
    phase7_corruption_manifest_to_json,
    phase7_corruption_specification_to_dict,
    phase7_run_summary_to_json,
)
from protoem_ct.robustness.blur import (
    apply_gaussian_blur,
    gaussian_blur_parameters_for_severity,
)
from protoem_ct.robustness.crop import apply_crop_fov_perturbation
from protoem_ct.robustness.geometry import SpatialGrid, build_spatial_grid, validate_binary_mask
from protoem_ct.robustness.intensity import apply_intensity_transform
from protoem_ct.robustness.noise import apply_gaussian_noise, gaussian_noise_parameters_for_severity
from protoem_ct.robustness.publication import (
    PHASE7_RUN_SUMMARY_NAME,
    Phase7PublicationCollisionError,
    Phase7PublicationError,
    Phase7PublicationInputs,
    Phase7PublicationIOError,
    Phase7PublicationPathError,
    build_phase7_geometry_records_collection_json,
    build_phase7_transform_results_collection_json,
    publish_phase7_artifacts,
)
from protoem_ct.robustness.resampling import (
    apply_anisotropic_downsampling,
    apply_slice_thickness_simulation,
)
from protoem_ct.uncertainty.artifacts import (
    phase7_calibration_result_to_json,
    phase7_degradation_result_to_json,
    phase7_lesion_subgroup_result_to_json,
    phase7_risk_coverage_result_to_json,
    phase7_uncertainty_result_to_json,
)
from protoem_ct.uncertainty.contracts import (
    BinaryPredictiveProbabilityMap,
    array_content_sha256,
    build_binary_predictive_probability_map,
    build_tta_probability_samples,
    build_tta_sample_manifest,
    build_tta_sample_record,
)
from protoem_ct.uncertainty.entropy import compute_predictive_entropy
from protoem_ct.uncertainty.publication import canonical_phase7_failure_detection_results_json
from protoem_ct.uncertainty.tta import TTAVarianceResult, compute_tta_probability_variance

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


def _raise_phase3_gate3_cli_error(exc: Exception) -> None:
    """Exit with a concise Phase 3 Gate 3 error without path leakage."""
    typer.secho(
        f"Phase 3 Gate 3 error: {type(exc).__name__}",
        err=True,
        fg=typer.colors.RED,
    )
    raise typer.Exit(code=1) from None


def _raise_phase4_protocol_cli_error(exc: Exception) -> None:
    """Exit with a concise Phase 4 protocol-generation error without path leakage."""
    typer.secho(
        f"Phase 4 few-shot protocol error: {type(exc).__name__}",
        err=True,
        fg=typer.colors.RED,
    )
    raise typer.Exit(code=1) from None


def _raise_phase5_retrieval_cli_error(exc: Exception) -> None:
    """Exit with a concise Phase 5 retrieval error without path or data leakage."""
    typer.secho(
        f"Phase 5 retrieval error: {type(exc).__name__}",
        err=True,
        fg=typer.colors.RED,
    )
    raise typer.Exit(code=1) from None


def _raise_phase6_protoem_cli_error(exc: Exception) -> None:
    """Exit with a concise Phase 6 ProtoEM error without path or data leakage."""
    typer.secho(
        f"Phase 6 ProtoEM error: {type(exc).__name__}",
        err=True,
        fg=typer.colors.RED,
    )
    raise typer.Exit(code=1) from None


def _raise_phase7_cli_error(exc: Exception) -> None:
    """Exit with a concise Phase 7 error without path or synthetic artifact leakage."""
    typer.secho(
        f"Phase 7 robustness/uncertainty error: {type(exc).__name__}",
        err=True,
        fg=typer.colors.RED,
    )
    raise typer.Exit(code=1) from None


def _raise_phase8_wave2_cli_error(exc: Exception) -> None:
    """Exit with a concise Phase 8 Wave 2 error without path or data leakage."""
    typer.secho(
        f"Phase 8 Wave 2 image inventory error: {type(exc).__name__}",
        err=True,
        fg=typer.colors.RED,
    )
    raise typer.Exit(code=1) from None


def _raise_phase8_wave3_cli_error(exc: Exception) -> None:
    """Exit with a concise Phase 8 Wave 3 error without path or data leakage."""
    typer.secho(
        f"Phase 8 Wave 3 policy publication error: {type(exc).__name__}",
        err=True,
        fg=typer.colors.RED,
    )
    raise typer.Exit(code=1) from None


def _raise_phase8_wave4_cli_error(exc: Exception) -> None:
    """Exit with a concise Phase 8 Wave 4 error without path or data leakage."""
    typer.secho(
        f"Phase 8 Wave 4 readiness error: {type(exc).__name__}",
        err=True,
        fg=typer.colors.RED,
    )
    raise typer.Exit(code=1) from None


def _raise_phase8_real_development_plan_cli_error(exc: Exception) -> None:
    """Exit with a concise Phase 8 real-development plan error without data leakage."""
    typer.secho(
        f"Phase 8 real-development plan error: {type(exc).__name__}",
        err=True,
        fg=typer.colors.RED,
    )
    raise typer.Exit(code=1) from None


def _raise_phase8_tiny_real_verification_cli_error(exc: Exception) -> None:
    """Exit with a concise Phase 8 tiny real-verification error without data leakage."""
    typer.secho(
        f"Phase 8 tiny real-development verification error: {type(exc).__name__}",
        err=True,
        fg=typer.colors.RED,
    )
    raise typer.Exit(code=1) from None


def _raise_phase8_bounded_pilot_cli_error(exc: Exception) -> None:
    """Exit with a concise Phase 8 bounded pilot error without data leakage."""
    typer.secho(
        f"Phase 8 bounded pilot error: {type(exc).__name__}",
        err=True,
        fg=typer.colors.RED,
    )
    raise typer.Exit(code=1) from None


def _raise_phase8_definitive_training_plan_cli_error(exc: Exception) -> None:
    """Exit with a concise Phase 8 definitive-training plan error without data leakage."""
    typer.secho(
        f"Phase 8 definitive-training plan error: {type(exc).__name__}",
        err=True,
        fg=typer.colors.RED,
    )
    raise typer.Exit(code=1) from None


def _raise_phase8_definitive_training_run_cli_error(exc: Exception) -> None:
    """Exit with a concise Phase 8 definitive-training run error without data leakage."""
    typer.secho(
        f"Phase 8 definitive-training run error: {type(exc).__name__}",
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


@app.command("generate-phase4-fewshot-protocol")
def generate_phase4_fewshot_protocol(
    manifest_path: Annotated[
        Path,
        typer.Option(
            "--manifest",
            help="Explicit absolute Phase 2 development manifest JSON path.",
        ),
    ] = ...,  # type: ignore[assignment]
    split_path: Annotated[
        Path,
        typer.Option(
            "--split",
            help="Explicit absolute Phase 2 development split JSON path.",
        ),
    ] = ...,  # type: ignore[assignment]
    lesion_artifact_path: Annotated[
        Path | None,
        typer.Option(
            "--lesion-artifact",
            help="Optional explicit absolute Phase 2 lesion-summary JSON path.",
        ),
    ] = None,
    output_root: Annotated[
        Path,
        typer.Option(
            "--output-root",
            help="Explicit absolute output root outside the repository.",
        ),
    ] = ...,  # type: ignore[assignment]
    initialization_reference_type: Annotated[
        str,
        typer.Option(
            "--initialization-reference-type",
            help="Explicit initialization reference type.",
        ),
    ] = ...,  # type: ignore[assignment]
    initialization_reference_identifier: Annotated[
        str,
        typer.Option(
            "--initialization-reference-identifier",
            help="Explicit initialization reference identifier.",
        ),
    ] = ...,  # type: ignore[assignment]
    initialization_artifact_sha256: Annotated[
        str | None,
        typer.Option(
            "--initialization-artifact-sha256",
            help="Optional explicit initialization provenance artifact SHA-256.",
        ),
    ] = None,
    initialization_checkpoint_sha256: Annotated[
        str | None,
        typer.Option(
            "--initialization-checkpoint-sha256",
            help="Optional explicit initialization checkpoint SHA-256.",
        ),
    ] = None,
    base_seed: Annotated[
        int,
        typer.Option(
            "--base-seed",
            help="Explicit nonnegative base seed for deterministic config derivation.",
        ),
    ] = ...,  # type: ignore[assignment]
    config_path: Annotated[
        Path,
        typer.Option(
            "--config",
            help="Path to the Phase 4 few-shot protocol OmegaConf YAML file.",
        ),
    ] = Path("configs/phase4_fewshot_protocol.yaml"),
) -> None:
    """Generate and publish deterministic Phase 4 few-shot protocol artifacts."""
    try:
        settings = load_phase4_fewshot_protocol_settings(config_path)
        initialization_reference = FewshotInitializationReference(
            reference_type=initialization_reference_type,
            reference_identifier=initialization_reference_identifier,
            artifact_sha256=initialization_artifact_sha256,
            checkpoint_sha256=initialization_checkpoint_sha256,
        )
        result = generate_and_publish_phase4_fewshot_protocol(
            manifest_path=manifest_path,
            split_path=split_path,
            lesion_artifact_path=lesion_artifact_path,
            output_root=output_root,
            initialization_reference=initialization_reference,
            base_seed=base_seed,
            settings=settings,
        )
    except (
        FewshotArtifactValidationError,
        FewshotPublicationCollisionError,
        FewshotPublicationConfigError,
        FewshotPublicationError,
        FewshotPublicationIOError,
        FewshotPublicationPathError,
        FewshotProtocolError,
    ) as exc:
        _raise_phase4_protocol_cli_error(exc)

    typer.echo("Phase 4 few-shot protocol generation success")
    typer.echo(f"support manifest count: {result.support_manifest_count}")
    typer.echo(f"adaptation config count: {result.adaptation_config_count}")
    typer.echo(f"protocol row count: {result.protocol_row_count}")
    typer.echo(f"protocol markdown data rows: {result.markdown_data_row_count}")
    typer.echo(f"source manifest hash: {result.source_development_manifest_hash}")
    typer.echo(f"source split hash: {result.source_development_split_hash}")
    typer.echo(f"immutable test cohort hash: {result.immutable_test_cohort_hash}")
    typer.echo(f"protocol table hash: {result.protocol_table_hash}")
    typer.echo(f"leakage check passed: {str(result.leakage_check_passed).lower()}")
    typer.echo(f"output root: {result.output_root}")


@app.command("run-phase5-retrieval")
def run_phase5_retrieval_command(
    config_path: Annotated[
        Path,
        typer.Option(
            "--config",
            help="Path to the Phase 5 foundation-retrieval OmegaConf YAML file.",
        ),
    ] = Path("configs/phase5_foundation_retrieval.yaml"),
    output_root: Annotated[
        Path,
        typer.Option(
            "--output-root",
            help="Explicit absolute external output root for the synthetic Phase 5 run.",
        ),
    ] = ...,  # type: ignore[assignment]
) -> None:
    """Run the bounded synthetic Phase 5 retrieval comparison and publish artifacts."""
    try:
        settings = load_phase5_foundation_retrieval_settings(config_path)
        result = run_and_publish_phase5_retrieval(
            output_root=output_root,
            settings=settings,
        )
    except (
        Phase5ComparisonCollisionError,
        Phase5ComparisonConfigError,
        Phase5ComparisonError,
        Phase5ComparisonIOError,
        Phase5ComparisonLeakageError,
        Phase5ComparisonPathError,
    ) as exc:
        _raise_phase5_retrieval_cli_error(exc)

    typer.echo("Phase 5 retrieval comparison success")
    typer.echo("mode: synthetic_only")
    typer.echo(f"comparison identity: {result.comparison_identity_sha256}")
    typer.echo(f"support set identity: {result.support_set_identity_sha256}")
    typer.echo(f"comparison artifact: {result.comparison_path}")
    typer.echo(f"comparison markdown: {result.markdown_path}")
    typer.echo(f"run summary artifact: {result.run_summary_path}")
    typer.echo(f"effective config artifact: {result.effective_config_path}")
    typer.echo(f"output root: {result.output_root}")


@app.command("run-phase6-protoem")
def run_phase6_protoem_command(
    config_path: Annotated[
        Path,
        typer.Option(
            "--config",
            help="Path to the Phase 6 ProtoEM-CT OmegaConf YAML file.",
        ),
    ] = Path("configs/phase6_protoem_ct.yaml"),
    output_root: Annotated[
        Path,
        typer.Option(
            "--output-root",
            help="Explicit absolute external output root for the synthetic Phase 6 run.",
        ),
    ] = ...,  # type: ignore[assignment]
) -> None:
    """Run the bounded synthetic Phase 6 ProtoEM-CT publication path."""

    try:
        settings = load_phase6_protoem_settings(config_path)
        result = run_and_publish_phase6_protoem(
            output_root=output_root,
            settings=settings,
        )
    except (
        Phase6PublicationCollisionError,
        Phase6PublicationConfigError,
        Phase6PublicationError,
        Phase6PublicationIOError,
        Phase6PublicationPathError,
    ) as exc:
        _raise_phase6_protoem_cli_error(exc)

    typer.echo("Phase 6 ProtoEM success")
    typer.echo("mode: synthetic_only")
    typer.echo(f"execution status: {result.execution_status}")
    typer.echo(f"config identity: {result.config_hash}")
    typer.echo(f"initialization identity: {result.initialization_identity_hash}")
    typer.echo(f"stopping reason: {result.stopping_reason}")
    typer.echo(f"run summary artifact: {result.run_summary_path}")
    typer.echo(f"effective config artifact: {result.effective_config_path}")
    typer.echo(f"objective trace artifact: {result.objective_trace_path}")
    typer.echo(f"output root: {result.output_root}")


PHASE7_CONFIG_ROOT_KEY: Final[str] = "phase7_robustness_uncertainty"
PHASE7_CONFIG_SCHEMA_VERSION: Final[str] = "v1"
_PHASE7_REQUIRED_CORRUPTIONS: Final[tuple[str, ...]] = (
    "hu_window_shift",
    "intensity_scale",
    "intensity_offset",
    "contrast_shift",
    "gaussian_noise",
    "gaussian_blur",
    "slice_thickness",
    "anisotropic_downsampling",
    "crop_fov",
)


class Phase7CliConfigError(ValueError):
    """Raised when the Phase 7 synthetic CLI configuration is invalid."""


@dataclass(frozen=True, slots=True)
class Phase7SyntheticSettings:
    """Validated bounded synthetic Phase 7 CLI settings."""

    effective_config_mapping: dict[str, JsonValue]
    deterministic_seed: int
    corruption_specs: tuple[Phase7CorruptionSpecification, ...]
    calibration_bin_count: int


@dataclass(frozen=True, slots=True)
class Phase7SyntheticPublicationResult:
    """Small command result for one bounded synthetic Phase 7 publication."""

    output_root: Path
    config_hash: str
    corruption_manifest_hash: str
    uncertainty_type: str
    calibration_ece: float | None
    risk_coverage_point_count: int
    transform_count: int
    geometry_record_count: int
    run_summary_path: Path
    reused_existing_output: bool


def load_phase7_robustness_uncertainty_settings(
    config_path: Path,
) -> Phase7SyntheticSettings:
    """Load versioned synthetic-only Phase 7 settings from an OmegaConf YAML file."""

    try:
        raw_config = OmegaConf.load(config_path)
    except OSError as exc:
        raise Phase7CliConfigError("Phase 7 config could not be read.") from exc
    if not isinstance(raw_config, DictConfig):
        raise Phase7CliConfigError("Phase 7 config must be an OmegaConf mapping.")
    container = OmegaConf.to_container(raw_config, resolve=True)
    if not isinstance(container, dict):
        raise Phase7CliConfigError("Phase 7 config must resolve to a mapping.")
    root = _expect_phase7_mapping(container.get(PHASE7_CONFIG_ROOT_KEY), PHASE7_CONFIG_ROOT_KEY)
    if set(root) != {
        "schema_version",
        "synthetic_mode_only",
        "deterministic_seed",
        "calibration_bin_count",
        "corruptions",
    }:
        raise Phase7CliConfigError("Phase 7 config root contains unexpected fields.")
    if root["schema_version"] != PHASE7_CONFIG_SCHEMA_VERSION:
        raise Phase7CliConfigError("Phase 7 config schema_version must be v1.")
    if root["synthetic_mode_only"] is not True:
        raise Phase7CliConfigError("Phase 7 CLI supports synthetic-only execution.")
    deterministic_seed = _expect_phase7_int(root["deterministic_seed"], "deterministic_seed")
    calibration_bin_count = _expect_phase7_int(
        root["calibration_bin_count"],
        "calibration_bin_count",
    )
    if calibration_bin_count <= 0:
        raise Phase7CliConfigError("calibration_bin_count must be positive.")
    corruptions = _expect_phase7_sequence(root["corruptions"], "corruptions")
    if len(corruptions) != len(_PHASE7_REQUIRED_CORRUPTIONS):
        raise Phase7CliConfigError("Phase 7 config must list every required corruption once.")
    specs = tuple(
        _phase7_corruption_specification_from_config(
            item,
            default_seed=deterministic_seed,
            index=index,
        )
        for index, item in enumerate(corruptions)
    )
    names = tuple(spec.corruption_name for spec in specs)
    if names != _PHASE7_REQUIRED_CORRUPTIONS:
        raise Phase7CliConfigError("Phase 7 corruptions must use canonical ordering.")
    effective_config = _json_value_mapping({PHASE7_CONFIG_ROOT_KEY: root})
    return Phase7SyntheticSettings(
        effective_config_mapping=effective_config,
        deterministic_seed=deterministic_seed,
        corruption_specs=specs,
        calibration_bin_count=calibration_bin_count,
    )


@app.command("run-phase7-robustness-uncertainty")
def run_phase7_robustness_uncertainty_command(
    config_path: Annotated[
        Path,
        typer.Option(
            "--config",
            help="Path to the Phase 7 robustness/uncertainty synthetic YAML file.",
        ),
    ] = Path("configs/phase7_robustness_uncertainty.yaml"),
    output_root: Annotated[
        Path,
        typer.Option(
            "--output-root",
            help="Explicit absolute external output root for the synthetic Phase 7 run.",
        ),
    ] = ...,  # type: ignore[assignment]
) -> None:
    """Run the bounded CPU synthetic Phase 7 robustness and uncertainty publication path."""

    try:
        settings = load_phase7_robustness_uncertainty_settings(config_path)
        result = run_and_publish_phase7_robustness_uncertainty(
            settings=settings,
            output_root=output_root,
        )
    except (
        Phase7CliConfigError,
        Phase7PublicationCollisionError,
        Phase7PublicationError,
        Phase7PublicationIOError,
        Phase7PublicationPathError,
        ValueError,
    ) as exc:
        _raise_phase7_cli_error(exc)

    typer.echo("Phase 7 robustness/uncertainty success")
    typer.echo("mode: synthetic_only")
    typer.echo(f"config identity: {result.config_hash}")
    typer.echo(f"corruption manifest: {result.corruption_manifest_hash}")
    typer.echo(f"uncertainty type: {result.uncertainty_type}")
    typer.echo(f"transform count: {result.transform_count}")
    typer.echo(f"geometry record count: {result.geometry_record_count}")
    typer.echo(f"calibration ece: {result.calibration_ece}")
    typer.echo(f"risk coverage points: {result.risk_coverage_point_count}")
    typer.echo(f"run summary artifact: {result.run_summary_path}")
    typer.echo(f"reused existing output: {str(result.reused_existing_output).lower()}")
    typer.echo(f"output root: {result.output_root}")


@app.command("run-phase8-wave2-image-inventory")
def run_phase8_wave2_image_inventory_command(
    dataset_root: Annotated[
        Path,
        typer.Option(
            "--dataset-root",
            help="Explicit absolute extracted 3D-IRCADb-01 3Dircadb1 root.",
        ),
    ] = ...,  # type: ignore[assignment]
    dataset_archive: Annotated[
        Path,
        typer.Option(
            "--dataset-archive",
            help="Explicit absolute outer 3Dircadb1.zip archive path.",
        ),
    ] = ...,  # type: ignore[assignment]
    output_root: Annotated[
        Path,
        typer.Option(
            "--output-root",
            help="Explicit absolute external Wave 2 output root outside the repository.",
        ),
    ] = ...,  # type: ignore[assignment]
    repository_root: Annotated[
        Path | None,
        typer.Option(
            "--repository-root",
            help="Explicit absolute repository root used only for output-root rejection.",
        ),
    ] = None,
) -> None:
    """Run Phase 8 Wave 2 image-only 3D-IRCADb discovery, QA, and manifest publication."""

    try:
        result = run_phase8_wave2_image_inventory(
            dataset_root=dataset_root,
            dataset_archive=dataset_archive,
            output_root=output_root,
            repository_root=repository_root or Path.cwd(),
        )
    except (Phase8Wave2PublicationError, ValueError, OSError) as exc:
        _raise_phase8_wave2_cli_error(exc)

    typer.echo("Phase 8 Wave 2 image-only inventory complete")
    typer.echo(f"case_count: {result.case_count}")
    typer.echo(f"qa_pass_count: {result.qa_pass_count}")
    typer.echo(f"qa_fail_count: {result.qa_fail_count}")
    typer.echo(f"manifest_hash: {result.manifest.manifest_hash}")
    for anonymous_case_id, reason_codes in result.anonymous_case_reason_codes:
        typer.echo(f"failed_case: {anonymous_case_id} reason_codes={','.join(reason_codes)}")
    for relative_name, digest in sorted(result.artifact_hashes.items()):
        typer.echo(f"artifact: {relative_name} sha256={digest}")


@app.command("run-phase8-wave3-policy")
def run_phase8_wave3_policy_command(
    wave2_artifact_root: Annotated[
        Path,
        typer.Option(
            "--wave2-artifact-root",
            help="Explicit absolute corrected Wave 2 image-only artifact root.",
        ),
    ] = ...,  # type: ignore[assignment]
    output_root: Annotated[
        Path,
        typer.Option(
            "--output-root",
            help="Explicit absolute external Wave 3 output root outside the repository.",
        ),
    ] = ...,  # type: ignore[assignment]
    repository_root: Annotated[
        Path | None,
        typer.Option(
            "--repository-root",
            help="Explicit absolute repository root used only for output-root rejection.",
        ),
    ] = None,
) -> None:
    """Run Phase 8 Wave 3 policy, domain-shift, and eligibility publication."""

    try:
        result = run_phase8_wave3_policy_publication(
            wave2_artifact_root=wave2_artifact_root,
            output_root=output_root,
            repository_root=repository_root or Path.cwd(),
        )
    except (Phase8Wave3PublicationError, ValueError, OSError) as exc:
        _raise_phase8_wave3_cli_error(exc)

    typer.echo("Phase 8 Wave 3 policy publication complete")
    typer.echo(f"case_count: {result.case_count}")
    typer.echo(f"image_qa_eligible_count: {result.image_qa_eligible_count}")
    typer.echo(f"inference_eligible_count: {result.inference_eligible_count}")
    typer.echo(f"label_compatibility_pending_count: {result.label_compatibility_pending_count}")
    typer.echo(f"evaluation_eligible_count: {result.evaluation_eligible_count}")
    typer.echo(f"deferred_count: {result.deferred_count}")
    typer.echo(f"label_mapping_policy_hash: {result.label_mapping_policy.policy_hash}")
    typer.echo(f"domain_shift_record_hash: {result.domain_shift_record.domain_shift_record_hash}")
    typer.echo(f"eligibility_policy_hash: {result.eligibility_policy.policy_hash}")
    typer.echo(f"cohort_accounting_hash: {result.cohort_accounting.accounting_hash}")
    for relative_name, digest in sorted(result.artifact_hashes.items()):
        typer.echo(f"artifact: {relative_name} sha256={digest}")


@app.command("run-phase8-wave4-readiness")
def run_phase8_wave4_readiness_command(
    wave2_artifact_root: Annotated[
        Path,
        typer.Option(
            "--wave2-artifact-root",
            help="Explicit absolute corrected Wave 2 image-only artifact root.",
        ),
    ] = ...,  # type: ignore[assignment]
    wave3_artifact_root: Annotated[
        Path,
        typer.Option(
            "--wave3-artifact-root",
            help="Explicit absolute approved Wave 3 policy artifact root.",
        ),
    ] = ...,  # type: ignore[assignment]
    internal_artifact_parent: Annotated[
        Path,
        typer.Option(
            "--internal-artifact-parent",
            help="Explicit absolute parent containing approved internal development artifacts.",
        ),
    ] = ...,  # type: ignore[assignment]
    output_root: Annotated[
        Path,
        typer.Option(
            "--output-root",
            help="Explicit absolute external Wave 4 output root outside the repository.",
        ),
    ] = ...,  # type: ignore[assignment]
    repository_root: Annotated[
        Path | None,
        typer.Option(
            "--repository-root",
            help="Explicit absolute repository root used only for output-root rejection.",
        ),
    ] = None,
) -> None:
    """Run Phase 8 Wave 4 guarded internal-evidence readiness publication."""

    try:
        result = run_phase8_wave4_readiness_publication(
            wave2_artifact_root=wave2_artifact_root,
            wave3_artifact_root=wave3_artifact_root,
            internal_artifact_parent=internal_artifact_parent,
            output_root=output_root,
            repository_root=repository_root or Path.cwd(),
        )
    except (Phase8FreezeError, ValueError, OSError) as exc:
        _raise_phase8_wave4_cli_error(exc)

    typer.echo("Phase 8 Wave 4 readiness publication complete")
    typer.echo(f"readiness_state: {result.wave4_state}")
    typer.echo(f"freeze_generated: {str(result.freeze_generated).lower()}")
    typer.echo(f"preregistration_generated: {str(result.preregistration_generated).lower()}")
    typer.echo(f"readiness_hash: {result.readiness_hash}")
    typer.echo(f"summary_hash: {result.summary_hash}")
    for relative_name, digest in sorted(result.artifact_hashes.items()):
        typer.echo(f"artifact: {relative_name} sha256={digest}")


def _read_json_mapping_strict(path: Path) -> dict[str, object]:
    resolved_path = path.resolve(strict=True)
    decoded = json.loads(resolved_path.read_text(encoding="utf-8"))
    if not isinstance(decoded, dict):
        raise Phase8RealDevelopmentRunnerError("JSON artifact root must be an object.")
    return cast(dict[str, object], decoded)


@app.command("plan-phase8-real-development-run")
def plan_phase8_real_development_run_command(
    manifest_path: Annotated[
        Path,
        typer.Option(
            "--manifest-path",
            help="Explicit absolute path to an approved Phase 2 dataset manifest JSON file.",
        ),
    ] = ...,  # type: ignore[assignment]
    split_path: Annotated[
        Path,
        typer.Option(
            "--split-path",
            help="Explicit absolute path to an approved Phase 2 development split JSON file.",
        ),
    ] = ...,  # type: ignore[assignment]
    expected_manifest_sha256: Annotated[
        str,
        typer.Option(
            "--expected-manifest-sha256",
            help="Caller-declared SHA-256 of the manifest file, verified before parsing.",
        ),
    ] = ...,  # type: ignore[assignment]
    expected_split_sha256: Annotated[
        str,
        typer.Option(
            "--expected-split-sha256",
            help="Caller-declared SHA-256 of the split file, verified before parsing.",
        ),
    ] = ...,  # type: ignore[assignment]
    candidate_inventory_path: Annotated[
        Path,
        typer.Option(
            "--candidate-inventory-path",
            help="Explicit absolute path to a Phase 8 fixed candidate inventory JSON file.",
        ),
    ] = ...,  # type: ignore[assignment]
    candidate_id: Annotated[
        str,
        typer.Option(
            "--candidate-id",
            help="Candidate identifier selected from the fixed candidate inventory.",
        ),
    ] = ...,  # type: ignore[assignment]
    preprocessing_decision_path: Annotated[
        Path,
        typer.Option(
            "--preprocessing-decision-path",
            help="Explicit absolute path to a Phase 8 preprocessing decision JSON file.",
        ),
    ] = ...,  # type: ignore[assignment]
    approved_development_artifact_set_identity: Annotated[
        str,
        typer.Option(
            "--approved-development-artifact-set-identity",
            help="Conservative identifier for the approved development artifact set.",
        ),
    ] = ...,  # type: ignore[assignment]
    output_root: Annotated[
        Path,
        typer.Option(
            "--output-root",
            help="Explicit absolute scaffold-plan output root outside the repository.",
        ),
    ] = ...,  # type: ignore[assignment]
    repository_root: Annotated[
        Path | None,
        typer.Option(
            "--repository-root",
            help="Explicit absolute repository root used only for output-root rejection.",
        ),
    ] = None,
) -> None:
    """Plan (never execute) a future real-development MONAI SegResNet run.

    This command is metadata-only: it never opens an image, label, prediction,
    or checkpoint file, and it never trains, infers, or computes a real metric.
    It reads exactly the manifest, split, candidate-inventory, and
    preprocessing-decision JSON files named on the command line and publishes a
    scaffold-only run plan.
    """

    try:
        input_binding = build_phase8_real_development_input_binding(
            manifest_path=manifest_path,
            split_path=split_path,
            expected_manifest_sha256=expected_manifest_sha256,
            expected_split_sha256=expected_split_sha256,
            approved_development_artifact_set_identity=(approved_development_artifact_set_identity),
        )
        candidate_inventory = phase8_fixed_candidate_inventory_from_mapping(
            _read_json_mapping_strict(candidate_inventory_path)
        )
        preprocessing_decision = phase8_preprocessing_decision_from_mapping(
            _read_json_mapping_strict(preprocessing_decision_path)
        )
        candidate_by_id = {
            candidate.candidate_id: candidate for candidate in candidate_inventory.candidates
        }
        candidate = candidate_by_id.get(candidate_id)
        if candidate is None:
            raise Phase8RealDevelopmentRunnerError(
                f"candidate_id {candidate_id!r} is not present in the fixed candidate inventory."
            )
        preprocessing_decision_reference = ArtifactReference(
            schema_name=preprocessing_decision.schema_name,
            schema_version=preprocessing_decision.schema_version,
            artifact_hash=preprocessing_decision.preprocessing_decision_hash,
            artifact_role="preprocessing_decision",
        )
        run_plan = build_phase8_real_development_run_plan(
            input_binding=input_binding,
            candidate_inventory=candidate_inventory,
            candidate_id=candidate_id,
            training_config_reference=candidate.training_config_reference,
            preprocessing_decision_reference=preprocessing_decision_reference,
            fixed_seeds=candidate.fixed_seeds,
            expected_checkpoint_metadata_schema=(
                PHASE8_CHECKPOINT_METADATA_SCHEMA_NAME,
                PHASE8_INTERNAL_EVIDENCE_SCHEMA_VERSION,
            ),
            expected_validation_evidence_schema=(
                PHASE8_VALIDATION_EVIDENCE_REFERENCE_SCHEMA_NAME,
                PHASE8_INTERNAL_EVIDENCE_SCHEMA_VERSION,
            ),
            expected_output_artifact_names=(
                "checkpoint_metadata",
                "validation_evidence",
                "preprocessing_evidence",
            ),
        )
        resolved_manifest_path = manifest_path.resolve(strict=True)
        manifest = phase2_artifact_from_json(
            resolved_manifest_path.read_text(encoding="utf-8"), DatasetManifest
        )
        case_bindings = build_phase8_real_development_case_bindings(
            run_plan=run_plan, manifest=manifest
        )
        result = run_phase8_real_development_plan_publication(
            input_binding=input_binding,
            run_plan=run_plan,
            case_bindings=case_bindings,
            output_root=output_root,
            repository_root=repository_root or Path.cwd(),
        )
    except (Phase8RealDevelopmentRunnerError, ValueError, OSError) as exc:
        _raise_phase8_real_development_plan_cli_error(exc)

    typer.echo("Phase 8 real-development plan publication complete")
    typer.echo("scaffold_only: true")
    typer.echo("pixel_access_not_started: true")
    typer.echo("training_executed: false")
    typer.echo("checkpoint_created: false")
    typer.echo("real_metrics_computed: false")
    typer.echo(f"input_binding_hash: {result.input_binding_hash}")
    typer.echo(f"run_plan_hash: {result.run_plan_hash}")
    typer.echo(f"case_binding_collection_hash: {result.case_binding_collection_hash}")
    typer.echo(f"summary_hash: {result.summary_hash}")
    for relative_name, digest in sorted(result.artifact_hashes.items()):
        typer.echo(f"artifact: {relative_name} sha256={digest}")


def _parse_tiny_real_package_versions(payload: str) -> dict[str, str]:
    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise Phase8TinyRealVerificationError(
            "--package-versions-json must be valid JSON."
        ) from exc
    if not isinstance(decoded, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in decoded.items()
    ):
        raise Phase8TinyRealVerificationError(
            "--package-versions-json must decode to a JSON object of string to string."
        )
    return cast(dict[str, str], decoded)


@app.command("run-phase8-tiny-real-development-verification")
def run_phase8_tiny_real_development_verification_command(
    approve_tiny_real_verification: Annotated[
        bool,
        typer.Option(
            "--approve-tiny-real-verification",
            help=(
                "Required explicit approval. Without this flag the command refuses "
                "before opening any file."
            ),
        ),
    ] = False,
    manifest_path: Annotated[
        Path,
        typer.Option(
            "--manifest-path",
            help="Explicit absolute path to an approved Phase 2 dataset manifest JSON file.",
        ),
    ] = ...,  # type: ignore[assignment]
    split_path: Annotated[
        Path,
        typer.Option(
            "--split-path",
            help="Explicit absolute path to an approved Phase 2 development split JSON file.",
        ),
    ] = ...,  # type: ignore[assignment]
    expected_manifest_sha256: Annotated[
        str,
        typer.Option(
            "--expected-manifest-sha256",
            help="Caller-declared SHA-256 of the manifest file, verified before parsing.",
        ),
    ] = ...,  # type: ignore[assignment]
    expected_split_sha256: Annotated[
        str,
        typer.Option(
            "--expected-split-sha256",
            help="Caller-declared SHA-256 of the split file, verified before parsing.",
        ),
    ] = ...,  # type: ignore[assignment]
    dataset_root: Annotated[
        Path,
        typer.Option(
            "--dataset-root",
            help="Explicit absolute, read-only, approved raw dataset root.",
        ),
    ] = ...,  # type: ignore[assignment]
    output_root: Annotated[
        Path,
        typer.Option(
            "--output-root",
            help=(
                "Explicit absolute, nonexistent, non-symlinked output root outside the repository."
            ),
        ),
    ] = ...,  # type: ignore[assignment]
    approved_development_artifact_set_identity: Annotated[
        str,
        typer.Option(
            "--approved-development-artifact-set-identity",
            help="Conservative identifier for the approved development artifact set.",
        ),
    ] = ...,  # type: ignore[assignment]
    git_commit: Annotated[
        str,
        typer.Option(
            "--git-commit",
            help="Originating Git commit hash (7-64 lowercase hex characters).",
        ),
    ] = ...,  # type: ignore[assignment]
    package_versions_json: Annotated[
        str,
        typer.Option(
            "--package-versions-json",
            help='JSON object of package name to version, e.g. {"torch": "2.2.0"}.',
        ),
    ] = ...,  # type: ignore[assignment]
    max_steps: Annotated[
        int,
        typer.Option(
            "--max-steps",
            help="Bounded optimizer step count; must satisfy 1 <= max_steps <= 2.",
        ),
    ] = 2,
    seed: Annotated[
        int,
        typer.Option("--seed", help="Deterministic seed for the bounded verification run."),
    ] = 1729,
    repository_root: Annotated[
        Path | None,
        typer.Option(
            "--repository-root",
            help="Explicit absolute repository root used only for output-root rejection.",
        ),
    ] = None,
) -> None:
    """Run a bounded, verification-only Phase 8 tiny real-development check.

    This command is a deliberately narrow engineering dry/overfit
    verification, not a real training run. It opens exactly one real train
    and one real validation NIfTI image/label pair, is CPU-only with AMP
    disabled, runs at most two optimizer steps over a single bounded spatial
    patch, and computes no Dice/IoU/HD95 or other scientific metric. Every
    published checkpoint is hard-coded not freeze eligible, not selection
    eligible, and not definitive training. Requires explicit
    ``--approve-tiny-real-verification``; without it, the command refuses
    before opening any file.
    """

    if not approve_tiny_real_verification:
        typer.secho(
            "Phase 8 tiny real-development verification requires --approve-tiny-real-verification.",
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    try:
        package_versions = _parse_tiny_real_package_versions(package_versions_json)
        result = run_phase8_tiny_real_development_verification(
            manifest_path=manifest_path,
            split_path=split_path,
            expected_manifest_sha256=expected_manifest_sha256,
            expected_split_sha256=expected_split_sha256,
            dataset_root=dataset_root,
            output_root=output_root,
            repository_root=repository_root or Path.cwd(),
            approved_development_artifact_set_identity=(approved_development_artifact_set_identity),
            git_commit=git_commit,
            package_versions=package_versions,
            max_steps=max_steps,
            seed=seed,
        )
    except (Phase8TinyRealVerificationError, ValueError, OSError) as exc:
        _raise_phase8_tiny_real_verification_cli_error(exc)

    typer.echo("Phase 8 tiny real-development verification complete")
    typer.echo("verification_only: true")
    typer.echo("scientific_metrics_computed: false")
    typer.echo("freeze_eligible: false")
    typer.echo("selection_eligible: false")
    typer.echo("definitive_training: false")
    typer.echo(f"config_hash: {result.config_hash}")
    typer.echo(f"access_ledger_hash: {result.access_ledger_hash}")
    typer.echo(f"checkpoint_metadata_hash: {result.checkpoint_metadata_hash}")
    typer.echo(f"summary_hash: {result.summary_hash}")
    for relative_name, digest in sorted(result.artifact_hashes.items()):
        typer.echo(f"artifact: {relative_name} sha256={digest}")


def _parse_bounded_pilot_package_versions(payload: str) -> dict[str, str]:
    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise Phase8BoundedPilotError("--package-versions-json must be valid JSON.") from exc
    if not isinstance(decoded, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in decoded.items()
    ):
        raise Phase8BoundedPilotError(
            "--package-versions-json must decode to a JSON object of string to string."
        )
    return cast(dict[str, str], decoded)


@app.command("run-phase8-bounded-real-development-pilot")
def run_phase8_bounded_real_development_pilot_command(
    approve_bounded_pilot: Annotated[
        bool,
        typer.Option(
            "--approve-bounded-pilot",
            help=(
                "Required explicit approval. Without this flag the command refuses "
                "before opening any file."
            ),
        ),
    ] = False,
    manifest_path: Annotated[
        Path,
        typer.Option(
            "--manifest-path", help="Explicit absolute path to an approved Phase 2 manifest JSON."
        ),
    ] = ...,  # type: ignore[assignment]
    split_path: Annotated[
        Path,
        typer.Option(
            "--split-path",
            help="Explicit absolute path to an approved Phase 2 development split JSON.",
        ),
    ] = ...,  # type: ignore[assignment]
    lesion_components_path: Annotated[
        Path,
        typer.Option(
            "--lesion-components-path",
            help="Explicit absolute path to the approved Phase 2 lesion-components JSON.",
        ),
    ] = ...,  # type: ignore[assignment]
    expected_manifest_sha256: Annotated[
        str,
        typer.Option(
            "--expected-manifest-sha256",
            help="Caller-declared SHA-256 of the manifest file, verified before parsing.",
        ),
    ] = ...,  # type: ignore[assignment]
    expected_split_sha256: Annotated[
        str,
        typer.Option(
            "--expected-split-sha256",
            help="Caller-declared SHA-256 of the split file, verified before parsing.",
        ),
    ] = ...,  # type: ignore[assignment]
    expected_lesion_components_sha256: Annotated[
        str,
        typer.Option(
            "--expected-lesion-components-sha256",
            help="Caller-declared SHA-256 of the lesion-components file, verified before parsing.",
        ),
    ] = ...,  # type: ignore[assignment]
    input_binding_path: Annotated[
        Path,
        typer.Option(
            "--input-binding-path",
            help="Explicit absolute path to the Substage 4B real-development input binding JSON.",
        ),
    ] = ...,  # type: ignore[assignment]
    candidate_inventory_path: Annotated[
        Path,
        typer.Option(
            "--candidate-inventory-path",
            help="Explicit absolute path to the Substage 4B fixed candidate inventory JSON.",
        ),
    ] = ...,  # type: ignore[assignment]
    preprocessing_decision_path: Annotated[
        Path,
        typer.Option(
            "--preprocessing-decision-path",
            help="Explicit absolute path to the Substage 4B preprocessing decision JSON.",
        ),
    ] = ...,  # type: ignore[assignment]
    expected_input_binding_sha256: Annotated[
        str,
        typer.Option(
            "--expected-input-binding-sha256",
            help="Caller-declared SHA-256 of the input binding file.",
        ),
    ] = ...,  # type: ignore[assignment]
    expected_candidate_inventory_sha256: Annotated[
        str,
        typer.Option(
            "--expected-candidate-inventory-sha256",
            help="Caller-declared SHA-256 of the candidate inventory file.",
        ),
    ] = ...,  # type: ignore[assignment]
    expected_preprocessing_decision_sha256: Annotated[
        str,
        typer.Option(
            "--expected-preprocessing-decision-sha256",
            help="Caller-declared SHA-256 of the preprocessing decision file.",
        ),
    ] = ...,  # type: ignore[assignment]
    dataset_root: Annotated[
        Path,
        typer.Option(
            "--dataset-root", help="Explicit absolute, read-only, approved raw dataset root."
        ),
    ] = ...,  # type: ignore[assignment]
    output_root: Annotated[
        Path,
        typer.Option(
            "--output-root",
            help=(
                "Explicit absolute, nonexistent, non-symlinked output root outside the repository."
            ),
        ),
    ] = ...,  # type: ignore[assignment]
    git_commit: Annotated[
        str,
        typer.Option(
            "--git-commit", help="Originating Git commit hash (7-64 lowercase hex characters)."
        ),
    ] = ...,  # type: ignore[assignment]
    package_versions_json: Annotated[
        str,
        typer.Option(
            "--package-versions-json",
            help='JSON object of package name to version, e.g. {"torch": "2.2.0"}.',
        ),
    ] = ...,  # type: ignore[assignment]
    seed: Annotated[
        int,
        typer.Option("--seed", help="Deterministic seed for the bounded pilot run."),
    ] = 1729,
    wall_clock_limit_seconds: Annotated[
        float,
        typer.Option(
            "--wall-clock-limit-seconds",
            help="Hard wall-clock watchdog limit in seconds (default 2700.0 = 45 minutes).",
        ),
    ] = DEFAULT_WALL_CLOCK_LIMIT_SECONDS,
    repository_root: Annotated[
        Path | None,
        typer.Option(
            "--repository-root",
            help="Explicit absolute repository root used only for output-root rejection.",
        ),
    ] = None,
) -> None:
    """Run a bounded, verification-scale Phase 8 real-data training pilot.

    This is a deliberately bounded, explicitly user-approved, CPU-only,
    verification-scale real-data training pilot -- not definitive training.
    It opens exactly two real train NIfTI pairs (one tumor-positive, one
    empty-target) and one real validation NIfTI pair, runs an exact 10:10
    foreground-aware patch sampling schedule over 20 optimizer steps, and
    performs one full-volume sliding-window validation forward pass. It
    computes no Dice/IoU/HD95/NSD or other scientific metric anywhere. Every
    published checkpoint is hard-coded not freeze eligible, not selection
    eligible, not definitive training, and not scientific-metric eligible.
    A hard 45-minute wall-clock watchdog aborts the run with no partial-
    success publication. Requires explicit ``--approve-bounded-pilot``;
    without it, the command refuses before opening any file.
    """

    if not approve_bounded_pilot:
        typer.secho(
            "Phase 8 bounded pilot requires --approve-bounded-pilot.",
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    try:
        package_versions = _parse_bounded_pilot_package_versions(package_versions_json)
        result = run_phase8_bounded_pilot_with_watchdog(
            manifest_path=manifest_path,
            split_path=split_path,
            lesion_components_path=lesion_components_path,
            expected_manifest_sha256=expected_manifest_sha256,
            expected_split_sha256=expected_split_sha256,
            expected_lesion_components_sha256=expected_lesion_components_sha256,
            input_binding_path=input_binding_path,
            candidate_inventory_path=candidate_inventory_path,
            preprocessing_decision_path=preprocessing_decision_path,
            expected_input_binding_sha256=expected_input_binding_sha256,
            expected_candidate_inventory_sha256=expected_candidate_inventory_sha256,
            expected_preprocessing_decision_sha256=expected_preprocessing_decision_sha256,
            dataset_root=dataset_root,
            output_root=output_root,
            repository_root=repository_root or Path.cwd(),
            git_commit=git_commit,
            package_versions=package_versions,
            seed=seed,
            wall_clock_limit_seconds=wall_clock_limit_seconds,
        )
    except (Phase8BoundedPilotError, ValueError, OSError) as exc:
        _raise_phase8_bounded_pilot_cli_error(exc)

    typer.echo("Phase 8 bounded real-development pilot complete")
    typer.echo("pilot_only: true")
    typer.echo("scientific_metrics_computed: false")
    typer.echo("freeze_eligible: false")
    typer.echo("selection_eligible: false")
    typer.echo("definitive_training: false")
    typer.echo(f"config_hash: {result.config_hash}")
    typer.echo(f"access_ledger_hash: {result.access_ledger_hash}")
    typer.echo(f"checkpoint_metadata_hash: {result.checkpoint_metadata_hash}")
    typer.echo(f"summary_hash: {result.summary_hash}")
    for relative_name, digest in sorted(result.artifact_hashes.items()):
        typer.echo(f"artifact: {relative_name} sha256={digest}")


@app.command("plan-phase8-definitive-development-training")
def plan_phase8_definitive_development_training_command(
    input_binding_hash: Annotated[
        str,
        typer.Option(
            "--input-binding-hash",
            help=(
                "SHA-256 of an already-verified Phase8RealDevelopmentInputBinding "
                "(Substage 2). No manifest or split file is opened by this command."
            ),
        ),
    ] = ...,  # type: ignore[assignment]
    candidate_inventory_path: Annotated[
        Path,
        typer.Option(
            "--candidate-inventory-path",
            help="Explicit absolute path to a Phase 8 fixed candidate inventory JSON file.",
        ),
    ] = ...,  # type: ignore[assignment]
    candidate_id: Annotated[
        str,
        typer.Option(
            "--candidate-id",
            help="Candidate identifier; must equal 'monai_segresnet_baseline'.",
        ),
    ] = ...,  # type: ignore[assignment]
    preprocessing_decision_path: Annotated[
        Path,
        typer.Option(
            "--preprocessing-decision-path",
            help="Explicit absolute path to a Phase 8 preprocessing decision JSON file.",
        ),
    ] = ...,  # type: ignore[assignment]
    release_scope_description: Annotated[
        str,
        typer.Option(
            "--release-scope-description",
            help="Conservative identifier describing what a future release would authorize.",
        ),
    ] = ...,  # type: ignore[assignment]
    authorized_max_training_steps: Annotated[
        int,
        typer.Option(
            "--authorized-max-training-steps",
            help="Explicit bounded training-step count a future release would authorize.",
        ),
    ] = ...,  # type: ignore[assignment]
    output_root: Annotated[
        Path,
        typer.Option(
            "--output-root",
            help="Explicit absolute definitive-training plan output root outside the repository.",
        ),
    ] = ...,  # type: ignore[assignment]
    repository_root: Annotated[
        Path | None,
        typer.Option(
            "--repository-root",
            help="Explicit absolute repository root used only for output-root rejection.",
        ),
    ] = None,
) -> None:
    """Plan (never execute) a future definitive MONAI SegResNet training run.

    This command is planning-only: it never opens an image, label,
    prediction, or checkpoint file, and it never trains, infers, or computes
    a real metric. It publishes a ``Phase8DefinitiveTrainingConfig`` whose
    ``execution_release_state`` is always ``"awaiting_explicit_user_approval"``
    alongside a ``Phase8DefinitiveExecutionRelease`` whose ``release_state``
    is always ``"not_released"``. No release is issued by this command, and
    there is no flag capable of changing either fixed state.
    """

    try:
        candidate_inventory = phase8_fixed_candidate_inventory_from_mapping(
            _read_json_mapping_strict(candidate_inventory_path)
        )
        preprocessing_decision = phase8_preprocessing_decision_from_mapping(
            _read_json_mapping_strict(preprocessing_decision_path)
        )
        candidate_by_id = {
            candidate.candidate_id: candidate for candidate in candidate_inventory.candidates
        }
        candidate = candidate_by_id.get(candidate_id)
        if candidate is None:
            raise Phase8DefinitiveTrainingError(
                f"candidate_id {candidate_id!r} is not present in the fixed candidate inventory."
            )
        if preprocessing_decision.candidate_id != candidate_id:
            raise Phase8DefinitiveTrainingError(
                "preprocessing_decision.candidate_id does not match --candidate-id."
            )
        preprocessing_decision_reference = ArtifactReference(
            schema_name=preprocessing_decision.schema_name,
            schema_version=preprocessing_decision.schema_version,
            artifact_hash=preprocessing_decision.preprocessing_decision_hash,
            artifact_role="preprocessing_decision",
        )
        config = build_phase8_definitive_training_config(
            input_binding_hash=input_binding_hash,
            fixed_candidate_inventory_hash=candidate_inventory.inventory_hash,
            preprocessing_decision_reference=preprocessing_decision_reference,
            training_config_reference=candidate.training_config_reference,
            fixed_seeds=candidate.fixed_seeds,
        )
        release = build_unreleased_definitive_execution_release(
            bound_config_hash=config.config_hash,
            release_scope_description=release_scope_description,
            authorized_max_training_steps=authorized_max_training_steps,
        )
        result = publish_phase8_definitive_training_plan(
            config=config,
            release=release,
            output_root=output_root,
            repository_root=repository_root or Path.cwd(),
        )
    except (Phase8DefinitiveTrainingError, ValueError, OSError) as exc:
        _raise_phase8_definitive_training_plan_cli_error(exc)

    typer.echo("Phase 8 definitive-training plan publication complete")
    typer.echo("execution_release_state: awaiting_explicit_user_approval")
    typer.echo("release_state: not_released")
    typer.echo("training_executed: false")
    typer.echo("checkpoint_created: false")
    typer.echo("real_metrics_computed: false")
    typer.echo(f"config_hash: {result.config_hash}")
    typer.echo(f"release_hash: {result.release_hash}")
    for relative_name, digest in sorted(result.artifact_hashes.items()):
        typer.echo(f"artifact: {relative_name} sha256={digest}")


@app.command("run-phase8-definitive-development-training")
def run_phase8_definitive_development_training_command(
    config_path: Annotated[
        Path,
        typer.Option(
            "--config-path",
            help="Explicit absolute path to a Phase8DefinitiveTrainingConfig JSON file.",
        ),
    ] = ...,  # type: ignore[assignment]
    release_path: Annotated[
        Path,
        typer.Option(
            "--release-path",
            help="Explicit absolute path to a Phase8DefinitiveExecutionRelease JSON file.",
        ),
    ] = ...,  # type: ignore[assignment]
) -> None:
    """Run (or, in practice, always refuse to run) definitive development training.

    Real execution requires an explicit ``Phase8DefinitiveExecutionRelease``
    artifact whose ``release_state`` is ``"released"`` and whose
    ``bound_config_hash`` matches the supplied config's ``config_hash``. No
    command or function in this codebase's own tooling can produce such a
    release; it can only be authorized by a separately released process
    outside this session. This command therefore refuses to proceed --
    before opening any manifest, split, or medical file -- whenever the
    release is absent, invalid, mismatched, or ``not_released``, which is
    every real invocation in this environment today.
    """

    try:
        config = phase8_definitive_training_config_from_mapping(
            _read_json_mapping_strict(config_path)
        )
        release = phase8_definitive_execution_release_from_mapping(
            _read_json_mapping_strict(release_path)
        )
        execute_phase8_definitive_training(config, release)
    except (Phase8DefinitiveTrainingError, ValueError, OSError) as exc:
        _raise_phase8_definitive_training_run_cli_error(exc)

    typer.echo("Phase 8 definitive-training run complete")


def run_and_publish_phase7_robustness_uncertainty(
    *,
    settings: Phase7SyntheticSettings,
    output_root: Path,
) -> Phase7SyntheticPublicationResult:
    """Build a bounded synthetic Phase 7 run and publish validated artifacts."""

    inputs, summary = _build_phase7_synthetic_publication_inputs(settings)
    published = publish_phase7_artifacts(output_root=output_root, inputs=inputs)
    return Phase7SyntheticPublicationResult(
        output_root=published.output_root,
        config_hash=cast(str, summary["config_hash"]),
        corruption_manifest_hash=cast(str, summary["corruption_manifest_hash"]),
        uncertainty_type=cast(str, summary["uncertainty_type"]),
        calibration_ece=cast(float | None, summary["calibration_ece"]),
        risk_coverage_point_count=cast(int, summary["risk_coverage_point_count"]),
        transform_count=cast(int, summary["transform_count"]),
        geometry_record_count=cast(int, summary["geometry_record_count"]),
        run_summary_path=published.output_root / PHASE7_RUN_SUMMARY_NAME,
        reused_existing_output=published.reused_existing_output,
    )


def _build_phase7_synthetic_publication_inputs(
    settings: Phase7SyntheticSettings,
) -> tuple[Phase7PublicationInputs, dict[str, JsonValue]]:
    shape = (4, 5, 6)
    image = np.linspace(-120.0, 140.0, num=np.prod(shape), dtype=np.float64).reshape(shape)
    reference_mask = np.ascontiguousarray((image > 35.0).astype(np.uint8))
    validate_binary_mask(reference_mask, expected_shape=shape)
    grid = build_spatial_grid(
        shape=shape,
        affine=np.diag([1.0, 1.0, 1.0, 1.0]),
        orientation=("R", "A", "S"),
    )
    foreground_probability = _synthetic_foreground_probability(image)
    baseline_prediction = np.ascontiguousarray(
        foreground_probability.reshape(shape) > 0.5,
        dtype=np.uint8,
    )

    config_json = canonical_json_bytes(settings.effective_config_mapping) + b"\n"
    config_hash = sha256_json(settings.effective_config_mapping)
    manifest = _build_phase7_corruption_manifest(
        settings=settings,
        config_hash=config_hash,
        input_image_hash=array_content_sha256(image),
        input_mask_hash=array_content_sha256(reference_mask),
    )
    transform_results: list[Phase7TransformResult] = []
    geometry_records: list[Phase7GeometryRecord] = []
    corrupted_prediction = baseline_prediction

    for spec in settings.corruption_specs:
        transform_result, geometry_record, transformed_image, transformed_mask = (
            _apply_phase7_synthetic_corruption(
                image=image,
                mask=reference_mask,
                grid=grid,
                specification=spec,
            )
        )
        transform_results.append(transform_result)
        if geometry_record is not None:
            geometry_records.append(geometry_record)
        if spec.corruption_name == "crop_fov":
            if transformed_mask is not None:
                validate_binary_mask(transformed_mask, expected_shape=shape)
            corrupted_prediction = np.ascontiguousarray(
                _synthetic_foreground_probability(transformed_image).reshape(shape) > 0.5,
                dtype=np.uint8,
            )

    if not geometry_records:
        raise Phase7CliConfigError("synthetic Phase 7 flow produced no geometry records.")
    common_grid_hash = geometry_records[-1].geometry_record_hash
    baseline_probability = _phase7_probability_map(
        foreground_probability,
        source_prediction_hash=array_content_sha256(baseline_prediction),
        common_grid_hash=common_grid_hash,
    )
    entropy = compute_predictive_entropy(baseline_probability)
    tta_result = _compute_phase7_synthetic_tta(
        baseline_probability,
        foreground_probability=foreground_probability,
        common_grid_hash=common_grid_hash,
        manifest_hash=manifest.corruption_manifest_hash,
    )
    calibration = compute_binary_calibration_ece(
        baseline_probability,
        reference_mask.reshape((1, 1, *shape)),
        bin_count=settings.calibration_bin_count,
        common_grid_geometry_record_hash=common_grid_hash,
    )
    risk = compute_risk_coverage(
        reference_mask=reference_mask.reshape((1, 1, *shape)),
        prediction_map=baseline_prediction.reshape((1, 1, *shape)),
        uncertainty_map=tta_result.foreground_variance_map,
        uncertainty_result_hash=tta_result.uncertainty_result.uncertainty_result_hash,
        common_grid_geometry_record_hash=common_grid_hash,
    )
    errors = np.ascontiguousarray(baseline_prediction != reference_mask, dtype=np.uint8)
    correlation = compute_uncertainty_error_correlation(
        uncertainty_values=tta_result.foreground_variance_map.reshape(-1),
        error_indicators=errors.reshape(-1),
    )
    auroc = compute_failure_detection_auroc(
        case_uncertainty_scores=np.asarray(
            [tta_result.mean_foreground_variance, float(np.mean(entropy.entropy_map))],
            dtype=np.float64,
        ),
        failure_indicators=np.asarray([0, 1], dtype=np.uint8),
    )
    baseline_dice = dice_from_counts(
        compute_binary_confusion_counts(
            reference_mask.astype(np.bool_),
            baseline_prediction.astype(np.bool_),
        )
    )
    corrupted_dice = dice_from_counts(
        compute_binary_confusion_counts(
            reference_mask.astype(np.bool_),
            corrupted_prediction.astype(np.bool_),
        )
    )
    degradation = build_phase7_degradation_result(
        baseline_artifact_hash=array_content_sha256(baseline_prediction),
        corrupted_artifact_hash=array_content_sha256(corrupted_prediction),
        metric_name="dice",
        metric_direction="higher_is_better",
        baseline_value=baseline_dice,
        corrupted_value=corrupted_dice,
    )
    subgroup = build_phase7_lesion_subgroup_result(
        (
            LesionSubgroupCase(
                case_id="synthetic_query_case",
                reference_mask=reference_mask,
                prediction_mask=baseline_prediction,
            ),
            LesionSubgroupCase(
                case_id="synthetic_empty_case",
                reference_mask=np.zeros(shape, dtype=np.uint8),
                prediction_mask=np.zeros(shape, dtype=np.uint8),
            ),
        ),
        metric_name="dice",
        common_grid_geometry_record_hash=common_grid_hash,
    )
    transform_results_json = build_phase7_transform_results_collection_json(
        tuple(transform_results)
    )
    geometry_records_json = build_phase7_geometry_records_collection_json(tuple(geometry_records))
    failure_detection_json = canonical_phase7_failure_detection_results_json(
        correlation_result=correlation,
        auroc_result=auroc,
    )
    uncertainty_result_json = phase7_uncertainty_result_to_json(tta_result.uncertainty_result)
    calibration_result_json = phase7_calibration_result_to_json(calibration.calibration_result)
    risk_coverage_result_json = phase7_risk_coverage_result_to_json(risk.risk_coverage_result)
    degradation_result_json = phase7_degradation_result_to_json(degradation)
    lesion_subgroup_result_json = phase7_lesion_subgroup_result_to_json(subgroup.result)
    manifest_json = phase7_corruption_manifest_to_json(manifest)
    transform_results_hash = _phase7_json_hash(transform_results_json)
    geometry_records_hash = _phase7_json_hash(geometry_records_json)
    failure_detection_hash = _phase7_json_hash(failure_detection_json)
    publication_payload_hash = _phase7_publication_payload_hash(
        config_hash=config_hash,
        manifest_hash=manifest.corruption_manifest_hash,
        transform_results_hash=transform_results_hash,
        geometry_records_hash=geometry_records_hash,
        uncertainty_hash=tta_result.uncertainty_result.uncertainty_result_hash,
        calibration_hash=calibration.calibration_result.calibration_result_hash,
        risk_hash=risk.risk_coverage_result.risk_coverage_result_hash,
        failure_detection_hash=failure_detection_hash,
        degradation_hash=degradation.degradation_result_hash,
        subgroup_hash=subgroup.result.lesion_subgroup_result_hash,
    )
    run_summary = _build_phase7_run_summary(
        config_hash=config_hash,
        manifest_hash=manifest.corruption_manifest_hash,
        transform_results_hash=transform_results_hash,
        geometry_records_hash=geometry_records_hash,
        phase6_run_summary_hash=sha256_json(
            {
                "schema_name": "phase6_synthetic_final_surface_reference",
                "schema_version": "v1",
                "prediction_hash": array_content_sha256(baseline_prediction),
                "probability_hash": baseline_probability.probability_map_identity_hash,
            }
        ),
        uncertainty_hash=tta_result.uncertainty_result.uncertainty_result_hash,
        calibration_hash=calibration.calibration_result.calibration_result_hash,
        risk_hash=risk.risk_coverage_result.risk_coverage_result_hash,
        failure_detection_hash=failure_detection_hash,
        degradation_hash=degradation.degradation_result_hash,
        subgroup_hash=subgroup.result.lesion_subgroup_result_hash,
        publication_payload_hash=publication_payload_hash,
    )
    inputs = Phase7PublicationInputs(
        effective_config_json=config_json,
        corruption_manifest_json=manifest_json,
        transform_results_json=transform_results_json,
        geometry_records_json=geometry_records_json,
        uncertainty_result_json=uncertainty_result_json,
        calibration_result_json=calibration_result_json,
        risk_coverage_result_json=risk_coverage_result_json,
        failure_detection_json=failure_detection_json,
        degradation_result_json=degradation_result_json,
        lesion_subgroup_result_json=lesion_subgroup_result_json,
        phase7_run_summary_json=phase7_run_summary_to_json(run_summary),
    )
    summary: dict[str, JsonValue] = {
        "calibration_ece": calibration.calibration_result.ece,
        "config_hash": config_hash,
        "corruption_manifest_hash": manifest.corruption_manifest_hash,
        "geometry_record_count": len(geometry_records),
        "risk_coverage_point_count": len(risk.risk_coverage_result.points),
        "transform_count": len(transform_results),
        "uncertainty_type": tta_result.uncertainty_result.uncertainty_type,
    }
    return inputs, summary


def _apply_phase7_synthetic_corruption(
    *,
    image: np.ndarray,
    mask: np.ndarray,
    grid: SpatialGrid,
    specification: Phase7CorruptionSpecification,
) -> tuple[Phase7TransformResult, Phase7GeometryRecord | None, np.ndarray, np.ndarray | None]:
    if specification.corruption_name in {
        "hu_window_shift",
        "intensity_scale",
        "intensity_offset",
        "contrast_shift",
    }:
        intensity_output = apply_intensity_transform(
            image,
            specification,
            mask=mask,
            spatial_grid=grid,
        )
        return (
            intensity_output.transform_result,
            intensity_output.geometry_record,
            intensity_output.image,
            intensity_output.mask,
        )
    if specification.corruption_name == "gaussian_noise":
        noise_output = apply_gaussian_noise(image, specification, mask=mask)
        return (
            noise_output.transform_result,
            None,
            noise_output.transformed_image,
            noise_output.transformed_mask,
        )
    if specification.corruption_name == "gaussian_blur":
        blur_output = apply_gaussian_blur(image, specification, mask=mask)
        return (
            blur_output.transform_result,
            None,
            blur_output.transformed_image,
            blur_output.transformed_mask,
        )
    if specification.corruption_name == "slice_thickness":
        slice_output = apply_slice_thickness_simulation(
            image=image,
            image_grid=grid,
            specification=specification,
            mask=mask,
            mask_grid=grid,
            target_common_grid=grid,
        )
        return (
            slice_output.transform_result,
            slice_output.geometry_record,
            slice_output.restored_image,
            slice_output.restored_mask,
        )
    if specification.corruption_name == "anisotropic_downsampling":
        anisotropic_output = apply_anisotropic_downsampling(
            image=image,
            image_grid=grid,
            specification=specification,
            mask=mask,
            mask_grid=grid,
            target_common_grid=grid,
        )
        return (
            anisotropic_output.transform_result,
            anisotropic_output.geometry_record,
            anisotropic_output.restored_image,
            anisotropic_output.restored_mask,
        )
    if specification.corruption_name == "crop_fov":
        crop_output = apply_crop_fov_perturbation(
            image=image,
            image_grid=grid,
            corruption_specification=specification,
            evaluation_mask=mask,
            mask_grid=grid,
        )
        return (
            crop_output.transform_result,
            crop_output.geometry_record,
            crop_output.restored_image,
            crop_output.restored_mask,
        )
    raise Phase7CliConfigError("unsupported Phase 7 corruption in synthetic flow.")


def _synthetic_foreground_probability(image: np.ndarray) -> np.ndarray:
    scaled = 1.0 / (1.0 + np.exp(-np.asarray(image, dtype=np.float64) / 45.0))
    return np.ascontiguousarray(scaled.reshape((1, 1, *image.shape)), dtype=np.float64)


def _phase7_probability_map(
    foreground_probability: np.ndarray,
    *,
    source_prediction_hash: str,
    common_grid_hash: str,
) -> BinaryPredictiveProbabilityMap:
    foreground = np.ascontiguousarray(foreground_probability, dtype=np.float64)
    background = np.ascontiguousarray(1.0 - foreground, dtype=np.float64)
    return build_binary_predictive_probability_map(
        foreground_probability_map=foreground,
        background_probability_map=background,
        source_prediction_identity_hash=source_prediction_hash,
        common_grid_identity_hash=common_grid_hash,
    )


def _compute_phase7_synthetic_tta(
    baseline_probability: BinaryPredictiveProbabilityMap,
    *,
    foreground_probability: np.ndarray,
    common_grid_hash: str,
    manifest_hash: str,
) -> TTAVarianceResult:
    shifted_foreground = np.ascontiguousarray(
        np.clip(foreground_probability + 0.025, 0.0, 1.0),
        dtype=np.float64,
    )
    sample_1 = baseline_probability
    sample_2 = _phase7_probability_map(
        shifted_foreground,
        source_prediction_hash=array_content_sha256(shifted_foreground > 0.5),
        common_grid_hash=common_grid_hash,
    )
    records = (
        build_tta_sample_record(
            sample_index=0,
            sample_id="synthetic_identity",
            probability_map_identity_hash=sample_1.probability_map_identity_hash,
            common_grid_identity_hash=common_grid_hash,
            transform_manifest_hash=manifest_hash,
        ),
        build_tta_sample_record(
            sample_index=1,
            sample_id="synthetic_shifted",
            probability_map_identity_hash=sample_2.probability_map_identity_hash,
            common_grid_identity_hash=common_grid_hash,
            transform_manifest_hash=manifest_hash,
        ),
    )
    manifest = build_tta_sample_manifest(records)
    samples = build_tta_probability_samples(
        sample_manifest=manifest,
        probability_maps=(sample_1, sample_2),
    )
    return compute_tta_probability_variance(samples)


def _build_phase7_corruption_manifest(
    *,
    settings: Phase7SyntheticSettings,
    config_hash: str,
    input_image_hash: str,
    input_mask_hash: str,
) -> Phase7CorruptionManifest:
    payload: dict[str, JsonValue] = {
        "config_hash": config_hash,
        "deterministic_seed": settings.deterministic_seed,
        "input_image_content_hash": input_image_hash,
        "input_mask_content_hash": input_mask_hash,
        "manifest_id": "phase7_synthetic_manifest",
        "schema_name": PHASE7_CORRUPTION_MANIFEST_SCHEMA_NAME,
        "schema_version": PHASE7_CORRUPTION_MANIFEST_SCHEMA_VERSION,
        "specifications": [
            phase7_corruption_specification_to_dict(spec) for spec in settings.corruption_specs
        ],
    }
    return Phase7CorruptionManifest(
        schema_name=PHASE7_CORRUPTION_MANIFEST_SCHEMA_NAME,
        schema_version=PHASE7_CORRUPTION_MANIFEST_SCHEMA_VERSION,
        corruption_manifest_hash=sha256_json(payload),
        manifest_id="phase7_synthetic_manifest",
        config_hash=config_hash,
        input_image_content_hash=input_image_hash,
        input_mask_content_hash=input_mask_hash,
        deterministic_seed=settings.deterministic_seed,
        specifications=settings.corruption_specs,
    )


def _build_phase7_run_summary(
    *,
    config_hash: str,
    manifest_hash: str,
    transform_results_hash: str,
    geometry_records_hash: str,
    phase6_run_summary_hash: str,
    uncertainty_hash: str,
    calibration_hash: str,
    risk_hash: str,
    failure_detection_hash: str,
    degradation_hash: str,
    subgroup_hash: str,
    publication_payload_hash: str,
) -> Phase7RunSummary:
    payload: dict[str, JsonValue] = {
        "calibration_result_hash": calibration_hash,
        "config_hash": config_hash,
        "corruption_manifest_hash": manifest_hash,
        "degradation_result_hash": degradation_hash,
        "execution_status": "completed",
        "failure_code": None,
        "failure_detection_result_hash": failure_detection_hash,
        "failure_message": None,
        "geometry_records_hash": geometry_records_hash,
        "lesion_subgroup_result_hash": subgroup_hash,
        "phase6_run_summary_hash": phase6_run_summary_hash,
        "publication_payload_hash": publication_payload_hash,
        "risk_coverage_result_hash": risk_hash,
        "schema_name": PHASE7_RUN_SUMMARY_SCHEMA_NAME,
        "schema_version": PHASE7_RUN_SUMMARY_SCHEMA_VERSION,
        "transform_results_hash": transform_results_hash,
        "uncertainty_result_hash": uncertainty_hash,
    }
    return Phase7RunSummary(
        schema_name=PHASE7_RUN_SUMMARY_SCHEMA_NAME,
        schema_version=PHASE7_RUN_SUMMARY_SCHEMA_VERSION,
        phase7_run_summary_hash=sha256_json(payload),
        config_hash=config_hash,
        corruption_manifest_hash=manifest_hash,
        transform_results_hash=transform_results_hash,
        geometry_records_hash=geometry_records_hash,
        phase6_run_summary_hash=phase6_run_summary_hash,
        uncertainty_result_hash=uncertainty_hash,
        calibration_result_hash=calibration_hash,
        risk_coverage_result_hash=risk_hash,
        failure_detection_result_hash=failure_detection_hash,
        degradation_result_hash=degradation_hash,
        lesion_subgroup_result_hash=subgroup_hash,
        publication_payload_hash=publication_payload_hash,
        execution_status="completed",
        failure_code=None,
        failure_message=None,
        duration_seconds=None,
        memory_availability_status="unavailable",
        peak_host_memory_bytes=None,
    )


def _phase7_json_hash(data: bytes | str) -> str:
    decoded = json.loads(data)
    if not isinstance(decoded, dict):
        raise Phase7CliConfigError("Phase 7 publication JSON root must be an object.")
    return sha256_json(decoded)


def _phase7_publication_payload_hash(
    *,
    config_hash: str,
    manifest_hash: str,
    transform_results_hash: str,
    geometry_records_hash: str,
    uncertainty_hash: str,
    calibration_hash: str,
    risk_hash: str,
    failure_detection_hash: str,
    degradation_hash: str,
    subgroup_hash: str,
) -> str:
    return sha256_json(
        {
            "config_hash": config_hash,
            "corruption_manifest_hash": manifest_hash,
            "degradation_result_hash": degradation_hash,
            "failure_detection_result_hash": failure_detection_hash,
            "geometry_records_hash": geometry_records_hash,
            "lesion_subgroup_result_hash": subgroup_hash,
            "risk_coverage_result_hash": risk_hash,
            "calibration_result_hash": calibration_hash,
            "schema_name": "phase7_publication_payload",
            "schema_version": "v1",
            "transform_results_hash": transform_results_hash,
            "uncertainty_result_hash": uncertainty_hash,
        }
    )


def _phase7_corruption_specification_from_config(
    value: object,
    *,
    default_seed: int,
    index: int,
) -> Phase7CorruptionSpecification:
    mapping = _expect_phase7_mapping(value, f"corruptions[{index}]")
    if set(mapping) != {
        "corruption_name",
        "severity",
        "parameters",
        "deterministic_seed",
        "changes_geometry",
        "image_interpolation",
        "mask_interpolation",
        "common_grid_restoration_required",
    }:
        raise Phase7CliConfigError("corruption config contains unexpected fields.")
    name = _expect_phase7_string(mapping["corruption_name"], "corruption_name")
    severity = _expect_phase7_string(mapping["severity"], "severity")
    parameters = _json_value_mapping(_expect_phase7_mapping(mapping["parameters"], "parameters"))
    if name == "gaussian_noise":
        expected_parameters = gaussian_noise_parameters_for_severity(severity)
        if parameters != expected_parameters:
            raise Phase7CliConfigError(
                "gaussian_noise parameters must match the explicit severity contract."
            )
    elif name == "gaussian_blur":
        expected_parameters = gaussian_blur_parameters_for_severity(severity)
        if parameters != expected_parameters:
            raise Phase7CliConfigError(
                "gaussian_blur parameters must match the explicit severity contract."
            )
    seed_value = mapping["deterministic_seed"]
    deterministic_seed = (
        default_seed + index
        if seed_value == "default_plus_index"
        else _expect_phase7_optional_int(seed_value, "deterministic_seed")
    )
    payload: dict[str, JsonValue] = {
        "changes_geometry": _expect_phase7_bool(mapping["changes_geometry"], "changes_geometry"),
        "common_grid_restoration_required": _expect_phase7_bool(
            mapping["common_grid_restoration_required"],
            "common_grid_restoration_required",
        ),
        "corruption_name": name,
        "deterministic_seed": deterministic_seed,
        "image_interpolation": _expect_phase7_string(
            mapping["image_interpolation"],
            "image_interpolation",
        ),
        "mask_interpolation": _expect_phase7_string(
            mapping["mask_interpolation"],
            "mask_interpolation",
        ),
        "parameters": parameters,
        "schema_name": PHASE7_CORRUPTION_SPECIFICATION_SCHEMA_NAME,
        "schema_version": PHASE7_CORRUPTION_SPECIFICATION_SCHEMA_VERSION,
        "severity": severity,
    }
    return Phase7CorruptionSpecification(
        schema_name=PHASE7_CORRUPTION_SPECIFICATION_SCHEMA_NAME,
        schema_version=PHASE7_CORRUPTION_SPECIFICATION_SCHEMA_VERSION,
        corruption_specification_hash=sha256_json(payload),
        corruption_name=name,
        severity=severity,
        parameters=parameters,
        deterministic_seed=deterministic_seed,
        changes_geometry=cast(bool, payload["changes_geometry"]),
        image_interpolation=cast(str, payload["image_interpolation"]),
        mask_interpolation=cast(str, payload["mask_interpolation"]),
        common_grid_restoration_required=cast(
            bool,
            payload["common_grid_restoration_required"],
        ),
    )


def _expect_phase7_mapping(value: object, field_name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise Phase7CliConfigError(f"{field_name} must be a mapping.")
    return cast(dict[str, object], value)


def _expect_phase7_sequence(value: object, field_name: str) -> list[object]:
    if not isinstance(value, list):
        raise Phase7CliConfigError(f"{field_name} must be a list.")
    return value


def _expect_phase7_string(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise Phase7CliConfigError(f"{field_name} must be a string.")
    return value


def _expect_phase7_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise Phase7CliConfigError(f"{field_name} must be boolean.")
    return value


def _expect_phase7_int(value: object, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise Phase7CliConfigError(f"{field_name} must be a nonnegative integer.")
    return value


def _expect_phase7_optional_int(value: object, field_name: str) -> int | None:
    if value is None:
        return None
    return _expect_phase7_int(value, field_name)


def _json_value_mapping(mapping: dict[str, object]) -> dict[str, JsonValue]:
    return cast(dict[str, JsonValue], json.loads(canonical_json_bytes(mapping)))


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

    from protoem_ct.tracking import LocalMlflowTrackingError, track_synthetic_run

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


@app.command("run-phase3-gate3-synthetic")
def run_phase3_gate3_synthetic_command(
    output_root: Annotated[
        Path,
        typer.Option(
            "--output-root",
            help="Absolute non-existing external output root for the synthetic Gate 3 run.",
        ),
    ] = ...,  # type: ignore[assignment]
    git_commit: Annotated[
        str,
        typer.Option(
            "--git-commit",
            help="Explicit Git commit recorded in the synthetic Gate 3 artifacts.",
        ),
    ] = ...,  # type: ignore[assignment]
    start_timestamp: Annotated[
        str,
        typer.Option(
            "--start-timestamp",
            help="Explicit UTC start timestamp recorded in the synthetic Gate 3 artifacts.",
        ),
    ] = ...,  # type: ignore[assignment]
    end_timestamp: Annotated[
        str,
        typer.Option(
            "--end-timestamp",
            help="Explicit UTC end timestamp recorded in the synthetic Gate 3 artifacts.",
        ),
    ] = ...,  # type: ignore[assignment]
) -> None:
    """Run the bounded synthetic Phase 3 Gate 3 orchestration outside the repository."""

    from protoem_ct.baselines.gate3 import (
        Phase3Gate3Error,
        run_phase3_gate3_synthetic,
    )

    try:
        result = run_phase3_gate3_synthetic(
            output_root=output_root,
            git_commit=git_commit,
            run_identifier="phase3_gate3_synthetic",
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            baseline_environment_lock_path=Path(
                "environments/phase3-baselines/intel-macos-cpu/uv.lock"
            ).resolve(strict=True),
        )
    except (Phase3Gate3Error, OSError) as exc:
        _raise_phase3_gate3_cli_error(exc)

    typer.echo("phase3 gate3 synthetic run success")
    typer.echo(f"report version: {PHASE3_GATE3_REPORT_VERSION}")
    typer.echo("selected device: cpu")
    typer.echo("amp enabled: false")
    typer.echo(f"fixture hash: {result.fixture_artifact_sha256}")
    typer.echo(f"nnunet checkpoint hash: {result.nnunet_result.checkpoint_sha256}")
    typer.echo(f"monai checkpoint hash: {result.monai_result.checkpoint_sha256}")
    typer.echo(f"gate3 report hash: {result.report.artifact_hash}")


if __name__ == "__main__":
    app()
