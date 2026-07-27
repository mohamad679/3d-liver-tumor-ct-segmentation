"""Synthetic-only Phase 2 end-to-end DAG.

This workflow orchestrates committed public CLI commands. It generates only synthetic fixtures and
does not authorize real LiTS, MSD Task03 Liver, DICOM, or 3D-IRCADb access.
"""

from pathlib import Path

from snakemake.exceptions import WorkflowError
import generate_phase2_synthetic_fixture as helpers

PHASE2_REPO_ROOT = Path(workflow.basedir).resolve()

configfile: str(PHASE2_REPO_ROOT / "configs" / "phase2_synthetic.yaml")

PHASE2_SYNTHETIC = config
PHASE2_GENERATED_ROOT_RAW = PHASE2_SYNTHETIC.get(
    "phase2_synthetic_generated_root",
    ".artifacts/phase2_synthetic",
)
PHASE2_GENERATED_ROOT_PATH = Path(PHASE2_GENERATED_ROOT_RAW)
if not PHASE2_GENERATED_ROOT_PATH.is_absolute():
    PHASE2_GENERATED_ROOT_PATH = PHASE2_REPO_ROOT / PHASE2_GENERATED_ROOT_PATH
if PHASE2_GENERATED_ROOT_PATH.exists() and not PHASE2_GENERATED_ROOT_PATH.is_dir():
    raise WorkflowError("phase2_synthetic_generated_root exists and is not a directory")
PHASE2_GENERATED_ROOT = PHASE2_GENERATED_ROOT_PATH.resolve(strict=False)

PHASE2_DATASET_ROOT = helpers.phase2_child_path(PHASE2_GENERATED_ROOT, "dataset")
PHASE2_CONTROL_ROOT = helpers.phase2_child_path(PHASE2_GENERATED_ROOT, "control")
PHASE2_ARTIFACT_ROOT = helpers.phase2_child_path(PHASE2_GENERATED_ROOT, "artifacts")
PHASE2_LOG_ROOT = helpers.phase2_child_path(PHASE2_GENERATED_ROOT, "logs")

PHASE2_LAYOUT = helpers.phase2_config(config, "layout")
PHASE2_SPLIT = helpers.phase2_config(config, "split")
PHASE2_ANON = helpers.phase2_config(config, "anonymous_ids")
PHASE2_GEOMETRY = helpers.phase2_config(config, "geometry_qa")
PHASE2_LESIONS = helpers.phase2_config(config, "lesions")
PHASE2_HISTOGRAM = helpers.phase2_config(config, "histogram")
PHASE2_REPORT = helpers.phase2_config(config, "report")
PHASE2_AUDIT = helpers.phase2_config(config, "audit")

PHASE2_FIXTURE_MARKER = PHASE2_CONTROL_ROOT / "fixture_complete.txt"
PHASE2_KEY_FILE = PHASE2_CONTROL_ROOT / "phase2_synthetic_hmac.key"
PHASE2_MANIFEST = PHASE2_ARTIFACT_ROOT / "dataset_manifest.json"
PHASE2_SPLIT_ARTIFACT = PHASE2_ARTIFACT_ROOT / "development_split.json"
PHASE2_GEOMETRY_QA = PHASE2_ARTIFACT_ROOT / "geometry_label_qa.json"
PHASE2_LESIONS_ARTIFACT = PHASE2_ARTIFACT_ROOT / "lesion_components.json"
PHASE2_SUMMARY = PHASE2_ARTIFACT_ROOT / "development_data_summary.json"
PHASE2_FINAL_QA = PHASE2_ARTIFACT_ROOT / "development_qa_report.json"
PHASE2_LEAKAGE_AUDIT = PHASE2_ARTIFACT_ROOT / "leakage_audit.json"
PHASE2_FIXTURE_LOG = PHASE2_LOG_ROOT / "generate_fixture.log"
PHASE2_INVENTORY_LOG = PHASE2_LOG_ROOT / "inventory_lits.log"
PHASE2_MANIFEST_LOG = PHASE2_LOG_ROOT / "build_lits_manifest.log"
PHASE2_SPLIT_LOG = PHASE2_LOG_ROOT / "build_development_split.log"
PHASE2_GEOMETRY_LOG = PHASE2_LOG_ROOT / "qa_geometry_labels.log"
PHASE2_LESION_LOG = PHASE2_LOG_ROOT / "summarize_lesions.log"
PHASE2_SUMMARY_LOG = PHASE2_LOG_ROOT / "summarize_development_data.log"
PHASE2_REPORT_LOG = PHASE2_LOG_ROOT / "build_development_qa_report.log"
PHASE2_AUDIT_LOG = PHASE2_LOG_ROOT / "audit_development_leakage.log"


rule phase2_synthetic_all:
    input:
        inventory=str(PHASE2_INVENTORY_LOG),
        manifest=str(PHASE2_MANIFEST),
        split=str(PHASE2_SPLIT_ARTIFACT),
        geometry=str(PHASE2_GEOMETRY_QA),
        lesion=str(PHASE2_LESIONS_ARTIFACT),
        summary=str(PHASE2_SUMMARY),
        report=str(PHASE2_FINAL_QA),
        audit=str(PHASE2_LEAKAGE_AUDIT),


rule phase2_generate_synthetic_fixture:
    output:
        marker=str(PHASE2_FIXTURE_MARKER),
    log:
        str(PHASE2_FIXTURE_LOG),
    params:
        script=str(PHASE2_REPO_ROOT / "workflow/scripts/generate_phase2_synthetic_fixture.py"),
        dataset_root=str(PHASE2_DATASET_ROOT),
        case_count=helpers.phase2_config(config, "case_count"),
        git_commit=lambda wildcards: helpers.phase2_nonempty_config(
            config,
            "phase2_synthetic_git_commit",
        ),
    run:
        helpers.phase2_generate_fixture(output, log, params)


rule phase2_inventory_lits:
    input:
        marker=rules.phase2_generate_synthetic_fixture.output.marker,
    output:
        log=str(PHASE2_INVENTORY_LOG),
    log:
        workflow=str(PHASE2_LOG_ROOT / "inventory_lits.workflow.log"),
    params:
        dataset_root=str(PHASE2_DATASET_ROOT),
        image_pattern=PHASE2_LAYOUT["image_pattern"],
        label_pattern=PHASE2_LAYOUT["label_pattern"],
        recursive=helpers.phase2_bool_text(PHASE2_LAYOUT["recursive"]),
        image_prefix=PHASE2_LAYOUT["image_prefix"],
        label_prefix=PHASE2_LAYOUT["label_prefix"],
    run:
        helpers.phase2_inventory_lits(output, params)


rule phase2_build_lits_manifest:
    input:
        marker=rules.phase2_generate_synthetic_fixture.output.marker,
        inventory=rules.phase2_inventory_lits.output.log,
    output:
        artifact=str(PHASE2_MANIFEST),
        log=str(PHASE2_MANIFEST_LOG),
    log:
        workflow=str(PHASE2_LOG_ROOT / "build_lits_manifest.workflow.log"),
    params:
        dataset_root=str(PHASE2_DATASET_ROOT),
        image_pattern=PHASE2_LAYOUT["image_pattern"],
        label_pattern=PHASE2_LAYOUT["label_pattern"],
        recursive=helpers.phase2_bool_text(PHASE2_LAYOUT["recursive"]),
        image_prefix=PHASE2_LAYOUT["image_prefix"],
        label_prefix=PHASE2_LAYOUT["label_prefix"],
        dataset_id=helpers.phase2_config(config, "dataset_id"),
        project_namespace=helpers.phase2_config(config, "project_namespace"),
        patient_id_prefix=PHASE2_ANON["patient_id_prefix"],
        case_id_prefix=PHASE2_ANON["case_id_prefix"],
        digest_length=PHASE2_ANON["digest_length"],
        key_file=str(PHASE2_KEY_FILE),
        git_commit=lambda wildcards: helpers.phase2_nonempty_config(
            config,
            "phase2_synthetic_git_commit",
        ),
        created_at_utc=helpers.phase2_config(config, "phase2_synthetic_created_at_utc"),
    run:
        helpers.phase2_build_lits_manifest(output, params)


rule phase2_build_development_split:
    input:
        manifest=rules.phase2_build_lits_manifest.output.artifact,
    output:
        artifact=str(PHASE2_SPLIT_ARTIFACT),
        log=str(PHASE2_SPLIT_LOG),
    log:
        workflow=str(PHASE2_LOG_ROOT / "build_development_split.workflow.log"),
    params:
        policy_version=PHASE2_SPLIT["policy_version"],
        split_seed=PHASE2_SPLIT["seed"],
        train_count=PHASE2_SPLIT["train_patient_count"],
        validation_count=PHASE2_SPLIT["validation_patient_count"],
        internal_test_count=PHASE2_SPLIT["internal_test_patient_count"],
        git_commit=lambda wildcards: helpers.phase2_nonempty_config(
            config,
            "phase2_synthetic_git_commit",
        ),
        created_at_utc=helpers.phase2_config(config, "phase2_synthetic_created_at_utc"),
    run:
        helpers.phase2_build_development_split(input, output, params)


rule phase2_geometry_label_qa:
    input:
        manifest=rules.phase2_build_lits_manifest.output.artifact,
        split=rules.phase2_build_development_split.output.artifact,
    output:
        artifact=str(PHASE2_GEOMETRY_QA),
        log=str(PHASE2_GEOMETRY_LOG),
    log:
        workflow=str(PHASE2_LOG_ROOT / "qa_geometry_labels.workflow.log"),
    params:
        dataset_root=str(PHASE2_DATASET_ROOT),
        allowed_labels=PHASE2_GEOMETRY["allowed_label_values"],
        tumor_label=PHASE2_GEOMETRY["tumor_label_value"],
        affine_tolerance=PHASE2_GEOMETRY["affine_tolerance"],
        git_commit=lambda wildcards: helpers.phase2_nonempty_config(
            config,
            "phase2_synthetic_git_commit",
        ),
        created_at_utc=helpers.phase2_config(config, "phase2_synthetic_created_at_utc"),
    run:
        helpers.phase2_geometry_label_qa(input, output, params)


rule phase2_summarize_lesions:
    input:
        manifest=rules.phase2_build_lits_manifest.output.artifact,
        geometry=rules.phase2_geometry_label_qa.output.artifact,
    output:
        artifact=str(PHASE2_LESIONS_ARTIFACT),
        log=str(PHASE2_LESION_LOG),
    log:
        workflow=str(PHASE2_LOG_ROOT / "summarize_lesions.workflow.log"),
    params:
        dataset_root=str(PHASE2_DATASET_ROOT),
        connectivity=PHASE2_LESIONS["connectivity"],
        tumor_label=PHASE2_LESIONS["tumor_label_value"],
        git_commit=lambda wildcards: helpers.phase2_nonempty_config(
            config,
            "phase2_synthetic_git_commit",
        ),
        created_at_utc=helpers.phase2_config(config, "phase2_synthetic_created_at_utc"),
    run:
        helpers.phase2_summarize_lesions(input, output, params)


rule phase2_summarize_development_data:
    input:
        manifest=rules.phase2_build_lits_manifest.output.artifact,
        geometry=rules.phase2_geometry_label_qa.output.artifact,
        lesion=rules.phase2_summarize_lesions.output.artifact,
    output:
        artifact=str(PHASE2_SUMMARY),
        log=str(PHASE2_SUMMARY_LOG),
    log:
        workflow=str(PHASE2_LOG_ROOT / "summarize_development_data.workflow.log"),
    params:
        dataset_root=str(PHASE2_DATASET_ROOT),
        histogram_min=PHASE2_HISTOGRAM["minimum"],
        histogram_max=PHASE2_HISTOGRAM["maximum"],
        histogram_bin_count=PHASE2_HISTOGRAM["bin_count"],
        git_commit=lambda wildcards: helpers.phase2_nonempty_config(
            config,
            "phase2_synthetic_git_commit",
        ),
        created_at_utc=helpers.phase2_config(config, "phase2_synthetic_created_at_utc"),
    run:
        helpers.phase2_summarize_development_data(input, output, params)


rule phase2_build_development_qa_report:
    input:
        manifest=rules.phase2_build_lits_manifest.output.artifact,
        split=rules.phase2_build_development_split.output.artifact,
        geometry=rules.phase2_geometry_label_qa.output.artifact,
        lesion=rules.phase2_summarize_lesions.output.artifact,
        summary=rules.phase2_summarize_development_data.output.artifact,
    output:
        artifact=str(PHASE2_FINAL_QA),
        log=str(PHASE2_REPORT_LOG),
    log:
        workflow=str(PHASE2_LOG_ROOT / "build_development_qa_report.workflow.log"),
    params:
        contract_version=PHASE2_REPORT["contract_version"],
        git_commit=lambda wildcards: helpers.phase2_nonempty_config(
            config,
            "phase2_synthetic_git_commit",
        ),
        created_at_utc=helpers.phase2_config(config, "phase2_synthetic_created_at_utc"),
    run:
        helpers.phase2_build_development_qa_report(input, output, params)


rule phase2_audit_development_leakage:
    input:
        manifest=rules.phase2_build_lits_manifest.output.artifact,
        split=rules.phase2_build_development_split.output.artifact,
        report=rules.phase2_build_development_qa_report.output.artifact,
    output:
        artifact=str(PHASE2_LEAKAGE_AUDIT),
        log=str(PHASE2_AUDIT_LOG),
    log:
        workflow=str(PHASE2_LOG_ROOT / "audit_development_leakage.workflow.log"),
    params:
        contract_version=PHASE2_AUDIT["contract_version"],
        git_commit=lambda wildcards: helpers.phase2_nonempty_config(
            config,
            "phase2_synthetic_git_commit",
        ),
        created_at_utc=helpers.phase2_config(config, "phase2_synthetic_created_at_utc"),
    run:
        helpers.phase2_audit_development_leakage(input, output, params)
