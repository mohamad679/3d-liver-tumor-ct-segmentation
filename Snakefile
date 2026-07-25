"""Phase 1 synthetic pipeline DAG.

This workflow only orchestrates the existing Phase 1 CLI stages. It requires
explicit run metadata and an absolute generated root supplied through
Snakemake --config.
"""

from pathlib import Path
import shutil

from snakemake.exceptions import WorkflowError


def _required_config_value(name):
    """Return a nonempty Snakemake config value or raise a workflow error."""
    value = config.get(name)
    if not isinstance(value, str) or not value.strip():
        raise WorkflowError(f"Missing required Snakemake --config value: {name}")
    return value


def _validate_child_path(root, path):
    """Require path to remain beneath root after lexical path construction."""
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise WorkflowError(f"Workflow output escapes generated_root: {path}") from exc
    return path


REPO_ROOT = Path(workflow.basedir).resolve()
DATA_CONFIG = REPO_ROOT / "configs" / "data" / "synthetic.yaml"
EXPERIMENT_CONFIG = REPO_ROOT / "configs" / "experiment" / "phase1.yaml"

GIT_COMMIT = _required_config_value("git_commit")
CREATED_AT_UTC = _required_config_value("created_at_utc")
GENERATED_ROOT_RAW = _required_config_value("generated_root")
GENERATED_ROOT_PATH = Path(GENERATED_ROOT_RAW)

if not GENERATED_ROOT_PATH.is_absolute():
    raise WorkflowError("generated_root must be an absolute path")
if GENERATED_ROOT_PATH.exists() and not GENERATED_ROOT_PATH.is_dir():
    raise WorkflowError("generated_root exists and is not a directory")

GENERATED_ROOT = GENERATED_ROOT_PATH.resolve(strict=False)
PROTOEM_CT = shutil.which("protoem-ct")
if PROTOEM_CT is None:
    raise WorkflowError("protoem-ct console script was not found on PATH")

DATA_ROOT = _validate_child_path(GENERATED_ROOT, GENERATED_ROOT / "data")
ARTIFACT_ROOT = _validate_child_path(GENERATED_ROOT, GENERATED_ROOT / "artifacts")
PREPROCESSED_ROOT = _validate_child_path(GENERATED_ROOT, GENERATED_ROOT / "preprocessed")
PREDICTION_ROOT = _validate_child_path(GENERATED_ROOT, GENERATED_ROOT / "predictions")
REPORT_ROOT = _validate_child_path(GENERATED_ROOT, GENERATED_ROOT / "report")

MANIFEST = DATA_ROOT / "generated" / "phase1" / "data" / "synthetic_manifest.json"
VALIDATION_ARTIFACT = ARTIFACT_ROOT / "validation.json"
PREPROCESS_ARTIFACT = ARTIFACT_ROOT / "preprocess.json"
INFERENCE_ARTIFACT = ARTIFACT_ROOT / "inference.json"
EVALUATION_ARTIFACT = ARTIFACT_ROOT / "evaluation.json"
REPORT_MARKDOWN = REPORT_ROOT / "phase1_report.md"
REPORT_ARTIFACT = REPORT_ROOT / "report_artifact.json"
LOG_ROOT = _validate_child_path(GENERATED_ROOT, GENERATED_ROOT / "logs")


rule all:
    input:
        report_markdown=str(REPORT_MARKDOWN),
        report_artifact=str(REPORT_ARTIFACT),


rule create_data:
    output:
        data_root=directory(str(DATA_ROOT)),
    log:
        str(LOG_ROOT / "create_data.log"),
    params:
        protoem_ct=PROTOEM_CT,
        config=str(DATA_CONFIG),
        git_commit=GIT_COMMIT,
        created_at_utc=CREATED_AT_UTC,
    run:
        shell(
            """
            {params.protoem_ct:q} create-data \
                --config {params.config:q} \
                --output-root {output.data_root:q} \
                --git-commit {params.git_commit:q} \
                --created-at-utc {params.created_at_utc:q}
            """
        )


rule validate:
    input:
        data_root=rules.create_data.output.data_root,
    output:
        artifact=str(VALIDATION_ARTIFACT),
    log:
        str(LOG_ROOT / "validate.log"),
    params:
        protoem_ct=PROTOEM_CT,
        manifest=str(MANIFEST),
        git_commit=GIT_COMMIT,
        created_at_utc=CREATED_AT_UTC,
    run:
        shell(
            """
            {params.protoem_ct:q} validate-data \
                --manifest {params.manifest:q} \
                --data-root {input.data_root:q} \
                --output {output.artifact:q} \
                --git-commit {params.git_commit:q} \
                --created-at-utc {params.created_at_utc:q}
            """
        )


rule preprocess:
    input:
        data_root=rules.create_data.output.data_root,
        validation_artifact=rules.validate.output.artifact,
    output:
        preprocessed_root=directory(str(PREPROCESSED_ROOT)),
        artifact=str(PREPROCESS_ARTIFACT),
    log:
        str(LOG_ROOT / "preprocess.log"),
    params:
        protoem_ct=PROTOEM_CT,
        manifest=str(MANIFEST),
        config=str(EXPERIMENT_CONFIG),
        git_commit=GIT_COMMIT,
        created_at_utc=CREATED_AT_UTC,
    run:
        shell(
            """
            {params.protoem_ct:q} preprocess-data \
                --manifest {params.manifest:q} \
                --validation-artifact {input.validation_artifact:q} \
                --data-root {input.data_root:q} \
                --output-root {output.preprocessed_root:q} \
                --artifact-output {output.artifact:q} \
                --config {params.config:q} \
                --git-commit {params.git_commit:q} \
                --created-at-utc {params.created_at_utc:q}
            """
        )


rule infer_dummy:
    input:
        preprocessed_root=rules.preprocess.output.preprocessed_root,
        preprocess_artifact=rules.preprocess.output.artifact,
    output:
        prediction_root=directory(str(PREDICTION_ROOT)),
        artifact=str(INFERENCE_ARTIFACT),
    log:
        str(LOG_ROOT / "infer_dummy.log"),
    params:
        protoem_ct=PROTOEM_CT,
        config=str(EXPERIMENT_CONFIG),
        git_commit=GIT_COMMIT,
        created_at_utc=CREATED_AT_UTC,
    run:
        shell(
            """
            {params.protoem_ct:q} infer-dummy \
                --preprocess-artifact {input.preprocess_artifact:q} \
                --preprocessed-root {input.preprocessed_root:q} \
                --output-root {output.prediction_root:q} \
                --artifact-output {output.artifact:q} \
                --config {params.config:q} \
                --git-commit {params.git_commit:q} \
                --created-at-utc {params.created_at_utc:q}
            """
        )


rule evaluate:
    input:
        preprocessed_root=rules.preprocess.output.preprocessed_root,
        prediction_root=rules.infer_dummy.output.prediction_root,
        preprocess_artifact=rules.preprocess.output.artifact,
        inference_artifact=rules.infer_dummy.output.artifact,
    output:
        artifact=str(EVALUATION_ARTIFACT),
    log:
        str(LOG_ROOT / "evaluate.log"),
    params:
        protoem_ct=PROTOEM_CT,
        git_commit=GIT_COMMIT,
        created_at_utc=CREATED_AT_UTC,
    run:
        shell(
            """
            {params.protoem_ct:q} evaluate \
                --preprocess-artifact {input.preprocess_artifact:q} \
                --inference-artifact {input.inference_artifact:q} \
                --preprocessed-root {input.preprocessed_root:q} \
                --prediction-root {input.prediction_root:q} \
                --artifact-output {output.artifact:q} \
                --git-commit {params.git_commit:q} \
                --created-at-utc {params.created_at_utc:q}
            """
        )


rule report:
    input:
        data_root=rules.create_data.output.data_root,
        validation_artifact=rules.validate.output.artifact,
        preprocess_artifact=rules.preprocess.output.artifact,
        inference_artifact=rules.infer_dummy.output.artifact,
        evaluation_artifact=rules.evaluate.output.artifact,
    output:
        report_markdown=str(REPORT_MARKDOWN),
        report_artifact=str(REPORT_ARTIFACT),
    log:
        str(LOG_ROOT / "report.log"),
    params:
        protoem_ct=PROTOEM_CT,
        manifest=str(MANIFEST),
        git_commit=GIT_COMMIT,
        created_at_utc=CREATED_AT_UTC,
    run:
        shell(
            """
            {params.protoem_ct:q} report \
                --manifest {params.manifest:q} \
                --validation-artifact {input.validation_artifact:q} \
                --preprocess-artifact {input.preprocess_artifact:q} \
                --inference-artifact {input.inference_artifact:q} \
                --evaluation-artifact {input.evaluation_artifact:q} \
                --report-output {output.report_markdown:q} \
                --artifact-output {output.report_artifact:q} \
                --git-commit {params.git_commit:q} \
                --created-at-utc {params.created_at_utc:q}
            """
        )
