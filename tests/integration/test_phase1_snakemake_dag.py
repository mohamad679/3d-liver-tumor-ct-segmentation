"""End-to-end integration tests for the Phase 1 Snakemake DAG."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from protoem_ct.artifacts import (
    EvaluationArtifact,
    InferenceArtifact,
    PreprocessArtifact,
    ReportArtifact,
    SyntheticManifest,
    ValidationArtifact,
    artifact_from_json,
    sha256_file,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SNAKEFILE = REPO_ROOT / "Snakefile"
TEST_GIT_COMMIT = "c5c54e8"
TEST_CREATED_AT_UTC = "2026-01-01T00:00:00Z"
RULE_NAMES = ("all", "create_data", "validate", "preprocess", "infer_dummy", "evaluate", "report")
DETERMINISTIC_JSON_NAMES = (
    "synthetic_manifest.json",
    "validation.json",
    "preprocess.json",
    "inference.json",
    "evaluation.json",
    "report_artifact.json",
)


@dataclass(frozen=True)
class DagPaths:
    """Expected Phase 1 output paths beneath one generated root."""

    root: Path
    data_root: Path
    manifest: Path
    artifacts_root: Path
    validation_artifact: Path
    preprocess_artifact: Path
    preprocessed_root: Path
    prediction_root: Path
    inference_artifact: Path
    evaluation_artifact: Path
    report_root: Path
    report_markdown: Path
    report_artifact: Path


def _paths(generated_root: Path) -> DagPaths:
    """Return the fixed Phase 1 output layout for a generated root."""
    data_root = generated_root / "data"
    artifacts_root = generated_root / "artifacts"
    report_root = generated_root / "report"
    return DagPaths(
        root=generated_root,
        data_root=data_root,
        manifest=data_root / "generated" / "phase1" / "data" / "synthetic_manifest.json",
        artifacts_root=artifacts_root,
        validation_artifact=artifacts_root / "validation.json",
        preprocess_artifact=artifacts_root / "preprocess.json",
        preprocessed_root=generated_root / "preprocessed",
        prediction_root=generated_root / "predictions",
        inference_artifact=artifacts_root / "inference.json",
        evaluation_artifact=artifacts_root / "evaluation.json",
        report_root=report_root,
        report_markdown=report_root / "phase1_report.md",
        report_artifact=report_root / "report_artifact.json",
    )


def _snakemake_base_command(*, cores: int = 1) -> list[str]:
    """Build the Snakemake command used by the integration tests."""
    return [
        "uv",
        "run",
        "--project",
        str(REPO_ROOT),
        "snakemake",
        "--snakefile",
        str(SNAKEFILE),
        "--cores",
        str(cores),
    ]


def _run_command(command: list[str], cwd: Path, *, expect_success: bool = True) -> str:
    """Run a subprocess and return combined stdout/stderr."""
    result = subprocess.run(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if expect_success:
        assert result.returncode == 0, result.stdout
    else:
        assert result.returncode != 0, result.stdout
    return result.stdout


def _run_snakemake(
    generated_root: Path,
    cwd: Path,
    *extra_args: str,
    git_commit: str = TEST_GIT_COMMIT,
    created_at_utc: str = TEST_CREATED_AT_UTC,
    expect_success: bool = True,
) -> str:
    """Run the real Phase 1 Snakemake workflow for one generated root."""
    command = [
        *_snakemake_base_command(),
        *extra_args,
        "--config",
        f"generated_root={generated_root}",
        f"git_commit={git_commit}",
        f"created_at_utc={created_at_utc}",
    ]
    return _run_command(command, cwd, expect_success=expect_success)


def _read_manifest(path: Path) -> SyntheticManifest:
    """Read a synthetic manifest through artifact helpers."""
    return artifact_from_json(path.read_text(encoding="utf-8"), SyntheticManifest)


def _read_validation(path: Path) -> ValidationArtifact:
    """Read a validation artifact through artifact helpers."""
    return artifact_from_json(path.read_text(encoding="utf-8"), ValidationArtifact)


def _read_preprocess(path: Path) -> PreprocessArtifact:
    """Read a preprocessing artifact through artifact helpers."""
    return artifact_from_json(path.read_text(encoding="utf-8"), PreprocessArtifact)


def _read_inference(path: Path) -> InferenceArtifact:
    """Read an inference artifact through artifact helpers."""
    return artifact_from_json(path.read_text(encoding="utf-8"), InferenceArtifact)


def _read_evaluation(path: Path) -> EvaluationArtifact:
    """Read an evaluation artifact through artifact helpers."""
    return artifact_from_json(path.read_text(encoding="utf-8"), EvaluationArtifact)


def _read_report_artifact(path: Path) -> ReportArtifact:
    """Read a report artifact through artifact helpers."""
    return artifact_from_json(path.read_text(encoding="utf-8"), ReportArtifact)


def _assert_expected_outputs_exist(paths: DagPaths) -> None:
    """Assert the fixed Phase 1 layout exists after a DAG run."""
    assert paths.data_root.is_dir()
    assert paths.artifacts_root.is_dir()
    assert paths.preprocessed_root.is_dir()
    assert paths.prediction_root.is_dir()
    assert paths.report_root.is_dir()
    assert paths.manifest.is_file()
    assert paths.validation_artifact.is_file()
    assert paths.preprocess_artifact.is_file()
    assert paths.inference_artifact.is_file()
    assert paths.evaluation_artifact.is_file()
    assert paths.report_markdown.is_file()
    assert paths.report_artifact.is_file()


def _assert_hash_linked_report(paths: DagPaths) -> ReportArtifact:
    """Assert the ReportArtifact records the Markdown report SHA-256."""
    report_artifact = _read_report_artifact(paths.report_artifact)
    assert report_artifact.report_sha256 == sha256_file(paths.report_markdown)
    return report_artifact


def _assert_linked_hashes_match(paths: DagPaths) -> None:
    """Assert all linked artifacts preserve matching config and manifest hashes."""
    manifest = _read_manifest(paths.manifest)
    validation = _read_validation(paths.validation_artifact)
    preprocess = _read_preprocess(paths.preprocess_artifact)
    inference = _read_inference(paths.inference_artifact)
    evaluation = _read_evaluation(paths.evaluation_artifact)
    report_artifact = _read_report_artifact(paths.report_artifact)

    manifest_hashes = {
        manifest.manifest_hash,
        validation.manifest_hash,
        preprocess.manifest_hash,
        inference.manifest_hash,
        evaluation.manifest_hash,
        report_artifact.manifest_hash,
    }
    assert validation.config_hash == manifest.config_hash
    assert preprocess.config_hash == inference.config_hash
    assert preprocess.config_hash == evaluation.config_hash
    assert preprocess.config_hash == report_artifact.config_hash
    assert len(manifest_hashes) == 1
    assert validation.invalid_case_count == 0
    assert manifest.case_ids == validation.validated_case_ids
    assert manifest.case_ids == preprocess.output_case_ids
    assert manifest.case_ids == inference.prediction_case_ids
    assert manifest.case_ids == evaluation.case_ids
    assert evaluation.evaluated_case_count == report_artifact.evaluated_case_count
    assert inference.method == "dummy"
    assert evaluation.method == "dummy"
    assert report_artifact.method == "dummy"


def _assert_generated_files_remain_under_root(paths: DagPaths) -> None:
    """Assert generated data, predictions, reports, and artifacts stay under root."""
    suffixes = {".json", ".md", ".nii", ".gz"}
    for path in paths.root.rglob("*"):
        if path.is_file() and path.suffix in suffixes:
            assert path.resolve().is_relative_to(paths.root.resolve())


def _artifact_bytes(paths: DagPaths) -> dict[str, bytes]:
    """Return deterministic JSON artifact bytes keyed by stable filenames."""
    return {
        "synthetic_manifest.json": paths.manifest.read_bytes(),
        "validation.json": paths.validation_artifact.read_bytes(),
        "preprocess.json": paths.preprocess_artifact.read_bytes(),
        "inference.json": paths.inference_artifact.read_bytes(),
        "evaluation.json": paths.evaluation_artifact.read_bytes(),
        "report_artifact.json": paths.report_artifact.read_bytes(),
    }


def test_phase1_snakemake_lists_rules_and_dry_run_resolves_complete_dag(tmp_path: Path) -> None:
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    generated_root = tmp_path / "generated-root"

    list_output = _run_snakemake(generated_root, cwd, "--list-rules")
    listed_rules = tuple(line.strip() for line in list_output.splitlines() if line.strip())
    assert set(listed_rules) == set(RULE_NAMES)
    assert len(listed_rules) == len(RULE_NAMES)

    dry_run_output = _run_snakemake(generated_root, cwd, "--dry-run", "--printshellcmds")
    for rule_name in RULE_NAMES[1:]:
        assert rule_name in dry_run_output


def test_phase1_snakemake_executes_from_empty_existing_generated_root(tmp_path: Path) -> None:
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    generated_root = tmp_path / "generated-root"
    generated_root.mkdir()
    paths = _paths(generated_root)

    first_output = _run_snakemake(generated_root, cwd, "--rerun-incomplete", "--printshellcmds")

    _assert_expected_outputs_exist(paths)
    _assert_hash_linked_report(paths)
    _assert_linked_hashes_match(paths)
    _assert_generated_files_remain_under_root(paths)
    report_text = paths.report_markdown.read_text(encoding="utf-8")
    assert "synthetic Phase 1 pipeline report" in report_text
    assert "dummy inference" in report_text
    for rule_name in RULE_NAMES[1:]:
        assert rule_name in first_output

    no_op_output = _run_snakemake(generated_root, cwd)
    assert "Nothing to be done" in no_op_output or "up to date" in no_op_output


def test_phase1_snakemake_executes_from_nonexistent_generated_root(tmp_path: Path) -> None:
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    generated_root = tmp_path / "missing-generated-root"
    paths = _paths(generated_root)

    _run_snakemake(generated_root, cwd, "--rerun-incomplete")

    _assert_expected_outputs_exist(paths)
    _assert_hash_linked_report(paths)
    _assert_linked_hashes_match(paths)


def test_phase1_snakemake_outputs_are_deterministic_across_independent_roots(
    tmp_path: Path,
) -> None:
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    first_paths = _paths(tmp_path / "first-root")
    second_paths = _paths(tmp_path / "second-root")

    _run_snakemake(first_paths.root, cwd, "--rerun-incomplete")
    _run_snakemake(second_paths.root, cwd, "--rerun-incomplete")

    assert first_paths.report_markdown.read_bytes() == second_paths.report_markdown.read_bytes()
    assert _artifact_bytes(first_paths) == _artifact_bytes(second_paths)
    assert tuple(_artifact_bytes(first_paths)) == DETERMINISTIC_JSON_NAMES


def test_phase1_snakemake_rebuilds_only_deleted_report_directory(tmp_path: Path) -> None:
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    paths = _paths(tmp_path / "generated-root")

    _run_snakemake(paths.root, cwd, "--rerun-incomplete")
    original_report = paths.report_markdown.read_bytes()
    upstream_artifacts = (
        paths.validation_artifact,
        paths.preprocess_artifact,
        paths.inference_artifact,
        paths.evaluation_artifact,
    )
    upstream_bytes = {path: path.read_bytes() for path in upstream_artifacts}
    upstream_mtimes = {path: path.stat().st_mtime_ns for path in upstream_artifacts}

    shutil.rmtree(paths.report_root)
    rebuild_output = _run_snakemake(paths.root, cwd)

    assert "report" in rebuild_output
    assert paths.report_markdown.read_bytes() == original_report
    assert _assert_hash_linked_report(paths)
    assert upstream_bytes == {path: path.read_bytes() for path in upstream_artifacts}
    assert upstream_mtimes == {path: path.stat().st_mtime_ns for path in upstream_artifacts}


def test_phase1_snakemake_rerun_incomplete_succeeds_when_outputs_are_complete(
    tmp_path: Path,
) -> None:
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    generated_root = tmp_path / "generated-root"

    _run_snakemake(generated_root, cwd, "--rerun-incomplete")
    output = _run_snakemake(generated_root, cwd, "--rerun-incomplete")

    assert "Nothing to be done" in output or "up to date" in output


def test_phase1_snakemake_missing_required_config_fails_clearly(tmp_path: Path) -> None:
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    command = [
        *_snakemake_base_command(),
        "--config",
        f"git_commit={TEST_GIT_COMMIT}",
        f"created_at_utc={TEST_CREATED_AT_UTC}",
    ]

    output = _run_command(command, cwd, expect_success=False)

    assert "Missing required Snakemake --config value: generated_root" in output
    assert "Traceback" not in output


def test_phase1_snakemake_relative_generated_root_fails_clearly(tmp_path: Path) -> None:
    cwd = tmp_path / "cwd"
    cwd.mkdir()

    output = _run_snakemake(Path("relative-generated-root"), cwd, expect_success=False)

    assert "generated_root must be an absolute path" in output
    assert "Traceback" not in output


def test_phase1_snakemake_regular_file_generated_root_fails_clearly(tmp_path: Path) -> None:
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    generated_root = tmp_path / "generated-root-file"
    generated_root.write_text("not a directory\n", encoding="utf-8")

    output = _run_snakemake(generated_root, cwd, expect_success=False)

    assert "generated_root exists and is not a directory" in output
    assert "Traceback" not in output
