"""Integration tests for the synthetic-only Phase 2 Snakemake DAG."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path
from typing import Any, NamedTuple, TypeVar, cast

import nibabel as nib
import numpy as np

from protoem_ct.artifacts import (
    DatasetManifest,
    DevelopmentDataSummaryArtifact,
    DevelopmentQaArtifact,
    DevelopmentSplitManifest,
    GeometryLabelQaArtifact,
    LeakageAuditArtifact,
    LesionComponentsArtifact,
    hash_dataset_manifest,
    hash_development_data_summary,
    hash_development_qa,
    hash_development_split,
    hash_geometry_label_qa,
    hash_leakage_audit,
    hash_lesion_components,
    phase2_artifact_from_json,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "phase2_synthetic.yaml"
SNAKEFILE = REPO_ROOT / "Snakefile"
SYNTHETIC_GIT_COMMIT = "syntheticgit0000000000000000000000000000000000000000"
EXPECTED_IMAGE_PATHS = tuple(f"dataset/volumes/volume-{index}.nii" for index in range(4))
EXPECTED_LABEL_PATHS = tuple(
    f"dataset/segmentations/segmentation-{index}.nii" for index in range(4)
)
EXPECTED_ARTIFACTS = (
    "dataset_manifest.json",
    "development_split.json",
    "geometry_label_qa.json",
    "lesion_components.json",
    "development_data_summary.json",
    "development_qa_report.json",
    "leakage_audit.json",
)
ArtifactT = TypeVar("ArtifactT")


class Phase2SyntheticRun(NamedTuple):
    """Resolved paths for one synthetic DAG execution."""

    root: Path
    dataset_root: Path
    control_root: Path
    artifact_root: Path
    log_root: Path


def _run_snakemake(
    root: Path, *, cwd: Path | None = None, include_git: bool = True
) -> subprocess.CompletedProcess[str]:
    snakemake = shutil.which("snakemake")
    if snakemake is None:
        raise RuntimeError("snakemake executable is not available")
    config_args = [
        "phase2_synthetic_generated_root=" + str(root),
    ]
    if include_git:
        config_args.append("phase2_synthetic_git_commit=" + SYNTHETIC_GIT_COMMIT)
    return subprocess.run(
        [
            snakemake,
            "--snakefile",
            str(SNAKEFILE),
            "--cores",
            "1",
            "phase2_synthetic_all",
            "--configfile",
            str(CONFIG_PATH),
            "--config",
            *config_args,
        ],
        cwd=cwd or REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _run_dag(root: Path, *, cwd: Path | None = None) -> Phase2SyntheticRun:
    result = _run_snakemake(root, cwd=cwd)
    assert result.returncode == 0, result.stderr or result.stdout
    return Phase2SyntheticRun(
        root=root,
        dataset_root=root / "dataset",
        control_root=root / "control",
        artifact_root=root / "artifacts",
        log_root=root / "logs",
    )


def _parse(path: Path, artifact_type: type[ArtifactT]) -> ArtifactT:
    artifact = phase2_artifact_from_json(path.read_text(encoding="utf-8"))
    assert isinstance(artifact, artifact_type)
    return artifact


def _load_chain(
    run: Phase2SyntheticRun,
) -> tuple[
    DatasetManifest,
    DevelopmentSplitManifest,
    GeometryLabelQaArtifact,
    LesionComponentsArtifact,
    DevelopmentDataSummaryArtifact,
    DevelopmentQaArtifact,
    LeakageAuditArtifact,
]:
    return (
        _parse(run.artifact_root / "dataset_manifest.json", DatasetManifest),
        _parse(run.artifact_root / "development_split.json", DevelopmentSplitManifest),
        _parse(run.artifact_root / "geometry_label_qa.json", GeometryLabelQaArtifact),
        _parse(run.artifact_root / "lesion_components.json", LesionComponentsArtifact),
        _parse(run.artifact_root / "development_data_summary.json", DevelopmentDataSummaryArtifact),
        _parse(run.artifact_root / "development_qa_report.json", DevelopmentQaArtifact),
        _parse(run.artifact_root / "leakage_audit.json", LeakageAuditArtifact),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_four_case_fixture(run: Phase2SyntheticRun) -> None:
    image_paths = tuple(
        sorted(
            path.relative_to(run.root).as_posix() for path in run.dataset_root.rglob("volume-*.nii")
        )
    )
    label_paths = tuple(
        sorted(
            path.relative_to(run.root).as_posix()
            for path in run.dataset_root.rglob("segmentation-*.nii")
        )
    )
    assert image_paths == EXPECTED_IMAGE_PATHS
    assert label_paths == EXPECTED_LABEL_PATHS

    image_hashes = {_sha256(run.root / relative_path) for relative_path in EXPECTED_IMAGE_PATHS}
    label_hashes = {_sha256(run.root / relative_path) for relative_path in EXPECTED_LABEL_PATHS}
    assert len(image_hashes) == 4
    assert len(label_hashes) == 4

    observed_tumor_counts: list[int] = []
    for image_relative, label_relative in zip(
        EXPECTED_IMAGE_PATHS, EXPECTED_LABEL_PATHS, strict=True
    ):
        image = cast(nib.Nifti1Image, nib.load(str(run.root / image_relative)))
        label = cast(nib.Nifti1Image, nib.load(str(run.root / label_relative)))
        assert image.shape == (6, 7, 5)
        assert label.shape == (6, 7, 5)
        image_zooms = cast(Any, image.header).get_zooms()
        label_zooms = cast(Any, label.header).get_zooms()
        assert tuple(float(value) for value in image_zooms[:3]) == (1.25, 1.5, 2.5)
        assert tuple(float(value) for value in label_zooms[:3]) == (1.25, 1.5, 2.5)
        label_data = np.asanyarray(label.dataobj)
        assert set(np.unique(label_data).tolist()) <= {0, 1}
        observed_tumor_counts.append(int(np.count_nonzero(label_data == 1)))
    assert observed_tumor_counts == [8, 12, 27, 0]


def _assert_phase2_artifact_chain(run: Phase2SyntheticRun) -> None:
    manifest, split, geometry, lesion, summary, report, audit = _load_chain(run)

    assert manifest.case_count == 4
    assert manifest.manifest_hash == hash_dataset_manifest(manifest)
    assert split.split_hash == hash_development_split(split)
    assert geometry.qa_artifact_hash == hash_geometry_label_qa(geometry)
    assert lesion.lesion_artifact_hash == hash_lesion_components(lesion)
    assert summary.summary_artifact_hash == hash_development_data_summary(summary)
    assert report.qa_artifact_hash == hash_development_qa(report)
    assert audit.audit_hash == hash_leakage_audit(audit)

    assert split.source_manifest_hash == manifest.manifest_hash
    assert geometry.manifest_hash == manifest.manifest_hash
    assert geometry.split_hash == split.split_hash
    assert lesion.manifest_hash == manifest.manifest_hash
    assert lesion.split_hash == split.split_hash
    assert lesion.geometry_qa_artifact_hash == geometry.qa_artifact_hash
    assert summary.manifest_hash == manifest.manifest_hash
    assert summary.split_hash == split.split_hash
    assert summary.geometry_qa_artifact_hash == geometry.qa_artifact_hash
    assert summary.lesion_artifact_hash == lesion.lesion_artifact_hash
    assert report.manifest_hash == manifest.manifest_hash
    assert report.split_hash == split.split_hash
    assert report.geometry_qa_artifact_hash == geometry.qa_artifact_hash
    assert report.lesion_artifact_hash == lesion.lesion_artifact_hash
    assert report.development_summary_artifact_hash == summary.summary_artifact_hash
    assert audit.manifest_hash == manifest.manifest_hash
    assert audit.split_hash == split.split_hash

    assignments = split.assignments
    assert split.train_patient_count == 2
    assert split.validation_patient_count == 1
    assert split.internal_test_patient_count == 1
    assert {assignment.partition for assignment in assignments} == {
        "train",
        "validation",
        "internal_test",
    }
    assert len(assignments) == 4
    assert len({assignment.anonymous_case_id for assignment in assignments}) == 4
    assert len({assignment.anonymous_patient_id for assignment in assignments}) == 4

    assert report.case_count == 4
    assert report.passed_case_count == 4
    assert report.failed_case_count == 0
    assert all(record.qa_passed for record in report.case_records)

    assert audit.audit_passed is True
    assert sum(audit.pairwise_patient_overlap_counts.values()) == 0
    assert sum(audit.pairwise_case_overlap_counts.values()) == 0
    assert audit.image_hash_cross_partition_overlap_count == 0
    assert audit.label_hash_cross_partition_overlap_count == 0
    assert audit.image_label_pair_cross_partition_overlap_count == 0
    assert audit.finding_codes == ()


def test_fixture_generator_creates_deterministic_synthetic_lits_files(tmp_path: Path) -> None:
    run = _run_dag(tmp_path / "generated")

    _assert_four_case_fixture(run)
    key_path = run.control_root / "phase2_synthetic_hmac.key"
    assert key_path.exists()
    assert key_path.relative_to(run.root).as_posix().startswith("control/")
    assert len(key_path.read_bytes()) >= 32

    serialized_text = "\n".join(
        path.read_text(encoding="utf-8") for path in run.artifact_root.glob("*.json")
    )
    log_text = "\n".join(path.read_text(encoding="utf-8") for path in run.log_root.glob("*.log"))
    key_text = key_path.read_text(encoding="utf-8")
    assert key_text not in serialized_text
    assert key_text not in log_text


def test_full_phase2_synthetic_dag_outputs_and_links(tmp_path: Path) -> None:
    run = _run_dag(tmp_path / "generated")

    assert (run.log_root / "inventory_lits.log").exists()
    for artifact_name in EXPECTED_ARTIFACTS:
        assert (run.artifact_root / artifact_name).exists()
    _assert_phase2_artifact_chain(run)


def test_two_independent_generated_roots_are_byte_identical(tmp_path: Path) -> None:
    first = _run_dag(tmp_path / "first")
    second_cwd = tmp_path / "cwd"
    second_cwd.mkdir()
    second = _run_dag(tmp_path / "second", cwd=second_cwd)

    assert (first.log_root / "inventory_lits.log").read_bytes() == (
        second.log_root / "inventory_lits.log"
    ).read_bytes()
    for artifact_name in EXPECTED_ARTIFACTS:
        assert (first.artifact_root / artifact_name).read_bytes() == (
            second.artifact_root / artifact_name
        ).read_bytes()
    for relative_path in (*EXPECTED_IMAGE_PATHS, *EXPECTED_LABEL_PATHS):
        assert _sha256(first.root / relative_path) == _sha256(second.root / relative_path)


def test_missing_phase2_git_commit_fails_before_fixture_processing(tmp_path: Path) -> None:
    root = tmp_path / "missing-git"

    result = _run_snakemake(root, include_git=False)

    assert result.returncode != 0
    assert "phase2_synthetic_git_commit" in (result.stderr + result.stdout)
    assert not list((root / "dataset").glob("**/*.nii"))


def test_default_generated_root_is_ignored_and_generated_files_are_not_git_visible(
    tmp_path: Path,
) -> None:
    ignored = subprocess.run(
        ["git", "check-ignore", "-v", ".artifacts/phase2_synthetic"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert ignored.returncode == 0

    run = _run_dag(tmp_path / "generated")
    assert run.root.is_absolute()
    visible = subprocess.run(
        [
            "git",
            "status",
            "--short",
            "--untracked-files=all",
            "--",
            "*.nii",
            "*.nii.gz",
            "*.dcm",
            "*.json",
            "*.csv",
            "*.tsv",
            "*.key",
            "*.log",
            ":(exclude)reports/**",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert visible.returncode == 0
    assert visible.stdout == ""
