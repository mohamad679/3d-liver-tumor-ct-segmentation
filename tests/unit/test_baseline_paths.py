from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from protoem_ct.artifacts.hashing import sha256_json
from protoem_ct.baselines.paths import (
    BaselineRunRootConflictError,
    BaselineRunRootCreationError,
    InvalidBaselineRunRootError,
    ValidatedBaselineRunPaths,
    create_validated_run_directory_tree,
    validate_baseline_run_root,
    validate_non_overlapping_run_roots,
)
from protoem_ct.baselines.provenance import (
    BaselineRunProvenance,
    baseline_run_provenance_to_json,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_valid_external_temporary_root(tmp_path: Path) -> None:
    parent = tmp_path / "runs"
    parent.mkdir()

    validated = validate_baseline_run_root(
        parent / "run-001",
        baseline_family="nnunet_v2",
        repository_root=REPO_ROOT,
    )

    assert validated.run_root == parent / "run-001"
    assert validated.predictions_dir == parent / "run-001" / "predictions"


def test_relative_path_rejected(tmp_path: Path) -> None:
    with pytest.raises(InvalidBaselineRunRootError):
        validate_baseline_run_root(
            Path("relative/run"),
            baseline_family="nnunet_v2",
            repository_root=REPO_ROOT,
        )


def test_repository_contained_path_rejected() -> None:
    with pytest.raises(InvalidBaselineRunRootError):
        validate_baseline_run_root(
            REPO_ROOT / "tmp-run",
            baseline_family="nnunet_v2",
            repository_root=REPO_ROOT,
        )


def test_repository_root_rejected() -> None:
    with pytest.raises(InvalidBaselineRunRootError):
        validate_baseline_run_root(
            REPO_ROOT,
            baseline_family="monai_segresnet",
            repository_root=REPO_ROOT,
        )


def test_existing_file_rejected(tmp_path: Path) -> None:
    parent = tmp_path / "runs"
    parent.mkdir()
    run_root = parent / "run-001"
    run_root.write_text("file", encoding="utf-8")

    with pytest.raises(InvalidBaselineRunRootError):
        validate_baseline_run_root(
            run_root,
            baseline_family="nnunet_v2",
            repository_root=REPO_ROOT,
        )


def test_existing_final_run_root_rejected(tmp_path: Path) -> None:
    parent = tmp_path / "runs"
    run_root = parent / "run-001"
    run_root.mkdir(parents=True)

    with pytest.raises(InvalidBaselineRunRootError):
        validate_baseline_run_root(
            run_root,
            baseline_family="monai_segresnet",
            repository_root=REPO_ROOT,
        )


@pytest.mark.skipif(not hasattr(Path, "symlink_to"), reason="Symlinks unsupported")
def test_symlink_root_rejected(tmp_path: Path) -> None:
    parent = tmp_path / "runs"
    parent.mkdir()
    target = tmp_path / "real-run"
    target.mkdir()
    symlink_root = parent / "run-link"
    symlink_root.symlink_to(target, target_is_directory=True)

    with pytest.raises(InvalidBaselineRunRootError):
        validate_baseline_run_root(
            symlink_root,
            baseline_family="nnunet_v2",
            repository_root=REPO_ROOT,
        )


@pytest.mark.skipif(not hasattr(Path, "symlink_to"), reason="Symlinks unsupported")
def test_symlinked_parent_component_rejected(tmp_path: Path) -> None:
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(InvalidBaselineRunRootError):
        validate_baseline_run_root(
            linked_parent / "run-001",
            baseline_family="nnunet_v2",
            repository_root=REPO_ROOT,
        )


def test_overlapping_run_roots_rejected(tmp_path: Path) -> None:
    parent = tmp_path / "runs"
    parent.mkdir()
    first_root = parent / "run-001"
    second_root = first_root / "nested"
    first = ValidatedBaselineRunPaths(
        baseline_family="nnunet_v2",
        run_root=first_root,
        checkpoints_dir=first_root / "checkpoints",
        predictions_dir=first_root / "predictions",
        metrics_dir=first_root / "metrics",
        mlflow_dir=first_root / "mlflow",
        logs_dir=first_root / "logs",
        temporary_dir=first_root / "temporary",
    )
    second = ValidatedBaselineRunPaths(
        baseline_family="monai_segresnet",
        run_root=second_root,
        checkpoints_dir=second_root / "checkpoints",
        predictions_dir=second_root / "predictions",
        metrics_dir=second_root / "metrics",
        mlflow_dir=second_root / "mlflow",
        logs_dir=second_root / "logs",
        temporary_dir=second_root / "temporary",
    )

    with pytest.raises(BaselineRunRootConflictError):
        validate_non_overlapping_run_roots((first, second))


def test_deterministic_child_paths(tmp_path: Path) -> None:
    parent = tmp_path / "runs"
    parent.mkdir()

    validated = validate_baseline_run_root(
        parent / "run-001",
        baseline_family="monai_segresnet",
        repository_root=REPO_ROOT,
    )

    assert validated.checkpoints_dir == validated.run_root / "checkpoints"
    assert validated.predictions_dir == validated.run_root / "predictions"
    assert validated.metrics_dir == validated.run_root / "metrics"
    assert validated.mlflow_dir == validated.run_root / "mlflow"
    assert validated.logs_dir == validated.run_root / "logs"
    assert validated.temporary_dir == validated.run_root / "temporary"


def test_successful_directory_tree_creation(tmp_path: Path) -> None:
    parent = tmp_path / "runs"
    parent.mkdir()
    validated = validate_baseline_run_root(
        parent / "run-001",
        baseline_family="nnunet_v2",
        repository_root=REPO_ROOT,
    )

    created = create_validated_run_directory_tree(validated)

    assert created.run_root.is_dir()
    assert created.checkpoints_dir.is_dir()
    assert created.predictions_dir.is_dir()
    assert created.metrics_dir.is_dir()
    assert created.mlflow_dir.is_dir()
    assert created.logs_dir.is_dir()
    assert created.temporary_dir.is_dir()


def test_no_overwrite_on_existing_path(tmp_path: Path) -> None:
    parent = tmp_path / "runs"
    parent.mkdir()
    validated = validate_baseline_run_root(
        parent / "run-001",
        baseline_family="nnunet_v2",
        repository_root=REPO_ROOT,
    )
    validated.run_root.mkdir()

    with pytest.raises(BaselineRunRootCreationError):
        create_validated_run_directory_tree(validated)


def test_failure_cleanup_preserves_preexisting_directories(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    parent = tmp_path / "runs"
    parent.mkdir()
    validated = validate_baseline_run_root(
        parent / "run-001",
        baseline_family="nnunet_v2",
        repository_root=REPO_ROOT,
    )
    original_mkdir = Path.mkdir

    def failing_mkdir(self: Path, *args: Any, **kwargs: Any) -> None:
        if self.name == "metrics":
            raise OSError("simulated mkdir failure")
        original_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", failing_mkdir)

    with pytest.raises(BaselineRunRootCreationError):
        create_validated_run_directory_tree(validated)

    assert parent.is_dir()
    assert not validated.run_root.exists()


def test_validation_performs_no_filesystem_creation(tmp_path: Path) -> None:
    parent = tmp_path / "runs"
    parent.mkdir()
    run_root = parent / "run-001"

    validate_baseline_run_root(
        run_root,
        baseline_family="nnunet_v2",
        repository_root=REPO_ROOT,
    )

    assert not run_root.exists()


def test_runtime_absolute_paths_do_not_appear_in_persisted_provenance(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "runs"
    parent.mkdir()
    validated = validate_baseline_run_root(
        parent / "run-001",
        baseline_family="nnunet_v2",
        repository_root=REPO_ROOT,
    )
    payload_without_hash = {
        "amp_enabled": False,
        "baseline_family": "nnunet_v2",
        "checkpoint_sha256": None,
        "config_sha256": "1" * 64,
        "contract_version": "baseline_run_provenance_v1",
        "dataset_manifest_sha256": "2" * 64,
        "development_split_sha256": "3" * 64,
        "end_timestamp": None,
        "failure_codes": [],
        "git_commit": "a" * 40,
        "metrics_artifact_sha256": None,
        "package_versions": {"numpy": "1.26.4"},
        "platform_machine": "x86_64",
        "platform_system": "Darwin",
        "prediction_manifest_sha256": None,
        "python_version": "3.11.9",
        "run_identifier": "run_001",
        "seed": 7,
        "selected_device": "cpu",
        "start_timestamp": "2026-07-28T12:00:00Z",
        "status": "planned",
    }
    provenance = BaselineRunProvenance(
        artifact_hash=sha256_json(payload_without_hash),
        baseline_family="nnunet_v2",
        run_identifier="run_001",
        git_commit="a" * 40,
        config_sha256="1" * 64,
        dataset_manifest_sha256="2" * 64,
        development_split_sha256="3" * 64,
        seed=7,
        python_version="3.11.9",
        platform_system="Darwin",
        platform_machine="x86_64",
        selected_device="cpu",
        amp_enabled=False,
        package_versions={"numpy": "1.26.4"},
        start_timestamp="2026-07-28T12:00:00Z",
        status="planned",
    )

    serialized = baseline_run_provenance_to_json(provenance)

    assert str(validated.run_root).encode("utf-8") not in serialized
