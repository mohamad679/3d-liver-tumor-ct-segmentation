"""Validated external path contracts for Phase 3 baseline runs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

BaselineFamily = Literal["nnunet_v2", "monai_segresnet"]

SUPPORTED_BASELINE_FAMILIES: tuple[BaselineFamily, ...] = (
    "nnunet_v2",
    "monai_segresnet",
)
BASELINE_CHILD_DIRECTORY_NAMES: tuple[str, ...] = (
    "checkpoints",
    "predictions",
    "metrics",
    "mlflow",
    "logs",
    "temporary",
)
_USER_ONLY_DIRECTORY_MODE = 0o700


class BaselinePathError(ValueError):
    """Base class for invalid Phase 3 baseline path contracts."""


class InvalidBaselineRunRootError(BaselinePathError):
    """Raised when a baseline run root violates the external path contract."""


class BaselineRunRootConflictError(BaselinePathError):
    """Raised when two validated run roots would overlap."""


class BaselineRunRootCreationError(BaselinePathError):
    """Raised when validated baseline directories cannot be created safely."""


@dataclass(frozen=True, slots=True)
class ValidatedBaselineRunPaths:
    """Validated runtime-only external locations for one baseline run."""

    baseline_family: BaselineFamily
    run_root: Path
    checkpoints_dir: Path
    predictions_dir: Path
    metrics_dir: Path
    mlflow_dir: Path
    logs_dir: Path
    temporary_dir: Path

    def create_directory_tree(self) -> ValidatedBaselineRunPaths:
        """Create only the approved baseline run directories."""

        return create_validated_run_directory_tree(self)


def validate_baseline_run_root(
    run_root: Path,
    *,
    baseline_family: BaselineFamily,
    repository_root: Path | None = None,
) -> ValidatedBaselineRunPaths:
    """Validate one external baseline run root without creating it."""

    _require_supported_baseline_family(baseline_family)
    if repository_root is None:
        repository_root = Path(__file__).resolve().parents[3]
    repo_root_resolved = repository_root.resolve(strict=True)
    root_path = Path(run_root)
    _require_absolute_path(root_path)
    _require_safe_path_components(root_path)
    _require_outside_repository(run_root=root_path, repository_root=repo_root_resolved)

    parent_path = root_path.parent
    if not parent_path.exists():
        raise InvalidBaselineRunRootError("Baseline run root parent directory must already exist.")
    if not parent_path.is_dir():
        raise InvalidBaselineRunRootError("Baseline run root parent directory must be a directory.")
    _require_no_symlink_components(parent_path)

    if root_path.exists():
        if root_path.is_symlink():
            raise InvalidBaselineRunRootError("Baseline run root must not be a symlink.")
        if root_path.is_file():
            raise InvalidBaselineRunRootError(
                "Baseline run root must not be an existing regular file."
            )
        raise InvalidBaselineRunRootError("Baseline run root must not already exist.")

    resolved_root = parent_path.resolve(strict=True) / root_path.name
    return ValidatedBaselineRunPaths(
        baseline_family=baseline_family,
        run_root=resolved_root,
        checkpoints_dir=resolved_root / "checkpoints",
        predictions_dir=resolved_root / "predictions",
        metrics_dir=resolved_root / "metrics",
        mlflow_dir=resolved_root / "mlflow",
        logs_dir=resolved_root / "logs",
        temporary_dir=resolved_root / "temporary",
    )


def validate_non_overlapping_run_roots(
    run_paths: tuple[ValidatedBaselineRunPaths, ...],
) -> tuple[ValidatedBaselineRunPaths, ...]:
    """Require that validated run roots do not overlap."""

    sorted_paths = sorted(run_paths, key=lambda item: str(item.run_root))
    for index, current in enumerate(sorted_paths):
        for other in sorted_paths[index + 1 :]:
            if _paths_overlap(current.run_root, other.run_root):
                raise BaselineRunRootConflictError(
                    f"Baseline run roots overlap: {current.run_root} and {other.run_root}."
                )
    return run_paths


def create_validated_run_directory_tree(
    run_paths: ValidatedBaselineRunPaths,
) -> ValidatedBaselineRunPaths:
    """Create the validated baseline run tree without overwriting anything."""

    created_directories: list[Path] = []
    all_directories = (
        run_paths.run_root,
        run_paths.checkpoints_dir,
        run_paths.predictions_dir,
        run_paths.metrics_dir,
        run_paths.mlflow_dir,
        run_paths.logs_dir,
        run_paths.temporary_dir,
    )
    try:
        if run_paths.run_root.exists():
            raise BaselineRunRootCreationError("Validated baseline run root already exists.")
        for directory in all_directories:
            directory.mkdir(mode=_USER_ONLY_DIRECTORY_MODE)
            created_directories.append(directory)
            _enforce_user_only_permissions(directory)
    except Exception as error:
        _cleanup_created_directories(created_directories)
        if isinstance(error, BaselinePathError):
            raise
        raise BaselineRunRootCreationError(
            f"Failed to create baseline run directory tree: {error}"
        ) from error
    return run_paths


def _require_supported_baseline_family(baseline_family: str) -> None:
    if baseline_family not in SUPPORTED_BASELINE_FAMILIES:
        raise InvalidBaselineRunRootError(f"Unsupported baseline family: {baseline_family!r}.")


def _require_absolute_path(path: Path) -> None:
    if not path.is_absolute():
        raise InvalidBaselineRunRootError("Baseline run root must be an absolute path.")


def _require_safe_path_components(path: Path) -> None:
    normalized = os.path.normpath(os.fspath(path))
    if ".." in Path(normalized).parts:
        raise InvalidBaselineRunRootError(
            "Baseline run root must not contain parent-directory traversal."
        )
    for component in path.parts[1:]:
        if component == "":
            raise InvalidBaselineRunRootError(
                "Baseline run root must not contain empty path components."
            )
        if any(ord(character) < 32 or ord(character) == 127 for character in component):
            raise InvalidBaselineRunRootError(
                "Baseline run root must not contain control characters."
            )


def _require_outside_repository(*, run_root: Path, repository_root: Path) -> None:
    normalized_root = run_root.resolve(strict=False)
    if normalized_root == repository_root:
        raise InvalidBaselineRunRootError("Baseline run root must not equal the repository root.")
    if _is_relative_to(normalized_root, repository_root):
        raise InvalidBaselineRunRootError(
            "Baseline run root must resolve outside the repository root."
        )


def _require_no_symlink_components(existing_path: Path) -> None:
    current = Path(existing_path.anchor)
    parts = existing_path.parts[1:]
    for component in parts:
        current = current / component
        if current.is_symlink():
            raise InvalidBaselineRunRootError(
                "Baseline run root must not traverse symlinked path components."
            )


def _enforce_user_only_permissions(directory: Path) -> None:
    if os.name != "posix":
        return
    directory.chmod(_USER_ONLY_DIRECTORY_MODE)


def _cleanup_created_directories(created_directories: list[Path]) -> None:
    for directory in reversed(created_directories):
        try:
            directory.rmdir()
        except FileNotFoundError:
            continue
        except OSError:
            continue


def _paths_overlap(first: Path, second: Path) -> bool:
    return _is_relative_to(first, second) or _is_relative_to(second, first)


def _is_relative_to(path: Path, candidate_parent: Path) -> bool:
    try:
        path.relative_to(candidate_parent)
        return True
    except ValueError:
        return False
