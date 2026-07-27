"""Safe path utilities for Phase 2 dataset and generated-output boundaries."""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath, PureWindowsPath

_DRIVE_PATTERN = re.compile(r"^[A-Za-z]:")
_URI_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")


class Phase2PathError(ValueError):
    """Base exception for Phase 2 safe-path and adapter boundary failures."""


class InvalidDatasetRootError(Phase2PathError):
    """Raised when an explicit dataset root is missing, unsafe, or invalid."""


class InvalidRelativeDatasetPathError(Phase2PathError):
    """Raised when a dataset-relative POSIX path is not safe and normalized."""


class DatasetPathEscapeError(Phase2PathError):
    """Raised when a path resolves outside its validated dataset root."""


class MissingDatasetPathError(Phase2PathError):
    """Raised when a dataset-relative path does not exist or dangles."""


class NonRegularDatasetFileError(Phase2PathError):
    """Raised when a required dataset file is not a regular file."""


class DuplicateResolvedPathError(Phase2PathError):
    """Raised when logical records resolve to the same canonical path."""


def validate_explicit_dataset_root(dataset_root: Path) -> Path:
    """Validate and canonicalize an explicitly supplied dataset root.

    The function never searches for a root, never creates one, and never reads environment or
    working-directory defaults. Callers must pass an absolute path.
    """
    if str(dataset_root) == "":
        msg = "dataset_root must not be empty"
        raise InvalidDatasetRootError(msg)
    if not dataset_root.is_absolute():
        msg = "dataset_root must be an explicit absolute path"
        raise InvalidDatasetRootError(msg)
    try:
        resolved_root = dataset_root.resolve(strict=True)
    except FileNotFoundError as exc:
        msg = "dataset_root does not exist"
        raise InvalidDatasetRootError(msg) from exc
    except OSError as exc:
        msg = f"dataset_root cannot be resolved: {exc}"
        raise InvalidDatasetRootError(msg) from exc
    if not resolved_root.is_dir():
        msg = "dataset_root must be an existing directory"
        raise InvalidDatasetRootError(msg)
    return resolved_root


def normalize_safe_relative_posix_path(value: str) -> str:
    """Return a canonical relative POSIX path without inspecting the filesystem."""
    if not isinstance(value, str) or value == "":
        msg = "relative dataset path must be a nonempty string"
        raise InvalidRelativeDatasetPathError(msg)
    if value != value.strip():
        msg = "relative dataset path must not contain leading or trailing whitespace"
        raise InvalidRelativeDatasetPathError(msg)
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        msg = "relative dataset path must not contain NUL or control characters"
        raise InvalidRelativeDatasetPathError(msg)
    if "\\" in value:
        msg = "relative dataset path must use POSIX separators"
        raise InvalidRelativeDatasetPathError(msg)
    if _DRIVE_PATTERN.match(value) is not None or _URI_PATTERN.match(value) is not None:
        msg = "relative dataset path must not be drive-letter or URI-like"
        raise InvalidRelativeDatasetPathError(msg)
    if value.startswith("~"):
        msg = "relative dataset path must not use home expansion"
        raise InvalidRelativeDatasetPathError(msg)
    if "//" in value:
        msg = "relative dataset path must not contain repeated separators"
        raise InvalidRelativeDatasetPathError(msg)

    path = PurePosixPath(value)
    if path.is_absolute():
        msg = "relative dataset path must not be absolute"
        raise InvalidRelativeDatasetPathError(msg)
    if str(path) != value:
        msg = "relative dataset path must already be normalized"
        raise InvalidRelativeDatasetPathError(msg)
    if value in {".", ".."}:
        msg = "relative dataset path must not be '.' or '..'"
        raise InvalidRelativeDatasetPathError(msg)
    if any(part in {"", ".", ".."} for part in path.parts):
        msg = "relative dataset path must not contain empty, '.', or '..' components"
        raise InvalidRelativeDatasetPathError(msg)
    if PureWindowsPath(value).is_absolute():
        msg = "relative dataset path must not be Windows-absolute"
        raise InvalidRelativeDatasetPathError(msg)
    return path.as_posix()


def resolve_existing_path_beneath_root(dataset_root: Path, relative_path: str) -> Path:
    """Resolve an existing relative path and require the target to stay beneath the root."""
    canonical_root = validate_explicit_dataset_root(dataset_root)
    normalized_relative_path = normalize_safe_relative_posix_path(relative_path)
    candidate = canonical_root / normalized_relative_path
    try:
        resolved_candidate = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        msg = f"dataset path is missing: {normalized_relative_path}"
        raise MissingDatasetPathError(msg) from exc
    except OSError as exc:
        msg = f"dataset path cannot be resolved: {normalized_relative_path}"
        raise MissingDatasetPathError(msg) from exc
    _require_beneath_root(canonical_root, resolved_candidate, normalized_relative_path)
    return resolved_candidate


def resolve_regular_file_beneath_root(dataset_root: Path, relative_path: str) -> Path:
    """Resolve a safe dataset-relative path and require a regular file target."""
    resolved_path = resolve_existing_path_beneath_root(dataset_root, relative_path)
    if not resolved_path.is_file():
        normalized_relative_path = normalize_safe_relative_posix_path(relative_path)
        msg = f"dataset path is not a regular file: {normalized_relative_path}"
        raise NonRegularDatasetFileError(msg)
    return resolved_path


def validate_explicit_external_output_root(
    output_root: Path,
    *,
    forbidden_roots: tuple[Path, ...] = (),
) -> Path:
    """Validate an explicit absolute generated-output root without creating it."""
    if str(output_root) == "":
        msg = "output_root must not be empty"
        raise InvalidDatasetRootError(msg)
    if not output_root.is_absolute():
        msg = "output_root must be an explicit absolute path"
        raise InvalidDatasetRootError(msg)

    try:
        resolved_output_root = _resolve_existing_parent_with_leaf(output_root)
    except FileNotFoundError as exc:
        msg = "output_root parent does not exist"
        raise InvalidDatasetRootError(msg) from exc
    except OSError as exc:
        msg = f"output_root cannot be resolved: {exc}"
        raise InvalidDatasetRootError(msg) from exc

    if output_root.exists() and not resolved_output_root.is_dir():
        msg = "output_root must be a directory when it already exists"
        raise InvalidDatasetRootError(msg)

    canonical_forbidden_roots = tuple(
        validate_explicit_dataset_root(root) for root in forbidden_roots
    )
    for forbidden_root in canonical_forbidden_roots:
        if _paths_are_equal_or_nested(resolved_output_root, forbidden_root):
            msg = "output_root must not equal or contain a forbidden input root"
            raise InvalidDatasetRootError(msg)
        if _paths_are_equal_or_nested(forbidden_root, resolved_output_root):
            msg = "output_root must not be inside a forbidden input root"
            raise InvalidDatasetRootError(msg)
    return resolved_output_root


def require_unique_resolved_paths(named_paths: tuple[tuple[str, Path], ...]) -> None:
    """Require every logical record name to map to a unique canonical resolved path."""
    resolved_by_path: dict[Path, list[str]] = {}
    for logical_name, path in named_paths:
        if not logical_name:
            msg = "logical path names must be nonempty"
            raise DuplicateResolvedPathError(msg)
        try:
            resolved_path = path.resolve(strict=True)
        except FileNotFoundError as exc:
            msg = f"path for {logical_name!r} is missing"
            raise MissingDatasetPathError(msg) from exc
        except OSError as exc:
            msg = f"path for {logical_name!r} cannot be resolved"
            raise MissingDatasetPathError(msg) from exc
        resolved_by_path.setdefault(resolved_path, []).append(logical_name)

    duplicate_groups = [
        tuple(sorted(names)) for names in resolved_by_path.values() if len(names) > 1
    ]
    if duplicate_groups:
        duplicate_groups.sort()
        formatted_groups = ", ".join(" + ".join(group) for group in duplicate_groups)
        msg = f"duplicate resolved paths for logical records: {formatted_groups}"
        raise DuplicateResolvedPathError(msg)


def _resolve_existing_parent_with_leaf(path: Path) -> Path:
    if path.exists():
        return path.resolve(strict=True)
    parent = path.parent.resolve(strict=True)
    return parent / path.name


def _require_beneath_root(root: Path, candidate: Path, logical_relative_path: str) -> None:
    if not _paths_are_equal_or_nested(candidate, root):
        msg = f"dataset path escapes the dataset root: {logical_relative_path}"
        raise DatasetPathEscapeError(msg)


def _paths_are_equal_or_nested(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
