"""LiTS-style filename discovery adapter for Phase 2 synthetic-tested inventory."""

from __future__ import annotations

import fnmatch
import os
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath

from protoem_ct.data.adapters.base import (
    AdapterCaseCandidate,
    AdapterInventory,
    AdapterLayoutSpec,
    InvalidAdapterInventoryError,
    resolve_adapter_inventory_files,
)
from protoem_ct.data.phase2_paths import (
    DuplicateResolvedPathError,
    Phase2PathError,
    normalize_safe_relative_posix_path,
    require_unique_resolved_paths,
    resolve_regular_file_beneath_root,
    validate_explicit_dataset_root,
)

LITS_STYLE_ADAPTER_NAME = "lits_style"
LITS_STYLE_ADAPTER_VERSION = "1"
SUPPORTED_LITS_SUFFIXES = (".nii.gz", ".nii")

_PREFIX_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_URI_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
_GLOB_METACHARACTERS = frozenset("*?[]!")


class LiTSAdapterError(Phase2PathError):
    """Base exception for LiTS-style discovery failures."""


class InvalidLiTSConventionError(LiTSAdapterError):
    """Raised when a LiTS-style filename convention is unsafe or unsupported."""


class LiTSDiscoveryError(LiTSAdapterError):
    """Raised when LiTS-style filesystem discovery fails."""


class MalformedLiTSFilenameError(LiTSDiscoveryError):
    """Raised when a matched filename does not follow the configured convention."""


class IncompleteLiTSPairError(LiTSDiscoveryError):
    """Raised when image and label keys do not form complete pairs."""


class AmbiguousLiTSPairingError(LiTSDiscoveryError):
    """Raised when matched files cannot be paired unambiguously."""


@dataclass(frozen=True, slots=True)
class LiTSFilenameConvention:
    """Explicit filename prefix and suffix convention for LiTS-style discovery.

    This configuration is not a Phase 2 artifact and must not be serialized as one. Prefixes are
    explicit input to the adapter; the adapter never infers them from arbitrary files.
    """

    image_prefix: str
    label_prefix: str
    allowed_suffixes: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_safe_prefix(self.image_prefix, "image_prefix")
        _require_safe_prefix(self.label_prefix, "label_prefix")
        if not isinstance(self.allowed_suffixes, tuple) or not self.allowed_suffixes:
            msg = "allowed_suffixes must be a nonempty tuple"
            raise InvalidLiTSConventionError(msg)
        if len(set(self.allowed_suffixes)) != len(self.allowed_suffixes):
            msg = "allowed_suffixes must not contain duplicates"
            raise InvalidLiTSConventionError(msg)
        for suffix in self.allowed_suffixes:
            if suffix not in SUPPORTED_LITS_SUFFIXES:
                msg = f"unsupported LiTS suffix: {suffix!r}"
                raise InvalidLiTSConventionError(msg)
        object.__setattr__(
            self,
            "allowed_suffixes",
            tuple(sorted(self.allowed_suffixes, key=len, reverse=True)),
        )


@dataclass(frozen=True, slots=True)
class _DiscoveredFile:
    source_case_key: str
    relative_path: str
    resolved_path: Path


@dataclass(frozen=True, slots=True)
class LiTSStyleAdapter:
    """Concrete read-only LiTS-style discovery adapter.

    Discovery inventories filesystem names only. It never opens, parses, hashes, memory-maps, or
    validates NIfTI contents and does not create anonymous manifests.
    """

    convention: LiTSFilenameConvention
    adapter_name: str = field(init=False, default=LITS_STYLE_ADAPTER_NAME)
    adapter_version: str = field(init=False, default=LITS_STYLE_ADAPTER_VERSION)

    def discover(self, dataset_root: Path, layout: AdapterLayoutSpec) -> AdapterInventory:
        """Discover deterministic image/label candidates beneath an explicit dataset root."""
        canonical_root = validate_explicit_dataset_root(dataset_root)
        image_files = _discover_files(
            canonical_root,
            layout.image_pattern,
            layout.recursive,
            prefix=self.convention.image_prefix,
            allowed_suffixes=self.convention.allowed_suffixes,
            side="image",
        )
        label_files = _discover_files(
            canonical_root,
            layout.label_pattern,
            layout.recursive,
            prefix=self.convention.label_prefix,
            allowed_suffixes=self.convention.allowed_suffixes,
            side="label",
        )
        if not image_files and not label_files:
            msg = "LiTS-style discovery found no matched image or label files"
            raise LiTSDiscoveryError(msg)
        _require_unique_discovered_keys(image_files, "image")
        _require_unique_discovered_keys(label_files, "label")
        _require_unique_resolved_files(image_files, "image")
        _require_unique_resolved_files(label_files, "label")
        _require_complete_key_pairs(image_files, label_files)

        image_by_key = {item.source_case_key: item for item in image_files}
        label_by_key = {item.source_case_key: item for item in label_files}
        candidates = tuple(
            AdapterCaseCandidate(
                source_case_key=key,
                image_relative_path=image_by_key[key].relative_path,
                label_relative_path=label_by_key[key].relative_path,
            )
            for key in sorted(image_by_key)
        )
        inventory = AdapterInventory(
            adapter_name=self.adapter_name,
            adapter_version=self.adapter_version,
            candidates=candidates,
        )
        try:
            resolve_adapter_inventory_files(canonical_root, inventory)
        except (DuplicateResolvedPathError, InvalidAdapterInventoryError) as exc:
            msg = "LiTS-style image and label files must resolve to unambiguous files"
            raise AmbiguousLiTSPairingError(msg) from exc
        return inventory


def _discover_files(
    dataset_root: Path,
    pattern: str,
    recursive: bool,
    *,
    prefix: str,
    allowed_suffixes: tuple[str, ...],
    side: str,
) -> tuple[_DiscoveredFile, ...]:
    relative_paths = _matching_relative_paths(dataset_root, pattern, recursive=recursive)
    discovered: list[_DiscoveredFile] = []
    for relative_path in relative_paths:
        source_case_key = _extract_source_case_key(
            relative_path,
            prefix=prefix,
            allowed_suffixes=allowed_suffixes,
            side=side,
        )
        resolved_path = resolve_regular_file_beneath_root(dataset_root, relative_path)
        discovered.append(
            _DiscoveredFile(
                source_case_key=source_case_key,
                relative_path=relative_path,
                resolved_path=resolved_path,
            )
        )
    return tuple(sorted(discovered, key=lambda item: item.relative_path))


def _matching_relative_paths(
    dataset_root: Path,
    pattern: str,
    *,
    recursive: bool,
) -> tuple[str, ...]:
    if recursive:
        values = tuple(_recursive_relative_matches(dataset_root, pattern))
    else:
        values = tuple(_nonrecursive_relative_matches(dataset_root, pattern))
    return tuple(sorted(normalize_safe_relative_posix_path(value) for value in values))


def _nonrecursive_relative_matches(dataset_root: Path, pattern: str) -> tuple[str, ...]:
    pattern_parts = PurePosixPath(pattern).parts
    current_dirs: tuple[Path, ...] = (dataset_root,)
    for pattern_part in pattern_parts[:-1]:
        next_dirs: list[Path] = []
        for current_dir in current_dirs:
            next_dirs.extend(_matching_child_directories(current_dir, pattern_part))
        current_dirs = tuple(sorted(next_dirs, key=lambda path: path.name))
        if not current_dirs:
            return ()

    final_part = pattern_parts[-1]
    matches: list[str] = []
    for current_dir in current_dirs:
        for child in _sorted_children(current_dir):
            if fnmatch.fnmatchcase(child.name, final_part):
                matches.append(_relative_posix_path(dataset_root, child))
    return tuple(matches)


def _recursive_relative_matches(dataset_root: Path, pattern: str) -> tuple[str, ...]:
    matches: list[str] = []
    pattern_has_directory = "/" in pattern
    for current_root, dirnames, filenames in os.walk(dataset_root, topdown=True, followlinks=False):
        current_path = Path(current_root)
        dirnames.sort()
        filenames.sort()
        for dirname in tuple(dirnames):
            directory_path = current_path / dirname
            relative_directory = _relative_posix_path(dataset_root, directory_path)
            if _matches_pattern(relative_directory, dirname, pattern, pattern_has_directory):
                matches.append(relative_directory)
            if directory_path.is_symlink():
                dirnames.remove(dirname)
        for filename in filenames:
            file_path = current_path / filename
            relative_file = _relative_posix_path(dataset_root, file_path)
            if _matches_pattern(relative_file, filename, pattern, pattern_has_directory):
                matches.append(relative_file)
    return tuple(matches)


def _matching_child_directories(current_dir: Path, pattern_part: str) -> tuple[Path, ...]:
    if _has_glob(pattern_part):
        children = _sorted_children(current_dir)
        return tuple(
            child
            for child in children
            if fnmatch.fnmatchcase(child.name, pattern_part)
            and child.is_dir()
            and not child.is_symlink()
        )
    child = current_dir / pattern_part
    if child.exists() and child.is_dir() and not child.is_symlink():
        return (child,)
    return ()


def _sorted_children(path: Path) -> tuple[Path, ...]:
    try:
        return tuple(sorted(path.iterdir(), key=lambda child: child.name))
    except FileNotFoundError:
        return ()


def _matches_pattern(
    relative_path: str,
    basename: str,
    pattern: str,
    pattern_has_directory: bool,
) -> bool:
    if fnmatch.fnmatchcase(relative_path, pattern):
        return True
    return not pattern_has_directory and fnmatch.fnmatchcase(basename, pattern)


def _extract_source_case_key(
    relative_path: str,
    *,
    prefix: str,
    allowed_suffixes: tuple[str, ...],
    side: str,
) -> str:
    filename = PurePosixPath(relative_path).name
    if not filename.startswith(prefix):
        msg = f"matched {side} filename does not start with configured prefix: {relative_path}"
        raise MalformedLiTSFilenameError(msg)
    suffix = _matching_suffix(filename, allowed_suffixes)
    if suffix is None:
        msg = f"matched {side} filename does not use an allowed NIfTI suffix: {relative_path}"
        raise MalformedLiTSFilenameError(msg)
    source_case_key = filename[len(prefix) : len(filename) - len(suffix)]
    if source_case_key == "":
        msg = f"matched {side} filename produced an empty source key: {relative_path}"
        raise MalformedLiTSFilenameError(msg)
    try:
        AdapterCaseCandidate(
            source_case_key=source_case_key,
            image_relative_path="synthetic-image-placeholder.nii",
            label_relative_path="synthetic-label-placeholder.nii",
        )
    except Phase2PathError as exc:
        msg = f"matched {side} filename produced an unsafe source key: {relative_path}"
        raise MalformedLiTSFilenameError(msg) from exc
    return source_case_key


def _matching_suffix(filename: str, allowed_suffixes: tuple[str, ...]) -> str | None:
    for suffix in allowed_suffixes:
        if filename.endswith(suffix):
            return suffix
    return None


def _require_unique_discovered_keys(files: tuple[_DiscoveredFile, ...], side: str) -> None:
    by_key: dict[str, list[str]] = {}
    for item in files:
        by_key.setdefault(item.source_case_key, []).append(item.relative_path)
    duplicate_keys = sorted(key for key, paths in by_key.items() if len(paths) > 1)
    if duplicate_keys:
        duplicate_descriptions = ", ".join(
            f"{key}: {', '.join(sorted(by_key[key]))}" for key in duplicate_keys
        )
        msg = f"multiple {side} files produced the same source key: {duplicate_descriptions}"
        raise AmbiguousLiTSPairingError(msg)


def _require_unique_resolved_files(files: tuple[_DiscoveredFile, ...], side: str) -> None:
    try:
        require_unique_resolved_paths(
            tuple((f"{item.source_case_key}:{side}", item.resolved_path) for item in files)
        )
    except DuplicateResolvedPathError as exc:
        msg = f"multiple {side} files resolve to the same file"
        raise AmbiguousLiTSPairingError(msg) from exc


def _require_complete_key_pairs(
    image_files: tuple[_DiscoveredFile, ...],
    label_files: tuple[_DiscoveredFile, ...],
) -> None:
    image_keys = {item.source_case_key for item in image_files}
    label_keys = {item.source_case_key for item in label_files}
    missing_labels = sorted(image_keys - label_keys)
    missing_images = sorted(label_keys - image_keys)
    if missing_labels or missing_images:
        details: list[str] = []
        if missing_labels:
            details.append(f"missing labels for keys: {', '.join(missing_labels)}")
        if missing_images:
            details.append(f"missing images for keys: {', '.join(missing_images)}")
        msg = "; ".join(details)
        raise IncompleteLiTSPairError(msg)


def _require_safe_prefix(value: object, field_name: str) -> None:
    if not isinstance(value, str) or value == "":
        msg = f"{field_name} must be a nonempty filename prefix"
        raise InvalidLiTSConventionError(msg)
    if value != value.strip():
        msg = f"{field_name} must not contain leading or trailing whitespace"
        raise InvalidLiTSConventionError(msg)
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        msg = f"{field_name} must not contain NUL or control characters"
        raise InvalidLiTSConventionError(msg)
    if "/" in value or "\\" in value:
        msg = f"{field_name} must not contain path separators"
        raise InvalidLiTSConventionError(msg)
    if _URI_PATTERN.match(value) is not None or PureWindowsPath(value).is_absolute():
        msg = f"{field_name} must not be URI-like or drive-letter form"
        raise InvalidLiTSConventionError(msg)
    if any(character in _GLOB_METACHARACTERS for character in value):
        msg = f"{field_name} must not contain glob metacharacters"
        raise InvalidLiTSConventionError(msg)
    if _PREFIX_PATTERN.fullmatch(value) is None:
        msg = f"{field_name} contains unsupported characters"
        raise InvalidLiTSConventionError(msg)


def _has_glob(value: str) -> bool:
    return any(character in value for character in "*?[")


def _relative_posix_path(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()
