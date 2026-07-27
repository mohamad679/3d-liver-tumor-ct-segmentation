"""3D-IRCADb-style discovery adapter for Phase 2 synthetic fixtures only.

This module provides a read-only adapter interface that can be exercised with synthetic filesystem
fixtures during Phase 2. Real 3D-IRCADb-01 data must not be inventoried, scanned, QA'd, split,
tuned on, or accessed until Phase 8 external validation. Discovery here does not authorize external
cohort use, external labels must not affect development decisions, and no real external inventory
output should be committed.
"""

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

IRCADB_STYLE_ADAPTER_NAME = "ircadb_style"
IRCADB_STYLE_ADAPTER_VERSION = "1"
SUPPORTED_IRCADB_SUFFIXES = (".nii.gz", ".nii")

_BASENAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,126}$")
_URI_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
_GLOB_ALLOWED_CHARS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-/*?[]!"
)
_GLOB_METACHARACTERS = frozenset("*?[]!")


class IrcadbAdapterError(Phase2PathError):
    """Base exception for 3D-IRCADb-style discovery failures."""


class InvalidIrcadbConventionError(IrcadbAdapterError):
    """Raised when a 3D-IRCADb-style convention is unsafe or unsupported."""


class IrcadbDiscoveryError(IrcadbAdapterError):
    """Raised when 3D-IRCADb-style filesystem discovery fails."""


class MalformedIrcadbCaseError(IrcadbDiscoveryError):
    """Raised when a discovered fixture case cannot produce a safe ephemeral key."""


class IncompleteIrcadbPairError(IrcadbDiscoveryError):
    """Raised when image and tumor-label files do not form complete pairs."""


class AmbiguousIrcadbPairingError(IrcadbDiscoveryError):
    """Raised when discovered files cannot be paired unambiguously."""


@dataclass(frozen=True, slots=True)
class IrcadbFilenameConvention:
    """Explicit 3D-IRCADb-style filename and case-directory convention.

    The convention is ephemeral configuration, not a Phase 2 artifact. It is supplied explicitly
    and is never inferred from dataset contents.
    """

    image_filename: str
    label_filename: str
    case_directory_pattern: str
    allowed_suffixes: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_safe_filename(self.image_filename, "image_filename")
        _require_safe_filename(self.label_filename, "label_filename")
        if self.image_filename == self.label_filename:
            msg = "image_filename and label_filename must differ"
            raise InvalidIrcadbConventionError(msg)
        _require_safe_case_directory_pattern(self.case_directory_pattern)
        if not isinstance(self.allowed_suffixes, tuple) or not self.allowed_suffixes:
            msg = "allowed_suffixes must be a nonempty tuple"
            raise InvalidIrcadbConventionError(msg)
        if len(set(self.allowed_suffixes)) != len(self.allowed_suffixes):
            msg = "allowed_suffixes must not contain duplicates"
            raise InvalidIrcadbConventionError(msg)
        for suffix in self.allowed_suffixes:
            if suffix not in SUPPORTED_IRCADB_SUFFIXES:
                msg = f"unsupported 3D-IRCADb-style suffix: {suffix!r}"
                raise InvalidIrcadbConventionError(msg)
        if _matching_suffix(self.image_filename, self.allowed_suffixes) is None:
            msg = "image_filename must use an allowed NIfTI suffix"
            raise InvalidIrcadbConventionError(msg)
        if _matching_suffix(self.label_filename, self.allowed_suffixes) is None:
            msg = "label_filename must use an allowed NIfTI suffix"
            raise InvalidIrcadbConventionError(msg)
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
class IrcadbStyleAdapter:
    """Concrete read-only 3D-IRCADb-style synthetic discovery adapter.

    Discovery inventories filesystem names only. It never opens, parses, hashes, memory-maps, or
    validates NIfTI/DICOM contents and does not create anonymous manifests.
    """

    convention: IrcadbFilenameConvention
    adapter_name: str = field(init=False, default=IRCADB_STYLE_ADAPTER_NAME)
    adapter_version: str = field(init=False, default=IRCADB_STYLE_ADAPTER_VERSION)

    def discover(self, dataset_root: Path, layout: AdapterLayoutSpec) -> AdapterInventory:
        """Discover deterministic image/tumor-label candidates beneath an explicit root."""
        canonical_root = validate_explicit_dataset_root(dataset_root)
        image_files = _discover_files(
            canonical_root,
            layout.image_pattern,
            layout.recursive,
            expected_filename=self.convention.image_filename,
            case_directory_pattern=self.convention.case_directory_pattern,
            side="image",
        )
        label_files = _discover_files(
            canonical_root,
            layout.label_pattern,
            layout.recursive,
            expected_filename=self.convention.label_filename,
            case_directory_pattern=self.convention.case_directory_pattern,
            side="label",
        )
        if not image_files and not label_files:
            msg = "3D-IRCADb-style discovery found no matched image or label files"
            raise IrcadbDiscoveryError(msg)
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
            msg = "3D-IRCADb-style image and label files must resolve to unambiguous files"
            raise AmbiguousIrcadbPairingError(msg) from exc
        return inventory


def _discover_files(
    dataset_root: Path,
    pattern: str,
    recursive: bool,
    *,
    expected_filename: str,
    case_directory_pattern: str,
    side: str,
) -> tuple[_DiscoveredFile, ...]:
    relative_paths = _matching_relative_paths(dataset_root, pattern, recursive=recursive)
    discovered: list[_DiscoveredFile] = []
    for relative_path in relative_paths:
        filename = PurePosixPath(relative_path).name
        if filename != expected_filename:
            msg = f"matched {side} file has unexpected basename: {relative_path}"
            raise MalformedIrcadbCaseError(msg)
        source_case_key = _extract_source_case_key(
            relative_path,
            case_directory_pattern=case_directory_pattern,
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
    case_directory_pattern: str,
    side: str,
) -> str:
    relative = PurePosixPath(relative_path)
    matching_case_directories = tuple(
        candidate
        for candidate in _ancestor_directories(relative)
        if _case_directory_matches(candidate, case_directory_pattern)
    )
    if not matching_case_directories:
        msg = f"matched {side} file has no configured case-directory match: {relative_path}"
        raise MalformedIrcadbCaseError(msg)
    if len(matching_case_directories) > 1:
        formatted = ", ".join(str(path) for path in matching_case_directories)
        msg = f"matched {side} file has ambiguous case-directory matches: {formatted}"
        raise AmbiguousIrcadbPairingError(msg)

    source_case_key = matching_case_directories[0].name
    if source_case_key == "":
        msg = f"matched {side} file produced an empty source key: {relative_path}"
        raise MalformedIrcadbCaseError(msg)
    try:
        AdapterCaseCandidate(
            source_case_key=source_case_key,
            image_relative_path="synthetic-image-placeholder.nii",
            label_relative_path="synthetic-label-placeholder.nii",
        )
    except Phase2PathError as exc:
        msg = f"matched {side} file produced an unsafe source key: {relative_path}"
        raise MalformedIrcadbCaseError(msg) from exc
    return source_case_key


def _ancestor_directories(relative_path: PurePosixPath) -> tuple[PurePosixPath, ...]:
    parents = tuple(parent for parent in relative_path.parents if str(parent) != ".")
    return tuple(reversed(parents))


def _case_directory_matches(candidate: PurePosixPath, pattern: str) -> bool:
    if "/" not in pattern:
        return fnmatch.fnmatchcase(candidate.name, pattern)
    return fnmatch.fnmatchcase(candidate.as_posix(), pattern)


def _matching_suffix(filename: str, allowed_suffixes: tuple[str, ...]) -> str | None:
    for suffix in tuple(sorted(allowed_suffixes, key=len, reverse=True)):
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
        raise AmbiguousIrcadbPairingError(msg)


def _require_unique_resolved_files(files: tuple[_DiscoveredFile, ...], side: str) -> None:
    try:
        require_unique_resolved_paths(
            tuple((f"{item.source_case_key}:{side}", item.resolved_path) for item in files)
        )
    except DuplicateResolvedPathError as exc:
        msg = f"multiple {side} files resolve to the same file"
        raise AmbiguousIrcadbPairingError(msg) from exc


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
        raise IncompleteIrcadbPairError(msg)


def _require_safe_filename(value: object, field_name: str) -> None:
    if not isinstance(value, str) or value == "":
        msg = f"{field_name} must be a nonempty basename"
        raise InvalidIrcadbConventionError(msg)
    if value != value.strip():
        msg = f"{field_name} must not contain leading or trailing whitespace"
        raise InvalidIrcadbConventionError(msg)
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        msg = f"{field_name} must not contain NUL or control characters"
        raise InvalidIrcadbConventionError(msg)
    if value in {".", ".."} or "/" in value or "\\" in value or ".." in value:
        msg = f"{field_name} must be a conservative basename, not a path"
        raise InvalidIrcadbConventionError(msg)
    if _URI_PATTERN.match(value) is not None or PureWindowsPath(value).is_absolute():
        msg = f"{field_name} must not be URI-like or drive-letter form"
        raise InvalidIrcadbConventionError(msg)
    if any(character in _GLOB_METACHARACTERS for character in value):
        msg = f"{field_name} must not contain glob metacharacters"
        raise InvalidIrcadbConventionError(msg)
    if _BASENAME_PATTERN.fullmatch(value) is None:
        msg = f"{field_name} contains unsupported characters"
        raise InvalidIrcadbConventionError(msg)


def _require_safe_case_directory_pattern(value: object) -> None:
    if not isinstance(value, str) or value == "":
        msg = "case_directory_pattern must be a nonempty relative POSIX glob pattern"
        raise InvalidIrcadbConventionError(msg)
    if value != value.strip():
        msg = "case_directory_pattern must not contain leading or trailing whitespace"
        raise InvalidIrcadbConventionError(msg)
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        msg = "case_directory_pattern must not contain NUL or control characters"
        raise InvalidIrcadbConventionError(msg)
    if "\\" in value:
        msg = "case_directory_pattern must use POSIX separators"
        raise InvalidIrcadbConventionError(msg)
    if _URI_PATTERN.match(value) is not None or PureWindowsPath(value).is_absolute():
        msg = "case_directory_pattern must not be drive-letter or URI-like"
        raise InvalidIrcadbConventionError(msg)
    if value.startswith("~") or value.startswith("/") or "//" in value:
        msg = "case_directory_pattern must be normalized relative POSIX form"
        raise InvalidIrcadbConventionError(msg)
    if any(character not in _GLOB_ALLOWED_CHARS for character in value):
        msg = "case_directory_pattern contains unsupported glob syntax"
        raise InvalidIrcadbConventionError(msg)
    path = PurePosixPath(value)
    if path.is_absolute() or value in {".", ".."}:
        msg = "case_directory_pattern must be relative and must not be '.' or '..'"
        raise InvalidIrcadbConventionError(msg)
    if any(part in {"", ".", ".."} for part in path.parts):
        msg = "case_directory_pattern must not contain empty, '.', or '..' components"
        raise InvalidIrcadbConventionError(msg)
    if "**" in path.parts or "**" in value:
        msg = "case_directory_pattern must not use recursive '**' glob syntax"
        raise InvalidIrcadbConventionError(msg)
    _require_balanced_glob_classes(value)


def _require_balanced_glob_classes(value: str) -> None:
    in_class = False
    class_has_content = False
    for character in value:
        if character == "[":
            if in_class:
                msg = "case_directory_pattern has nested glob character classes"
                raise InvalidIrcadbConventionError(msg)
            in_class = True
            class_has_content = False
        elif character == "]":
            if not in_class or not class_has_content:
                msg = "case_directory_pattern has an invalid glob character class"
                raise InvalidIrcadbConventionError(msg)
            in_class = False
        elif in_class and character != "!":
            class_has_content = True
    if in_class:
        msg = "case_directory_pattern has an unterminated glob character class"
        raise InvalidIrcadbConventionError(msg)


def _has_glob(value: str) -> bool:
    return any(character in value for character in "*?[")


def _relative_posix_path(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()
