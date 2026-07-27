"""Base contracts for Phase 2 dataset adapters.

This module does not implement LiTS or 3D-IRCADb discovery. It defines shared, read-only
contracts that concrete adapters must satisfy in later substages.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Protocol, runtime_checkable

from protoem_ct.data.phase2_paths import (
    DuplicateResolvedPathError,
    Phase2PathError,
    normalize_safe_relative_posix_path,
    require_unique_resolved_paths,
    resolve_regular_file_beneath_root,
)

_ADAPTER_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")
_SOURCE_CASE_KEY_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,126}$")
_URI_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
_GLOB_ALLOWED_CHARS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-/*?[]!"
)


class InvalidAdapterLayoutError(Phase2PathError):
    """Raised when adapter layout rules are unsafe or unsupported."""


class InvalidAdapterCandidateError(Phase2PathError):
    """Raised when an ephemeral adapter candidate violates the discovery contract."""


class InvalidAdapterInventoryError(Phase2PathError):
    """Raised when an in-memory adapter inventory is ambiguous or inconsistent."""


@dataclass(frozen=True, slots=True)
class AdapterLayoutSpec:
    """Configurable future-discovery layout rules.

    Supported glob syntax is intentionally conservative: literals, ``*``, ``?``, and simple
    character classes such as ``[0-9]`` or ``[!a-z]``. Recursive discovery is controlled only by the
    ``recursive`` flag; ``**`` is rejected and no globbing is executed in this substage.
    """

    image_pattern: str
    label_pattern: str
    recursive: bool

    def __post_init__(self) -> None:
        _require_safe_glob_pattern(self.image_pattern, "image_pattern")
        _require_safe_glob_pattern(self.label_pattern, "label_pattern")
        if self.image_pattern == self.label_pattern:
            msg = "image_pattern and label_pattern must differ"
            raise InvalidAdapterLayoutError(msg)


@dataclass(frozen=True, slots=True)
class AdapterCaseCandidate:
    """Ephemeral image/label pairing candidate produced by adapter discovery.

    ``source_case_key`` is an internal matching key only. It is not anonymous, is not safe for
    persistence, and must never contain, log, or commit raw or identifying source values.
    """

    source_case_key: str
    image_relative_path: str
    label_relative_path: str

    def __post_init__(self) -> None:
        _require_source_case_key(self.source_case_key)
        image_path = normalize_safe_relative_posix_path(self.image_relative_path)
        label_path = normalize_safe_relative_posix_path(self.label_relative_path)
        if image_path == label_path:
            msg = "image_relative_path and label_relative_path must differ"
            raise InvalidAdapterCandidateError(msg)


@dataclass(frozen=True, slots=True)
class AdapterInventory:
    """In-memory adapter discovery result.

    This is not an anonymous ``DatasetManifest`` and must not be persisted as a Phase 2 artifact.
    """

    adapter_name: str
    adapter_version: str
    candidates: tuple[AdapterCaseCandidate, ...]

    def __post_init__(self) -> None:
        _require_adapter_identifier(self.adapter_name, "adapter_name")
        _require_adapter_identifier(self.adapter_version, "adapter_version")
        if not isinstance(self.candidates, tuple) or not self.candidates:
            msg = "candidates must be a nonempty tuple"
            raise InvalidAdapterInventoryError(msg)
        if any(not isinstance(candidate, AdapterCaseCandidate) for candidate in self.candidates):
            msg = "candidates must contain only AdapterCaseCandidate entries"
            raise InvalidAdapterInventoryError(msg)
        expected = tuple(sorted(self.candidates, key=lambda candidate: candidate.source_case_key))
        if self.candidates != expected:
            msg = "candidates must be sorted by source_case_key"
            raise InvalidAdapterInventoryError(msg)
        _require_unique_candidate_values(
            (candidate.source_case_key for candidate in self.candidates),
            "source_case_key",
        )
        _require_unique_candidate_values(
            (
                (candidate.image_relative_path, candidate.label_relative_path)
                for candidate in self.candidates
            ),
            "image/label pair",
        )
        _require_unique_candidate_values(
            (candidate.image_relative_path for candidate in self.candidates),
            "image_relative_path",
        )
        _require_unique_candidate_values(
            (candidate.label_relative_path for candidate in self.candidates),
            "label_relative_path",
        )


@dataclass(frozen=True, slots=True)
class ResolvedAdapterCase:
    """Resolved regular files for one ephemeral adapter candidate."""

    source_case_key: str
    image_path: Path
    label_path: Path

    def __post_init__(self) -> None:
        _require_source_case_key(self.source_case_key)
        if not isinstance(self.image_path, Path) or not self.image_path.is_absolute():
            msg = "image_path must be an absolute resolved Path"
            raise InvalidAdapterInventoryError(msg)
        if not isinstance(self.label_path, Path) or not self.label_path.is_absolute():
            msg = "label_path must be an absolute resolved Path"
            raise InvalidAdapterInventoryError(msg)
        if self.image_path == self.label_path:
            msg = "image_path and label_path must differ"
            raise InvalidAdapterInventoryError(msg)


@runtime_checkable
class DatasetAdapter(Protocol):
    """Runtime-checkable protocol for read-only dataset discovery adapters.

    Implementations must accept only an explicit validated dataset root, never search outside that
    root, never modify source data, reject ambiguous image/label pairings, return deterministic
    inventory ordering, avoid logging or persisting source identifiers, reject symlink escapes, make
    no assumptions about the current working directory, and perform discovery only. Manifest
    anonymization and QA are separate later steps.
    """

    adapter_name: str
    adapter_version: str

    def discover(self, dataset_root: Path, layout: AdapterLayoutSpec) -> AdapterInventory:
        """Discover candidate image/label pairs beneath an explicit dataset root."""


def resolve_adapter_inventory_files(
    dataset_root: Path,
    inventory: AdapterInventory,
) -> tuple[ResolvedAdapterCase, ...]:
    """Resolve every inventory candidate to regular files without opening or hashing contents."""
    resolved_cases = tuple(
        ResolvedAdapterCase(
            source_case_key=candidate.source_case_key,
            image_path=resolve_regular_file_beneath_root(
                dataset_root,
                candidate.image_relative_path,
            ),
            label_path=resolve_regular_file_beneath_root(
                dataset_root,
                candidate.label_relative_path,
            ),
        )
        for candidate in inventory.candidates
    )

    require_unique_resolved_paths(
        tuple((f"{case.source_case_key}:image", case.image_path) for case in resolved_cases)
    )
    require_unique_resolved_paths(
        tuple((f"{case.source_case_key}:label", case.label_path) for case in resolved_cases)
    )
    try:
        require_unique_resolved_paths(
            tuple(
                item
                for case in resolved_cases
                for item in (
                    (f"{case.source_case_key}:image", case.image_path),
                    (f"{case.source_case_key}:label", case.label_path),
                )
            )
        )
    except DuplicateResolvedPathError as exc:
        msg = "resolved image files and label files must not alias each other"
        raise InvalidAdapterInventoryError(msg) from exc
    return resolved_cases


def _require_adapter_identifier(value: object, field_name: str) -> None:
    if not isinstance(value, str) or _ADAPTER_ID_PATTERN.fullmatch(value) is None:
        msg = f"{field_name} must use lowercase letters, digits, underscores, dots, or hyphens"
        raise InvalidAdapterInventoryError(msg)


def _require_source_case_key(value: object) -> None:
    if not isinstance(value, str) or value.strip() == "":
        msg = "source_case_key must be a nonempty internal matching key"
        raise InvalidAdapterCandidateError(msg)
    if value != value.strip() or any(
        ord(character) < 32 or ord(character) == 127 for character in value
    ):
        msg = "source_case_key must not contain surrounding whitespace or control characters"
        raise InvalidAdapterCandidateError(msg)
    if "/" in value or "\\" in value:
        msg = "source_case_key must not contain path separators"
        raise InvalidAdapterCandidateError(msg)
    if _URI_PATTERN.match(value) is not None:
        msg = "source_case_key must not be URI-like"
        raise InvalidAdapterCandidateError(msg)
    if _SOURCE_CASE_KEY_PATTERN.fullmatch(value) is None:
        msg = "source_case_key contains unsupported characters"
        raise InvalidAdapterCandidateError(msg)


def _require_safe_glob_pattern(value: object, field_name: str) -> None:
    if not isinstance(value, str) or value == "":
        msg = f"{field_name} must be a nonempty relative POSIX glob pattern"
        raise InvalidAdapterLayoutError(msg)
    if value != value.strip():
        msg = f"{field_name} must not contain leading or trailing whitespace"
        raise InvalidAdapterLayoutError(msg)
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        msg = f"{field_name} must not contain NUL or control characters"
        raise InvalidAdapterLayoutError(msg)
    if "\\" in value:
        msg = f"{field_name} must use POSIX separators"
        raise InvalidAdapterLayoutError(msg)
    if _URI_PATTERN.match(value) is not None or PureWindowsPath(value).is_absolute():
        msg = f"{field_name} must not be drive-letter or URI-like"
        raise InvalidAdapterLayoutError(msg)
    if value.startswith("~") or value.startswith("/") or "//" in value:
        msg = f"{field_name} must be normalized relative POSIX form"
        raise InvalidAdapterLayoutError(msg)
    if any(character not in _GLOB_ALLOWED_CHARS for character in value):
        msg = f"{field_name} contains unsupported glob syntax"
        raise InvalidAdapterLayoutError(msg)
    path = PurePosixPath(value)
    if path.is_absolute() or value in {".", ".."}:
        msg = f"{field_name} must be relative and must not be '.' or '..'"
        raise InvalidAdapterLayoutError(msg)
    if any(part in {"", ".", ".."} for part in path.parts):
        msg = f"{field_name} must not contain empty, '.', or '..' components"
        raise InvalidAdapterLayoutError(msg)
    if "**" in path.parts or "**" in value:
        msg = f"{field_name} must not use recursive '**' glob syntax"
        raise InvalidAdapterLayoutError(msg)
    _require_balanced_glob_classes(value, field_name)


def _require_balanced_glob_classes(value: str, field_name: str) -> None:
    in_class = False
    class_has_content = False
    for character in value:
        if character == "[":
            if in_class:
                msg = f"{field_name} has nested glob character classes"
                raise InvalidAdapterLayoutError(msg)
            in_class = True
            class_has_content = False
        elif character == "]":
            if not in_class or not class_has_content:
                msg = f"{field_name} has an invalid glob character class"
                raise InvalidAdapterLayoutError(msg)
            in_class = False
        elif in_class and character != "!":
            class_has_content = True
    if in_class:
        msg = f"{field_name} has an unterminated glob character class"
        raise InvalidAdapterLayoutError(msg)


def _require_unique_candidate_values(values: Iterable[object], field_name: str) -> None:
    seen: set[object] = set()
    for value in values:
        if value in seen:
            msg = f"{field_name} values must be unique"
            raise InvalidAdapterInventoryError(msg)
        seen.add(value)
