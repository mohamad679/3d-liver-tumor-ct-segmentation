"""Image-only 3D-IRCADb-01 layout discovery for Phase 8 Wave 2.

This adapter is read-only and records only filesystem layout facts that are safe before external
label access. It never opens DICOM, JPG, mask, labelled, or mesh content.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final

from protoem_ct.data.phase2_paths import (
    InvalidDatasetRootError,
    InvalidRelativeDatasetPathError,
    Phase2PathError,
    normalize_safe_relative_posix_path,
    validate_explicit_dataset_root,
)

IRCADB_PHASE8_ADAPTER_NAME: Final[str] = "phase8_ircadb_image_only"
IRCADB_PHASE8_ADAPTER_VERSION: Final[str] = "v1"
IRCADB_PHASE8_COHORT_IDENTITY: Final[str] = "3d-ircadb-01"
IRCADB_PHASE8_EXPECTED_ROOT_NAME: Final[str] = "3Dircadb1"
IRCADB_PHASE8_EXPECTED_CASE_COUNT: Final[int] = 20
IRCADB_PHASE8_CASE_PREFIX: Final[str] = "3Dircadb1."
IRCADB_PHASE8_PATIENT_ARCHIVE_NAME: Final[str] = "PATIENT_DICOM.zip"
IRCADB_PHASE8_PROHIBITED_ARCHIVE_NAMES: Final[tuple[str, ...]] = (
    "LABELLED_DICOM.zip",
    "MASKS_DICOM.zip",
    "MESHES_VTK.zip",
)
IRCADB_PHASE8_PROHIBITED_JPG_PATTERN: Final[re.Pattern[str]] = re.compile(r"^liver_[0-9]+\.jpg$")

_CASE_NAME_RE = re.compile(r"^3Dircadb1\.(0*[1-9][0-9]*)$")


class Phase8IrcadbAdapterError(Phase2PathError):
    """Base exception for Phase 8 3D-IRCADb-01 adapter failures."""


class Phase8IrcadbRootError(Phase8IrcadbAdapterError):
    """Raised when the explicit 3D-IRCADb-01 root is missing, unsafe, or unexpected."""


class Phase8IrcadbLayoutError(Phase8IrcadbAdapterError):
    """Raised when the discovered 3D-IRCADb-01 layout violates the frozen convention."""


class Phase8IrcadbZipMemberPathError(Phase8IrcadbAdapterError):
    """Raised when a ZIP member path is unsafe to use for later image-only streaming."""


@dataclass(frozen=True, slots=True)
class Phase8IrcadbProhibitedPresence:
    """Presence flags for label-bearing or otherwise prohibited nested files.

    These flags must be derived from outer filesystem names only. The referenced files must not be
    opened, listed internally, extracted, or hashed during Wave 2 adapter discovery.
    """

    masks_dicom_zip: bool
    labelled_dicom_zip: bool
    meshes_vtk_zip: bool
    liver_jpg: bool


@dataclass(frozen=True, slots=True)
class Phase8IrcadbCase:
    """One anonymous image-only 3D-IRCADb-01 case layout record."""

    anonymous_case_id: str
    source_case_ordinal: int
    source_case_directory: str
    patient_dicom_zip_relative_path: str
    prohibited_presence: Phase8IrcadbProhibitedPresence


@dataclass(frozen=True, slots=True)
class Phase8IrcadbInventory:
    """Deterministic image-only inventory of the expected 3D-IRCADb-01 case directories."""

    adapter_name: str
    adapter_version: str
    cohort_identity: str
    expected_case_count: int
    cases: tuple[Phase8IrcadbCase, ...]


def discover_phase8_ircadb_image_layout(dataset_root: Path) -> Phase8IrcadbInventory:
    """Discover the real 3D-IRCADb-01 nested archive layout beneath an explicit root.

    The function requires an absolute caller-provided root named ``3Dircadb1``. It does not search
    parent directories, infer repository-relative locations, or inspect archive internals.
    """

    canonical_root = _validate_ircadb_root(dataset_root)
    case_dirs = _discover_case_directories(canonical_root)
    _require_expected_ordinals(case_dirs)
    cases = tuple(
        _build_case_record(canonical_root, ordinal, case_dir) for ordinal, case_dir in case_dirs
    )
    return Phase8IrcadbInventory(
        adapter_name=IRCADB_PHASE8_ADAPTER_NAME,
        adapter_version=IRCADB_PHASE8_ADAPTER_VERSION,
        cohort_identity=IRCADB_PHASE8_COHORT_IDENTITY,
        expected_case_count=IRCADB_PHASE8_EXPECTED_CASE_COUNT,
        cases=cases,
    )


def normalize_safe_ircadb_zip_member_path(member_name: str) -> str:
    """Normalize a ZIP member path and reject traversal, absolute, or platform-specific forms."""

    try:
        normalized = normalize_safe_relative_posix_path(member_name)
    except InvalidRelativeDatasetPathError as exc:
        msg = "ZIP member path must be a normalized relative POSIX path"
        raise Phase8IrcadbZipMemberPathError(msg) from exc
    if is_ignored_macos_metadata_path(normalized):
        msg = "ZIP member path must not be macOS metadata"
        raise Phase8IrcadbZipMemberPathError(msg)
    return normalized


def is_ignored_macos_metadata_path(relative_path: str) -> bool:
    """Return whether a relative path is known macOS metadata that should be ignored."""

    path = PurePosixPath(relative_path)
    return any(
        part == "__MACOSX" or part.startswith("._") or part == ".DS_Store" for part in path.parts
    )


def _validate_ircadb_root(dataset_root: Path) -> Path:
    try:
        canonical_root = validate_explicit_dataset_root(dataset_root)
    except InvalidDatasetRootError as exc:
        msg = "3D-IRCADb-01 dataset root must be an explicit existing absolute directory"
        raise Phase8IrcadbRootError(msg) from exc
    if canonical_root.name != IRCADB_PHASE8_EXPECTED_ROOT_NAME:
        msg = "3D-IRCADb-01 dataset root must be the extracted 3Dircadb1 directory"
        raise Phase8IrcadbRootError(msg)
    if dataset_root.is_symlink():
        msg = "3D-IRCADb-01 dataset root must not be a symlink"
        raise Phase8IrcadbRootError(msg)
    return canonical_root


def _discover_case_directories(root: Path) -> tuple[tuple[int, Path], ...]:
    ordinals: dict[int, list[Path]] = {}
    unexpected: list[str] = []
    try:
        children = tuple(sorted(root.iterdir(), key=lambda path: path.name))
    except OSError as exc:
        msg = "3D-IRCADb-01 dataset root cannot be listed"
        raise Phase8IrcadbRootError(msg) from exc
    for child in children:
        if is_ignored_macos_metadata_path(child.name):
            continue
        match = _CASE_NAME_RE.fullmatch(child.name)
        if match is None:
            unexpected.append(child.name)
            continue
        if child.is_symlink():
            msg = f"3D-IRCADb-01 case directory must not be a symlink: {child.name}"
            raise Phase8IrcadbLayoutError(msg)
        if not child.is_dir():
            msg = f"3D-IRCADb-01 case entry must be a directory: {child.name}"
            raise Phase8IrcadbLayoutError(msg)
        ordinals.setdefault(int(match.group(1)), []).append(child)
    if unexpected:
        msg = f"3D-IRCADb-01 root contains unexpected entries: {sorted(unexpected)!r}"
        raise Phase8IrcadbLayoutError(msg)
    duplicate_ordinals = sorted(ordinal for ordinal, paths in ordinals.items() if len(paths) > 1)
    if duplicate_ordinals:
        msg = f"3D-IRCADb-01 duplicate case ordinals: {duplicate_ordinals!r}"
        raise Phase8IrcadbLayoutError(msg)
    misspelled = sorted(
        paths[0].name
        for ordinal, paths in ordinals.items()
        if paths[0].name != f"{IRCADB_PHASE8_CASE_PREFIX}{ordinal}"
    )
    if misspelled:
        msg = f"3D-IRCADb-01 case directories use unexpected ordinal spelling: {misspelled!r}"
        raise Phase8IrcadbLayoutError(msg)
    return tuple(sorted((ordinal, paths[0]) for ordinal, paths in ordinals.items()))


def _require_expected_ordinals(case_dirs: tuple[tuple[int, Path], ...]) -> None:
    discovered = {ordinal for ordinal, _case_dir in case_dirs}
    expected = set(range(1, IRCADB_PHASE8_EXPECTED_CASE_COUNT + 1))
    missing = sorted(expected - discovered)
    unexpected = sorted(discovered - expected)
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append(f"missing ordinals: {missing!r}")
        if unexpected:
            details.append(f"unexpected ordinals: {unexpected!r}")
        msg = "; ".join(details)
        raise Phase8IrcadbLayoutError(msg)


def _build_case_record(root: Path, ordinal: int, case_dir: Path) -> Phase8IrcadbCase:
    _require_no_unexpected_case_entries(case_dir)
    patient_archive = _require_regular_nonsymlink_file(case_dir, IRCADB_PHASE8_PATIENT_ARCHIVE_NAME)
    return Phase8IrcadbCase(
        anonymous_case_id=f"ext-ircadb-{ordinal:03d}",
        source_case_ordinal=ordinal,
        source_case_directory=case_dir.name,
        patient_dicom_zip_relative_path=_relative_posix_path(root, patient_archive),
        prohibited_presence=Phase8IrcadbProhibitedPresence(
            masks_dicom_zip=_file_presence_flag(case_dir, "MASKS_DICOM.zip"),
            labelled_dicom_zip=_file_presence_flag(case_dir, "LABELLED_DICOM.zip"),
            meshes_vtk_zip=_file_presence_flag(case_dir, "MESHES_VTK.zip"),
            liver_jpg=any(
                IRCADB_PHASE8_PROHIBITED_JPG_PATTERN.fullmatch(child.name) is not None
                for child in _case_children(case_dir)
                if not is_ignored_macos_metadata_path(child.name)
            ),
        ),
    )


def _require_no_unexpected_case_entries(case_dir: Path) -> None:
    allowed = {
        IRCADB_PHASE8_PATIENT_ARCHIVE_NAME,
        *IRCADB_PHASE8_PROHIBITED_ARCHIVE_NAMES,
        "LICENSE.txt",
    }
    unexpected: list[str] = []
    for child in _case_children(case_dir):
        if is_ignored_macos_metadata_path(child.name):
            continue
        if child.name in allowed or IRCADB_PHASE8_PROHIBITED_JPG_PATTERN.fullmatch(child.name):
            if child.is_symlink():
                msg = f"3D-IRCADb-01 case entry must not be a symlink: {case_dir.name}/{child.name}"
                raise Phase8IrcadbLayoutError(msg)
            continue
        unexpected.append(child.name)
    if unexpected:
        msg = (
            f"3D-IRCADb-01 case {case_dir.name} contains unexpected entries: {sorted(unexpected)!r}"
        )
        raise Phase8IrcadbLayoutError(msg)


def _require_regular_nonsymlink_file(case_dir: Path, filename: str) -> Path:
    path = case_dir / filename
    if path.is_symlink():
        msg = f"3D-IRCADb-01 case entry must not be a symlink: {case_dir.name}/{filename}"
        raise Phase8IrcadbLayoutError(msg)
    if not path.exists():
        msg = f"3D-IRCADb-01 case is missing required PATIENT_DICOM.zip: {case_dir.name}"
        raise Phase8IrcadbLayoutError(msg)
    if not path.is_file():
        msg = f"3D-IRCADb-01 case PATIENT_DICOM.zip is not a regular file: {case_dir.name}"
        raise Phase8IrcadbLayoutError(msg)
    return path


def _file_presence_flag(case_dir: Path, filename: str) -> bool:
    path = case_dir / filename
    if path.is_symlink():
        msg = (
            "3D-IRCADb-01 prohibited archive path must not be a symlink: "
            f"{case_dir.name}/{filename}"
        )
        raise Phase8IrcadbLayoutError(msg)
    if not path.exists():
        return False
    if not path.is_file():
        msg = (
            "3D-IRCADb-01 prohibited archive path is not a regular file: "
            f"{case_dir.name}/{filename}"
        )
        raise Phase8IrcadbLayoutError(msg)
    return True


def _case_children(case_dir: Path) -> tuple[Path, ...]:
    try:
        return tuple(sorted(case_dir.iterdir(), key=lambda path: path.name))
    except OSError as exc:
        msg = f"3D-IRCADb-01 case directory cannot be listed: {case_dir.name}"
        raise Phase8IrcadbLayoutError(msg) from exc


def _relative_posix_path(root: Path, path: Path) -> str:
    relative_path = path.relative_to(root).as_posix()
    return normalize_safe_relative_posix_path(relative_path)
