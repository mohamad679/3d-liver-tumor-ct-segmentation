"""Synthetic tests for Phase 8 image-only 3D-IRCADb-01 discovery."""

from __future__ import annotations

from pathlib import Path

import pytest

from protoem_ct.external.ircadb import (
    IRCADB_PHASE8_ADAPTER_NAME,
    IRCADB_PHASE8_ADAPTER_VERSION,
    IRCADB_PHASE8_COHORT_IDENTITY,
    IRCADB_PHASE8_EXPECTED_CASE_COUNT,
    IRCADB_PHASE8_EXPECTED_ROOT_NAME,
    Phase8IrcadbLayoutError,
    Phase8IrcadbRootError,
    Phase8IrcadbZipMemberPathError,
    discover_phase8_ircadb_image_layout,
    is_ignored_macos_metadata_path,
    normalize_safe_ircadb_zip_member_path,
)


def _write_file(path: Path, body: bytes = b"synthetic placeholder\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)


def _make_root(tmp_path: Path) -> Path:
    root = tmp_path / IRCADB_PHASE8_EXPECTED_ROOT_NAME
    root.mkdir(parents=True)
    return root


def _make_case(root: Path, ordinal: int) -> None:
    case_dir = root / f"3Dircadb1.{ordinal}"
    _write_file(case_dir / "PATIENT_DICOM.zip")
    _write_file(case_dir / "MASKS_DICOM.zip", b"must not be opened\n")
    _write_file(case_dir / "LABELLED_DICOM.zip", b"must not be opened\n")
    _write_file(case_dir / "MESHES_VTK.zip", b"must not be opened\n")
    _write_file(case_dir / "LICENSE.txt")
    _write_file(case_dir / f"liver_{ordinal}.jpg", b"must not be opened\n")


def _make_complete_layout(tmp_path: Path) -> Path:
    root = _make_root(tmp_path)
    for ordinal in range(1, IRCADB_PHASE8_EXPECTED_CASE_COUNT + 1):
        _make_case(root, ordinal)
    return root


def test_complete_expected_layout_discovers_anonymous_image_only_inventory(tmp_path: Path) -> None:
    root = _make_complete_layout(tmp_path)
    _write_file(root / "._3Dircadb1")
    _write_file(root / "3Dircadb1.1" / "._MASKS_DICOM.zip")

    inventory = discover_phase8_ircadb_image_layout(root)

    assert inventory.adapter_name == IRCADB_PHASE8_ADAPTER_NAME
    assert inventory.adapter_version == IRCADB_PHASE8_ADAPTER_VERSION
    assert inventory.cohort_identity == IRCADB_PHASE8_COHORT_IDENTITY
    assert inventory.expected_case_count == IRCADB_PHASE8_EXPECTED_CASE_COUNT
    assert len(inventory.cases) == 20
    assert inventory.cases[0].anonymous_case_id == "ext-ircadb-001"
    assert inventory.cases[-1].anonymous_case_id == "ext-ircadb-020"
    assert inventory.cases[0].source_case_ordinal == 1
    assert inventory.cases[0].source_case_directory == "3Dircadb1.1"
    assert inventory.cases[0].patient_dicom_zip_relative_path == ("3Dircadb1.1/PATIENT_DICOM.zip")
    assert inventory.cases[0].prohibited_presence.masks_dicom_zip is True
    assert inventory.cases[0].prohibited_presence.labelled_dicom_zip is True
    assert inventory.cases[0].prohibited_presence.meshes_vtk_zip is True
    assert inventory.cases[0].prohibited_presence.liver_jpg is True


def test_explicit_root_is_required_and_must_be_extracted_dataset_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _make_complete_layout(tmp_path)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(Phase8IrcadbRootError, match="explicit"):
        discover_phase8_ircadb_image_layout(Path(IRCADB_PHASE8_EXPECTED_ROOT_NAME))
    with pytest.raises(Phase8IrcadbRootError, match="3Dircadb1"):
        discover_phase8_ircadb_image_layout(tmp_path)
    with pytest.raises(Phase8IrcadbRootError, match="explicit"):
        discover_phase8_ircadb_image_layout(tmp_path / "missing" / IRCADB_PHASE8_EXPECTED_ROOT_NAME)

    assert discover_phase8_ircadb_image_layout(root).cases[0].anonymous_case_id == "ext-ircadb-001"


def test_missing_unexpected_duplicate_and_misspelled_ordinals_are_rejected(
    tmp_path: Path,
) -> None:
    missing_root = _make_complete_layout(tmp_path / "missing")
    (missing_root / "3Dircadb1.20" / "PATIENT_DICOM.zip").unlink()

    with pytest.raises(Phase8IrcadbLayoutError, match="missing required PATIENT_DICOM"):
        discover_phase8_ircadb_image_layout(missing_root)

    missing_ordinal_root = _make_complete_layout(tmp_path / "missing-ordinal")
    (missing_ordinal_root / "3Dircadb1.20").rename(missing_ordinal_root / "._3Dircadb1.20")
    with pytest.raises(Phase8IrcadbLayoutError, match="missing ordinals"):
        discover_phase8_ircadb_image_layout(missing_ordinal_root)

    unexpected_root = _make_complete_layout(tmp_path / "unexpected")
    (unexpected_root / "3Dircadb1.21").mkdir()
    with pytest.raises(Phase8IrcadbLayoutError, match="unexpected ordinals"):
        discover_phase8_ircadb_image_layout(unexpected_root)

    duplicate_root = _make_complete_layout(tmp_path / "duplicate")
    (duplicate_root / "3Dircadb1.01").mkdir()
    with pytest.raises(Phase8IrcadbLayoutError, match="duplicate case ordinals"):
        discover_phase8_ircadb_image_layout(duplicate_root)

    misspelled_root = _make_complete_layout(tmp_path / "misspelled")
    (misspelled_root / "3Dircadb1.1").rename(misspelled_root / "3Dircadb1.01")
    with pytest.raises(
        Phase8IrcadbLayoutError, match="missing ordinals|unexpected ordinal spelling"
    ):
        discover_phase8_ircadb_image_layout(misspelled_root)


def test_unexpected_root_or_case_entries_are_rejected_but_appledouble_is_ignored(
    tmp_path: Path,
) -> None:
    root = _make_complete_layout(tmp_path)
    _write_file(root / "README.txt")
    with pytest.raises(Phase8IrcadbLayoutError, match="unexpected entries"):
        discover_phase8_ircadb_image_layout(root)

    apple_root = _make_complete_layout(tmp_path / "apple")
    _write_file(apple_root / "3Dircadb1.2" / "unexpected.txt")
    with pytest.raises(Phase8IrcadbLayoutError, match="unexpected entries"):
        discover_phase8_ircadb_image_layout(apple_root)

    ignored_root = _make_complete_layout(tmp_path / "ignored")
    _write_file(ignored_root / "._3Dircadb1")
    _write_file(ignored_root / "3Dircadb1.2" / "._unexpected.txt")
    assert len(discover_phase8_ircadb_image_layout(ignored_root).cases) == 20


@pytest.mark.skipif(not hasattr(Path, "symlink_to"), reason="symlinks unsupported")
def test_symlinks_are_rejected_for_root_case_and_case_entries(tmp_path: Path) -> None:
    root = _make_complete_layout(tmp_path)
    outside = tmp_path / "outside.zip"
    _write_file(outside)
    patient_archive = root / "3Dircadb1.1" / "PATIENT_DICOM.zip"
    patient_archive.unlink()
    try:
        patient_archive.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks unavailable on this filesystem")

    with pytest.raises(Phase8IrcadbLayoutError, match="symlink"):
        discover_phase8_ircadb_image_layout(root)

    root_with_case_link = _make_complete_layout(tmp_path / "case-link")
    (root_with_case_link / "3Dircadb1.20").rename(root_with_case_link / "real-case")
    (root_with_case_link / "3Dircadb1.20").symlink_to(
        root_with_case_link / "real-case",
        target_is_directory=True,
    )
    with pytest.raises(Phase8IrcadbLayoutError, match="symlink"):
        discover_phase8_ircadb_image_layout(root_with_case_link)


def test_zip_member_path_safety_helper_rejects_traversal_and_metadata() -> None:
    assert normalize_safe_ircadb_zip_member_path("series/slice001.dcm") == "series/slice001.dcm"
    assert is_ignored_macos_metadata_path("series/._slice001.dcm") is True
    assert is_ignored_macos_metadata_path("__MACOSX/slice001.dcm") is True

    for unsafe in (
        "../slice001.dcm",
        "series/../slice001.dcm",
        "/series/slice001.dcm",
        "series\\slice001.dcm",
        "file://series/slice001.dcm",
        "C:/series/slice001.dcm",
        "series/._slice001.dcm",
        "__MACOSX/slice001.dcm",
    ):
        with pytest.raises(Phase8IrcadbZipMemberPathError):
            normalize_safe_ircadb_zip_member_path(unsafe)
