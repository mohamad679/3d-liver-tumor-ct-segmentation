"""Synthetic tests for Phase 8 Wave 2 image-only publication."""

from __future__ import annotations

import struct
import zipfile
from pathlib import Path

import numpy as np
import pytest

from protoem_ct.external.manifest import phase8_external_image_manifest_from_json
from protoem_ct.external.wave2 import (
    PHASE8_WAVE2_MANIFEST_FILENAME,
    PHASE8_WAVE2_QA_FILENAME,
    PHASE8_WAVE2_SUMMARY_FILENAME,
    Phase8Wave2PublicationError,
    run_phase8_wave2_image_inventory,
)


def test_wave2_publication_uses_anonymous_outputs_only(tmp_path: Path) -> None:
    dataset_root = tmp_path / "raw" / "3Dircadb1"
    dataset_archive = tmp_path / "download" / "3Dircadb1.zip"
    output_root = tmp_path / "runs" / "phase8_wave2_inventory_v1"
    repository_root = tmp_path / "repo"
    repository_root.mkdir()
    output_root.parent.mkdir(parents=True)
    dataset_archive.parent.mkdir(parents=True)
    dataset_archive.write_bytes(b"synthetic outer archive")
    _write_synthetic_ircadb_root(dataset_root)

    result = run_phase8_wave2_image_inventory(
        dataset_root=dataset_root,
        dataset_archive=dataset_archive,
        output_root=output_root,
        repository_root=repository_root,
    )

    assert result.case_count == 20
    assert result.qa_pass_count == 20
    assert result.qa_fail_count == 0
    assert result.anonymous_case_reason_codes == ()
    assert sorted(result.artifact_hashes) == [
        PHASE8_WAVE2_MANIFEST_FILENAME,
        PHASE8_WAVE2_SUMMARY_FILENAME,
        PHASE8_WAVE2_QA_FILENAME,
        "phase8_wave2_ircadb_layout.json",
    ]
    manifest = phase8_external_image_manifest_from_json(
        (output_root / PHASE8_WAVE2_MANIFEST_FILENAME).read_bytes()
    )
    assert manifest.manifest_hash == result.manifest.manifest_hash
    text = "\n".join(path.read_text(encoding="utf-8") for path in output_root.iterdir())
    assert "/Volumes" not in text
    assert "PatientName" not in text
    assert "PatientID" not in text
    assert "MASKS_DICOM/" not in text
    assert "LABELLED_DICOM/" not in text
    assert "MESHES_VTK/" not in text


def test_wave2_publication_rejects_repository_output_root(tmp_path: Path) -> None:
    dataset_root = tmp_path / "raw" / "3Dircadb1"
    dataset_archive = tmp_path / "download" / "3Dircadb1.zip"
    repository_root = tmp_path / "repo"
    repository_root.mkdir()
    dataset_archive.parent.mkdir(parents=True)
    dataset_archive.write_bytes(b"synthetic outer archive")
    _write_synthetic_ircadb_root(dataset_root)

    with pytest.raises(Phase8Wave2PublicationError):
        run_phase8_wave2_image_inventory(
            dataset_root=dataset_root,
            dataset_archive=dataset_archive,
            output_root=repository_root / "generated",
            repository_root=repository_root,
        )


def _write_synthetic_ircadb_root(dataset_root: Path) -> None:
    for ordinal in range(1, 21):
        case_dir = dataset_root / f"3Dircadb1.{ordinal}"
        case_dir.mkdir(parents=True)
        (case_dir / "LICENSE.txt").write_text("synthetic license\n", encoding="utf-8")
        (case_dir / "MASKS_DICOM.zip").write_bytes(b"not opened")
        (case_dir / "LABELLED_DICOM.zip").write_bytes(b"not opened")
        (case_dir / "MESHES_VTK.zip").write_bytes(b"not opened")
        (case_dir / f"liver_{ordinal}.jpg").write_bytes(b"not opened")
        with zipfile.ZipFile(case_dir / "PATIENT_DICOM.zip", "w") as archive:
            archive.writestr("slice-000.dcm", _dicom_slice(0))
            archive.writestr("slice-001.dcm", _dicom_slice(1))


def _dicom_slice(index: int) -> bytes:
    pixel_values = np.arange(4, dtype=np.uint16) + np.uint16(index * 10)
    elements: list[tuple[int, int, str, bytes]] = [
        (0x0002, 0x0010, "UI", _ui("1.2.840.10008.1.2.1")),
        (0x0008, 0x0018, "UI", _ui(f"1.2.826.0.1.3680043.10.543.{index}")),
        (0x0008, 0x0060, "CS", _text("CT")),
        (0x0020, 0x000E, "UI", _ui("1.2.826.0.1.3680043.10.543.100")),
        (0x0020, 0x0032, "DS", _ds((0.0, 0.0, float(index) * 2.5))),
        (0x0020, 0x0037, "DS", _ds((1.0, 0.0, 0.0, 0.0, 1.0, 0.0))),
        (0x0028, 0x0010, "US", struct.pack("<H", 2)),
        (0x0028, 0x0011, "US", struct.pack("<H", 2)),
        (0x0028, 0x0030, "DS", _ds((0.7, 0.8))),
        (0x0028, 0x0100, "US", struct.pack("<H", 16)),
        (0x0028, 0x0103, "US", struct.pack("<H", 0)),
        (0x0028, 0x1052, "DS", _text("-1000")),
        (0x0028, 0x1053, "DS", _text("1")),
        (0x7FE0, 0x0010, "OW", pixel_values.tobytes()),
    ]
    return b"\0" * 128 + b"DICM" + b"".join(_element(*item) for item in elements)


def _element(group: int, element: int, vr: str, value: bytes) -> bytes:
    padded = value if len(value) % 2 == 0 else value + b" "
    prefix = struct.pack("<HH", group, element) + vr.encode("ascii")
    if vr == "OW":
        return prefix + b"\0\0" + struct.pack("<I", len(padded)) + padded
    return prefix + struct.pack("<H", len(padded)) + padded


def _text(value: str) -> bytes:
    data = value.encode("ascii")
    return data if len(data) % 2 == 0 else data + b" "


def _ui(value: str) -> bytes:
    data = value.encode("ascii")
    return data if len(data) % 2 == 0 else data + b"\0"


def _ds(values: tuple[float, ...]) -> bytes:
    return _text("\\".join(f"{value:g}" for value in values))
