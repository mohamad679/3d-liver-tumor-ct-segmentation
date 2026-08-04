"""Synthetic tests for Phase 8 image-only DICOM QA."""

from __future__ import annotations

import hashlib
import struct
import zipfile
from pathlib import Path

import numpy as np

from protoem_ct.external.image_qa import (
    PHASE8_IMAGE_QA_SCHEMA_NAME,
    PHASE8_IMAGE_QA_SCHEMA_VERSION,
    Phase8ImageQaReasonCode,
    hash_phase8_image_qa_result,
    phase8_image_qa_result_to_dict,
    run_patient_dicom_zip_image_qa,
)


def test_patient_dicom_zip_image_qa_passes_for_synthetic_ct(tmp_path: Path) -> None:
    archive = tmp_path / "PATIENT_DICOM.zip"
    _write_patient_zip(archive, [_dicom_slice(index) for index in range(3)])

    result = run_patient_dicom_zip_image_qa(
        patient_dicom_zip=archive,
        anonymous_case_id="ext-ircadb-001",
    )

    assert result.schema_name == PHASE8_IMAGE_QA_SCHEMA_NAME
    assert result.schema_version == PHASE8_IMAGE_QA_SCHEMA_VERSION
    assert result.qa_status == "passed"
    assert result.reason_codes == ()
    assert result.image_archive_sha256 == _sha256_file(archive)
    assert result.image_member_count == 3
    assert result.image_slice_count == 3
    assert result.image_series_count == 1
    assert result.image_series_identity is None
    assert result.rows == 2
    assert result.columns == 2
    assert result.volume_shape == (3, 2, 2)
    assert result.voxel_spacing == (0.7, 0.8, 2.5)
    assert result.intensity_summary == {
        "max_hu": -977.0,
        "mean_hu": -988.5,
        "min_hu": -1000.0,
        "voxel_count": 12,
    }
    assert result.qa_result_hash == hash_phase8_image_qa_result(result)


def test_image_qa_rejects_unsafe_zip_member_without_persisting_member_path(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "PATIENT_DICOM.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../escape.dcm", _dicom_slice(0))

    result = run_patient_dicom_zip_image_qa(
        patient_dicom_zip=archive,
        anonymous_case_id="ext-ircadb-001",
    )

    assert result.qa_status == "failed"
    assert result.reason_codes == (Phase8ImageQaReasonCode.UNSAFE_MEMBER_PATH.value,)
    assert "../escape.dcm" not in str(phase8_image_qa_result_to_dict(result))


def test_image_qa_ignores_apple_double_members(tmp_path: Path) -> None:
    archive = tmp_path / "PATIENT_DICOM.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("._slice-000.dcm", b"apple metadata")
        handle.writestr("slice-000.dcm", _dicom_slice(0))
        handle.writestr("slice-001.dcm", _dicom_slice(1))

    result = run_patient_dicom_zip_image_qa(
        patient_dicom_zip=archive,
        anonymous_case_id="ext-ircadb-001",
    )

    assert result.image_member_count == 2
    assert Phase8ImageQaReasonCode.READABLE_DICOM_FAILED.value not in result.reason_codes


def test_image_qa_records_duplicate_sop_without_persisting_uid(tmp_path: Path) -> None:
    archive = tmp_path / "PATIENT_DICOM.zip"
    duplicate_uid = "1.2.826.0.1.3680043.10.543.8"
    _write_patient_zip(
        archive,
        [
            _dicom_slice(0, sop_instance_uid=duplicate_uid),
            _dicom_slice(1, sop_instance_uid=duplicate_uid),
        ],
    )

    result = run_patient_dicom_zip_image_qa(
        patient_dicom_zip=archive,
        anonymous_case_id="ext-ircadb-001",
    )

    assert result.qa_status == "failed"
    assert Phase8ImageQaReasonCode.DUPLICATE_SOP_INSTANCE_UID.value in result.reason_codes
    assert duplicate_uid not in str(phase8_image_qa_result_to_dict(result))


def test_image_qa_records_header_and_order_failures(tmp_path: Path) -> None:
    archive = tmp_path / "PATIENT_DICOM.zip"
    _write_patient_zip(
        archive,
        [
            _dicom_slice(0),
            _dicom_slice(
                1,
                modality="MR",
                series_instance_uid="1.2.826.0.1.3680043.10.543.900",
                rows=3,
                pixel_spacing=(0.7, 0.9),
                image_orientation_patient=(1.0, 0.0, 0.0, 0.0, -1.0, 0.0),
                rescale_slope=2.0,
            ),
            _dicom_slice(2, image_position_patient=(0.0, 0.0, 2.5)),
        ],
    )

    result = run_patient_dicom_zip_image_qa(
        patient_dicom_zip=archive,
        anonymous_case_id="ext-ircadb-001",
    )

    assert result.qa_status == "failed"
    assert Phase8ImageQaReasonCode.MULTIPLE_IMAGE_SERIES.value in result.reason_codes
    assert Phase8ImageQaReasonCode.NON_CT_MODALITY.value in result.reason_codes
    assert Phase8ImageQaReasonCode.INCONSISTENT_ROWS_COLUMNS.value in result.reason_codes
    assert Phase8ImageQaReasonCode.INCONSISTENT_PIXEL_SPACING.value in result.reason_codes
    assert Phase8ImageQaReasonCode.INCONSISTENT_IMAGE_ORIENTATION.value in result.reason_codes
    assert Phase8ImageQaReasonCode.INCONSISTENT_RESCALING.value in result.reason_codes
    assert Phase8ImageQaReasonCode.DUPLICATE_SLICE_POSITION.value in result.reason_codes


def test_image_qa_records_missing_rescale_and_pixel_data(tmp_path: Path) -> None:
    missing_rescale = tmp_path / "missing-rescale.zip"
    _write_patient_zip(
        missing_rescale,
        [_dicom_slice(0, rescale_slope=None), _dicom_slice(1, rescale_slope=None)],
    )

    missing_rescale_result = run_patient_dicom_zip_image_qa(
        patient_dicom_zip=missing_rescale,
        anonymous_case_id="ext-ircadb-001",
    )

    assert Phase8ImageQaReasonCode.MISSING_RESCALING.value in missing_rescale_result.reason_codes
    assert missing_rescale_result.intensity_summary is None

    missing_pixels = tmp_path / "missing-pixels.zip"
    _write_patient_zip(missing_pixels, [_dicom_slice(0, include_pixel_data=False)])

    missing_pixels_result = run_patient_dicom_zip_image_qa(
        patient_dicom_zip=missing_pixels,
        anonymous_case_id="ext-ircadb-001",
    )

    assert missing_pixels_result.reason_codes == (Phase8ImageQaReasonCode.MISSING_PIXEL_DATA.value,)


def test_image_qa_result_hash_is_self_validating(tmp_path: Path) -> None:
    archive = tmp_path / "PATIENT_DICOM.zip"
    _write_patient_zip(archive, [_dicom_slice(0), _dicom_slice(1)])

    result = run_patient_dicom_zip_image_qa(
        patient_dicom_zip=archive,
        anonymous_case_id="ext-ircadb-001",
    )
    payload = phase8_image_qa_result_to_dict(result)

    assert payload["qa_result_hash"] == result.qa_result_hash
    assert result.qa_result_hash == hash_phase8_image_qa_result(result)
    assert "/Volumes" not in str(payload)
    assert "PatientName" not in str(payload)
    assert "PatientID" not in str(payload)


def _write_patient_zip(path: Path, slices: list[bytes]) -> None:
    with zipfile.ZipFile(path, "w") as handle:
        for index, data in enumerate(slices):
            handle.writestr(f"slice-{index:03d}.dcm", data)


def _dicom_slice(
    index: int,
    *,
    sop_instance_uid: str | None = None,
    series_instance_uid: str = "1.2.826.0.1.3680043.10.543.100",
    modality: str = "CT",
    rows: int = 2,
    columns: int = 2,
    pixel_spacing: tuple[float, float] = (0.7, 0.8),
    image_position_patient: tuple[float, float, float] | None = None,
    image_orientation_patient: tuple[float, float, float, float, float, float] = (
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
    ),
    rescale_slope: float | None = 1.0,
    rescale_intercept: float | None = -1000.0,
    include_pixel_data: bool = True,
) -> bytes:
    position = image_position_patient or (0.0, 0.0, float(index) * 2.5)
    pixel_values = np.arange(rows * columns, dtype=np.uint16) + np.uint16(index * 10)
    elements: list[tuple[int, int, str, bytes]] = [
        (0x0002, 0x0010, "UI", _ui("1.2.840.10008.1.2.1")),
        (0x0008, 0x0018, "UI", _ui(sop_instance_uid or f"1.2.826.0.1.3680043.10.543.{index}")),
        (0x0008, 0x0060, "CS", _text(modality)),
        (0x0020, 0x000E, "UI", _ui(series_instance_uid)),
        (0x0020, 0x0032, "DS", _ds(position)),
        (0x0020, 0x0037, "DS", _ds(image_orientation_patient)),
        (0x0028, 0x0010, "US", struct.pack("<H", rows)),
        (0x0028, 0x0011, "US", struct.pack("<H", columns)),
        (0x0028, 0x0030, "DS", _ds(pixel_spacing)),
        (0x0028, 0x0100, "US", struct.pack("<H", 16)),
        (0x0028, 0x0101, "US", struct.pack("<H", 16)),
        (0x0028, 0x0102, "US", struct.pack("<H", 15)),
        (0x0028, 0x0103, "US", struct.pack("<H", 0)),
    ]
    if rescale_intercept is not None:
        elements.append((0x0028, 0x1052, "DS", _text(str(rescale_intercept))))
    if rescale_slope is not None:
        elements.append((0x0028, 0x1053, "DS", _text(str(rescale_slope))))
    if include_pixel_data:
        elements.append((0x7FE0, 0x0010, "OW", pixel_values.tobytes()))
    return b"\0" * 128 + b"DICM" + b"".join(_element(*item) for item in elements)


def _element(group: int, element: int, vr: str, value: bytes) -> bytes:
    padded = value if len(value) % 2 == 0 else value + b" "
    prefix = struct.pack("<HH", group, element) + vr.encode("ascii")
    if vr in {"OB", "OD", "OF", "OL", "OV", "OW", "SQ", "UC", "UR", "UT", "UN"}:
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


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
