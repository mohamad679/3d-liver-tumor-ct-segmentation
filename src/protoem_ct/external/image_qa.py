"""Image-only DICOM QA for Phase 8 external 3D-IRCADb CT archives.

The public QA surface deliberately exposes only anonymous counts, geometry, aggregate intensity
summaries, and structured reason codes. DICOM identifiers are used only transiently for consistency
checks and duplicate detection; they are never returned by this module.
"""

from __future__ import annotations

import hashlib
import math
import struct
import zipfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Final, cast

import numpy as np

from protoem_ct.artifacts.hashing import JsonValue, sha256_json
from protoem_ct.data.phase2_paths import normalize_safe_relative_posix_path

PHASE8_IMAGE_QA_SCHEMA_NAME: Final[str] = "phase8_external_image_qa"
PHASE8_IMAGE_QA_SCHEMA_VERSION: Final[str] = "v1"

_DICOM_PREAMBLE_SIZE: Final[int] = 128
_EXPLICIT_VR_LONG_LENGTH_VRS: Final[frozenset[str]] = frozenset(
    {"OB", "OD", "OF", "OL", "OV", "OW", "SQ", "UC", "UR", "UT", "UN"}
)
_EXPLICIT_LE_TRANSFER_SYNTAX: Final[str] = "1.2.840.10008.1.2.1"
_IMPLICIT_LE_TRANSFER_SYNTAX: Final[str] = "1.2.840.10008.1.2"
_SUPPORTED_TRANSFER_SYNTAXES: Final[frozenset[str]] = frozenset(
    {_EXPLICIT_LE_TRANSFER_SYNTAX, _IMPLICIT_LE_TRANSFER_SYNTAX}
)
_ALLOWED_DICOM_TAGS: Final[frozenset[tuple[int, int]]] = frozenset(
    {
        (0x0002, 0x0010),  # TransferSyntaxUID
        (0x0008, 0x0018),  # SOPInstanceUID
        (0x0008, 0x0060),  # Modality
        (0x0020, 0x000E),  # SeriesInstanceUID
        (0x0020, 0x0032),  # ImagePositionPatient
        (0x0020, 0x0037),  # ImageOrientationPatient
        (0x0028, 0x0010),  # Rows
        (0x0028, 0x0011),  # Columns
        (0x0028, 0x0030),  # PixelSpacing
        (0x0028, 0x0100),  # BitsAllocated
        (0x0028, 0x0103),  # PixelRepresentation
        (0x0028, 0x1052),  # RescaleIntercept
        (0x0028, 0x1053),  # RescaleSlope
        (0x7FE0, 0x0010),  # PixelData
    }
)


class Phase8ImageQaError(ValueError):
    """Base exception for Phase 8 image-only QA failures."""


class Phase8DicomParseError(Phase8ImageQaError):
    """Raised when an allowlisted DICOM field cannot be parsed safely."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: Phase8ImageQaReasonCode | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code


class Phase8ImageQaReasonCode(StrEnum):
    """Structured image-only QA reason codes for anonymous public artifacts."""

    DUPLICATE_SOP_INSTANCE_UID = "duplicate_sop_instance_uid"
    DUPLICATE_SLICE_POSITION = "duplicate_slice_position"
    EMPTY_IMAGE_ARCHIVE = "empty_image_archive"
    EMPTY_IMAGE_VOLUME = "empty_image_volume"
    INCONSISTENT_IMAGE_ORIENTATION = "inconsistent_image_orientation"
    INCONSISTENT_PIXEL_SPACING = "inconsistent_pixel_spacing"
    INCONSISTENT_RESCALING = "inconsistent_rescaling"
    INCONSISTENT_ROWS_COLUMNS = "inconsistent_rows_columns"
    MISSING_PIXEL_DATA = "missing_pixel_data"
    MISSING_RESCALING = "missing_rescaling"
    MISSING_SLICE_POSITION = "missing_slice_position"
    MULTIPLE_IMAGE_SERIES = "multiple_image_series"
    NON_CT_MODALITY = "non_ct_modality"
    NON_FINITE_HU_PIXELS = "non_finite_hu_pixels"
    NON_MONOTONIC_SLICE_POSITIONS = "non_monotonic_slice_positions"
    PIXEL_DATA_DECODE_FAILED = "pixel_data_decode_failed"
    READABLE_ARCHIVE_FAILED = "readable_archive_failed"
    READABLE_DICOM_FAILED = "readable_dicom_failed"
    SINGLE_SLICE_POSITION_UNDETERMINED = "single_slice_position_undetermined"
    UNSAFE_MEMBER_PATH = "unsafe_member_path"
    UNSUPPORTED_TRANSFER_SYNTAX = "unsupported_transfer_syntax"


@dataclass(frozen=True, slots=True)
class Phase8DicomSlice:
    """Transient safe DICOM slice fields required for image-only QA."""

    member_name: str
    sop_instance_uid: str
    series_instance_uid: str
    modality: str
    rows: int
    columns: int
    pixel_spacing: tuple[float, float]
    image_position_patient: tuple[float, float, float] | None
    image_orientation_patient: tuple[float, float, float, float, float, float] | None
    rescale_slope: float | None
    rescale_intercept: float | None
    bits_allocated: int
    pixel_representation: int
    pixel_data: bytes


@dataclass(frozen=True, slots=True)
class Phase8ImageQaResult:
    """Anonymous image-only QA result for one CT image archive."""

    schema_name: str
    schema_version: str
    anonymous_case_id: str
    qa_status: str
    reason_codes: tuple[str, ...]
    image_archive_sha256: str
    image_member_count: int
    image_slice_count: int
    image_series_count: int
    image_series_identity: str | None
    rows: int | None
    columns: int | None
    volume_shape: tuple[int, int, int] | None
    voxel_spacing: tuple[float, float, float] | None
    intensity_summary: dict[str, float | int] | None
    qa_result_hash: str

    def __post_init__(self) -> None:
        if self.schema_name != PHASE8_IMAGE_QA_SCHEMA_NAME:
            msg = "schema_name must be phase8_external_image_qa"
            raise Phase8ImageQaError(msg)
        if self.schema_version != PHASE8_IMAGE_QA_SCHEMA_VERSION:
            msg = "schema_version must be v1"
            raise Phase8ImageQaError(msg)
        expected_hash = hash_phase8_image_qa_result(self)
        if self.qa_result_hash != expected_hash:
            msg = "qa_result_hash does not match deterministic content"
            raise Phase8ImageQaError(msg)


def run_patient_dicom_zip_image_qa(
    *,
    patient_dicom_zip: Path,
    anonymous_case_id: str,
) -> Phase8ImageQaResult:
    """Evaluate image-only QA over one ``PATIENT_DICOM.zip`` file.

    The function reads only the supplied image archive. It never discovers sibling files and never
    opens masks, labels, meshes, or JPG previews.
    """

    reason_codes: set[Phase8ImageQaReasonCode] = set()
    archive_hash = _sha256_regular_file(patient_dicom_zip)
    try:
        with zipfile.ZipFile(patient_dicom_zip, "r") as archive:
            members = _safe_zip_members(archive)
            slices = tuple(_read_dicom_slice(archive.read(member), member) for member in members)
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile):
        reason_codes.add(Phase8ImageQaReasonCode.READABLE_ARCHIVE_FAILED)
        return _build_result(
            anonymous_case_id=anonymous_case_id,
            image_archive_sha256=archive_hash,
            image_member_count=0,
            slices=(),
            reason_codes=reason_codes,
        )
    except Phase8DicomParseError as exc:
        code = exc.reason_code or Phase8ImageQaReasonCode.READABLE_DICOM_FAILED
        reason_codes.add(code)
        return _build_result(
            anonymous_case_id=anonymous_case_id,
            image_archive_sha256=archive_hash,
            image_member_count=0,
            slices=(),
            reason_codes=reason_codes,
        )
    except Phase8ImageQaError as exc:
        code = (
            Phase8ImageQaReasonCode.UNSAFE_MEMBER_PATH
            if isinstance(
                exc,
                _UnsafeZipMemberError,
            )
            else Phase8ImageQaReasonCode.READABLE_DICOM_FAILED
        )
        reason_codes.add(code)
        return _build_result(
            anonymous_case_id=anonymous_case_id,
            image_archive_sha256=archive_hash,
            image_member_count=0,
            slices=(),
            reason_codes=reason_codes,
        )

    if not slices:
        reason_codes.add(Phase8ImageQaReasonCode.EMPTY_IMAGE_ARCHIVE)
        return _build_result(
            anonymous_case_id=anonymous_case_id,
            image_archive_sha256=archive_hash,
            image_member_count=0,
            slices=slices,
            reason_codes=reason_codes,
        )

    reason_codes.update(_evaluate_header_consistency(slices))
    volume, volume_reasons = _build_hu_volume(slices)
    reason_codes.update(volume_reasons)
    return _build_result(
        anonymous_case_id=anonymous_case_id,
        image_archive_sha256=archive_hash,
        image_member_count=len(slices),
        slices=slices,
        reason_codes=reason_codes,
        hu_volume=volume,
    )


def phase8_image_qa_result_to_dict(result: Phase8ImageQaResult) -> dict[str, JsonValue]:
    """Convert an image QA result to a deterministic JSON-compatible mapping."""

    intensity_summary = (
        cast(dict[str, JsonValue], result.intensity_summary)
        if result.intensity_summary is not None
        else None
    )
    return {
        "anonymous_case_id": result.anonymous_case_id,
        "columns": result.columns,
        "image_archive_sha256": result.image_archive_sha256,
        "image_member_count": result.image_member_count,
        "image_series_count": result.image_series_count,
        "image_series_identity": result.image_series_identity,
        "image_slice_count": result.image_slice_count,
        "intensity_summary": intensity_summary,
        "qa_result_hash": result.qa_result_hash,
        "qa_status": result.qa_status,
        "reason_codes": list(result.reason_codes),
        "rows": result.rows,
        "schema_name": result.schema_name,
        "schema_version": result.schema_version,
        "volume_shape": list(result.volume_shape) if result.volume_shape is not None else None,
        "voxel_spacing": (list(result.voxel_spacing) if result.voxel_spacing is not None else None),
    }


def hash_phase8_image_qa_result(result: Phase8ImageQaResult) -> str:
    """Return the lowercase SHA-256 self-hash for an image QA result."""

    payload = phase8_image_qa_result_to_dict_without_hash(result)
    return sha256_json(payload)


def phase8_image_qa_result_to_dict_without_hash(
    result: Phase8ImageQaResult,
) -> dict[str, JsonValue]:
    """Return the canonical QA payload excluding ``qa_result_hash``."""

    payload = phase8_image_qa_result_to_dict(result)
    payload.pop("qa_result_hash")
    return payload


class _UnsafeZipMemberError(Phase8ImageQaError):
    """Raised when a ZIP member name is unsafe for image-only QA."""


def _build_result(
    *,
    anonymous_case_id: str,
    image_archive_sha256: str,
    image_member_count: int,
    slices: tuple[Phase8DicomSlice, ...],
    reason_codes: set[Phase8ImageQaReasonCode],
    hu_volume: np.ndarray | None = None,
) -> Phase8ImageQaResult:
    ordered_reasons = tuple(sorted(code.value for code in reason_codes))
    qa_status = "passed" if not ordered_reasons else "failed"
    rows = slices[0].rows if slices else None
    columns = slices[0].columns if slices else None
    series_values = tuple(sorted({item.series_instance_uid for item in slices}))
    series_identity = None
    voxel_spacing = _voxel_spacing(slices)
    intensity_summary = _intensity_summary(hu_volume) if hu_volume is not None else None
    volume_shape = (
        cast(tuple[int, int, int], tuple(int(value) for value in hu_volume.shape))
        if hu_volume is not None
        else None
    )
    intensity_summary_payload = (
        cast(dict[str, JsonValue], intensity_summary) if intensity_summary is not None else None
    )
    result_hash = sha256_json(
        {
            "anonymous_case_id": anonymous_case_id,
            "columns": columns,
            "image_archive_sha256": image_archive_sha256,
            "image_member_count": image_member_count,
            "image_series_count": len(series_values),
            "image_series_identity": series_identity,
            "image_slice_count": len(slices),
            "intensity_summary": intensity_summary_payload,
            "qa_status": qa_status,
            "reason_codes": list(ordered_reasons),
            "rows": rows,
            "schema_name": PHASE8_IMAGE_QA_SCHEMA_NAME,
            "schema_version": PHASE8_IMAGE_QA_SCHEMA_VERSION,
            "volume_shape": list(volume_shape) if volume_shape is not None else None,
            "voxel_spacing": list(voxel_spacing) if voxel_spacing is not None else None,
        }
    )
    return Phase8ImageQaResult(
        schema_name=PHASE8_IMAGE_QA_SCHEMA_NAME,
        schema_version=PHASE8_IMAGE_QA_SCHEMA_VERSION,
        anonymous_case_id=anonymous_case_id,
        qa_status=qa_status,
        reason_codes=ordered_reasons,
        image_archive_sha256=image_archive_sha256,
        image_member_count=image_member_count,
        image_slice_count=len(slices),
        image_series_count=len(series_values),
        image_series_identity=series_identity,
        rows=rows,
        columns=columns,
        volume_shape=volume_shape,
        voxel_spacing=voxel_spacing,
        intensity_summary=intensity_summary,
        qa_result_hash=result_hash,
    )


def _evaluate_header_consistency(
    slices: tuple[Phase8DicomSlice, ...],
) -> set[Phase8ImageQaReasonCode]:
    reason_codes: set[Phase8ImageQaReasonCode] = set()
    if len({item.series_instance_uid for item in slices}) != 1:
        reason_codes.add(Phase8ImageQaReasonCode.MULTIPLE_IMAGE_SERIES)
    if len({item.sop_instance_uid for item in slices}) != len(slices):
        reason_codes.add(Phase8ImageQaReasonCode.DUPLICATE_SOP_INSTANCE_UID)
    if any(item.modality != "CT" for item in slices):
        reason_codes.add(Phase8ImageQaReasonCode.NON_CT_MODALITY)
    if len({(item.rows, item.columns) for item in slices}) != 1:
        reason_codes.add(Phase8ImageQaReasonCode.INCONSISTENT_ROWS_COLUMNS)
    if len({_rounded_tuple(item.pixel_spacing) for item in slices}) != 1:
        reason_codes.add(Phase8ImageQaReasonCode.INCONSISTENT_PIXEL_SPACING)
    orientations = [item.image_orientation_patient for item in slices]
    if (
        any(item is None for item in orientations)
        or len({_rounded_tuple(item) for item in orientations if item is not None}) != 1
    ):
        reason_codes.add(Phase8ImageQaReasonCode.INCONSISTENT_IMAGE_ORIENTATION)
    if any(item.rescale_slope is None or item.rescale_intercept is None for item in slices):
        reason_codes.add(Phase8ImageQaReasonCode.MISSING_RESCALING)
    elif len({(item.rescale_slope, item.rescale_intercept) for item in slices}) != 1:
        reason_codes.add(Phase8ImageQaReasonCode.INCONSISTENT_RESCALING)

    positions = _slice_axis_positions(slices)
    if positions is None:
        reason_codes.add(Phase8ImageQaReasonCode.MISSING_SLICE_POSITION)
    elif len(positions) == 1:
        reason_codes.add(Phase8ImageQaReasonCode.SINGLE_SLICE_POSITION_UNDETERMINED)
    else:
        sorted_positions = sorted(positions)
        deltas = [
            right - left
            for left, right in zip(sorted_positions, sorted_positions[1:], strict=False)
        ]
        if any(abs(delta) <= 1e-6 for delta in deltas):
            reason_codes.add(Phase8ImageQaReasonCode.DUPLICATE_SLICE_POSITION)
        if any(delta <= 0.0 for delta in deltas):
            reason_codes.add(Phase8ImageQaReasonCode.NON_MONOTONIC_SLICE_POSITIONS)
    return reason_codes


def _build_hu_volume(
    slices: tuple[Phase8DicomSlice, ...],
) -> tuple[np.ndarray | None, set[Phase8ImageQaReasonCode]]:
    reason_codes: set[Phase8ImageQaReasonCode] = set()
    if not slices:
        return None, reason_codes
    if len({(item.rows, item.columns) for item in slices}) != 1:
        return None, reason_codes
    if any(item.rescale_slope is None or item.rescale_intercept is None for item in slices):
        return None, reason_codes

    arrays: list[np.ndarray] = []
    for item in sorted(slices, key=lambda item: _slice_sort_key(item)):
        try:
            raw = _pixel_array(item)
        except Phase8ImageQaError:
            reason_codes.add(Phase8ImageQaReasonCode.PIXEL_DATA_DECODE_FAILED)
            return None, reason_codes
        rescale_slope = item.rescale_slope
        rescale_intercept = item.rescale_intercept
        if rescale_slope is None or rescale_intercept is None:
            return None, reason_codes
        hu = raw.astype(np.float64) * float(rescale_slope) + float(rescale_intercept)
        if not np.isfinite(hu).all():
            reason_codes.add(Phase8ImageQaReasonCode.NON_FINITE_HU_PIXELS)
            return None, reason_codes
        arrays.append(hu)
    volume = np.stack(arrays, axis=0)
    if volume.size == 0:
        reason_codes.add(Phase8ImageQaReasonCode.EMPTY_IMAGE_VOLUME)
        return None, reason_codes
    return volume, reason_codes


def _pixel_array(item: Phase8DicomSlice) -> np.ndarray:
    if item.bits_allocated != 16:
        msg = "only 16-bit DICOM pixel data is supported"
        raise Phase8DicomParseError(msg)
    dtype = np.dtype("<i2") if item.pixel_representation == 1 else np.dtype("<u2")
    expected_values = item.rows * item.columns
    expected_bytes = expected_values * dtype.itemsize
    if len(item.pixel_data) < expected_bytes:
        msg = "pixel data is shorter than rows*columns"
        raise Phase8DicomParseError(msg)
    array = np.frombuffer(item.pixel_data[:expected_bytes], dtype=dtype)
    return array.reshape((item.rows, item.columns))


def _read_dicom_slice(data: bytes, member_name: str) -> Phase8DicomSlice:
    elements = _parse_dicom_elements(data)
    transfer_syntax = _text(elements, (0x0002, 0x0010)) or _EXPLICIT_LE_TRANSFER_SYNTAX
    if transfer_syntax not in _SUPPORTED_TRANSFER_SYNTAXES:
        msg = "unsupported transfer syntax"
        raise Phase8DicomParseError(
            msg,
            reason_code=Phase8ImageQaReasonCode.UNSUPPORTED_TRANSFER_SYNTAX,
        )
    pixel_data = elements.get((0x7FE0, 0x0010))
    if pixel_data is None:
        msg = "pixel data is missing"
        raise Phase8DicomParseError(
            msg,
            reason_code=Phase8ImageQaReasonCode.MISSING_PIXEL_DATA,
        )
    return Phase8DicomSlice(
        member_name=member_name,
        sop_instance_uid=_required_text(elements, (0x0008, 0x0018), "SOPInstanceUID"),
        series_instance_uid=_required_text(elements, (0x0020, 0x000E), "SeriesInstanceUID"),
        modality=_required_text(elements, (0x0008, 0x0060), "Modality"),
        rows=_required_uint16(elements, (0x0028, 0x0010), "Rows"),
        columns=_required_uint16(elements, (0x0028, 0x0011), "Columns"),
        pixel_spacing=cast(
            tuple[float, float],
            _required_float_tuple(elements, (0x0028, 0x0030), 2, "PixelSpacing"),
        ),
        image_position_patient=cast(
            tuple[float, float, float] | None,
            _optional_float_tuple(elements, (0x0020, 0x0032), 3),
        ),
        image_orientation_patient=cast(
            tuple[float, float, float, float, float, float] | None,
            _optional_float_tuple(elements, (0x0020, 0x0037), 6),
        ),
        rescale_slope=_optional_float(elements, (0x0028, 0x1053)),
        rescale_intercept=_optional_float(elements, (0x0028, 0x1052)),
        bits_allocated=_required_uint16(elements, (0x0028, 0x0100), "BitsAllocated"),
        pixel_representation=_required_uint16(
            elements,
            (0x0028, 0x0103),
            "PixelRepresentation",
        ),
        pixel_data=pixel_data,
    )


def _parse_dicom_elements(data: bytes) -> dict[tuple[int, int], bytes]:
    offset = 0
    if len(data) >= _DICOM_PREAMBLE_SIZE + 4 and data[_DICOM_PREAMBLE_SIZE:132] == b"DICM":
        offset = 132
    elements: dict[tuple[int, int], bytes] = {}
    explicit_vr = True
    while offset + 8 <= len(data):
        group, element = struct.unpack_from("<HH", data, offset)
        offset += 4
        if group == 0xFFFE:
            break
        if explicit_vr:
            vr_bytes = data[offset : offset + 2]
            vr = vr_bytes.decode("ascii", errors="ignore")
            offset += 2
            if vr in _EXPLICIT_VR_LONG_LENGTH_VRS:
                offset += 2
                if offset + 4 > len(data):
                    break
                length = struct.unpack_from("<I", data, offset)[0]
                offset += 4
            elif vr.isalpha():
                if offset + 2 > len(data):
                    break
                length = struct.unpack_from("<H", data, offset)[0]
                offset += 2
            else:
                explicit_vr = False
                offset -= 2
                if offset + 4 > len(data):
                    break
                length = struct.unpack_from("<I", data, offset)[0]
                offset += 4
        else:
            length = struct.unpack_from("<I", data, offset)[0]
            offset += 4
        if length == 0xFFFFFFFF or offset + length > len(data):
            break
        if (group, element) in _ALLOWED_DICOM_TAGS:
            elements[(group, element)] = data[offset : offset + length]
        offset += length
        if offset % 2 == 1:
            offset += 1
    if not elements:
        msg = "DICOM file contains no parseable allowlisted elements"
        raise Phase8DicomParseError(msg)
    return elements


def _safe_zip_members(archive: zipfile.ZipFile) -> tuple[str, ...]:
    members: list[str] = []
    for info in archive.infolist():
        name = info.filename
        if name.endswith("/") or _is_apple_double_member(name):
            continue
        try:
            normalized = normalize_safe_relative_posix_path(name)
        except ValueError as exc:
            msg = "unsafe ZIP member path"
            raise _UnsafeZipMemberError(msg) from exc
        if PureWindowsPath(normalized).is_absolute() or PurePosixPath(normalized).is_absolute():
            msg = "unsafe ZIP member path"
            raise _UnsafeZipMemberError(msg)
        members.append(normalized)
    return tuple(sorted(members))


def _is_apple_double_member(name: str) -> bool:
    return any(part.startswith("._") or part == "__MACOSX" for part in PurePosixPath(name).parts)


def _sha256_regular_file(path: Path) -> str:
    if not path.is_file():
        msg = "patient_dicom_zip must be a regular file"
        raise Phase8ImageQaError(msg)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _required_text(
    elements: dict[tuple[int, int], bytes],
    tag: tuple[int, int],
    name: str,
) -> str:
    value = _text(elements, tag)
    if value is None or value == "":
        msg = f"required DICOM field is missing: {name}"
        raise Phase8DicomParseError(msg)
    return value


def _text(elements: dict[tuple[int, int], bytes], tag: tuple[int, int]) -> str | None:
    raw = elements.get(tag)
    if raw is None:
        return None
    return raw.rstrip(b" \0").decode("ascii", errors="strict")


def _required_uint16(
    elements: dict[tuple[int, int], bytes],
    tag: tuple[int, int],
    name: str,
) -> int:
    raw = elements.get(tag)
    if raw is None or len(raw) < 2:
        msg = f"required DICOM uint16 field is missing: {name}"
        raise Phase8DicomParseError(msg)
    return int(struct.unpack_from("<H", raw, 0)[0])


def _required_float_tuple(
    elements: dict[tuple[int, int], bytes],
    tag: tuple[int, int],
    count: int,
    name: str,
) -> tuple[float, ...]:
    value = _optional_float_tuple(elements, tag, count)
    if value is None:
        msg = f"required DICOM float tuple field is missing: {name}"
        raise Phase8DicomParseError(msg)
    return value


def _optional_float_tuple(
    elements: dict[tuple[int, int], bytes],
    tag: tuple[int, int],
    count: int,
) -> tuple[float, ...] | None:
    value = _text(elements, tag)
    if value is None:
        return None
    parts = value.split("\\")
    if len(parts) != count:
        return None
    floats = tuple(float(part) for part in parts)
    if not all(math.isfinite(item) for item in floats):
        return None
    return floats


def _optional_float(elements: dict[tuple[int, int], bytes], tag: tuple[int, int]) -> float | None:
    value = _text(elements, tag)
    if value is None or value == "":
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _rounded_tuple(values: tuple[float, ...], digits: int = 6) -> tuple[float, ...]:
    return tuple(round(value, digits) for value in values)


def _slice_axis_positions(slices: tuple[Phase8DicomSlice, ...]) -> list[float] | None:
    if any(item.image_position_patient is None for item in slices):
        return None
    orientation = slices[0].image_orientation_patient
    if orientation is None:
        return None
    row = np.asarray(orientation[:3], dtype=np.float64)
    column = np.asarray(orientation[3:], dtype=np.float64)
    normal = np.cross(row, column)
    if not np.isfinite(normal).all() or float(np.linalg.norm(normal)) == 0.0:
        return None
    return [
        float(np.dot(np.asarray(item.image_position_patient, dtype=np.float64), normal))
        for item in slices
        if item.image_position_patient is not None
    ]


def _slice_sort_key(item: Phase8DicomSlice) -> tuple[int, float | str]:
    position = _slice_axis_positions((item,))
    if position is not None:
        return (0, position[0])
    return (1, item.member_name)


def _voxel_spacing(slices: tuple[Phase8DicomSlice, ...]) -> tuple[float, float, float] | None:
    if not slices:
        return None
    positions = _slice_axis_positions(slices)
    if positions is None or len(positions) < 2:
        return None
    sorted_positions = sorted(positions)
    deltas = [
        right - left for left, right in zip(sorted_positions, sorted_positions[1:], strict=False)
    ]
    positive = [delta for delta in deltas if delta > 1e-6]
    if not positive:
        return None
    slice_spacing = float(np.median(np.asarray(positive, dtype=np.float64)))
    row_spacing, column_spacing = slices[0].pixel_spacing
    return (float(row_spacing), float(column_spacing), slice_spacing)


def _intensity_summary(volume: np.ndarray) -> dict[str, float | int]:
    return {
        "max_hu": float(np.max(volume)),
        "mean_hu": float(np.mean(volume)),
        "min_hu": float(np.min(volume)),
        "voxel_count": int(volume.size),
    }
