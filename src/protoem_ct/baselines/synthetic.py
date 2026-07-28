"""Deterministic Phase 3 synthetic fixture generation and artifact contracts."""

from __future__ import annotations

import gzip
import io
import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast

import nibabel as nib
import numpy as np
import numpy.typing as npt

from protoem_ct.artifacts.hashing import canonical_json_bytes, sha256_file, sha256_json

BASELINE_SYNTHETIC_FIXTURE_VERSION = "baseline_synthetic_fixture_v1"
BASELINE_SYNTHETIC_DATASET_NAME = "Dataset901_ProtoEMCTSynthetic"
BASELINE_SYNTHETIC_DATASET_ID = 901
BASELINE_SYNTHETIC_FILE_ENDING = ".nii.gz"
BASELINE_SYNTHETIC_SHAPE: tuple[int, int, int] = (24, 24, 16)
BASELINE_SYNTHETIC_SPACING_MM: tuple[float, float, float] = (1.0, 1.0, 2.5)
BASELINE_SYNTHETIC_ORIGIN_MM: tuple[float, float, float] = (12.0, -8.0, 5.0)
BASELINE_SYNTHETIC_AFFINE: tuple[tuple[float, float, float, float], ...] = (
    (1.0, 0.0, 0.0, 12.0),
    (0.0, 1.0, 0.0, -8.0),
    (0.0, 0.0, 2.5, 5.0),
    (0.0, 0.0, 0.0, 1.0),
)
BASELINE_SYNTHETIC_TRAINING_CASE_IDENTIFIERS: tuple[str, ...] = (
    "synthetic_train_001",
    "synthetic_train_002",
    "synthetic_train_003",
    "synthetic_train_004",
    "synthetic_train_005",
)
BASELINE_SYNTHETIC_TEST_CASE_IDENTIFIERS: tuple[str, ...] = (
    "synthetic_test_001",
    "synthetic_test_002",
)
_CASE_IDENTIFIER_PATTERN = __import__("re").compile(r"^[a-z0-9](?:[a-z0-9_-]{0,62}[a-z0-9])?$")
_SHA256_PATTERN = __import__("re").compile(r"^[0-9a-f]{64}$")
_WINDOWS_ABSOLUTE_PATTERN = __import__("re").compile(r"^[A-Za-z]:[\\/]")
_ALLOWED_ROLES = {
    "dataset_metadata",
    "training_image",
    "training_label",
    "test_image",
    "test_label",
}
_ALLOWED_ARTIFACT_FIELDS = {
    "affine",
    "artifact_hash",
    "contract_version",
    "dataset_id",
    "dataset_name",
    "file_records",
    "image_shape",
    "test_case_identifiers",
    "training_case_identifiers",
    "voxel_spacing_mm",
}
_ALLOWED_FILE_RECORD_FIELDS = {
    "byte_size",
    "case_identifier",
    "channel_index",
    "relative_path",
    "role",
    "sha256",
}
_FIXED_GZIP_MTIME = 0

SyntheticFixtureRole = Literal[
    "dataset_metadata",
    "training_image",
    "training_label",
    "test_image",
    "test_label",
]

ImageArray = npt.NDArray[np.int16]
LabelArray = npt.NDArray[np.uint8]
AffineArray = npt.NDArray[np.float64]


class BaselineSyntheticFixtureError(ValueError):
    """Base error for Phase 3 synthetic fixture generation and parsing."""


class BaselineSyntheticFixtureValidationError(BaselineSyntheticFixtureError):
    """Raised when fixture inputs or persisted content violate the contract."""


class BaselineSyntheticFixtureSerializationError(BaselineSyntheticFixtureError):
    """Raised when a fixture artifact cannot be serialized or parsed safely."""


class BaselineSyntheticFixtureVersionError(BaselineSyntheticFixtureError):
    """Raised when a fixture artifact uses an unsupported contract version."""


class BaselineSyntheticFixtureHashError(BaselineSyntheticFixtureError):
    """Raised when a persisted fixture artifact hash does not match its content."""


@dataclass(frozen=True, slots=True)
class BaselineSyntheticFixtureFileRecord:
    """Immutable description of one generated fixture file."""

    relative_path: str
    role: SyntheticFixtureRole
    case_identifier: str | None
    channel_index: int | None
    sha256: str
    byte_size: int

    def __post_init__(self) -> None:
        _require_relative_posix_path(self.relative_path)
        if self.role not in _ALLOWED_ROLES:
            raise BaselineSyntheticFixtureValidationError(f"Unsupported file role: {self.role!r}.")
        if self.case_identifier is not None:
            _require_case_identifier(self.case_identifier)
        if self.role == "dataset_metadata":
            if self.case_identifier is not None or self.channel_index is not None:
                raise BaselineSyntheticFixtureValidationError(
                    "dataset_metadata records must not contain case_identifier or channel_index."
                )
        elif self.role.endswith("_image"):
            if self.case_identifier is None or self.channel_index != 0:
                raise BaselineSyntheticFixtureValidationError(
                    "Image records must contain case_identifier and channel_index 0."
                )
        else:
            if self.case_identifier is None or self.channel_index is not None:
                raise BaselineSyntheticFixtureValidationError(
                    "Label records must contain case_identifier and null channel_index."
                )
        _require_sha256(self.sha256, field_name="sha256")
        if self.byte_size <= 0:
            raise BaselineSyntheticFixtureValidationError("byte_size must be positive.")
        _require_no_absolute_path_string(self.relative_path, field_name="relative_path")


@dataclass(frozen=True, slots=True)
class BaselineSyntheticFixtureArtifact:
    """Immutable deterministic manifest for the fixed Phase 3 synthetic fixture."""

    artifact_hash: str
    contract_version: str
    dataset_name: str
    dataset_id: int
    image_shape: tuple[int, int, int]
    voxel_spacing_mm: tuple[float, float, float]
    affine: tuple[tuple[float, float, float, float], ...]
    training_case_identifiers: tuple[str, ...]
    test_case_identifiers: tuple[str, ...]
    file_records: tuple[BaselineSyntheticFixtureFileRecord, ...]

    def __post_init__(self) -> None:
        if self.contract_version != BASELINE_SYNTHETIC_FIXTURE_VERSION:
            raise BaselineSyntheticFixtureVersionError(
                f"Unsupported synthetic fixture version: {self.contract_version!r}."
            )
        _require_sha256(self.artifact_hash, field_name="artifact_hash")
        if self.dataset_name != BASELINE_SYNTHETIC_DATASET_NAME:
            raise BaselineSyntheticFixtureValidationError(
                f"dataset_name must equal {BASELINE_SYNTHETIC_DATASET_NAME!r}."
            )
        if self.dataset_id != BASELINE_SYNTHETIC_DATASET_ID:
            raise BaselineSyntheticFixtureValidationError(
                f"dataset_id must equal {BASELINE_SYNTHETIC_DATASET_ID}."
            )
        if self.image_shape != BASELINE_SYNTHETIC_SHAPE:
            raise BaselineSyntheticFixtureValidationError(
                f"image_shape must equal {BASELINE_SYNTHETIC_SHAPE!r}."
            )
        if tuple(float(value) for value in self.voxel_spacing_mm) != BASELINE_SYNTHETIC_SPACING_MM:
            raise BaselineSyntheticFixtureValidationError(
                f"voxel_spacing_mm must equal {BASELINE_SYNTHETIC_SPACING_MM!r}."
            )
        if self.affine != BASELINE_SYNTHETIC_AFFINE:
            raise BaselineSyntheticFixtureValidationError(
                "affine must equal the fixed synthetic affine."
            )
        if self.training_case_identifiers != BASELINE_SYNTHETIC_TRAINING_CASE_IDENTIFIERS:
            raise BaselineSyntheticFixtureValidationError(
                "training_case_identifiers must equal the fixed training-case tuple."
            )
        if self.test_case_identifiers != BASELINE_SYNTHETIC_TEST_CASE_IDENTIFIERS:
            raise BaselineSyntheticFixtureValidationError(
                "test_case_identifiers must equal the fixed test-case tuple."
            )
        _validate_file_records(self.file_records)
        _require_no_absolute_paths(_artifact_payload(self))
        expected_hash = sha256_json(_artifact_payload_without_hash(self))
        if self.artifact_hash != expected_hash:
            raise BaselineSyntheticFixtureHashError(
                "artifact_hash does not match the deterministic fixture artifact content."
            )


@dataclass(frozen=True, slots=True)
class BaselineSyntheticFixtureGenerationResult:
    """Published synthetic fixture result."""

    artifact: BaselineSyntheticFixtureArtifact
    total_generated_file_count: int


@dataclass(frozen=True, slots=True)
class _SyntheticCase:
    case_identifier: str
    split: Literal["train", "test"]
    image: ImageArray
    label: LabelArray


def generate_baseline_synthetic_fixture(
    output_root: Path,
) -> BaselineSyntheticFixtureGenerationResult:
    """Generate and publish the fixed synthetic Phase 3 fixture outside the repository."""

    validated_output_root = _validate_output_root(output_root)
    temporary_root = Path(
        tempfile.mkdtemp(
            prefix=f".{validated_output_root.name}.tmp-",
            dir=str(validated_output_root.parent),
        )
    )
    created_temporary_root = True
    try:
        artifact = _materialize_fixture_root(temporary_root)
        if validated_output_root.exists():
            raise BaselineSyntheticFixtureValidationError(
                "Synthetic fixture output root must not already exist."
            )
        os.rename(temporary_root, validated_output_root)
        created_temporary_root = False
        return BaselineSyntheticFixtureGenerationResult(
            artifact=artifact,
            total_generated_file_count=len(artifact.file_records) + 1,
        )
    except Exception:
        if created_temporary_root and temporary_root.exists():
            shutil.rmtree(temporary_root)
        raise


def baseline_synthetic_fixture_to_json(artifact: BaselineSyntheticFixtureArtifact) -> bytes:
    """Serialize the synthetic fixture artifact deterministically."""

    return canonical_json_bytes(_artifact_payload(artifact)) + b"\n"


def baseline_synthetic_fixture_from_json(
    payload: bytes | str,
) -> BaselineSyntheticFixtureArtifact:
    """Parse and validate the synthetic fixture artifact."""

    parsed = _parse_json_object(payload, context="baseline synthetic fixture")
    _require_exact_fields(
        parsed,
        expected=_ALLOWED_ARTIFACT_FIELDS,
        context="baseline synthetic fixture",
    )
    file_records_value = parsed["file_records"]
    if not isinstance(file_records_value, list):
        raise BaselineSyntheticFixtureSerializationError("file_records must be a JSON array.")
    file_records = tuple(
        _file_record_from_json_value(file_record_value) for file_record_value in file_records_value
    )
    artifact = BaselineSyntheticFixtureArtifact(
        artifact_hash=_expect_string(parsed["artifact_hash"], field_name="artifact_hash"),
        contract_version=_expect_string(
            parsed["contract_version"],
            field_name="contract_version",
        ),
        dataset_name=_expect_string(parsed["dataset_name"], field_name="dataset_name"),
        dataset_id=_expect_int(parsed["dataset_id"], field_name="dataset_id"),
        image_shape=_expect_int_tuple3(parsed["image_shape"], field_name="image_shape"),
        voxel_spacing_mm=_expect_float_tuple3(
            parsed["voxel_spacing_mm"],
            field_name="voxel_spacing_mm",
        ),
        affine=_expect_affine(parsed["affine"], field_name="affine"),
        training_case_identifiers=_expect_identifier_tuple(
            parsed["training_case_identifiers"],
            field_name="training_case_identifiers",
        ),
        test_case_identifiers=_expect_identifier_tuple(
            parsed["test_case_identifiers"],
            field_name="test_case_identifiers",
        ),
        file_records=file_records,
    )
    expected_hash = sha256_json(_artifact_payload_without_hash(artifact))
    if artifact.artifact_hash != expected_hash:
        raise BaselineSyntheticFixtureHashError(
            "artifact_hash does not match the parsed synthetic fixture content."
        )
    return artifact


def synthetic_dataset_json() -> dict[str, Any]:
    """Return the fixed nnU-Net raw-dataset metadata payload."""

    return {
        "channel_names": {"0": "CT"},
        "file_ending": BASELINE_SYNTHETIC_FILE_ENDING,
        "labels": {"background": 0, "tumor": 1},
        "name": BASELINE_SYNTHETIC_DATASET_NAME,
        "numTest": len(BASELINE_SYNTHETIC_TEST_CASE_IDENTIFIERS),
        "numTraining": len(BASELINE_SYNTHETIC_TRAINING_CASE_IDENTIFIERS),
    }


def synthetic_affine_array() -> AffineArray:
    """Return the fixed synthetic affine as a float64 array."""

    return np.asarray(BASELINE_SYNTHETIC_AFFINE, dtype=np.float64)


def synthetic_cases() -> tuple[_SyntheticCase, ...]:
    """Return the fixed synthetic cases in deterministic split order."""

    return (
        _SyntheticCase(
            case_identifier="synthetic_train_001",
            split="train",
            image=_generate_image("synthetic_train_001", _empty_label()),
            label=_empty_label(),
        ),
        _SyntheticCase(
            case_identifier="synthetic_train_002",
            split="train",
            image=_generate_image("synthetic_train_002", _central_lesion_label()),
            label=_central_lesion_label(),
        ),
        _SyntheticCase(
            case_identifier="synthetic_train_003",
            split="train",
            image=_generate_image("synthetic_train_003", _two_separated_training_lesions_label()),
            label=_two_separated_training_lesions_label(),
        ),
        _SyntheticCase(
            case_identifier="synthetic_train_004",
            split="train",
            image=_generate_image("synthetic_train_004", _boundary_touching_label()),
            label=_boundary_touching_label(),
        ),
        _SyntheticCase(
            case_identifier="synthetic_train_005",
            split="train",
            image=_generate_image("synthetic_train_005", _irregular_training_label()),
            label=_irregular_training_label(),
        ),
        _SyntheticCase(
            case_identifier="synthetic_test_001",
            split="test",
            image=_generate_image("synthetic_test_001", _single_test_lesion_label()),
            label=_single_test_lesion_label(),
        ),
        _SyntheticCase(
            case_identifier="synthetic_test_002",
            split="test",
            image=_generate_image("synthetic_test_002", _two_separated_test_lesions_label()),
            label=_two_separated_test_lesions_label(),
        ),
    )


def _materialize_fixture_root(root: Path) -> BaselineSyntheticFixtureArtifact:
    dataset_root = root / BASELINE_SYNTHETIC_DATASET_NAME
    images_tr = dataset_root / "imagesTr"
    labels_tr = dataset_root / "labelsTr"
    images_ts = dataset_root / "imagesTs"
    labels_ts = dataset_root / "labelsTs"
    images_tr.mkdir(parents=True)
    labels_tr.mkdir()
    images_ts.mkdir()
    labels_ts.mkdir()

    dataset_json_path = dataset_root / "dataset.json"
    dataset_json_bytes = canonical_json_bytes(synthetic_dataset_json()) + b"\n"
    dataset_json_path.write_bytes(dataset_json_bytes)

    for case in synthetic_cases():
        if case.split == "train":
            image_path = images_tr / f"{case.case_identifier}_0000{BASELINE_SYNTHETIC_FILE_ENDING}"
            label_path = labels_tr / f"{case.case_identifier}{BASELINE_SYNTHETIC_FILE_ENDING}"
        else:
            image_path = images_ts / f"{case.case_identifier}_0000{BASELINE_SYNTHETIC_FILE_ENDING}"
            label_path = labels_ts / f"{case.case_identifier}{BASELINE_SYNTHETIC_FILE_ENDING}"
        image_path.write_bytes(_serialize_nifti_gz(case.image, dtype=np.int16))
        label_path.write_bytes(_serialize_nifti_gz(case.label, dtype=np.uint8))

    artifact = _build_fixture_artifact(root)
    artifact_path = root / "baseline_synthetic_fixture.json"
    artifact_path.write_bytes(baseline_synthetic_fixture_to_json(artifact))
    return artifact


def _build_fixture_artifact(root: Path) -> BaselineSyntheticFixtureArtifact:
    file_records: list[BaselineSyntheticFixtureFileRecord] = []
    for relative_path in _expected_record_paths():
        path = root / relative_path
        role, case_identifier, channel_index = _role_and_case_for_path(relative_path)
        file_records.append(
            BaselineSyntheticFixtureFileRecord(
                relative_path=relative_path,
                role=role,
                case_identifier=case_identifier,
                channel_index=channel_index,
                sha256=sha256_file(path),
                byte_size=path.stat().st_size,
            )
        )
    payload_without_hash = {
        "affine": [list(row) for row in BASELINE_SYNTHETIC_AFFINE],
        "contract_version": BASELINE_SYNTHETIC_FIXTURE_VERSION,
        "dataset_id": BASELINE_SYNTHETIC_DATASET_ID,
        "dataset_name": BASELINE_SYNTHETIC_DATASET_NAME,
        "file_records": [_file_record_payload(file_record) for file_record in file_records],
        "image_shape": list(BASELINE_SYNTHETIC_SHAPE),
        "test_case_identifiers": list(BASELINE_SYNTHETIC_TEST_CASE_IDENTIFIERS),
        "training_case_identifiers": list(BASELINE_SYNTHETIC_TRAINING_CASE_IDENTIFIERS),
        "voxel_spacing_mm": list(BASELINE_SYNTHETIC_SPACING_MM),
    }
    return BaselineSyntheticFixtureArtifact(
        artifact_hash=sha256_json(payload_without_hash),
        contract_version=BASELINE_SYNTHETIC_FIXTURE_VERSION,
        dataset_name=BASELINE_SYNTHETIC_DATASET_NAME,
        dataset_id=BASELINE_SYNTHETIC_DATASET_ID,
        image_shape=BASELINE_SYNTHETIC_SHAPE,
        voxel_spacing_mm=BASELINE_SYNTHETIC_SPACING_MM,
        affine=BASELINE_SYNTHETIC_AFFINE,
        training_case_identifiers=BASELINE_SYNTHETIC_TRAINING_CASE_IDENTIFIERS,
        test_case_identifiers=BASELINE_SYNTHETIC_TEST_CASE_IDENTIFIERS,
        file_records=tuple(file_records),
    )


def _expected_record_paths() -> tuple[str, ...]:
    paths = [f"{BASELINE_SYNTHETIC_DATASET_NAME}/dataset.json"]
    for case_identifier in BASELINE_SYNTHETIC_TRAINING_CASE_IDENTIFIERS:
        paths.append(
            f"{BASELINE_SYNTHETIC_DATASET_NAME}/imagesTr/{case_identifier}_0000{BASELINE_SYNTHETIC_FILE_ENDING}"
        )
        paths.append(
            f"{BASELINE_SYNTHETIC_DATASET_NAME}/labelsTr/{case_identifier}{BASELINE_SYNTHETIC_FILE_ENDING}"
        )
    for case_identifier in BASELINE_SYNTHETIC_TEST_CASE_IDENTIFIERS:
        paths.append(
            f"{BASELINE_SYNTHETIC_DATASET_NAME}/imagesTs/{case_identifier}_0000{BASELINE_SYNTHETIC_FILE_ENDING}"
        )
        paths.append(
            f"{BASELINE_SYNTHETIC_DATASET_NAME}/labelsTs/{case_identifier}{BASELINE_SYNTHETIC_FILE_ENDING}"
        )
    return tuple(sorted(paths))


def _role_and_case_for_path(
    relative_path: str,
) -> tuple[SyntheticFixtureRole, str | None, int | None]:
    if relative_path == f"{BASELINE_SYNTHETIC_DATASET_NAME}/dataset.json":
        return ("dataset_metadata", None, None)
    pure_path = PurePosixPath(relative_path)
    stem = pure_path.name
    if pure_path.parts[1] == "imagesTr":
        return ("training_image", stem.removesuffix(f"_0000{BASELINE_SYNTHETIC_FILE_ENDING}"), 0)
    if pure_path.parts[1] == "labelsTr":
        return ("training_label", stem.removesuffix(BASELINE_SYNTHETIC_FILE_ENDING), None)
    if pure_path.parts[1] == "imagesTs":
        return ("test_image", stem.removesuffix(f"_0000{BASELINE_SYNTHETIC_FILE_ENDING}"), 0)
    if pure_path.parts[1] == "labelsTs":
        return ("test_label", stem.removesuffix(BASELINE_SYNTHETIC_FILE_ENDING), None)
    raise BaselineSyntheticFixtureValidationError(f"Unexpected fixture path: {relative_path!r}.")


def _serialize_nifti_gz(
    array: npt.NDArray[np.generic],
    *,
    dtype: type[np.int16] | type[np.uint8],
) -> bytes:
    affine = synthetic_affine_array()
    cast_array = np.asarray(array, dtype=dtype)
    image = nib.Nifti1Image(cast_array, affine)  # type: ignore[no-untyped-call]
    image.set_qform(affine, code=1)  # type: ignore[no-untyped-call]
    image.set_sform(affine, code=1)  # type: ignore[no-untyped-call]
    header = image.header
    header.set_data_dtype(cast_array.dtype)  # type: ignore[no-untyped-call]
    header.set_xyzt_units("mm")  # type: ignore[no-untyped-call]
    header["scl_slope"] = 1.0
    header["scl_inter"] = 0.0
    header["descrip"] = b"protoem-ct baseline synthetic"
    nii_bytes = image.to_bytes()
    buffer = io.BytesIO()
    with gzip.GzipFile(
        filename="",
        fileobj=buffer,
        mode="wb",
        mtime=_FIXED_GZIP_MTIME,
    ) as gzip_file:
        gzip_file.write(nii_bytes)
    return buffer.getvalue()


def _generate_image(case_identifier: str, label: LabelArray) -> ImageArray:
    indices = np.indices(BASELINE_SYNTHETIC_SHAPE, dtype=np.int32)
    x_axis = indices[0]
    y_axis = indices[1]
    z_axis = indices[2]
    case_index = _all_case_identifiers().index(case_identifier) + 1
    base = (
        np.int32(-860)
        + np.int32(17) * x_axis
        - np.int32(11) * y_axis
        + np.int32(29) * z_axis
        + np.int32(13) * case_index
        + ((x_axis * y_axis + z_axis * 3 + case_index) % 19)
        - ((x_axis + z_axis + case_index) % 7) * 5
    )
    correlated = base + label.astype(np.int32, copy=False) * np.int32(340)
    return cast(ImageArray, correlated.astype(np.int16))


def _empty_label() -> LabelArray:
    return np.zeros(BASELINE_SYNTHETIC_SHAPE, dtype=np.uint8)


def _central_lesion_label() -> LabelArray:
    label = _empty_label()
    label[10:14, 10:14, 6:9] = 1
    return label


def _two_separated_training_lesions_label() -> LabelArray:
    label = _empty_label()
    label[4:7, 4:7, 3:5] = 1
    label[15:20, 15:18, 8:11] = 1
    return label


def _boundary_touching_label() -> LabelArray:
    label = _empty_label()
    label[0:3, 6:11, 4:8] = 1
    return label


def _irregular_training_label() -> LabelArray:
    indices = np.indices(BASELINE_SYNTHETIC_SHAPE, dtype=np.int32)
    x_axis = indices[0]
    y_axis = indices[1]
    z_axis = indices[2]
    ellipsoid = (
        ((x_axis - 16) * (x_axis - 16)) / 9.0
        + ((y_axis - 8) * (y_axis - 8)) / 16.0
        + ((z_axis - 7) * (z_axis - 7)) / 4.0
    ) <= 1.0
    extension = (
        (x_axis >= 14)
        & (x_axis <= 18)
        & (y_axis >= 8)
        & (y_axis <= 10)
        & (z_axis >= 5)
        & (z_axis <= 8)
    )
    tail = (
        (x_axis >= 17)
        & (x_axis <= 19)
        & (y_axis >= 10)
        & (y_axis <= 13)
        & (z_axis >= 7)
        & (z_axis <= 9)
    )
    return np.asarray(ellipsoid | extension | tail, dtype=np.uint8)


def _single_test_lesion_label() -> LabelArray:
    label = _empty_label()
    label[7:11, 13:18, 5:8] = 1
    return label


def _two_separated_test_lesions_label() -> LabelArray:
    label = _empty_label()
    label[5:8, 16:19, 2:4] = 1
    label[17:21, 4:8, 9:12] = 1
    return label


def _all_case_identifiers() -> tuple[str, ...]:
    return (
        *BASELINE_SYNTHETIC_TRAINING_CASE_IDENTIFIERS,
        *BASELINE_SYNTHETIC_TEST_CASE_IDENTIFIERS,
    )


def _validate_output_root(output_root: Path) -> Path:
    repository_root = Path(__file__).resolve().parents[3]
    path = Path(output_root)
    if not path.is_absolute():
        raise BaselineSyntheticFixtureValidationError("output_root must be an absolute path.")
    _require_safe_absolute_path(path)
    if path.exists():
        raise BaselineSyntheticFixtureValidationError("output_root must not already exist.")
    parent = path.parent
    if not parent.exists():
        raise BaselineSyntheticFixtureValidationError(
            "output_root parent directory must already exist."
        )
    if not parent.is_dir():
        raise BaselineSyntheticFixtureValidationError(
            "output_root parent directory must be a directory."
        )
    _require_no_symlink_components(parent)
    normalized_path = path.resolve(strict=False)
    if normalized_path == repository_root or _is_relative_to(normalized_path, repository_root):
        raise BaselineSyntheticFixtureValidationError(
            "output_root must resolve outside the repository root."
        )
    return parent.resolve(strict=True) / path.name


def _validate_file_records(file_records: tuple[BaselineSyntheticFixtureFileRecord, ...]) -> None:
    expected_paths = _expected_record_paths()
    observed_paths = tuple(file_record.relative_path for file_record in file_records)
    if observed_paths != tuple(sorted(observed_paths)):
        raise BaselineSyntheticFixtureValidationError(
            "file_records must be sorted by ascending relative_path."
        )
    if len(set(observed_paths)) != len(observed_paths):
        raise BaselineSyntheticFixtureValidationError(
            "file_records must not contain duplicate paths."
        )
    if observed_paths != expected_paths:
        raise BaselineSyntheticFixtureValidationError(
            "file_records must match the fixed synthetic fixture layout."
        )
    training_cases = set(BASELINE_SYNTHETIC_TRAINING_CASE_IDENTIFIERS)
    test_cases = set(BASELINE_SYNTHETIC_TEST_CASE_IDENTIFIERS)
    seen_training_images: set[str] = set()
    seen_training_labels: set[str] = set()
    seen_test_images: set[str] = set()
    seen_test_labels: set[str] = set()
    metadata_count = 0
    for file_record in file_records:
        case_identifier = file_record.case_identifier
        if file_record.role == "dataset_metadata":
            metadata_count += 1
            continue
        if case_identifier is None:
            raise BaselineSyntheticFixtureValidationError(
                "Non-metadata records must contain case_identifier."
            )
        if file_record.role.startswith("training_"):
            if case_identifier not in training_cases:
                raise BaselineSyntheticFixtureValidationError(
                    "training file records must reference declared training cases."
                )
            if file_record.role == "training_image":
                seen_training_images.add(case_identifier)
            else:
                seen_training_labels.add(case_identifier)
        elif file_record.role.startswith("test_"):
            if case_identifier not in test_cases:
                raise BaselineSyntheticFixtureValidationError(
                    "test file records must reference declared test cases."
                )
            if file_record.role == "test_image":
                seen_test_images.add(case_identifier)
            else:
                seen_test_labels.add(case_identifier)
        else:
            raise BaselineSyntheticFixtureValidationError(
                f"Unsupported file role: {file_record.role!r}."
            )
    if metadata_count != 1:
        raise BaselineSyntheticFixtureValidationError(
            "file_records must contain exactly one dataset_metadata record."
        )
    if seen_training_images != training_cases or seen_training_labels != training_cases:
        raise BaselineSyntheticFixtureValidationError(
            "file_records must contain one training image and one training label per training case."
        )
    if seen_test_images != test_cases or seen_test_labels != test_cases:
        raise BaselineSyntheticFixtureValidationError(
            "file_records must contain one test image and one test label per test case."
        )


def _artifact_payload(artifact: BaselineSyntheticFixtureArtifact) -> dict[str, Any]:
    return {
        "affine": [list(row) for row in artifact.affine],
        "artifact_hash": artifact.artifact_hash,
        "contract_version": artifact.contract_version,
        "dataset_id": artifact.dataset_id,
        "dataset_name": artifact.dataset_name,
        "file_records": [
            _file_record_payload(file_record) for file_record in artifact.file_records
        ],
        "image_shape": list(artifact.image_shape),
        "test_case_identifiers": list(artifact.test_case_identifiers),
        "training_case_identifiers": list(artifact.training_case_identifiers),
        "voxel_spacing_mm": list(artifact.voxel_spacing_mm),
    }


def _artifact_payload_without_hash(artifact: BaselineSyntheticFixtureArtifact) -> dict[str, Any]:
    payload = _artifact_payload(artifact)
    del payload["artifact_hash"]
    return payload


def _file_record_payload(file_record: BaselineSyntheticFixtureFileRecord) -> dict[str, Any]:
    return {
        "byte_size": file_record.byte_size,
        "case_identifier": file_record.case_identifier,
        "channel_index": file_record.channel_index,
        "relative_path": file_record.relative_path,
        "role": file_record.role,
        "sha256": file_record.sha256,
    }


def _file_record_from_json_value(value: Any) -> BaselineSyntheticFixtureFileRecord:
    if not isinstance(value, dict):
        raise BaselineSyntheticFixtureSerializationError("file_records entries must be objects.")
    _require_exact_fields(
        value,
        expected=_ALLOWED_FILE_RECORD_FIELDS,
        context="baseline synthetic fixture file record",
    )
    return BaselineSyntheticFixtureFileRecord(
        relative_path=_expect_string(value["relative_path"], field_name="relative_path"),
        role=cast(SyntheticFixtureRole, _expect_string(value["role"], field_name="role")),
        case_identifier=_expect_optional_string(
            value["case_identifier"],
            field_name="case_identifier",
        ),
        channel_index=_expect_optional_int(value["channel_index"], field_name="channel_index"),
        sha256=_expect_string(value["sha256"], field_name="sha256"),
        byte_size=_expect_int(value["byte_size"], field_name="byte_size"),
    )


def _require_case_identifier(value: str) -> None:
    if not _CASE_IDENTIFIER_PATTERN.fullmatch(value):
        raise BaselineSyntheticFixtureValidationError(
            "case identifiers must match the conservative ASCII identifier policy."
        )
    _require_no_absolute_path_string(value, field_name="case_identifier")


def _require_sha256(value: str, *, field_name: str) -> None:
    if not _SHA256_PATTERN.fullmatch(value):
        raise BaselineSyntheticFixtureValidationError(
            f"{field_name} must be exactly 64 lowercase hexadecimal characters."
        )


def _require_relative_posix_path(value: str) -> None:
    if not value or "\\" in value:
        raise BaselineSyntheticFixtureValidationError(
            "relative_path must be a relative POSIX path without backslashes."
        )
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise BaselineSyntheticFixtureValidationError(
            "relative_path must not contain control characters."
        )
    pure_path = PurePosixPath(value)
    if pure_path.is_absolute():
        raise BaselineSyntheticFixtureValidationError("relative_path must not be absolute.")
    if ".." in pure_path.parts:
        raise BaselineSyntheticFixtureValidationError(
            "relative_path must not contain parent-directory traversal."
        )


def _require_safe_absolute_path(path: Path) -> None:
    normalized = os.path.normpath(os.fspath(path))
    if ".." in Path(normalized).parts:
        raise BaselineSyntheticFixtureValidationError(
            "output_root must not contain parent-directory traversal."
        )
    for component in path.parts[1:]:
        if component == "":
            raise BaselineSyntheticFixtureValidationError(
                "output_root must not contain empty path components."
            )
        if any(ord(character) < 32 or ord(character) == 127 for character in component):
            raise BaselineSyntheticFixtureValidationError(
                "output_root must not contain control characters."
            )


def _require_no_symlink_components(path: Path) -> None:
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current = current / component
        if current.is_symlink():
            raise BaselineSyntheticFixtureValidationError(
                "output_root must not traverse symlinked path components."
            )


def _is_relative_to(path: Path, candidate_parent: Path) -> bool:
    try:
        path.relative_to(candidate_parent)
        return True
    except ValueError:
        return False


def _require_no_absolute_paths(value: Any) -> None:
    if isinstance(value, str):
        _require_no_absolute_path_string(value, field_name="persisted string")
        return
    if isinstance(value, Mapping):
        for item in value.values():
            _require_no_absolute_paths(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _require_no_absolute_paths(item)


def _require_no_absolute_path_string(value: str, *, field_name: str) -> None:
    if value.startswith("/") or _WINDOWS_ABSOLUTE_PATTERN.match(value):
        raise BaselineSyntheticFixtureValidationError(
            f"{field_name} must not contain an absolute path."
        )


def _parse_json_object(payload: bytes | str, *, context: str) -> dict[str, Any]:
    text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
    try:
        parsed = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_bad_json_constant,
        )
    except Exception as error:
        raise BaselineSyntheticFixtureSerializationError(
            f"Failed to parse {context} JSON: {error}"
        ) from error
    if not isinstance(parsed, dict):
        raise BaselineSyntheticFixtureSerializationError(f"{context} JSON must encode an object.")
    return parsed


def _require_exact_fields(
    payload: Mapping[str, Any],
    *,
    expected: set[str],
    context: str,
) -> None:
    unknown_fields = set(payload) - expected
    if unknown_fields:
        raise BaselineSyntheticFixtureSerializationError(
            f"Unknown {context} field(s): {', '.join(sorted(unknown_fields))}."
        )
    missing_fields = expected - set(payload)
    if missing_fields:
        raise BaselineSyntheticFixtureSerializationError(
            f"Missing {context} field(s): {', '.join(sorted(missing_fields))}."
        )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BaselineSyntheticFixtureSerializationError(f"Duplicate JSON object key: {key!r}.")
        result[key] = value
    return result


def _bad_json_constant(constant: str) -> Any:
    raise BaselineSyntheticFixtureSerializationError(f"Invalid JSON constant: {constant!r}.")


def _expect_string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise BaselineSyntheticFixtureSerializationError(f"{field_name} must be a string.")
    return value


def _expect_optional_string(value: Any, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _expect_string(value, field_name=field_name)


def _expect_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BaselineSyntheticFixtureSerializationError(f"{field_name} must be an integer.")
    return value


def _expect_optional_int(value: Any, *, field_name: str) -> int | None:
    if value is None:
        return None
    return _expect_int(value, field_name=field_name)


def _expect_number(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BaselineSyntheticFixtureSerializationError(f"{field_name} must be a number.")
    scalar = float(value)
    if not math.isfinite(scalar):
        raise BaselineSyntheticFixtureSerializationError(f"{field_name} must be finite.")
    return scalar


def _expect_int_tuple3(value: Any, *, field_name: str) -> tuple[int, int, int]:
    if not isinstance(value, list) or len(value) != 3:
        raise BaselineSyntheticFixtureSerializationError(f"{field_name} must be a length-3 array.")
    return tuple(_expect_int(item, field_name=field_name) for item in value)  # type: ignore[return-value]


def _expect_float_tuple3(value: Any, *, field_name: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise BaselineSyntheticFixtureSerializationError(f"{field_name} must be a length-3 array.")
    return tuple(_expect_number(item, field_name=field_name) for item in value)  # type: ignore[return-value]


def _expect_affine(
    value: Any,
    *,
    field_name: str,
) -> tuple[tuple[float, float, float, float], ...]:
    if not isinstance(value, list) or len(value) != 4:
        raise BaselineSyntheticFixtureSerializationError(f"{field_name} must be a 4x4 array.")
    rows: list[tuple[float, float, float, float]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != 4:
            raise BaselineSyntheticFixtureSerializationError(f"{field_name} must be a 4x4 array.")
        rows.append(tuple(_expect_number(item, field_name=field_name) for item in row))  # type: ignore[arg-type]
    return tuple(rows)


def _expect_identifier_tuple(value: Any, *, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise BaselineSyntheticFixtureSerializationError(
            f"{field_name} must be an array of case identifiers."
        )
    identifiers = tuple(_expect_string(item, field_name=field_name) for item in value)
    for identifier in identifiers:
        _require_case_identifier(identifier)
    return identifiers
