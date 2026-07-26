"""Read-only Phase 2 geometry and label QA over anonymous development artifacts."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final, cast

import nibabel as nib
import numpy as np
import numpy.typing as npt
from nibabel.filebasedimages import ImageFileError

import protoem_ct.artifacts.hashing as artifact_hashing
from protoem_ct.artifacts import (
    GEOMETRY_LABEL_QA_FAILURE_CODES,
    GEOMETRY_LABEL_QA_STAGE,
    PHASE2_DEVELOPMENT_COHORT_ROLE,
    PHASE2_SCHEMA_VERSION,
    DatasetManifest,
    DevelopmentSplitManifest,
    GeometryLabelQaArtifact,
    GeometryLabelQaCaseRecord,
    Phase2ArtifactError,
    SplitAssignment,
    hash_dataset_manifest,
    hash_development_split,
    hash_geometry_label_qa,
    phase2_artifact_from_json,
    phase2_artifact_to_json,
    sha256_json,
)
from protoem_ct.data.phase2_paths import (
    Phase2PathError,
    require_unique_resolved_paths,
    resolve_regular_file_beneath_root,
    validate_explicit_dataset_root,
)

GEOMETRY_LABEL_QA_CONFIG_PAYLOAD_TYPE: Final[str] = "phase2-geometry-label-qa-config-v1"
_ZERO_SHA256: Final[str] = "0" * 64

Array = npt.NDArray[np.generic]
FloatArray = npt.NDArray[np.floating[Any]]


class GeometryLabelQaError(ValueError):
    """Base exception for Phase 2 geometry and label QA failures."""


class InvalidGeometryLabelQaConfigError(GeometryLabelQaError):
    """Raised when explicit geometry-label QA configuration is invalid."""


class InvalidGeometryLabelQaInputError(GeometryLabelQaError):
    """Raised when manifest, split, or explicit metadata inputs are invalid."""


class GeometryLabelQaHashMismatchError(InvalidGeometryLabelQaInputError):
    """Raised when a stored artifact hash does not match recomputed content."""


class GeometryLabelQaAssignmentError(InvalidGeometryLabelQaInputError):
    """Raised when split assignments do not match the source manifest exactly."""


class GeometryLabelQaSourceIntegrityError(GeometryLabelQaError):
    """Raised when source files are missing, aliased, changed, or unsafe."""


class GeometryLabelQaNiftiReadError(GeometryLabelQaError):
    """Raised when an integrity-verified source cannot be read as NIfTI."""


class UnsafeGeometryLabelQaOutputPathError(GeometryLabelQaError):
    """Raised when the requested QA output path violates containment policy."""


class ExistingGeometryLabelQaOutputError(GeometryLabelQaError):
    """Raised when QA publication would overwrite an existing output."""


class GeometryLabelQaPublicationError(GeometryLabelQaError):
    """Raised when validated QA JSON cannot be published atomically."""


@dataclass(frozen=True, slots=True)
class GeometryLabelQaConfig:
    """Explicit scientific settings for geometry and label QA.

    No label policy or affine tolerance is inferred from the data. The configuration contains no
    dataset paths, source identifiers, environment values, or fitted statistics.
    """

    allowed_label_values: tuple[int, ...]
    tumor_label_value: int
    affine_tolerance: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.allowed_label_values, tuple)
            or not self.allowed_label_values
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in self.allowed_label_values
            )
        ):
            msg = "allowed_label_values must be a nonempty tuple of integer labels"
            raise InvalidGeometryLabelQaConfigError(msg)
        if tuple(sorted(set(self.allowed_label_values))) != self.allowed_label_values:
            msg = "allowed_label_values must be unique and sorted"
            raise InvalidGeometryLabelQaConfigError(msg)
        if isinstance(self.tumor_label_value, bool) or not isinstance(self.tumor_label_value, int):
            msg = "tumor_label_value must be an integer"
            raise InvalidGeometryLabelQaConfigError(msg)
        if self.tumor_label_value not in self.allowed_label_values:
            msg = "tumor_label_value must be included in allowed_label_values"
            raise InvalidGeometryLabelQaConfigError(msg)
        if (
            isinstance(self.affine_tolerance, bool)
            or not isinstance(self.affine_tolerance, float)
            or not math.isfinite(self.affine_tolerance)
            or self.affine_tolerance <= 0.0
        ):
            msg = "affine_tolerance must be a positive finite float"
            raise InvalidGeometryLabelQaConfigError(msg)


def geometry_label_qa_config_hash_payload(config: GeometryLabelQaConfig) -> dict[str, object]:
    """Return the canonical payload for a geometry-label QA configuration hash."""
    return {
        "affine_tolerance": config.affine_tolerance,
        "allowed_label_values": list(config.allowed_label_values),
        "payload_type": GEOMETRY_LABEL_QA_CONFIG_PAYLOAD_TYPE,
        "schema_version": PHASE2_SCHEMA_VERSION,
        "tumor_label_value": config.tumor_label_value,
    }


def hash_geometry_label_qa_config(config: GeometryLabelQaConfig) -> str:
    """Return the deterministic SHA-256 hash of explicit geometry-label QA settings."""
    return sha256_json(geometry_label_qa_config_hash_payload(config))


def run_geometry_label_qa(
    manifest_path: Path,
    split_path: Path,
    *,
    dataset_root: Path,
    output_path: Path,
    config: GeometryLabelQaConfig,
    git_commit: str,
    created_at_utc: str,
) -> GeometryLabelQaArtifact:
    """Run read-only geometry and label QA and publish an intermediate QA artifact."""
    source_manifest_path = _validate_input_json_path(manifest_path, "manifest")
    source_split_path = _validate_input_json_path(split_path, "split")
    canonical_root = validate_explicit_dataset_root(dataset_root)
    safe_output_path = _validate_qa_output_path(
        output_path,
        dataset_root=canonical_root,
        input_paths=(source_manifest_path, source_split_path),
    )
    _require_explicit_metadata(git_commit, "git_commit")
    _require_explicit_metadata(created_at_utc, "created_at_utc")

    manifest = _load_and_verify_dataset_manifest(source_manifest_path.read_text(encoding="utf-8"))
    split = _load_and_verify_development_split(source_split_path.read_text(encoding="utf-8"))
    assignments_by_case = _verify_manifest_split_linkage(manifest, split)

    verified_sources = _resolve_and_verify_sources(canonical_root, manifest)
    case_records = tuple(
        sorted(
            (
                _qa_case(
                    case=case,
                    assignment=assignments_by_case[case.anonymous_case_id],
                    image_path=verified_sources[case.anonymous_case_id][0],
                    label_path=verified_sources[case.anonymous_case_id][1],
                    config=config,
                )
                for case in manifest.cases
            ),
            key=lambda record: (record.anonymous_patient_id, record.anonymous_case_id),
        )
    )
    artifact_without_hash = GeometryLabelQaArtifact(
        schema_version=PHASE2_SCHEMA_VERSION,
        stage=GEOMETRY_LABEL_QA_STAGE,
        created_at_utc=created_at_utc,
        git_commit=git_commit,
        config_hash=hash_geometry_label_qa_config(config),
        manifest_hash=manifest.manifest_hash,
        split_hash=split.split_hash,
        case_count=len(case_records),
        passed_case_count=sum(1 for record in case_records if record.qa_passed),
        failed_case_count=sum(1 for record in case_records if not record.qa_passed),
        case_records=case_records,
        qa_artifact_hash=_ZERO_SHA256,
    )
    qa_hash = hash_geometry_label_qa(artifact_without_hash)
    artifact = replace(artifact_without_hash, qa_artifact_hash=qa_hash)
    if hash_geometry_label_qa(artifact) != qa_hash:
        msg = "geometry-label QA artifact hash verification failed"
        raise GeometryLabelQaPublicationError(msg)

    _publish_qa_json(artifact, safe_output_path)
    return artifact


def _load_and_verify_dataset_manifest(manifest_text: str) -> DatasetManifest:
    try:
        artifact = phase2_artifact_from_json(manifest_text)
    except Phase2ArtifactError as exc:
        msg = "input manifest is not a valid Phase 2 dataset manifest"
        raise InvalidGeometryLabelQaInputError(msg) from exc
    if not isinstance(artifact, DatasetManifest):
        msg = "input artifact must be a DatasetManifest"
        raise InvalidGeometryLabelQaInputError(msg)
    if artifact.cohort_role != PHASE2_DEVELOPMENT_COHORT_ROLE:
        msg = "input dataset manifest must use the development cohort role"
        raise InvalidGeometryLabelQaInputError(msg)
    if hash_dataset_manifest(artifact) != artifact.manifest_hash:
        msg = "input dataset manifest hash does not match its content"
        raise GeometryLabelQaHashMismatchError(msg)
    return artifact


def _load_and_verify_development_split(split_text: str) -> DevelopmentSplitManifest:
    try:
        artifact = phase2_artifact_from_json(split_text)
    except Phase2ArtifactError as exc:
        msg = "input split is not a valid Phase 2 development split"
        raise InvalidGeometryLabelQaInputError(msg) from exc
    if not isinstance(artifact, DevelopmentSplitManifest):
        msg = "input artifact must be a DevelopmentSplitManifest"
        raise InvalidGeometryLabelQaInputError(msg)
    if hash_development_split(artifact) != artifact.split_hash:
        msg = "input split hash does not match its content"
        raise GeometryLabelQaHashMismatchError(msg)
    return artifact


def _verify_manifest_split_linkage(
    manifest: DatasetManifest,
    split: DevelopmentSplitManifest,
) -> dict[str, SplitAssignment]:
    if split.source_manifest_hash != manifest.manifest_hash:
        msg = "split source_manifest_hash must equal manifest_hash"
        raise GeometryLabelQaAssignmentError(msg)
    manifest_cases = {case.anonymous_case_id: case for case in manifest.cases}
    assignments_by_case = {
        assignment.anonymous_case_id: assignment for assignment in split.assignments
    }
    if set(assignments_by_case) != set(manifest_cases):
        msg = "split assignments must match manifest cases exactly"
        raise GeometryLabelQaAssignmentError(msg)
    for case_id, assignment in assignments_by_case.items():
        if assignment.anonymous_patient_id != manifest_cases[case_id].anonymous_patient_id:
            msg = "split assignment patient ID must match the manifest case patient ID"
            raise GeometryLabelQaAssignmentError(msg)
    _require_no_partition_overlap(split.assignments)
    return assignments_by_case


def _require_no_partition_overlap(assignments: tuple[SplitAssignment, ...]) -> None:
    patient_partitions: dict[str, str] = {}
    case_partitions: dict[str, str] = {}
    for assignment in assignments:
        previous_patient_partition = patient_partitions.setdefault(
            assignment.anonymous_patient_id,
            assignment.partition,
        )
        previous_case_partition = case_partitions.setdefault(
            assignment.anonymous_case_id,
            assignment.partition,
        )
        if previous_patient_partition != assignment.partition:
            msg = "one patient cannot appear in multiple split partitions"
            raise GeometryLabelQaAssignmentError(msg)
        if previous_case_partition != assignment.partition:
            msg = "one case cannot appear in multiple split partitions"
            raise GeometryLabelQaAssignmentError(msg)


def _resolve_and_verify_sources(
    dataset_root: Path,
    manifest: DatasetManifest,
) -> dict[str, tuple[Path, Path]]:
    resolved: dict[str, tuple[Path, Path]] = {}
    named_paths: list[tuple[str, Path]] = []
    try:
        for case in manifest.cases:
            image_path = resolve_regular_file_beneath_root(
                dataset_root,
                case.relative_image_path,
            )
            label_path = resolve_regular_file_beneath_root(
                dataset_root,
                case.relative_label_path,
            )
            resolved[case.anonymous_case_id] = (image_path, label_path)
            named_paths.extend(
                (
                    (f"{case.anonymous_case_id}:image", image_path),
                    (f"{case.anonymous_case_id}:label", label_path),
                )
            )
        require_unique_resolved_paths(tuple(named_paths))
    except Phase2PathError as exc:
        msg = "source file path violates the Phase 2 containment contract"
        raise GeometryLabelQaSourceIntegrityError(msg) from exc

    try:
        for case in manifest.cases:
            image_path, label_path = resolved[case.anonymous_case_id]
            if artifact_hashing.sha256_file(image_path) != case.image_sha256:
                msg = "source image file hash does not match the manifest"
                raise GeometryLabelQaSourceIntegrityError(msg)
            if artifact_hashing.sha256_file(label_path) != case.label_sha256:
                msg = "source label file hash does not match the manifest"
                raise GeometryLabelQaSourceIntegrityError(msg)
    except artifact_hashing.HashingError as exc:
        msg = "failed to hash an integrity-checked source file"
        raise GeometryLabelQaSourceIntegrityError(msg) from exc
    return resolved


def _qa_case(
    *,
    case: Any,
    assignment: SplitAssignment,
    image_path: Path,
    label_path: Path,
    config: GeometryLabelQaConfig,
) -> GeometryLabelQaCaseRecord:
    image = _load_nifti(image_path, "image")
    label = _load_nifti(label_path, "label")
    image_array = _nifti_array(image, "image")
    label_array = _nifti_array(label, "label")
    image_affine = _affine_tuple(image.affine, "image_affine")
    label_affine = _affine_tuple(label.affine, "label_affine")
    image_spacing = _spacing_tuple(image_affine, "image_spacing")
    label_spacing = _spacing_tuple(label_affine, "label_spacing")
    image_orientation = _orientation_tuple(image_affine, "image_orientation")
    label_orientation = _orientation_tuple(label_affine, "label_orientation")

    image_shape = tuple(int(dimension) for dimension in image_array.shape)
    label_shape = tuple(int(dimension) for dimension in label_array.shape)
    image_finite = _array_is_finite(image_array)
    label_finite = _array_is_finite(label_array)
    label_integer = label_finite and _label_is_integer_valued(label_array)
    observed_label_values = _observed_label_values(label_array)
    observed_integer_values = tuple(
        int(value) for value in observed_label_values if isinstance(value, int)
    )
    tumor_voxel_count = _tumor_voxel_count(label_array, config.tumor_label_value)

    failure_reasons = _failure_reasons(
        image_ndim=image_array.ndim,
        label_ndim=label_array.ndim,
        image_finite=image_finite,
        label_finite=label_finite,
        label_integer=label_integer,
        observed_integer_values=observed_integer_values,
        image_shape=image_shape,
        label_shape=label_shape,
        image_affine=image_affine,
        label_affine=label_affine,
        image_spacing=image_spacing,
        label_spacing=label_spacing,
        allowed_label_values=config.allowed_label_values,
        affine_tolerance=config.affine_tolerance,
    )
    return GeometryLabelQaCaseRecord(
        anonymous_patient_id=case.anonymous_patient_id,
        anonymous_case_id=case.anonymous_case_id,
        partition=assignment.partition,
        dimensionality=int(image_array.ndim),
        image_shape=image_shape,
        label_shape=label_shape,
        image_dtype=str(image.get_data_dtype()),
        label_dtype=str(label.get_data_dtype()),
        image_affine=image_affine,
        label_affine=label_affine,
        image_orientation=image_orientation,
        label_orientation=label_orientation,
        image_spacing=image_spacing,
        label_spacing=label_spacing,
        image_finite=image_finite,
        label_finite=label_finite,
        observed_label_values=observed_label_values,
        allowed_label_values=config.allowed_label_values,
        image_label_shape_match=image_shape == label_shape,
        image_label_affine_match=_allclose_affine(
            image_affine,
            label_affine,
            tolerance=config.affine_tolerance,
        ),
        tumor_label_value=config.tumor_label_value,
        tumor_voxel_count=tumor_voxel_count,
        empty_tumor=tumor_voxel_count == 0,
        qa_passed=not failure_reasons,
        failure_reasons=failure_reasons,
    )


def _load_nifti(path: Path, role: str) -> Any:
    try:
        return nib.load(str(path), mmap=False)
    except (ImageFileError, OSError, ValueError) as exc:
        msg = f"{role} source is not a readable NIfTI file"
        raise GeometryLabelQaNiftiReadError(msg) from exc


def _nifti_array(image: Any, role: str) -> Array:
    try:
        dataobj = image.dataobj
        if hasattr(dataobj, "get_unscaled"):
            return cast(Array, np.asarray(dataobj.get_unscaled()))
        return cast(Array, np.asarray(dataobj))
    except (OSError, ValueError, TypeError) as exc:
        msg = f"{role} NIfTI data cannot be loaded"
        raise GeometryLabelQaNiftiReadError(msg) from exc


def _affine_tuple(value: object, field_name: str) -> tuple[tuple[float, float, float, float], ...]:
    array = np.asarray(value, dtype=float)
    if array.shape != (4, 4) or not bool(np.isfinite(array).all()):
        msg = f"{field_name} must be a finite 4x4 affine"
        raise GeometryLabelQaNiftiReadError(msg)
    rows = tuple(tuple(float(item) for item in row) for row in array.tolist())
    return cast(tuple[tuple[float, float, float, float], ...], rows)


def _spacing_tuple(
    affine: tuple[tuple[float, float, float, float], ...],
    field_name: str,
) -> tuple[float, float, float]:
    matrix = np.asarray(affine, dtype=float)[:3, :3]
    spacing = np.sqrt(np.sum(matrix * matrix, axis=0))
    if spacing.shape != (3,) or not bool(np.isfinite(spacing).all()) or bool(np.any(spacing <= 0)):
        msg = f"{field_name} must contain three positive finite values"
        raise GeometryLabelQaNiftiReadError(msg)
    return cast(tuple[float, float, float], tuple(float(value) for value in spacing.tolist()))


def _orientation_tuple(
    affine: tuple[tuple[float, float, float, float], ...],
    field_name: str,
) -> tuple[str, str, str]:
    axis_codes = nib.aff2axcodes(np.asarray(affine, dtype=float))  # type: ignore[no-untyped-call]
    if len(axis_codes) != 3 or any(code is None for code in axis_codes):
        msg = f"{field_name} could not be derived from affine"
        raise GeometryLabelQaNiftiReadError(msg)
    return cast(tuple[str, str, str], tuple(str(code) for code in axis_codes))


def _array_is_finite(array: Array) -> bool:
    try:
        return bool(np.isfinite(array).all())
    except TypeError:
        return False


def _label_is_integer_valued(array: Array) -> bool:
    if np.iscomplexobj(cast(Any, array)):
        return False
    try:
        return bool(np.equal(cast(Any, array), np.round(cast(Any, array))).all())
    except TypeError:
        return False


def _observed_label_values(array: Array) -> tuple[int | float, ...]:
    try:
        unique_values = np.unique(array)
    except TypeError:
        return ()
    values: list[int | float] = []
    for raw_value in unique_values.tolist():
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        if value.is_integer():
            values.append(int(value))
        else:
            values.append(value)
    return tuple(sorted(set(values)))


def _tumor_voxel_count(array: Array, tumor_label_value: int) -> int:
    try:
        return int(np.count_nonzero(array == tumor_label_value))
    except TypeError:
        return 0


def _failure_reasons(
    *,
    image_ndim: int,
    label_ndim: int,
    image_finite: bool,
    label_finite: bool,
    label_integer: bool,
    observed_integer_values: tuple[int, ...],
    image_shape: tuple[int, ...],
    label_shape: tuple[int, ...],
    image_affine: tuple[tuple[float, float, float, float], ...],
    label_affine: tuple[tuple[float, float, float, float], ...],
    image_spacing: tuple[float, float, float],
    label_spacing: tuple[float, float, float],
    allowed_label_values: tuple[int, ...],
    affine_tolerance: float,
) -> tuple[str, ...]:
    observed: list[str] = []
    if image_ndim != 3:
        observed.append("image_not_3d")
    if label_ndim != 3:
        observed.append("label_not_3d")
    if not image_finite:
        observed.append("image_nonfinite")
    if not label_finite:
        observed.append("label_nonfinite")
    if not label_integer:
        observed.append("label_noninteger")
    if label_integer and any(
        value not in allowed_label_values for value in observed_integer_values
    ):
        observed.append("label_value_not_allowed")
    if image_shape != label_shape:
        observed.append("shape_mismatch")
    if not _allclose_affine(image_affine, label_affine, tolerance=affine_tolerance):
        observed.append("affine_mismatch")
    if not _allclose_spacing(image_spacing, label_spacing, tolerance=affine_tolerance):
        observed.append("spacing_mismatch")
    return tuple(code for code in GEOMETRY_LABEL_QA_FAILURE_CODES if code in observed)


def _allclose_affine(
    first: tuple[tuple[float, float, float, float], ...],
    second: tuple[tuple[float, float, float, float], ...],
    *,
    tolerance: float,
) -> bool:
    return bool(
        np.allclose(
            np.asarray(first, dtype=float),
            np.asarray(second, dtype=float),
            atol=tolerance,
            rtol=0.0,
            equal_nan=False,
        )
    )


def _allclose_spacing(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
    *,
    tolerance: float,
) -> bool:
    return bool(
        np.allclose(
            np.asarray(first, dtype=float),
            np.asarray(second, dtype=float),
            atol=tolerance,
            rtol=0.0,
            equal_nan=False,
        )
    )


def _validate_input_json_path(path: Path, role: str) -> Path:
    if str(path) == "":
        msg = f"input {role} path must not be empty"
        raise InvalidGeometryLabelQaInputError(msg)
    if not path.is_absolute():
        msg = f"input {role} path must be explicit and absolute"
        raise InvalidGeometryLabelQaInputError(msg)
    try:
        resolved_path = path.resolve(strict=True)
    except FileNotFoundError as exc:
        msg = f"input {role} JSON does not exist"
        raise InvalidGeometryLabelQaInputError(msg) from exc
    except OSError as exc:
        msg = f"input {role} JSON cannot be resolved"
        raise InvalidGeometryLabelQaInputError(msg) from exc
    if not resolved_path.is_file():
        msg = f"input {role} JSON must be a regular file"
        raise InvalidGeometryLabelQaInputError(msg)
    return resolved_path


def _validate_qa_output_path(
    output_path: Path,
    *,
    dataset_root: Path,
    input_paths: tuple[Path, Path],
) -> Path:
    if str(output_path) == "":
        msg = "geometry-label QA output path must not be empty"
        raise UnsafeGeometryLabelQaOutputPathError(msg)
    if not output_path.is_absolute():
        msg = "geometry-label QA output path must be explicit and absolute"
        raise UnsafeGeometryLabelQaOutputPathError(msg)
    if any(part == ".." for part in output_path.parts):
        msg = "geometry-label QA output path must not contain parent traversal"
        raise UnsafeGeometryLabelQaOutputPathError(msg)
    resolved_non_strict = output_path.resolve(strict=False)
    if any(resolved_non_strict == input_path for input_path in input_paths):
        msg = "geometry-label QA output path must not equal an input JSON path"
        raise UnsafeGeometryLabelQaOutputPathError(msg)
    if output_path.exists() and output_path.is_file():
        msg = "geometry-label QA output path already exists"
        raise ExistingGeometryLabelQaOutputError(msg)

    safe_output_path = _resolve_path_with_existing_ancestor(output_path)
    if _paths_are_equal_or_nested(safe_output_path, dataset_root):
        msg = "geometry-label QA output path must not equal or be inside the dataset root"
        raise UnsafeGeometryLabelQaOutputPathError(msg)
    if _paths_are_equal_or_nested(dataset_root, safe_output_path):
        msg = "geometry-label QA output path must not contain the dataset root"
        raise UnsafeGeometryLabelQaOutputPathError(msg)
    if any(safe_output_path == input_path for input_path in input_paths):
        msg = "geometry-label QA output path must not equal an input JSON path"
        raise UnsafeGeometryLabelQaOutputPathError(msg)
    if output_path.exists():
        msg = "geometry-label QA output path already exists"
        raise ExistingGeometryLabelQaOutputError(msg)
    return safe_output_path


def _resolve_path_with_existing_ancestor(path: Path) -> Path:
    missing_parts: list[str] = []
    current = path
    while not current.exists():
        missing_parts.append(current.name)
        parent = current.parent
        if parent == current:
            msg = "geometry-label QA output path has no existing parent"
            raise UnsafeGeometryLabelQaOutputPathError(msg)
        current = parent
    if current.is_file():
        msg = "geometry-label QA output parent must not be a regular file"
        raise UnsafeGeometryLabelQaOutputPathError(msg)
    if not current.is_dir():
        msg = "geometry-label QA output parent must resolve beneath a directory"
        raise UnsafeGeometryLabelQaOutputPathError(msg)
    try:
        resolved = current.resolve(strict=True)
    except OSError as exc:
        msg = "geometry-label QA output path cannot be resolved"
        raise UnsafeGeometryLabelQaOutputPathError(msg) from exc
    for part in reversed(missing_parts):
        resolved = resolved / part
    return resolved


def _publish_qa_json(artifact: GeometryLabelQaArtifact, output_path: Path) -> None:
    try:
        text = phase2_artifact_to_json(artifact)
    except Phase2ArtifactError as exc:
        msg = "failed to serialize geometry-label QA artifact"
        raise GeometryLabelQaPublicationError(msg) from exc
    temp_path = output_path.with_name(f".{output_path.name}.tmp")
    created_temp = False
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if temp_path.exists():
            msg = "temporary geometry-label QA output already exists"
            raise ExistingGeometryLabelQaOutputError(msg)
        temp_path.write_text(text, encoding="utf-8", newline="\n")
        created_temp = True
        if output_path.exists():
            msg = "geometry-label QA output path already exists"
            raise ExistingGeometryLabelQaOutputError(msg)
        os.link(temp_path, output_path)
        temp_path.unlink()
        created_temp = False
    except ExistingGeometryLabelQaOutputError:
        raise
    except OSError as exc:
        msg = "failed to publish geometry-label QA JSON"
        raise GeometryLabelQaPublicationError(msg) from exc
    finally:
        if created_temp and temp_path.exists():
            temp_path.unlink()


def _require_explicit_metadata(value: object, field_name: str) -> None:
    if not isinstance(value, str) or value == "" or value != value.strip():
        msg = f"{field_name} must be supplied explicitly as a nonempty string"
        raise InvalidGeometryLabelQaInputError(msg)
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        msg = f"{field_name} must not contain control characters"
        raise InvalidGeometryLabelQaInputError(msg)


def _paths_are_equal_or_nested(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


__all__ = [
    "GEOMETRY_LABEL_QA_CONFIG_PAYLOAD_TYPE",
    "GeometryLabelQaAssignmentError",
    "GeometryLabelQaConfig",
    "GeometryLabelQaError",
    "GeometryLabelQaHashMismatchError",
    "GeometryLabelQaNiftiReadError",
    "GeometryLabelQaPublicationError",
    "GeometryLabelQaSourceIntegrityError",
    "ExistingGeometryLabelQaOutputError",
    "InvalidGeometryLabelQaConfigError",
    "InvalidGeometryLabelQaInputError",
    "UnsafeGeometryLabelQaOutputPathError",
    "geometry_label_qa_config_hash_payload",
    "hash_geometry_label_qa_config",
    "run_geometry_label_qa",
]
