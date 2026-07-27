"""Read-only Phase 2 deterministic 3D connected-component lesion summaries."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final, cast

import nibabel as nib
import numpy as np
import numpy.typing as npt
from nibabel.filebasedimages import ImageFileError
from scipy.ndimage import label as scipy_label  # type: ignore[import-untyped]

import protoem_ct.artifacts.hashing as artifact_hashing
from protoem_ct.artifacts import (
    LESION_COMPONENTS_STAGE,
    LESION_COMPONENTS_UPSTREAM_FAILURE_REASON,
    PHASE2_DEVELOPMENT_COHORT_ROLE,
    PHASE2_SCHEMA_VERSION,
    DatasetCaseRecord,
    DatasetManifest,
    GeometryLabelQaArtifact,
    GeometryLabelQaCaseRecord,
    LesionComponentCaseRecord,
    LesionComponentsArtifact,
    LesionSummaryRecord,
    Phase2ArtifactError,
    hash_dataset_manifest,
    hash_geometry_label_qa,
    hash_lesion_components,
    phase2_artifact_from_json,
    phase2_artifact_to_json,
    sha256_json,
)
from protoem_ct.data._phase2_publication import (
    Phase2PublicationExistingOutputError,
    Phase2PublicationIOError,
    publish_text_no_overwrite,
)
from protoem_ct.data.phase2_paths import (
    Phase2PathError,
    require_unique_resolved_paths,
    resolve_regular_file_beneath_root,
    validate_explicit_dataset_root,
)

LESION_COMPONENTS_CONFIG_PAYLOAD_TYPE: Final[str] = "phase2-lesion-components-config-v1"
SUPPORTED_LESION_CONNECTIVITY: Final[tuple[int, int, int]] = (6, 18, 26)
_ZERO_SHA256: Final[str] = "0" * 64

Array = npt.NDArray[np.generic]
BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.integer[Any]]


class LesionComponentsError(ValueError):
    """Base exception for Phase 2 lesion connected-component failures."""


class InvalidLesionComponentsConfigError(LesionComponentsError):
    """Raised when explicit lesion-component configuration is invalid."""


class InvalidLesionComponentsInputError(LesionComponentsError):
    """Raised when manifest, geometry QA, or explicit metadata inputs are invalid."""


class LesionComponentsHashMismatchError(InvalidLesionComponentsInputError):
    """Raised when a stored input artifact hash does not match recomputed content."""


class LesionComponentsLinkageError(InvalidLesionComponentsInputError):
    """Raised when manifest and geometry QA case linkage is inconsistent."""


class LesionComponentsSourceIntegrityError(LesionComponentsError):
    """Raised when label files are missing, aliased, changed, or unsafe."""


class LesionComponentsNiftiReadError(LesionComponentsError):
    """Raised when an integrity-verified label cannot be read as NIfTI."""


class UnsafeLesionComponentsOutputPathError(LesionComponentsError):
    """Raised when the requested lesion output path violates containment policy."""


class ExistingLesionComponentsOutputError(LesionComponentsError):
    """Raised when lesion publication would overwrite an existing output."""


class LesionComponentsPublicationError(LesionComponentsError):
    """Raised when validated lesion JSON cannot be published atomically."""


@dataclass(frozen=True, slots=True)
class LesionComponentsConfig:
    """Explicit settings for deterministic 3D connected-component lesion summaries.

    Connectivity is supplied by the caller for Phase 2 synthetic verification. This module does
    not choose a final real-data connectivity, fit preprocessing, or alter labels.
    """

    connectivity: int
    tumor_label_value: int

    def __post_init__(self) -> None:
        if self.connectivity not in SUPPORTED_LESION_CONNECTIVITY:
            msg = f"connectivity must be one of {SUPPORTED_LESION_CONNECTIVITY!r}"
            raise InvalidLesionComponentsConfigError(msg)
        if isinstance(self.tumor_label_value, bool) or not isinstance(self.tumor_label_value, int):
            msg = "tumor_label_value must be an integer"
            raise InvalidLesionComponentsConfigError(msg)


def lesion_components_config_hash_payload(config: LesionComponentsConfig) -> dict[str, object]:
    """Return the canonical payload for a lesion-components configuration hash."""
    return {
        "connectivity": config.connectivity,
        "payload_type": LESION_COMPONENTS_CONFIG_PAYLOAD_TYPE,
        "schema_version": PHASE2_SCHEMA_VERSION,
        "tumor_label_value": config.tumor_label_value,
    }


def hash_lesion_components_config(config: LesionComponentsConfig) -> str:
    """Return the deterministic SHA-256 hash of explicit lesion-component settings."""
    return sha256_json(lesion_components_config_hash_payload(config))


def run_lesion_component_analysis(
    manifest_path: Path,
    geometry_qa_artifact_path: Path,
    *,
    dataset_root: Path,
    output_path: Path,
    config: LesionComponentsConfig,
    git_commit: str,
    created_at_utc: str,
) -> LesionComponentsArtifact:
    """Run read-only connected-component lesion analysis and publish an artifact."""
    source_manifest_path = _validate_input_json_path(manifest_path, "manifest")
    source_geometry_path = _validate_input_json_path(
        geometry_qa_artifact_path,
        "geometry QA artifact",
    )
    canonical_root = validate_explicit_dataset_root(dataset_root)
    safe_output_path = _validate_lesion_output_path(
        output_path,
        dataset_root=canonical_root,
        input_paths=(source_manifest_path, source_geometry_path),
    )
    _require_explicit_metadata(git_commit, "git_commit")
    _require_explicit_metadata(created_at_utc, "created_at_utc")

    manifest = _load_and_verify_dataset_manifest(source_manifest_path.read_text(encoding="utf-8"))
    geometry = _load_and_verify_geometry_qa(source_geometry_path.read_text(encoding="utf-8"))
    _verify_manifest_geometry_linkage(manifest, geometry, config)

    verified_labels = _resolve_and_verify_passing_labels(canonical_root, manifest, geometry)
    case_records = tuple(
        sorted(
            (
                _lesion_case_record(
                    case=case,
                    geometry_case=geometry_case,
                    label_path=verified_labels.get(case.anonymous_case_id),
                    config=config,
                )
                for case, geometry_case in zip(manifest.cases, geometry.case_records, strict=True)
            ),
            key=lambda record: (record.anonymous_patient_id, record.anonymous_case_id),
        )
    )
    artifact_without_hash = LesionComponentsArtifact(
        schema_version=PHASE2_SCHEMA_VERSION,
        stage=LESION_COMPONENTS_STAGE,
        created_at_utc=created_at_utc,
        git_commit=git_commit,
        config_hash=hash_lesion_components_config(config),
        manifest_hash=manifest.manifest_hash,
        split_hash=geometry.split_hash,
        geometry_qa_artifact_hash=geometry.qa_artifact_hash,
        connectivity=config.connectivity,
        case_count=len(case_records),
        analyzed_case_count=sum(1 for record in case_records if record.analysis_performed),
        skipped_case_count=sum(1 for record in case_records if not record.analysis_performed),
        case_records=case_records,
        lesion_artifact_hash=_ZERO_SHA256,
    )
    lesion_hash = hash_lesion_components(artifact_without_hash)
    artifact = replace(artifact_without_hash, lesion_artifact_hash=lesion_hash)
    if hash_lesion_components(artifact) != lesion_hash:
        msg = "lesion-components artifact hash verification failed"
        raise LesionComponentsPublicationError(msg)

    _publish_lesion_json(artifact, safe_output_path)
    return artifact


def _load_and_verify_dataset_manifest(manifest_text: str) -> DatasetManifest:
    try:
        artifact = phase2_artifact_from_json(manifest_text)
    except Phase2ArtifactError as exc:
        msg = "input manifest is not a valid Phase 2 dataset manifest"
        raise InvalidLesionComponentsInputError(msg) from exc
    if not isinstance(artifact, DatasetManifest):
        msg = "input artifact must be a DatasetManifest"
        raise InvalidLesionComponentsInputError(msg)
    if artifact.cohort_role != PHASE2_DEVELOPMENT_COHORT_ROLE:
        msg = "input dataset manifest must use the development cohort role"
        raise InvalidLesionComponentsInputError(msg)
    if hash_dataset_manifest(artifact) != artifact.manifest_hash:
        msg = "input dataset manifest hash does not match its content"
        raise LesionComponentsHashMismatchError(msg)
    return artifact


def _load_and_verify_geometry_qa(geometry_text: str) -> GeometryLabelQaArtifact:
    try:
        artifact = phase2_artifact_from_json(geometry_text)
    except Phase2ArtifactError as exc:
        msg = "input geometry QA artifact is not a valid Phase 2 artifact"
        raise InvalidLesionComponentsInputError(msg) from exc
    if not isinstance(artifact, GeometryLabelQaArtifact):
        msg = "input artifact must be a GeometryLabelQaArtifact"
        raise InvalidLesionComponentsInputError(msg)
    if hash_geometry_label_qa(artifact) != artifact.qa_artifact_hash:
        msg = "input geometry QA artifact hash does not match its content"
        raise LesionComponentsHashMismatchError(msg)
    return artifact


def _verify_manifest_geometry_linkage(
    manifest: DatasetManifest,
    geometry: GeometryLabelQaArtifact,
    config: LesionComponentsConfig,
) -> None:
    if geometry.manifest_hash != manifest.manifest_hash:
        msg = "geometry QA manifest_hash must equal the dataset manifest hash"
        raise LesionComponentsLinkageError(msg)
    manifest_key = tuple(
        (case.anonymous_patient_id, case.anonymous_case_id) for case in manifest.cases
    )
    geometry_key = tuple(
        (case.anonymous_patient_id, case.anonymous_case_id) for case in geometry.case_records
    )
    if manifest_key != geometry_key:
        msg = "geometry QA case ordering and identifiers must match manifest cases exactly"
        raise LesionComponentsLinkageError(msg)
    for record in geometry.case_records:
        if record.tumor_label_value != config.tumor_label_value:
            msg = "geometry QA tumor label must match lesion-components configuration"
            raise LesionComponentsLinkageError(msg)
        if config.tumor_label_value not in record.allowed_label_values:
            msg = "configured tumor label must be present in geometry QA allowed labels"
            raise LesionComponentsLinkageError(msg)


def _resolve_and_verify_passing_labels(
    dataset_root: Path,
    manifest: DatasetManifest,
    geometry: GeometryLabelQaArtifact,
) -> dict[str, Path]:
    passing_case_ids = {
        record.anonymous_case_id for record in geometry.case_records if record.qa_passed
    }
    resolved: dict[str, Path] = {}
    named_paths: list[tuple[str, Path]] = []
    try:
        for case in manifest.cases:
            if case.anonymous_case_id not in passing_case_ids:
                continue
            label_path = resolve_regular_file_beneath_root(
                dataset_root,
                case.relative_label_path,
            )
            resolved[case.anonymous_case_id] = label_path
            named_paths.append((f"{case.anonymous_case_id}:label", label_path))
        require_unique_resolved_paths(tuple(named_paths))
    except Phase2PathError as exc:
        msg = "source label path violates the Phase 2 containment contract"
        raise LesionComponentsSourceIntegrityError(msg) from exc

    try:
        for case in manifest.cases:
            if case.anonymous_case_id not in passing_case_ids:
                continue
            if artifact_hashing.sha256_file(resolved[case.anonymous_case_id]) != case.label_sha256:
                msg = "source label file hash does not match the manifest"
                raise LesionComponentsSourceIntegrityError(msg)
    except artifact_hashing.HashingError as exc:
        msg = "failed to hash an integrity-checked label file"
        raise LesionComponentsSourceIntegrityError(msg) from exc
    return resolved


def _lesion_case_record(
    *,
    case: DatasetCaseRecord,
    geometry_case: GeometryLabelQaCaseRecord,
    label_path: Path | None,
    config: LesionComponentsConfig,
) -> LesionComponentCaseRecord:
    if not geometry_case.qa_passed:
        return LesionComponentCaseRecord(
            anonymous_patient_id=case.anonymous_patient_id,
            anonymous_case_id=case.anonymous_case_id,
            partition=geometry_case.partition,
            analysis_performed=False,
            connectivity=config.connectivity,
            tumor_label_value=config.tumor_label_value,
            voxel_volume_mm3=None,
            tumor_voxel_count=None,
            tumor_physical_volume_mm3=None,
            lesion_count=None,
            lesions=(),
            qa_passed=False,
            failure_reasons=(LESION_COMPONENTS_UPSTREAM_FAILURE_REASON,),
        )
    if label_path is None:
        msg = "passing geometry QA case is missing a verified label path"
        raise LesionComponentsSourceIntegrityError(msg)

    label = _load_nifti(label_path)
    label_array = _nifti_array(label)
    label_affine = _affine_tuple(label.affine, "label_affine")
    label_spacing = _spacing_tuple(label_affine, "label_spacing")
    label_orientation = _orientation_tuple(label_affine, "label_orientation")
    _verify_geometry_consistency(
        label_array=label_array,
        label_affine=label_affine,
        label_spacing=label_spacing,
        label_orientation=label_orientation,
        geometry_case=geometry_case,
        tumor_label_value=config.tumor_label_value,
    )

    voxel_volume = float(math.prod(label_spacing))
    tumor_mask = cast(BoolArray, label_array == config.tumor_label_value)
    lesions = _summarize_components(
        tumor_mask=tumor_mask,
        voxel_volume_mm3=voxel_volume,
        connectivity=config.connectivity,
    )
    tumor_voxel_count = sum(lesion.voxel_count for lesion in lesions)
    tumor_physical_volume = float(tumor_voxel_count * voxel_volume)
    return LesionComponentCaseRecord(
        anonymous_patient_id=case.anonymous_patient_id,
        anonymous_case_id=case.anonymous_case_id,
        partition=geometry_case.partition,
        analysis_performed=True,
        connectivity=config.connectivity,
        tumor_label_value=config.tumor_label_value,
        voxel_volume_mm3=voxel_volume,
        tumor_voxel_count=tumor_voxel_count,
        tumor_physical_volume_mm3=tumor_physical_volume,
        lesion_count=len(lesions),
        lesions=lesions,
        qa_passed=True,
        failure_reasons=(),
    )


def _load_nifti(path: Path) -> Any:
    try:
        return nib.load(str(path), mmap=False)
    except (ImageFileError, OSError, ValueError) as exc:
        msg = "label source is not a readable NIfTI file"
        raise LesionComponentsNiftiReadError(msg) from exc


def _nifti_array(image: Any) -> Array:
    try:
        dataobj = image.dataobj
        if hasattr(dataobj, "get_unscaled"):
            return cast(Array, np.asarray(dataobj.get_unscaled()))
        return cast(Array, np.asarray(dataobj))
    except (OSError, ValueError, TypeError) as exc:
        msg = "label NIfTI data cannot be loaded"
        raise LesionComponentsNiftiReadError(msg) from exc


def _affine_tuple(value: object, field_name: str) -> tuple[tuple[float, float, float, float], ...]:
    array = np.asarray(value, dtype=float)
    if array.shape != (4, 4) or not bool(np.isfinite(array).all()):
        msg = f"{field_name} must be a finite 4x4 affine"
        raise LesionComponentsNiftiReadError(msg)
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
        raise LesionComponentsNiftiReadError(msg)
    return cast(tuple[float, float, float], tuple(float(value) for value in spacing.tolist()))


def _orientation_tuple(
    affine: tuple[tuple[float, float, float, float], ...],
    field_name: str,
) -> tuple[str, str, str]:
    axis_codes = nib.aff2axcodes(np.asarray(affine, dtype=float))  # type: ignore[no-untyped-call]
    if len(axis_codes) != 3 or any(code is None for code in axis_codes):
        msg = f"{field_name} could not be derived from affine"
        raise LesionComponentsNiftiReadError(msg)
    return cast(tuple[str, str, str], tuple(str(code) for code in axis_codes))


def _verify_geometry_consistency(
    *,
    label_array: Array,
    label_affine: tuple[tuple[float, float, float, float], ...],
    label_spacing: tuple[float, float, float],
    label_orientation: tuple[str, str, str],
    geometry_case: GeometryLabelQaCaseRecord,
    tumor_label_value: int,
) -> None:
    if label_array.ndim != 3:
        msg = "label dimensionality changed after geometry QA"
        raise LesionComponentsNiftiReadError(msg)
    if not bool(np.isfinite(label_array).all()):
        msg = "label finite-value status changed after geometry QA"
        raise LesionComponentsSourceIntegrityError(msg)
    if not _label_is_integer_valued(label_array):
        msg = "label integer-value status changed after geometry QA"
        raise LesionComponentsSourceIntegrityError(msg)
    if tuple(int(dimension) for dimension in label_array.shape) != geometry_case.label_shape:
        msg = "label shape does not match geometry QA artifact"
        raise LesionComponentsSourceIntegrityError(msg)
    if label_affine != geometry_case.label_affine:
        msg = "label affine does not match geometry QA artifact"
        raise LesionComponentsSourceIntegrityError(msg)
    if label_spacing != geometry_case.label_spacing:
        msg = "label spacing does not match geometry QA artifact"
        raise LesionComponentsSourceIntegrityError(msg)
    if label_orientation != geometry_case.label_orientation:
        msg = "label orientation does not match geometry QA artifact"
        raise LesionComponentsSourceIntegrityError(msg)
    observed_tumor_count = int(np.count_nonzero(label_array == tumor_label_value))
    if observed_tumor_count != geometry_case.tumor_voxel_count:
        msg = "label tumor voxel count does not match geometry QA artifact"
        raise LesionComponentsSourceIntegrityError(msg)


def _label_is_integer_valued(array: Array) -> bool:
    if np.iscomplexobj(cast(Any, array)):
        return False
    try:
        return bool(np.equal(cast(Any, array), np.round(cast(Any, array))).all())
    except TypeError:
        return False


def _summarize_components(
    *,
    tumor_mask: BoolArray,
    voxel_volume_mm3: float,
    connectivity: int,
) -> tuple[LesionSummaryRecord, ...]:
    labeled_array, component_count = scipy_label(
        tumor_mask, structure=_connectivity_structure(connectivity)
    )
    if component_count == 0:
        return ()

    labeled = cast(IntArray, labeled_array)
    components: list[tuple[int, tuple[int, int, int], int]] = []
    for raw_component_index in range(1, int(component_count) + 1):
        coordinates = np.argwhere(labeled == raw_component_index)
        voxel_count = int(coordinates.shape[0])
        min_coordinate = min(
            cast(tuple[int, int, int], tuple(int(value) for value in coordinate))
            for coordinate in coordinates.tolist()
        )
        components.append((voxel_count, min_coordinate, raw_component_index))

    ordered = sorted(components, key=lambda item: (-item[0], item[1], item[2]))
    return tuple(
        LesionSummaryRecord(
            lesion_index=index,
            voxel_count=voxel_count,
            physical_volume_mm3=float(voxel_count * voxel_volume_mm3),
        )
        for index, (voxel_count, _min_coordinate, _component_index) in enumerate(ordered, start=1)
    )


def _connectivity_structure(connectivity: int) -> BoolArray:
    grid = np.indices((3, 3, 3), dtype=int) - 1
    manhattan_distance = np.abs(grid).sum(axis=0)
    if connectivity == 6:
        return cast(BoolArray, manhattan_distance <= 1)
    if connectivity == 18:
        return cast(BoolArray, manhattan_distance <= 2)
    if connectivity == 26:
        return cast(BoolArray, manhattan_distance <= 3)
    msg = f"connectivity must be one of {SUPPORTED_LESION_CONNECTIVITY!r}"
    raise InvalidLesionComponentsConfigError(msg)


def _validate_input_json_path(path: Path, role: str) -> Path:
    if str(path) == "":
        msg = f"input {role} path must not be empty"
        raise InvalidLesionComponentsInputError(msg)
    if not path.is_absolute():
        msg = f"input {role} path must be explicit and absolute"
        raise InvalidLesionComponentsInputError(msg)
    try:
        resolved_path = path.resolve(strict=True)
    except FileNotFoundError as exc:
        msg = f"input {role} JSON does not exist"
        raise InvalidLesionComponentsInputError(msg) from exc
    except OSError as exc:
        msg = f"input {role} JSON cannot be resolved"
        raise InvalidLesionComponentsInputError(msg) from exc
    if not resolved_path.is_file():
        msg = f"input {role} JSON must be a regular file"
        raise InvalidLesionComponentsInputError(msg)
    return resolved_path


def _validate_lesion_output_path(
    output_path: Path,
    *,
    dataset_root: Path,
    input_paths: tuple[Path, Path],
) -> Path:
    if str(output_path) == "":
        msg = "lesion-components output path must not be empty"
        raise UnsafeLesionComponentsOutputPathError(msg)
    if not output_path.is_absolute():
        msg = "lesion-components output path must be explicit and absolute"
        raise UnsafeLesionComponentsOutputPathError(msg)
    if any(part == ".." for part in output_path.parts):
        msg = "lesion-components output path must not contain parent traversal"
        raise UnsafeLesionComponentsOutputPathError(msg)
    resolved_non_strict = output_path.resolve(strict=False)
    if any(resolved_non_strict == input_path for input_path in input_paths):
        msg = "lesion-components output path must not equal an input JSON path"
        raise UnsafeLesionComponentsOutputPathError(msg)
    if output_path.exists() and output_path.is_file():
        msg = "lesion-components output path already exists"
        raise ExistingLesionComponentsOutputError(msg)

    safe_output_path = _resolve_path_with_existing_ancestor(output_path)
    if _paths_are_equal_or_nested(safe_output_path, dataset_root):
        msg = "lesion-components output path must not equal or be inside the dataset root"
        raise UnsafeLesionComponentsOutputPathError(msg)
    if _paths_are_equal_or_nested(dataset_root, safe_output_path):
        msg = "lesion-components output path must not contain the dataset root"
        raise UnsafeLesionComponentsOutputPathError(msg)
    if any(safe_output_path == input_path for input_path in input_paths):
        msg = "lesion-components output path must not equal an input JSON path"
        raise UnsafeLesionComponentsOutputPathError(msg)
    if output_path.exists():
        msg = "lesion-components output path already exists"
        raise ExistingLesionComponentsOutputError(msg)
    return safe_output_path


def _resolve_path_with_existing_ancestor(path: Path) -> Path:
    missing_parts: list[str] = []
    current = path
    while not current.exists():
        missing_parts.append(current.name)
        parent = current.parent
        if parent == current:
            msg = "lesion-components output path has no existing parent"
            raise UnsafeLesionComponentsOutputPathError(msg)
        current = parent
    if current.is_file():
        msg = "lesion-components output parent must not be a regular file"
        raise UnsafeLesionComponentsOutputPathError(msg)
    if not current.is_dir():
        msg = "lesion-components output parent must resolve beneath a directory"
        raise UnsafeLesionComponentsOutputPathError(msg)
    try:
        resolved = current.resolve(strict=True)
    except OSError as exc:
        msg = "lesion-components output path cannot be resolved"
        raise UnsafeLesionComponentsOutputPathError(msg) from exc
    for part in reversed(missing_parts):
        resolved = resolved / part
    return resolved


def _publish_lesion_json(artifact: LesionComponentsArtifact, output_path: Path) -> None:
    try:
        text = phase2_artifact_to_json(artifact)
    except Phase2ArtifactError as exc:
        msg = "failed to serialize lesion-components artifact"
        raise LesionComponentsPublicationError(msg) from exc
    try:
        publish_text_no_overwrite(
            text=text,
            output_path=output_path,
            temporary_exists_message="temporary lesion-components output already exists",
            final_exists_message="lesion-components output path already exists",
        )
    except Phase2PublicationExistingOutputError as exc:
        raise ExistingLesionComponentsOutputError(str(exc)) from exc
    except Phase2PublicationIOError as exc:
        msg = "failed to publish lesion-components JSON"
        raise LesionComponentsPublicationError(msg) from exc


def _require_explicit_metadata(value: object, field_name: str) -> None:
    if not isinstance(value, str) or value == "" or value != value.strip():
        msg = f"{field_name} must be supplied explicitly as a nonempty string"
        raise InvalidLesionComponentsInputError(msg)
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        msg = f"{field_name} must not contain control characters"
        raise InvalidLesionComponentsInputError(msg)


def _paths_are_equal_or_nested(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


__all__ = [
    "LESION_COMPONENTS_CONFIG_PAYLOAD_TYPE",
    "SUPPORTED_LESION_CONNECTIVITY",
    "ExistingLesionComponentsOutputError",
    "InvalidLesionComponentsConfigError",
    "InvalidLesionComponentsInputError",
    "LesionComponentsConfig",
    "LesionComponentsError",
    "LesionComponentsHashMismatchError",
    "LesionComponentsLinkageError",
    "LesionComponentsNiftiReadError",
    "LesionComponentsPublicationError",
    "LesionComponentsSourceIntegrityError",
    "UnsafeLesionComponentsOutputPathError",
    "hash_lesion_components_config",
    "lesion_components_config_hash_payload",
    "run_lesion_component_analysis",
]
