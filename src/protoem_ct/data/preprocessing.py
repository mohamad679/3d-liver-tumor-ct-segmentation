"""Deterministic Phase 1 synthetic preprocessing stage."""

from __future__ import annotations

import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any, cast

import nibabel as nib
import numpy as np
import numpy.typing as npt
from nibabel.filebasedimages import ImageFileError
from omegaconf import DictConfig, OmegaConf

from protoem_ct.artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    ArtifactError,
    ArtifactValidationError,
    PreprocessArtifact,
    SyntheticManifest,
    ValidationArtifact,
    artifact_from_json,
    artifact_to_json,
    hash_config,
    hash_manifest,
)
from protoem_ct.artifacts.hashing import HashingError, JsonValue
from protoem_ct.data.synthetic import synthetic_manifest_hash_payload
from protoem_ct.data.validation import NiftiValidationError, validate_nifti_pair

IMAGE_OUTPUT_DTYPE = "float32"
LABEL_OUTPUT_DTYPE = "uint8"

_CONFIG_SECTION = "preprocessing"
_EXPECTED_CONFIG_KEYS = {
    "clip_min",
    "clip_max",
    "output_min",
    "output_max",
    "image_dtype",
    "label_dtype",
}

Array = npt.NDArray[np.generic]
Float32Array = npt.NDArray[np.float32]
UInt8Array = npt.NDArray[np.uint8]
AffineArray = npt.NDArray[np.float64]


class PreprocessingError(ValueError):
    """Base error for Phase 1 synthetic preprocessing failures."""


class PreprocessingConfigError(PreprocessingError):
    """Raised when preprocessing configuration is invalid."""


class PreprocessingInputArtifactError(PreprocessingError):
    """Raised when input artifacts are missing, altered, or inconsistent."""


class PreprocessingPathError(PreprocessingError):
    """Raised when preprocessing input or output paths are unsafe."""


class PreprocessingFailureError(PreprocessingError):
    """Raised when a case cannot be preprocessed."""


class PreprocessingOutputCollisionError(PreprocessingError):
    """Raised when preprocessing would overwrite an existing output."""


@dataclass(frozen=True, slots=True)
class PreprocessingConfig:
    """Validated immutable configuration for deterministic preprocessing."""

    clip_min: float
    clip_max: float
    output_min: float
    output_max: float
    image_dtype: str
    label_dtype: str


@dataclass(frozen=True, slots=True)
class _InputCasePaths:
    case_id: str
    image_path: Path
    label_path: Path


@dataclass(frozen=True, slots=True)
class _OutputRootPlan:
    final_root: Path
    parent: Path
    existed_empty: bool


def load_preprocessing_config(config_path: Path) -> PreprocessingConfig:
    """Load and validate preprocessing settings from an OmegaConf YAML file."""
    try:
        raw_config = OmegaConf.load(config_path)
    except OSError as exc:
        msg = f"failed to read preprocessing config at {config_path}: {exc}"
        raise PreprocessingConfigError(msg) from exc
    except Exception as exc:
        msg = f"failed to parse preprocessing config at {config_path}: {exc}"
        raise PreprocessingConfigError(msg) from exc

    if not isinstance(raw_config, DictConfig):
        msg = "preprocessing config must contain a mapping"
        raise PreprocessingConfigError(msg)

    container = OmegaConf.to_container(raw_config, resolve=True)
    if not isinstance(container, dict):
        msg = "preprocessing config must resolve to a mapping"
        raise PreprocessingConfigError(msg)

    resolved = cast(dict[str, object], container)
    config_mapping = _extract_preprocessing_mapping(resolved)
    unknown_keys = set(config_mapping) - _EXPECTED_CONFIG_KEYS
    missing_keys = _EXPECTED_CONFIG_KEYS - set(config_mapping)
    if unknown_keys:
        msg = f"unknown preprocessing config keys: {sorted(unknown_keys)}"
        raise PreprocessingConfigError(msg)
    if missing_keys:
        msg = f"missing preprocessing config keys: {sorted(missing_keys)}"
        raise PreprocessingConfigError(msg)

    clip_min = _validated_finite_float(config_mapping["clip_min"], "clip_min")
    clip_max = _validated_finite_float(config_mapping["clip_max"], "clip_max")
    output_min = _validated_finite_float(config_mapping["output_min"], "output_min")
    output_max = _validated_finite_float(config_mapping["output_max"], "output_max")
    if clip_min >= clip_max:
        msg = "clip_min must be less than clip_max"
        raise PreprocessingConfigError(msg)
    if output_min >= output_max:
        msg = "output_min must be less than output_max"
        raise PreprocessingConfigError(msg)

    return PreprocessingConfig(
        clip_min=clip_min,
        clip_max=clip_max,
        output_min=output_min,
        output_max=output_max,
        image_dtype=_validated_dtype(
            config_mapping["image_dtype"],
            expected=IMAGE_OUTPUT_DTYPE,
            field_name="image_dtype",
        ),
        label_dtype=_validated_dtype(
            config_mapping["label_dtype"],
            expected=LABEL_OUTPUT_DTYPE,
            field_name="label_dtype",
        ),
    )


def preprocessing_config_hash_payload(config: PreprocessingConfig) -> dict[str, JsonValue]:
    """Return the canonical JSON payload used for the preprocessing config hash."""
    return {
        "clip_max": config.clip_max,
        "clip_min": config.clip_min,
        "image_dtype": config.image_dtype,
        "label_dtype": config.label_dtype,
        "output_max": config.output_max,
        "output_min": config.output_min,
    }


def preprocessing_parameters_payload(config: PreprocessingConfig) -> dict[str, JsonValue]:
    """Return preprocessing parameters recorded in the preprocessing artifact."""
    payload = preprocessing_config_hash_payload(config)
    payload["resampling"] = False
    return payload


def preprocess_synthetic_dataset(
    manifest_path: Path,
    validation_artifact_path: Path,
    *,
    data_root: Path,
    output_root: Path,
    artifact_output_path: Path,
    config_path: Path,
    git_commit: str,
    created_at_utc: str,
) -> PreprocessArtifact:
    """Preprocess a validated synthetic manifest into deterministic NIfTI outputs."""
    _require_nonempty_text(git_commit, "git_commit")
    _require_nonempty_text(created_at_utc, "created_at_utc")

    config = load_preprocessing_config(config_path)
    config_hash = hash_config(preprocessing_config_hash_payload(config))
    manifest = _load_synthetic_manifest(manifest_path)
    _verify_manifest_hash(manifest)
    validation_artifact = _load_validation_artifact(validation_artifact_path)
    _verify_validation_artifact(validation_artifact, manifest)

    data_root_resolved = _resolve_existing_directory(data_root, "data_root")
    output_plan = _prepare_output_root(output_root, data_root=data_root_resolved)
    artifact_output = _prepare_artifact_output_path(artifact_output_path)
    input_cases = _resolve_input_cases(manifest, data_root=data_root_resolved)
    output_image_paths, output_label_paths = _relative_output_paths(manifest.case_ids)

    staging_root: Path | None = None
    artifact_temp_path: Path | None = None
    published_output_root = False
    published_artifact = False
    try:
        staging_root = Path(
            tempfile.mkdtemp(
                prefix=f".{output_plan.final_root.name}.tmp-",
                dir=output_plan.parent,
            )
        )
        staged_output_root = staging_root / "outputs"
        _preprocess_cases(
            input_cases,
            output_image_paths=output_image_paths,
            output_label_paths=output_label_paths,
            output_root=staged_output_root,
            config=config,
        )
        artifact = PreprocessArtifact(
            schema_version=ARTIFACT_SCHEMA_VERSION,
            stage="preprocess",
            created_at_utc=created_at_utc,
            git_commit=git_commit,
            config_hash=config_hash,
            manifest_hash=manifest.manifest_hash,
            output_case_ids=manifest.case_ids,
            output_image_paths=tuple(path.as_posix() for path in output_image_paths),
            output_label_paths=tuple(path.as_posix() for path in output_label_paths),
            target_spacing=manifest.spacing,
            preprocessing_parameters=MappingProxyType(preprocessing_parameters_payload(config)),
        )
        artifact_temp_path = _write_artifact_to_temp(artifact, artifact_output)
        _publish_output_root(staged_output_root, output_plan)
        published_output_root = True
        _publish_artifact(artifact_temp_path, artifact_output)
        published_artifact = True
        return artifact
    except PreprocessingError:
        _cleanup_after_failure(
            output_plan.final_root,
            artifact_output,
            artifact_temp_path=artifact_temp_path,
            published_output_root=published_output_root,
            published_artifact=published_artifact,
        )
        raise
    except Exception as exc:
        _cleanup_after_failure(
            output_plan.final_root,
            artifact_output,
            artifact_temp_path=artifact_temp_path,
            published_output_root=published_output_root,
            published_artifact=published_artifact,
        )
        msg = f"preprocessing failed: {exc}"
        raise PreprocessingFailureError(msg) from exc
    finally:
        if staging_root is not None and staging_root.exists():
            shutil.rmtree(staging_root, ignore_errors=True)


def _extract_preprocessing_mapping(resolved: dict[str, object]) -> dict[str, object]:
    if _CONFIG_SECTION in resolved:
        section = resolved[_CONFIG_SECTION]
        if not isinstance(section, dict):
            msg = "preprocessing config section must contain a mapping"
            raise PreprocessingConfigError(msg)
        return cast(dict[str, object], section)
    return resolved


def _validated_finite_float(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = f"{field_name} must be a finite number"
        raise PreprocessingConfigError(msg)
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        msg = f"{field_name} must be a finite number"
        raise PreprocessingConfigError(msg)
    return numeric_value


def _validated_dtype(value: object, *, expected: str, field_name: str) -> str:
    if value != expected:
        msg = f"{field_name} must be fixed to {expected!r}"
        raise PreprocessingConfigError(msg)
    return expected


def _load_synthetic_manifest(manifest_path: Path) -> SyntheticManifest:
    try:
        text = manifest_path.read_text(encoding="utf-8")
    except OSError as exc:
        msg = f"failed to read synthetic manifest at {manifest_path}: {exc}"
        raise PreprocessingInputArtifactError(msg) from exc

    try:
        return artifact_from_json(text, SyntheticManifest)
    except ArtifactValidationError as exc:
        if "artifact path" in str(exc):
            msg = f"unsafe synthetic manifest path at {manifest_path}: {exc}"
            raise PreprocessingPathError(msg) from exc
        msg = f"invalid synthetic manifest at {manifest_path}: {exc}"
        raise PreprocessingInputArtifactError(msg) from exc
    except ArtifactError as exc:
        msg = f"invalid synthetic manifest at {manifest_path}: {exc}"
        raise PreprocessingInputArtifactError(msg) from exc


def _load_validation_artifact(validation_artifact_path: Path) -> ValidationArtifact:
    try:
        text = validation_artifact_path.read_text(encoding="utf-8")
    except OSError as exc:
        msg = f"failed to read validation artifact at {validation_artifact_path}: {exc}"
        raise PreprocessingInputArtifactError(msg) from exc

    try:
        return artifact_from_json(text, ValidationArtifact)
    except ArtifactError as exc:
        msg = f"invalid validation artifact at {validation_artifact_path}: {exc}"
        raise PreprocessingInputArtifactError(msg) from exc


def _verify_manifest_hash(manifest: SyntheticManifest) -> None:
    try:
        recomputed_hash = hash_manifest(synthetic_manifest_hash_payload(manifest))
    except (ArtifactError, HashingError) as exc:
        msg = f"failed to recompute synthetic manifest hash: {exc}"
        raise PreprocessingInputArtifactError(msg) from exc

    if recomputed_hash != manifest.manifest_hash:
        msg = (
            "synthetic manifest hash mismatch: "
            f"stored={manifest.manifest_hash}, recomputed={recomputed_hash}"
        )
        raise PreprocessingInputArtifactError(msg)


def _verify_validation_artifact(
    validation_artifact: ValidationArtifact,
    manifest: SyntheticManifest,
) -> None:
    if validation_artifact.config_hash != manifest.config_hash:
        msg = (
            "validation artifact config hash does not match manifest config hash: "
            f"validation={validation_artifact.config_hash}, manifest={manifest.config_hash}"
        )
        raise PreprocessingInputArtifactError(msg)
    if validation_artifact.manifest_hash != manifest.manifest_hash:
        msg = (
            "validation artifact manifest hash does not match manifest hash: "
            f"validation={validation_artifact.manifest_hash}, manifest={manifest.manifest_hash}"
        )
        raise PreprocessingInputArtifactError(msg)
    if validation_artifact.invalid_case_count != 0:
        msg = "validation artifact invalid_case_count must equal zero"
        raise PreprocessingInputArtifactError(msg)
    if validation_artifact.validated_case_ids != manifest.case_ids:
        msg = "validation artifact validated_case_ids must match manifest case_ids in order"
        raise PreprocessingInputArtifactError(msg)
    if validation_artifact.valid_case_count != len(manifest.case_ids):
        msg = "validation artifact valid_case_count must match manifest case count"
        raise PreprocessingInputArtifactError(msg)


def _resolve_existing_directory(path: Path, role: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        msg = f"{role} does not exist or cannot be resolved: {path}"
        raise PreprocessingPathError(msg) from exc
    if not resolved.is_dir():
        msg = f"{role} is not a directory: {path}"
        raise PreprocessingPathError(msg)
    return resolved


def _resolve_input_cases(
    manifest: SyntheticManifest,
    *,
    data_root: Path,
) -> tuple[_InputCasePaths, ...]:
    seen_paths: set[Path] = set()
    image_paths = _resolve_manifest_paths(
        manifest.image_paths,
        data_root=data_root,
        role="image",
        seen_paths=seen_paths,
    )
    label_paths = _resolve_manifest_paths(
        manifest.label_paths,
        data_root=data_root,
        role="label",
        seen_paths=seen_paths,
    )
    return tuple(
        _InputCasePaths(case_id=case_id, image_path=image_path, label_path=label_path)
        for case_id, image_path, label_path in zip(
            manifest.case_ids,
            image_paths,
            label_paths,
            strict=True,
        )
    )


def _resolve_manifest_paths(
    paths: tuple[str, ...],
    *,
    data_root: Path,
    role: str,
    seen_paths: set[Path],
) -> tuple[Path, ...]:
    resolved_paths: list[Path] = []
    for relative_path in paths:
        resolved_path = _resolve_manifest_path(relative_path, data_root=data_root, role=role)
        if resolved_path in seen_paths:
            msg = f"duplicate resolved input path in manifest: {relative_path}"
            raise PreprocessingPathError(msg)
        seen_paths.add(resolved_path)
        resolved_paths.append(resolved_path)
    return tuple(resolved_paths)


def _resolve_manifest_path(relative_path: str, *, data_root: Path, role: str) -> Path:
    _require_relative_manifest_path(relative_path, role)
    posix_path = PurePosixPath(relative_path)
    candidate_path = data_root.joinpath(*posix_path.parts)
    try:
        resolved_path = candidate_path.resolve(strict=True)
    except OSError as exc:
        msg = f"{role} path does not exist or cannot be resolved: {relative_path}"
        raise PreprocessingPathError(msg) from exc

    try:
        resolved_path.relative_to(data_root)
    except ValueError as exc:
        msg = f"{role} path escapes data_root: {relative_path}"
        raise PreprocessingPathError(msg) from exc

    if not resolved_path.is_file():
        msg = f"{role} path is not a regular file: {relative_path}"
        raise PreprocessingPathError(msg)
    return resolved_path


def _require_relative_manifest_path(relative_path: str, role: str) -> None:
    if not isinstance(relative_path, str) or not relative_path:
        msg = f"{role} path must be a nonempty relative path"
        raise PreprocessingPathError(msg)
    if relative_path.startswith("~"):
        msg = f"{role} path must not use a home-relative path: {relative_path}"
        raise PreprocessingPathError(msg)
    posix_path = PurePosixPath(relative_path)
    windows_path = PureWindowsPath(relative_path)
    if posix_path.is_absolute() or windows_path.is_absolute():
        msg = f"{role} path must be relative: {relative_path}"
        raise PreprocessingPathError(msg)
    if ".." in posix_path.parts or ".." in windows_path.parts:
        msg = f"{role} path must not contain parent traversal: {relative_path}"
        raise PreprocessingPathError(msg)


def _prepare_output_root(output_root: Path, *, data_root: Path) -> _OutputRootPlan:
    if _contains_parent_traversal(output_root):
        msg = f"output_root must not contain parent traversal: {output_root}"
        raise PreprocessingPathError(msg)
    if output_root.exists() and not output_root.is_dir():
        msg = f"output root exists and is not a directory: {output_root}"
        raise PreprocessingOutputCollisionError(msg)
    if output_root.exists() and any(output_root.iterdir()):
        msg = f"output root must be empty or nonexistent: {output_root}"
        raise PreprocessingOutputCollisionError(msg)

    if output_root.exists():
        final_root = output_root.resolve(strict=True)
        parent = final_root.parent
        existed_empty = True
    else:
        try:
            parent = output_root.parent.resolve(strict=True)
        except OSError as exc:
            msg = f"output_root parent does not exist or cannot be resolved: {output_root.parent}"
            raise PreprocessingPathError(msg) from exc
        if not parent.is_dir():
            msg = f"output_root parent is not a directory: {output_root.parent}"
            raise PreprocessingPathError(msg)
        final_root = parent / output_root.name
        existed_empty = False

    try:
        final_root.relative_to(data_root)
    except ValueError:
        pass
    else:
        msg = f"output root must not resolve inside data_root: {output_root}"
        raise PreprocessingPathError(msg)

    return _OutputRootPlan(final_root=final_root, parent=parent, existed_empty=existed_empty)


def _prepare_artifact_output_path(artifact_output_path: Path) -> Path:
    if not artifact_output_path.name:
        msg = "artifact_output_path must name a JSON output file"
        raise PreprocessingPathError(msg)
    if _contains_parent_traversal(artifact_output_path):
        msg = f"artifact_output_path must not contain parent traversal: {artifact_output_path}"
        raise PreprocessingPathError(msg)
    if artifact_output_path.exists() or artifact_output_path.is_symlink():
        msg = f"refusing to overwrite existing preprocessing artifact: {artifact_output_path}"
        raise PreprocessingOutputCollisionError(msg)
    artifact_output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        parent = artifact_output_path.parent.resolve(strict=True)
    except OSError as exc:
        msg = f"failed to resolve preprocessing artifact parent: {artifact_output_path.parent}"
        raise PreprocessingPathError(msg) from exc
    if not parent.is_dir():
        msg = f"preprocessing artifact parent is not a directory: {artifact_output_path.parent}"
        raise PreprocessingPathError(msg)
    return parent / artifact_output_path.name


def _relative_output_paths(
    case_ids: tuple[str, ...],
) -> tuple[tuple[PurePosixPath, ...], tuple[PurePosixPath, ...]]:
    image_paths: list[PurePosixPath] = []
    label_paths: list[PurePosixPath] = []
    for case_id in case_ids:
        image_paths.append(_relative_case_output_path("images", case_id))
        label_paths.append(_relative_case_output_path("labels", case_id))
    return tuple(image_paths), tuple(label_paths)


def _relative_case_output_path(directory: str, case_id: str) -> PurePosixPath:
    filename = f"{case_id}.nii"
    posix_filename = PurePosixPath(filename)
    windows_filename = PureWindowsPath(filename)
    if len(posix_filename.parts) != 1 or len(windows_filename.parts) != 1:
        msg = f"case_id cannot be used as a deterministic output file name: {case_id}"
        raise PreprocessingPathError(msg)
    return PurePosixPath(directory) / filename


def _preprocess_cases(
    input_cases: tuple[_InputCasePaths, ...],
    *,
    output_image_paths: tuple[PurePosixPath, ...],
    output_label_paths: tuple[PurePosixPath, ...],
    output_root: Path,
    config: PreprocessingConfig,
) -> None:
    for input_case, output_image_path, output_label_path in zip(
        input_cases,
        output_image_paths,
        output_label_paths,
        strict=True,
    ):
        _preprocess_case(
            input_case,
            output_image_path=output_image_path,
            output_label_path=output_label_path,
            output_root=output_root,
            config=config,
        )


def _preprocess_case(
    input_case: _InputCasePaths,
    *,
    output_image_path: PurePosixPath,
    output_label_path: PurePosixPath,
    output_root: Path,
    config: PreprocessingConfig,
) -> None:
    try:
        validate_nifti_pair(input_case.image_path, input_case.label_path)
    except NiftiValidationError as exc:
        msg = f"case {input_case.case_id} failed input NIfTI validation: {exc}"
        raise PreprocessingFailureError(msg) from exc

    image = _load_nifti(input_case.image_path, "image", case_id=input_case.case_id)
    label = _load_nifti(input_case.label_path, "label", case_id=input_case.case_id)
    image_array = _as_array(image, "image", case_id=input_case.case_id)
    label_array = _as_array(label, "label", case_id=input_case.case_id)
    _require_finite_values(image_array, "image", case_id=input_case.case_id)
    _require_finite_values(label_array, "label", case_id=input_case.case_id)
    _require_binary_label_values(label_array, case_id=input_case.case_id)

    processed_image = _clip_and_scale_image(image_array, config)
    processed_label = cast(UInt8Array, np.asarray(label_array, dtype=np.uint8))
    image_affine = _as_affine(image, "image", case_id=input_case.case_id)
    label_affine = _as_affine(label, "label", case_id=input_case.case_id)

    _write_nifti_file(
        _safe_join_under_root(output_root, output_image_path),
        processed_image,
        image_affine,
        np.dtype(np.float32),
    )
    _write_nifti_file(
        _safe_join_under_root(output_root, output_label_path),
        processed_label,
        label_affine,
        np.dtype(np.uint8),
    )


def _load_nifti(path: Path, role: str, *, case_id: str) -> Any:
    try:
        return cast(Any, nib.load(str(path)))
    except ImageFileError as exc:
        msg = f"case {case_id} {role} path is not a readable NIfTI file: {path}"
        raise PreprocessingFailureError(msg) from exc
    except OSError as exc:
        msg = f"case {case_id} failed to read {role} NIfTI file at {path}: {exc}"
        raise PreprocessingFailureError(msg) from exc
    except ValueError as exc:
        msg = f"case {case_id} has invalid {role} NIfTI data at {path}: {exc}"
        raise PreprocessingFailureError(msg) from exc


def _as_array(image: Any, role: str, *, case_id: str) -> Array:
    try:
        return cast(Array, np.asanyarray(image.dataobj))
    except ValueError as exc:
        msg = f"case {case_id} has invalid {role} array data: {exc}"
        raise PreprocessingFailureError(msg) from exc


def _as_affine(image: Any, role: str, *, case_id: str) -> AffineArray:
    affine = cast(AffineArray, np.asarray(image.affine, dtype=np.float64))
    if affine.shape != (4, 4):
        msg = f"case {case_id} {role} affine must be 4x4, got {affine.shape}"
        raise PreprocessingFailureError(msg)
    if not np.isfinite(affine).all():
        msg = f"case {case_id} {role} affine contains NaN or Infinity"
        raise PreprocessingFailureError(msg)
    return affine


def _require_finite_values(array: Array, role: str, *, case_id: str) -> None:
    try:
        has_only_finite_values = bool(np.isfinite(array).all())
    except TypeError as exc:
        msg = f"case {case_id} {role} data contains values that cannot be checked for finiteness"
        raise PreprocessingFailureError(msg) from exc
    if not has_only_finite_values:
        msg = f"case {case_id} {role} data contains NaN or infinite values"
        raise PreprocessingFailureError(msg)


def _require_binary_label_values(label_array: Array, *, case_id: str) -> None:
    if np.iscomplexobj(label_array):
        msg = f"case {case_id} label data must contain only real binary values drawn from {{0, 1}}"
        raise PreprocessingFailureError(msg)
    unique_values = cast(Array, np.unique(label_array))
    invalid_values = unique_values[~np.isin(unique_values, [0, 1])]
    if invalid_values.size:
        msg = f"case {case_id} label data must be binary with values from {{0, 1}}"
        raise PreprocessingFailureError(msg)


def _clip_and_scale_image(image_array: Array, config: PreprocessingConfig) -> Float32Array:
    image_float = np.asarray(image_array, dtype=np.float64)
    clipped = np.clip(image_float, config.clip_min, config.clip_max)
    unit_scaled = (clipped - config.clip_min) / (config.clip_max - config.clip_min)
    scaled = unit_scaled * (config.output_max - config.output_min) + config.output_min
    return cast(Float32Array, scaled.astype(np.float32, copy=False))


def _write_nifti_file(
    path: Path,
    data: Array,
    affine: AffineArray,
    dtype: np.dtype[Any],
) -> None:
    if path.exists() or path.is_symlink():
        msg = f"refusing to overwrite existing preprocessing output file: {path}"
        raise PreprocessingOutputCollisionError(msg)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(data.astype(dtype, copy=False), affine)  # type: ignore[no-untyped-call]
    image.set_qform(affine, code=1)  # type: ignore[no-untyped-call]
    image.set_sform(affine, code=1)  # type: ignore[no-untyped-call]
    header = image.header
    header.set_data_dtype(dtype)  # type: ignore[no-untyped-call]
    header.set_xyzt_units("mm")  # type: ignore[no-untyped-call]
    header["descrip"] = b"protoem-ct phase1 preprocess"
    nib.save(image, str(path))


def _safe_join_under_root(output_root: Path, relative_path: PurePosixPath) -> Path:
    if relative_path.is_absolute() or ".." in relative_path.parts:
        msg = f"output path must be relative and stay under output root: {relative_path}"
        raise PreprocessingPathError(msg)
    path = output_root.joinpath(*relative_path.parts)
    resolved_path = path.resolve(strict=False)
    root_resolved = output_root.resolve(strict=False)
    try:
        resolved_path.relative_to(root_resolved)
    except ValueError as exc:
        msg = f"output path escapes output root: {relative_path}"
        raise PreprocessingPathError(msg) from exc
    return resolved_path


def _write_artifact_to_temp(
    artifact: PreprocessArtifact,
    artifact_output_path: Path,
) -> Path:
    descriptor, temp_path_text = tempfile.mkstemp(
        prefix=f".{artifact_output_path.name}.",
        suffix=".tmp",
        dir=artifact_output_path.parent,
    )
    os.close(descriptor)
    temp_path = Path(temp_path_text)
    try:
        artifact_to_json(artifact, temp_path)
    except ArtifactError as exc:
        temp_path.unlink(missing_ok=True)
        msg = f"failed to write staged preprocessing artifact: {exc}"
        raise PreprocessingFailureError(msg) from exc
    return temp_path


def _publish_output_root(staged_output_root: Path, output_plan: _OutputRootPlan) -> None:
    if output_plan.final_root.exists():
        if not output_plan.final_root.is_dir() or any(output_plan.final_root.iterdir()):
            msg = f"output root must be empty before publish: {output_plan.final_root}"
            raise PreprocessingOutputCollisionError(msg)
        output_plan.final_root.rmdir()
    elif output_plan.existed_empty:
        msg = f"previously empty output root disappeared before publish: {output_plan.final_root}"
        raise PreprocessingOutputCollisionError(msg)
    staged_output_root.replace(output_plan.final_root)


def _publish_artifact(artifact_temp_path: Path, artifact_output_path: Path) -> None:
    if artifact_output_path.exists() or artifact_output_path.is_symlink():
        msg = f"refusing to overwrite existing preprocessing artifact: {artifact_output_path}"
        raise PreprocessingOutputCollisionError(msg)
    artifact_temp_path.replace(artifact_output_path)


def _cleanup_after_failure(
    output_root: Path,
    artifact_output_path: Path,
    *,
    artifact_temp_path: Path | None,
    published_output_root: bool,
    published_artifact: bool,
) -> None:
    if artifact_temp_path is not None and artifact_temp_path.exists():
        artifact_temp_path.unlink(missing_ok=True)
    if published_artifact and artifact_output_path.exists():
        artifact_output_path.unlink(missing_ok=True)
    if published_output_root and output_root.exists():
        shutil.rmtree(output_root, ignore_errors=True)


def _contains_parent_traversal(path: Path) -> bool:
    value = str(path)
    return ".." in PurePosixPath(value).parts or ".." in PureWindowsPath(value).parts


def _require_nonempty_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        msg = f"{field_name} must be a nonempty string"
        raise PreprocessingInputArtifactError(msg)
