"""Deterministic Phase 1 dummy inference stage."""

from __future__ import annotations

import hashlib
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
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
    InferenceArtifact,
    PreprocessArtifact,
    artifact_from_json,
    artifact_to_json,
    sha256_file,
)

DUMMY_INFERENCE_METHOD = "dummy"
DUMMY_INFERENCE_STAGE = "infer-dummy"
PREDICTION_DTYPE = "uint8"

_CONFIG_SECTION = "dummy_inference"
_EXPECTED_CONFIG_KEYS = {"method", "prediction_dtype", "seed", "threshold"}
_SPATIAL_PERTURBATION_SCALE = 0.05

Array = npt.NDArray[np.generic]
AffineArray = npt.NDArray[np.float64]
UInt8Array = npt.NDArray[np.uint8]


class DummyInferenceError(ValueError):
    """Base error for Phase 1 deterministic dummy inference failures."""


class DummyInferenceConfigError(DummyInferenceError):
    """Raised when dummy-inference configuration is invalid."""


class DummyInferenceInputArtifactError(DummyInferenceError):
    """Raised when the preprocessing artifact is invalid or inconsistent."""


class DummyInferencePathError(DummyInferenceError):
    """Raised when an input or output path is unsafe."""


class DummyInferenceImageError(DummyInferenceError):
    """Raised when a preprocessed image is missing or invalid."""


class DummyInferenceOutputCollisionError(DummyInferenceError):
    """Raised when dummy inference would overwrite existing outputs."""


class DummyInferenceFailureError(DummyInferenceError):
    """Raised when dummy inference cannot complete all cases."""


@dataclass(frozen=True, slots=True)
class DummyInferenceConfig:
    """Validated immutable configuration for deterministic dummy inference."""

    seed: int
    threshold: float
    prediction_dtype: str
    method: str


@dataclass(frozen=True, slots=True)
class _InputCasePath:
    case_id: str
    relative_image_path: PurePosixPath
    image_path: Path


@dataclass(frozen=True, slots=True)
class _OutputRootPlan:
    final_root: Path
    parent: Path
    existed_empty: bool


def load_dummy_inference_config(config_path: Path) -> DummyInferenceConfig:
    """Load and validate dummy-inference settings from an OmegaConf YAML file."""
    try:
        raw_config = OmegaConf.load(config_path)
    except OSError as exc:
        msg = f"failed to read dummy-inference config: {exc}"
        raise DummyInferenceConfigError(msg) from exc
    except Exception as exc:
        msg = f"failed to parse dummy-inference config: {exc}"
        raise DummyInferenceConfigError(msg) from exc

    if not isinstance(raw_config, DictConfig):
        msg = "dummy-inference config must contain a mapping"
        raise DummyInferenceConfigError(msg)

    container = OmegaConf.to_container(raw_config, resolve=True)
    if not isinstance(container, dict):
        msg = "dummy-inference config must resolve to a mapping"
        raise DummyInferenceConfigError(msg)

    resolved = cast(dict[str, object], container)
    config_mapping = _extract_dummy_inference_mapping(resolved)
    unknown_keys = set(config_mapping) - _EXPECTED_CONFIG_KEYS
    missing_keys = _EXPECTED_CONFIG_KEYS - set(config_mapping)
    if unknown_keys:
        msg = f"unknown dummy-inference config keys: {sorted(unknown_keys)}"
        raise DummyInferenceConfigError(msg)
    if missing_keys:
        msg = f"missing dummy-inference config keys: {sorted(missing_keys)}"
        raise DummyInferenceConfigError(msg)

    return DummyInferenceConfig(
        seed=_validated_seed(config_mapping["seed"]),
        threshold=_validated_threshold(config_mapping["threshold"]),
        prediction_dtype=_validated_prediction_dtype(config_mapping["prediction_dtype"]),
        method=_validated_method(config_mapping["method"]),
    )


def run_dummy_inference(
    preprocess_artifact_path: Path,
    *,
    preprocessed_root: Path,
    output_root: Path,
    artifact_output_path: Path,
    config_path: Path,
    git_commit: str,
    created_at_utc: str,
) -> InferenceArtifact:
    """Run deterministic dummy inference from persisted preprocessed images."""
    _require_nonempty_text(git_commit, "git_commit")
    _require_nonempty_text(created_at_utc, "created_at_utc")

    config = load_dummy_inference_config(config_path)
    preprocess_artifact = _load_preprocess_artifact(preprocess_artifact_path)
    _verify_preprocess_artifact(preprocess_artifact)
    preprocessed_root_resolved = _resolve_existing_directory(
        preprocessed_root,
        "preprocessed_root",
    )
    output_plan = _prepare_output_root(
        output_root,
        preprocessed_root=preprocessed_root_resolved,
    )
    artifact_output = _prepare_artifact_output_path(artifact_output_path)
    input_cases = _resolve_input_cases(
        preprocess_artifact,
        preprocessed_root=preprocessed_root_resolved,
    )
    prediction_paths = _relative_prediction_paths(preprocess_artifact.output_case_ids)

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
        prediction_hashes = _write_predictions(
            input_cases,
            prediction_paths=prediction_paths,
            output_root=staged_output_root,
            config=config,
        )
        artifact = InferenceArtifact(
            schema_version=ARTIFACT_SCHEMA_VERSION,
            stage=DUMMY_INFERENCE_STAGE,
            created_at_utc=created_at_utc,
            git_commit=git_commit,
            config_hash=preprocess_artifact.config_hash,
            manifest_hash=preprocess_artifact.manifest_hash,
            prediction_case_ids=preprocess_artifact.output_case_ids,
            prediction_paths=tuple(path.as_posix() for path in prediction_paths),
            prediction_hashes=prediction_hashes,
            method=config.method,
            deterministic_seed=config.seed,
            threshold=config.threshold,
            prediction_dtype=config.prediction_dtype,
        )
        artifact_temp_path = _write_artifact_to_temp(artifact, artifact_output)
        _publish_output_root(staged_output_root, output_plan)
        published_output_root = True
        _publish_artifact(artifact_temp_path, artifact_output)
        published_artifact = True
        return artifact
    except DummyInferenceError:
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
        msg = f"dummy inference failed: {exc}"
        raise DummyInferenceFailureError(msg) from exc
    finally:
        if staging_root is not None and staging_root.exists():
            shutil.rmtree(staging_root, ignore_errors=True)


def _extract_dummy_inference_mapping(resolved: dict[str, object]) -> dict[str, object]:
    if _CONFIG_SECTION in resolved:
        section = resolved[_CONFIG_SECTION]
        if not isinstance(section, dict):
            msg = "dummy_inference config section must contain a mapping"
            raise DummyInferenceConfigError(msg)
        return cast(dict[str, object], section)
    return resolved


def _validated_seed(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        msg = "seed must be a nonnegative integer"
        raise DummyInferenceConfigError(msg)
    return value


def _validated_threshold(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = "threshold must be a finite number in [0, 1]"
        raise DummyInferenceConfigError(msg)
    threshold = float(value)
    if not math.isfinite(threshold):
        msg = "threshold must be a finite number in [0, 1]"
        raise DummyInferenceConfigError(msg)
    if threshold < 0.0 or threshold > 1.0:
        msg = "threshold must be in [0, 1]"
        raise DummyInferenceConfigError(msg)
    return threshold


def _validated_prediction_dtype(value: object) -> str:
    if value != PREDICTION_DTYPE:
        msg = f"prediction_dtype must be fixed to {PREDICTION_DTYPE!r}"
        raise DummyInferenceConfigError(msg)
    return PREDICTION_DTYPE


def _validated_method(value: object) -> str:
    if value != DUMMY_INFERENCE_METHOD:
        msg = f"method must be fixed to {DUMMY_INFERENCE_METHOD!r}"
        raise DummyInferenceConfigError(msg)
    return DUMMY_INFERENCE_METHOD


def _load_preprocess_artifact(preprocess_artifact_path: Path) -> PreprocessArtifact:
    try:
        text = preprocess_artifact_path.read_text(encoding="utf-8")
    except OSError as exc:
        msg = "failed to read preprocessing artifact"
        raise DummyInferenceInputArtifactError(msg) from exc

    try:
        return artifact_from_json(text, PreprocessArtifact)
    except ArtifactValidationError as exc:
        if "artifact path" in str(exc):
            msg = f"unsafe preprocessing artifact path: {exc}"
            raise DummyInferencePathError(msg) from exc
        msg = f"invalid preprocessing artifact: {exc}"
        raise DummyInferenceInputArtifactError(msg) from exc
    except ArtifactError as exc:
        msg = f"invalid preprocessing artifact: {exc}"
        raise DummyInferenceInputArtifactError(msg) from exc


def _verify_preprocess_artifact(artifact: PreprocessArtifact) -> None:
    if len(set(artifact.output_case_ids)) != len(artifact.output_case_ids):
        msg = "preprocessing artifact output_case_ids must be unique"
        raise DummyInferenceInputArtifactError(msg)
    if len(set(artifact.output_label_paths)) != len(artifact.output_label_paths):
        msg = "preprocessing artifact output_label_paths must be unique"
        raise DummyInferenceInputArtifactError(msg)


def _resolve_existing_directory(path: Path, role: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        msg = f"{role} does not exist or cannot be resolved"
        raise DummyInferencePathError(msg) from exc
    if not resolved.is_dir():
        msg = f"{role} is not a directory"
        raise DummyInferencePathError(msg)
    return resolved


def _resolve_input_cases(
    artifact: PreprocessArtifact,
    *,
    preprocessed_root: Path,
) -> tuple[_InputCasePath, ...]:
    seen_image_paths: set[Path] = set()
    input_cases: list[_InputCasePath] = []
    for case_id, relative_path in zip(
        artifact.output_case_ids,
        artifact.output_image_paths,
        strict=True,
    ):
        relative_image_path = _relative_posix_path(relative_path, "preprocessed image")
        image_path = _resolve_preprocessed_image_path(
            relative_image_path,
            preprocessed_root=preprocessed_root,
        )
        if image_path in seen_image_paths:
            msg = f"duplicate resolved preprocessed image path: {relative_path}"
            raise DummyInferencePathError(msg)
        seen_image_paths.add(image_path)
        input_cases.append(
            _InputCasePath(
                case_id=case_id,
                relative_image_path=relative_image_path,
                image_path=image_path,
            )
        )
    return tuple(input_cases)


def _resolve_preprocessed_image_path(
    relative_path: PurePosixPath,
    *,
    preprocessed_root: Path,
) -> Path:
    candidate_path = preprocessed_root.joinpath(*relative_path.parts)
    try:
        resolved_path = candidate_path.resolve(strict=True)
    except OSError as exc:
        msg = f"preprocessed image is missing or cannot be resolved: {relative_path.as_posix()}"
        raise DummyInferenceImageError(msg) from exc

    try:
        resolved_path.relative_to(preprocessed_root)
    except ValueError as exc:
        msg = f"preprocessed image path escapes preprocessed_root: {relative_path.as_posix()}"
        raise DummyInferencePathError(msg) from exc

    if not resolved_path.is_file():
        msg = f"preprocessed image path is not a regular file: {relative_path.as_posix()}"
        raise DummyInferenceImageError(msg)
    return resolved_path


def _relative_posix_path(value: str, role: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        msg = f"{role} path must be a nonempty relative path"
        raise DummyInferencePathError(msg)
    if value.startswith("~"):
        msg = f"{role} path must not use a home-relative path: {value}"
        raise DummyInferencePathError(msg)
    posix_path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    if posix_path.is_absolute() or windows_path.is_absolute():
        msg = f"{role} path must be relative: {value}"
        raise DummyInferencePathError(msg)
    if ".." in posix_path.parts or ".." in windows_path.parts:
        msg = f"{role} path must not contain parent traversal: {value}"
        raise DummyInferencePathError(msg)
    return posix_path


def _prepare_output_root(
    output_root: Path,
    *,
    preprocessed_root: Path,
) -> _OutputRootPlan:
    if _contains_parent_traversal(output_root):
        msg = "output_root must not contain parent traversal"
        raise DummyInferencePathError(msg)
    if output_root.is_symlink():
        msg = "output root must not be a symlink"
        raise DummyInferenceOutputCollisionError(msg)
    if output_root.exists() and not output_root.is_dir():
        msg = "output root exists and is not a directory"
        raise DummyInferenceOutputCollisionError(msg)
    if output_root.exists() and any(output_root.iterdir()):
        msg = "output root must be empty or nonexistent"
        raise DummyInferenceOutputCollisionError(msg)

    if output_root.exists():
        final_root = output_root.resolve(strict=True)
        parent = final_root.parent
        existed_empty = True
    else:
        try:
            parent = output_root.parent.resolve(strict=True)
        except OSError as exc:
            msg = "output_root parent does not exist or cannot be resolved"
            raise DummyInferencePathError(msg) from exc
        if not parent.is_dir():
            msg = "output_root parent is not a directory"
            raise DummyInferencePathError(msg)
        final_root = parent / output_root.name
        existed_empty = False

    try:
        final_root.relative_to(preprocessed_root)
    except ValueError:
        pass
    else:
        msg = "output root must not resolve inside preprocessed_root"
        raise DummyInferencePathError(msg)

    return _OutputRootPlan(final_root=final_root, parent=parent, existed_empty=existed_empty)


def _prepare_artifact_output_path(artifact_output_path: Path) -> Path:
    if not artifact_output_path.name:
        msg = "artifact_output_path must name a JSON output file"
        raise DummyInferencePathError(msg)
    if _contains_parent_traversal(artifact_output_path):
        msg = "artifact_output_path must not contain parent traversal"
        raise DummyInferencePathError(msg)
    if artifact_output_path.exists() or artifact_output_path.is_symlink():
        msg = "refusing to overwrite existing inference artifact"
        raise DummyInferenceOutputCollisionError(msg)
    artifact_output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        parent = artifact_output_path.parent.resolve(strict=True)
    except OSError as exc:
        msg = "failed to resolve inference artifact parent"
        raise DummyInferencePathError(msg) from exc
    if not parent.is_dir():
        msg = "inference artifact parent is not a directory"
        raise DummyInferencePathError(msg)
    return parent / artifact_output_path.name


def _relative_prediction_paths(case_ids: tuple[str, ...]) -> tuple[PurePosixPath, ...]:
    return tuple(_relative_case_prediction_path(case_id) for case_id in case_ids)


def _relative_case_prediction_path(case_id: str) -> PurePosixPath:
    filename = f"{case_id}.nii"
    posix_filename = PurePosixPath(filename)
    windows_filename = PureWindowsPath(filename)
    if len(posix_filename.parts) != 1 or len(windows_filename.parts) != 1:
        msg = f"case_id cannot be used as a deterministic prediction file name: {case_id}"
        raise DummyInferencePathError(msg)
    return PurePosixPath("predictions") / filename


def _write_predictions(
    input_cases: tuple[_InputCasePath, ...],
    *,
    prediction_paths: tuple[PurePosixPath, ...],
    output_root: Path,
    config: DummyInferenceConfig,
) -> tuple[str, ...]:
    prediction_hashes: list[str] = []
    for input_case, prediction_path in zip(input_cases, prediction_paths, strict=True):
        output_path = _safe_join_under_root(output_root, prediction_path)
        _write_prediction(
            input_case,
            output_path=output_path,
            config=config,
        )
        prediction_hashes.append(sha256_file(output_path))
    return tuple(prediction_hashes)


def _write_prediction(
    input_case: _InputCasePath,
    *,
    output_path: Path,
    config: DummyInferenceConfig,
) -> None:
    image = _load_nifti(input_case)
    image_array = _as_array(image, input_case)
    _require_3d_image(image_array, input_case)
    _require_real_finite_values(image_array, input_case)
    affine = _as_affine(image, input_case)
    prediction = _make_prediction(
        image_array,
        case_id=input_case.case_id,
        config=config,
    )
    _write_nifti_file(output_path, prediction, affine)


def _load_nifti(input_case: _InputCasePath) -> Any:
    try:
        return cast(Any, nib.load(str(input_case.image_path)))
    except ImageFileError as exc:
        msg = (
            f"case {input_case.case_id} preprocessed image is not a readable NIfTI file: "
            f"{input_case.relative_image_path.as_posix()}"
        )
        raise DummyInferenceImageError(msg) from exc
    except OSError as exc:
        msg = (
            f"case {input_case.case_id} failed to read preprocessed image: "
            f"{input_case.relative_image_path.as_posix()}"
        )
        raise DummyInferenceImageError(msg) from exc
    except ValueError as exc:
        msg = (
            f"case {input_case.case_id} has invalid preprocessed image data: "
            f"{input_case.relative_image_path.as_posix()}"
        )
        raise DummyInferenceImageError(msg) from exc


def _as_array(image: Any, input_case: _InputCasePath) -> Array:
    try:
        return cast(Array, np.asanyarray(image.dataobj))
    except ValueError as exc:
        msg = f"case {input_case.case_id} has invalid preprocessed image array data"
        raise DummyInferenceImageError(msg) from exc


def _require_3d_image(array: Array, input_case: _InputCasePath) -> None:
    if array.ndim != 3:
        msg = f"case {input_case.case_id} preprocessed image must be 3D, got {array.shape}"
        raise DummyInferenceImageError(msg)


def _require_real_finite_values(array: Array, input_case: _InputCasePath) -> None:
    if np.iscomplexobj(array):
        msg = f"case {input_case.case_id} preprocessed image must contain only real values"
        raise DummyInferenceImageError(msg)
    try:
        has_only_finite_values = bool(np.isfinite(array).all())
    except TypeError as exc:
        msg = f"case {input_case.case_id} preprocessed image values cannot be checked"
        raise DummyInferenceImageError(msg) from exc
    if not has_only_finite_values:
        msg = f"case {input_case.case_id} preprocessed image contains NaN or infinite values"
        raise DummyInferenceImageError(msg)


def _as_affine(image: Any, input_case: _InputCasePath) -> AffineArray:
    affine = cast(AffineArray, np.asarray(image.affine, dtype=np.float64))
    if affine.shape != (4, 4):
        msg = f"case {input_case.case_id} preprocessed image affine must be 4x4"
        raise DummyInferenceImageError(msg)
    if not np.isfinite(affine).all():
        msg = f"case {input_case.case_id} preprocessed image affine contains NaN or Infinity"
        raise DummyInferenceImageError(msg)
    return affine


def _make_prediction(
    image_array: Array,
    *,
    case_id: str,
    config: DummyInferenceConfig,
) -> UInt8Array:
    image_float = np.asarray(image_array, dtype=np.float64)
    image_min = float(image_float.min())
    image_max = float(image_float.max())
    if image_max > image_min:
        normalized = (image_float - image_min) / (image_max - image_min)
    else:
        normalized = np.zeros(image_float.shape, dtype=np.float64)

    rng = np.random.default_rng(_case_seed(config.seed, case_id))
    perturbation = rng.uniform(
        low=-_SPATIAL_PERTURBATION_SCALE,
        high=_SPATIAL_PERTURBATION_SCALE,
        size=image_float.shape,
    )
    score = np.clip(normalized + perturbation, 0.0, 1.0)
    return cast(UInt8Array, (score >= config.threshold).astype(np.uint8, copy=False))


def _case_seed(seed: int, case_id: str) -> int:
    payload = f"{seed}:{case_id}".encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _write_nifti_file(path: Path, data: UInt8Array, affine: AffineArray) -> None:
    if path.exists() or path.is_symlink():
        msg = "refusing to overwrite existing prediction output file"
        raise DummyInferenceOutputCollisionError(msg)
    path.parent.mkdir(parents=True, exist_ok=True)
    dtype = np.dtype(np.uint8)
    image = nib.Nifti1Image(data.astype(dtype, copy=False), affine)  # type: ignore[no-untyped-call]
    image.set_qform(affine, code=1)  # type: ignore[no-untyped-call]
    image.set_sform(affine, code=1)  # type: ignore[no-untyped-call]
    header = image.header
    header.set_data_dtype(dtype)  # type: ignore[no-untyped-call]
    header.set_xyzt_units("mm")  # type: ignore[no-untyped-call]
    header["descrip"] = b"protoem-ct phase1 infer-dummy"
    nib.save(image, str(path))


def _safe_join_under_root(output_root: Path, relative_path: PurePosixPath) -> Path:
    if relative_path.is_absolute() or ".." in relative_path.parts:
        msg = f"output path must be relative and stay under output root: {relative_path}"
        raise DummyInferencePathError(msg)
    path = output_root.joinpath(*relative_path.parts)
    resolved_path = path.resolve(strict=False)
    root_resolved = output_root.resolve(strict=False)
    try:
        resolved_path.relative_to(root_resolved)
    except ValueError as exc:
        msg = f"output path escapes output root: {relative_path}"
        raise DummyInferencePathError(msg) from exc
    return resolved_path


def _write_artifact_to_temp(
    artifact: InferenceArtifact,
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
        msg = f"failed to write staged inference artifact: {exc}"
        raise DummyInferenceFailureError(msg) from exc
    return temp_path


def _publish_output_root(staged_output_root: Path, output_plan: _OutputRootPlan) -> None:
    if output_plan.final_root.exists():
        if not output_plan.final_root.is_dir() or any(output_plan.final_root.iterdir()):
            msg = "output root must be empty before publish"
            raise DummyInferenceOutputCollisionError(msg)
        output_plan.final_root.rmdir()
    elif output_plan.existed_empty:
        msg = "previously empty output root disappeared before publish"
        raise DummyInferenceOutputCollisionError(msg)
    staged_output_root.replace(output_plan.final_root)


def _publish_artifact(artifact_temp_path: Path, artifact_output_path: Path) -> None:
    if artifact_output_path.exists() or artifact_output_path.is_symlink():
        msg = "refusing to overwrite existing inference artifact"
        raise DummyInferenceOutputCollisionError(msg)
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
        raise DummyInferenceInputArtifactError(msg)
