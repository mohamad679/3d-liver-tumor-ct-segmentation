"""Deterministic Phase 1 synthetic CT dataset generation.

Image arrays are stored as float32 NIfTI volumes and binary tumor labels are
stored as uint8 NIfTI volumes. Manifest hashes are computed from the serialized
SyntheticManifest payload with the ``manifest_hash`` field omitted, so the hash
never recursively includes itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, cast

import nibabel as nib
import numpy as np
import numpy.typing as npt
from omegaconf import DictConfig, ListConfig, OmegaConf

from protoem_ct.artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    SyntheticManifest,
    artifact_to_dict,
    artifact_to_json,
    hash_config,
    hash_manifest,
)
from protoem_ct.artifacts.hashing import JsonValue

IMAGE_DTYPE = np.float32
LABEL_DTYPE = np.uint8
MANIFEST_FILENAME = "synthetic_manifest.json"

_EXPECTED_CONFIG_KEYS = {
    "dataset_id",
    "seed",
    "case_count",
    "shape",
    "spacing",
    "generated_data_root",
}
_MANIFEST_HASH_PLACEHOLDER = "0" * 64

ImageArray = npt.NDArray[np.float32]
LabelArray = npt.NDArray[np.uint8]
AffineArray = npt.NDArray[np.float64]


class SyntheticDataError(ValueError):
    """Base error for Phase 1 synthetic data generation failures."""


class SyntheticConfigError(SyntheticDataError):
    """Raised when the synthetic data configuration is invalid."""


class SyntheticOutputError(SyntheticDataError):
    """Raised when synthetic data output paths are unsafe or already populated."""


@dataclass(frozen=True, slots=True)
class SyntheticDataConfig:
    """Validated immutable configuration for the deterministic synthetic dataset."""

    dataset_id: str
    seed: int
    case_count: int
    shape: tuple[int, int, int]
    spacing: tuple[float, float, float]
    generated_data_root: PurePosixPath


@dataclass(frozen=True, slots=True)
class SyntheticCase:
    """In-memory deterministic arrays for one synthetic CT case."""

    case_id: str
    image: ImageArray
    label: LabelArray
    affine: AffineArray


@dataclass(frozen=True, slots=True)
class SyntheticDatasetResult:
    """Paths and hashes produced by one synthetic dataset generation run."""

    manifest: SyntheticManifest
    manifest_path: Path
    manifest_relative_path: PurePosixPath
    config_hash: str
    manifest_hash: str


def load_synthetic_config(config_path: Path) -> SyntheticDataConfig:
    """Load and validate a synthetic data configuration from an OmegaConf YAML file."""
    try:
        raw_config = OmegaConf.load(config_path)
    except OSError as exc:
        msg = f"failed to read synthetic config at {config_path}: {exc}"
        raise SyntheticConfigError(msg) from exc
    except Exception as exc:
        msg = f"failed to parse synthetic config at {config_path}: {exc}"
        raise SyntheticConfigError(msg) from exc

    if not isinstance(raw_config, DictConfig):
        msg = "synthetic config must contain a mapping"
        raise SyntheticConfigError(msg)

    config_keys = {str(key) for key in raw_config}
    unknown_keys = config_keys - _EXPECTED_CONFIG_KEYS
    missing_keys = _EXPECTED_CONFIG_KEYS - config_keys
    if unknown_keys:
        msg = f"unknown synthetic config keys: {sorted(unknown_keys)}"
        raise SyntheticConfigError(msg)
    if missing_keys:
        msg = f"missing synthetic config keys: {sorted(missing_keys)}"
        raise SyntheticConfigError(msg)

    container = OmegaConf.to_container(raw_config, resolve=True)
    if not isinstance(container, dict):
        msg = "synthetic config must resolve to a mapping"
        raise SyntheticConfigError(msg)
    resolved = cast(dict[str, object], container)

    return SyntheticDataConfig(
        dataset_id=_validated_dataset_id(resolved["dataset_id"]),
        seed=_validated_seed(resolved["seed"]),
        case_count=_validated_case_count(resolved["case_count"]),
        shape=_validated_shape(resolved["shape"]),
        spacing=_validated_spacing(resolved["spacing"]),
        generated_data_root=_validated_generated_data_root(resolved["generated_data_root"]),
    )


def synthetic_config_hash_payload(config: SyntheticDataConfig) -> dict[str, JsonValue]:
    """Return the canonical JSON payload used for the synthetic config hash."""
    return {
        "case_count": config.case_count,
        "dataset_id": config.dataset_id,
        "generated_data_root": config.generated_data_root.as_posix(),
        "seed": config.seed,
        "shape": list(config.shape),
        "spacing": list(config.spacing),
    }


def synthetic_manifest_hash_payload(manifest: SyntheticManifest) -> dict[str, JsonValue]:
    """Return the manifest hash payload, excluding only the self-referential hash field."""
    payload = artifact_to_dict(manifest)
    del payload["manifest_hash"]
    return payload


def generate_synthetic_case(config: SyntheticDataConfig, case_index: int) -> SyntheticCase:
    """Generate deterministic arrays for a single synthetic case index."""
    if case_index < 0 or case_index >= config.case_count:
        msg = f"case_index must be in [0, {config.case_count}), got {case_index}"
        raise SyntheticDataError(msg)

    rng = np.random.default_rng(config.seed)
    case: SyntheticCase | None = None
    for index in range(case_index + 1):
        case = _generate_case_from_rng(config, rng, index)
    if case is None:
        msg = "failed to generate synthetic case"
        raise SyntheticDataError(msg)
    return case


def create_synthetic_dataset(
    config: SyntheticDataConfig,
    *,
    output_root: Path,
    git_commit: str,
    created_at_utc: str,
) -> SyntheticDatasetResult:
    """Generate deterministic synthetic NIfTI files and a hashed manifest.

    The output root must be empty or nonexistent. All generated paths are placed
    beneath ``output_root / config.generated_data_root`` and all manifest paths
    are relative POSIX paths.
    """
    _require_nonempty_text(git_commit, "git_commit")
    _require_nonempty_text(created_at_utc, "created_at_utc")
    root = _prepare_output_root(output_root)
    generated_root = _safe_join_under_root(root, config.generated_data_root)
    images_dir = generated_root / "images"
    labels_dir = generated_root / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    config_hash = hash_config(synthetic_config_hash_payload(config))
    rng = np.random.default_rng(config.seed)
    case_ids: list[str] = []
    image_paths: list[str] = []
    label_paths: list[str] = []
    for index in range(config.case_count):
        case = _generate_case_from_rng(config, rng, index)
        image_relative = config.generated_data_root / "images" / f"{case.case_id}_ct.nii"
        label_relative = config.generated_data_root / "labels" / f"{case.case_id}_tumor.nii"
        image_path = _safe_join_under_root(root, image_relative)
        label_path = _safe_join_under_root(root, label_relative)
        _write_nifti_file(image_path, case.image, case.affine)
        _write_nifti_file(label_path, case.label, case.affine)
        case_ids.append(case.case_id)
        image_paths.append(image_relative.as_posix())
        label_paths.append(label_relative.as_posix())

    manifest_without_hash = SyntheticManifest(
        schema_version=ARTIFACT_SCHEMA_VERSION,
        stage="synthetic-manifest",
        created_at_utc=created_at_utc,
        git_commit=git_commit,
        config_hash=config_hash,
        manifest_hash=_MANIFEST_HASH_PLACEHOLDER,
        dataset_id=config.dataset_id,
        seed=config.seed,
        case_ids=tuple(sorted(case_ids)),
        image_paths=tuple(image_paths),
        label_paths=tuple(label_paths),
        shape=config.shape,
        spacing=config.spacing,
    )
    manifest_hash = hash_manifest(synthetic_manifest_hash_payload(manifest_without_hash))
    manifest = SyntheticManifest(
        schema_version=manifest_without_hash.schema_version,
        stage=manifest_without_hash.stage,
        created_at_utc=manifest_without_hash.created_at_utc,
        git_commit=manifest_without_hash.git_commit,
        config_hash=manifest_without_hash.config_hash,
        manifest_hash=manifest_hash,
        dataset_id=manifest_without_hash.dataset_id,
        seed=manifest_without_hash.seed,
        case_ids=manifest_without_hash.case_ids,
        image_paths=manifest_without_hash.image_paths,
        label_paths=manifest_without_hash.label_paths,
        shape=manifest_without_hash.shape,
        spacing=manifest_without_hash.spacing,
    )
    manifest_relative_path = config.generated_data_root / MANIFEST_FILENAME
    manifest_path = _safe_join_under_root(root, manifest_relative_path)
    _require_absent_target(manifest_path)
    artifact_to_json(manifest, manifest_path)

    return SyntheticDatasetResult(
        manifest=manifest,
        manifest_path=manifest_path,
        manifest_relative_path=manifest_relative_path,
        config_hash=config_hash,
        manifest_hash=manifest_hash,
    )


def _validated_dataset_id(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        msg = "dataset_id must be a nonempty string"
        raise SyntheticConfigError(msg)
    return value


def _validated_seed(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        msg = "seed must be a nonnegative integer"
        raise SyntheticConfigError(msg)
    return value


def _validated_case_count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        msg = "case_count must be at least 1"
        raise SyntheticConfigError(msg)
    return value


def _validated_shape(value: object) -> tuple[int, int, int]:
    if not isinstance(value, list | ListConfig) or len(value) != 3:
        msg = "shape must contain exactly three positive integers"
        raise SyntheticConfigError(msg)
    shape = tuple(value)
    if any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in shape):
        msg = "shape must contain exactly three positive integers"
        raise SyntheticConfigError(msg)
    if math.prod(shape) < 2:
        msg = "shape must contain at least two voxels"
        raise SyntheticConfigError(msg)
    return cast(tuple[int, int, int], shape)


def _validated_spacing(value: object) -> tuple[float, float, float]:
    if not isinstance(value, list | ListConfig) or len(value) != 3:
        msg = "spacing must contain exactly three positive finite numbers"
        raise SyntheticConfigError(msg)
    spacing_values = tuple(value)
    if any(
        isinstance(item, bool)
        or not isinstance(item, int | float)
        or not math.isfinite(float(item))
        or float(item) <= 0.0
        for item in spacing_values
    ):
        msg = "spacing must contain exactly three positive finite numbers"
        raise SyntheticConfigError(msg)
    return cast(tuple[float, float, float], tuple(float(item) for item in spacing_values))


def _validated_generated_data_root(value: object) -> PurePosixPath:
    if not isinstance(value, str) or not value.strip():
        msg = "generated_data_root must be a nonempty relative path"
        raise SyntheticConfigError(msg)
    if value.startswith("~"):
        msg = f"generated_data_root must not use a home-relative path: {value}"
        raise SyntheticConfigError(msg)
    posix_path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    if posix_path.is_absolute() or windows_path.is_absolute():
        msg = f"generated_data_root must be relative: {value}"
        raise SyntheticConfigError(msg)
    if ".." in posix_path.parts or ".." in windows_path.parts:
        msg = f"generated_data_root must not contain parent traversal: {value}"
        raise SyntheticConfigError(msg)
    return posix_path


def _require_nonempty_text(value: str, field_name: str) -> None:
    if not value.strip():
        msg = f"{field_name} must be a nonempty string"
        raise SyntheticDataError(msg)


def _prepare_output_root(output_root: Path) -> Path:
    if output_root.exists() and not output_root.is_dir():
        msg = f"output root exists and is not a directory: {output_root}"
        raise SyntheticOutputError(msg)
    if output_root.exists() and any(output_root.iterdir()):
        msg = f"output root must be empty or nonexistent: {output_root}"
        raise SyntheticOutputError(msg)
    output_root.mkdir(parents=True, exist_ok=True)
    return output_root.resolve(strict=True)


def _safe_join_under_root(output_root: Path, relative_path: PurePosixPath) -> Path:
    if relative_path.is_absolute() or ".." in relative_path.parts:
        msg = f"generated path must be relative and stay under output root: {relative_path}"
        raise SyntheticOutputError(msg)
    path = output_root.joinpath(*relative_path.parts)
    resolved_path = path.resolve(strict=False)
    try:
        resolved_path.relative_to(output_root)
    except ValueError as exc:
        msg = f"generated path escapes output root: {relative_path}"
        raise SyntheticOutputError(msg) from exc
    return resolved_path


def _generate_case_from_rng(
    config: SyntheticDataConfig,
    rng: np.random.Generator,
    case_index: int,
) -> SyntheticCase:
    label = _generate_label(config.shape, rng)
    gradient = _deterministic_gradient(config.shape)
    noise = rng.normal(loc=-120.0 + case_index, scale=25.0, size=config.shape).astype(IMAGE_DTYPE)
    image = (noise + gradient + label.astype(IMAGE_DTYPE) * 180.0).astype(IMAGE_DTYPE)
    return SyntheticCase(
        case_id=f"synthetic-{case_index:03d}",
        image=image,
        label=label,
        affine=_affine_from_spacing(config.spacing),
    )


def _generate_label(shape: tuple[int, int, int], rng: np.random.Generator) -> LabelArray:
    voxel_count = math.prod(shape)
    if voxel_count < 2:
        msg = "shape must contain at least two voxels to create binary foreground and background"
        raise SyntheticDataError(msg)
    foreground_count = min(voxel_count - 1, max(1, voxel_count // 8))
    foreground_indices = rng.choice(voxel_count, size=foreground_count, replace=False)
    label = np.zeros(voxel_count, dtype=LABEL_DTYPE)
    label[foreground_indices] = 1
    return label.reshape(shape)


def _deterministic_gradient(shape: tuple[int, int, int]) -> ImageArray:
    axes = np.indices(shape, dtype=IMAGE_DTYPE)
    gradient = axes[0] * 3.0 + axes[1] * 2.0 + axes[2] * 1.0
    return cast(ImageArray, gradient.astype(IMAGE_DTYPE))


def _affine_from_spacing(spacing: tuple[float, float, float]) -> AffineArray:
    return np.array(
        [
            [spacing[0], 0.0, 0.0, 0.0],
            [0.0, spacing[1], 0.0, 0.0],
            [0.0, 0.0, spacing[2], 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _write_nifti_file(path: Path, data: npt.NDArray[Any], affine: AffineArray) -> None:
    _require_absent_target(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(data, affine)  # type: ignore[no-untyped-call]
    image.set_qform(affine, code=1)  # type: ignore[no-untyped-call]
    image.set_sform(affine, code=1)  # type: ignore[no-untyped-call]
    header = image.header
    header.set_data_dtype(data.dtype)  # type: ignore[no-untyped-call]
    header.set_xyzt_units("mm")  # type: ignore[no-untyped-call]
    header["descrip"] = b"protoem-ct phase1 synthetic"
    nib.save(image, str(path))


def _require_absent_target(path: Path) -> None:
    if path.exists():
        msg = f"refusing to overwrite existing output file: {path}"
        raise SyntheticOutputError(msg)
