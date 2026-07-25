"""Validation stage for Phase 1 synthetic manifests."""

from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, cast

import nibabel as nib
import numpy as np

from protoem_ct.artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    ArtifactError,
    SyntheticManifest,
    ValidationArtifact,
    artifact_from_json,
    artifact_to_json,
    hash_manifest,
)
from protoem_ct.artifacts.hashing import HashingError
from protoem_ct.data.synthetic import synthetic_manifest_hash_payload
from protoem_ct.data.validation import NiftiValidationError, validate_nifti_pair


class ManifestValidationError(ValueError):
    """Base error for Phase 1 synthetic manifest validation failures."""


class ManifestValidationManifestError(ManifestValidationError):
    """Raised when the input manifest is missing, altered, or incompatible."""


class ManifestValidationPathError(ManifestValidationError):
    """Raised when a manifest path is unsafe, duplicated, or missing."""


class ManifestValidationNiftiError(ManifestValidationError):
    """Raised when a manifest case fails NIfTI validation."""


class ManifestValidationOutputError(ManifestValidationError):
    """Raised when the validation artifact output path is unsafe or occupied."""


def validate_synthetic_manifest(
    manifest_path: Path,
    *,
    data_root: Path,
    output_path: Path,
    git_commit: str,
    created_at_utc: str,
    affine_tolerance: float = 1e-5,
) -> ValidationArtifact:
    """Validate a synthetic manifest and write a deterministic validation artifact."""
    _require_nonempty_text(git_commit, "git_commit")
    _require_nonempty_text(created_at_utc, "created_at_utc")
    _require_nonnegative_tolerance(affine_tolerance)
    _require_output_available(output_path)

    manifest = _load_synthetic_manifest(manifest_path)
    _verify_manifest_hash(manifest)
    data_root_resolved = _resolve_data_root(data_root)
    image_paths = _resolve_manifest_paths(
        manifest.image_paths,
        data_root=data_root_resolved,
        role="image",
    )
    label_paths = _resolve_manifest_paths(
        manifest.label_paths,
        data_root=data_root_resolved,
        role="label",
    )

    validated_case_ids: list[str] = []
    for case_id, image_path, label_path in zip(
        manifest.case_ids,
        image_paths,
        label_paths,
        strict=True,
    ):
        try:
            result = validate_nifti_pair(
                image_path,
                label_path,
                affine_tolerance=affine_tolerance,
            )
        except NiftiValidationError as exc:
            msg = f"case {case_id} failed NIfTI validation: {exc}"
            raise ManifestValidationNiftiError(msg) from exc

        if result.shape != manifest.shape:
            msg = (
                f"case {case_id} shape {result.shape} does not match manifest shape "
                f"{manifest.shape}"
            )
            raise ManifestValidationNiftiError(msg)

        spacing = _voxel_spacing_from_affine(image_path)
        if not np.allclose(
            np.asarray(spacing, dtype=float),
            np.asarray(manifest.spacing, dtype=float),
            atol=affine_tolerance,
            rtol=0.0,
            equal_nan=False,
        ):
            msg = (
                f"case {case_id} voxel spacing {spacing} does not match manifest spacing "
                f"{manifest.spacing} with absolute tolerance {affine_tolerance}"
            )
            raise ManifestValidationNiftiError(msg)

        validated_case_ids.append(case_id)

    artifact = ValidationArtifact(
        schema_version=ARTIFACT_SCHEMA_VERSION,
        stage="validate",
        created_at_utc=created_at_utc,
        git_commit=git_commit,
        config_hash=manifest.config_hash,
        manifest_hash=manifest.manifest_hash,
        valid_case_count=len(validated_case_ids),
        invalid_case_count=0,
        validated_case_ids=tuple(validated_case_ids),
    )
    _write_validation_artifact(artifact, output_path)
    return artifact


def _load_synthetic_manifest(manifest_path: Path) -> SyntheticManifest:
    try:
        text = manifest_path.read_text(encoding="utf-8")
    except OSError as exc:
        msg = f"failed to read synthetic manifest at {manifest_path}: {exc}"
        raise ManifestValidationManifestError(msg) from exc

    try:
        return artifact_from_json(text, SyntheticManifest)
    except ArtifactError as exc:
        msg = f"invalid synthetic manifest at {manifest_path}: {exc}"
        raise ManifestValidationManifestError(msg) from exc


def _verify_manifest_hash(manifest: SyntheticManifest) -> None:
    try:
        recomputed_hash = hash_manifest(synthetic_manifest_hash_payload(manifest))
    except (ArtifactError, HashingError) as exc:
        msg = f"failed to recompute synthetic manifest hash: {exc}"
        raise ManifestValidationManifestError(msg) from exc

    if recomputed_hash != manifest.manifest_hash:
        msg = (
            "synthetic manifest hash mismatch: "
            f"stored={manifest.manifest_hash}, recomputed={recomputed_hash}"
        )
        raise ManifestValidationManifestError(msg)


def _resolve_data_root(data_root: Path) -> Path:
    try:
        resolved = data_root.resolve(strict=True)
    except OSError as exc:
        msg = f"data_root does not exist or cannot be resolved: {data_root}"
        raise ManifestValidationPathError(msg) from exc
    if not resolved.is_dir():
        msg = f"data_root is not a directory: {data_root}"
        raise ManifestValidationPathError(msg)
    return resolved


def _resolve_manifest_paths(
    paths: tuple[str, ...],
    *,
    data_root: Path,
    role: str,
) -> tuple[Path, ...]:
    resolved_paths: list[Path] = []
    seen_paths: set[Path] = set()
    for relative_path in paths:
        resolved_path = _resolve_manifest_path(
            relative_path,
            data_root=data_root,
            role=role,
        )
        if resolved_path in seen_paths:
            msg = f"duplicate resolved {role} path in manifest: {relative_path}"
            raise ManifestValidationPathError(msg)
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
        raise ManifestValidationPathError(msg) from exc

    try:
        resolved_path.relative_to(data_root)
    except ValueError as exc:
        msg = f"{role} path escapes data_root: {relative_path}"
        raise ManifestValidationPathError(msg) from exc

    if not resolved_path.is_file():
        msg = f"{role} path is not a regular file: {relative_path}"
        raise ManifestValidationPathError(msg)
    return resolved_path


def _require_relative_manifest_path(relative_path: str, role: str) -> None:
    if not isinstance(relative_path, str) or not relative_path:
        msg = f"{role} path must be a nonempty relative path"
        raise ManifestValidationPathError(msg)
    if relative_path.startswith("~"):
        msg = f"{role} path must not use a home-relative path: {relative_path}"
        raise ManifestValidationPathError(msg)
    posix_path = PurePosixPath(relative_path)
    windows_path = PureWindowsPath(relative_path)
    if posix_path.is_absolute() or windows_path.is_absolute():
        msg = f"{role} path must be relative: {relative_path}"
        raise ManifestValidationPathError(msg)
    if ".." in posix_path.parts or ".." in windows_path.parts:
        msg = f"{role} path must not contain parent traversal: {relative_path}"
        raise ManifestValidationPathError(msg)


def _voxel_spacing_from_affine(image_path: Path) -> tuple[float, float, float]:
    try:
        image = cast(Any, nib.load(str(image_path)))
        affine = np.asarray(image.affine, dtype=float)
    except (OSError, ValueError) as exc:
        msg = f"failed to load validated image affine at {image_path}: {exc}"
        raise ManifestValidationNiftiError(msg) from exc

    if affine.shape != (4, 4):
        msg = f"image affine must be 4x4, got {affine.shape}"
        raise ManifestValidationNiftiError(msg)
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    if not np.isfinite(spacing).all():
        msg = "image affine-derived voxel spacing contains NaN or Infinity"
        raise ManifestValidationNiftiError(msg)
    return (float(spacing[0]), float(spacing[1]), float(spacing[2]))


def _require_output_available(output_path: Path) -> None:
    if not output_path.name:
        msg = "output_path must name a JSON output file"
        raise ManifestValidationOutputError(msg)
    if ".." in output_path.parts:
        msg = f"output_path must not contain parent traversal: {output_path}"
        raise ManifestValidationOutputError(msg)
    if output_path.exists():
        msg = f"refusing to overwrite existing validation artifact: {output_path}"
        raise ManifestValidationOutputError(msg)


def _write_validation_artifact(artifact: ValidationArtifact, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        parent = output_path.parent.resolve(strict=True)
    except OSError as exc:
        msg = f"failed to resolve validation artifact parent: {output_path.parent}"
        raise ManifestValidationOutputError(msg) from exc

    resolved_output = parent / output_path.name
    try:
        resolved_output.relative_to(parent)
    except ValueError as exc:
        msg = f"validation artifact output escapes its parent: {output_path}"
        raise ManifestValidationOutputError(msg) from exc
    if resolved_output.exists():
        msg = f"refusing to overwrite existing validation artifact: {output_path}"
        raise ManifestValidationOutputError(msg)

    try:
        artifact_to_json(artifact, resolved_output)
    except ArtifactError as exc:
        msg = f"failed to write validation artifact: {exc}"
        raise ManifestValidationOutputError(msg) from exc


def _require_nonempty_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        msg = f"{field_name} must be a nonempty string"
        raise ManifestValidationError(msg)


def _require_nonnegative_tolerance(affine_tolerance: float) -> None:
    if affine_tolerance < 0:
        msg = f"affine_tolerance must be non-negative, got {affine_tolerance}"
        raise ManifestValidationError(msg)
