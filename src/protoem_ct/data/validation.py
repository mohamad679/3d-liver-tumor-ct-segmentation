"""NIfTI image and label pair validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import nibabel as nib
import numpy as np
import numpy.typing as npt
from nibabel.filebasedimages import ImageFileError

Array = npt.NDArray[np.generic]
AffineArray = npt.NDArray[np.floating[Any]]


class _LoadedNifti(Protocol):
    dataobj: object
    affine: object


@dataclass(frozen=True, slots=True)
class PairValidationResult:
    """Result returned after a NIfTI image and label pair passes validation."""

    image_path: Path
    label_path: Path
    shape: tuple[int, ...]
    affine_tolerance: float
    label_values: tuple[int, ...]


class NiftiValidationError(ValueError):
    """Base error for project-specific NIfTI validation failures."""


class ShapeMismatchError(NiftiValidationError):
    """Raised when image and label arrays have different shapes."""


class AffineMismatchError(NiftiValidationError):
    """Raised when image and label affine matrices differ beyond tolerance."""


class InvalidLabelError(NiftiValidationError):
    """Raised when label values are not exactly binary."""


class NonFiniteValueError(NiftiValidationError):
    """Raised when image or label data contains NaN or infinite values."""


def validate_nifti_pair(
    image_path: Path,
    label_path: Path,
    *,
    affine_tolerance: float = 1e-5,
) -> PairValidationResult:
    """Validate that a NIfTI image and binary label mask are aligned."""
    if affine_tolerance < 0:
        msg = f"affine_tolerance must be non-negative, got {affine_tolerance}"
        raise NiftiValidationError(msg)

    _require_regular_file(image_path, "image")
    _require_regular_file(label_path, "label")

    image_array, image_affine = _load_nifti_array(image_path, "image")
    label_array, label_affine = _load_nifti_array(label_path, "label")

    if image_array.shape != label_array.shape:
        msg = (
            "image and label shapes differ: "
            f"image shape={image_array.shape}, label shape={label_array.shape}"
        )
        raise ShapeMismatchError(msg)

    if not np.allclose(
        image_affine,
        label_affine,
        atol=affine_tolerance,
        rtol=0.0,
        equal_nan=False,
    ):
        msg = f"image and label affines differ beyond absolute tolerance {affine_tolerance}"
        raise AffineMismatchError(msg)

    _require_finite_values(image_array, "image")
    _require_finite_values(label_array, "label")
    label_values = _require_binary_label_values(label_array)

    return PairValidationResult(
        image_path=image_path,
        label_path=label_path,
        shape=tuple(int(dimension) for dimension in image_array.shape),
        affine_tolerance=affine_tolerance,
        label_values=label_values,
    )


def _require_regular_file(path: Path, role: str) -> None:
    if not path.exists():
        msg = f"{role} path does not exist: {path}"
        raise NiftiValidationError(msg)
    if not path.is_file():
        msg = f"{role} path is not a regular file: {path}"
        raise NiftiValidationError(msg)


def _load_nifti_array(path: Path, role: str) -> tuple[Array, AffineArray]:
    try:
        image = cast(_LoadedNifti, nib.load(str(path)))
        array = cast(Array, np.asanyarray(image.dataobj))
        affine = cast(AffineArray, np.asarray(image.affine, dtype=float))
    except ImageFileError as exc:
        msg = f"{role} path is not a readable NIfTI file: {path}"
        raise NiftiValidationError(msg) from exc
    except OSError as exc:
        msg = f"failed to read {role} NIfTI file at {path}: {exc}"
        raise NiftiValidationError(msg) from exc
    except ValueError as exc:
        msg = f"invalid {role} NIfTI data at {path}: {exc}"
        raise NiftiValidationError(msg) from exc

    return array, affine


def _require_finite_values(array: Array, role: str) -> None:
    try:
        has_only_finite_values = bool(np.isfinite(array).all())
    except TypeError as exc:
        msg = f"{role} data contains values that cannot be checked for finiteness"
        raise NonFiniteValueError(msg) from exc

    if not has_only_finite_values:
        msg = f"{role} data contains NaN or infinite values"
        raise NonFiniteValueError(msg)


def _require_binary_label_values(label_array: Array) -> tuple[int, ...]:
    if np.iscomplexobj(label_array):
        msg = "label data must contain only real binary values drawn from {0, 1}"
        raise InvalidLabelError(msg)

    unique_values = cast(Array, np.unique(label_array))
    invalid_values = unique_values[~np.isin(unique_values, [0, 1])]
    if invalid_values.size:
        msg = f"label data must be binary with values from {{0, 1}}, got {invalid_values}"
        raise InvalidLabelError(msg)

    return tuple(sorted(int(value) for value in unique_values.tolist()))
