"""Unit tests for public NIfTI pair validation behavior."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from protoem_ct.data import (
    AffineMismatchError,
    InvalidLabelError,
    NiftiValidationError,
    NonFiniteValueError,
    ShapeMismatchError,
    validate_nifti_pair,
)

NiftiArray = npt.NDArray[np.generic]
AffineArray = npt.NDArray[np.float64]
WriteNiftiFile = Callable[[Path, NiftiArray, AffineArray], Path]


def _write_valid_pair(
    tmp_path: Path,
    valid_ct_image: NiftiArray,
    valid_binary_tumor_mask: NiftiArray,
    matching_affine: AffineArray,
    write_nifti_file: WriteNiftiFile,
) -> tuple[Path, Path]:
    """Write a valid synthetic image-label pair and return both paths."""
    image_path = write_nifti_file(tmp_path / "image.nii.gz", valid_ct_image, matching_affine)
    label_path = write_nifti_file(
        tmp_path / "label.nii.gz",
        valid_binary_tumor_mask,
        matching_affine,
    )
    return image_path, label_path


def test_valid_image_and_binary_label(
    tmp_path: Path,
    valid_ct_image: NiftiArray,
    valid_binary_tumor_mask: NiftiArray,
    matching_affine: AffineArray,
    write_nifti_file: WriteNiftiFile,
) -> None:
    image_path, label_path = _write_valid_pair(
        tmp_path,
        valid_ct_image,
        valid_binary_tumor_mask,
        matching_affine,
        write_nifti_file,
    )

    result = validate_nifti_pair(image_path, label_path)

    assert result.image_path == image_path
    assert result.label_path == label_path
    assert result.shape == (2, 3, 4)
    assert result.affine_tolerance == 1e-5
    assert result.label_values == (0, 1)


def test_label_values_are_deterministically_sorted(
    tmp_path: Path,
    valid_ct_image: NiftiArray,
    matching_affine: AffineArray,
    write_nifti_file: WriteNiftiFile,
) -> None:
    label = np.ones((2, 3, 4), dtype=np.uint8)
    label[0, 0, 0] = 0
    image_path = write_nifti_file(tmp_path / "image.nii.gz", valid_ct_image, matching_affine)
    label_path = write_nifti_file(tmp_path / "label.nii.gz", label, matching_affine)

    result = validate_nifti_pair(image_path, label_path)

    assert result.label_values == (0, 1)


def test_all_zero_label_is_accepted(
    tmp_path: Path,
    valid_ct_image: NiftiArray,
    matching_affine: AffineArray,
    write_nifti_file: WriteNiftiFile,
) -> None:
    label = np.zeros((2, 3, 4), dtype=np.uint8)
    image_path = write_nifti_file(tmp_path / "image.nii.gz", valid_ct_image, matching_affine)
    label_path = write_nifti_file(tmp_path / "label.nii.gz", label, matching_affine)

    result = validate_nifti_pair(image_path, label_path)

    assert result.label_values == (0,)


def test_all_one_label_is_accepted(
    tmp_path: Path,
    valid_ct_image: NiftiArray,
    matching_affine: AffineArray,
    write_nifti_file: WriteNiftiFile,
) -> None:
    label = np.ones((2, 3, 4), dtype=np.uint8)
    image_path = write_nifti_file(tmp_path / "image.nii.gz", valid_ct_image, matching_affine)
    label_path = write_nifti_file(tmp_path / "label.nii.gz", label, matching_affine)

    result = validate_nifti_pair(image_path, label_path)

    assert result.label_values == (1,)


def test_shape_mismatch_raises_shape_mismatch_error(
    tmp_path: Path,
    valid_ct_image: NiftiArray,
    matching_affine: AffineArray,
    write_nifti_file: WriteNiftiFile,
) -> None:
    label = np.zeros((2, 3, 5), dtype=np.uint8)
    image_path = write_nifti_file(tmp_path / "image.nii.gz", valid_ct_image, matching_affine)
    label_path = write_nifti_file(tmp_path / "label.nii.gz", label, matching_affine)

    with pytest.raises(ShapeMismatchError):
        validate_nifti_pair(image_path, label_path)


def test_affine_mismatch_raises_affine_mismatch_error(
    tmp_path: Path,
    valid_ct_image: NiftiArray,
    valid_binary_tumor_mask: NiftiArray,
    matching_affine: AffineArray,
    write_nifti_file: WriteNiftiFile,
) -> None:
    label_affine = matching_affine.copy()
    label_affine[0, 3] += 1e-3
    image_path = write_nifti_file(tmp_path / "image.nii.gz", valid_ct_image, matching_affine)
    label_path = write_nifti_file(tmp_path / "label.nii.gz", valid_binary_tumor_mask, label_affine)

    with pytest.raises(AffineMismatchError):
        validate_nifti_pair(image_path, label_path)


def test_affine_difference_within_tolerance_is_accepted(
    tmp_path: Path,
    valid_ct_image: NiftiArray,
    valid_binary_tumor_mask: NiftiArray,
    matching_affine: AffineArray,
    write_nifti_file: WriteNiftiFile,
) -> None:
    label_affine = matching_affine.copy()
    label_affine[0, 3] += 5e-6
    image_path = write_nifti_file(tmp_path / "image.nii.gz", valid_ct_image, matching_affine)
    label_path = write_nifti_file(tmp_path / "label.nii.gz", valid_binary_tumor_mask, label_affine)

    result = validate_nifti_pair(image_path, label_path)

    assert result.label_values == (0, 1)


def test_invalid_label_value_raises_invalid_label_error(
    tmp_path: Path,
    valid_ct_image: NiftiArray,
    valid_binary_tumor_mask: NiftiArray,
    matching_affine: AffineArray,
    write_nifti_file: WriteNiftiFile,
) -> None:
    label = valid_binary_tumor_mask.copy()
    label[0, 0, 0] = 2
    image_path = write_nifti_file(tmp_path / "image.nii.gz", valid_ct_image, matching_affine)
    label_path = write_nifti_file(tmp_path / "label.nii.gz", label, matching_affine)

    with pytest.raises(InvalidLabelError):
        validate_nifti_pair(image_path, label_path)


def test_negative_label_value_raises_invalid_label_error(
    tmp_path: Path,
    valid_ct_image: NiftiArray,
    valid_binary_tumor_mask: NiftiArray,
    matching_affine: AffineArray,
    write_nifti_file: WriteNiftiFile,
) -> None:
    label = valid_binary_tumor_mask.astype(np.int16)
    label[0, 0, 0] = -1
    image_path = write_nifti_file(tmp_path / "image.nii.gz", valid_ct_image, matching_affine)
    label_path = write_nifti_file(tmp_path / "label.nii.gz", label, matching_affine)

    with pytest.raises(InvalidLabelError):
        validate_nifti_pair(image_path, label_path)


def test_nan_in_image_raises_non_finite_value_error(
    tmp_path: Path,
    valid_ct_image: NiftiArray,
    valid_binary_tumor_mask: NiftiArray,
    matching_affine: AffineArray,
    write_nifti_file: WriteNiftiFile,
) -> None:
    image = valid_ct_image.copy()
    image[0, 0, 0] = np.nan
    image_path = write_nifti_file(tmp_path / "image.nii.gz", image, matching_affine)
    label_path = write_nifti_file(
        tmp_path / "label.nii.gz",
        valid_binary_tumor_mask,
        matching_affine,
    )

    with pytest.raises(NonFiniteValueError):
        validate_nifti_pair(image_path, label_path)


def test_nan_in_label_raises_non_finite_value_error(
    tmp_path: Path,
    valid_ct_image: NiftiArray,
    valid_binary_tumor_mask: NiftiArray,
    matching_affine: AffineArray,
    write_nifti_file: WriteNiftiFile,
) -> None:
    label = valid_binary_tumor_mask.astype(np.float32)
    label[0, 0, 0] = np.nan
    image_path = write_nifti_file(tmp_path / "image.nii.gz", valid_ct_image, matching_affine)
    label_path = write_nifti_file(tmp_path / "label.nii.gz", label, matching_affine)

    with pytest.raises(NonFiniteValueError):
        validate_nifti_pair(image_path, label_path)


def test_positive_infinity_in_image_raises_non_finite_value_error(
    tmp_path: Path,
    valid_ct_image: NiftiArray,
    valid_binary_tumor_mask: NiftiArray,
    matching_affine: AffineArray,
    write_nifti_file: WriteNiftiFile,
) -> None:
    image = valid_ct_image.copy()
    image[0, 0, 0] = np.inf
    image_path = write_nifti_file(tmp_path / "image.nii.gz", image, matching_affine)
    label_path = write_nifti_file(
        tmp_path / "label.nii.gz",
        valid_binary_tumor_mask,
        matching_affine,
    )

    with pytest.raises(NonFiniteValueError):
        validate_nifti_pair(image_path, label_path)


def test_negative_infinity_in_image_raises_non_finite_value_error(
    tmp_path: Path,
    valid_ct_image: NiftiArray,
    valid_binary_tumor_mask: NiftiArray,
    matching_affine: AffineArray,
    write_nifti_file: WriteNiftiFile,
) -> None:
    image = valid_ct_image.copy()
    image[0, 0, 0] = -np.inf
    image_path = write_nifti_file(tmp_path / "image.nii.gz", image, matching_affine)
    label_path = write_nifti_file(
        tmp_path / "label.nii.gz",
        valid_binary_tumor_mask,
        matching_affine,
    )

    with pytest.raises(NonFiniteValueError):
        validate_nifti_pair(image_path, label_path)


def test_positive_infinity_in_label_raises_non_finite_value_error(
    tmp_path: Path,
    valid_ct_image: NiftiArray,
    valid_binary_tumor_mask: NiftiArray,
    matching_affine: AffineArray,
    write_nifti_file: WriteNiftiFile,
) -> None:
    label = valid_binary_tumor_mask.astype(np.float32)
    label[0, 0, 0] = np.inf
    image_path = write_nifti_file(tmp_path / "image.nii.gz", valid_ct_image, matching_affine)
    label_path = write_nifti_file(tmp_path / "label.nii.gz", label, matching_affine)

    with pytest.raises(NonFiniteValueError):
        validate_nifti_pair(image_path, label_path)


def test_negative_infinity_in_label_raises_non_finite_value_error(
    tmp_path: Path,
    valid_ct_image: NiftiArray,
    valid_binary_tumor_mask: NiftiArray,
    matching_affine: AffineArray,
    write_nifti_file: WriteNiftiFile,
) -> None:
    label = valid_binary_tumor_mask.astype(np.float32)
    label[0, 0, 0] = -np.inf
    image_path = write_nifti_file(tmp_path / "image.nii.gz", valid_ct_image, matching_affine)
    label_path = write_nifti_file(tmp_path / "label.nii.gz", label, matching_affine)

    with pytest.raises(NonFiniteValueError):
        validate_nifti_pair(image_path, label_path)


def test_missing_image_path_raises_nifti_validation_error(
    tmp_path: Path,
    valid_binary_tumor_mask: NiftiArray,
    matching_affine: AffineArray,
    write_nifti_file: WriteNiftiFile,
) -> None:
    label_path = write_nifti_file(
        tmp_path / "label.nii.gz",
        valid_binary_tumor_mask,
        matching_affine,
    )

    with pytest.raises(NiftiValidationError):
        validate_nifti_pair(tmp_path / "missing-image.nii.gz", label_path)


def test_missing_label_path_raises_nifti_validation_error(
    tmp_path: Path,
    valid_ct_image: NiftiArray,
    matching_affine: AffineArray,
    write_nifti_file: WriteNiftiFile,
) -> None:
    image_path = write_nifti_file(tmp_path / "image.nii.gz", valid_ct_image, matching_affine)

    with pytest.raises(NiftiValidationError):
        validate_nifti_pair(image_path, tmp_path / "missing-label.nii.gz")


def test_directory_instead_of_file_raises_nifti_validation_error(
    tmp_path: Path,
    valid_binary_tumor_mask: NiftiArray,
    matching_affine: AffineArray,
    write_nifti_file: WriteNiftiFile,
) -> None:
    label_path = write_nifti_file(
        tmp_path / "label.nii.gz",
        valid_binary_tumor_mask,
        matching_affine,
    )

    with pytest.raises(NiftiValidationError):
        validate_nifti_pair(tmp_path, label_path)


def test_invalid_unreadable_nifti_raises_nifti_validation_error(
    tmp_path: Path,
    valid_binary_tumor_mask: NiftiArray,
    matching_affine: AffineArray,
    write_nifti_file: WriteNiftiFile,
) -> None:
    invalid_image_path = tmp_path / "invalid.nii.gz"
    invalid_image_path.write_text("not a nifti file", encoding="utf-8")
    label_path = write_nifti_file(
        tmp_path / "label.nii.gz",
        valid_binary_tumor_mask,
        matching_affine,
    )

    with pytest.raises(NiftiValidationError):
        validate_nifti_pair(invalid_image_path, label_path)


def test_negative_affine_tolerance_raises_nifti_validation_error(
    tmp_path: Path,
    valid_ct_image: NiftiArray,
    valid_binary_tumor_mask: NiftiArray,
    matching_affine: AffineArray,
    write_nifti_file: WriteNiftiFile,
) -> None:
    image_path, label_path = _write_valid_pair(
        tmp_path,
        valid_ct_image,
        valid_binary_tumor_mask,
        matching_affine,
        write_nifti_file,
    )

    with pytest.raises(NiftiValidationError):
        validate_nifti_pair(image_path, label_path, affine_tolerance=-1e-5)
