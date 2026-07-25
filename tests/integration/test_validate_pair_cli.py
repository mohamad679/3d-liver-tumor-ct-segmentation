"""Integration tests for the validate-pair CLI command."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from typer.testing import CliRunner

from protoem_ct.cli.main import app

NiftiArray = npt.NDArray[np.generic]
AffineArray = npt.NDArray[np.float64]
WriteNiftiFile = Callable[[Path, NiftiArray, AffineArray], Path]


def _invoke_validate_pair(image_path: Path, label_path: Path, *extra_args: str) -> Any:
    """Invoke the public CLI command with a synthetic image-label pair."""
    runner = CliRunner()
    return runner.invoke(
        app,
        ["validate-pair", str(image_path), str(label_path), *extra_args],
    )


def test_cli_success_outputs_validation_result(
    tmp_path: Path,
    valid_ct_image: NiftiArray,
    valid_binary_tumor_mask: NiftiArray,
    matching_affine: AffineArray,
    write_nifti_file: WriteNiftiFile,
) -> None:
    image_path = write_nifti_file(tmp_path / "image.nii.gz", valid_ct_image, matching_affine)
    label_path = write_nifti_file(
        tmp_path / "label.nii.gz",
        valid_binary_tumor_mask,
        matching_affine,
    )

    result = _invoke_validate_pair(image_path, label_path)

    assert result.exit_code == 0
    assert "validation success" in result.output
    assert "shape: (2, 3, 4)" in result.output
    assert "label values: [0, 1]" in result.output


def test_cli_invalid_pair_exits_nonzero_without_traceback(
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

    result = _invoke_validate_pair(image_path, label_path)
    error_output = getattr(result, "stderr", "") or result.output

    assert result.exit_code != 0
    assert "label data must be binary" in error_output
    assert "Traceback" not in result.output
    assert "Traceback" not in error_output


def test_cli_accepts_affine_tolerance_option(
    tmp_path: Path,
    valid_ct_image: NiftiArray,
    valid_binary_tumor_mask: NiftiArray,
    matching_affine: AffineArray,
    write_nifti_file: WriteNiftiFile,
) -> None:
    label_affine = matching_affine.copy()
    label_affine[0, 3] += 5e-2
    image_path = write_nifti_file(tmp_path / "image.nii.gz", valid_ct_image, matching_affine)
    label_path = write_nifti_file(tmp_path / "label.nii.gz", valid_binary_tumor_mask, label_affine)

    result = _invoke_validate_pair(image_path, label_path, "--affine-tolerance", "0.1")

    assert result.exit_code == 0
    assert "validation success" in result.output
    assert "label values: [0, 1]" in result.output
