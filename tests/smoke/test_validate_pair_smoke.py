"""Fast smoke test for the validate-pair CLI path."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt
from typer.testing import CliRunner

from protoem_ct.cli.main import app

NiftiArray = npt.NDArray[np.generic]
AffineArray = npt.NDArray[np.float64]
WriteNiftiFile = Callable[[Path, NiftiArray, AffineArray], Path]


def test_validate_pair_smoke(
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

    result = CliRunner().invoke(app, ["validate-pair", str(image_path), str(label_path)])

    assert result.exit_code == 0
    assert result.output == "validation success\nshape: (2, 3, 4)\nlabel values: [0, 1]\n"
