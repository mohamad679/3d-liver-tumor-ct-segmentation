from __future__ import annotations

import gzip
import inspect
import json
import struct
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import nibabel as nib
import numpy as np
import pytest
from scipy import ndimage  # type: ignore[import-untyped]

from protoem_ct.baselines import synthetic as synthetic_module
from protoem_ct.baselines.synthetic import (
    BASELINE_SYNTHETIC_AFFINE,
    BASELINE_SYNTHETIC_DATASET_NAME,
    BASELINE_SYNTHETIC_SHAPE,
    BASELINE_SYNTHETIC_SPACING_MM,
    BASELINE_SYNTHETIC_TEST_CASE_IDENTIFIERS,
    BASELINE_SYNTHETIC_TRAINING_CASE_IDENTIFIERS,
    BaselineSyntheticFixtureArtifact,
    BaselineSyntheticFixtureFileRecord,
    BaselineSyntheticFixtureHashError,
    BaselineSyntheticFixtureSerializationError,
    BaselineSyntheticFixtureValidationError,
    BaselineSyntheticFixtureVersionError,
    baseline_synthetic_fixture_from_json,
    baseline_synthetic_fixture_to_json,
    generate_baseline_synthetic_fixture,
    synthetic_affine_array,
    synthetic_cases,
    synthetic_dataset_json,
)


def _lesion_count(label: np.ndarray[Any, Any]) -> int:
    structure = ndimage.generate_binary_structure(3, 3)
    _labeled, count = ndimage.label(label, structure=structure)
    return int(count)


def _relative_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): __import__("hashlib")
        .sha256(path.read_bytes())
        .hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_exact_case_counts_and_identifiers() -> None:
    cases = synthetic_cases()

    assert (
        tuple(case.case_identifier for case in cases[:5])
        == BASELINE_SYNTHETIC_TRAINING_CASE_IDENTIFIERS
    )
    assert (
        tuple(case.case_identifier for case in cases[5:])
        == BASELINE_SYNTHETIC_TEST_CASE_IDENTIFIERS
    )
    assert len(cases) == 7


def test_exact_shape_spacing_affine_qform_and_sform(tmp_path: Path) -> None:
    output_root = tmp_path / "fixture-root"
    result = generate_baseline_synthetic_fixture(output_root)

    assert result.artifact.image_shape == BASELINE_SYNTHETIC_SHAPE
    assert result.artifact.voxel_spacing_mm == BASELINE_SYNTHETIC_SPACING_MM
    assert result.artifact.affine == BASELINE_SYNTHETIC_AFFINE

    image_path = (
        output_root
        / BASELINE_SYNTHETIC_DATASET_NAME
        / "imagesTr"
        / f"{BASELINE_SYNTHETIC_TRAINING_CASE_IDENTIFIERS[1]}_0000.nii.gz"
    )
    image = cast(Any, nib.load(str(image_path)))
    qform, qform_code = image.get_qform(coded=True)
    sform, sform_code = image.get_sform(coded=True)
    assert image.shape == BASELINE_SYNTHETIC_SHAPE
    assert (
        tuple(float(value) for value in image.header.get_zooms()[:3])
        == BASELINE_SYNTHETIC_SPACING_MM
    )
    assert np.array_equal(qform, synthetic_affine_array())
    assert np.array_equal(sform, synthetic_affine_array())
    assert qform_code != 0
    assert sform_code != 0


def test_exact_image_and_label_dtypes_and_label_values(tmp_path: Path) -> None:
    output_root = tmp_path / "fixture-root"
    generate_baseline_synthetic_fixture(output_root)

    image_path = (
        output_root
        / BASELINE_SYNTHETIC_DATASET_NAME
        / "imagesTr"
        / f"{BASELINE_SYNTHETIC_TRAINING_CASE_IDENTIFIERS[1]}_0000.nii.gz"
    )
    label_path = (
        output_root
        / BASELINE_SYNTHETIC_DATASET_NAME
        / "labelsTr"
        / f"{BASELINE_SYNTHETIC_TRAINING_CASE_IDENTIFIERS[1]}.nii.gz"
    )
    image = cast(Any, nib.load(str(image_path)))
    label = cast(Any, nib.load(str(label_path)))
    image_data = np.asanyarray(image.dataobj)
    label_data = np.asanyarray(label.dataobj)

    assert image.get_data_dtype() == np.dtype(np.int16)
    assert label.get_data_dtype() == np.dtype(np.uint8)
    assert image_data.dtype == np.int16
    assert label_data.dtype == np.uint8
    assert set(np.unique(label_data).tolist()) <= {0, 1}


def test_empty_one_two_boundary_irregular_and_test_lesion_patterns() -> None:
    cases = {case.case_identifier: case for case in synthetic_cases()}

    assert np.count_nonzero(cases["synthetic_train_001"].label) == 0
    assert _lesion_count(cases["synthetic_train_002"].label) == 1
    assert _lesion_count(cases["synthetic_train_003"].label) == 2
    assert np.any(cases["synthetic_train_004"].label[0, :, :] == 1)
    assert _lesion_count(cases["synthetic_train_005"].label) == 1
    assert _lesion_count(cases["synthetic_test_001"].label) == 1
    assert _lesion_count(cases["synthetic_test_002"].label) == 2
    lesion_sizes = sorted(int(np.count_nonzero(case.label)) for case in cases.values())
    assert len(set(lesion_sizes)) > 1


def test_deterministic_coordinate_based_image_generation_and_repeat_equality() -> None:
    first_cases = synthetic_cases()
    second_cases = synthetic_cases()

    first = first_cases[1]
    second = second_cases[1]
    assert np.array_equal(first.image, second.image)
    assert np.array_equal(first.label, second.label)
    x_axis, y_axis, z_axis = 10, 10, 6
    expected = (
        -860
        + 17 * x_axis
        - 11 * y_axis
        + 29 * z_axis
        + 13 * 2
        + ((x_axis * y_axis + z_axis * 3 + 2) % 19)
        - ((x_axis + z_axis + 2) % 7) * 5
        + 340
    )
    assert int(first.image[x_axis, y_axis, z_axis]) == expected


def test_no_random_api_usage_in_production_implementation() -> None:
    source = inspect.getsource(synthetic_module).lower()

    assert "random" not in source
    assert "uuid" not in source


def test_repeated_relative_file_hash_equality_across_output_roots(tmp_path: Path) -> None:
    first_root = tmp_path / "first-root"
    second_root = tmp_path / "second-root"
    generate_baseline_synthetic_fixture(first_root)
    generate_baseline_synthetic_fixture(second_root)

    assert _relative_hashes(first_root) == _relative_hashes(second_root)


def test_deterministic_compressed_nifti_bytes_and_fixed_gzip_metadata(tmp_path: Path) -> None:
    first_root = tmp_path / "first-root"
    second_root = tmp_path / "second-root"
    generate_baseline_synthetic_fixture(first_root)
    generate_baseline_synthetic_fixture(second_root)

    relative_path = (
        f"{BASELINE_SYNTHETIC_DATASET_NAME}/imagesTr/"
        f"{BASELINE_SYNTHETIC_TRAINING_CASE_IDENTIFIERS[1]}_0000.nii.gz"
    )
    first_bytes = (first_root / relative_path).read_bytes()
    second_bytes = (second_root / relative_path).read_bytes()

    assert first_bytes == second_bytes
    assert first_bytes[:2] == b"\x1f\x8b"
    assert first_bytes[3] == 0
    assert struct.unpack("<I", first_bytes[4:8])[0] == 0
    assert gzip.decompress(first_bytes) == gzip.decompress(second_bytes)


def test_dataset_json_is_deterministic() -> None:
    assert synthetic_dataset_json() == {
        "channel_names": {"0": "CT"},
        "file_ending": ".nii.gz",
        "labels": {"background": 0, "tumor": 1},
        "name": BASELINE_SYNTHETIC_DATASET_NAME,
        "numTest": 2,
        "numTraining": 5,
    }


def test_artifact_json_hash_round_trip_and_tamper_detection(tmp_path: Path) -> None:
    output_root = tmp_path / "fixture-root"
    result = generate_baseline_synthetic_fixture(output_root)
    artifact_json = baseline_synthetic_fixture_to_json(result.artifact)

    parsed = baseline_synthetic_fixture_from_json(artifact_json)
    assert parsed.artifact_hash == result.artifact.artifact_hash

    payload = json.loads(artifact_json)
    payload["file_records"][0]["byte_size"] += 1
    with pytest.raises(BaselineSyntheticFixtureHashError):
        baseline_synthetic_fixture_from_json(json.dumps(payload).encode("utf-8"))


def test_unknown_field_version_and_role_rejection(tmp_path: Path) -> None:
    artifact = generate_baseline_synthetic_fixture(tmp_path / "fixture-root").artifact
    payload = json.loads(baseline_synthetic_fixture_to_json(artifact))

    payload_with_unknown = dict(payload)
    payload_with_unknown["unknown_field"] = "value"
    with pytest.raises(BaselineSyntheticFixtureSerializationError):
        baseline_synthetic_fixture_from_json(json.dumps(payload_with_unknown).encode("utf-8"))

    payload_with_version = dict(payload)
    payload_with_version["contract_version"] = "baseline_synthetic_fixture_v999"
    with pytest.raises(BaselineSyntheticFixtureVersionError):
        baseline_synthetic_fixture_from_json(json.dumps(payload_with_version).encode("utf-8"))

    payload_with_role = json.loads(baseline_synthetic_fixture_to_json(artifact))
    payload_with_role["file_records"][0]["role"] = "invalid_role"
    with pytest.raises(BaselineSyntheticFixtureValidationError):
        baseline_synthetic_fixture_from_json(json.dumps(payload_with_role).encode("utf-8"))


def test_relative_path_duplicate_unsorted_and_absolute_path_rejection(tmp_path: Path) -> None:
    artifact = generate_baseline_synthetic_fixture(tmp_path / "fixture-root").artifact
    first_record = artifact.file_records[0]

    with pytest.raises(BaselineSyntheticFixtureValidationError):
        BaselineSyntheticFixtureFileRecord(
            relative_path="/abs/path",
            role="dataset_metadata",
            case_identifier=None,
            channel_index=None,
            sha256="a" * 64,
            byte_size=1,
        )

    with pytest.raises(BaselineSyntheticFixtureValidationError):
        cast(
            Any,
            replace,
        )(
            artifact,
            file_records=tuple(reversed(artifact.file_records)),
        )

    duplicate = BaselineSyntheticFixtureFileRecord(
        relative_path=first_record.relative_path,
        role=first_record.role,
        case_identifier=first_record.case_identifier,
        channel_index=first_record.channel_index,
        sha256=first_record.sha256,
        byte_size=first_record.byte_size,
    )
    with pytest.raises(BaselineSyntheticFixtureValidationError):
        cast(
            Any,
            replace,
        )(
            artifact,
            file_records=(artifact.file_records[0], duplicate, *artifact.file_records[2:]),
        )


def test_no_absolute_path_persisted(tmp_path: Path) -> None:
    output_root = tmp_path / "fixture-root"
    result = generate_baseline_synthetic_fixture(output_root)
    artifact_text = baseline_synthetic_fixture_to_json(result.artifact).decode("utf-8")

    assert str(output_root) not in artifact_text
    assert str(tmp_path) not in artifact_text


def test_no_overwrite_behavior(tmp_path: Path) -> None:
    output_root = tmp_path / "fixture-root"
    output_root.mkdir()
    before = output_root / "marker.txt"
    before.write_text("keep\n", encoding="utf-8")

    with pytest.raises(BaselineSyntheticFixtureValidationError):
        generate_baseline_synthetic_fixture(output_root)

    assert before.read_text(encoding="utf-8") == "keep\n"


@pytest.mark.skipif(not hasattr(Path, "symlink_to"), reason="Symlinks unsupported")
def test_symlink_destination_rejection(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    output_root = tmp_path / "fixture-root"
    output_root.symlink_to(target, target_is_directory=True)

    with pytest.raises(BaselineSyntheticFixtureValidationError):
        generate_baseline_synthetic_fixture(output_root)


@pytest.mark.skipif(not hasattr(Path, "symlink_to"), reason="Symlinks unsupported")
def test_symlinked_parent_component_rejection(tmp_path: Path) -> None:
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(BaselineSyntheticFixtureValidationError):
        generate_baseline_synthetic_fixture(linked_parent / "fixture-root")


def test_failure_cleanup_and_no_partial_final_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "fixture-root"

    def failing_materialize(root: Path) -> BaselineSyntheticFixtureArtifact:
        marker = root / "partial.txt"
        marker.write_text("partial\n", encoding="utf-8")
        raise BaselineSyntheticFixtureValidationError("fail")

    monkeypatch.setattr(synthetic_module, "_materialize_fixture_root", failing_materialize)

    with pytest.raises(BaselineSyntheticFixtureValidationError):
        generate_baseline_synthetic_fixture(output_root)

    assert not output_root.exists()
    assert not any(path.name.startswith(".fixture-root.tmp-") for path in tmp_path.iterdir())


def test_no_heavy_ml_imports_from_baselines_synthetic(
    run_import_guard: Callable[[tuple[str, ...], tuple[str, ...]], None],
) -> None:
    run_import_guard(
        ("protoem_ct.baselines.synthetic",),
        ("torch", "torchvision", "monai", "nnunetv2", "SimpleITK", "mlflow"),
    )
