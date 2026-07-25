"""Unit tests for deterministic Phase 1 synthetic data generation."""

from __future__ import annotations

import json
import math
from pathlib import Path, PurePosixPath
from typing import Any, cast

import nibabel as nib
import numpy as np
import pytest

from protoem_ct.artifacts import artifact_from_json, hash_config, hash_manifest, sha256_file
from protoem_ct.artifacts.schemas import SyntheticManifest
from protoem_ct.data import synthetic as synthetic_module
from protoem_ct.data.synthetic import (
    IMAGE_DTYPE,
    LABEL_DTYPE,
    MANIFEST_FILENAME,
    SyntheticConfigError,
    SyntheticDataConfig,
    SyntheticOutputError,
    create_synthetic_dataset,
    generate_synthetic_case,
    load_synthetic_config,
    synthetic_config_hash_payload,
    synthetic_manifest_hash_payload,
)
from protoem_ct.data.validation import validate_nifti_pair

GIT_COMMIT = "f0b9066"
CREATED_AT_UTC = "2026-07-25T00:00:00Z"


def _config(**overrides: object) -> SyntheticDataConfig:
    values: dict[str, object] = {
        "dataset_id": "phase1-synthetic",
        "seed": 1729,
        "case_count": 3,
        "shape": (8, 8, 6),
        "spacing": (1.5, 1.5, 2.0),
        "generated_data_root": PurePosixPath("generated/phase1/data"),
    }
    values.update(overrides)
    return SyntheticDataConfig(**values)  # type: ignore[arg-type]


def _write_config(path: Path, **overrides: object) -> Path:
    values: dict[str, object] = {
        "dataset_id": "phase1-synthetic",
        "seed": 1729,
        "case_count": 3,
        "shape": [8, 8, 6],
        "spacing": [1.5, 1.5, 2.0],
        "generated_data_root": "generated/phase1/data",
    }
    values.update(overrides)
    path.write_text(json.dumps(values, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_valid_config_loading() -> None:
    config = load_synthetic_config(Path("configs/data/synthetic.yaml"))

    assert config == _config()


def test_unknown_config_key_rejected(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path / "synthetic.yaml", unexpected=True)

    with pytest.raises(SyntheticConfigError, match="unknown"):
        load_synthetic_config(config_path)


@pytest.mark.parametrize("dataset_id", ["", "   "])
def test_invalid_dataset_id_rejected(tmp_path: Path, dataset_id: str) -> None:
    config_path = _write_config(tmp_path / "synthetic.yaml", dataset_id=dataset_id)

    with pytest.raises(SyntheticConfigError, match="dataset_id"):
        load_synthetic_config(config_path)


def test_negative_seed_rejected(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path / "synthetic.yaml", seed=-1)

    with pytest.raises(SyntheticConfigError, match="seed"):
        load_synthetic_config(config_path)


@pytest.mark.parametrize("case_count", [0, -2])
def test_invalid_case_count_rejected(tmp_path: Path, case_count: int) -> None:
    config_path = _write_config(tmp_path / "synthetic.yaml", case_count=case_count)

    with pytest.raises(SyntheticConfigError, match="case_count"):
        load_synthetic_config(config_path)


@pytest.mark.parametrize("shape", [[8, 8], [8, 8, 0], [8, -1, 6], [8.0, 8, 6]])
def test_invalid_shape_rejected(tmp_path: Path, shape: list[object]) -> None:
    config_path = _write_config(tmp_path / "synthetic.yaml", shape=shape)

    with pytest.raises(SyntheticConfigError, match="shape"):
        load_synthetic_config(config_path)


@pytest.mark.parametrize(
    "spacing",
    [
        [1.5, 1.5],
        [1.5, 0.0, 2.0],
        [1.5, -1.0, 2.0],
        [1.5, math.inf, 2.0],
    ],
)
def test_invalid_spacing_rejected(tmp_path: Path, spacing: list[float]) -> None:
    config_path = _write_config(tmp_path / "synthetic.yaml", spacing=spacing)

    with pytest.raises(SyntheticConfigError, match="spacing"):
        load_synthetic_config(config_path)


def test_absolute_generated_path_rejected(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path / "synthetic.yaml", generated_data_root="/tmp/data")

    with pytest.raises(SyntheticConfigError, match="relative"):
        load_synthetic_config(config_path)


@pytest.mark.parametrize(
    "generated_data_root",
    ["../generated/data", "generated/../data", r"generated\..\data"],
)
def test_parent_traversal_rejected(tmp_path: Path, generated_data_root: str) -> None:
    config_path = _write_config(
        tmp_path / "synthetic.yaml",
        generated_data_root=generated_data_root,
    )

    with pytest.raises(SyntheticConfigError, match="parent traversal"):
        load_synthetic_config(config_path)


def test_deterministic_arrays_and_masks() -> None:
    config = _config()
    first = generate_synthetic_case(config, 0)
    second = generate_synthetic_case(config, 0)
    different_seed = generate_synthetic_case(_config(seed=1730), 0)

    assert first.image.dtype == IMAGE_DTYPE
    assert first.label.dtype == LABEL_DTYPE
    np.testing.assert_array_equal(first.image, second.image)
    np.testing.assert_array_equal(first.label, second.label)
    assert not np.array_equal(first.image, different_seed.image)


def test_binary_labels_contain_both_classes() -> None:
    case = generate_synthetic_case(_config(), 0)

    assert set(np.unique(case.label).tolist()) == {0, 1}


def test_stable_case_ids() -> None:
    config = _config(case_count=4)

    assert [generate_synthetic_case(config, index).case_id for index in range(4)] == [
        "synthetic-000",
        "synthetic-001",
        "synthetic-002",
        "synthetic-003",
    ]


def test_deterministic_file_hashes_across_separate_output_directories(tmp_path: Path) -> None:
    config = _config()
    first = create_synthetic_dataset(
        config,
        output_root=tmp_path / "first",
        git_commit=GIT_COMMIT,
        created_at_utc=CREATED_AT_UTC,
    )
    second = create_synthetic_dataset(
        config,
        output_root=tmp_path / "second",
        git_commit=GIT_COMMIT,
        created_at_utc=CREATED_AT_UTC,
    )

    first_paths = [*first.manifest.image_paths, *first.manifest.label_paths]
    second_paths = [*second.manifest.image_paths, *second.manifest.label_paths]
    assert first_paths == second_paths
    assert [sha256_file(tmp_path / "first" / relative_path) for relative_path in first_paths] == [
        sha256_file(tmp_path / "second" / relative_path) for relative_path in second_paths
    ]
    assert sha256_file(first.manifest_path) == sha256_file(second.manifest_path)


def test_existing_nonempty_output_directory_rejected(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    output_root.mkdir()
    (output_root / "existing.txt").write_text("occupied\n", encoding="utf-8")

    with pytest.raises(SyntheticOutputError, match="empty or nonexistent"):
        create_synthetic_dataset(
            _config(),
            output_root=output_root,
            git_commit=GIT_COMMIT,
            created_at_utc=CREATED_AT_UTC,
        )


def test_existing_target_file_rejected(tmp_path: Path) -> None:
    target_path = tmp_path / "existing.nii"
    target_path.write_bytes(b"existing\n")

    with pytest.raises(SyntheticOutputError, match="overwrite existing output file"):
        synthetic_module._write_nifti_file(
            target_path,
            np.zeros((2, 1, 1), dtype=IMAGE_DTYPE),
            np.eye(4, dtype=np.float64),
        )


def test_manifest_contains_only_relative_paths(tmp_path: Path) -> None:
    result = create_synthetic_dataset(
        _config(),
        output_root=tmp_path / "output",
        git_commit=GIT_COMMIT,
        created_at_utc=CREATED_AT_UTC,
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    for relative_path in [*manifest["image_paths"], *manifest["label_paths"]]:
        assert not Path(relative_path).is_absolute()
        assert ".." not in PurePosixPath(relative_path).parts
    assert str(tmp_path) not in result.manifest_path.read_text(encoding="utf-8")


def test_manifest_hash_is_stable(tmp_path: Path) -> None:
    first = create_synthetic_dataset(
        _config(),
        output_root=tmp_path / "first",
        git_commit=GIT_COMMIT,
        created_at_utc=CREATED_AT_UTC,
    )
    second = create_synthetic_dataset(
        _config(),
        output_root=tmp_path / "second",
        git_commit=GIT_COMMIT,
        created_at_utc=CREATED_AT_UTC,
    )

    assert first.manifest_hash == second.manifest_hash
    assert first.manifest_hash == hash_manifest(synthetic_manifest_hash_payload(first.manifest))
    assert "manifest_hash" not in synthetic_manifest_hash_payload(first.manifest)


def test_config_hash_is_stable() -> None:
    first = _config()
    second = _config()

    assert hash_config(synthetic_config_hash_payload(first)) == hash_config(
        synthetic_config_hash_payload(second)
    )


def test_manifest_json_round_trips_and_ends_with_newline(tmp_path: Path) -> None:
    result = create_synthetic_dataset(
        _config(),
        output_root=tmp_path / "output",
        git_commit=GIT_COMMIT,
        created_at_utc=CREATED_AT_UTC,
    )
    text = result.manifest_path.read_text(encoding="utf-8")
    manifest = artifact_from_json(text, SyntheticManifest)

    assert manifest == result.manifest
    assert (
        result.manifest_relative_path == PurePosixPath("generated/phase1/data") / MANIFEST_FILENAME
    )
    assert text.endswith("\n")
    assert list(json.loads(text)) == sorted(json.loads(text))


def test_output_files_pass_existing_validate_nifti_pair(tmp_path: Path) -> None:
    result = create_synthetic_dataset(
        _config(),
        output_root=tmp_path / "output",
        git_commit=GIT_COMMIT,
        created_at_utc=CREATED_AT_UTC,
    )

    for image_relative, label_relative in zip(
        result.manifest.image_paths,
        result.manifest.label_paths,
        strict=True,
    ):
        validation = validate_nifti_pair(
            tmp_path / "output" / image_relative,
            tmp_path / "output" / label_relative,
        )
        image = cast(Any, nib.load(str(tmp_path / "output" / image_relative)))
        label = cast(Any, nib.load(str(tmp_path / "output" / label_relative)))
        assert validation.shape == _config().shape
        assert validation.label_values == (0, 1)
        assert image.header.get_data_dtype() == IMAGE_DTYPE
        assert label.header.get_data_dtype() == LABEL_DTYPE
