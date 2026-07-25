"""Unit tests for the Phase 1 synthetic preprocessing stage."""

from __future__ import annotations

import json
import math
from pathlib import Path, PurePosixPath
from typing import Any, cast

import nibabel as nib
import numpy as np
import pytest

from protoem_ct.artifacts import (
    PreprocessArtifact,
    SyntheticManifest,
    ValidationArtifact,
    artifact_from_json,
    artifact_to_dict,
    hash_config,
    hash_manifest,
    sha256_file,
)
from protoem_ct.data.manifest_validation import validate_synthetic_manifest
from protoem_ct.data.preprocessing import (
    PreprocessingConfig,
    PreprocessingConfigError,
    PreprocessingFailureError,
    PreprocessingInputArtifactError,
    PreprocessingOutputCollisionError,
    PreprocessingPathError,
    load_preprocessing_config,
    preprocess_synthetic_dataset,
    preprocessing_config_hash_payload,
)
from protoem_ct.data.synthetic import (
    SyntheticDataConfig,
    SyntheticDatasetResult,
    create_synthetic_dataset,
)

GIT_COMMIT = "728d38b"
CREATED_AT_UTC = "2026-07-25T01:00:00Z"
VALIDATION_CREATED_AT_UTC = "2026-07-25T01:05:00Z"
PREPROCESS_CREATED_AT_UTC = "2026-07-25T01:10:00Z"
NIFTI1_HEADER_AND_EXTENDER_SIZE = 352


def _synthetic_config(**overrides: object) -> SyntheticDataConfig:
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


def _preprocessing_values(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "clip_min": -1000.0,
        "clip_max": 1000.0,
        "output_min": 0.0,
        "output_max": 1.0,
        "image_dtype": "float32",
        "label_dtype": "uint8",
    }
    values.update(overrides)
    return values


def _write_preprocessing_config(path: Path, **overrides: object) -> Path:
    values = _preprocessing_values(**overrides)
    lines = ["preprocessing:"]
    for key in sorted(values):
        lines.append(f"  {key}: {_yaml_scalar(values[key])}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


def _yaml_scalar(value: object) -> str:
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and math.isnan(value):
        return ".nan"
    if isinstance(value, float) and math.isinf(value):
        return ".inf" if value > 0 else "-.inf"
    return str(value)


def _dataset(tmp_path: Path) -> SyntheticDatasetResult:
    return create_synthetic_dataset(
        _synthetic_config(),
        output_root=tmp_path / "data-root",
        git_commit=GIT_COMMIT,
        created_at_utc=CREATED_AT_UTC,
    )


def _data_root(_result: SyntheticDatasetResult, tmp_path: Path) -> Path:
    return tmp_path / "data-root"


def _validate(
    result: SyntheticDatasetResult,
    tmp_path: Path,
) -> tuple[Path, ValidationArtifact]:
    validation_path = tmp_path / "artifacts" / "validation.json"
    validation = validate_synthetic_manifest(
        result.manifest_path,
        data_root=_data_root(result, tmp_path),
        output_path=validation_path,
        git_commit=GIT_COMMIT,
        created_at_utc=VALIDATION_CREATED_AT_UTC,
    )
    return validation_path, validation


def _preprocess(
    result: SyntheticDatasetResult,
    validation_path: Path,
    tmp_path: Path,
    *,
    output_root: Path | None = None,
    artifact_output_path: Path | None = None,
    config_path: Path = Path("configs/experiment/phase1.yaml"),
) -> PreprocessArtifact:
    return preprocess_synthetic_dataset(
        result.manifest_path,
        validation_path,
        data_root=_data_root(result, tmp_path),
        output_root=output_root or tmp_path / "preprocessed",
        artifact_output_path=artifact_output_path or tmp_path / "artifacts" / "preprocess.json",
        config_path=config_path,
        git_commit=GIT_COMMIT,
        created_at_utc=PREPROCESS_CREATED_AT_UTC,
    )


def _manifest_payload(result: SyntheticDatasetResult) -> dict[str, Any]:
    return cast(dict[str, Any], artifact_to_dict(result.manifest))


def _write_manifest_payload(
    result: SyntheticDatasetResult,
    payload: dict[str, Any],
    *,
    recompute_manifest_hash: bool,
) -> SyntheticManifest:
    _write_manifest_json(result, payload, recompute_manifest_hash=recompute_manifest_hash)
    return artifact_from_json(
        result.manifest_path.read_text(encoding="utf-8"),
        SyntheticManifest,
    )


def _write_manifest_json(
    result: SyntheticDatasetResult,
    payload: dict[str, Any],
    *,
    recompute_manifest_hash: bool,
) -> None:
    if recompute_manifest_hash:
        payload["manifest_hash"] = "0" * 64
        hash_payload = {key: value for key, value in payload.items() if key != "manifest_hash"}
        payload["manifest_hash"] = hash_manifest(hash_payload)
    result.manifest_path.write_text(
        json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_validation_payload(
    validation_path: Path,
    validation: ValidationArtifact,
    **overrides: object,
) -> None:
    payload = cast(dict[str, Any], artifact_to_dict(validation))
    payload.update(overrides)
    validation_path.write_text(
        json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _load_nifti(path: Path) -> Any:
    return cast(Any, nib.load(str(path)))


def _array(path: Path) -> np.ndarray[Any, Any]:
    return cast(np.ndarray[Any, Any], np.asanyarray(_load_nifti(path).dataobj))


def _corrupt_nifti_header(path: Path) -> None:
    """Replace the NIfTI-1 header without truncating the test input."""
    payload = path.read_bytes()
    if len(payload) < NIFTI1_HEADER_AND_EXTENDER_SIZE:
        msg = f"test NIfTI is unexpectedly short: {path}"
        raise AssertionError(msg)
    path.write_bytes(
        bytes(NIFTI1_HEADER_AND_EXTENDER_SIZE) + payload[NIFTI1_HEADER_AND_EXTENDER_SIZE:]
    )


def test_valid_preprocessing_configuration() -> None:
    config = load_preprocessing_config(Path("configs/experiment/phase1.yaml"))

    assert config == PreprocessingConfig(
        clip_min=-1000.0,
        clip_max=1000.0,
        output_min=0.0,
        output_max=1.0,
        image_dtype="float32",
        label_dtype="uint8",
    )


def test_unknown_configuration_key_rejected(tmp_path: Path) -> None:
    config_path = _write_preprocessing_config(tmp_path / "phase1.yaml", unexpected=True)

    with pytest.raises(PreprocessingConfigError, match="unknown"):
        load_preprocessing_config(config_path)


def test_invalid_clipping_range_rejected(tmp_path: Path) -> None:
    config_path = _write_preprocessing_config(
        tmp_path / "phase1.yaml",
        clip_min=1000.0,
        clip_max=1000.0,
    )

    with pytest.raises(PreprocessingConfigError, match="clip_min"):
        load_preprocessing_config(config_path)


def test_invalid_output_range_rejected(tmp_path: Path) -> None:
    config_path = _write_preprocessing_config(
        tmp_path / "phase1.yaml",
        output_min=1.0,
        output_max=0.0,
    )

    with pytest.raises(PreprocessingConfigError, match="output_min"):
        load_preprocessing_config(config_path)


@pytest.mark.parametrize("field_name", ["clip_min", "clip_max", "output_min", "output_max"])
def test_non_finite_configuration_rejected(tmp_path: Path, field_name: str) -> None:
    config_path = _write_preprocessing_config(tmp_path / "phase1.yaml", **{field_name: math.inf})

    with pytest.raises(PreprocessingConfigError, match=field_name):
        load_preprocessing_config(config_path)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [("image_dtype", "float64"), ("label_dtype", "int16")],
)
def test_unsupported_dtype_rejected(tmp_path: Path, field_name: str, value: str) -> None:
    config_path = _write_preprocessing_config(tmp_path / "phase1.yaml", **{field_name: value})

    with pytest.raises(PreprocessingConfigError, match=field_name):
        load_preprocessing_config(config_path)


def test_valid_dataset_preprocessing_preserves_geometry_labels_and_inputs(
    tmp_path: Path,
) -> None:
    result = _dataset(tmp_path)
    validation_path, _validation = _validate(result, tmp_path)
    data_root = _data_root(result, tmp_path)
    input_hashes = {
        relative_path: sha256_file(data_root / relative_path)
        for relative_path in [*result.manifest.image_paths, *result.manifest.label_paths]
    }
    input_labels = {
        label_relative: _array(data_root / label_relative)
        for label_relative in result.manifest.label_paths
    }
    input_affines = {
        relative_path: np.asarray(_load_nifti(data_root / relative_path).affine)
        for relative_path in [*result.manifest.image_paths, *result.manifest.label_paths]
    }

    artifact = _preprocess(result, validation_path, tmp_path)

    assert artifact.output_case_ids == result.manifest.case_ids
    assert artifact.output_image_paths == (
        "images/synthetic-000.nii",
        "images/synthetic-001.nii",
        "images/synthetic-002.nii",
    )
    assert artifact.output_label_paths == (
        "labels/synthetic-000.nii",
        "labels/synthetic-001.nii",
        "labels/synthetic-002.nii",
    )
    assert artifact.target_spacing == result.manifest.spacing
    assert artifact.preprocessing_parameters["resampling"] is False

    for image_relative, label_relative, output_image_relative, output_label_relative in zip(
        result.manifest.image_paths,
        result.manifest.label_paths,
        artifact.output_image_paths,
        artifact.output_label_paths,
        strict=True,
    ):
        output_image_path = tmp_path / "preprocessed" / output_image_relative
        output_label_path = tmp_path / "preprocessed" / output_label_relative
        output_image = _load_nifti(output_image_path)
        output_label = _load_nifti(output_label_path)
        output_image_array = _array(output_image_path)
        output_label_array = _array(output_label_path)

        assert output_image_path.suffix == ".nii"
        assert output_label_path.suffix == ".nii"
        assert output_image_array.shape == result.manifest.shape
        assert output_label_array.shape == result.manifest.shape
        np.testing.assert_array_equal(output_image.affine, input_affines[image_relative])
        np.testing.assert_array_equal(output_label.affine, input_affines[label_relative])
        assert output_image.header.get_data_dtype() == np.dtype(np.float32)
        assert output_label.header.get_data_dtype() == np.dtype(np.uint8)
        assert float(output_image_array.min()) >= 0.0
        assert float(output_image_array.max()) <= 1.0
        assert set(np.unique(output_label_array).tolist()) == {0, 1}
        np.testing.assert_array_equal(output_label_array, input_labels[label_relative])

    assert input_hashes == {
        relative_path: sha256_file(data_root / relative_path)
        for relative_path in [*result.manifest.image_paths, *result.manifest.label_paths]
    }


def test_deterministic_sha256_hashes_across_separate_output_roots(tmp_path: Path) -> None:
    result = _dataset(tmp_path)
    validation_path, _validation = _validate(result, tmp_path)
    first_artifact_path = tmp_path / "first-artifacts" / "preprocess.json"
    second_artifact_path = tmp_path / "second-artifacts" / "preprocess.json"

    first = _preprocess(
        result,
        validation_path,
        tmp_path,
        output_root=tmp_path / "first-output",
        artifact_output_path=first_artifact_path,
    )
    second = _preprocess(
        result,
        validation_path,
        tmp_path,
        output_root=tmp_path / "second-output",
        artifact_output_path=second_artifact_path,
    )

    assert first == second
    for relative_path in [*first.output_image_paths, *first.output_label_paths]:
        assert sha256_file(tmp_path / "first-output" / relative_path) == sha256_file(
            tmp_path / "second-output" / relative_path
        )
    assert sha256_file(first_artifact_path) == sha256_file(second_artifact_path)


def test_manifest_and_config_hash_linkage_is_preserved(tmp_path: Path) -> None:
    result = _dataset(tmp_path)
    validation_path, validation = _validate(result, tmp_path)
    config = load_preprocessing_config(Path("configs/experiment/phase1.yaml"))

    artifact = _preprocess(result, validation_path, tmp_path)
    persisted = artifact_from_json(
        (tmp_path / "artifacts" / "preprocess.json").read_text(encoding="utf-8"),
        PreprocessArtifact,
    )

    assert persisted == artifact
    assert validation.config_hash == result.manifest.config_hash
    assert validation.manifest_hash == result.manifest.manifest_hash
    assert artifact.config_hash == hash_config(preprocessing_config_hash_payload(config))
    assert artifact.manifest_hash == result.manifest.manifest_hash


def test_validation_artifact_hash_mismatch_rejected(tmp_path: Path) -> None:
    result = _dataset(tmp_path)
    validation_path, validation = _validate(result, tmp_path)
    _write_validation_payload(validation_path, validation, manifest_hash="1" * 64)

    with pytest.raises(PreprocessingInputArtifactError, match="manifest hash"):
        _preprocess(result, validation_path, tmp_path)


def test_invalid_case_count_greater_than_zero_rejected(tmp_path: Path) -> None:
    result = _dataset(tmp_path)
    validation_path, validation = _validate(result, tmp_path)
    _write_validation_payload(validation_path, validation, invalid_case_count=1)

    with pytest.raises(PreprocessingInputArtifactError, match="invalid_case_count"):
        _preprocess(result, validation_path, tmp_path)


def test_validated_case_order_mismatch_rejected(tmp_path: Path) -> None:
    result = _dataset(tmp_path)
    validation_path, validation = _validate(result, tmp_path)
    _write_validation_payload(
        validation_path,
        validation,
        validated_case_ids=list(reversed(validation.validated_case_ids)),
    )

    with pytest.raises(PreprocessingInputArtifactError, match="validated_case_ids"):
        _preprocess(result, validation_path, tmp_path)


def test_missing_input_rejected(tmp_path: Path) -> None:
    result = _dataset(tmp_path)
    validation_path, _validation = _validate(result, tmp_path)
    (_data_root(result, tmp_path) / result.manifest.image_paths[0]).unlink()

    with pytest.raises(PreprocessingPathError, match="image path"):
        _preprocess(result, validation_path, tmp_path)


def test_unsafe_path_traversal_rejected(tmp_path: Path) -> None:
    result = _dataset(tmp_path)
    validation_path, _validation = _validate(result, tmp_path)
    payload = _manifest_payload(result)
    image_paths = cast(list[str], payload["image_paths"])
    image_paths[0] = "../escape.nii"
    _write_manifest_json(result, payload, recompute_manifest_hash=True)

    with pytest.raises(PreprocessingPathError, match="parent traversal"):
        _preprocess(result, validation_path, tmp_path)


def test_symlink_escape_rejected_when_supported(tmp_path: Path) -> None:
    result = _dataset(tmp_path)
    validation_path, validation = _validate(result, tmp_path)
    data_root = _data_root(result, tmp_path)
    outside_image = tmp_path / "outside.nii"
    outside_image.write_bytes((data_root / result.manifest.image_paths[0]).read_bytes())
    symlink_path = data_root / "generated/phase1/data/images/escape.nii"
    try:
        symlink_path.symlink_to(outside_image)
    except OSError as exc:
        pytest.skip(f"platform does not support this symlink test: {exc}")
    payload = _manifest_payload(result)
    image_paths = cast(list[str], payload["image_paths"])
    image_paths[0] = "generated/phase1/data/images/escape.nii"
    manifest = _write_manifest_payload(result, payload, recompute_manifest_hash=True)
    _write_validation_payload(validation_path, validation, manifest_hash=manifest.manifest_hash)

    with pytest.raises(PreprocessingPathError, match="escapes data_root"):
        _preprocess(result, validation_path, tmp_path)


def test_duplicate_resolved_input_path_rejected(tmp_path: Path) -> None:
    result = _dataset(tmp_path)
    validation_path, validation = _validate(result, tmp_path)
    payload = _manifest_payload(result)
    image_paths = cast(list[str], payload["image_paths"])
    image_paths[1] = image_paths[0]
    manifest = _write_manifest_payload(result, payload, recompute_manifest_hash=True)
    _write_validation_payload(validation_path, validation, manifest_hash=manifest.manifest_hash)

    with pytest.raises(PreprocessingPathError, match="duplicate resolved input path"):
        _preprocess(result, validation_path, tmp_path)


def test_nonempty_output_root_rejected(tmp_path: Path) -> None:
    result = _dataset(tmp_path)
    validation_path, _validation = _validate(result, tmp_path)
    output_root = tmp_path / "preprocessed"
    output_root.mkdir()
    (output_root / "existing.txt").write_text("occupied\n", encoding="utf-8")

    with pytest.raises(PreprocessingOutputCollisionError, match="empty or nonexistent"):
        _preprocess(result, validation_path, tmp_path, output_root=output_root)


def test_existing_artifact_output_rejected(tmp_path: Path) -> None:
    result = _dataset(tmp_path)
    validation_path, _validation = _validate(result, tmp_path)
    artifact_output_path = tmp_path / "artifacts" / "preprocess.json"
    artifact_output_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(PreprocessingOutputCollisionError, match="overwrite"):
        _preprocess(result, validation_path, tmp_path, artifact_output_path=artifact_output_path)


def test_output_root_inside_input_data_root_rejected(tmp_path: Path) -> None:
    result = _dataset(tmp_path)
    validation_path, _validation = _validate(result, tmp_path)

    with pytest.raises(PreprocessingPathError, match="inside data_root"):
        _preprocess(
            result,
            validation_path,
            tmp_path,
            output_root=_data_root(result, tmp_path) / "preprocessed",
        )


def test_failed_processing_leaves_no_partial_output_or_artifact(tmp_path: Path) -> None:
    result = _dataset(tmp_path)
    validation_path, _validation = _validate(result, tmp_path)
    data_root = _data_root(result, tmp_path)
    _corrupt_nifti_header(data_root / result.manifest.image_paths[1])
    input_hashes = {
        relative_path: sha256_file(data_root / relative_path)
        for relative_path in [*result.manifest.image_paths, *result.manifest.label_paths]
    }
    output_root = tmp_path / "preprocessed"
    artifact_output_path = tmp_path / "artifacts" / "preprocess.json"

    with pytest.raises(PreprocessingFailureError):
        _preprocess(
            result,
            validation_path,
            tmp_path,
            output_root=output_root,
            artifact_output_path=artifact_output_path,
        )

    assert not output_root.exists()
    assert not artifact_output_path.exists()
    assert not any(path.name.startswith(f".{output_root.name}.tmp-") for path in tmp_path.iterdir())
    assert input_hashes == {
        relative_path: sha256_file(data_root / relative_path)
        for relative_path in [*result.manifest.image_paths, *result.manifest.label_paths]
    }
