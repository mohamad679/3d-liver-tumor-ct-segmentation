"""Unit tests for deterministic Phase 1 dummy inference."""

from __future__ import annotations

import json
import math
from pathlib import Path, PurePosixPath
from typing import Any, cast

import nibabel as nib
import numpy as np
import pytest

from protoem_ct.artifacts import (
    InferenceArtifact,
    PreprocessArtifact,
    SyntheticManifest,
    ValidationArtifact,
    artifact_from_json,
    artifact_to_dict,
    sha256_file,
)
from protoem_ct.data.manifest_validation import validate_synthetic_manifest
from protoem_ct.data.preprocessing import preprocess_synthetic_dataset
from protoem_ct.data.synthetic import (
    SyntheticDataConfig,
    SyntheticDatasetResult,
    create_synthetic_dataset,
)
from protoem_ct.evaluation.dummy_inference import (
    DummyInferenceConfig,
    DummyInferenceConfigError,
    DummyInferenceImageError,
    DummyInferenceInputArtifactError,
    DummyInferenceOutputCollisionError,
    DummyInferencePathError,
    load_dummy_inference_config,
    run_dummy_inference,
)

GIT_COMMIT = "8ce4f1e"
CREATED_AT_UTC = "2026-07-25T01:00:00Z"
VALIDATION_CREATED_AT_UTC = "2026-07-25T01:05:00Z"
PREPROCESS_CREATED_AT_UTC = "2026-07-25T01:10:00Z"
INFERENCE_CREATED_AT_UTC = "2026-07-25T01:15:00Z"
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


def _dummy_config_values(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "method": "dummy",
        "prediction_dtype": "uint8",
        "seed": 2718,
        "threshold": 0.5,
    }
    values.update(overrides)
    return values


def _write_dummy_config(path: Path, **overrides: object) -> Path:
    values = _dummy_config_values(**overrides)
    lines = ["dummy_inference:"]
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


def _data_root(tmp_path: Path) -> Path:
    return tmp_path / "data-root"


def _validate(
    result: SyntheticDatasetResult,
    tmp_path: Path,
) -> tuple[Path, ValidationArtifact]:
    validation_path = tmp_path / "artifacts" / "validation.json"
    validation = validate_synthetic_manifest(
        result.manifest_path,
        data_root=_data_root(tmp_path),
        output_path=validation_path,
        git_commit=GIT_COMMIT,
        created_at_utc=VALIDATION_CREATED_AT_UTC,
    )
    return validation_path, validation


def _preprocess(
    result: SyntheticDatasetResult,
    validation_path: Path,
    tmp_path: Path,
) -> tuple[Path, PreprocessArtifact]:
    artifact_path = tmp_path / "artifacts" / "preprocess.json"
    artifact = preprocess_synthetic_dataset(
        result.manifest_path,
        validation_path,
        data_root=_data_root(tmp_path),
        output_root=tmp_path / "preprocessed",
        artifact_output_path=artifact_path,
        config_path=Path("configs/experiment/phase1.yaml"),
        git_commit=GIT_COMMIT,
        created_at_utc=PREPROCESS_CREATED_AT_UTC,
    )
    return artifact_path, artifact


def _prepared(tmp_path: Path) -> tuple[SyntheticManifest, Path, PreprocessArtifact]:
    result = _dataset(tmp_path)
    validation_path, _validation = _validate(result, tmp_path)
    preprocess_artifact_path, preprocess_artifact = _preprocess(result, validation_path, tmp_path)
    return result.manifest, preprocess_artifact_path, preprocess_artifact


def _run_dummy(
    preprocess_artifact_path: Path,
    tmp_path: Path,
    *,
    output_root: Path | None = None,
    artifact_output_path: Path | None = None,
    config_path: Path = Path("configs/experiment/phase1.yaml"),
) -> InferenceArtifact:
    return run_dummy_inference(
        preprocess_artifact_path,
        preprocessed_root=tmp_path / "preprocessed",
        output_root=output_root or tmp_path / "predictions",
        artifact_output_path=artifact_output_path or tmp_path / "artifacts" / "inference.json",
        config_path=config_path,
        git_commit=GIT_COMMIT,
        created_at_utc=INFERENCE_CREATED_AT_UTC,
    )


def _write_preprocess_payload(
    path: Path,
    artifact: PreprocessArtifact,
    **overrides: object,
) -> None:
    payload = cast(dict[str, Any], artifact_to_dict(artifact))
    payload.update(overrides)
    path.write_text(
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


def test_valid_dummy_inference_configuration() -> None:
    config = load_dummy_inference_config(Path("configs/experiment/phase1.yaml"))

    assert config == DummyInferenceConfig(
        seed=2718,
        threshold=0.5,
        prediction_dtype="uint8",
        method="dummy",
    )


def test_unknown_configuration_key_rejected(tmp_path: Path) -> None:
    config_path = _write_dummy_config(tmp_path / "phase1.yaml", unexpected=True)

    with pytest.raises(DummyInferenceConfigError, match="unknown"):
        load_dummy_inference_config(config_path)


def test_invalid_method_rejected(tmp_path: Path) -> None:
    config_path = _write_dummy_config(tmp_path / "phase1.yaml", method="copy-label")

    with pytest.raises(DummyInferenceConfigError, match="method"):
        load_dummy_inference_config(config_path)


def test_negative_seed_rejected(tmp_path: Path) -> None:
    config_path = _write_dummy_config(tmp_path / "phase1.yaml", seed=-1)

    with pytest.raises(DummyInferenceConfigError, match="seed"):
        load_dummy_inference_config(config_path)


@pytest.mark.parametrize("threshold", [-0.1, 1.1, math.nan, math.inf])
def test_invalid_threshold_rejected(tmp_path: Path, threshold: float) -> None:
    config_path = _write_dummy_config(tmp_path / "phase1.yaml", threshold=threshold)

    with pytest.raises(DummyInferenceConfigError, match="threshold"):
        load_dummy_inference_config(config_path)


def test_unsupported_dtype_rejected(tmp_path: Path) -> None:
    config_path = _write_dummy_config(tmp_path / "phase1.yaml", prediction_dtype="float32")

    with pytest.raises(DummyInferenceConfigError, match="prediction_dtype"):
        load_dummy_inference_config(config_path)


def test_valid_inference_succeeds_and_records_artifact_linkage(tmp_path: Path) -> None:
    _manifest, preprocess_artifact_path, preprocess_artifact = _prepared(tmp_path)

    artifact = _run_dummy(preprocess_artifact_path, tmp_path)
    persisted = artifact_from_json(
        (tmp_path / "artifacts" / "inference.json").read_text(encoding="utf-8"),
        InferenceArtifact,
    )

    assert persisted == artifact
    assert artifact.stage == "infer-dummy"
    assert artifact.prediction_case_ids == preprocess_artifact.output_case_ids
    assert artifact.config_hash == preprocess_artifact.config_hash
    assert artifact.manifest_hash == preprocess_artifact.manifest_hash
    assert artifact.method == "dummy"
    assert artifact.deterministic_seed == 2718
    assert artifact.threshold == 0.5
    assert artifact.prediction_dtype == "uint8"
    assert artifact.prediction_paths == (
        "predictions/synthetic-000.nii",
        "predictions/synthetic-001.nii",
        "predictions/synthetic-002.nii",
    )
    for relative_path, prediction_hash in zip(
        artifact.prediction_paths,
        artifact.prediction_hashes,
        strict=True,
    ):
        prediction_path = tmp_path / "predictions" / relative_path
        assert prediction_path.is_file()
        assert sha256_file(prediction_path) == prediction_hash


def test_prediction_outputs_are_binary_uint8_and_preserve_inputs_and_geometry(
    tmp_path: Path,
) -> None:
    _manifest, preprocess_artifact_path, preprocess_artifact = _prepared(tmp_path)
    preprocessed_root = tmp_path / "preprocessed"
    input_hashes = {
        relative_path: sha256_file(preprocessed_root / relative_path)
        for relative_path in [
            *preprocess_artifact.output_image_paths,
            *preprocess_artifact.output_label_paths,
        ]
    }

    artifact = _run_dummy(preprocess_artifact_path, tmp_path)

    for image_relative, prediction_relative in zip(
        preprocess_artifact.output_image_paths,
        artifact.prediction_paths,
        strict=True,
    ):
        input_image = _load_nifti(preprocessed_root / image_relative)
        prediction_path = tmp_path / "predictions" / prediction_relative
        prediction_image = _load_nifti(prediction_path)
        prediction_array = _array(prediction_path)

        assert prediction_path.suffix == ".nii"
        assert prediction_image.header.get_data_dtype() == np.dtype(np.uint8)
        assert prediction_array.shape == _array(preprocessed_root / image_relative).shape
        np.testing.assert_array_equal(prediction_image.affine, input_image.affine)
        assert set(np.unique(prediction_array).tolist()) <= {0, 1}

    assert input_hashes == {
        relative_path: sha256_file(preprocessed_root / relative_path)
        for relative_path in [
            *preprocess_artifact.output_image_paths,
            *preprocess_artifact.output_label_paths,
        ]
    }


def test_deterministic_predictions_across_independent_output_roots(tmp_path: Path) -> None:
    _manifest, preprocess_artifact_path, _preprocess_artifact = _prepared(tmp_path)

    first = _run_dummy(
        preprocess_artifact_path,
        tmp_path,
        output_root=tmp_path / "first-predictions",
        artifact_output_path=tmp_path / "first-artifacts" / "inference.json",
    )
    second = _run_dummy(
        preprocess_artifact_path,
        tmp_path,
        output_root=tmp_path / "second-predictions",
        artifact_output_path=tmp_path / "second-artifacts" / "inference.json",
    )

    assert first == second
    assert first.prediction_hashes == second.prediction_hashes
    for relative_path in first.prediction_paths:
        assert sha256_file(tmp_path / "first-predictions" / relative_path) == sha256_file(
            tmp_path / "second-predictions" / relative_path
        )


def test_different_seeds_change_at_least_one_prediction(tmp_path: Path) -> None:
    _manifest, preprocess_artifact_path, _preprocess_artifact = _prepared(tmp_path)
    alternate_config_path = _write_dummy_config(tmp_path / "alternate.yaml", seed=999)

    first = _run_dummy(
        preprocess_artifact_path,
        tmp_path,
        output_root=tmp_path / "first-predictions",
        artifact_output_path=tmp_path / "first-artifacts" / "inference.json",
    )
    second = _run_dummy(
        preprocess_artifact_path,
        tmp_path,
        output_root=tmp_path / "second-predictions",
        artifact_output_path=tmp_path / "second-artifacts" / "inference.json",
        config_path=alternate_config_path,
    )

    assert first.prediction_hashes != second.prediction_hashes


def test_no_label_file_is_read_during_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _manifest, preprocess_artifact_path, preprocess_artifact = _prepared(tmp_path)
    original_load = nib.load
    loaded_paths: list[str] = []

    def tracking_load(filename: str | Path, *args: Any, **kwargs: Any) -> Any:
        filename_text = str(filename)
        loaded_paths.append(filename_text)
        if "labels" in filename_text:
            msg = f"label file was read during dummy inference: {filename_text}"
            raise AssertionError(msg)
        return original_load(filename, *args, **kwargs)

    monkeypatch.setattr(nib, "load", tracking_load)

    _run_dummy(preprocess_artifact_path, tmp_path)

    assert len(loaded_paths) == len(preprocess_artifact.output_image_paths)
    assert all("labels" not in path for path in loaded_paths)


def test_malformed_preprocessing_artifact_rejected(tmp_path: Path) -> None:
    _manifest, _preprocess_artifact_path, _preprocess_artifact = _prepared(tmp_path)
    malformed_path = tmp_path / "artifacts" / "malformed-preprocess.json"
    malformed_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(DummyInferenceInputArtifactError, match="invalid preprocessing artifact"):
        _run_dummy(malformed_path, tmp_path)


def test_mismatched_case_path_lengths_rejected(tmp_path: Path) -> None:
    _manifest, preprocess_artifact_path, preprocess_artifact = _prepared(tmp_path)
    _write_preprocess_payload(
        preprocess_artifact_path,
        preprocess_artifact,
        output_image_paths=list(preprocess_artifact.output_image_paths[:-1]),
    )

    with pytest.raises(DummyInferenceInputArtifactError, match="equal lengths"):
        _run_dummy(preprocess_artifact_path, tmp_path)


def test_duplicate_case_id_rejected(tmp_path: Path) -> None:
    _manifest, preprocess_artifact_path, preprocess_artifact = _prepared(tmp_path)
    duplicate_case_ids = list(preprocess_artifact.output_case_ids)
    duplicate_case_ids[1] = duplicate_case_ids[0]
    _write_preprocess_payload(
        preprocess_artifact_path,
        preprocess_artifact,
        output_case_ids=duplicate_case_ids,
    )

    with pytest.raises(DummyInferenceInputArtifactError, match="output_case_ids"):
        _run_dummy(preprocess_artifact_path, tmp_path)


def test_duplicate_resolved_image_path_rejected(tmp_path: Path) -> None:
    _manifest, preprocess_artifact_path, preprocess_artifact = _prepared(tmp_path)
    duplicate_image_paths = list(preprocess_artifact.output_image_paths)
    duplicate_image_paths[1] = duplicate_image_paths[0]
    _write_preprocess_payload(
        preprocess_artifact_path,
        preprocess_artifact,
        output_image_paths=duplicate_image_paths,
    )

    with pytest.raises(DummyInferencePathError, match="duplicate resolved"):
        _run_dummy(preprocess_artifact_path, tmp_path)


def test_missing_image_rejected(tmp_path: Path) -> None:
    _manifest, preprocess_artifact_path, preprocess_artifact = _prepared(tmp_path)
    (tmp_path / "preprocessed" / preprocess_artifact.output_image_paths[0]).unlink()

    with pytest.raises(DummyInferenceImageError, match="missing"):
        _run_dummy(preprocess_artifact_path, tmp_path)


def test_absolute_path_rejected(tmp_path: Path) -> None:
    _manifest, preprocess_artifact_path, preprocess_artifact = _prepared(tmp_path)
    image_paths = list(preprocess_artifact.output_image_paths)
    image_paths[0] = "/absolute/escape.nii"
    _write_preprocess_payload(
        preprocess_artifact_path,
        preprocess_artifact,
        output_image_paths=image_paths,
    )

    with pytest.raises(DummyInferencePathError, match="relative"):
        _run_dummy(preprocess_artifact_path, tmp_path)


def test_parent_traversal_rejected(tmp_path: Path) -> None:
    _manifest, preprocess_artifact_path, preprocess_artifact = _prepared(tmp_path)
    image_paths = list(preprocess_artifact.output_image_paths)
    image_paths[0] = "../escape.nii"
    _write_preprocess_payload(
        preprocess_artifact_path,
        preprocess_artifact,
        output_image_paths=image_paths,
    )

    with pytest.raises(DummyInferencePathError, match="parent traversal"):
        _run_dummy(preprocess_artifact_path, tmp_path)


def test_symlink_escape_rejected_when_supported(tmp_path: Path) -> None:
    _manifest, preprocess_artifact_path, preprocess_artifact = _prepared(tmp_path)
    preprocessed_root = tmp_path / "preprocessed"
    outside_image = tmp_path / "outside.nii"
    outside_image.write_bytes(
        (preprocessed_root / preprocess_artifact.output_image_paths[0]).read_bytes()
    )
    symlink_path = preprocessed_root / "images" / "escape.nii"
    try:
        symlink_path.symlink_to(outside_image)
    except OSError as exc:
        pytest.skip(f"platform does not support this symlink test: {exc}")

    image_paths = list(preprocess_artifact.output_image_paths)
    image_paths[0] = "images/escape.nii"
    _write_preprocess_payload(
        preprocess_artifact_path,
        preprocess_artifact,
        output_image_paths=image_paths,
    )

    with pytest.raises(DummyInferencePathError, match="escapes preprocessed_root"):
        _run_dummy(preprocess_artifact_path, tmp_path)


def test_nonempty_output_root_rejected(tmp_path: Path) -> None:
    _manifest, preprocess_artifact_path, _preprocess_artifact = _prepared(tmp_path)
    output_root = tmp_path / "predictions"
    output_root.mkdir()
    (output_root / "existing.txt").write_text("occupied\n", encoding="utf-8")

    with pytest.raises(DummyInferenceOutputCollisionError, match="empty or nonexistent"):
        _run_dummy(preprocess_artifact_path, tmp_path, output_root=output_root)


def test_existing_artifact_output_rejected(tmp_path: Path) -> None:
    _manifest, preprocess_artifact_path, _preprocess_artifact = _prepared(tmp_path)
    artifact_output_path = tmp_path / "artifacts" / "inference.json"
    artifact_output_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(DummyInferenceOutputCollisionError, match="overwrite"):
        _run_dummy(
            preprocess_artifact_path,
            tmp_path,
            artifact_output_path=artifact_output_path,
        )


def test_output_root_inside_preprocessed_root_rejected(tmp_path: Path) -> None:
    _manifest, preprocess_artifact_path, _preprocess_artifact = _prepared(tmp_path)

    with pytest.raises(DummyInferencePathError, match="inside preprocessed_root"):
        _run_dummy(
            preprocess_artifact_path,
            tmp_path,
            output_root=tmp_path / "preprocessed" / "predictions",
        )


def test_failure_leaves_no_partial_output_artifact_or_staging(tmp_path: Path) -> None:
    _manifest, preprocess_artifact_path, preprocess_artifact = _prepared(tmp_path)
    output_root = tmp_path / "predictions"
    artifact_output_path = tmp_path / "artifacts" / "inference.json"
    preprocessed_root = tmp_path / "preprocessed"
    _corrupt_nifti_header(preprocessed_root / preprocess_artifact.output_image_paths[1])
    input_hashes = {
        relative_path: sha256_file(preprocessed_root / relative_path)
        for relative_path in [
            *preprocess_artifact.output_image_paths,
            *preprocess_artifact.output_label_paths,
        ]
    }

    with pytest.raises(DummyInferenceImageError):
        _run_dummy(
            preprocess_artifact_path,
            tmp_path,
            output_root=output_root,
            artifact_output_path=artifact_output_path,
        )

    assert not output_root.exists()
    assert not artifact_output_path.exists()
    assert not any(path.name.startswith(f".{output_root.name}.tmp-") for path in tmp_path.iterdir())
    assert input_hashes == {
        relative_path: sha256_file(preprocessed_root / relative_path)
        for relative_path in [
            *preprocess_artifact.output_image_paths,
            *preprocess_artifact.output_label_paths,
        ]
    }
