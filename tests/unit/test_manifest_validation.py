"""Unit tests for the Phase 1 synthetic manifest validation stage."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any, cast

import nibabel as nib
import numpy as np
import pytest

from protoem_ct.artifacts import (
    SyntheticManifest,
    ValidationArtifact,
    artifact_from_json,
    artifact_to_dict,
    hash_manifest,
)
from protoem_ct.data.manifest_validation import (
    ManifestValidationError,
    ManifestValidationManifestError,
    ManifestValidationNiftiError,
    ManifestValidationOutputError,
    ManifestValidationPathError,
    validate_synthetic_manifest,
)
from protoem_ct.data.synthetic import (
    SyntheticDataConfig,
    SyntheticDatasetResult,
    create_synthetic_dataset,
)

GIT_COMMIT = "f0b34b3"
CREATED_AT_UTC = "2026-07-25T01:00:00Z"
VALIDATION_CREATED_AT_UTC = "2026-07-25T01:05:00Z"


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


def _dataset(tmp_path: Path) -> SyntheticDatasetResult:
    return create_synthetic_dataset(
        _config(),
        output_root=tmp_path / "data-root",
        git_commit=GIT_COMMIT,
        created_at_utc=CREATED_AT_UTC,
    )


def _validate(result: SyntheticDatasetResult, output_path: Path) -> ValidationArtifact:
    return validate_synthetic_manifest(
        result.manifest_path,
        data_root=result.manifest_path.parents[3],
        output_path=output_path,
        git_commit=GIT_COMMIT,
        created_at_utc=VALIDATION_CREATED_AT_UTC,
    )


def _payload(result: SyntheticDatasetResult) -> dict[str, Any]:
    return cast(dict[str, Any], artifact_to_dict(result.manifest))


def _write_payload(
    manifest_path: Path,
    payload: dict[str, Any],
    *,
    recompute_manifest_hash: bool,
) -> None:
    if recompute_manifest_hash:
        payload["manifest_hash"] = "0" * 64
        hash_payload = {key: value for key, value in payload.items() if key != "manifest_hash"}
        payload["manifest_hash"] = hash_manifest(hash_payload)
    manifest_path.write_text(
        json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _rewrite_manifest(
    result: SyntheticDatasetResult,
    **overrides: object,
) -> SyntheticManifest:
    payload = _payload(result)
    payload.update(overrides)
    _write_payload(result.manifest_path, payload, recompute_manifest_hash=True)
    return artifact_from_json(result.manifest_path.read_text(encoding="utf-8"), SyntheticManifest)


def _data_root(result: SyntheticDatasetResult) -> Path:
    return result.manifest_path.parents[3]


def _write_label(path: Path, values: np.ndarray[Any, Any]) -> None:
    label = cast(Any, nib.load(str(path)))
    image = nib.Nifti1Image(values, label.affine)  # type: ignore[no-untyped-call]
    nib.save(image, str(path))


def test_valid_manifest_and_generated_dataset_are_accepted(tmp_path: Path) -> None:
    result = _dataset(tmp_path)

    artifact = _validate(result, tmp_path / "artifacts" / "validation.json")

    assert artifact.valid_case_count == 3
    assert artifact.invalid_case_count == 0
    assert artifact.validated_case_ids == result.manifest.case_ids


def test_validation_artifact_fields_are_correct(tmp_path: Path) -> None:
    result = _dataset(tmp_path)
    output_path = tmp_path / "artifacts" / "validation.json"

    artifact = _validate(result, output_path)
    persisted = artifact_from_json(output_path.read_text(encoding="utf-8"), ValidationArtifact)

    assert persisted == artifact
    assert artifact.schema_version == "1"
    assert artifact.stage == "validate"
    assert artifact.created_at_utc == VALIDATION_CREATED_AT_UTC
    assert artifact.git_commit == GIT_COMMIT
    assert artifact.config_hash == result.manifest.config_hash
    assert artifact.manifest_hash == result.manifest.manifest_hash


def test_validated_case_order_is_deterministic_manifest_order(tmp_path: Path) -> None:
    result = _dataset(tmp_path)
    case_ids = tuple(reversed(result.manifest.case_ids))
    image_paths = tuple(reversed(result.manifest.image_paths))
    label_paths = tuple(reversed(result.manifest.label_paths))
    manifest = _rewrite_manifest(
        result,
        case_ids=list(case_ids),
        image_paths=list(image_paths),
        label_paths=list(label_paths),
    )

    artifact = _validate(result, tmp_path / "artifacts" / "validation.json")

    assert artifact.validated_case_ids == manifest.case_ids


def test_artifact_json_is_stable_for_identical_explicit_inputs(tmp_path: Path) -> None:
    result = _dataset(tmp_path)
    first_path = tmp_path / "first" / "validation.json"
    second_path = tmp_path / "second" / "validation.json"

    _validate(result, first_path)
    _validate(result, second_path)

    assert first_path.read_text(encoding="utf-8") == second_path.read_text(encoding="utf-8")


def test_altered_manifest_hash_is_rejected(tmp_path: Path) -> None:
    result = _dataset(tmp_path)
    payload = _payload(result)
    payload["dataset_id"] = "altered"
    _write_payload(result.manifest_path, payload, recompute_manifest_hash=False)

    with pytest.raises(ManifestValidationManifestError, match="hash mismatch"):
        _validate(result, tmp_path / "artifacts" / "validation.json")


def test_incorrect_manifest_stage_is_rejected(tmp_path: Path) -> None:
    result = _dataset(tmp_path)
    payload = _payload(result)
    payload["stage"] = "preprocess"
    _write_payload(result.manifest_path, payload, recompute_manifest_hash=True)

    with pytest.raises(ManifestValidationManifestError, match="incorrect artifact stage"):
        _validate(result, tmp_path / "artifacts" / "validation.json")


def test_missing_image_is_rejected(tmp_path: Path) -> None:
    result = _dataset(tmp_path)
    (_data_root(result) / result.manifest.image_paths[0]).unlink()

    with pytest.raises(ManifestValidationPathError, match="image path"):
        _validate(result, tmp_path / "artifacts" / "validation.json")


def test_missing_label_is_rejected(tmp_path: Path) -> None:
    result = _dataset(tmp_path)
    (_data_root(result) / result.manifest.label_paths[0]).unlink()

    with pytest.raises(ManifestValidationPathError, match="label path"):
        _validate(result, tmp_path / "artifacts" / "validation.json")


def test_invalid_nifti_pair_is_rejected(tmp_path: Path) -> None:
    result = _dataset(tmp_path)
    label_path = _data_root(result) / result.manifest.label_paths[0]
    _write_label(label_path, np.full(result.manifest.shape, 2, dtype=np.uint8))

    with pytest.raises(ManifestValidationNiftiError, match="failed NIfTI validation"):
        _validate(result, tmp_path / "artifacts" / "validation.json")


def test_manifest_shape_mismatch_is_rejected(tmp_path: Path) -> None:
    result = _dataset(tmp_path)
    _rewrite_manifest(result, shape=[7, 8, 6])

    with pytest.raises(ManifestValidationNiftiError, match="shape"):
        _validate(result, tmp_path / "artifacts" / "validation.json")


def test_manifest_spacing_mismatch_is_rejected(tmp_path: Path) -> None:
    result = _dataset(tmp_path)
    _rewrite_manifest(result, spacing=[1.6, 1.5, 2.0])

    with pytest.raises(ManifestValidationNiftiError, match="voxel spacing"):
        _validate(result, tmp_path / "artifacts" / "validation.json")


def test_path_traversal_is_rejected(tmp_path: Path) -> None:
    result = _dataset(tmp_path)
    payload = _payload(result)
    payload["image_paths"][0] = "../escape.nii"
    _write_payload(result.manifest_path, payload, recompute_manifest_hash=True)

    with pytest.raises(ManifestValidationManifestError, match="parent traversal"):
        _validate(result, tmp_path / "artifacts" / "validation.json")


def test_absolute_manifest_path_is_rejected(tmp_path: Path) -> None:
    result = _dataset(tmp_path)
    payload = _payload(result)
    payload["image_paths"][0] = str(_data_root(result) / result.manifest.image_paths[0])
    _write_payload(result.manifest_path, payload, recompute_manifest_hash=True)

    with pytest.raises(ManifestValidationManifestError, match="relative"):
        _validate(result, tmp_path / "artifacts" / "validation.json")


def test_symlink_escape_is_rejected_where_supported(tmp_path: Path) -> None:
    result = _dataset(tmp_path)
    outside_image = tmp_path / "outside.nii"
    outside_image.write_bytes((_data_root(result) / result.manifest.image_paths[0]).read_bytes())
    symlink_path = _data_root(result) / "generated/phase1/data/images/escape.nii"
    try:
        symlink_path.symlink_to(outside_image)
    except OSError as exc:
        pytest.skip(f"platform does not support this symlink test: {exc}")
    payload = _payload(result)
    payload["image_paths"][0] = "generated/phase1/data/images/escape.nii"
    _write_payload(result.manifest_path, payload, recompute_manifest_hash=True)

    with pytest.raises(ManifestValidationPathError, match="escapes data_root"):
        _validate(result, tmp_path / "artifacts" / "validation.json")


def test_duplicate_resolved_image_path_is_rejected(tmp_path: Path) -> None:
    result = _dataset(tmp_path)
    payload = _payload(result)
    payload["image_paths"][1] = payload["image_paths"][0]
    _write_payload(result.manifest_path, payload, recompute_manifest_hash=True)

    with pytest.raises(ManifestValidationPathError, match="duplicate resolved image path"):
        _validate(result, tmp_path / "artifacts" / "validation.json")


def test_duplicate_resolved_label_path_is_rejected(tmp_path: Path) -> None:
    result = _dataset(tmp_path)
    payload = _payload(result)
    payload["label_paths"][1] = payload["label_paths"][0]
    _write_payload(result.manifest_path, payload, recompute_manifest_hash=True)

    with pytest.raises(ManifestValidationPathError, match="duplicate resolved label path"):
        _validate(result, tmp_path / "artifacts" / "validation.json")


def test_negative_affine_tolerance_is_rejected(tmp_path: Path) -> None:
    result = _dataset(tmp_path)

    with pytest.raises(ManifestValidationError, match="affine_tolerance"):
        validate_synthetic_manifest(
            result.manifest_path,
            data_root=_data_root(result),
            output_path=tmp_path / "artifacts" / "validation.json",
            git_commit=GIT_COMMIT,
            created_at_utc=VALIDATION_CREATED_AT_UTC,
            affine_tolerance=-1.0,
        )


def test_existing_output_file_is_rejected(tmp_path: Path) -> None:
    result = _dataset(tmp_path)
    output_path = tmp_path / "artifacts" / "validation.json"
    output_path.parent.mkdir()
    output_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ManifestValidationOutputError, match="overwrite"):
        _validate(result, output_path)


def test_failed_validation_leaves_no_partial_artifact(tmp_path: Path) -> None:
    result = _dataset(tmp_path)
    label_path = _data_root(result) / result.manifest.label_paths[0]
    _write_label(label_path, np.full(result.manifest.shape, 2, dtype=np.uint8))
    output_path = tmp_path / "artifacts" / "validation.json"

    with pytest.raises(ManifestValidationNiftiError):
        _validate(result, output_path)

    assert not output_path.exists()
