"""Unit tests for Phase 2 geometry and label QA."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest

from protoem_ct.artifacts import (
    DATASET_MANIFEST_TYPE,
    DEVELOPMENT_SPLIT_MANIFEST_TYPE,
    GEOMETRY_LABEL_QA_STAGE,
    PHASE2_DEVELOPMENT_COHORT_ROLE,
    PHASE2_SCHEMA_VERSION,
    DatasetCaseRecord,
    DatasetManifest,
    DevelopmentSplitManifest,
    GeometryLabelQaArtifact,
    GeometryLabelQaCaseRecord,
    Phase2ArtifactSerializationError,
    Phase2ArtifactValidationError,
    SplitAssignment,
    geometry_label_qa_hash_payload,
    hash_dataset_manifest,
    hash_development_split,
    hash_geometry_label_qa,
    phase2_artifact_from_json,
    phase2_artifact_to_json,
    sha256_file,
)
from protoem_ct.data import (
    ExistingGeometryLabelQaOutputError,
    GeometryLabelQaAssignmentError,
    GeometryLabelQaConfig,
    GeometryLabelQaHashMismatchError,
    GeometryLabelQaNiftiReadError,
    GeometryLabelQaSourceIntegrityError,
    InvalidGeometryLabelQaConfigError,
    UnsafeGeometryLabelQaOutputPathError,
    hash_geometry_label_qa_config,
    run_geometry_label_qa,
)
from protoem_ct.data import geometry_label_qa as qa_module

CREATED_AT_UTC = "2026-07-26T00:00:00Z"
GIT_COMMIT = "4b50df7"
NiftiArray = npt.NDArray[np.generic]
AffineArray = npt.NDArray[np.float64]
WriteNiftiFile = Callable[[Path, NiftiArray, AffineArray], Path]


def _config(**overrides: object) -> GeometryLabelQaConfig:
    values: dict[str, object] = {
        "allowed_label_values": (0, 1),
        "tumor_label_value": 1,
        "affine_tolerance": 1e-5,
    }
    values.update(overrides)
    return GeometryLabelQaConfig(**cast(Any, values))


def _write_case(
    root: Path,
    *,
    index: int,
    image_data: NiftiArray,
    label_data: NiftiArray,
    image_affine: AffineArray,
    label_affine: AffineArray,
    write_nifti_file: WriteNiftiFile,
) -> tuple[str, str]:
    image_relative = f"images/case-{index:04d}.nii.gz"
    label_relative = f"labels/case-{index:04d}.nii.gz"
    image_path = root / image_relative
    label_path = root / label_relative
    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    write_nifti_file(image_path, image_data, image_affine)
    write_nifti_file(label_path, label_data, label_affine)
    return image_relative, label_relative


def _case_record(
    root: Path,
    index: int,
    image_relative: str,
    label_relative: str,
) -> DatasetCaseRecord:
    return DatasetCaseRecord(
        anonymous_patient_id=f"anon-p{index:04d}",
        anonymous_case_id=f"anon-c{index:04d}",
        relative_image_path=image_relative,
        relative_label_path=label_relative,
        image_sha256=sha256_file(root / image_relative),
        label_sha256=sha256_file(root / label_relative),
        cohort_role=PHASE2_DEVELOPMENT_COHORT_ROLE,
    )


def _manifest(root: Path, cases: tuple[DatasetCaseRecord, ...]) -> DatasetManifest:
    without_hash = DatasetManifest(
        schema_version=PHASE2_SCHEMA_VERSION,
        manifest_type=DATASET_MANIFEST_TYPE,
        dataset_id="lits-development",
        cohort_role=PHASE2_DEVELOPMENT_COHORT_ROLE,
        adapter_name="lits_style",
        adapter_version="1",
        generated_at_utc="2026-07-25T00:00:00Z",
        git_commit=GIT_COMMIT,
        dataset_root_fingerprint="0" * 64,
        manifest_hash="0" * 64,
        case_count=len(cases),
        cases=cases,
    )
    return replace(without_hash, manifest_hash=hash_dataset_manifest(without_hash))


def _split(
    manifest: DatasetManifest,
    *,
    source_hash: str | None = None,
) -> DevelopmentSplitManifest:
    partitions = ("train", "validation", "internal_test")
    assignments = tuple(
        SplitAssignment(
            anonymous_patient_id=case.anonymous_patient_id,
            anonymous_case_id=case.anonymous_case_id,
            partition=partitions[(index - 1) % 3],
        )
        for index, case in enumerate(manifest.cases, start=1)
    )
    train = sum(1 for assignment in assignments if assignment.partition == "train")
    validation = sum(1 for assignment in assignments if assignment.partition == "validation")
    internal_test = sum(1 for assignment in assignments if assignment.partition == "internal_test")
    without_hash = DevelopmentSplitManifest(
        schema_version=PHASE2_SCHEMA_VERSION,
        manifest_type=DEVELOPMENT_SPLIT_MANIFEST_TYPE,
        generated_at_utc=CREATED_AT_UTC,
        git_commit=GIT_COMMIT,
        source_manifest_hash=source_hash or manifest.manifest_hash,
        split_policy_version="patient_hash_rank_v1",
        split_seed=1729,
        split_hash="0" * 64,
        assignments=assignments,
        train_patient_count=train,
        validation_patient_count=validation,
        internal_test_patient_count=internal_test,
        train_case_count=train,
        validation_case_count=validation,
        internal_test_case_count=internal_test,
    )
    return replace(without_hash, split_hash=hash_development_split(without_hash))


def _write_artifacts(
    tmp_path: Path,
    manifest: DatasetManifest,
    split: DevelopmentSplitManifest,
) -> tuple[Path, Path]:
    manifest_path = tmp_path / "artifacts" / "manifest.json"
    split_path = tmp_path / "artifacts" / "split.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(phase2_artifact_to_json(manifest), encoding="utf-8", newline="\n")
    split_path.write_text(phase2_artifact_to_json(split), encoding="utf-8", newline="\n")
    return manifest_path, split_path


def _fixture(
    tmp_path: Path,
    write_nifti_file: WriteNiftiFile,
    matching_affine: AffineArray,
    valid_ct_image: NiftiArray,
    valid_binary_tumor_mask: NiftiArray,
    *,
    case_count: int = 1,
) -> tuple[Path, DatasetManifest, DevelopmentSplitManifest, Path, Path]:
    root = tmp_path / "dataset"
    cases: list[DatasetCaseRecord] = []
    for index in range(1, case_count + 1):
        case_image = cast(NiftiArray, np.add(valid_ct_image, np.float32(index * 0.001)))
        image_relative, label_relative = _write_case(
            root,
            index=index,
            image_data=case_image,
            label_data=valid_binary_tumor_mask,
            image_affine=matching_affine,
            label_affine=matching_affine,
            write_nifti_file=write_nifti_file,
        )
        cases.append(_case_record(root, index, image_relative, label_relative))
    manifest = _manifest(root, tuple(cases))
    split = _split(manifest)
    manifest_path, split_path = _write_artifacts(tmp_path, manifest, split)
    return root, manifest, split, manifest_path, split_path


def _run(
    manifest_path: Path,
    split_path: Path,
    root: Path,
    output: Path,
    *,
    config: GeometryLabelQaConfig | None = None,
) -> GeometryLabelQaArtifact:
    return run_geometry_label_qa(
        manifest_path,
        split_path,
        dataset_root=root,
        output_path=output,
        config=config or _config(),
        git_commit=GIT_COMMIT,
        created_at_utc=CREATED_AT_UTC,
    )


def test_config_validation_immutability_and_hash() -> None:
    config = _config()
    assert hash_geometry_label_qa_config(config) == hash_geometry_label_qa_config(_config())

    with pytest.raises(InvalidGeometryLabelQaConfigError):
        _config(allowed_label_values=())
    with pytest.raises(InvalidGeometryLabelQaConfigError):
        _config(allowed_label_values=(0, 1, 1))
    with pytest.raises(InvalidGeometryLabelQaConfigError):
        _config(allowed_label_values=(1, 0))
    with pytest.raises(InvalidGeometryLabelQaConfigError):
        _config(allowed_label_values=(0, 1.5))
    with pytest.raises(InvalidGeometryLabelQaConfigError):
        _config(tumor_label_value=2)
    for tolerance in (0.0, -1.0, math.nan, math.inf):
        with pytest.raises(InvalidGeometryLabelQaConfigError):
            _config(affine_tolerance=tolerance)
    with pytest.raises(FrozenInstanceError):
        cast(Any, config).affine_tolerance = 1.0


def test_valid_geometry_label_qa_records_expected_fields(
    tmp_path: Path,
    write_nifti_file: WriteNiftiFile,
    matching_affine: AffineArray,
    valid_ct_image: NiftiArray,
    valid_binary_tumor_mask: NiftiArray,
) -> None:
    root, manifest, split, manifest_path, split_path = _fixture(
        tmp_path,
        write_nifti_file,
        matching_affine,
        valid_ct_image,
        valid_binary_tumor_mask,
    )
    image_bytes = (root / manifest.cases[0].relative_image_path).read_bytes()
    label_bytes = (root / manifest.cases[0].relative_label_path).read_bytes()

    artifact = _run(manifest_path, split_path, root, tmp_path / "out" / "qa.json")
    loaded = phase2_artifact_from_json(
        (tmp_path / "out" / "qa.json").read_text(encoding="utf-8"),
        GeometryLabelQaArtifact,
    )

    assert loaded == artifact
    assert artifact.stage == GEOMETRY_LABEL_QA_STAGE
    assert artifact.manifest_hash == manifest.manifest_hash
    assert artifact.split_hash == split.split_hash
    assert artifact.qa_artifact_hash == hash_geometry_label_qa(artifact)
    assert "qa_artifact_hash" not in geometry_label_qa_hash_payload(artifact)
    assert artifact.case_count == 1
    assert artifact.passed_case_count == 1
    record = artifact.case_records[0]
    assert record.partition == "train"
    assert record.dimensionality == 3
    assert record.image_shape == (2, 3, 4)
    assert record.label_shape == (2, 3, 4)
    assert record.image_dtype == "float32"
    assert record.label_dtype == "uint8"
    assert record.image_orientation == ("R", "A", "S")
    assert record.label_orientation == ("R", "A", "S")
    assert record.image_spacing == (1.0, 1.5, 2.0)
    assert record.label_spacing == (1.0, 1.5, 2.0)
    assert record.observed_label_values == (0, 1)
    assert record.allowed_label_values == (0, 1)
    assert record.tumor_voxel_count == 2
    assert not record.empty_tumor
    assert record.qa_passed
    assert record.failure_reasons == ()
    assert (root / manifest.cases[0].relative_image_path).read_bytes() == image_bytes
    assert (root / manifest.cases[0].relative_label_path).read_bytes() == label_bytes


def test_empty_tumor_may_pass(
    tmp_path: Path,
    write_nifti_file: WriteNiftiFile,
    matching_affine: AffineArray,
    valid_ct_image: NiftiArray,
) -> None:
    empty_label = np.zeros((2, 3, 4), dtype=np.uint8)
    root = tmp_path / "dataset"
    image_relative, label_relative = _write_case(
        root,
        index=1,
        image_data=valid_ct_image,
        label_data=empty_label,
        image_affine=matching_affine,
        label_affine=matching_affine,
        write_nifti_file=write_nifti_file,
    )
    manifest = _manifest(root, (_case_record(root, 1, image_relative, label_relative),))
    split = _split(manifest)
    manifest_path, split_path = _write_artifacts(tmp_path, manifest, split)

    artifact = _run(manifest_path, split_path, root, tmp_path / "qa.json")

    assert artifact.passed_case_count == 1
    assert artifact.case_records[0].empty_tumor
    assert artifact.case_records[0].tumor_voxel_count == 0


def test_case_level_failures_are_recorded_and_do_not_abort(
    tmp_path: Path,
    write_nifti_file: WriteNiftiFile,
    matching_affine: AffineArray,
    valid_ct_image: NiftiArray,
    valid_binary_tumor_mask: NiftiArray,
) -> None:
    root = tmp_path / "dataset"
    first = _write_case(
        root,
        index=1,
        image_data=valid_ct_image,
        label_data=valid_binary_tumor_mask,
        image_affine=matching_affine,
        label_affine=matching_affine,
        write_nifti_file=write_nifti_file,
    )
    bad_image = np.array([[0.0, np.nan], [np.inf, 1.0]], dtype=np.float32)
    bad_label = np.array([[0.0, 0.5], [2.0, np.nan]], dtype=np.float32)
    bad_label_affine = matching_affine.copy()
    bad_label_affine[0, 0] = 1.25
    second = _write_case(
        root,
        index=2,
        image_data=bad_image,
        label_data=bad_label,
        image_affine=matching_affine,
        label_affine=bad_label_affine,
        write_nifti_file=write_nifti_file,
    )
    cases = (
        _case_record(root, 1, first[0], first[1]),
        _case_record(root, 2, second[0], second[1]),
    )
    manifest = _manifest(root, cases)
    split = _split(manifest)
    manifest_path, split_path = _write_artifacts(tmp_path, manifest, split)

    artifact = _run(manifest_path, split_path, root, tmp_path / "qa.json")

    assert artifact.case_count == 2
    assert artifact.passed_case_count == 1
    assert artifact.failed_case_count == 1
    failed = artifact.case_records[1]
    assert failed.failure_reasons == (
        "image_not_3d",
        "label_not_3d",
        "image_nonfinite",
        "label_nonfinite",
        "label_noninteger",
        "affine_mismatch",
        "spacing_mismatch",
    )
    assert failed.observed_label_values == (0, 0.5, 2)
    assert not failed.qa_passed


def test_disallowed_label_shape_and_affine_failures_are_recorded(
    tmp_path: Path,
    write_nifti_file: WriteNiftiFile,
    matching_affine: AffineArray,
    valid_ct_image: NiftiArray,
) -> None:
    root = tmp_path / "dataset"
    label = np.zeros((2, 3, 5), dtype=np.uint8)
    label[0, 0, 0] = 2
    label_affine = matching_affine.copy()
    label_affine[1, 3] = 10.0
    image_relative, label_relative = _write_case(
        root,
        index=1,
        image_data=valid_ct_image,
        label_data=label,
        image_affine=matching_affine,
        label_affine=label_affine,
        write_nifti_file=write_nifti_file,
    )
    manifest = _manifest(root, (_case_record(root, 1, image_relative, label_relative),))
    split = _split(manifest)
    manifest_path, split_path = _write_artifacts(tmp_path, manifest, split)

    artifact = _run(manifest_path, split_path, root, tmp_path / "qa.json")

    assert artifact.failed_case_count == 1
    assert artifact.case_records[0].failure_reasons == (
        "label_value_not_allowed",
        "shape_mismatch",
        "affine_mismatch",
    )


def test_artifact_serialization_contract_rejects_bad_values() -> None:
    record = GeometryLabelQaCaseRecord(
        anonymous_patient_id="anon-p0001",
        anonymous_case_id="anon-c0001",
        partition="train",
        dimensionality=3,
        image_shape=(2, 3, 4),
        label_shape=(2, 3, 4),
        image_dtype="float32",
        label_dtype="uint8",
        image_affine=((1.0, 0.0, 0.0, 0.0),) * 4,
        label_affine=((1.0, 0.0, 0.0, 0.0),) * 4,
        image_orientation=("R", "A", "S"),
        label_orientation=("R", "A", "S"),
        image_spacing=(1.0, 1.0, 1.0),
        label_spacing=(1.0, 1.0, 1.0),
        image_finite=True,
        label_finite=True,
        observed_label_values=(0, 1),
        allowed_label_values=(0, 1),
        image_label_shape_match=True,
        image_label_affine_match=True,
        tumor_label_value=1,
        tumor_voxel_count=0,
        empty_tumor=True,
        qa_passed=True,
        failure_reasons=(),
    )
    artifact = GeometryLabelQaArtifact(
        schema_version=PHASE2_SCHEMA_VERSION,
        stage=GEOMETRY_LABEL_QA_STAGE,
        created_at_utc=CREATED_AT_UTC,
        git_commit=GIT_COMMIT,
        config_hash="0" * 64,
        manifest_hash="1" * 64,
        split_hash="2" * 64,
        case_count=1,
        passed_case_count=1,
        failed_case_count=0,
        case_records=(record,),
        qa_artifact_hash="3" * 64,
    )
    text = phase2_artifact_to_json(artifact)

    assert phase2_artifact_from_json(text, GeometryLabelQaArtifact) == artifact
    assert phase2_artifact_to_json(artifact) == text
    raw = json.loads(text)
    raw["unexpected"] = True
    with pytest.raises(Phase2ArtifactValidationError):
        phase2_artifact_from_json(json.dumps(raw), GeometryLabelQaArtifact)
    with pytest.raises(Phase2ArtifactSerializationError):
        phase2_artifact_from_json(text.replace("1.0", "NaN", 1), GeometryLabelQaArtifact)


def test_integrity_and_structural_failures_leave_no_output(
    tmp_path: Path,
    write_nifti_file: WriteNiftiFile,
    matching_affine: AffineArray,
    valid_ct_image: NiftiArray,
    valid_binary_tumor_mask: NiftiArray,
) -> None:
    root, manifest, split, manifest_path, split_path = _fixture(
        tmp_path,
        write_nifti_file,
        matching_affine,
        valid_ct_image,
        valid_binary_tumor_mask,
    )
    output = tmp_path / "out" / "qa.json"

    tampered_manifest = replace(manifest, manifest_hash="f" * 64)
    manifest_path.write_text(phase2_artifact_to_json(tampered_manifest), encoding="utf-8")
    with pytest.raises(GeometryLabelQaHashMismatchError):
        _run(manifest_path, split_path, root, output)
    assert not output.exists()

    manifest_path.write_text(phase2_artifact_to_json(manifest), encoding="utf-8")
    tampered_split = replace(split, split_hash="f" * 64)
    split_path.write_text(phase2_artifact_to_json(tampered_split), encoding="utf-8")
    with pytest.raises(GeometryLabelQaHashMismatchError):
        _run(manifest_path, split_path, root, output)
    assert not output.exists()

    wrong_source_split = _split(manifest, source_hash="e" * 64)
    split_path.write_text(phase2_artifact_to_json(wrong_source_split), encoding="utf-8")
    with pytest.raises(GeometryLabelQaAssignmentError):
        _run(manifest_path, split_path, root, output)
    assert not output.exists()

    split_path.write_text(phase2_artifact_to_json(split), encoding="utf-8")
    (root / manifest.cases[0].relative_image_path).write_bytes(b"changed")
    with pytest.raises(GeometryLabelQaSourceIntegrityError):
        _run(manifest_path, split_path, root, output)
    assert not output.exists()


def test_source_path_nifti_and_output_safety_failures(
    tmp_path: Path,
    write_nifti_file: WriteNiftiFile,
    matching_affine: AffineArray,
    valid_ct_image: NiftiArray,
    valid_binary_tumor_mask: NiftiArray,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, manifest, split, manifest_path, split_path = _fixture(
        tmp_path,
        write_nifti_file,
        matching_affine,
        valid_ct_image,
        valid_binary_tumor_mask,
    )
    output = tmp_path / "out" / "qa.json"

    existing = tmp_path / "existing.json"
    existing.write_text("occupied\n", encoding="utf-8")
    with pytest.raises(ExistingGeometryLabelQaOutputError):
        _run(manifest_path, split_path, root, existing)
    with pytest.raises(UnsafeGeometryLabelQaOutputPathError):
        _run(manifest_path, split_path, root, root / "qa.json")
    with pytest.raises(UnsafeGeometryLabelQaOutputPathError):
        _run(manifest_path, split_path, root, Path("relative.json"))
    parent_file = tmp_path / "parent-file"
    parent_file.write_text("not a dir\n", encoding="utf-8")
    with pytest.raises(UnsafeGeometryLabelQaOutputPathError):
        _run(manifest_path, split_path, root, parent_file / "qa.json")

    image_path = root / manifest.cases[0].relative_image_path
    image_path.unlink()
    with pytest.raises(GeometryLabelQaSourceIntegrityError):
        _run(manifest_path, split_path, root, output)
    assert not output.exists()

    write_nifti_file(image_path, valid_ct_image, matching_affine)
    with pytest.raises(Phase2ArtifactValidationError):
        bad_case = replace(
            manifest.cases[0],
            relative_image_path="../escape.nii.gz",
            image_sha256=sha256_file(image_path),
        )
        bad_manifest = replace(manifest, cases=(bad_case,))
        manifest_path.write_text(phase2_artifact_to_json(bad_manifest), encoding="utf-8")

    manifest_path.write_text(phase2_artifact_to_json(manifest), encoding="utf-8")
    unreadable = root / manifest.cases[0].relative_image_path
    unreadable.write_bytes(b"not-nifti")
    altered_case = replace(manifest.cases[0], image_sha256=sha256_file(unreadable))
    altered_manifest = _manifest(root, (altered_case,))
    manifest_path.write_text(phase2_artifact_to_json(altered_manifest), encoding="utf-8")
    altered_split = _split(altered_manifest)
    split_path.write_text(phase2_artifact_to_json(altered_split), encoding="utf-8")
    with pytest.raises(GeometryLabelQaNiftiReadError):
        _run(manifest_path, split_path, root, output)
    assert not output.exists()

    def _fail_serialize(artifact: object) -> str:
        raise Phase2ArtifactSerializationError("synthetic serialization failure")

    root, manifest, split, manifest_path, split_path = _fixture(
        tmp_path / "fresh",
        write_nifti_file,
        matching_affine,
        valid_ct_image,
        valid_binary_tumor_mask,
    )
    output = tmp_path / "fresh-out" / "qa.json"
    monkeypatch.setattr(qa_module, "phase2_artifact_to_json", _fail_serialize)
    with pytest.raises(qa_module.GeometryLabelQaPublicationError):
        _run(manifest_path, split_path, root, output)
    assert not output.exists()
    assert not (output.parent / ".qa.json.tmp").exists()


def test_current_working_directory_and_output_location_do_not_affect_json(
    tmp_path: Path,
    write_nifti_file: WriteNiftiFile,
    matching_affine: AffineArray,
    valid_ct_image: NiftiArray,
    valid_binary_tumor_mask: NiftiArray,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _manifest_value, _split_value, manifest_path, split_path = _fixture(
        tmp_path,
        write_nifti_file,
        matching_affine,
        valid_ct_image,
        valid_binary_tumor_mask,
        case_count=3,
    )
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)

    first = _run(manifest_path, split_path, root, tmp_path / "first" / "qa.json")
    second = _run(manifest_path, split_path, root, tmp_path / "second" / "qa.json")

    assert phase2_artifact_to_json(first) == phase2_artifact_to_json(second)
    json_text = phase2_artifact_to_json(first)
    for forbidden in (
        str(root),
        str(tmp_path),
        "images/",
        "labels/",
        "case-",
        "source_case_key",
        "dataset_root",
        "/Users/",
        "/private/",
    ):
        assert forbidden not in json_text
