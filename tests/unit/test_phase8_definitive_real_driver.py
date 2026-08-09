"""Unit tests for the Phase 8 definitive real-training driver (Package C).

All tests operate on small, synthetic, in-memory / ``tmp_path`` fixtures
only. No real dataset, checkpoint, or ``/Volumes`` path is referenced
anywhere in this file, and no real 500-step definitive training is executed
-- orchestration-level tests monkeypatch the heavy
:mod:`protoem_ct.external.definitive_pipeline` training/validation entry
points with fast, pure-Python fakes, mirroring the approach in
``tests/unit/test_phase8_definitive_training_orchestration.py``.
"""

from __future__ import annotations

import contextlib
import hashlib
import multiprocessing
import time
from pathlib import Path
from typing import Any, cast

import nibabel as nib
import numpy as np
import pytest

from protoem_ct.artifacts import (
    DatasetCaseRecord,
    DatasetManifest,
    DevelopmentSplitManifest,
    LesionComponentCaseRecord,
    LesionComponentsArtifact,
    LesionSummaryRecord,
    SplitAssignment,
    hash_dataset_manifest,
    hash_development_split,
    hash_lesion_components,
)
from protoem_ct.artifacts.hashing import sha256_file
from protoem_ct.artifacts.phase2_schemas import phase2_artifact_to_json
from protoem_ct.external import definitive_pipeline as pipeline
from protoem_ct.external import definitive_real_training_driver as driver

_FAKE_SHA256 = "a" * 64
_FAKE_GIT_COMMIT = "b" * 40


# ---------------------------------------------------------------------------
# Synthetic manifest/split/lesion-components fixture builders
# ---------------------------------------------------------------------------


def _deterministic_sha256(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def _case_record(index: int) -> DatasetCaseRecord:
    case_id = f"case-{index:04d}"
    return DatasetCaseRecord(
        anonymous_patient_id=f"patient-{index:04d}",
        anonymous_case_id=case_id,
        relative_image_path=f"imagesTr/{case_id}.nii.gz",
        relative_label_path=f"labelsTr/{case_id}.nii.gz",
        image_sha256=_deterministic_sha256("image", case_id),
        label_sha256=_deterministic_sha256("label", case_id),
        cohort_role="development",
    )


def _build_manifest(case_count: int) -> DatasetManifest:
    cases = tuple(_case_record(i) for i in range(case_count))
    draft = DatasetManifest(
        schema_version="2",
        manifest_type="phase2-dataset-manifest",
        dataset_id="synthetic-test-dataset",
        cohort_role="development",
        adapter_name="synthetic",
        adapter_version="v1",
        generated_at_utc="2026-01-01T00:00:00Z",
        git_commit=_FAKE_GIT_COMMIT[:40],
        dataset_root_fingerprint=_FAKE_SHA256,
        manifest_hash="0" * 64,
        case_count=case_count,
        cases=cases,
    )
    manifest_hash = hash_dataset_manifest(draft)
    return DatasetManifest(
        schema_version=draft.schema_version,
        manifest_type=draft.manifest_type,
        dataset_id=draft.dataset_id,
        cohort_role=draft.cohort_role,
        adapter_name=draft.adapter_name,
        adapter_version=draft.adapter_version,
        generated_at_utc=draft.generated_at_utc,
        git_commit=draft.git_commit,
        dataset_root_fingerprint=draft.dataset_root_fingerprint,
        manifest_hash=manifest_hash,
        case_count=draft.case_count,
        cases=draft.cases,
    )


def _build_split(
    *, train_count: int, validation_count: int, internal_test_count: int = 0
) -> DevelopmentSplitManifest:
    assignments: list[SplitAssignment] = []
    for i in range(train_count):
        assignments.append(
            SplitAssignment(
                anonymous_patient_id=f"patient-{i:04d}",
                anonymous_case_id=f"case-{i:04d}",
                partition="train",
            )
        )
    for i in range(train_count, train_count + validation_count):
        assignments.append(
            SplitAssignment(
                anonymous_patient_id=f"patient-{i:04d}",
                anonymous_case_id=f"case-{i:04d}",
                partition="validation",
            )
        )
    for i in range(
        train_count + validation_count, train_count + validation_count + internal_test_count
    ):
        assignments.append(
            SplitAssignment(
                anonymous_patient_id=f"patient-{i:04d}",
                anonymous_case_id=f"case-{i:04d}",
                partition="internal_test",
            )
        )
    sorted_assignments = tuple(
        sorted(assignments, key=lambda a: (a.anonymous_patient_id, a.anonymous_case_id))
    )
    draft = DevelopmentSplitManifest(
        schema_version="2",
        manifest_type="phase2-development-split-manifest",
        generated_at_utc="2026-01-01T00:00:00Z",
        git_commit=_FAKE_GIT_COMMIT[:40],
        source_manifest_hash=_FAKE_SHA256,
        split_policy_version="v1",
        split_seed=1729,
        split_hash="0" * 64,
        assignments=sorted_assignments,
        train_patient_count=train_count,
        validation_patient_count=validation_count,
        internal_test_patient_count=internal_test_count,
        train_case_count=train_count,
        validation_case_count=validation_count,
        internal_test_case_count=internal_test_count,
    )
    split_hash = hash_development_split(draft)
    return DevelopmentSplitManifest(
        schema_version=draft.schema_version,
        manifest_type=draft.manifest_type,
        generated_at_utc=draft.generated_at_utc,
        git_commit=draft.git_commit,
        source_manifest_hash=draft.source_manifest_hash,
        split_policy_version=draft.split_policy_version,
        split_seed=draft.split_seed,
        split_hash=split_hash,
        assignments=draft.assignments,
        train_patient_count=draft.train_patient_count,
        validation_patient_count=draft.validation_patient_count,
        internal_test_patient_count=draft.internal_test_patient_count,
        train_case_count=draft.train_case_count,
        validation_case_count=draft.validation_case_count,
        internal_test_case_count=draft.internal_test_case_count,
    )


def _build_lesion_components(
    *, manifest: DatasetManifest, split: DevelopmentSplitManifest, positive_case_ids: set[str]
) -> LesionComponentsArtifact:
    partition_by_case_id = {a.anonymous_case_id: a.partition for a in split.assignments}
    records: list[LesionComponentCaseRecord] = []
    for case in manifest.cases:
        partition = partition_by_case_id.get(case.anonymous_case_id, "train")
        is_positive = case.anonymous_case_id in positive_case_ids
        records.append(
            LesionComponentCaseRecord(
                anonymous_patient_id=case.anonymous_patient_id,
                anonymous_case_id=case.anonymous_case_id,
                partition=partition,
                analysis_performed=True,
                connectivity=26,
                tumor_label_value=2,
                voxel_volume_mm3=1.0,
                tumor_voxel_count=10 if is_positive else 0,
                tumor_physical_volume_mm3=10.0 if is_positive else 0.0,
                lesion_count=1 if is_positive else 0,
                lesions=(
                    (LesionSummaryRecord(lesion_index=1, voxel_count=10, physical_volume_mm3=10.0),)
                    if is_positive
                    else ()
                ),
                qa_passed=True,
                failure_reasons=(),
            )
        )
    sorted_records = tuple(
        sorted(records, key=lambda r: (r.anonymous_patient_id, r.anonymous_case_id))
    )
    draft = LesionComponentsArtifact(
        schema_version="2",
        stage="lesion_components",
        created_at_utc="2026-01-01T00:00:00Z",
        git_commit=_FAKE_GIT_COMMIT[:40],
        config_hash=_FAKE_SHA256,
        manifest_hash=manifest.manifest_hash,
        split_hash=split.split_hash,
        geometry_qa_artifact_hash=_FAKE_SHA256,
        connectivity=26,
        case_count=len(sorted_records),
        analyzed_case_count=len(sorted_records),
        skipped_case_count=0,
        case_records=sorted_records,
        lesion_artifact_hash="0" * 64,
    )
    lesion_artifact_hash = hash_lesion_components(draft)
    return LesionComponentsArtifact(
        schema_version=draft.schema_version,
        stage=draft.stage,
        created_at_utc=draft.created_at_utc,
        git_commit=draft.git_commit,
        config_hash=draft.config_hash,
        manifest_hash=draft.manifest_hash,
        split_hash=draft.split_hash,
        geometry_qa_artifact_hash=draft.geometry_qa_artifact_hash,
        connectivity=draft.connectivity,
        case_count=draft.case_count,
        analyzed_case_count=draft.analyzed_case_count,
        skipped_case_count=draft.skipped_case_count,
        case_records=draft.case_records,
        lesion_artifact_hash=lesion_artifact_hash,
    )


def _write_artifact_and_hash(artifact: Any, path: Path) -> str:
    phase2_artifact_to_json(artifact, path)
    return sha256_file(path)


def _write_metadata_fixture(
    tmp_path: Path,
    *,
    train_count: int,
    validation_count: int,
    internal_test_count: int = 0,
    positive_case_ids: set[str] | None = None,
) -> tuple[Path, Path, Path, str, str, str, str, str]:
    total = train_count + validation_count + internal_test_count
    manifest = _build_manifest(total)
    split = _build_split(
        train_count=train_count,
        validation_count=validation_count,
        internal_test_count=internal_test_count,
    )
    if positive_case_ids is None:
        positive_case_ids = {"case-0000"}
    lesion_components = _build_lesion_components(
        manifest=manifest, split=split, positive_case_ids=positive_case_ids
    )

    manifest_path = tmp_path / "manifest.json"
    split_path = tmp_path / "split.json"
    lesion_path = tmp_path / "lesion_components.json"
    manifest_sha256 = _write_artifact_and_hash(manifest, manifest_path)
    split_sha256 = _write_artifact_and_hash(split, split_path)
    lesion_sha256 = _write_artifact_and_hash(lesion_components, lesion_path)
    return (
        manifest_path,
        split_path,
        lesion_path,
        manifest_sha256,
        split_sha256,
        lesion_sha256,
        manifest.manifest_hash,
        split.split_hash,
    )


@pytest.fixture(autouse=True)
def _pin_required_hashes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure every test starts from a clean, test-local set of required hashes."""
    monkeypatch.setattr(driver, "REQUIRED_MANIFEST_SHA256", "f" * 64)
    monkeypatch.setattr(driver, "REQUIRED_SPLIT_SHA256", "f" * 64)
    monkeypatch.setattr(driver, "REQUIRED_MANIFEST_EMBEDDED_HASH", "f" * 64)
    monkeypatch.setattr(driver, "REQUIRED_SPLIT_EMBEDDED_HASH", "f" * 64)


def _pin(
    monkeypatch: pytest.MonkeyPatch,
    *,
    manifest_sha256: str,
    split_sha256: str,
    manifest_embedded_hash: str | None = None,
    split_embedded_hash: str | None = None,
) -> None:
    monkeypatch.setattr(driver, "REQUIRED_MANIFEST_SHA256", manifest_sha256)
    monkeypatch.setattr(driver, "REQUIRED_SPLIT_SHA256", split_sha256)
    if manifest_embedded_hash is not None:
        monkeypatch.setattr(driver, "REQUIRED_MANIFEST_EMBEDDED_HASH", manifest_embedded_hash)
    if split_embedded_hash is not None:
        monkeypatch.setattr(driver, "REQUIRED_SPLIT_EMBEDDED_HASH", split_embedded_hash)


# ---------------------------------------------------------------------------
# A. Metadata verification and case-ID selection
# ---------------------------------------------------------------------------


def test_load_and_verify_metadata_succeeds_and_selects_positive_case_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (
        manifest_path,
        split_path,
        lesion_path,
        manifest_sha256,
        split_sha256,
        lesion_sha256,
        manifest_embedded_hash,
        split_embedded_hash,
    ) = _write_metadata_fixture(
        tmp_path,
        train_count=4,
        validation_count=2,
        internal_test_count=1,
        positive_case_ids={"case-0000", "case-0002"},
    )
    _pin(
        monkeypatch,
        manifest_sha256=manifest_sha256,
        split_sha256=split_sha256,
        manifest_embedded_hash=manifest_embedded_hash,
        split_embedded_hash=split_embedded_hash,
    )
    monkeypatch.setattr(driver, "REQUIRED_TRAIN_CASE_COUNT", 4)
    monkeypatch.setattr(driver, "REQUIRED_VALIDATION_CASE_COUNT", 2)

    metadata = driver.load_and_verify_definitive_real_training_metadata(
        manifest_path=manifest_path,
        split_path=split_path,
        lesion_components_path=lesion_path,
        expected_lesion_components_sha256=lesion_sha256,
    )

    assert len(metadata.train_case_references) == 4
    assert len(metadata.validation_case_references) == 2
    assert metadata.positive_case_ids == frozenset({"case-0000", "case-0002"})
    assert "case-0006" in metadata.internal_test_case_ids


def test_load_and_verify_metadata_rejects_manifest_hash_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (
        manifest_path,
        split_path,
        lesion_path,
        manifest_sha256,
        split_sha256,
        lesion_sha256,
        manifest_embedded_hash,
        split_embedded_hash,
    ) = _write_metadata_fixture(tmp_path, train_count=2, validation_count=2)
    _pin(monkeypatch, manifest_sha256="0" * 64, split_sha256=split_sha256)

    with pytest.raises(driver.Phase8DefinitiveRealTrainingPrerequisitesError):
        driver.load_and_verify_definitive_real_training_metadata(
            manifest_path=manifest_path,
            split_path=split_path,
            lesion_components_path=lesion_path,
            expected_lesion_components_sha256=lesion_sha256,
        )


def test_load_and_verify_metadata_rejects_split_hash_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (
        manifest_path,
        split_path,
        lesion_path,
        manifest_sha256,
        split_sha256,
        lesion_sha256,
        manifest_embedded_hash,
        split_embedded_hash,
    ) = _write_metadata_fixture(tmp_path, train_count=2, validation_count=2)
    _pin(monkeypatch, manifest_sha256=manifest_sha256, split_sha256="0" * 64)

    with pytest.raises(driver.Phase8DefinitiveRealTrainingPrerequisitesError):
        driver.load_and_verify_definitive_real_training_metadata(
            manifest_path=manifest_path,
            split_path=split_path,
            lesion_components_path=lesion_path,
            expected_lesion_components_sha256=lesion_sha256,
        )


def test_load_and_verify_metadata_rejects_lesion_components_hash_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (
        manifest_path,
        split_path,
        lesion_path,
        manifest_sha256,
        split_sha256,
        _lesion_sha256,
        manifest_embedded_hash,
        split_embedded_hash,
    ) = _write_metadata_fixture(tmp_path, train_count=2, validation_count=2)
    _pin(
        monkeypatch,
        manifest_sha256=manifest_sha256,
        split_sha256=split_sha256,
        manifest_embedded_hash=manifest_embedded_hash,
        split_embedded_hash=split_embedded_hash,
    )

    with pytest.raises(driver.Phase8DefinitiveRealTrainingPrerequisitesError):
        driver.load_and_verify_definitive_real_training_metadata(
            manifest_path=manifest_path,
            split_path=split_path,
            lesion_components_path=lesion_path,
            expected_lesion_components_sha256="0" * 64,
        )


@pytest.mark.parametrize("train_count,validation_count", [(90, 20), (91, 19), (92, 20)])
def test_load_and_verify_metadata_rejects_wrong_partition_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, train_count: int, validation_count: int
) -> None:
    (
        manifest_path,
        split_path,
        lesion_path,
        manifest_sha256,
        split_sha256,
        lesion_sha256,
        manifest_embedded_hash,
        split_embedded_hash,
    ) = _write_metadata_fixture(
        tmp_path, train_count=train_count, validation_count=validation_count
    )
    _pin(
        monkeypatch,
        manifest_sha256=manifest_sha256,
        split_sha256=split_sha256,
        manifest_embedded_hash=manifest_embedded_hash,
        split_embedded_hash=split_embedded_hash,
    )

    # Patch the required counts down to keep the fixture small in this test
    # while still exercising the mismatch path against a *different* pair of
    # required constants.
    monkeypatch.setattr(driver, "REQUIRED_TRAIN_CASE_COUNT", 91)
    monkeypatch.setattr(driver, "REQUIRED_VALIDATION_CASE_COUNT", 20)

    with pytest.raises(driver.Phase8DefinitiveRealTrainingCaseSelectionError):
        driver.load_and_verify_definitive_real_training_metadata(
            manifest_path=manifest_path,
            split_path=split_path,
            lesion_components_path=lesion_path,
            expected_lesion_components_sha256=lesion_sha256,
        )


def test_load_and_verify_metadata_rejects_no_positive_train_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (
        manifest_path,
        split_path,
        lesion_path,
        manifest_sha256,
        split_sha256,
        lesion_sha256,
        manifest_embedded_hash,
        split_embedded_hash,
    ) = _write_metadata_fixture(
        tmp_path, train_count=2, validation_count=2, positive_case_ids=set()
    )
    _pin(
        monkeypatch,
        manifest_sha256=manifest_sha256,
        split_sha256=split_sha256,
        manifest_embedded_hash=manifest_embedded_hash,
        split_embedded_hash=split_embedded_hash,
    )

    with pytest.raises(driver.Phase8DefinitiveRealTrainingCaseSelectionError):
        driver.load_and_verify_definitive_real_training_metadata(
            manifest_path=manifest_path,
            split_path=split_path,
            lesion_components_path=lesion_path,
            expected_lesion_components_sha256=lesion_sha256,
        )


# ---------------------------------------------------------------------------
# B. Output-root path safety
# ---------------------------------------------------------------------------


def test_validate_output_root_rejects_existing_directory(tmp_path: Path) -> None:
    existing = tmp_path / "already_here"
    existing.mkdir()
    with pytest.raises(driver.Phase8DefinitiveRealTrainingOutputRootError):
        driver.validate_definitive_real_training_output_root(existing, repository_root=tmp_path)


def test_validate_output_root_rejects_relative_path(tmp_path: Path) -> None:
    with pytest.raises(driver.Phase8DefinitiveRealTrainingOutputRootError):
        driver.validate_definitive_real_training_output_root(
            Path("relative/path"), repository_root=tmp_path
        )


def test_validate_output_root_accepts_nonexistent_absolute_path(tmp_path: Path) -> None:
    repository_root = tmp_path / "repo"
    repository_root.mkdir()
    candidate = tmp_path / "new_output_root"
    paths = driver.validate_definitive_real_training_output_root(
        candidate, repository_root=repository_root
    )
    assert paths.output_root == candidate.resolve()
    assert paths.checkpoints_dir == candidate.resolve() / "checkpoints"
    assert paths.patches_dir == candidate.resolve() / "patches"


# ---------------------------------------------------------------------------
# C. Real geometry / hash validation and preprocessing
# ---------------------------------------------------------------------------


def _write_nifti(path: Path, array: np.ndarray, affine: np.ndarray) -> None:
    image = cast(Any, nib).Nifti1Image(array.astype(np.float32), affine)
    nib.save(image, str(path))


def _ras_affine(spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> np.ndarray:
    return np.diag([spacing[0], spacing[1], spacing[2], 1.0])


def test_validate_raw_case_pair_geometry_and_hash_accepts_valid_ras_pair(tmp_path: Path) -> None:
    shape = (8, 8, 8)
    image_array = np.zeros(shape, dtype=np.float32)
    label_array = np.zeros(shape, dtype=np.float32)
    label_array[2:4, 2:4, 2:4] = 2.0
    affine = _ras_affine()

    image_path = tmp_path / "image.nii.gz"
    label_path = tmp_path / "label.nii.gz"
    _write_nifti(image_path, image_array, affine)
    _write_nifti(label_path, label_array, affine)

    result_image, result_label, result_affine = driver._validate_raw_case_pair_geometry_and_hash(
        image_path=image_path,
        label_path=label_path,
        expected_image_sha256=sha256_file(image_path),
        expected_label_sha256=sha256_file(label_path),
    )
    assert result_image.shape == shape
    assert result_label.shape == shape
    assert result_affine.shape == (4, 4)


def test_validate_raw_case_pair_geometry_and_hash_rejects_hash_mismatch(tmp_path: Path) -> None:
    shape = (8, 8, 8)
    affine = _ras_affine()
    image_path = tmp_path / "image.nii.gz"
    label_path = tmp_path / "label.nii.gz"
    _write_nifti(image_path, np.zeros(shape, dtype=np.float32), affine)
    _write_nifti(label_path, np.zeros(shape, dtype=np.float32), affine)

    with pytest.raises(driver.Phase8DefinitiveRealTrainingHashMismatchError):
        driver._validate_raw_case_pair_geometry_and_hash(
            image_path=image_path,
            label_path=label_path,
            expected_image_sha256="0" * 64,
            expected_label_sha256=sha256_file(label_path),
        )


def test_validate_raw_case_pair_geometry_and_hash_rejects_non_ras_orientation(
    tmp_path: Path,
) -> None:
    shape = (8, 8, 8)
    affine = np.diag([-1.0, -1.0, 1.0, 1.0])  # LPS, not RAS.
    image_path = tmp_path / "image.nii.gz"
    label_path = tmp_path / "label.nii.gz"
    _write_nifti(image_path, np.zeros(shape, dtype=np.float32), affine)
    _write_nifti(label_path, np.zeros(shape, dtype=np.float32), affine)

    with pytest.raises(driver.Phase8DefinitiveRealTrainingOrientationError):
        driver._validate_raw_case_pair_geometry_and_hash(
            image_path=image_path,
            label_path=label_path,
            expected_image_sha256=sha256_file(image_path),
            expected_label_sha256=sha256_file(label_path),
        )


def test_validate_raw_case_pair_geometry_and_hash_rejects_invalid_label_domain(
    tmp_path: Path,
) -> None:
    shape = (8, 8, 8)
    affine = _ras_affine()
    label_array = np.zeros(shape, dtype=np.float32)
    label_array[0, 0, 0] = 5.0  # outside {0, 1, 2}.
    image_path = tmp_path / "image.nii.gz"
    label_path = tmp_path / "label.nii.gz"
    _write_nifti(image_path, np.zeros(shape, dtype=np.float32), affine)
    _write_nifti(label_path, label_array, affine)

    with pytest.raises(driver.Phase8DefinitiveRealTrainingLabelDomainError):
        driver._validate_raw_case_pair_geometry_and_hash(
            image_path=image_path,
            label_path=label_path,
            expected_image_sha256=sha256_file(image_path),
            expected_label_sha256=sha256_file(label_path),
        )


def test_validate_relative_path_prefix_rejects_non_train_prefix() -> None:
    with pytest.raises(driver.Phase8DefinitiveRealTrainingDataError):
        driver._validate_relative_path_prefix(
            "imagesTs/case.nii.gz", field_name="relative_image_path"
        )


def test_preprocess_real_case_produces_locked_patch_scaleable_output() -> None:
    config = pipeline.build_definitive_config_v1()
    shape = (20, 20, 20)
    rng = np.random.default_rng(0)
    image_array = rng.uniform(-500.0, 500.0, size=shape).astype(np.float32)
    label_array = np.zeros(shape, dtype=np.float32)
    label_array[5:8, 5:8, 5:8] = 2.0
    affine = _ras_affine((0.5, 0.5, 0.5))

    scaled_image, binary_label = driver._preprocess_real_case(
        image_array=image_array, label_array=label_array, affine=affine, config=config
    )

    assert scaled_image.shape == binary_label.shape
    assert scaled_image.min() >= -1.0 - 1e-6
    assert scaled_image.max() <= 1.0 + 1e-6
    assert binary_label.dtype == bool
    assert bool(np.any(binary_label))


# ---------------------------------------------------------------------------
# D/E. Real loaders (end-to-end preprocessing through the loader closures)
# ---------------------------------------------------------------------------


def _build_manifest_from_records(records: dict[str, DatasetCaseRecord]) -> DatasetManifest:
    cases = tuple(
        sorted(records.values(), key=lambda r: (r.anonymous_patient_id, r.anonymous_case_id))
    )
    draft = DatasetManifest(
        schema_version="2",
        manifest_type="phase2-dataset-manifest",
        dataset_id="synthetic-test-dataset",
        cohort_role="development",
        adapter_name="synthetic",
        adapter_version="v1",
        generated_at_utc="2026-01-01T00:00:00Z",
        git_commit=_FAKE_GIT_COMMIT[:40],
        dataset_root_fingerprint=_FAKE_SHA256,
        manifest_hash="0" * 64,
        case_count=len(cases),
        cases=cases,
    )
    manifest_hash = hash_dataset_manifest(draft)
    return DatasetManifest(
        schema_version=draft.schema_version,
        manifest_type=draft.manifest_type,
        dataset_id=draft.dataset_id,
        cohort_role=draft.cohort_role,
        adapter_name=draft.adapter_name,
        adapter_version=draft.adapter_version,
        generated_at_utc=draft.generated_at_utc,
        git_commit=draft.git_commit,
        dataset_root_fingerprint=draft.dataset_root_fingerprint,
        manifest_hash=manifest_hash,
        case_count=draft.case_count,
        cases=draft.cases,
    )


def _write_real_dataset(
    tmp_path: Path, *, case_ids: list[str], positive_case_ids: set[str]
) -> tuple[Path, dict[str, DatasetCaseRecord]]:
    dataset_root = tmp_path / "dataset"
    (dataset_root / "imagesTr").mkdir(parents=True)
    (dataset_root / "labelsTr").mkdir(parents=True)

    shape = (16, 16, 16)
    affine = _ras_affine()
    records: dict[str, DatasetCaseRecord] = {}
    for case_index, case_id in enumerate(case_ids):
        rng = np.random.default_rng(case_index)
        image_array = rng.uniform(-100.0, 100.0, size=shape).astype(np.float32)
        label_array = np.zeros(shape, dtype=np.float32)
        if case_id in positive_case_ids:
            label_array[4:6, 4:6, 4:6] = 2.0
        image_path = dataset_root / "imagesTr" / f"{case_id}.nii.gz"
        label_path = dataset_root / "labelsTr" / f"{case_id}.nii.gz"
        _write_nifti(image_path, image_array, affine)
        _write_nifti(label_path, label_array, affine)
        records[case_id] = DatasetCaseRecord(
            anonymous_patient_id=f"patient-{case_id}",
            anonymous_case_id=case_id,
            relative_image_path=f"imagesTr/{case_id}.nii.gz",
            relative_label_path=f"labelsTr/{case_id}.nii.gz",
            image_sha256=sha256_file(image_path),
            label_sha256=sha256_file(label_path),
            cohort_role="development",
        )
    return dataset_root, records


def test_real_train_case_loader_returns_preprocessed_case(tmp_path: Path) -> None:
    case_ids = ["case-a", "case-b"]
    dataset_root, records = _write_real_dataset(
        tmp_path, case_ids=case_ids, positive_case_ids={"case-a"}
    )
    manifest = _build_manifest_from_records(records)
    config = pipeline.build_definitive_config_v1()
    ledger = driver._AccessLedgerRecorder()
    loader = driver._build_real_train_case_loader(
        manifest=manifest, dataset_root=dataset_root, config=config, ledger=ledger
    )

    case = loader("case-a")
    assert case.case_id == "case-a"
    assert bool(np.any(case.label_binary))
    assert len(ledger.records) == 1
    assert ledger.records[0].anonymous_case_id == "case-a"
    assert ledger.records[0].partition == "train"

    ledger_dict = ledger.to_dict()
    ledger_text = str(ledger_dict)
    assert str(dataset_root) not in ledger_text
    assert "imagesTr" not in ledger_text


def test_real_train_case_loader_rejects_unknown_case_id(tmp_path: Path) -> None:
    dataset_root, records = _write_real_dataset(
        tmp_path, case_ids=["case-a"], positive_case_ids={"case-a"}
    )
    manifest = _build_manifest_from_records(records)
    config = pipeline.build_definitive_config_v1()
    ledger = driver._AccessLedgerRecorder()
    loader = driver._build_real_train_case_loader(
        manifest=manifest, dataset_root=dataset_root, config=config, ledger=ledger
    )
    with pytest.raises(driver.Phase8DefinitiveRealTrainingDataError):
        loader("case-does-not-exist")


# ---------------------------------------------------------------------------
# F. Orchestration wiring (heavy pipeline calls monkeypatched with fakes)
# ---------------------------------------------------------------------------


def _fake_full_validation_result(
    *, candidate_step: int, case_ids: tuple[str, ...], mean_dice: float
) -> pipeline.Phase8DefinitiveFullValidationResult:
    case_results = tuple(
        pipeline.Phase8DefinitiveValidationRunResult(
            case_id=case_id, tumor_dice=mean_dice, tumor_iou=0.0, prediction_finite=True
        )
        for case_id in case_ids
    )
    return pipeline.Phase8DefinitiveFullValidationResult(
        candidate_step=candidate_step,
        case_results=case_results,
        case_ids=tuple(sorted(case_ids)),
        mean_tumor_dice=mean_dice,
    )


def test_run_phase8_definitive_real_training_wires_pipeline_calls_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (
        manifest_path,
        split_path,
        lesion_path,
        manifest_sha256,
        split_sha256,
        lesion_sha256,
        manifest_embedded_hash,
        split_embedded_hash,
    ) = _write_metadata_fixture(
        tmp_path,
        train_count=91,
        validation_count=20,
        positive_case_ids={"case-0000"},
    )
    _pin(
        monkeypatch,
        manifest_sha256=manifest_sha256,
        split_sha256=split_sha256,
        manifest_embedded_hash=manifest_embedded_hash,
        split_embedded_hash=split_embedded_hash,
    )

    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    output_root = tmp_path / "output_root"
    repository_root = tmp_path / "repo"
    repository_root.mkdir()

    call_log: list[str] = []

    def _fake_materialize(
        schedule: pipeline.Phase8DefinitivePatchSchedule, **_kwargs: Any
    ) -> pipeline.Phase8DefinitivePatchMaterializationResult:
        call_log.append("materialize")
        return pipeline.Phase8DefinitivePatchMaterializationResult(
            schedule_hash=schedule.schedule_hash, published_patches=(), processed_case_ids=()
        )

    fake_state_dict_250 = {"step": 250}
    fake_state_dict_500 = {"step": 500}

    def _fake_train_from_materialized(
        schedule: pipeline.Phase8DefinitivePatchSchedule, **_kwargs: Any
    ) -> pipeline.Phase8DefinitiveContinuousTrainingRunResult:
        call_log.append("train")
        return pipeline.Phase8DefinitiveContinuousTrainingRunResult(
            step_results=(),
            all_steps_finite=True,
            checkpoint_snapshots={250: fake_state_dict_250, 500: fake_state_dict_500},
        )

    published: dict[int, Path] = {}

    def _fake_publish_checkpoint(
        *,
        state_dict: Any,
        optimizer_step: int,
        output_root: Path,
        config: pipeline.Phase8DefinitiveConfig,
        development_manifest_hash: str,
        development_split_hash: str,
        originating_git_commit: str,
        package_environment_reference: Any,
        preprocessing_evidence_hash: str,
        completion_status: str = "completed",
        seed: int | None = None,
    ) -> pipeline.Phase8DefinitiveCheckpointPublicationResult:
        call_log.append(f"publish_{optimizer_step}")
        checkpoint_metadata = pipeline.build_definitive_checkpoint_metadata(
            checkpoint_sha256="e" * 64,
            checkpoint_byte_size=128,
            config=config,
            development_manifest_hash=development_manifest_hash,
            development_split_hash=development_split_hash,
            originating_git_commit=originating_git_commit,
            package_environment_reference=package_environment_reference,
            completion_status=completion_status,
            preprocessing_evidence_hash=preprocessing_evidence_hash,
        )
        output_path = Path(output_root) / f"fake_checkpoint_{optimizer_step}.pt"
        published[optimizer_step] = output_path
        return pipeline.Phase8DefinitiveCheckpointPublicationResult(
            optimizer_step=optimizer_step,
            output_path=output_path,
            checkpoint_sha256=f"{optimizer_step:064d}"[-64:],
            checkpoint_byte_size=128,
            checkpoint_metadata=checkpoint_metadata,
        )

    def _fake_import_torch() -> Any:
        class _FakeTorch:
            pass

        return _FakeTorch()

    def _fake_import_monai() -> Any:
        class _FakeMonai:
            pass

        return _FakeMonai()

    def _fake_build_model(**_kwargs: Any) -> Any:
        class _FakeModel:
            def load_state_dict(self, state_dict: Any) -> None:
                self.state_dict = state_dict

        return _FakeModel()

    def _fake_full_validation(
        case_references: Any, *, candidate_step: int, **_kwargs: Any
    ) -> pipeline.Phase8DefinitiveFullValidationResult:
        call_log.append(f"validate_{candidate_step}")
        case_ids = tuple(reference.case_id for reference in case_references)
        mean_dice = 0.4 if candidate_step == 250 else 0.7
        return _fake_full_validation_result(
            candidate_step=candidate_step, case_ids=case_ids, mean_dice=mean_dice
        )

    monkeypatch.setattr(pipeline, "materialize_definitive_training_patches", _fake_materialize)
    monkeypatch.setattr(
        pipeline, "run_definitive_training_from_materialized_patches", _fake_train_from_materialized
    )
    monkeypatch.setattr(
        pipeline, "publish_definitive_checkpoint_snapshot", _fake_publish_checkpoint
    )
    monkeypatch.setattr(pipeline, "_import_torch", _fake_import_torch)
    monkeypatch.setattr(pipeline, "_import_monai", _fake_import_monai)
    monkeypatch.setattr(pipeline, "_build_definitive_segresnet_model", _fake_build_model)
    monkeypatch.setattr(pipeline, "run_definitive_full_validation", _fake_full_validation)

    result = driver.run_phase8_definitive_real_training(
        manifest_path=manifest_path,
        split_path=split_path,
        lesion_components_path=lesion_path,
        expected_lesion_components_sha256=lesion_sha256,
        dataset_root=dataset_root,
        output_root=output_root,
        repository_root=repository_root,
        git_commit=_FAKE_GIT_COMMIT,
        package_versions={"torch": "2.2.0"},
    )

    assert call_log == [
        "materialize",
        "train",
        "publish_250",
        "publish_500",
        "validate_250",
        "validate_500",
    ]
    assert result.selected_checkpoint_step == 500
    assert result.mean_tumor_dice_step_250 == pytest.approx(0.4)
    assert result.mean_tumor_dice_step_500 == pytest.approx(0.7)
    assert (
        output_root / driver.PHASE8_DEFINITIVE_REAL_TRAINING_SELECTION_EVIDENCE_FILENAME
    ).exists()
    assert (output_root / driver.PHASE8_DEFINITIVE_REAL_TRAINING_ACCESS_LEDGER_FILENAME).exists()


def test_run_phase8_definitive_real_training_rejects_non_finite_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (
        manifest_path,
        split_path,
        lesion_path,
        manifest_sha256,
        split_sha256,
        lesion_sha256,
        manifest_embedded_hash,
        split_embedded_hash,
    ) = _write_metadata_fixture(tmp_path, train_count=91, validation_count=20)
    _pin(
        monkeypatch,
        manifest_sha256=manifest_sha256,
        split_sha256=split_sha256,
        manifest_embedded_hash=manifest_embedded_hash,
        split_embedded_hash=split_embedded_hash,
    )
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    repository_root = tmp_path / "repo"
    repository_root.mkdir()

    def _fake_materialize(
        schedule: pipeline.Phase8DefinitivePatchSchedule, **_kwargs: Any
    ) -> pipeline.Phase8DefinitivePatchMaterializationResult:
        return pipeline.Phase8DefinitivePatchMaterializationResult(
            schedule_hash=schedule.schedule_hash, published_patches=(), processed_case_ids=()
        )

    def _fake_train_from_materialized(
        schedule: pipeline.Phase8DefinitivePatchSchedule, **_kwargs: Any
    ) -> pipeline.Phase8DefinitiveContinuousTrainingRunResult:
        return pipeline.Phase8DefinitiveContinuousTrainingRunResult(
            step_results=(), all_steps_finite=False, checkpoint_snapshots={}
        )

    monkeypatch.setattr(pipeline, "materialize_definitive_training_patches", _fake_materialize)
    monkeypatch.setattr(
        pipeline, "run_definitive_training_from_materialized_patches", _fake_train_from_materialized
    )

    with pytest.raises(driver.Phase8DefinitiveRealTrainingRuntimeError):
        driver.run_phase8_definitive_real_training(
            manifest_path=manifest_path,
            split_path=split_path,
            lesion_components_path=lesion_path,
            expected_lesion_components_sha256=lesion_sha256,
            dataset_root=dataset_root,
            output_root=tmp_path / "output_root",
            repository_root=repository_root,
            git_commit=_FAKE_GIT_COMMIT,
            package_versions={"torch": "2.2.0"},
        )


def test_run_phase8_definitive_real_training_rejects_invalid_git_commit(tmp_path: Path) -> None:
    with pytest.raises(driver.Phase8DefinitiveRealTrainingRuntimeError):
        driver.run_phase8_definitive_real_training(
            manifest_path=tmp_path / "manifest.json",
            split_path=tmp_path / "split.json",
            lesion_components_path=tmp_path / "lesion_components.json",
            expected_lesion_components_sha256="0" * 64,
            dataset_root=tmp_path,
            output_root=tmp_path / "output_root",
            repository_root=tmp_path,
            git_commit="not-a-git-commit",
            package_versions={},
        )


# ---------------------------------------------------------------------------
# G. Process-level watchdog
# ---------------------------------------------------------------------------


def test_watchdog_kills_blocking_child_and_fails_closed() -> None:
    ctx = multiprocessing.get_context("fork")
    call_count = ctx.Value("i", 0)

    def _blocking_target(**_kwargs: object) -> Any:
        with call_count.get_lock():
            call_count.value += 1
        time.sleep(2.0)
        raise AssertionError("unreachable: the child must be killed before returning")

    with pytest.raises(driver.Phase8DefinitiveRealTrainingWatchdogTimeoutError):
        driver.run_phase8_definitive_real_training_with_watchdog(
            manifest_path=Path("/nonexistent/manifest.json"),
            split_path=Path("/nonexistent/split.json"),
            lesion_components_path=Path("/nonexistent/lesion_components.json"),
            expected_lesion_components_sha256="0" * 64,
            dataset_root=Path("/nonexistent/dataset"),
            output_root=Path("/nonexistent/output_root"),
            repository_root=Path("/nonexistent/repo"),
            git_commit=_FAKE_GIT_COMMIT,
            package_versions={},
            wall_clock_limit_seconds=0.2,
            _target=_blocking_target,
            _multiprocessing_context=ctx,
            _termination_grace_seconds=1.0,
        )

    assert multiprocessing.active_children() == []
    assert call_count.value == 1


def test_watchdog_timeout_never_surfaces_a_result() -> None:
    ctx = multiprocessing.get_context("fork")

    def _blocking_target(**_kwargs: object) -> Any:
        time.sleep(2.0)
        raise AssertionError("unreachable")

    outcome: dict[str, object] = {}
    with contextlib.suppress(driver.Phase8DefinitiveRealTrainingWatchdogTimeoutError):
        outcome["result"] = driver.run_phase8_definitive_real_training_with_watchdog(
            manifest_path=Path("/nonexistent/manifest.json"),
            split_path=Path("/nonexistent/split.json"),
            lesion_components_path=Path("/nonexistent/lesion_components.json"),
            expected_lesion_components_sha256="0" * 64,
            dataset_root=Path("/nonexistent/dataset"),
            output_root=Path("/nonexistent/output_root"),
            repository_root=Path("/nonexistent/repo"),
            git_commit=_FAKE_GIT_COMMIT,
            package_versions={},
            wall_clock_limit_seconds=0.2,
            _target=_blocking_target,
            _multiprocessing_context=ctx,
            _termination_grace_seconds=1.0,
        )

    assert "result" not in outcome
    assert multiprocessing.active_children() == []


def test_watchdog_propagates_child_exception() -> None:
    ctx = multiprocessing.get_context("fork")

    def _raising_target(**_kwargs: object) -> Any:
        raise driver.Phase8DefinitiveRealTrainingDataError("synthetic failure")

    with pytest.raises(driver.Phase8DefinitiveRealTrainingDataError):
        driver.run_phase8_definitive_real_training_with_watchdog(
            manifest_path=Path("/nonexistent/manifest.json"),
            split_path=Path("/nonexistent/split.json"),
            lesion_components_path=Path("/nonexistent/lesion_components.json"),
            expected_lesion_components_sha256="0" * 64,
            dataset_root=Path("/nonexistent/dataset"),
            output_root=Path("/nonexistent/output_root"),
            repository_root=Path("/nonexistent/repo"),
            git_commit=_FAKE_GIT_COMMIT,
            package_versions={},
            wall_clock_limit_seconds=5.0,
            _target=_raising_target,
            _multiprocessing_context=ctx,
            _termination_grace_seconds=1.0,
        )


# ---------------------------------------------------------------------------
# H. Static scan: driver must never embed the real, approved required hashes
#    outside their two named constants and must not read internal_test paths.
# ---------------------------------------------------------------------------


def test_source_hardcodes_real_manifest_and_split_hashes_only_once_each() -> None:
    source_path = Path(driver.__file__)
    source_text = source_path.read_text(encoding="utf-8")
    assert (
        source_text.count("c24244951e050050cf25c4b321f67d61c2087fc0c93fdcf9d112e0e488e1384b") == 1
    )
    assert (
        source_text.count("936376cd7b5e6070397c2fef16e5125c60fd6569ff3188d7e9bb5428a46ffadb") == 1
    )
