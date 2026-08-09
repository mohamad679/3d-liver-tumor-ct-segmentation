"""Unit tests for the Phase 8 tiny real-development verification (Substage 3)."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pytest
from click.utils import strip_ansi  # type: ignore[attr-defined]
from typer.testing import CliRunner

from protoem_ct.artifacts import (
    DATASET_MANIFEST_TYPE,
    DEVELOPMENT_SPLIT_MANIFEST_TYPE,
    PHASE2_DEVELOPMENT_COHORT_ROLE,
    PHASE2_SCHEMA_VERSION,
    DatasetCaseRecord,
    DatasetManifest,
    DevelopmentSplitManifest,
    SplitAssignment,
    hash_dataset_manifest,
    hash_development_split,
    phase2_artifact_to_json,
)
from protoem_ct.artifacts.hashing import sha256_file, sha256_json
from protoem_ct.cli.main import app
from protoem_ct.external.internal_evidence import (
    Phase8CheckpointMetadata,
    Phase8InternalEvidenceValidationError,
)
from protoem_ct.external.tiny_real_verification import (
    DEFAULT_PATCH_SIZE,
    MAX_OPTIMIZER_STEPS,
    Phase8TinyRealAccessLedger,
    Phase8TinyRealCheckpointMetadata,
    Phase8TinyRealSelectedCase,
    Phase8TinyRealVerificationConfigError,
    Phase8TinyRealVerificationError,
    Phase8TinyRealVerificationOutputRootError,
    build_phase8_tiny_real_access_ledger,
    build_phase8_tiny_real_checkpoint_metadata,
    build_phase8_tiny_real_verification_config,
    phase8_tiny_real_access_ledger_to_dict,
    phase8_tiny_real_checkpoint_metadata_to_dict,
    phase8_tiny_real_verification_config_to_dict,
    select_phase8_tiny_real_development_cases,
    validate_phase8_tiny_real_output_root,
)

requires_baseline_environment = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None or importlib.util.find_spec("monai") is None,
    reason="Phase 8 tiny real-development verification requires the isolated baseline environment.",
)

GIT_COMMIT = "a" * 40
CREATED_AT_UTC = "2026-08-04T00:00:00Z"
PACKAGE_VERSIONS = {"torch": "2.2.0", "monai": "1.3.0"}


# ---------------------------------------------------------------------------
# Synthetic Phase 2 manifest/split fixtures (mirrors test_phase8_real_development_runner.py)
# ---------------------------------------------------------------------------


def _case(index: int) -> DatasetCaseRecord:
    return DatasetCaseRecord(
        anonymous_patient_id=f"anon-p{index:04d}",
        anonymous_case_id=f"anon-c{index:04d}",
        relative_image_path=f"images/case-{index:04d}.nii.gz",
        relative_label_path=f"labels/case-{index:04d}.nii.gz",
        image_sha256=f"{index:064x}",
        label_sha256=f"{index + 500:064x}",
        cohort_role=PHASE2_DEVELOPMENT_COHORT_ROLE,
    )


def _manifest(case_count: int = 6) -> DatasetManifest:
    cases = tuple(_case(index) for index in range(1, case_count + 1))
    without_hash = DatasetManifest(
        schema_version=PHASE2_SCHEMA_VERSION,
        manifest_type=DATASET_MANIFEST_TYPE,
        dataset_id="synthetic-dev",
        cohort_role=PHASE2_DEVELOPMENT_COHORT_ROLE,
        adapter_name="lits_style",
        adapter_version="1",
        generated_at_utc=CREATED_AT_UTC,
        git_commit=GIT_COMMIT,
        dataset_root_fingerprint="0" * 64,
        manifest_hash="0" * 64,
        case_count=len(cases),
        cases=cases,
    )
    return replace(without_hash, manifest_hash=hash_dataset_manifest(without_hash))


def _split(manifest: DatasetManifest) -> DevelopmentSplitManifest:
    # cases 1-3 -> train, 4-5 -> validation, 6 -> internal_test (1:1 patient:case).
    assignments: list[SplitAssignment] = []
    for index, case in enumerate(manifest.cases, start=1):
        if index <= 3:
            partition = "train"
        elif index <= 5:
            partition = "validation"
        else:
            partition = "internal_test"
        assignments.append(
            SplitAssignment(
                anonymous_patient_id=case.anonymous_patient_id,
                anonymous_case_id=case.anonymous_case_id,
                partition=partition,
            )
        )
    assignments_tuple = tuple(
        sorted(assignments, key=lambda item: (item.anonymous_patient_id, item.anonymous_case_id))
    )
    patient_counts = {
        partition: sum(1 for item in assignments_tuple if item.partition == partition)
        for partition in ("train", "validation", "internal_test")
    }
    without_hash = DevelopmentSplitManifest(
        schema_version=PHASE2_SCHEMA_VERSION,
        manifest_type=DEVELOPMENT_SPLIT_MANIFEST_TYPE,
        generated_at_utc=CREATED_AT_UTC,
        git_commit=GIT_COMMIT,
        source_manifest_hash=manifest.manifest_hash,
        split_policy_version="v1",
        split_seed=1729,
        split_hash="0" * 64,
        assignments=assignments_tuple,
        train_patient_count=patient_counts["train"],
        validation_patient_count=patient_counts["validation"],
        internal_test_patient_count=patient_counts["internal_test"],
        train_case_count=patient_counts["train"],
        validation_case_count=patient_counts["validation"],
        internal_test_case_count=patient_counts["internal_test"],
    )
    return replace(without_hash, split_hash=hash_development_split(without_hash))


def _write_manifest_and_split(tmp_path: Path) -> tuple[Path, Path, DatasetManifest]:
    manifest = _manifest()
    split = _split(manifest)
    manifest_path = tmp_path / "manifest.json"
    split_path = tmp_path / "split.json"
    phase2_artifact_to_json(manifest, manifest_path)
    phase2_artifact_to_json(split, split_path)
    return manifest_path, split_path, manifest


def _write_dataset_root(tmp_path: Path, *, shape: tuple[int, int, int] = (26, 26, 18)) -> Path:
    dataset_root = tmp_path / "dataset"
    (dataset_root / "images").mkdir(parents=True)
    (dataset_root / "labels").mkdir(parents=True)
    affine = np.eye(4, dtype=np.float64)
    for index in range(1, 7):
        rng = np.random.default_rng(index)
        image_array = rng.uniform(-500.0, 500.0, size=shape).astype(np.float32)
        label_array = np.zeros(shape, dtype=np.uint8)
        label_array[0, 0, 0] = 1
        image = nib.Nifti1Image(image_array, affine)  # type: ignore[no-untyped-call]
        label = nib.Nifti1Image(label_array, affine)  # type: ignore[no-untyped-call]
        nib.save(image, str(dataset_root / "images" / f"case-{index:04d}.nii.gz"))
        nib.save(label, str(dataset_root / "labels" / f"case-{index:04d}.nii.gz"))
    return dataset_root


def _write_nan_dataset_root(tmp_path: Path, *, shape: tuple[int, int, int] = (26, 26, 18)) -> Path:
    dataset_root = _write_dataset_root(tmp_path, shape=shape)
    affine = np.eye(4, dtype=np.float64)
    nan_image_array = np.full(shape, np.nan, dtype=np.float32)
    nan_image = nib.Nifti1Image(nan_image_array, affine)  # type: ignore[no-untyped-call]
    nib.save(nan_image, str(dataset_root / "images" / "case-0001.nii.gz"))
    return dataset_root


# ---------------------------------------------------------------------------
# 1-2: Compute-limit config validation (Section C)
# ---------------------------------------------------------------------------


def test_config_builds_with_hard_coded_limits() -> None:
    config = build_phase8_tiny_real_verification_config(
        max_optimizer_steps=2,
        seed=1729,
        originating_git_commit=GIT_COMMIT,
        package_versions=PACKAGE_VERSIONS,
    )
    assert config.device_type == "cpu"
    assert config.amp_enabled is False
    assert config.train_case_count == 1
    assert config.validation_case_count == 1
    assert config.batch_size == 1
    assert config.num_workers == 0
    assert config.max_epochs == 1
    assert config.train_patches_per_step == 1
    assert config.max_validation_patches == 1
    assert config.fit_scope == "no_data_dependent_fit"
    assert config.hu_window_min == -1000.0
    assert config.hu_window_max == 1000.0
    assert config.patch_size == DEFAULT_PATCH_SIZE
    round_trip = phase8_tiny_real_verification_config_to_dict(config)
    assert round_trip["config_hash"] == config.config_hash
    assert sha256_json({k: v for k, v in round_trip.items() if k != "config_hash"}) == (
        config.config_hash
    )


@pytest.mark.parametrize("max_steps", [0, 3, -1])
def test_config_rejects_out_of_bound_step_counts(max_steps: int) -> None:
    with pytest.raises(Phase8TinyRealVerificationConfigError):
        build_phase8_tiny_real_verification_config(
            max_optimizer_steps=max_steps,
            seed=1729,
            originating_git_commit=GIT_COMMIT,
            package_versions=PACKAGE_VERSIONS,
        )


def test_config_accepts_boundary_step_counts() -> None:
    for steps in range(1, MAX_OPTIMIZER_STEPS + 1):
        config = build_phase8_tiny_real_verification_config(
            max_optimizer_steps=steps,
            seed=1729,
            originating_git_commit=GIT_COMMIT,
            package_versions=PACKAGE_VERSIONS,
        )
        assert config.max_optimizer_steps == steps


@pytest.mark.parametrize(
    "patch_size",
    [(33, 24, 16), (24, 65, 16), (24, 24, 65), (0, 24, 16)],
)
def test_config_rejects_out_of_bound_patch_dims(patch_size: tuple[int, int, int]) -> None:
    with pytest.raises(Phase8TinyRealVerificationConfigError):
        build_phase8_tiny_real_verification_config(
            max_optimizer_steps=2,
            seed=1729,
            originating_git_commit=GIT_COMMIT,
            package_versions=PACKAGE_VERSIONS,
            patch_size=patch_size,
        )


def test_config_rejects_invalid_git_commit() -> None:
    with pytest.raises(Phase8TinyRealVerificationConfigError):
        build_phase8_tiny_real_verification_config(
            max_optimizer_steps=2,
            seed=1729,
            originating_git_commit="not-hex",
            package_versions=PACKAGE_VERSIONS,
        )


def test_config_device_is_hard_coded_cpu_and_cannot_be_overridden() -> None:
    # There is no device parameter on the builder at all; construct the frozen
    # dataclass directly with a non-cpu device to prove the type itself rejects it.
    config = build_phase8_tiny_real_verification_config(
        max_optimizer_steps=2,
        seed=1729,
        originating_git_commit=GIT_COMMIT,
        package_versions=PACKAGE_VERSIONS,
    )
    with pytest.raises(Phase8TinyRealVerificationConfigError):
        replace(config, device_type="cuda", config_hash="0" * 64)


def test_config_amp_cannot_be_enabled() -> None:
    config = build_phase8_tiny_real_verification_config(
        max_optimizer_steps=2,
        seed=1729,
        originating_git_commit=GIT_COMMIT,
        package_versions=PACKAGE_VERSIONS,
    )
    with pytest.raises(Phase8TinyRealVerificationConfigError):
        replace(config, amp_enabled=True, config_hash="0" * 64)


# ---------------------------------------------------------------------------
# 3-5: Deterministic case selection
# ---------------------------------------------------------------------------


def test_case_selection_selects_exactly_one_train_and_one_validation() -> None:
    manifest = _manifest()
    split = _split(manifest)
    train_case, validation_case = select_phase8_tiny_real_development_cases(
        manifest=manifest, split=split
    )
    assert train_case.partition == "train"
    assert validation_case.partition == "validation"
    assert train_case.anonymous_case_id == "anon-c0001"
    assert validation_case.anonymous_case_id == "anon-c0004"


@dataclass(frozen=True, slots=True)
class _FakeSplit:
    assignments: tuple[SplitAssignment, ...]


def test_case_selection_deterministic_regardless_of_input_order() -> None:
    manifest = _manifest()
    split = _split(manifest)
    sorted_train, sorted_validation = select_phase8_tiny_real_development_cases(
        manifest=manifest, split=split
    )

    scrambled_assignments = tuple(reversed(split.assignments))
    scrambled_split = _FakeSplit(assignments=scrambled_assignments)
    scrambled_train, scrambled_validation = select_phase8_tiny_real_development_cases(
        manifest=manifest,
        split=scrambled_split,  # type: ignore[arg-type]
    )

    assert scrambled_train == sorted_train
    assert scrambled_validation == sorted_validation


def test_case_selection_never_selects_internal_test_case() -> None:
    manifest = _manifest()
    split = _split(manifest)
    internal_test_case_ids = {
        a.anonymous_case_id for a in split.assignments if a.partition == "internal_test"
    }
    assert internal_test_case_ids  # sanity: fixture actually has an internal_test case

    train_case, validation_case = select_phase8_tiny_real_development_cases(
        manifest=manifest, split=split
    )
    assert train_case.anonymous_case_id not in internal_test_case_ids
    assert validation_case.anonymous_case_id not in internal_test_case_ids


# ---------------------------------------------------------------------------
# 6-8: Output-root path safety
# ---------------------------------------------------------------------------


def test_output_root_rejects_symlink(tmp_path: Path) -> None:
    repository_root = tmp_path / "repo"
    repository_root.mkdir()
    real_target = tmp_path / "real-target"
    real_target.mkdir()
    output_root = tmp_path / "symlinked-output"
    output_root.symlink_to(real_target, target_is_directory=True)
    with pytest.raises(Phase8TinyRealVerificationOutputRootError):
        validate_phase8_tiny_real_output_root(output_root, repository_root=repository_root)


def test_output_root_rejects_existing_nonempty_directory(tmp_path: Path) -> None:
    repository_root = tmp_path / "repo"
    repository_root.mkdir()
    output_root = tmp_path / "runs" / "phase8-tiny-real"
    output_root.mkdir(parents=True)
    (output_root / "preexisting.json").write_text("{}", encoding="utf-8")
    with pytest.raises(Phase8TinyRealVerificationOutputRootError):
        validate_phase8_tiny_real_output_root(output_root, repository_root=repository_root)


def test_output_root_rejects_existing_empty_directory(tmp_path: Path) -> None:
    repository_root = tmp_path / "repo"
    repository_root.mkdir()
    output_root = tmp_path / "runs" / "phase8-tiny-real"
    output_root.mkdir(parents=True)
    with pytest.raises(Phase8TinyRealVerificationOutputRootError):
        validate_phase8_tiny_real_output_root(output_root, repository_root=repository_root)


def test_output_root_rejects_path_inside_repository(tmp_path: Path) -> None:
    repository_root = tmp_path / "repo"
    repository_root.mkdir()
    output_root = repository_root / "runs" / "phase8-tiny-real"
    with pytest.raises(Phase8TinyRealVerificationOutputRootError):
        validate_phase8_tiny_real_output_root(output_root, repository_root=repository_root)


def test_output_root_accepts_valid_nonexistent_external_root(tmp_path: Path) -> None:
    repository_root = tmp_path / "repo"
    repository_root.mkdir()
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    output_root = runs_root / "phase8-tiny-real"
    paths = validate_phase8_tiny_real_output_root(output_root, repository_root=repository_root)
    assert paths.output_root == output_root.resolve()
    assert paths.checkpoints_dir == paths.output_root / "checkpoints"
    assert not paths.output_root.exists()


# ---------------------------------------------------------------------------
# 9-10: Access ledger anonymity
# ---------------------------------------------------------------------------


def _selected_case(*, case_id: str, patient_id: str, partition: str) -> Phase8TinyRealSelectedCase:
    return Phase8TinyRealSelectedCase(
        anonymous_patient_id=patient_id,
        anonymous_case_id=case_id,
        partition=partition,
        relative_image_path=f"images/{case_id}.nii.gz",
        relative_label_path=f"labels/{case_id}.nii.gz",
    )


def _scan_for_absolute_path_like_strings(value: Any) -> list[str]:
    findings: list[str] = []
    if isinstance(value, str):
        if value.startswith("/") or value.startswith("~") or "\\" in value:
            findings.append(value)
    elif isinstance(value, dict):
        for item in value.values():
            findings.extend(_scan_for_absolute_path_like_strings(item))
    elif isinstance(value, list):
        for item in value:
            findings.extend(_scan_for_absolute_path_like_strings(item))
    return findings


def test_access_ledger_contains_anonymous_ids_only() -> None:
    train_case = _selected_case(case_id="anon-c0001", patient_id="anon-p0001", partition="train")
    validation_case = _selected_case(
        case_id="anon-c0004", patient_id="anon-p0004", partition="validation"
    )
    ledger = build_phase8_tiny_real_access_ledger(
        train_case=train_case, validation_case=validation_case
    )
    assert ledger.internal_test_opened is False
    assert ledger.external_data_opened is False
    assert ledger.file_access_count == 4

    payload = phase8_tiny_real_access_ledger_to_dict(ledger)
    assert _scan_for_absolute_path_like_strings(payload) == []
    serialized = json.dumps(payload)
    assert "images/" not in serialized
    assert "labels/" not in serialized
    assert train_case.relative_image_path not in serialized


def test_access_ledger_rejects_internal_test_opened_true() -> None:
    with pytest.raises(Phase8TinyRealVerificationError):
        Phase8TinyRealAccessLedger(
            schema_name="phase8_tiny_real_access_ledger",
            schema_version="v1",
            records=(),
            internal_test_opened=True,
            external_data_opened=False,
            file_access_count=0,
            ledger_hash="0" * 64,
        )


# ---------------------------------------------------------------------------
# 11-13: Checkpoint safety
# ---------------------------------------------------------------------------


def test_checkpoint_metadata_remains_verification_only_and_not_freeze_eligible() -> None:
    config = build_phase8_tiny_real_verification_config(
        max_optimizer_steps=2,
        seed=1729,
        originating_git_commit=GIT_COMMIT,
        package_versions=PACKAGE_VERSIONS,
    )
    metadata = build_phase8_tiny_real_checkpoint_metadata(
        config=config,
        development_manifest_hash="a" * 64,
        development_split_hash="b" * 64,
        checkpoint_sha256="c" * 64,
        checkpoint_byte_size=1024,
        executed_step_count=2,
    )
    assert metadata.tiny_verification_only is True
    assert metadata.selection_eligible is False
    assert metadata.definitive_training is False
    assert metadata.checkpoint_metadata.freeze_eligible is False
    assert metadata.checkpoint_metadata.completion_status == "synthetic_smoke"

    payload = phase8_tiny_real_checkpoint_metadata_to_dict(metadata)
    assert payload["wrapper_hash"] == metadata.wrapper_hash
    assert _scan_for_absolute_path_like_strings(payload) == []


def test_checkpoint_metadata_ineligibility_flags_cannot_be_flipped() -> None:
    config = build_phase8_tiny_real_verification_config(
        max_optimizer_steps=2,
        seed=1729,
        originating_git_commit=GIT_COMMIT,
        package_versions=PACKAGE_VERSIONS,
    )
    metadata = build_phase8_tiny_real_checkpoint_metadata(
        config=config,
        development_manifest_hash="a" * 64,
        development_split_hash="b" * 64,
        checkpoint_sha256="c" * 64,
        checkpoint_byte_size=1024,
        executed_step_count=2,
    )
    for kwargs in (
        {"selection_eligible": True},
        {"definitive_training": True},
        {"tiny_verification_only": False},
    ):
        with pytest.raises(Phase8TinyRealVerificationError):
            Phase8TinyRealCheckpointMetadata(
                schema_name=metadata.schema_name,
                schema_version=metadata.schema_version,
                checkpoint_metadata=metadata.checkpoint_metadata,
                tiny_verification_only=kwargs.get(
                    "tiny_verification_only", metadata.tiny_verification_only
                ),
                selection_eligible=kwargs.get("selection_eligible", metadata.selection_eligible),
                definitive_training=kwargs.get("definitive_training", metadata.definitive_training),
                wrapper_hash="0" * 64,
            )


def test_internal_evidence_package_rejects_tiny_checkpoint_as_freeze_eligible() -> None:
    """The existing Phase8CheckpointMetadata contract itself must reject this.

    completion_status='synthetic_smoke' plus freeze_eligible=True must raise
    Phase8InternalEvidenceValidationError, because freeze_eligible requires
    completion_status == 'completed'. This is the structural mechanism that
    makes the tiny verification checkpoint permanently non-freeze-eligible.
    """
    from protoem_ct.external.internal_evidence import ArtifactReference

    package_environment_reference = ArtifactReference(
        schema_name="phase8_tiny_real_verification_config",
        schema_version="v1",
        artifact_hash="a" * 64,
        artifact_role="tiny_verification_config",
    )
    with pytest.raises(Phase8InternalEvidenceValidationError):
        Phase8CheckpointMetadata(
            schema_name="phase8_checkpoint_metadata",
            schema_version="v1",
            checkpoint_sha256="c" * 64,
            checkpoint_byte_size=1024,
            serialization_format="torch_state_dict",
            model_family="monai_segresnet",
            candidate_id="phase8_tiny_real_verification",
            seed=1729,
            training_adaptation_mode="baseline_training",
            originating_git_commit=GIT_COMMIT,
            package_environment_reference=package_environment_reference,
            training_config_hash="d" * 64,
            preprocessing_evidence_hash="e" * 64,
            development_manifest_hash="f" * 64,
            development_split_hash="1" * 64,
            validation_metric_artifact_reference=None,
            device_type="cpu",
            amp_state="disabled",
            completion_status="synthetic_smoke",
            failure_codes=(),
            no_external_data=True,
            freeze_eligible=True,
            checkpoint_metadata_hash="0" * 64,
        )


# ---------------------------------------------------------------------------
# 14: Static scan for external dataset literals / no leaked absolute paths
# ---------------------------------------------------------------------------


def test_source_contains_no_external_dataset_literals() -> None:
    from protoem_ct.external import tiny_real_verification

    source_path = Path(tiny_real_verification.__file__)
    source_text = source_path.read_text(encoding="utf-8").lower()
    forbidden_literals = ("3d-ircadb", "ircadb", "/volumes/lexar", "datasets/external")
    for literal in forbidden_literals:
        assert literal not in source_text, f"forbidden external-dataset literal found: {literal!r}"


# ---------------------------------------------------------------------------
# 15: Fail-closed on invalid input (no torch import reached, no artifacts written)
# ---------------------------------------------------------------------------


def test_orchestration_fails_closed_on_nonfinite_train_image(tmp_path: Path) -> None:
    from protoem_ct.external.tiny_real_verification import (
        Phase8TinyRealVerificationDataError,
        run_phase8_tiny_real_development_verification,
    )

    manifest_path, split_path, _manifest_obj = _write_manifest_and_split(tmp_path)
    dataset_root = _write_nan_dataset_root(tmp_path)
    repository_root = tmp_path / "repo"
    repository_root.mkdir()
    output_root = tmp_path / "output"

    with pytest.raises(Phase8TinyRealVerificationDataError):
        run_phase8_tiny_real_development_verification(
            manifest_path=manifest_path,
            split_path=split_path,
            expected_manifest_sha256=sha256_file(manifest_path),
            expected_split_sha256=sha256_file(split_path),
            dataset_root=dataset_root,
            output_root=output_root,
            repository_root=repository_root,
            approved_development_artifact_set_identity="phase2-real-lits-v2",
            git_commit=GIT_COMMIT,
            package_versions=PACKAGE_VERSIONS,
            max_steps=2,
            seed=1729,
        )
    assert not output_root.exists()


def test_orchestration_rejects_scope_broadening_before_any_file_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from protoem_ct.external.tiny_real_verification import (
        run_phase8_tiny_real_development_verification,
    )

    def _forbidden_load(*args: object, **kwargs: object) -> None:
        raise AssertionError("nib.load must not be called when scope validation fails first.")

    # protoem_ct.external.tiny_real_verification imports the same nibabel
    # module object as `nib` here, so patching this attribute also patches
    # the loader the orchestration function would call.
    monkeypatch.setattr(nib, "load", _forbidden_load)

    manifest_path, split_path, _manifest_obj = _write_manifest_and_split(tmp_path)
    dataset_root = tmp_path / "does-not-exist"
    repository_root = tmp_path / "repo"
    repository_root.mkdir()
    output_root = tmp_path / "output"

    with pytest.raises(Phase8TinyRealVerificationConfigError):
        run_phase8_tiny_real_development_verification(
            manifest_path=manifest_path,
            split_path=split_path,
            expected_manifest_sha256=sha256_file(manifest_path),
            expected_split_sha256=sha256_file(split_path),
            dataset_root=dataset_root,
            output_root=output_root,
            repository_root=repository_root,
            approved_development_artifact_set_identity="phase2-real-lits-v2",
            git_commit=GIT_COMMIT,
            package_versions=PACKAGE_VERSIONS,
            max_steps=3,
            seed=1729,
        )
    assert not output_root.exists()


def test_orchestration_fails_closed_on_out_of_domain_label_value(tmp_path: Path) -> None:
    from protoem_ct.external.tiny_real_verification import (
        Phase8TinyRealVerificationLabelDomainError,
        run_phase8_tiny_real_development_verification,
    )

    manifest_path, split_path, _manifest_obj = _write_manifest_and_split(tmp_path)
    dataset_root = _write_dataset_root(tmp_path)
    # Introduce a label value outside the approved raw LiTS/MSD domain
    # {0, 1, 2} for the first (deterministically-selected) train case.
    affine = np.eye(4, dtype=np.float64)
    shape = (26, 26, 18)
    label_array = np.zeros(shape, dtype=np.uint8)
    label_array[0, 0, 0] = 9
    label = nib.Nifti1Image(label_array, affine)  # type: ignore[no-untyped-call]
    nib.save(label, str(dataset_root / "labels" / "case-0001.nii.gz"))
    repository_root = tmp_path / "repo"
    repository_root.mkdir()
    output_root = tmp_path / "output"

    with pytest.raises(Phase8TinyRealVerificationLabelDomainError):
        run_phase8_tiny_real_development_verification(
            manifest_path=manifest_path,
            split_path=split_path,
            expected_manifest_sha256=sha256_file(manifest_path),
            expected_split_sha256=sha256_file(split_path),
            dataset_root=dataset_root,
            output_root=output_root,
            repository_root=repository_root,
            approved_development_artifact_set_identity="phase2-real-lits-v2",
            git_commit=GIT_COMMIT,
            package_versions=PACKAGE_VERSIONS,
            max_steps=2,
            seed=1729,
        )
    assert not output_root.exists()


@requires_baseline_environment
def test_raw_three_class_label_is_accepted_and_binarized_to_tumor_foreground(
    tmp_path: Path,
) -> None:
    """Regression test for the raw LiTS/MSD label-domain defect.

    The real Task03_Liver labels are 0=background, 1=liver, 2=tumor, not
    strictly binary. This proves a raw label containing all three values is
    accepted (not rejected as an invalid binary label) and that the bounded
    patch loader binarizes it to foreground=(label==2) rather than treating
    "liver" (1) as foreground.
    """

    from protoem_ct.external.tiny_real_verification import (
        Phase8TinyRealVerificationConfig,
        _load_bounded_patch,
        _validate_raw_case_pair_geometry,
        build_phase8_tiny_real_verification_config,
    )

    shape = (26, 26, 18)
    affine = np.eye(4, dtype=np.float64)
    image_array = np.random.default_rng(7).uniform(-500.0, 500.0, size=shape).astype(np.float32)
    label_array = np.zeros(shape, dtype=np.uint8)
    label_array[1, 1, 1] = 1  # liver: must NOT become foreground
    label_array[2, 2, 2] = 2  # tumor: must become foreground
    image_path = tmp_path / "image.nii.gz"
    label_path = tmp_path / "label.nii.gz"
    nib.save(nib.Nifti1Image(image_array, affine), str(image_path))  # type: ignore[no-untyped-call]
    nib.save(nib.Nifti1Image(label_array, affine), str(label_path))  # type: ignore[no-untyped-call]

    _validate_raw_case_pair_geometry(image_path, label_path)

    config: Phase8TinyRealVerificationConfig = build_phase8_tiny_real_verification_config(
        max_optimizer_steps=2,
        seed=1729,
        originating_git_commit=GIT_COMMIT,
        package_versions=PACKAGE_VERSIONS,
        patch_size=(4, 4, 4),
    )
    import torch  # type: ignore[import-not-found]

    _image_tensor, label_tensor = _load_bounded_patch(
        image_path=image_path, label_path=label_path, config=config, torch=torch
    )
    label_values = set(torch.unique(label_tensor).tolist())
    assert label_values <= {0, 1}
    assert int(label_tensor[0, 1, 1, 1]) == 0  # liver voxel -> background
    assert int(label_tensor[0, 2, 2, 2]) == 1  # tumor voxel -> foreground


# ---------------------------------------------------------------------------
# 16: CLI help and approval gating
# ---------------------------------------------------------------------------


def test_cli_help_mentions_verification_only_bounded_and_approval() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["run-phase8-tiny-real-development-verification", "--help"])
    assert result.exit_code == 0, result.output
    lowered = strip_ansi(result.output).lower()
    assert "verification-only" in lowered or "verification_only" in lowered
    assert "bounded" in lowered
    assert "approve-tiny-real-verification" in lowered


def test_cli_refuses_without_approval_before_opening_any_file(tmp_path: Path) -> None:
    manifest_path, split_path, _manifest_obj = _write_manifest_and_split(tmp_path)
    dataset_root = tmp_path / "does-not-exist"
    output_root = tmp_path / "output"

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "run-phase8-tiny-real-development-verification",
            "--manifest-path",
            str(manifest_path),
            "--split-path",
            str(split_path),
            "--expected-manifest-sha256",
            sha256_file(manifest_path),
            "--expected-split-sha256",
            sha256_file(split_path),
            "--dataset-root",
            str(dataset_root),
            "--output-root",
            str(output_root),
            "--approved-development-artifact-set-identity",
            "phase2-real-lits-v2",
            "--git-commit",
            GIT_COMMIT,
            "--package-versions-json",
            json.dumps(PACKAGE_VERSIONS),
        ],
    )
    assert result.exit_code != 0
    assert not output_root.exists()


# ---------------------------------------------------------------------------
# 17-19: Full synthetic pipeline (requires torch/monai)
# ---------------------------------------------------------------------------


@requires_baseline_environment
def test_full_synthetic_verification_produces_all_artifacts(tmp_path: Path) -> None:
    from protoem_ct.external.tiny_real_verification import (
        PHASE8_TINY_REAL_ACCESS_LEDGER_FILENAME,
        PHASE8_TINY_REAL_CHECKPOINT_METADATA_FILENAME,
        PHASE8_TINY_REAL_CHECKPOINT_SUBDIRECTORY_NAME,
        PHASE8_TINY_REAL_CONFIG_FILENAME,
        PHASE8_TINY_REAL_VERIFICATION_SUMMARY_FILENAME,
        run_phase8_tiny_real_development_verification,
    )

    manifest_path, split_path, _manifest_obj = _write_manifest_and_split(tmp_path)
    dataset_root = _write_dataset_root(tmp_path)
    repository_root = tmp_path / "repo"
    repository_root.mkdir()
    output_root = tmp_path / "output"

    result = run_phase8_tiny_real_development_verification(
        manifest_path=manifest_path,
        split_path=split_path,
        expected_manifest_sha256=sha256_file(manifest_path),
        expected_split_sha256=sha256_file(split_path),
        dataset_root=dataset_root,
        output_root=output_root,
        repository_root=repository_root,
        approved_development_artifact_set_identity="phase2-real-lits-v2",
        git_commit=GIT_COMMIT,
        package_versions=PACKAGE_VERSIONS,
        max_steps=2,
        seed=1729,
    )

    assert (output_root / PHASE8_TINY_REAL_CONFIG_FILENAME).is_file()
    assert (output_root / PHASE8_TINY_REAL_ACCESS_LEDGER_FILENAME).is_file()
    assert (output_root / PHASE8_TINY_REAL_CHECKPOINT_METADATA_FILENAME).is_file()
    assert (output_root / PHASE8_TINY_REAL_VERIFICATION_SUMMARY_FILENAME).is_file()
    checkpoint_dir = output_root / PHASE8_TINY_REAL_CHECKPOINT_SUBDIRECTORY_NAME
    checkpoint_files = list(checkpoint_dir.glob("*.pt"))
    assert len(checkpoint_files) == 1

    summary_text = (output_root / PHASE8_TINY_REAL_VERIFICATION_SUMMARY_FILENAME).read_text(
        encoding="utf-8"
    )
    summary = json.loads(summary_text)
    assert summary["verification_only"] is True
    assert summary["checkpoint_round_trip_verified"] is True
    assert summary["executed_step_count"] == 2
    # The disclaimer field is expected to *name* dice/iou/hd95 in order to disclaim them; only
    # the disclaimer field may reference those terms, and it must not report a numeric value.
    disclaimer = summary["non_scientific_disclaimer"]
    non_disclaimer_fields = {k: v for k, v in summary.items() if k != "non_scientific_disclaimer"}
    non_disclaimer_text = json.dumps(non_disclaimer_fields).lower()
    assert "dice" not in non_disclaimer_text
    assert "iou" not in non_disclaimer_text
    assert "dice" in disclaimer.lower()
    assert "iou" in disclaimer.lower()
    assert _scan_for_absolute_path_like_strings(summary) == []

    assert result.config_hash
    assert result.access_ledger_hash
    assert result.checkpoint_metadata_hash
    assert result.summary_hash


@requires_baseline_environment
def test_cli_smoke_execution_creates_expected_files_only_under_tmp_path(
    tmp_path: Path,
) -> None:
    manifest_path, split_path, _manifest_obj = _write_manifest_and_split(tmp_path)
    dataset_root = _write_dataset_root(tmp_path)
    repository_root = tmp_path / "repo"
    repository_root.mkdir()
    output_root = tmp_path / "output"

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "run-phase8-tiny-real-development-verification",
            "--approve-tiny-real-verification",
            "--manifest-path",
            str(manifest_path),
            "--split-path",
            str(split_path),
            "--expected-manifest-sha256",
            sha256_file(manifest_path),
            "--expected-split-sha256",
            sha256_file(split_path),
            "--dataset-root",
            str(dataset_root),
            "--output-root",
            str(output_root),
            "--approved-development-artifact-set-identity",
            "phase2-real-lits-v2",
            "--git-commit",
            GIT_COMMIT,
            "--package-versions-json",
            json.dumps(PACKAGE_VERSIONS),
            "--repository-root",
            str(repository_root),
        ],
    )
    assert result.exit_code == 0, result.output
    assert output_root.is_dir()
    published_files = sorted(p.name for p in output_root.iterdir() if p.is_file())
    assert "phase8_tiny_real_verification_config.json" in published_files
    assert "phase8_tiny_real_access_ledger.json" in published_files
    assert "phase8_tiny_real_checkpoint_metadata.json" in published_files
    assert "phase8_tiny_real_verification_summary.json" in published_files
    assert not any(repository_root.rglob("*.json"))


@requires_baseline_environment
def test_internal_test_case_is_never_opened_by_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from protoem_ct.external.tiny_real_verification import (
        run_phase8_tiny_real_development_verification,
    )

    manifest_path, split_path, manifest = _write_manifest_and_split(tmp_path)
    split = _split(manifest)
    internal_test_records = {
        record.relative_image_path
        for record in manifest.cases
        if record.anonymous_case_id
        in {a.anonymous_case_id for a in split.assignments if a.partition == "internal_test"}
    }
    dataset_root = _write_dataset_root(tmp_path)

    original_load = nib.load
    opened_relative_names: list[str] = []

    def _spy_load(path: str, *args: object, **kwargs: object) -> object:
        opened_relative_names.append(Path(path).name)
        return original_load(path, *args, **kwargs)

    # protoem_ct.external.tiny_real_verification imports the same nibabel
    # module object as `nib` here, so patching this attribute also patches
    # the loader the orchestration function would call.
    monkeypatch.setattr(nib, "load", _spy_load)

    repository_root = tmp_path / "repo"
    repository_root.mkdir()
    output_root = tmp_path / "output"

    run_phase8_tiny_real_development_verification(
        manifest_path=manifest_path,
        split_path=split_path,
        expected_manifest_sha256=sha256_file(manifest_path),
        expected_split_sha256=sha256_file(split_path),
        dataset_root=dataset_root,
        output_root=output_root,
        repository_root=repository_root,
        approved_development_artifact_set_identity="phase2-real-lits-v2",
        git_commit=GIT_COMMIT,
        package_versions=PACKAGE_VERSIONS,
        max_steps=2,
        seed=1729,
    )

    internal_test_basenames = {Path(p).name for p in internal_test_records}
    for name in opened_relative_names:
        assert name not in internal_test_basenames
