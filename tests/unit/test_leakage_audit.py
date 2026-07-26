"""Unit tests for deterministic Phase 2 development leakage-audit generation."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

import protoem_ct.artifacts.hashing as artifact_hashing
from protoem_ct.artifacts import (
    DATASET_MANIFEST_TYPE,
    DEVELOPMENT_SPLIT_MANIFEST_TYPE,
    PHASE2_DEVELOPMENT_COHORT_ROLE,
    PHASE2_SCHEMA_VERSION,
    DatasetCaseRecord,
    DatasetManifest,
    DevelopmentSplitManifest,
    LeakageAuditArtifact,
    Phase2ArtifactSerializationError,
    Phase2ArtifactValidationError,
    SplitAssignment,
    hash_dataset_manifest,
    hash_development_split,
    hash_leakage_audit,
    leakage_audit_hash_payload,
    phase2_artifact_from_json,
    phase2_artifact_to_dict,
    phase2_artifact_to_json,
)
from protoem_ct.data.leakage_audit import (
    DEVELOPMENT_LEAKAGE_AUDIT_CONTRACT_VERSION,
    DevelopmentLeakageAuditConfig,
    DevelopmentLeakageAuditHashMismatchError,
    DevelopmentLeakageAuditLinkageError,
    DevelopmentLeakageAuditPublicationError,
    ExistingDevelopmentLeakageAuditOutputError,
    InvalidDevelopmentLeakageAuditConfigError,
    InvalidDevelopmentLeakageAuditInputError,
    UnsafeDevelopmentLeakageAuditOutputPathError,
    _compute_audit_inputs,
    development_leakage_audit_config_hash_payload,
    hash_development_leakage_audit_config,
    run_development_leakage_audit,
)

CREATED_AT_UTC = "2026-07-25T00:00:00Z"
GIT_COMMIT = "091a64c"
HASH0 = "0" * 64
HASH1 = "1" * 64
HASH2 = "2" * 64
HASH3 = "3" * 64
HASH4 = "4" * 64
HASH5 = "5" * 64
HASH6 = "6" * 64
HASH7 = "7" * 64


def _case(
    index: int,
    *,
    label_hash: str | None = None,
    image_hash: str | None = None,
) -> DatasetCaseRecord:
    return DatasetCaseRecord(
        anonymous_patient_id=f"anon-p{index:03d}",
        anonymous_case_id=f"anon-c{index:03d}",
        relative_image_path=f"images/case-{index:03d}.nii.gz",
        relative_label_path=f"labels/case-{index:03d}.nii.gz",
        image_sha256=image_hash or f"{index:064x}",
        label_sha256=label_hash or f"{index + 10:064x}",
        cohort_role=PHASE2_DEVELOPMENT_COHORT_ROLE,
    )


def _manifest(cases: tuple[DatasetCaseRecord, ...] | None = None) -> DatasetManifest:
    manifest_cases = cases or (_case(1), _case(2), _case(3))
    without_hash = DatasetManifest(
        schema_version=PHASE2_SCHEMA_VERSION,
        manifest_type=DATASET_MANIFEST_TYPE,
        dataset_id="synthetic-dev",
        cohort_role=PHASE2_DEVELOPMENT_COHORT_ROLE,
        adapter_name="lits_style",
        adapter_version="1",
        generated_at_utc=CREATED_AT_UTC,
        git_commit=GIT_COMMIT,
        dataset_root_fingerprint=HASH1,
        manifest_hash=HASH0,
        case_count=len(manifest_cases),
        cases=manifest_cases,
    )
    return replace(without_hash, manifest_hash=hash_dataset_manifest(without_hash))


def _assignments() -> tuple[SplitAssignment, ...]:
    return (
        SplitAssignment("anon-p001", "anon-c001", "train"),
        SplitAssignment("anon-p002", "anon-c002", "validation"),
        SplitAssignment("anon-p003", "anon-c003", "internal_test"),
    )


def _split(
    manifest: DatasetManifest,
    assignments: tuple[SplitAssignment, ...] | None = None,
) -> DevelopmentSplitManifest:
    split_assignments = assignments or _assignments()
    patient_counts = {
        partition: len(
            {
                assignment.anonymous_patient_id
                for assignment in split_assignments
                if assignment.partition == partition
            }
        )
        for partition in ("train", "validation", "internal_test")
    }
    case_counts = {
        partition: sum(1 for assignment in split_assignments if assignment.partition == partition)
        for partition in ("train", "validation", "internal_test")
    }
    without_hash = DevelopmentSplitManifest(
        schema_version=PHASE2_SCHEMA_VERSION,
        manifest_type=DEVELOPMENT_SPLIT_MANIFEST_TYPE,
        generated_at_utc=CREATED_AT_UTC,
        git_commit=GIT_COMMIT,
        source_manifest_hash=manifest.manifest_hash,
        split_policy_version="patient_hash_rank_v1",
        split_seed=1729,
        split_hash=HASH0,
        assignments=tuple(
            sorted(
                split_assignments,
                key=lambda item: (item.anonymous_patient_id, item.anonymous_case_id),
            )
        ),
        train_patient_count=patient_counts["train"],
        validation_patient_count=patient_counts["validation"],
        internal_test_patient_count=patient_counts["internal_test"],
        train_case_count=case_counts["train"],
        validation_case_count=case_counts["validation"],
        internal_test_case_count=case_counts["internal_test"],
    )
    return replace(without_hash, split_hash=hash_development_split(without_hash))


def _write_inputs(
    tmp_path: Path,
    *,
    manifest: DatasetManifest | None = None,
    split: DevelopmentSplitManifest | None = None,
) -> tuple[Path, Path]:
    manifest_artifact = manifest or _manifest()
    split_artifact = split or _split(manifest_artifact)
    tmp_path.mkdir(parents=True, exist_ok=True)
    manifest_path = tmp_path / "manifest.json"
    split_path = tmp_path / "split.json"
    manifest_path.write_text(phase2_artifact_to_json(manifest_artifact), encoding="utf-8")
    split_path.write_text(phase2_artifact_to_json(split_artifact), encoding="utf-8")
    return manifest_path, split_path


def _run(paths: tuple[Path, Path], output: Path) -> LeakageAuditArtifact:
    return run_development_leakage_audit(
        paths[0],
        paths[1],
        output_path=output,
        config=DevelopmentLeakageAuditConfig(DEVELOPMENT_LEAKAGE_AUDIT_CONTRACT_VERSION),
        git_commit=GIT_COMMIT,
        created_at_utc=CREATED_AT_UTC,
    )


def test_config_schema_hash_and_immutability(tmp_path: Path) -> None:
    config = DevelopmentLeakageAuditConfig(DEVELOPMENT_LEAKAGE_AUDIT_CONTRACT_VERSION)
    assert development_leakage_audit_config_hash_payload(config) == {
        "audit_contract_version": DEVELOPMENT_LEAKAGE_AUDIT_CONTRACT_VERSION,
        "payload_type": "phase2-development-leakage-audit-config-v1",
        "schema_version": PHASE2_SCHEMA_VERSION,
    }
    assert hash_development_leakage_audit_config(config) == hash_development_leakage_audit_config(
        config
    )
    with pytest.raises(InvalidDevelopmentLeakageAuditConfigError):
        DevelopmentLeakageAuditConfig("unsupported")
    with pytest.raises(FrozenInstanceError):
        config.audit_contract_version = "other"  # type: ignore[misc]

    artifact = _run(_write_inputs(tmp_path), tmp_path / "out" / "audit.json")
    parsed = phase2_artifact_from_json(
        phase2_artifact_to_json(artifact),
        LeakageAuditArtifact,
    )
    assert parsed == artifact
    artifact_dict = phase2_artifact_to_dict(artifact)
    artifact_dict["unknown"] = True
    with pytest.raises(Phase2ArtifactValidationError):
        phase2_artifact_from_json(json.dumps(artifact_dict))


def test_clean_audit_passes_with_zero_findings(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    output = tmp_path / "out" / "audit.json"

    artifact = _run(paths, output)

    assert artifact.audit_passed is True
    assert artifact.assignment_complete is True
    assert artifact.manifest_case_count == 3
    assert artifact.manifest_patient_count == 3
    assert dict(artifact.patient_counts_by_partition) == {
        "train": 1,
        "validation": 1,
        "internal_test": 1,
    }
    assert all(value == 0 for value in artifact.pairwise_patient_overlap_counts.values())
    assert all(value == 0 for value in artifact.pairwise_case_overlap_counts.values())
    assert artifact.image_hash_cross_partition_overlap_count == 0
    assert artifact.label_hash_cross_partition_overlap_count == 0
    assert artifact.image_label_pair_cross_partition_overlap_count == 0
    assert artifact.finding_codes == ()
    assert artifact.audit_hash == hash_leakage_audit(artifact)
    assert "audit_hash" not in leakage_audit_hash_payload(artifact)
    assert output.read_text(encoding="utf-8").endswith("\n")


@pytest.mark.parametrize(
    ("assignments", "expected_code"),
    [
        (
            (
                SplitAssignment("anon-p001", "anon-c001", "train"),
                SplitAssignment("anon-p002", "anon-c002", "validation"),
                SplitAssignment("anon-p999", "anon-c999", "internal_test"),
            ),
            "missing_case_assignment",
        ),
        (
            (
                SplitAssignment("anon-p001", "anon-c001", "train"),
                SplitAssignment("anon-p002", "anon-c002", "validation"),
                SplitAssignment("anon-p999", "anon-c999", "internal_test"),
            ),
            "extra_case_assignment",
        ),
        (
            (
                SplitAssignment("anon-p999", "anon-c001", "train"),
                SplitAssignment("anon-p002", "anon-c002", "validation"),
                SplitAssignment("anon-p003", "anon-c003", "internal_test"),
            ),
            "patient_id_mismatch",
        ),
        (
            (
                SplitAssignment("anon-p001", "anon-c001", "train"),
                SplitAssignment("anon-p002", "anon-c002", "validation"),
            ),
            "empty_partition",
        ),
    ],
)
def test_assignment_findings_publish_with_audit_failed(
    tmp_path: Path,
    assignments: tuple[SplitAssignment, ...],
    expected_code: str,
) -> None:
    manifest = _manifest()
    split = _split(manifest, assignments)

    artifact = _run(_write_inputs(tmp_path, manifest=manifest, split=split), tmp_path / "out.json")

    assert artifact.audit_passed is False
    assert expected_code in artifact.finding_codes
    assert artifact.critical_finding_count == len(artifact.finding_codes)
    assert artifact.unresolved_findings == artifact.finding_codes


def test_label_hash_cross_partition_overlap_is_detected(tmp_path: Path) -> None:
    manifest = _manifest((_case(1, label_hash=HASH7), _case(2, label_hash=HASH7), _case(3)))
    split = _split(manifest)

    artifact = _run(_write_inputs(tmp_path, manifest=manifest, split=split), tmp_path / "out.json")

    assert artifact.audit_passed is False
    assert artifact.label_hash_cross_partition_overlap_count == 1
    assert artifact.finding_codes == ("label_hash_cross_partition_overlap",)
    assert artifact.duplicate_label_hash_findings == ("label_hash_cross_partition_overlap",)


def test_detector_covers_schema_impossible_overlap_and_identity_codes() -> None:
    clean_manifest = _manifest()
    duplicate_image_cases = (
        _case(1, image_hash=HASH5, label_hash=HASH6),
        _case(2, image_hash=HASH5, label_hash=HASH6),
        _case(3),
    )
    unsafe_manifest = _unsafe_manifest(clean_manifest, duplicate_image_cases)
    split = _split(clean_manifest)

    audit_inputs = _compute_audit_inputs(unsafe_manifest, split)

    assert "image_hash_cross_partition_overlap" in audit_inputs.finding_codes
    assert "image_label_pair_cross_partition_overlap" in audit_inputs.finding_codes


def test_multiple_findings_follow_fixed_order(tmp_path: Path) -> None:
    manifest = _manifest((_case(1, label_hash=HASH7), _case(2, label_hash=HASH7), _case(3)))
    assignments = (
        SplitAssignment("anon-p999", "anon-c001", "train"),
        SplitAssignment("anon-p002", "anon-c002", "validation"),
    )
    split = _split(manifest, assignments)

    artifact = _run(_write_inputs(tmp_path, manifest=manifest, split=split), tmp_path / "out.json")

    assert artifact.finding_codes == (
        "missing_case_assignment",
        "patient_id_mismatch",
        "empty_partition",
        "label_hash_cross_partition_overlap",
    )


def test_structural_integrity_failures_reject_without_output(tmp_path: Path) -> None:
    manifest = _manifest()
    split = _split(manifest)
    paths = _write_inputs(tmp_path, manifest=manifest, split=split)
    tampered_manifest = replace(manifest, manifest_hash=HASH7)
    paths = _write_inputs(tmp_path / "tampered-manifest", manifest=tampered_manifest, split=split)
    with pytest.raises(DevelopmentLeakageAuditHashMismatchError):
        _run(paths, tmp_path / "bad-manifest.json")

    bad_split = replace(split, split_hash=HASH7)
    paths = _write_inputs(tmp_path / "tampered-split", manifest=manifest, split=bad_split)
    with pytest.raises(DevelopmentLeakageAuditHashMismatchError):
        _run(paths, tmp_path / "bad-split.json")

    unlinked = replace(split, source_manifest_hash=HASH7, split_hash=HASH0)
    unlinked = replace(unlinked, split_hash=hash_development_split(unlinked))
    paths = _write_inputs(tmp_path / "unlinked", manifest=manifest, split=unlinked)
    with pytest.raises(DevelopmentLeakageAuditLinkageError):
        _run(paths, tmp_path / "unlinked.json")

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{bad json", encoding="utf-8")
    with pytest.raises(InvalidDevelopmentLeakageAuditInputError):
        _run((malformed, paths[1]), tmp_path / "malformed-out.json")


def test_output_safety_and_publication_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_inputs(tmp_path)
    with pytest.raises(UnsafeDevelopmentLeakageAuditOutputPathError):
        _run(paths, Path("relative.json"))
    with pytest.raises(UnsafeDevelopmentLeakageAuditOutputPathError):
        _run(paths, tmp_path / "out.md")
    with pytest.raises(UnsafeDevelopmentLeakageAuditOutputPathError):
        _run(paths, paths[0])
    existing = tmp_path / "existing.json"
    existing.write_text("occupied\n", encoding="utf-8")
    with pytest.raises(ExistingDevelopmentLeakageAuditOutputError):
        _run(paths, existing)
    parent_file = tmp_path / "parent-file"
    parent_file.write_text("not a directory\n", encoding="utf-8")
    with pytest.raises(UnsafeDevelopmentLeakageAuditOutputPathError):
        _run(paths, parent_file / "audit.json")

    def _fail_serialize(_artifact: object) -> str:
        raise Phase2ArtifactSerializationError("synthetic serialization failure")

    monkeypatch.setattr("protoem_ct.data.leakage_audit.phase2_artifact_to_json", _fail_serialize)
    failed_output = tmp_path / "failed" / "audit.json"
    with pytest.raises(DevelopmentLeakageAuditPublicationError):
        _run(paths, failed_output)
    assert not failed_output.exists()
    assert not list((tmp_path / "failed").glob("*.tmp"))


def test_boundaries_no_source_hashing_or_paths_and_deterministic_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_inputs(tmp_path / "inputs")
    first = tmp_path / "first" / "audit.json"
    second = tmp_path / "second" / "audit.json"

    def _forbidden_file_hash(_path: Path) -> str:
        raise AssertionError("source file hashing must not be called")

    monkeypatch.setattr(artifact_hashing, "sha256_file", _forbidden_file_hash)
    first_artifact = _run(paths, first)
    second_artifact = _run(paths, second)

    assert first.read_bytes() == second.read_bytes()
    assert first_artifact == second_artifact
    text = first.read_text(encoding="utf-8")
    assert "images/" not in text
    assert "labels/" not in text
    assert "case-" not in text
    assert ".nii" not in text


def _unsafe_manifest(
    template: DatasetManifest,
    cases: tuple[DatasetCaseRecord, ...],
) -> DatasetManifest:
    manifest = object.__new__(DatasetManifest)
    for field_name, value in phase2_artifact_to_dict(template).items():
        object.__setattr__(manifest, field_name, value)
    object.__setattr__(manifest, "cases", cases)
    object.__setattr__(manifest, "case_count", len(cases))
    return manifest
