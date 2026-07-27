"""Unit tests for deterministic Phase 2 development split generation."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any, cast

import pytest

from protoem_ct.artifacts import (
    DATASET_MANIFEST_TYPE,
    DEVELOPMENT_SPLIT_MANIFEST_TYPE,
    PHASE2_DEVELOPMENT_COHORT_ROLE,
    PHASE2_SCHEMA_VERSION,
    DatasetCaseRecord,
    DatasetManifest,
    DevelopmentSplitManifest,
    Phase2ArtifactSerializationError,
    Phase2ArtifactStageError,
    development_split_hash_payload,
    hash_dataset_manifest,
    hash_development_split,
    phase2_artifact_from_json,
    phase2_artifact_to_dict,
    phase2_artifact_to_json,
)
from protoem_ct.data import (
    PATIENT_HASH_RANK_POLICY_VERSION,
    DevelopmentSplitPolicy,
    ExistingSplitOutputError,
    IncompatibleSplitPatientCountsError,
    InvalidDevelopmentSplitPolicyError,
    InvalidSplitInputManifestError,
    SplitManifestHashMismatchError,
    SplitPublicationError,
    UnsafeSplitOutputPathError,
    build_development_split,
    patient_ranking_digest,
)
from protoem_ct.data import development_split as split_builder

GENERATED_AT_UTC = "2026-07-26T00:00:00Z"
GIT_COMMIT = "30ef8ab"


def _case(index: int) -> DatasetCaseRecord:
    return DatasetCaseRecord(
        anonymous_patient_id=f"anon-p{index:04d}",
        anonymous_case_id=f"anon-c{index:04d}",
        relative_image_path=f"images/case-{index:04d}.nii.gz",
        relative_label_path=f"labels/case-{index:04d}.nii.gz",
        image_sha256=f"{index:064x}",
        label_sha256=f"{index + 1000:064x}",
        cohort_role=PHASE2_DEVELOPMENT_COHORT_ROLE,
    )


def _manifest(case_count: int = 9, **overrides: object) -> DatasetManifest:
    cases = tuple(_case(index) for index in range(1, case_count + 1))
    manifest_without_hash = DatasetManifest(
        schema_version=PHASE2_SCHEMA_VERSION,
        manifest_type=DATASET_MANIFEST_TYPE,
        dataset_id="lits-development",
        cohort_role=PHASE2_DEVELOPMENT_COHORT_ROLE,
        adapter_name="lits_style",
        adapter_version="1",
        generated_at_utc="2026-07-25T00:00:00Z",
        git_commit="30ef8ab",
        dataset_root_fingerprint="0" * 64,
        manifest_hash="0" * 64,
        case_count=len(cases),
        cases=cases,
    )
    manifest = replace(
        manifest_without_hash, manifest_hash=hash_dataset_manifest(manifest_without_hash)
    )
    return cast(DatasetManifest, cast(Any, replace)(manifest, **overrides))


def _policy(**overrides: object) -> DevelopmentSplitPolicy:
    values: dict[str, object] = {
        "policy_version": PATIENT_HASH_RANK_POLICY_VERSION,
        "split_seed": 1729,
        "train_patient_count": 5,
        "validation_patient_count": 2,
        "internal_test_patient_count": 2,
    }
    values.update(overrides)
    return DevelopmentSplitPolicy(**cast(Any, values))


def _write_manifest(path: Path, manifest: DatasetManifest | None = None) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = phase2_artifact_to_json(manifest or _manifest())
    path.write_text(text, encoding="utf-8", newline="\n")
    return path.read_bytes()


def _build(
    manifest_path: Path,
    output_path: Path,
    *,
    policy: DevelopmentSplitPolicy | None = None,
    generated_at_utc: str = GENERATED_AT_UTC,
    git_commit: str = GIT_COMMIT,
) -> DevelopmentSplitManifest:
    result = build_development_split(
        manifest_path,
        policy=policy or _policy(),
        output_path=output_path,
        git_commit=git_commit,
        generated_at_utc=generated_at_utc,
    )
    return result.split_manifest


def _partition_sets(
    split_manifest: DevelopmentSplitManifest,
) -> dict[str, tuple[set[str], set[str]]]:
    result: dict[str, tuple[set[str], set[str]]] = {
        "train": (set(), set()),
        "validation": (set(), set()),
        "internal_test": (set(), set()),
    }
    for assignment in split_manifest.assignments:
        patients, cases = result[assignment.partition]
        patients.add(assignment.anonymous_patient_id)
        cases.add(assignment.anonymous_case_id)
    return result


def test_policy_validation_and_immutability() -> None:
    policy = _policy()
    assert policy.total_patient_count == 9

    with pytest.raises(InvalidDevelopmentSplitPolicyError):
        _policy(split_seed=-1)
    with pytest.raises(InvalidDevelopmentSplitPolicyError):
        _policy(policy_version="patient-split-v0")
    with pytest.raises(InvalidDevelopmentSplitPolicyError):
        _policy(train_patient_count=0)
    with pytest.raises(InvalidDevelopmentSplitPolicyError):
        _policy(validation_patient_count=-1)

    with pytest.raises(FrozenInstanceError):
        cast(Any, policy).split_seed = 1


def test_ranking_is_deterministic_and_seed_sensitive() -> None:
    patient_id = "anon-p0001"

    first = patient_ranking_digest(
        patient_id,
        policy_version=PATIENT_HASH_RANK_POLICY_VERSION,
        split_seed=1729,
    )
    second = patient_ranking_digest(
        patient_id,
        policy_version=PATIENT_HASH_RANK_POLICY_VERSION,
        split_seed=1729,
    )
    changed = patient_ranking_digest(
        patient_id,
        policy_version=PATIENT_HASH_RANK_POLICY_VERSION,
        split_seed=1730,
    )

    assert first == second
    assert first != changed
    assert patient_id not in first


def test_same_seed_assignments_identical_and_different_seed_changes_fixture(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path)

    first = _build(manifest_path, tmp_path / "out-a" / "split.json")
    second = _build(manifest_path, tmp_path / "out-b" / "split.json")
    changed = _build(
        manifest_path,
        tmp_path / "out-c" / "split.json",
        policy=_policy(split_seed=1),
    )

    assert first.assignments == second.assignments
    assert first.assignments != changed.assignments


def test_assignment_invariants_and_partition_counts(tmp_path: Path) -> None:
    manifest = _manifest()
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, manifest)

    split_manifest = _build(manifest_path, tmp_path / "split.json")
    partitions = _partition_sets(split_manifest)

    assert split_manifest.train_patient_count == 5
    assert split_manifest.validation_patient_count == 2
    assert split_manifest.internal_test_patient_count == 2
    assert split_manifest.train_case_count == 5
    assert split_manifest.validation_case_count == 2
    assert split_manifest.internal_test_case_count == 2
    assert {assignment.anonymous_patient_id for assignment in split_manifest.assignments} == {
        case.anonymous_patient_id for case in manifest.cases
    }
    assert {assignment.anonymous_case_id for assignment in split_manifest.assignments} == {
        case.anonymous_case_id for case in manifest.cases
    }
    assert partitions["train"][0].isdisjoint(partitions["validation"][0])
    assert partitions["train"][0].isdisjoint(partitions["internal_test"][0])
    assert partitions["validation"][0].isdisjoint(partitions["internal_test"][0])
    assert partitions["train"][1].isdisjoint(partitions["validation"][1])
    assert partitions["train"][1].isdisjoint(partitions["internal_test"][1])
    assert partitions["validation"][1].isdisjoint(partitions["internal_test"][1])
    assert split_manifest.assignments == tuple(
        sorted(
            split_manifest.assignments,
            key=lambda assignment: (
                assignment.anonymous_patient_id,
                assignment.anonymous_case_id,
            ),
        )
    )


def test_split_is_independent_of_cwd_and_manifest_location(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    workdir = tmp_path / "work"
    workdir.mkdir()
    manifest = _manifest()
    manifest_a = root_a / "manifest.json"
    manifest_b = root_b / "manifest.json"
    _write_manifest(manifest_a, manifest)
    _write_manifest(manifest_b, manifest)

    monkeypatch.chdir(workdir)

    first = _build(manifest_a, tmp_path / "out-a" / "split.json")
    second = _build(manifest_b, tmp_path / "out-b" / "split.json")

    assert phase2_artifact_to_json(first) == phase2_artifact_to_json(second)


def test_split_does_not_open_manifest_referenced_files(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path)

    split_manifest = _build(manifest_path, tmp_path / "split.json")

    assert split_manifest.train_case_count == 5
    assert not (tmp_path / "images").exists()
    assert not (tmp_path / "labels").exists()


def test_manifest_integrity_failures_are_rejected(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path)
    tampered = phase2_artifact_from_json(manifest_path.read_text(encoding="utf-8"), DatasetManifest)
    tampered = replace(tampered, manifest_hash="f" * 64)
    manifest_path.write_text(phase2_artifact_to_json(tampered), encoding="utf-8")
    with pytest.raises(SplitManifestHashMismatchError):
        _build(manifest_path, tmp_path / "split.json")

    wrong_role = phase2_artifact_to_dict(_manifest())
    wrong_role["cohort_role"] = "external"
    manifest_path.write_text(json.dumps(wrong_role), encoding="utf-8")
    with pytest.raises(InvalidSplitInputManifestError):
        _build(manifest_path, tmp_path / "wrong-role.json")

    wrong_type = phase2_artifact_to_dict(_manifest())
    wrong_type["manifest_type"] = DEVELOPMENT_SPLIT_MANIFEST_TYPE
    manifest_path.write_text(json.dumps(wrong_type), encoding="utf-8")
    with pytest.raises(InvalidSplitInputManifestError):
        _build(manifest_path, tmp_path / "wrong-type.json")

    manifest_path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(InvalidSplitInputManifestError):
        _build(manifest_path, tmp_path / "malformed.json")

    with pytest.raises(InvalidSplitInputManifestError):
        _build(tmp_path / "missing.json", tmp_path / "missing-out.json")


def test_wrong_artifact_type_is_rejected(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    split_like = {
        "assignments": [],
        "generated_at_utc": GENERATED_AT_UTC,
        "git_commit": GIT_COMMIT,
        "internal_test_case_count": 0,
        "internal_test_patient_count": 0,
        "manifest_type": DEVELOPMENT_SPLIT_MANIFEST_TYPE,
        "schema_version": PHASE2_SCHEMA_VERSION,
        "source_manifest_hash": "0" * 64,
        "split_hash": "0" * 64,
        "split_policy_version": PATIENT_HASH_RANK_POLICY_VERSION,
        "split_seed": 0,
        "train_case_count": 0,
        "train_patient_count": 0,
        "validation_case_count": 0,
        "validation_patient_count": 0,
    }
    manifest_path.write_text(str(split_like).replace("'", '"'), encoding="utf-8")

    with pytest.raises((InvalidSplitInputManifestError, Phase2ArtifactStageError)):
        _build(manifest_path, tmp_path / "split.json")


def test_incompatible_patient_counts_are_rejected(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path)

    with pytest.raises(IncompatibleSplitPatientCountsError):
        _build(
            manifest_path,
            tmp_path / "small.json",
            policy=_policy(train_patient_count=4),
        )
    with pytest.raises(IncompatibleSplitPatientCountsError):
        _build(
            manifest_path,
            tmp_path / "large.json",
            policy=_policy(train_patient_count=6),
        )


def test_split_hash_artifact_and_json_contract(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_bytes = _write_manifest(manifest_path)
    source_manifest = phase2_artifact_from_json(
        manifest_path.read_text(encoding="utf-8"), DatasetManifest
    )

    split_manifest = _build(manifest_path, tmp_path / "split.json")
    loaded = phase2_artifact_from_json(
        (tmp_path / "split.json").read_text(encoding="utf-8"),
        DevelopmentSplitManifest,
    )

    assert loaded == split_manifest
    assert split_manifest.source_manifest_hash == source_manifest.manifest_hash
    assert split_manifest.split_hash == hash_development_split(split_manifest)
    assert "split_hash" not in development_split_hash_payload(split_manifest)
    assert manifest_path.read_bytes() == manifest_bytes


def test_seed_timestamp_and_location_reproducibility(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path)

    baseline = _build(manifest_path, tmp_path / "a" / "split.json")
    seed_changed = _build(
        manifest_path,
        tmp_path / "b" / "split.json",
        policy=_policy(split_seed=1),
    )
    timestamp_changed = _build(
        manifest_path,
        tmp_path / "c" / "split.json",
        generated_at_utc="2026-07-27T00:00:00Z",
    )
    independent = _build(manifest_path, tmp_path / "d" / "split.json")

    assert seed_changed.split_hash != baseline.split_hash
    assert timestamp_changed.split_hash != baseline.split_hash
    assert phase2_artifact_to_json(timestamp_changed) != phase2_artifact_to_json(baseline)
    assert phase2_artifact_to_json(independent) == phase2_artifact_to_json(baseline)


def test_serialized_split_contains_no_source_paths_or_local_roots(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path)

    split_manifest = _build(manifest_path, tmp_path / "split.json")
    json_text = phase2_artifact_to_json(split_manifest)

    for forbidden in (
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


def test_output_safety_and_atomic_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path)

    existing = tmp_path / "existing.json"
    existing.write_text("occupied\n", encoding="utf-8")
    with pytest.raises(ExistingSplitOutputError):
        _build(manifest_path, existing)
    with pytest.raises(UnsafeSplitOutputPathError):
        _build(manifest_path, manifest_path)
    with pytest.raises(UnsafeSplitOutputPathError):
        _build(manifest_path, Path("relative.json"))
    with pytest.raises(UnsafeSplitOutputPathError):
        _build(manifest_path, tmp_path / "a" / ".." / "split.json")

    parent_file = tmp_path / "parent-file"
    parent_file.write_text("not a dir\n", encoding="utf-8")
    with pytest.raises(UnsafeSplitOutputPathError):
        _build(manifest_path, parent_file / "split.json")

    output = tmp_path / "out" / "split.json"

    def _fail_serialize(artifact: object) -> str:
        raise Phase2ArtifactSerializationError("synthetic serialization failure")

    monkeypatch.setattr(split_builder, "phase2_artifact_to_json", _fail_serialize)
    with pytest.raises(SplitPublicationError):
        _build(manifest_path, output)
    assert not output.exists()
    assert not (output.parent / ".split.json.tmp").exists()
