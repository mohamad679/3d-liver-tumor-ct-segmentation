"""Unit tests for the Research-v2 R0 protocol contract."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from protoem_ct.research_v2.protocol import (
    AccessPurpose,
    DataAccessViolation,
    DataPartition,
    PartitionIsolationError,
    Phase2ArtifactIdentity,
    ProtocolValidationError,
    assert_data_access,
    assert_protocol_artifact_lock,
    audit_partition_isolation,
    build_nnunet_splits,
    load_protocol,
)
from protoem_ct.research_v2.r0_audit import write_nnunet_split_files

BASE = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = BASE / "configs" / "research_v2" / "protocol_v2.yaml"

EXPECTED_MANIFEST_ARTIFACT_HASH = (
    "c24244951e050050cf25c4b321f67d61c2087fc0c93fdcf9d112e0e488e1384b"
)
EXPECTED_MANIFEST_FILE_SHA256 = (
    "0e21a7d555e20b6011091bc18e462bc150cc23a8f46e522dbd8c01b014a44ae3"
)
EXPECTED_SPLIT_ARTIFACT_HASH = (
    "936376cd7b5e6070397c2fef16e5125c60fd6569ff3188d7e9bb5428a46ffadb"
)
EXPECTED_SPLIT_FILE_SHA256 = (
    "416ca83e85c8598fc4f7065316153193211bf6b57875fbda4cafc01c360445f7"
)


def _protocol() -> Any:
    return load_protocol(PROTOCOL_PATH)


def _unlocked_protocol() -> Any:
    """Return the production protocol with only artifact locks cleared for synthetic fixtures."""

    return replace(
        _protocol(),
        existing_manifest_hash=None,
        existing_manifest_file_sha256=None,
        existing_split_hash=None,
        existing_split_file_sha256=None,
    )


def _artifacts() -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_hash = "a" * 64
    split_hash = "b" * 64
    cases: list[dict[str, Any]] = []
    assignments: list[dict[str, Any]] = []
    index = 0
    for partition, count in (("train", 91), ("validation", 20), ("internal_test", 20)):
        for _ in range(count):
            index += 1
            patient_id = f"anon-p{index:04d}"
            case_id = f"anon-c{index:04d}"
            cases.append(
                {
                    "anonymous_patient_id": patient_id,
                    "anonymous_case_id": case_id,
                    "image_sha256": f"{index:064x}",
                    "label_sha256": f"{index + 10000:064x}",
                }
            )
            assignments.append(
                {
                    "anonymous_patient_id": patient_id,
                    "anonymous_case_id": case_id,
                    "partition": partition,
                }
            )
    return (
        {"manifest_hash": manifest_hash, "cases": cases},
        {
            "source_manifest_hash": manifest_hash,
            "split_hash": split_hash,
            "assignments": assignments,
        },
    )


def test_protocol_loads_expected_locked_counts() -> None:
    protocol = _protocol()
    assert protocol.base_commit == "ae05bf41665f71339da533068fd5b5324c634669"
    assert (protocol.train_count, protocol.validation_count, protocol.internal_test_count) == (
        91,
        20,
        20,
    )
    assert protocol.nnunet_fold_count == 5
    assert protocol.existing_manifest_hash == EXPECTED_MANIFEST_ARTIFACT_HASH
    assert protocol.existing_manifest_file_sha256 == EXPECTED_MANIFEST_FILE_SHA256
    assert protocol.existing_split_hash == EXPECTED_SPLIT_ARTIFACT_HASH
    assert protocol.existing_split_file_sha256 == EXPECTED_SPLIT_FILE_SHA256


def test_gate_policy_v2_allows_r1_without_gpu_but_requires_gpu_before_training() -> None:
    raw = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    policy = raw["gate_policy"]
    assert policy["version"] == "research_v2_gate_policy.v2"
    assert policy["r0_research_readiness"] == {
        "required_for": ["R1"],
        "requires_real_data_audit": True,
        "requires_python311_software_ci": True,
        "requires_cuda_gpu": False,
    }
    assert policy["gpu_training_readiness"] == {
        "required_before": ["R2_training", "R3_training"],
        "requires_cuda_gpu": True,
    }


def test_access_policy_is_fail_closed() -> None:
    protocol = _protocol()
    assert_data_access(
        protocol,
        purpose=AccessPurpose.FIT,
        partitions=[DataPartition.TRAIN],
    )
    assert_data_access(
        protocol,
        purpose=AccessPurpose.TUNE,
        partitions=[DataPartition.TRAIN, DataPartition.VALIDATION],
    )
    with pytest.raises(DataAccessViolation):
        assert_data_access(
            protocol,
            purpose=AccessPurpose.TUNE,
            partitions=[DataPartition.INTERNAL_TEST],
        )
    with pytest.raises(DataAccessViolation):
        assert_data_access(
            protocol,
            purpose=AccessPurpose.TUNE,
            partitions=[DataPartition.HISTORICAL_EXTERNAL],
        )
    with pytest.raises(DataAccessViolation):
        assert_data_access(
            protocol,
            purpose=AccessPurpose.LOCKED_EVALUATION,
            partitions=[DataPartition.INTERNAL_TEST],
            candidate_locked=False,
        )
    assert_data_access(
        protocol,
        purpose=AccessPurpose.LOCKED_EVALUATION,
        partitions=[DataPartition.INTERNAL_TEST],
        candidate_locked=True,
    )


def test_partition_audit_accepts_disjoint_fixture_and_is_deterministic() -> None:
    protocol = _unlocked_protocol()
    manifest, split = _artifacts()
    first = audit_partition_isolation(protocol, manifest=manifest, split=split)
    second = audit_partition_isolation(protocol, manifest=manifest, split=split)
    assert first.patient_counts == {"internal_test": 20, "train": 91, "validation": 20}
    assert first.case_counts == {"internal_test": 20, "train": 91, "validation": 20}
    assert first.audit_hash == second.audit_hash


def test_partition_audit_rejects_cross_partition_image_hash_collision() -> None:
    protocol = _unlocked_protocol()
    manifest, split = _artifacts()
    train_case = next(
        case for case in manifest["cases"] if case["anonymous_case_id"] == "anon-c0001"
    )
    validation_case = next(
        case for case in manifest["cases"] if case["anonymous_case_id"] == "anon-c0092"
    )
    validation_case["image_sha256"] = train_case["image_sha256"]
    with pytest.raises(PartitionIsolationError, match="image hash is shared"):
        audit_partition_isolation(protocol, manifest=manifest, split=split)


def test_partition_audit_rejects_patient_crossing_partitions() -> None:
    protocol = _unlocked_protocol()
    manifest, split = _artifacts()
    broken = copy.deepcopy(split)
    broken["assignments"][91]["anonymous_patient_id"] = "anon-p0001"
    with pytest.raises(PartitionIsolationError):
        audit_partition_isolation(protocol, manifest=manifest, split=broken)


def test_nnunet_folds_use_train_only_and_cover_each_train_case_once() -> None:
    _manifest, split = _artifacts()
    folds = build_nnunet_splits(split, fold_count=5)
    assert len(folds) == 5
    val_cases = [case_id for fold in folds for case_id in fold["val"]]
    assert len(val_cases) == 91
    assert len(set(val_cases)) == 91
    assert all(int(case_id.removeprefix("anon-c")) <= 91 for case_id in val_cases)
    for fold in folds:
        assert set(fold["train"]).isdisjoint(fold["val"])
        assert len(fold["train"]) + len(fold["val"]) == 91


def test_nnunet_split_file_is_direct_list_and_metadata_is_separate(tmp_path: Path) -> None:
    _manifest, split = _artifacts()
    folds = build_nnunet_splits(split, fold_count=5)
    metadata = {
        "fold_count": 5,
        "folds_sha256": "f" * 64,
        "protocol_id": "test",
    }
    write_nnunet_split_files(output_dir=tmp_path, folds=folds, metadata=metadata)

    direct = json.loads((tmp_path / "splits_final.json").read_text(encoding="utf-8"))
    meta = json.loads((tmp_path / "splits_final.meta.json").read_text(encoding="utf-8"))
    assert isinstance(direct, list)
    assert direct == folds
    assert meta == metadata
    assert "folds" not in meta


def test_protocol_rejects_a_policy_that_exposes_internal_test_to_tuning(tmp_path: Path) -> None:
    raw = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    raw["access_policy"]["tune"].append("internal_test")
    bad_path = tmp_path / "bad.yaml"
    bad_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ProtocolValidationError, match="access policy"):
        load_protocol(bad_path)


def test_protocol_artifact_lock_requires_artifact_and_file_hashes(tmp_path: Path) -> None:
    identity = Phase2ArtifactIdentity(
        manifest_artifact_hash="a" * 64,
        manifest_file_sha256="c" * 64,
        split_artifact_hash="b" * 64,
        split_file_sha256="d" * 64,
        source_manifest_hash="a" * 64,
        manifest_case_count=131,
    )
    with pytest.raises(PartitionIsolationError, match="incomplete"):
        assert_protocol_artifact_lock(_unlocked_protocol(), identity)

    raw = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    existing = raw["development_data"]["existing_split"]
    existing["manifest_hash"] = identity.manifest_artifact_hash
    existing["manifest_file_sha256"] = identity.manifest_file_sha256
    existing["split_hash"] = identity.split_artifact_hash
    existing["split_file_sha256"] = identity.split_file_sha256
    locked_path = tmp_path / "locked.yaml"
    locked_path.write_text(json.dumps(raw), encoding="utf-8")
    assert_protocol_artifact_lock(load_protocol(locked_path), identity)


def test_partition_audit_rejects_case_count_mismatch(tmp_path: Path) -> None:
    protocol_raw = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    protocol_raw["development_data"]["counts"] = {
        "train": 90,
        "validation": 20,
        "internal_test": 20,
    }
    existing = protocol_raw["development_data"]["existing_split"]
    existing["manifest_hash"] = None
    existing["manifest_file_sha256"] = None
    existing["split_hash"] = None
    existing["split_file_sha256"] = None
    protocol_path = tmp_path / "wrong-counts.yaml"
    protocol_path.write_text(json.dumps(protocol_raw), encoding="utf-8")
    manifest, split = _artifacts()
    with pytest.raises(PartitionIsolationError, match="patient counts"):
        audit_partition_isolation(load_protocol(protocol_path), manifest=manifest, split=split)
