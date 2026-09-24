"""Research-v2 R0 protocol validation, access guards, and split audits."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Final, cast

RESEARCH_V2_SCHEMA_VERSION: Final = "research_v2_protocol.v1"
_NNUNET_FOLD_DOMAIN: Final = "protoem-ct.research-v2.nnunet-fold.v1"
_DEVELOPMENT_PARTITIONS: Final = frozenset({"train", "validation", "internal_test"})


class ProtocolValidationError(ValueError):
    """Raised when the Research-v2 protocol is malformed or inconsistent."""


class DataAccessViolation(PermissionError):
    """Raised when a stage requests a forbidden data scope."""


class PartitionIsolationError(ValueError):
    """Raised when artifact identity or partition isolation is violated."""


class DataPartition(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    INTERNAL_TEST = "internal_test"
    HISTORICAL_EXTERNAL = "historical_external"
    NEW_EXTERNAL = "new_external"


class AccessPurpose(StrEnum):
    FIT = "fit"
    TUNE = "tune"
    FOLD_PLANNING = "fold_planning"
    LOCKED_EVALUATION = "locked_evaluation"
    EXTERNAL_CONFIRMATION = "external_confirmation"
    EXPLORATORY_EXTERNAL = "exploratory_external"


@dataclass(frozen=True, slots=True)
class ResearchV2Protocol:
    schema_version: str
    protocol_id: str
    base_commit: str
    train_count: int
    validation_count: int
    internal_test_count: int
    allowed_access: Mapping[AccessPurpose, frozenset[DataPartition]]
    candidate_lock_required_for: frozenset[AccessPurpose]
    existing_manifest_hash: str | None
    existing_manifest_file_sha256: str | None
    existing_split_hash: str | None
    existing_split_file_sha256: str | None
    nnunet_fold_count: int


@dataclass(frozen=True, slots=True)
class PartitionIsolationReport:
    patient_counts: Mapping[str, int]
    case_counts: Mapping[str, int]
    manifest_hash: str
    split_hash: str
    audit_hash: str


@dataclass(frozen=True, slots=True)
class Phase2ArtifactIdentity:
    manifest_artifact_hash: str
    manifest_file_sha256: str
    split_artifact_hash: str
    split_file_sha256: str
    source_manifest_hash: str
    manifest_case_count: int


@dataclass(frozen=True, slots=True)
class VerifiedPhase2Artifacts:
    manifest: Mapping[str, Any]
    split: Mapping[str, Any]
    identity: Phase2ArtifactIdentity


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def load_protocol(path: Path) -> ResearchV2Protocol:
    try:
        root = _mapping(json.loads(path.read_text(encoding="utf-8")), "protocol root")
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProtocolValidationError("protocol must be readable JSON-compatible YAML") from exc

    schema_version = _string(root.get("schema_version"), "schema_version")
    if schema_version != RESEARCH_V2_SCHEMA_VERSION:
        raise ProtocolValidationError(f"unsupported schema_version: {schema_version!r}")
    protocol_id = _string(root.get("protocol_id"), "protocol_id")
    base_commit = _hex_digest(root.get("base_commit"), "base_commit", {40})

    development = _mapping(root.get("development_data"), "development_data")
    counts = _mapping(development.get("counts"), "development_data.counts")
    train_count = _positive_int(counts.get("train"), "development_data.counts.train")
    validation_count = _positive_int(counts.get("validation"), "development_data.counts.validation")
    internal_test_count = _positive_int(
        counts.get("internal_test"), "development_data.counts.internal_test"
    )

    existing = _mapping(development.get("existing_split"), "development_data.existing_split")
    if existing.get("replace_existing_split") is not False:
        raise ProtocolValidationError("existing development split must not be replaced")

    allowed_access = _load_access_policy(root)
    lock_required = _load_candidate_lock(root)

    nnunet = _mapping(root.get("nnunet"), "nnunet")
    fold_count = _positive_int(nnunet.get("fold_count"), "nnunet.fold_count")
    if fold_count != 5:
        raise ProtocolValidationError("R2 requires five nnU-Net folds")

    return ResearchV2Protocol(
        schema_version=schema_version,
        protocol_id=protocol_id,
        base_commit=base_commit,
        train_count=train_count,
        validation_count=validation_count,
        internal_test_count=internal_test_count,
        allowed_access=allowed_access,
        candidate_lock_required_for=lock_required,
        existing_manifest_hash=_optional_sha256(existing.get("manifest_hash"), "manifest_hash"),
        existing_manifest_file_sha256=_optional_sha256(
            existing.get("manifest_file_sha256"), "manifest_file_sha256"
        ),
        existing_split_hash=_optional_sha256(existing.get("split_hash"), "split_hash"),
        existing_split_file_sha256=_optional_sha256(
            existing.get("split_file_sha256"), "split_file_sha256"
        ),
        nnunet_fold_count=fold_count,
    )


def assert_data_access(
    protocol: ResearchV2Protocol,
    *,
    purpose: AccessPurpose,
    partitions: Sequence[DataPartition],
    candidate_locked: bool = False,
) -> None:
    if not partitions:
        raise DataAccessViolation("at least one partition must be requested explicitly")
    requested = frozenset(partitions)
    allowed = protocol.allowed_access[purpose]
    if not requested.issubset(allowed):
        denied = ", ".join(sorted(item.value for item in requested - allowed))
        raise DataAccessViolation(f"{purpose.value} cannot access: {denied}")
    if purpose in protocol.candidate_lock_required_for and not candidate_locked:
        raise DataAccessViolation(f"{purpose.value} requires a frozen candidate")


def load_verified_phase2_artifacts(
    manifest_path: Path,
    split_path: Path,
) -> VerifiedPhase2Artifacts:
    """Recompute schema artifact hashes and raw-file SHA-256 values independently."""

    from protoem_ct.artifacts import (
        DatasetManifest,
        DevelopmentSplitManifest,
        hash_dataset_manifest,
        hash_development_split,
        phase2_artifact_from_json,
        phase2_artifact_to_dict,
        sha256_file,
    )

    try:
        manifest_obj = phase2_artifact_from_json(
            manifest_path.read_text(encoding="utf-8"), DatasetManifest
        )
        split_obj = phase2_artifact_from_json(
            split_path.read_text(encoding="utf-8"), DevelopmentSplitManifest
        )
        manifest_obj = cast(DatasetManifest, manifest_obj)
        split_obj = cast(DevelopmentSplitManifest, split_obj)
        manifest_artifact_hash = hash_dataset_manifest(manifest_obj)
        split_artifact_hash = hash_development_split(split_obj)
        manifest_file_sha256 = sha256_file(manifest_path)
        split_file_sha256 = sha256_file(split_path)
    except (OSError, UnicodeError, ValueError) as exc:
        raise PartitionIsolationError("failed to load or hash Phase 2 artifacts") from exc

    if manifest_artifact_hash != manifest_obj.manifest_hash:
        raise PartitionIsolationError("manifest artifact hash does not match recomputed content")
    if split_artifact_hash != split_obj.split_hash:
        raise PartitionIsolationError("split artifact hash does not match recomputed content")
    if split_obj.source_manifest_hash != manifest_artifact_hash:
        raise PartitionIsolationError("split source_manifest_hash does not match manifest")

    return VerifiedPhase2Artifacts(
        manifest=phase2_artifact_to_dict(manifest_obj),
        split=phase2_artifact_to_dict(split_obj),
        identity=Phase2ArtifactIdentity(
            manifest_artifact_hash=manifest_artifact_hash,
            manifest_file_sha256=manifest_file_sha256,
            split_artifact_hash=split_artifact_hash,
            split_file_sha256=split_file_sha256,
            source_manifest_hash=split_obj.source_manifest_hash,
            manifest_case_count=manifest_obj.case_count,
        ),
    )


def assert_protocol_artifact_lock(
    protocol: ResearchV2Protocol,
    identity: Phase2ArtifactIdentity,
) -> None:
    expected = {
        "manifest_artifact_hash": protocol.existing_manifest_hash,
        "manifest_file_sha256": protocol.existing_manifest_file_sha256,
        "split_artifact_hash": protocol.existing_split_hash,
        "split_file_sha256": protocol.existing_split_file_sha256,
    }
    observed = {
        "manifest_artifact_hash": identity.manifest_artifact_hash,
        "manifest_file_sha256": identity.manifest_file_sha256,
        "split_artifact_hash": identity.split_artifact_hash,
        "split_file_sha256": identity.split_file_sha256,
    }
    unlocked = sorted(key for key, value in expected.items() if value is None)
    if unlocked:
        message = "protocol artifact lock is incomplete: " + ", ".join(unlocked)
        raise PartitionIsolationError(message)
    mismatched = sorted(key for key, value in expected.items() if value != observed[key])
    if mismatched:
        raise PartitionIsolationError(
            "protocol artifact lock differs from verified files: " + ", ".join(mismatched)
        )


def audit_partition_isolation(
    protocol: ResearchV2Protocol,
    *,
    manifest: Mapping[str, Any],
    split: Mapping[str, Any],
) -> PartitionIsolationReport:
    manifest_hash = _hex_digest(manifest.get("manifest_hash"), "manifest.manifest_hash", {64})
    split_hash = _hex_digest(split.get("split_hash"), "split.split_hash", {64})
    source_hash = _hex_digest(split.get("source_manifest_hash"), "split.source_manifest_hash", {64})
    if source_hash != manifest_hash:
        raise PartitionIsolationError("split source_manifest_hash does not match manifest_hash")
    if protocol.existing_manifest_hash not in (None, manifest_hash):
        raise PartitionIsolationError("manifest hash differs from the protocol lock")
    if protocol.existing_split_hash not in (None, split_hash):
        raise PartitionIsolationError("split hash differs from the protocol lock")

    cases_raw = manifest.get("cases")
    assignments_raw = split.get("assignments")
    if not isinstance(cases_raw, list) or not cases_raw:
        raise PartitionIsolationError("manifest.cases must be a nonempty list")
    if not isinstance(assignments_raw, list) or not assignments_raw:
        raise PartitionIsolationError("split.assignments must be a nonempty list")

    cases: dict[str, tuple[str, str, str]] = {}
    for index, raw in enumerate(cases_raw):
        case = _mapping(raw, f"manifest.cases[{index}]")
        case_id = _string(case.get("anonymous_case_id"), "anonymous_case_id")
        if case_id in cases:
            raise PartitionIsolationError(f"duplicate manifest case id: {case_id}")
        cases[case_id] = (
            _string(case.get("anonymous_patient_id"), "anonymous_patient_id"),
            _hex_digest(case.get("image_sha256"), "image_sha256", {64}),
            _hex_digest(case.get("label_sha256"), "label_sha256", {64}),
        )

    assignments: dict[str, tuple[str, str]] = {}
    patient_partition: dict[str, str] = {}
    for index, raw in enumerate(assignments_raw):
        assignment = _mapping(raw, f"split.assignments[{index}]")
        case_id = _string(assignment.get("anonymous_case_id"), "anonymous_case_id")
        patient_id = _string(assignment.get("anonymous_patient_id"), "anonymous_patient_id")
        partition = _string(assignment.get("partition"), "partition")
        if partition not in _DEVELOPMENT_PARTITIONS:
            raise PartitionIsolationError(f"unknown development partition: {partition}")
        if case_id in assignments:
            raise PartitionIsolationError(f"duplicate split case id: {case_id}")
        assignments[case_id] = (patient_id, partition)
        previous = patient_partition.setdefault(patient_id, partition)
        if previous != partition:
            raise PartitionIsolationError(f"patient appears in multiple partitions: {patient_id}")

    if set(cases) != set(assignments):
        raise PartitionIsolationError("split assignments must cover exactly the manifest cases")

    image_partition: dict[str, str] = {}
    label_partition: dict[str, str] = {}
    case_counts: Counter[str] = Counter()
    patients: dict[str, set[str]] = defaultdict(set)
    for case_id, (manifest_patient, image_hash, label_hash) in cases.items():
        assigned_patient, partition = assignments[case_id]
        if assigned_patient != manifest_patient:
            raise PartitionIsolationError(f"patient mismatch for case: {case_id}")
        _assert_hash_partition(image_partition, image_hash, partition, "image")
        _assert_hash_partition(label_partition, label_hash, partition, "label")
        case_counts[partition] += 1
        patients[partition].add(manifest_patient)

    patient_counts = Counter({partition: len(ids) for partition, ids in patients.items()})
    expected = {
        "train": protocol.train_count,
        "validation": protocol.validation_count,
        "internal_test": protocol.internal_test_count,
    }
    if dict(patient_counts) != expected:
        observed = dict(patient_counts)
        raise PartitionIsolationError(
            f"patient counts differ from protocol: observed={observed!r} expected={expected!r}"
        )
    if dict(case_counts) != expected:
        observed_cases = dict(case_counts)
        raise PartitionIsolationError(
            f"case counts differ from protocol: observed={observed_cases!r} expected={expected!r}"
        )

    payload = {
        "case_counts": dict(sorted(case_counts.items())),
        "manifest_hash": manifest_hash,
        "patient_counts": dict(sorted(patient_counts.items())),
        "split_hash": split_hash,
    }
    return PartitionIsolationReport(
        patient_counts=dict(sorted(patient_counts.items())),
        case_counts=dict(sorted(case_counts.items())),
        manifest_hash=manifest_hash,
        split_hash=split_hash,
        audit_hash=canonical_json_sha256(payload),
    )


def build_nnunet_splits(
    split: Mapping[str, Any],
    *,
    fold_count: int = 5,
) -> list[dict[str, list[str]]]:
    if isinstance(fold_count, bool) or not isinstance(fold_count, int) or fold_count < 2:
        raise ProtocolValidationError("fold_count must be an integer >= 2")
    assignments = split.get("assignments")
    if not isinstance(assignments, list) or not assignments:
        raise ProtocolValidationError("split.assignments must be a nonempty list")

    patient_cases: dict[str, list[str]] = defaultdict(list)
    for index, raw in enumerate(assignments):
        item = _mapping(raw, f"split.assignments[{index}]")
        if item.get("partition") == "train":
            patient_cases[_string(item.get("anonymous_patient_id"), "anonymous_patient_id")].append(
                _string(item.get("anonymous_case_id"), "anonymous_case_id")
            )
    if len(patient_cases) < fold_count:
        raise ProtocolValidationError("development train has fewer patients than requested folds")

    ranked = sorted(
        patient_cases,
        key=lambda patient_id: (
            canonical_json_sha256({"domain": _NNUNET_FOLD_DOMAIN, "patient_id": patient_id}),
            patient_id,
        ),
    )
    fold_patients = [set[str]() for _ in range(fold_count)]
    for index, patient_id in enumerate(ranked):
        fold_patients[index % fold_count].add(patient_id)

    all_cases = {case_id for case_ids in patient_cases.values() for case_id in case_ids}
    folds: list[dict[str, list[str]]] = []
    for val_patients in fold_patients:
        val_cases = {case_id for patient_id in val_patients for case_id in patient_cases[patient_id]}
        train_cases = all_cases - val_cases
        if train_cases & val_cases:
            raise ProtocolValidationError("nnU-Net fold contains overlapping train/val cases")
        folds.append({"train": sorted(train_cases), "val": sorted(val_cases)})
    return folds


def _load_access_policy(root: Mapping[str, Any]) -> dict[AccessPurpose, frozenset[DataPartition]]:
    raw_policy = _mapping(root.get("access_policy"), "access_policy")
    result: dict[AccessPurpose, frozenset[DataPartition]] = {}
    for purpose in AccessPurpose:
        raw = raw_policy.get(purpose.value)
        if not isinstance(raw, list) or not raw:
            raise ProtocolValidationError(f"access_policy.{purpose.value} must be a nonempty list")
        try:
            result[purpose] = frozenset(DataPartition(str(item)) for item in raw)
        except ValueError as exc:
            raise ProtocolValidationError("access policy contains an unknown partition") from exc
    expected = {
        AccessPurpose.FIT: frozenset({DataPartition.TRAIN}),
        AccessPurpose.TUNE: frozenset({DataPartition.TRAIN, DataPartition.VALIDATION}),
        AccessPurpose.FOLD_PLANNING: frozenset({DataPartition.TRAIN}),
        AccessPurpose.LOCKED_EVALUATION: frozenset({DataPartition.INTERNAL_TEST}),
        AccessPurpose.EXTERNAL_CONFIRMATION: frozenset({DataPartition.NEW_EXTERNAL}),
        AccessPurpose.EXPLORATORY_EXTERNAL: frozenset({DataPartition.HISTORICAL_EXTERNAL}),
    }
    if result != expected:
        raise ProtocolValidationError("access policy does not match the preregistered R2 contract")
    return result


def _load_candidate_lock(root: Mapping[str, Any]) -> frozenset[AccessPurpose]:
    raw = _mapping(root.get("candidate_lock"), "candidate_lock").get("required_for")
    if not isinstance(raw, list):
        raise ProtocolValidationError("candidate_lock.required_for must be a list")
    try:
        result = frozenset(AccessPurpose(str(item)) for item in raw)
    except ValueError as exc:
        raise ProtocolValidationError("candidate lock contains an unknown purpose") from exc
    expected = frozenset({AccessPurpose.LOCKED_EVALUATION, AccessPurpose.EXTERNAL_CONFIRMATION})
    if result != expected:
        raise ProtocolValidationError("candidate lock must guard internal and external evaluation")
    return result


def _assert_hash_partition(seen: dict[str, str], digest: str, partition: str, kind: str) -> None:
    previous = seen.setdefault(digest, partition)
    if previous != partition:
        raise PartitionIsolationError(
            f"{kind} hash is shared across partitions: {previous} vs {partition}"
        )


def _mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ProtocolValidationError(f"{field_name} must be an object")
    return cast(Mapping[str, Any], value)


def _string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProtocolValidationError(f"{field_name} must be a nonempty string")
    return value


def _positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ProtocolValidationError(f"{field_name} must be a positive integer")
    return value


def _optional_sha256(value: object, field_name: str) -> str | None:
    return None if value is None else _hex_digest(value, field_name, {64})


def _hex_digest(value: object, field_name: str, lengths: set[int]) -> str:
    text = _string(value, field_name).lower()
    if len(text) not in lengths or any(char not in "0123456789abcdef" for char in text):
        expected = "/".join(str(length) for length in sorted(lengths))
        raise ProtocolValidationError(f"{field_name} must be a {expected}-character hex digest")
    return text
