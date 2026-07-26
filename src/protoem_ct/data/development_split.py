"""Deterministic patient-level Phase 2 development split generation."""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from protoem_ct.artifacts import (
    DEVELOPMENT_SPLIT_MANIFEST_TYPE,
    PHASE2_DEVELOPMENT_COHORT_ROLE,
    PHASE2_SCHEMA_VERSION,
    DatasetManifest,
    DevelopmentSplitManifest,
    Phase2ArtifactError,
    Phase2ArtifactSerializationError,
    SplitAssignment,
    hash_dataset_manifest,
    hash_development_split,
    phase2_artifact_from_json,
    phase2_artifact_to_json,
    sha256_json,
)

PATIENT_HASH_RANK_POLICY_VERSION: Final[str] = "patient_hash_rank_v1"
PATIENT_RANKING_DOMAIN: Final[str] = "protoem-ct.phase2.development-split.patient-rank.v1"
_ZERO_SHA256 = "0" * 64


class DevelopmentSplitError(ValueError):
    """Base exception for deterministic development split generation failures."""


class InvalidDevelopmentSplitPolicyError(DevelopmentSplitError):
    """Raised when a split policy is unsupported or internally invalid."""


class InvalidSplitInputManifestError(DevelopmentSplitError):
    """Raised when the input dataset manifest cannot be used for split generation."""


class SplitManifestHashMismatchError(InvalidSplitInputManifestError):
    """Raised when the stored manifest hash does not match the recomputed hash."""


class IncompatibleSplitPatientCountsError(DevelopmentSplitError):
    """Raised when explicit partition patient counts do not match the manifest patients."""


class SplitAssignmentInvariantError(DevelopmentSplitError):
    """Raised when generated assignments violate patient- or case-level invariants."""


class UnsafeSplitOutputPathError(DevelopmentSplitError):
    """Raised when the requested split output path violates output safety policy."""


class ExistingSplitOutputError(DevelopmentSplitError):
    """Raised when split publication would overwrite an existing output."""


class SplitPublicationError(DevelopmentSplitError):
    """Raised when validated split JSON cannot be published atomically."""


@dataclass(frozen=True, slots=True)
class DevelopmentSplitPolicy:
    """Explicit patient-level split policy with no hidden ratios or dataset statistics."""

    policy_version: str
    split_seed: int
    train_patient_count: int
    validation_patient_count: int
    internal_test_patient_count: int

    def __post_init__(self) -> None:
        if self.policy_version != PATIENT_HASH_RANK_POLICY_VERSION:
            msg = f"unsupported development split policy_version: {self.policy_version!r}"
            raise InvalidDevelopmentSplitPolicyError(msg)
        if (
            isinstance(self.split_seed, bool)
            or not isinstance(self.split_seed, int)
            or self.split_seed < 0
        ):
            msg = "split_seed must be a nonnegative integer"
            raise InvalidDevelopmentSplitPolicyError(msg)
        _require_positive_count(self.train_patient_count, "train_patient_count")
        _require_positive_count(self.validation_patient_count, "validation_patient_count")
        _require_positive_count(self.internal_test_patient_count, "internal_test_patient_count")

    @property
    def total_patient_count(self) -> int:
        """Return the explicit total requested patient count."""
        return (
            self.train_patient_count
            + self.validation_patient_count
            + self.internal_test_patient_count
        )


@dataclass(frozen=True, slots=True)
class DevelopmentSplitResult:
    """Safe result summary for a published development split artifact."""

    split_manifest: DevelopmentSplitManifest
    output_path: Path
    policy_version: str
    split_seed: int
    train_patient_count: int
    validation_patient_count: int
    internal_test_patient_count: int
    total_case_count: int
    source_manifest_hash: str
    split_hash: str


def build_development_split(
    manifest_path: Path,
    *,
    policy: DevelopmentSplitPolicy,
    output_path: Path,
    git_commit: str,
    generated_at_utc: str,
) -> DevelopmentSplitResult:
    """Build and publish a deterministic patient-level development split artifact."""
    source_manifest_path = _validate_input_manifest_path(manifest_path)
    safe_output_path = _validate_split_output_path(
        output_path,
        input_manifest_path=source_manifest_path,
    )
    _require_explicit_metadata(git_commit, "git_commit")
    _require_explicit_metadata(generated_at_utc, "generated_at_utc")

    manifest_text = source_manifest_path.read_text(encoding="utf-8")
    manifest = _load_and_verify_dataset_manifest(manifest_text)
    assignments = _assign_manifest_cases(manifest, policy)
    _verify_assignment_invariants(manifest, assignments, policy)

    patient_counts = _partition_patient_counts(assignments)
    case_counts = _partition_case_counts(assignments)
    split_without_hash = DevelopmentSplitManifest(
        schema_version=PHASE2_SCHEMA_VERSION,
        manifest_type=DEVELOPMENT_SPLIT_MANIFEST_TYPE,
        generated_at_utc=generated_at_utc,
        git_commit=git_commit,
        source_manifest_hash=manifest.manifest_hash,
        split_policy_version=policy.policy_version,
        split_seed=policy.split_seed,
        split_hash=_ZERO_SHA256,
        assignments=assignments,
        train_patient_count=patient_counts["train"],
        validation_patient_count=patient_counts["validation"],
        internal_test_patient_count=patient_counts["internal_test"],
        train_case_count=case_counts["train"],
        validation_case_count=case_counts["validation"],
        internal_test_case_count=case_counts["internal_test"],
    )
    split_hash = hash_development_split(split_without_hash)
    split_manifest = DevelopmentSplitManifest(
        schema_version=split_without_hash.schema_version,
        manifest_type=split_without_hash.manifest_type,
        generated_at_utc=split_without_hash.generated_at_utc,
        git_commit=split_without_hash.git_commit,
        source_manifest_hash=split_without_hash.source_manifest_hash,
        split_policy_version=split_without_hash.split_policy_version,
        split_seed=split_without_hash.split_seed,
        split_hash=split_hash,
        assignments=split_without_hash.assignments,
        train_patient_count=split_without_hash.train_patient_count,
        validation_patient_count=split_without_hash.validation_patient_count,
        internal_test_patient_count=split_without_hash.internal_test_patient_count,
        train_case_count=split_without_hash.train_case_count,
        validation_case_count=split_without_hash.validation_case_count,
        internal_test_case_count=split_without_hash.internal_test_case_count,
    )
    if hash_development_split(split_manifest) != split_hash:
        msg = "split hash verification failed"
        raise SplitPublicationError(msg)

    _publish_split_json(split_manifest, safe_output_path)
    return DevelopmentSplitResult(
        split_manifest=split_manifest,
        output_path=safe_output_path,
        policy_version=split_manifest.split_policy_version,
        split_seed=split_manifest.split_seed,
        train_patient_count=split_manifest.train_patient_count,
        validation_patient_count=split_manifest.validation_patient_count,
        internal_test_patient_count=split_manifest.internal_test_patient_count,
        total_case_count=len(split_manifest.assignments),
        source_manifest_hash=split_manifest.source_manifest_hash,
        split_hash=split_manifest.split_hash,
    )


def patient_ranking_digest(
    anonymous_patient_id: str,
    *,
    policy_version: str,
    split_seed: int,
) -> str:
    """Return the deterministic canonical SHA-256 ranking digest for one anonymous patient."""
    return sha256_json(
        {
            "anonymous_patient_id": anonymous_patient_id,
            "domain": PATIENT_RANKING_DOMAIN,
            "policy_version": policy_version,
            "split_seed": str(split_seed),
        }
    )


def _load_and_verify_dataset_manifest(manifest_text: str) -> DatasetManifest:
    try:
        artifact = phase2_artifact_from_json(manifest_text)
    except Phase2ArtifactError as exc:
        msg = "input manifest is not a valid Phase 2 dataset manifest"
        raise InvalidSplitInputManifestError(msg) from exc
    if not isinstance(artifact, DatasetManifest):
        msg = "input artifact must be a DatasetManifest"
        raise InvalidSplitInputManifestError(msg)
    if artifact.cohort_role != PHASE2_DEVELOPMENT_COHORT_ROLE:
        msg = "input dataset manifest must use the development cohort role"
        raise InvalidSplitInputManifestError(msg)
    recomputed_hash = hash_dataset_manifest(artifact)
    if recomputed_hash != artifact.manifest_hash:
        msg = "input dataset manifest hash does not match its content"
        raise SplitManifestHashMismatchError(msg)
    return artifact


def _assign_manifest_cases(
    manifest: DatasetManifest,
    policy: DevelopmentSplitPolicy,
) -> tuple[SplitAssignment, ...]:
    patients = tuple(sorted({case.anonymous_patient_id for case in manifest.cases}))
    if policy.total_patient_count != len(patients):
        msg = "explicit partition patient counts must equal the manifest patient count"
        raise IncompatibleSplitPatientCountsError(msg)

    ranked_patients = tuple(
        sorted(
            patients,
            key=lambda patient_id: (
                patient_ranking_digest(
                    patient_id,
                    policy_version=policy.policy_version,
                    split_seed=policy.split_seed,
                ),
                patient_id,
            ),
        )
    )
    train_end = policy.train_patient_count
    validation_end = train_end + policy.validation_patient_count
    patient_to_partition = (
        {patient_id: "train" for patient_id in ranked_patients[:train_end]}
        | {patient_id: "validation" for patient_id in ranked_patients[train_end:validation_end]}
        | {patient_id: "internal_test" for patient_id in ranked_patients[validation_end:]}
    )

    assignments = tuple(
        SplitAssignment(
            anonymous_patient_id=case.anonymous_patient_id,
            anonymous_case_id=case.anonymous_case_id,
            partition=patient_to_partition[case.anonymous_patient_id],
        )
        for case in manifest.cases
    )
    return tuple(
        sorted(
            assignments,
            key=lambda assignment: (
                assignment.anonymous_patient_id,
                assignment.anonymous_case_id,
            ),
        )
    )


def _verify_assignment_invariants(
    manifest: DatasetManifest,
    assignments: tuple[SplitAssignment, ...],
    policy: DevelopmentSplitPolicy,
) -> None:
    manifest_patient_ids = {case.anonymous_patient_id for case in manifest.cases}
    manifest_case_ids = {case.anonymous_case_id for case in manifest.cases}
    assigned_patient_ids = {assignment.anonymous_patient_id for assignment in assignments}
    assigned_case_ids = {assignment.anonymous_case_id for assignment in assignments}
    if assigned_patient_ids != manifest_patient_ids or assigned_case_ids != manifest_case_ids:
        msg = "split assignments must cover every manifest patient and case exactly once"
        raise SplitAssignmentInvariantError(msg)
    if len(assignments) != len(manifest.cases) or len(assigned_case_ids) != len(assignments):
        msg = "split assignments must contain one entry per manifest case"
        raise SplitAssignmentInvariantError(msg)
    patient_partitions: dict[str, str] = {}
    for assignment in assignments:
        previous = patient_partitions.setdefault(
            assignment.anonymous_patient_id,
            assignment.partition,
        )
        if previous != assignment.partition:
            msg = "one patient cannot be assigned to multiple partitions"
            raise SplitAssignmentInvariantError(msg)
    counts = _partition_patient_counts(assignments)
    if (
        counts["train"] != policy.train_patient_count
        or counts["validation"] != policy.validation_patient_count
        or counts["internal_test"] != policy.internal_test_patient_count
    ):
        msg = "computed partition patient counts do not match the requested policy"
        raise SplitAssignmentInvariantError(msg)


def _partition_patient_counts(assignments: tuple[SplitAssignment, ...]) -> dict[str, int]:
    counts = Counter(
        (assignment.partition, assignment.anonymous_patient_id) for assignment in assignments
    )
    partition_counts = {"train": 0, "validation": 0, "internal_test": 0}
    for partition, _patient_id in counts:
        partition_counts[partition] += 1
    return partition_counts


def _partition_case_counts(assignments: tuple[SplitAssignment, ...]) -> dict[str, int]:
    counts = Counter(assignment.partition for assignment in assignments)
    return {
        "train": counts["train"],
        "validation": counts["validation"],
        "internal_test": counts["internal_test"],
    }


def _validate_input_manifest_path(path: Path) -> Path:
    if str(path) == "":
        msg = "input manifest path must not be empty"
        raise InvalidSplitInputManifestError(msg)
    if not path.is_absolute():
        msg = "input manifest path must be explicit and absolute"
        raise InvalidSplitInputManifestError(msg)
    try:
        resolved_path = path.resolve(strict=True)
    except FileNotFoundError as exc:
        msg = "input manifest does not exist"
        raise InvalidSplitInputManifestError(msg) from exc
    except OSError as exc:
        msg = "input manifest cannot be resolved"
        raise InvalidSplitInputManifestError(msg) from exc
    if not resolved_path.is_file():
        msg = "input manifest must be a regular JSON file"
        raise InvalidSplitInputManifestError(msg)
    return resolved_path


def _validate_split_output_path(output_path: Path, *, input_manifest_path: Path) -> Path:
    if str(output_path) == "":
        msg = "split output path must not be empty"
        raise UnsafeSplitOutputPathError(msg)
    if not output_path.is_absolute():
        msg = "split output path must be explicit and absolute"
        raise UnsafeSplitOutputPathError(msg)
    if any(part == ".." for part in output_path.parts):
        msg = "split output path must not contain parent traversal"
        raise UnsafeSplitOutputPathError(msg)
    if output_path.resolve(strict=False) == input_manifest_path:
        msg = "split output path must not equal the input manifest path"
        raise UnsafeSplitOutputPathError(msg)
    if output_path.exists() and output_path.is_file():
        msg = "split output path already exists"
        raise ExistingSplitOutputError(msg)

    safe_output_path = _resolve_path_with_existing_ancestor(output_path)
    if safe_output_path == input_manifest_path:
        msg = "split output path must not equal the input manifest path"
        raise UnsafeSplitOutputPathError(msg)
    if output_path.exists():
        msg = "split output path already exists"
        raise ExistingSplitOutputError(msg)
    return safe_output_path


def _resolve_path_with_existing_ancestor(path: Path) -> Path:
    missing_parts: list[str] = []
    current = path
    while not current.exists():
        missing_parts.append(current.name)
        parent = current.parent
        if parent == current:
            msg = "split output path has no existing parent"
            raise UnsafeSplitOutputPathError(msg)
        current = parent
    if current.is_file():
        msg = "split output parent must not be a regular file"
        raise UnsafeSplitOutputPathError(msg)
    if not current.is_dir():
        msg = "split output parent must resolve beneath a directory"
        raise UnsafeSplitOutputPathError(msg)
    try:
        resolved = current.resolve(strict=True)
    except OSError as exc:
        msg = "split output path cannot be resolved"
        raise UnsafeSplitOutputPathError(msg) from exc
    for part in reversed(missing_parts):
        resolved = resolved / part
    return resolved


def _publish_split_json(split_manifest: DevelopmentSplitManifest, output_path: Path) -> None:
    try:
        text = phase2_artifact_to_json(split_manifest)
    except Phase2ArtifactSerializationError as exc:
        msg = "failed to serialize split artifact"
        raise SplitPublicationError(msg) from exc
    temp_path = output_path.with_name(f".{output_path.name}.tmp")
    created_temp = False
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if temp_path.exists():
            msg = "temporary split output already exists"
            raise ExistingSplitOutputError(msg)
        temp_path.write_text(text, encoding="utf-8", newline="\n")
        created_temp = True
        if output_path.exists():
            msg = "split output path already exists"
            raise ExistingSplitOutputError(msg)
        os.link(temp_path, output_path)
        temp_path.unlink()
        created_temp = False
    except ExistingSplitOutputError:
        raise
    except OSError as exc:
        msg = "failed to publish split JSON"
        raise SplitPublicationError(msg) from exc
    finally:
        if created_temp and temp_path.exists():
            temp_path.unlink()


def _require_positive_count(value: object, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        msg = f"{field_name} must be a positive integer"
        raise InvalidDevelopmentSplitPolicyError(msg)


def _require_explicit_metadata(value: object, field_name: str) -> None:
    if not isinstance(value, str) or value == "" or value != value.strip():
        msg = f"{field_name} must be supplied explicitly as a nonempty string"
        raise InvalidSplitInputManifestError(msg)
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        msg = f"{field_name} must not contain control characters"
        raise InvalidSplitInputManifestError(msg)
