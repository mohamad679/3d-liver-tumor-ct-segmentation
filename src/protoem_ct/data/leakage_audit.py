"""Deterministic Phase 2 development leakage-audit generation from JSON artifacts only."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final

from protoem_ct.artifacts import (
    LEAKAGE_AUDIT_FINDING_CODES,
    LEAKAGE_AUDIT_STAGE,
    PHASE2_DEVELOPMENT_COHORT_ROLE,
    PHASE2_SCHEMA_VERSION,
    DatasetManifest,
    DevelopmentSplitManifest,
    LeakageAuditArtifact,
    Phase2ArtifactError,
    SplitAssignment,
    hash_dataset_manifest,
    hash_development_split,
    hash_leakage_audit,
    phase2_artifact_from_json,
    phase2_artifact_to_json,
    sha256_json,
)
from protoem_ct.data._phase2_publication import (
    Phase2PublicationExistingOutputError,
    Phase2PublicationIOError,
    publish_text_no_overwrite,
)

DEVELOPMENT_LEAKAGE_AUDIT_CONTRACT_VERSION: Final[str] = "development_leakage_audit_v1"
DEVELOPMENT_LEAKAGE_AUDIT_CONFIG_PAYLOAD_TYPE: Final[str] = (
    "phase2-development-leakage-audit-config-v1"
)
LEAKAGE_AUDIT_PARTITIONS: Final[tuple[str, ...]] = (
    "train",
    "validation",
    "internal_test",
)
LEAKAGE_AUDIT_PARTITION_PAIRS: Final[tuple[tuple[str, str], ...]] = (
    ("train", "validation"),
    ("train", "internal_test"),
    ("validation", "internal_test"),
)
LEAKAGE_AUDIT_PAIR_KEYS: Final[tuple[str, ...]] = tuple(
    f"{left}__{right}" for left, right in LEAKAGE_AUDIT_PARTITION_PAIRS
)
NEAR_DUPLICATE_POLICY: Final[str] = (
    "exact persisted SHA-256 cross-partition reuse is audited; near-duplicate policy requires "
    "later explicit approval"
)
LITS_MSD_EQUIVALENCE_WARNING: Final[str] = "LiTS and MSD Task03 Liver are not independent cohorts."
_ZERO_SHA256: Final[str] = "0" * 64


class DevelopmentLeakageAuditError(ValueError):
    """Base exception for deterministic development leakage-audit failures."""


class InvalidDevelopmentLeakageAuditConfigError(DevelopmentLeakageAuditError):
    """Raised when leakage-audit configuration is unsupported."""


class InvalidDevelopmentLeakageAuditInputError(DevelopmentLeakageAuditError):
    """Raised when a manifest or split input is malformed or incompatible."""


class DevelopmentLeakageAuditHashMismatchError(InvalidDevelopmentLeakageAuditInputError):
    """Raised when an input artifact's stored hash does not match recomputed content."""


class DevelopmentLeakageAuditLinkageError(InvalidDevelopmentLeakageAuditInputError):
    """Raised when manifest and split artifacts do not link to each other."""


class UnsafeDevelopmentLeakageAuditOutputPathError(DevelopmentLeakageAuditError):
    """Raised when the requested leakage-audit output path is unsafe."""


class ExistingDevelopmentLeakageAuditOutputError(DevelopmentLeakageAuditError):
    """Raised when leakage-audit publication would overwrite an existing file."""


class DevelopmentLeakageAuditPublicationError(DevelopmentLeakageAuditError):
    """Raised when a validated leakage-audit artifact cannot be published atomically."""


@dataclass(frozen=True, slots=True)
class DevelopmentLeakageAuditConfig:
    """Explicit leakage-audit contract with no hidden thresholds or scientific settings."""

    audit_contract_version: str

    def __post_init__(self) -> None:
        if self.audit_contract_version != DEVELOPMENT_LEAKAGE_AUDIT_CONTRACT_VERSION:
            msg = f"unsupported audit_contract_version: {self.audit_contract_version!r}"
            raise InvalidDevelopmentLeakageAuditConfigError(msg)


def development_leakage_audit_config_hash_payload(
    config: DevelopmentLeakageAuditConfig,
) -> dict[str, object]:
    """Return the canonical payload for a development leakage-audit config hash."""
    return {
        "audit_contract_version": config.audit_contract_version,
        "payload_type": DEVELOPMENT_LEAKAGE_AUDIT_CONFIG_PAYLOAD_TYPE,
        "schema_version": PHASE2_SCHEMA_VERSION,
    }


def hash_development_leakage_audit_config(config: DevelopmentLeakageAuditConfig) -> str:
    """Return the deterministic SHA-256 hash for explicit leakage-audit configuration."""
    return sha256_json(development_leakage_audit_config_hash_payload(config))


def run_development_leakage_audit(
    manifest_path: Path,
    split_path: Path,
    *,
    output_path: Path,
    config: DevelopmentLeakageAuditConfig,
    git_commit: str,
    created_at_utc: str,
) -> LeakageAuditArtifact:
    """Run a deterministic leakage audit from anonymous manifest and split JSON only."""
    manifest_json_path = _validate_input_json_path(manifest_path, "manifest")
    split_json_path = _validate_input_json_path(split_path, "split")
    safe_output_path = _validate_audit_output_path(
        output_path,
        input_paths=(manifest_json_path, split_json_path),
    )
    _require_explicit_metadata(git_commit, "git_commit")
    _require_explicit_metadata(created_at_utc, "created_at_utc")

    manifest = _load_and_verify_manifest(manifest_json_path.read_text(encoding="utf-8"))
    split = _load_and_verify_split(split_json_path.read_text(encoding="utf-8"))
    if split.source_manifest_hash != manifest.manifest_hash:
        msg = "split source_manifest_hash must equal manifest hash"
        raise DevelopmentLeakageAuditLinkageError(msg)

    audit_inputs = _compute_audit_inputs(manifest, split)
    artifact_without_hash = LeakageAuditArtifact(
        schema_version=PHASE2_SCHEMA_VERSION,
        stage=LEAKAGE_AUDIT_STAGE,
        created_at_utc=created_at_utc,
        git_commit=git_commit,
        config_hash=hash_development_leakage_audit_config(config),
        manifest_hash=manifest.manifest_hash,
        split_hash=split.split_hash,
        split_policy_version=split.split_policy_version,
        split_seed=split.split_seed,
        manifest_case_count=len(manifest.cases),
        manifest_patient_count=len({case.anonymous_patient_id for case in manifest.cases}),
        assignment_complete=audit_inputs.assignment_complete,
        patient_counts_by_partition=audit_inputs.patient_counts_by_partition,
        case_counts_by_partition=audit_inputs.case_counts_by_partition,
        pairwise_patient_overlap_counts=audit_inputs.pairwise_patient_overlap_counts,
        pairwise_case_overlap_counts=audit_inputs.pairwise_case_overlap_counts,
        image_hash_cross_partition_overlap_count=audit_inputs.image_hash_overlap_count,
        label_hash_cross_partition_overlap_count=audit_inputs.label_hash_overlap_count,
        image_label_pair_cross_partition_overlap_count=audit_inputs.image_label_pair_overlap_count,
        duplicate_image_hash_findings=(
            ("image_hash_cross_partition_overlap",) if audit_inputs.image_hash_overlap_count else ()
        ),
        duplicate_label_hash_findings=(
            ("label_hash_cross_partition_overlap",) if audit_inputs.label_hash_overlap_count else ()
        ),
        finding_codes=audit_inputs.finding_codes,
        near_duplicate_policy=NEAR_DUPLICATE_POLICY,
        lits_msd_equivalence_warning=LITS_MSD_EQUIVALENCE_WARNING,
        external_data_accessed=False,
        preprocessing_fitted_on_nontraining_data=False,
        unresolved_findings=audit_inputs.finding_codes,
        critical_finding_count=len(audit_inputs.finding_codes),
        audit_passed=not audit_inputs.finding_codes,
        audit_hash=_ZERO_SHA256,
    )
    audit_hash = hash_leakage_audit(artifact_without_hash)
    artifact = replace(artifact_without_hash, audit_hash=audit_hash)
    if hash_leakage_audit(artifact) != audit_hash:
        msg = "leakage-audit hash verification failed"
        raise DevelopmentLeakageAuditPublicationError(msg)
    _publish_audit_json(artifact, safe_output_path)
    return artifact


@dataclass(frozen=True, slots=True)
class _AuditInputs:
    assignment_complete: bool
    patient_counts_by_partition: Mapping[str, int]
    case_counts_by_partition: Mapping[str, int]
    pairwise_patient_overlap_counts: Mapping[str, int]
    pairwise_case_overlap_counts: Mapping[str, int]
    image_hash_overlap_count: int
    label_hash_overlap_count: int
    image_label_pair_overlap_count: int
    finding_codes: tuple[str, ...]


def _load_and_verify_manifest(text: str) -> DatasetManifest:
    try:
        artifact = phase2_artifact_from_json(text)
    except Phase2ArtifactError as exc:
        msg = "input manifest is not a valid Phase 2 dataset manifest"
        raise InvalidDevelopmentLeakageAuditInputError(msg) from exc
    if not isinstance(artifact, DatasetManifest):
        msg = "input artifact must be a DatasetManifest"
        raise InvalidDevelopmentLeakageAuditInputError(msg)
    if artifact.cohort_role != PHASE2_DEVELOPMENT_COHORT_ROLE:
        msg = "input dataset manifest must use the development cohort role"
        raise InvalidDevelopmentLeakageAuditInputError(msg)
    if hash_dataset_manifest(artifact) != artifact.manifest_hash:
        msg = "input dataset manifest hash does not match its content"
        raise DevelopmentLeakageAuditHashMismatchError(msg)
    return artifact


def _load_and_verify_split(text: str) -> DevelopmentSplitManifest:
    try:
        artifact = phase2_artifact_from_json(text)
    except Phase2ArtifactError as exc:
        msg = "input split is not a valid Phase 2 development split"
        raise InvalidDevelopmentLeakageAuditInputError(msg) from exc
    if not isinstance(artifact, DevelopmentSplitManifest):
        msg = "input artifact must be a DevelopmentSplitManifest"
        raise InvalidDevelopmentLeakageAuditInputError(msg)
    if hash_development_split(artifact) != artifact.split_hash:
        msg = "input split hash does not match its content"
        raise DevelopmentLeakageAuditHashMismatchError(msg)
    return artifact


def _compute_audit_inputs(
    manifest: DatasetManifest,
    split: DevelopmentSplitManifest,
) -> _AuditInputs:
    manifest_cases_by_case_id = {case.anonymous_case_id: case for case in manifest.cases}
    manifest_case_counts = Counter(case.anonymous_case_id for case in manifest.cases)
    manifest_patient_counts = Counter(case.anonymous_patient_id for case in manifest.cases)
    manifest_pair_counts = Counter(
        (case.anonymous_patient_id, case.anonymous_case_id) for case in manifest.cases
    )
    assignment_case_counts = Counter(
        assignment.anonymous_case_id for assignment in split.assignments
    )
    assignment_patient_partitions = _partitions_by_patient(split.assignments)
    assignment_case_partitions = _partitions_by_case(split.assignments)

    finding_flags: set[str] = set()
    if any(count > 1 for count in manifest_patient_counts.values()):
        finding_flags.add("duplicate_anonymous_patient_id")
    if any(count > 1 for count in manifest_case_counts.values()):
        finding_flags.add("duplicate_anonymous_case_id")
    if any(count > 1 for count in manifest_pair_counts.values()):
        finding_flags.add("duplicate_patient_case_pair")
    if any(len(partitions) > 1 for partitions in assignment_patient_partitions.values()):
        finding_flags.add("conflicting_patient_partition")
    if any(len(partitions) > 1 for partitions in assignment_case_partitions.values()):
        finding_flags.add("conflicting_case_partition")

    missing_cases = tuple(
        sorted(
            case_id
            for case_id in manifest_cases_by_case_id
            if assignment_case_counts.get(case_id, 0) != 1
        )
    )
    extra_cases = tuple(
        sorted(
            case_id
            for case_id in assignment_case_counts
            if case_id not in manifest_cases_by_case_id
        )
    )
    patient_mismatches = tuple(
        sorted(
            assignment.anonymous_case_id
            for assignment in split.assignments
            if assignment.anonymous_case_id in manifest_cases_by_case_id
            and manifest_cases_by_case_id[assignment.anonymous_case_id].anonymous_patient_id
            != assignment.anonymous_patient_id
        )
    )
    if missing_cases:
        finding_flags.add("missing_case_assignment")
    if extra_cases:
        finding_flags.add("extra_case_assignment")
    if patient_mismatches:
        finding_flags.add("patient_id_mismatch")

    patient_sets = _patient_sets_by_partition(split.assignments)
    case_sets = _case_sets_by_partition(split.assignments)
    patient_counts = {
        partition: len(patient_sets[partition]) for partition in LEAKAGE_AUDIT_PARTITIONS
    }
    case_counts = {partition: len(case_sets[partition]) for partition in LEAKAGE_AUDIT_PARTITIONS}
    if any(
        patient_counts[partition] == 0 or case_counts[partition] == 0
        for partition in LEAKAGE_AUDIT_PARTITIONS
    ):
        finding_flags.add("empty_partition")

    patient_overlaps = _pairwise_overlap_counts(patient_sets)
    case_overlaps = _pairwise_overlap_counts(case_sets)
    if any(count != 0 for count in patient_overlaps.values()):
        finding_flags.add("patient_partition_overlap")
    if any(count != 0 for count in case_overlaps.values()):
        finding_flags.add("case_partition_overlap")

    case_partition = _single_partition_by_case(split.assignments)
    image_overlap = _cross_partition_hash_overlap_count(
        (case.image_sha256, case_partition[case.anonymous_case_id])
        for case in manifest.cases
        if case.anonymous_case_id in case_partition
    )
    label_overlap = _cross_partition_hash_overlap_count(
        (case.label_sha256, case_partition[case.anonymous_case_id])
        for case in manifest.cases
        if case.anonymous_case_id in case_partition
    )
    pair_overlap = _cross_partition_hash_overlap_count(
        ((case.image_sha256, case.label_sha256), case_partition[case.anonymous_case_id])
        for case in manifest.cases
        if case.anonymous_case_id in case_partition
    )
    if image_overlap:
        finding_flags.add("image_hash_cross_partition_overlap")
    if label_overlap:
        finding_flags.add("label_hash_cross_partition_overlap")
    if pair_overlap:
        finding_flags.add("image_label_pair_cross_partition_overlap")

    assignment_complete = not (missing_cases or extra_cases or patient_mismatches)
    ordered_findings = tuple(code for code in LEAKAGE_AUDIT_FINDING_CODES if code in finding_flags)
    return _AuditInputs(
        assignment_complete=assignment_complete,
        patient_counts_by_partition=patient_counts,
        case_counts_by_partition=case_counts,
        pairwise_patient_overlap_counts=patient_overlaps,
        pairwise_case_overlap_counts=case_overlaps,
        image_hash_overlap_count=image_overlap,
        label_hash_overlap_count=label_overlap,
        image_label_pair_overlap_count=pair_overlap,
        finding_codes=ordered_findings,
    )


def _partitions_by_patient(
    assignments: Iterable[SplitAssignment],
) -> dict[str, set[str]]:
    values: dict[str, set[str]] = defaultdict(set)
    for assignment in assignments:
        values[assignment.anonymous_patient_id].add(assignment.partition)
    return values


def _partitions_by_case(assignments: Iterable[SplitAssignment]) -> dict[str, set[str]]:
    values: dict[str, set[str]] = defaultdict(set)
    for assignment in assignments:
        values[assignment.anonymous_case_id].add(assignment.partition)
    return values


def _patient_sets_by_partition(
    assignments: Iterable[SplitAssignment],
) -> dict[str, set[str]]:
    values = {partition: set[str]() for partition in LEAKAGE_AUDIT_PARTITIONS}
    for assignment in assignments:
        values[assignment.partition].add(assignment.anonymous_patient_id)
    return values


def _case_sets_by_partition(assignments: Iterable[SplitAssignment]) -> dict[str, set[str]]:
    values = {partition: set[str]() for partition in LEAKAGE_AUDIT_PARTITIONS}
    for assignment in assignments:
        values[assignment.partition].add(assignment.anonymous_case_id)
    return values


def _single_partition_by_case(assignments: Iterable[SplitAssignment]) -> dict[str, str]:
    values: dict[str, str] = {}
    for assignment in assignments:
        values.setdefault(assignment.anonymous_case_id, assignment.partition)
    return values


def _pairwise_overlap_counts(partition_sets: Mapping[str, set[str]]) -> dict[str, int]:
    return {
        f"{left}__{right}": len(partition_sets[left] & partition_sets[right])
        for left, right in LEAKAGE_AUDIT_PARTITION_PAIRS
    }


def _cross_partition_hash_overlap_count(
    values: Iterable[tuple[object, str]],
) -> int:
    partitions_by_hash: dict[object, set[str]] = defaultdict(set)
    for hash_value, partition in values:
        partitions_by_hash[hash_value].add(partition)
    return sum(1 for partitions in partitions_by_hash.values() if len(partitions) > 1)


def _validate_input_json_path(path: Path, role: str) -> Path:
    if str(path) == "":
        msg = f"input {role} path must not be empty"
        raise InvalidDevelopmentLeakageAuditInputError(msg)
    if not path.is_absolute():
        msg = f"input {role} path must be explicit and absolute"
        raise InvalidDevelopmentLeakageAuditInputError(msg)
    try:
        resolved_path = path.resolve(strict=True)
    except FileNotFoundError as exc:
        msg = f"input {role} JSON does not exist"
        raise InvalidDevelopmentLeakageAuditInputError(msg) from exc
    except OSError as exc:
        msg = f"input {role} JSON cannot be resolved"
        raise InvalidDevelopmentLeakageAuditInputError(msg) from exc
    if not resolved_path.is_file():
        msg = f"input {role} JSON must be a regular file"
        raise InvalidDevelopmentLeakageAuditInputError(msg)
    return resolved_path


def _validate_audit_output_path(output_path: Path, *, input_paths: tuple[Path, ...]) -> Path:
    if str(output_path) == "":
        msg = "leakage-audit output path must not be empty"
        raise UnsafeDevelopmentLeakageAuditOutputPathError(msg)
    if not output_path.is_absolute():
        msg = "leakage-audit output path must be explicit and absolute"
        raise UnsafeDevelopmentLeakageAuditOutputPathError(msg)
    if output_path.suffix != ".json":
        msg = "leakage-audit output path must use a .json suffix"
        raise UnsafeDevelopmentLeakageAuditOutputPathError(msg)
    if any(part == ".." for part in output_path.parts):
        msg = "leakage-audit output path must not contain parent traversal"
        raise UnsafeDevelopmentLeakageAuditOutputPathError(msg)
    if any(output_path.resolve(strict=False) == input_path for input_path in input_paths):
        msg = "leakage-audit output path must not equal an input JSON path"
        raise UnsafeDevelopmentLeakageAuditOutputPathError(msg)
    if output_path.exists() and output_path.is_file():
        msg = "leakage-audit output path already exists"
        raise ExistingDevelopmentLeakageAuditOutputError(msg)

    safe_output_path = _resolve_path_with_existing_ancestor(output_path)
    if any(safe_output_path == input_path for input_path in input_paths):
        msg = "leakage-audit output path must not equal an input JSON path"
        raise UnsafeDevelopmentLeakageAuditOutputPathError(msg)
    if output_path.exists():
        msg = "leakage-audit output path already exists"
        raise ExistingDevelopmentLeakageAuditOutputError(msg)
    return safe_output_path


def _resolve_path_with_existing_ancestor(path: Path) -> Path:
    missing_parts: list[str] = []
    current = path
    while not current.exists():
        missing_parts.append(current.name)
        parent = current.parent
        if parent == current:
            msg = "leakage-audit output path has no existing parent"
            raise UnsafeDevelopmentLeakageAuditOutputPathError(msg)
        current = parent
    if current.is_file():
        msg = "leakage-audit output parent must not be a regular file"
        raise UnsafeDevelopmentLeakageAuditOutputPathError(msg)
    if not current.is_dir():
        msg = "leakage-audit output parent must resolve beneath a directory"
        raise UnsafeDevelopmentLeakageAuditOutputPathError(msg)
    try:
        resolved = current.resolve(strict=True)
    except OSError as exc:
        msg = "leakage-audit output path cannot be resolved"
        raise UnsafeDevelopmentLeakageAuditOutputPathError(msg) from exc
    for part in reversed(missing_parts):
        resolved = resolved / part
    return resolved


def _publish_audit_json(artifact: LeakageAuditArtifact, output_path: Path) -> None:
    try:
        text = phase2_artifact_to_json(artifact)
    except Phase2ArtifactError as exc:
        msg = "failed to serialize leakage-audit artifact"
        raise DevelopmentLeakageAuditPublicationError(msg) from exc
    try:
        publish_text_no_overwrite(
            text=text,
            output_path=output_path,
            temporary_exists_message="temporary leakage-audit output already exists",
            final_exists_message="leakage-audit output path already exists",
        )
    except Phase2PublicationExistingOutputError as exc:
        raise ExistingDevelopmentLeakageAuditOutputError(str(exc)) from exc
    except Phase2PublicationIOError as exc:
        msg = "failed to publish leakage-audit JSON"
        raise DevelopmentLeakageAuditPublicationError(msg) from exc


def _require_explicit_metadata(value: object, field_name: str) -> None:
    if not isinstance(value, str) or value == "" or value != value.strip():
        msg = f"{field_name} must be supplied explicitly as a nonempty string"
        raise InvalidDevelopmentLeakageAuditInputError(msg)
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        msg = f"{field_name} must not contain control characters"
        raise InvalidDevelopmentLeakageAuditInputError(msg)


__all__ = [
    "DEVELOPMENT_LEAKAGE_AUDIT_CONFIG_PAYLOAD_TYPE",
    "DEVELOPMENT_LEAKAGE_AUDIT_CONTRACT_VERSION",
    "DevelopmentLeakageAuditConfig",
    "DevelopmentLeakageAuditError",
    "DevelopmentLeakageAuditHashMismatchError",
    "DevelopmentLeakageAuditLinkageError",
    "DevelopmentLeakageAuditPublicationError",
    "ExistingDevelopmentLeakageAuditOutputError",
    "InvalidDevelopmentLeakageAuditConfigError",
    "InvalidDevelopmentLeakageAuditInputError",
    "LEAKAGE_AUDIT_PAIR_KEYS",
    "LEAKAGE_AUDIT_PARTITIONS",
    "LEAKAGE_AUDIT_PARTITION_PAIRS",
    "UnsafeDevelopmentLeakageAuditOutputPathError",
    "development_leakage_audit_config_hash_payload",
    "hash_development_leakage_audit_config",
    "run_development_leakage_audit",
]
