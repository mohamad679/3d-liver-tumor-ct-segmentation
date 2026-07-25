"""Phase 2 immutable data contracts and deterministic JSON serialization."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any, ClassVar, NoReturn, TypeAlias, TypeVar, cast, overload

from protoem_ct.artifacts.hashing import JsonValue
from protoem_ct.artifacts.schemas import (
    ArtifactError,
    ArtifactSchemaVersionError,
    ArtifactSerializationError,
    ArtifactStageError,
    ArtifactValidationError,
)

PHASE2_SCHEMA_VERSION = "2"
PHASE2_DEVELOPMENT_COHORT_ROLE = "development"
DATASET_MANIFEST_TYPE = "phase2-dataset-manifest"
DEVELOPMENT_SPLIT_MANIFEST_TYPE = "phase2-development-split-manifest"
DEVELOPMENT_QA_STAGE = "phase2-development-qa"
LEAKAGE_AUDIT_STAGE = "phase2-leakage-audit"
SUPPORTED_SPLIT_PARTITIONS = ("train", "validation", "internal_test")
SUPPORTED_CONNECTIVITY_VALUES = (6, 18, 26)

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{1,126}[a-z0-9]$")
_DATE_PATTERN = re.compile(r"\b(?:19|20)\d{2}[-_]?(?:0[1-9]|1[0-2])[-_]?(?:0[1-9]|[12]\d|3[01])\b")
_IDENTIFIER_KEYWORDS = (
    "accession",
    "birth",
    "dateofbirth",
    "dicom",
    "dob",
    "medicalrecord",
    "mrn",
    "name",
    "patientname",
    "record",
)
_ORIENTATION_AXIS_CODES = frozenset({"L", "R", "P", "A", "I", "S"})
_ORIENTATION_GROUPS = {"L": "x", "R": "x", "P": "y", "A": "y", "I": "z", "S": "z"}
_PAIRWISE_PARTITION_KEYS = (
    "train__validation",
    "train__internal_test",
    "validation__internal_test",
)


class Phase2ArtifactError(ArtifactError):
    """Base error for Phase 2 artifact contract failures."""


class Phase2ArtifactValidationError(ArtifactValidationError, Phase2ArtifactError):
    """Raised when Phase 2 artifact content violates the contract."""


class Phase2ArtifactSerializationError(ArtifactSerializationError, Phase2ArtifactError):
    """Raised when Phase 2 artifact JSON cannot be parsed or serialized."""


class Phase2ArtifactSchemaVersionError(ArtifactSchemaVersionError, Phase2ArtifactError):
    """Raised when Phase 2 artifact JSON uses an unsupported schema version."""


class Phase2ArtifactStageError(ArtifactStageError, Phase2ArtifactError):
    """Raised when Phase 2 artifact JSON has the wrong manifest type or stage."""


@dataclass(frozen=True, slots=True)
class DatasetCaseRecord:
    """Anonymous read-only record for one Phase 2 development case."""

    anonymous_patient_id: str
    anonymous_case_id: str
    relative_image_path: str
    relative_label_path: str
    image_sha256: str
    label_sha256: str
    cohort_role: str

    def __post_init__(self) -> None:
        _require_safe_anonymous_id(self.anonymous_patient_id, "anonymous_patient_id")
        _require_safe_anonymous_id(self.anonymous_case_id, "anonymous_case_id")
        _require_safe_relative_posix_path(self.relative_image_path, "relative_image_path")
        _require_safe_relative_posix_path(self.relative_label_path, "relative_label_path")
        if self.relative_image_path == self.relative_label_path:
            msg = "relative_image_path and relative_label_path must differ"
            raise Phase2ArtifactValidationError(msg)
        _require_sha256(self.image_sha256, "image_sha256")
        _require_sha256(self.label_sha256, "label_sha256")
        _require_development_cohort_role(self.cohort_role, "cohort_role")


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    """Anonymous manifest for the LiTS development cohort contract."""

    schema_version: str
    manifest_type: str
    dataset_id: str
    cohort_role: str
    adapter_name: str
    adapter_version: str
    generated_at_utc: str
    git_commit: str
    dataset_root_fingerprint: str
    manifest_hash: str
    case_count: int
    cases: tuple[DatasetCaseRecord, ...]

    _expected_manifest_type: ClassVar[str] = DATASET_MANIFEST_TYPE

    def __post_init__(self) -> None:
        _require_phase2_schema_version(self.schema_version)
        _require_manifest_type(self.manifest_type, self._expected_manifest_type)
        _require_nonempty_string(self.dataset_id, "dataset_id")
        _require_development_cohort_role(self.cohort_role, "cohort_role")
        _require_nonempty_string(self.adapter_name, "adapter_name")
        _require_nonempty_string(self.adapter_version, "adapter_version")
        _require_nonempty_string(self.generated_at_utc, "generated_at_utc")
        _require_nonempty_string(self.git_commit, "git_commit")
        _require_sha256(self.dataset_root_fingerprint, "dataset_root_fingerprint")
        _require_sha256(self.manifest_hash, "manifest_hash")
        _require_nonnegative_int(self.case_count, "case_count")
        _require_tuple_of(self.cases, DatasetCaseRecord, "cases", allow_empty=False)
        if self.case_count != len(self.cases):
            msg = "case_count must equal the number of cases"
            raise Phase2ArtifactValidationError(msg)
        _require_sorted_cases(self.cases)
        _require_unique_values(
            (case.anonymous_patient_id for case in self.cases),
            "anonymous_patient_id",
        )
        _require_unique_values((case.anonymous_case_id for case in self.cases), "anonymous_case_id")
        _require_unique_values((case.relative_image_path for case in self.cases), "image paths")
        _require_unique_values((case.relative_label_path for case in self.cases), "label paths")
        _require_unique_values((case.image_sha256 for case in self.cases), "image hashes")
        _require_unique_values(
            ((case.image_sha256, case.label_sha256) for case in self.cases),
            "image/label hash pairings",
        )
        for case in self.cases:
            if case.cohort_role != self.cohort_role:
                msg = "all case cohort roles must match manifest cohort_role"
                raise Phase2ArtifactValidationError(msg)


@dataclass(frozen=True, slots=True)
class SplitAssignment:
    """Patient-level partition assignment for one anonymous case."""

    anonymous_patient_id: str
    anonymous_case_id: str
    partition: str

    def __post_init__(self) -> None:
        _require_safe_anonymous_id(self.anonymous_patient_id, "anonymous_patient_id")
        _require_safe_anonymous_id(self.anonymous_case_id, "anonymous_case_id")
        if self.partition not in SUPPORTED_SPLIT_PARTITIONS:
            msg = f"partition must be one of {SUPPORTED_SPLIT_PARTITIONS!r}"
            raise Phase2ArtifactValidationError(msg)


@dataclass(frozen=True, slots=True)
class DevelopmentSplitManifest:
    """Persisted patient-level development split artifact."""

    schema_version: str
    manifest_type: str
    generated_at_utc: str
    git_commit: str
    source_manifest_hash: str
    split_policy_version: str
    split_seed: int
    split_hash: str
    assignments: tuple[SplitAssignment, ...]
    train_patient_count: int
    validation_patient_count: int
    internal_test_patient_count: int
    train_case_count: int
    validation_case_count: int
    internal_test_case_count: int

    _expected_manifest_type: ClassVar[str] = DEVELOPMENT_SPLIT_MANIFEST_TYPE

    def __post_init__(self) -> None:
        _require_phase2_schema_version(self.schema_version)
        _require_manifest_type(self.manifest_type, self._expected_manifest_type)
        _require_nonempty_string(self.generated_at_utc, "generated_at_utc")
        _require_nonempty_string(self.git_commit, "git_commit")
        _require_sha256(self.source_manifest_hash, "source_manifest_hash")
        _require_nonempty_string(self.split_policy_version, "split_policy_version")
        _require_nonnegative_int(self.split_seed, "split_seed")
        _require_sha256(self.split_hash, "split_hash")
        _require_tuple_of(self.assignments, SplitAssignment, "assignments", allow_empty=False)
        _require_sorted_assignments(self.assignments)

        patient_to_partition: dict[str, str] = {}
        case_ids: set[str] = set()
        patient_counts = {partition: 0 for partition in SUPPORTED_SPLIT_PARTITIONS}
        case_counts = {partition: 0 for partition in SUPPORTED_SPLIT_PARTITIONS}
        for assignment in self.assignments:
            prior_partition = patient_to_partition.get(assignment.anonymous_patient_id)
            if prior_partition is not None:
                msg = "anonymous_patient_id assignments must be unique"
                raise Phase2ArtifactValidationError(msg)
            patient_to_partition[assignment.anonymous_patient_id] = assignment.partition
            if assignment.anonymous_case_id in case_ids:
                msg = "anonymous_case_id assignments must be unique"
                raise Phase2ArtifactValidationError(msg)
            case_ids.add(assignment.anonymous_case_id)
            patient_counts[assignment.partition] += 1
            case_counts[assignment.partition] += 1

        _require_partition_counts(
            patient_counts,
            {
                "train": self.train_patient_count,
                "validation": self.validation_patient_count,
                "internal_test": self.internal_test_patient_count,
            },
            "patient",
        )
        _require_partition_counts(
            case_counts,
            {
                "train": self.train_case_count,
                "validation": self.validation_case_count,
                "internal_test": self.internal_test_case_count,
            },
            "case",
        )


@dataclass(frozen=True, slots=True)
class LesionSummaryRecord:
    """Per-lesion volume summary from a documented 3D connectivity convention."""

    lesion_index: int
    voxel_count: int
    physical_volume_mm3: float

    def __post_init__(self) -> None:
        _require_positive_int(self.lesion_index, "lesion_index")
        _require_nonnegative_int(self.voxel_count, "voxel_count")
        _require_nonnegative_finite_float(self.physical_volume_mm3, "physical_volume_mm3")


@dataclass(frozen=True, slots=True)
class CaseQaRecord:
    """Per-case Phase 2 QA record populated by later QA execution."""

    anonymous_patient_id: str
    anonymous_case_id: str
    dimensionality: int
    shape: tuple[int, int, int]
    image_dtype: str
    label_dtype: str
    affine: tuple[tuple[float, float, float, float], ...]
    orientation: tuple[str, str, str]
    spacing: tuple[float, float, float]
    image_finite: bool
    label_finite: bool
    allowed_label_values: tuple[int, ...]
    image_label_shape_match: bool
    image_label_affine_match: bool
    intensity_min: float
    intensity_max: float
    intensity_mean: float
    intensity_std: float
    histogram_bin_edges: tuple[float, ...]
    histogram_counts: tuple[int, ...]
    tumor_voxel_count: int
    tumor_physical_volume_mm3: float
    lesion_count: int
    lesions: tuple[LesionSummaryRecord, ...]
    empty_tumor: bool
    qa_passed: bool
    failure_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_safe_anonymous_id(self.anonymous_patient_id, "anonymous_patient_id")
        _require_safe_anonymous_id(self.anonymous_case_id, "anonymous_case_id")
        if self.dimensionality != 3:
            msg = "dimensionality must be fixed to 3"
            raise Phase2ArtifactValidationError(msg)
        _require_shape_3d(self.shape, "shape")
        _require_nonempty_string(self.image_dtype, "image_dtype")
        _require_nonempty_string(self.label_dtype, "label_dtype")
        _require_affine_4x4(self.affine, "affine")
        _require_orientation(self.orientation, "orientation")
        _require_spacing_3d(self.spacing, "spacing")
        _require_bool(self.image_finite, "image_finite")
        _require_bool(self.label_finite, "label_finite")
        _require_allowed_label_values(self.allowed_label_values, "allowed_label_values")
        _require_bool(self.image_label_shape_match, "image_label_shape_match")
        _require_bool(self.image_label_affine_match, "image_label_affine_match")
        _require_finite_float(self.intensity_min, "intensity_min")
        _require_finite_float(self.intensity_max, "intensity_max")
        _require_finite_float(self.intensity_mean, "intensity_mean")
        _require_nonnegative_finite_float(self.intensity_std, "intensity_std")
        if self.intensity_min > self.intensity_max:
            msg = "intensity_min must be less than or equal to intensity_max"
            raise Phase2ArtifactValidationError(msg)
        _require_histogram(self.histogram_bin_edges, self.histogram_counts)
        _require_nonnegative_int(self.tumor_voxel_count, "tumor_voxel_count")
        _require_nonnegative_finite_float(
            self.tumor_physical_volume_mm3,
            "tumor_physical_volume_mm3",
        )
        _require_nonnegative_int(self.lesion_count, "lesion_count")
        _require_tuple_of(self.lesions, LesionSummaryRecord, "lesions", allow_empty=True)
        if self.lesion_count != len(self.lesions):
            msg = "lesion_count must equal the number of lesion records"
            raise Phase2ArtifactValidationError(msg)
        _require_consecutive_lesion_indices(self.lesions)
        _require_bool(self.empty_tumor, "empty_tumor")
        _require_bool(self.qa_passed, "qa_passed")
        _require_string_tuple(self.failure_reasons, "failure_reasons", allow_empty=True)
        if self.qa_passed and self.failure_reasons:
            msg = "qa_passed=True requires no failure_reasons"
            raise Phase2ArtifactValidationError(msg)
        if not self.qa_passed and not self.failure_reasons:
            msg = "qa_passed=False requires at least one failure reason"
            raise Phase2ArtifactValidationError(msg)
        if self.empty_tumor:
            if self.tumor_voxel_count != 0 or self.lesion_count != 0 or self.lesions:
                msg = "empty_tumor=True requires zero tumor voxels and zero lesions"
                raise Phase2ArtifactValidationError(msg)
        elif self.tumor_voxel_count == 0 or self.lesion_count == 0:
            msg = "empty_tumor=False requires positive tumor voxels and at least one lesion"
            raise Phase2ArtifactValidationError(msg)


@dataclass(frozen=True, slots=True)
class DevelopmentQaArtifact:
    """Machine-readable Phase 2 development QA artifact."""

    schema_version: str
    stage: str
    created_at_utc: str
    git_commit: str
    config_hash: str
    manifest_hash: str
    split_hash: str
    connectivity: int
    affine_tolerance: float
    histogram_bin_edges: tuple[float, ...]
    case_count: int
    passed_case_count: int
    failed_case_count: int
    case_records: tuple[CaseQaRecord, ...]
    qa_artifact_hash: str

    _expected_stage: ClassVar[str] = DEVELOPMENT_QA_STAGE

    def __post_init__(self) -> None:
        _require_phase2_schema_version(self.schema_version)
        _require_stage(self.stage, self._expected_stage)
        _require_nonempty_string(self.created_at_utc, "created_at_utc")
        _require_nonempty_string(self.git_commit, "git_commit")
        _require_sha256(self.config_hash, "config_hash")
        _require_sha256(self.manifest_hash, "manifest_hash")
        _require_sha256(self.split_hash, "split_hash")
        if self.connectivity not in SUPPORTED_CONNECTIVITY_VALUES:
            msg = f"connectivity must be one of {SUPPORTED_CONNECTIVITY_VALUES!r}"
            raise Phase2ArtifactValidationError(msg)
        _require_positive_finite_float(self.affine_tolerance, "affine_tolerance")
        _require_histogram_edges(self.histogram_bin_edges, "histogram_bin_edges")
        _require_nonnegative_int(self.case_count, "case_count")
        _require_nonnegative_int(self.passed_case_count, "passed_case_count")
        _require_nonnegative_int(self.failed_case_count, "failed_case_count")
        _require_tuple_of(self.case_records, CaseQaRecord, "case_records", allow_empty=False)
        if self.case_count != len(self.case_records):
            msg = "case_count must equal the number of case_records"
            raise Phase2ArtifactValidationError(msg)
        if self.passed_case_count != sum(1 for case in self.case_records if case.qa_passed):
            msg = "passed_case_count must equal the number of passing case records"
            raise Phase2ArtifactValidationError(msg)
        if self.failed_case_count != sum(1 for case in self.case_records if not case.qa_passed):
            msg = "failed_case_count must equal the number of failing case records"
            raise Phase2ArtifactValidationError(msg)
        if self.case_count != self.passed_case_count + self.failed_case_count:
            msg = "case_count must equal passed_case_count + failed_case_count"
            raise Phase2ArtifactValidationError(msg)
        _require_sorted_case_qa_records(self.case_records)
        _require_unique_values(
            (case.anonymous_case_id for case in self.case_records),
            "anonymous_case_id",
        )
        for case in self.case_records:
            if case.histogram_bin_edges != self.histogram_bin_edges:
                msg = "all case histogram_bin_edges must match the QA artifact edges"
                raise Phase2ArtifactValidationError(msg)
        _require_sha256(self.qa_artifact_hash, "qa_artifact_hash")


@dataclass(frozen=True, slots=True)
class LeakageAuditArtifact:
    """Machine-readable Phase 2 leakage-audit evidence."""

    schema_version: str
    stage: str
    created_at_utc: str
    git_commit: str
    manifest_hash: str
    split_hash: str
    split_policy_version: str
    split_seed: int
    patient_counts_by_partition: Mapping[str, int]
    case_counts_by_partition: Mapping[str, int]
    pairwise_patient_overlap_counts: Mapping[str, int]
    pairwise_case_overlap_counts: Mapping[str, int]
    duplicate_image_hash_findings: tuple[str, ...]
    duplicate_label_hash_findings: tuple[str, ...]
    near_duplicate_policy: str
    lits_msd_equivalence_warning: str
    external_data_accessed: bool
    preprocessing_fitted_on_nontraining_data: bool
    unresolved_findings: tuple[str, ...]
    critical_finding_count: int
    audit_passed: bool
    audit_hash: str

    _expected_stage: ClassVar[str] = LEAKAGE_AUDIT_STAGE

    def __post_init__(self) -> None:
        _require_phase2_schema_version(self.schema_version)
        _require_stage(self.stage, self._expected_stage)
        _require_nonempty_string(self.created_at_utc, "created_at_utc")
        _require_nonempty_string(self.git_commit, "git_commit")
        _require_sha256(self.manifest_hash, "manifest_hash")
        _require_sha256(self.split_hash, "split_hash")
        _require_nonempty_string(self.split_policy_version, "split_policy_version")
        _require_nonnegative_int(self.split_seed, "split_seed")
        patient_counts = _freeze_nonnegative_int_mapping(
            self.patient_counts_by_partition,
            "patient_counts_by_partition",
            required_keys=SUPPORTED_SPLIT_PARTITIONS,
        )
        case_counts = _freeze_nonnegative_int_mapping(
            self.case_counts_by_partition,
            "case_counts_by_partition",
            required_keys=SUPPORTED_SPLIT_PARTITIONS,
        )
        patient_overlaps = _freeze_nonnegative_int_mapping(
            self.pairwise_patient_overlap_counts,
            "pairwise_patient_overlap_counts",
            required_keys=_PAIRWISE_PARTITION_KEYS,
        )
        case_overlaps = _freeze_nonnegative_int_mapping(
            self.pairwise_case_overlap_counts,
            "pairwise_case_overlap_counts",
            required_keys=_PAIRWISE_PARTITION_KEYS,
        )
        object.__setattr__(self, "patient_counts_by_partition", patient_counts)
        object.__setattr__(self, "case_counts_by_partition", case_counts)
        object.__setattr__(self, "pairwise_patient_overlap_counts", patient_overlaps)
        object.__setattr__(self, "pairwise_case_overlap_counts", case_overlaps)
        _require_string_tuple(
            self.duplicate_image_hash_findings,
            "duplicate_image_hash_findings",
            allow_empty=True,
        )
        _require_string_tuple(
            self.duplicate_label_hash_findings,
            "duplicate_label_hash_findings",
            allow_empty=True,
        )
        _require_nonempty_string(self.near_duplicate_policy, "near_duplicate_policy")
        _require_lits_msd_warning(self.lits_msd_equivalence_warning)
        _require_bool(self.external_data_accessed, "external_data_accessed")
        _require_bool(
            self.preprocessing_fitted_on_nontraining_data,
            "preprocessing_fitted_on_nontraining_data",
        )
        _require_string_tuple(self.unresolved_findings, "unresolved_findings", allow_empty=True)
        _require_nonnegative_int(self.critical_finding_count, "critical_finding_count")
        _require_bool(self.audit_passed, "audit_passed")
        _require_sha256(self.audit_hash, "audit_hash")
        if self.audit_passed:
            _require_all_zero(patient_overlaps, "pairwise_patient_overlap_counts")
            _require_all_zero(case_overlaps, "pairwise_case_overlap_counts")
            if self.critical_finding_count != 0:
                msg = "audit_passed=True requires critical_finding_count to equal zero"
                raise Phase2ArtifactValidationError(msg)
            if self.unresolved_findings:
                msg = "audit_passed=True requires no unresolved findings"
                raise Phase2ArtifactValidationError(msg)
            if self.external_data_accessed:
                msg = "audit_passed=True requires external_data_accessed=False"
                raise Phase2ArtifactValidationError(msg)
            if self.preprocessing_fitted_on_nontraining_data:
                msg = "audit_passed=True requires preprocessing_fitted_on_nontraining_data=False"
                raise Phase2ArtifactValidationError(msg)


Phase2Artifact: TypeAlias = (
    DatasetCaseRecord
    | DatasetManifest
    | SplitAssignment
    | DevelopmentSplitManifest
    | LesionSummaryRecord
    | CaseQaRecord
    | DevelopmentQaArtifact
    | LeakageAuditArtifact
)
Phase2ArtifactT = TypeVar("Phase2ArtifactT", bound=Phase2Artifact)

_PHASE2_ARTIFACT_TYPES: tuple[type[Phase2Artifact], ...] = (
    DatasetCaseRecord,
    DatasetManifest,
    SplitAssignment,
    DevelopmentSplitManifest,
    LesionSummaryRecord,
    CaseQaRecord,
    DevelopmentQaArtifact,
    LeakageAuditArtifact,
)
_MANIFEST_TYPES: dict[str, type[Phase2Artifact]] = {
    DATASET_MANIFEST_TYPE: DatasetManifest,
    DEVELOPMENT_SPLIT_MANIFEST_TYPE: DevelopmentSplitManifest,
}
_STAGE_TYPES: dict[str, type[Phase2Artifact]] = {
    DEVELOPMENT_QA_STAGE: DevelopmentQaArtifact,
    LEAKAGE_AUDIT_STAGE: LeakageAuditArtifact,
}
_NESTED_RECORD_FIELDS = {
    "cases": DatasetCaseRecord,
    "assignments": SplitAssignment,
    "lesions": LesionSummaryRecord,
    "case_records": CaseQaRecord,
}
_TUPLE_FIELDS = {
    "shape",
    "affine",
    "orientation",
    "spacing",
    "allowed_label_values",
    "histogram_bin_edges",
    "histogram_counts",
    "failure_reasons",
    "duplicate_image_hash_findings",
    "duplicate_label_hash_findings",
    "unresolved_findings",
}
_MAPPING_FIELDS = {
    "patient_counts_by_partition",
    "case_counts_by_partition",
    "pairwise_patient_overlap_counts",
    "pairwise_case_overlap_counts",
}


def phase2_artifact_to_dict(artifact: Phase2Artifact) -> dict[str, JsonValue]:
    """Convert a Phase 2 dataclass into a JSON-compatible dictionary."""
    if not isinstance(artifact, _PHASE2_ARTIFACT_TYPES):
        msg = f"unsupported Phase 2 artifact type: {type(artifact).__name__}"
        raise Phase2ArtifactSerializationError(msg)
    return cast(dict[str, JsonValue], _jsonify_dataclass(artifact))


def phase2_artifact_to_json(artifact: Phase2Artifact, path: Path | None = None) -> str:
    """Serialize a Phase 2 artifact with sorted keys, UTF-8, and one trailing newline."""
    try:
        text = json.dumps(
            phase2_artifact_to_dict(artifact),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        msg = f"failed to serialize Phase 2 artifact: {exc}"
        raise Phase2ArtifactSerializationError(msg) from exc

    text_with_newline = f"{text}\n"
    if path is not None:
        path.write_text(text_with_newline, encoding="utf-8", newline="\n")
    return text_with_newline


@overload
def phase2_artifact_from_json(
    json_text: str,
    artifact_type: type[Phase2ArtifactT],
) -> Phase2ArtifactT: ...


@overload
def phase2_artifact_from_json(
    json_text: str,
    artifact_type: None = None,
) -> Phase2Artifact: ...


def phase2_artifact_from_json(
    json_text: str,
    artifact_type: type[Phase2ArtifactT] | None = None,
) -> Phase2ArtifactT | Phase2Artifact:
    """Parse deterministic Phase 2 JSON into a frozen dataclass contract."""
    try:
        raw_value = json.loads(
            json_text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_bad_json_constant,
        )
    except json.JSONDecodeError as exc:
        msg = f"invalid Phase 2 artifact JSON: {exc}"
        raise Phase2ArtifactSerializationError(msg) from exc

    if not isinstance(raw_value, dict):
        msg = "Phase 2 artifact JSON must contain an object"
        raise Phase2ArtifactSerializationError(msg)

    raw = cast(dict[str, object], raw_value)
    if artifact_type is None:
        resolved_type = _resolve_phase2_type(raw)
    else:
        resolved_type = cast(type[Phase2Artifact], artifact_type)
    _require_phase2_schema_when_present(raw, resolved_type)
    return _construct_phase2_artifact(raw, resolved_type)


def _resolve_phase2_type(raw: dict[str, object]) -> type[Phase2Artifact]:
    manifest_type = raw.get("manifest_type")
    if isinstance(manifest_type, str):
        if manifest_type not in _MANIFEST_TYPES:
            msg = f"unsupported Phase 2 manifest_type: {manifest_type!r}"
            raise Phase2ArtifactStageError(msg)
        return _MANIFEST_TYPES[manifest_type]
    stage = raw.get("stage")
    if isinstance(stage, str):
        if stage not in _STAGE_TYPES:
            msg = f"unsupported Phase 2 artifact stage: {stage!r}"
            raise Phase2ArtifactStageError(msg)
        return _STAGE_TYPES[stage]
    msg = "Phase 2 artifact JSON requires artifact_type when it has no manifest_type or stage"
    raise Phase2ArtifactStageError(msg)


def _require_phase2_schema_when_present(
    raw: dict[str, object],
    artifact_type: type[Phase2Artifact],
) -> None:
    if "schema_version" in {field.name for field in fields(artifact_type)}:
        _require_phase2_schema_version(raw.get("schema_version"))


def _construct_phase2_artifact(
    raw: dict[str, object],
    artifact_type: type[Phase2Artifact],
) -> Phase2Artifact:
    if artifact_type not in _PHASE2_ARTIFACT_TYPES:
        msg = f"unsupported Phase 2 artifact type: {artifact_type.__name__}"
        raise Phase2ArtifactSerializationError(msg)
    expected_fields = {field.name for field in fields(artifact_type)}
    extra_fields = set(raw) - expected_fields
    missing_fields = expected_fields - set(raw)
    if extra_fields:
        msg = f"unexpected fields for {artifact_type.__name__}: {sorted(extra_fields)}"
        raise Phase2ArtifactValidationError(msg)
    if missing_fields:
        msg = f"missing fields for {artifact_type.__name__}: {sorted(missing_fields)}"
        raise Phase2ArtifactValidationError(msg)

    kwargs: dict[str, object] = {}
    for key, value in raw.items():
        if key in _NESTED_RECORD_FIELDS:
            if not isinstance(value, list):
                msg = f"{key} must be a JSON array"
                raise Phase2ArtifactValidationError(msg)
            nested_type = _NESTED_RECORD_FIELDS[key]
            kwargs[key] = tuple(_construct_nested_record(item, nested_type, key) for item in value)
        elif key in _TUPLE_FIELDS:
            if not isinstance(value, list):
                msg = f"{key} must be a JSON array"
                raise Phase2ArtifactValidationError(msg)
            kwargs[key] = _deep_tuple(value)
        elif key in _MAPPING_FIELDS:
            if not isinstance(value, dict):
                msg = f"{key} must be a JSON object"
                raise Phase2ArtifactValidationError(msg)
            kwargs[key] = MappingProxyType(dict(value))
        else:
            kwargs[key] = value

    return cast(Phase2Artifact, cast(Any, artifact_type)(**kwargs))


def _construct_nested_record(
    value: object,
    nested_type: type[Phase2Artifact],
    field_name: str,
) -> Phase2Artifact:
    if not isinstance(value, dict):
        msg = f"{field_name} entries must be JSON objects"
        raise Phase2ArtifactValidationError(msg)
    return _construct_phase2_artifact(cast(dict[str, object], value), nested_type)


def _jsonify_dataclass(value: Phase2Artifact) -> JsonValue:
    result: dict[str, JsonValue] = {}
    for field in fields(value):
        result[field.name] = _jsonify(getattr(value, field.name), field.name)
    return result


def _jsonify(value: object, location: str) -> JsonValue:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            msg = f"{location} must not contain NaN or Infinity"
            raise Phase2ArtifactSerializationError(msg)
        return value
    if is_dataclass(value) and isinstance(value, _PHASE2_ARTIFACT_TYPES):
        return _jsonify_dataclass(value)
    if isinstance(value, tuple):
        return [_jsonify(item, f"{location}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, Mapping):
        result: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                msg = f"{location} contains a non-string dictionary key: {key!r}"
                raise Phase2ArtifactSerializationError(msg)
            result[key] = _jsonify(item, f"{location}.{key}")
        return result
    msg = f"{location} is not JSON-compatible: {type(value).__name__}"
    raise Phase2ArtifactSerializationError(msg)


def _deep_tuple(value: object) -> object:
    if isinstance(value, list):
        return tuple(_deep_tuple(item) for item in value)
    return value


def _require_phase2_schema_version(value: object) -> None:
    if value != PHASE2_SCHEMA_VERSION:
        msg = f"unsupported Phase 2 schema_version: {value!r}; expected {PHASE2_SCHEMA_VERSION!r}"
        raise Phase2ArtifactSchemaVersionError(msg)


def _require_manifest_type(value: object, expected_manifest_type: str) -> None:
    if value != expected_manifest_type:
        msg = f"incorrect manifest_type: {value!r}; expected {expected_manifest_type!r}"
        raise Phase2ArtifactStageError(msg)


def _require_stage(value: object, expected_stage: str) -> None:
    if value != expected_stage:
        msg = f"incorrect Phase 2 artifact stage: {value!r}; expected {expected_stage!r}"
        raise Phase2ArtifactStageError(msg)


def _require_nonempty_string(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value:
        msg = f"{field_name} must be a nonempty string"
        raise Phase2ArtifactValidationError(msg)


def _require_safe_anonymous_id(value: object, field_name: str) -> None:
    _require_nonempty_string(value, field_name)
    text = cast(str, value)
    lowered = text.lower()
    if _SAFE_ID_PATTERN.fullmatch(text) is None:
        msg = f"{field_name} must use lowercase letters, digits, underscores, and hyphens only"
        raise Phase2ArtifactValidationError(msg)
    if any(separator in text for separator in ("/", "\\", ":", "~")) or text in {".", ".."}:
        msg = f"{field_name} must not be path-like"
        raise Phase2ArtifactValidationError(msg)
    if _DATE_PATTERN.search(lowered) is not None:
        msg = f"{field_name} must not contain date-like identifiers"
        raise Phase2ArtifactValidationError(msg)
    compact = lowered.replace("-", "").replace("_", "")
    if any(keyword in compact for keyword in _IDENTIFIER_KEYWORDS):
        msg = f"{field_name} must not contain obvious medical identifier keywords"
        raise Phase2ArtifactValidationError(msg)


def _require_safe_relative_posix_path(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value:
        msg = f"{field_name} must be a nonempty relative POSIX path"
        raise Phase2ArtifactValidationError(msg)
    if value.startswith("~") or "\\" in value:
        msg = f"{field_name} must be a normalized relative POSIX path"
        raise Phase2ArtifactValidationError(msg)
    posix_path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    if posix_path.is_absolute() or windows_path.is_absolute():
        msg = f"{field_name} must not be absolute"
        raise Phase2ArtifactValidationError(msg)
    if str(posix_path) != value or value in {".", ".."}:
        msg = f"{field_name} must be normalized and must not be '.' or '..'"
        raise Phase2ArtifactValidationError(msg)
    if ".." in posix_path.parts or any(part == "" for part in posix_path.parts):
        msg = f"{field_name} must not contain empty segments or parent traversal"
        raise Phase2ArtifactValidationError(msg)


def _require_sha256(value: object, field_name: str) -> None:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        msg = f"{field_name} must be a lowercase SHA-256 hexadecimal string"
        raise Phase2ArtifactValidationError(msg)


def _require_development_cohort_role(value: object, field_name: str) -> None:
    if value != PHASE2_DEVELOPMENT_COHORT_ROLE:
        msg = f"{field_name} must be fixed to {PHASE2_DEVELOPMENT_COHORT_ROLE!r} for Phase 2"
        raise Phase2ArtifactValidationError(msg)


def _require_nonnegative_int(value: object, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        msg = f"{field_name} must be a nonnegative integer"
        raise Phase2ArtifactValidationError(msg)


def _require_positive_int(value: object, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        msg = f"{field_name} must be a positive integer"
        raise Phase2ArtifactValidationError(msg)


def _require_finite_float(value: object, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, float) or not math.isfinite(value):
        msg = f"{field_name} must be a finite float"
        raise Phase2ArtifactValidationError(msg)


def _require_positive_finite_float(value: object, field_name: str) -> None:
    _require_finite_float(value, field_name)
    if cast(float, value) <= 0.0:
        msg = f"{field_name} must be positive"
        raise Phase2ArtifactValidationError(msg)


def _require_nonnegative_finite_float(value: object, field_name: str) -> None:
    _require_finite_float(value, field_name)
    if cast(float, value) < 0.0:
        msg = f"{field_name} must be nonnegative"
        raise Phase2ArtifactValidationError(msg)


def _require_bool(value: object, field_name: str) -> None:
    if not isinstance(value, bool):
        msg = f"{field_name} must be a boolean"
        raise Phase2ArtifactValidationError(msg)


def _require_string_tuple(value: object, field_name: str, *, allow_empty: bool) -> None:
    if not isinstance(value, tuple):
        msg = f"{field_name} must be a tuple of strings"
        raise Phase2ArtifactValidationError(msg)
    if not allow_empty and not value:
        msg = f"{field_name} must not be empty"
        raise Phase2ArtifactValidationError(msg)
    if any(not isinstance(item, str) or not item for item in value):
        msg = f"{field_name} must contain only nonempty strings"
        raise Phase2ArtifactValidationError(msg)


def _require_tuple_of(
    value: object,
    item_type: type[object],
    field_name: str,
    *,
    allow_empty: bool,
) -> None:
    if not isinstance(value, tuple):
        msg = f"{field_name} must be a tuple"
        raise Phase2ArtifactValidationError(msg)
    if not allow_empty and not value:
        msg = f"{field_name} must not be empty"
        raise Phase2ArtifactValidationError(msg)
    if any(not isinstance(item, item_type) for item in value):
        msg = f"{field_name} must contain only {item_type.__name__} entries"
        raise Phase2ArtifactValidationError(msg)


def _require_shape_3d(value: object, field_name: str) -> None:
    if (
        not isinstance(value, tuple)
        or len(value) != 3
        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in value)
    ):
        msg = f"{field_name} must contain exactly three positive integers"
        raise Phase2ArtifactValidationError(msg)


def _require_spacing_3d(value: object, field_name: str) -> None:
    if (
        not isinstance(value, tuple)
        or len(value) != 3
        or any(
            isinstance(item, bool)
            or not isinstance(item, float)
            or not math.isfinite(item)
            or item <= 0.0
            for item in value
        )
    ):
        msg = f"{field_name} must contain exactly three positive finite floats"
        raise Phase2ArtifactValidationError(msg)


def _require_affine_4x4(value: object, field_name: str) -> None:
    if not isinstance(value, tuple) or len(value) != 4:
        msg = f"{field_name} must be exactly 4x4"
        raise Phase2ArtifactValidationError(msg)
    for row in value:
        if (
            not isinstance(row, tuple)
            or len(row) != 4
            or any(
                isinstance(item, bool) or not isinstance(item, float) or not math.isfinite(item)
                for item in row
            )
        ):
            msg = f"{field_name} must contain finite float values in exactly four rows"
            raise Phase2ArtifactValidationError(msg)


def _require_orientation(value: object, field_name: str) -> None:
    if (
        not isinstance(value, tuple)
        or len(value) != 3
        or any(not isinstance(item, str) or item not in _ORIENTATION_AXIS_CODES for item in value)
    ):
        msg = f"{field_name} must contain three valid axis codes"
        raise Phase2ArtifactValidationError(msg)
    groups = {_ORIENTATION_GROUPS[item] for item in value}
    if len(groups) != 3:
        msg = f"{field_name} must contain one axis code from each 3D anatomical axis"
        raise Phase2ArtifactValidationError(msg)


def _require_allowed_label_values(value: object, field_name: str) -> None:
    if (
        not isinstance(value, tuple)
        or not value
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value)
    ):
        msg = f"{field_name} must be a nonempty tuple of nonnegative integers"
        raise Phase2ArtifactValidationError(msg)
    if tuple(sorted(set(value))) != value:
        msg = f"{field_name} must be unique and sorted"
        raise Phase2ArtifactValidationError(msg)


def _require_histogram_edges(edges: object, field_name: str) -> None:
    if (
        not isinstance(edges, tuple)
        or len(edges) < 2
        or any(
            isinstance(item, bool) or not isinstance(item, float) or not math.isfinite(item)
            for item in edges
        )
    ):
        msg = f"{field_name} must contain at least two finite float edges"
        raise Phase2ArtifactValidationError(msg)
    if any(edges[index] >= edges[index + 1] for index in range(len(edges) - 1)):
        msg = f"{field_name} must be strictly increasing"
        raise Phase2ArtifactValidationError(msg)


def _require_histogram(edges: object, counts: object) -> None:
    _require_histogram_edges(edges, "histogram_bin_edges")
    if not isinstance(counts, tuple) or any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in counts
    ):
        msg = "histogram_counts must contain nonnegative integers"
        raise Phase2ArtifactValidationError(msg)
    if len(cast(tuple[float, ...], edges)) != len(counts) + 1:
        msg = "histogram_bin_edges count must equal histogram_counts count plus one"
        raise Phase2ArtifactValidationError(msg)


def _require_consecutive_lesion_indices(lesions: tuple[LesionSummaryRecord, ...]) -> None:
    expected_indices = tuple(range(1, len(lesions) + 1))
    observed_indices = tuple(lesion.lesion_index for lesion in lesions)
    if observed_indices != expected_indices:
        msg = "lesion indices must be deterministic and consecutive starting at 1"
        raise Phase2ArtifactValidationError(msg)


def _require_sorted_cases(cases: tuple[DatasetCaseRecord, ...]) -> None:
    expected = tuple(
        sorted(cases, key=lambda case: (case.anonymous_patient_id, case.anonymous_case_id))
    )
    if cases != expected:
        msg = "cases must be ordered by anonymous_patient_id and anonymous_case_id"
        raise Phase2ArtifactValidationError(msg)


def _require_sorted_assignments(assignments: tuple[SplitAssignment, ...]) -> None:
    expected = tuple(
        sorted(
            assignments,
            key=lambda assignment: (
                assignment.anonymous_patient_id,
                assignment.anonymous_case_id,
            ),
        )
    )
    if assignments != expected:
        msg = "assignments must be ordered by anonymous_patient_id and anonymous_case_id"
        raise Phase2ArtifactValidationError(msg)


def _require_sorted_case_qa_records(case_records: tuple[CaseQaRecord, ...]) -> None:
    expected = tuple(
        sorted(case_records, key=lambda case: (case.anonymous_patient_id, case.anonymous_case_id))
    )
    if case_records != expected:
        msg = "case_records must be ordered by anonymous_patient_id and anonymous_case_id"
        raise Phase2ArtifactValidationError(msg)


def _require_unique_values(values: Iterable[object], field_name: str) -> None:
    seen: set[object] = set()
    for value in values:
        if value in seen:
            msg = f"{field_name} must be unique"
            raise Phase2ArtifactValidationError(msg)
        seen.add(value)


def _require_partition_counts(
    computed: Mapping[str, int],
    stored: Mapping[str, int],
    count_kind: str,
) -> None:
    for partition, stored_count in stored.items():
        _require_nonnegative_int(stored_count, f"{partition}_{count_kind}_count")
        if computed[partition] != stored_count:
            msg = f"{partition}_{count_kind}_count must equal computed assignment count"
            raise Phase2ArtifactValidationError(msg)


def _freeze_nonnegative_int_mapping(
    value: object,
    field_name: str,
    *,
    required_keys: tuple[str, ...],
) -> Mapping[str, int]:
    if not isinstance(value, Mapping):
        msg = f"{field_name} must be a mapping"
        raise Phase2ArtifactValidationError(msg)
    raw = dict(value)
    if set(raw) != set(required_keys):
        msg = f"{field_name} must contain exactly these keys: {required_keys!r}"
        raise Phase2ArtifactValidationError(msg)
    for key, item in raw.items():
        if not isinstance(key, str):
            msg = f"{field_name} keys must be strings"
            raise Phase2ArtifactValidationError(msg)
        _require_nonnegative_int(item, f"{field_name}.{key}")
    return MappingProxyType(raw)


def _require_all_zero(values: Mapping[str, int], field_name: str) -> None:
    if any(value != 0 for value in values.values()):
        msg = f"a passing audit requires every {field_name} value to equal zero"
        raise Phase2ArtifactValidationError(msg)


def _require_lits_msd_warning(value: object) -> None:
    _require_nonempty_string(value, "lits_msd_equivalence_warning")
    text = cast(str, value).lower()
    if "lits" not in text or "msd" not in text or "independent" not in text:
        msg = (
            "lits_msd_equivalence_warning must explicitly warn that LiTS and MSD are not "
            "independent"
        )
        raise Phase2ArtifactValidationError(msg)


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            msg = f"duplicate JSON object key: {key}"
            raise Phase2ArtifactSerializationError(msg)
        result[key] = value
    return result


def _bad_json_constant(value: str) -> NoReturn:
    msg = f"Phase 2 artifact JSON must not contain {value}"
    raise Phase2ArtifactSerializationError(msg)
