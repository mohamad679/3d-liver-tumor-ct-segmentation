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
GEOMETRY_LABEL_QA_STAGE = "geometry_label_qa"
LESION_COMPONENTS_STAGE = "lesion_components"
DEVELOPMENT_DATA_SUMMARY_STAGE = "development_data_summary"
DEVELOPMENT_QA_STAGE = "phase2-development-qa"
LEAKAGE_AUDIT_STAGE = "phase2-leakage-audit"
SUPPORTED_SPLIT_PARTITIONS = ("train", "validation", "internal_test")
SUPPORTED_CONNECTIVITY_VALUES = (6, 18, 26)
LEAKAGE_AUDIT_FINDING_CODES = (
    "missing_case_assignment",
    "extra_case_assignment",
    "patient_id_mismatch",
    "empty_partition",
    "patient_partition_overlap",
    "case_partition_overlap",
    "image_hash_cross_partition_overlap",
    "label_hash_cross_partition_overlap",
    "image_label_pair_cross_partition_overlap",
    "duplicate_anonymous_patient_id",
    "duplicate_anonymous_case_id",
    "duplicate_patient_case_pair",
    "conflicting_patient_partition",
    "conflicting_case_partition",
)
LESION_COMPONENTS_UPSTREAM_FAILURE_REASON = "upstream_geometry_label_qa_failed"
DEVELOPMENT_DATA_SUMMARY_UPSTREAM_FAILURE_REASON = "upstream_geometry_label_qa_failed"
GEOMETRY_LABEL_QA_FAILURE_CODES = (
    "image_not_3d",
    "label_not_3d",
    "image_nonfinite",
    "label_nonfinite",
    "label_noninteger",
    "label_value_not_allowed",
    "shape_mismatch",
    "affine_mismatch",
    "spacing_mismatch",
)

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
    partition: str
    dimensionality: int
    shape: tuple[int, ...]
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
    intensity_min: float | None
    intensity_max: float | None
    intensity_mean: float | None
    intensity_std: float | None
    histogram_bin_edges: tuple[float, ...]
    histogram_counts: tuple[int, ...]
    tumor_voxel_count: int
    tumor_physical_volume_mm3: float | None
    lesion_count: int | None
    lesions: tuple[LesionSummaryRecord, ...]
    empty_tumor: bool
    qa_passed: bool
    failure_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_safe_anonymous_id(self.anonymous_patient_id, "anonymous_patient_id")
        _require_safe_anonymous_id(self.anonymous_case_id, "anonymous_case_id")
        if self.partition not in SUPPORTED_SPLIT_PARTITIONS:
            msg = f"partition must be one of {SUPPORTED_SPLIT_PARTITIONS!r}"
            raise Phase2ArtifactValidationError(msg)
        _require_positive_int(self.dimensionality, "dimensionality")
        _require_positive_shape(self.shape, "shape")
        if self.qa_passed and (self.dimensionality != 3 or len(self.shape) != 3):
            msg = "qa_passed=True requires 3D dimensionality and shape"
            raise Phase2ArtifactValidationError(msg)
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
        _require_histogram_edges(self.histogram_bin_edges, "histogram_bin_edges")
        _require_nonnegative_int(self.tumor_voxel_count, "tumor_voxel_count")
        _require_tuple_of(self.lesions, LesionSummaryRecord, "lesions", allow_empty=True)
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
        self._validate_optional_measurements()
        if self.empty_tumor != (self.tumor_voxel_count == 0):
            msg = "empty_tumor must be consistent with tumor_voxel_count"
            raise Phase2ArtifactValidationError(msg)

    def _validate_optional_measurements(self) -> None:
        intensity_fields = (
            self.intensity_min,
            self.intensity_max,
            self.intensity_mean,
            self.intensity_std,
        )
        intensity_available = all(value is not None for value in intensity_fields)
        if intensity_available:
            _require_finite_float(self.intensity_min, "intensity_min")
            _require_finite_float(self.intensity_max, "intensity_max")
            _require_finite_float(self.intensity_mean, "intensity_mean")
            _require_nonnegative_finite_float(self.intensity_std, "intensity_std")
            if cast(float, self.intensity_min) > cast(float, self.intensity_max):
                msg = "intensity_min must be less than or equal to intensity_max"
                raise Phase2ArtifactValidationError(msg)
            _require_histogram(self.histogram_bin_edges, self.histogram_counts)
        else:
            if any(value is not None for value in intensity_fields) or self.histogram_counts:
                msg = "unavailable intensity measurements must all be absent together"
                raise Phase2ArtifactValidationError(msg)
        lesion_available = (
            self.tumor_physical_volume_mm3 is not None and self.lesion_count is not None
        )
        if lesion_available:
            _require_nonnegative_finite_float(
                self.tumor_physical_volume_mm3,
                "tumor_physical_volume_mm3",
            )
            _require_nonnegative_int(self.lesion_count, "lesion_count")
            if self.lesion_count != len(self.lesions):
                msg = "lesion_count must equal the number of lesion records"
                raise Phase2ArtifactValidationError(msg)
            if self.empty_tumor:
                if (
                    self.tumor_voxel_count != 0
                    or self.tumor_physical_volume_mm3 != 0.0
                    or self.lesion_count != 0
                    or self.lesions
                ):
                    msg = "empty_tumor=True requires zero tumor voxels, volume, and lesions"
                    raise Phase2ArtifactValidationError(msg)
            elif self.tumor_voxel_count == 0 or self.lesion_count == 0:
                msg = "empty_tumor=False requires positive tumor voxels and at least one lesion"
                raise Phase2ArtifactValidationError(msg)
        elif (
            self.tumor_physical_volume_mm3 is not None
            or self.lesion_count is not None
            or self.lesions
        ):
            msg = "unavailable lesion measurements must all be absent together"
            raise Phase2ArtifactValidationError(msg)
        if self.qa_passed and (not intensity_available or not lesion_available):
            msg = "qa_passed=True requires complete intensity and lesion measurements"
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
    geometry_qa_artifact_hash: str
    lesion_artifact_hash: str
    development_summary_artifact_hash: str
    connectivity: int
    affine_tolerance: float | None
    histogram_bin_edges: tuple[float, ...]
    histogram_bin_count: int
    case_count: int
    passed_case_count: int
    failed_case_count: int
    case_records: tuple[CaseQaRecord, ...]
    aggregate_image_voxel_count: int
    aggregate_intensity_min: float | None
    aggregate_intensity_max: float | None
    aggregate_intensity_mean: float | None
    aggregate_intensity_std: float | None
    aggregate_histogram_counts: tuple[int, ...]
    aggregate_below_histogram_range_count: int
    aggregate_above_histogram_range_count: int
    total_tumor_voxel_count: int
    total_tumor_physical_volume_mm3: float
    total_lesion_count: int
    lesion_volume_min_mm3: float | None
    lesion_volume_max_mm3: float | None
    lesion_volume_mean_mm3: float | None
    lesion_volume_median_mm3: float | None
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
        _require_sha256(self.geometry_qa_artifact_hash, "geometry_qa_artifact_hash")
        _require_sha256(self.lesion_artifact_hash, "lesion_artifact_hash")
        _require_sha256(
            self.development_summary_artifact_hash,
            "development_summary_artifact_hash",
        )
        if self.connectivity not in SUPPORTED_CONNECTIVITY_VALUES:
            msg = f"connectivity must be one of {SUPPORTED_CONNECTIVITY_VALUES!r}"
            raise Phase2ArtifactValidationError(msg)
        if self.affine_tolerance is not None:
            _require_positive_finite_float(self.affine_tolerance, "affine_tolerance")
        _require_histogram_edges(self.histogram_bin_edges, "histogram_bin_edges")
        _require_positive_int(self.histogram_bin_count, "histogram_bin_count")
        if len(self.histogram_bin_edges) != self.histogram_bin_count + 1:
            msg = "histogram_bin_edges count must equal histogram_bin_count plus one"
            raise Phase2ArtifactValidationError(msg)
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
            if case.histogram_counts and len(case.histogram_counts) != self.histogram_bin_count:
                msg = "all available case histogram_counts must match histogram_bin_count"
                raise Phase2ArtifactValidationError(msg)
        self._validate_summary_aggregates()
        _require_sha256(self.qa_artifact_hash, "qa_artifact_hash")

    def _validate_summary_aggregates(self) -> None:
        _require_nonnegative_int(self.aggregate_image_voxel_count, "aggregate_image_voxel_count")
        _require_nonnegative_int_tuple(
            self.aggregate_histogram_counts,
            "aggregate_histogram_counts",
            allow_empty=False,
        )
        if len(self.aggregate_histogram_counts) != self.histogram_bin_count:
            msg = "aggregate_histogram_counts length must equal histogram_bin_count"
            raise Phase2ArtifactValidationError(msg)
        _require_nonnegative_int(
            self.aggregate_below_histogram_range_count,
            "aggregate_below_histogram_range_count",
        )
        _require_nonnegative_int(
            self.aggregate_above_histogram_range_count,
            "aggregate_above_histogram_range_count",
        )
        aggregate_counted_voxels = (
            sum(self.aggregate_histogram_counts)
            + self.aggregate_below_histogram_range_count
            + self.aggregate_above_histogram_range_count
        )
        if aggregate_counted_voxels != self.aggregate_image_voxel_count:
            msg = "aggregate histogram and out-of-range counts must equal aggregate voxels"
            raise Phase2ArtifactValidationError(msg)
        if self.aggregate_image_voxel_count == 0:
            if any(
                value is not None
                for value in (
                    self.aggregate_intensity_min,
                    self.aggregate_intensity_max,
                    self.aggregate_intensity_mean,
                    self.aggregate_intensity_std,
                )
            ):
                msg = "zero aggregate voxels require absent aggregate intensity measurements"
                raise Phase2ArtifactValidationError(msg)
        else:
            _require_finite_float(self.aggregate_intensity_min, "aggregate_intensity_min")
            _require_finite_float(self.aggregate_intensity_max, "aggregate_intensity_max")
            _require_finite_float(self.aggregate_intensity_mean, "aggregate_intensity_mean")
            _require_nonnegative_finite_float(
                self.aggregate_intensity_std,
                "aggregate_intensity_std",
            )
            aggregate_min = cast(float, self.aggregate_intensity_min)
            aggregate_max = cast(float, self.aggregate_intensity_max)
            if aggregate_min > aggregate_max:
                msg = (
                    "aggregate_intensity_min must be less than or equal to aggregate_intensity_max"
                )
                raise Phase2ArtifactValidationError(msg)
        _require_nonnegative_int(self.total_tumor_voxel_count, "total_tumor_voxel_count")
        _require_nonnegative_finite_float(
            self.total_tumor_physical_volume_mm3,
            "total_tumor_physical_volume_mm3",
        )
        _require_nonnegative_int(self.total_lesion_count, "total_lesion_count")
        lesion_fields = (
            self.lesion_volume_min_mm3,
            self.lesion_volume_max_mm3,
            self.lesion_volume_mean_mm3,
            self.lesion_volume_median_mm3,
        )
        if self.total_lesion_count == 0:
            if any(value is not None for value in lesion_fields):
                msg = "zero total lesions require absent lesion volume summary statistics"
                raise Phase2ArtifactValidationError(msg)
            return
        for value, field_name in (
            (self.lesion_volume_min_mm3, "lesion_volume_min_mm3"),
            (self.lesion_volume_max_mm3, "lesion_volume_max_mm3"),
            (self.lesion_volume_mean_mm3, "lesion_volume_mean_mm3"),
            (self.lesion_volume_median_mm3, "lesion_volume_median_mm3"),
        ):
            _require_positive_finite_float(value, field_name)


@dataclass(frozen=True, slots=True)
class GeometryLabelQaCaseRecord:
    """Intermediate per-case geometry and label QA record before lesion and histogram QA."""

    anonymous_patient_id: str
    anonymous_case_id: str
    partition: str
    dimensionality: int
    image_shape: tuple[int, ...]
    label_shape: tuple[int, ...]
    image_dtype: str
    label_dtype: str
    image_affine: tuple[tuple[float, float, float, float], ...]
    label_affine: tuple[tuple[float, float, float, float], ...]
    image_orientation: tuple[str, str, str]
    label_orientation: tuple[str, str, str]
    image_spacing: tuple[float, float, float]
    label_spacing: tuple[float, float, float]
    image_finite: bool
    label_finite: bool
    observed_label_values: tuple[int | float, ...]
    allowed_label_values: tuple[int, ...]
    image_label_shape_match: bool
    image_label_affine_match: bool
    tumor_label_value: int
    tumor_voxel_count: int
    empty_tumor: bool
    qa_passed: bool
    failure_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_safe_anonymous_id(self.anonymous_patient_id, "anonymous_patient_id")
        _require_safe_anonymous_id(self.anonymous_case_id, "anonymous_case_id")
        if self.partition not in SUPPORTED_SPLIT_PARTITIONS:
            msg = f"partition must be one of {SUPPORTED_SPLIT_PARTITIONS!r}"
            raise Phase2ArtifactValidationError(msg)
        _require_positive_int(self.dimensionality, "dimensionality")
        _require_positive_shape(self.image_shape, "image_shape")
        _require_positive_shape(self.label_shape, "label_shape")
        _require_nonempty_string(self.image_dtype, "image_dtype")
        _require_nonempty_string(self.label_dtype, "label_dtype")
        _require_affine_4x4(self.image_affine, "image_affine")
        _require_affine_4x4(self.label_affine, "label_affine")
        _require_orientation(self.image_orientation, "image_orientation")
        _require_orientation(self.label_orientation, "label_orientation")
        _require_spacing_3d(self.image_spacing, "image_spacing")
        _require_spacing_3d(self.label_spacing, "label_spacing")
        _require_bool(self.image_finite, "image_finite")
        _require_bool(self.label_finite, "label_finite")
        _require_observed_label_values(self.observed_label_values, "observed_label_values")
        _require_allowed_label_values(self.allowed_label_values, "allowed_label_values")
        _require_bool(self.image_label_shape_match, "image_label_shape_match")
        _require_bool(self.image_label_affine_match, "image_label_affine_match")
        if (
            isinstance(self.tumor_label_value, bool)
            or not isinstance(self.tumor_label_value, int)
            or self.tumor_label_value < 0
        ):
            msg = "tumor_label_value must be a nonnegative integer"
            raise Phase2ArtifactValidationError(msg)
        if self.tumor_label_value not in self.allowed_label_values:
            msg = "tumor_label_value must be included in allowed_label_values"
            raise Phase2ArtifactValidationError(msg)
        _require_nonnegative_int(self.tumor_voxel_count, "tumor_voxel_count")
        _require_bool(self.empty_tumor, "empty_tumor")
        if self.empty_tumor != (self.tumor_voxel_count == 0):
            msg = "empty_tumor must be consistent with tumor_voxel_count"
            raise Phase2ArtifactValidationError(msg)
        _require_bool(self.qa_passed, "qa_passed")
        _require_string_tuple(self.failure_reasons, "failure_reasons", allow_empty=True)
        _require_geometry_label_failure_reasons(self.failure_reasons)
        if self.qa_passed and self.failure_reasons:
            msg = "qa_passed=True requires no failure_reasons"
            raise Phase2ArtifactValidationError(msg)
        if not self.qa_passed and not self.failure_reasons:
            msg = "qa_passed=False requires at least one deterministic failure reason"
            raise Phase2ArtifactValidationError(msg)


@dataclass(frozen=True, slots=True)
class GeometryLabelQaArtifact:
    """Intermediate Phase 2 geometry and label QA artifact."""

    schema_version: str
    stage: str
    created_at_utc: str
    git_commit: str
    config_hash: str
    manifest_hash: str
    split_hash: str
    case_count: int
    passed_case_count: int
    failed_case_count: int
    case_records: tuple[GeometryLabelQaCaseRecord, ...]
    qa_artifact_hash: str

    _expected_stage: ClassVar[str] = GEOMETRY_LABEL_QA_STAGE

    def __post_init__(self) -> None:
        _require_phase2_schema_version(self.schema_version)
        _require_stage(self.stage, self._expected_stage)
        _require_nonempty_string(self.created_at_utc, "created_at_utc")
        _require_nonempty_string(self.git_commit, "git_commit")
        _require_sha256(self.config_hash, "config_hash")
        _require_sha256(self.manifest_hash, "manifest_hash")
        _require_sha256(self.split_hash, "split_hash")
        _require_nonnegative_int(self.case_count, "case_count")
        _require_nonnegative_int(self.passed_case_count, "passed_case_count")
        _require_nonnegative_int(self.failed_case_count, "failed_case_count")
        _require_tuple_of(
            self.case_records,
            GeometryLabelQaCaseRecord,
            "case_records",
            allow_empty=False,
        )
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
        _require_sorted_case_records(self.case_records)
        _require_unique_values(
            (case.anonymous_case_id for case in self.case_records),
            "anonymous_case_id",
        )
        _require_sha256(self.qa_artifact_hash, "qa_artifact_hash")


@dataclass(frozen=True, slots=True)
class LesionComponentCaseRecord:
    """Intermediate per-case connected-component lesion summary record.

    Component ordering is deterministic for Phase 2 synthetic verification: lesions are sorted by
    descending voxel count, then by the lexicographically smallest voxel coordinate in the
    component. This is not a final scientific approval for real-data execution.
    """

    anonymous_patient_id: str
    anonymous_case_id: str
    partition: str
    analysis_performed: bool
    connectivity: int
    tumor_label_value: int
    voxel_volume_mm3: float | None
    tumor_voxel_count: int | None
    tumor_physical_volume_mm3: float | None
    lesion_count: int | None
    lesions: tuple[LesionSummaryRecord, ...]
    qa_passed: bool
    failure_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_safe_anonymous_id(self.anonymous_patient_id, "anonymous_patient_id")
        _require_safe_anonymous_id(self.anonymous_case_id, "anonymous_case_id")
        if self.partition not in SUPPORTED_SPLIT_PARTITIONS:
            msg = f"partition must be one of {SUPPORTED_SPLIT_PARTITIONS!r}"
            raise Phase2ArtifactValidationError(msg)
        _require_bool(self.analysis_performed, "analysis_performed")
        if self.connectivity not in SUPPORTED_CONNECTIVITY_VALUES:
            msg = f"connectivity must be one of {SUPPORTED_CONNECTIVITY_VALUES!r}"
            raise Phase2ArtifactValidationError(msg)
        if isinstance(self.tumor_label_value, bool) or not isinstance(self.tumor_label_value, int):
            msg = "tumor_label_value must be an integer"
            raise Phase2ArtifactValidationError(msg)
        _require_tuple_of(self.lesions, LesionSummaryRecord, "lesions", allow_empty=True)
        _require_bool(self.qa_passed, "qa_passed")
        _require_string_tuple(self.failure_reasons, "failure_reasons", allow_empty=True)

        if self.analysis_performed:
            self._validate_analyzed_measurements()
            return
        self._validate_skipped_measurements()

    def _validate_analyzed_measurements(self) -> None:
        if not self.qa_passed:
            msg = "analysis_performed=True requires qa_passed=True"
            raise Phase2ArtifactValidationError(msg)
        if self.failure_reasons:
            msg = "analysis_performed=True requires no failure_reasons"
            raise Phase2ArtifactValidationError(msg)
        if self.voxel_volume_mm3 is None:
            msg = "analysis_performed=True requires voxel_volume_mm3"
            raise Phase2ArtifactValidationError(msg)
        _require_positive_finite_float(self.voxel_volume_mm3, "voxel_volume_mm3")
        if self.tumor_voxel_count is None:
            msg = "analysis_performed=True requires tumor_voxel_count"
            raise Phase2ArtifactValidationError(msg)
        _require_nonnegative_int(self.tumor_voxel_count, "tumor_voxel_count")
        if self.tumor_physical_volume_mm3 is None:
            msg = "analysis_performed=True requires tumor_physical_volume_mm3"
            raise Phase2ArtifactValidationError(msg)
        _require_nonnegative_finite_float(
            self.tumor_physical_volume_mm3,
            "tumor_physical_volume_mm3",
        )
        if self.lesion_count is None:
            msg = "analysis_performed=True requires lesion_count"
            raise Phase2ArtifactValidationError(msg)
        _require_nonnegative_int(self.lesion_count, "lesion_count")
        if self.lesion_count != len(self.lesions):
            msg = "lesion_count must equal the number of lesion records"
            raise Phase2ArtifactValidationError(msg)
        _require_consecutive_lesion_indices(self.lesions)
        lesion_voxel_sum = sum(lesion.voxel_count for lesion in self.lesions)
        if lesion_voxel_sum != self.tumor_voxel_count:
            msg = "sum of lesion voxel counts must equal tumor_voxel_count"
            raise Phase2ArtifactValidationError(msg)
        lesion_volume_sum = sum(lesion.physical_volume_mm3 for lesion in self.lesions)
        if not math.isclose(
            lesion_volume_sum,
            self.tumor_physical_volume_mm3,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            msg = "sum of lesion physical volumes must equal tumor_physical_volume_mm3"
            raise Phase2ArtifactValidationError(msg)
        if self.tumor_voxel_count == 0:
            if self.tumor_physical_volume_mm3 != 0.0 or self.lesion_count != 0 or self.lesions:
                msg = "empty tumor requires zero physical volume and zero lesions"
                raise Phase2ArtifactValidationError(msg)
            return
        if self.tumor_physical_volume_mm3 <= 0.0 or self.lesion_count == 0:
            msg = "nonempty tumor requires positive physical volume and at least one lesion"
            raise Phase2ArtifactValidationError(msg)

    def _validate_skipped_measurements(self) -> None:
        if self.qa_passed:
            msg = "analysis_performed=False requires qa_passed=False"
            raise Phase2ArtifactValidationError(msg)
        if self.failure_reasons != (LESION_COMPONENTS_UPSTREAM_FAILURE_REASON,):
            msg = "skipped lesion records require the fixed upstream geometry-label QA failure"
            raise Phase2ArtifactValidationError(msg)
        if (
            self.voxel_volume_mm3 is not None
            or self.tumor_voxel_count is not None
            or self.tumor_physical_volume_mm3 is not None
            or self.lesion_count is not None
            or self.lesions
        ):
            msg = "skipped lesion records must not contain fabricated lesion measurements"
            raise Phase2ArtifactValidationError(msg)


@dataclass(frozen=True, slots=True)
class LesionComponentsArtifact:
    """Intermediate Phase 2 connected-component lesion-summary artifact."""

    schema_version: str
    stage: str
    created_at_utc: str
    git_commit: str
    config_hash: str
    manifest_hash: str
    split_hash: str
    geometry_qa_artifact_hash: str
    connectivity: int
    case_count: int
    analyzed_case_count: int
    skipped_case_count: int
    case_records: tuple[LesionComponentCaseRecord, ...]
    lesion_artifact_hash: str

    _expected_stage: ClassVar[str] = LESION_COMPONENTS_STAGE

    def __post_init__(self) -> None:
        _require_phase2_schema_version(self.schema_version)
        _require_stage(self.stage, self._expected_stage)
        _require_nonempty_string(self.created_at_utc, "created_at_utc")
        _require_nonempty_string(self.git_commit, "git_commit")
        _require_sha256(self.config_hash, "config_hash")
        _require_sha256(self.manifest_hash, "manifest_hash")
        _require_sha256(self.split_hash, "split_hash")
        _require_sha256(self.geometry_qa_artifact_hash, "geometry_qa_artifact_hash")
        if self.connectivity not in SUPPORTED_CONNECTIVITY_VALUES:
            msg = f"connectivity must be one of {SUPPORTED_CONNECTIVITY_VALUES!r}"
            raise Phase2ArtifactValidationError(msg)
        _require_nonnegative_int(self.case_count, "case_count")
        _require_nonnegative_int(self.analyzed_case_count, "analyzed_case_count")
        _require_nonnegative_int(self.skipped_case_count, "skipped_case_count")
        _require_tuple_of(
            self.case_records,
            LesionComponentCaseRecord,
            "case_records",
            allow_empty=False,
        )
        if self.case_count != len(self.case_records):
            msg = "case_count must equal the number of case_records"
            raise Phase2ArtifactValidationError(msg)
        if self.analyzed_case_count != sum(
            1 for case in self.case_records if case.analysis_performed
        ):
            msg = "analyzed_case_count must equal analyzed case records"
            raise Phase2ArtifactValidationError(msg)
        if self.skipped_case_count != sum(
            1 for case in self.case_records if not case.analysis_performed
        ):
            msg = "skipped_case_count must equal skipped case records"
            raise Phase2ArtifactValidationError(msg)
        if self.case_count != self.analyzed_case_count + self.skipped_case_count:
            msg = "case_count must equal analyzed_case_count + skipped_case_count"
            raise Phase2ArtifactValidationError(msg)
        _require_sorted_lesion_component_records(self.case_records)
        _require_unique_values(
            (case.anonymous_case_id for case in self.case_records),
            "anonymous_case_id",
        )
        for case in self.case_records:
            if case.connectivity != self.connectivity:
                msg = "all case connectivity values must match the artifact connectivity"
                raise Phase2ArtifactValidationError(msg)
        _require_sha256(self.lesion_artifact_hash, "lesion_artifact_hash")


@dataclass(frozen=True, slots=True)
class CtHistogramCaseRecord:
    """Intermediate per-case fixed-bin CT histogram summary record.

    Histogram bins follow NumPy semantics: bins are left-closed and right-open, except
    the final bin includes its right edge. Values below and above the configured range are
    counted separately and are never clipped into edge bins.
    """

    anonymous_patient_id: str
    anonymous_case_id: str
    partition: str
    analysis_performed: bool
    image_voxel_count: int | None
    intensity_min: float | None
    intensity_max: float | None
    intensity_mean: float | None
    intensity_std: float | None
    histogram_counts: tuple[int, ...]
    below_histogram_range_count: int | None
    above_histogram_range_count: int | None
    qa_passed: bool
    failure_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_safe_anonymous_id(self.anonymous_patient_id, "anonymous_patient_id")
        _require_safe_anonymous_id(self.anonymous_case_id, "anonymous_case_id")
        if self.partition not in SUPPORTED_SPLIT_PARTITIONS:
            msg = f"partition must be one of {SUPPORTED_SPLIT_PARTITIONS!r}"
            raise Phase2ArtifactValidationError(msg)
        _require_bool(self.analysis_performed, "analysis_performed")
        _require_bool(self.qa_passed, "qa_passed")
        _require_string_tuple(self.failure_reasons, "failure_reasons", allow_empty=True)
        _require_nonnegative_int_tuple(self.histogram_counts, "histogram_counts", allow_empty=True)
        if self.analysis_performed:
            self._validate_analyzed_measurements()
            return
        self._validate_skipped_measurements()

    def _validate_analyzed_measurements(self) -> None:
        if not self.qa_passed:
            msg = "analysis_performed=True requires qa_passed=True"
            raise Phase2ArtifactValidationError(msg)
        if self.failure_reasons:
            msg = "analysis_performed=True requires no failure_reasons"
            raise Phase2ArtifactValidationError(msg)
        if self.image_voxel_count is None:
            msg = "analysis_performed=True requires image_voxel_count"
            raise Phase2ArtifactValidationError(msg)
        _require_positive_int(self.image_voxel_count, "image_voxel_count")
        for field_name in (
            "intensity_min",
            "intensity_max",
            "intensity_mean",
            "intensity_std",
        ):
            if getattr(self, field_name) is None:
                msg = f"analysis_performed=True requires {field_name}"
                raise Phase2ArtifactValidationError(msg)
        _require_finite_float(self.intensity_min, "intensity_min")
        _require_finite_float(self.intensity_max, "intensity_max")
        _require_finite_float(self.intensity_mean, "intensity_mean")
        _require_nonnegative_finite_float(self.intensity_std, "intensity_std")
        if cast(float, self.intensity_min) > cast(float, self.intensity_max):
            msg = "intensity_min must be less than or equal to intensity_max"
            raise Phase2ArtifactValidationError(msg)
        if not self.histogram_counts:
            msg = "analysis_performed=True requires histogram_counts"
            raise Phase2ArtifactValidationError(msg)
        if self.below_histogram_range_count is None:
            msg = "analysis_performed=True requires below_histogram_range_count"
            raise Phase2ArtifactValidationError(msg)
        if self.above_histogram_range_count is None:
            msg = "analysis_performed=True requires above_histogram_range_count"
            raise Phase2ArtifactValidationError(msg)
        _require_nonnegative_int(
            self.below_histogram_range_count,
            "below_histogram_range_count",
        )
        _require_nonnegative_int(
            self.above_histogram_range_count,
            "above_histogram_range_count",
        )
        counted_voxels = (
            sum(self.histogram_counts)
            + self.below_histogram_range_count
            + self.above_histogram_range_count
        )
        if counted_voxels != self.image_voxel_count:
            msg = "histogram and out-of-range counts must equal image_voxel_count"
            raise Phase2ArtifactValidationError(msg)

    def _validate_skipped_measurements(self) -> None:
        if self.qa_passed:
            msg = "analysis_performed=False requires qa_passed=False"
            raise Phase2ArtifactValidationError(msg)
        if self.failure_reasons != (DEVELOPMENT_DATA_SUMMARY_UPSTREAM_FAILURE_REASON,):
            msg = "skipped CT histogram records require the fixed upstream QA failure"
            raise Phase2ArtifactValidationError(msg)
        if (
            self.image_voxel_count is not None
            or self.intensity_min is not None
            or self.intensity_max is not None
            or self.intensity_mean is not None
            or self.intensity_std is not None
            or self.histogram_counts
            or self.below_histogram_range_count is not None
            or self.above_histogram_range_count is not None
        ):
            msg = "skipped CT histogram records must not contain fabricated measurements"
            raise Phase2ArtifactValidationError(msg)


@dataclass(frozen=True, slots=True)
class DevelopmentDataSummaryArtifact:
    """Intermediate Phase 2 fixed CT histogram and deterministic dataset-summary artifact."""

    schema_version: str
    stage: str
    created_at_utc: str
    git_commit: str
    config_hash: str
    manifest_hash: str
    split_hash: str
    geometry_qa_artifact_hash: str
    lesion_artifact_hash: str
    histogram_bin_edges: tuple[float, ...]
    histogram_bin_count: int
    case_count: int
    analyzed_case_count: int
    skipped_case_count: int
    case_records: tuple[CtHistogramCaseRecord, ...]
    aggregate_image_voxel_count: int
    aggregate_intensity_min: float | None
    aggregate_intensity_max: float | None
    aggregate_intensity_mean: float | None
    aggregate_intensity_std: float | None
    aggregate_histogram_counts: tuple[int, ...]
    aggregate_below_histogram_range_count: int
    aggregate_above_histogram_range_count: int
    total_tumor_voxel_count: int
    total_tumor_physical_volume_mm3: float
    total_lesion_count: int
    lesion_volume_min_mm3: float | None
    lesion_volume_max_mm3: float | None
    lesion_volume_mean_mm3: float | None
    lesion_volume_median_mm3: float | None
    summary_artifact_hash: str

    _expected_stage: ClassVar[str] = DEVELOPMENT_DATA_SUMMARY_STAGE

    def __post_init__(self) -> None:
        _require_phase2_schema_version(self.schema_version)
        _require_stage(self.stage, self._expected_stage)
        _require_nonempty_string(self.created_at_utc, "created_at_utc")
        _require_nonempty_string(self.git_commit, "git_commit")
        _require_sha256(self.config_hash, "config_hash")
        _require_sha256(self.manifest_hash, "manifest_hash")
        _require_sha256(self.split_hash, "split_hash")
        _require_sha256(self.geometry_qa_artifact_hash, "geometry_qa_artifact_hash")
        _require_sha256(self.lesion_artifact_hash, "lesion_artifact_hash")
        _require_histogram_edges(self.histogram_bin_edges, "histogram_bin_edges")
        _require_positive_int(self.histogram_bin_count, "histogram_bin_count")
        if len(self.histogram_bin_edges) != self.histogram_bin_count + 1:
            msg = "histogram_bin_edges count must equal histogram_bin_count plus one"
            raise Phase2ArtifactValidationError(msg)
        _require_nonnegative_int(self.case_count, "case_count")
        _require_nonnegative_int(self.analyzed_case_count, "analyzed_case_count")
        _require_nonnegative_int(self.skipped_case_count, "skipped_case_count")
        _require_tuple_of(
            self.case_records,
            CtHistogramCaseRecord,
            "case_records",
            allow_empty=False,
        )
        self._validate_case_counts()
        _require_sorted_ct_histogram_records(self.case_records)
        _require_unique_values(
            (case.anonymous_case_id for case in self.case_records),
            "anonymous_case_id",
        )
        self._validate_histogram_aggregates()
        self._validate_intensity_aggregates()
        self._validate_lesion_aggregates()
        _require_sha256(self.summary_artifact_hash, "summary_artifact_hash")

    def _validate_case_counts(self) -> None:
        if self.case_count != len(self.case_records):
            msg = "case_count must equal the number of case_records"
            raise Phase2ArtifactValidationError(msg)
        analyzed = sum(1 for case in self.case_records if case.analysis_performed)
        skipped = sum(1 for case in self.case_records if not case.analysis_performed)
        if self.analyzed_case_count != analyzed:
            msg = "analyzed_case_count must equal analyzed case records"
            raise Phase2ArtifactValidationError(msg)
        if self.skipped_case_count != skipped:
            msg = "skipped_case_count must equal skipped case records"
            raise Phase2ArtifactValidationError(msg)
        if self.case_count != self.analyzed_case_count + self.skipped_case_count:
            msg = "case_count must equal analyzed_case_count + skipped_case_count"
            raise Phase2ArtifactValidationError(msg)

    def _validate_histogram_aggregates(self) -> None:
        _require_nonnegative_int(self.aggregate_image_voxel_count, "aggregate_image_voxel_count")
        _require_nonnegative_int_tuple(
            self.aggregate_histogram_counts,
            "aggregate_histogram_counts",
            allow_empty=False,
        )
        if len(self.aggregate_histogram_counts) != self.histogram_bin_count:
            msg = "aggregate_histogram_counts length must equal histogram_bin_count"
            raise Phase2ArtifactValidationError(msg)
        _require_nonnegative_int(
            self.aggregate_below_histogram_range_count,
            "aggregate_below_histogram_range_count",
        )
        _require_nonnegative_int(
            self.aggregate_above_histogram_range_count,
            "aggregate_above_histogram_range_count",
        )
        analyzed_records = tuple(case for case in self.case_records if case.analysis_performed)
        for case in analyzed_records:
            if len(case.histogram_counts) != self.histogram_bin_count:
                msg = "analyzed case histogram_counts length must equal histogram_bin_count"
                raise Phase2ArtifactValidationError(msg)
        summed_histogram = tuple(
            sum(case.histogram_counts[index] for case in analyzed_records)
            for index in range(self.histogram_bin_count)
        )
        if self.aggregate_histogram_counts != summed_histogram:
            msg = "aggregate_histogram_counts must equal the elementwise case histogram sum"
            raise Phase2ArtifactValidationError(msg)
        if self.aggregate_image_voxel_count != sum(
            cast(int, case.image_voxel_count) for case in analyzed_records
        ):
            msg = "aggregate_image_voxel_count must equal analyzed case voxel counts"
            raise Phase2ArtifactValidationError(msg)
        if self.aggregate_below_histogram_range_count != sum(
            cast(int, case.below_histogram_range_count) for case in analyzed_records
        ):
            msg = "aggregate_below_histogram_range_count must equal analyzed case counts"
            raise Phase2ArtifactValidationError(msg)
        if self.aggregate_above_histogram_range_count != sum(
            cast(int, case.above_histogram_range_count) for case in analyzed_records
        ):
            msg = "aggregate_above_histogram_range_count must equal analyzed case counts"
            raise Phase2ArtifactValidationError(msg)

    def _validate_intensity_aggregates(self) -> None:
        fields_to_check = (
            "aggregate_intensity_min",
            "aggregate_intensity_max",
            "aggregate_intensity_mean",
            "aggregate_intensity_std",
        )
        if self.analyzed_case_count == 0:
            if self.aggregate_image_voxel_count != 0 or any(
                getattr(self, field_name) is not None for field_name in fields_to_check
            ):
                msg = "zero analyzed cases require absent aggregate intensity measurements"
                raise Phase2ArtifactValidationError(msg)
            return
        for field_name in fields_to_check:
            if getattr(self, field_name) is None:
                msg = f"analyzed cases require {field_name}"
                raise Phase2ArtifactValidationError(msg)
        _require_finite_float(self.aggregate_intensity_min, "aggregate_intensity_min")
        _require_finite_float(self.aggregate_intensity_max, "aggregate_intensity_max")
        _require_finite_float(self.aggregate_intensity_mean, "aggregate_intensity_mean")
        _require_nonnegative_finite_float(
            self.aggregate_intensity_std,
            "aggregate_intensity_std",
        )
        if cast(float, self.aggregate_intensity_min) > cast(float, self.aggregate_intensity_max):
            msg = "aggregate_intensity_min must be less than or equal to aggregate_intensity_max"
            raise Phase2ArtifactValidationError(msg)

    def _validate_lesion_aggregates(self) -> None:
        _require_nonnegative_int(self.total_tumor_voxel_count, "total_tumor_voxel_count")
        _require_nonnegative_finite_float(
            self.total_tumor_physical_volume_mm3,
            "total_tumor_physical_volume_mm3",
        )
        _require_nonnegative_int(self.total_lesion_count, "total_lesion_count")
        lesion_fields = (
            "lesion_volume_min_mm3",
            "lesion_volume_max_mm3",
            "lesion_volume_mean_mm3",
            "lesion_volume_median_mm3",
        )
        if self.total_lesion_count == 0:
            if any(getattr(self, field_name) is not None for field_name in lesion_fields):
                msg = "zero total lesions require absent lesion volume summary statistics"
                raise Phase2ArtifactValidationError(msg)
            return
        for field_name in lesion_fields:
            if getattr(self, field_name) is None:
                msg = f"nonzero total lesions require {field_name}"
                raise Phase2ArtifactValidationError(msg)
            _require_positive_finite_float(getattr(self, field_name), field_name)
        if cast(float, self.lesion_volume_min_mm3) > cast(float, self.lesion_volume_max_mm3):
            msg = "lesion_volume_min_mm3 must be less than or equal to lesion_volume_max_mm3"
            raise Phase2ArtifactValidationError(msg)


@dataclass(frozen=True, slots=True)
class LeakageAuditArtifact:
    """Machine-readable Phase 2 leakage-audit evidence."""

    schema_version: str
    stage: str
    created_at_utc: str
    git_commit: str
    config_hash: str
    manifest_hash: str
    split_hash: str
    split_policy_version: str
    split_seed: int
    manifest_case_count: int
    manifest_patient_count: int
    assignment_complete: bool
    patient_counts_by_partition: Mapping[str, int]
    case_counts_by_partition: Mapping[str, int]
    pairwise_patient_overlap_counts: Mapping[str, int]
    pairwise_case_overlap_counts: Mapping[str, int]
    image_hash_cross_partition_overlap_count: int
    label_hash_cross_partition_overlap_count: int
    image_label_pair_cross_partition_overlap_count: int
    duplicate_image_hash_findings: tuple[str, ...]
    duplicate_label_hash_findings: tuple[str, ...]
    finding_codes: tuple[str, ...]
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
        _require_sha256(self.config_hash, "config_hash")
        _require_sha256(self.manifest_hash, "manifest_hash")
        _require_sha256(self.split_hash, "split_hash")
        _require_nonempty_string(self.split_policy_version, "split_policy_version")
        _require_nonnegative_int(self.split_seed, "split_seed")
        _require_positive_int(self.manifest_case_count, "manifest_case_count")
        _require_positive_int(self.manifest_patient_count, "manifest_patient_count")
        _require_bool(self.assignment_complete, "assignment_complete")
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
        _require_nonnegative_int(
            self.image_hash_cross_partition_overlap_count,
            "image_hash_cross_partition_overlap_count",
        )
        _require_nonnegative_int(
            self.label_hash_cross_partition_overlap_count,
            "label_hash_cross_partition_overlap_count",
        )
        _require_nonnegative_int(
            self.image_label_pair_cross_partition_overlap_count,
            "image_label_pair_cross_partition_overlap_count",
        )
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
        _require_leakage_finding_codes(self.finding_codes)
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
        self._validate_finding_consistency(
            patient_counts,
            case_counts,
            patient_overlaps,
            case_overlaps,
        )
        if self.audit_passed:
            _require_all_zero(patient_overlaps, "pairwise_patient_overlap_counts")
            _require_all_zero(case_overlaps, "pairwise_case_overlap_counts")
            if not self.assignment_complete:
                msg = "audit_passed=True requires assignment_complete=True"
                raise Phase2ArtifactValidationError(msg)
            if any(value == 0 for value in patient_counts.values()) or any(
                value == 0 for value in case_counts.values()
            ):
                msg = "audit_passed=True requires every partition to be nonempty"
                raise Phase2ArtifactValidationError(msg)
            if self.image_hash_cross_partition_overlap_count != 0:
                msg = "audit_passed=True requires zero cross-partition image hash overlap"
                raise Phase2ArtifactValidationError(msg)
            if self.label_hash_cross_partition_overlap_count != 0:
                msg = "audit_passed=True requires zero cross-partition label hash overlap"
                raise Phase2ArtifactValidationError(msg)
            if self.image_label_pair_cross_partition_overlap_count != 0:
                msg = "audit_passed=True requires zero cross-partition image/label pair overlap"
                raise Phase2ArtifactValidationError(msg)
            if self.finding_codes:
                msg = "audit_passed=True requires no finding_codes"
                raise Phase2ArtifactValidationError(msg)
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

    def _validate_finding_consistency(
        self,
        patient_counts: Mapping[str, int],
        case_counts: Mapping[str, int],
        patient_overlaps: Mapping[str, int],
        case_overlaps: Mapping[str, int],
    ) -> None:
        if self.critical_finding_count != len(self.finding_codes):
            msg = "critical_finding_count must equal the number of finding_codes"
            raise Phase2ArtifactValidationError(msg)
        if tuple(self.unresolved_findings) != tuple(self.finding_codes):
            msg = "unresolved_findings must match deterministic finding_codes"
            raise Phase2ArtifactValidationError(msg)
        expected_presence = {
            "empty_partition": any(value == 0 for value in patient_counts.values())
            or any(value == 0 for value in case_counts.values()),
            "patient_partition_overlap": any(value != 0 for value in patient_overlaps.values()),
            "case_partition_overlap": any(value != 0 for value in case_overlaps.values()),
            "image_hash_cross_partition_overlap": (
                self.image_hash_cross_partition_overlap_count != 0
            ),
            "label_hash_cross_partition_overlap": (
                self.label_hash_cross_partition_overlap_count != 0
            ),
            "image_label_pair_cross_partition_overlap": (
                self.image_label_pair_cross_partition_overlap_count != 0
            ),
        }
        for code, should_be_present in expected_presence.items():
            if should_be_present != (code in self.finding_codes):
                msg = f"finding_codes must consistently represent {code}"
                raise Phase2ArtifactValidationError(msg)
        expected_image_findings = (
            ("image_hash_cross_partition_overlap",)
            if self.image_hash_cross_partition_overlap_count
            else ()
        )
        if self.duplicate_image_hash_findings != expected_image_findings:
            msg = "duplicate_image_hash_findings must match image hash overlap count"
            raise Phase2ArtifactValidationError(msg)
        expected_label_findings = (
            ("label_hash_cross_partition_overlap",)
            if self.label_hash_cross_partition_overlap_count
            else ()
        )
        if self.duplicate_label_hash_findings != expected_label_findings:
            msg = "duplicate_label_hash_findings must match label hash overlap count"
            raise Phase2ArtifactValidationError(msg)


Phase2Artifact: TypeAlias = (
    DatasetCaseRecord
    | DatasetManifest
    | SplitAssignment
    | DevelopmentSplitManifest
    | LesionSummaryRecord
    | CaseQaRecord
    | DevelopmentQaArtifact
    | GeometryLabelQaCaseRecord
    | GeometryLabelQaArtifact
    | LesionComponentCaseRecord
    | LesionComponentsArtifact
    | CtHistogramCaseRecord
    | DevelopmentDataSummaryArtifact
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
    GeometryLabelQaCaseRecord,
    GeometryLabelQaArtifact,
    LesionComponentCaseRecord,
    LesionComponentsArtifact,
    CtHistogramCaseRecord,
    DevelopmentDataSummaryArtifact,
    LeakageAuditArtifact,
)
_MANIFEST_TYPES: dict[str, type[Phase2Artifact]] = {
    DATASET_MANIFEST_TYPE: DatasetManifest,
    DEVELOPMENT_SPLIT_MANIFEST_TYPE: DevelopmentSplitManifest,
}
_STAGE_TYPES: dict[str, type[Phase2Artifact]] = {
    GEOMETRY_LABEL_QA_STAGE: GeometryLabelQaArtifact,
    LESION_COMPONENTS_STAGE: LesionComponentsArtifact,
    DEVELOPMENT_DATA_SUMMARY_STAGE: DevelopmentDataSummaryArtifact,
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
    "image_shape",
    "label_shape",
    "image_affine",
    "label_affine",
    "image_orientation",
    "label_orientation",
    "image_spacing",
    "label_spacing",
    "observed_label_values",
    "allowed_label_values",
    "histogram_bin_edges",
    "histogram_counts",
    "aggregate_histogram_counts",
    "failure_reasons",
    "duplicate_image_hash_findings",
    "duplicate_label_hash_findings",
    "finding_codes",
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
        nested_type = _nested_record_type(artifact_type, key)
        if nested_type is not None:
            if not isinstance(value, list):
                msg = f"{key} must be a JSON array"
                raise Phase2ArtifactValidationError(msg)
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


def _nested_record_type(
    artifact_type: type[Phase2Artifact],
    field_name: str,
) -> type[Phase2Artifact] | None:
    if artifact_type is GeometryLabelQaArtifact and field_name == "case_records":
        return GeometryLabelQaCaseRecord
    if artifact_type is LesionComponentsArtifact and field_name == "case_records":
        return LesionComponentCaseRecord
    if artifact_type is DevelopmentDataSummaryArtifact and field_name == "case_records":
        return CtHistogramCaseRecord
    return _NESTED_RECORD_FIELDS.get(field_name)


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


def _require_nonnegative_int_tuple(
    value: object,
    field_name: str,
    *,
    allow_empty: bool,
) -> None:
    if not isinstance(value, tuple):
        msg = f"{field_name} must be a tuple of nonnegative integers"
        raise Phase2ArtifactValidationError(msg)
    if not allow_empty and not value:
        msg = f"{field_name} must not be empty"
        raise Phase2ArtifactValidationError(msg)
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value):
        msg = f"{field_name} must contain nonnegative integers"
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


def _require_positive_shape(value: object, field_name: str) -> None:
    if (
        not isinstance(value, tuple)
        or not value
        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in value)
    ):
        msg = f"{field_name} must contain positive integers"
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


def _require_observed_label_values(value: object, field_name: str) -> None:
    if not isinstance(value, tuple):
        msg = f"{field_name} must be a tuple"
        raise Phase2ArtifactValidationError(msg)
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int | float) or not math.isfinite(item):
            msg = f"{field_name} must contain finite numeric label values"
            raise Phase2ArtifactValidationError(msg)
        if isinstance(item, float) and item.is_integer():
            msg = f"{field_name} must store integer-valued labels as integers"
            raise Phase2ArtifactValidationError(msg)
    if tuple(sorted(set(value))) != value:
        msg = f"{field_name} must be unique and sorted"
        raise Phase2ArtifactValidationError(msg)


def _require_geometry_label_failure_reasons(value: tuple[str, ...]) -> None:
    code_order = {code: index for index, code in enumerate(GEOMETRY_LABEL_QA_FAILURE_CODES)}
    previous_order = -1
    seen: set[str] = set()
    for code in value:
        if code not in code_order:
            msg = f"failure_reasons contains unsupported geometry-label QA code: {code!r}"
            raise Phase2ArtifactValidationError(msg)
        if code in seen:
            msg = "failure_reasons must not contain duplicate codes"
            raise Phase2ArtifactValidationError(msg)
        seen.add(code)
        current_order = code_order[code]
        if current_order <= previous_order:
            msg = "failure_reasons must follow the deterministic geometry-label code order"
            raise Phase2ArtifactValidationError(msg)
        previous_order = current_order


def _require_leakage_finding_codes(value: object) -> None:
    _require_string_tuple(value, "finding_codes", allow_empty=True)
    codes = cast(tuple[str, ...], value)
    code_order = {code: index for index, code in enumerate(LEAKAGE_AUDIT_FINDING_CODES)}
    previous_order = -1
    seen: set[str] = set()
    for code in codes:
        if code not in code_order:
            msg = f"finding_codes contains unsupported leakage-audit code: {code!r}"
            raise Phase2ArtifactValidationError(msg)
        if code in seen:
            msg = "finding_codes must not contain duplicate codes"
            raise Phase2ArtifactValidationError(msg)
        seen.add(code)
        current_order = code_order[code]
        if current_order <= previous_order:
            msg = "finding_codes must follow the deterministic leakage-audit code order"
            raise Phase2ArtifactValidationError(msg)
        previous_order = current_order


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
    expected = tuple(sorted(case_records, key=_case_record_sort_key))
    if case_records != expected:
        msg = "case_records must be ordered by anonymous_patient_id and anonymous_case_id"
        raise Phase2ArtifactValidationError(msg)


def _require_sorted_case_records(
    case_records: tuple[GeometryLabelQaCaseRecord, ...],
) -> None:
    expected = tuple(sorted(case_records, key=_case_record_sort_key))
    if case_records != expected:
        msg = "case_records must be ordered by anonymous_patient_id and anonymous_case_id"
        raise Phase2ArtifactValidationError(msg)


def _require_sorted_lesion_component_records(
    case_records: tuple[LesionComponentCaseRecord, ...],
) -> None:
    expected = tuple(sorted(case_records, key=_case_record_sort_key))
    if case_records != expected:
        msg = "case_records must be ordered by anonymous_patient_id and anonymous_case_id"
        raise Phase2ArtifactValidationError(msg)


def _require_sorted_ct_histogram_records(
    case_records: tuple[CtHistogramCaseRecord, ...],
) -> None:
    expected = tuple(sorted(case_records, key=_case_record_sort_key))
    if case_records != expected:
        msg = "case_records must be ordered by anonymous_patient_id and anonymous_case_id"
        raise Phase2ArtifactValidationError(msg)


def _case_record_sort_key(
    case: (
        CaseQaRecord | GeometryLabelQaCaseRecord | LesionComponentCaseRecord | CtHistogramCaseRecord
    ),
) -> tuple[str, str]:
    return (case.anonymous_patient_id, case.anonymous_case_id)


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
