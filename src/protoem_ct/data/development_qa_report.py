"""Assemble the final Phase 2 DevelopmentQaArtifact JSON report from saved artifacts."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final

from protoem_ct.artifacts import (
    DEVELOPMENT_QA_STAGE,
    PHASE2_DEVELOPMENT_COHORT_ROLE,
    PHASE2_SCHEMA_VERSION,
    CaseQaRecord,
    CtHistogramCaseRecord,
    DatasetManifest,
    DevelopmentDataSummaryArtifact,
    DevelopmentQaArtifact,
    DevelopmentSplitManifest,
    GeometryLabelQaArtifact,
    GeometryLabelQaCaseRecord,
    LesionComponentCaseRecord,
    LesionComponentsArtifact,
    Phase2ArtifactError,
    hash_dataset_manifest,
    hash_development_data_summary,
    hash_development_qa,
    hash_development_split,
    hash_geometry_label_qa,
    hash_lesion_components,
    phase2_artifact_from_json,
    phase2_artifact_to_json,
    sha256_json,
)

DEVELOPMENT_QA_REPORT_CONTRACT_VERSION: Final[str] = "development_qa_report_v1"
DEVELOPMENT_QA_REPORT_CONFIG_PAYLOAD_TYPE: Final[str] = "phase2-development-qa-report-config-v1"
_ZERO_SHA256: Final[str] = "0" * 64


class DevelopmentQaReportError(ValueError):
    """Base exception for Phase 2 final development-QA report assembly failures."""


class InvalidDevelopmentQaReportConfigError(DevelopmentQaReportError):
    """Raised when report assembly configuration is unsupported."""


class InvalidDevelopmentQaReportInputError(DevelopmentQaReportError):
    """Raised when an input artifact path or artifact type is invalid."""


class DevelopmentQaReportHashMismatchError(InvalidDevelopmentQaReportInputError):
    """Raised when a stored input artifact hash does not match recomputed content."""


class DevelopmentQaReportLinkageError(InvalidDevelopmentQaReportInputError):
    """Raised when the five saved artifacts are not linked consistently."""


class DevelopmentQaReportCaseAssemblyError(DevelopmentQaReportError):
    """Raised when final per-case QA assembly would violate invariants."""


class DevelopmentQaReportAggregateError(DevelopmentQaReportError):
    """Raised when final aggregate fields cannot be copied consistently."""


class UnsafeDevelopmentQaReportOutputPathError(DevelopmentQaReportError):
    """Raised when the requested report output path violates JSON output safety."""


class ExistingDevelopmentQaReportOutputError(DevelopmentQaReportError):
    """Raised when report publication would overwrite an existing output."""


class DevelopmentQaReportPublicationError(DevelopmentQaReportError):
    """Raised when validated final report JSON cannot be published atomically."""


@dataclass(frozen=True, slots=True)
class DevelopmentQaReportConfig:
    """Explicit JSON-report assembly contract with no hidden scientific settings."""

    report_contract_version: str

    def __post_init__(self) -> None:
        if self.report_contract_version != DEVELOPMENT_QA_REPORT_CONTRACT_VERSION:
            msg = f"unsupported report_contract_version: {self.report_contract_version!r}"
            raise InvalidDevelopmentQaReportConfigError(msg)


def development_qa_report_config_hash_payload(
    config: DevelopmentQaReportConfig,
) -> dict[str, object]:
    """Return the canonical payload for a final development-QA report config hash."""
    return {
        "payload_type": DEVELOPMENT_QA_REPORT_CONFIG_PAYLOAD_TYPE,
        "report_contract_version": config.report_contract_version,
        "schema_version": PHASE2_SCHEMA_VERSION,
    }


def hash_development_qa_report_config(config: DevelopmentQaReportConfig) -> str:
    """Return the deterministic SHA-256 hash of explicit report assembly settings."""
    return sha256_json(development_qa_report_config_hash_payload(config))


def assemble_development_qa_report(
    manifest_path: Path,
    split_path: Path,
    geometry_qa_artifact_path: Path,
    lesion_artifact_path: Path,
    development_summary_artifact_path: Path,
    *,
    output_path: Path,
    config: DevelopmentQaReportConfig,
    git_commit: str,
    created_at_utc: str,
) -> DevelopmentQaArtifact:
    """Assemble and publish one deterministic final DevelopmentQaArtifact JSON report."""
    input_paths = (
        _validate_input_json_path(manifest_path, "manifest"),
        _validate_input_json_path(split_path, "split"),
        _validate_input_json_path(geometry_qa_artifact_path, "geometry QA artifact"),
        _validate_input_json_path(lesion_artifact_path, "lesion artifact"),
        _validate_input_json_path(
            development_summary_artifact_path,
            "development summary artifact",
        ),
    )
    safe_output_path = _validate_report_output_path(output_path, input_paths=input_paths)
    _require_explicit_metadata(git_commit, "git_commit")
    _require_explicit_metadata(created_at_utc, "created_at_utc")

    manifest = _load_and_verify_dataset_manifest(input_paths[0].read_text(encoding="utf-8"))
    split = _load_and_verify_split(input_paths[1].read_text(encoding="utf-8"))
    geometry = _load_and_verify_geometry(input_paths[2].read_text(encoding="utf-8"))
    lesion = _load_and_verify_lesion(input_paths[3].read_text(encoding="utf-8"))
    summary = _load_and_verify_summary(input_paths[4].read_text(encoding="utf-8"))
    _verify_linkage(manifest, split, geometry, lesion, summary)

    assignments_by_case = {
        assignment.anonymous_case_id: assignment for assignment in split.assignments
    }
    final_case_records = tuple(
        sorted(
            (
                _assemble_case_record(
                    geometry_case=geometry_case,
                    lesion_case=lesion_case,
                    summary_case=summary_case,
                    partition=assignments_by_case[geometry_case.anonymous_case_id].partition,
                    histogram_bin_edges=summary.histogram_bin_edges,
                )
                for geometry_case, lesion_case, summary_case in zip(
                    geometry.case_records,
                    lesion.case_records,
                    summary.case_records,
                    strict=True,
                )
            ),
            key=lambda record: (record.anonymous_patient_id, record.anonymous_case_id),
        )
    )
    _verify_final_case_counts(final_case_records, geometry, lesion, summary)

    artifact_without_hash = DevelopmentQaArtifact(
        schema_version=PHASE2_SCHEMA_VERSION,
        stage=DEVELOPMENT_QA_STAGE,
        created_at_utc=created_at_utc,
        git_commit=git_commit,
        config_hash=hash_development_qa_report_config(config),
        manifest_hash=manifest.manifest_hash,
        split_hash=split.split_hash,
        geometry_qa_artifact_hash=geometry.qa_artifact_hash,
        lesion_artifact_hash=lesion.lesion_artifact_hash,
        development_summary_artifact_hash=summary.summary_artifact_hash,
        connectivity=lesion.connectivity,
        affine_tolerance=None,
        histogram_bin_edges=summary.histogram_bin_edges,
        histogram_bin_count=summary.histogram_bin_count,
        case_count=len(final_case_records),
        passed_case_count=sum(1 for record in final_case_records if record.qa_passed),
        failed_case_count=sum(1 for record in final_case_records if not record.qa_passed),
        case_records=final_case_records,
        aggregate_image_voxel_count=summary.aggregate_image_voxel_count,
        aggregate_intensity_min=summary.aggregate_intensity_min,
        aggregate_intensity_max=summary.aggregate_intensity_max,
        aggregate_intensity_mean=summary.aggregate_intensity_mean,
        aggregate_intensity_std=summary.aggregate_intensity_std,
        aggregate_histogram_counts=summary.aggregate_histogram_counts,
        aggregate_below_histogram_range_count=summary.aggregate_below_histogram_range_count,
        aggregate_above_histogram_range_count=summary.aggregate_above_histogram_range_count,
        total_tumor_voxel_count=summary.total_tumor_voxel_count,
        total_tumor_physical_volume_mm3=summary.total_tumor_physical_volume_mm3,
        total_lesion_count=summary.total_lesion_count,
        lesion_volume_min_mm3=summary.lesion_volume_min_mm3,
        lesion_volume_max_mm3=summary.lesion_volume_max_mm3,
        lesion_volume_mean_mm3=summary.lesion_volume_mean_mm3,
        lesion_volume_median_mm3=summary.lesion_volume_median_mm3,
        qa_artifact_hash=_ZERO_SHA256,
    )
    qa_hash = hash_development_qa(artifact_without_hash)
    artifact = replace(artifact_without_hash, qa_artifact_hash=qa_hash)
    if hash_development_qa(artifact) != qa_hash:
        msg = "final development-QA artifact hash verification failed"
        raise DevelopmentQaReportPublicationError(msg)

    _verify_final_aggregates(artifact, summary)
    _publish_report_json(artifact, safe_output_path)
    return artifact


def _load_and_verify_dataset_manifest(text: str) -> DatasetManifest:
    try:
        artifact = phase2_artifact_from_json(text)
    except Phase2ArtifactError as exc:
        msg = "input manifest is not a valid Phase 2 dataset manifest"
        raise InvalidDevelopmentQaReportInputError(msg) from exc
    if not isinstance(artifact, DatasetManifest):
        msg = "input artifact must be a DatasetManifest"
        raise InvalidDevelopmentQaReportInputError(msg)
    if artifact.cohort_role != PHASE2_DEVELOPMENT_COHORT_ROLE:
        msg = "input dataset manifest must use the development cohort role"
        raise InvalidDevelopmentQaReportInputError(msg)
    if hash_dataset_manifest(artifact) != artifact.manifest_hash:
        msg = "input dataset manifest hash does not match its content"
        raise DevelopmentQaReportHashMismatchError(msg)
    return artifact


def _load_and_verify_split(text: str) -> DevelopmentSplitManifest:
    try:
        artifact = phase2_artifact_from_json(text)
    except Phase2ArtifactError as exc:
        msg = "input split is not a valid Phase 2 development split"
        raise InvalidDevelopmentQaReportInputError(msg) from exc
    if not isinstance(artifact, DevelopmentSplitManifest):
        msg = "input artifact must be a DevelopmentSplitManifest"
        raise InvalidDevelopmentQaReportInputError(msg)
    if hash_development_split(artifact) != artifact.split_hash:
        msg = "input split hash does not match its content"
        raise DevelopmentQaReportHashMismatchError(msg)
    return artifact


def _load_and_verify_geometry(text: str) -> GeometryLabelQaArtifact:
    try:
        artifact = phase2_artifact_from_json(text)
    except Phase2ArtifactError as exc:
        msg = "input geometry QA artifact is not valid Phase 2 JSON"
        raise InvalidDevelopmentQaReportInputError(msg) from exc
    if not isinstance(artifact, GeometryLabelQaArtifact):
        msg = "input artifact must be a GeometryLabelQaArtifact"
        raise InvalidDevelopmentQaReportInputError(msg)
    if hash_geometry_label_qa(artifact) != artifact.qa_artifact_hash:
        msg = "input geometry QA artifact hash does not match its content"
        raise DevelopmentQaReportHashMismatchError(msg)
    return artifact


def _load_and_verify_lesion(text: str) -> LesionComponentsArtifact:
    try:
        artifact = phase2_artifact_from_json(text)
    except Phase2ArtifactError as exc:
        msg = "input lesion artifact is not valid Phase 2 JSON"
        raise InvalidDevelopmentQaReportInputError(msg) from exc
    if not isinstance(artifact, LesionComponentsArtifact):
        msg = "input artifact must be a LesionComponentsArtifact"
        raise InvalidDevelopmentQaReportInputError(msg)
    if hash_lesion_components(artifact) != artifact.lesion_artifact_hash:
        msg = "input lesion artifact hash does not match its content"
        raise DevelopmentQaReportHashMismatchError(msg)
    return artifact


def _load_and_verify_summary(text: str) -> DevelopmentDataSummaryArtifact:
    try:
        artifact = phase2_artifact_from_json(text)
    except Phase2ArtifactError as exc:
        msg = "input development summary artifact is not valid Phase 2 JSON"
        raise InvalidDevelopmentQaReportInputError(msg) from exc
    if not isinstance(artifact, DevelopmentDataSummaryArtifact):
        msg = "input artifact must be a DevelopmentDataSummaryArtifact"
        raise InvalidDevelopmentQaReportInputError(msg)
    if hash_development_data_summary(artifact) != artifact.summary_artifact_hash:
        msg = "input development summary artifact hash does not match its content"
        raise DevelopmentQaReportHashMismatchError(msg)
    return artifact


def _verify_linkage(
    manifest: DatasetManifest,
    split: DevelopmentSplitManifest,
    geometry: GeometryLabelQaArtifact,
    lesion: LesionComponentsArtifact,
    summary: DevelopmentDataSummaryArtifact,
) -> None:
    if split.source_manifest_hash != manifest.manifest_hash:
        msg = "split source_manifest_hash must equal manifest hash"
        raise DevelopmentQaReportLinkageError(msg)
    if geometry.manifest_hash != manifest.manifest_hash or geometry.split_hash != split.split_hash:
        msg = "geometry QA artifact must link to the exact manifest and split"
        raise DevelopmentQaReportLinkageError(msg)
    if (
        lesion.manifest_hash != manifest.manifest_hash
        or lesion.split_hash != split.split_hash
        or lesion.geometry_qa_artifact_hash != geometry.qa_artifact_hash
    ):
        msg = "lesion artifact must link to the exact manifest, split, and geometry QA artifact"
        raise DevelopmentQaReportLinkageError(msg)
    if (
        summary.manifest_hash != manifest.manifest_hash
        or summary.split_hash != split.split_hash
        or summary.geometry_qa_artifact_hash != geometry.qa_artifact_hash
        or summary.lesion_artifact_hash != lesion.lesion_artifact_hash
    ):
        msg = "development summary must link to the exact upstream artifacts"
        raise DevelopmentQaReportLinkageError(msg)
    _verify_case_sets(manifest, split, geometry, lesion, summary)
    _verify_partition_disjointness(split)


def _verify_case_sets(
    manifest: DatasetManifest,
    split: DevelopmentSplitManifest,
    geometry: GeometryLabelQaArtifact,
    lesion: LesionComponentsArtifact,
    summary: DevelopmentDataSummaryArtifact,
) -> None:
    manifest_key = tuple(
        (case.anonymous_patient_id, case.anonymous_case_id) for case in manifest.cases
    )
    split_key = tuple(
        (assignment.anonymous_patient_id, assignment.anonymous_case_id)
        for assignment in split.assignments
    )
    geometry_key = tuple(
        (case.anonymous_patient_id, case.anonymous_case_id) for case in geometry.case_records
    )
    if split_key != manifest_key or geometry_key != manifest_key:
        msg = "manifest, split, and geometry case identifiers must match exactly"
        raise DevelopmentQaReportLinkageError(msg)
    geometry_partition_key = tuple(
        (case.anonymous_patient_id, case.anonymous_case_id, case.partition)
        for case in geometry.case_records
    )
    lesion_key = tuple(
        (case.anonymous_patient_id, case.anonymous_case_id, case.partition)
        for case in lesion.case_records
    )
    summary_key = tuple(
        (case.anonymous_patient_id, case.anonymous_case_id, case.partition)
        for case in summary.case_records
    )
    split_partition_key = tuple(
        (assignment.anonymous_patient_id, assignment.anonymous_case_id, assignment.partition)
        for assignment in split.assignments
    )
    if (
        geometry_partition_key != split_partition_key
        or lesion_key != geometry_partition_key
        or summary_key != geometry_partition_key
    ):
        msg = "case ordering, identifiers, and partitions must match across all artifacts"
        raise DevelopmentQaReportLinkageError(msg)


def _verify_partition_disjointness(split: DevelopmentSplitManifest) -> None:
    partitions = ("train", "validation", "internal_test")
    patients_by_partition = {partition: set[str]() for partition in partitions}
    cases_by_partition = {partition: set[str]() for partition in partitions}
    for assignment in split.assignments:
        patients_by_partition[assignment.partition].add(assignment.anonymous_patient_id)
        cases_by_partition[assignment.partition].add(assignment.anonymous_case_id)
    partition_pairs = (
        ("train", "validation"),
        ("train", "internal_test"),
        ("validation", "internal_test"),
    )
    for left, right in partition_pairs:
        if patients_by_partition[left] & patients_by_partition[right]:
            msg = "patient overlap detected across split partitions"
            raise DevelopmentQaReportLinkageError(msg)
        if cases_by_partition[left] & cases_by_partition[right]:
            msg = "case overlap detected across split partitions"
            raise DevelopmentQaReportLinkageError(msg)


def _assemble_case_record(
    *,
    geometry_case: GeometryLabelQaCaseRecord,
    lesion_case: LesionComponentCaseRecord,
    summary_case: CtHistogramCaseRecord,
    partition: str,
    histogram_bin_edges: tuple[float, ...],
) -> CaseQaRecord:
    if geometry_case.qa_passed:
        if not lesion_case.analysis_performed or not summary_case.analysis_performed:
            msg = "passing geometry QA requires performed lesion and CT summary analyses"
            raise DevelopmentQaReportCaseAssemblyError(msg)
        if lesion_case.failure_reasons or summary_case.failure_reasons:
            msg = "performed upstream analyses must not contain failure reasons"
            raise DevelopmentQaReportCaseAssemblyError(msg)
        qa_passed = True
        failure_reasons: tuple[str, ...] = ()
    else:
        if lesion_case.analysis_performed or summary_case.analysis_performed:
            msg = "failed geometry QA cases must remain upstream-skipped downstream"
            raise DevelopmentQaReportCaseAssemblyError(msg)
        qa_passed = False
        failure_reasons = geometry_case.failure_reasons

    return CaseQaRecord(
        anonymous_patient_id=geometry_case.anonymous_patient_id,
        anonymous_case_id=geometry_case.anonymous_case_id,
        partition=partition,
        dimensionality=geometry_case.dimensionality,
        shape=geometry_case.image_shape,
        image_dtype=geometry_case.image_dtype,
        label_dtype=geometry_case.label_dtype,
        affine=geometry_case.image_affine,
        orientation=geometry_case.image_orientation,
        spacing=geometry_case.image_spacing,
        image_finite=geometry_case.image_finite,
        label_finite=geometry_case.label_finite,
        allowed_label_values=geometry_case.allowed_label_values,
        image_label_shape_match=geometry_case.image_label_shape_match,
        image_label_affine_match=geometry_case.image_label_affine_match,
        intensity_min=summary_case.intensity_min,
        intensity_max=summary_case.intensity_max,
        intensity_mean=summary_case.intensity_mean,
        intensity_std=summary_case.intensity_std,
        histogram_bin_edges=histogram_bin_edges,
        histogram_counts=summary_case.histogram_counts,
        tumor_voxel_count=geometry_case.tumor_voxel_count,
        tumor_physical_volume_mm3=lesion_case.tumor_physical_volume_mm3,
        lesion_count=lesion_case.lesion_count,
        lesions=lesion_case.lesions,
        empty_tumor=geometry_case.empty_tumor,
        qa_passed=qa_passed,
        failure_reasons=failure_reasons,
    )


def _verify_final_case_counts(
    case_records: tuple[CaseQaRecord, ...],
    geometry: GeometryLabelQaArtifact,
    lesion: LesionComponentsArtifact,
    summary: DevelopmentDataSummaryArtifact,
) -> None:
    passed = sum(1 for record in case_records if record.qa_passed)
    failed = sum(1 for record in case_records if not record.qa_passed)
    if passed != geometry.passed_case_count or failed != geometry.failed_case_count:
        msg = "final QA pass/fail counts must match geometry QA counts"
        raise DevelopmentQaReportCaseAssemblyError(msg)
    if passed != lesion.analyzed_case_count or failed != lesion.skipped_case_count:
        msg = "final QA counts must match lesion analyzed/skipped counts"
        raise DevelopmentQaReportCaseAssemblyError(msg)
    if passed != summary.analyzed_case_count or failed != summary.skipped_case_count:
        msg = "final QA counts must match summary analyzed/skipped counts"
        raise DevelopmentQaReportCaseAssemblyError(msg)


def _verify_final_aggregates(
    artifact: DevelopmentQaArtifact,
    summary: DevelopmentDataSummaryArtifact,
) -> None:
    copied_fields = (
        "histogram_bin_edges",
        "histogram_bin_count",
        "aggregate_image_voxel_count",
        "aggregate_intensity_min",
        "aggregate_intensity_max",
        "aggregate_intensity_mean",
        "aggregate_intensity_std",
        "aggregate_histogram_counts",
        "aggregate_below_histogram_range_count",
        "aggregate_above_histogram_range_count",
        "total_tumor_voxel_count",
        "total_tumor_physical_volume_mm3",
        "total_lesion_count",
        "lesion_volume_min_mm3",
        "lesion_volume_max_mm3",
        "lesion_volume_mean_mm3",
        "lesion_volume_median_mm3",
    )
    for field_name in copied_fields:
        if getattr(artifact, field_name) != getattr(summary, field_name):
            msg = "final aggregate fields must be copied exactly from development summary"
            raise DevelopmentQaReportAggregateError(msg)


def _validate_input_json_path(path: Path, role: str) -> Path:
    if str(path) == "":
        msg = f"input {role} path must not be empty"
        raise InvalidDevelopmentQaReportInputError(msg)
    if not path.is_absolute():
        msg = f"input {role} path must be explicit and absolute"
        raise InvalidDevelopmentQaReportInputError(msg)
    try:
        resolved_path = path.resolve(strict=True)
    except FileNotFoundError as exc:
        msg = f"input {role} JSON does not exist"
        raise InvalidDevelopmentQaReportInputError(msg) from exc
    except OSError as exc:
        msg = f"input {role} JSON cannot be resolved"
        raise InvalidDevelopmentQaReportInputError(msg) from exc
    if not resolved_path.is_file():
        msg = f"input {role} JSON must be a regular file"
        raise InvalidDevelopmentQaReportInputError(msg)
    return resolved_path


def _validate_report_output_path(output_path: Path, *, input_paths: tuple[Path, ...]) -> Path:
    if str(output_path) == "":
        msg = "development-QA report output path must not be empty"
        raise UnsafeDevelopmentQaReportOutputPathError(msg)
    if not output_path.is_absolute():
        msg = "development-QA report output path must be explicit and absolute"
        raise UnsafeDevelopmentQaReportOutputPathError(msg)
    if output_path.suffix != ".json":
        msg = "development-QA report output path must use a .json suffix"
        raise UnsafeDevelopmentQaReportOutputPathError(msg)
    if any(part == ".." for part in output_path.parts):
        msg = "development-QA report output path must not contain parent traversal"
        raise UnsafeDevelopmentQaReportOutputPathError(msg)
    resolved_non_strict = output_path.resolve(strict=False)
    if any(resolved_non_strict == input_path for input_path in input_paths):
        msg = "development-QA report output path must not equal an input JSON path"
        raise UnsafeDevelopmentQaReportOutputPathError(msg)
    if output_path.exists() and output_path.is_file():
        msg = "development-QA report output path already exists"
        raise ExistingDevelopmentQaReportOutputError(msg)

    safe_output_path = _resolve_path_with_existing_ancestor(output_path)
    if any(safe_output_path == input_path for input_path in input_paths):
        msg = "development-QA report output path must not equal an input JSON path"
        raise UnsafeDevelopmentQaReportOutputPathError(msg)
    if output_path.exists():
        msg = "development-QA report output path already exists"
        raise ExistingDevelopmentQaReportOutputError(msg)
    return safe_output_path


def _resolve_path_with_existing_ancestor(path: Path) -> Path:
    missing_parts: list[str] = []
    current = path
    while not current.exists():
        missing_parts.append(current.name)
        parent = current.parent
        if parent == current:
            msg = "development-QA report output path has no existing parent"
            raise UnsafeDevelopmentQaReportOutputPathError(msg)
        current = parent
    if current.is_file():
        msg = "development-QA report output parent must not be a regular file"
        raise UnsafeDevelopmentQaReportOutputPathError(msg)
    if not current.is_dir():
        msg = "development-QA report output parent must resolve beneath a directory"
        raise UnsafeDevelopmentQaReportOutputPathError(msg)
    try:
        resolved = current.resolve(strict=True)
    except OSError as exc:
        msg = "development-QA report output path cannot be resolved"
        raise UnsafeDevelopmentQaReportOutputPathError(msg) from exc
    for part in reversed(missing_parts):
        resolved = resolved / part
    return resolved


def _publish_report_json(artifact: DevelopmentQaArtifact, output_path: Path) -> None:
    try:
        text = phase2_artifact_to_json(artifact)
    except Phase2ArtifactError as exc:
        msg = "failed to serialize final development-QA report"
        raise DevelopmentQaReportPublicationError(msg) from exc
    temp_path = output_path.with_name(f".{output_path.name}.tmp")
    created_temp = False
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if temp_path.exists():
            msg = "temporary development-QA report output already exists"
            raise ExistingDevelopmentQaReportOutputError(msg)
        temp_path.write_text(text, encoding="utf-8", newline="\n")
        created_temp = True
        if output_path.exists():
            msg = "development-QA report output path already exists"
            raise ExistingDevelopmentQaReportOutputError(msg)
        os.link(temp_path, output_path)
        temp_path.unlink()
        created_temp = False
    except ExistingDevelopmentQaReportOutputError:
        raise
    except OSError as exc:
        msg = "failed to publish final development-QA report JSON"
        raise DevelopmentQaReportPublicationError(msg) from exc
    finally:
        if created_temp and temp_path.exists():
            temp_path.unlink()


def _require_explicit_metadata(value: object, field_name: str) -> None:
    if not isinstance(value, str) or value == "" or value != value.strip():
        msg = f"{field_name} must be supplied explicitly as a nonempty string"
        raise InvalidDevelopmentQaReportInputError(msg)
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        msg = f"{field_name} must not contain control characters"
        raise InvalidDevelopmentQaReportInputError(msg)


__all__ = [
    "DEVELOPMENT_QA_REPORT_CONFIG_PAYLOAD_TYPE",
    "DEVELOPMENT_QA_REPORT_CONTRACT_VERSION",
    "DevelopmentQaReportAggregateError",
    "DevelopmentQaReportCaseAssemblyError",
    "DevelopmentQaReportConfig",
    "DevelopmentQaReportError",
    "DevelopmentQaReportHashMismatchError",
    "DevelopmentQaReportLinkageError",
    "DevelopmentQaReportPublicationError",
    "ExistingDevelopmentQaReportOutputError",
    "InvalidDevelopmentQaReportConfigError",
    "InvalidDevelopmentQaReportInputError",
    "UnsafeDevelopmentQaReportOutputPathError",
    "assemble_development_qa_report",
    "development_qa_report_config_hash_payload",
    "hash_development_qa_report_config",
]
