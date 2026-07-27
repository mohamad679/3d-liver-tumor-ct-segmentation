"""Read-only Phase 2 fixed CT histograms and development dataset summaries."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final, cast

import nibabel as nib
import numpy as np
import numpy.typing as npt
from nibabel.filebasedimages import ImageFileError

import protoem_ct.artifacts.hashing as artifact_hashing
from protoem_ct.artifacts import (
    DEVELOPMENT_DATA_SUMMARY_STAGE,
    DEVELOPMENT_DATA_SUMMARY_UPSTREAM_FAILURE_REASON,
    PHASE2_DEVELOPMENT_COHORT_ROLE,
    PHASE2_SCHEMA_VERSION,
    CtHistogramCaseRecord,
    DatasetManifest,
    DevelopmentDataSummaryArtifact,
    GeometryLabelQaArtifact,
    GeometryLabelQaCaseRecord,
    LesionComponentCaseRecord,
    LesionComponentsArtifact,
    Phase2ArtifactError,
    hash_dataset_manifest,
    hash_development_data_summary,
    hash_geometry_label_qa,
    hash_lesion_components,
    phase2_artifact_from_json,
    phase2_artifact_to_json,
    sha256_json,
)
from protoem_ct.data._phase2_publication import (
    Phase2PublicationExistingOutputError,
    Phase2PublicationIOError,
    publish_text_no_overwrite,
)
from protoem_ct.data.phase2_paths import (
    Phase2PathError,
    require_unique_resolved_paths,
    resolve_regular_file_beneath_root,
    validate_explicit_dataset_root,
)

DEVELOPMENT_SUMMARY_CONFIG_PAYLOAD_TYPE: Final[str] = "phase2-development-summary-config-v1"
MAX_HISTOGRAM_BIN_COUNT: Final[int] = 100_000
_ZERO_SHA256: Final[str] = "0" * 64

Array = npt.NDArray[np.generic]
FloatArray = npt.NDArray[np.float64]


class DevelopmentSummaryError(ValueError):
    """Base exception for Phase 2 development-summary failures."""


class InvalidDevelopmentSummaryConfigError(DevelopmentSummaryError):
    """Raised when explicit development-summary configuration is invalid."""


class InvalidDevelopmentSummaryInputError(DevelopmentSummaryError):
    """Raised when manifest, QA artifacts, or explicit metadata inputs are invalid."""


class DevelopmentSummaryHashMismatchError(InvalidDevelopmentSummaryInputError):
    """Raised when a stored input artifact hash does not match recomputed content."""


class DevelopmentSummaryLinkageError(InvalidDevelopmentSummaryInputError):
    """Raised when linked Phase 2 input artifacts are inconsistent."""


class DevelopmentSummarySourceIntegrityError(DevelopmentSummaryError):
    """Raised when image files are missing, aliased, changed, or unsafe."""


class DevelopmentSummaryNiftiReadError(DevelopmentSummaryError):
    """Raised when an integrity-verified image cannot be read as NIfTI."""


class UnsafeDevelopmentSummaryOutputPathError(DevelopmentSummaryError):
    """Raised when the requested summary output path violates containment policy."""


class ExistingDevelopmentSummaryOutputError(DevelopmentSummaryError):
    """Raised when summary publication would overwrite an existing output."""


class DevelopmentSummaryPublicationError(DevelopmentSummaryError):
    """Raised when validated summary JSON cannot be published atomically."""


@dataclass(frozen=True, slots=True)
class DevelopmentSummaryConfig:
    """Explicit fixed-bin CT histogram settings.

    Bins follow NumPy histogram semantics: bins are left-closed and right-open, except the final
    bin includes its right edge. Values below ``histogram_min`` and above ``histogram_max`` are
    counted separately and are never clipped into edge bins. The range and bin count are explicit
    inputs; this module never fits them from data.
    """

    histogram_min: float
    histogram_max: float
    histogram_bin_count: int

    def __post_init__(self) -> None:
        for field_name in ("histogram_min", "histogram_max"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, float) or not math.isfinite(value):
                msg = f"{field_name} must be a finite float"
                raise InvalidDevelopmentSummaryConfigError(msg)
        if self.histogram_min >= self.histogram_max:
            msg = "histogram_min must be strictly less than histogram_max"
            raise InvalidDevelopmentSummaryConfigError(msg)
        if (
            isinstance(self.histogram_bin_count, bool)
            or not isinstance(self.histogram_bin_count, int)
            or self.histogram_bin_count <= 0
            or self.histogram_bin_count > MAX_HISTOGRAM_BIN_COUNT
        ):
            msg = (
                "histogram_bin_count must be a positive integer no larger than "
                f"{MAX_HISTOGRAM_BIN_COUNT}"
            )
            raise InvalidDevelopmentSummaryConfigError(msg)

    @property
    def histogram_bin_edges(self) -> tuple[float, ...]:
        """Return deterministic float64 histogram bin edges."""
        edges = np.linspace(
            np.float64(self.histogram_min),
            np.float64(self.histogram_max),
            self.histogram_bin_count + 1,
            dtype=np.float64,
        )
        return tuple(float(value) for value in edges.tolist())


def development_summary_config_hash_payload(config: DevelopmentSummaryConfig) -> dict[str, object]:
    """Return the canonical payload for a development-summary configuration hash."""
    return {
        "histogram_bin_count": config.histogram_bin_count,
        "histogram_max": config.histogram_max,
        "histogram_min": config.histogram_min,
        "payload_type": DEVELOPMENT_SUMMARY_CONFIG_PAYLOAD_TYPE,
        "schema_version": PHASE2_SCHEMA_VERSION,
    }


def hash_development_summary_config(config: DevelopmentSummaryConfig) -> str:
    """Return the deterministic SHA-256 hash of explicit summary settings."""
    return sha256_json(development_summary_config_hash_payload(config))


@dataclass(frozen=True, slots=True)
class _StreamingStats:
    count: int
    mean: float
    m2: float
    minimum: float | None
    maximum: float | None


def run_development_data_summary(
    manifest_path: Path,
    geometry_qa_artifact_path: Path,
    lesion_artifact_path: Path,
    *,
    dataset_root: Path,
    output_path: Path,
    config: DevelopmentSummaryConfig,
    git_commit: str,
    created_at_utc: str,
) -> DevelopmentDataSummaryArtifact:
    """Run read-only fixed CT histograms and publish a development summary artifact."""
    source_manifest_path = _validate_input_json_path(manifest_path, "manifest")
    source_geometry_path = _validate_input_json_path(
        geometry_qa_artifact_path,
        "geometry QA artifact",
    )
    source_lesion_path = _validate_input_json_path(lesion_artifact_path, "lesion artifact")
    canonical_root = validate_explicit_dataset_root(dataset_root)
    safe_output_path = _validate_summary_output_path(
        output_path,
        dataset_root=canonical_root,
        input_paths=(source_manifest_path, source_geometry_path, source_lesion_path),
    )
    _require_explicit_metadata(git_commit, "git_commit")
    _require_explicit_metadata(created_at_utc, "created_at_utc")

    manifest = _load_and_verify_dataset_manifest(source_manifest_path.read_text(encoding="utf-8"))
    geometry = _load_and_verify_geometry_qa(source_geometry_path.read_text(encoding="utf-8"))
    lesion = _load_and_verify_lesion_components(source_lesion_path.read_text(encoding="utf-8"))
    _verify_input_linkage(manifest, geometry, lesion)

    verified_images = _resolve_and_verify_passing_images(canonical_root, manifest, geometry)
    histogram_edges = config.histogram_bin_edges
    case_records = tuple(
        sorted(
            (
                _summary_case_record(
                    geometry_case=geometry_case,
                    image_path=verified_images.get(geometry_case.anonymous_case_id),
                    histogram_edges=histogram_edges,
                )
                for geometry_case in geometry.case_records
            ),
            key=lambda record: (record.anonymous_patient_id, record.anonymous_case_id),
        )
    )
    aggregate_stats = _aggregate_case_statistics(case_records, config.histogram_bin_count)
    lesion_totals = _aggregate_lesion_summaries(lesion.case_records)

    artifact_without_hash = DevelopmentDataSummaryArtifact(
        schema_version=PHASE2_SCHEMA_VERSION,
        stage=DEVELOPMENT_DATA_SUMMARY_STAGE,
        created_at_utc=created_at_utc,
        git_commit=git_commit,
        config_hash=hash_development_summary_config(config),
        manifest_hash=manifest.manifest_hash,
        split_hash=geometry.split_hash,
        geometry_qa_artifact_hash=geometry.qa_artifact_hash,
        lesion_artifact_hash=lesion.lesion_artifact_hash,
        histogram_bin_edges=histogram_edges,
        histogram_bin_count=config.histogram_bin_count,
        case_count=len(case_records),
        analyzed_case_count=sum(1 for record in case_records if record.analysis_performed),
        skipped_case_count=sum(1 for record in case_records if not record.analysis_performed),
        case_records=case_records,
        aggregate_image_voxel_count=aggregate_stats["count"],
        aggregate_intensity_min=aggregate_stats["minimum"],
        aggregate_intensity_max=aggregate_stats["maximum"],
        aggregate_intensity_mean=aggregate_stats["mean"],
        aggregate_intensity_std=aggregate_stats["std"],
        aggregate_histogram_counts=cast(tuple[int, ...], aggregate_stats["histogram"]),
        aggregate_below_histogram_range_count=aggregate_stats["below"],
        aggregate_above_histogram_range_count=aggregate_stats["above"],
        total_tumor_voxel_count=lesion_totals["tumor_voxels"],
        total_tumor_physical_volume_mm3=lesion_totals["tumor_volume"],
        total_lesion_count=lesion_totals["lesion_count"],
        lesion_volume_min_mm3=lesion_totals["lesion_min"],
        lesion_volume_max_mm3=lesion_totals["lesion_max"],
        lesion_volume_mean_mm3=lesion_totals["lesion_mean"],
        lesion_volume_median_mm3=lesion_totals["lesion_median"],
        summary_artifact_hash=_ZERO_SHA256,
    )
    summary_hash = hash_development_data_summary(artifact_without_hash)
    artifact = replace(artifact_without_hash, summary_artifact_hash=summary_hash)
    if hash_development_data_summary(artifact) != summary_hash:
        msg = "development-data-summary artifact hash verification failed"
        raise DevelopmentSummaryPublicationError(msg)

    _verify_lesion_totals_against_artifact(artifact, lesion)
    _publish_summary_json(artifact, safe_output_path)
    return artifact


def _load_and_verify_dataset_manifest(manifest_text: str) -> DatasetManifest:
    try:
        artifact = phase2_artifact_from_json(manifest_text)
    except Phase2ArtifactError as exc:
        msg = "input manifest is not a valid Phase 2 dataset manifest"
        raise InvalidDevelopmentSummaryInputError(msg) from exc
    if not isinstance(artifact, DatasetManifest):
        msg = "input artifact must be a DatasetManifest"
        raise InvalidDevelopmentSummaryInputError(msg)
    if artifact.cohort_role != PHASE2_DEVELOPMENT_COHORT_ROLE:
        msg = "input dataset manifest must use the development cohort role"
        raise InvalidDevelopmentSummaryInputError(msg)
    if hash_dataset_manifest(artifact) != artifact.manifest_hash:
        msg = "input dataset manifest hash does not match its content"
        raise DevelopmentSummaryHashMismatchError(msg)
    return artifact


def _load_and_verify_geometry_qa(geometry_text: str) -> GeometryLabelQaArtifact:
    try:
        artifact = phase2_artifact_from_json(geometry_text)
    except Phase2ArtifactError as exc:
        msg = "input geometry QA artifact is not a valid Phase 2 artifact"
        raise InvalidDevelopmentSummaryInputError(msg) from exc
    if not isinstance(artifact, GeometryLabelQaArtifact):
        msg = "input artifact must be a GeometryLabelQaArtifact"
        raise InvalidDevelopmentSummaryInputError(msg)
    if hash_geometry_label_qa(artifact) != artifact.qa_artifact_hash:
        msg = "input geometry QA artifact hash does not match its content"
        raise DevelopmentSummaryHashMismatchError(msg)
    return artifact


def _load_and_verify_lesion_components(lesion_text: str) -> LesionComponentsArtifact:
    try:
        artifact = phase2_artifact_from_json(lesion_text)
    except Phase2ArtifactError as exc:
        msg = "input lesion artifact is not a valid Phase 2 artifact"
        raise InvalidDevelopmentSummaryInputError(msg) from exc
    if not isinstance(artifact, LesionComponentsArtifact):
        msg = "input artifact must be a LesionComponentsArtifact"
        raise InvalidDevelopmentSummaryInputError(msg)
    if hash_lesion_components(artifact) != artifact.lesion_artifact_hash:
        msg = "input lesion artifact hash does not match its content"
        raise DevelopmentSummaryHashMismatchError(msg)
    return artifact


def _verify_input_linkage(
    manifest: DatasetManifest,
    geometry: GeometryLabelQaArtifact,
    lesion: LesionComponentsArtifact,
) -> None:
    if geometry.manifest_hash != manifest.manifest_hash:
        msg = "geometry QA manifest_hash must equal the dataset manifest hash"
        raise DevelopmentSummaryLinkageError(msg)
    if lesion.manifest_hash != manifest.manifest_hash:
        msg = "lesion artifact manifest_hash must equal the dataset manifest hash"
        raise DevelopmentSummaryLinkageError(msg)
    if lesion.split_hash != geometry.split_hash:
        msg = "lesion artifact split_hash must equal the geometry QA split_hash"
        raise DevelopmentSummaryLinkageError(msg)
    if lesion.geometry_qa_artifact_hash != geometry.qa_artifact_hash:
        msg = "lesion artifact must link to the exact geometry QA artifact hash"
        raise DevelopmentSummaryLinkageError(msg)
    manifest_key = tuple(
        (case.anonymous_patient_id, case.anonymous_case_id) for case in manifest.cases
    )
    geometry_key = tuple(
        (case.anonymous_patient_id, case.anonymous_case_id, case.partition)
        for case in geometry.case_records
    )
    lesion_key = tuple(
        (case.anonymous_patient_id, case.anonymous_case_id, case.partition)
        for case in lesion.case_records
    )
    if tuple((patient, case) for patient, case, _partition in geometry_key) != manifest_key:
        msg = "geometry QA case ordering and identifiers must match manifest cases exactly"
        raise DevelopmentSummaryLinkageError(msg)
    if lesion_key != geometry_key:
        msg = "lesion artifact case ordering, identifiers, and partitions must match geometry QA"
        raise DevelopmentSummaryLinkageError(msg)
    for geometry_case, lesion_case in zip(
        geometry.case_records,
        lesion.case_records,
        strict=True,
    ):
        if geometry_case.qa_passed != lesion_case.analysis_performed:
            msg = "lesion analyzed/skipped status must match geometry QA pass status"
            raise DevelopmentSummaryLinkageError(msg)


def _resolve_and_verify_passing_images(
    dataset_root: Path,
    manifest: DatasetManifest,
    geometry: GeometryLabelQaArtifact,
) -> dict[str, Path]:
    passing_case_ids = {
        record.anonymous_case_id for record in geometry.case_records if record.qa_passed
    }
    case_by_id = {case.anonymous_case_id: case for case in manifest.cases}
    resolved: dict[str, Path] = {}
    named_paths: list[tuple[str, Path]] = []
    try:
        for case_id in sorted(passing_case_ids):
            case = case_by_id[case_id]
            image_path = resolve_regular_file_beneath_root(dataset_root, case.relative_image_path)
            resolved[case_id] = image_path
            named_paths.append((f"{case_id}:image", image_path))
        require_unique_resolved_paths(tuple(named_paths))
    except Phase2PathError as exc:
        msg = "source image path violates the Phase 2 containment contract"
        raise DevelopmentSummarySourceIntegrityError(msg) from exc

    try:
        for case_id in sorted(passing_case_ids):
            case = case_by_id[case_id]
            if artifact_hashing.sha256_file(resolved[case_id]) != case.image_sha256:
                msg = "source image file hash does not match the manifest"
                raise DevelopmentSummarySourceIntegrityError(msg)
    except artifact_hashing.HashingError as exc:
        msg = "failed to hash an integrity-checked image file"
        raise DevelopmentSummarySourceIntegrityError(msg) from exc
    return resolved


def _summary_case_record(
    *,
    geometry_case: GeometryLabelQaCaseRecord,
    image_path: Path | None,
    histogram_edges: tuple[float, ...],
) -> CtHistogramCaseRecord:
    if not geometry_case.qa_passed:
        return CtHistogramCaseRecord(
            anonymous_patient_id=geometry_case.anonymous_patient_id,
            anonymous_case_id=geometry_case.anonymous_case_id,
            partition=geometry_case.partition,
            analysis_performed=False,
            image_voxel_count=None,
            intensity_min=None,
            intensity_max=None,
            intensity_mean=None,
            intensity_std=None,
            histogram_counts=(),
            below_histogram_range_count=None,
            above_histogram_range_count=None,
            qa_passed=False,
            failure_reasons=(DEVELOPMENT_DATA_SUMMARY_UPSTREAM_FAILURE_REASON,),
        )
    if image_path is None:
        msg = "passing geometry QA case is missing a verified image path"
        raise DevelopmentSummarySourceIntegrityError(msg)

    image = _load_nifti(image_path)
    _verify_image_metadata(image, geometry_case)
    stats = _stream_image_statistics(image, histogram_edges)
    return CtHistogramCaseRecord(
        anonymous_patient_id=geometry_case.anonymous_patient_id,
        anonymous_case_id=geometry_case.anonymous_case_id,
        partition=geometry_case.partition,
        analysis_performed=True,
        image_voxel_count=stats["count"],
        intensity_min=stats["minimum"],
        intensity_max=stats["maximum"],
        intensity_mean=stats["mean"],
        intensity_std=stats["std"],
        histogram_counts=cast(tuple[int, ...], stats["histogram"]),
        below_histogram_range_count=stats["below"],
        above_histogram_range_count=stats["above"],
        qa_passed=True,
        failure_reasons=(),
    )


def _load_nifti(path: Path) -> Any:
    try:
        return nib.load(str(path), mmap=False)
    except (ImageFileError, OSError, ValueError) as exc:
        msg = "image source is not a readable NIfTI file"
        raise DevelopmentSummaryNiftiReadError(msg) from exc


def _verify_image_metadata(image: Any, geometry_case: GeometryLabelQaCaseRecord) -> None:
    shape = tuple(int(dimension) for dimension in image.shape)
    if len(shape) != 3 or shape != geometry_case.image_shape:
        msg = "image shape does not match geometry QA artifact"
        raise DevelopmentSummarySourceIntegrityError(msg)
    if str(image.get_data_dtype()) != geometry_case.image_dtype:
        msg = "image dtype does not match geometry QA artifact"
        raise DevelopmentSummarySourceIntegrityError(msg)
    affine = _affine_tuple(image.affine, "image_affine")
    spacing = _spacing_tuple(affine, "image_spacing")
    orientation = _orientation_tuple(affine, "image_orientation")
    if affine != geometry_case.image_affine:
        msg = "image affine does not match geometry QA artifact"
        raise DevelopmentSummarySourceIntegrityError(msg)
    if spacing != geometry_case.image_spacing:
        msg = "image spacing does not match geometry QA artifact"
        raise DevelopmentSummarySourceIntegrityError(msg)
    if orientation != geometry_case.image_orientation:
        msg = "image orientation does not match geometry QA artifact"
        raise DevelopmentSummarySourceIntegrityError(msg)
    if not geometry_case.image_finite:
        msg = "geometry QA did not pass finite-image validation"
        raise DevelopmentSummarySourceIntegrityError(msg)


def _stream_image_statistics(image: Any, edges: tuple[float, ...]) -> dict[str, Any]:
    histogram_edges = np.asarray(edges, dtype=np.float64)
    histogram = np.zeros(len(edges) - 1, dtype=np.int64)
    below = 0
    above = 0
    stats = _StreamingStats(count=0, mean=0.0, m2=0.0, minimum=None, maximum=None)
    try:
        volume = np.asanyarray(image.dataobj)
        if volume.shape != tuple(int(dimension) for dimension in image.shape):
            msg = "image NIfTI data shape does not match header shape"
            raise DevelopmentSummarySourceIntegrityError(msg)
        for z_index in range(int(volume.shape[2])):
            slice_array = np.asarray(volume[..., z_index], dtype=np.float64)
            if not bool(np.isfinite(slice_array).all()):
                msg = "image finite-value status changed after geometry QA"
                raise DevelopmentSummarySourceIntegrityError(msg)
            flat = cast(FloatArray, slice_array.reshape(-1))
            stats = _combine_stats(stats, _stats_from_batch(flat))
            histogram += np.histogram(flat, bins=histogram_edges)[0].astype(np.int64)
            below += int(np.count_nonzero(flat < histogram_edges[0]))
            above += int(np.count_nonzero(flat > histogram_edges[-1]))
        del volume
    except DevelopmentSummarySourceIntegrityError:
        raise
    except (OSError, ValueError, TypeError, IndexError) as exc:
        msg = "image NIfTI data cannot be loaded"
        raise DevelopmentSummaryNiftiReadError(msg) from exc
    if stats.count <= 0:
        msg = "image contains no voxels"
        raise DevelopmentSummarySourceIntegrityError(msg)
    variance = max(stats.m2 / stats.count, 0.0)
    return {
        "above": above,
        "below": below,
        "count": stats.count,
        "histogram": tuple(int(value) for value in histogram.tolist()),
        "maximum": stats.maximum,
        "mean": stats.mean,
        "minimum": stats.minimum,
        "std": float(math.sqrt(variance)),
    }


def _stats_from_batch(values: FloatArray) -> _StreamingStats:
    count = int(values.size)
    if count == 0:
        return _StreamingStats(count=0, mean=0.0, m2=0.0, minimum=None, maximum=None)
    mean = float(np.mean(values, dtype=np.float64))
    centered = values - mean
    return _StreamingStats(
        count=count,
        mean=mean,
        m2=float(np.sum(centered * centered, dtype=np.float64)),
        minimum=float(np.min(values)),
        maximum=float(np.max(values)),
    )


def _combine_stats(left: _StreamingStats, right: _StreamingStats) -> _StreamingStats:
    if right.count == 0:
        return left
    if left.count == 0:
        return right
    total = left.count + right.count
    delta = right.mean - left.mean
    mean = left.mean + delta * right.count / total
    m2 = left.m2 + right.m2 + delta * delta * left.count * right.count / total
    minimum = min(cast(float, left.minimum), cast(float, right.minimum))
    maximum = max(cast(float, left.maximum), cast(float, right.maximum))
    return _StreamingStats(count=total, mean=mean, m2=m2, minimum=minimum, maximum=maximum)


def _aggregate_case_statistics(
    case_records: tuple[CtHistogramCaseRecord, ...],
    histogram_bin_count: int,
) -> dict[str, Any]:
    histogram = [0] * histogram_bin_count
    below = 0
    above = 0
    stats = _StreamingStats(count=0, mean=0.0, m2=0.0, minimum=None, maximum=None)
    for record in case_records:
        if not record.analysis_performed:
            continue
        for index, count in enumerate(record.histogram_counts):
            histogram[index] += count
        below += cast(int, record.below_histogram_range_count)
        above += cast(int, record.above_histogram_range_count)
        case_stats = _StreamingStats(
            count=cast(int, record.image_voxel_count),
            mean=cast(float, record.intensity_mean),
            m2=(cast(float, record.intensity_std) ** 2) * cast(int, record.image_voxel_count),
            minimum=record.intensity_min,
            maximum=record.intensity_max,
        )
        stats = _combine_stats(stats, case_stats)
    if stats.count == 0:
        return {
            "above": above,
            "below": below,
            "count": 0,
            "histogram": tuple(histogram),
            "maximum": None,
            "mean": None,
            "minimum": None,
            "std": None,
        }
    return {
        "above": above,
        "below": below,
        "count": stats.count,
        "histogram": tuple(histogram),
        "maximum": stats.maximum,
        "mean": stats.mean,
        "minimum": stats.minimum,
        "std": float(math.sqrt(max(stats.m2 / stats.count, 0.0))),
    }


def _aggregate_lesion_summaries(
    case_records: tuple[LesionComponentCaseRecord, ...],
) -> dict[str, Any]:
    tumor_voxels = 0
    tumor_volume = 0.0
    lesion_volumes: list[float] = []
    for record in case_records:
        if not record.analysis_performed:
            continue
        tumor_voxels += cast(int, record.tumor_voxel_count)
        tumor_volume += cast(float, record.tumor_physical_volume_mm3)
        lesion_volumes.extend(lesion.physical_volume_mm3 for lesion in record.lesions)

    lesion_count = len(lesion_volumes)
    if lesion_count == 0:
        return {
            "lesion_count": 0,
            "lesion_max": None,
            "lesion_mean": None,
            "lesion_median": None,
            "lesion_min": None,
            "tumor_volume": float(tumor_volume),
            "tumor_voxels": tumor_voxels,
        }
    ordered_volumes = sorted(lesion_volumes)
    middle = lesion_count // 2
    if lesion_count % 2:
        median = ordered_volumes[middle]
    else:
        median = (ordered_volumes[middle - 1] + ordered_volumes[middle]) / 2.0
    return {
        "lesion_count": lesion_count,
        "lesion_max": float(max(ordered_volumes)),
        "lesion_mean": float(sum(ordered_volumes) / lesion_count),
        "lesion_median": float(median),
        "lesion_min": float(min(ordered_volumes)),
        "tumor_volume": float(tumor_volume),
        "tumor_voxels": tumor_voxels,
    }


def _verify_lesion_totals_against_artifact(
    summary: DevelopmentDataSummaryArtifact,
    lesion: LesionComponentsArtifact,
) -> None:
    totals = _aggregate_lesion_summaries(lesion.case_records)
    if (
        summary.total_tumor_voxel_count != totals["tumor_voxels"]
        or not math.isclose(
            summary.total_tumor_physical_volume_mm3,
            totals["tumor_volume"],
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or summary.total_lesion_count != totals["lesion_count"]
        or summary.lesion_volume_min_mm3 != totals["lesion_min"]
        or summary.lesion_volume_max_mm3 != totals["lesion_max"]
        or summary.lesion_volume_mean_mm3 != totals["lesion_mean"]
        or summary.lesion_volume_median_mm3 != totals["lesion_median"]
    ):
        msg = "development summary lesion totals do not match lesion artifact records"
        raise DevelopmentSummaryLinkageError(msg)


def _affine_tuple(value: object, field_name: str) -> tuple[tuple[float, float, float, float], ...]:
    array = np.asarray(value, dtype=float)
    if array.shape != (4, 4) or not bool(np.isfinite(array).all()):
        msg = f"{field_name} must be a finite 4x4 affine"
        raise DevelopmentSummaryNiftiReadError(msg)
    rows = tuple(tuple(float(item) for item in row) for row in array.tolist())
    return cast(tuple[tuple[float, float, float, float], ...], rows)


def _spacing_tuple(
    affine: tuple[tuple[float, float, float, float], ...],
    field_name: str,
) -> tuple[float, float, float]:
    matrix = np.asarray(affine, dtype=float)[:3, :3]
    spacing = np.sqrt(np.sum(matrix * matrix, axis=0))
    if spacing.shape != (3,) or not bool(np.isfinite(spacing).all()) or bool(np.any(spacing <= 0)):
        msg = f"{field_name} must contain three positive finite values"
        raise DevelopmentSummaryNiftiReadError(msg)
    return cast(tuple[float, float, float], tuple(float(value) for value in spacing.tolist()))


def _orientation_tuple(
    affine: tuple[tuple[float, float, float, float], ...],
    field_name: str,
) -> tuple[str, str, str]:
    axis_codes = nib.aff2axcodes(np.asarray(affine, dtype=float))  # type: ignore[no-untyped-call]
    if len(axis_codes) != 3 or any(code is None for code in axis_codes):
        msg = f"{field_name} could not be derived from affine"
        raise DevelopmentSummaryNiftiReadError(msg)
    return cast(tuple[str, str, str], tuple(str(code) for code in axis_codes))


def _validate_input_json_path(path: Path, role: str) -> Path:
    if str(path) == "":
        msg = f"input {role} path must not be empty"
        raise InvalidDevelopmentSummaryInputError(msg)
    if not path.is_absolute():
        msg = f"input {role} path must be explicit and absolute"
        raise InvalidDevelopmentSummaryInputError(msg)
    try:
        resolved_path = path.resolve(strict=True)
    except FileNotFoundError as exc:
        msg = f"input {role} JSON does not exist"
        raise InvalidDevelopmentSummaryInputError(msg) from exc
    except OSError as exc:
        msg = f"input {role} JSON cannot be resolved"
        raise InvalidDevelopmentSummaryInputError(msg) from exc
    if not resolved_path.is_file():
        msg = f"input {role} JSON must be a regular file"
        raise InvalidDevelopmentSummaryInputError(msg)
    return resolved_path


def _validate_summary_output_path(
    output_path: Path,
    *,
    dataset_root: Path,
    input_paths: tuple[Path, Path, Path],
) -> Path:
    if str(output_path) == "":
        msg = "development-summary output path must not be empty"
        raise UnsafeDevelopmentSummaryOutputPathError(msg)
    if not output_path.is_absolute():
        msg = "development-summary output path must be explicit and absolute"
        raise UnsafeDevelopmentSummaryOutputPathError(msg)
    if any(part == ".." for part in output_path.parts):
        msg = "development-summary output path must not contain parent traversal"
        raise UnsafeDevelopmentSummaryOutputPathError(msg)
    resolved_non_strict = output_path.resolve(strict=False)
    if any(resolved_non_strict == input_path for input_path in input_paths):
        msg = "development-summary output path must not equal an input JSON path"
        raise UnsafeDevelopmentSummaryOutputPathError(msg)
    if output_path.exists() and output_path.is_file():
        msg = "development-summary output path already exists"
        raise ExistingDevelopmentSummaryOutputError(msg)

    safe_output_path = _resolve_path_with_existing_ancestor(output_path)
    if _paths_are_equal_or_nested(safe_output_path, dataset_root):
        msg = "development-summary output path must not equal or be inside the dataset root"
        raise UnsafeDevelopmentSummaryOutputPathError(msg)
    if _paths_are_equal_or_nested(dataset_root, safe_output_path):
        msg = "development-summary output path must not contain the dataset root"
        raise UnsafeDevelopmentSummaryOutputPathError(msg)
    if any(safe_output_path == input_path for input_path in input_paths):
        msg = "development-summary output path must not equal an input JSON path"
        raise UnsafeDevelopmentSummaryOutputPathError(msg)
    if output_path.exists():
        msg = "development-summary output path already exists"
        raise ExistingDevelopmentSummaryOutputError(msg)
    return safe_output_path


def _resolve_path_with_existing_ancestor(path: Path) -> Path:
    missing_parts: list[str] = []
    current = path
    while not current.exists():
        missing_parts.append(current.name)
        parent = current.parent
        if parent == current:
            msg = "development-summary output path has no existing parent"
            raise UnsafeDevelopmentSummaryOutputPathError(msg)
        current = parent
    if current.is_file():
        msg = "development-summary output parent must not be a regular file"
        raise UnsafeDevelopmentSummaryOutputPathError(msg)
    if not current.is_dir():
        msg = "development-summary output parent must resolve beneath a directory"
        raise UnsafeDevelopmentSummaryOutputPathError(msg)
    try:
        resolved = current.resolve(strict=True)
    except OSError as exc:
        msg = "development-summary output path cannot be resolved"
        raise UnsafeDevelopmentSummaryOutputPathError(msg) from exc
    for part in reversed(missing_parts):
        resolved = resolved / part
    return resolved


def _publish_summary_json(artifact: DevelopmentDataSummaryArtifact, output_path: Path) -> None:
    try:
        text = phase2_artifact_to_json(artifact)
    except Phase2ArtifactError as exc:
        msg = "failed to serialize development-summary artifact"
        raise DevelopmentSummaryPublicationError(msg) from exc
    try:
        publish_text_no_overwrite(
            text=text,
            output_path=output_path,
            temporary_exists_message="temporary development-summary output already exists",
            final_exists_message="development-summary output path already exists",
        )
    except Phase2PublicationExistingOutputError as exc:
        raise ExistingDevelopmentSummaryOutputError(str(exc)) from exc
    except Phase2PublicationIOError as exc:
        msg = "failed to publish development-summary JSON"
        raise DevelopmentSummaryPublicationError(msg) from exc


def _require_explicit_metadata(value: object, field_name: str) -> None:
    if not isinstance(value, str) or value == "" or value != value.strip():
        msg = f"{field_name} must be supplied explicitly as a nonempty string"
        raise InvalidDevelopmentSummaryInputError(msg)
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        msg = f"{field_name} must not contain control characters"
        raise InvalidDevelopmentSummaryInputError(msg)


def _paths_are_equal_or_nested(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


__all__ = [
    "DEVELOPMENT_SUMMARY_CONFIG_PAYLOAD_TYPE",
    "MAX_HISTOGRAM_BIN_COUNT",
    "DevelopmentSummaryConfig",
    "DevelopmentSummaryError",
    "DevelopmentSummaryHashMismatchError",
    "DevelopmentSummaryLinkageError",
    "DevelopmentSummaryNiftiReadError",
    "DevelopmentSummaryPublicationError",
    "DevelopmentSummarySourceIntegrityError",
    "ExistingDevelopmentSummaryOutputError",
    "InvalidDevelopmentSummaryConfigError",
    "InvalidDevelopmentSummaryInputError",
    "UnsafeDevelopmentSummaryOutputPathError",
    "development_summary_config_hash_payload",
    "hash_development_summary_config",
    "run_development_data_summary",
]
