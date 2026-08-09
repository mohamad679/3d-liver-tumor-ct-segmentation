"""Phase 8 Wave 2 image-only 3D-IRCADb inventory publication."""

from __future__ import annotations

import hashlib
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

from protoem_ct.artifacts.hashing import JsonValue, canonical_json_bytes, sha256_file, sha256_json
from protoem_ct.data._phase2_publication import (
    Phase2PublicationExistingOutputError,
    Phase2PublicationIOError,
    publish_text_no_overwrite,
)
from protoem_ct.data.phase2_paths import (
    InvalidDatasetRootError,
    validate_explicit_external_output_root,
)
from protoem_ct.external.image_qa import (
    Phase8ImageQaReasonCode,
    Phase8ImageQaResult,
    phase8_image_qa_result_to_dict,
    run_patient_dicom_zip_image_qa,
)
from protoem_ct.external.ircadb import (
    IRCADB_PHASE8_ADAPTER_NAME,
    IRCADB_PHASE8_ADAPTER_VERSION,
    Phase8IrcadbCase,
    Phase8IrcadbInventory,
    discover_phase8_ircadb_image_layout,
    normalize_safe_ircadb_zip_member_path,
)
from protoem_ct.external.manifest import (
    PHASE8_EXTERNAL_IMAGE_CASE_SCHEMA_NAME,
    PHASE8_EXTERNAL_IMAGE_CASE_SCHEMA_VERSION,
    Phase8ExternalImageCase,
    Phase8ExternalImageManifest,
    build_phase8_external_image_manifest,
    phase8_external_image_manifest_to_json,
)

PHASE8_WAVE2_LAYOUT_FILENAME: Final[str] = "phase8_wave2_ircadb_layout.json"
PHASE8_WAVE2_QA_FILENAME: Final[str] = "phase8_wave2_image_qa_results.json"
PHASE8_WAVE2_MANIFEST_FILENAME: Final[str] = "phase8_external_image_manifest.json"
PHASE8_WAVE2_SUMMARY_FILENAME: Final[str] = "phase8_wave2_generation_summary.json"


class Phase8Wave2PublicationError(ValueError):
    """Raised when Phase 8 Wave 2 image-only publication fails safely."""


@dataclass(frozen=True, slots=True)
class Phase8Wave2PublicationResult:
    """Anonymous aggregate result for one Wave 2 image-only publication."""

    output_root: Path
    case_count: int
    qa_pass_count: int
    qa_fail_count: int
    anonymous_case_reason_codes: tuple[tuple[str, tuple[str, ...]], ...]
    artifact_hashes: dict[str, str]
    manifest: Phase8ExternalImageManifest


def run_phase8_wave2_image_inventory(
    *,
    dataset_root: Path,
    dataset_archive: Path,
    output_root: Path,
    repository_root: Path,
) -> Phase8Wave2PublicationResult:
    """Run image-only Wave 2 discovery, QA, manifest creation, and publication."""

    canonical_output_root = _validate_wave2_output_root(
        output_root=output_root,
        dataset_root=dataset_root,
        repository_root=repository_root,
    )
    canonical_dataset_archive = _validate_dataset_archive(dataset_archive)
    _require_output_root_empty_or_absent(canonical_output_root)
    dataset_archive_hash = sha256_file(canonical_dataset_archive)
    dataset_archive_size = canonical_dataset_archive.stat().st_size
    inventory = discover_phase8_ircadb_image_layout(dataset_root)
    qa_results: list[Phase8ImageQaResult] = []
    manifest_cases: list[Phase8ExternalImageCase] = []
    for case in inventory.cases:
        patient_archive = dataset_root / case.patient_dicom_zip_relative_path
        qa_result = run_patient_dicom_zip_image_qa(
            patient_dicom_zip=patient_archive,
            anonymous_case_id=case.anonymous_case_id,
        )
        qa_results.append(qa_result)
        manifest_cases.append(
            _manifest_case_from_qa(
                case=case,
                qa_result=qa_result,
                member_integrity_hash=_patient_archive_member_integrity_hash(patient_archive),
                archive_size_bytes=patient_archive.stat().st_size,
            )
        )
    manifest = build_phase8_external_image_manifest(
        dataset_archive_sha256=dataset_archive_hash,
        dataset_archive_size_bytes=dataset_archive_size,
        adapter_name=IRCADB_PHASE8_ADAPTER_NAME,
        adapter_version=IRCADB_PHASE8_ADAPTER_VERSION,
        cases=tuple(manifest_cases),
    )
    artifacts = {
        PHASE8_WAVE2_LAYOUT_FILENAME: _layout_inventory_payload(inventory),
        PHASE8_WAVE2_QA_FILENAME: _qa_results_payload(qa_results),
        PHASE8_WAVE2_SUMMARY_FILENAME: _summary_payload(
            manifest=manifest,
            qa_results=tuple(qa_results),
        ),
    }
    _publish_json(canonical_output_root / PHASE8_WAVE2_MANIFEST_FILENAME, manifest)
    for filename, payload in artifacts.items():
        _publish_json(canonical_output_root / filename, payload)
    artifact_hashes = {
        relative.name: sha256_file(relative)
        for relative in sorted(canonical_output_root.iterdir(), key=lambda path: path.name)
        if relative.is_file()
    }
    reason_codes = tuple(
        (result.anonymous_case_id, result.reason_codes)
        for result in sorted(qa_results, key=lambda item: item.anonymous_case_id)
        if result.reason_codes
    )
    return Phase8Wave2PublicationResult(
        output_root=canonical_output_root,
        case_count=len(inventory.cases),
        qa_pass_count=sum(1 for result in qa_results if result.qa_status == "passed"),
        qa_fail_count=sum(1 for result in qa_results if result.qa_status == "failed"),
        anonymous_case_reason_codes=reason_codes,
        artifact_hashes=artifact_hashes,
        manifest=manifest,
    )


def _validate_wave2_output_root(
    *,
    output_root: Path,
    dataset_root: Path,
    repository_root: Path,
) -> Path:
    if output_root.exists() and output_root.is_symlink():
        raise Phase8Wave2PublicationError("output_root must not be a symlink.")
    try:
        return validate_explicit_external_output_root(
            output_root,
            forbidden_roots=(dataset_root, repository_root),
        )
    except InvalidDatasetRootError as exc:
        raise Phase8Wave2PublicationError("invalid Wave 2 output root.") from exc


def _require_output_root_empty_or_absent(output_root: Path) -> None:
    if output_root.exists() and any(output_root.iterdir()):
        raise Phase8Wave2PublicationError("Wave 2 output root already contains files.")


def _validate_dataset_archive(dataset_archive: Path) -> Path:
    if not dataset_archive.is_absolute():
        raise Phase8Wave2PublicationError("dataset_archive must be an explicit absolute path.")
    if dataset_archive.is_symlink():
        raise Phase8Wave2PublicationError("dataset_archive must not be a symlink.")
    resolved = dataset_archive.resolve(strict=True)
    if not resolved.is_file():
        raise Phase8Wave2PublicationError("dataset_archive must be a regular file.")
    return resolved


def _manifest_case_from_qa(
    *,
    case: Phase8IrcadbCase,
    qa_result: Phase8ImageQaResult,
    member_integrity_hash: str,
    archive_size_bytes: int,
) -> Phase8ExternalImageCase:
    return Phase8ExternalImageCase(
        schema_name=PHASE8_EXTERNAL_IMAGE_CASE_SCHEMA_NAME,
        schema_version=PHASE8_EXTERNAL_IMAGE_CASE_SCHEMA_VERSION,
        anonymous_case_id=case.anonymous_case_id,
        source_case_ordinal=case.source_case_ordinal,
        image_archive_relative_path=case.patient_dicom_zip_relative_path,
        image_archive_sha256=qa_result.image_archive_sha256,
        image_archive_size_bytes=archive_size_bytes,
        image_member_count=qa_result.image_member_count,
        image_member_integrity_hash=member_integrity_hash,
        image_series_identity=_safe_manifest_image_series_identity(
            case=case,
            qa_result=qa_result,
            member_integrity_hash=member_integrity_hash,
        ),
        image_slice_count=qa_result.image_slice_count,
        image_rows=qa_result.rows,
        image_columns=qa_result.columns,
        volume_shape_zyx=qa_result.volume_shape,
        voxel_spacing_xyz_mm=qa_result.voxel_spacing,
        orientation_validation_state=_validation_state(
            qa_result,
            {Phase8ImageQaReasonCode.INCONSISTENT_IMAGE_ORIENTATION.value},
        ),
        slice_order_validation_state=_validation_state(
            qa_result,
            {
                Phase8ImageQaReasonCode.MISSING_SLICE_POSITION.value,
                Phase8ImageQaReasonCode.SINGLE_SLICE_POSITION_UNDETERMINED.value,
                Phase8ImageQaReasonCode.DUPLICATE_SLICE_POSITION.value,
                Phase8ImageQaReasonCode.NON_MONOTONIC_SLICE_POSITIONS.value,
            },
        ),
        sop_instance_consistency_state=_validation_state(
            qa_result,
            {Phase8ImageQaReasonCode.DUPLICATE_SOP_INSTANCE_UID.value},
        ),
        series_consistency_state=_validation_state(
            qa_result,
            {Phase8ImageQaReasonCode.MULTIPLE_IMAGE_SERIES.value},
        ),
        qa_status=qa_result.qa_status,
        qa_reason_codes=qa_result.reason_codes,
        inclusion_eligible_for_inference=qa_result.qa_status == "passed",
    )


def _validation_state(result: Phase8ImageQaResult, failure_reasons: set[str]) -> str:
    if result.qa_status == "passed":
        return "passed"
    if failure_reasons.intersection(result.reason_codes):
        return "failed"
    return "not_evaluated"


def _safe_manifest_image_series_identity(
    *,
    case: Phase8IrcadbCase,
    qa_result: Phase8ImageQaResult,
    member_integrity_hash: str,
) -> str | None:
    if qa_result.qa_status != "passed":
        return None
    digest = sha256_json(
        {
            "anonymous_case_id": case.anonymous_case_id,
            "identity_type": "phase8-image-series-archive-layout-hash",
            "image_archive_sha256": qa_result.image_archive_sha256,
            "image_member_integrity_hash": member_integrity_hash,
            "image_slice_count": qa_result.image_slice_count,
            "source_case_ordinal": case.source_case_ordinal,
        }
    )
    return f"image-series-{digest}"


def _patient_archive_member_integrity_hash(patient_archive: Path) -> str:
    digest = hashlib.sha256()
    with zipfile.ZipFile(patient_archive, "r") as archive:
        for info in sorted(archive.infolist(), key=lambda item: item.filename):
            if info.filename.endswith("/"):
                continue
            try:
                normalized = normalize_safe_ircadb_zip_member_path(info.filename)
            except ValueError:
                normalized = "unsafe-member"
            digest.update(normalized.encode("utf-8"))
            digest.update(str(info.file_size).encode("ascii"))
            digest.update(str(info.CRC).encode("ascii"))
    return digest.hexdigest()


def _layout_inventory_payload(inventory: Phase8IrcadbInventory) -> dict[str, JsonValue]:
    return {
        "adapter_name": inventory.adapter_name,
        "adapter_version": inventory.adapter_version,
        "case_count": len(inventory.cases),
        "cohort_identity": inventory.cohort_identity,
        "expected_case_count": inventory.expected_case_count,
        "cases": [
            {
                "anonymous_case_id": case.anonymous_case_id,
                "patient_dicom_zip_relative_path": case.patient_dicom_zip_relative_path,
                "prohibited_presence": {
                    "labelled_dicom_zip": case.prohibited_presence.labelled_dicom_zip,
                    "liver_jpg": case.prohibited_presence.liver_jpg,
                    "masks_dicom_zip": case.prohibited_presence.masks_dicom_zip,
                    "meshes_vtk_zip": case.prohibited_presence.meshes_vtk_zip,
                },
                "source_case_ordinal": case.source_case_ordinal,
            }
            for case in inventory.cases
        ],
        "schema_name": "phase8_ircadb_image_layout",
        "schema_version": "v1",
    }


def _qa_results_payload(results: list[Phase8ImageQaResult]) -> dict[str, JsonValue]:
    return {
        "case_count": len(results),
        "cases": [phase8_image_qa_result_to_dict(result) for result in results],
        "schema_name": "phase8_external_image_qa_collection",
        "schema_version": "v1",
    }


def _summary_payload(
    *,
    manifest: Phase8ExternalImageManifest,
    qa_results: tuple[Phase8ImageQaResult, ...],
) -> dict[str, JsonValue]:
    return {
        "anonymous_failed_cases": [
            {
                "anonymous_case_id": result.anonymous_case_id,
                "reason_codes": list(result.reason_codes),
            }
            for result in qa_results
            if result.reason_codes
        ],
        "case_count": len(qa_results),
        "label_accessed": False,
        "manifest_hash": manifest.manifest_hash,
        "qa_fail_count": sum(1 for result in qa_results if result.qa_status == "failed"),
        "qa_pass_count": sum(1 for result in qa_results if result.qa_status == "passed"),
        "raw_data_modified": False,
        "schema_name": "phase8_wave2_generation_summary",
        "schema_version": "v1",
    }


def _publish_json(path: Path, value: object) -> None:
    if isinstance(value, Phase8ExternalImageManifest):
        text = phase8_external_image_manifest_to_json(value).decode("utf-8")
    else:
        text = canonical_json_bytes(cast(JsonValue, value)).decode("utf-8") + "\n"
    try:
        publish_text_no_overwrite(
            text=text,
            output_path=path,
            temporary_exists_message="temporary Phase 8 Wave 2 output already exists",
            final_exists_message="Phase 8 Wave 2 output already exists",
        )
    except (Phase2PublicationExistingOutputError, Phase2PublicationIOError) as exc:
        raise Phase8Wave2PublicationError("failed to publish Wave 2 artifact.") from exc
