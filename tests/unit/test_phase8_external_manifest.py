"""Synthetic tests for Phase 8 anonymous external image manifests."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, cast

import pytest

from protoem_ct.artifacts.hashing import canonical_json_bytes, sha256_json
from protoem_ct.external.manifest import (
    PHASE8_DICOM_SAFE_HEADER_ALLOWLIST,
    PHASE8_EXTERNAL_IMAGE_CASE_SCHEMA_NAME,
    PHASE8_EXTERNAL_IMAGE_CASE_SCHEMA_VERSION,
    PHASE8_EXTERNAL_IMAGE_MANIFEST_SCHEMA_NAME,
    PHASE8_EXTERNAL_IMAGE_MANIFEST_SCHEMA_VERSION,
    PHASE8_IRCADB_CASE_COUNT,
    Phase8ExternalImageCase,
    Phase8ExternalImageManifest,
    Phase8ExternalManifestHashError,
    Phase8ExternalManifestPublicationError,
    Phase8ExternalManifestSerializationError,
    Phase8ExternalManifestValidationError,
    anonymous_ircadb_case_id,
    build_phase8_external_image_manifest,
    hash_phase8_external_image_manifest,
    phase8_external_image_manifest_from_json,
    phase8_external_image_manifest_to_dict,
    phase8_external_image_manifest_to_json,
    publish_phase8_external_image_manifest,
)

ARCHIVE_HASH = "a" * 64


def _case(ordinal: int, **overrides: object) -> Phase8ExternalImageCase:
    case = Phase8ExternalImageCase(
        schema_name=PHASE8_EXTERNAL_IMAGE_CASE_SCHEMA_NAME,
        schema_version=PHASE8_EXTERNAL_IMAGE_CASE_SCHEMA_VERSION,
        anonymous_case_id=anonymous_ircadb_case_id(ordinal),
        source_case_ordinal=ordinal,
        image_archive_relative_path=f"3Dircadb1.{ordinal}/PATIENT_DICOM.zip",
        image_archive_sha256=sha256_json({"archive": ordinal}),
        image_archive_size_bytes=1024 + ordinal,
        image_member_count=8 + ordinal,
        image_member_integrity_hash=sha256_json({"members": ordinal}),
        image_series_identity=f"image-series-{sha256_json({'series': ordinal})[:16]}",
        image_slice_count=4 + ordinal,
        image_rows=32,
        image_columns=48,
        volume_shape_zyx=(4 + ordinal, 32, 48),
        voxel_spacing_xyz_mm=(0.75, 0.75, 2.5),
        orientation_validation_state="passed",
        slice_order_validation_state="passed",
        sop_instance_consistency_state="passed",
        series_consistency_state="passed",
        qa_status="passed",
        qa_reason_codes=(),
        inclusion_eligible_for_inference=True,
    )
    return cast(Phase8ExternalImageCase, cast(Any, dataclasses.replace)(case, **overrides))


def _cases() -> tuple[Phase8ExternalImageCase, ...]:
    return tuple(_case(index) for index in range(1, PHASE8_IRCADB_CASE_COUNT + 1))


def _manifest() -> Phase8ExternalImageManifest:
    return build_phase8_external_image_manifest(
        dataset_archive_sha256=ARCHIVE_HASH,
        dataset_archive_size_bytes=820270584,
        adapter_name="ircadb_phase8_image_only",
        adapter_version="v1",
        cases=_cases(),
    )


def test_anonymous_case_id_uses_public_ordinal_only() -> None:
    assert anonymous_ircadb_case_id(1) == "ext-ircadb-001"
    assert anonymous_ircadb_case_id(20) == "ext-ircadb-020"

    with pytest.raises(Phase8ExternalManifestValidationError):
        _case(1, anonymous_case_id="patient-001")


def test_manifest_round_trip_and_canonical_bytes() -> None:
    manifest = _manifest()
    encoded = phase8_external_image_manifest_to_json(manifest)

    restored = phase8_external_image_manifest_from_json(encoded)

    assert restored == manifest
    assert restored.manifest_hash == hash_phase8_external_image_manifest(restored)
    assert encoded == phase8_external_image_manifest_to_json(restored)
    assert encoded == canonical_json_bytes(json.loads(encoded)) + b"\n"
    assert phase8_external_image_manifest_to_dict(restored)["schema_name"] == (
        PHASE8_EXTERNAL_IMAGE_MANIFEST_SCHEMA_NAME
    )
    assert restored.schema_version == PHASE8_EXTERNAL_IMAGE_MANIFEST_SCHEMA_VERSION


def test_unknown_fields_and_invalid_self_hash_are_rejected() -> None:
    payload = phase8_external_image_manifest_to_dict(_manifest())
    payload["created_at_utc"] = "2026-08-04T00:00:00Z"

    with pytest.raises(Phase8ExternalManifestSerializationError, match="extra"):
        phase8_external_image_manifest_from_json(canonical_json_bytes(payload))

    payload = phase8_external_image_manifest_to_dict(_manifest())
    payload["manifest_hash"] = "0" * 64
    with pytest.raises(Phase8ExternalManifestHashError, match="manifest_hash"):
        phase8_external_image_manifest_from_json(canonical_json_bytes(payload))


def test_manifest_requires_exact_ircadb_case_ordinal_set() -> None:
    with pytest.raises(Phase8ExternalManifestValidationError, match="exactly 20"):
        build_phase8_external_image_manifest(
            dataset_archive_sha256=ARCHIVE_HASH,
            dataset_archive_size_bytes=1,
            adapter_name="ircadb_phase8_image_only",
            adapter_version="v1",
            cases=_cases()[:-1],
        )

    duplicate = _cases()[:-1] + (_case(1),)
    with pytest.raises(Phase8ExternalManifestValidationError, match="1 through 20"):
        build_phase8_external_image_manifest(
            dataset_archive_sha256=ARCHIVE_HASH,
            dataset_archive_size_bytes=1,
            adapter_name="ircadb_phase8_image_only",
            adapter_version="v1",
            cases=duplicate,
        )


def test_input_order_is_canonicalized() -> None:
    ordered = _manifest()
    reversed_manifest = build_phase8_external_image_manifest(
        dataset_archive_sha256=ARCHIVE_HASH,
        dataset_archive_size_bytes=820270584,
        adapter_name="ircadb_phase8_image_only",
        adapter_version="v1",
        cases=tuple(reversed(_cases())),
    )

    assert reversed_manifest == ordered
    assert [case.source_case_ordinal for case in reversed_manifest.cases] == list(range(1, 21))
    assert phase8_external_image_manifest_to_json(reversed_manifest) == (
        phase8_external_image_manifest_to_json(ordered)
    )


def test_safe_fields_reject_absolute_paths_phi_tokens_and_label_archives() -> None:
    with pytest.raises(Phase8ExternalManifestValidationError, match="absolute"):
        _case(1, image_archive_relative_path="/Volumes/Lexar/PATIENT_DICOM.zip")

    with pytest.raises(Phase8ExternalManifestValidationError, match="PHI"):
        build_phase8_external_image_manifest(
            dataset_archive_sha256=ARCHIVE_HASH,
            dataset_archive_size_bytes=1,
            adapter_name="patientid",
            adapter_version="v1",
            cases=_cases(),
        )

    with pytest.raises(Phase8ExternalManifestValidationError, match="PATIENT_DICOM.zip"):
        _case(1, image_archive_relative_path="3Dircadb1.1/MASKS_DICOM.zip")


def test_qa_reason_codes_control_inference_eligibility() -> None:
    failed = _case(
        1,
        qa_status="failed",
        qa_reason_codes=("duplicate_sop_instance_uid", "slice_order_not_determinable"),
        inclusion_eligible_for_inference=False,
        slice_order_validation_state="failed",
        sop_instance_consistency_state="failed",
    )

    assert failed.qa_reason_codes == (
        "duplicate_sop_instance_uid",
        "slice_order_not_determinable",
    )

    with pytest.raises(Phase8ExternalManifestValidationError, match="requires qa_status"):
        _case(1, qa_status="failed", qa_reason_codes=("bad_archive",))

    with pytest.raises(Phase8ExternalManifestValidationError, match="require qa_reason_codes"):
        _case(1, qa_status="failed", qa_reason_codes=(), inclusion_eligible_for_inference=False)


def test_failed_case_can_record_unavailable_geometry_without_fabrication() -> None:
    failed = _case(
        1,
        image_series_identity=None,
        image_slice_count=0,
        image_rows=None,
        image_columns=None,
        volume_shape_zyx=None,
        voxel_spacing_xyz_mm=None,
        orientation_validation_state="not_evaluated",
        slice_order_validation_state="not_evaluated",
        sop_instance_consistency_state="not_evaluated",
        series_consistency_state="not_evaluated",
        qa_status="failed",
        qa_reason_codes=("readable_archive_failed",),
        inclusion_eligible_for_inference=False,
    )

    payload = phase8_external_image_manifest_to_dict(
        build_phase8_external_image_manifest(
            dataset_archive_sha256=ARCHIVE_HASH,
            dataset_archive_size_bytes=820270584,
            adapter_name="ircadb_phase8_image_only",
            adapter_version="v1",
            cases=(failed,) + _cases()[1:],
        )
    )

    first_case = cast(dict[str, object], cast(list[object], payload["cases"])[0])
    assert first_case["volume_shape_zyx"] is None
    assert first_case["voxel_spacing_xyz_mm"] is None


def test_no_raw_dicom_header_dictionary_is_exposed() -> None:
    manifest = phase8_external_image_manifest_to_dict(_manifest())
    text = json.dumps(manifest, sort_keys=True)

    assert "PatientID" not in text
    assert "PatientName" not in text
    assert "AccessionNumber" not in text
    assert "InstitutionName" not in text
    assert "dicom_headers" not in text
    assert "MASKS_DICOM" not in text
    assert "LABELLED_DICOM" not in text
    assert "MESHES_VTK" not in text
    assert "liver_" not in text
    assert (
        frozenset(
            {
                "Columns",
                "ImageOrientationPatient",
                "ImagePositionPatient",
                "Modality",
                "PixelSpacing",
                "RescaleIntercept",
                "RescaleSlope",
                "Rows",
                "SliceThickness",
                "SpacingBetweenSlices",
            }
        )
        == PHASE8_DICOM_SAFE_HEADER_ALLOWLIST
    )


def test_publication_writes_canonical_json_without_overwrite(tmp_path: Path) -> None:
    manifest = _manifest()
    output_path = tmp_path / "phase8_external_image_manifest.json"

    publish_phase8_external_image_manifest(manifest, output_path)

    assert output_path.read_bytes() == phase8_external_image_manifest_to_json(manifest)
    with pytest.raises(Phase8ExternalManifestPublicationError, match="already exists"):
        publish_phase8_external_image_manifest(manifest, output_path)
    with pytest.raises(Phase8ExternalManifestPublicationError, match="absolute"):
        publish_phase8_external_image_manifest(manifest, Path("relative.json"))
