"""Synthetic tests for Phase 8 Wave 3 policy publication."""

from __future__ import annotations

from pathlib import Path

import pytest

from protoem_ct.artifacts.hashing import canonical_json_bytes
from protoem_ct.external.domain_shift import phase8_domain_shift_record_from_json
from protoem_ct.external.eligibility import (
    phase8_eligibility_accounting_from_json,
    phase8_eligibility_policy_from_json,
)
from protoem_ct.external.label_mapping import phase8_label_mapping_policy_from_json
from protoem_ct.external.manifest import (
    PHASE8_EXTERNAL_IMAGE_CASE_SCHEMA_NAME,
    PHASE8_EXTERNAL_IMAGE_CASE_SCHEMA_VERSION,
    Phase8ExternalImageCase,
    Phase8ExternalImageManifest,
    build_phase8_external_image_manifest,
    phase8_external_image_manifest_to_json,
)
from protoem_ct.external.wave2 import (
    PHASE8_WAVE2_LAYOUT_FILENAME,
    PHASE8_WAVE2_MANIFEST_FILENAME,
    PHASE8_WAVE2_QA_FILENAME,
    PHASE8_WAVE2_SUMMARY_FILENAME,
)
from protoem_ct.external.wave3 import (
    PHASE8_WAVE3_COHORT_ACCOUNTING_FILENAME,
    PHASE8_WAVE3_DOMAIN_SHIFT_RECORD_FILENAME,
    PHASE8_WAVE3_ELIGIBILITY_POLICY_FILENAME,
    PHASE8_WAVE3_LABEL_MAPPING_POLICY_FILENAME,
    PHASE8_WAVE3_SUMMARY_FILENAME,
    Phase8Wave3PublicationError,
    run_phase8_wave3_policy_publication,
)


def test_wave3_publication_freezes_policy_and_defers_label_compatibility(
    tmp_path: Path,
) -> None:
    wave2_root = tmp_path / "runs" / "phase8_wave2"
    output_root = tmp_path / "runs" / "phase8_wave3"
    repository_root = tmp_path / "repo"
    repository_root.mkdir()
    manifest = _write_wave2_artifacts(wave2_root)

    result = run_phase8_wave3_policy_publication(
        wave2_artifact_root=wave2_root,
        output_root=output_root,
        repository_root=repository_root,
    )

    assert result.case_count == 20
    assert result.image_qa_eligible_count == 20
    assert result.inference_eligible_count == 20
    assert result.label_compatibility_pending_count == 20
    assert result.evaluation_eligible_count == 0
    assert result.deferred_count == 20
    assert sorted(result.artifact_hashes) == [
        PHASE8_WAVE3_COHORT_ACCOUNTING_FILENAME,
        PHASE8_WAVE3_DOMAIN_SHIFT_RECORD_FILENAME,
        PHASE8_WAVE3_ELIGIBILITY_POLICY_FILENAME,
        PHASE8_WAVE3_LABEL_MAPPING_POLICY_FILENAME,
        PHASE8_WAVE3_SUMMARY_FILENAME,
    ]
    policy = phase8_label_mapping_policy_from_json(
        (output_root / PHASE8_WAVE3_LABEL_MAPPING_POLICY_FILENAME).read_bytes()
    )
    eligibility_policy = phase8_eligibility_policy_from_json(
        (output_root / PHASE8_WAVE3_ELIGIBILITY_POLICY_FILENAME).read_bytes()
    )
    domain_shift = phase8_domain_shift_record_from_json(
        (output_root / PHASE8_WAVE3_DOMAIN_SHIFT_RECORD_FILENAME).read_bytes()
    )
    accounting = phase8_eligibility_accounting_from_json(
        (output_root / PHASE8_WAVE3_COHORT_ACCOUNTING_FILENAME).read_bytes()
    )
    assert policy.verification_state == "expected_documented_not_empirically_verified"
    assert eligibility_policy.policy_hash == result.eligibility_policy.policy_hash
    assert domain_shift.external_manifest_hash == manifest.manifest_hash
    assert accounting.evaluation_eligible_count == 0
    assert accounting.label_compatibility_pending_count == 20
    text = "\n".join(path.read_text(encoding="utf-8") for path in output_root.iterdir())
    assert "/Volumes" not in text
    assert "MASKS_DICOM.zip" not in text
    assert "LABELLED_DICOM.zip" not in text
    assert "MESHES_VTK.zip" not in text


def test_wave3_publication_rejects_repository_output_root(tmp_path: Path) -> None:
    wave2_root = tmp_path / "runs" / "phase8_wave2"
    repository_root = tmp_path / "repo"
    repository_root.mkdir()
    _write_wave2_artifacts(wave2_root)

    with pytest.raises(Phase8Wave3PublicationError):
        run_phase8_wave3_policy_publication(
            wave2_artifact_root=wave2_root,
            output_root=repository_root / "phase8_wave3",
            repository_root=repository_root,
        )


def _write_wave2_artifacts(wave2_root: Path) -> Phase8ExternalImageManifest:
    wave2_root.mkdir(parents=True)
    cases = tuple(_manifest_case(index) for index in range(1, 21))
    manifest = build_phase8_external_image_manifest(
        dataset_archive_sha256="a" * 64,
        dataset_archive_size_bytes=123456,
        adapter_name="phase8_ircadb_image_layout",
        adapter_version="v1",
        cases=cases,
    )
    (wave2_root / PHASE8_WAVE2_MANIFEST_FILENAME).write_bytes(
        phase8_external_image_manifest_to_json(manifest)
    )
    _write_json(
        wave2_root / PHASE8_WAVE2_QA_FILENAME,
        {
            "case_count": 20,
            "cases": [],
            "schema_name": "phase8_external_image_qa_collection",
            "schema_version": "v1",
        },
    )
    _write_json(
        wave2_root / PHASE8_WAVE2_LAYOUT_FILENAME,
        {
            "case_count": 20,
            "cases": [],
            "schema_name": "phase8_ircadb_image_layout",
            "schema_version": "v1",
        },
    )
    _write_json(
        wave2_root / PHASE8_WAVE2_SUMMARY_FILENAME,
        {
            "case_count": 20,
            "label_accessed": False,
            "manifest_hash": manifest.manifest_hash,
            "qa_fail_count": 0,
            "qa_pass_count": 20,
            "raw_data_modified": False,
            "schema_name": "phase8_wave2_generation_summary",
            "schema_version": "v1",
        },
    )
    return manifest


def _manifest_case(index: int) -> Phase8ExternalImageCase:
    return Phase8ExternalImageCase(
        schema_name=PHASE8_EXTERNAL_IMAGE_CASE_SCHEMA_NAME,
        schema_version=PHASE8_EXTERNAL_IMAGE_CASE_SCHEMA_VERSION,
        anonymous_case_id=f"ext-ircadb-{index:03d}",
        source_case_ordinal=index,
        image_archive_relative_path=f"3Dircadb1.{index}/PATIENT_DICOM.zip",
        image_archive_sha256=f"{index:064x}",
        image_archive_size_bytes=1000 + index,
        image_member_count=2,
        image_member_integrity_hash=f"{index + 100:064x}",
        image_series_identity=f"image-series-{index:016x}",
        image_slice_count=2 + index,
        image_rows=512,
        image_columns=512,
        volume_shape_zyx=(2 + index, 512, 512),
        voxel_spacing_xyz_mm=(0.7, 0.8, 2.5),
        orientation_validation_state="passed",
        slice_order_validation_state="passed",
        sop_instance_consistency_state="passed",
        series_consistency_state="passed",
        qa_status="passed",
        qa_reason_codes=(),
        inclusion_eligible_for_inference=True,
    )


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")
