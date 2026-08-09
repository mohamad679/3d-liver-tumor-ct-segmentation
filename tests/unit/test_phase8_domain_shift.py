"""Synthetic tests for Phase 8 aggregate-only domain-shift contracts."""

from __future__ import annotations

import dataclasses
import json
from typing import Any, cast

import pytest

from protoem_ct.artifacts.hashing import canonical_json_bytes, sha256_json
from protoem_ct.external.domain_shift import (
    PHASE8_DOMAIN_SHIFT_DIMENSIONS,
    PHASE8_DOMAIN_SHIFT_RECORD_SCHEMA_NAME,
    PHASE8_DOMAIN_SHIFT_RECORD_SCHEMA_VERSION,
    PHASE8_FROZEN_AGGREGATE_REFERENCE_SCHEMA_NAME,
    PHASE8_FROZEN_AGGREGATE_REFERENCE_SCHEMA_VERSION,
    Phase8DomainShiftHashError,
    Phase8DomainShiftSerializationError,
    Phase8DomainShiftValidationError,
    Phase8FrozenAggregateReference,
    build_phase8_domain_shift_record,
    hash_phase8_domain_shift_record,
    phase8_domain_shift_record_from_json,
    phase8_domain_shift_record_to_dict,
    phase8_domain_shift_record_to_json,
)
from protoem_ct.external.manifest import (
    PHASE8_EXTERNAL_IMAGE_CASE_SCHEMA_NAME,
    PHASE8_EXTERNAL_IMAGE_CASE_SCHEMA_VERSION,
    PHASE8_IRCADB_CASE_COUNT,
    Phase8ExternalImageCase,
    Phase8ExternalImageManifest,
    anonymous_ircadb_case_id,
    build_phase8_external_image_manifest,
)

ARCHIVE_HASH = "a" * 64
REFERENCE_HASH = "b" * 64


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
        image_slice_count=20 + ordinal,
        image_rows=512,
        image_columns=512,
        volume_shape_zyx=(20 + ordinal, 512, 512),
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


def _manifest() -> Phase8ExternalImageManifest:
    failed = _case(
        20,
        image_series_identity=None,
        image_slice_count=0,
        image_rows=None,
        image_columns=None,
        volume_shape_zyx=None,
        voxel_spacing_xyz_mm=None,
        orientation_validation_state="failed",
        slice_order_validation_state="failed",
        qa_status="failed",
        qa_reason_codes=("missing_rescaling", "non_ct_modality"),
        inclusion_eligible_for_inference=False,
    )
    cases = tuple(_case(index) for index in range(1, PHASE8_IRCADB_CASE_COUNT)) + (failed,)
    return build_phase8_external_image_manifest(
        dataset_archive_sha256=ARCHIVE_HASH,
        dataset_archive_size_bytes=820270584,
        adapter_name="ircadb_phase8_image_only",
        adapter_version="v1",
        cases=cases,
    )


def _reference() -> Phase8FrozenAggregateReference:
    return Phase8FrozenAggregateReference(
        schema_name=PHASE8_FROZEN_AGGREGATE_REFERENCE_SCHEMA_NAME,
        schema_version=PHASE8_FROZEN_AGGREGATE_REFERENCE_SCHEMA_VERSION,
        reference_role="internal_preprocessing_contract",
        referenced_schema_name="phase6_frozen_preprocessing_config",
        referenced_schema_version="v1",
        artifact_hash=REFERENCE_HASH,
        approval_state="approved",
    )


def test_domain_shift_record_round_trip_and_canonical_hash() -> None:
    record = build_phase8_domain_shift_record(external_manifest=_manifest())
    encoded = phase8_domain_shift_record_to_json(record)

    restored = phase8_domain_shift_record_from_json(encoded)

    assert restored == record
    assert restored.schema_name == PHASE8_DOMAIN_SHIFT_RECORD_SCHEMA_NAME
    assert restored.schema_version == PHASE8_DOMAIN_SHIFT_RECORD_SCHEMA_VERSION
    assert restored.domain_shift_record_hash == hash_phase8_domain_shift_record(restored)
    assert encoded == phase8_domain_shift_record_to_json(restored)
    assert encoded == canonical_json_bytes(json.loads(encoded)) + b"\n"
    assert (
        tuple(item.dimension_name for item in restored.dimensions) == PHASE8_DOMAIN_SHIFT_DIMENSIONS
    )


def test_domain_shift_uses_aggregates_only_and_marks_unavailable_internal_comparison() -> None:
    record = build_phase8_domain_shift_record(external_manifest=_manifest())
    payload = phase8_domain_shift_record_to_dict(record)
    text = json.dumps(payload, sort_keys=True)

    assert record.aggregate_only is True
    assert record.labels_accessed is False
    assert record.raw_images_accessed_by_domain_shift_generation is False
    assert record.scanner_vendor_site_recorded is False
    assert "ext-ircadb-" not in text
    assert "PATIENT_DICOM" not in text
    assert "/Volumes" not in text
    assert "PatientName" not in text
    assert "MASKS_DICOM" not in text

    comparison = record.dimensions[-1]
    assert comparison.dimension_name == "differences_vs_frozen_internal_preprocessing_contract"
    assert comparison.availability_status == "unavailable"
    assert comparison.unavailable_reason_codes == ("approved_internal_aggregate_reference_absent",)
    assert comparison.external_aggregate_summary is None


def test_domain_shift_aggregate_values_are_from_manifest_fields() -> None:
    record = build_phase8_domain_shift_record(external_manifest=_manifest())
    summaries = {
        dimension.dimension_name: dimension.external_aggregate_summary
        for dimension in record.dimensions
    }

    assert summaries["image_matrix"] == {
        "available_case_count": 19,
        "columns_max": 512,
        "columns_min": 512,
        "rows_max": 512,
        "rows_min": 512,
        "unavailable_case_count": 1,
        "unique_matrix_count": 1,
    }
    assert summaries["image_qa_compatibility"] == {
        "failed_case_count": 1,
        "inclusion_eligible_for_inference_count": 19,
        "passed_case_count": 19,
        "reason_code_counts": {"missing_rescaling": 1, "non_ct_modality": 1},
    }
    assert summaries["orientation"] == {
        "orientation_failed_case_count": 1,
        "orientation_not_evaluated_case_count": 0,
        "orientation_passed_case_count": 19,
        "orientation_value_status": "unavailable_not_persisted_in_image_manifest",
    }


def test_internal_comparison_requires_approved_aggregate_reference() -> None:
    record = build_phase8_domain_shift_record(
        external_manifest=_manifest(),
        internal_preprocessing_reference=_reference(),
        internal_preprocessing_difference_summary={
            "spacing_policy_relation": "external_spacing_range_compared_to_frozen_target_spacing",
            "slice_axis_policy_relation": "external_slice_counts_compared_to_config_only",
        },
    )

    comparison = record.dimensions[-1]
    assert comparison.availability_status == "available"
    assert comparison.comparison_status == "compared_to_approved_aggregate_reference"
    assert comparison.approved_reference == _reference()

    with pytest.raises(Phase8DomainShiftValidationError, match="requires both"):
        build_phase8_domain_shift_record(
            external_manifest=_manifest(),
            internal_preprocessing_reference=_reference(),
        )


def test_unknown_fields_bad_hash_and_forbidden_values_are_rejected() -> None:
    payload = phase8_domain_shift_record_to_dict(
        build_phase8_domain_shift_record(external_manifest=_manifest())
    )
    payload["created_at_utc"] = "2026-08-04T00:00:00Z"
    with pytest.raises(Phase8DomainShiftSerializationError, match="extra"):
        phase8_domain_shift_record_from_json(canonical_json_bytes(payload))

    payload = phase8_domain_shift_record_to_dict(
        build_phase8_domain_shift_record(external_manifest=_manifest())
    )
    payload["domain_shift_record_hash"] = "0" * 64
    with pytest.raises(Phase8DomainShiftHashError, match="domain_shift_record_hash"):
        phase8_domain_shift_record_from_json(canonical_json_bytes(payload))

    with pytest.raises(Phase8DomainShiftValidationError, match="scanner/vendor/site"):
        build_phase8_domain_shift_record(
            external_manifest=_manifest(),
            internal_preprocessing_reference=_reference(),
            internal_preprocessing_difference_summary={"scanner_vendor_site": "scanner_a"},
        )
