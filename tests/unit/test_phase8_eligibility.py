"""Synthetic tests for Phase 8 eligibility and label-compatibility accounting."""

from __future__ import annotations

import dataclasses
import json
from typing import Any, cast

import pytest

from protoem_ct.artifacts.hashing import canonical_json_bytes, sha256_json
from protoem_ct.external.eligibility import (
    PHASE8_ELIGIBILITY_ACCOUNTING_SCHEMA_NAME,
    PHASE8_ELIGIBILITY_ACCOUNTING_SCHEMA_VERSION,
    PHASE8_ELIGIBILITY_CASE_SCHEMA_NAME,
    PHASE8_ELIGIBILITY_CASE_SCHEMA_VERSION,
    Phase8EligibilityCase,
    Phase8EligibilityHashError,
    Phase8EligibilitySerializationError,
    Phase8EligibilityValidationError,
    build_phase8_label_compatible_case,
    build_phase8_post_label_eligibility_accounting,
    build_phase8_prelabel_eligibility_accounting,
    hash_phase8_eligibility_accounting,
    hash_phase8_eligibility_policy,
    phase8_eligibility_accounting_from_json,
    phase8_eligibility_accounting_to_dict,
    phase8_eligibility_accounting_to_json,
    phase8_eligibility_policy_payload,
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
LEDGER_HASH = "b" * 64


def _image_case(ordinal: int, **overrides: object) -> Phase8ExternalImageCase:
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


def _image_manifest(
    cases: tuple[Phase8ExternalImageCase, ...] | None = None,
) -> Phase8ExternalImageManifest:
    return build_phase8_external_image_manifest(
        dataset_archive_sha256=ARCHIVE_HASH,
        dataset_archive_size_bytes=820270584,
        adapter_name="ircadb_phase8_image_only",
        adapter_version="v1",
        cases=cases
        if cases is not None
        else tuple(_image_case(index) for index in range(1, PHASE8_IRCADB_CASE_COUNT + 1)),
    )


def test_prelabel_accounting_preserves_20_anonymous_ids_and_defers_evaluation() -> None:
    accounting = build_phase8_prelabel_eligibility_accounting(image_manifest=_image_manifest())

    assert accounting.schema_name == PHASE8_ELIGIBILITY_ACCOUNTING_SCHEMA_NAME
    assert accounting.schema_version == PHASE8_ELIGIBILITY_ACCOUNTING_SCHEMA_VERSION
    assert accounting.case_count == 20
    assert [case.anonymous_case_id for case in accounting.cases] == [
        f"ext-ircadb-{index:03d}" for index in range(1, 21)
    ]
    assert accounting.discovered_count == 20
    assert accounting.image_readable_count == 20
    assert accounting.image_qa_eligible_count == 20
    assert accounting.inference_eligible_count == 20
    assert accounting.label_compatibility_pending_count == 20
    assert accounting.evaluation_eligible_count == 0
    assert accounting.deferred_count == 20
    assert accounting.excluded_count == 0
    assert accounting.cases[0].label_compatibility_status == "pending"
    assert accounting.cases[0].reason_codes == ("label_compatibility_pending",)


def test_round_trip_canonical_bytes_and_self_hash_validation() -> None:
    accounting = build_phase8_prelabel_eligibility_accounting(image_manifest=_image_manifest())
    encoded = phase8_eligibility_accounting_to_json(accounting)

    restored = phase8_eligibility_accounting_from_json(encoded)

    assert restored == accounting
    assert hash_phase8_eligibility_accounting(restored) == accounting.accounting_hash
    assert encoded == phase8_eligibility_accounting_to_json(restored)
    assert encoded == canonical_json_bytes(json.loads(encoded)) + b"\n"
    assert encoded.endswith(b"\n")


def test_unknown_fields_and_bad_self_hash_rejected() -> None:
    payload = phase8_eligibility_accounting_to_dict(
        build_phase8_prelabel_eligibility_accounting(image_manifest=_image_manifest())
    )
    payload["created_at_utc"] = "2026-08-04T00:00:00Z"

    with pytest.raises(Phase8EligibilitySerializationError, match="extra"):
        phase8_eligibility_accounting_from_json(canonical_json_bytes(payload))

    payload = phase8_eligibility_accounting_to_dict(
        build_phase8_prelabel_eligibility_accounting(image_manifest=_image_manifest())
    )
    payload["accounting_hash"] = "0" * 64
    with pytest.raises(Phase8EligibilityHashError, match="accounting_hash"):
        phase8_eligibility_accounting_from_json(canonical_json_bytes(payload))


def test_failed_image_case_is_retained_with_explicit_exclusion_reasons() -> None:
    failed = _image_case(
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
        image_member_count=0,
    )

    accounting = build_phase8_prelabel_eligibility_accounting(
        image_manifest=_image_manifest((failed,) + tuple(_image_case(i) for i in range(2, 21)))
    )

    first = accounting.cases[0]
    assert first.anonymous_case_id == "ext-ircadb-001"
    assert first.discovered is True
    assert first.image_readable is False
    assert first.image_qa_eligible is False
    assert first.inference_eligible is False
    assert first.excluded is True
    assert first.deferred is False
    assert first.evaluation_eligible is False
    assert first.reason_codes == ("image_not_readable", "image_qa_failed", "inference_ineligible")
    assert accounting.excluded_count == 1
    assert accounting.deferred_count == 19
    assert accounting.case_count == 20


def test_prelabel_accounting_never_claims_evaluation_eligibility_from_unseen_labels() -> None:
    accounting = build_phase8_prelabel_eligibility_accounting(image_manifest=_image_manifest())
    case = accounting.cases[0]

    with pytest.raises(Phase8EligibilityValidationError, match="evaluation_eligible"):
        dataclasses.replace(
            case,
            evaluation_eligible=True,
            deferred=False,
            label_compatibility_status="pending",
            reason_codes=(),
        )


def test_post_label_compatibility_can_make_evaluation_eligible_after_ledger_hash() -> None:
    prelabel = build_phase8_prelabel_eligibility_accounting(image_manifest=_image_manifest())
    compatible_cases = tuple(
        build_phase8_label_compatible_case(
            case,
            compatible=True,
            label_compatibility_reason_codes=("compatible_binary_tumor_target",),
        )
        for case in prelabel.cases
    )

    postlabel = build_phase8_post_label_eligibility_accounting(
        previous_accounting=prelabel,
        label_access_ledger_hash=LEDGER_HASH,
        compatibility_updates=compatible_cases,
    )

    assert postlabel.accounting_state == "post_label_access_accounted"
    assert postlabel.label_access_ledger_hash == LEDGER_HASH
    assert postlabel.label_compatibility_pending_count == 0
    assert postlabel.evaluation_eligible_count == 20
    assert postlabel.excluded_count == 0
    assert postlabel.deferred_count == 0
    assert postlabel.cases[0].label_compatibility_status == "compatible"
    assert postlabel.cases[0].evaluation_eligible is True


def test_post_label_incompatibility_excludes_without_changing_image_accounting() -> None:
    prelabel = build_phase8_prelabel_eligibility_accounting(image_manifest=_image_manifest())
    updates = list(
        build_phase8_label_compatible_case(
            case,
            compatible=True,
            label_compatibility_reason_codes=("compatible_binary_tumor_target",),
        )
        for case in prelabel.cases
    )
    updates[2] = build_phase8_label_compatible_case(
        prelabel.cases[2],
        compatible=False,
        label_compatibility_reason_codes=("label_geometry_incompatible",),
    )

    postlabel = build_phase8_post_label_eligibility_accounting(
        previous_accounting=prelabel,
        label_access_ledger_hash=LEDGER_HASH,
        compatibility_updates=tuple(updates),
    )

    assert postlabel.evaluation_eligible_count == 19
    assert postlabel.excluded_count == 1
    assert postlabel.cases[2].anonymous_case_id == "ext-ircadb-003"
    assert postlabel.cases[2].reason_codes == ("label_incompatible",)

    tampered = dataclasses.replace(
        updates[0],
        image_qa_eligible=False,
        label_compatibility_status="incompatible",
        label_compatibility_reason_codes=("label_geometry_incompatible",),
        evaluation_eligible=False,
        excluded=True,
        reason_codes=("image_qa_failed", "label_incompatible"),
    )
    with pytest.raises(Phase8EligibilityValidationError, match="image_qa_eligible"):
        build_phase8_post_label_eligibility_accounting(
            previous_accounting=prelabel,
            label_access_ledger_hash=LEDGER_HASH,
            compatibility_updates=(tampered,) + tuple(updates[1:]),
        )


def test_duplicate_ids_counts_duplicate_reason_codes_and_unknown_codes_rejected() -> None:
    accounting = build_phase8_prelabel_eligibility_accounting(image_manifest=_image_manifest())
    payload = phase8_eligibility_accounting_to_dict(accounting)
    cases = cast(list[dict[str, object]], payload["cases"])
    cases[1]["anonymous_case_id"] = "ext-ircadb-001"
    payload["accounting_hash"] = sha256_json(
        {k: v for k, v in payload.items() if k != "accounting_hash"}
    )
    with pytest.raises(Phase8EligibilityValidationError, match="duplicate anonymous IDs"):
        phase8_eligibility_accounting_from_json(canonical_json_bytes(payload))

    with pytest.raises(Phase8EligibilityValidationError, match="duplicates"):
        Phase8EligibilityCase(
            schema_name=PHASE8_ELIGIBILITY_CASE_SCHEMA_NAME,
            schema_version=PHASE8_ELIGIBILITY_CASE_SCHEMA_VERSION,
            anonymous_case_id="ext-ircadb-001",
            discovered=True,
            image_readable=True,
            image_qa_eligible=True,
            inference_eligible=True,
            label_compatibility_status="pending",
            label_compatibility_reason_codes=("pending_label_access",),
            evaluation_eligible=False,
            excluded=False,
            deferred=True,
            reason_codes=("label_compatibility_pending", "label_compatibility_pending"),
        )

    with pytest.raises(Phase8EligibilityValidationError, match="unknown"):
        Phase8EligibilityCase(
            schema_name=PHASE8_ELIGIBILITY_CASE_SCHEMA_NAME,
            schema_version=PHASE8_ELIGIBILITY_CASE_SCHEMA_VERSION,
            anonymous_case_id="ext-ircadb-001",
            discovered=True,
            image_readable=True,
            image_qa_eligible=True,
            inference_eligible=True,
            label_compatibility_status="pending",
            label_compatibility_reason_codes=("pending_label_access",),
            evaluation_eligible=False,
            excluded=False,
            deferred=True,
            reason_codes=("label_compatibility_pending", "raw_patient_id_found"),
        )


def test_identity_excludes_paths_phi_raw_ids_and_timestamps() -> None:
    accounting = build_phase8_prelabel_eligibility_accounting(image_manifest=_image_manifest())
    payload_text = json.dumps(
        phase8_eligibility_accounting_to_dict(accounting),
        sort_keys=True,
    )

    assert "/Volumes" not in payload_text
    assert "PATIENT_DICOM.zip" not in payload_text
    assert "MASKS_DICOM" not in payload_text
    assert "LABELLED_DICOM" not in payload_text
    assert "PatientID" not in payload_text
    assert "created_at" not in payload_text
    assert phase8_eligibility_policy_payload()["schema_name"] == "phase8_eligibility_policy"
    assert accounting.preregistered_policy_hash == hash_phase8_eligibility_policy()
