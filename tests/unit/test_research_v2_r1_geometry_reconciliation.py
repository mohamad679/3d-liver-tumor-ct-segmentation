from __future__ import annotations

import numpy as np

from protoem_ct.research_v2.r1_geometry_reconciliation import (
    R1_SOURCE_AFFINE_ABS_TOLERANCE_MM,
    diagnose_effective_geometry,
    reconcile_geometry_mismatches,
)


def _record(*, translation_mm: float = 0.0) -> dict[str, object]:
    image_affine = np.eye(4, dtype=float)
    label_affine = np.eye(4, dtype=float)
    label_affine[0, 3] = translation_mm
    return {
        "anonymous_case_id": "case-fixture",
        "partition": "train",
        "image_shape": [8, 8, 8],
        "label_shape": [8, 8, 8],
        "image_affine": image_affine.tolist(),
        "label_affine": label_affine.tolist(),
        "image_spacing_mm": [1.0, 1.0, 1.0],
        "label_spacing_mm": [1.0, 1.0, 1.0],
        "image_orientation": ["R", "A", "S"],
        "label_orientation": ["R", "A", "S"],
    }


def test_phase2_source_tolerance_is_one_ten_thousandth_mm() -> None:
    assert R1_SOURCE_AFFINE_ABS_TOLERANCE_MM == 1e-4


def test_historical_precision_scale_is_safe_for_effective_source_geometry() -> None:
    diagnostic = diagnose_effective_geometry(_record(translation_mm=3.0e-5))
    assert diagnostic.shape_match
    assert diagnostic.orientation_match
    assert diagnostic.effective_affine_match_at_approved_tolerance
    assert diagnostic.spacing_match_at_approved_tolerance
    assert diagnostic.within_historical_phase2_envelope


def test_reconciliation_retains_qform_observation_but_resolves_gate_failure() -> None:
    records = [_record(translation_mm=3.0e-5)]
    mismatches = [
        {
            "anonymous_case_id": "case-fixture",
            "partition": "train",
            "reason": "label_qform_sform_conflict",
            "resolution_status": "unresolved",
            "resolution": None,
        },
        {
            "anonymous_case_id": "case-fixture",
            "partition": "train",
            "reason": "image_label_affine_mismatch",
            "resolution_status": "unresolved",
            "resolution": None,
        },
    ]

    reconciled, summary = reconcile_geometry_mismatches(records=records, mismatches=mismatches)

    assert reconciled[0]["resolution_status"] == "resolved_nonblocking_observation"
    assert reconciled[1]["resolution_status"] == "resolved_preexisting_phase2_precision_envelope"
    assert summary["resolved_qform_sform_observation_case_count"] == 1
    assert summary["resolved_effective_affine_case_count"] == 1
    assert summary["unresolved_mismatch_record_count"] == 0
    assert summary["source_arrays_opened_by_reconciliation"] == 0


def test_geometry_outside_approved_source_tolerance_remains_unresolved() -> None:
    records = [_record(translation_mm=2.0e-4)]
    mismatches = [
        {
            "anonymous_case_id": "case-fixture",
            "partition": "train",
            "reason": "image_label_affine_mismatch",
            "resolution_status": "unresolved",
            "resolution": None,
        }
    ]

    reconciled, summary = reconcile_geometry_mismatches(records=records, mismatches=mismatches)

    assert reconciled[0]["resolution_status"] == "unresolved"
    assert summary["unresolved_mismatch_record_count"] == 1


def test_unknown_mismatch_reason_is_never_auto_resolved() -> None:
    records = [_record()]
    mismatches = [
        {
            "anonymous_case_id": "case-fixture",
            "partition": "train",
            "reason": "unexpected_future_reason",
            "resolution_status": "unresolved",
            "resolution": None,
        }
    ]

    reconciled, summary = reconcile_geometry_mismatches(records=records, mismatches=mismatches)

    assert reconciled[0]["resolution_status"] == "unresolved"
    assert summary["unresolved_mismatch_record_count"] == 1
