#!/usr/bin/env python3
"""Reconcile R1 geometry mismatch records without reopening medical arrays."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from protoem_ct.research_v2.r1_geometry_reconciliation import reconcile_geometry_mismatches


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_list(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise RuntimeError(f"expected JSON list of objects: {path.name}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audit_dir", type=Path)
    args = parser.parse_args()

    root = args.audit_dir.expanduser().resolve()
    records_path = root / "r1_case_records_no_paths.json"
    mismatches_path = root / "r1_mismatches.json"
    summary_path = root / "r1_real_audit_summary.json"
    for path in (records_path, mismatches_path, summary_path):
        if not path.is_file():
            raise RuntimeError(f"required R1 evidence file missing: {path.name}")

    audit_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(audit_summary, dict):
        raise RuntimeError("r1_real_audit_summary.json must contain one JSON object")

    array_access = audit_summary.get("array_access", {})
    expected_access = {
        "attempted_train_cases": 91,
        "attempted_validation_cases": 20,
        "successfully_opened_train_cases": 91,
        "successfully_opened_validation_cases": 20,
        "unique_authorized_case_arrays_opened": 111,
        "internal_test_arrays_opened": 0,
        "external_arrays_opened": 0,
    }
    for key, expected in expected_access.items():
        if int(array_access.get(key, -1)) != expected:
            raise RuntimeError(
                f"cannot reconcile geometry: array-access evidence {key}="
                f"{array_access.get(key)!r}, expected {expected}"
            )

    if audit_summary.get("transform_trace_failures") not in ([], None):
        raise RuntimeError("cannot reconcile geometry while transform trace failures exist")

    records = _load_list(records_path)
    mismatches = _load_list(mismatches_path)
    if len(records) != 111:
        raise RuntimeError(f"expected 111 sanitized case records, observed {len(records)}")

    reconciled, reconciliation = reconcile_geometry_mismatches(
        records=records,
        mismatches=mismatches,
    )
    reconciliation["source_evidence"] = {
        "case_records_sha256": _sha256(records_path),
        "mismatches_sha256": _sha256(mismatches_path),
        "real_audit_summary_sha256": _sha256(summary_path),
    }
    reconciliation["original_real_audit_status"] = audit_summary.get("status")
    reconciliation["original_unresolved_mismatch_count"] = audit_summary.get(
        "unresolved_mismatch_count"
    )
    reconciliation["internal_test_arrays_opened"] = int(
        array_access["internal_test_arrays_opened"]
    )
    reconciliation["external_arrays_opened"] = int(array_access["external_arrays_opened"])

    reconciled_path = root / "r1_mismatches.reconciled.json"
    reconciliation_path = root / "r1_geometry_reconciliation.json"
    reconciled_path.write_text(
        json.dumps(reconciled, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    reconciliation_path.write_text(
        json.dumps(reconciliation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    unresolved = int(reconciliation["unresolved_mismatch_record_count"])
    print("===== R1 GEOMETRY RECONCILIATION =====")
    print("source_arrays_opened_by_reconciliation=0")
    print(f"original_mismatch_record_count={len(mismatches)}")
    print(
        "resolved_qform_sform_observation_case_count="
        f"{reconciliation['resolved_qform_sform_observation_case_count']}"
    )
    print(
        "resolved_effective_affine_case_count="
        f"{reconciliation['resolved_effective_affine_case_count']}"
    )
    print(f"unresolved_mismatch_record_count={unresolved}")
    print(f"internal_test_arrays_opened={reconciliation['internal_test_arrays_opened']}")
    print(f"external_arrays_opened={reconciliation['external_arrays_opened']}")
    print(f"RECONCILIATION_JSON={reconciliation_path}")
    print(f"RECONCILED_MISMATCHES_JSON={reconciled_path}")
    if unresolved:
        print("R1_GEOMETRY_RECONCILIATION=BLOCKED")
        return 40
    print("R1_GEOMETRY_RECONCILIATION=PASS")
    print("NEXT=LOCAL_OVERLAY_REVIEW")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
