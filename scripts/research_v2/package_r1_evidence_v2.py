#!/usr/bin/env python3
"""Validate and package reconciled sanitized Research-v2 R1 evidence.

The archive contains JSON evidence only. Raw NIfTI arrays and local tri-planar PNG overlays are
explicitly excluded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

EXPECTED_MANIFEST_HASH = "c24244951e050050cf25c4b321f67d61c2087fc0c93fdcf9d112e0e488e1384b"
EXPECTED_SPLIT_HASH = "936376cd7b5e6070397c2fef16e5125c60fd6569ff3188d7e9bb5428a46ffadb"
EXPECTED_TRAIN = 91
EXPECTED_VALIDATION = 20
EXPECTED_OVERLAYS = 10

REQUIRED_JSON_FILES = (
    "r1_case_records_no_paths.json",
    "r1_mismatches.json",
    "r1_mismatches.reconciled.json",
    "r1_geometry_reconciliation.json",
    "r1_transform_traces.json",
    "r1_overlay_review.json",
    "r1_real_audit_summary.json",
    "r1_evidence_hashes.json",
    "r1_overlay_review.completed.json",
    "r1_overlay_review_summary.json",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    root = args.evidence_dir.expanduser().resolve()
    if not root.is_dir():
        raise RuntimeError(f"evidence directory does not exist: {root}")
    missing = [name for name in REQUIRED_JSON_FILES if not (root / name).is_file()]
    if missing:
        raise RuntimeError(f"required sanitized evidence files are missing: {missing}")

    summary = _load_json(root / "r1_real_audit_summary.json")
    reconciliation = _load_json(root / "r1_geometry_reconciliation.json")
    overlay_summary = _load_json(root / "r1_overlay_review_summary.json")
    reconciled_mismatches = _load_json(root / "r1_mismatches.reconciled.json")
    traces = _load_json(root / "r1_transform_traces.json")
    hashes = _load_json(root / "r1_evidence_hashes.json")

    if summary.get("manifest_artifact_hash") != EXPECTED_MANIFEST_HASH:
        raise RuntimeError("manifest logical hash differs from the R0 lock")
    if summary.get("split_artifact_hash") != EXPECTED_SPLIT_HASH:
        raise RuntimeError("split logical hash differs from the R0 lock")

    access = summary.get("array_access")
    if not isinstance(access, dict):
        raise RuntimeError("array_access summary is missing")
    required_access = {
        "attempted_train_cases": EXPECTED_TRAIN,
        "attempted_validation_cases": EXPECTED_VALIDATION,
        "successfully_opened_train_cases": EXPECTED_TRAIN,
        "successfully_opened_validation_cases": EXPECTED_VALIDATION,
        "unique_authorized_case_arrays_opened": EXPECTED_TRAIN + EXPECTED_VALIDATION,
        "internal_test_arrays_opened": 0,
        "external_arrays_opened": 0,
    }
    for key, expected in required_access.items():
        if access.get(key) != expected:
            raise RuntimeError(f"array access evidence mismatch: {key}={access.get(key)!r}")

    if summary.get("transform_trace_failures") != []:
        raise RuntimeError("R1 has unresolved transform-trace failures")
    if summary.get("overlay_train_case_count") != EXPECTED_OVERLAYS:
        raise RuntimeError("R1 requires exactly 10 selected train overlays")
    if summary.get("training_performed") is not False:
        raise RuntimeError("R1 evidence unexpectedly reports training")
    if summary.get("threshold_tuning_performed") is not False:
        raise RuntimeError("R1 evidence unexpectedly reports threshold tuning")

    if reconciliation.get("source_arrays_opened_by_reconciliation") != 0:
        raise RuntimeError("geometry reconciliation unexpectedly opened source arrays")
    if reconciliation.get("unresolved_mismatch_record_count") != 0:
        raise RuntimeError("R1 geometry reconciliation still has unresolved mismatches")
    if reconciliation.get("internal_test_arrays_opened") != 0:
        raise RuntimeError("R1 reconciliation evidence reports internal-test array access")
    if reconciliation.get("external_arrays_opened") != 0:
        raise RuntimeError("R1 reconciliation evidence reports external array access")
    if reconciliation.get("original_mismatch_record_count") != len(_load_json(root / "r1_mismatches.json")):
        raise RuntimeError("geometry reconciliation does not link to the original mismatch artifact")

    unresolved = [
        item
        for item in reconciled_mismatches
        if not isinstance(item, dict)
        or str(item.get("resolution_status", "")).startswith("unresolved")
    ]
    if unresolved:
        raise RuntimeError("reconciled mismatch artifact still contains unresolved entries")

    if overlay_summary.get("reviewed_count") != EXPECTED_OVERLAYS:
        raise RuntimeError("human overlay review count is not exactly 10")
    if overlay_summary.get("all_alignment_confirmed") is not True:
        raise RuntimeError("one or more local overlays failed human alignment review")
    if overlay_summary.get("blocking_case_ids") != []:
        raise RuntimeError("overlay review contains blocking cases")
    if len(traces) != EXPECTED_OVERLAYS:
        raise RuntimeError("transform trace count must equal overlay case count")

    if isinstance(hashes, dict):
        for filename, expected_hash in hashes.items():
            path = root / str(filename)
            if path.is_file() and _sha256(path) != expected_hash:
                raise RuntimeError(f"original evidence file hash changed after audit: {filename}")

    handoff = {
        "schema_version": "research_v2_r1_handoff.v2",
        "status": "READY_FOR_GATE_REVIEW",
        "manifest_artifact_hash": EXPECTED_MANIFEST_HASH,
        "split_artifact_hash": EXPECTED_SPLIT_HASH,
        "audited_train_cases": EXPECTED_TRAIN,
        "audited_validation_cases": EXPECTED_VALIDATION,
        "internal_test_arrays_opened": 0,
        "external_arrays_opened": 0,
        "original_mismatch_record_count": reconciliation["original_mismatch_record_count"],
        "unresolved_mismatch_count_after_reconciliation": 0,
        "reviewed_train_overlays": EXPECTED_OVERLAYS,
        "all_overlay_alignment_confirmed": True,
        "training_performed": False,
        "threshold_tuning_performed": False,
        "medical_images_in_handoff": False,
        "raw_labels_in_handoff": False,
        "overlay_pngs_in_handoff": False,
    }
    handoff_path = root / "r1_handoff_summary.v2.json"
    handoff_path.write_text(json.dumps(handoff, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else root / "research_v2_r1_sanitized_evidence_v2.zip"
    )
    if output.exists():
        raise RuntimeError(f"refusing to overwrite existing evidence archive: {output}")

    archive_files = (*REQUIRED_JSON_FILES, "r1_handoff_summary.v2.json")
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for filename in archive_files:
            archive.write(root / filename, arcname=filename)

    result = {
        **handoff,
        "archive_filename": output.name,
        "archive_sha256": _sha256(output),
        "archive_size_bytes": output.stat().st_size,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"R1_SANITIZED_EVIDENCE_ZIP={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
