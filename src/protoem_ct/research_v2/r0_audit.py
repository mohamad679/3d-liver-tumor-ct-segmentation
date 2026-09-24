"""Materialize Research-v2 R0 verification evidence from the real Phase 2 artifacts."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from protoem_ct.research_v2.protocol import (
    PartitionIsolationError,
    assert_protocol_artifact_lock,
    audit_partition_isolation,
    build_nnunet_splits,
    canonical_json_sha256,
    load_protocol,
    load_verified_phase2_artifacts,
)


def run_r0_artifact_audit(
    *,
    protocol_path: Path,
    manifest_path: Path,
    split_path: Path,
    output_dir: Path,
    require_protocol_lock: bool,
) -> dict[str, Any]:
    """Verify real artifacts and write deterministic identity, isolation, and fold evidence."""

    protocol = load_protocol(protocol_path)
    verified = load_verified_phase2_artifacts(manifest_path, split_path)
    if require_protocol_lock:
        assert_protocol_artifact_lock(protocol, verified.identity)

    isolation = audit_partition_isolation(
        protocol,
        manifest=verified.manifest,
        split=verified.split,
    )
    folds = build_nnunet_splits(verified.split, fold_count=protocol.nnunet_fold_count)
    _verify_fold_coverage(folds, expected_train_case_count=protocol.train_count)

    identity_payload = {
        "artifact_identity": asdict(verified.identity),
        "protocol_id": protocol.protocol_id,
        "protocol_lock_required": require_protocol_lock,
        "protocol_lock_status": "PASS" if require_protocol_lock else "DISCOVERY_ONLY",
    }
    isolation_payload = {
        "audit_hash": isolation.audit_hash,
        "case_counts": dict(isolation.case_counts),
        "manifest_artifact_hash": isolation.manifest_hash,
        "patient_counts": dict(isolation.patient_counts),
        "protocol_id": protocol.protocol_id,
        "split_artifact_hash": isolation.split_hash,
        "status": "PASS",
    }
    folds_payload = {
        "description": "nnU-Net folds derived only from the verified 91-case development train",
        "fold_count": protocol.nnunet_fold_count,
        "folds": folds,
        "protocol_id": protocol.protocol_id,
        "source_split_artifact_hash": verified.identity.split_artifact_hash,
    }
    folds_payload["folds_sha256"] = canonical_json_sha256(folds_payload["folds"])

    _write_json(output_dir / "artifact_identity.json", identity_payload)
    _write_json(output_dir / "partition_isolation_audit.json", isolation_payload)
    _write_json(output_dir / "splits_final.json", folds_payload)
    return {
        "artifact_identity": identity_payload,
        "partition_isolation": isolation_payload,
        "splits_final": folds_payload,
    }


def _verify_fold_coverage(
    folds: list[dict[str, list[str]]],
    *,
    expected_train_case_count: int,
) -> None:
    validation_cases = [case_id for fold in folds for case_id in fold["val"]]
    if len(validation_cases) != expected_train_case_count:
        raise PartitionIsolationError("fold validation coverage does not match train case count")
    if len(set(validation_cases)) != expected_train_case_count:
        raise PartitionIsolationError("a train case appears in more than one fold validation set")
    for fold in folds:
        train_cases = set(fold["train"])
        val_cases = set(fold["val"])
        if train_cases & val_cases:
            raise PartitionIsolationError("fold train and validation case sets overlap")
        if len(train_cases | val_cases) != expected_train_case_count:
            raise PartitionIsolationError("fold does not cover every development-train case")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require-protocol-lock", action="store_true")
    return parser


def main() -> int:
    """CLI entry point."""

    args = _parser().parse_args()
    run_r0_artifact_audit(
        protocol_path=args.protocol,
        manifest_path=args.manifest,
        split_path=args.split,
        output_dir=args.output_dir,
        require_protocol_lock=args.require_protocol_lock,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
