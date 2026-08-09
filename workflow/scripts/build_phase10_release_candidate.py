"""Build Phase 10 release-candidate metadata from final package files."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from build_phase10_artifact_inventory import sha256_file, sha256_json

SCHEMA_VERSION = "v1"


class Phase10ReleaseError(RuntimeError):
    """Raised when release-candidate inputs are missing."""


def git_output(args: list[str]) -> str:
    """Return stripped Git command output."""
    return subprocess.check_output(["git", *args], text=True).strip()


def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object."""
    if not path.exists():
        raise Phase10ReleaseError(f"Missing release input: {path}")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise Phase10ReleaseError(f"Expected JSON object: {path}")
    return loaded


def write_json(path: Path, payload: dict[str, Any]) -> str:
    """Write deterministic JSON with self-hash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    full_payload = dict(payload)
    full_payload["self_hash"] = sha256_json(full_payload)
    path.write_text(json.dumps(full_payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return str(full_payload["self_hash"])


def tracked_file_record(path: Path) -> dict[str, str]:
    """Return a release manifest record for one file."""
    if not path.exists():
        raise Phase10ReleaseError(f"Missing release file: {path}")
    return {"path": str(path), "sha256": sha256_file(path)}


def build_release_candidate(output: Path) -> dict[str, Any]:
    """Build release-candidate metadata."""
    inventory = load_json(Path("reports/phase10/artifact_inventory.json"))
    b_manifest = load_json(Path("reports/phase10/phase10_b_outputs_manifest.json"))
    report_manifest = load_json(Path("reports/phase10/final_report_manifest.json"))
    files = [
        Path("reports/final_report.md"),
        Path("reports/final_report.html"),
        Path("reports/phase10/artifact_inventory.json"),
        Path("reports/phase10/phase10_b_outputs_manifest.json"),
        Path("reports/phase10/final_report_manifest.json"),
        Path("docs/PHASE10_REPRODUCTION.md"),
        Path("MODEL_CARD.md"),
        Path("DATA_CARD.md"),
        Path("docs/PHASE10_CV_PROJECT_SUMMARY.md"),
        Path("docs/PHASE10_MOTIVATION_EVIDENCE_PARAGRAPH.md"),
        Path("docs/PHASE10_RELEASE_CHECKLIST.md"),
        Path("README.md"),
        Path("ACCEPTANCE_CHECKLIST.md"),
        Path("docs/PHASE_PLAN.md"),
        Path("docs/DECISIONS.md"),
    ]
    payload = {
        "schema_name": "phase10_release_candidate",
        "schema_version": SCHEMA_VERSION,
        "source_package_commit": git_output(["rev-parse", "HEAD"]),
        "release_candidate_containing_commit_resolution": (
            "git log -1 --format=%H -- reports/phase10/release_candidate.json"
        ),
        "git_commit_identity_policy": (
            "The commit containing this tracked metadata file cannot be embedded inside the "
            "same file without changing the commit hash. Resolve the containing commit with "
            "release_candidate_containing_commit_resolution."
        ),
        "git_branch": git_output(["branch", "--show-current"]),
        "artifact_inventory_hash": inventory["self_hash"],
        "phase10_b_outputs_manifest_hash": b_manifest["self_hash"],
        "final_report_manifest_hash": report_manifest["self_hash"],
        "final_report_html_sha256": report_manifest["report_html_sha256"],
        "phase9_status": "skipped_by_user_decision_not_executed",
        "phase10_status": "complete_p10r_pass_gate8_pass_no_push",
        "gate8_status": "pass_final_report_generated_from_machine_readable_artifacts",
        "training_rerun": False,
        "inference_rerun": False,
        "external_label_tuning": False,
        "raw_medical_data_committed": False,
        "checkpoints_or_weights_committed": False,
        "predictions_committed": False,
        "secrets_committed": False,
        "lits_msd_policy": "single_lits_derived_development_source",
        "external_validation_interpretation": "negative_result_no_strong_generalization_claim",
        "files": [tracked_file_record(path) for path in files],
    }
    self_hash = write_json(output, payload)
    payload["self_hash"] = self_hash
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("reports/phase10/release_candidate.json")
    )
    args = parser.parse_args()
    payload = build_release_candidate(args.output)
    print(f"Wrote {args.output} (self_hash={payload['self_hash']})")


if __name__ == "__main__":
    main()
