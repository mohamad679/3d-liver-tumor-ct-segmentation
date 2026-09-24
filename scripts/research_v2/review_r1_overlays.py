#!/usr/bin/env python3
"""Record human review of local-only R1 tri-planar overlays.

This helper does not open medical arrays. It only verifies the already-rendered PNG hashes and
records the reviewer's alignment decision for each anonymous train case.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence_dir", type=Path)
    args = parser.parse_args()
    root = args.evidence_dir.expanduser().resolve()
    review_path = root / "r1_overlay_review.json"
    overlays_dir = root / "overlays_local_only"
    entries: list[dict[str, Any]] = json.loads(review_path.read_text(encoding="utf-8"))
    if len(entries) < 10:
        raise RuntimeError("R1 requires at least 10 train overlays")

    completed: list[dict[str, Any]] = []
    print("Open each PNG locally and inspect CT/tumor alignment in all three panels.")
    print("Answer y only when the overlay is anatomically aligned; n records a blocking mismatch.")
    for entry in entries:
        overlay = overlays_dir / str(entry["overlay_filename"])
        if not overlay.is_file():
            raise RuntimeError(f"overlay file missing: {overlay.name}")
        observed_hash = _sha256(overlay)
        if observed_hash != entry["overlay_sha256"]:
            raise RuntimeError(f"overlay hash changed: {overlay.name}")
        while True:
            answer = input(
                f"{entry['anonymous_case_id']} ({overlay.name}) aligned in axial/sagittal/coronal? "
                "[y/n]: "
            ).strip().lower()
            if answer in {"y", "n"}:
                break
        notes = input("optional note (press Enter for none): ").strip()
        completed.append(
            {
                **entry,
                "alignment_confirmed": answer == "y",
                "notes": notes or None,
            }
        )

    output = root / "r1_overlay_review.completed.json"
    output.write_text(json.dumps(completed, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    all_confirmed = all(bool(entry["alignment_confirmed"]) for entry in completed)
    summary = {
        "schema_version": "research_v2_r1_overlay_review.v1",
        "reviewed_count": len(completed),
        "all_alignment_confirmed": all_confirmed,
        "review_artifact_sha256": _sha256(output),
        "blocking_case_ids": [
            entry["anonymous_case_id"]
            for entry in completed
            if not bool(entry["alignment_confirmed"])
        ],
    }
    summary_path = root / "r1_overlay_review_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if all_confirmed else 40


if __name__ == "__main__":
    raise SystemExit(main())
