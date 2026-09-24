#!/usr/bin/env python3
"""Prepare and hash the deterministic train-only R2 overfit subset.

This script does not decode NIfTI arrays. It reads the locked manifest/split and sanitized R1
metadata, verifies only the selected train files by SHA-256, and optionally copies only those files
to a private staging directory for later Kaggle transfer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from protoem_ct.artifacts import (
    DatasetManifest,
    DevelopmentSplitManifest,
    hash_dataset_manifest,
    hash_development_split,
    phase2_artifact_from_json,
    sha256_file,
)
from protoem_ct.data.phase2_paths import resolve_regular_file_beneath_root
from protoem_ct.research_v2.r2_selection import (
    assert_closed_r1_expected_selection,
    select_r2_train_cases,
)

EXPECTED_MANIFEST_HASH = "c24244951e050050cf25c4b321f67d61c2087fc0c93fdcf9d112e0e488e1384b"
EXPECTED_MANIFEST_FILE_SHA256 = "0e21a7d555e20b6011091bc18e462bc150cc23a8f46e522dbd8c01b014a44ae3"
EXPECTED_SPLIT_HASH = "936376cd7b5e6070397c2fef16e5125c60fd6569ff3188d7e9bb5428a46ffadb"
EXPECTED_SPLIT_FILE_SHA256 = "416ca83e85c8598fc4f7065316153193211bf6b57875fbda4cafc01c360445f7"
DEFAULT_MANIFEST = Path("/Volumes/Lexar/ProtoEM-CT/runs/phase2_real_lits_v2/manifest.json")
DEFAULT_SPLIT = Path("/Volumes/Lexar/ProtoEM-CT/runs/phase2_real_lits_v2/split.json")
DEFAULT_SEARCH_ROOT = Path("/Volumes/Lexar/ProtoEM-CT")


def _load_locked_artifacts(
    manifest_path: Path, split_path: Path
) -> tuple[DatasetManifest, DevelopmentSplitManifest]:
    if sha256_file(manifest_path) != EXPECTED_MANIFEST_FILE_SHA256:
        raise RuntimeError("manifest raw SHA-256 does not match the R0 lock")
    if sha256_file(split_path) != EXPECTED_SPLIT_FILE_SHA256:
        raise RuntimeError("split raw SHA-256 does not match the R0 lock")
    manifest = phase2_artifact_from_json(manifest_path.read_text(encoding="utf-8"))
    split = phase2_artifact_from_json(split_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, DatasetManifest):
        raise RuntimeError("locked manifest did not parse as DatasetManifest")
    if not isinstance(split, DevelopmentSplitManifest):
        raise RuntimeError("locked split did not parse as DevelopmentSplitManifest")
    if manifest.manifest_hash != EXPECTED_MANIFEST_HASH:
        raise RuntimeError("manifest logical hash differs from the R0 lock")
    if split.split_hash != EXPECTED_SPLIT_HASH:
        raise RuntimeError("split logical hash differs from the R0 lock")
    if hash_dataset_manifest(manifest) != EXPECTED_MANIFEST_HASH:
        raise RuntimeError("independent manifest logical-hash recomputation failed")
    if hash_development_split(split) != EXPECTED_SPLIT_HASH:
        raise RuntimeError("independent split logical-hash recomputation failed")
    if split.source_manifest_hash != manifest.manifest_hash:
        raise RuntimeError("split source_manifest_hash differs from the locked manifest")
    return manifest, split


def _candidate_root_for_match(match: Path, relative_path: str) -> Path:
    candidate = match
    for _ in Path(relative_path).parts:
        candidate = candidate.parent
    return candidate


def _detect_dataset_root(search_root: Path, relative_path: str) -> Path:
    if not search_root.is_dir():
        raise RuntimeError(f"dataset search root does not exist: {search_root}")
    basename = Path(relative_path).name
    candidates: set[Path] = set()
    for match in search_root.rglob(basename):
        if not match.is_file():
            continue
        candidate = _candidate_root_for_match(match, relative_path)
        try:
            expected = (candidate / relative_path).resolve(strict=True)
            resolved_match = match.resolve(strict=True)
        except OSError:
            continue
        if expected == resolved_match:
            candidates.add(candidate.resolve())
    if len(candidates) != 1:
        raise RuntimeError(
            "could not uniquely auto-detect the dataset root from the selected train file; "
            f"candidates={[str(path) for path in sorted(candidates)]}"
        )
    return next(iter(candidates))


def _load_r1_records(path: Path) -> list[dict[str, Any]]:
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, list) or not all(isinstance(item, dict) for item in parsed):
        raise RuntimeError("R1 case records must be a JSON list of objects")
    return parsed


def _pair_identity(image_sha256: str, label_sha256: str) -> str:
    payload = f"{image_sha256}:{label_sha256}".encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _private_copy(source: Path, destination: Path, expected_sha256: str) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise RuntimeError(f"refusing to overwrite private staged file: {destination}")
    shutil.copy2(source, destination)
    observed = sha256_file(destination)
    if observed != expected_sha256:
        raise RuntimeError(f"staged file SHA-256 mismatch: {destination.name}")
    return observed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r1-case-records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--search-root", type=Path, default=DEFAULT_SEARCH_ROOT)
    parser.add_argument("--private-staging-dir", type=Path, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = output_dir / "r2_selected_train_cases.json"
    if evidence_path.exists():
        raise RuntimeError(f"refusing to overwrite existing evidence: {evidence_path}")

    manifest, split = _load_locked_artifacts(args.manifest, args.split)
    r1_records = _load_r1_records(args.r1_case_records)
    selected = select_r2_train_cases(r1_records)
    assert_closed_r1_expected_selection(selected)

    assignments = {item.anonymous_case_id: item for item in split.assignments}
    manifest_cases = {item.anonymous_case_id: item for item in manifest.cases}
    for selected_case in selected:
        assignment = assignments.get(selected_case.anonymous_case_id)
        if assignment is None or assignment.partition != "train":
            raise RuntimeError("selected R2 case is not assigned to train in the locked split")
        if assignment.anonymous_patient_id != selected_case.anonymous_patient_id:
            raise RuntimeError("selected R2 patient identity differs from the locked split")
        manifest_case = manifest_cases.get(selected_case.anonymous_case_id)
        if manifest_case is None:
            raise RuntimeError("selected R2 case is absent from the locked manifest")
        if manifest_case.anonymous_patient_id != selected_case.anonymous_patient_id:
            raise RuntimeError("selected R2 patient identity differs from the locked manifest")

    first_manifest_case = manifest_cases[selected[0].anonymous_case_id]
    dataset_root = (
        args.dataset_root.expanduser().resolve()
        if args.dataset_root is not None
        else _detect_dataset_root(
            args.search_root.expanduser().resolve(),
            first_manifest_case.relative_image_path,
        )
    )

    staging_root = (
        args.private_staging_dir.expanduser().resolve()
        if args.private_staging_dir is not None
        else None
    )
    if staging_root is not None:
        staging_root.mkdir(parents=True, exist_ok=True)

    case_evidence: list[dict[str, Any]] = []
    for selected_case in selected:
        manifest_case = manifest_cases[selected_case.anonymous_case_id]
        image_path = resolve_regular_file_beneath_root(
            dataset_root, manifest_case.relative_image_path
        )
        label_path = resolve_regular_file_beneath_root(
            dataset_root, manifest_case.relative_label_path
        )
        observed_image_sha256 = sha256_file(image_path)
        observed_label_sha256 = sha256_file(label_path)
        if observed_image_sha256 != manifest_case.image_sha256:
            raise RuntimeError("selected train image SHA-256 changed from the locked manifest")
        if observed_label_sha256 != manifest_case.label_sha256:
            raise RuntimeError("selected train label SHA-256 changed from the locked manifest")

        staged_image_sha256: str | None = None
        staged_label_sha256: str | None = None
        staged_image_relative_path: str | None = None
        staged_label_relative_path: str | None = None
        if staging_root is not None:
            case_root = staging_root / selected_case.anonymous_case_id
            staged_image = case_root / "image.nii.gz"
            staged_label = case_root / "label.nii.gz"
            staged_image_sha256 = _private_copy(
                image_path, staged_image, manifest_case.image_sha256
            )
            staged_label_sha256 = _private_copy(
                label_path, staged_label, manifest_case.label_sha256
            )
            staged_image_relative_path = str(staged_image.relative_to(staging_root))
            staged_label_relative_path = str(staged_label.relative_to(staging_root))

        case_evidence.append(
            {
                "role": selected_case.role,
                "anonymous_case_id": selected_case.anonymous_case_id,
                "anonymous_patient_id": selected_case.anonymous_patient_id,
                "partition": "train",
                "tumor_voxel_count": selected_case.tumor_voxel_count,
                "tumor_volume_mm3": selected_case.tumor_volume_mm3,
                "tumor_lesion_count_26c": selected_case.tumor_lesion_count_26c,
                "source_relative_image_path": manifest_case.relative_image_path,
                "source_relative_label_path": manifest_case.relative_label_path,
                "image_sha256": manifest_case.image_sha256,
                "label_sha256": manifest_case.label_sha256,
                "image_label_pair_identity_sha256": _pair_identity(
                    manifest_case.image_sha256, manifest_case.label_sha256
                ),
                "source_hash_verified": True,
                "staged_image_relative_path": staged_image_relative_path,
                "staged_label_relative_path": staged_label_relative_path,
                "staged_image_sha256": staged_image_sha256,
                "staged_label_sha256": staged_label_sha256,
                "staged_hash_verified": staging_root is not None,
            }
        )

    evidence = {
        "schema_version": "research_v2_r2_selected_train_cases.v1",
        "status": "PASS",
        "selection_source": "closed R1 sanitized metadata; train partition only",
        "selection_rule": (
            "positive q20/q50/q80 rank by tumor_voxel_count with case-id tie-break; "
            "lexicographically first train-empty case"
        ),
        "manifest_artifact_hash": EXPECTED_MANIFEST_HASH,
        "manifest_file_sha256": EXPECTED_MANIFEST_FILE_SHA256,
        "split_artifact_hash": EXPECTED_SPLIT_HASH,
        "split_file_sha256": EXPECTED_SPLIT_FILE_SHA256,
        "selected_case_count": len(case_evidence),
        "selected_cases": case_evidence,
        "medical_arrays_decoded_by_selection_script": 0,
        "validation_arrays_opened": 0,
        "internal_test_arrays_opened": 0,
        "external_arrays_opened": 0,
        "private_staging_performed": staging_root is not None,
        "private_staging_file_count": len(case_evidence) * 2 if staging_root is not None else 0,
        "warning": (
            "Selected images/labels and any private staging directory contain medical data and "
            "must not be committed to Git or published."
        ),
    }
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(evidence, indent=2, sort_keys=True))
    print(f"R2_SELECTION_EVIDENCE={evidence_path}")
    if staging_root is not None:
        print(f"R2_PRIVATE_STAGING_DIR={staging_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
