"""Synthetic tests for guarded Phase 8 Wave 4 readiness publication."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from protoem_ct.artifacts.hashing import canonical_json_bytes
from protoem_ct.external.freeze import (
    PHASE8_DECISION_FREEZE_FILENAME,
    PHASE8_EXTERNAL_PREREGISTRATION_FILENAME,
    PHASE8_READINESS_FILENAME,
    PHASE8_WAVE4_SUMMARY_FILENAME,
    Phase8FreezeError,
    run_phase8_wave4_readiness_publication,
)
from protoem_ct.external.manifest import (
    PHASE8_EXTERNAL_IMAGE_CASE_SCHEMA_NAME,
    PHASE8_EXTERNAL_IMAGE_CASE_SCHEMA_VERSION,
    Phase8ExternalImageCase,
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
)


def test_wave4_blocked_readiness_writes_only_allowed_artifacts(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    wave2 = _write_wave2(tmp_path / "wave2")
    wave3 = _write_wave3(tmp_path / "wave3")
    internal = tmp_path / "runs"
    internal.mkdir()

    result = run_phase8_wave4_readiness_publication(
        wave2_artifact_root=wave2,
        wave3_artifact_root=wave3,
        internal_artifact_parent=internal,
        output_root=tmp_path / "wave4",
        repository_root=repo,
    )

    assert result.wave4_state == "BLOCKED"
    assert result.freeze_generated is False
    assert result.preregistration_generated is False
    assert sorted(path.name for path in (tmp_path / "wave4").iterdir()) == [
        PHASE8_READINESS_FILENAME,
        PHASE8_WAVE4_SUMMARY_FILENAME,
    ]
    assert not (tmp_path / "wave4" / PHASE8_DECISION_FREEZE_FILENAME).exists()
    assert not (tmp_path / "wave4" / PHASE8_EXTERNAL_PREREGISTRATION_FILENAME).exists()

    readiness = json.loads((tmp_path / "wave4" / PHASE8_READINESS_FILENAME).read_text())
    assert set(readiness["category_readiness"]) == {
        "bootstrap_configuration",
        "checkpoint_metadata",
        "label_mapping_policy",
        "metric_configuration",
        "model_selection_decision",
        "preprocessing_decision",
        "publication_configuration",
        "support_policy",
        "threshold_decision",
    }
    assert (
        readiness["category_readiness"]["label_mapping_policy"][
            "legally_scientifically_freezeable_now"
        ]
        is True
    )
    assert (
        readiness["category_readiness"]["checkpoint_metadata"][
            "legally_scientifically_freezeable_now"
        ]
        is False
    )
    assert "real trained checkpoint metadata" in " ".join(readiness["unresolved_blockers"])
    assert readiness["external_data_access"]["external_labels_read"] is False


def test_wave4_synthetic_checkpoint_file_does_not_unblock_readiness(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    wave2 = _write_wave2(tmp_path / "wave2")
    wave3 = _write_wave3(tmp_path / "wave3")
    internal = tmp_path / "runs"
    internal.mkdir()
    (internal / "synthetic_smoke.ckpt").write_bytes(b"not a real checkpoint")

    result = run_phase8_wave4_readiness_publication(
        wave2_artifact_root=wave2,
        wave3_artifact_root=wave3,
        internal_artifact_parent=internal,
        output_root=tmp_path / "wave4",
        repository_root=repo,
    )

    readiness = json.loads((tmp_path / "wave4" / PHASE8_READINESS_FILENAME).read_text())
    assert result.wave4_state == "BLOCKED"
    assert readiness["checkpoint_candidate_count"] == 1
    assert (
        readiness["category_readiness"]["checkpoint_metadata"][
            "legally_scientifically_freezeable_now"
        ]
        is False
    )


def test_wave4_rejects_repository_output_root(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    wave2 = _write_wave2(tmp_path / "wave2")
    wave3 = _write_wave3(tmp_path / "wave3")
    internal = tmp_path / "runs"
    internal.mkdir()

    with pytest.raises(Phase8FreezeError):
        run_phase8_wave4_readiness_publication(
            wave2_artifact_root=wave2,
            wave3_artifact_root=wave3,
            internal_artifact_parent=internal,
            output_root=repo / "wave4",
            repository_root=repo,
        )


def test_wave4_json_is_deterministic_across_roots(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    wave2 = _write_wave2(tmp_path / "wave2")
    wave3 = _write_wave3(tmp_path / "wave3")
    internal = tmp_path / "runs"
    internal.mkdir()

    run_phase8_wave4_readiness_publication(
        wave2_artifact_root=wave2,
        wave3_artifact_root=wave3,
        internal_artifact_parent=internal,
        output_root=tmp_path / "wave4a",
        repository_root=repo,
    )
    run_phase8_wave4_readiness_publication(
        wave2_artifact_root=wave2,
        wave3_artifact_root=wave3,
        internal_artifact_parent=internal,
        output_root=tmp_path / "wave4b",
        repository_root=repo,
    )

    for filename in (PHASE8_READINESS_FILENAME, PHASE8_WAVE4_SUMMARY_FILENAME):
        assert (tmp_path / "wave4a" / filename).read_bytes() == (
            tmp_path / "wave4b" / filename
        ).read_bytes()


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    config_dir = repo / "configs"
    config_dir.mkdir()
    (config_dir / "phase6_protoem_ct.yaml").write_text("synthetic_mode_only: true\n")
    return repo


def _write_wave2(root: Path) -> Path:
    root.mkdir()
    cases = tuple(_manifest_case(index) for index in range(1, 21))
    manifest = build_phase8_external_image_manifest(
        dataset_archive_sha256="a" * 64,
        dataset_archive_size_bytes=123456,
        adapter_name="phase8_ircadb_image_layout",
        adapter_version="v1",
        cases=cases,
    )
    (root / PHASE8_WAVE2_MANIFEST_FILENAME).write_bytes(
        phase8_external_image_manifest_to_json(manifest)
    )
    _write_json(
        root / PHASE8_WAVE2_QA_FILENAME,
        {"case_count": 20, "schema_name": "phase8_external_image_qa_collection"},
    )
    _write_json(
        root / PHASE8_WAVE2_LAYOUT_FILENAME,
        {"case_count": 20, "schema_name": "phase8_ircadb_image_layout"},
    )
    _write_json(
        root / PHASE8_WAVE2_SUMMARY_FILENAME,
        {"case_count": 20, "label_accessed": False, "manifest_hash": manifest.manifest_hash},
    )
    return root


def _write_wave3(root: Path) -> Path:
    root.mkdir()
    for filename, schema_name in (
        (PHASE8_WAVE3_COHORT_ACCOUNTING_FILENAME, "phase8_eligibility_accounting"),
        (PHASE8_WAVE3_DOMAIN_SHIFT_RECORD_FILENAME, "phase8_domain_shift_record"),
        (PHASE8_WAVE3_ELIGIBILITY_POLICY_FILENAME, "phase8_eligibility_policy"),
        (PHASE8_WAVE3_LABEL_MAPPING_POLICY_FILENAME, "phase8_label_mapping_policy"),
        (PHASE8_WAVE3_SUMMARY_FILENAME, "phase8_wave3_generation_summary"),
    ):
        _write_json(root / filename, {"schema_name": schema_name, "schema_version": "v1"})
    return root


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
