"""Unit tests for the Phase 8 Package D definitive-freeze wiring module.

All tests operate on synthetic, tmp_path-scoped inputs. No real dataset,
``/Volumes`` path, medical file, or Package C checkpoint is referenced
anywhere in this file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from protoem_ct.external import definitive_freeze as freeze
from protoem_ct.external.artifacts import (
    Phase8ArtifactHashError,
    phase8_decision_freeze_from_json,
)
from protoem_ct.external.internal_evidence import phase8_support_policy_from_mapping

_FAKE_HASH_A = "a" * 64
_FAKE_HASH_B = "b" * 64
_FAKE_HASH_C = "c" * 64
_FAKE_HASH_D = "d" * 64
_FAKE_HASH_E = "e" * 64
_FAKE_HASH_F = "f" * 64
_FAKE_HASH_1 = "1" * 64
_FAKE_HASH_2 = "2" * 64


def _build_freeze_kwargs(*, support_policy_hash: str) -> dict[str, str]:
    return {
        "definitive_config_hash": _FAKE_HASH_A,
        "spacing_provenance_hash": _FAKE_HASH_B,
        "selection_evidence_hash": _FAKE_HASH_C,
        "selected_checkpoint_metadata_hash": _FAKE_HASH_D,
        "selected_checkpoint_sha256": _FAKE_HASH_E,
        "training_policy_hash": _FAKE_HASH_F,
        "support_policy_hash": support_policy_hash,
        "label_mapping_policy_hash": _FAKE_HASH_1,
        "metric_configuration_hash": _FAKE_HASH_2,
        "bootstrap_configuration_hash": _FAKE_HASH_1,
        "publication_configuration_hash": _FAKE_HASH_2,
    }


def test_build_no_support_policy_self_validates_and_round_trips() -> None:
    policy = freeze.build_definitive_no_support_policy(
        selected_candidate_id="phase8_definitive_segresnet",
        selected_checkpoint_hash=_FAKE_HASH_E,
    )
    assert policy.policy_type == "no_support"
    assert policy.external_support_labels_forbidden is True
    assert policy.external_prototype_construction_forbidden is True
    assert policy.external_support_state == "forbidden"
    assert policy.freeze_status == "freeze_ready"
    assert policy.internal_support_manifest_reference is None
    assert policy.support_k is None
    assert policy.support_seed_reference is None

    from protoem_ct.external.internal_evidence import phase8_support_policy_to_dict

    round_tripped = phase8_support_policy_from_mapping(phase8_support_policy_to_dict(policy))
    assert round_tripped == policy


def test_build_decision_freeze_contains_exactly_nine_categories() -> None:
    support_policy = freeze.build_definitive_no_support_policy(
        selected_candidate_id="phase8_definitive_segresnet",
        selected_checkpoint_hash=_FAKE_HASH_E,
    )
    decision_freeze = freeze.build_definitive_decision_freeze(
        **_build_freeze_kwargs(support_policy_hash=support_policy.support_policy_hash)
    )
    categories = sorted(r.category for r in decision_freeze.decision_references)
    assert categories == [
        "bootstrap_configuration",
        "checkpoint_metadata",
        "label_mapping_policy",
        "metric_configuration",
        "model_selection_decision",
        "preprocessing_decision",
        "publication_configuration",
        "support_policy",
        "threshold_decision",
    ]
    assert decision_freeze.freeze_state == "frozen"
    assert decision_freeze.frozen_before_external_evaluation is True
    for reference in decision_freeze.decision_references:
        assert reference.frozen is True
        assert reference.freeze_state == "frozen"


def test_decision_freeze_round_trips_through_json() -> None:
    support_policy = freeze.build_definitive_no_support_policy(
        selected_candidate_id="phase8_definitive_segresnet",
        selected_checkpoint_hash=_FAKE_HASH_E,
    )
    decision_freeze = freeze.build_definitive_decision_freeze(
        **_build_freeze_kwargs(support_policy_hash=support_policy.support_policy_hash)
    )
    from protoem_ct.external.artifacts import phase8_decision_freeze_to_json

    reconstructed = phase8_decision_freeze_from_json(
        phase8_decision_freeze_to_json(decision_freeze)
    )
    assert reconstructed == decision_freeze


def test_decision_freeze_rejects_tampered_reference() -> None:
    support_policy = freeze.build_definitive_no_support_policy(
        selected_candidate_id="phase8_definitive_segresnet",
        selected_checkpoint_hash=_FAKE_HASH_E,
    )
    decision_freeze = freeze.build_definitive_decision_freeze(
        **_build_freeze_kwargs(support_policy_hash=support_policy.support_policy_hash)
    )
    from dataclasses import replace

    from protoem_ct.external.artifacts import Phase8DecisionFreeze

    tampered_references = tuple(
        replace(r, source_artifact_hash=_FAKE_HASH_1) if r.category == "checkpoint_metadata" else r
        for r in decision_freeze.decision_references
    )
    with pytest.raises(Phase8ArtifactHashError):
        Phase8DecisionFreeze(
            schema_name=decision_freeze.schema_name,
            schema_version=decision_freeze.schema_version,
            freeze_inventory_hash=decision_freeze.freeze_inventory_hash,
            freeze_state=decision_freeze.freeze_state,
            frozen_before_external_evaluation=decision_freeze.frozen_before_external_evaluation,
            decision_references=tampered_references,
        )


def test_run_publication_writes_both_artifacts_and_rejects_overwrite(tmp_path: Path) -> None:
    repository_root = tmp_path / "repo"
    repository_root.mkdir()
    output_parent = tmp_path / "external_run_root"
    output_parent.mkdir()
    output_root = output_parent / "phase8_definitive_freeze_v1"

    result = freeze.run_phase8_definitive_freeze_publication(
        output_root=output_root,
        repository_root=repository_root,
        definitive_config_hash=_FAKE_HASH_A,
        spacing_provenance_hash=_FAKE_HASH_B,
        selection_evidence_hash=_FAKE_HASH_C,
        selected_checkpoint_metadata_hash=_FAKE_HASH_D,
        selected_checkpoint_sha256=_FAKE_HASH_E,
        selected_candidate_id="phase8_definitive_segresnet",
        training_policy_hash=_FAKE_HASH_F,
        label_mapping_policy_hash=_FAKE_HASH_1,
        metric_configuration_hash=_FAKE_HASH_2,
        bootstrap_configuration_hash=_FAKE_HASH_1,
        publication_configuration_hash=_FAKE_HASH_2,
    )

    assert result.support_policy_path.is_file()
    assert result.decision_freeze_path.is_file()
    published_freeze = json.loads(result.decision_freeze_path.read_text())
    assert published_freeze["freeze_state"] == "frozen"
    published_support = json.loads(result.support_policy_path.read_text())
    assert published_support["policy_type"] == "no_support"
    assert published_support["selected_checkpoint_hash"] == _FAKE_HASH_E

    # Reject-on-overwrite: a second call into the same root must fail closed.
    with pytest.raises(freeze.Phase8DefinitiveFreezeError):
        freeze.run_phase8_definitive_freeze_publication(
            output_root=output_root,
            repository_root=repository_root,
            definitive_config_hash=_FAKE_HASH_A,
            spacing_provenance_hash=_FAKE_HASH_B,
            selection_evidence_hash=_FAKE_HASH_C,
            selected_checkpoint_metadata_hash=_FAKE_HASH_D,
            selected_checkpoint_sha256=_FAKE_HASH_E,
            selected_candidate_id="phase8_definitive_segresnet",
            training_policy_hash=_FAKE_HASH_F,
            label_mapping_policy_hash=_FAKE_HASH_1,
            metric_configuration_hash=_FAKE_HASH_2,
            bootstrap_configuration_hash=_FAKE_HASH_1,
            publication_configuration_hash=_FAKE_HASH_2,
        )


def test_run_publication_rejects_output_root_inside_repository(tmp_path: Path) -> None:
    repository_root = tmp_path / "repo"
    repository_root.mkdir()
    output_root = repository_root / "phase8_definitive_freeze_v1"

    with pytest.raises(freeze.Phase8DefinitiveFreezeError):
        freeze.run_phase8_definitive_freeze_publication(
            output_root=output_root,
            repository_root=repository_root,
            definitive_config_hash=_FAKE_HASH_A,
            spacing_provenance_hash=_FAKE_HASH_B,
            selection_evidence_hash=_FAKE_HASH_C,
            selected_checkpoint_metadata_hash=_FAKE_HASH_D,
            selected_checkpoint_sha256=_FAKE_HASH_E,
            selected_candidate_id="phase8_definitive_segresnet",
            training_policy_hash=_FAKE_HASH_F,
            label_mapping_policy_hash=_FAKE_HASH_1,
            metric_configuration_hash=_FAKE_HASH_2,
            bootstrap_configuration_hash=_FAKE_HASH_1,
            publication_configuration_hash=_FAKE_HASH_2,
        )


def test_run_publication_rejects_nonempty_existing_output_root(tmp_path: Path) -> None:
    repository_root = tmp_path / "repo"
    repository_root.mkdir()
    output_root = tmp_path / "external_run_root" / "phase8_definitive_freeze_v1"
    output_root.mkdir(parents=True)
    (output_root / "stray.txt").write_text("x")

    with pytest.raises(freeze.Phase8DefinitiveFreezeError):
        freeze.run_phase8_definitive_freeze_publication(
            output_root=output_root,
            repository_root=repository_root,
            definitive_config_hash=_FAKE_HASH_A,
            spacing_provenance_hash=_FAKE_HASH_B,
            selection_evidence_hash=_FAKE_HASH_C,
            selected_checkpoint_metadata_hash=_FAKE_HASH_D,
            selected_checkpoint_sha256=_FAKE_HASH_E,
            selected_candidate_id="phase8_definitive_segresnet",
            training_policy_hash=_FAKE_HASH_F,
            label_mapping_policy_hash=_FAKE_HASH_1,
            metric_configuration_hash=_FAKE_HASH_2,
            bootstrap_configuration_hash=_FAKE_HASH_1,
            publication_configuration_hash=_FAKE_HASH_2,
        )


def test_no_medical_file_access_or_external_dataset_open_in_source() -> None:
    """The module may *mention* the external dataset path in prose, but must never open one."""

    source_text = Path(freeze.__file__).read_text(encoding="utf-8")
    assert ".nii" not in source_text
    assert "3D-IRCADb" not in source_text
    assert "nib.load" not in source_text
    assert "nibabel" not in source_text
    assert "open(" not in source_text
