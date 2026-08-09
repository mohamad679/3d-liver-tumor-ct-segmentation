"""Unit tests for Phase 8 Package E preregistration publication wiring."""

from __future__ import annotations

from pathlib import Path

import pytest

from protoem_ct.artifacts.hashing import canonical_json_bytes, sha256_file, sha256_json
from protoem_ct.external.artifacts import (
    FrozenDecisionReference,
    Phase8DecisionFreeze,
    phase8_decision_freeze_to_json,
)
from protoem_ct.external.definitive_freeze import build_definitive_no_support_policy
from protoem_ct.external.eligibility import hash_phase8_eligibility_policy
from protoem_ct.external.freeze import PHASE8_EXTERNAL_PREREGISTRATION_FILENAME
from protoem_ct.external.internal_evidence import (
    Phase8SupportPolicy,
    phase8_support_policy_to_dict,
)
from protoem_ct.external.label_mapping import (
    build_default_phase8_label_mapping_policy,
    phase8_label_mapping_policy_to_json,
)
from protoem_ct.external.preregistration import (
    phase8_external_preregistration_from_json,
    phase8_external_preregistration_to_dict,
)
from protoem_ct.external.preregistration_publication import (
    Phase8PreregistrationPublicationError,
    build_phase8_external_preregistration,
    run_phase8_external_preregistration_publication,
    sha256_json_bytes,
)
from protoem_ct.external.statistical_policy import build_phase8_statistical_policy_bundle


def _build_support_policy() -> Phase8SupportPolicy:
    return build_definitive_no_support_policy(
        selected_candidate_id="unit-test-candidate",
        selected_checkpoint_hash="3" * 64,
    )


def _support_policy_json_bytes(policy: Phase8SupportPolicy) -> bytes:
    return canonical_json_bytes(phase8_support_policy_to_dict(policy)) + b"\n"


_CATEGORY_SOURCE_SCHEMAS: dict[str, str] = {
    "bootstrap_configuration": "phase8_bootstrap_configuration",
    "checkpoint_metadata": "phase8_checkpoint_metadata",
    "label_mapping_policy": "phase8_external_label_mapping_policy",
    "metric_configuration": "phase8_metric_configuration",
    "model_selection_decision": "phase8_definitive_checkpoint_selection_evidence",
    "preprocessing_decision": "phase8_definitive_config",
    "publication_configuration": "phase8_publication_configuration",
    "support_policy": "phase8_support_policy",
    "threshold_decision": "phase8_definitive_config",
}


def _reference(category: str, source_artifact_hash: str) -> FrozenDecisionReference:
    return FrozenDecisionReference(
        schema_name="phase8_frozen_decision_reference",
        schema_version="v1",
        category=category,
        source_schema_name=_CATEGORY_SOURCE_SCHEMAS[category],
        source_schema_version="v1",
        source_phase="phase8_package_c",
        source_artifact_hash=source_artifact_hash,
        rationale_code="unit-test-rationale",
        provenance_reference=None,
        frozen=True,
        freeze_state="frozen",
    )


def _build_frozen_decision_freeze(
    *,
    label_mapping_hash: str | None = None,
    support_policy_hash: str | None = None,
) -> Phase8DecisionFreeze:
    statistical_policy = build_phase8_statistical_policy_bundle()
    label_mapping_policy = build_default_phase8_label_mapping_policy()
    resolved_label_mapping_hash = label_mapping_hash or sha256_json_bytes(
        phase8_label_mapping_policy_to_json(label_mapping_policy)
    )
    resolved_support_policy_hash = (
        support_policy_hash or _build_support_policy().support_policy_hash
    )
    placeholder = "1" * 64
    references = tuple(
        _reference(category, placeholder)
        for category in (
            "checkpoint_metadata",
            "model_selection_decision",
            "preprocessing_decision",
            "threshold_decision",
        )
    ) + (
        _reference("support_policy", resolved_support_policy_hash),
        _reference("label_mapping_policy", resolved_label_mapping_hash),
        _reference(
            "bootstrap_configuration",
            sha256_json(statistical_policy["bootstrap_configuration"]),
        ),
        _reference(
            "metric_configuration",
            sha256_json(statistical_policy["metric_configuration"]),
        ),
        _reference(
            "publication_configuration",
            sha256_json(statistical_policy["publication_configuration"]),
        ),
    )
    ordered_references = sorted(references, key=lambda item: item.category)
    payload = {
        "decision_references": [
            {
                "category": r.category,
                "freeze_state": r.freeze_state,
                "frozen": r.frozen,
                "provenance_reference": r.provenance_reference,
                "rationale_code": r.rationale_code,
                "schema_name": r.schema_name,
                "schema_version": r.schema_version,
                "source_artifact_hash": r.source_artifact_hash,
                "source_phase": r.source_phase,
                "source_schema_name": r.source_schema_name,
                "source_schema_version": r.source_schema_version,
            }
            for r in ordered_references
        ],
        "freeze_state": "frozen",
        "frozen_before_external_evaluation": True,
        "schema_name": "phase8_decision_freeze",
        "schema_version": "v1",
    }
    freeze_inventory_hash = sha256_json(payload)
    return Phase8DecisionFreeze(
        schema_name="phase8_decision_freeze",
        schema_version="v1",
        freeze_inventory_hash=freeze_inventory_hash,
        freeze_state="frozen",
        frozen_before_external_evaluation=True,
        decision_references=references,
    )


def test_build_phase8_external_preregistration_is_draft_with_expected_references() -> None:
    freeze = _build_frozen_decision_freeze()

    preregistration = build_phase8_external_preregistration(decision_freeze=freeze)

    assert preregistration.lifecycle_state == "draft"
    assert preregistration.external_cohort_identifier == "3d_ircadb_01"
    assert preregistration.frozen_decision_inventory_hash == freeze.freeze_inventory_hash
    assert preregistration.anonymous_manifest_reference.reference_state == "unresolved"
    assert preregistration.anonymous_manifest_reference.artifact_hash is None
    assert preregistration.domain_shift_record_reference.reference_state == "unresolved"
    assert preregistration.label_mapping_policy_reference.reference_state == "resolved"
    assert (
        preregistration.eligibility_exclusion_policy_reference.artifact_hash
        == hash_phase8_eligibility_policy()
    )
    assert preregistration.no_tuning_declaration is True
    assert preregistration.external_label_access_state == "unavailable_before_prediction_lock"
    assert preregistration.prediction_lock_requirement == "required_before_label_access"
    assert preregistration.robustness_uncertainty_inclusion.inclusion_state == "not_included"
    assert preregistration.robustness_uncertainty_inclusion.policy_reference is None

    round_tripped = phase8_external_preregistration_from_json(
        canonical_json_bytes(phase8_external_preregistration_to_dict(preregistration))
    )
    assert round_tripped == preregistration


def test_build_phase8_external_preregistration_rejects_stale_label_mapping_freeze() -> None:
    freeze = _build_frozen_decision_freeze(label_mapping_hash="2" * 64)

    with pytest.raises(Phase8PreregistrationPublicationError, match="label_mapping_policy"):
        build_phase8_external_preregistration(decision_freeze=freeze)


def test_run_phase8_external_preregistration_publication_end_to_end(tmp_path: Path) -> None:
    freeze = _build_frozen_decision_freeze()
    freeze_path = tmp_path / "phase8_decision_freeze.json"
    freeze_path.write_bytes(phase8_decision_freeze_to_json(freeze))

    support_policy_path = tmp_path / "phase8_definitive_support_policy.json"
    support_policy_path.write_bytes(_support_policy_json_bytes(_build_support_policy()))

    output_root = tmp_path / "output_root"
    result = run_phase8_external_preregistration_publication(
        freeze_artifact_path=freeze_path,
        expected_freeze_artifact_hash=sha256_file(freeze_path),
        support_policy_artifact_path=support_policy_path,
        expected_support_policy_artifact_hash=sha256_file(support_policy_path),
        output_root=output_root,
    )

    assert result.output_root == output_root.resolve()
    assert result.output_path == output_root.resolve() / PHASE8_EXTERNAL_PREREGISTRATION_FILENAME
    assert result.output_path.is_file()

    published = phase8_external_preregistration_from_json(result.output_path.read_bytes())
    assert published.preregistration_hash == result.preregistration_hash
    assert published.lifecycle_state == "draft"


def test_run_phase8_external_preregistration_publication_rejects_hash_mismatch(
    tmp_path: Path,
) -> None:
    freeze = _build_frozen_decision_freeze()
    freeze_path = tmp_path / "phase8_decision_freeze.json"
    freeze_path.write_bytes(phase8_decision_freeze_to_json(freeze))
    support_policy_path = tmp_path / "phase8_definitive_support_policy.json"
    support_policy_path.write_bytes(_support_policy_json_bytes(_build_support_policy()))

    with pytest.raises(Phase8PreregistrationPublicationError, match="SHA-256"):
        run_phase8_external_preregistration_publication(
            freeze_artifact_path=freeze_path,
            expected_freeze_artifact_hash="0" * 64,
            support_policy_artifact_path=support_policy_path,
            expected_support_policy_artifact_hash="0" * 64,
            output_root=tmp_path / "output_root_2",
        )


def test_run_phase8_external_preregistration_publication_rejects_stale_support_policy(
    tmp_path: Path,
) -> None:
    freeze = _build_frozen_decision_freeze(support_policy_hash="4" * 64)
    freeze_path = tmp_path / "phase8_decision_freeze.json"
    freeze_path.write_bytes(phase8_decision_freeze_to_json(freeze))
    support_policy_path = tmp_path / "phase8_definitive_support_policy.json"
    support_policy_path.write_bytes(_support_policy_json_bytes(_build_support_policy()))

    with pytest.raises(Phase8PreregistrationPublicationError, match="support policy"):
        run_phase8_external_preregistration_publication(
            freeze_artifact_path=freeze_path,
            expected_freeze_artifact_hash=sha256_file(freeze_path),
            support_policy_artifact_path=support_policy_path,
            expected_support_policy_artifact_hash=sha256_file(support_policy_path),
            output_root=tmp_path / "output_root_4",
        )


def test_run_phase8_external_preregistration_publication_refuses_existing_output_root(
    tmp_path: Path,
) -> None:
    freeze = _build_frozen_decision_freeze()
    freeze_path = tmp_path / "phase8_decision_freeze.json"
    freeze_path.write_bytes(phase8_decision_freeze_to_json(freeze))
    support_policy_path = tmp_path / "phase8_definitive_support_policy.json"
    support_policy_path.write_bytes(_support_policy_json_bytes(_build_support_policy()))

    output_root = tmp_path / "output_root_3"
    output_root.mkdir()

    with pytest.raises(Phase8PreregistrationPublicationError, match="already exists"):
        run_phase8_external_preregistration_publication(
            freeze_artifact_path=freeze_path,
            expected_freeze_artifact_hash=sha256_file(freeze_path),
            support_policy_artifact_path=support_policy_path,
            expected_support_policy_artifact_hash=sha256_file(support_policy_path),
            output_root=output_root,
        )
