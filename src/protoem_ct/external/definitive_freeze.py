"""Phase 8 Package D: definitive freeze publication (minimal wiring only).

This module performs no scientific decision-making. Every value it accepts
is either a caller-supplied, already-computed real hash/identifier from the
completed Package C run (or an earlier approved Wave 2/3/4 artifact), or a
locked literal already approved elsewhere in this repository (the
``no_support`` policy type from
:mod:`protoem_ct.external.definitive_training`). It wires those already-real
values into the existing, unmodified freeze contracts
(:class:`protoem_ct.external.artifacts.Phase8DecisionFreeze`,
:class:`protoem_ct.external.artifacts.FrozenDecisionReference`, and
:class:`protoem_ct.external.internal_evidence.Phase8SupportPolicy`) and
publishes them, reject-on-overwrite, beneath an explicit external output
root. It never opens a medical image/label file, never accesses
``/Volumes/Lexar/ProtoEM-CT/datasets/external``, and never recomputes a
training/validation metric.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from protoem_ct.artifacts.hashing import JsonValue, canonical_json_bytes, sha256_file, sha256_json
from protoem_ct.data._phase2_publication import (
    Phase2PublicationExistingOutputError,
    Phase2PublicationIOError,
    publish_text_no_overwrite,
)
from protoem_ct.data.phase2_paths import (
    InvalidDatasetRootError,
    validate_explicit_external_output_root,
)
from protoem_ct.external.artifacts import (
    FrozenDecisionReference,
    Phase8DecisionFreeze,
    frozen_decision_reference_to_dict,
    phase8_decision_freeze_to_dict,
)
from protoem_ct.external.definitive_training import REQUIRED_SUPPORT_POLICY_TYPE
from protoem_ct.external.internal_evidence import (
    PHASE8_SUPPORT_POLICY_SCHEMA_NAME,
    Phase8SupportPolicy,
    phase8_support_policy_to_dict,
)

PHASE8_DEFINITIVE_FREEZE_SUPPORT_POLICY_FILENAME: Final[str] = (
    "phase8_definitive_support_policy.json"
)
PHASE8_DEFINITIVE_FREEZE_DECISION_FREEZE_FILENAME: Final[str] = "phase8_decision_freeze.json"


class Phase8DefinitiveFreezeError(ValueError):
    """Raised when definitive freeze publication fails safely."""


@dataclass(frozen=True, slots=True)
class Phase8DefinitiveFreezePublicationResult:
    """Result of one guarded definitive-freeze publication."""

    output_root: Path
    support_policy_hash: str
    decision_freeze_hash: str
    support_policy_path: Path
    decision_freeze_path: Path


def build_definitive_no_support_policy(
    *, selected_candidate_id: str, selected_checkpoint_hash: str
) -> Phase8SupportPolicy:
    """Build the locked ``no_support`` policy bound to Package C's real candidate.

    Reuses :class:`Phase8SupportPolicy` and the already-approved
    ``REQUIRED_SUPPORT_POLICY_TYPE`` (``"no_support"``) literal, defined and
    locked in :mod:`protoem_ct.external.definitive_training`
    (Substage 4A, committed at ``809edeb``) as the sole support-policy type
    ever concretely constructed anywhere in this repository for the single
    definitive SegResNet baseline candidate. No new support decision is made
    here; only the caller-supplied real ``candidate_id``/``checkpoint_hash``
    (taken from Package C's own published checkpoint metadata) are bound in.
    """

    payload: dict[str, JsonValue] = {
        "external_prototype_construction_forbidden": True,
        "external_support_labels_forbidden": True,
        "external_support_state": "forbidden",
        "freeze_status": "freeze_ready",
        "internal_support_manifest_reference": None,
        "policy_type": REQUIRED_SUPPORT_POLICY_TYPE,
        "schema_name": PHASE8_SUPPORT_POLICY_SCHEMA_NAME,
        "schema_version": "v1",
        "selected_candidate_id": selected_candidate_id,
        "selected_checkpoint_hash": selected_checkpoint_hash,
        "support_k": None,
        "support_seed_reference": None,
    }
    return Phase8SupportPolicy(
        schema_name=PHASE8_SUPPORT_POLICY_SCHEMA_NAME,
        schema_version="v1",
        policy_type=REQUIRED_SUPPORT_POLICY_TYPE,
        selected_candidate_id=selected_candidate_id,
        selected_checkpoint_hash=selected_checkpoint_hash,
        internal_support_manifest_reference=None,
        support_k=None,
        support_seed_reference=None,
        external_support_labels_forbidden=True,
        external_prototype_construction_forbidden=True,
        external_support_state="forbidden",
        freeze_status="freeze_ready",
        support_policy_hash=sha256_json(payload),
    )


def _reference(
    *,
    category: str,
    source_schema_name: str,
    source_artifact_hash: str,
    source_phase: str,
    rationale_code: str,
    provenance_reference: str,
) -> FrozenDecisionReference:
    return FrozenDecisionReference(
        schema_name="phase8_frozen_decision_reference",
        schema_version="v1",
        category=category,
        source_schema_name=source_schema_name,
        source_schema_version="v1",
        source_phase=source_phase,
        source_artifact_hash=source_artifact_hash,
        rationale_code=rationale_code,
        provenance_reference=provenance_reference,
        frozen=True,
        freeze_state="frozen",
    )


def build_definitive_decision_freeze(
    *,
    definitive_config_hash: str,
    spacing_provenance_hash: str,
    selection_evidence_hash: str,
    selected_checkpoint_metadata_hash: str,
    selected_checkpoint_sha256: str,
    training_policy_hash: str,
    support_policy_hash: str,
    label_mapping_policy_hash: str,
    metric_configuration_hash: str,
    bootstrap_configuration_hash: str,
    publication_configuration_hash: str,
) -> Phase8DecisionFreeze:
    """Assemble the Phase 8 freeze inventory from already-real evidence hashes.

    Every ``source_artifact_hash`` passed in is an already-published, already
    -verified real hash (Package C's config/selection-evidence/checkpoint
    metadata, or an earlier approved Wave 3/Wave 4 artifact hash). This
    function performs no new scientific decision; it only assembles the nine
    required :class:`FrozenDecisionReference` categories into one
    :class:`Phase8DecisionFreeze`.
    """

    references = (
        _reference(
            category="preprocessing_decision",
            source_schema_name="phase8_definitive_config",
            source_artifact_hash=definitive_config_hash,
            source_phase="phase8_package_c",
            rationale_code="phase8-definitive-train-spacing-v1",
            provenance_reference=spacing_provenance_hash,
        ),
        _reference(
            category="threshold_decision",
            source_schema_name="phase8_definitive_config",
            source_artifact_hash=definitive_config_hash,
            source_phase="phase8_package_c",
            rationale_code="fixed-constant-threshold-not-tuned-on-external-data",
            provenance_reference=selection_evidence_hash,
        ),
        _reference(
            category="model_selection_decision",
            source_schema_name="phase8_definitive_checkpoint_selection_evidence",
            source_artifact_hash=selection_evidence_hash,
            source_phase="phase8_package_c",
            rationale_code="higher-mean-tumor-dice-tie-break-step-250",
            provenance_reference=training_policy_hash,
        ),
        _reference(
            category="checkpoint_metadata",
            source_schema_name="phase8_checkpoint_metadata",
            source_artifact_hash=selected_checkpoint_metadata_hash,
            source_phase="phase8_package_c",
            rationale_code="selected-checkpoint-step-500",
            provenance_reference=selected_checkpoint_sha256,
        ),
        _reference(
            category="support_policy",
            source_schema_name="phase8_support_policy",
            source_artifact_hash=support_policy_hash,
            source_phase="phase8_package_c",
            rationale_code="no-support-locked-baseline-candidate",
            provenance_reference=definitive_config_hash,
        ),
        _reference(
            category="label_mapping_policy",
            source_schema_name="phase8_external_label_mapping_policy",
            source_artifact_hash=label_mapping_policy_hash,
            source_phase="phase8_wave3",
            rationale_code="wave3-approved-prelabel-policy",
            provenance_reference="phase8-wave3-policy-v1",
        ),
        _reference(
            category="metric_configuration",
            source_schema_name="phase8_metric_configuration",
            source_artifact_hash=metric_configuration_hash,
            source_phase="phase8_wave4",
            rationale_code="wave4-preregistered-statistical-policy",
            provenance_reference="phase8-statistical-policy-bundle-v1",
        ),
        _reference(
            category="bootstrap_configuration",
            source_schema_name="phase8_bootstrap_configuration",
            source_artifact_hash=bootstrap_configuration_hash,
            source_phase="phase8_wave4",
            rationale_code="wave4-preregistered-statistical-policy",
            provenance_reference="phase8-statistical-policy-bundle-v1",
        ),
        _reference(
            category="publication_configuration",
            source_schema_name="phase8_publication_configuration",
            source_artifact_hash=publication_configuration_hash,
            source_phase="phase8_wave4",
            rationale_code="wave4-preregistered-statistical-policy",
            provenance_reference="phase8-statistical-policy-bundle-v1",
        ),
    )
    return Phase8DecisionFreeze(
        schema_name="phase8_decision_freeze",
        schema_version="v1",
        freeze_inventory_hash=sha256_json(
            {
                "decision_references": [
                    frozen_decision_reference_to_dict(item)
                    for item in sorted(references, key=lambda item: item.category)
                ],
                "freeze_state": "frozen",
                "frozen_before_external_evaluation": True,
                "schema_name": "phase8_decision_freeze",
                "schema_version": "v1",
            }
        ),
        freeze_state="frozen",
        frozen_before_external_evaluation=True,
        decision_references=references,
    )


def _validate_output_root(output_root: Path, *, repository_root: Path) -> Path:
    if output_root.exists() and output_root.is_symlink():
        raise Phase8DefinitiveFreezeError("output_root must not be a symlink.")
    try:
        canonical = validate_explicit_external_output_root(
            output_root, forbidden_roots=(repository_root,)
        )
    except InvalidDatasetRootError as exc:
        raise Phase8DefinitiveFreezeError("invalid definitive freeze output root.") from exc
    if canonical.exists() and any(canonical.iterdir()):
        raise Phase8DefinitiveFreezeError("definitive freeze output root already contains files.")
    return canonical


def _publish_json(path: Path, payload: dict[str, JsonValue]) -> None:
    try:
        publish_text_no_overwrite(
            text=(canonical_json_bytes(payload) + b"\n").decode("utf-8"),
            output_path=path,
            temporary_exists_message="temporary Phase 8 definitive freeze artifact already exists.",
            final_exists_message="Phase 8 definitive freeze artifact already exists.",
        )
    except (Phase2PublicationExistingOutputError, Phase2PublicationIOError) as exc:
        raise Phase8DefinitiveFreezeError(
            "failed to publish Phase 8 definitive freeze artifact."
        ) from exc


def run_phase8_definitive_freeze_publication(
    *,
    output_root: Path,
    repository_root: Path,
    definitive_config_hash: str,
    spacing_provenance_hash: str,
    selection_evidence_hash: str,
    selected_checkpoint_metadata_hash: str,
    selected_checkpoint_sha256: str,
    selected_candidate_id: str,
    training_policy_hash: str,
    label_mapping_policy_hash: str,
    metric_configuration_hash: str,
    bootstrap_configuration_hash: str,
    publication_configuration_hash: str,
) -> Phase8DefinitiveFreezePublicationResult:
    """Publish the support-policy and decision-freeze artifacts, reject-on-overwrite.

    Every argument is a caller-supplied, already-real hash/identifier; this
    function opens no medical file and accesses no external dataset path.
    """

    canonical_repository_root = repository_root.resolve(strict=True)
    canonical_output_root = _validate_output_root(
        output_root, repository_root=canonical_repository_root
    )

    support_policy = build_definitive_no_support_policy(
        selected_candidate_id=selected_candidate_id,
        selected_checkpoint_hash=selected_checkpoint_sha256,
    )
    decision_freeze = build_definitive_decision_freeze(
        definitive_config_hash=definitive_config_hash,
        spacing_provenance_hash=spacing_provenance_hash,
        selection_evidence_hash=selection_evidence_hash,
        selected_checkpoint_metadata_hash=selected_checkpoint_metadata_hash,
        selected_checkpoint_sha256=selected_checkpoint_sha256,
        training_policy_hash=training_policy_hash,
        support_policy_hash=support_policy.support_policy_hash,
        label_mapping_policy_hash=label_mapping_policy_hash,
        metric_configuration_hash=metric_configuration_hash,
        bootstrap_configuration_hash=bootstrap_configuration_hash,
        publication_configuration_hash=publication_configuration_hash,
    )

    support_policy_path = canonical_output_root / PHASE8_DEFINITIVE_FREEZE_SUPPORT_POLICY_FILENAME
    decision_freeze_path = canonical_output_root / PHASE8_DEFINITIVE_FREEZE_DECISION_FREEZE_FILENAME
    _publish_json(support_policy_path, phase8_support_policy_to_dict(support_policy))
    _publish_json(decision_freeze_path, phase8_decision_freeze_to_dict(decision_freeze))

    return Phase8DefinitiveFreezePublicationResult(
        output_root=canonical_output_root,
        support_policy_hash=sha256_file(support_policy_path),
        decision_freeze_hash=sha256_file(decision_freeze_path),
        support_policy_path=support_policy_path,
        decision_freeze_path=decision_freeze_path,
    )


__all__ = [
    "PHASE8_DEFINITIVE_FREEZE_DECISION_FREEZE_FILENAME",
    "PHASE8_DEFINITIVE_FREEZE_SUPPORT_POLICY_FILENAME",
    "Phase8DefinitiveFreezeError",
    "Phase8DefinitiveFreezePublicationResult",
    "build_definitive_decision_freeze",
    "build_definitive_no_support_policy",
    "run_phase8_definitive_freeze_publication",
]
