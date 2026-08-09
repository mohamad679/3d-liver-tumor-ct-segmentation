"""Phase 8 Package E: external-evaluation preregistration publication wiring.

This module wires the already-committed, already-tested Phase 8 preregistration schema
(``preregistration.py``) to the already-frozen Package D decision-freeze artifact
(``phase8_decision_freeze.json``). It introduces no new scientific policy: every referenced value is
either read verbatim from the frozen decision inventory or recomputed from an already-committed
repository contract (label-mapping policy, eligibility policy, statistical policy bundle) and
cross-checked against the freeze's recorded hash.

This module performs no external-dataset filesystem access. It reads exactly the two caller-supplied
frozen-decision JSON files (whose SHA-256 the caller must supply and which this module verifies
before use) and writes one non-medical preregistration JSON artifact. The external cohort's
configured root is recorded nowhere in this module -- it never appears in any hashed/published
payload.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from protoem_ct.artifacts.hashing import JsonValue, sha256_file, sha256_json
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
    phase8_decision_freeze_from_json,
)
from protoem_ct.external.definitive_training import REQUIRED_SUPPORT_POLICY_TYPE
from protoem_ct.external.eligibility import (
    PHASE8_ELIGIBILITY_POLICY_SCHEMA_NAME,
    hash_phase8_eligibility_policy,
)
from protoem_ct.external.freeze import PHASE8_EXTERNAL_PREREGISTRATION_FILENAME
from protoem_ct.external.internal_evidence import phase8_support_policy_from_mapping
from protoem_ct.external.label_mapping import (
    build_default_phase8_label_mapping_policy,
    phase8_label_mapping_policy_to_json,
)
from protoem_ct.external.preregistration import (
    PHASE8_ARTIFACT_REFERENCE_SCHEMA_NAME,
    PHASE8_ARTIFACT_REFERENCE_SCHEMA_VERSION,
    PHASE8_EXTERNAL_PREREGISTRATION_SCHEMA_NAME,
    PHASE8_EXTERNAL_PREREGISTRATION_SCHEMA_VERSION,
    PHASE8_ROBUSTNESS_UNCERTAINTY_INCLUSION_SCHEMA_NAME,
    PHASE8_ROBUSTNESS_UNCERTAINTY_INCLUSION_SCHEMA_VERSION,
    Phase8ExternalPreregistration,
    phase8_external_preregistration_from_mapping,
    phase8_external_preregistration_to_json,
)
from protoem_ct.external.statistical_policy import (
    PHASE8_SEGMENTATION_METRICS,
    build_phase8_statistical_policy_bundle,
)

PHASE8_PREREGISTRATION_EXTERNAL_COHORT_IDENTIFIER: Final[str] = "3d_ircadb_01"
_LABEL_MAPPING_SOURCE_SCHEMA_NAME: Final[str] = "phase8_external_label_mapping_policy"

_FREEZE_TO_COMPONENT: Final[tuple[tuple[str, str], ...]] = (
    ("bootstrap_configuration", "bootstrap_configuration"),
    ("metric_configuration", "metric_configuration"),
    ("publication_configuration", "publication_configuration"),
)


class Phase8PreregistrationPublicationError(ValueError):
    """Raised when guarded Package E preregistration publication fails safely."""


@dataclass(frozen=True, slots=True)
class Phase8PreregistrationPublicationResult:
    """Result of one guarded Package E preregistration publication."""

    output_root: Path
    output_path: Path
    preregistration_hash: str
    freeze_artifact_hash: str
    support_policy_artifact_hash: str


def build_phase8_external_preregistration(
    *,
    decision_freeze: Phase8DecisionFreeze,
) -> Phase8ExternalPreregistration:
    """Build the draft Package E preregistration from the frozen decision inventory only.

    ``lifecycle_state`` is ``draft`` because the anonymous image manifest and domain-shift record do
    not exist yet -- no external data has been accessed. Every resolved reference below is either
    the frozen decision inventory itself or an already-committed repository policy contract,
    cross-checked against the hash the freeze inventory already recorded for it.
    """

    references_by_category = {ref.category: ref for ref in decision_freeze.decision_references}

    label_mapping_policy = build_default_phase8_label_mapping_policy()
    label_mapping_artifact_hash = sha256_json_bytes(
        phase8_label_mapping_policy_to_json(label_mapping_policy)
    )
    _require_matches_freeze(
        references_by_category,
        category="label_mapping_policy",
        computed_hash=label_mapping_artifact_hash,
    )

    statistical_policy = build_phase8_statistical_policy_bundle()
    for freeze_category, component_name in _FREEZE_TO_COMPONENT:
        component = statistical_policy[component_name]
        assert isinstance(component, dict)  # noqa: S101 - internal invariant, not user input
        _require_matches_freeze(
            references_by_category,
            category=freeze_category,
            computed_hash=sha256_json(component),
        )

    bootstrap_component = statistical_policy["bootstrap_configuration"]
    comparison_component = statistical_policy["comparison_configuration"]
    qualitative_component = statistical_policy["qualitative_output_policy"]
    assert isinstance(bootstrap_component, dict)  # noqa: S101
    assert isinstance(comparison_component, dict)  # noqa: S101
    assert isinstance(qualitative_component, dict)  # noqa: S101

    payload: dict[str, JsonValue] = {
        "schema_name": PHASE8_EXTERNAL_PREREGISTRATION_SCHEMA_NAME,
        "schema_version": PHASE8_EXTERNAL_PREREGISTRATION_SCHEMA_VERSION,
        "lifecycle_state": "draft",
        "external_cohort_identifier": PHASE8_PREREGISTRATION_EXTERNAL_COHORT_IDENTIFIER,
        "frozen_decision_inventory_hash": decision_freeze.freeze_inventory_hash,
        "anonymous_manifest_reference": _reference_payload(
            referenced_schema_name="phase8_external_image_manifest",
            artifact_hash=None,
            reference_state="unresolved",
        ),
        "eligibility_exclusion_policy_reference": _reference_payload(
            referenced_schema_name=PHASE8_ELIGIBILITY_POLICY_SCHEMA_NAME,
            artifact_hash=hash_phase8_eligibility_policy(),
            reference_state="resolved",
        ),
        "label_mapping_policy_reference": _reference_payload(
            referenced_schema_name=_LABEL_MAPPING_SOURCE_SCHEMA_NAME,
            artifact_hash=label_mapping_artifact_hash,
            reference_state="resolved",
        ),
        "domain_shift_record_reference": _reference_payload(
            referenced_schema_name="phase8_domain_shift_record",
            artifact_hash=None,
            reference_state="unresolved",
        ),
        "preregistered_segmentation_metrics": list(PHASE8_SEGMENTATION_METRICS),
        "bootstrap_policy_config_reference": _reference_payload(
            referenced_schema_name="phase8_bootstrap_configuration",
            artifact_hash=sha256_json(bootstrap_component),
            reference_state="resolved",
        ),
        "internal_external_comparison_policy_reference": _reference_payload(
            referenced_schema_name="phase8_internal_external_comparison_policy",
            artifact_hash=sha256_json(comparison_component),
            reference_state="resolved",
        ),
        "qualitative_output_policy_reference": _reference_payload(
            referenced_schema_name="phase8_qualitative_output_policy",
            artifact_hash=sha256_json(qualitative_component),
            reference_state="resolved",
        ),
        "robustness_uncertainty_inclusion": {
            "schema_name": PHASE8_ROBUSTNESS_UNCERTAINTY_INCLUSION_SCHEMA_NAME,
            "schema_version": PHASE8_ROBUSTNESS_UNCERTAINTY_INCLUSION_SCHEMA_VERSION,
            "inclusion_state": "not_included",
            "policy_reference": None,
        },
        "permitted_deviation_policy": "no_deviations_without_versioned_addendum",
        "no_tuning_declaration": True,
        "external_label_access_state": "unavailable_before_prediction_lock",
        "prediction_lock_requirement": "required_before_label_access",
    }
    preregistration_hash = sha256_json(payload)
    return phase8_external_preregistration_from_mapping(
        {"preregistration_hash": preregistration_hash, **payload}
    )


def run_phase8_external_preregistration_publication(
    *,
    freeze_artifact_path: Path,
    expected_freeze_artifact_hash: str,
    support_policy_artifact_path: Path,
    expected_support_policy_artifact_hash: str,
    output_root: Path,
) -> Phase8PreregistrationPublicationResult:
    """Verify the frozen Package D inputs and publish the guarded Package E preregistration.

    Reads exactly the two supplied frozen-decision JSON files after verifying their SHA-256 against
    the caller-supplied expected value (fail-closed on mismatch). Touches no other filesystem path
    except the output root. Never opens, lists, or stats the external dataset root.
    """

    freeze_hash = _verify_and_hash_input(
        freeze_artifact_path,
        expected_hash=expected_freeze_artifact_hash,
        label="freeze artifact",
    )
    support_policy_hash = _verify_and_hash_input(
        support_policy_artifact_path,
        expected_hash=expected_support_policy_artifact_hash,
        label="support policy artifact",
    )

    decision_freeze = phase8_decision_freeze_from_json(freeze_artifact_path.read_bytes())
    if (
        decision_freeze.freeze_state != "frozen"
        or not decision_freeze.frozen_before_external_evaluation
    ):
        raise Phase8PreregistrationPublicationError(
            "decision freeze inventory is not in a frozen, evaluation-eligible state."
        )
    _require_support_policy_matches_freeze(
        decision_freeze=decision_freeze,
        support_policy_artifact_path=support_policy_artifact_path,
    )

    preregistration = build_phase8_external_preregistration(decision_freeze=decision_freeze)

    canonical_output_root = _validate_output_root(output_root)
    _require_output_root_absent(canonical_output_root)
    output_path = canonical_output_root / PHASE8_EXTERNAL_PREREGISTRATION_FILENAME
    try:
        publish_text_no_overwrite(
            text=phase8_external_preregistration_to_json(preregistration).decode("utf-8"),
            output_path=output_path,
            temporary_exists_message="temporary Package E preregistration artifact already exists.",
            final_exists_message="Package E preregistration artifact already exists.",
        )
    except (Phase2PublicationExistingOutputError, Phase2PublicationIOError) as exc:
        raise Phase8PreregistrationPublicationError(
            "failed to publish Package E preregistration artifact."
        ) from exc

    return Phase8PreregistrationPublicationResult(
        output_root=canonical_output_root,
        output_path=output_path,
        preregistration_hash=preregistration.preregistration_hash,
        freeze_artifact_hash=freeze_hash,
        support_policy_artifact_hash=support_policy_hash,
    )


def sha256_json_bytes(data: bytes) -> str:
    """Return the lowercase SHA-256 hex digest of already-canonical JSON bytes."""

    return hashlib.sha256(data).hexdigest()


def _reference_payload(
    *,
    referenced_schema_name: str,
    artifact_hash: str | None,
    reference_state: str,
) -> dict[str, JsonValue]:
    return {
        "schema_name": PHASE8_ARTIFACT_REFERENCE_SCHEMA_NAME,
        "schema_version": PHASE8_ARTIFACT_REFERENCE_SCHEMA_VERSION,
        "referenced_schema_name": referenced_schema_name,
        "referenced_schema_version": "v1",
        "artifact_hash": artifact_hash,
        "reference_state": reference_state,
    }


def _require_matches_freeze(
    references_by_category: dict[str, FrozenDecisionReference],
    *,
    category: str,
    computed_hash: str,
) -> None:
    reference = references_by_category.get(category)
    if reference is None or reference.source_artifact_hash != computed_hash:
        raise Phase8PreregistrationPublicationError(
            f"committed {category} contract no longer matches the frozen decision inventory."
        )


def _require_support_policy_matches_freeze(
    *,
    decision_freeze: Phase8DecisionFreeze,
    support_policy_artifact_path: Path,
) -> None:
    references_by_category = {ref.category: ref for ref in decision_freeze.decision_references}
    frozen_reference = references_by_category.get("support_policy")
    support_policy = phase8_support_policy_from_mapping(
        json.loads(support_policy_artifact_path.read_bytes())
    )
    if support_policy.policy_type != REQUIRED_SUPPORT_POLICY_TYPE:
        raise Phase8PreregistrationPublicationError(
            f"support policy artifact policy_type must be {REQUIRED_SUPPORT_POLICY_TYPE!r}."
        )
    if (
        frozen_reference is None
        or frozen_reference.source_artifact_hash != support_policy.support_policy_hash
    ):
        raise Phase8PreregistrationPublicationError(
            "support policy artifact no longer matches the frozen decision inventory."
        )


def _verify_and_hash_input(path: Path, *, expected_hash: str, label: str) -> str:
    if not path.is_absolute():
        raise Phase8PreregistrationPublicationError(f"{label} path must be absolute.")
    if path.is_symlink():
        raise Phase8PreregistrationPublicationError(f"{label} path must not be a symlink.")
    if not path.is_file():
        raise Phase8PreregistrationPublicationError(f"{label} path must be a regular file.")
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise Phase8PreregistrationPublicationError(
            f"{label} SHA-256 does not match the expected frozen value."
        )
    return actual_hash


def _validate_output_root(output_root: Path) -> Path:
    if output_root.exists() and output_root.is_symlink():
        raise Phase8PreregistrationPublicationError("output_root must not be a symlink.")
    try:
        return validate_explicit_external_output_root(output_root, forbidden_roots=())
    except InvalidDatasetRootError as exc:
        raise Phase8PreregistrationPublicationError("invalid Package E output root.") from exc


def _require_output_root_absent(output_root: Path) -> None:
    if output_root.exists():
        raise Phase8PreregistrationPublicationError(
            "Package E output root already exists; refusing to overwrite or silently version."
        )


__all__ = [
    "PHASE8_PREREGISTRATION_EXTERNAL_COHORT_IDENTIFIER",
    "Phase8PreregistrationPublicationError",
    "Phase8PreregistrationPublicationResult",
    "build_phase8_external_preregistration",
    "run_phase8_external_preregistration_publication",
]
