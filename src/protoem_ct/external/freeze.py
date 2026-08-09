"""Phase 8 internal-evidence readiness audit and guarded freeze generation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

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
from protoem_ct.external.artifacts import PHASE8_REQUIRED_DECISION_CATEGORIES
from protoem_ct.external.statistical_policy import (
    build_phase8_statistical_policy_bundle,
    phase8_statistical_policy_component_hash,
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

PHASE8_READINESS_FILENAME: Final[str] = "phase8_internal_evidence_readiness.json"
PHASE8_DECISION_FREEZE_FILENAME: Final[str] = "phase8_decision_freeze.json"
PHASE8_EXTERNAL_PREREGISTRATION_FILENAME: Final[str] = "phase8_external_preregistration.json"
PHASE8_WAVE4_SUMMARY_FILENAME: Final[str] = "phase8_wave4_generation_summary.json"
PHASE8_READINESS_SCHEMA_NAME: Final[str] = "phase8_internal_evidence_readiness"
PHASE8_READINESS_SCHEMA_VERSION: Final[str] = "v1"
PHASE8_WAVE4_SUMMARY_SCHEMA_NAME: Final[str] = "phase8_wave4_generation_summary"
PHASE8_WAVE4_SUMMARY_SCHEMA_VERSION: Final[str] = "v1"

_WAVE2_REQUIRED: Final[tuple[str, ...]] = (
    PHASE8_WAVE2_LAYOUT_FILENAME,
    PHASE8_WAVE2_MANIFEST_FILENAME,
    PHASE8_WAVE2_QA_FILENAME,
    PHASE8_WAVE2_SUMMARY_FILENAME,
)
_WAVE3_REQUIRED: Final[tuple[str, ...]] = (
    PHASE8_WAVE3_COHORT_ACCOUNTING_FILENAME,
    PHASE8_WAVE3_DOMAIN_SHIFT_RECORD_FILENAME,
    PHASE8_WAVE3_ELIGIBILITY_POLICY_FILENAME,
    PHASE8_WAVE3_LABEL_MAPPING_POLICY_FILENAME,
    PHASE8_WAVE3_SUMMARY_FILENAME,
)


class Phase8FreezeError(ValueError):
    """Raised when Phase 8 Wave 4 readiness publication fails safely."""


@dataclass(frozen=True, slots=True)
class Phase8Wave4PublicationResult:
    """Result of one guarded Phase 8 Wave 4 readiness publication."""

    output_root: Path
    wave4_state: str
    readiness_hash: str
    summary_hash: str
    artifact_hashes: dict[str, str]
    freeze_generated: bool
    preregistration_generated: bool


def build_phase8_internal_evidence_readiness(
    *,
    wave2_artifact_root: Path,
    wave3_artifact_root: Path,
    internal_artifact_parent: Path,
    repository_root: Path,
) -> dict[str, JsonValue]:
    """Build a conservative readiness artifact from metadata-only development evidence."""

    wave2_root = _validate_input_root(wave2_artifact_root, required_filenames=_WAVE2_REQUIRED)
    wave3_root = _validate_input_root(wave3_artifact_root, required_filenames=_WAVE3_REQUIRED)
    internal_parent = _validate_internal_parent(internal_artifact_parent)
    repo_root = repository_root.resolve(strict=True)

    wave2_hashes = _hashes(wave2_root, _WAVE2_REQUIRED)
    wave3_hashes = _hashes(wave3_root, _WAVE3_REQUIRED)
    statistical_policy = build_phase8_statistical_policy_bundle()
    checkpoint_candidates = _checkpoint_candidates(internal_parent)
    categories = {
        "preprocessing_decision": _category(
            present=True,
            candidate_artifact=_relative_candidate(
                repo_root / "configs/phase6_protoem_ct.yaml", repo_root
            ),
            schema_version="phase6_protoem_ct/v1",
            identity_hash=_optional_file_hash(repo_root / "configs/phase6_protoem_ct.yaml"),
            development_only_provenance="repository_config_synthetic_mode_only",
            synthetic_status="synthetic_only",
            compatibility="incompatible_with_real_external_freeze",
            ambiguity=(
                "fixed internal preprocessing contract absent for selected external inference"
            ),
            freezeable=False,
            blockers=("real preprocessing decision linked to selected model is missing",),
        ),
        "support_policy": _category(
            present=False,
            candidate_artifact=None,
            schema_version=None,
            identity_hash=None,
            development_only_provenance="none_found",
            synthetic_status="missing",
            compatibility="unresolved_for_protoem_external_inference",
            ambiguity=(
                "external support-label use versus no-support/internal-only support policy is "
                "unresolved"
            ),
            freezeable=False,
            blockers=("immutable external-label-free support or no-support policy is missing",),
        ),
        "threshold_decision": _category(
            present=True,
            candidate_artifact=_relative_candidate(
                repo_root / "configs/phase6_protoem_ct.yaml", repo_root
            ),
            schema_version="phase6_protoem_ct/v1",
            identity_hash=_optional_file_hash(repo_root / "configs/phase6_protoem_ct.yaml"),
            development_only_provenance="repository_config_synthetic_mode_only",
            synthetic_status="synthetic_only",
            compatibility="incompatible_with_real_external_freeze",
            ambiguity=(
                "confidence threshold exists only in synthetic Phase 6 config, not a real selected "
                "model decision"
            ),
            freezeable=False,
            blockers=("real fixed threshold decision with development-only provenance is missing",),
        ),
        "checkpoint_metadata": _category(
            present=bool(checkpoint_candidates),
            candidate_artifact=(
                cast(str, checkpoint_candidates[0]["candidate_artifact"])
                if checkpoint_candidates
                else None
            ),
            schema_version=None,
            identity_hash=cast(str | None, checkpoint_candidates[0]["identity_hash"])
            if checkpoint_candidates
            else None,
            development_only_provenance="checkpoint_file_scan_only"
            if checkpoint_candidates
            else "none_found",
            synthetic_status="unresolved_candidate_requires_metadata"
            if checkpoint_candidates
            else "missing",
            compatibility="unresolved_for_protoem_external_inference",
            ambiguity=(
                "checkpoint metadata must prove real training, manifest/config hashes, method "
                "identity, mode, and selection rule"
            ),
            freezeable=False,
            blockers=(
                "real trained checkpoint metadata with development manifest/config hashes is "
                "missing",
                "selection rule and selected candidate are not recorded",
            ),
        ),
        "model_selection_decision": _category(
            present=False,
            candidate_artifact=None,
            schema_version=None,
            identity_hash=None,
            development_only_provenance="none_found",
            synthetic_status="missing",
            compatibility="unresolved_for_protoem_external_inference",
            ambiguity=(
                "no recorded selected candidate resolves multiple possible methods or checkpoints"
            ),
            freezeable=False,
            blockers=("real development-only model-selection artifact is missing",),
        ),
        "label_mapping_policy": _category(
            present=True,
            candidate_artifact=PHASE8_WAVE3_LABEL_MAPPING_POLICY_FILENAME,
            schema_version="phase8_label_mapping_policy/v1",
            identity_hash=wave3_hashes[PHASE8_WAVE3_LABEL_MAPPING_POLICY_FILENAME],
            development_only_provenance="wave3_prelabel_policy_only",
            synthetic_status="real_external_policy_expected_not_empirically_verified",
            compatibility="compatible_prelabel_binary_tumor_mapping_policy",
            ambiguity="label layout remains empirically unverified until authorized label ledger",
            freezeable=True,
            blockers=(),
        ),
        "metric_configuration": _category(
            present=True,
            candidate_artifact="repository_phase3_metric_contract_and_phase8_statistical_policy",
            schema_version="phase8_metric_configuration/v1",
            identity_hash=phase8_statistical_policy_component_hash("metric_configuration"),
            development_only_provenance="docs_decisions_phase3_metric_contract",
            synthetic_status="repository_contract_no_external_results",
            compatibility="compatible_with_binary_tumor_external_metrics",
            ambiguity=None,
            freezeable=False,
            blockers=(
                "metric configuration is newly proposed in Wave 4 and cannot be frozen without "
                "complete decision package",
            ),
        ),
        "bootstrap_configuration": _category(
            present=True,
            candidate_artifact="repository_phase8_statistical_policy",
            schema_version="phase8_bootstrap_configuration/v1",
            identity_hash=phase8_statistical_policy_component_hash("bootstrap_configuration"),
            development_only_provenance="phase8_preregistration_policy_no_external_results",
            synthetic_status="policy_only_no_results",
            compatibility="compatible_case_patient_level_external_ci",
            ambiguity=None,
            freezeable=False,
            blockers=(
                "bootstrap configuration is newly proposed in Wave 4 and cannot be frozen "
                "without complete decision package",
            ),
        ),
        "publication_configuration": _category(
            present=True,
            candidate_artifact="repository_phase8_statistical_policy",
            schema_version="phase8_publication_configuration/v1",
            identity_hash=phase8_statistical_policy_component_hash("publication_configuration"),
            development_only_provenance="phase8_preregistration_policy_no_external_results",
            synthetic_status="policy_only_no_results",
            compatibility="compatible_json_first_external_publication",
            ambiguity=None,
            freezeable=False,
            blockers=(
                "publication configuration is newly proposed in Wave 4 and cannot be frozen "
                "without complete decision package",
            ),
        ),
    }
    missing = sorted(set(PHASE8_REQUIRED_DECISION_CATEGORIES) - set(categories))
    if missing:
        raise Phase8FreezeError(f"internal readiness builder omitted categories: {missing!r}")
    blockers = sorted(
        {
            blocker
            for record in categories.values()
            for blocker in cast(list[str], record["blockers"])
        }
    )
    payload: dict[str, JsonValue] = {
        "all_categories_freezeable": not blockers,
        "approved_wave2_file_hashes": cast(dict[str, JsonValue], wave2_hashes),
        "approved_wave3_file_hashes": cast(dict[str, JsonValue], wave3_hashes),
        "category_readiness": cast(dict[str, JsonValue], categories),
        "checkpoint_candidate_count": len(checkpoint_candidates),
        "checkpoint_candidates": cast(list[JsonValue], checkpoint_candidates[:10]),
        "external_data_access": {
            "external_labels_read": False,
            "external_predictions_read": False,
            "raw_external_data_read": False,
            "raw_development_images_read": False,
        },
        "readiness_state": "READY" if not blockers else "BLOCKED",
        "schema_name": PHASE8_READINESS_SCHEMA_NAME,
        "schema_version": PHASE8_READINESS_SCHEMA_VERSION,
        "statistical_policy": statistical_policy,
        "unresolved_blockers": cast(list[JsonValue], blockers),
    }
    payload["readiness_hash"] = sha256_json(payload)
    return payload


def run_phase8_wave4_readiness_publication(
    *,
    wave2_artifact_root: Path,
    wave3_artifact_root: Path,
    internal_artifact_parent: Path,
    output_root: Path,
    repository_root: Path,
) -> Phase8Wave4PublicationResult:
    """Publish guarded Wave 4 readiness and stop unless every category is freezeable."""

    wave2_root = _validate_input_root(wave2_artifact_root, required_filenames=_WAVE2_REQUIRED)
    wave3_root = _validate_input_root(wave3_artifact_root, required_filenames=_WAVE3_REQUIRED)
    internal_parent = _validate_internal_parent(internal_artifact_parent)
    repo_root = repository_root.resolve(strict=True)
    canonical_output_root = _validate_output_root(
        output_root=output_root,
        forbidden_roots=(wave2_root, wave3_root, repo_root),
    )
    _require_output_root_empty_or_absent(canonical_output_root)
    readiness = build_phase8_internal_evidence_readiness(
        wave2_artifact_root=wave2_root,
        wave3_artifact_root=wave3_root,
        internal_artifact_parent=internal_parent,
        repository_root=repo_root,
    )
    _publish_json(canonical_output_root / PHASE8_READINESS_FILENAME, readiness)
    readiness_hash = sha256_file(canonical_output_root / PHASE8_READINESS_FILENAME)
    state = cast(str, readiness["readiness_state"])
    freeze_generated = state == "READY"
    preregistration_generated = state == "READY"
    if freeze_generated or preregistration_generated:
        raise Phase8FreezeError(
            "evaluation-ready freeze generation is not implemented for unresolved audits."
        )
    summary: dict[str, JsonValue] = {
        "artifact_hashes_before_summary": {PHASE8_READINESS_FILENAME: readiness_hash},
        "external_labels_read": False,
        "external_predictions_read": False,
        "freeze_generated": freeze_generated,
        "preregistration_generated": preregistration_generated,
        "raw_external_data_read": False,
        "readiness_hash": readiness_hash,
        "readiness_state": state,
        "schema_name": PHASE8_WAVE4_SUMMARY_SCHEMA_NAME,
        "schema_version": PHASE8_WAVE4_SUMMARY_SCHEMA_VERSION,
        "unresolved_blocker_count": len(cast(list[str], readiness["unresolved_blockers"])),
        "wave5_started": False,
    }
    summary["summary_identity_hash"] = sha256_json(summary)
    _publish_json(canonical_output_root / PHASE8_WAVE4_SUMMARY_FILENAME, summary)
    artifact_hashes = _artifact_hashes(canonical_output_root)
    return Phase8Wave4PublicationResult(
        output_root=canonical_output_root,
        wave4_state=state,
        readiness_hash=readiness_hash,
        summary_hash=artifact_hashes[PHASE8_WAVE4_SUMMARY_FILENAME],
        artifact_hashes=artifact_hashes,
        freeze_generated=freeze_generated,
        preregistration_generated=preregistration_generated,
    )


def _category(
    *,
    present: bool,
    candidate_artifact: str | None,
    schema_version: str | None,
    identity_hash: str | None,
    development_only_provenance: str,
    synthetic_status: str,
    compatibility: str,
    ambiguity: str | None,
    freezeable: bool,
    blockers: Iterable[str],
) -> dict[str, JsonValue]:
    return {
        "blockers": list(blockers),
        "candidate_artifact": candidate_artifact,
        "compatibility_with_intended_protoem_external_inference": compatibility,
        "development_only_provenance": development_only_provenance,
        "identity_or_file_hash": identity_hash,
        "present": present,
        "schema_version": schema_version,
        "synthetic_versus_real_data_status": synthetic_status,
        "unresolved_ambiguity": ambiguity,
        "legally_scientifically_freezeable_now": freezeable,
    }


def _validate_input_root(root: Path, *, required_filenames: tuple[str, ...]) -> Path:
    if not root.is_absolute():
        raise Phase8FreezeError("input artifact roots must be absolute paths.")
    if root.is_symlink():
        raise Phase8FreezeError("input artifact roots must not be symlinks.")
    resolved = root.resolve(strict=True)
    if not resolved.is_dir():
        raise Phase8FreezeError("input artifact root must be a directory.")
    for filename in required_filenames:
        path = resolved / filename
        if path.is_symlink() or not path.is_file():
            raise Phase8FreezeError("input artifact root is missing required regular JSON files.")
    return resolved


def _validate_internal_parent(path: Path) -> Path:
    if not path.is_absolute():
        raise Phase8FreezeError("internal_artifact_parent must be an absolute path.")
    if path.is_symlink():
        raise Phase8FreezeError("internal_artifact_parent must not be a symlink.")
    resolved = path.resolve(strict=True)
    if not resolved.is_dir():
        raise Phase8FreezeError("internal_artifact_parent must be a directory.")
    return resolved


def _validate_output_root(*, output_root: Path, forbidden_roots: tuple[Path, ...]) -> Path:
    if output_root.exists() and output_root.is_symlink():
        raise Phase8FreezeError("output_root must not be a symlink.")
    try:
        return validate_explicit_external_output_root(output_root, forbidden_roots=forbidden_roots)
    except InvalidDatasetRootError as exc:
        raise Phase8FreezeError("invalid Wave 4 output root.") from exc


def _require_output_root_empty_or_absent(output_root: Path) -> None:
    if output_root.exists() and any(output_root.iterdir()):
        raise Phase8FreezeError("Wave 4 output root already contains files.")


def _publish_json(path: Path, payload: dict[str, JsonValue]) -> None:
    try:
        publish_text_no_overwrite(
            text=(canonical_json_bytes(payload) + b"\n").decode("utf-8"),
            output_path=path,
            temporary_exists_message="temporary Wave 4 artifact already exists.",
            final_exists_message="Wave 4 artifact already exists.",
        )
    except (Phase2PublicationExistingOutputError, Phase2PublicationIOError) as exc:
        raise Phase8FreezeError("failed to publish Wave 4 JSON artifact.") from exc


def _hashes(root: Path, filenames: tuple[str, ...]) -> dict[str, str]:
    return {filename: sha256_file(root / filename) for filename in sorted(filenames)}


def _artifact_hashes(root: Path) -> dict[str, str]:
    return {path.name: sha256_file(path) for path in sorted(root.glob("*.json")) if path.is_file()}


def _optional_file_hash(path: Path) -> str | None:
    return sha256_file(path) if path.is_file() else None


def _relative_candidate(path: Path, repository_root: Path) -> str:
    try:
        return (
            path.resolve(strict=True).relative_to(repository_root.resolve(strict=True)).as_posix()
        )
    except (FileNotFoundError, ValueError):
        return path.name


def _checkpoint_candidates(root: Path) -> list[dict[str, JsonValue]]:
    candidates: list[dict[str, JsonValue]] = []
    for path in sorted(root.rglob("*")):
        if (
            path.is_symlink()
            or not path.is_file()
            or path.suffix.lower() not in {".ckpt", ".pt", ".pth"}
        ):
            continue
        candidates.append(
            {
                "candidate_artifact": path.name,
                "identity_hash": sha256_file(path),
                "metadata_status": "file_hash_only_not_sufficient_for_freeze",
            }
        )
        if len(candidates) >= 10:
            break
    return candidates


__all__ = [
    "PHASE8_DECISION_FREEZE_FILENAME",
    "PHASE8_EXTERNAL_PREREGISTRATION_FILENAME",
    "PHASE8_READINESS_FILENAME",
    "PHASE8_WAVE4_SUMMARY_FILENAME",
    "Phase8FreezeError",
    "Phase8Wave4PublicationResult",
    "build_phase8_internal_evidence_readiness",
    "run_phase8_wave4_readiness_publication",
]
