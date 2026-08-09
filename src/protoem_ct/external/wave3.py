"""Phase 8 Wave 3 policy, domain-shift, and eligibility publication."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

from protoem_ct.artifacts.hashing import JsonValue, canonical_json_bytes, sha256_file
from protoem_ct.data._phase2_publication import (
    Phase2PublicationExistingOutputError,
    Phase2PublicationIOError,
    publish_text_no_overwrite,
)
from protoem_ct.data.phase2_paths import (
    InvalidDatasetRootError,
    validate_explicit_external_output_root,
)
from protoem_ct.external.domain_shift import (
    Phase8DomainShiftRecord,
    build_phase8_domain_shift_record,
    phase8_domain_shift_record_to_json,
)
from protoem_ct.external.eligibility import (
    Phase8EligibilityAccounting,
    Phase8EligibilityPolicy,
    build_phase8_eligibility_policy,
    build_phase8_prelabel_eligibility_accounting,
    phase8_eligibility_accounting_to_json,
    phase8_eligibility_policy_to_json,
)
from protoem_ct.external.label_mapping import (
    Phase8LabelMappingPolicy,
    build_default_phase8_label_mapping_policy,
    phase8_label_mapping_policy_to_json,
)
from protoem_ct.external.manifest import (
    Phase8ExternalImageManifest,
    phase8_external_image_manifest_from_json,
)
from protoem_ct.external.wave2 import (
    PHASE8_WAVE2_LAYOUT_FILENAME,
    PHASE8_WAVE2_MANIFEST_FILENAME,
    PHASE8_WAVE2_QA_FILENAME,
    PHASE8_WAVE2_SUMMARY_FILENAME,
)

PHASE8_WAVE3_LABEL_MAPPING_POLICY_FILENAME: Final[str] = "phase8_external_label_mapping_policy.json"
PHASE8_WAVE3_DOMAIN_SHIFT_RECORD_FILENAME: Final[str] = "phase8_external_domain_shift_record.json"
PHASE8_WAVE3_ELIGIBILITY_POLICY_FILENAME: Final[str] = "phase8_external_eligibility_policy.json"
PHASE8_WAVE3_COHORT_ACCOUNTING_FILENAME: Final[str] = "phase8_external_cohort_accounting.json"
PHASE8_WAVE3_SUMMARY_FILENAME: Final[str] = "phase8_wave3_generation_summary.json"
PHASE8_WAVE3_SUMMARY_SCHEMA_NAME: Final[str] = "phase8_wave3_generation_summary"
PHASE8_WAVE3_SUMMARY_SCHEMA_VERSION: Final[str] = "v1"

_WAVE2_REQUIRED_FILENAMES: Final[tuple[str, ...]] = (
    PHASE8_WAVE2_LAYOUT_FILENAME,
    PHASE8_WAVE2_MANIFEST_FILENAME,
    PHASE8_WAVE2_QA_FILENAME,
    PHASE8_WAVE2_SUMMARY_FILENAME,
)


class Phase8Wave3PublicationError(ValueError):
    """Raised when Phase 8 Wave 3 publication fails safely."""


@dataclass(frozen=True, slots=True)
class Phase8Wave3PublicationResult:
    """Aggregate result for one Wave 3 policy publication."""

    output_root: Path
    case_count: int
    discovered_count: int
    image_qa_eligible_count: int
    inference_eligible_count: int
    label_compatibility_pending_count: int
    evaluation_eligible_count: int
    excluded_count: int
    deferred_count: int
    artifact_hashes: dict[str, str]
    label_mapping_policy: Phase8LabelMappingPolicy
    eligibility_policy: Phase8EligibilityPolicy
    domain_shift_record: Phase8DomainShiftRecord
    cohort_accounting: Phase8EligibilityAccounting


def run_phase8_wave3_policy_publication(
    *,
    wave2_artifact_root: Path,
    output_root: Path,
    repository_root: Path,
) -> Phase8Wave3PublicationResult:
    """Publish Wave 3 real policy artifacts from persisted Wave 2 image-only JSON."""

    canonical_wave2_root = _validate_wave2_artifact_root(wave2_artifact_root)
    canonical_output_root = _validate_wave3_output_root(
        output_root=output_root,
        wave2_artifact_root=canonical_wave2_root,
        repository_root=repository_root,
    )
    _require_output_root_empty_or_absent(canonical_output_root)
    input_hashes = _wave2_input_hashes(canonical_wave2_root)
    manifest = phase8_external_image_manifest_from_json(
        (canonical_wave2_root / PHASE8_WAVE2_MANIFEST_FILENAME).read_bytes()
    )
    _validate_wave2_supporting_artifacts(canonical_wave2_root, manifest)

    label_mapping_policy = build_default_phase8_label_mapping_policy()
    eligibility_policy = build_phase8_eligibility_policy()
    domain_shift_record = build_phase8_domain_shift_record(external_manifest=manifest)
    cohort_accounting = build_phase8_prelabel_eligibility_accounting(
        image_manifest=manifest,
        preregistered_policy_hash=eligibility_policy.policy_hash,
    )
    artifact_payloads = {
        PHASE8_WAVE3_LABEL_MAPPING_POLICY_FILENAME: phase8_label_mapping_policy_to_json(
            label_mapping_policy
        ),
        PHASE8_WAVE3_DOMAIN_SHIFT_RECORD_FILENAME: phase8_domain_shift_record_to_json(
            domain_shift_record
        ),
        PHASE8_WAVE3_ELIGIBILITY_POLICY_FILENAME: phase8_eligibility_policy_to_json(
            eligibility_policy
        ),
        PHASE8_WAVE3_COHORT_ACCOUNTING_FILENAME: phase8_eligibility_accounting_to_json(
            cohort_accounting
        ),
    }
    for filename, data in artifact_payloads.items():
        _publish_json_bytes(canonical_output_root / filename, data)
    output_hashes_without_summary = _artifact_hashes(canonical_output_root)
    summary_payload = _summary_payload(
        input_hashes=input_hashes,
        output_hashes_without_summary=output_hashes_without_summary,
        manifest=manifest,
        label_mapping_policy=label_mapping_policy,
        eligibility_policy=eligibility_policy,
        domain_shift_record=domain_shift_record,
        cohort_accounting=cohort_accounting,
    )
    _publish_json_bytes(
        canonical_output_root / PHASE8_WAVE3_SUMMARY_FILENAME,
        canonical_json_bytes(summary_payload) + b"\n",
    )
    artifact_hashes = _artifact_hashes(canonical_output_root)
    return Phase8Wave3PublicationResult(
        output_root=canonical_output_root,
        case_count=cohort_accounting.case_count,
        discovered_count=cohort_accounting.discovered_count,
        image_qa_eligible_count=cohort_accounting.image_qa_eligible_count,
        inference_eligible_count=cohort_accounting.inference_eligible_count,
        label_compatibility_pending_count=cohort_accounting.label_compatibility_pending_count,
        evaluation_eligible_count=cohort_accounting.evaluation_eligible_count,
        excluded_count=cohort_accounting.excluded_count,
        deferred_count=cohort_accounting.deferred_count,
        artifact_hashes=artifact_hashes,
        label_mapping_policy=label_mapping_policy,
        eligibility_policy=eligibility_policy,
        domain_shift_record=domain_shift_record,
        cohort_accounting=cohort_accounting,
    )


def _validate_wave2_artifact_root(wave2_artifact_root: Path) -> Path:
    if not wave2_artifact_root.is_absolute():
        raise Phase8Wave3PublicationError("wave2_artifact_root must be an absolute path.")
    if wave2_artifact_root.is_symlink():
        raise Phase8Wave3PublicationError("wave2_artifact_root must not be a symlink.")
    resolved = wave2_artifact_root.resolve(strict=True)
    if not resolved.is_dir():
        raise Phase8Wave3PublicationError("wave2_artifact_root must be a directory.")
    for filename in _WAVE2_REQUIRED_FILENAMES:
        path = resolved / filename
        if path.is_symlink() or not path.is_file():
            raise Phase8Wave3PublicationError(
                "wave2_artifact_root is missing a required regular Wave 2 JSON artifact."
            )
    return resolved


def _validate_wave3_output_root(
    *,
    output_root: Path,
    wave2_artifact_root: Path,
    repository_root: Path,
) -> Path:
    if output_root.exists() and output_root.is_symlink():
        raise Phase8Wave3PublicationError("output_root must not be a symlink.")
    try:
        return validate_explicit_external_output_root(
            output_root,
            forbidden_roots=(wave2_artifact_root, repository_root),
        )
    except InvalidDatasetRootError as exc:
        raise Phase8Wave3PublicationError("invalid Wave 3 output root.") from exc


def _require_output_root_empty_or_absent(output_root: Path) -> None:
    if output_root.exists() and any(output_root.iterdir()):
        raise Phase8Wave3PublicationError("Wave 3 output root already contains files.")


def _wave2_input_hashes(wave2_artifact_root: Path) -> dict[str, str]:
    return {
        filename: sha256_file(wave2_artifact_root / filename)
        for filename in sorted(_WAVE2_REQUIRED_FILENAMES)
    }


def _validate_wave2_supporting_artifacts(
    wave2_artifact_root: Path,
    manifest: Phase8ExternalImageManifest,
) -> None:
    qa_collection = _read_json_mapping(wave2_artifact_root / PHASE8_WAVE2_QA_FILENAME)
    if qa_collection.get("schema_name") != "phase8_external_image_qa_collection":
        raise Phase8Wave3PublicationError("Wave 2 QA artifact has an unexpected schema.")
    if qa_collection.get("case_count") != manifest.case_count:
        raise Phase8Wave3PublicationError("Wave 2 QA case count does not match manifest.")
    layout = _read_json_mapping(wave2_artifact_root / PHASE8_WAVE2_LAYOUT_FILENAME)
    if layout.get("schema_name") != "phase8_ircadb_image_layout":
        raise Phase8Wave3PublicationError("Wave 2 layout artifact has an unexpected schema.")
    if layout.get("case_count") != manifest.case_count:
        raise Phase8Wave3PublicationError("Wave 2 layout case count does not match manifest.")
    summary = _read_json_mapping(wave2_artifact_root / PHASE8_WAVE2_SUMMARY_FILENAME)
    if summary.get("schema_name") != "phase8_wave2_generation_summary":
        raise Phase8Wave3PublicationError("Wave 2 generation summary has an unexpected schema.")
    if summary.get("manifest_hash") != manifest.manifest_hash:
        raise Phase8Wave3PublicationError("Wave 2 summary manifest hash does not match manifest.")
    if summary.get("label_accessed") is not False:
        raise Phase8Wave3PublicationError("Wave 2 summary must report no label access.")


def _summary_payload(
    *,
    input_hashes: dict[str, str],
    output_hashes_without_summary: dict[str, str],
    manifest: Phase8ExternalImageManifest,
    label_mapping_policy: Phase8LabelMappingPolicy,
    eligibility_policy: Phase8EligibilityPolicy,
    domain_shift_record: Phase8DomainShiftRecord,
    cohort_accounting: Phase8EligibilityAccounting,
) -> dict[str, JsonValue]:
    return {
        "artifact_hashes_before_summary": cast(dict[str, JsonValue], output_hashes_without_summary),
        "case_count": cohort_accounting.case_count,
        "cohort_accounting_hash": cohort_accounting.accounting_hash,
        "deferred_count": cohort_accounting.deferred_count,
        "domain_shift_record_hash": domain_shift_record.domain_shift_record_hash,
        "eligibility_policy_hash": eligibility_policy.policy_hash,
        "evaluation_eligible_count": cohort_accounting.evaluation_eligible_count,
        "external_image_manifest_hash": manifest.manifest_hash,
        "image_qa_eligible_count": cohort_accounting.image_qa_eligible_count,
        "inference_eligible_count": cohort_accounting.inference_eligible_count,
        "input_wave2_artifact_hashes": cast(dict[str, JsonValue], input_hashes),
        "label_accessed": False,
        "label_compatibility_pending_count": (cohort_accounting.label_compatibility_pending_count),
        "label_mapping_policy_hash": label_mapping_policy.policy_hash,
        "raw_data_modified": False,
        "schema_name": PHASE8_WAVE3_SUMMARY_SCHEMA_NAME,
        "schema_version": PHASE8_WAVE3_SUMMARY_SCHEMA_VERSION,
        "wave4_released": False,
    }


def _read_json_mapping(path: Path) -> dict[str, JsonValue]:
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Phase8Wave3PublicationError("failed to read required Wave 2 JSON artifact.") from exc
    if not isinstance(decoded, dict):
        raise Phase8Wave3PublicationError("required Wave 2 JSON artifact root must be an object.")
    return cast(dict[str, JsonValue], decoded)


def _publish_json_bytes(path: Path, data: bytes) -> None:
    try:
        publish_text_no_overwrite(
            text=data.decode("utf-8"),
            output_path=path,
            temporary_exists_message="temporary Phase 8 Wave 3 output already exists",
            final_exists_message="Phase 8 Wave 3 output already exists",
        )
    except (Phase2PublicationExistingOutputError, Phase2PublicationIOError) as exc:
        raise Phase8Wave3PublicationError("failed to publish Wave 3 artifact.") from exc


def _artifact_hashes(output_root: Path) -> dict[str, str]:
    return {
        path.name: sha256_file(path)
        for path in sorted(output_root.iterdir(), key=lambda item: item.name)
        if path.is_file()
    }


__all__ = [
    "PHASE8_WAVE3_COHORT_ACCOUNTING_FILENAME",
    "PHASE8_WAVE3_DOMAIN_SHIFT_RECORD_FILENAME",
    "PHASE8_WAVE3_ELIGIBILITY_POLICY_FILENAME",
    "PHASE8_WAVE3_LABEL_MAPPING_POLICY_FILENAME",
    "PHASE8_WAVE3_SUMMARY_FILENAME",
    "PHASE8_WAVE3_SUMMARY_SCHEMA_NAME",
    "PHASE8_WAVE3_SUMMARY_SCHEMA_VERSION",
    "Phase8Wave3PublicationError",
    "Phase8Wave3PublicationResult",
    "run_phase8_wave3_policy_publication",
]
