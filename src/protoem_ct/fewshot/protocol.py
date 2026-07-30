"""Deterministic Phase 4 adaptation-config construction and protocol-table generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePath
from typing import Final

from protoem_ct.artifacts.hashing import sha256_json
from protoem_ct.fewshot.artifacts import (
    FEWSHOT_ADAPTATION_CONFIG_VERSION,
    FEWSHOT_PROTOCOL_TABLE_VERSION,
    SUPPORTED_ADAPTATION_MODES,
    FewshotAdaptationConfig,
    FewshotInitializationReference,
    FewshotProtocolTable,
    FewshotProtocolTableRow,
    FewshotSupportManifest,
    hash_fewshot_support_manifest,
)
from protoem_ct.fewshot.support import (
    DEFAULT_REPLICATE_COUNT,
    build_default_support_replicate_plans,
)

FIXED_PROTOCOL_K_VALUES: Final[tuple[int, ...]] = (1, 2, 5, 10, 20)
FIXED_PROTOCOL_ADAPTATION_MODES: Final[tuple[str, ...]] = tuple(sorted(SUPPORTED_ADAPTATION_MODES))
PLANNED_PROTOCOL_RUN_STATUS: Final[str] = "planned"
FIXED_PROTOCOL_ROW_COUNT: Final[int] = (
    len(FIXED_PROTOCOL_K_VALUES) * DEFAULT_REPLICATE_COUNT * len(FIXED_PROTOCOL_ADAPTATION_MODES)
)
_PLANNED_OUTPUT_IDENTIFIER_VERSION: Final[str] = "fewshot_protocol_output_id_v1"
_DERIVED_SEED_VERSION: Final[str] = "fewshot_protocol_seed_v1"


class FewshotProtocolError(ValueError):
    """Base error for deterministic Phase 4 protocol construction."""


class FewshotProtocolValidationError(FewshotProtocolError):
    """Raised when support manifests, configs, or protocol rows are malformed."""


@dataclass(frozen=True, slots=True)
class FewshotAdaptationSettings:
    """Explicit bounded settings for deterministic Phase 4 adaptation configs."""

    baseline_family: str = "monai_segresnet"
    optimizer_name: str = "adam"
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    max_adaptation_steps: int = 8
    support_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    amp_enabled: bool = False


DEFAULT_FEWSHOT_ADAPTATION_SETTINGS = FewshotAdaptationSettings()


def build_fewshot_adaptation_config(
    support_manifest: FewshotSupportManifest,
    *,
    adaptation_mode: str,
    initialization_reference: FewshotInitializationReference,
    seed: int,
    settings: FewshotAdaptationSettings = DEFAULT_FEWSHOT_ADAPTATION_SETTINGS,
) -> FewshotAdaptationConfig:
    """Construct one deterministic adaptation-config artifact from explicit inputs."""

    _validate_support_manifest(support_manifest)
    if adaptation_mode not in SUPPORTED_ADAPTATION_MODES:
        raise FewshotProtocolValidationError(f"Unsupported adaptation_mode: {adaptation_mode!r}.")
    _require_seed(seed, field_name="seed")
    payload_without_hash = {
        "schema_version": FEWSHOT_ADAPTATION_CONFIG_VERSION,
        "adaptation_mode": adaptation_mode,
        "seed": seed,
        "baseline_family": settings.baseline_family,
        "initialization_reference": _initialization_reference_to_dict(initialization_reference),
        "optimizer_name": settings.optimizer_name,
        "learning_rate": settings.learning_rate,
        "weight_decay": settings.weight_decay,
        "max_adaptation_steps": settings.max_adaptation_steps,
        "support_batch_size": settings.support_batch_size,
        "gradient_accumulation_steps": settings.gradient_accumulation_steps,
        "amp_enabled": settings.amp_enabled,
    }
    artifact_hash = sha256_json(payload_without_hash)
    return FewshotAdaptationConfig(
        artifact_hash=artifact_hash,
        schema_version=FEWSHOT_ADAPTATION_CONFIG_VERSION,
        adaptation_mode=adaptation_mode,
        seed=seed,
        baseline_family=settings.baseline_family,
        initialization_reference=initialization_reference,
        optimizer_name=settings.optimizer_name,
        learning_rate=settings.learning_rate,
        weight_decay=settings.weight_decay,
        max_adaptation_steps=settings.max_adaptation_steps,
        support_batch_size=settings.support_batch_size,
        gradient_accumulation_steps=settings.gradient_accumulation_steps,
        amp_enabled=settings.amp_enabled,
    )


def build_all_fewshot_adaptation_configs(
    support_manifests_by_k: dict[int, tuple[FewshotSupportManifest, ...]],
    *,
    initialization_reference: FewshotInitializationReference,
    base_seed: int,
    settings: FewshotAdaptationSettings = DEFAULT_FEWSHOT_ADAPTATION_SETTINGS,
    adaptation_modes: tuple[str, ...] = FIXED_PROTOCOL_ADAPTATION_MODES,
) -> dict[tuple[int, str, str], FewshotAdaptationConfig]:
    """Construct all deterministic configs for the fixed manifest set."""

    normalized_manifests = _validate_support_manifest_set(support_manifests_by_k)
    _require_seed(base_seed, field_name="base_seed")
    _validate_adaptation_modes(adaptation_modes)
    configs: dict[tuple[int, str, str], FewshotAdaptationConfig] = {}
    for support_manifest in normalized_manifests:
        for adaptation_mode in adaptation_modes:
            key = (
                support_manifest.requested_k,
                support_manifest.replicate_id,
                adaptation_mode,
            )
            if key in configs:
                raise FewshotProtocolValidationError(
                    f"Duplicate adaptation-config identity detected for {key!r}."
                )
            configs[key] = build_fewshot_adaptation_config(
                support_manifest,
                adaptation_mode=adaptation_mode,
                initialization_reference=initialization_reference,
                seed=_derive_adaptation_seed(
                    support_manifest=support_manifest,
                    adaptation_mode=adaptation_mode,
                    base_seed=base_seed,
                ),
                settings=settings,
            )
    return configs


def build_fewshot_protocol_table(
    support_manifests_by_k: dict[int, tuple[FewshotSupportManifest, ...]],
    *,
    initialization_reference: FewshotInitializationReference,
    base_seed: int,
    settings: FewshotAdaptationSettings = DEFAULT_FEWSHOT_ADAPTATION_SETTINGS,
    adaptation_modes: tuple[str, ...] = FIXED_PROTOCOL_ADAPTATION_MODES,
) -> FewshotProtocolTable:
    """Construct and validate the complete deterministic protocol table."""

    normalized_manifests = _validate_support_manifest_set(support_manifests_by_k)
    configs = build_all_fewshot_adaptation_configs(
        support_manifests_by_k,
        initialization_reference=initialization_reference,
        base_seed=base_seed,
        settings=settings,
        adaptation_modes=adaptation_modes,
    )
    rows: list[FewshotProtocolTableRow] = []
    support_identity_by_hash: dict[str, tuple[int, str]] = {}
    for support_manifest in normalized_manifests:
        support_identity = (
            support_manifest.requested_k,
            support_manifest.replicate_id,
        )
        existing_identity = support_identity_by_hash.get(support_manifest.artifact_hash)
        if existing_identity is None:
            support_identity_by_hash[support_manifest.artifact_hash] = support_identity
        elif existing_identity != support_identity:
            raise FewshotProtocolValidationError(
                "support_manifest_hash must not identify different support-manifest identities."
            )
        for adaptation_mode in adaptation_modes:
            key = (
                support_manifest.requested_k,
                support_manifest.replicate_id,
                adaptation_mode,
            )
            config = configs[key]
            row = FewshotProtocolTableRow(
                requested_k=support_manifest.requested_k,
                replicate_id=support_manifest.replicate_id,
                support_manifest_hash=support_manifest.artifact_hash,
                adaptation_mode=adaptation_mode,
                adaptation_config_hash=config.artifact_hash,
                initialization_reference=initialization_reference,
                planned_output_identifier=_build_planned_output_identifier(
                    support_manifest=support_manifest,
                    adaptation_mode=adaptation_mode,
                    adaptation_config=config,
                ),
                run_status=PLANNED_PROTOCOL_RUN_STATUS,
            )
            _validate_planned_output_identifier(row.planned_output_identifier, support_manifest)
            rows.append(row)
    normalized_rows = tuple(sorted(rows, key=_protocol_row_sort_key))
    _validate_protocol_rows(normalized_rows, normalized_manifests, adaptation_modes)
    first_manifest = normalized_manifests[0]
    payload_without_hash = {
        "schema_version": FEWSHOT_PROTOCOL_TABLE_VERSION,
        "source_development_manifest_hash": first_manifest.source_development_manifest_hash,
        "source_development_split_hash": first_manifest.source_development_split_hash,
        "source_lesion_summary_hash": first_manifest.source_lesion_summary_hash,
        "immutable_test_cohort_hash": first_manifest.immutable_test_cohort_hash,
        "rows": [_protocol_row_to_dict(row) for row in normalized_rows],
    }
    artifact_hash = sha256_json(payload_without_hash)
    return FewshotProtocolTable(
        artifact_hash=artifact_hash,
        schema_version=FEWSHOT_PROTOCOL_TABLE_VERSION,
        source_development_manifest_hash=first_manifest.source_development_manifest_hash,
        source_development_split_hash=first_manifest.source_development_split_hash,
        source_lesion_summary_hash=first_manifest.source_lesion_summary_hash,
        immutable_test_cohort_hash=first_manifest.immutable_test_cohort_hash,
        rows=normalized_rows,
    )


def _validate_support_manifest_set(
    support_manifests_by_k: dict[int, tuple[FewshotSupportManifest, ...]],
) -> tuple[FewshotSupportManifest, ...]:
    if tuple(sorted(support_manifests_by_k)) != FIXED_PROTOCOL_K_VALUES:
        raise FewshotProtocolValidationError(
            f"support_manifests_by_k must contain exactly {FIXED_PROTOCOL_K_VALUES!r}."
        )
    normalized: list[FewshotSupportManifest] = []
    shared_manifest_hash: str | None = None
    shared_split_hash: str | None = None
    shared_lesion_hash: str | None = None
    shared_immutable_hash: str | None = None
    seen_identities: set[tuple[int, str]] = set()
    seen_hashes: set[str] = set()
    for requested_k in FIXED_PROTOCOL_K_VALUES:
        manifests = support_manifests_by_k[requested_k]
        if len(manifests) != DEFAULT_REPLICATE_COUNT:
            raise FewshotProtocolValidationError(
                f"requested_k={requested_k} must have exactly {DEFAULT_REPLICATE_COUNT} manifests."
            )
        expected_replicate_ids = tuple(
            plan.replicate_id for plan in build_default_support_replicate_plans(requested_k)
        )
        actual_replicate_ids = tuple(sorted(item.replicate_id for item in manifests))
        if actual_replicate_ids != expected_replicate_ids:
            raise FewshotProtocolValidationError(
                f"requested_k={requested_k} must use replicate IDs {expected_replicate_ids!r}."
            )
        for support_manifest in sorted(
            manifests,
            key=lambda item: (item.requested_k, item.replicate_id, item.artifact_hash),
        ):
            _validate_support_manifest(support_manifest)
            if support_manifest.requested_k != requested_k:
                raise FewshotProtocolValidationError(
                    "Support manifest requested_k must match its support_manifests_by_k key."
                )
            identity = (support_manifest.requested_k, support_manifest.replicate_id)
            if identity in seen_identities:
                raise FewshotProtocolValidationError(
                    f"Duplicate support-manifest identity detected for {identity!r}."
                )
            if support_manifest.artifact_hash in seen_hashes:
                raise FewshotProtocolValidationError(
                    "Support-manifest hashes must be unique across the fixed manifest set."
                )
            seen_identities.add(identity)
            seen_hashes.add(support_manifest.artifact_hash)
            shared_manifest_hash = _require_shared_value(
                shared_manifest_hash,
                support_manifest.source_development_manifest_hash,
                field_name="source_development_manifest_hash",
            )
            shared_split_hash = _require_shared_value(
                shared_split_hash,
                support_manifest.source_development_split_hash,
                field_name="source_development_split_hash",
            )
            shared_lesion_hash = _require_shared_optional_value(
                shared_lesion_hash,
                support_manifest.source_lesion_summary_hash,
                field_name="source_lesion_summary_hash",
            )
            shared_immutable_hash = _require_shared_value(
                shared_immutable_hash,
                support_manifest.immutable_test_cohort_hash,
                field_name="immutable_test_cohort_hash",
            )
            normalized.append(support_manifest)
    return tuple(normalized)


def _validate_support_manifest(support_manifest: FewshotSupportManifest) -> None:
    expected_hash = hash_fewshot_support_manifest(support_manifest)
    if expected_hash != support_manifest.artifact_hash:
        raise FewshotProtocolValidationError(
            "Support manifest artifact_hash does not match its deterministic content."
        )


def _validate_adaptation_modes(adaptation_modes: tuple[str, ...]) -> None:
    if tuple(sorted(adaptation_modes)) != FIXED_PROTOCOL_ADAPTATION_MODES:
        raise FewshotProtocolValidationError(
            f"adaptation_modes must equal {FIXED_PROTOCOL_ADAPTATION_MODES!r}."
        )


def _derive_adaptation_seed(
    *,
    support_manifest: FewshotSupportManifest,
    adaptation_mode: str,
    base_seed: int,
) -> int:
    digest = sha256_json(
        {
            "derivation_version": _DERIVED_SEED_VERSION,
            "base_seed": base_seed,
            "requested_k": support_manifest.requested_k,
            "replicate_id": support_manifest.replicate_id,
            "support_manifest_hash": support_manifest.artifact_hash,
            "adaptation_mode": adaptation_mode,
        }
    )
    return int(digest[:16], 16)


def _build_planned_output_identifier(
    *,
    support_manifest: FewshotSupportManifest,
    adaptation_mode: str,
    adaptation_config: FewshotAdaptationConfig,
) -> str:
    digest = sha256_json(
        {
            "identifier_version": _PLANNED_OUTPUT_IDENTIFIER_VERSION,
            "requested_k": support_manifest.requested_k,
            "replicate_id": support_manifest.replicate_id,
            "support_manifest_hash": support_manifest.artifact_hash,
            "adaptation_mode": adaptation_mode,
            "adaptation_config_hash": adaptation_config.artifact_hash,
        }
    )
    return (
        f"fs_k{support_manifest.requested_k:02d}_"
        f"{support_manifest.replicate_id}_"
        f"{adaptation_mode}_"
        f"{digest[:16]}"
    )


def _validate_planned_output_identifier(
    identifier: str,
    support_manifest: FewshotSupportManifest,
) -> None:
    path = PurePath(identifier)
    if path.is_absolute() or "/" in identifier or "\\" in identifier or ":" in identifier:
        raise FewshotProtocolValidationError(
            "planned_output_identifier must not contain an absolute or filesystem path."
        )
    for assignment in support_manifest.assignments:
        if assignment.anonymous_patient_id in identifier:
            raise FewshotProtocolValidationError(
                "planned_output_identifier must not contain anonymous_patient_id values."
            )
        if assignment.anonymous_case_id in identifier:
            raise FewshotProtocolValidationError(
                "planned_output_identifier must not contain anonymous_case_id values."
            )


def _validate_protocol_rows(
    rows: tuple[FewshotProtocolTableRow, ...],
    support_manifests: tuple[FewshotSupportManifest, ...],
    adaptation_modes: tuple[str, ...],
) -> None:
    if len(rows) != FIXED_PROTOCOL_ROW_COUNT:
        raise FewshotProtocolValidationError(
            f"Protocol table must contain exactly {FIXED_PROTOCOL_ROW_COUNT} rows."
        )
    if tuple(sorted(rows, key=_protocol_row_sort_key)) != rows:
        raise FewshotProtocolValidationError(
            "Protocol table rows must be in canonical deterministic order."
        )
    if len({row.planned_output_identifier for row in rows}) != len(rows):
        raise FewshotProtocolValidationError(
            "Protocol table planned_output_identifier values must be unique."
        )
    expected_combinations = {
        (support_manifest.requested_k, support_manifest.replicate_id, adaptation_mode)
        for support_manifest in support_manifests
        for adaptation_mode in adaptation_modes
    }
    actual_combinations = {(row.requested_k, row.replicate_id, row.adaptation_mode) for row in rows}
    if actual_combinations != expected_combinations:
        raise FewshotProtocolValidationError(
            "Protocol table must contain every required K/replicate/adaptation-mode combination "
            "exactly once."
        )
    if len(actual_combinations) != len(rows):
        raise FewshotProtocolValidationError("Protocol table rows must be unique.")
    manifest_hash_by_identity = {
        (
            support_manifest.requested_k,
            support_manifest.replicate_id,
        ): support_manifest.artifact_hash
        for support_manifest in support_manifests
    }
    for row in rows:
        expected_manifest_hash = manifest_hash_by_identity[(row.requested_k, row.replicate_id)]
        if row.support_manifest_hash != expected_manifest_hash:
            raise FewshotProtocolValidationError(
                "Protocol row support_manifest_hash must match the fixed support-manifest identity."
            )
        if row.run_status != PLANNED_PROTOCOL_RUN_STATUS:
            raise FewshotProtocolValidationError(
                f"Protocol row run_status must equal {PLANNED_PROTOCOL_RUN_STATUS!r}."
            )


def _protocol_row_sort_key(row: FewshotProtocolTableRow) -> tuple[object, ...]:
    return (
        row.requested_k,
        row.replicate_id,
        row.adaptation_mode,
        row.planned_output_identifier,
        row.support_manifest_hash,
        row.adaptation_config_hash,
        row.initialization_reference.reference_type,
        row.initialization_reference.reference_identifier,
        row.run_status,
    )


def _protocol_row_to_dict(row: FewshotProtocolTableRow) -> dict[str, object]:
    return {
        "requested_k": row.requested_k,
        "replicate_id": row.replicate_id,
        "support_manifest_hash": row.support_manifest_hash,
        "adaptation_mode": row.adaptation_mode,
        "adaptation_config_hash": row.adaptation_config_hash,
        "initialization_reference": _initialization_reference_to_dict(row.initialization_reference),
        "planned_output_identifier": row.planned_output_identifier,
        "run_status": row.run_status,
    }


def _initialization_reference_to_dict(
    reference: FewshotInitializationReference,
) -> dict[str, object]:
    return {
        "reference_type": reference.reference_type,
        "reference_identifier": reference.reference_identifier,
        "artifact_sha256": reference.artifact_sha256,
        "checkpoint_sha256": reference.checkpoint_sha256,
    }


def _require_seed(value: int, *, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise FewshotProtocolValidationError(f"{field_name} must be a nonnegative integer.")


def _require_shared_value(
    current: str | None,
    new_value: str,
    *,
    field_name: str,
) -> str:
    if current is None:
        return new_value
    if current != new_value:
        raise FewshotProtocolValidationError(
            f"All support manifests must share the same {field_name}."
        )
    return current


def _require_shared_optional_value(
    current: str | None,
    new_value: str | None,
    *,
    field_name: str,
) -> str | None:
    if current is None and new_value is None:
        return None
    if current is None:
        return new_value
    if current != new_value:
        raise FewshotProtocolValidationError(
            f"All support manifests must share the same {field_name}."
        )
    return current


__all__ = [
    "DEFAULT_FEWSHOT_ADAPTATION_SETTINGS",
    "FIXED_PROTOCOL_ADAPTATION_MODES",
    "FIXED_PROTOCOL_K_VALUES",
    "FIXED_PROTOCOL_ROW_COUNT",
    "PLANNED_PROTOCOL_RUN_STATUS",
    "FewshotAdaptationSettings",
    "FewshotProtocolError",
    "FewshotProtocolValidationError",
    "build_all_fewshot_adaptation_configs",
    "build_fewshot_adaptation_config",
    "build_fewshot_protocol_table",
]
