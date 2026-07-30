"""Deterministic Phase 4 filesystem publication and CLI generation helpers."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

from omegaconf import DictConfig, OmegaConf

from protoem_ct.artifacts import (
    DatasetManifest,
    DevelopmentSplitManifest,
    LesionComponentsArtifact,
    Phase2ArtifactError,
    canonical_json_bytes,
    phase2_artifact_from_json,
)
from protoem_ct.data import InvalidDatasetRootError, validate_explicit_external_output_root
from protoem_ct.fewshot.artifacts import (
    FewshotAdaptationConfig,
    FewshotInitializationReference,
    FewshotProtocolTable,
    FewshotSupportManifest,
    fewshot_adaptation_config_to_json,
    fewshot_protocol_table_from_json,
    fewshot_protocol_table_to_json,
    fewshot_support_manifest_to_json,
)
from protoem_ct.fewshot.protocol import (
    DEFAULT_FEWSHOT_ADAPTATION_SETTINGS,
    FIXED_PROTOCOL_ADAPTATION_MODES,
    FIXED_PROTOCOL_K_VALUES,
    FIXED_PROTOCOL_ROW_COUNT,
    FewshotAdaptationSettings,
    FewshotProtocolValidationError,
    build_all_fewshot_adaptation_configs,
    build_fewshot_protocol_table,
)
from protoem_ct.fewshot.support import (
    DEFAULT_REPLICATE_COUNT,
    generate_complete_fixed_support_manifest_set,
)

_PHASE4_CONFIG_ROOT_KEY: Final[str] = "phase4_fewshot_protocol"
_PHASE4_EXPECTED_CONFIG_KEYS: Final[frozenset[str]] = frozenset(
    {
        "amp_enabled",
        "baseline_family",
        "gradient_accumulation_steps",
        "learning_rate",
        "max_adaptation_steps",
        "optimizer_name",
        "support_batch_size",
        "weight_decay",
    }
)
_GENERATION_SUMMARY_VERSION: Final[str] = "phase4_protocol_generation_summary_v1"
_PROTOCOL_MARKDOWN_NAME: Final[str] = "fewshot_protocol_table.md"
_PROTOCOL_JSON_NAME: Final[str] = "fewshot_protocol_table.json"
_GENERATION_SUMMARY_NAME: Final[str] = "generation_summary.json"


class FewshotPublicationError(ValueError):
    """Base error for Phase 4 publication and CLI generation."""


class FewshotPublicationConfigError(FewshotPublicationError):
    """Raised when the Phase 4 config file is missing or invalid."""


class FewshotPublicationPathError(FewshotPublicationError):
    """Raised when publication paths violate the explicit output-root contract."""


class FewshotPublicationCollisionError(FewshotPublicationError):
    """Raised when publication would overwrite non-identical existing artifacts."""


class FewshotPublicationIOError(FewshotPublicationError):
    """Raised when staged Phase 4 publication fails."""


@dataclass(frozen=True, slots=True)
class Phase4FewshotPublicationResult:
    """Deterministic summary of one Phase 4 publication plan."""

    output_root: Path
    support_manifest_count: int
    adaptation_config_count: int
    protocol_row_count: int
    markdown_data_row_count: int
    source_development_manifest_hash: str
    source_development_split_hash: str
    source_lesion_summary_hash: str | None
    immutable_test_cohort_hash: str
    leakage_check_passed: bool
    patient_overlap_count: int
    case_overlap_count: int
    reused_existing_output: bool
    protocol_table_hash: str


def load_phase4_fewshot_protocol_settings(config_path: Path) -> FewshotAdaptationSettings:
    """Load bounded Phase 4 adaptation settings from an OmegaConf YAML file."""

    try:
        raw_config = OmegaConf.load(config_path)
    except OSError as exc:
        raise FewshotPublicationConfigError(
            f"failed to read Phase 4 config at {config_path}: {exc}"
        ) from exc
    except Exception as exc:
        raise FewshotPublicationConfigError(
            f"failed to parse Phase 4 config at {config_path}: {exc}"
        ) from exc
    if not isinstance(raw_config, DictConfig):
        raise FewshotPublicationConfigError("Phase 4 config must contain a mapping.")
    container = OmegaConf.to_container(raw_config, resolve=True)
    if not isinstance(container, dict):
        raise FewshotPublicationConfigError("Phase 4 config must resolve to a mapping.")
    resolved = cast(dict[str, object], container)
    unknown_root_keys = set(resolved) - {_PHASE4_CONFIG_ROOT_KEY}
    if unknown_root_keys:
        raise FewshotPublicationConfigError(
            f"unknown Phase 4 config keys: {sorted(unknown_root_keys)!r}."
        )
    if _PHASE4_CONFIG_ROOT_KEY not in resolved:
        raise FewshotPublicationConfigError(
            f"missing Phase 4 config key: {_PHASE4_CONFIG_ROOT_KEY!r}."
        )
    settings_mapping = resolved[_PHASE4_CONFIG_ROOT_KEY]
    if not isinstance(settings_mapping, dict):
        raise FewshotPublicationConfigError("phase4_fewshot_protocol must resolve to a mapping.")
    typed_settings = cast(dict[str, object], settings_mapping)
    unknown_keys = set(typed_settings) - _PHASE4_EXPECTED_CONFIG_KEYS
    missing_keys = _PHASE4_EXPECTED_CONFIG_KEYS - set(typed_settings)
    if unknown_keys:
        raise FewshotPublicationConfigError(
            f"unknown Phase 4 setting keys: {sorted(unknown_keys)!r}."
        )
    if missing_keys:
        raise FewshotPublicationConfigError(
            f"missing Phase 4 setting keys: {sorted(missing_keys)!r}."
        )
    return FewshotAdaptationSettings(
        baseline_family=_expect_nonempty_string(
            typed_settings["baseline_family"],
            field_name="baseline_family",
        ),
        optimizer_name=_expect_nonempty_string(
            typed_settings["optimizer_name"],
            field_name="optimizer_name",
        ),
        learning_rate=_expect_positive_float(
            typed_settings["learning_rate"],
            field_name="learning_rate",
        ),
        weight_decay=_expect_nonnegative_float(
            typed_settings["weight_decay"],
            field_name="weight_decay",
        ),
        max_adaptation_steps=_expect_positive_int(
            typed_settings["max_adaptation_steps"],
            field_name="max_adaptation_steps",
        ),
        support_batch_size=_expect_positive_int(
            typed_settings["support_batch_size"],
            field_name="support_batch_size",
        ),
        gradient_accumulation_steps=_expect_positive_int(
            typed_settings["gradient_accumulation_steps"],
            field_name="gradient_accumulation_steps",
        ),
        amp_enabled=_expect_bool(typed_settings["amp_enabled"], field_name="amp_enabled"),
    )


def generate_and_publish_phase4_fewshot_protocol(
    *,
    manifest_path: Path,
    split_path: Path,
    lesion_artifact_path: Path | None,
    output_root: Path,
    initialization_reference: FewshotInitializationReference,
    base_seed: int,
    settings: FewshotAdaptationSettings = DEFAULT_FEWSHOT_ADAPTATION_SETTINGS,
) -> Phase4FewshotPublicationResult:
    """Generate, validate, and publish deterministic Phase 4 protocol artifacts."""

    _require_nonnegative_int(base_seed, field_name="base_seed")
    manifest = _load_phase2_artifact(manifest_path, DatasetManifest)
    split = _load_phase2_artifact(split_path, DevelopmentSplitManifest)
    lesion_artifact = (
        _load_phase2_artifact(lesion_artifact_path, LesionComponentsArtifact)
        if lesion_artifact_path is not None
        else None
    )
    support_manifests_by_k = generate_complete_fixed_support_manifest_set(
        manifest,
        split,
        lesion_artifact,
    )
    leakage_result = _verify_zero_support_internal_test_overlap(support_manifests_by_k, split)
    if not leakage_result.leakage_check_passed:
        raise FewshotProtocolValidationError(
            "Support/internal-test overlap detected during Phase 4 publication."
        )
    adaptation_configs = build_all_fewshot_adaptation_configs(
        support_manifests_by_k,
        initialization_reference=initialization_reference,
        base_seed=base_seed,
        settings=settings,
    )
    protocol_table = build_fewshot_protocol_table(
        support_manifests_by_k,
        initialization_reference=initialization_reference,
        base_seed=base_seed,
        settings=settings,
    )
    artifact_bytes = _build_publication_bytes(
        support_manifests_by_k=support_manifests_by_k,
        adaptation_configs=adaptation_configs,
        protocol_table=protocol_table,
        leakage_result=leakage_result,
    )
    safe_output_root = _validate_phase4_output_root(output_root)
    reused_existing_output = _publish_directory_tree_if_absent_or_equal(
        output_root=safe_output_root,
        artifact_bytes=artifact_bytes,
    )
    return Phase4FewshotPublicationResult(
        output_root=safe_output_root,
        support_manifest_count=sum(len(items) for items in support_manifests_by_k.values()),
        adaptation_config_count=len(adaptation_configs),
        protocol_row_count=len(protocol_table.rows),
        markdown_data_row_count=len(protocol_table.rows),
        source_development_manifest_hash=protocol_table.source_development_manifest_hash,
        source_development_split_hash=protocol_table.source_development_split_hash,
        source_lesion_summary_hash=protocol_table.source_lesion_summary_hash,
        immutable_test_cohort_hash=protocol_table.immutable_test_cohort_hash,
        leakage_check_passed=leakage_result.leakage_check_passed,
        patient_overlap_count=leakage_result.patient_overlap_count,
        case_overlap_count=leakage_result.case_overlap_count,
        reused_existing_output=reused_existing_output,
        protocol_table_hash=protocol_table.artifact_hash,
    )


@dataclass(frozen=True, slots=True)
class _LeakageCheckResult:
    leakage_check_passed: bool
    patient_overlap_count: int
    case_overlap_count: int


def _build_publication_bytes(
    *,
    support_manifests_by_k: dict[int, tuple[FewshotSupportManifest, ...]],
    adaptation_configs: dict[tuple[int, str, str], FewshotAdaptationConfig],
    protocol_table: FewshotProtocolTable,
    leakage_result: _LeakageCheckResult,
) -> dict[str, bytes]:
    support_count = sum(len(items) for items in support_manifests_by_k.values())
    if support_count != len(FIXED_PROTOCOL_K_VALUES) * DEFAULT_REPLICATE_COUNT:
        raise FewshotProtocolValidationError("Expected exactly 15 support manifests.")
    if len(adaptation_configs) != FIXED_PROTOCOL_ROW_COUNT:
        raise FewshotProtocolValidationError("Expected exactly 45 adaptation configs.")
    if len(protocol_table.rows) != FIXED_PROTOCOL_ROW_COUNT:
        raise FewshotProtocolValidationError("Expected exactly 45 protocol-table rows.")

    artifact_bytes: dict[str, bytes] = {}
    for requested_k in FIXED_PROTOCOL_K_VALUES:
        manifests = tuple(
            sorted(
                support_manifests_by_k[requested_k],
                key=lambda item: (item.replicate_id, item.artifact_hash),
            )
        )
        for support_manifest in manifests:
            relative_path = (
                f"support_manifests/k{requested_k:02d}/"
                f"fewshot_support_manifest__{support_manifest.manifest_id}.json"
            )
            artifact_bytes[relative_path] = fewshot_support_manifest_to_json(support_manifest)
    for key in sorted(adaptation_configs):
        requested_k, replicate_id, adaptation_mode = key
        config = adaptation_configs[key]
        relative_path = (
            "adaptation_configs/"
            f"fewshot_adaptation_config__k{requested_k:02d}_{replicate_id}_{adaptation_mode}_"
            f"{config.artifact_hash[:16]}.json"
        )
        artifact_bytes[relative_path] = fewshot_adaptation_config_to_json(config)

    protocol_table_json = fewshot_protocol_table_to_json(protocol_table)
    validated_protocol_table = fewshot_protocol_table_from_json(protocol_table_json)
    artifact_bytes[f"protocol/{_PROTOCOL_JSON_NAME}"] = protocol_table_json
    artifact_bytes[f"protocol/{_PROTOCOL_MARKDOWN_NAME}"] = _render_protocol_markdown(
        validated_protocol_table
    )
    artifact_bytes[_GENERATION_SUMMARY_NAME] = _generation_summary_json_bytes(
        protocol_table=validated_protocol_table,
        support_manifest_count=support_count,
        adaptation_config_count=len(adaptation_configs),
        leakage_result=leakage_result,
    )
    return dict(sorted(artifact_bytes.items()))


def _render_protocol_markdown(protocol_table: FewshotProtocolTable) -> bytes:
    lines = [
        "# Few-Shot Protocol Table",
        "",
        "| K | Replicate ID | Adaptation Mode | Support Hash | Config Hash | "
        "Planned Output Identifier | Run Status |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in protocol_table.rows:
        lines.append(
            "| "
            f"{row.requested_k} | "
            f"{row.replicate_id} | "
            f"{row.adaptation_mode} | "
            f"{row.support_manifest_hash[:12]} | "
            f"{row.adaptation_config_hash[:12]} | "
            f"{row.planned_output_identifier} | "
            f"{row.run_status} |"
        )
    if len(protocol_table.rows) != FIXED_PROTOCOL_ROW_COUNT:
        raise FewshotPublicationError("Protocol Markdown must contain exactly 45 data rows.")
    return ("\n".join(lines) + "\n").encode("utf-8")


def _generation_summary_json_bytes(
    *,
    protocol_table: FewshotProtocolTable,
    support_manifest_count: int,
    adaptation_config_count: int,
    leakage_result: _LeakageCheckResult,
) -> bytes:
    summary = {
        "summary_version": _GENERATION_SUMMARY_VERSION,
        "required_k_values": list(FIXED_PROTOCOL_K_VALUES),
        "replicate_count": DEFAULT_REPLICATE_COUNT,
        "adaptation_modes": list(FIXED_PROTOCOL_ADAPTATION_MODES),
        "adaptation_mode_count": len(FIXED_PROTOCOL_ADAPTATION_MODES),
        "support_manifest_count": support_manifest_count,
        "adaptation_config_count": adaptation_config_count,
        "protocol_row_count": len(protocol_table.rows),
        "protocol_markdown_data_row_count": len(protocol_table.rows),
        "source_development_manifest_hash": protocol_table.source_development_manifest_hash,
        "source_development_split_hash": protocol_table.source_development_split_hash,
        "source_lesion_summary_hash": protocol_table.source_lesion_summary_hash,
        "immutable_test_cohort_hash": protocol_table.immutable_test_cohort_hash,
        "protocol_table_hash": protocol_table.artifact_hash,
        "leakage_check_passed": leakage_result.leakage_check_passed,
        "internal_test_patient_overlap_count": leakage_result.patient_overlap_count,
        "internal_test_case_overlap_count": leakage_result.case_overlap_count,
    }
    return canonical_json_bytes(summary) + b"\n"


def _verify_zero_support_internal_test_overlap(
    support_manifests_by_k: dict[int, tuple[FewshotSupportManifest, ...]],
    split: DevelopmentSplitManifest,
) -> _LeakageCheckResult:
    internal_test_patient_ids = {
        assignment.anonymous_patient_id
        for assignment in split.assignments
        if assignment.partition == "internal_test"
    }
    internal_test_case_ids = {
        assignment.anonymous_case_id
        for assignment in split.assignments
        if assignment.partition == "internal_test"
    }
    patient_overlap_count = 0
    case_overlap_count = 0
    for manifests in support_manifests_by_k.values():
        for support_manifest in manifests:
            support_patient_ids = {
                assignment.anonymous_patient_id for assignment in support_manifest.assignments
            }
            support_case_ids = {
                assignment.anonymous_case_id for assignment in support_manifest.assignments
            }
            patient_overlap_count += len(support_patient_ids & internal_test_patient_ids)
            case_overlap_count += len(support_case_ids & internal_test_case_ids)
    return _LeakageCheckResult(
        leakage_check_passed=(patient_overlap_count == 0 and case_overlap_count == 0),
        patient_overlap_count=patient_overlap_count,
        case_overlap_count=case_overlap_count,
    )


def _load_phase2_artifact(path: Path | None, artifact_type: type[Any]) -> Any:
    if path is None:
        raise FewshotPublicationError("artifact path must not be None.")
    try:
        return phase2_artifact_from_json(path.read_text(encoding="utf-8"), artifact_type)
    except (OSError, Phase2ArtifactError) as exc:
        raise FewshotPublicationError(
            f"failed to load persisted Phase 2 artifact {artifact_type.__name__}"
        ) from exc


def _validate_phase4_output_root(output_root: Path) -> Path:
    repository_root = Path(__file__).resolve().parents[3]
    try:
        resolved_output_root = validate_explicit_external_output_root(
            output_root,
            forbidden_roots=(repository_root,),
        )
    except InvalidDatasetRootError as exc:
        raise FewshotPublicationPathError(str(exc)) from exc
    _require_no_symlink_components(output_root)
    return resolved_output_root


def _publish_directory_tree_if_absent_or_equal(
    *,
    output_root: Path,
    artifact_bytes: dict[str, bytes],
) -> bool:
    expected_directory_paths = _expected_directory_paths(artifact_bytes)
    if output_root.exists():
        if not output_root.is_dir():
            raise FewshotPublicationPathError("output_root must be a directory when it exists.")
        _require_no_symlinks_in_tree(output_root)
        if any(output_root.iterdir()):
            if _existing_tree_matches(
                output_root=output_root,
                artifact_bytes=artifact_bytes,
                expected_directory_paths=expected_directory_paths,
            ):
                return True
            raise FewshotPublicationCollisionError(
                "output_root already contains non-identical published artifacts."
            )

    staging_root = output_root.parent / f".{output_root.name}.phase4-publication.tmp"
    if staging_root.exists():
        raise FewshotPublicationCollisionError("staging output root already exists.")
    try:
        staging_root.mkdir(mode=0o700)
        for relative_directory in expected_directory_paths:
            (staging_root / relative_directory).mkdir(parents=True, exist_ok=True)
        for relative_path, payload in artifact_bytes.items():
            _write_bytes_in_staging_root(
                staging_root=staging_root,
                relative_path=relative_path,
                payload=payload,
            )
        if output_root.exists():
            if any(output_root.iterdir()):
                raise FewshotPublicationCollisionError(
                    "output_root became non-empty before staged publication."
                )
            output_root.rmdir()
        os.replace(staging_root, output_root)
    except FewshotPublicationError:
        _remove_tree_if_present(staging_root)
        raise
    except OSError as exc:
        _remove_tree_if_present(staging_root)
        raise FewshotPublicationIOError("failed to publish Phase 4 protocol artifacts.") from exc
    return False


def _write_bytes_in_staging_root(
    *,
    staging_root: Path,
    relative_path: str,
    payload: bytes,
) -> None:
    normalized_relative_path = _normalize_relative_publication_path(relative_path)
    output_path = staging_root / normalized_relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(payload)


def _existing_tree_matches(
    *,
    output_root: Path,
    artifact_bytes: dict[str, bytes],
    expected_directory_paths: set[str],
) -> bool:
    existing_directory_paths: set[str] = set()
    existing_file_paths: dict[str, bytes] = {}
    for path in sorted(output_root.rglob("*")):
        relative_path = path.relative_to(output_root).as_posix()
        if path.is_symlink():
            raise FewshotPublicationPathError("output_root must not contain symlinked paths.")
        if path.is_dir():
            existing_directory_paths.add(relative_path)
        elif path.is_file():
            existing_file_paths[relative_path] = path.read_bytes()
        else:
            raise FewshotPublicationPathError("output_root must contain only directories or files.")
    if existing_directory_paths != expected_directory_paths:
        return False
    return existing_file_paths == artifact_bytes


def _expected_directory_paths(artifact_bytes: dict[str, bytes]) -> set[str]:
    directories: set[str] = set()
    for relative_path in artifact_bytes:
        parts = relative_path.split("/")[:-1]
        for index in range(1, len(parts) + 1):
            directories.add("/".join(parts[:index]))
    return directories


def _normalize_relative_publication_path(relative_path: str) -> str:
    from protoem_ct.data.phase2_paths import normalize_safe_relative_posix_path

    try:
        return normalize_safe_relative_posix_path(relative_path)
    except Exception as exc:
        raise FewshotPublicationPathError("relative publication path is unsafe.") from exc


def _require_no_symlink_components(path: Path) -> None:
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current = current / component
        if current.exists() and current.is_symlink():
            raise FewshotPublicationPathError("output_root must not traverse symlinked paths.")
        if current.exists() and not current.is_dir() and current != path:
            raise FewshotPublicationPathError("output_root parent chain must contain directories.")


def _require_no_symlinks_in_tree(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_symlink():
            raise FewshotPublicationPathError("output_root must not contain symlinked paths.")


def _remove_tree_if_present(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _expect_nonempty_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise FewshotPublicationConfigError(f"{field_name} must be a non-empty string.")
    return value


def _expect_positive_float(value: object, *, field_name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool) or float(value) <= 0.0:
        raise FewshotPublicationConfigError(f"{field_name} must be a positive number.")
    return float(value)


def _expect_nonnegative_float(value: object, *, field_name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool) or float(value) < 0.0:
        raise FewshotPublicationConfigError(f"{field_name} must be a nonnegative number.")
    return float(value)


def _expect_positive_int(value: object, *, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise FewshotPublicationConfigError(f"{field_name} must be a positive integer.")
    return value


def _expect_bool(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise FewshotPublicationConfigError(f"{field_name} must be a boolean.")
    return value


def _require_nonnegative_int(value: int, *, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise FewshotPublicationError(f"{field_name} must be a nonnegative integer.")


__all__ = [
    "FewshotPublicationCollisionError",
    "FewshotPublicationConfigError",
    "FewshotPublicationError",
    "FewshotPublicationIOError",
    "FewshotPublicationPathError",
    "Phase4FewshotPublicationResult",
    "generate_and_publish_phase4_fewshot_protocol",
    "load_phase4_fewshot_protocol_settings",
]
