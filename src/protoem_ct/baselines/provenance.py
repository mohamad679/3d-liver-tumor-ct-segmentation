"""Deterministic provenance artifacts for Phase 3 baseline runs."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from protoem_ct.artifacts.hashing import canonical_json_bytes, sha256_json

BASELINE_RUN_PROVENANCE_VERSION = "baseline_run_provenance_v1"
SUPPORTED_BASELINE_RUN_STATUSES: tuple[str, ...] = (
    "planned",
    "running",
    "completed",
    "failed",
)
_SUPPORTED_BASELINE_FAMILIES = {"nnunet_v2", "monai_segresnet"}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_RUN_IDENTIFIER_RE = re.compile(r"^[a-z0-9](?:[a-z0-9_-]{0,62}[a-z0-9])?$")
_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")
_FAILURE_CODE_RE = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
_PACKAGE_NAME_RE = re.compile(r"[-_.]+")
_ALLOWED_FIELDS = {
    "artifact_hash",
    "baseline_family",
    "run_identifier",
    "git_commit",
    "config_sha256",
    "dataset_manifest_sha256",
    "development_split_sha256",
    "seed",
    "python_version",
    "platform_system",
    "platform_machine",
    "selected_device",
    "amp_enabled",
    "package_versions",
    "start_timestamp",
    "end_timestamp",
    "status",
    "checkpoint_sha256",
    "prediction_manifest_sha256",
    "metrics_artifact_sha256",
    "failure_codes",
    "contract_version",
}


class BaselineProvenanceError(ValueError):
    """Base error for baseline provenance artifacts."""


class BaselineProvenanceValidationError(BaselineProvenanceError):
    """Raised when baseline provenance content is invalid."""


class BaselineProvenanceSerializationError(BaselineProvenanceError):
    """Raised when baseline provenance JSON cannot be parsed safely."""


class BaselineProvenanceVersionError(BaselineProvenanceError):
    """Raised when an unsupported provenance version is encountered."""


class BaselineProvenanceHashError(BaselineProvenanceError):
    """Raised when a persisted artifact hash does not match its content."""


@dataclass(frozen=True, slots=True)
class BaselineRunProvenance:
    """Versioned immutable provenance for one baseline run."""

    artifact_hash: str
    baseline_family: str
    run_identifier: str
    git_commit: str
    config_sha256: str
    dataset_manifest_sha256: str
    development_split_sha256: str
    seed: int
    python_version: str
    platform_system: str
    platform_machine: str
    selected_device: str
    amp_enabled: bool
    package_versions: Mapping[str, str] = field(default_factory=dict)
    start_timestamp: str = ""
    end_timestamp: str | None = None
    status: str = "planned"
    checkpoint_sha256: str | None = None
    prediction_manifest_sha256: str | None = None
    metrics_artifact_sha256: str | None = None
    failure_codes: tuple[str, ...] = ()
    contract_version: str = BASELINE_RUN_PROVENANCE_VERSION

    def __post_init__(self) -> None:
        _validate_contract_version(self.contract_version)
        if self.baseline_family not in _SUPPORTED_BASELINE_FAMILIES:
            raise BaselineProvenanceValidationError(
                f"Unsupported baseline family: {self.baseline_family!r}."
            )
        _require_sha256(self.artifact_hash, field_name="artifact_hash")
        _require_conservative_identifier(self.run_identifier)
        _require_no_absolute_path(self.run_identifier, field_name="run_identifier")
        _require_git_commit(self.git_commit)
        _require_sha256(self.config_sha256, field_name="config_sha256")
        _require_sha256(self.dataset_manifest_sha256, field_name="dataset_manifest_sha256")
        _require_sha256(self.development_split_sha256, field_name="development_split_sha256")
        _require_seed(self.seed)
        _require_non_path_string(self.python_version, field_name="python_version")
        _require_non_path_string(self.platform_system, field_name="platform_system")
        _require_non_path_string(self.platform_machine, field_name="platform_machine")
        _require_non_path_string(self.selected_device, field_name="selected_device")
        normalized_packages = _normalize_package_versions(self.package_versions)
        object.__setattr__(
            self,
            "package_versions",
            MappingProxyType(dict(sorted(normalized_packages.items()))),
        )
        _require_timestamp(self.start_timestamp, field_name="start_timestamp")
        if self.end_timestamp is not None:
            _require_timestamp(self.end_timestamp, field_name="end_timestamp")
        _require_status(self.status)
        _require_optional_sha256(self.checkpoint_sha256, field_name="checkpoint_sha256")
        _require_optional_sha256(
            self.prediction_manifest_sha256,
            field_name="prediction_manifest_sha256",
        )
        _require_optional_sha256(
            self.metrics_artifact_sha256,
            field_name="metrics_artifact_sha256",
        )
        normalized_failure_codes = _normalize_failure_codes(self.failure_codes)
        object.__setattr__(self, "failure_codes", normalized_failure_codes)
        _require_status_constraints(self)
        expected_hash = _compute_artifact_hash(_artifact_payload_without_hash(self))
        if self.artifact_hash != expected_hash:
            raise BaselineProvenanceHashError(
                "artifact_hash does not match the deterministic artifact content."
            )


def baseline_run_provenance_to_json(provenance: BaselineRunProvenance) -> bytes:
    """Serialize provenance deterministically with one trailing newline."""

    try:
        payload = _artifact_payload(provenance)
        return canonical_json_bytes(payload) + b"\n"
    except Exception as error:
        if isinstance(error, BaselineProvenanceError):
            raise
        raise BaselineProvenanceSerializationError(
            f"Failed to serialize baseline provenance: {error}"
        ) from error


def baseline_run_provenance_from_json(payload: bytes | str) -> BaselineRunProvenance:
    """Parse and validate baseline provenance JSON."""

    text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
    try:
        parsed = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_bad_json_constant,
        )
    except Exception as error:
        raise BaselineProvenanceSerializationError(
            f"Failed to parse baseline provenance JSON: {error}"
        ) from error
    if not isinstance(parsed, dict):
        raise BaselineProvenanceSerializationError(
            "Baseline provenance JSON must encode an object."
        )
    unknown_fields = set(parsed) - _ALLOWED_FIELDS
    if unknown_fields:
        names = ", ".join(sorted(unknown_fields))
        raise BaselineProvenanceSerializationError(
            f"Unknown baseline provenance field(s): {names}."
        )
    missing_fields = _ALLOWED_FIELDS - set(parsed)
    if missing_fields:
        names = ", ".join(sorted(missing_fields))
        raise BaselineProvenanceSerializationError(
            f"Missing baseline provenance field(s): {names}."
        )
    contract_version = parsed["contract_version"]
    _validate_contract_version(contract_version)
    provenance = BaselineRunProvenance(
        artifact_hash=_expect_string(parsed["artifact_hash"], field_name="artifact_hash"),
        baseline_family=_expect_string(parsed["baseline_family"], field_name="baseline_family"),
        run_identifier=_expect_string(parsed["run_identifier"], field_name="run_identifier"),
        git_commit=_expect_string(parsed["git_commit"], field_name="git_commit"),
        config_sha256=_expect_string(parsed["config_sha256"], field_name="config_sha256"),
        dataset_manifest_sha256=_expect_string(
            parsed["dataset_manifest_sha256"], field_name="dataset_manifest_sha256"
        ),
        development_split_sha256=_expect_string(
            parsed["development_split_sha256"],
            field_name="development_split_sha256",
        ),
        seed=_expect_int(parsed["seed"], field_name="seed"),
        python_version=_expect_string(parsed["python_version"], field_name="python_version"),
        platform_system=_expect_string(parsed["platform_system"], field_name="platform_system"),
        platform_machine=_expect_string(parsed["platform_machine"], field_name="platform_machine"),
        selected_device=_expect_string(parsed["selected_device"], field_name="selected_device"),
        amp_enabled=_expect_bool(parsed["amp_enabled"], field_name="amp_enabled"),
        package_versions=_expect_string_mapping(
            parsed["package_versions"], field_name="package_versions"
        ),
        start_timestamp=_expect_string(parsed["start_timestamp"], field_name="start_timestamp"),
        end_timestamp=_expect_optional_string(parsed["end_timestamp"], field_name="end_timestamp"),
        status=_expect_string(parsed["status"], field_name="status"),
        checkpoint_sha256=_expect_optional_string(
            parsed["checkpoint_sha256"], field_name="checkpoint_sha256"
        ),
        prediction_manifest_sha256=_expect_optional_string(
            parsed["prediction_manifest_sha256"],
            field_name="prediction_manifest_sha256",
        ),
        metrics_artifact_sha256=_expect_optional_string(
            parsed["metrics_artifact_sha256"],
            field_name="metrics_artifact_sha256",
        ),
        failure_codes=_expect_string_tuple(parsed["failure_codes"], field_name="failure_codes"),
        contract_version=contract_version,
    )
    expected_hash = _compute_artifact_hash(_artifact_payload_without_hash(provenance))
    if expected_hash != provenance.artifact_hash:
        raise BaselineProvenanceHashError(
            "artifact_hash does not match the parsed baseline provenance content."
        )
    return provenance


def _artifact_payload(provenance: BaselineRunProvenance) -> dict[str, Any]:
    return {
        "amp_enabled": provenance.amp_enabled,
        "artifact_hash": provenance.artifact_hash,
        "baseline_family": provenance.baseline_family,
        "checkpoint_sha256": provenance.checkpoint_sha256,
        "config_sha256": provenance.config_sha256,
        "contract_version": provenance.contract_version,
        "dataset_manifest_sha256": provenance.dataset_manifest_sha256,
        "development_split_sha256": provenance.development_split_sha256,
        "end_timestamp": provenance.end_timestamp,
        "failure_codes": list(provenance.failure_codes),
        "git_commit": provenance.git_commit,
        "metrics_artifact_sha256": provenance.metrics_artifact_sha256,
        "package_versions": dict(provenance.package_versions),
        "platform_machine": provenance.platform_machine,
        "platform_system": provenance.platform_system,
        "prediction_manifest_sha256": provenance.prediction_manifest_sha256,
        "python_version": provenance.python_version,
        "run_identifier": provenance.run_identifier,
        "seed": provenance.seed,
        "selected_device": provenance.selected_device,
        "start_timestamp": provenance.start_timestamp,
        "status": provenance.status,
    }


def _artifact_payload_without_hash(
    provenance: BaselineRunProvenance,
) -> dict[str, Any]:
    payload = _artifact_payload(provenance)
    del payload["artifact_hash"]
    return payload


def _compute_artifact_hash(payload_without_hash: Mapping[str, Any]) -> str:
    return sha256_json(payload_without_hash)


def _validate_contract_version(contract_version: str) -> None:
    if contract_version != BASELINE_RUN_PROVENANCE_VERSION:
        raise BaselineProvenanceVersionError(
            f"Unsupported baseline provenance version: {contract_version!r}."
        )


def _require_git_commit(value: str) -> None:
    if not _GIT_COMMIT_RE.fullmatch(value):
        raise BaselineProvenanceValidationError(
            "git_commit must be exactly 40 lowercase hexadecimal characters."
        )


def _require_sha256(value: str, *, field_name: str) -> None:
    if not _SHA256_RE.fullmatch(value):
        raise BaselineProvenanceValidationError(
            f"{field_name} must be exactly 64 lowercase hexadecimal characters."
        )


def _require_optional_sha256(value: str | None, *, field_name: str) -> None:
    if value is not None:
        _require_sha256(value, field_name=field_name)


def _require_seed(seed: int) -> None:
    if not 0 <= seed <= 0xFFFFFFFF:
        raise BaselineProvenanceValidationError("seed must be between 0 and 2^32-1 inclusive.")


def _require_conservative_identifier(value: str) -> None:
    if not _RUN_IDENTIFIER_RE.fullmatch(value):
        raise BaselineProvenanceValidationError(
            "run_identifier must match the conservative baseline identifier grammar."
        )


def _require_timestamp(value: str, *, field_name: str) -> None:
    if not _TIMESTAMP_RE.fullmatch(value):
        raise BaselineProvenanceValidationError(
            f"{field_name} must be an explicit UTC ISO-8601 timestamp ending in Z."
        )


def _require_status(value: str) -> None:
    if value not in SUPPORTED_BASELINE_RUN_STATUSES:
        raise BaselineProvenanceValidationError(f"Unsupported baseline run status: {value!r}.")


def _require_non_path_string(value: str, *, field_name: str) -> None:
    if not value:
        raise BaselineProvenanceValidationError(f"{field_name} must not be empty.")
    _require_no_absolute_path(value, field_name=field_name)
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise BaselineProvenanceValidationError(
            f"{field_name} must not contain control characters."
        )


def _require_no_absolute_path(value: str, *, field_name: str) -> None:
    if value.startswith("/"):
        raise BaselineProvenanceValidationError(f"{field_name} must not contain an absolute path.")
    if re.match(r"^[A-Za-z]:[\\/]", value):
        raise BaselineProvenanceValidationError(f"{field_name} must not contain an absolute path.")


def _normalize_package_versions(package_versions: Mapping[str, str]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for name, version in package_versions.items():
        if not isinstance(name, str) or not isinstance(version, str):
            raise BaselineProvenanceValidationError("package_versions must map strings to strings.")
        normalized_name = _normalize_package_name(name)
        if name != normalized_name:
            raise BaselineProvenanceValidationError(
                "package_versions keys must already be normalized."
            )
        _require_non_path_string(version, field_name=f"package_versions[{name!r}]")
        if normalized_name in normalized:
            raise BaselineProvenanceValidationError(
                f"Duplicate normalized package name: {normalized_name!r}."
            )
        normalized[normalized_name] = version
    return normalized


def _normalize_package_name(value: str) -> str:
    stripped = value.strip().lower()
    normalized = _PACKAGE_NAME_RE.sub("-", stripped)
    if stripped != normalized or not normalized:
        return normalized
    return normalized


def _normalize_failure_codes(failure_codes: tuple[str, ...]) -> tuple[str, ...]:
    normalized = tuple(sorted(failure_codes))
    if tuple(failure_codes) != normalized:
        raise BaselineProvenanceValidationError("failure_codes must be sorted deterministically.")
    if len(set(normalized)) != len(normalized):
        raise BaselineProvenanceValidationError("failure_codes must not contain duplicates.")
    for code in normalized:
        if not _FAILURE_CODE_RE.fullmatch(code):
            raise BaselineProvenanceValidationError(
                "failure_codes must use the machine-readable failure-code grammar."
            )
        _require_no_absolute_path(code, field_name="failure_codes")
    return normalized


def _require_status_constraints(provenance: BaselineRunProvenance) -> None:
    final_hashes_present = any(
        value is not None
        for value in (
            provenance.checkpoint_sha256,
            provenance.prediction_manifest_sha256,
            provenance.metrics_artifact_sha256,
        )
    )
    if provenance.status == "planned":
        if provenance.end_timestamp is not None or final_hashes_present:
            raise BaselineProvenanceValidationError(
                "planned provenance must not include an end timestamp or final artifact hashes."
            )
        if provenance.failure_codes:
            raise BaselineProvenanceValidationError(
                "planned provenance must not include failure codes."
            )
        return
    if provenance.status == "running":
        if provenance.end_timestamp is not None:
            raise BaselineProvenanceValidationError(
                "running provenance must not include an end timestamp."
            )
        if final_hashes_present:
            raise BaselineProvenanceValidationError(
                "running provenance must not include final artifact hashes."
            )
        if provenance.failure_codes:
            raise BaselineProvenanceValidationError(
                "running provenance must not include failure codes."
            )
        return
    if provenance.status == "completed":
        if provenance.end_timestamp is None:
            raise BaselineProvenanceValidationError(
                "completed provenance requires an end timestamp."
            )
        if not (
            provenance.checkpoint_sha256
            and provenance.prediction_manifest_sha256
            and provenance.metrics_artifact_sha256
        ):
            raise BaselineProvenanceValidationError(
                "completed provenance requires checkpoint, prediction-manifest, and metrics hashes."
            )
        if provenance.failure_codes:
            raise BaselineProvenanceValidationError(
                "completed provenance must not include failure codes."
            )
        return
    if provenance.status == "failed":
        if provenance.end_timestamp is None:
            raise BaselineProvenanceValidationError("failed provenance requires an end timestamp.")
        if not provenance.failure_codes:
            raise BaselineProvenanceValidationError(
                "failed provenance requires at least one failure code."
            )
        return
    raise BaselineProvenanceValidationError(
        f"Unsupported baseline run status: {provenance.status!r}."
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BaselineProvenanceSerializationError(f"Duplicate JSON object key: {key!r}.")
        result[key] = value
    return result


def _bad_json_constant(constant: str) -> Any:
    raise BaselineProvenanceSerializationError(
        f"Invalid JSON constant in baseline provenance: {constant!r}."
    )


def _expect_string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise BaselineProvenanceSerializationError(f"{field_name} must be a string.")
    return value


def _expect_optional_string(value: Any, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _expect_string(value, field_name=field_name)


def _expect_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BaselineProvenanceSerializationError(f"{field_name} must be an integer.")
    return value


def _expect_bool(value: Any, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise BaselineProvenanceSerializationError(f"{field_name} must be a boolean.")
    return value


def _expect_string_mapping(value: Any, *, field_name: str) -> Mapping[str, str]:
    if not isinstance(value, dict):
        raise BaselineProvenanceSerializationError(
            f"{field_name} must be an object mapping strings to strings."
        )
    result: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str):
            raise BaselineProvenanceSerializationError(f"{field_name} must map strings to strings.")
        result[key] = item
    return result


def _expect_string_tuple(value: Any, *, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise BaselineProvenanceSerializationError(f"{field_name} must be a JSON array of strings.")
    items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise BaselineProvenanceSerializationError(f"{field_name} must contain only strings.")
        items.append(item)
    return tuple(items)
