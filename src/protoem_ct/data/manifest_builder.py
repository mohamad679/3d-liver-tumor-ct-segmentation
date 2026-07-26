"""Anonymous Phase 2 LiTS development-manifest builder.

This module supports only LiTS-style development-cohort inventory and manifest publication.
It does not authorize real-data execution, does not create source-key mappings, and does not
support external-cohort manifest publication.
"""

from __future__ import annotations

import hmac
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import protoem_ct.artifacts.hashing as hashing
from protoem_ct.artifacts import (
    DATASET_MANIFEST_TYPE,
    PHASE2_DEVELOPMENT_COHORT_ROLE,
    PHASE2_SCHEMA_VERSION,
    DatasetCaseRecord,
    DatasetManifest,
    phase2_artifact_to_json,
)
from protoem_ct.data.adapters.base import AdapterLayoutSpec, resolve_adapter_inventory_files
from protoem_ct.data.adapters.lits import LiTSFilenameConvention, LiTSStyleAdapter
from protoem_ct.data.phase2_paths import (
    validate_explicit_dataset_root,
)

MIN_ID_KEY_BYTES: Final[int] = 32
MAX_ID_KEY_FILE_BYTES: Final[int] = 64 * 1024
MIN_ANONYMOUS_ID_ENTROPY_BITS: Final[int] = 128
PATIENT_ID_HMAC_DOMAIN: Final[str] = "protoem-ct.phase2.lits.patient.v1"
CASE_ID_HMAC_DOMAIN: Final[str] = "protoem-ct.phase2.lits.case.v1"

_SAFE_CONFIG_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")
_SAFE_PREFIX_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{1,63}$")
_URI_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
_ZERO_SHA256 = "0" * 64


class ManifestBuilderError(ValueError):
    """Base exception for Phase 2 LiTS manifest-builder failures."""


class InvalidAnonymousIdConfigError(ManifestBuilderError):
    """Raised when anonymous-ID configuration is unsafe or insufficient."""


class InvalidIdKeyMaterialError(ManifestBuilderError):
    """Raised when explicit HMAC key material is missing, unsafe, or too small."""


class AnonymousIdCollisionError(ManifestBuilderError):
    """Raised when generated anonymous patient or case identifiers collide."""


class InvalidManifestMetadataError(ManifestBuilderError):
    """Raised when explicit manifest metadata is missing or unsafe."""


class SourceHashingError(ManifestBuilderError):
    """Raised when a source image or label file cannot be hashed."""


class UnsafeManifestOutputPathError(ManifestBuilderError):
    """Raised when the requested manifest output path violates containment policy."""


class ExistingManifestOutputError(ManifestBuilderError):
    """Raised when manifest publication would overwrite an existing output."""


class ManifestPublicationError(ManifestBuilderError):
    """Raised when validated manifest JSON cannot be published atomically."""


@dataclass(frozen=True, slots=True)
class AnonymousIdConfig:
    """Nonsecret deterministic anonymous-ID settings for LiTS development manifests."""

    project_namespace: str
    patient_id_prefix: str
    case_id_prefix: str
    digest_length: int

    def __post_init__(self) -> None:
        _require_safe_config_value(self.project_namespace, "project_namespace")
        _require_safe_prefix(self.patient_id_prefix, "patient_id_prefix")
        _require_safe_prefix(self.case_id_prefix, "case_id_prefix")
        if self.patient_id_prefix == self.case_id_prefix:
            msg = "patient_id_prefix and case_id_prefix must differ"
            raise InvalidAnonymousIdConfigError(msg)
        if (
            isinstance(self.digest_length, bool)
            or not isinstance(self.digest_length, int)
            or self.digest_length * 4 < MIN_ANONYMOUS_ID_ENTROPY_BITS
            or self.digest_length > 64
        ):
            msg = "digest_length must represent at least 128 and at most 256 bits"
            raise InvalidAnonymousIdConfigError(msg)


@dataclass(frozen=True, slots=True)
class InventoryDryRunResult:
    """Safe dry-run LiTS inventory summary without source keys, paths, IDs, or hashes."""

    adapter_name: str
    adapter_version: str
    case_count: int
    image_count: int
    label_count: int


@dataclass(frozen=True, slots=True)
class ManifestBuildResult:
    """Safe result summary for a published anonymous LiTS development manifest."""

    manifest: DatasetManifest
    output_path: Path
    dataset_id: str
    case_count: int
    dataset_root_fingerprint: str
    manifest_hash: str


def read_id_key_file(path: Path) -> bytes:
    """Read explicit HMAC key material from an absolute regular file.

    Symlink key files are rejected to keep the source of key material unambiguous. The key bytes and
    key-file path are never serialized by this module.
    """
    if str(path) == "":
        msg = "id key file path must not be empty"
        raise InvalidIdKeyMaterialError(msg)
    if not path.is_absolute():
        msg = "id key file path must be explicit and absolute"
        raise InvalidIdKeyMaterialError(msg)
    if path.is_symlink():
        msg = "id key file must not be a symlink"
        raise InvalidIdKeyMaterialError(msg)
    try:
        resolved_path = path.resolve(strict=True)
    except FileNotFoundError as exc:
        msg = "id key file does not exist"
        raise InvalidIdKeyMaterialError(msg) from exc
    except OSError as exc:
        msg = "id key file cannot be resolved"
        raise InvalidIdKeyMaterialError(msg) from exc
    if not resolved_path.is_file():
        msg = "id key path must be a regular file"
        raise InvalidIdKeyMaterialError(msg)
    size = resolved_path.stat().st_size
    if size > MAX_ID_KEY_FILE_BYTES:
        msg = "id key file exceeds the maximum supported size"
        raise InvalidIdKeyMaterialError(msg)
    key_material = resolved_path.read_bytes()
    _require_id_key_material(key_material)
    return key_material


def dry_run_lits_inventory(
    dataset_root: Path,
    *,
    layout: AdapterLayoutSpec,
    convention: LiTSFilenameConvention,
) -> InventoryDryRunResult:
    """Run LiTS-style discovery only and return a nonidentifying inventory summary."""
    adapter = LiTSStyleAdapter(convention=convention)
    inventory = adapter.discover(dataset_root, layout)
    case_count = len(inventory.candidates)
    return InventoryDryRunResult(
        adapter_name=inventory.adapter_name,
        adapter_version=inventory.adapter_version,
        case_count=case_count,
        image_count=case_count,
        label_count=case_count,
    )


def build_lits_development_manifest(
    dataset_root: Path,
    *,
    layout: AdapterLayoutSpec,
    convention: LiTSFilenameConvention,
    dataset_id: str,
    generated_at_utc: str,
    git_commit: str,
    anonymous_id_config: AnonymousIdConfig,
    id_key: bytes,
    output_path: Path,
) -> ManifestBuildResult:
    """Publish an anonymous LiTS development manifest from explicit synthetic-tested inputs."""
    canonical_root = validate_explicit_dataset_root(dataset_root)
    safe_output_path = _validate_manifest_output_path(output_path, dataset_root=canonical_root)
    _require_manifest_metadata(dataset_id, "dataset_id")
    _require_manifest_metadata(generated_at_utc, "generated_at_utc")
    _require_manifest_metadata(git_commit, "git_commit")
    _require_id_key_material(id_key)

    adapter = LiTSStyleAdapter(convention=convention)
    inventory = adapter.discover(canonical_root, layout)
    resolved_cases = resolve_adapter_inventory_files(canonical_root, inventory)
    resolved_by_key = {case.source_case_key: case for case in resolved_cases}

    case_records: list[DatasetCaseRecord] = []
    patient_ids: set[str] = set()
    case_ids: set[str] = set()
    for candidate in inventory.candidates:
        resolved_case = resolved_by_key[candidate.source_case_key]
        image_sha256 = _hash_source_file(resolved_case.image_path, logical_side="image")
        label_sha256 = _hash_source_file(resolved_case.label_path, logical_side="label")
        anonymous_patient_id = anonymous_lits_patient_id(
            candidate.source_case_key,
            config=anonymous_id_config,
            id_key=id_key,
        )
        anonymous_case_id = anonymous_lits_case_id(
            candidate.source_case_key,
            config=anonymous_id_config,
            id_key=id_key,
        )
        if anonymous_patient_id == anonymous_case_id:
            msg = "anonymous patient and case IDs must differ"
            raise AnonymousIdCollisionError(msg)
        if anonymous_patient_id in patient_ids:
            msg = "anonymous patient ID collision detected"
            raise AnonymousIdCollisionError(msg)
        if anonymous_case_id in case_ids:
            msg = "anonymous case ID collision detected"
            raise AnonymousIdCollisionError(msg)
        patient_ids.add(anonymous_patient_id)
        case_ids.add(anonymous_case_id)
        case_records.append(
            DatasetCaseRecord(
                anonymous_patient_id=anonymous_patient_id,
                anonymous_case_id=anonymous_case_id,
                relative_image_path=candidate.image_relative_path,
                relative_label_path=candidate.label_relative_path,
                image_sha256=image_sha256,
                label_sha256=label_sha256,
                cohort_role=PHASE2_DEVELOPMENT_COHORT_ROLE,
            )
        )

    ordered_cases = tuple(
        sorted(case_records, key=lambda case: (case.anonymous_patient_id, case.anonymous_case_id))
    )
    dataset_root_fingerprint = hashing.hash_dataset_root_fingerprint(
        adapter_name=inventory.adapter_name,
        adapter_version=inventory.adapter_version,
        cases=ordered_cases,
    )
    manifest_without_hash = DatasetManifest(
        schema_version=PHASE2_SCHEMA_VERSION,
        manifest_type=DATASET_MANIFEST_TYPE,
        dataset_id=dataset_id,
        cohort_role=PHASE2_DEVELOPMENT_COHORT_ROLE,
        adapter_name=inventory.adapter_name,
        adapter_version=inventory.adapter_version,
        generated_at_utc=generated_at_utc,
        git_commit=git_commit,
        dataset_root_fingerprint=dataset_root_fingerprint,
        manifest_hash=_ZERO_SHA256,
        case_count=len(ordered_cases),
        cases=ordered_cases,
    )
    manifest_hash = hashing.hash_dataset_manifest(manifest_without_hash)
    manifest = DatasetManifest(
        schema_version=manifest_without_hash.schema_version,
        manifest_type=manifest_without_hash.manifest_type,
        dataset_id=manifest_without_hash.dataset_id,
        cohort_role=manifest_without_hash.cohort_role,
        adapter_name=manifest_without_hash.adapter_name,
        adapter_version=manifest_without_hash.adapter_version,
        generated_at_utc=manifest_without_hash.generated_at_utc,
        git_commit=manifest_without_hash.git_commit,
        dataset_root_fingerprint=manifest_without_hash.dataset_root_fingerprint,
        manifest_hash=manifest_hash,
        case_count=manifest_without_hash.case_count,
        cases=manifest_without_hash.cases,
    )
    if hashing.hash_dataset_manifest(manifest) != manifest_hash:
        msg = "manifest hash verification failed"
        raise ManifestPublicationError(msg)

    _publish_manifest_json(manifest, safe_output_path)
    return ManifestBuildResult(
        manifest=manifest,
        output_path=safe_output_path,
        dataset_id=manifest.dataset_id,
        case_count=manifest.case_count,
        dataset_root_fingerprint=manifest.dataset_root_fingerprint,
        manifest_hash=manifest.manifest_hash,
    )


def anonymous_lits_patient_id(
    source_case_key: str,
    *,
    config: AnonymousIdConfig,
    id_key: bytes,
) -> str:
    """Return the deterministic one-way patient ID for a LiTS source case key."""
    return _hmac_identifier(
        source_case_key,
        config=config,
        id_key=id_key,
        domain=PATIENT_ID_HMAC_DOMAIN,
        prefix=config.patient_id_prefix,
    )


def anonymous_lits_case_id(
    source_case_key: str,
    *,
    config: AnonymousIdConfig,
    id_key: bytes,
) -> str:
    """Return the deterministic one-way case ID for a LiTS source case key."""
    return _hmac_identifier(
        source_case_key,
        config=config,
        id_key=id_key,
        domain=CASE_ID_HMAC_DOMAIN,
        prefix=config.case_id_prefix,
    )


def _hmac_identifier(
    source_case_key: str,
    *,
    config: AnonymousIdConfig,
    id_key: bytes,
    domain: str,
    prefix: str,
) -> str:
    _require_id_key_material(id_key)
    if not isinstance(source_case_key, str) or source_case_key == "":
        msg = "source case key must be nonempty"
        raise AnonymousIdCollisionError(msg)
    message = "\0".join((config.project_namespace, domain, source_case_key)).encode("utf-8")
    digest = hmac.new(id_key, message, digestmod="sha256").hexdigest()[: config.digest_length]
    return f"{prefix}{digest}"


def _hash_source_file(path: Path, *, logical_side: str) -> str:
    try:
        return hashing.sha256_file(path)
    except (OSError, hashing.HashingError) as exc:
        msg = f"failed to hash {logical_side} source file"
        raise SourceHashingError(msg) from exc


def _validate_manifest_output_path(output_path: Path, *, dataset_root: Path) -> Path:
    if str(output_path) == "":
        msg = "manifest output path must not be empty"
        raise UnsafeManifestOutputPathError(msg)
    if not output_path.is_absolute():
        msg = "manifest output path must be explicit and absolute"
        raise UnsafeManifestOutputPathError(msg)
    if any(part == ".." for part in output_path.parts):
        msg = "manifest output path must not contain parent traversal"
        raise UnsafeManifestOutputPathError(msg)
    if output_path.exists() and output_path.is_file():
        msg = "manifest output path already exists"
        raise ExistingManifestOutputError(msg)

    safe_output_path = _resolve_path_with_existing_ancestor(output_path)
    if _paths_are_equal_or_nested(safe_output_path, dataset_root):
        msg = "manifest output path must not equal or be inside the dataset root"
        raise UnsafeManifestOutputPathError(msg)
    if _paths_are_equal_or_nested(dataset_root, safe_output_path):
        msg = "manifest output path must not contain the dataset root"
        raise UnsafeManifestOutputPathError(msg)
    if output_path.exists():
        msg = "manifest output path already exists"
        raise ExistingManifestOutputError(msg)
    return safe_output_path


def _resolve_path_with_existing_ancestor(path: Path) -> Path:
    missing_parts: list[str] = []
    current = path
    while not current.exists():
        missing_parts.append(current.name)
        parent = current.parent
        if parent == current:
            msg = "manifest output path has no existing parent"
            raise UnsafeManifestOutputPathError(msg)
        current = parent
    if current.is_file():
        msg = "manifest output parent must not be a regular file"
        raise UnsafeManifestOutputPathError(msg)
    if not current.is_dir():
        msg = "manifest output parent must resolve beneath a directory"
        raise UnsafeManifestOutputPathError(msg)
    try:
        resolved = current.resolve(strict=True)
    except OSError as exc:
        msg = "manifest output path cannot be resolved"
        raise UnsafeManifestOutputPathError(msg) from exc
    for part in reversed(missing_parts):
        resolved = resolved / part
    return resolved


def _publish_manifest_json(manifest: DatasetManifest, output_path: Path) -> None:
    text = phase2_artifact_to_json(manifest)
    temp_path = output_path.with_name(f".{output_path.name}.tmp")
    created_temp = False
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if temp_path.exists():
            msg = "temporary manifest output already exists"
            raise ExistingManifestOutputError(msg)
        temp_path.write_text(text, encoding="utf-8", newline="\n")
        created_temp = True
        if output_path.exists():
            msg = "manifest output path already exists"
            raise ExistingManifestOutputError(msg)
        os.link(temp_path, output_path)
        temp_path.unlink()
        created_temp = False
    except ExistingManifestOutputError:
        raise
    except OSError as exc:
        msg = "failed to publish manifest JSON"
        raise ManifestPublicationError(msg) from exc
    finally:
        if created_temp and temp_path.exists():
            temp_path.unlink()


def _require_id_key_material(value: object) -> None:
    if not isinstance(value, bytes):
        msg = "id key material must be bytes"
        raise InvalidIdKeyMaterialError(msg)
    if len(value) < MIN_ID_KEY_BYTES:
        msg = "id key material must contain at least 32 bytes"
        raise InvalidIdKeyMaterialError(msg)


def _require_manifest_metadata(value: object, field_name: str) -> None:
    if not isinstance(value, str) or value == "" or value != value.strip():
        msg = f"{field_name} must be supplied explicitly as a nonempty string"
        raise InvalidManifestMetadataError(msg)
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        msg = f"{field_name} must not contain control characters"
        raise InvalidManifestMetadataError(msg)
    if field_name == "dataset_id" and _SAFE_CONFIG_PATTERN.fullmatch(value) is None:
        msg = "dataset_id must use conservative lowercase metadata characters"
        raise InvalidManifestMetadataError(msg)


def _require_safe_config_value(value: object, field_name: str) -> None:
    if not isinstance(value, str) or _SAFE_CONFIG_PATTERN.fullmatch(value) is None:
        msg = f"{field_name} must use conservative lowercase metadata characters"
        raise InvalidAnonymousIdConfigError(msg)
    _require_no_path_or_uri_like_text(value, field_name)


def _require_safe_prefix(value: object, field_name: str) -> None:
    if not isinstance(value, str) or _SAFE_PREFIX_PATTERN.fullmatch(value) is None:
        msg = f"{field_name} must use conservative lowercase ID-prefix characters"
        raise InvalidAnonymousIdConfigError(msg)
    _require_no_path_or_uri_like_text(value, field_name)


def _require_no_path_or_uri_like_text(value: str, field_name: str) -> None:
    if value != value.strip():
        msg = f"{field_name} must not contain leading or trailing whitespace"
        raise InvalidAnonymousIdConfigError(msg)
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        msg = f"{field_name} must not contain control characters"
        raise InvalidAnonymousIdConfigError(msg)
    if "/" in value or "\\" in value or ":" in value or value in {".", ".."}:
        msg = f"{field_name} must not be path-like"
        raise InvalidAnonymousIdConfigError(msg)
    if _URI_PATTERN.match(value) is not None:
        msg = f"{field_name} must not be URI-like"
        raise InvalidAnonymousIdConfigError(msg)


def _paths_are_equal_or_nested(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
