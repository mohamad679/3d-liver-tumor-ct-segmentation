"""Deterministic JSON and file hashing utilities for project artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from protoem_ct.artifacts.phase2_schemas import (
        DatasetCaseRecord,
        DatasetManifest,
        DevelopmentQaArtifact,
        DevelopmentSplitManifest,
        GeometryLabelQaArtifact,
        LeakageAuditArtifact,
    )

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


class HashingError(ValueError):
    """Base error for project-specific hashing failures."""


class HashInputError(HashingError):
    """Raised when a value cannot be represented as canonical project JSON."""


class HashFileError(HashingError):
    """Raised when a file cannot be hashed as a regular file."""


def canonical_json_bytes(value: object) -> bytes:
    """Return canonical UTF-8 JSON bytes for a JSON-compatible value."""
    checked_value = _require_json_value(value, "value")
    text = json.dumps(
        checked_value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return text.encode("utf-8")


def sha256_json(value: object) -> str:
    """Return the lowercase SHA-256 hex digest of a canonical JSON value."""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 hex digest of a regular file."""
    if not path.exists():
        msg = f"file does not exist: {path}"
        raise HashFileError(msg)
    if not path.is_file():
        msg = f"path is not a regular file: {path}"
        raise HashFileError(msg)

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def hash_config(config: Mapping[str, object]) -> str:
    """Return the canonical SHA-256 hash for a configuration mapping."""
    return sha256_json(config)


def hash_manifest(manifest: Mapping[str, object]) -> str:
    """Return the canonical SHA-256 hash for a manifest mapping."""
    return sha256_json(manifest)


def dataset_root_fingerprint_payload(
    *,
    adapter_name: str,
    adapter_version: str,
    cases: Sequence[DatasetCaseRecord],
) -> dict[str, JsonValue]:
    """Return the deterministic metadata payload for a dataset-root fingerprint.

    The payload deliberately excludes the absolute dataset root and all timestamps. It is derived
    only from adapter identity plus ordered safe relative paths and file hashes already validated by
    ``DatasetCaseRecord``.
    """
    return {
        "adapter_name": adapter_name,
        "adapter_version": adapter_version,
        "cases": [
            {
                "cohort_role": case.cohort_role,
                "image_sha256": case.image_sha256,
                "label_sha256": case.label_sha256,
                "relative_image_path": case.relative_image_path,
                "relative_label_path": case.relative_label_path,
            }
            for case in cases
        ],
        "payload_type": "phase2-dataset-root-fingerprint",
        "schema_version": "2",
    }


def dataset_manifest_hash_payload(manifest: DatasetManifest) -> dict[str, JsonValue]:
    """Return the dataset-manifest hash payload, excluding ``manifest_hash``."""
    from protoem_ct.artifacts.phase2_schemas import phase2_artifact_to_dict

    payload = phase2_artifact_to_dict(manifest)
    payload.pop("manifest_hash")
    return payload


def development_split_hash_payload(
    manifest: DevelopmentSplitManifest,
) -> dict[str, JsonValue]:
    """Return the development-split hash payload, excluding ``split_hash``."""
    from protoem_ct.artifacts.phase2_schemas import phase2_artifact_to_dict

    payload = phase2_artifact_to_dict(manifest)
    payload.pop("split_hash")
    return payload


def development_qa_hash_payload(artifact: DevelopmentQaArtifact) -> dict[str, JsonValue]:
    """Return the development-QA hash payload, excluding ``qa_artifact_hash``."""
    from protoem_ct.artifacts.phase2_schemas import phase2_artifact_to_dict

    payload = phase2_artifact_to_dict(artifact)
    payload.pop("qa_artifact_hash")
    return payload


def geometry_label_qa_hash_payload(artifact: GeometryLabelQaArtifact) -> dict[str, JsonValue]:
    """Return the geometry-label-QA hash payload, excluding ``qa_artifact_hash``."""
    from protoem_ct.artifacts.phase2_schemas import phase2_artifact_to_dict

    payload = phase2_artifact_to_dict(artifact)
    payload.pop("qa_artifact_hash")
    return payload


def leakage_audit_hash_payload(artifact: LeakageAuditArtifact) -> dict[str, JsonValue]:
    """Return the leakage-audit hash payload, excluding ``audit_hash``."""
    from protoem_ct.artifacts.phase2_schemas import phase2_artifact_to_dict

    payload = phase2_artifact_to_dict(artifact)
    payload.pop("audit_hash")
    return payload


def hash_dataset_root_fingerprint(
    *,
    adapter_name: str,
    adapter_version: str,
    cases: Sequence[DatasetCaseRecord],
) -> str:
    """Return the lowercase SHA-256 hash for a Phase 2 dataset-root fingerprint."""
    return sha256_json(
        dataset_root_fingerprint_payload(
            adapter_name=adapter_name,
            adapter_version=adapter_version,
            cases=cases,
        )
    )


def hash_dataset_manifest(manifest: DatasetManifest) -> str:
    """Return the lowercase SHA-256 hash for a Phase 2 dataset manifest."""
    return sha256_json(dataset_manifest_hash_payload(manifest))


def hash_development_split(manifest: DevelopmentSplitManifest) -> str:
    """Return the lowercase SHA-256 hash for a Phase 2 development split artifact."""
    return sha256_json(development_split_hash_payload(manifest))


def hash_development_qa(artifact: DevelopmentQaArtifact) -> str:
    """Return the lowercase SHA-256 hash for a Phase 2 development QA artifact."""
    return sha256_json(development_qa_hash_payload(artifact))


def hash_geometry_label_qa(artifact: GeometryLabelQaArtifact) -> str:
    """Return the lowercase SHA-256 hash for a Phase 2 geometry-label QA artifact."""
    return sha256_json(geometry_label_qa_hash_payload(artifact))


def hash_leakage_audit(artifact: LeakageAuditArtifact) -> str:
    """Return the lowercase SHA-256 hash for a Phase 2 leakage-audit artifact."""
    return sha256_json(leakage_audit_hash_payload(artifact))


def _require_json_value(value: object, location: str) -> JsonValue:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            msg = f"{location} must not contain NaN or Infinity"
            raise HashInputError(msg)
        return value
    if isinstance(value, Mapping):
        checked_mapping: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                msg = f"{location} contains a non-string dictionary key: {key!r}"
                raise HashInputError(msg)
            checked_mapping[key] = _require_json_value(item, f"{location}.{key}")
        return checked_mapping
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        if not isinstance(value, list):
            msg = f"{location} must use lists for JSON arrays, got {type(value).__name__}"
            raise HashInputError(msg)
        return [
            _require_json_value(item, f"{location}[{index}]") for index, item in enumerate(value)
        ]

    msg = f"{location} is not JSON-compatible: {type(value).__name__}"
    raise HashInputError(msg)
