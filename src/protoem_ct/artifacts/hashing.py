"""Deterministic JSON and file hashing utilities for project artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypeAlias

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
