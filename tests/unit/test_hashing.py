"""Unit tests for deterministic artifact hashing."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pytest

from protoem_ct.artifacts import HashFileError, HashInputError
from protoem_ct.artifacts.hashing import (
    canonical_json_bytes,
    hash_config,
    hash_manifest,
    sha256_file,
    sha256_json,
)


def test_canonical_json_bytes_are_deterministic() -> None:
    value = {"z": [2, 1], "a": {"b": True, "c": None}}

    assert canonical_json_bytes(value) == b'{"a":{"b":true,"c":null},"z":[2,1]}'


def test_dictionary_key_order_does_not_change_hash() -> None:
    first = {"a": 1, "b": {"c": 2, "d": 3}}
    second = {"b": {"d": 3, "c": 2}, "a": 1}

    assert sha256_json(first) == sha256_json(second)
    assert hash_config(first) == hash_config(second)
    assert hash_manifest(first) == hash_manifest(second)


def test_list_order_changes_hash() -> None:
    assert sha256_json({"case_ids": ["case-001", "case-002"]}) != sha256_json(
        {"case_ids": ["case-002", "case-001"]}
    )


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_json_numbers_are_rejected(value: float) -> None:
    with pytest.raises(HashInputError):
        canonical_json_bytes({"value": value})


def test_streamed_file_hash_matches_known_sha256(tmp_path: Path) -> None:
    payload = b"protoem-ct deterministic file hash\n"
    path = tmp_path / "payload.txt"
    path.write_bytes(payload)

    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()


def test_missing_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(HashFileError):
        sha256_file(tmp_path / "missing.txt")


def test_directory_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(HashFileError):
        sha256_file(tmp_path)
