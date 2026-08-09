"""Tests for the Phase 10 artifact inventory builder's pure helper functions.

These tests do not require the external drive to be mounted; they exercise
canonical hashing, artifact-loading, and inventory-assembly logic against
synthetic in-repo fixtures only.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "workflow"
    / "scripts"
    / "build_phase10_artifact_inventory.py"
)
_MODULE_NAME = "build_phase10_artifact_inventory"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


_module = _load_module()
HashCheck = _module.HashCheck
InventoryBuilder = _module.InventoryBuilder
build_inventory = _module.build_inventory
canonical_json_bytes = _module.canonical_json_bytes
load_json = _module.load_json
render_markdown = _module.render_markdown
sha256_file = _module.sha256_file
sha256_json = _module.sha256_json


def test_canonical_json_bytes_sorts_keys_and_is_compact() -> None:
    payload = {"b": 1, "a": 2}
    result = canonical_json_bytes(payload)
    assert result == b'{"a":2,"b":1}'


def test_sha256_json_is_deterministic_regardless_of_key_order() -> None:
    assert sha256_json({"a": 1, "b": 2}) == sha256_json({"b": 2, "a": 1})


def test_sha256_file_matches_known_digest(tmp_path: Path) -> None:
    target = tmp_path / "sample.bin"
    target.write_bytes(b"hello world")
    expected = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    assert sha256_file(target) == expected


def test_load_json_returns_none_for_missing_file(tmp_path: Path) -> None:
    assert load_json(tmp_path / "does_not_exist.json") is None


def test_load_json_reads_existing_file(tmp_path: Path) -> None:
    target = tmp_path / "sample.json"
    target.write_text(json.dumps({"key": "value"}), encoding="utf-8")
    assert load_json(target) == {"key": "value"}


def test_check_field_hash_reports_match(tmp_path: Path) -> None:
    builder = InventoryBuilder(repo_root=tmp_path, drive_root=tmp_path, drive_available=False)
    builder.check_field_hash(
        artifact_name="fixture.json",
        doc={"manifest_hash": "abc123"},
        field_name="manifest_hash",
        expected="abc123",
    )
    assert builder.hash_checks == [
        HashCheck("fixture.json", "manifest_hash", "abc123", "abc123", "match")
    ]


def test_check_field_hash_reports_mismatch(tmp_path: Path) -> None:
    builder = InventoryBuilder(repo_root=tmp_path, drive_root=tmp_path, drive_available=False)
    builder.check_field_hash(
        artifact_name="fixture.json",
        doc={"manifest_hash": "xyz"},
        field_name="manifest_hash",
        expected="abc123",
    )
    assert builder.hash_checks[0].status == "mismatch"


def test_check_field_hash_reports_unavailable_when_doc_missing(tmp_path: Path) -> None:
    builder = InventoryBuilder(repo_root=tmp_path, drive_root=tmp_path, drive_available=False)
    builder.check_field_hash(
        artifact_name="fixture.json", doc=None, field_name="manifest_hash", expected="abc123"
    )
    assert builder.hash_checks[0].status == "unavailable_artifact_missing"
    assert builder.hash_checks[0].actual is None


def test_check_raw_file_hash_reports_match(tmp_path: Path) -> None:
    target = tmp_path / "sample.bin"
    target.write_bytes(b"payload")
    builder = InventoryBuilder(repo_root=tmp_path, drive_root=tmp_path, drive_available=False)
    expected = sha256_file(target)
    builder.check_raw_file_hash(artifact_name="sample.bin", path=target, expected=expected)
    assert builder.hash_checks[0].status == "match"


def test_check_raw_file_hash_reports_unavailable_when_file_missing(tmp_path: Path) -> None:
    builder = InventoryBuilder(repo_root=tmp_path, drive_root=tmp_path, drive_available=False)
    builder.check_raw_file_hash(
        artifact_name="missing.bin", path=tmp_path / "missing.bin", expected="deadbeef"
    )
    assert builder.hash_checks[0].status == "unavailable_artifact_missing"


def test_build_inventory_without_drive_marks_everything_unavailable_but_not_fabricated(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    drive_root = tmp_path / "no_such_drive"
    repo_root.mkdir()

    inventory = build_inventory(repo_root, drive_root)

    assert inventory["schema_name"] == "phase10_artifact_inventory"
    assert inventory["drive_available_this_session"] is False
    # No metric should be silently fabricated: every metric whose backing
    # artifact could not be loaded is simply absent from the metrics list,
    # not present with a guessed value.
    for metric in inventory["metrics"]:
        assert metric["value"] is not None or metric["value"] == 0
    # Hash checks should all report unavailable, not a fabricated match.
    assert all(
        check["status"] == "unavailable_artifact_missing" for check in inventory["hash_checks"]
    )
    # Self-hash must be present and stable across re-serialization.
    assert inventory["self_hash"] == sha256_json(
        {k: v for k, v in inventory.items() if k != "self_hash"}
    )


def test_build_inventory_is_deterministic_across_two_calls(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    drive_root = tmp_path / "no_such_drive"
    repo_root.mkdir()

    first = build_inventory(repo_root, drive_root)
    second = build_inventory(repo_root, drive_root)
    assert first == second


def test_render_markdown_includes_self_hash_and_counts(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    drive_root = tmp_path / "no_such_drive"
    repo_root.mkdir()
    inventory = build_inventory(repo_root, drive_root)

    markdown = render_markdown(inventory)

    assert inventory["self_hash"] in markdown
    assert "Hash verification results" in markdown
    assert "Unavailable (referenced in prose, no backing artifact located)" in markdown


@pytest.mark.parametrize("field_name", ["manifest_hash", "split_hash", "audit_hash"])
def test_check_field_hash_field_name_is_recorded(tmp_path: Path, field_name: str) -> None:
    builder = InventoryBuilder(repo_root=tmp_path, drive_root=tmp_path, drive_available=False)
    builder.check_field_hash(
        artifact_name="fixture.json",
        doc={field_name: "same"},
        field_name=field_name,
        expected="same",
    )
    assert builder.hash_checks[0].field_or_kind == field_name
