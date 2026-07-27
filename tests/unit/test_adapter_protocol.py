"""Unit tests for Phase 2 adapter protocol and inventory contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, ClassVar, cast

import pytest

from protoem_ct.artifacts import Phase2ArtifactSerializationError, phase2_artifact_to_json
from protoem_ct.data import (
    AdapterCaseCandidate,
    AdapterInventory,
    AdapterLayoutSpec,
    DatasetAdapter,
    DatasetPathEscapeError,
    DuplicateResolvedPathError,
    InvalidAdapterCandidateError,
    InvalidAdapterInventoryError,
    InvalidAdapterLayoutError,
    ResolvedAdapterCase,
    resolve_adapter_inventory_files,
)


def _candidate(index: int = 1, **overrides: object) -> AdapterCaseCandidate:
    values: dict[str, object] = {
        "source_case_key": f"case-{index:03d}",
        "image_relative_path": f"images/case-{index:03d}.txt",
        "label_relative_path": f"labels/case-{index:03d}.txt",
    }
    values.update(overrides)
    return cast(AdapterCaseCandidate, cast(Any, AdapterCaseCandidate)(**values))


def _inventory(*candidates: AdapterCaseCandidate) -> AdapterInventory:
    return AdapterInventory(
        adapter_name="synthetic-adapter",
        adapter_version="1.0.0",
        candidates=candidates or (_candidate(1), _candidate(2)),
    )


def _write_inventory_files(root: Path, inventory: AdapterInventory) -> None:
    for candidate in inventory.candidates:
        image_path = root / candidate.image_relative_path
        label_path = root / candidate.label_relative_path
        image_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_text(f"image {candidate.source_case_key}\n", encoding="utf-8")
        label_path.write_text(f"label {candidate.source_case_key}\n", encoding="utf-8")


def test_valid_configurable_layout_patterns_are_accepted() -> None:
    layout = AdapterLayoutSpec(
        image_pattern="images/case-*.txt",
        label_pattern="labels/case-[0-9][0-9][0-9].txt",
        recursive=True,
    )

    assert layout.image_pattern == "images/case-*.txt"
    assert layout.recursive is True


def test_layout_rejects_equal_patterns_and_unsafe_or_unsupported_globs() -> None:
    with pytest.raises(InvalidAdapterLayoutError):
        AdapterLayoutSpec("images/*.txt", "images/*.txt", recursive=False)

    for pattern in (
        "/images/*.txt",
        "../images/*.txt",
        "images\\*.txt",
        "file://images/*.txt",
        "C:/images/*.txt",
        "images/**/case.txt",
        "images/{case}.txt",
        "images/[].txt",
    ):
        with pytest.raises(InvalidAdapterLayoutError):
            AdapterLayoutSpec(pattern, "labels/*.txt", recursive=False)


def test_layout_validation_does_not_execute_filesystem_globbing(tmp_path: Path) -> None:
    layout = AdapterLayoutSpec("missing-images/*.txt", "missing-labels/*.txt", recursive=False)

    assert layout.image_pattern == "missing-images/*.txt"
    assert not (tmp_path / "missing-images").exists()


def test_valid_candidate_is_accepted_and_unsafe_source_keys_are_rejected() -> None:
    candidate = _candidate()

    assert candidate.source_case_key == "case-001"

    for source_key in ("", "   ", "case/001", "case\\001", "file://case", "case 001", "case\n001"):
        with pytest.raises(InvalidAdapterCandidateError):
            _candidate(source_case_key=source_key)


def test_candidate_rejects_equal_image_and_label_paths() -> None:
    with pytest.raises(InvalidAdapterCandidateError):
        _candidate(label_relative_path="images/case-001.txt")


def test_inventory_accepts_deterministic_candidates_and_is_immutable() -> None:
    inventory = _inventory()

    assert tuple(candidate.source_case_key for candidate in inventory.candidates) == (
        "case-001",
        "case-002",
    )
    with pytest.raises(FrozenInstanceError):
        cast(Any, inventory).adapter_name = "other"


def test_inventory_rejects_unsorted_and_duplicate_candidates() -> None:
    with pytest.raises(InvalidAdapterInventoryError, match="sorted"):
        _inventory(_candidate(2), _candidate(1))

    with pytest.raises(InvalidAdapterInventoryError, match="source_case_key"):
        _inventory(_candidate(1), _candidate(2, source_case_key="case-001"))

    with pytest.raises(InvalidAdapterInventoryError, match="image_relative_path"):
        _inventory(_candidate(1), _candidate(2, image_relative_path="images/case-001.txt"))

    with pytest.raises(InvalidAdapterInventoryError, match="label_relative_path"):
        _inventory(_candidate(1), _candidate(2, label_relative_path="labels/case-001.txt"))

    with pytest.raises(InvalidAdapterInventoryError, match="image/label pair"):
        _inventory(
            _candidate(1),
            _candidate(
                2,
                image_relative_path="images/case-001.txt",
                label_relative_path="labels/case-001.txt",
            ),
        )


class _SyntheticFakeAdapter:
    adapter_name: ClassVar[str] = "synthetic-adapter"
    adapter_version: ClassVar[str] = "1.0.0"

    def discover(self, dataset_root: Path, layout: AdapterLayoutSpec) -> AdapterInventory:
        assert layout.image_pattern
        assert dataset_root.is_absolute()
        return _inventory()


class _IncompleteFakeAdapter:
    adapter_name: ClassVar[str] = "synthetic-adapter"
    adapter_version: ClassVar[str] = "1.0.0"


def test_runtime_checkable_dataset_adapter_protocol() -> None:
    assert isinstance(_SyntheticFakeAdapter(), DatasetAdapter)
    assert not isinstance(_IncompleteFakeAdapter(), DatasetAdapter)


def test_no_concrete_real_data_adapter_exists() -> None:
    import protoem_ct.data.adapters as adapters

    names = set(adapters.__all__)

    assert "LiTSAdapter" not in names
    assert "IrcadAdapter" not in names
    assert "IRCADAdapter" not in names
    assert "ThreeDircadbAdapter" not in names


def test_valid_inventory_resolution_preserves_order_and_placeholder_bytes(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    inventory = _inventory()
    _write_inventory_files(root, inventory)
    image_before = (root / "images/case-001.txt").read_bytes()
    label_before = (root / "labels/case-001.txt").read_bytes()

    resolved = resolve_adapter_inventory_files(root, inventory)

    assert tuple(case.source_case_key for case in resolved) == ("case-001", "case-002")
    assert all(isinstance(case, ResolvedAdapterCase) for case in resolved)
    assert (root / "images/case-001.txt").read_bytes() == image_before
    assert (root / "labels/case-001.txt").read_bytes() == label_before


def test_inventory_resolution_does_not_open_or_hash_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    inventory = _inventory(_candidate(1))
    _write_inventory_files(root, inventory)

    def _fail_open(self: Path, *args: object, **kwargs: object) -> object:
        msg = f"unexpected file open: {self}"
        raise AssertionError(msg)

    monkeypatch.setattr(Path, "open", _fail_open)

    resolved = resolve_adapter_inventory_files(root, inventory)

    assert resolved[0].source_case_key == "case-001"


def test_inventory_resolution_rejects_duplicate_resolved_image_paths(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    image_target = root / "shared-image.txt"
    image_target.write_text("image\n", encoding="utf-8")
    (root / "images").mkdir()
    (root / "labels").mkdir()
    (root / "labels/case-001.txt").write_text("label one\n", encoding="utf-8")
    (root / "labels/case-002.txt").write_text("label two\n", encoding="utf-8")
    try:
        (root / "images/case-001.txt").symlink_to(image_target)
        (root / "images/case-002.txt").symlink_to(image_target)
    except OSError:
        pytest.skip("symlinks unavailable on this filesystem")

    with pytest.raises(DuplicateResolvedPathError, match="duplicate resolved paths"):
        resolve_adapter_inventory_files(root, _inventory())


def test_inventory_resolution_rejects_duplicate_resolved_label_paths(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    label_target = root / "shared-label.txt"
    label_target.write_text("label\n", encoding="utf-8")
    (root / "images").mkdir()
    (root / "labels").mkdir()
    (root / "images/case-001.txt").write_text("image one\n", encoding="utf-8")
    (root / "images/case-002.txt").write_text("image two\n", encoding="utf-8")
    try:
        (root / "labels/case-001.txt").symlink_to(label_target)
        (root / "labels/case-002.txt").symlink_to(label_target)
    except OSError:
        pytest.skip("symlinks unavailable on this filesystem")

    with pytest.raises(DuplicateResolvedPathError, match="duplicate resolved paths"):
        resolve_adapter_inventory_files(root, _inventory())


def test_inventory_resolution_rejects_image_aliasing_any_label(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    inventory = _inventory()
    _write_inventory_files(root, inventory)
    (root / "images/case-002.txt").unlink()
    try:
        (root / "images/case-002.txt").symlink_to(root / "labels/case-001.txt")
    except OSError:
        pytest.skip("symlinks unavailable on this filesystem")

    with pytest.raises(InvalidAdapterInventoryError, match="image files and label files"):
        resolve_adapter_inventory_files(root, inventory)


def test_inventory_resolution_rejects_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    inventory = _inventory(_candidate(1))
    _write_inventory_files(root, inventory)
    outside = tmp_path / "outside.txt"
    outside.write_text("outside\n", encoding="utf-8")
    (root / "images/case-001.txt").unlink()
    try:
        (root / "images/case-001.txt").symlink_to(outside)
    except OSError:
        pytest.skip("symlinks unavailable on this filesystem")

    with pytest.raises(DatasetPathEscapeError):
        resolve_adapter_inventory_files(root, inventory)


def test_phase2_artifact_serialization_does_not_support_adapter_ephemeral_types(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    inventory = _inventory(_candidate(1))
    _write_inventory_files(root, inventory)
    resolved = resolve_adapter_inventory_files(root, inventory)[0]

    for value in (_candidate(1), inventory, resolved):
        with pytest.raises(Phase2ArtifactSerializationError):
            phase2_artifact_to_json(cast(Any, value))


def test_source_case_key_does_not_appear_in_persisted_artifact_examples() -> None:
    with pytest.raises(Phase2ArtifactSerializationError) as error:
        phase2_artifact_to_json(cast(Any, _candidate(1)))

    assert "case-001" not in str(error.value)
