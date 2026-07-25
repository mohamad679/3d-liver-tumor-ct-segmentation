"""Synthetic filesystem tests for the Phase 2 LiTS-style discovery adapter."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, cast

import pytest

from protoem_ct.artifacts import Phase2ArtifactSerializationError, phase2_artifact_to_json
from protoem_ct.data import (
    AdapterInventory,
    AdapterLayoutSpec,
    AmbiguousLiTSPairingError,
    DatasetAdapter,
    DatasetPathEscapeError,
    IncompleteLiTSPairError,
    InvalidLiTSConventionError,
    LiTSDiscoveryError,
    LiTSFilenameConvention,
    LiTSStyleAdapter,
    MalformedLiTSFilenameError,
    MissingDatasetPathError,
    NonRegularDatasetFileError,
)

LEGACY_LAYOUT = AdapterLayoutSpec(
    image_pattern="volume-*.nii*",
    label_pattern="segmentation-*.nii*",
    recursive=False,
)
RECURSIVE_LEGACY_LAYOUT = AdapterLayoutSpec(
    image_pattern="volume-*.nii*",
    label_pattern="segmentation-*.nii*",
    recursive=True,
)
MSD_LAYOUT = AdapterLayoutSpec(
    image_pattern="imagesTr/liver_*.nii.gz",
    label_pattern="labelsTr/liver_*.nii.gz",
    recursive=False,
)


def _legacy_convention() -> LiTSFilenameConvention:
    return LiTSFilenameConvention(
        image_prefix="volume-",
        label_prefix="segmentation-",
        allowed_suffixes=(".nii.gz", ".nii"),
    )


def _msd_convention() -> LiTSFilenameConvention:
    return LiTSFilenameConvention(
        image_prefix="liver_",
        label_prefix="liver_",
        allowed_suffixes=(".nii.gz",),
    )


def _adapter(
    convention: LiTSFilenameConvention | None = None,
) -> LiTSStyleAdapter:
    return LiTSStyleAdapter(convention=convention or _legacy_convention())


def _write_placeholder(path: Path, text: str = "synthetic placeholder\n") -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))
    return path.read_bytes()


def _discover(
    root: Path,
    *,
    adapter: LiTSStyleAdapter | None = None,
    layout: AdapterLayoutSpec = LEGACY_LAYOUT,
) -> AdapterInventory:
    return (adapter or _adapter()).discover(root, layout)


def _source_keys(inventory: AdapterInventory) -> tuple[str, ...]:
    return tuple(candidate.source_case_key for candidate in inventory.candidates)


def test_valid_legacy_convention_and_equal_msd_prefixes_are_accepted() -> None:
    legacy = _legacy_convention()
    msd = _msd_convention()

    assert legacy.image_prefix == "volume-"
    assert legacy.label_prefix == "segmentation-"
    assert msd.image_prefix == msd.label_prefix == "liver_"
    assert legacy.allowed_suffixes == (".nii.gz", ".nii")


@pytest.mark.parametrize(
    "prefix",
    ["", " volume-", "volume- ", "vol/ume-", "vol\\ume-", "file://x", "C:", "vol*"],
)
def test_unsafe_or_empty_prefix_rejected(prefix: str) -> None:
    with pytest.raises(InvalidLiTSConventionError):
        LiTSFilenameConvention(prefix, "segmentation-", (".nii",))


def test_unsupported_and_duplicate_suffixes_rejected_and_convention_immutable() -> None:
    with pytest.raises(InvalidLiTSConventionError):
        LiTSFilenameConvention("volume-", "segmentation-", (".dcm",))
    with pytest.raises(InvalidLiTSConventionError):
        LiTSFilenameConvention("volume-", "segmentation-", (".nii", ".nii"))

    convention = _legacy_convention()
    with pytest.raises(FrozenInstanceError):
        cast(Any, convention).image_prefix = "other-"


def test_valid_flat_nii_pairs_are_discovered(tmp_path: Path) -> None:
    _write_placeholder(tmp_path / "volume-0.nii")
    _write_placeholder(tmp_path / "segmentation-0.nii")

    inventory = _discover(tmp_path)

    assert _source_keys(inventory) == ("0",)
    assert inventory.candidates[0].image_relative_path == "volume-0.nii"
    assert inventory.candidates[0].label_relative_path == "segmentation-0.nii"


def test_valid_flat_nii_gz_pairs_are_discovered(tmp_path: Path) -> None:
    _write_placeholder(tmp_path / "volume-0.nii.gz")
    _write_placeholder(tmp_path / "segmentation-0.nii.gz")

    inventory = _discover(tmp_path)

    assert _source_keys(inventory) == ("0",)
    assert inventory.candidates[0].image_relative_path == "volume-0.nii.gz"
    assert inventory.candidates[0].label_relative_path == "segmentation-0.nii.gz"


def test_mixed_suffix_pairs_are_handled_deterministically(tmp_path: Path) -> None:
    _write_placeholder(tmp_path / "volume-10.nii")
    _write_placeholder(tmp_path / "segmentation-10.nii.gz")
    _write_placeholder(tmp_path / "volume-2.nii.gz")
    _write_placeholder(tmp_path / "segmentation-2.nii")

    inventory = _discover(tmp_path)

    assert _source_keys(inventory) == ("10", "2")
    assert tuple(candidate.image_relative_path for candidate in inventory.candidates) == (
        "volume-10.nii",
        "volume-2.nii.gz",
    )


def test_multiple_cases_sorted_by_exact_key_and_leading_zeros_preserved(tmp_path: Path) -> None:
    for key in ("2", "000", "10"):
        _write_placeholder(tmp_path / f"volume-{key}.nii.gz")
        _write_placeholder(tmp_path / f"segmentation-{key}.nii.gz")

    inventory = _discover(tmp_path)

    assert _source_keys(inventory) == ("000", "10", "2")


def test_msd_style_imagestr_labelstr_layout_is_discovered(tmp_path: Path) -> None:
    _write_placeholder(tmp_path / "imagesTr/liver_000.nii.gz")
    _write_placeholder(tmp_path / "labelsTr/liver_000.nii.gz")

    inventory = _discover(tmp_path, adapter=_adapter(_msd_convention()), layout=MSD_LAYOUT)

    assert _source_keys(inventory) == ("000",)
    assert inventory.candidates[0].image_relative_path == "imagesTr/liver_000.nii.gz"
    assert inventory.candidates[0].label_relative_path == "labelsTr/liver_000.nii.gz"


def test_msd_layout_is_independent_of_current_working_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "dataset"
    other = tmp_path / "other"
    root.mkdir()
    other.mkdir()
    _write_placeholder(root / "imagesTr/liver_001.nii.gz")
    _write_placeholder(root / "labelsTr/liver_001.nii.gz")

    monkeypatch.chdir(other)

    inventory = _discover(root, adapter=_adapter(_msd_convention()), layout=MSD_LAYOUT)

    assert _source_keys(inventory) == ("001",)


def test_recursive_discovery_finds_nested_pairs(tmp_path: Path) -> None:
    _write_placeholder(tmp_path / "a/images/volume-1.nii.gz")
    _write_placeholder(tmp_path / "b/labels/segmentation-1.nii.gz")

    inventory = _discover(tmp_path, layout=RECURSIVE_LEGACY_LAYOUT)

    assert inventory.candidates[0].image_relative_path == "a/images/volume-1.nii.gz"
    assert inventory.candidates[0].label_relative_path == "b/labels/segmentation-1.nii.gz"


def test_nonrecursive_discovery_does_not_scan_unrelated_nested_directories(tmp_path: Path) -> None:
    _write_placeholder(tmp_path / "nested/volume-1.nii")
    _write_placeholder(tmp_path / "nested/segmentation-1.nii")

    with pytest.raises(LiTSDiscoveryError):
        _discover(tmp_path, layout=LEGACY_LAYOUT)


def test_recursive_traversal_does_not_follow_outside_directory_symlink(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    root = tmp_path / "dataset"
    root.mkdir()
    _write_placeholder(outside / "volume-1.nii")
    _write_placeholder(outside / "segmentation-1.nii")
    try:
        (root / "linked-outside").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable on this filesystem")

    with pytest.raises(LiTSDiscoveryError):
        _discover(root, layout=RECURSIVE_LEGACY_LAYOUT)


def test_internal_file_symlink_is_accepted_where_supported(tmp_path: Path) -> None:
    target_image = tmp_path / "targets/image.nii"
    target_label = tmp_path / "targets/label.nii"
    _write_placeholder(target_image, "image target\n")
    _write_placeholder(target_label, "label target\n")
    try:
        (tmp_path / "volume-1.nii").symlink_to(target_image)
        (tmp_path / "segmentation-1.nii").symlink_to(target_label)
    except OSError:
        pytest.skip("symlinks unavailable on this filesystem")

    inventory = _discover(tmp_path)

    assert _source_keys(inventory) == ("1",)


def test_outside_file_symlink_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    outside = tmp_path / "outside-image.nii"
    _write_placeholder(outside)
    _write_placeholder(root / "segmentation-1.nii")
    try:
        (root / "volume-1.nii").symlink_to(outside)
    except OSError:
        pytest.skip("symlinks unavailable on this filesystem")

    with pytest.raises(DatasetPathEscapeError):
        _discover(root)


def test_missing_image_and_missing_label_are_rejected(tmp_path: Path) -> None:
    _write_placeholder(tmp_path / "segmentation-1.nii")
    with pytest.raises(IncompleteLiTSPairError, match="missing images"):
        _discover(tmp_path)

    other_root = tmp_path / "other"
    other_root.mkdir()
    _write_placeholder(other_root / "volume-1.nii")
    with pytest.raises(IncompleteLiTSPairError, match="missing labels"):
        _discover(other_root)


def test_empty_discovery_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(LiTSDiscoveryError, match="no matched"):
        _discover(tmp_path)


def test_malformed_prefix_and_empty_or_unsafe_keys_are_rejected(tmp_path: Path) -> None:
    _write_placeholder(tmp_path / "bad-1.nii")
    _write_placeholder(tmp_path / "segmentation-1.nii")
    with pytest.raises(MalformedLiTSFilenameError, match="configured prefix"):
        _discover(tmp_path, layout=AdapterLayoutSpec("bad-*.nii", "segmentation-*.nii", False))

    empty_key_root = tmp_path / "empty-key"
    empty_key_root.mkdir()
    _write_placeholder(empty_key_root / "volume-.nii")
    _write_placeholder(empty_key_root / "segmentation-.nii")
    with pytest.raises(MalformedLiTSFilenameError, match="empty source key"):
        _discover(empty_key_root)

    unsafe_key_root = tmp_path / "unsafe-key"
    unsafe_key_root.mkdir()
    _write_placeholder(unsafe_key_root / "volume-bad key.nii")
    _write_placeholder(unsafe_key_root / "segmentation-bad key.nii")
    with pytest.raises(MalformedLiTSFilenameError, match="unsafe source key"):
        _discover(unsafe_key_root)


def test_duplicate_image_and_label_keys_are_rejected(tmp_path: Path) -> None:
    _write_placeholder(tmp_path / "volume-1.nii")
    _write_placeholder(tmp_path / "volume-1.nii.gz")
    _write_placeholder(tmp_path / "segmentation-1.nii")
    with pytest.raises(AmbiguousLiTSPairingError, match="multiple image files"):
        _discover(tmp_path)

    label_root = tmp_path / "label-duplicates"
    label_root.mkdir()
    _write_placeholder(label_root / "volume-1.nii")
    _write_placeholder(label_root / "segmentation-1.nii")
    _write_placeholder(label_root / "segmentation-1.nii.gz")
    with pytest.raises(AmbiguousLiTSPairingError, match="multiple label files"):
        _discover(label_root)


def test_same_resolved_file_used_as_image_and_label_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "target/shared.nii.gz"
    _write_placeholder(target)
    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    try:
        (tmp_path / "images/liver_000.nii.gz").symlink_to(target)
        (tmp_path / "labels/liver_000.nii.gz").symlink_to(target)
    except OSError:
        pytest.skip("symlinks unavailable on this filesystem")

    layout = AdapterLayoutSpec("images/liver_*.nii.gz", "labels/liver_*.nii.gz", False)
    with pytest.raises(AmbiguousLiTSPairingError):
        _discover(tmp_path, adapter=_adapter(_msd_convention()), layout=layout)


def test_image_resolving_to_another_record_label_is_rejected(tmp_path: Path) -> None:
    _write_placeholder(tmp_path / "volume-0.nii")
    _write_placeholder(tmp_path / "segmentation-0.nii")
    _write_placeholder(tmp_path / "segmentation-1.nii")
    try:
        (tmp_path / "volume-1.nii").symlink_to(tmp_path / "segmentation-0.nii")
    except OSError:
        pytest.skip("symlinks unavailable on this filesystem")

    with pytest.raises(AmbiguousLiTSPairingError):
        _discover(tmp_path)


def test_matched_directory_and_dangling_symlink_are_rejected(tmp_path: Path) -> None:
    (tmp_path / "volume-1.nii").mkdir()
    _write_placeholder(tmp_path / "segmentation-1.nii")
    with pytest.raises(NonRegularDatasetFileError):
        _discover(tmp_path)

    dangling_root = tmp_path / "dangling"
    dangling_root.mkdir()
    _write_placeholder(dangling_root / "segmentation-1.nii")
    try:
        (dangling_root / "volume-1.nii").symlink_to(dangling_root / "missing.nii")
    except OSError:
        pytest.skip("symlinks unavailable on this filesystem")
    with pytest.raises(MissingDatasetPathError):
        _discover(dangling_root)


def test_ambiguous_pairing_error_message_is_deterministic(tmp_path: Path) -> None:
    for name in ("volume-1.nii.gz", "volume-1.nii", "segmentation-1.nii"):
        _write_placeholder(tmp_path / name)

    with pytest.raises(AmbiguousLiTSPairingError) as first_error:
        _discover(tmp_path)
    with pytest.raises(AmbiguousLiTSPairingError) as second_error:
        _discover(tmp_path)

    assert str(first_error.value) == str(second_error.value)


def test_directory_creation_order_does_not_change_inventory(tmp_path: Path) -> None:
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    for key in ("2", "1"):
        _write_placeholder(root_a / f"volume-{key}.nii")
        _write_placeholder(root_a / f"segmentation-{key}.nii")
    for key in ("1", "2"):
        _write_placeholder(root_b / f"segmentation-{key}.nii")
        _write_placeholder(root_b / f"volume-{key}.nii")

    assert _discover(root_a) == _discover(root_b)


def test_repeated_discovery_is_equal_and_inventory_immutable(tmp_path: Path) -> None:
    _write_placeholder(tmp_path / "volume-1.nii")
    _write_placeholder(tmp_path / "segmentation-1.nii")

    first = _discover(tmp_path)
    second = _discover(tmp_path)

    assert first == second
    with pytest.raises(FrozenInstanceError):
        cast(Any, first).adapter_name = "other"


def test_discovery_does_not_open_load_or_hash_file_contents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_before = _write_placeholder(tmp_path / "volume-1.nii", "image bytes\n")
    label_before = _write_placeholder(tmp_path / "segmentation-1.nii", "label bytes\n")

    def _fail(*args: object, **kwargs: object) -> object:
        raise AssertionError("unexpected content load or hash")

    import nibabel as nib

    import protoem_ct.artifacts.hashing as hashing

    with monkeypatch.context() as context:
        context.setattr(nib, "load", _fail)
        context.setattr(hashing, "sha256_file", _fail)
        context.setattr(Path, "open", _fail)

        inventory = _discover(tmp_path)

    assert _source_keys(inventory) == ("1",)
    assert (tmp_path / "volume-1.nii").read_bytes() == image_before
    assert (tmp_path / "segmentation-1.nii").read_bytes() == label_before


def test_adapter_protocol_and_serialization_boundaries(tmp_path: Path) -> None:
    _write_placeholder(tmp_path / "volume-1.nii")
    _write_placeholder(tmp_path / "segmentation-1.nii")
    adapter = _adapter()
    inventory = adapter.discover(tmp_path, LEGACY_LAYOUT)

    assert isinstance(adapter, DatasetAdapter)
    with pytest.raises(Phase2ArtifactSerializationError) as error:
        phase2_artifact_to_json(cast(Any, inventory))
    assert "1" not in str(error.value)


def test_no_concrete_3d_ircadb_adapter_exists() -> None:
    import protoem_ct.data.adapters as adapters

    names = set(adapters.__all__)

    assert "IRCAD" not in "".join(names).upper()
