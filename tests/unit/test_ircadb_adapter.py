"""Synthetic filesystem tests for the Phase 2 3D-IRCADb-style adapter."""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, cast

import pytest

from protoem_ct.artifacts import (
    PHASE2_DEVELOPMENT_COHORT_ROLE,
    DatasetCaseRecord,
    Phase2ArtifactSerializationError,
    phase2_artifact_to_json,
)
from protoem_ct.data import (
    IRCADB_STYLE_ADAPTER_NAME,
    IRCADB_STYLE_ADAPTER_VERSION,
    SUPPORTED_IRCADB_SUFFIXES,
    AdapterInventory,
    AdapterLayoutSpec,
    DatasetAdapter,
    DatasetPathEscapeError,
    LiTSFilenameConvention,
    LiTSStyleAdapter,
    MissingDatasetPathError,
    NonRegularDatasetFileError,
)
from protoem_ct.data import (
    AmbiguousIrcadbPairingError as TopLevelAmbiguousIrcadbPairingError,
)
from protoem_ct.data import (
    IncompleteIrcadbPairError as TopLevelIncompleteIrcadbPairError,
)
from protoem_ct.data import (
    InvalidIrcadbConventionError as TopLevelInvalidIrcadbConventionError,
)
from protoem_ct.data import (
    IrcadbAdapterError as TopLevelIrcadbAdapterError,
)
from protoem_ct.data import (
    IrcadbDiscoveryError as TopLevelIrcadbDiscoveryError,
)
from protoem_ct.data import (
    IrcadbFilenameConvention as TopLevelIrcadbFilenameConvention,
)
from protoem_ct.data import (
    IrcadbStyleAdapter as TopLevelIrcadbStyleAdapter,
)
from protoem_ct.data import (
    MalformedIrcadbCaseError as TopLevelMalformedIrcadbCaseError,
)
from protoem_ct.data.adapters import IrcadbStyleAdapter as AdapterPackageIrcadbStyleAdapter
from protoem_ct.data.adapters.ircadb import (
    AmbiguousIrcadbPairingError,
    IncompleteIrcadbPairError,
    InvalidIrcadbConventionError,
    IrcadbAdapterError,
    IrcadbDiscoveryError,
    IrcadbFilenameConvention,
    IrcadbStyleAdapter,
    MalformedIrcadbCaseError,
)

HASH0 = "0" * 64
HASH1 = "1" * 64

IRCADB_ARCHIVE_LAYOUT = AdapterLayoutSpec(
    image_pattern="3Dircadb1.*/PATIENT_DICOM/image.nii.gz",
    label_pattern="3Dircadb1.*/MASKS_DICOM/liver_tumor.nii.gz",
    recursive=False,
)
NESTED_CASE_LAYOUT = AdapterLayoutSpec(
    image_pattern="cases/case_*/image.nii.gz",
    label_pattern="cases/case_*/tumor.nii.gz",
    recursive=False,
)
NII_CASE_LAYOUT = AdapterLayoutSpec(
    image_pattern="cases/case_*/image.nii",
    label_pattern="cases/case_*/tumor.nii",
    recursive=False,
)
RECURSIVE_CASE_LAYOUT = AdapterLayoutSpec(
    image_pattern="image.nii.gz",
    label_pattern="tumor.nii.gz",
    recursive=True,
)


def _archive_convention() -> IrcadbFilenameConvention:
    return IrcadbFilenameConvention(
        image_filename="image.nii.gz",
        label_filename="liver_tumor.nii.gz",
        case_directory_pattern="3Dircadb1.*",
        allowed_suffixes=(".nii.gz", ".nii"),
    )


def _nested_convention(suffix: str = ".nii.gz") -> IrcadbFilenameConvention:
    return IrcadbFilenameConvention(
        image_filename=f"image{suffix}",
        label_filename=f"tumor{suffix}",
        case_directory_pattern="cases/case_*",
        allowed_suffixes=(suffix,),
    )


def _recursive_convention() -> IrcadbFilenameConvention:
    return IrcadbFilenameConvention(
        image_filename="image.nii.gz",
        label_filename="tumor.nii.gz",
        case_directory_pattern="case_*",
        allowed_suffixes=(".nii.gz",),
    )


def _adapter(
    convention: IrcadbFilenameConvention | None = None,
) -> IrcadbStyleAdapter:
    return IrcadbStyleAdapter(convention=convention or _archive_convention())


def _write_placeholder(path: Path, text: str = "synthetic placeholder\n") -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))
    return path.read_bytes()


def _discover(
    root: Path,
    *,
    adapter: IrcadbStyleAdapter | None = None,
    layout: AdapterLayoutSpec = IRCADB_ARCHIVE_LAYOUT,
) -> AdapterInventory:
    return (adapter or _adapter()).discover(root, layout)


def _source_keys(inventory: AdapterInventory) -> tuple[str, ...]:
    return tuple(candidate.source_case_key for candidate in inventory.candidates)


def test_ircadb_public_exports_are_available_without_authorizing_real_data_use() -> None:
    import protoem_ct.data as data
    import protoem_ct.data.adapters as adapters
    import protoem_ct.data.adapters.ircadb as ircadb_module
    from protoem_ct.data import IrcadbStyleAdapter as ImportedTopLevelAdapter
    from protoem_ct.data.adapters import IrcadbStyleAdapter as ImportedPackageAdapter

    expected_names = {
        "IrcadbStyleAdapter",
        "IrcadbFilenameConvention",
        "IRCADB_STYLE_ADAPTER_NAME",
        "IRCADB_STYLE_ADAPTER_VERSION",
        "SUPPORTED_IRCADB_SUFFIXES",
        "IrcadbAdapterError",
        "InvalidIrcadbConventionError",
        "IrcadbDiscoveryError",
        "MalformedIrcadbCaseError",
        "IncompleteIrcadbPairError",
        "AmbiguousIrcadbPairingError",
    }

    assert expected_names <= set(adapters.__all__)
    assert expected_names <= set(data.__all__)
    assert ImportedPackageAdapter is IrcadbStyleAdapter
    assert ImportedTopLevelAdapter is ImportedPackageAdapter
    assert AdapterPackageIrcadbStyleAdapter is ImportedPackageAdapter
    assert TopLevelIrcadbStyleAdapter is ImportedTopLevelAdapter
    assert TopLevelIrcadbFilenameConvention is IrcadbFilenameConvention
    assert TopLevelIrcadbAdapterError is IrcadbAdapterError
    assert TopLevelInvalidIrcadbConventionError is InvalidIrcadbConventionError
    assert TopLevelIrcadbDiscoveryError is IrcadbDiscoveryError
    assert TopLevelMalformedIrcadbCaseError is MalformedIrcadbCaseError
    assert TopLevelIncompleteIrcadbPairError is IncompleteIrcadbPairError
    assert TopLevelAmbiguousIrcadbPairingError is AmbiguousIrcadbPairingError
    assert IRCADB_STYLE_ADAPTER_NAME == "ircadb_style"
    assert IRCADB_STYLE_ADAPTER_VERSION == "1"
    assert SUPPORTED_IRCADB_SUFFIXES == (".nii.gz", ".nii")
    assert isinstance(_adapter(_nested_convention()), DatasetAdapter)
    assert "Phase 8" in (ircadb_module.__doc__ or "")
    assert "never opens" in (IrcadbStyleAdapter.__doc__ or "")


def test_valid_explicit_convention_accepted() -> None:
    convention = _archive_convention()

    assert convention.image_filename == "image.nii.gz"
    assert convention.label_filename == "liver_tumor.nii.gz"
    assert convention.case_directory_pattern == "3Dircadb1.*"
    assert convention.allowed_suffixes == (".nii.gz", ".nii")


def test_convention_rejects_equal_pathlike_unsafe_and_duplicate_values() -> None:
    with pytest.raises(InvalidIrcadbConventionError, match="must differ"):
        IrcadbFilenameConvention("image.nii.gz", "image.nii.gz", "case_*", (".nii.gz",))

    for filename in (
        "/image.nii.gz",
        "dir/image.nii.gz",
        "dir\\image.nii.gz",
        "../image.nii.gz",
        " image.nii.gz",
        "image.nii.gz ",
        "file://image.nii.gz",
        "C:image.nii.gz",
        "image*.nii.gz",
        "image\n.nii.gz",
    ):
        with pytest.raises(InvalidIrcadbConventionError):
            IrcadbFilenameConvention(filename, "tumor.nii.gz", "case_*", (".nii.gz",))

    for pattern in ("/case_*", "../case_*", "case\\*", "file://case_*", "cases/**", "cases/{}"):
        with pytest.raises(InvalidIrcadbConventionError):
            IrcadbFilenameConvention("image.nii.gz", "tumor.nii.gz", pattern, (".nii.gz",))

    with pytest.raises(InvalidIrcadbConventionError):
        IrcadbFilenameConvention("image.dcm", "tumor.nii.gz", "case_*", (".dcm",))
    with pytest.raises(InvalidIrcadbConventionError):
        IrcadbFilenameConvention("image.nii", "tumor.nii", "case_*", (".nii", ".nii"))

    convention = _archive_convention()
    with pytest.raises(FrozenInstanceError):
        cast(Any, convention).image_filename = "other.nii.gz"


def test_valid_one_case_archive_like_structure_discovered(tmp_path: Path) -> None:
    _write_placeholder(tmp_path / "3Dircadb1.1/PATIENT_DICOM/image.nii.gz")
    _write_placeholder(tmp_path / "3Dircadb1.1/MASKS_DICOM/liver_tumor.nii.gz")

    inventory = _discover(tmp_path)

    assert _source_keys(inventory) == ("3Dircadb1.1",)
    assert inventory.candidates[0].image_relative_path == "3Dircadb1.1/PATIENT_DICOM/image.nii.gz"
    assert (
        inventory.candidates[0].label_relative_path == "3Dircadb1.1/MASKS_DICOM/liver_tumor.nii.gz"
    )


def test_multiple_cases_sorted_deterministically_and_leading_zeros_preserved(
    tmp_path: Path,
) -> None:
    for key in ("case_010", "case_001", "case_02"):
        _write_placeholder(tmp_path / f"cases/{key}/image.nii.gz")
        _write_placeholder(tmp_path / f"cases/{key}/tumor.nii.gz")

    inventory = _discover(
        tmp_path,
        adapter=_adapter(_nested_convention()),
        layout=NESTED_CASE_LAYOUT,
    )

    assert _source_keys(inventory) == ("case_001", "case_010", "case_02")


def test_nested_layout_and_nii_suffix_are_supported(tmp_path: Path) -> None:
    _write_placeholder(tmp_path / "cases/case_001/image.nii")
    _write_placeholder(tmp_path / "cases/case_001/tumor.nii")

    inventory = _discover(
        tmp_path,
        adapter=_adapter(_nested_convention(".nii")),
        layout=NII_CASE_LAYOUT,
    )

    assert _source_keys(inventory) == ("case_001",)
    assert inventory.candidates[0].image_relative_path == "cases/case_001/image.nii"


def test_repeated_discovery_creation_order_and_cwd_do_not_affect_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    other = tmp_path / "other"
    other.mkdir()
    for key in ("case_002", "case_001"):
        _write_placeholder(root_a / f"cases/{key}/image.nii.gz")
        _write_placeholder(root_a / f"cases/{key}/tumor.nii.gz")
    for key in ("case_001", "case_002"):
        _write_placeholder(root_b / f"cases/{key}/tumor.nii.gz")
        _write_placeholder(root_b / f"cases/{key}/image.nii.gz")

    monkeypatch.chdir(other)

    first = _discover(root_a, adapter=_adapter(_nested_convention()), layout=NESTED_CASE_LAYOUT)
    second = _discover(root_a, adapter=_adapter(_nested_convention()), layout=NESTED_CASE_LAYOUT)
    assert first == second
    assert first == _discover(
        root_b, adapter=_adapter(_nested_convention()), layout=NESTED_CASE_LAYOUT
    )


def test_missing_image_missing_label_and_empty_discovery_rejected(tmp_path: Path) -> None:
    _write_placeholder(tmp_path / "cases/case_001/tumor.nii.gz")
    with pytest.raises(IncompleteIrcadbPairError, match="missing images"):
        _discover(tmp_path, adapter=_adapter(_nested_convention()), layout=NESTED_CASE_LAYOUT)

    image_only_root = tmp_path / "image-only"
    _write_placeholder(image_only_root / "cases/case_001/image.nii.gz")
    with pytest.raises(IncompleteIrcadbPairError, match="missing labels"):
        _discover(
            image_only_root, adapter=_adapter(_nested_convention()), layout=NESTED_CASE_LAYOUT
        )

    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    with pytest.raises(IrcadbDiscoveryError, match="no matched"):
        _discover(empty_root, adapter=_adapter(_nested_convention()), layout=NESTED_CASE_LAYOUT)


def test_duplicate_image_and_label_keys_are_rejected(tmp_path: Path) -> None:
    _write_placeholder(tmp_path / "a/case_001/image.nii.gz")
    _write_placeholder(tmp_path / "b/case_001/image.nii.gz")
    _write_placeholder(tmp_path / "a/case_001/tumor.nii.gz")
    layout = AdapterLayoutSpec("*/case_*/image.nii.gz", "*/case_*/tumor.nii.gz", False)
    convention = IrcadbFilenameConvention(
        "image.nii.gz",
        "tumor.nii.gz",
        "*/case_*",
        (".nii.gz",),
    )
    with pytest.raises(AmbiguousIrcadbPairingError, match="multiple image files"):
        _discover(tmp_path, adapter=_adapter(convention), layout=layout)

    label_root = tmp_path / "labels"
    _write_placeholder(label_root / "a/case_001/image.nii.gz")
    _write_placeholder(label_root / "a/case_001/tumor.nii.gz")
    _write_placeholder(label_root / "b/case_001/tumor.nii.gz")
    with pytest.raises(AmbiguousIrcadbPairingError, match="multiple label files"):
        _discover(label_root, adapter=_adapter(convention), layout=layout)


def test_malformed_case_key_and_unexpected_matched_basename_rejected(tmp_path: Path) -> None:
    _write_placeholder(tmp_path / "cases/bad key/image.nii.gz")
    _write_placeholder(tmp_path / "cases/bad key/tumor.nii.gz")
    layout = AdapterLayoutSpec("cases/*/image.nii.gz", "cases/*/tumor.nii.gz", False)
    convention = IrcadbFilenameConvention("image.nii.gz", "tumor.nii.gz", "cases/*", (".nii.gz",))
    with pytest.raises(MalformedIrcadbCaseError, match="unsafe source key"):
        _discover(tmp_path, adapter=_adapter(convention), layout=layout)

    unexpected_root = tmp_path / "unexpected"
    _write_placeholder(unexpected_root / "cases/case_001/image-copy.nii.gz")
    _write_placeholder(unexpected_root / "cases/case_001/tumor.nii.gz")
    bad_layout = AdapterLayoutSpec("cases/case_*/image*.nii.gz", "cases/case_*/tumor.nii.gz", False)
    with pytest.raises(MalformedIrcadbCaseError, match="unexpected basename"):
        _discover(unexpected_root, adapter=_adapter(_nested_convention()), layout=bad_layout)


def test_same_resolved_file_and_cross_case_aliases_are_rejected(tmp_path: Path) -> None:
    target = tmp_path / "target/shared.nii.gz"
    _write_placeholder(target)
    (tmp_path / "cases/case_001").mkdir(parents=True)
    try:
        (tmp_path / "cases/case_001/image.nii.gz").symlink_to(target)
        (tmp_path / "cases/case_001/tumor.nii.gz").symlink_to(target)
    except OSError:
        pytest.skip("symlinks unavailable on this filesystem")
    with pytest.raises(AmbiguousIrcadbPairingError):
        _discover(tmp_path, adapter=_adapter(_nested_convention()), layout=NESTED_CASE_LAYOUT)

    alias_root = tmp_path / "alias"
    _write_placeholder(alias_root / "cases/case_001/image.nii.gz")
    _write_placeholder(alias_root / "cases/case_001/tumor.nii.gz")
    _write_placeholder(alias_root / "cases/case_002/tumor.nii.gz")
    try:
        (alias_root / "cases/case_002/image.nii.gz").symlink_to(
            alias_root / "cases/case_001/tumor.nii.gz"
        )
    except OSError:
        pytest.skip("symlinks unavailable on this filesystem")
    with pytest.raises(AmbiguousIrcadbPairingError):
        _discover(alias_root, adapter=_adapter(_nested_convention()), layout=NESTED_CASE_LAYOUT)


def test_matched_directory_dangling_and_outside_symlinks_rejected(tmp_path: Path) -> None:
    (tmp_path / "cases/case_001/image.nii.gz").mkdir(parents=True)
    _write_placeholder(tmp_path / "cases/case_001/tumor.nii.gz")
    with pytest.raises(NonRegularDatasetFileError):
        _discover(tmp_path, adapter=_adapter(_nested_convention()), layout=NESTED_CASE_LAYOUT)

    dangling_root = tmp_path / "dangling"
    _write_placeholder(dangling_root / "cases/case_001/tumor.nii.gz")
    try:
        (dangling_root / "cases/case_001/image.nii.gz").symlink_to(
            dangling_root / "cases/case_001/missing.nii.gz"
        )
    except OSError:
        pytest.skip("symlinks unavailable on this filesystem")
    with pytest.raises(MissingDatasetPathError):
        _discover(dangling_root, adapter=_adapter(_nested_convention()), layout=NESTED_CASE_LAYOUT)

    outside_root = tmp_path / "outside-link"
    outside = tmp_path / "outside-image.nii.gz"
    _write_placeholder(outside)
    _write_placeholder(outside_root / "cases/case_001/tumor.nii.gz")
    try:
        (outside_root / "cases/case_001/image.nii.gz").symlink_to(outside)
    except OSError:
        pytest.skip("symlinks unavailable on this filesystem")
    with pytest.raises(DatasetPathEscapeError):
        _discover(outside_root, adapter=_adapter(_nested_convention()), layout=NESTED_CASE_LAYOUT)


def test_recursive_directory_symlink_outside_root_is_not_followed(tmp_path: Path) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    _write_placeholder(outside / "case_001/image.nii.gz")
    _write_placeholder(outside / "case_001/tumor.nii.gz")
    try:
        (root / "linked-outside").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable on this filesystem")

    with pytest.raises(IrcadbDiscoveryError):
        _discover(root, adapter=_adapter(_recursive_convention()), layout=RECURSIVE_CASE_LAYOUT)


def test_internal_file_symlink_accepted_and_placeholder_bytes_unchanged(tmp_path: Path) -> None:
    image_target = tmp_path / "targets/image.nii.gz"
    label_target = tmp_path / "targets/tumor.nii.gz"
    image_before = _write_placeholder(image_target, "image bytes\n")
    label_before = _write_placeholder(label_target, "label bytes\n")
    (tmp_path / "cases/case_001").mkdir(parents=True)
    try:
        (tmp_path / "cases/case_001/image.nii.gz").symlink_to(image_target)
        (tmp_path / "cases/case_001/tumor.nii.gz").symlink_to(label_target)
    except OSError:
        pytest.skip("symlinks unavailable on this filesystem")

    inventory = _discover(
        tmp_path, adapter=_adapter(_nested_convention()), layout=NESTED_CASE_LAYOUT
    )

    assert _source_keys(inventory) == ("case_001",)
    assert image_target.read_bytes() == image_before
    assert label_target.read_bytes() == label_before


def test_ambiguous_error_ordering_is_deterministic(tmp_path: Path) -> None:
    _write_placeholder(tmp_path / "z/case_001/image.nii.gz")
    _write_placeholder(tmp_path / "a/case_001/image.nii.gz")
    _write_placeholder(tmp_path / "a/case_001/tumor.nii.gz")
    layout = AdapterLayoutSpec("*/case_*/image.nii.gz", "*/case_*/tumor.nii.gz", False)
    convention = IrcadbFilenameConvention(
        "image.nii.gz",
        "tumor.nii.gz",
        "*/case_*",
        (".nii.gz",),
    )

    with pytest.raises(AmbiguousIrcadbPairingError) as first_error:
        _discover(tmp_path, adapter=_adapter(convention), layout=layout)
    with pytest.raises(AmbiguousIrcadbPairingError) as second_error:
        _discover(tmp_path, adapter=_adapter(convention), layout=layout)

    assert str(first_error.value) == str(second_error.value)


def test_discovery_does_not_open_parse_or_hash_contents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_before = _write_placeholder(tmp_path / "cases/case_001/image.nii.gz", "image bytes\n")
    label_before = _write_placeholder(tmp_path / "cases/case_001/tumor.nii.gz", "label bytes\n")

    def _fail(*args: object, **kwargs: object) -> object:
        raise AssertionError("unexpected content read, parse, or hash")

    import nibabel as nib

    import protoem_ct.artifacts.hashing as hashing

    with monkeypatch.context() as context:
        context.setattr(nib, "load", _fail)
        context.setattr(hashing, "sha256_file", _fail)
        context.setattr(Path, "open", _fail)
        if importlib.util.find_spec("pydicom") is not None:
            pydicom = importlib.import_module("pydicom")
            context.setattr(cast(Any, pydicom), "dcmread", _fail)

        inventory = _discover(
            tmp_path,
            adapter=_adapter(_nested_convention()),
            layout=NESTED_CASE_LAYOUT,
        )

    assert _source_keys(inventory) == ("case_001",)
    assert (tmp_path / "cases/case_001/image.nii.gz").read_bytes() == image_before
    assert (tmp_path / "cases/case_001/tumor.nii.gz").read_bytes() == label_before


def test_protocol_serialization_boundary_and_source_key_not_persisted(tmp_path: Path) -> None:
    _write_placeholder(tmp_path / "cases/case_001/image.nii.gz")
    _write_placeholder(tmp_path / "cases/case_001/tumor.nii.gz")
    adapter = _adapter(_nested_convention())
    inventory = adapter.discover(tmp_path, NESTED_CASE_LAYOUT)

    assert isinstance(adapter, DatasetAdapter)
    with pytest.raises(Phase2ArtifactSerializationError) as error:
        phase2_artifact_to_json(cast(Any, inventory))
    assert "case_001" not in str(error.value)

    artifact_json = phase2_artifact_to_json(
        DatasetCaseRecord(
            anonymous_patient_id="anon-p001",
            anonymous_case_id="anon-c001",
            relative_image_path="images/anon-c001.nii.gz",
            relative_label_path="labels/anon-c001.nii.gz",
            image_sha256=HASH0,
            label_sha256=HASH1,
            cohort_role=PHASE2_DEVELOPMENT_COHORT_ROLE,
        )
    )
    assert "case_001" not in artifact_json


def test_no_real_dataset_root_string_and_lits_behavior_remains_unchanged(tmp_path: Path) -> None:
    dataset_name = "3D-" + "IRCADb-01"
    forbidden_root_markers = (f"/{dataset_name}", f"\\{dataset_name}", f"{dataset_name}/")
    for path in (
        Path("src/protoem_ct/data/adapters/ircadb.py"),
        Path("tests/unit/test_ircadb_adapter.py"),
    ):
        text = path.read_text(encoding="utf-8")
        for marker in forbidden_root_markers:
            assert marker not in text

    _write_placeholder(tmp_path / "volume-1.nii")
    _write_placeholder(tmp_path / "segmentation-1.nii")
    lits_inventory = LiTSStyleAdapter(
        LiTSFilenameConvention("volume-", "segmentation-", (".nii",))
    ).discover(
        tmp_path,
        AdapterLayoutSpec("volume-*.nii", "segmentation-*.nii", recursive=False),
    )

    assert _source_keys(lits_inventory) == ("1",)
