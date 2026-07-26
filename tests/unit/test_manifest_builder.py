"""Unit tests for Phase 2 anonymous LiTS manifest building."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, cast

import pytest

import protoem_ct.artifacts.hashing as artifact_hashing
from protoem_ct.artifacts import (
    DatasetManifest,
    Phase2ArtifactSerializationError,
    Phase2ArtifactValidationError,
    hash_dataset_manifest,
    hash_dataset_root_fingerprint,
    phase2_artifact_from_json,
    phase2_artifact_to_json,
)
from protoem_ct.artifacts.hashing import HashFileError
from protoem_ct.data import (
    SUPPORTED_LITS_SUFFIXES,
    AdapterLayoutSpec,
    AnonymousIdCollisionError,
    AnonymousIdConfig,
    DatasetPathEscapeError,
    ExistingManifestOutputError,
    InvalidAnonymousIdConfigError,
    InvalidIdKeyMaterialError,
    InvalidManifestMetadataError,
    InventoryDryRunResult,
    LiTSFilenameConvention,
    SourceHashingError,
    UnsafeManifestOutputPathError,
    anonymous_lits_case_id,
    anonymous_lits_patient_id,
    build_lits_development_manifest,
    dry_run_lits_inventory,
    read_id_key_file,
)
from protoem_ct.data import manifest_builder as builder

LEGACY_LAYOUT = AdapterLayoutSpec("volume-*.nii*", "segmentation-*.nii*", recursive=False)
MSD_LAYOUT = AdapterLayoutSpec("imagesTr/liver_*.nii.gz", "labelsTr/liver_*.nii.gz", False)
KEY = b"k" * 32
OTHER_KEY = b"z" * 32
GENERATED_AT_UTC = "2026-07-25T00:00:00Z"
GIT_COMMIT = "3b82869"


def _legacy_convention() -> LiTSFilenameConvention:
    return LiTSFilenameConvention("volume-", "segmentation-", SUPPORTED_LITS_SUFFIXES)


def _msd_convention() -> LiTSFilenameConvention:
    return LiTSFilenameConvention("liver_", "liver_", (".nii.gz",))


def _config(**overrides: object) -> AnonymousIdConfig:
    values: dict[str, object] = {
        "project_namespace": "protoem-phase2",
        "patient_id_prefix": "anon-p-",
        "case_id_prefix": "anon-c-",
        "digest_length": 32,
    }
    values.update(overrides)
    return AnonymousIdConfig(**cast(Any, values))


def _write_placeholder(path: Path, text: str) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))
    return path.read_bytes()


def _write_legacy_case(root: Path, key: str, *, suffix: str = ".nii.gz") -> tuple[bytes, bytes]:
    image = _write_placeholder(root / f"volume-{key}{suffix}", f"image-{key}\n")
    label = _write_placeholder(root / f"segmentation-{key}{suffix}", f"label-{key}\n")
    return image, label


def _write_msd_case(root: Path, key: str) -> tuple[bytes, bytes]:
    image = _write_placeholder(root / f"imagesTr/liver_{key}.nii.gz", f"image-{key}\n")
    label = _write_placeholder(root / f"labelsTr/liver_{key}.nii.gz", f"label-{key}\n")
    return image, label


def _build_manifest(
    dataset_root: Path,
    output_path: Path,
    *,
    layout: AdapterLayoutSpec = LEGACY_LAYOUT,
    convention: LiTSFilenameConvention | None = None,
    generated_at_utc: str = GENERATED_AT_UTC,
    git_commit: str = GIT_COMMIT,
    id_key: bytes = KEY,
    anonymous_id_config: AnonymousIdConfig | None = None,
) -> DatasetManifest:
    result = build_lits_development_manifest(
        dataset_root,
        layout=layout,
        convention=convention or _legacy_convention(),
        dataset_id="lits-development",
        generated_at_utc=generated_at_utc,
        git_commit=git_commit,
        anonymous_id_config=anonymous_id_config or _config(),
        id_key=id_key,
        output_path=output_path,
    )
    return result.manifest


def test_anonymous_id_config_validation_and_immutability() -> None:
    config = _config()
    assert config.digest_length == 32

    bad_values = (
        {"project_namespace": "bad namespace"},
        {"project_namespace": "file://bad"},
        {"patient_id_prefix": "anon/p-"},
        {"case_id_prefix": "anon c-"},
        {"case_id_prefix": "anon-p-"},
        {"digest_length": 31},
    )
    for overrides in bad_values:
        with pytest.raises(InvalidAnonymousIdConfigError):
            _config(**overrides)

    with pytest.raises(FrozenInstanceError):
        cast(Any, config).digest_length = 64


def test_hmac_identifiers_are_deterministic_domain_separated_and_one_way() -> None:
    config = _config(project_namespace="namespace-a")

    patient_id = anonymous_lits_patient_id("000", config=config, id_key=KEY)
    case_id = anonymous_lits_case_id("000", config=config, id_key=KEY)

    assert anonymous_lits_patient_id("000", config=config, id_key=KEY) == patient_id
    assert patient_id != case_id
    assert anonymous_lits_patient_id("000", config=config, id_key=OTHER_KEY) != patient_id
    assert anonymous_lits_patient_id("000", config=_config(), id_key=KEY) != patient_id
    assert "000" not in patient_id
    assert patient_id.startswith("anon-p-")
    assert case_id.startswith("anon-c-")


def test_key_file_validation_and_boundaries(tmp_path: Path) -> None:
    key_file = tmp_path / "synthetic.key"
    before = b"s" * 32
    key_file.write_bytes(before)

    assert read_id_key_file(key_file) == before
    assert key_file.read_bytes() == before

    bad_key = tmp_path / "short.key"
    bad_key.write_bytes(b"short")
    with pytest.raises(InvalidIdKeyMaterialError):
        read_id_key_file(bad_key)
    with pytest.raises(InvalidIdKeyMaterialError):
        read_id_key_file(tmp_path / "missing.key")
    with pytest.raises(InvalidIdKeyMaterialError):
        read_id_key_file(tmp_path)
    with pytest.raises(InvalidIdKeyMaterialError):
        read_id_key_file(Path("relative.key"))

    oversized = tmp_path / "oversized.key"
    oversized.write_bytes(b"x" * (builder.MAX_ID_KEY_FILE_BYTES + 1))
    with pytest.raises(InvalidIdKeyMaterialError):
        read_id_key_file(oversized)


def test_dry_run_inventory_returns_safe_summary_without_open_or_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_legacy_case(tmp_path, "1")

    def _fail(*args: object, **kwargs: object) -> object:
        raise AssertionError("unexpected file content access")

    import protoem_ct.artifacts.hashing as hashing

    with monkeypatch.context() as context:
        context.setattr(hashing, "sha256_file", _fail)
        context.setattr(Path, "open", _fail)
        first = dry_run_lits_inventory(
            tmp_path,
            layout=LEGACY_LAYOUT,
            convention=_legacy_convention(),
        )
        second = dry_run_lits_inventory(
            tmp_path,
            layout=LEGACY_LAYOUT,
            convention=_legacy_convention(),
        )

    assert first == second
    assert first == InventoryDryRunResult("lits_style", "1", 1, 1, 1)
    assert "source_case_key" not in repr(first)


def test_key_file_is_not_read_during_dry_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_legacy_case(tmp_path, "1")

    def _fail_read_key_file(path: Path) -> bytes:
        raise AssertionError(f"unexpected key read: {path}")

    monkeypatch.setattr(builder, "read_id_key_file", _fail_read_key_file)

    result = dry_run_lits_inventory(tmp_path, layout=LEGACY_LAYOUT, convention=_legacy_convention())

    assert result.case_count == 1


def test_valid_legacy_manifest_publication_and_hash_verification(tmp_path: Path) -> None:
    _write_legacy_case(tmp_path / "dataset", "2")
    output = tmp_path / "out" / "manifest.json"

    manifest = _build_manifest(tmp_path / "dataset", output)

    assert output.is_file()
    assert manifest == phase2_artifact_from_json(
        output.read_text(encoding="utf-8"), DatasetManifest
    )
    assert manifest.manifest_hash == hash_dataset_manifest(manifest)
    assert manifest.dataset_root_fingerprint == hash_dataset_root_fingerprint(
        adapter_name=manifest.adapter_name,
        adapter_version=manifest.adapter_version,
        cases=manifest.cases,
    )
    assert manifest.cohort_role == "development"
    assert manifest.case_count == 1


def test_valid_msd_layout_manifest_publication(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _write_msd_case(root, "000")
    output = tmp_path / "out" / "manifest.json"

    manifest = _build_manifest(
        root,
        output,
        layout=MSD_LAYOUT,
        convention=_msd_convention(),
    )

    assert manifest.case_count == 1
    assert manifest.cases[0].relative_image_path == "imagesTr/liver_000.nii.gz"
    assert manifest.cases[0].relative_label_path == "labelsTr/liver_000.nii.gz"


def test_manifest_ordering_and_ids_are_stable_unique_and_source_free(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    for key in ("2", "000", "10"):
        _write_legacy_case(root, key)

    manifest = _build_manifest(root, tmp_path / "out" / "manifest.json")
    json_text = phase2_artifact_to_json(manifest)

    assert manifest.cases == tuple(
        sorted(manifest.cases, key=lambda case: (case.anonymous_patient_id, case.anonymous_case_id))
    )
    assert len({case.anonymous_patient_id for case in manifest.cases}) == 3
    assert len({case.anonymous_case_id for case in manifest.cases}) == 3
    for forbidden in ("source_case_key", "anonymous_id_config"):
        assert forbidden not in json_text
    assert str(root) not in json_text
    assert KEY.hex() not in json_text


def test_manifest_json_is_byte_identical_across_equivalent_external_outputs(tmp_path: Path) -> None:
    root_a = tmp_path / "a" / "dataset"
    root_b = tmp_path / "b" / "dataset"
    for root in (root_a, root_b):
        _write_legacy_case(root, "1")
        _write_legacy_case(root, "2")

    manifest_a = _build_manifest(root_a, tmp_path / "out-a" / "manifest.json")
    manifest_b = _build_manifest(root_b, tmp_path / "out-b" / "manifest.json")

    assert phase2_artifact_to_json(manifest_a) == phase2_artifact_to_json(manifest_b)


def test_file_timestamp_and_git_changes_affect_expected_hashes(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _write_legacy_case(root, "1")

    baseline = _build_manifest(root, tmp_path / "out-a" / "manifest.json")
    timestamp_changed = _build_manifest(
        root,
        tmp_path / "out-b" / "manifest.json",
        generated_at_utc="2026-07-26T00:00:00Z",
    )
    commit_changed = _build_manifest(
        root, tmp_path / "out-c" / "manifest.json", git_commit="abc1234"
    )

    assert timestamp_changed.manifest_hash != baseline.manifest_hash
    assert timestamp_changed.dataset_root_fingerprint == baseline.dataset_root_fingerprint
    assert commit_changed.manifest_hash != baseline.manifest_hash
    assert commit_changed.dataset_root_fingerprint == baseline.dataset_root_fingerprint

    (root / "volume-1.nii.gz").write_bytes(b"changed image bytes\n")
    bytes_changed = _build_manifest(root, tmp_path / "out-d" / "manifest.json")
    assert bytes_changed.dataset_root_fingerprint != baseline.dataset_root_fingerprint
    assert bytes_changed.manifest_hash != baseline.manifest_hash


def test_directory_creation_order_does_not_change_manifest(tmp_path: Path) -> None:
    root_a = tmp_path / "a" / "dataset"
    root_b = tmp_path / "b" / "dataset"
    for key in ("2", "1"):
        _write_legacy_case(root_a, key)
    for key in ("1", "2"):
        _write_legacy_case(root_b, key)

    assert phase2_artifact_to_json(
        _build_manifest(root_a, tmp_path / "out-a" / "manifest.json")
    ) == phase2_artifact_to_json(_build_manifest(root_b, tmp_path / "out-b" / "manifest.json"))


def test_manifest_publication_rejects_unsafe_or_existing_outputs(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _write_legacy_case(root, "1")

    existing = tmp_path / "existing.json"
    existing.write_text("occupied\n", encoding="utf-8")
    with pytest.raises(ExistingManifestOutputError):
        _build_manifest(root, existing)

    with pytest.raises(UnsafeManifestOutputPathError):
        _build_manifest(root, root / "manifest.json")
    with pytest.raises(UnsafeManifestOutputPathError):
        _build_manifest(root, root)
    with pytest.raises(UnsafeManifestOutputPathError):
        _build_manifest(root, tmp_path)

    parent_file = tmp_path / "parent-file"
    parent_file.write_text("not a directory\n", encoding="utf-8")
    with pytest.raises(UnsafeManifestOutputPathError):
        _build_manifest(root, parent_file / "manifest.json")
    with pytest.raises(UnsafeManifestOutputPathError):
        _build_manifest(root, Path("relative-manifest.json"))


def test_hashing_error_collision_and_bad_metadata_leave_no_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "dataset"
    _write_legacy_case(root, "1")
    output = tmp_path / "out" / "manifest.json"

    def _fail_hash(path: Path) -> str:
        raise HashFileError("synthetic hash failure")

    monkeypatch.setattr(artifact_hashing, "sha256_file", _fail_hash)
    with pytest.raises(SourceHashingError):
        _build_manifest(root, output)
    assert not output.exists()
    assert not (output.parent / ".manifest.json.tmp").exists()

    monkeypatch.undo()
    with pytest.raises(InvalidManifestMetadataError):
        build_lits_development_manifest(
            root,
            layout=LEGACY_LAYOUT,
            convention=_legacy_convention(),
            dataset_id="Bad Dataset",
            generated_at_utc=GENERATED_AT_UTC,
            git_commit=GIT_COMMIT,
            anonymous_id_config=_config(),
            id_key=KEY,
            output_path=output,
        )
    assert not output.exists()


def test_anonymization_collision_and_duplicate_hash_contract_leave_no_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "dataset"
    for key in ("1", "2"):
        _write_legacy_case(root, key)
    output = tmp_path / "out" / "manifest.json"

    def _colliding_id(
        source_case_key: str,
        *,
        config: AnonymousIdConfig,
        id_key: bytes,
        domain: str,
        prefix: str,
    ) -> str:
        return f"{prefix}{'0' * 32}"

    monkeypatch.setattr(builder, "_hmac_identifier", _colliding_id)
    with pytest.raises(AnonymousIdCollisionError):
        _build_manifest(root, output)
    assert not output.exists()

    monkeypatch.undo()
    (root / "volume-2.nii.gz").write_bytes((root / "volume-1.nii.gz").read_bytes())
    with pytest.raises(Phase2ArtifactValidationError, match="image hashes"):
        _build_manifest(root, output)
    assert not output.exists()


def test_source_files_remain_unchanged_and_symlink_escape_rejected(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    image_before, label_before = _write_legacy_case(root, "1")
    _build_manifest(root, tmp_path / "out" / "manifest.json")

    assert (root / "volume-1.nii.gz").read_bytes() == image_before
    assert (root / "segmentation-1.nii.gz").read_bytes() == label_before

    escape_root = tmp_path / "escape"
    escape_root.mkdir()
    outside = tmp_path / "outside.nii.gz"
    outside.write_text("outside\n", encoding="utf-8")
    try:
        (escape_root / "volume-1.nii.gz").symlink_to(outside)
        _write_placeholder(escape_root / "segmentation-1.nii.gz", "label\n")
    except OSError:
        pytest.skip("symlinks unavailable on this filesystem")
    with pytest.raises(DatasetPathEscapeError):
        _build_manifest(escape_root, tmp_path / "escape-out" / "manifest.json")


def test_serialization_boundaries_for_results_and_secrets(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _write_legacy_case(root, "1")
    manifest = _build_manifest(root, tmp_path / "out" / "manifest.json")
    json_text = phase2_artifact_to_json(manifest)

    with pytest.raises(Phase2ArtifactSerializationError):
        phase2_artifact_to_json(
            cast(
                Any,
                dry_run_lits_inventory(
                    root,
                    layout=LEGACY_LAYOUT,
                    convention=_legacy_convention(),
                ),
            )
        )
    assert KEY.decode("utf-8") not in json_text
    assert "synthetic.key" not in json_text
    assert str(root) not in json_text
