"""Integration tests for Phase 2 LiTS manifest CLI commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from protoem_ct.artifacts import (
    DatasetManifest,
    hash_dataset_manifest,
    hash_dataset_root_fingerprint,
    phase2_artifact_from_json,
)
from protoem_ct.cli.main import app

GENERATED_AT_UTC = "2026-07-25T00:00:00Z"
GIT_COMMIT = "3b82869"
SECRET = b"cli-synthetic-secret-key-material-32"


def _invoke(*args: str) -> Any:
    runner = CliRunner()
    return runner.invoke(app, [*args])


def _write_placeholder(path: Path, text: str) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))
    return path.read_bytes()


def _write_legacy_case(root: Path, key: str) -> None:
    _write_placeholder(root / f"volume-{key}.nii.gz", f"image-{key}\n")
    _write_placeholder(root / f"segmentation-{key}.nii.gz", f"label-{key}\n")


def _write_msd_case(root: Path, key: str) -> None:
    _write_placeholder(root / f"imagesTr/liver_{key}.nii.gz", f"image-{key}\n")
    _write_placeholder(root / f"labelsTr/liver_{key}.nii.gz", f"label-{key}\n")


def _write_key_file(path: Path, key: bytes = SECRET) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(key)


def _legacy_inventory_args(root: Path) -> list[str]:
    return [
        "inventory-lits",
        "--dataset-root",
        str(root),
        "--image-pattern",
        "volume-*.nii*",
        "--label-pattern",
        "segmentation-*.nii*",
        "--no-recursive",
        "--image-prefix",
        "volume-",
        "--label-prefix",
        "segmentation-",
    ]


def _msd_inventory_args(root: Path) -> list[str]:
    return [
        "inventory-lits",
        "--dataset-root",
        str(root),
        "--image-pattern",
        "imagesTr/liver_*.nii.gz",
        "--label-pattern",
        "labelsTr/liver_*.nii.gz",
        "--no-recursive",
        "--image-prefix",
        "liver_",
        "--label-prefix",
        "liver_",
    ]


def _legacy_build_args(root: Path, output: Path, key_file: Path) -> list[str]:
    return [
        "build-lits-manifest",
        "--dataset-root",
        str(root),
        "--image-pattern",
        "volume-*.nii*",
        "--label-pattern",
        "segmentation-*.nii*",
        "--no-recursive",
        "--image-prefix",
        "volume-",
        "--label-prefix",
        "segmentation-",
        "--dataset-id",
        "lits-development",
        "--project-namespace",
        "protoem-phase2",
        "--patient-id-prefix",
        "anon-p-",
        "--case-id-prefix",
        "anon-c-",
        "--digest-length",
        "32",
        "--id-key-file",
        str(key_file),
        "--output",
        str(output),
        "--git-commit",
        GIT_COMMIT,
        "--generated-at-utc",
        GENERATED_AT_UTC,
    ]


def _msd_build_args(root: Path, output: Path, key_file: Path) -> list[str]:
    args = _legacy_build_args(root, output, key_file)
    replacements = {
        "volume-*.nii*": "imagesTr/liver_*.nii.gz",
        "segmentation-*.nii*": "labelsTr/liver_*.nii.gz",
        "volume-": "liver_",
        "segmentation-": "liver_",
    }
    return [replacements.get(item, item) for item in args]


def _error_text(result: Any) -> str:
    return getattr(result, "stderr", "") or result.output


def test_legacy_and_msd_dry_run_inventory_outputs_safe_summary(tmp_path: Path) -> None:
    legacy_root = tmp_path / "legacy"
    msd_root = tmp_path / "msd"
    _write_legacy_case(legacy_root, "legacycase")
    _write_msd_case(msd_root, "msdcase")

    legacy = _invoke(*_legacy_inventory_args(legacy_root))
    msd = _invoke(*_msd_inventory_args(msd_root))

    for result, root, forbidden_key in (
        (legacy, legacy_root, "legacycase"),
        (msd, msd_root, "msdcase"),
    ):
        assert result.exit_code == 0
        assert "inventory success" in result.output
        assert "adapter: lits_style@1" in result.output
        assert "case count: 1" in result.output
        assert "image count: 1" in result.output
        assert "label count: 1" in result.output
        assert "no files were opened or hashed" in result.output
        assert str(root) not in result.output
        assert forbidden_key not in result.output
        assert "volume-" not in result.output
        assert "liver_" not in result.output


def test_legacy_manifest_cli_publication_outputs_only_permitted_summary(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    key_file = tmp_path / "keys" / "synthetic.key"
    output = tmp_path / "generated" / "manifest.json"
    _write_legacy_case(root, "001")
    _write_key_file(key_file)

    result = _invoke(*_legacy_build_args(root, output, key_file))

    assert result.exit_code == 0
    assert "manifest generation success" in result.output
    assert "dataset ID: lits-development" in result.output
    assert "case count: 1" in result.output
    assert f"manifest output path: {output}" in result.output
    assert "dataset-root fingerprint: " in result.output
    assert "manifest hash: " in result.output
    assert "LiTS development cohort" in result.output
    assert "LiTS and MSD Task03 Liver are not independent cohorts" in result.output
    assert str(root) not in result.output
    assert str(key_file) not in result.output
    assert SECRET.decode("utf-8") not in result.output
    assert "001" not in result.output

    manifest = phase2_artifact_from_json(output.read_text(encoding="utf-8"), DatasetManifest)
    assert manifest.manifest_hash == hash_dataset_manifest(manifest)
    assert manifest.dataset_root_fingerprint == hash_dataset_root_fingerprint(
        adapter_name=manifest.adapter_name,
        adapter_version=manifest.adapter_version,
        cases=manifest.cases,
    )


def test_msd_style_manifest_cli_publication_parses(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    key_file = tmp_path / "keys" / "synthetic.key"
    output = tmp_path / "generated" / "manifest.json"
    _write_msd_case(root, "msdcase")
    _write_key_file(key_file)

    result = _invoke(*_msd_build_args(root, output, key_file))

    assert result.exit_code == 0
    manifest = phase2_artifact_from_json(output.read_text(encoding="utf-8"), DatasetManifest)
    assert manifest.case_count == 1
    assert manifest.cases[0].relative_image_path == "imagesTr/liver_msdcase.nii.gz"
    assert "msdcase" not in result.output
    assert str(root) not in result.output


def test_cli_errors_exit_nonzero_without_traceback_or_secret_leak(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    key_file = tmp_path / "keys" / "synthetic.key"
    output = tmp_path / "generated" / "manifest.json"
    _write_placeholder(root / "volume-001.nii.gz", "image\n")
    _write_key_file(key_file)

    missing_pair = _invoke(*_legacy_inventory_args(root))
    assert missing_pair.exit_code != 0
    assert "Traceback" not in _error_text(missing_pair)
    assert str(root) not in _error_text(missing_pair)
    assert "001" not in _error_text(missing_pair)

    short_key = tmp_path / "keys" / "short.key"
    short_key.write_bytes(b"short")
    invalid_key = _invoke(*_legacy_build_args(root, output, short_key))
    assert invalid_key.exit_code != 0
    assert "Traceback" not in _error_text(invalid_key)
    assert str(short_key) not in _error_text(invalid_key)
    assert SECRET.decode("utf-8") not in _error_text(invalid_key)

    _write_placeholder(root / "segmentation-001.nii.gz", "label\n")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("occupied\n", encoding="utf-8")
    existing_output = _invoke(*_legacy_build_args(root, output, key_file))
    assert existing_output.exit_code != 0
    assert "Traceback" not in _error_text(existing_output)
    assert str(root) not in _error_text(existing_output)

    inside_output = _invoke(*_legacy_build_args(root, root / "manifest.json", key_file))
    assert inside_output.exit_code != 0
    assert "Traceback" not in _error_text(inside_output)
    assert str(key_file) not in _error_text(inside_output)
