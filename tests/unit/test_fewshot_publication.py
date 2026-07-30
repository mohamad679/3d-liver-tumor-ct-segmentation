"""Unit tests for deterministic Phase 4 publication and filesystem safety."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

import pytest

from protoem_ct.artifacts import phase2_artifact_to_json
from protoem_ct.fewshot import (
    FewshotPublicationCollisionError,
    FewshotPublicationIOError,
    FewshotPublicationPathError,
    generate_and_publish_phase4_fewshot_protocol,
    load_phase4_fewshot_protocol_settings,
)
from protoem_ct.fewshot import publication as publication_module


def _load_helpers() -> Any:
    helper_path = Path(__file__).with_name("test_fewshot_protocol.py")
    spec = importlib.util.spec_from_file_location("_fewshot_protocol_helpers", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("few-shot protocol helpers could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_HELPERS = _load_helpers()


def _write_phase2_fixture(root: Path) -> tuple[Path, Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    manifest = _HELPERS._manifest()
    split = _HELPERS._split(manifest)
    lesion = _HELPERS._lesion_artifact(manifest, split)
    manifest_path = root / "manifest.json"
    split_path = root / "split.json"
    lesion_path = root / "lesion.json"
    manifest_path.write_text(phase2_artifact_to_json(manifest), encoding="utf-8")
    split_path.write_text(phase2_artifact_to_json(split), encoding="utf-8")
    lesion_path.write_text(phase2_artifact_to_json(lesion), encoding="utf-8")
    return manifest_path, split_path, lesion_path


def _publish(root: Path, output_root: Path) -> Any:
    manifest_path, split_path, lesion_path = _write_phase2_fixture(root)
    return generate_and_publish_phase4_fewshot_protocol(
        manifest_path=manifest_path,
        split_path=split_path,
        lesion_artifact_path=lesion_path,
        output_root=output_root,
        initialization_reference=_HELPERS._reference(),
        base_seed=1729,
        settings=load_phase4_fewshot_protocol_settings(
            Path("configs/phase4_fewshot_protocol.yaml").resolve()
        ),
    )


def test_atomic_publication_failure_leaves_no_partial_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "published"

    def _fail_replace(_source: Path, _dest: Path) -> None:
        raise OSError("synthetic replace failure")

    monkeypatch.setattr("protoem_ct.fewshot.publication.os.replace", _fail_replace)
    with pytest.raises(FewshotPublicationIOError):
        _publish(source_root, output_root)

    assert not output_root.exists()
    assert not (tmp_path / ".published.phase4-publication.tmp").exists()


def test_relative_publication_path_rejects_parent_traversal() -> None:
    with pytest.raises(FewshotPublicationPathError):
        publication_module._normalize_relative_publication_path("../escape.json")


def test_symlink_output_root_escape_rejected(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    real_root = tmp_path / "real"
    real_root.mkdir()
    symlink_root = tmp_path / "symlink-root"
    symlink_root.symlink_to(real_root, target_is_directory=True)

    with pytest.raises(FewshotPublicationPathError):
        _publish(source_root, symlink_root)


def test_unsafe_overwrite_rejected_and_identical_regeneration_allowed(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "published"

    first = _publish(source_root, output_root)
    second = _publish(source_root, output_root)

    assert first.support_manifest_count == 15
    assert second.reused_existing_output is True

    summary_path = output_root / "generation_summary.json"
    summary_path.write_text('{"tampered": true}\n', encoding="utf-8")

    with pytest.raises(FewshotPublicationCollisionError):
        _publish(source_root, output_root)


def test_existing_output_root_with_symlinked_child_rejected(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "published"
    output_root.mkdir()
    (output_root / "protocol").mkdir()
    external_target = tmp_path / "outside"
    external_target.mkdir()
    (output_root / "protocol" / "link").symlink_to(external_target, target_is_directory=True)

    with pytest.raises(FewshotPublicationPathError):
        _publish(source_root, output_root)


def test_published_root_contains_expected_directories_only(tmp_path: Path) -> None:
    output_root = tmp_path / "published"
    result = _publish(tmp_path / "source", output_root)

    assert result.leakage_check_passed is True
    assert sorted(path.name for path in output_root.iterdir()) == [
        "adaptation_configs",
        "generation_summary.json",
        "protocol",
        "support_manifests",
    ]
    assert sorted(path.name for path in (output_root / "support_manifests").iterdir()) == [
        "k01",
        "k02",
        "k05",
        "k10",
        "k20",
    ]
    assert os.path.isabs(str(output_root))
