"""Regression tests for shared Phase 2 JSON publication behavior."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import protoem_ct.data._phase2_publication as publication


def _publish(output_path: Path, text: str = '{"ok": true}\n') -> None:
    publication.publish_text_no_overwrite(
        text=text,
        output_path=output_path,
        temporary_exists_message="temporary output already exists",
        final_exists_message="final output already exists",
    )


def test_hard_link_publication_writes_exact_bytes_and_removes_temp(tmp_path: Path) -> None:
    output = tmp_path / "artifact.json"
    text = '{"a": 1}\n'

    _publish(output, text)

    assert output.read_text(encoding="utf-8") == text
    assert not (tmp_path / ".artifact.json.tmp").exists()


def test_unsupported_hard_link_fallback_publishes_identical_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hard_link_output = tmp_path / "hard-link.json"
    fallback_output = tmp_path / "fallback.json"
    text = '{"a": 1, "b": [2, 3]}\n'
    _publish(hard_link_output, text)

    def _unsupported_link(_source: Path, _dest: Path) -> None:
        raise OSError("synthetic hard-link unsupported")

    monkeypatch.setattr("protoem_ct.data._phase2_publication.os.link", _unsupported_link)
    _publish(fallback_output, text)

    assert fallback_output.read_bytes() == hard_link_output.read_bytes()
    assert not (tmp_path / ".fallback.json.tmp").exists()


def test_existing_final_output_is_not_overwritten_when_fallback_is_used(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "artifact.json"
    output.write_text("existing\n", encoding="utf-8")

    def _unsupported_link(_source: Path, _dest: Path) -> None:
        raise OSError("synthetic hard-link unsupported")

    monkeypatch.setattr("protoem_ct.data._phase2_publication.os.link", _unsupported_link)
    with pytest.raises(publication.Phase2PublicationExistingOutputError) as exc_info:
        _publish(output, '{"new": true}\n')

    assert str(exc_info.value) == "final output already exists"
    assert output.read_text(encoding="utf-8") == "existing\n"
    assert not (tmp_path / ".artifact.json.tmp").exists()


def test_publication_failure_removes_temporary_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "artifact.json"

    def _unsupported_link(_source: Path, _dest: Path) -> None:
        raise OSError("synthetic hard-link unsupported")

    def _failed_rename(_source: Path, _dest: Path) -> None:
        raise OSError("synthetic rename failure")

    monkeypatch.setattr("protoem_ct.data._phase2_publication.os.link", _unsupported_link)
    monkeypatch.setattr("protoem_ct.data._phase2_publication.os.rename", _failed_rename)

    with pytest.raises(publication.Phase2PublicationIOError) as exc_info:
        _publish(output)

    assert str(exc_info.value) == "failed to publish Phase 2 JSON artifact"
    assert not output.exists()
    assert not (tmp_path / ".artifact.json.tmp").exists()


def test_source_input_and_external_artifact_are_not_modified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_input = tmp_path / "source-input.json"
    external_artifact = tmp_path / "external-artifact.json"
    source_input.write_text("source\n", encoding="utf-8")
    external_artifact.write_text("external\n", encoding="utf-8")

    def _unsupported_link(_source: Path, _dest: Path) -> None:
        raise OSError("synthetic hard-link unsupported")

    monkeypatch.setattr("protoem_ct.data._phase2_publication.os.link", _unsupported_link)
    _publish(tmp_path / "published.json", '{"published": true}\n')

    assert source_input.read_text(encoding="utf-8") == "source\n"
    assert external_artifact.read_text(encoding="utf-8") == "external\n"


def test_error_messages_do_not_leak_paths_ids_or_artifact_contents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "secret-pat_123-case_456.json"

    def _unsupported_link(_source: Path, _dest: Path) -> None:
        raise OSError("synthetic hard-link unsupported")

    def _failed_rename(_source: Path, _dest: Path) -> None:
        raise OSError("synthetic rename failure")

    monkeypatch.setattr("protoem_ct.data._phase2_publication.os.link", _unsupported_link)
    monkeypatch.setattr("protoem_ct.data._phase2_publication.os.rename", _failed_rename)

    with pytest.raises(publication.Phase2PublicationIOError) as exc_info:
        _publish(output, '{"artifact": "secret contents"}\n')

    message = str(exc_info.value)
    assert str(tmp_path) not in message
    assert "pat_123" not in message
    assert "case_456" not in message
    assert "secret contents" not in message
    assert os.linesep not in message
