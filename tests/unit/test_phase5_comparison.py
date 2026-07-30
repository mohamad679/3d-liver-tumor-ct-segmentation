"""Unit tests for deterministic Phase 5 comparison artifacts and method execution."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from protoem_ct.retrieval import (
    FIXED_METHOD_ORDER,
    FIXED_NEAREST_SUPPORT_TOP_K,
    METRICS_UNAVAILABLE_STATUS,
    NEAREST_SUPPORT_RETRIEVAL_METHOD,
    NO_RETRIEVAL_METHOD,
    PHASE5_RUN_SUMMARY_JSON_NAME,
    Phase5ComparisonCollisionError,
    Phase5ComparisonIOError,
    Phase5ComparisonPathError,
    Phase5ComparisonValidationError,
    Phase5RunSummary,
    build_phase5_synthetic_fixture,
    hash_phase5_comparison_table,
    load_phase5_foundation_retrieval_settings,
    phase5_comparison_table_from_json,
    phase5_comparison_table_identity_payload,
    phase5_comparison_table_to_json,
    phase5_run_summary_from_json,
    phase5_run_summary_to_json,
    run_and_publish_phase5_retrieval,
    run_phase5_comparison,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _config_path() -> Path:
    return REPO_ROOT / "configs" / "phase5_foundation_retrieval.yaml"


def test_exactly_two_fixed_comparison_methods() -> None:
    fixture = build_phase5_synthetic_fixture()
    comparison = run_phase5_comparison(
        query_feature_encoding=fixture.query_feature_encoding,
        support_records=fixture.support_records,
        query_patient_id=fixture.query_patient_id,
        query_case_id=fixture.query_case_id,
        query_identity=fixture.query_identity,
        dataset_manifest_hash=fixture.dataset_manifest_hash,
        query_reference_mask=fixture.query_reference_mask,
        emit_confidence_margin=True,
        nearest_support_top_k=FIXED_NEAREST_SUPPORT_TOP_K,
    )

    assert (
        tuple(item.method_name for item in comparison.ordered_method_definitions)
        == FIXED_METHOD_ORDER
    )
    assert tuple(item.method_name for item in comparison.records) == FIXED_METHOD_ORDER


def test_no_retrieval_uses_all_allowed_supports_without_retrieval() -> None:
    fixture = build_phase5_synthetic_fixture()
    comparison = run_phase5_comparison(
        query_feature_encoding=fixture.query_feature_encoding,
        support_records=tuple(reversed(fixture.support_records)),
        query_patient_id=fixture.query_patient_id,
        query_case_id=fixture.query_case_id,
        query_identity=fixture.query_identity,
        dataset_manifest_hash=fixture.dataset_manifest_hash,
        query_reference_mask=fixture.query_reference_mask,
        emit_confidence_margin=True,
    )

    record = comparison.records[0]
    assert record.method_name == NO_RETRIEVAL_METHOD
    assert record.retrieval_result_identity_sha256 is None
    assert record.support_count_used == 2
    assert record.selected_support_identifiers == ("support_case_001", "support_case_002")


def test_nearest_support_uses_exact_top_k_one_result() -> None:
    fixture = build_phase5_synthetic_fixture()
    comparison = run_phase5_comparison(
        query_feature_encoding=fixture.query_feature_encoding,
        support_records=fixture.support_records,
        query_patient_id=fixture.query_patient_id,
        query_case_id=fixture.query_case_id,
        query_identity=fixture.query_identity,
        dataset_manifest_hash=fixture.dataset_manifest_hash,
        query_reference_mask=fixture.query_reference_mask,
        emit_confidence_margin=True,
    )

    record = comparison.records[1]
    assert record.method_name == NEAREST_SUPPORT_RETRIEVAL_METHOD
    assert record.support_count_used == 1
    assert record.selected_support_identifiers == ("support_case_001",)
    assert record.retrieved_support_identifier == "support_case_001"
    assert record.retrieval_result_identity_sha256 is not None


def test_deterministic_method_ordering_and_identity() -> None:
    fixture = build_phase5_synthetic_fixture()
    first = run_phase5_comparison(
        query_feature_encoding=fixture.query_feature_encoding,
        support_records=fixture.support_records,
        query_patient_id=fixture.query_patient_id,
        query_case_id=fixture.query_case_id,
        query_identity=fixture.query_identity,
        dataset_manifest_hash=fixture.dataset_manifest_hash,
        query_reference_mask=fixture.query_reference_mask,
        emit_confidence_margin=True,
    )
    second = run_phase5_comparison(
        query_feature_encoding=fixture.query_feature_encoding,
        support_records=tuple(reversed(fixture.support_records)),
        query_patient_id=fixture.query_patient_id,
        query_case_id=fixture.query_case_id,
        query_identity=fixture.query_identity,
        dataset_manifest_hash=fixture.dataset_manifest_hash,
        query_reference_mask=fixture.query_reference_mask,
        emit_confidence_margin=True,
    )

    assert hash_phase5_comparison_table(first) == hash_phase5_comparison_table(second)
    assert phase5_comparison_table_identity_payload(
        first
    ) == phase5_comparison_table_identity_payload(second)


def test_metrics_with_known_synthetic_arrays() -> None:
    fixture = build_phase5_synthetic_fixture()
    comparison = run_phase5_comparison(
        query_feature_encoding=fixture.query_feature_encoding,
        support_records=fixture.support_records,
        query_patient_id=fixture.query_patient_id,
        query_case_id=fixture.query_case_id,
        query_identity=fixture.query_identity,
        dataset_manifest_hash=fixture.dataset_manifest_hash,
        query_reference_mask=fixture.query_reference_mask,
        emit_confidence_margin=True,
    )

    no_retrieval_record, nearest_record = comparison.records
    assert no_retrieval_record.dice == pytest.approx(1.0)
    assert no_retrieval_record.iou == pytest.approx(1.0)
    assert nearest_record.dice == pytest.approx(1.0)
    assert nearest_record.iou == pytest.approx(1.0)


def test_metrics_unavailable_without_reference_mask() -> None:
    fixture = build_phase5_synthetic_fixture()
    comparison = run_phase5_comparison(
        query_feature_encoding=fixture.query_feature_encoding,
        support_records=fixture.support_records,
        query_patient_id=fixture.query_patient_id,
        query_case_id=fixture.query_case_id,
        query_identity=fixture.query_identity,
        dataset_manifest_hash=fixture.dataset_manifest_hash,
        query_reference_mask=None,
        emit_confidence_margin=True,
    )

    for record in comparison.records:
        assert record.metrics_availability_status == METRICS_UNAVAILABLE_STATUS
        assert record.dice is None
        assert record.iou is None


def test_canonical_mapping_round_trip_and_unknown_field_rejection() -> None:
    fixture = build_phase5_synthetic_fixture()
    comparison = run_phase5_comparison(
        query_feature_encoding=fixture.query_feature_encoding,
        support_records=fixture.support_records,
        query_patient_id=fixture.query_patient_id,
        query_case_id=fixture.query_case_id,
        query_identity=fixture.query_identity,
        dataset_manifest_hash=fixture.dataset_manifest_hash,
        query_reference_mask=fixture.query_reference_mask,
        emit_confidence_margin=True,
    )
    round_trip = phase5_comparison_table_from_json(phase5_comparison_table_to_json(comparison))

    assert round_trip == comparison

    bad_payload = (
        phase5_comparison_table_to_json(comparison)
        .decode("utf-8")
        .replace(
            '"support_set_identity_sha256"',
            '"unexpected":true,"support_set_identity_sha256"',
            1,
        )
    )
    with pytest.raises(Phase5ComparisonValidationError):
        phase5_comparison_table_from_json(bad_payload)


def test_run_summary_round_trip() -> None:
    summary = Phase5RunSummary(
        schema_name="phase5_run_summary",
        schema_version="v1",
        comparison_identity_sha256="a" * 64,
        support_set_identity_sha256="b" * 64,
        method_count=2,
        nearest_support_top_k=1,
        metrics_reference_available=True,
        foundation_model_adapter_enabled=False,
        published_filenames=(
            "effective_config.json",
            "phase5_comparison.json",
            "phase5_comparison_table.md",
            "phase5_run_summary.json",
        ),
    )

    assert phase5_run_summary_from_json(phase5_run_summary_to_json(summary)) == summary


def test_atomic_publication_failure_leaves_no_partial_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = load_phase5_foundation_retrieval_settings(_config_path())
    output_root = tmp_path / "published"

    def _fail_replace(_source: Path, _dest: Path) -> None:
        raise OSError("synthetic replace failure")

    monkeypatch.setattr(
        "protoem_ct.retrieval.publication.os.replace",
        _fail_replace,
    )
    with pytest.raises(Phase5ComparisonIOError):
        run_and_publish_phase5_retrieval(output_root=output_root, settings=settings)

    assert not output_root.exists()
    assert not (tmp_path / ".published.phase5-publication.tmp").exists()


def test_byte_identical_regeneration_and_unsafe_output_rejection(tmp_path: Path) -> None:
    settings = load_phase5_foundation_retrieval_settings(_config_path())
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"

    run_and_publish_phase5_retrieval(output_root=first_root, settings=settings)
    run_and_publish_phase5_retrieval(output_root=second_root, settings=settings)

    assert {
        path.relative_to(first_root).as_posix(): path.read_bytes()
        for path in first_root.rglob("*")
        if path.is_file()
    } == {
        path.relative_to(second_root).as_posix(): path.read_bytes()
        for path in second_root.rglob("*")
        if path.is_file()
    }

    (first_root / PHASE5_RUN_SUMMARY_JSON_NAME).write_text('{"tampered":true}\n', encoding="utf-8")
    with pytest.raises(Phase5ComparisonCollisionError):
        run_and_publish_phase5_retrieval(output_root=first_root, settings=settings)

    with pytest.raises(Phase5ComparisonPathError):
        run_and_publish_phase5_retrieval(
            output_root=REPO_ROOT / "reports" / "phase5", settings=settings
        )


def test_existing_absolute_external_temp_directory_is_accepted() -> None:
    settings = load_phase5_foundation_retrieval_settings(_config_path())
    output_root = Path(tempfile.mkdtemp(prefix="protoem-ct-phase5-existing.", dir="/tmp"))

    try:
        result = run_and_publish_phase5_retrieval(output_root=output_root, settings=settings)

        assert result.output_root == output_root.resolve(strict=True)
        assert result.comparison_path.exists()
    finally:
        shutil.rmtree(output_root.resolve(strict=False), ignore_errors=True)


def test_tmp_lexical_and_resolved_forms_are_consistent_when_tmp_is_alias() -> None:
    lexical_root = Path(tempfile.mkdtemp(prefix="protoem-ct-phase5-tmp-alias.", dir="/tmp"))
    resolved_root = lexical_root.resolve(strict=True)
    settings = load_phase5_foundation_retrieval_settings(_config_path())

    try:
        first = run_and_publish_phase5_retrieval(output_root=lexical_root, settings=settings)
        second = run_and_publish_phase5_retrieval(output_root=resolved_root, settings=settings)

        assert first.output_root == resolved_root
        assert second.output_root == resolved_root
        assert second.reused_existing_output is True
    finally:
        shutil.rmtree(resolved_root, ignore_errors=True)


def test_repository_internal_output_root_remains_rejected() -> None:
    settings = load_phase5_foundation_retrieval_settings(_config_path())

    with pytest.raises(Phase5ComparisonPathError):
        run_and_publish_phase5_retrieval(
            output_root=REPO_ROOT / "src" / "phase5-forbidden-output",
            settings=settings,
        )


@pytest.mark.skipif(not hasattr(Path, "symlink_to"), reason="Symlinks unsupported")
def test_symlink_escape_rejected(tmp_path: Path) -> None:
    settings = load_phase5_foundation_retrieval_settings(_config_path())
    real_root = tmp_path / "real"
    real_root.mkdir()
    symlink_root = tmp_path / "linked"
    symlink_root.symlink_to(real_root, target_is_directory=True)

    with pytest.raises(Phase5ComparisonPathError):
        run_and_publish_phase5_retrieval(output_root=symlink_root, settings=settings)


@pytest.mark.skipif(not hasattr(Path, "symlink_to"), reason="Symlinks unsupported")
def test_symlink_parent_escape_rejected(tmp_path: Path) -> None:
    settings = load_phase5_foundation_retrieval_settings(_config_path())
    real_root = tmp_path / "real"
    real_root.mkdir()
    symlink_parent = tmp_path / "linked-parent"
    symlink_parent.symlink_to(real_root, target_is_directory=True)

    with pytest.raises(Phase5ComparisonPathError):
        run_and_publish_phase5_retrieval(
            output_root=symlink_parent / "published",
            settings=settings,
        )


def test_parent_traversal_output_root_rejected(tmp_path: Path) -> None:
    settings = load_phase5_foundation_retrieval_settings(_config_path())
    lexical_parent = tmp_path / "lexical-parent"
    lexical_parent.mkdir()

    with pytest.raises(Phase5ComparisonPathError):
        run_and_publish_phase5_retrieval(
            output_root=lexical_parent / ".." / "published",
            settings=settings,
        )


def test_identical_regeneration_on_same_root_is_byte_identical(tmp_path: Path) -> None:
    settings = load_phase5_foundation_retrieval_settings(_config_path())
    output_root = tmp_path / "published"

    first = run_and_publish_phase5_retrieval(output_root=output_root, settings=settings)
    first_bytes = {
        path.relative_to(first.output_root).as_posix(): path.read_bytes()
        for path in first.output_root.rglob("*")
        if path.is_file()
    }
    second = run_and_publish_phase5_retrieval(output_root=output_root, settings=settings)
    second_bytes = {
        path.relative_to(second.output_root).as_posix(): path.read_bytes()
        for path in second.output_root.rglob("*")
        if path.is_file()
    }

    assert second.reused_existing_output is True
    assert second_bytes == first_bytes
