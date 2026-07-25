"""Unit tests for deterministic Phase 1 report generation."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any, cast

import nibabel as nib
import pytest

from protoem_ct.artifacts import (
    EvaluationArtifact,
    InferenceArtifact,
    PreprocessArtifact,
    ReportArtifact,
    SyntheticManifest,
    ValidationArtifact,
    artifact_from_json,
    artifact_to_dict,
    sha256_file,
)
from protoem_ct.data.manifest_validation import validate_synthetic_manifest
from protoem_ct.data.preprocessing import preprocess_synthetic_dataset
from protoem_ct.data.synthetic import (
    SyntheticDataConfig,
    SyntheticDatasetResult,
    create_synthetic_dataset,
)
from protoem_ct.evaluation.dummy_inference import run_dummy_inference
from protoem_ct.evaluation.metrics import evaluate_predictions
from protoem_ct.reporting import (
    ReportInputArtifactError,
    ReportOutputCollisionError,
    ReportPathError,
    ReportRenderingError,
    generate_synthetic_report,
)
from protoem_ct.reporting import generate as report_generate

GIT_COMMIT = "c3cf8ae"
CREATED_AT_UTC = "2026-07-25T01:00:00Z"
VALIDATION_CREATED_AT_UTC = "2026-07-25T01:05:00Z"
PREPROCESS_CREATED_AT_UTC = "2026-07-25T01:10:00Z"
INFERENCE_CREATED_AT_UTC = "2026-07-25T01:15:00Z"
EVALUATION_CREATED_AT_UTC = "2026-07-25T01:20:00Z"
REPORT_CREATED_AT_UTC = "2026-07-25T01:25:00Z"


def _synthetic_config(**overrides: object) -> SyntheticDataConfig:
    values: dict[str, object] = {
        "dataset_id": "phase1-synthetic",
        "seed": 1729,
        "case_count": 3,
        "shape": (8, 8, 6),
        "spacing": (1.5, 1.5, 2.0),
        "generated_data_root": PurePosixPath("generated/phase1/data"),
    }
    values.update(overrides)
    return SyntheticDataConfig(**values)  # type: ignore[arg-type]


def _dataset(root: Path) -> SyntheticDatasetResult:
    root.mkdir(parents=True, exist_ok=True)
    return create_synthetic_dataset(
        _synthetic_config(),
        output_root=root / "data-root",
        git_commit=GIT_COMMIT,
        created_at_utc=CREATED_AT_UTC,
    )


def _prepared(
    root: Path,
) -> tuple[
    Path,
    SyntheticManifest,
    Path,
    ValidationArtifact,
    Path,
    PreprocessArtifact,
    Path,
    InferenceArtifact,
    Path,
    EvaluationArtifact,
]:
    result = _dataset(root)
    validation_path = root / "artifacts" / "validation.json"
    validation = validate_synthetic_manifest(
        result.manifest_path,
        data_root=root / "data-root",
        output_path=validation_path,
        git_commit=GIT_COMMIT,
        created_at_utc=VALIDATION_CREATED_AT_UTC,
    )
    preprocess_path = root / "artifacts" / "preprocess.json"
    preprocess = preprocess_synthetic_dataset(
        result.manifest_path,
        validation_path,
        data_root=root / "data-root",
        output_root=root / "preprocessed",
        artifact_output_path=preprocess_path,
        config_path=Path("configs/experiment/phase1.yaml"),
        git_commit=GIT_COMMIT,
        created_at_utc=PREPROCESS_CREATED_AT_UTC,
    )
    inference_path = root / "artifacts" / "inference.json"
    inference = run_dummy_inference(
        preprocess_path,
        preprocessed_root=root / "preprocessed",
        output_root=root / "predictions",
        artifact_output_path=inference_path,
        config_path=Path("configs/experiment/phase1.yaml"),
        git_commit=GIT_COMMIT,
        created_at_utc=INFERENCE_CREATED_AT_UTC,
    )
    evaluation_path = root / "artifacts" / "evaluation.json"
    evaluation = evaluate_predictions(
        preprocess_path,
        inference_path,
        preprocessed_root=root / "preprocessed",
        prediction_root=root / "predictions",
        artifact_output_path=evaluation_path,
        git_commit=GIT_COMMIT,
        created_at_utc=EVALUATION_CREATED_AT_UTC,
    )
    return (
        result.manifest_path,
        result.manifest,
        validation_path,
        validation,
        preprocess_path,
        preprocess,
        inference_path,
        inference,
        evaluation_path,
        evaluation,
    )


def _generate_report(root: Path) -> tuple[ReportArtifact, Path, Path]:
    (
        manifest_path,
        _manifest,
        validation_path,
        _validation,
        preprocess_path,
        _preprocess,
        inference_path,
        _inference,
        evaluation_path,
        _evaluation,
    ) = _prepared(root)
    report_path = root / "reports" / "synthetic_report.md"
    report_artifact_path = root / "artifacts" / "report.json"
    artifact = generate_synthetic_report(
        manifest_path,
        validation_path,
        preprocess_path,
        inference_path,
        evaluation_path,
        report_output_path=report_path,
        artifact_output_path=report_artifact_path,
        git_commit=GIT_COMMIT,
        created_at_utc=REPORT_CREATED_AT_UTC,
    )
    return artifact, report_path, report_artifact_path


def _write_payload(path: Path, artifact: object, **overrides: object) -> None:
    payload = cast(dict[str, Any], artifact_to_dict(cast(Any, artifact)))
    payload.update(overrides)
    path.write_text(
        json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _input_paths(root: Path) -> tuple[Path, Path, Path, Path, Path]:
    manifest_path = root / "data-root" / "generated" / "phase1" / "data" / "synthetic_manifest.json"
    return (
        manifest_path,
        root / "artifacts" / "validation.json",
        root / "artifacts" / "preprocess.json",
        root / "artifacts" / "inference.json",
        root / "artifacts" / "evaluation.json",
    )


def _run_generate_with_existing_inputs(
    root: Path,
    *,
    report_output_path: Path | None = None,
    artifact_output_path: Path | None = None,
) -> ReportArtifact:
    manifest_path, validation_path, preprocess_path, inference_path, evaluation_path = _input_paths(
        root
    )
    return generate_synthetic_report(
        manifest_path,
        validation_path,
        preprocess_path,
        inference_path,
        evaluation_path,
        report_output_path=report_output_path or root / "reports" / "synthetic_report.md",
        artifact_output_path=artifact_output_path or root / "artifacts" / "report.json",
        git_commit=GIT_COMMIT,
        created_at_utc=REPORT_CREATED_AT_UTC,
    )


def test_valid_linked_artifacts_generate_report(tmp_path: Path) -> None:
    artifact, report_path, report_artifact_path = _generate_report(tmp_path)
    persisted = artifact_from_json(report_artifact_path.read_text(encoding="utf-8"), ReportArtifact)

    assert report_path.is_file()
    assert report_artifact_path.is_file()
    assert persisted == artifact
    assert artifact.stage == "reporting"
    assert artifact.report_format == "markdown"
    assert artifact.method == "dummy"
    assert artifact.evaluated_case_count == 3


def test_report_generation_reads_json_artifacts_only_and_never_uses_nibabel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepared(tmp_path)
    manifest_path, validation_path, preprocess_path, inference_path, evaluation_path = _input_paths(
        tmp_path
    )

    def fail_load(*_args: object, **_kwargs: object) -> object:
        msg = "nibabel load must not be called during report generation"
        raise AssertionError(msg)

    monkeypatch.setattr(nib, "load", fail_load)

    generate_synthetic_report(
        manifest_path,
        validation_path,
        preprocess_path,
        inference_path,
        evaluation_path,
        report_output_path=tmp_path / "reports" / "synthetic_report.md",
        artifact_output_path=tmp_path / "artifacts" / "report.json",
        git_commit=GIT_COMMIT,
        created_at_utc=REPORT_CREATED_AT_UTC,
    )


def test_report_has_fixed_sections_required_fields_and_warnings(tmp_path: Path) -> None:
    artifact, report_path, _report_artifact_path = _generate_report(tmp_path)
    text = report_path.read_text(encoding="utf-8")
    evaluation = artifact_from_json(
        (tmp_path / "artifacts" / "evaluation.json").read_text(encoding="utf-8"),
        EvaluationArtifact,
    )
    headers = [
        "# ProtoEM-CT Phase 1 Synthetic Pipeline Report",
        "## Run Metadata",
        "## Dataset Summary",
        "## Validation Summary",
        "## Preprocessing Summary",
        "## Dummy-Inference Summary",
        "## Evaluation Summary",
        "## Ordered Per-Case Metric Table",
        "## Artifact Linkage",
        "## Reproducibility",
    ]

    positions = [text.index(header) for header in headers]

    assert positions == sorted(positions)
    assert "synthetic Phase 1 pipeline report" in text
    assert "dummy inference" in text
    assert "pipeline verification" in text
    assert "not scientific model results" in text
    assert "- Schema version: 1" in text
    assert f"- Git commit: {GIT_COMMIT}" in text
    assert f"- Created at UTC: {REPORT_CREATED_AT_UTC}" in text
    assert "- Dataset ID: phase1-synthetic" in text
    assert "- Case count: 3" in text
    assert "- Shape: 8 x 8 x 6" in text
    assert "- Spacing: 1.500000 x 1.500000 x 2.000000" in text
    assert "- Valid case count: 3" in text
    assert "- Invalid case count: 0" in text
    assert "  - clip_min: -1000.000000" in text
    assert "  - output_max: 1.000000" in text
    assert "- Method: dummy" in text
    assert "- Deterministic seed: 2718" in text
    assert "- Threshold: 0.500000" in text
    assert f"- Macro Dice: {artifact.macro_mean_dice:.6f}" in text
    assert f"- Macro IoU: {artifact.macro_mean_iou:.6f}" in text
    assert f"- Micro Dice: {artifact.micro_dice:.6f}" in text
    assert f"- Micro IoU: {artifact.micro_iou:.6f}" in text
    assert f"- Total TP: {evaluation.total_true_positives}" in text
    assert f"- Total FP: {evaluation.total_false_positives}" in text
    assert f"- Total FN: {evaluation.total_false_negatives}" in text
    assert f"- Total TN: {evaluation.total_true_negatives}" in text
    assert f"- Pipeline config hash: {artifact.config_hash}" in text
    assert f"- Manifest hash: {artifact.manifest_hash}" in text
    assert text.endswith("\n")
    assert not text.endswith("\n\n")


def test_ordered_per_case_table_is_deterministic_with_fixed_numeric_format(
    tmp_path: Path,
) -> None:
    artifact, report_path, _report_artifact_path = _generate_report(tmp_path)
    text = report_path.read_text(encoding="utf-8")
    rows = [line for line in text.splitlines() if line.startswith("| synthetic-")]
    evaluation = artifact_from_json(
        (tmp_path / "artifacts" / "evaluation.json").read_text(encoding="utf-8"),
        EvaluationArtifact,
    )

    assert artifact.evaluated_case_count == len(rows)
    assert [row.split("|")[1].strip() for row in rows] == [
        metric.case_id for metric in evaluation.case_metrics
    ]
    for row in rows:
        dice_value = row.split("|")[-3].strip()
        iou_value = row.split("|")[-2].strip()
        assert len(dice_value.split(".")[-1]) == 6
        assert len(iou_value.split(".")[-1]) == 6


def test_report_contains_no_absolute_paths_nan_infinity_or_forbidden_claims(
    tmp_path: Path,
) -> None:
    _artifact, report_path, _report_artifact_path = _generate_report(tmp_path)
    text = report_path.read_text(encoding="utf-8")
    lowered = text.lower()

    assert str(tmp_path) not in text
    assert "NaN" not in text
    assert "Infinity" not in text
    for forbidden in (
        "model performance",
        "clinical validity",
        "state of the art",
        "generalization",
    ):
        assert forbidden not in lowered


def test_repeated_generation_produces_byte_identical_markdown(tmp_path: Path) -> None:
    first_artifact, first_report_path, _first_artifact_path = _generate_report(tmp_path / "first")
    second_artifact, second_report_path, _second_artifact_path = _generate_report(
        tmp_path / "second"
    )

    assert first_report_path.read_bytes() == second_report_path.read_bytes()
    assert first_artifact.report_sha256 == second_artifact.report_sha256


def test_report_sha256_matches_persisted_report_and_artifact_round_trip(tmp_path: Path) -> None:
    artifact, report_path, report_artifact_path = _generate_report(tmp_path)
    persisted = artifact_from_json(report_artifact_path.read_text(encoding="utf-8"), ReportArtifact)

    assert sha256_file(report_path) == artifact.report_sha256
    assert persisted.report_sha256 == artifact.report_sha256
    assert persisted.reported_metric_names == ("dice", "iou")


def test_config_hash_mismatch_rejected(tmp_path: Path) -> None:
    (
        _manifest_path,
        _manifest,
        _validation_path,
        _validation,
        _preprocess_path,
        _preprocess,
        _inference_path,
        _inference,
        evaluation_path,
        evaluation,
    ) = _prepared(tmp_path)
    _write_payload(evaluation_path, evaluation, config_hash="1" * 64)

    with pytest.raises(ReportInputArtifactError, match="config hashes"):
        _run_generate_with_existing_inputs(tmp_path)


def test_manifest_hash_mismatch_rejected(tmp_path: Path) -> None:
    (
        _manifest_path,
        _manifest,
        _validation_path,
        _validation,
        _preprocess_path,
        _preprocess,
        inference_path,
        inference,
        _evaluation_path,
        _evaluation,
    ) = _prepared(tmp_path)
    _write_payload(inference_path, inference, manifest_hash="1" * 64)

    with pytest.raises(ReportInputArtifactError, match="manifest hashes"):
        _run_generate_with_existing_inputs(tmp_path)


def test_incorrect_stage_rejected(tmp_path: Path) -> None:
    (
        _manifest_path,
        _manifest,
        validation_path,
        validation,
        _preprocess_path,
        _preprocess,
        _inference_path,
        _inference,
        _evaluation_path,
        _evaluation,
    ) = _prepared(tmp_path)
    _write_payload(validation_path, validation, stage="evaluate")

    with pytest.raises(ReportInputArtifactError, match="invalid validation artifact"):
        _run_generate_with_existing_inputs(tmp_path)


def test_case_order_mismatch_rejected(tmp_path: Path) -> None:
    (
        _manifest_path,
        _manifest,
        validation_path,
        validation,
        _preprocess_path,
        _preprocess,
        _inference_path,
        _inference,
        _evaluation_path,
        _evaluation,
    ) = _prepared(tmp_path)
    _write_payload(
        validation_path,
        validation,
        validated_case_ids=list(reversed(validation.validated_case_ids)),
    )

    with pytest.raises(ReportInputArtifactError, match="case IDs"):
        _run_generate_with_existing_inputs(tmp_path)


def test_case_count_mismatch_rejected(tmp_path: Path) -> None:
    (
        _manifest_path,
        _manifest,
        validation_path,
        validation,
        _preprocess_path,
        _preprocess,
        _inference_path,
        _inference,
        _evaluation_path,
        _evaluation,
    ) = _prepared(tmp_path)
    _write_payload(validation_path, validation, valid_case_count=2)

    with pytest.raises(ReportInputArtifactError, match="case count"):
        _run_generate_with_existing_inputs(tmp_path)


def test_invalid_case_count_above_zero_rejected(tmp_path: Path) -> None:
    (
        _manifest_path,
        _manifest,
        validation_path,
        validation,
        _preprocess_path,
        _preprocess,
        _inference_path,
        _inference,
        _evaluation_path,
        _evaluation,
    ) = _prepared(tmp_path)
    _write_payload(validation_path, validation, invalid_case_count=1)

    with pytest.raises(ReportInputArtifactError, match="invalid_case_count"):
        _run_generate_with_existing_inputs(tmp_path)


def test_non_dummy_method_rejected(tmp_path: Path) -> None:
    (
        _manifest_path,
        _manifest,
        _validation_path,
        _validation,
        _preprocess_path,
        _preprocess,
        inference_path,
        inference,
        _evaluation_path,
        _evaluation,
    ) = _prepared(tmp_path)
    _write_payload(inference_path, inference, method="model")

    with pytest.raises(ReportInputArtifactError, match="invalid inference artifact"):
        _run_generate_with_existing_inputs(tmp_path)


def test_malformed_json_rejected(tmp_path: Path) -> None:
    _prepared(tmp_path)
    manifest_path, validation_path, preprocess_path, inference_path, evaluation_path = _input_paths(
        tmp_path
    )
    evaluation_path.write_text("{\n", encoding="utf-8")

    with pytest.raises(ReportInputArtifactError, match="invalid evaluation artifact"):
        generate_synthetic_report(
            manifest_path,
            validation_path,
            preprocess_path,
            inference_path,
            evaluation_path,
            report_output_path=tmp_path / "reports" / "synthetic_report.md",
            artifact_output_path=tmp_path / "artifacts" / "report.json",
            git_commit=GIT_COMMIT,
            created_at_utc=REPORT_CREATED_AT_UTC,
        )


def test_missing_input_artifact_rejected(tmp_path: Path) -> None:
    _prepared(tmp_path)
    manifest_path, validation_path, preprocess_path, inference_path, _evaluation_path = (
        _input_paths(tmp_path)
    )

    with pytest.raises(ReportInputArtifactError, match="failed to read evaluation"):
        generate_synthetic_report(
            manifest_path,
            validation_path,
            preprocess_path,
            inference_path,
            tmp_path / "artifacts" / "missing-evaluation.json",
            report_output_path=tmp_path / "reports" / "synthetic_report.md",
            artifact_output_path=tmp_path / "artifacts" / "report.json",
            git_commit=GIT_COMMIT,
            created_at_utc=REPORT_CREATED_AT_UTC,
        )


def test_existing_report_output_rejected(tmp_path: Path) -> None:
    _prepared(tmp_path)
    report_path = tmp_path / "reports" / "synthetic_report.md"
    report_path.parent.mkdir()
    report_path.write_text("existing\n", encoding="utf-8")

    with pytest.raises(ReportOutputCollisionError, match="overwrite"):
        _run_generate_with_existing_inputs(tmp_path, report_output_path=report_path)


def test_existing_artifact_output_rejected(tmp_path: Path) -> None:
    _prepared(tmp_path)
    artifact_path = tmp_path / "artifacts" / "report.json"
    artifact_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ReportOutputCollisionError, match="overwrite"):
        _run_generate_with_existing_inputs(tmp_path, artifact_output_path=artifact_path)


def test_identical_output_paths_rejected(tmp_path: Path) -> None:
    _prepared(tmp_path)
    output_path = tmp_path / "reports" / "same-output.md"

    with pytest.raises(ReportPathError, match="different files"):
        _run_generate_with_existing_inputs(
            tmp_path,
            report_output_path=output_path,
            artifact_output_path=output_path,
        )


def test_parent_traversal_rejected(tmp_path: Path) -> None:
    _prepared(tmp_path)
    manifest_path, validation_path, preprocess_path, inference_path, evaluation_path = _input_paths(
        tmp_path
    )

    with pytest.raises(ReportPathError, match="parent traversal"):
        generate_synthetic_report(
            manifest_path,
            validation_path,
            preprocess_path,
            inference_path,
            evaluation_path,
            report_output_path=Path("../synthetic_report.md"),
            artifact_output_path=tmp_path / "artifacts" / "report.json",
            git_commit=GIT_COMMIT,
            created_at_utc=REPORT_CREATED_AT_UTC,
        )


def test_failed_rendering_leaves_no_partial_report_artifact_or_temp_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepared(tmp_path)
    manifest_path, validation_path, preprocess_path, inference_path, evaluation_path = _input_paths(
        tmp_path
    )
    report_path = tmp_path / "reports" / "synthetic_report.md"
    report_artifact_path = tmp_path / "artifacts" / "report.json"

    def fail_render(*_args: object, **_kwargs: object) -> str:
        msg = "forced rendering failure"
        raise ReportRenderingError(msg)

    monkeypatch.setattr(report_generate, "_render_markdown_report", fail_render)

    with pytest.raises(ReportRenderingError, match="forced rendering failure"):
        generate_synthetic_report(
            manifest_path,
            validation_path,
            preprocess_path,
            inference_path,
            evaluation_path,
            report_output_path=report_path,
            artifact_output_path=report_artifact_path,
            git_commit=GIT_COMMIT,
            created_at_utc=REPORT_CREATED_AT_UTC,
        )

    assert not report_path.exists()
    assert not report_artifact_path.exists()
    assert not report_path.parent.exists()
    assert not any(path.name.startswith(".synthetic_report.md.") for path in tmp_path.rglob("*"))
    assert not any(path.name.startswith(".report.json.") for path in tmp_path.rglob("*"))
