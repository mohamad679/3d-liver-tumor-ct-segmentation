from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from protoem_ct.artifacts.hashing import sha256_file, sha256_json
from protoem_ct.baselines import _log_paths
from protoem_ct.baselines import gate3 as gate3_module
from protoem_ct.baselines import nnunet as nnunet_module
from protoem_ct.baselines._mlflow_local import Phase3MlflowVerificationResult
from protoem_ct.baselines.gate3 import (
    PHASE3_GATE3_REPORT_VERSION,
    Phase3Gate3ExecutionError,
    Phase3Gate3Report,
    Phase3Gate3ValidationError,
    _audit_nnunet_logs,
    _require_mlflow_verification,
    _verify_persisted_baseline_provenance,
    phase3_gate3_report_from_json,
    phase3_gate3_report_to_json,
)
from protoem_ct.baselines.metrics import (
    baseline_metric_report_to_json,
    build_baseline_metric_report,
    compute_baseline_case_metrics,
)
from protoem_ct.baselines.nnunet import _write_log
from protoem_ct.baselines.paths import ValidatedBaselineRunPaths
from protoem_ct.baselines.predictions import (
    BASELINE_PREDICTION_RECORD_VERSION,
    BaselinePredictionRecord,
    baseline_prediction_manifest_to_json,
    build_baseline_prediction_manifest,
)
from protoem_ct.baselines.provenance import (
    BaselineRunProvenance,
    baseline_run_provenance_to_json,
)


def _report() -> Phase3Gate3Report:
    payload: dict[str, Any] = {
        "amp_enabled": False,
        "baseline_environment_lock_sha256": "1" * 64,
        "both_checkpoint_sha256": {"monai_segresnet": "2" * 64, "nnunet_v2": "3" * 64},
        "both_metric_report_sha256": {"monai_segresnet": "4" * 64, "nnunet_v2": "5" * 64},
        "both_prediction_manifest_sha256": {"monai_segresnet": "6" * 64, "nnunet_v2": "7" * 64},
        "both_provenance_sha256": {"monai_segresnet": "8" * 64, "nnunet_v2": "9" * 64},
        "contract_version": PHASE3_GATE3_REPORT_VERSION,
        "cpu_shape_smoke_status_by_baseline": {"monai_segresnet": True, "nnunet_v2": True},
        "deterministic_metric_json_verified_by_baseline": {
            "monai_segresnet": True,
            "nnunet_v2": True,
        },
        "failure_codes": [],
        "git_commit": "a" * 40,
        "initial_loss_by_baseline": {"monai_segresnet": 1.0, "nnunet_v2": 1.0},
        "final_loss_by_baseline": {"monai_segresnet": 0.5, "nnunet_v2": 0.4},
        "best_loss_by_baseline": {"monai_segresnet": 0.5, "nnunet_v2": 0.4},
        "mlflow_metadata_verified_by_baseline": {"monai_segresnet": True, "nnunet_v2": True},
        "monai_run_config_sha256": "b" * 64,
        "nnunet_execution_strategy_identifier": (
            "official_plan_preprocess_plus_trainer_network_tiny_cpu_v1"
        ),
        "nnunet_run_config_sha256": "c" * 64,
        "overall_status": "completed",
        "overfit_status_by_baseline": {"monai_segresnet": True, "nnunet_v2": True},
        "run_identifier": "phase3_gate3_test",
        "selected_device": "cpu",
        "synthetic_fixture_artifact_sha256": "d" * 64,
        "training_step_count_by_baseline": {"monai_segresnet": 5, "nnunet_v2": 6},
    }
    return Phase3Gate3Report(
        artifact_hash=sha256_json(payload),
        contract_version=cast(str, payload["contract_version"]),
        run_identifier=cast(str, payload["run_identifier"]),
        git_commit=cast(str, payload["git_commit"]),
        baseline_environment_lock_sha256=cast(str, payload["baseline_environment_lock_sha256"]),
        synthetic_fixture_artifact_sha256=cast(str, payload["synthetic_fixture_artifact_sha256"]),
        selected_device="cpu",
        amp_enabled=False,
        nnunet_execution_strategy_identifier=cast(
            str, payload["nnunet_execution_strategy_identifier"]
        ),
        nnunet_run_config_sha256=cast(str, payload["nnunet_run_config_sha256"]),
        monai_run_config_sha256=cast(str, payload["monai_run_config_sha256"]),
        both_checkpoint_sha256=cast(dict[str, str], payload["both_checkpoint_sha256"]),
        both_prediction_manifest_sha256=cast(
            dict[str, str], payload["both_prediction_manifest_sha256"]
        ),
        both_metric_report_sha256=cast(dict[str, str], payload["both_metric_report_sha256"]),
        both_provenance_sha256=cast(dict[str, str], payload["both_provenance_sha256"]),
        cpu_shape_smoke_status_by_baseline=cast(
            dict[str, bool], payload["cpu_shape_smoke_status_by_baseline"]
        ),
        overfit_status_by_baseline=cast(dict[str, bool], payload["overfit_status_by_baseline"]),
        initial_loss_by_baseline=cast(dict[str, float], payload["initial_loss_by_baseline"]),
        final_loss_by_baseline=cast(dict[str, float], payload["final_loss_by_baseline"]),
        best_loss_by_baseline=cast(dict[str, float], payload["best_loss_by_baseline"]),
        training_step_count_by_baseline=cast(
            dict[str, int], payload["training_step_count_by_baseline"]
        ),
        deterministic_metric_json_verified_by_baseline=cast(
            dict[str, bool], payload["deterministic_metric_json_verified_by_baseline"]
        ),
        mlflow_metadata_verified_by_baseline=cast(
            dict[str, bool], payload["mlflow_metadata_verified_by_baseline"]
        ),
        overall_status="completed",
        failure_codes=(),
    )


def test_gate3_report_round_trip() -> None:
    report = _report()
    parsed = phase3_gate3_report_from_json(phase3_gate3_report_to_json(report))
    assert parsed.artifact_hash == report.artifact_hash


def test_gate3_report_tamper_changes_hash_validation() -> None:
    payload = json.loads(phase3_gate3_report_to_json(_report()))
    payload["selected_device"] = "cpu:tampered"
    with pytest.raises(Phase3Gate3ValidationError):
        phase3_gate3_report_from_json(json.dumps(payload).encode("utf-8"))


def test_gate3_requires_matching_mlflow_verification_result() -> None:
    verification = Phase3MlflowVerificationResult(
        database_filename="tracking.db",
        database_byte_size=1024,
        experiment_name="phase3_monai_segresnet",
        run_identifier="phase3_gate3_test_monai_segresnet",
        baseline_family="monai_segresnet",
        parameter_count=12,
        metric_count=3,
        verified=True,
    )

    _require_mlflow_verification(
        verification,
        baseline_family="monai_segresnet",
        run_identifier="phase3_gate3_test_monai_segresnet",
    )
    with pytest.raises(Phase3Gate3ExecutionError) as exc_info:
        _require_mlflow_verification(
            verification,
            baseline_family="nnunet_v2",
            run_identifier="phase3_gate3_test_nnunet_v2",
        )

    assert exc_info.value.failure_code == "phase3_mlflow_metadata_invalid"


def _write_supporting_artifacts(
    tmp_path: Path, *, checkpoint_override: str | None = None
) -> tuple[BaselineRunProvenance, Path, Path, Path, Path]:
    checkpoint_path = tmp_path / "checkpoint.bin"
    checkpoint_path.write_bytes(b"checkpoint\n")
    checkpoint_sha256 = checkpoint_override or sha256_file(checkpoint_path)
    prediction_manifest = build_baseline_prediction_manifest(
        baseline_family="nnunet_v2",
        run_identifier="phase3_gate3_test",
        dataset_manifest_sha256="1" * 64,
        development_split_sha256="2" * 64,
        checkpoint_sha256=checkpoint_sha256,
        prediction_records=(
            BaselinePredictionRecord(
                contract_version=BASELINE_PREDICTION_RECORD_VERSION,
                case_identifier="case_001",
                prediction_path="case_001.nii.gz",
                prediction_sha256="3" * 64,
                byte_size=123,
                shape=(24, 24, 16),
                voxel_spacing_mm=(1.0, 1.0, 1.0),
                affine=(
                    (1.0, 0.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0, 0.0),
                    (0.0, 0.0, 0.0, 1.0),
                ),
            ),
        ),
    )
    prediction_manifest_path = tmp_path / "prediction_manifest.json"
    prediction_manifest_path.write_bytes(baseline_prediction_manifest_to_json(prediction_manifest))
    metric_report = build_baseline_metric_report(
        baseline_family="nnunet_v2",
        run_identifier="phase3_gate3_test",
        dataset_manifest_sha256="1" * 64,
        development_split_sha256="2" * 64,
        prediction_manifest_sha256=prediction_manifest.artifact_hash,
        metric_config_sha256="4" * 64,
        nsd_tolerance_mm=1.0,
        case_records=(
            compute_baseline_case_metrics(
                case_identifier="case_001",
                ground_truth_mask=np.zeros((3, 3, 3), dtype=np.uint8),
                prediction_mask=np.zeros((3, 3, 3), dtype=np.uint8),
                voxel_spacing_mm=(1.0, 1.0, 1.0),
                nsd_tolerance_mm=1.0,
            ),
        ),
    )
    metric_report_path = tmp_path / "metric_report.json"
    metric_report_path.write_bytes(baseline_metric_report_to_json(metric_report))
    payload = {
        "amp_enabled": False,
        "baseline_family": "nnunet_v2",
        "checkpoint_sha256": checkpoint_sha256,
        "config_sha256": "5" * 64,
        "contract_version": "baseline_run_provenance_v1",
        "dataset_manifest_sha256": "1" * 64,
        "development_split_sha256": "2" * 64,
        "end_timestamp": "2026-07-29T01:00:00Z",
        "failure_codes": [],
        "git_commit": "a" * 40,
        "metrics_artifact_sha256": metric_report.artifact_hash,
        "package_versions": {"torch": "2.2.2"},
        "platform_machine": "arm64",
        "platform_system": "Darwin",
        "prediction_manifest_sha256": prediction_manifest.artifact_hash,
        "python_version": "3.11.9",
        "run_identifier": "phase3_gate3_test",
        "seed": 1729,
        "selected_device": "cpu",
        "start_timestamp": "2026-07-29T00:00:00Z",
        "status": "completed",
    }
    provenance = BaselineRunProvenance(
        artifact_hash=sha256_json(payload),
        baseline_family="nnunet_v2",
        run_identifier="phase3_gate3_test",
        git_commit="a" * 40,
        config_sha256="5" * 64,
        dataset_manifest_sha256="1" * 64,
        development_split_sha256="2" * 64,
        seed=1729,
        python_version="3.11.9",
        platform_system="Darwin",
        platform_machine="arm64",
        selected_device="cpu",
        amp_enabled=False,
        package_versions={"torch": "2.2.2"},
        start_timestamp="2026-07-29T00:00:00Z",
        end_timestamp="2026-07-29T01:00:00Z",
        status="completed",
        checkpoint_sha256=checkpoint_sha256,
        prediction_manifest_sha256=prediction_manifest.artifact_hash,
        metrics_artifact_sha256=metric_report.artifact_hash,
        failure_codes=(),
        contract_version="baseline_run_provenance_v1",
    )
    provenance_path = tmp_path / "provenance.json"
    return (
        provenance,
        provenance_path,
        checkpoint_path,
        prediction_manifest_path,
        metric_report_path,
    )


def test_gate3_requires_persisted_completed_provenance_file(tmp_path: Path) -> None:
    provenance, provenance_path, checkpoint_path, prediction_manifest_path, metric_report_path = (
        _write_supporting_artifacts(tmp_path)
    )

    with pytest.raises(Phase3Gate3ExecutionError) as exc_info:
        _verify_persisted_baseline_provenance(
            provenance_path=provenance_path,
            expected_provenance=provenance,
            checkpoint_path=checkpoint_path,
            prediction_manifest_path=prediction_manifest_path,
            metric_report_path=metric_report_path,
        )

    assert exc_info.value.failure_code == "phase3_provenance_missing"


def test_gate3_rejects_tampered_persisted_provenance(tmp_path: Path) -> None:
    provenance, provenance_path, checkpoint_path, prediction_manifest_path, metric_report_path = (
        _write_supporting_artifacts(tmp_path)
    )
    payload = json.loads(baseline_run_provenance_to_json(provenance))
    payload["checkpoint_sha256"] = "f" * 64
    provenance_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(Phase3Gate3ExecutionError) as exc_info:
        _verify_persisted_baseline_provenance(
            provenance_path=provenance_path,
            expected_provenance=provenance,
            checkpoint_path=checkpoint_path,
            prediction_manifest_path=prediction_manifest_path,
            metric_report_path=metric_report_path,
        )

    assert exc_info.value.failure_code == "phase3_provenance_hash_mismatch"


def test_gate3_rejects_provenance_cross_link_mismatch(tmp_path: Path) -> None:
    provenance, provenance_path, checkpoint_path, prediction_manifest_path, metric_report_path = (
        _write_supporting_artifacts(tmp_path, checkpoint_override="f" * 64)
    )
    provenance_path.write_bytes(baseline_run_provenance_to_json(provenance))

    with pytest.raises(Phase3Gate3ExecutionError) as exc_info:
        _verify_persisted_baseline_provenance(
            provenance_path=provenance_path,
            expected_provenance=provenance,
            checkpoint_path=checkpoint_path,
            prediction_manifest_path=prediction_manifest_path,
            metric_report_path=metric_report_path,
        )

    assert exc_info.value.failure_code == "phase3_provenance_link_mismatch"


def _log_run_paths(tmp_path: Path) -> ValidatedBaselineRunPaths:
    run_root = tmp_path / "nnunet_v2_run"
    logs_dir = run_root / "logs"
    logs_dir.mkdir(parents=True)
    return ValidatedBaselineRunPaths(
        baseline_family="nnunet_v2",
        run_root=run_root,
        checkpoints_dir=run_root / "checkpoints",
        predictions_dir=run_root / "predictions",
        metrics_dir=run_root / "metrics",
        mlflow_dir=run_root / "mlflow",
        logs_dir=logs_dir,
        temporary_dir=run_root / "temporary",
    )


def test_gate3_log_audit_accepts_exact_preserved_https_url(tmp_path: Path) -> None:
    run_paths = _log_run_paths(tmp_path)
    url = "https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/resenc_presets.md"
    (run_paths.logs_dir / "stdout.log").write_text(f"Documentation: {url}\n", encoding="utf-8")

    _audit_nnunet_logs(run_paths=run_paths)


def test_gate3_log_audit_rejects_path_without_exposing_it(tmp_path: Path) -> None:
    run_paths = _log_run_paths(tmp_path)
    (run_paths.logs_dir / "stderr.log").write_text(
        'File "/private/runtime path/module.py", line 17\n',
        encoding="utf-8",
    )

    with pytest.raises(Phase3Gate3ExecutionError) as exc_info:
        _audit_nnunet_logs(run_paths=run_paths)

    assert exc_info.value.failure_code == "phase3_log_redaction_failed"
    assert "/private/" not in str(exc_info.value)


def test_log_writer_and_gate3_audit_call_same_canonical_detector(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_paths = _log_run_paths(tmp_path)
    calls: list[str] = []

    def fake_detector(value: str) -> bool:
        calls.append(value)
        return False

    monkeypatch.setattr(_log_paths, "_contains_absolute_filesystem_path", fake_detector)
    _write_log(
        run_paths.logs_dir / "stdout.log",
        "safe output\n",
        stage_name="shared-detector",
        log_redaction_roots=None,
        runtime_environment={},
        working_directory=run_paths.temporary_dir,
        logs_directory=run_paths.logs_dir,
    )
    _audit_nnunet_logs(run_paths=run_paths)

    assert calls == ["safe output\n", "safe output\n"]


def test_no_older_log_path_grammar_remains_in_nnunet_or_gate3() -> None:
    nnunet_source = Path(nnunet_module.__file__).read_text(encoding="utf-8")
    gate3_source = Path(gate3_module.__file__).read_text(encoding="utf-8")

    assert "_POSIX_ABSOLUTE_PATH_RE" not in nnunet_source
    assert "_POSIX_ABSOLUTE_PATH_RE" not in gate3_source
    assert "_WINDOWS_ABSOLUTE_PATH_RE" not in gate3_source
    assert "_contains_absolute_filesystem_path" in nnunet_source
    assert "_contains_absolute_filesystem_path" in gate3_source
