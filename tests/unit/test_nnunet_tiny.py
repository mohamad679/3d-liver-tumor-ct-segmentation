from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from protoem_ct.artifacts.hashing import sha256_file, sha256_json
from protoem_ct.baselines.metrics import (
    baseline_metric_report_to_json,
    build_baseline_metric_report,
    compute_baseline_case_metrics,
)
from protoem_ct.baselines.nnunet_tiny import (
    NNUNET_TINY_STRATEGY_IDENTIFIER,
    _publish_completed_provenance,
)
from protoem_ct.baselines.predictions import (
    BASELINE_PREDICTION_RECORD_VERSION,
    BaselinePredictionRecord,
    baseline_prediction_manifest_to_json,
    build_baseline_prediction_manifest,
)
from protoem_ct.baselines.provenance import (
    BaselineRunProvenance,
    baseline_run_provenance_from_json,
    baseline_run_provenance_to_json,
)
from protoem_ct.data._phase2_publication import Phase2PublicationExistingOutputError


def test_nnunet_strategy_identifier_is_fixed() -> None:
    assert (
        NNUNET_TINY_STRATEGY_IDENTIFIER
        == "official_plan_preprocess_plus_trainer_network_tiny_cpu_v1"
    )


def test_importing_nnunet_tiny_module_does_not_import_heavy_modules(
    run_import_guard: Callable[[tuple[str, ...], tuple[str, ...]], None],
) -> None:
    run_import_guard(
        ("protoem_ct.baselines.nnunet_tiny",),
        ("torch", "nnunetv2", "mlflow"),
    )


def test_publish_completed_provenance_is_deterministic_and_no_overwrite(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoint_final.pth"
    checkpoint_path.write_bytes(b"checkpoint\n")
    checkpoint_sha256 = sha256_file(checkpoint_path)

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

    output_path = tmp_path / "nnunet_v2_provenance.json"
    _publish_completed_provenance(
        provenance=provenance,
        output_path=output_path,
        checkpoint_path=checkpoint_path,
        prediction_manifest_path=prediction_manifest_path,
        metric_report_path=metric_report_path,
    )

    assert output_path.name == "nnunet_v2_provenance.json"
    assert output_path.read_bytes() == baseline_run_provenance_to_json(provenance)
    assert (
        baseline_run_provenance_from_json(output_path.read_bytes()).artifact_hash
        == provenance.artifact_hash
    )

    with pytest.raises(Phase2PublicationExistingOutputError):
        _publish_completed_provenance(
            provenance=provenance,
            output_path=output_path,
            checkpoint_path=checkpoint_path,
            prediction_manifest_path=prediction_manifest_path,
            metric_report_path=metric_report_path,
        )
