from __future__ import annotations

import importlib
import json
import sys
from typing import Any, cast

import pytest

from protoem_ct.artifacts.hashing import sha256_json
from protoem_ct.baselines.provenance import (
    BASELINE_RUN_PROVENANCE_VERSION,
    BaselineProvenanceHashError,
    BaselineProvenanceSerializationError,
    BaselineProvenanceValidationError,
    BaselineProvenanceVersionError,
    BaselineRunProvenance,
    baseline_run_provenance_from_json,
    baseline_run_provenance_to_json,
)


def _build_provenance(
    *,
    baseline_family: str = "nnunet_v2",
    status: str = "planned",
    end_timestamp: str | None = None,
    checkpoint_sha256: str | None = None,
    prediction_manifest_sha256: str | None = None,
    metrics_artifact_sha256: str | None = None,
    failure_codes: tuple[str, ...] = (),
    package_versions: dict[str, str] | None = None,
    overrides: dict[str, object] | None = None,
) -> BaselineRunProvenance:
    payload_without_hash: dict[str, Any] = {
        "amp_enabled": False,
        "baseline_family": baseline_family,
        "checkpoint_sha256": checkpoint_sha256,
        "config_sha256": "1" * 64,
        "contract_version": BASELINE_RUN_PROVENANCE_VERSION,
        "dataset_manifest_sha256": "2" * 64,
        "development_split_sha256": "3" * 64,
        "end_timestamp": end_timestamp,
        "failure_codes": list(failure_codes),
        "git_commit": "a" * 40,
        "metrics_artifact_sha256": metrics_artifact_sha256,
        "package_versions": package_versions or {"numpy": "1.26.4", "scipy": "1.17.1"},
        "platform_machine": "x86_64",
        "platform_system": "Darwin",
        "prediction_manifest_sha256": prediction_manifest_sha256,
        "python_version": "3.11.9",
        "run_identifier": "baseline_run_001",
        "seed": 17,
        "selected_device": "cpu",
        "start_timestamp": "2026-07-28T12:00:00Z",
        "status": status,
    }
    if overrides:
        payload_without_hash.update(overrides)
    return BaselineRunProvenance(
        artifact_hash=sha256_json(payload_without_hash),
        baseline_family=str(payload_without_hash["baseline_family"]),
        run_identifier=str(payload_without_hash["run_identifier"]),
        git_commit=str(payload_without_hash["git_commit"]),
        config_sha256=str(payload_without_hash["config_sha256"]),
        dataset_manifest_sha256=str(payload_without_hash["dataset_manifest_sha256"]),
        development_split_sha256=str(payload_without_hash["development_split_sha256"]),
        seed=cast(int, payload_without_hash["seed"]),
        python_version=str(payload_without_hash["python_version"]),
        platform_system=str(payload_without_hash["platform_system"]),
        platform_machine=str(payload_without_hash["platform_machine"]),
        selected_device=str(payload_without_hash["selected_device"]),
        amp_enabled=bool(payload_without_hash["amp_enabled"]),
        package_versions=cast(dict[str, str], payload_without_hash["package_versions"]),
        start_timestamp=str(payload_without_hash["start_timestamp"]),
        end_timestamp=cast(str | None, payload_without_hash["end_timestamp"]),
        status=str(payload_without_hash["status"]),
        checkpoint_sha256=cast(str | None, payload_without_hash["checkpoint_sha256"]),
        prediction_manifest_sha256=cast(
            str | None, payload_without_hash["prediction_manifest_sha256"]
        ),
        metrics_artifact_sha256=cast(str | None, payload_without_hash["metrics_artifact_sha256"]),
        failure_codes=tuple(cast(list[str], payload_without_hash["failure_codes"])),
        contract_version=str(payload_without_hash["contract_version"]),
    )


def test_valid_records_for_both_baseline_families() -> None:
    assert _build_provenance(baseline_family="nnunet_v2").baseline_family == "nnunet_v2"
    assert _build_provenance(baseline_family="monai_segresnet").baseline_family == "monai_segresnet"


def test_deterministic_byte_identical_serialization() -> None:
    provenance = _build_provenance()
    first = baseline_run_provenance_to_json(provenance)
    second = baseline_run_provenance_to_json(provenance)

    assert first == second
    assert first.endswith(b"\n")


def test_hash_recomputed_on_round_trip() -> None:
    provenance = _build_provenance()
    parsed = baseline_run_provenance_from_json(baseline_run_provenance_to_json(provenance))

    assert parsed.artifact_hash == provenance.artifact_hash


def test_tamper_detection() -> None:
    provenance = _build_provenance()
    payload = json.loads(baseline_run_provenance_to_json(provenance))
    payload["selected_device"] = "cpu:tampered"

    with pytest.raises(BaselineProvenanceHashError):
        baseline_run_provenance_from_json(json.dumps(payload).encode("utf-8"))


def test_invalid_git_commit_and_hashes_rejected() -> None:
    with pytest.raises(BaselineProvenanceValidationError):
        _build_provenance(overrides={"git_commit": "ABC"})
    with pytest.raises(BaselineProvenanceValidationError):
        _build_provenance(overrides={"config_sha256": "abc"})


def test_invalid_seed_rejected() -> None:
    with pytest.raises(BaselineProvenanceValidationError):
        _build_provenance(overrides={"seed": -1})


def test_invalid_timestamp_rejected() -> None:
    with pytest.raises(BaselineProvenanceValidationError):
        _build_provenance(overrides={"start_timestamp": "2026-07-28 12:00:00"})


def test_invalid_run_identifier_rejected() -> None:
    with pytest.raises(BaselineProvenanceValidationError):
        _build_provenance(overrides={"run_identifier": "../patient123"})


def test_invalid_package_mapping_rejected() -> None:
    with pytest.raises(BaselineProvenanceValidationError):
        _build_provenance(package_versions={"NumPy": "1.26.4"})


@pytest.mark.parametrize(
    ("field_name", "override_value"),
    [
        ("python_version", "/abs/path"),
        ("platform_system", "/abs/path"),
        ("platform_machine", "/abs/path"),
        ("selected_device", "/abs/path"),
        ("run_identifier", "/abs/path"),
    ],
)
def test_absolute_path_rejected_in_string_bearing_fields(
    field_name: str,
    override_value: str,
) -> None:
    with pytest.raises(BaselineProvenanceValidationError):
        _build_provenance(overrides={field_name: override_value})


def test_planned_running_completed_failed_lifecycle_constraints() -> None:
    _build_provenance(status="planned")
    _build_provenance(status="running")
    _build_provenance(
        status="completed",
        end_timestamp="2026-07-28T12:05:00Z",
        checkpoint_sha256="4" * 64,
        prediction_manifest_sha256="5" * 64,
        metrics_artifact_sha256="6" * 64,
    )
    _build_provenance(
        status="failed",
        end_timestamp="2026-07-28T12:05:00Z",
        failure_codes=("missing_files",),
    )


def test_completed_requires_final_hashes() -> None:
    with pytest.raises(BaselineProvenanceValidationError):
        _build_provenance(
            status="completed",
            end_timestamp="2026-07-28T12:05:00Z",
            checkpoint_sha256="4" * 64,
        )


def test_failed_requires_failure_codes() -> None:
    with pytest.raises(BaselineProvenanceValidationError):
        _build_provenance(
            status="failed",
            end_timestamp="2026-07-28T12:05:00Z",
        )


def test_unknown_field_rejected() -> None:
    provenance = _build_provenance()
    payload = json.loads(baseline_run_provenance_to_json(provenance))
    payload["unknown_field"] = "value"

    with pytest.raises(BaselineProvenanceSerializationError):
        baseline_run_provenance_from_json(json.dumps(payload).encode("utf-8"))


def test_unknown_version_rejected() -> None:
    provenance = _build_provenance()
    payload = json.loads(baseline_run_provenance_to_json(provenance))
    payload["contract_version"] = "baseline_run_provenance_v999"

    with pytest.raises(BaselineProvenanceVersionError):
        baseline_run_provenance_from_json(json.dumps(payload).encode("utf-8"))


def test_importing_baselines_package_does_not_import_heavy_dependencies() -> None:
    for name in (
        "torch",
        "monai",
        "nnunetv2",
        "torchvision",
        "SimpleITK",
        "protoem_ct.baselines",
    ):
        sys.modules.pop(name, None)

    importlib.import_module("protoem_ct.baselines")

    assert "torch" not in sys.modules
    assert "monai" not in sys.modules
    assert "nnunetv2" not in sys.modules
    assert "torchvision" not in sys.modules
    assert "SimpleITK" not in sys.modules
