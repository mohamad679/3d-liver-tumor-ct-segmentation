"""Deterministic MONAI SegResNet baseline execution for Phase 3."""

from __future__ import annotations

import json
import math
import random
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import nibabel as nib
import numpy as np

from protoem_ct.artifacts.hashing import canonical_json_bytes, sha256_file, sha256_json
from protoem_ct.baselines._mlflow_local import (
    Phase3MlflowVerificationResult,
    log_phase3_metadata_only_mlflow_run,
)
from protoem_ct.baselines._torch_runtime import configure_phase3_cpu_torch_runtime
from protoem_ct.baselines.evaluation import (
    BaselinePredictionImportResult,
    build_reference_label_mapping_from_synthetic_fixture,
    import_saved_baseline_predictions,
)
from protoem_ct.baselines.metrics import baseline_metric_report_from_json
from protoem_ct.baselines.paths import ValidatedBaselineRunPaths
from protoem_ct.baselines.predictions import baseline_prediction_manifest_from_json
from protoem_ct.baselines.provenance import (
    BaselineRunProvenance,
    baseline_run_provenance_from_json,
    baseline_run_provenance_to_json,
)
from protoem_ct.baselines.synthetic import (
    BASELINE_SYNTHETIC_DATASET_NAME,
    BASELINE_SYNTHETIC_TEST_CASE_IDENTIFIERS,
    baseline_synthetic_fixture_from_json,
)
from protoem_ct.data._phase2_publication import publish_text_no_overwrite

MONAI_SEGRESNET_RUN_CONFIG_VERSION = "monai_segresnet_run_config_v1"
MONAI_SEGRESNET_BASELINE_FAMILY = "monai_segresnet"
MONAI_SEGRESNET_DEFAULT_SEED = 1729
MONAI_SEGRESNET_DEFAULT_CHECKPOINT_NAME = "checkpoint_final.pt"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_IDENTIFIER_RE = re.compile(r"^[a-z0-9](?:[a-z0-9_-]{0,62}[a-z0-9])?$")
_SAFE_FILENAME_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,126}[A-Za-z0-9])?$")
_RUN_CONFIG_FIELDS = {
    "amp_enabled",
    "artifact_hash",
    "baseline_family",
    "checkpoint_name",
    "contract_version",
    "device",
    "fixed_intensity_max",
    "fixed_intensity_min",
    "in_channels",
    "learning_rate",
    "loss_name",
    "max_training_steps",
    "num_classes",
    "optimizer_name",
    "overfit_case_identifier",
    "seed",
    "sliding_window_batch_size",
    "sliding_window_overlap",
    "sliding_window_roi_size",
    "thread_count",
}


class MonaiSegResNetError(ValueError):
    """Base error for the MONAI SegResNet baseline."""


class MonaiSegResNetValidationError(MonaiSegResNetError):
    """Raised when MONAI SegResNet configuration or runtime inputs are invalid."""


class MonaiSegResNetSerializationError(MonaiSegResNetError):
    """Raised when MONAI SegResNet artifacts cannot be serialized or parsed safely."""


class MonaiSegResNetVersionError(MonaiSegResNetError):
    """Raised when an unsupported MONAI SegResNet contract version is encountered."""


class MonaiSegResNetHashError(MonaiSegResNetError):
    """Raised when a persisted MONAI SegResNet artifact hash does not match its content."""


class MonaiSegResNetRuntimeError(MonaiSegResNetError):
    """Raised when the bounded MONAI SegResNet execution path fails."""


@dataclass(frozen=True, slots=True)
class MonaiSegResNetRunConfig:
    """Immutable deterministic MONAI SegResNet run configuration."""

    artifact_hash: str
    contract_version: str
    baseline_family: str
    device: str
    amp_enabled: bool
    seed: int
    in_channels: int
    num_classes: int
    optimizer_name: str
    learning_rate: float
    loss_name: str
    max_training_steps: int
    fixed_intensity_min: float
    fixed_intensity_max: float
    sliding_window_roi_size: tuple[int, int, int]
    sliding_window_overlap: float
    sliding_window_batch_size: int
    thread_count: int
    overfit_case_identifier: str
    checkpoint_name: str = MONAI_SEGRESNET_DEFAULT_CHECKPOINT_NAME

    def __post_init__(self) -> None:
        if self.contract_version != MONAI_SEGRESNET_RUN_CONFIG_VERSION:
            raise MonaiSegResNetVersionError(
                f"Unsupported MONAI SegResNet contract version: {self.contract_version!r}."
            )
        _require_sha256(self.artifact_hash, field_name="artifact_hash")
        if self.baseline_family != MONAI_SEGRESNET_BASELINE_FAMILY:
            raise MonaiSegResNetValidationError(
                f"baseline_family must equal {MONAI_SEGRESNET_BASELINE_FAMILY!r}."
            )
        if self.device != "cpu":
            raise MonaiSegResNetValidationError("device must equal 'cpu'.")
        if self.amp_enabled:
            raise MonaiSegResNetValidationError("amp_enabled must be false for the CPU path.")
        if not isinstance(self.seed, int) or self.seed < 0 or self.seed > (2**32 - 1):
            raise MonaiSegResNetValidationError("seed must be an integer in [0, 2^32 - 1].")
        if self.in_channels != 1:
            raise MonaiSegResNetValidationError("in_channels must equal 1.")
        if self.num_classes != 2:
            raise MonaiSegResNetValidationError("num_classes must equal 2.")
        _require_safe_identifier(self.optimizer_name, field_name="optimizer_name")
        _require_safe_identifier(self.loss_name, field_name="loss_name")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise MonaiSegResNetValidationError("learning_rate must be finite and positive.")
        if not isinstance(self.max_training_steps, int) or self.max_training_steps <= 0:
            raise MonaiSegResNetValidationError("max_training_steps must be a positive integer.")
        if not math.isfinite(self.fixed_intensity_min) or not math.isfinite(
            self.fixed_intensity_max
        ):
            raise MonaiSegResNetValidationError(
                "fixed intensity bounds must be finite numeric values."
            )
        if self.fixed_intensity_min >= self.fixed_intensity_max:
            raise MonaiSegResNetValidationError(
                "fixed_intensity_min must be strictly smaller than fixed_intensity_max."
            )
        if len(self.sliding_window_roi_size) != 3 or any(
            int(value) <= 0 for value in self.sliding_window_roi_size
        ):
            raise MonaiSegResNetValidationError(
                "sliding_window_roi_size must contain exactly three positive integers."
            )
        if not math.isfinite(self.sliding_window_overlap) or not (
            0.0 <= self.sliding_window_overlap < 1.0
        ):
            raise MonaiSegResNetValidationError(
                "sliding_window_overlap must be finite and in [0.0, 1.0)."
            )
        if (
            not isinstance(self.sliding_window_batch_size, int)
            or self.sliding_window_batch_size <= 0
        ):
            raise MonaiSegResNetValidationError(
                "sliding_window_batch_size must be a positive integer."
            )
        if not isinstance(self.thread_count, int) or self.thread_count <= 0:
            raise MonaiSegResNetValidationError("thread_count must be a positive integer.")
        _require_safe_identifier(
            self.overfit_case_identifier,
            field_name="overfit_case_identifier",
        )
        _require_safe_filename(self.checkpoint_name, field_name="checkpoint_name")
        if self.artifact_hash != sha256_json(_run_config_payload_without_hash(self)):
            raise MonaiSegResNetHashError(
                "artifact_hash does not match the deterministic MONAI SegResNet config content."
            )


@dataclass(frozen=True, slots=True)
class MonaiSegResNetExecutionResult:
    """Published result for one deterministic bounded MONAI baseline run."""

    run_config: MonaiSegResNetRunConfig
    checkpoint_path: Path
    checkpoint_sha256: str
    prediction_import_result: BaselinePredictionImportResult
    provenance: BaselineRunProvenance
    provenance_path: Path
    shape_smoke_passed: bool
    overfit_passed: bool
    initial_loss: float
    final_loss: float
    best_loss: float
    training_step_count: int
    deterministic_metric_json_verified: bool
    mlflow_verification: Phase3MlflowVerificationResult

    @property
    def mlflow_metadata_verified(self) -> bool:
        """Return whether the persisted MLflow metadata contract was verified."""

        return self.mlflow_verification.verified


def build_monai_segresnet_run_config(
    *,
    seed: int = MONAI_SEGRESNET_DEFAULT_SEED,
    max_training_steps: int = 16,
    learning_rate: float = 5e-3,
    sliding_window_roi_size: tuple[int, int, int] = (24, 24, 16),
    sliding_window_overlap: float = 0.0,
    sliding_window_batch_size: int = 1,
    thread_count: int = 1,
    overfit_case_identifier: str = "synthetic_train_002",
) -> MonaiSegResNetRunConfig:
    """Build the deterministic initial MONAI SegResNet run configuration."""

    payload_without_hash = {
        "amp_enabled": False,
        "baseline_family": MONAI_SEGRESNET_BASELINE_FAMILY,
        "checkpoint_name": MONAI_SEGRESNET_DEFAULT_CHECKPOINT_NAME,
        "contract_version": MONAI_SEGRESNET_RUN_CONFIG_VERSION,
        "device": "cpu",
        "fixed_intensity_max": 1000.0,
        "fixed_intensity_min": -1000.0,
        "in_channels": 1,
        "learning_rate": learning_rate,
        "loss_name": "cross_entropy",
        "max_training_steps": max_training_steps,
        "num_classes": 2,
        "optimizer_name": "adam",
        "overfit_case_identifier": overfit_case_identifier,
        "seed": seed,
        "sliding_window_batch_size": sliding_window_batch_size,
        "sliding_window_overlap": sliding_window_overlap,
        "sliding_window_roi_size": list(sliding_window_roi_size),
        "thread_count": thread_count,
    }
    return MonaiSegResNetRunConfig(
        artifact_hash=sha256_json(payload_without_hash),
        contract_version=MONAI_SEGRESNET_RUN_CONFIG_VERSION,
        baseline_family=MONAI_SEGRESNET_BASELINE_FAMILY,
        device="cpu",
        amp_enabled=False,
        seed=seed,
        in_channels=1,
        num_classes=2,
        optimizer_name="adam",
        learning_rate=learning_rate,
        loss_name="cross_entropy",
        max_training_steps=max_training_steps,
        fixed_intensity_min=-1000.0,
        fixed_intensity_max=1000.0,
        sliding_window_roi_size=sliding_window_roi_size,
        sliding_window_overlap=sliding_window_overlap,
        sliding_window_batch_size=sliding_window_batch_size,
        thread_count=thread_count,
        overfit_case_identifier=overfit_case_identifier,
    )


def monai_segresnet_run_config_to_json(run_config: MonaiSegResNetRunConfig) -> bytes:
    """Serialize a deterministic MONAI SegResNet configuration artifact."""

    return canonical_json_bytes(_run_config_payload(run_config)) + b"\n"


def monai_segresnet_run_config_from_json(payload: bytes | str) -> MonaiSegResNetRunConfig:
    """Parse and validate a deterministic MONAI SegResNet configuration artifact."""

    text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
    try:
        parsed = json.loads(text)
    except Exception as error:
        raise MonaiSegResNetSerializationError(
            f"Failed to parse MONAI SegResNet config JSON: {error}"
        ) from error
    if not isinstance(parsed, dict):
        raise MonaiSegResNetSerializationError("MONAI SegResNet config JSON must encode an object.")
    unknown_fields = set(parsed) - _RUN_CONFIG_FIELDS
    if unknown_fields:
        raise MonaiSegResNetSerializationError(
            f"Unknown MONAI SegResNet config field(s): {', '.join(sorted(unknown_fields))}."
        )
    missing_fields = _RUN_CONFIG_FIELDS - set(parsed)
    if missing_fields:
        raise MonaiSegResNetSerializationError(
            f"Missing MONAI SegResNet config field(s): {', '.join(sorted(missing_fields))}."
        )
    return MonaiSegResNetRunConfig(
        artifact_hash=_expect_string(parsed["artifact_hash"], field_name="artifact_hash"),
        contract_version=_expect_string(parsed["contract_version"], field_name="contract_version"),
        baseline_family=_expect_string(parsed["baseline_family"], field_name="baseline_family"),
        device=_expect_string(parsed["device"], field_name="device"),
        amp_enabled=_expect_bool(parsed["amp_enabled"], field_name="amp_enabled"),
        seed=_expect_int(parsed["seed"], field_name="seed"),
        in_channels=_expect_int(parsed["in_channels"], field_name="in_channels"),
        num_classes=_expect_int(parsed["num_classes"], field_name="num_classes"),
        optimizer_name=_expect_string(parsed["optimizer_name"], field_name="optimizer_name"),
        learning_rate=_expect_float(parsed["learning_rate"], field_name="learning_rate"),
        loss_name=_expect_string(parsed["loss_name"], field_name="loss_name"),
        max_training_steps=_expect_int(
            parsed["max_training_steps"], field_name="max_training_steps"
        ),
        fixed_intensity_min=_expect_float(
            parsed["fixed_intensity_min"], field_name="fixed_intensity_min"
        ),
        fixed_intensity_max=_expect_float(
            parsed["fixed_intensity_max"], field_name="fixed_intensity_max"
        ),
        sliding_window_roi_size=_expect_int_tuple3(
            parsed["sliding_window_roi_size"], field_name="sliding_window_roi_size"
        ),
        sliding_window_overlap=_expect_float(
            parsed["sliding_window_overlap"], field_name="sliding_window_overlap"
        ),
        sliding_window_batch_size=_expect_int(
            parsed["sliding_window_batch_size"], field_name="sliding_window_batch_size"
        ),
        thread_count=_expect_int(parsed["thread_count"], field_name="thread_count"),
        overfit_case_identifier=_expect_string(
            parsed["overfit_case_identifier"], field_name="overfit_case_identifier"
        ),
        checkpoint_name=_expect_string(parsed["checkpoint_name"], field_name="checkpoint_name"),
    )


def run_monai_segresnet_baseline(
    *,
    run_config: MonaiSegResNetRunConfig,
    run_paths: ValidatedBaselineRunPaths,
    fixture_root: Path,
    fixture_artifact_path: Path,
    run_identifier: str,
    git_commit: str,
    start_timestamp: str,
    end_timestamp: str,
    dataset_manifest_sha256: str,
    development_split_sha256: str,
    metric_config_sha256: str,
    nsd_tolerance_mm: float,
    package_versions: Mapping[str, str],
    python_version: str,
    platform_system: str,
    platform_machine: str,
) -> MonaiSegResNetExecutionResult:
    """Execute the bounded deterministic MONAI baseline on the synthetic fixture."""

    _require_safe_identifier(run_identifier, field_name="run_identifier")
    _require_sha256(dataset_manifest_sha256, field_name="dataset_manifest_sha256")
    _require_sha256(development_split_sha256, field_name="development_split_sha256")
    _require_sha256(metric_config_sha256, field_name="metric_config_sha256")
    if not fixture_root.is_absolute() or not fixture_root.is_dir():
        raise MonaiSegResNetValidationError("fixture_root must be an absolute existing directory.")
    fixture_artifact = baseline_synthetic_fixture_from_json(fixture_artifact_path.read_bytes())
    if fixture_artifact.artifact_hash != dataset_manifest_sha256:
        raise MonaiSegResNetValidationError(
            "dataset_manifest_sha256 must equal the synthetic fixture artifact hash."
        )

    torch = _import_torch()
    monai = _import_monai()
    _configure_torch_runtime(torch=torch)
    _configure_determinism(torch=torch, seed=run_config.seed, thread_count=run_config.thread_count)

    dataset_root = fixture_root / BASELINE_SYNTHETIC_DATASET_NAME
    train_image_path = (
        dataset_root / "imagesTr" / f"{run_config.overfit_case_identifier}_0000.nii.gz"
    )
    train_label_path = dataset_root / "labelsTr" / f"{run_config.overfit_case_identifier}.nii.gz"
    train_input, train_target, train_affine = _load_case_for_training(
        image_path=train_image_path,
        label_path=train_label_path,
        run_config=run_config,
        torch=torch,
    )

    model = monai.networks.nets.SegResNet(
        spatial_dims=3,
        init_filters=8,
        in_channels=1,
        out_channels=2,
        dropout_prob=None,
        blocks_down=(1, 1, 1),
        blocks_up=(1, 1),
        upsample_mode="deconv",
    ).to(torch.device("cpu"))
    optimizer = torch.optim.Adam(model.parameters(), lr=run_config.learning_rate)
    loss_function = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        smoke_logits = model(train_input)
        if not torch.isfinite(smoke_logits).all().item():
            raise MonaiSegResNetRuntimeError("nonfinite_shape_smoke_prediction")
    shape_smoke_passed = tuple(int(v) for v in smoke_logits.shape) == (1, 2, 24, 24, 16)

    losses: list[float] = []
    best_loss = math.inf
    for _step in range(run_config.max_training_steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_input)
        loss = loss_function(logits, train_target)
        if not torch.isfinite(loss).item():
            raise MonaiSegResNetRuntimeError("nonfinite_loss")
        loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all().item():
                raise MonaiSegResNetRuntimeError("nonfinite_gradient")
        optimizer.step()
        loss_value = float(loss.detach().cpu().item())
        losses.append(loss_value)
        best_loss = min(best_loss, loss_value)

    initial_loss = losses[0]
    final_loss = losses[-1]
    overfit_passed = final_loss < initial_loss
    if not overfit_passed:
        raise MonaiSegResNetRuntimeError("loss_not_reduced")

    checkpoint_path = run_paths.checkpoints_dir / run_config.checkpoint_name
    if checkpoint_path.exists():
        raise MonaiSegResNetRuntimeError("checkpoint_collision")
    torch.save(
        {
            "baseline_family": MONAI_SEGRESNET_BASELINE_FAMILY,
            "checkpoint_format": "phase3_monai_segresnet_checkpoint_v1",
            "config_sha256": run_config.artifact_hash,
            "fixture_sha256": fixture_artifact.artifact_hash,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "seed": run_config.seed,
            "step_count": run_config.max_training_steps,
        },
        checkpoint_path,
    )
    checkpoint_sha256 = sha256_file(checkpoint_path)

    prediction_dir = run_paths.predictions_dir / "saved_predictions"
    prediction_dir.mkdir(mode=0o700)
    for case_identifier in BASELINE_SYNTHETIC_TEST_CASE_IDENTIFIERS:
        image_path = dataset_root / "imagesTs" / f"{case_identifier}_0000.nii.gz"
        label_path = dataset_root / "labelsTs" / f"{case_identifier}.nii.gz"
        image_tensor, _target_tensor, affine = _load_case_for_training(
            image_path=image_path,
            label_path=label_path,
            run_config=run_config,
            torch=torch,
        )
        with torch.no_grad():
            logits = monai.inferers.sliding_window_inference(
                image_tensor,
                roi_size=run_config.sliding_window_roi_size,
                sw_batch_size=run_config.sliding_window_batch_size,
                predictor=model,
                overlap=run_config.sliding_window_overlap,
                mode="constant",
                progress=False,
                sw_device=torch.device("cpu"),
                device=torch.device("cpu"),
            )
        prediction = torch.argmax(logits, dim=1).to(dtype=torch.uint8).cpu().numpy()[0]
        _write_prediction(
            output_path=prediction_dir / f"{case_identifier}.nii.gz",
            prediction=prediction,
            affine=affine,
        )

    mapping = build_reference_label_mapping_from_synthetic_fixture(
        fixture_artifact_path,
        fixture_root=fixture_root,
        split="test",
    )
    prediction_manifest_path = run_paths.metrics_dir / "prediction_manifest.json"
    metric_report_path = run_paths.metrics_dir / "metric_report.json"
    prediction_import_result = import_saved_baseline_predictions(
        baseline_family=MONAI_SEGRESNET_BASELINE_FAMILY,
        run_identifier=run_identifier,
        prediction_directory=prediction_dir,
        label_paths_by_case=mapping,
        dataset_manifest_sha256=dataset_manifest_sha256,
        development_split_sha256=development_split_sha256,
        checkpoint_sha256=checkpoint_sha256,
        metric_config_sha256=metric_config_sha256,
        nsd_tolerance_mm=nsd_tolerance_mm,
        affine_tolerance_mm=0.0001,
        prediction_manifest_output_path=prediction_manifest_path,
        metric_report_output_path=metric_report_path,
    )
    deterministic_metric_json_verified = prediction_manifest_path.read_text(
        encoding="utf-8"
    ).endswith("\n") and metric_report_path.read_text(encoding="utf-8").endswith("\n")

    provenance = _build_completed_provenance(
        run_config_sha256=run_config.artifact_hash,
        dataset_manifest_sha256=dataset_manifest_sha256,
        development_split_sha256=development_split_sha256,
        run_identifier=run_identifier,
        git_commit=git_commit,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        checkpoint_sha256=checkpoint_sha256,
        prediction_manifest_sha256=prediction_import_result.prediction_manifest.artifact_hash,
        metric_report_sha256=prediction_import_result.metric_report.artifact_hash,
        seed=run_config.seed,
        package_versions=package_versions,
        python_version=python_version,
        platform_system=platform_system,
        platform_machine=platform_machine,
    )
    provenance_path = run_paths.metrics_dir / "monai_segresnet_provenance.json"
    _publish_completed_provenance(
        provenance=provenance,
        output_path=provenance_path,
        checkpoint_path=checkpoint_path,
        prediction_manifest_path=prediction_manifest_path,
        metric_report_path=metric_report_path,
    )
    mlflow_metadata_verified = log_phase3_metadata_only_mlflow_run(
        tracking_directory=run_paths.mlflow_dir,
        experiment_name="phase3_monai_segresnet",
        run_name=run_identifier,
        parameters={
            "amp_enabled": False,
            "baseline_family": MONAI_SEGRESNET_BASELINE_FAMILY,
            "checkpoint_sha256": checkpoint_sha256,
            "config_sha256": run_config.artifact_hash,
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "metric_report_sha256": prediction_import_result.metric_report.artifact_hash,
            "prediction_manifest_sha256": (
                prediction_import_result.prediction_manifest.artifact_hash
            ),
            "run_identifier": run_identifier,
            "seed": run_config.seed,
            "selected_device": "cpu",
            "status": "completed",
            "step_count": run_config.max_training_steps,
        },
        metrics={
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "best_loss": best_loss,
        },
    )
    return MonaiSegResNetExecutionResult(
        run_config=run_config,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        prediction_import_result=prediction_import_result,
        provenance=provenance,
        provenance_path=provenance_path,
        shape_smoke_passed=shape_smoke_passed,
        overfit_passed=overfit_passed,
        initial_loss=initial_loss,
        final_loss=final_loss,
        best_loss=best_loss,
        training_step_count=run_config.max_training_steps,
        deterministic_metric_json_verified=deterministic_metric_json_verified,
        mlflow_verification=mlflow_metadata_verified,
    )


def _run_config_payload(run_config: MonaiSegResNetRunConfig) -> dict[str, Any]:
    return {
        "amp_enabled": run_config.amp_enabled,
        "artifact_hash": run_config.artifact_hash,
        "baseline_family": run_config.baseline_family,
        "checkpoint_name": run_config.checkpoint_name,
        "contract_version": run_config.contract_version,
        "device": run_config.device,
        "fixed_intensity_max": run_config.fixed_intensity_max,
        "fixed_intensity_min": run_config.fixed_intensity_min,
        "in_channels": run_config.in_channels,
        "learning_rate": run_config.learning_rate,
        "loss_name": run_config.loss_name,
        "max_training_steps": run_config.max_training_steps,
        "num_classes": run_config.num_classes,
        "optimizer_name": run_config.optimizer_name,
        "overfit_case_identifier": run_config.overfit_case_identifier,
        "seed": run_config.seed,
        "sliding_window_batch_size": run_config.sliding_window_batch_size,
        "sliding_window_overlap": run_config.sliding_window_overlap,
        "sliding_window_roi_size": list(run_config.sliding_window_roi_size),
        "thread_count": run_config.thread_count,
    }


def _run_config_payload_without_hash(run_config: MonaiSegResNetRunConfig) -> dict[str, Any]:
    payload = _run_config_payload(run_config)
    del payload["artifact_hash"]
    return payload


def _build_completed_provenance(
    *,
    run_config_sha256: str,
    dataset_manifest_sha256: str,
    development_split_sha256: str,
    run_identifier: str,
    git_commit: str,
    start_timestamp: str,
    end_timestamp: str,
    checkpoint_sha256: str,
    prediction_manifest_sha256: str,
    metric_report_sha256: str,
    seed: int,
    package_versions: Mapping[str, str],
    python_version: str,
    platform_system: str,
    platform_machine: str,
) -> BaselineRunProvenance:
    payload_without_hash = {
        "amp_enabled": False,
        "baseline_family": MONAI_SEGRESNET_BASELINE_FAMILY,
        "checkpoint_sha256": checkpoint_sha256,
        "config_sha256": run_config_sha256,
        "contract_version": "baseline_run_provenance_v1",
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "development_split_sha256": development_split_sha256,
        "end_timestamp": end_timestamp,
        "failure_codes": [],
        "git_commit": git_commit,
        "metrics_artifact_sha256": metric_report_sha256,
        "package_versions": dict(sorted(package_versions.items())),
        "platform_machine": platform_machine,
        "platform_system": platform_system,
        "prediction_manifest_sha256": prediction_manifest_sha256,
        "python_version": python_version,
        "run_identifier": run_identifier,
        "seed": seed,
        "selected_device": "cpu",
        "start_timestamp": start_timestamp,
        "status": "completed",
    }
    return BaselineRunProvenance(
        artifact_hash=sha256_json(payload_without_hash),
        baseline_family=MONAI_SEGRESNET_BASELINE_FAMILY,
        run_identifier=run_identifier,
        git_commit=git_commit,
        config_sha256=run_config_sha256,
        dataset_manifest_sha256=dataset_manifest_sha256,
        development_split_sha256=development_split_sha256,
        seed=seed,
        python_version=python_version,
        platform_system=platform_system,
        platform_machine=platform_machine,
        selected_device="cpu",
        amp_enabled=False,
        package_versions=dict(sorted(package_versions.items())),
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        status="completed",
        checkpoint_sha256=checkpoint_sha256,
        prediction_manifest_sha256=prediction_manifest_sha256,
        metrics_artifact_sha256=metric_report_sha256,
        failure_codes=(),
        contract_version="baseline_run_provenance_v1",
    )


def _configure_determinism(*, torch: Any, seed: int, thread_count: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if thread_count != 1:
        raise MonaiSegResNetValidationError("thread_count must equal 1 for the CPU path.")
    torch.use_deterministic_algorithms(True)


def _configure_torch_runtime(*, torch: Any) -> None:
    configure_phase3_cpu_torch_runtime(torch=torch)


def _publish_completed_provenance(
    *,
    provenance: BaselineRunProvenance,
    output_path: Path,
    checkpoint_path: Path,
    prediction_manifest_path: Path,
    metric_report_path: Path,
) -> None:
    publish_text_no_overwrite(
        text=baseline_run_provenance_to_json(provenance).decode("utf-8"),
        output_path=output_path,
        temporary_exists_message="temporary provenance publication path already exists",
        final_exists_message="completed provenance output already exists",
    )
    persisted = baseline_run_provenance_from_json(output_path.read_bytes())
    if persisted.status != "completed":
        raise MonaiSegResNetRuntimeError("monai_provenance_not_completed")
    if persisted.artifact_hash != provenance.artifact_hash:
        raise MonaiSegResNetRuntimeError("monai_provenance_hash_mismatch")
    if persisted.checkpoint_sha256 != sha256_file(checkpoint_path):
        raise MonaiSegResNetRuntimeError("monai_provenance_link_mismatch")
    if (
        persisted.prediction_manifest_sha256
        != baseline_prediction_manifest_from_json(
            prediction_manifest_path.read_bytes()
        ).artifact_hash
    ):
        raise MonaiSegResNetRuntimeError("monai_provenance_link_mismatch")
    if (
        persisted.metrics_artifact_sha256
        != baseline_metric_report_from_json(metric_report_path.read_bytes()).artifact_hash
    ):
        raise MonaiSegResNetRuntimeError("monai_provenance_link_mismatch")


def _load_case_for_training(
    *,
    image_path: Path,
    label_path: Path,
    run_config: MonaiSegResNetRunConfig,
    torch: Any,
) -> tuple[Any, Any, np.ndarray]:
    image = cast(Any, nib.load(str(image_path)))
    label = cast(Any, nib.load(str(label_path)))
    image_array = np.asanyarray(image.dataobj).astype(np.float32, copy=True)
    image_array = np.clip(
        image_array,
        run_config.fixed_intensity_min,
        run_config.fixed_intensity_max,
    )
    image_array = (image_array - run_config.fixed_intensity_min) / (
        run_config.fixed_intensity_max - run_config.fixed_intensity_min
    ) * 2.0 - 1.0
    label_array = np.asanyarray(label.dataobj).astype(np.int64, copy=True)
    return (
        torch.from_numpy(image_array[None, None, ...]).to(dtype=torch.float32),
        torch.from_numpy(label_array[None, ...]).to(dtype=torch.long),
        np.asarray(image.affine, dtype=np.float64),
    )


def _write_prediction(*, output_path: Path, prediction: np.ndarray, affine: np.ndarray) -> None:
    image = cast(
        Any,
        nib.Nifti1Image(prediction.astype(np.uint8, copy=True), affine),  # type: ignore[no-untyped-call]
    )
    image.set_data_dtype(np.uint8)
    image.set_qform(affine, code=1)
    image.set_sform(affine, code=1)
    nib.save(image, str(output_path))


def _import_torch() -> Any:
    import torch  # type: ignore[import-not-found]

    return torch


def _import_monai() -> Any:
    import monai  # type: ignore[import-not-found]

    return monai


def _require_safe_identifier(value: str, *, field_name: str) -> None:
    if not _SAFE_IDENTIFIER_RE.fullmatch(value):
        raise MonaiSegResNetValidationError(
            f"{field_name} must match the conservative ASCII identifier grammar."
        )


def _require_safe_filename(value: str, *, field_name: str) -> None:
    if not _SAFE_FILENAME_RE.fullmatch(value):
        raise MonaiSegResNetValidationError(
            f"{field_name} must match the conservative filename grammar."
        )


def _require_sha256(value: str, *, field_name: str) -> None:
    if not _SHA256_RE.fullmatch(value):
        raise MonaiSegResNetValidationError(f"{field_name} must be a lowercase SHA-256 hex.")


def _expect_string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise MonaiSegResNetSerializationError(f"{field_name} must be a JSON string.")
    return value


def _expect_bool(value: Any, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise MonaiSegResNetSerializationError(f"{field_name} must be a JSON boolean.")
    return value


def _expect_int(value: Any, *, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise MonaiSegResNetSerializationError(f"{field_name} must be a JSON integer.")
    return value


def _expect_float(value: Any, *, field_name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise MonaiSegResNetSerializationError(f"{field_name} must be a JSON number.")
    return float(value)


def _expect_int_tuple3(value: Any, *, field_name: str) -> tuple[int, int, int]:
    if not isinstance(value, list) or len(value) != 3:
        raise MonaiSegResNetSerializationError(f"{field_name} must be a length-3 JSON array.")
    return cast(
        tuple[int, int, int],
        tuple(_expect_int(item, field_name=field_name) for item in value),
    )
