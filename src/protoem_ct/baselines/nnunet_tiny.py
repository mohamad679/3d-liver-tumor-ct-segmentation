"""Deterministic bounded nnU-Net v2 tiny execution for Phase 3."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import nibabel as nib
import numpy as np

from protoem_ct.artifacts.hashing import sha256_file, sha256_json
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
from protoem_ct.baselines.nnunet import (
    NnUNetV2RunConfig,
    build_nnunet_v2_plan_and_preprocess_command,
    build_nnunet_v2_runtime_environment,
    execute_nnunet_command,
)
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

NNUNET_TINY_STRATEGY_IDENTIFIER = "official_plan_preprocess_plus_trainer_network_tiny_cpu_v1"


class NnUNetTinyError(ValueError):
    """Base error for the bounded nnU-Net tiny execution path."""


class NnUNetTinyRuntimeError(NnUNetTinyError):
    """Raised when the bounded nnU-Net tiny execution path fails."""


@dataclass(frozen=True, slots=True)
class NnUNetTinyExecutionResult:
    """Published result for one deterministic bounded nnU-Net tiny run."""

    strategy_identifier: str
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


def run_nnunet_tiny_baseline(
    *,
    run_config: NnUNetV2RunConfig,
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
    training_steps: int = 12,
    timeout_seconds: float = 300.0,
) -> NnUNetTinyExecutionResult:
    """Run the bounded deterministic nnU-Net tiny path on the synthetic fixture."""

    fixture_artifact = baseline_synthetic_fixture_from_json(fixture_artifact_path.read_bytes())
    if fixture_artifact.artifact_hash != dataset_manifest_sha256:
        raise NnUNetTinyRuntimeError(
            "dataset_manifest_sha256 must equal the synthetic fixture artifact hash."
        )

    runtime = build_nnunet_v2_runtime_environment(raw_root=fixture_root, run_paths=run_paths)
    execute_nnunet_command(
        build_nnunet_v2_plan_and_preprocess_command(run_config),
        runtime_environment=runtime.environment,
        working_directory=run_paths.temporary_dir,
        logs_directory=run_paths.logs_dir,
        timeout_seconds=timeout_seconds,
        stage_name="nnunet_plan_and_preprocess",
        required_output_paths=(
            runtime.preprocessed_root
            / BASELINE_SYNTHETIC_DATASET_NAME
            / f"{run_config.plans_identifier}.json",
        ),
    )
    _write_single_case_split(
        runtime.preprocessed_root / BASELINE_SYNTHETIC_DATASET_NAME / "splits_final.json"
    )

    torch = _import_torch()
    _configure_torch_runtime(torch=torch)
    _configure_determinism(torch=torch, seed=run_config.seed)

    trainer = _build_trainer(
        run_config=run_config,
        runtime_environment=runtime.environment,
        torch=torch,
    )
    trainer.enable_deep_supervision = False
    trainer.initialize()

    dataset = trainer.dataset_class(trainer.preprocessed_dataset_folder, ["synthetic_train_002"])
    image_data, seg_data, _seg_prev, _properties = dataset.load_case("synthetic_train_002")
    input_tensor = torch.from_numpy(np.asarray(image_data, dtype=np.float32)[None, ...])
    target_tensor = torch.from_numpy(np.asarray(seg_data[0:1, ...], dtype=np.int64)[None, ...])
    with torch.no_grad():
        smoke_logits = trainer.network(input_tensor)
        if isinstance(smoke_logits, (list, tuple)):
            smoke_logits = smoke_logits[0]
        if not torch.isfinite(smoke_logits).all().item():
            raise NnUNetTinyRuntimeError("nnunet_nonfinite_shape_smoke_prediction")
    shape_smoke_passed = tuple(int(v) for v in smoke_logits.shape[2:]) == tuple(
        int(v) for v in input_tensor.shape[2:]
    )

    losses: list[float] = []
    best_loss = math.inf
    trainer.network.train()
    for _step in range(training_steps):
        trainer.optimizer.zero_grad(set_to_none=True)
        logits = trainer.network(input_tensor)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        loss = trainer.loss(logits, target_tensor)
        if not torch.isfinite(loss).item():
            raise NnUNetTinyRuntimeError("nnunet_nonfinite_loss")
        loss.backward()
        trainer.optimizer.step()
        loss_value = float(loss.detach().cpu().item())
        losses.append(loss_value)
        best_loss = min(best_loss, loss_value)

    initial_loss = losses[0]
    final_loss = losses[-1]
    overfit_passed = final_loss < initial_loss
    if not overfit_passed:
        raise NnUNetTinyRuntimeError("nnunet_loss_not_reduced")

    checkpoint_path = run_paths.checkpoints_dir / run_config.checkpoint_name
    if checkpoint_path.exists():
        raise NnUNetTinyRuntimeError("nnunet_checkpoint_collision")
    torch.save(
        {
            "checkpoint_format": "phase3_nnunet_tiny_checkpoint_v1",
            "config_sha256": run_config.artifact_hash,
            "fixture_sha256": fixture_artifact.artifact_hash,
            "seed": run_config.seed,
            "step_count": training_steps,
            "network_weights": trainer.network.state_dict(),
            "trainer_name": trainer.__class__.__name__,
            "init_args": {"configuration": run_config.configuration},
            "inference_allowed_mirroring_axes": None,
        },
        checkpoint_path,
    )
    checkpoint_sha256 = sha256_file(checkpoint_path)

    predictor = _build_predictor(trainer=trainer, torch=torch)
    prediction_dir = run_paths.predictions_dir / "saved_predictions"
    prediction_dir.mkdir(mode=0o700)
    dataset_root = fixture_root / BASELINE_SYNTHETIC_DATASET_NAME
    for case_identifier in BASELINE_SYNTHETIC_TEST_CASE_IDENTIFIERS:
        image_path = dataset_root / "imagesTs" / f"{case_identifier}_0000.nii.gz"
        image = cast(Any, nib.load(str(image_path)))
        image_array = np.asanyarray(image.dataobj).astype(np.float32, copy=False)[None, ...]
        seg = predictor.predict_single_npy_array(
            np.transpose(image_array, (0, 3, 2, 1)),
            {"spacing": tuple(float(v) for v in image.header.get_zooms()[:3])[::-1]},
            output_file_truncated=None,
            save_or_return_probabilities=False,
        )
        pred_img = cast(
            Any,
            nib.Nifti1Image(  # type: ignore[no-untyped-call]
                np.transpose(np.asarray(seg, dtype=np.uint8), (2, 1, 0)),
                image.affine,
                image.header.copy(),
            ),
        )
        pred_img.set_data_dtype(np.uint8)
        nib.save(pred_img, str(prediction_dir / f"{case_identifier}.nii.gz"))

    mapping = build_reference_label_mapping_from_synthetic_fixture(
        fixture_artifact_path,
        fixture_root=fixture_root,
        split="test",
    )
    prediction_manifest_path = run_paths.metrics_dir / "prediction_manifest.json"
    metric_report_path = run_paths.metrics_dir / "metric_report.json"
    prediction_import_result = import_saved_baseline_predictions(
        baseline_family="nnunet_v2",
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

    provenance_payload = {
        "amp_enabled": False,
        "baseline_family": "nnunet_v2",
        "checkpoint_sha256": checkpoint_sha256,
        "config_sha256": run_config.artifact_hash,
        "contract_version": "baseline_run_provenance_v1",
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "development_split_sha256": development_split_sha256,
        "end_timestamp": end_timestamp,
        "failure_codes": [],
        "git_commit": git_commit,
        "metrics_artifact_sha256": prediction_import_result.metric_report.artifact_hash,
        "package_versions": dict(sorted(package_versions.items())),
        "platform_machine": platform_machine,
        "platform_system": platform_system,
        "prediction_manifest_sha256": prediction_import_result.prediction_manifest.artifact_hash,
        "python_version": python_version,
        "run_identifier": run_identifier,
        "seed": run_config.seed,
        "selected_device": "cpu",
        "start_timestamp": start_timestamp,
        "status": "completed",
    }
    provenance = BaselineRunProvenance(
        artifact_hash=sha256_json(provenance_payload),
        baseline_family="nnunet_v2",
        run_identifier=run_identifier,
        git_commit=git_commit,
        config_sha256=run_config.artifact_hash,
        dataset_manifest_sha256=dataset_manifest_sha256,
        development_split_sha256=development_split_sha256,
        seed=run_config.seed,
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
        prediction_manifest_sha256=prediction_import_result.prediction_manifest.artifact_hash,
        metrics_artifact_sha256=prediction_import_result.metric_report.artifact_hash,
        failure_codes=(),
        contract_version="baseline_run_provenance_v1",
    )
    provenance_path = run_paths.metrics_dir / "nnunet_v2_provenance.json"
    _publish_completed_provenance(
        provenance=provenance,
        output_path=provenance_path,
        checkpoint_path=checkpoint_path,
        prediction_manifest_path=prediction_manifest_path,
        metric_report_path=metric_report_path,
    )
    mlflow_metadata_verified = log_phase3_metadata_only_mlflow_run(
        tracking_directory=run_paths.mlflow_dir,
        experiment_name="phase3_nnunet_tiny",
        run_name=run_identifier,
        parameters={
            "amp_enabled": False,
            "baseline_family": "nnunet_v2",
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
            "step_count": training_steps,
            "strategy_identifier": NNUNET_TINY_STRATEGY_IDENTIFIER,
        },
        metrics={
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "best_loss": best_loss,
        },
    )
    return NnUNetTinyExecutionResult(
        strategy_identifier=NNUNET_TINY_STRATEGY_IDENTIFIER,
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
        training_step_count=training_steps,
        deterministic_metric_json_verified=deterministic_metric_json_verified,
        mlflow_verification=mlflow_metadata_verified,
    )


def _configure_determinism(*, torch: Any, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
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
        raise NnUNetTinyRuntimeError("nnunet_provenance_not_completed")
    if persisted.artifact_hash != provenance.artifact_hash:
        raise NnUNetTinyRuntimeError("nnunet_provenance_hash_mismatch")
    if persisted.checkpoint_sha256 != sha256_file(checkpoint_path):
        raise NnUNetTinyRuntimeError("nnunet_provenance_link_mismatch")
    if (
        persisted.prediction_manifest_sha256
        != baseline_prediction_manifest_from_json(
            prediction_manifest_path.read_bytes()
        ).artifact_hash
    ):
        raise NnUNetTinyRuntimeError("nnunet_provenance_link_mismatch")
    if (
        persisted.metrics_artifact_sha256
        != baseline_metric_report_from_json(metric_report_path.read_bytes()).artifact_hash
    ):
        raise NnUNetTinyRuntimeError("nnunet_provenance_link_mismatch")


def _write_single_case_split(path: Path) -> None:
    path.write_text(
        json.dumps([{"train": ["synthetic_train_002"], "val": ["synthetic_train_002"]}]),
        encoding="utf-8",
    )


def _build_trainer(
    *,
    run_config: NnUNetV2RunConfig,
    runtime_environment: Mapping[str, str],
    torch: Any,
) -> Any:
    from nnunetv2.run.run_training import get_trainer_from_args  # type: ignore[import-not-found]

    for key in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
        if key in runtime_environment:
            import os

            os.environ[key] = runtime_environment[key]
    return get_trainer_from_args(
        str(run_config.dataset_id),
        run_config.configuration,
        run_config.fold,
        run_config.trainer_class,
        run_config.plans_identifier,
        continue_training=False,
        device=torch.device("cpu"),
    )


def _build_predictor(*, trainer: Any, torch: Any) -> Any:
    from nnunetv2.inference.predict_from_raw_data import (  # type: ignore[import-not-found]
        nnUNetPredictor,
    )

    trainer.network.eval()
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=False,
        device=torch.device("cpu"),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.manual_initialization(
        trainer.network,
        trainer.plans_manager,
        trainer.configuration_manager,
        [trainer.network.state_dict()],
        trainer.dataset_json,
        trainer.__class__.__name__,
        None,
    )
    return predictor


def _import_torch() -> Any:
    import torch  # type: ignore[import-not-found]

    return torch
