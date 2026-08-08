"""Unit tests for the Phase 8 definitive training/validation orchestration.

Covers: the locked ``Phase8DefinitiveTrainingExecutionPolicy`` (Package A
"G"), continuous training with checkpoint snapshots ("H"), checkpoint
snapshot serialization/publication ("I"), full 20-case development
validation ("J"), checkpoint selection ("K"), the machine-readable
selection-evidence artifact ("L"), and the process-level training watchdog
("M"). These extend :mod:`tests.unit.test_phase8_definitive_pipeline`, kept
in a separate file to avoid further growing that already-large file.

All tests operate on small, synthetic, in-memory NumPy arrays only. No real
dataset, checkpoint, or ``/Volumes`` path is referenced anywhere in this
file. Orchestration-level tests (full-validation, selection, selection
evidence) monkeypatch ``run_definitive_validation`` with a fast, pure-Python
fake so they can exercise loader/uniqueness/mean/selection logic without
requiring the isolated torch/monai baseline environment; only the
training-step and checkpoint-serialization tests are gated behind
``requires_baseline_environment``.
"""

from __future__ import annotations

import contextlib
import dataclasses
import importlib.util
import inspect
import math
import multiprocessing
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from protoem_ct.external import definitive_pipeline
from protoem_ct.external.definitive_pipeline import (
    Phase8DefinitiveCheckpointPublicationError,
    Phase8DefinitiveCheckpointSelectionEvidence,
    Phase8DefinitiveFullValidationResult,
    Phase8DefinitivePipelineConfigError,
    Phase8DefinitivePipelineSelectionError,
    Phase8DefinitivePipelineValidationError,
    Phase8DefinitiveTrainCase,
    Phase8DefinitiveTrainCaseReference,
    Phase8DefinitiveTrainingExecutionPolicy,
    Phase8DefinitiveTrainingWatchdogTimeoutError,
    Phase8DefinitiveValidationCase,
    Phase8DefinitiveValidationCaseReference,
    Phase8DefinitiveValidationRunResult,
    build_definitive_checkpoint_selection_evidence,
    build_definitive_config_v1,
    build_definitive_training_execution_policy_v1,
    publish_definitive_checkpoint_snapshot,
    run_definitive_continuous_training_with_snapshots,
    run_definitive_continuous_training_with_snapshots_and_watchdog,
    run_definitive_full_validation,
    run_definitive_training_from_references,
    select_definitive_checkpoint,
)
from protoem_ct.external.internal_evidence import ArtifactReference

requires_baseline_environment = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None or importlib.util.find_spec("monai") is None,
    reason="Phase 8 definitive pipeline torch/monai tests require the isolated baseline "
    "environment.",
)

_FAKE_SHA256 = "a" * 64
_FAKE_GIT_COMMIT = "b" * 40


def _package_environment_reference() -> ArtifactReference:
    return ArtifactReference(
        schema_name="phase3_cpu_environment",
        schema_version="v1",
        artifact_hash=_FAKE_SHA256,
        artifact_role="package_environment",
    )


def _make_synthetic_train_case(
    case_id: str, *, positive: bool, shape: tuple[int, int, int] = (160, 160, 80)
) -> Phase8DefinitiveTrainCase:
    rng = np.random.default_rng(abs(hash(case_id)) % (2**32))
    image = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
    label = np.zeros(shape, dtype=bool)
    if positive:
        label[140:144, 140:144, 70:74] = True
    return Phase8DefinitiveTrainCase(case_id=case_id, image=image, label_binary=label)


class _LazyCaseLoader:
    """A call-counting, call-logging loader that builds cases on demand only."""

    def __init__(self, *, positive_ids: set[str]) -> None:
        self._positive_ids = positive_ids
        self.calls: list[str] = []
        self.construction_count = 0

    def __call__(self, case_id: str) -> Phase8DefinitiveTrainCase:
        self.calls.append(case_id)
        self.construction_count += 1
        return _make_synthetic_train_case(case_id, positive=case_id in self._positive_ids)


# ---------------------------------------------------------------------------
# G. Locked training execution policy
# ---------------------------------------------------------------------------


def test_build_definitive_training_execution_policy_v1_is_valid_and_self_verifies() -> None:
    policy = build_definitive_training_execution_policy_v1()
    assert policy.total_optimizer_steps == 500
    assert policy.candidate_checkpoint_steps == (250, 500)
    assert policy.early_stopping is False
    assert policy.resume_policy == "none"
    assert policy.maximum_wall_clock_seconds == 36000.0
    assert policy.validation_expected_case_count == 20
    assert policy.full_validation_required_for_every_candidate is True
    assert policy.subset_validation_for_checkpoint_selection is False
    assert policy.checkpoint_selection_metric == "mean_tumor_dice"
    assert policy.tie_break_policy == "earliest_checkpoint_step"
    assert policy.internal_test_used_for_selection is False
    assert policy.external_data_used_for_selection is False
    assert policy.external_labels_used_for_selection is False
    assert len(policy.policy_hash) == 64


@pytest.mark.parametrize(
    "field_name,bad_value",
    [
        ("total_optimizer_steps", 501),
        ("total_optimizer_steps", 250),
        ("candidate_checkpoint_steps", (250, 375, 500)),
        ("candidate_checkpoint_steps", (500,)),
        ("candidate_checkpoint_steps", (100, 500)),
        ("early_stopping", True),
        ("resume_policy", "latest"),
        ("maximum_wall_clock_seconds", 3600.0),
        ("maximum_wall_clock_seconds", 72000.0),
        ("validation_expected_case_count", 19),
        ("validation_expected_case_count", 21),
        ("subset_validation_for_checkpoint_selection", True),
        ("internal_test_used_for_selection", True),
        ("external_data_used_for_selection", True),
        ("external_labels_used_for_selection", True),
    ],
)
def test_tampered_training_policy_field_fails_closed(field_name: str, bad_value: object) -> None:
    policy = build_definitive_training_execution_policy_v1()
    payload = {f.name: getattr(policy, f.name) for f in dataclasses.fields(policy)}
    payload[field_name] = bad_value
    with pytest.raises(Phase8DefinitivePipelineConfigError):
        Phase8DefinitiveTrainingExecutionPolicy(**payload)


def test_tampered_training_policy_hash_fails_closed() -> None:
    policy = build_definitive_training_execution_policy_v1()
    payload = {f.name: getattr(policy, f.name) for f in dataclasses.fields(policy)}
    payload["policy_hash"] = "0" * 64
    with pytest.raises(Phase8DefinitivePipelineConfigError):
        Phase8DefinitiveTrainingExecutionPolicy(**payload)


# ---------------------------------------------------------------------------
# H. Continuous training with checkpoint snapshots
# ---------------------------------------------------------------------------


@requires_baseline_environment
def test_continuous_training_snapshots_at_exactly_policy_steps() -> None:
    config = build_definitive_config_v1()
    policy = build_definitive_training_execution_policy_v1()

    case_references = [
        Phase8DefinitiveTrainCaseReference(case_id="case_pos"),
        Phase8DefinitiveTrainCaseReference(case_id="case_neg"),
    ]
    loader = _LazyCaseLoader(positive_ids={"case_pos"})

    result = run_definitive_continuous_training_with_snapshots(
        case_references,
        positive_case_ids={"case_pos"},
        loader=loader,
        config=config,
        policy=policy,
    )

    assert result.all_steps_finite is True
    assert len(result.step_results) == 500
    assert set(result.checkpoint_snapshots.keys()) == {250, 500}
    for snapshot in result.checkpoint_snapshots.values():
        assert snapshot is not None


@requires_baseline_environment
def test_continuous_training_rejects_tampered_policy_steps() -> None:
    config = build_definitive_config_v1()
    policy = build_definitive_training_execution_policy_v1()
    tampered_policy = object.__new__(Phase8DefinitiveTrainingExecutionPolicy)
    for field in dataclasses.fields(policy):
        object.__setattr__(tampered_policy, field.name, getattr(policy, field.name))
    object.__setattr__(tampered_policy, "total_optimizer_steps", 3)

    case_references = [Phase8DefinitiveTrainCaseReference(case_id="only_case")]
    loader = _LazyCaseLoader(positive_ids={"only_case"})

    with pytest.raises(Phase8DefinitivePipelineConfigError):
        run_definitive_continuous_training_with_snapshots(
            case_references,
            positive_case_ids={"only_case"},
            loader=loader,
            config=config,
            policy=tampered_policy,
        )


@requires_baseline_environment
def test_continuous_training_core_seeds_exactly_once(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch  # type: ignore[import-not-found]

    config = build_definitive_config_v1()
    case_references = [
        Phase8DefinitiveTrainCaseReference(case_id="case_pos"),
        Phase8DefinitiveTrainCaseReference(case_id="case_neg"),
    ]
    loader = _LazyCaseLoader(positive_ids={"case_pos"})

    seed_calls: list[int] = []
    real_manual_seed = torch.manual_seed

    def _spy_manual_seed(seed: int) -> Any:
        seed_calls.append(seed)
        return real_manual_seed(seed)

    monkeypatch.setattr(torch, "manual_seed", _spy_manual_seed)

    result = definitive_pipeline._run_definitive_continuous_training_with_snapshots_core(
        case_references,
        positive_case_ids={"case_pos"},
        loader=loader,
        config=config,
        total_optimizer_steps=6,
        candidate_checkpoint_steps=(3, 6),
    )

    assert result.all_steps_finite is True
    # Exactly one torch.manual_seed call across the entire run: the model and
    # optimizer are seeded once and never reinitialized at a candidate step.
    assert len(seed_calls) == 1


@requires_baseline_environment
def test_continuous_training_sampling_matches_non_snapshot_path_through_early_steps() -> None:
    """The RNG stream is not reset partway through a continuous run.

    Compares the first few steps of a continuous-with-snapshots run against
    an independent call to the existing, already-tested
    :func:`run_definitive_training_from_references` with the same
    seed/config/case pool: both use the identical deterministic
    positive/negative selection and sampling scheme, so if the continuous
    run's early steps matched a fresh/reset RNG stream instead of a truly
    continuous one, the loss values would diverge.
    """

    config = build_definitive_config_v1()
    case_references = [
        Phase8DefinitiveTrainCaseReference(case_id="case_pos"),
        Phase8DefinitiveTrainCaseReference(case_id="case_neg"),
    ]

    loader_continuous = _LazyCaseLoader(positive_ids={"case_pos"})
    continuous_result = definitive_pipeline._run_definitive_continuous_training_with_snapshots_core(
        case_references,
        positive_case_ids={"case_pos"},
        loader=loader_continuous,
        config=config,
        total_optimizer_steps=5,
        candidate_checkpoint_steps=(2, 5),
    )

    loader_reference = _LazyCaseLoader(positive_ids={"case_pos"})
    reference_result = run_definitive_training_from_references(
        case_references,
        positive_case_ids={"case_pos"},
        loader=loader_reference,
        config=config,
        max_steps=5,
    )

    assert continuous_result.all_steps_finite is True
    assert reference_result.all_steps_finite is True
    continuous_losses = [s.loss_value for s in continuous_result.step_results]
    reference_losses = [s.loss_value for s in reference_result.step_results]
    assert continuous_losses == reference_losses
    assert loader_continuous.calls == loader_reference.calls


@requires_baseline_environment
def test_continuous_training_snapshot_deep_copy_is_independent() -> None:
    import torch

    config = build_definitive_config_v1()
    case_references = [
        Phase8DefinitiveTrainCaseReference(case_id="case_pos"),
        Phase8DefinitiveTrainCaseReference(case_id="case_neg"),
    ]
    loader = _LazyCaseLoader(positive_ids={"case_pos"})

    result = definitive_pipeline._run_definitive_continuous_training_with_snapshots_core(
        case_references,
        positive_case_ids={"case_pos"},
        loader=loader,
        config=config,
        total_optimizer_steps=6,
        candidate_checkpoint_steps=(3, 6),
    )

    assert set(result.checkpoint_snapshots.keys()) == {3, 6}
    snapshot_3 = result.checkpoint_snapshots[3]
    snapshot_6 = result.checkpoint_snapshots[6]

    # Training continued for 3 more steps between the two snapshots, so at
    # least one parameter tensor must differ -- if the "snapshot" had been a
    # bare reference into the live model rather than an independent deep
    # copy, both entries would be bit-for-bit identical (the final weights).
    any_differs = False
    for key in snapshot_3:
        if not torch.equal(snapshot_3[key], snapshot_6[key]):
            any_differs = True
            break
    assert any_differs is True

    # Snapshots do not alias each other's storage.
    for key in snapshot_3:
        assert snapshot_3[key].data_ptr() != snapshot_6[key].data_ptr()


# ---------------------------------------------------------------------------
# I. Checkpoint snapshot serialization and publication
# ---------------------------------------------------------------------------


@requires_baseline_environment
def test_publish_definitive_checkpoint_snapshot_rejects_overwrite(tmp_path: Path) -> None:
    import torch

    config = build_definitive_config_v1()
    state_dict = {"weight": torch.zeros(4, 4)}

    output_root = tmp_path / "checkpoints"
    output_root.mkdir()

    first = publish_definitive_checkpoint_snapshot(
        state_dict=state_dict,
        optimizer_step=250,
        output_root=output_root,
        config=config,
        development_manifest_hash=_FAKE_SHA256,
        development_split_hash=_FAKE_SHA256,
        originating_git_commit=_FAKE_GIT_COMMIT,
        package_environment_reference=_package_environment_reference(),
        preprocessing_evidence_hash=_FAKE_SHA256,
    )
    assert first.output_path.exists()
    assert len(first.checkpoint_sha256) == 64
    assert first.checkpoint_metadata.freeze_eligible is False

    with pytest.raises(Phase8DefinitiveCheckpointPublicationError):
        publish_definitive_checkpoint_snapshot(
            state_dict=state_dict,
            optimizer_step=250,
            output_root=output_root,
            config=config,
            development_manifest_hash=_FAKE_SHA256,
            development_split_hash=_FAKE_SHA256,
            originating_git_commit=_FAKE_GIT_COMMIT,
            package_environment_reference=_package_environment_reference(),
            preprocessing_evidence_hash=_FAKE_SHA256,
        )


@requires_baseline_environment
def test_publish_definitive_checkpoint_snapshot_rejects_unknown_step(tmp_path: Path) -> None:
    import torch

    config = build_definitive_config_v1()
    state_dict = {"weight": torch.zeros(2, 2)}
    output_root = tmp_path / "checkpoints"
    output_root.mkdir()

    with pytest.raises(Phase8DefinitiveCheckpointPublicationError):
        publish_definitive_checkpoint_snapshot(
            state_dict=state_dict,
            optimizer_step=300,
            output_root=output_root,
            config=config,
            development_manifest_hash=_FAKE_SHA256,
            development_split_hash=_FAKE_SHA256,
            originating_git_commit=_FAKE_GIT_COMMIT,
            package_environment_reference=_package_environment_reference(),
            preprocessing_evidence_hash=_FAKE_SHA256,
        )


# ---------------------------------------------------------------------------
# J. Full development-validation orchestration -- fake validator fixtures
# ---------------------------------------------------------------------------


def _validation_case_ids(count: int = 20) -> list[str]:
    return [f"vcase_{i:03d}" for i in range(count)]


class _CountingValidationLoader:
    """A call-counting loader that never retains a case past one call."""

    def __init__(self, case_ids: list[str]) -> None:
        self._known = set(case_ids)
        self.calls: list[str] = []

    def __call__(self, case_id: str) -> Phase8DefinitiveValidationCase:
        assert case_id in self._known
        self.calls.append(case_id)
        return Phase8DefinitiveValidationCase(
            case_id=case_id,
            image=np.zeros((4, 4, 4), dtype=np.float32),
            label_binary=np.zeros((4, 4, 4), dtype=bool),
        )


def _install_fake_validator(
    monkeypatch: pytest.MonkeyPatch,
    *,
    dice_by_case: dict[str, float],
    iou_by_case: dict[str, float] | None = None,
    finite_by_case: dict[str, bool] | None = None,
) -> None:
    def _fake_run_definitive_validation(
        model: Any, case: Phase8DefinitiveValidationCase, *, config: Any
    ) -> Phase8DefinitiveValidationRunResult:
        finite = True if finite_by_case is None else finite_by_case.get(case.case_id, True)
        iou = 0.0 if iou_by_case is None else iou_by_case.get(case.case_id, 0.0)
        dice = dice_by_case[case.case_id]
        if not finite:
            return Phase8DefinitiveValidationRunResult(
                case_id=case.case_id,
                tumor_dice=float("nan"),
                tumor_iou=float("nan"),
                prediction_finite=False,
            )
        return Phase8DefinitiveValidationRunResult(
            case_id=case.case_id, tumor_dice=dice, tumor_iou=iou, prediction_finite=True
        )

    monkeypatch.setattr(
        definitive_pipeline, "run_definitive_validation", _fake_run_definitive_validation
    )


def _full_validation_result(
    *,
    candidate_step: int,
    case_ids: list[str],
    dice_values: list[float],
) -> Phase8DefinitiveFullValidationResult:
    case_results = tuple(
        Phase8DefinitiveValidationRunResult(
            case_id=case_id, tumor_dice=dice, tumor_iou=0.0, prediction_finite=True
        )
        for case_id, dice in zip(case_ids, dice_values, strict=True)
    )
    mean_dice = sum(dice_values) / len(dice_values)
    return Phase8DefinitiveFullValidationResult(
        candidate_step=candidate_step,
        case_results=case_results,
        case_ids=tuple(sorted(case_ids)),
        mean_tumor_dice=mean_dice,
    )


# ---------------------------------------------------------------------------
# J. Full development-validation orchestration -- tests
# ---------------------------------------------------------------------------


def test_run_definitive_full_validation_succeeds_with_exactly_twenty_unique_cases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = build_definitive_config_v1()
    policy = build_definitive_training_execution_policy_v1()
    case_ids = _validation_case_ids(20)
    dice_by_case = {case_id: 0.5 for case_id in case_ids}
    _install_fake_validator(monkeypatch, dice_by_case=dice_by_case)

    loader = _CountingValidationLoader(case_ids)
    case_references = [Phase8DefinitiveValidationCaseReference(case_id=c) for c in case_ids]

    result = run_definitive_full_validation(
        case_references,
        loader=loader,
        model=object(),
        config=config,
        policy=policy,
        candidate_step=250,
    )

    assert len(result.case_results) == 20
    assert result.mean_tumor_dice == pytest.approx(0.5)
    # One at a time: every case is loaded exactly once, in order, and no
    # persistent cache reuses a case across the run.
    assert loader.calls == case_ids


def test_run_definitive_full_validation_loader_called_one_at_a_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = build_definitive_config_v1()
    policy = build_definitive_training_execution_policy_v1()
    case_ids = _validation_case_ids(20)
    dice_by_case = {case_id: 0.5 for case_id in case_ids}
    _install_fake_validator(monkeypatch, dice_by_case=dice_by_case)

    live_case_count = {"max": 0, "current": 0}
    loader = _CountingValidationLoader(case_ids)
    real_loader_call = loader.__call__

    def _tracking_loader(case_id: str) -> Phase8DefinitiveValidationCase:
        live_case_count["current"] += 1
        live_case_count["max"] = max(live_case_count["max"], live_case_count["current"])
        case = real_loader_call(case_id)
        live_case_count["current"] -= 1
        return case

    case_references = [Phase8DefinitiveValidationCaseReference(case_id=c) for c in case_ids]

    run_definitive_full_validation(
        case_references,
        loader=_tracking_loader,
        model=object(),
        config=config,
        policy=policy,
        candidate_step=250,
    )

    # At most one case's arrays are "live" (inside a loader call) at once.
    assert live_case_count["max"] == 1
    assert len(loader.calls) == 20


@pytest.mark.parametrize("case_count", [19, 21])
def test_run_definitive_full_validation_rejects_wrong_case_count(
    monkeypatch: pytest.MonkeyPatch, case_count: int
) -> None:
    config = build_definitive_config_v1()
    policy = build_definitive_training_execution_policy_v1()
    case_ids = _validation_case_ids(case_count)
    _install_fake_validator(monkeypatch, dice_by_case={case_id: 0.5 for case_id in case_ids})
    loader = _CountingValidationLoader(case_ids)
    case_references = [Phase8DefinitiveValidationCaseReference(case_id=c) for c in case_ids]

    with pytest.raises(Phase8DefinitivePipelineValidationError):
        run_definitive_full_validation(
            case_references,
            loader=loader,
            model=object(),
            config=config,
            policy=policy,
            candidate_step=250,
        )


def test_run_definitive_full_validation_rejects_duplicate_case_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = build_definitive_config_v1()
    policy = build_definitive_training_execution_policy_v1()
    unique_ids = _validation_case_ids(19)
    case_ids = [*unique_ids, unique_ids[0]]  # 20 total, only 19 unique.
    assert len(case_ids) == 20
    _install_fake_validator(monkeypatch, dice_by_case={case_id: 0.5 for case_id in unique_ids})
    loader = _CountingValidationLoader(unique_ids)
    case_references = [Phase8DefinitiveValidationCaseReference(case_id=c) for c in case_ids]

    with pytest.raises(Phase8DefinitivePipelineValidationError):
        run_definitive_full_validation(
            case_references,
            loader=loader,
            model=object(),
            config=config,
            policy=policy,
            candidate_step=250,
        )


def test_run_definitive_full_validation_computes_exact_mean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = build_definitive_config_v1()
    policy = build_definitive_training_execution_policy_v1()
    case_ids = _validation_case_ids(20)
    dice_values = [0.1 * (i % 10) for i in range(20)]
    dice_by_case = dict(zip(case_ids, dice_values, strict=True))
    _install_fake_validator(monkeypatch, dice_by_case=dice_by_case)
    loader = _CountingValidationLoader(case_ids)
    case_references = [Phase8DefinitiveValidationCaseReference(case_id=c) for c in case_ids]

    result = run_definitive_full_validation(
        case_references,
        loader=loader,
        model=object(),
        config=config,
        policy=policy,
        candidate_step=500,
    )

    assert result.mean_tumor_dice == pytest.approx(sum(dice_values) / len(dice_values))


def test_run_definitive_full_validation_fails_closed_on_non_finite_prediction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = build_definitive_config_v1()
    policy = build_definitive_training_execution_policy_v1()
    case_ids = _validation_case_ids(20)
    dice_by_case = {case_id: 0.5 for case_id in case_ids}
    finite_by_case = {case_id: True for case_id in case_ids}
    finite_by_case[case_ids[3]] = False
    _install_fake_validator(monkeypatch, dice_by_case=dice_by_case, finite_by_case=finite_by_case)
    loader = _CountingValidationLoader(case_ids)
    case_references = [Phase8DefinitiveValidationCaseReference(case_id=c) for c in case_ids]

    with pytest.raises(Phase8DefinitivePipelineValidationError):
        run_definitive_full_validation(
            case_references,
            loader=loader,
            model=object(),
            config=config,
            policy=policy,
            candidate_step=250,
        )


# ---------------------------------------------------------------------------
# K. Checkpoint selection
# ---------------------------------------------------------------------------


def test_select_definitive_checkpoint_higher_mean_wins() -> None:
    config = build_definitive_config_v1()
    policy = build_definitive_training_execution_policy_v1()
    case_ids = _validation_case_ids(20)

    result_250 = _full_validation_result(
        candidate_step=250, case_ids=case_ids, dice_values=[0.4] * 20
    )
    result_500 = _full_validation_result(
        candidate_step=500, case_ids=case_ids, dice_values=[0.6] * 20
    )

    selection = select_definitive_checkpoint(
        result_step_250=result_250, result_step_500=result_500, policy=policy, config=config
    )

    assert selection.mean_tumor_dice_step_250 == pytest.approx(0.4)
    assert selection.mean_tumor_dice_step_500 == pytest.approx(0.6)
    assert selection.selected_step == 500


def test_select_definitive_checkpoint_exact_tie_selects_step_250() -> None:
    config = build_definitive_config_v1()
    policy = build_definitive_training_execution_policy_v1()
    case_ids = _validation_case_ids(20)

    result_250 = _full_validation_result(
        candidate_step=250, case_ids=case_ids, dice_values=[0.5] * 20
    )
    result_500 = _full_validation_result(
        candidate_step=500, case_ids=case_ids, dice_values=[0.5] * 20
    )
    assert result_250.mean_tumor_dice == result_500.mean_tumor_dice

    selection = select_definitive_checkpoint(
        result_step_250=result_250, result_step_500=result_500, policy=policy, config=config
    )
    assert selection.selected_step == 250


def test_select_definitive_checkpoint_ignores_iou_favoring_the_other_candidate() -> None:
    config = build_definitive_config_v1()
    policy = build_definitive_training_execution_policy_v1()
    case_ids = _validation_case_ids(20)

    # Dice favors step 250; IoU (never read by selection) favors step 500.
    case_results_250 = tuple(
        Phase8DefinitiveValidationRunResult(
            case_id=case_id, tumor_dice=0.9, tumor_iou=0.1, prediction_finite=True
        )
        for case_id in case_ids
    )
    case_results_500 = tuple(
        Phase8DefinitiveValidationRunResult(
            case_id=case_id, tumor_dice=0.2, tumor_iou=0.99, prediction_finite=True
        )
        for case_id in case_ids
    )
    result_250 = Phase8DefinitiveFullValidationResult(
        candidate_step=250,
        case_results=case_results_250,
        case_ids=tuple(sorted(case_ids)),
        mean_tumor_dice=0.9,
    )
    result_500 = Phase8DefinitiveFullValidationResult(
        candidate_step=500,
        case_results=case_results_500,
        case_ids=tuple(sorted(case_ids)),
        mean_tumor_dice=0.2,
    )

    selection = select_definitive_checkpoint(
        result_step_250=result_250, result_step_500=result_500, policy=policy, config=config
    )
    assert selection.selected_step == 250


def test_select_definitive_checkpoint_signature_has_no_loss_parameter() -> None:
    signature = inspect.signature(select_definitive_checkpoint)
    assert "loss" not in signature.parameters
    assert set(signature.parameters) == {"result_step_250", "result_step_500", "policy", "config"}


def test_select_definitive_checkpoint_rejects_missing_candidate() -> None:
    config = build_definitive_config_v1()
    policy = build_definitive_training_execution_policy_v1()
    case_ids = _validation_case_ids(20)
    result_250 = _full_validation_result(
        candidate_step=250, case_ids=case_ids, dice_values=[0.5] * 20
    )

    with pytest.raises(Phase8DefinitivePipelineSelectionError):
        select_definitive_checkpoint(
            result_step_250=result_250, result_step_500=None, policy=policy, config=config
        )
    with pytest.raises(Phase8DefinitivePipelineSelectionError):
        select_definitive_checkpoint(
            result_step_250=None, result_step_500=result_250, policy=policy, config=config
        )


def test_select_definitive_checkpoint_rejects_differing_case_sets() -> None:
    config = build_definitive_config_v1()
    policy = build_definitive_training_execution_policy_v1()
    case_ids_a = _validation_case_ids(20)
    case_ids_b = [f"other_{i:03d}" for i in range(20)]

    result_250 = _full_validation_result(
        candidate_step=250, case_ids=case_ids_a, dice_values=[0.5] * 20
    )
    result_500 = _full_validation_result(
        candidate_step=500, case_ids=case_ids_b, dice_values=[0.6] * 20
    )

    with pytest.raises(Phase8DefinitivePipelineSelectionError):
        select_definitive_checkpoint(
            result_step_250=result_250, result_step_500=result_500, policy=policy, config=config
        )


def test_select_definitive_checkpoint_rejects_non_finite_dice() -> None:
    config = build_definitive_config_v1()
    policy = build_definitive_training_execution_policy_v1()
    case_ids = _validation_case_ids(20)

    dice_values = [0.5] * 20
    case_results = tuple(
        Phase8DefinitiveValidationRunResult(
            case_id=case_id,
            tumor_dice=math.nan if index == 0 else dice,
            tumor_iou=0.0,
            prediction_finite=True,
        )
        for index, (case_id, dice) in enumerate(zip(case_ids, dice_values, strict=True))
    )
    result_250 = Phase8DefinitiveFullValidationResult(
        candidate_step=250,
        case_results=case_results,
        case_ids=tuple(sorted(case_ids)),
        mean_tumor_dice=0.5,
    )
    result_500 = _full_validation_result(
        candidate_step=500, case_ids=case_ids, dice_values=[0.6] * 20
    )

    with pytest.raises(Phase8DefinitivePipelineSelectionError):
        select_definitive_checkpoint(
            result_step_250=result_250, result_step_500=result_500, policy=policy, config=config
        )


def test_select_definitive_checkpoint_rejects_wrong_candidate_case_count() -> None:
    config = build_definitive_config_v1()
    policy = build_definitive_training_execution_policy_v1()
    case_ids_250 = _validation_case_ids(19)
    case_ids_500 = _validation_case_ids(20)

    result_250 = _full_validation_result(
        candidate_step=250, case_ids=case_ids_250, dice_values=[0.5] * 19
    )
    result_500 = _full_validation_result(
        candidate_step=500, case_ids=case_ids_500, dice_values=[0.6] * 20
    )

    with pytest.raises(Phase8DefinitivePipelineSelectionError):
        select_definitive_checkpoint(
            result_step_250=result_250, result_step_500=result_500, policy=policy, config=config
        )


# ---------------------------------------------------------------------------
# L. Selection-evidence artifact
# ---------------------------------------------------------------------------


def test_build_definitive_checkpoint_selection_evidence_records_hashes_and_flags() -> None:
    config = build_definitive_config_v1()
    policy = build_definitive_training_execution_policy_v1()
    case_ids = _validation_case_ids(20)

    result_250 = _full_validation_result(
        candidate_step=250, case_ids=case_ids, dice_values=[0.4] * 20
    )
    result_500 = _full_validation_result(
        candidate_step=500, case_ids=case_ids, dice_values=[0.6] * 20
    )
    selection = select_definitive_checkpoint(
        result_step_250=result_250, result_step_500=result_500, policy=policy, config=config
    )

    checkpoint_hash_250 = "c" * 64
    checkpoint_hash_500 = "d" * 64
    evidence = build_definitive_checkpoint_selection_evidence(
        selection=selection,
        checkpoint_hash_step_250=checkpoint_hash_250,
        checkpoint_hash_step_500=checkpoint_hash_500,
    )

    assert isinstance(evidence, Phase8DefinitiveCheckpointSelectionEvidence)
    assert evidence.candidate_checkpoint_hash_step_250 == checkpoint_hash_250
    assert evidence.candidate_checkpoint_hash_step_500 == checkpoint_hash_500
    assert evidence.selected_checkpoint_step == 500
    assert evidence.selected_checkpoint_hash == checkpoint_hash_500
    assert evidence.selection_metric == "mean_tumor_dice"
    assert evidence.tie_break_policy == "earliest_checkpoint_step"
    assert evidence.internal_test_used is False
    assert evidence.external_data_used is False
    assert evidence.external_labels_used is False
    assert evidence.selection_completed is True
    assert len(evidence.evidence_hash) == 64


def test_selection_evidence_tampered_flag_fails_closed() -> None:
    config = build_definitive_config_v1()
    policy = build_definitive_training_execution_policy_v1()
    case_ids = _validation_case_ids(20)
    result_250 = _full_validation_result(
        candidate_step=250, case_ids=case_ids, dice_values=[0.4] * 20
    )
    result_500 = _full_validation_result(
        candidate_step=500, case_ids=case_ids, dice_values=[0.6] * 20
    )
    selection = select_definitive_checkpoint(
        result_step_250=result_250, result_step_500=result_500, policy=policy, config=config
    )
    evidence = build_definitive_checkpoint_selection_evidence(
        selection=selection,
        checkpoint_hash_step_250="c" * 64,
        checkpoint_hash_step_500="d" * 64,
    )
    payload = {f.name: getattr(evidence, f.name) for f in dataclasses.fields(evidence)}

    for field_name, bad_value in (
        ("internal_test_used", True),
        ("external_data_used", True),
        ("external_labels_used", True),
    ):
        tampered = dict(payload)
        tampered[field_name] = bad_value
        with pytest.raises(Phase8DefinitivePipelineValidationError):
            Phase8DefinitiveCheckpointSelectionEvidence(**tampered)


# ---------------------------------------------------------------------------
# M. Process-level training watchdog
# ---------------------------------------------------------------------------
#
# Mirrors the pattern in test_phase8_definitive_training_pilot.py's
# supervised-pilot watchdog tests: fake, non-medical ``_target`` callables and
# ``multiprocessing.get_context("fork")`` so closures can be used directly.


def test_training_watchdog_kills_blocking_child_and_fails_closed() -> None:
    config = build_definitive_config_v1()
    policy = build_definitive_training_execution_policy_v1()
    ctx = multiprocessing.get_context("fork")
    call_count = ctx.Value("i", 0)

    def _blocking_target(**_kwargs: object) -> Any:
        with call_count.get_lock():
            call_count.value += 1
        time.sleep(2.0)  # far longer than the tiny test deadline below.
        raise AssertionError("unreachable: the child must be killed before returning")

    case_references = [Phase8DefinitiveTrainCaseReference(case_id="only_case")]
    loader = _LazyCaseLoader(positive_ids={"only_case"})

    with pytest.raises(Phase8DefinitiveTrainingWatchdogTimeoutError) as exc_info:
        run_definitive_continuous_training_with_snapshots_and_watchdog(
            case_references,
            positive_case_ids={"only_case"},
            loader=loader,
            config=config,
            policy=policy,
            wall_clock_limit_seconds=0.2,
            _target=_blocking_target,
            _multiprocessing_context=ctx,
            _termination_grace_seconds=1.0,
        )

    assert isinstance(exc_info.value, Phase8DefinitiveTrainingWatchdogTimeoutError)
    assert multiprocessing.active_children() == []
    # No automatic retry: the fake target was invoked exactly once.
    assert call_count.value == 1


def test_training_watchdog_timeout_never_surfaces_a_result() -> None:
    config = build_definitive_config_v1()
    policy = build_definitive_training_execution_policy_v1()
    ctx = multiprocessing.get_context("fork")

    def _blocking_target(**_kwargs: object) -> Any:
        time.sleep(2.0)
        raise AssertionError("unreachable")

    case_references = [Phase8DefinitiveTrainCaseReference(case_id="only_case")]
    loader = _LazyCaseLoader(positive_ids={"only_case"})

    outcome: dict[str, object] = {}
    with contextlib.suppress(Phase8DefinitiveTrainingWatchdogTimeoutError):
        outcome["result"] = run_definitive_continuous_training_with_snapshots_and_watchdog(
            case_references,
            positive_case_ids={"only_case"},
            loader=loader,
            config=config,
            policy=policy,
            wall_clock_limit_seconds=0.2,
            _target=_blocking_target,
            _multiprocessing_context=ctx,
            _termination_grace_seconds=1.0,
        )

    # No selection-evidence/result was ever produced by the timed-out call.
    assert "result" not in outcome
    assert multiprocessing.active_children() == []


# ---------------------------------------------------------------------------
# Static scan: no external-dataset / real-filesystem / real-checkpoint literals
# ---------------------------------------------------------------------------


def test_source_contains_no_external_dataset_or_filesystem_literals() -> None:
    source_path = Path(definitive_pipeline.__file__)
    source_text = source_path.read_text(encoding="utf-8").lower()
    forbidden_literals = (
        "/volumes/",
        "task03_liver",
        "task03liver",
        "3d-ircadb",
        "ircadb",
        "patient_dicom",
        "liver_",
    )
    for literal in forbidden_literals:
        assert literal not in source_text, f"forbidden external-dataset literal found: {literal!r}"
