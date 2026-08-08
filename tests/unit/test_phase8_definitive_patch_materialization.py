"""Unit tests for the Phase 8 definitive patch-materialization design.

Covers ``P8-DEFINITIVE-PATCH-MATERIALIZATION-V1``: the deterministic 500-step
patch-request schedule (Section N), sequential one-case-at-a-time patch
materialization (Section O), and training from pre-materialized patches
(Section P). Kept in a separate file from
:mod:`tests.unit.test_phase8_definitive_pipeline` and
:mod:`tests.unit.test_phase8_definitive_training_orchestration` to avoid
further growing those already-large files.

All tests operate on small, synthetic, in-memory NumPy arrays and
``tmp_path`` only. No real dataset, checkpoint, or ``/Volumes`` path is
referenced anywhere in this file, and no real definitive training is
executed.
"""

from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from protoem_ct.external import definitive_pipeline
from protoem_ct.external.definitive_pipeline import (
    Phase8DefinitivePatchMaterializationError,
    Phase8DefinitivePatchSchedule,
    Phase8DefinitivePatchScheduleError,
    Phase8DefinitiveTrainCase,
    Phase8DefinitiveTrainCaseReference,
    Phase8DefinitiveTrainingStepResult,
    build_definitive_config_v1,
    build_definitive_patch_request_schedule,
    build_definitive_training_execution_policy_v1,
    materialize_definitive_training_patches,
)

requires_baseline_environment = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None or importlib.util.find_spec("monai") is None,
    reason="Phase 8 definitive pipeline torch/monai tests require the isolated baseline "
    "environment.",
)

_SHAPE = (160, 160, 80)


def _make_synthetic_train_case(case_id: str, *, positive: bool) -> Phase8DefinitiveTrainCase:
    rng = np.random.default_rng(abs(hash(case_id)) % (2**32))
    image = rng.uniform(-1.0, 1.0, size=_SHAPE).astype(np.float32)
    label = np.zeros(_SHAPE, dtype=bool)
    if positive:
        label[140:144, 140:144, 70:74] = True
    return Phase8DefinitiveTrainCase(case_id=case_id, image=image, label_binary=label)


class _LazyCaseLoader:
    """A call-counting, call-logging loader that builds cases on demand only."""

    def __init__(self, *, positive_ids: set[str]) -> None:
        self._positive_ids = positive_ids
        self.calls: list[str] = []

    def __call__(self, case_id: str) -> Phase8DefinitiveTrainCase:
        self.calls.append(case_id)
        return _make_synthetic_train_case(case_id, positive=case_id in self._positive_ids)


class _SequentialGuardLoader:
    """Fails if a new case is loaded before the previous case's patches are all on disk.

    ``requirement_counts`` maps ``case_id`` to the number of ``.npz`` patch
    files that case must contribute. Before each loader call, this asserts
    that exactly the sum of requirement counts for every *already-loaded*
    case is present on disk -- i.e. the previously loaded case was fully
    materialized and released before the next case was ever fetched, so at
    most one full case is live at once.
    """

    def __init__(
        self, *, positive_ids: set[str], patch_root: Path, requirement_counts: dict[str, int]
    ) -> None:
        self._positive_ids = positive_ids
        self._patch_root = patch_root
        self._requirement_counts = requirement_counts
        self.calls: list[str] = []
        self._expected_files_before_call = 0

    def __call__(self, case_id: str) -> Phase8DefinitiveTrainCase:
        existing_files = len(list(self._patch_root.glob("*.npz")))
        assert existing_files == self._expected_files_before_call, (
            "a new case was loaded before the previous case's patches were fully "
            "published; more than one full case would be live at once."
        )
        self.calls.append(case_id)
        self._expected_files_before_call += self._requirement_counts[case_id]
        return _make_synthetic_train_case(case_id, positive=case_id in self._positive_ids)


def _requirement_counts(schedule: Phase8DefinitivePatchSchedule) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in schedule.entries:
        counts[entry.positive_case_id] = counts.get(entry.positive_case_id, 0) + 1
        counts[entry.negative_case_id] = counts.get(entry.negative_case_id, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# N. Deterministic patch-request schedule
# ---------------------------------------------------------------------------


def test_schedule_has_exactly_500_entries_with_one_positive_one_negative_each() -> None:
    config = build_definitive_config_v1()
    policy = build_definitive_training_execution_policy_v1()
    case_references = [Phase8DefinitiveTrainCaseReference(case_id=f"case_{i}") for i in range(4)]

    schedule = build_definitive_patch_request_schedule(
        case_references, positive_case_ids={"case_0"}, config=config, policy=policy
    )

    assert schedule.total_optimizer_steps == 500
    assert len(schedule.entries) == 500
    known_ids = {reference.case_id for reference in case_references}
    for index, entry in enumerate(schedule.entries):
        assert entry.step_index == index
        assert entry.positive_case_id == "case_0"
        assert entry.negative_case_id in known_ids
    assert len(schedule.schedule_hash) == 64


def test_schedule_generation_is_deterministic() -> None:
    config = build_definitive_config_v1()
    policy = build_definitive_training_execution_policy_v1()
    case_references = [Phase8DefinitiveTrainCaseReference(case_id=f"case_{i}") for i in range(5)]

    schedule_a = build_definitive_patch_request_schedule(
        case_references, positive_case_ids={"case_0", "case_2"}, config=config, policy=policy
    )
    schedule_b = build_definitive_patch_request_schedule(
        case_references, positive_case_ids={"case_0", "case_2"}, config=config, policy=policy
    )

    assert schedule_a.entries == schedule_b.entries
    assert schedule_a.schedule_hash == schedule_b.schedule_hash


def test_schedule_hash_is_deterministic_and_tamper_fails_closed() -> None:
    config = build_definitive_config_v1()
    policy = build_definitive_training_execution_policy_v1()
    case_references = [Phase8DefinitiveTrainCaseReference(case_id=f"case_{i}") for i in range(3)]
    schedule = build_definitive_patch_request_schedule(
        case_references, positive_case_ids={"case_0"}, config=config, policy=policy
    )

    payload = {f.name: getattr(schedule, f.name) for f in dataclasses.fields(schedule)}
    payload["schedule_hash"] = "0" * 64
    # schedule_hash is verified by the same shared _require_self_hash helper
    # used by the locked Config/Policy dataclasses, so a tampered hash raises
    # the same shared Phase8DefinitivePipelineConfigError, not a schedule-
    # specific error subclass.
    with pytest.raises(definitive_pipeline.Phase8DefinitivePipelineConfigError):
        Phase8DefinitivePatchSchedule(**payload)


def test_schedule_depends_on_case_reference_order_with_no_internal_reordering() -> None:
    config = build_definitive_config_v1()
    policy = build_definitive_training_execution_policy_v1()
    refs_a = [Phase8DefinitiveTrainCaseReference(case_id=c) for c in ("case_0", "case_1", "case_2")]
    refs_b = [Phase8DefinitiveTrainCaseReference(case_id=c) for c in ("case_2", "case_1", "case_0")]

    schedule_a = build_definitive_patch_request_schedule(
        refs_a, positive_case_ids={"case_0"}, config=config, policy=policy
    )
    schedule_b = build_definitive_patch_request_schedule(
        refs_b, positive_case_ids={"case_0"}, config=config, policy=policy
    )

    # Documented deterministic rule: case_references order is preserved
    # exactly as supplied, with no internal sorting/renumbering, so a
    # different caller-supplied order deterministically produces a different
    # (but still fully deterministic) schedule.
    assert schedule_a.case_reference_ids == ("case_0", "case_1", "case_2")
    assert schedule_b.case_reference_ids == ("case_2", "case_1", "case_0")
    assert schedule_a.schedule_hash != schedule_b.schedule_hash


def test_schedule_positive_rotation_matches_committed_formula() -> None:
    config = build_definitive_config_v1()
    policy = build_definitive_training_execution_policy_v1()
    case_references = [
        Phase8DefinitiveTrainCaseReference(case_id=c)
        for c in ("case_0", "case_1", "case_2", "case_3")
    ]
    positive_ids = {"case_1", "case_3"}

    schedule = build_definitive_patch_request_schedule(
        case_references, positive_case_ids=positive_ids, config=config, policy=policy
    )

    positive_refs = [
        reference.case_id for reference in case_references if reference.case_id in positive_ids
    ]
    expected = [positive_refs[i % len(positive_refs)] for i in range(500)]
    actual = [entry.positive_case_id for entry in schedule.entries]
    assert actual == expected


def test_schedule_negative_selection_matches_dedicated_rng_stream() -> None:
    config = build_definitive_config_v1()
    policy = build_definitive_training_execution_policy_v1()
    case_references = [
        Phase8DefinitiveTrainCaseReference(case_id=c)
        for c in ("case_0", "case_1", "case_2", "case_3", "case_4")
    ]

    schedule = build_definitive_patch_request_schedule(
        case_references, positive_case_ids={"case_0"}, config=config, policy=policy
    )

    reference_rng = np.random.default_rng(config.seed)
    expected_indices = [int(reference_rng.integers(0, len(case_references))) for _ in range(500)]
    expected = [case_references[index].case_id for index in expected_indices]
    actual = [entry.negative_case_id for entry in schedule.entries]
    assert actual == expected


def test_schedule_rejects_unknown_positive_case_id() -> None:
    config = build_definitive_config_v1()
    policy = build_definitive_training_execution_policy_v1()
    case_references = [Phase8DefinitiveTrainCaseReference(case_id="case_a")]
    with pytest.raises(Phase8DefinitivePatchScheduleError):
        build_definitive_patch_request_schedule(
            case_references, positive_case_ids={"case_missing"}, config=config, policy=policy
        )


# ---------------------------------------------------------------------------
# O. Sequential patch materialization
# ---------------------------------------------------------------------------


def _small_schedule(
    case_references: list[Phase8DefinitiveTrainCaseReference],
    *,
    positive_case_ids: set[str],
    config: Any,
    policy: Any,
    total_steps: int = 6,
) -> Phase8DefinitivePatchSchedule:
    return definitive_pipeline._build_definitive_patch_request_schedule_core(
        case_references,
        positive_case_ids=positive_case_ids,
        config=config,
        policy_hash=policy.policy_hash,
        seed=config.seed,
        total_optimizer_steps=total_steps,
    )


def test_materialization_loads_one_case_at_a_time_and_only_once(tmp_path: Path) -> None:
    config = build_definitive_config_v1()
    policy = build_definitive_training_execution_policy_v1()
    case_references = [
        Phase8DefinitiveTrainCaseReference(case_id=c)
        for c in ("case_pos", "case_neg_a", "case_neg_b")
    ]
    schedule = _small_schedule(
        case_references, positive_case_ids={"case_pos"}, config=config, policy=policy, total_steps=8
    )
    patch_root = tmp_path / "patches"
    patch_root.mkdir()

    loader = _SequentialGuardLoader(
        positive_ids={"case_pos"},
        patch_root=patch_root,
        requirement_counts=_requirement_counts(schedule),
    )

    result = materialize_definitive_training_patches(
        schedule,
        case_references=case_references,
        positive_case_ids={"case_pos"},
        loader=loader,
        patch_root=patch_root,
        config=config,
    )

    # Each required case was fetched exactly once (no repeated preprocessing).
    assert len(loader.calls) == len(set(loader.calls))
    assert set(loader.calls) <= {"case_pos", "case_neg_a", "case_neg_b"}
    assert result.processed_case_ids == tuple(loader.calls)


def test_materialization_produces_all_required_patches_with_correct_shape_and_labels(
    tmp_path: Path,
) -> None:
    config = build_definitive_config_v1()
    policy = build_definitive_training_execution_policy_v1()
    case_references = [
        Phase8DefinitiveTrainCaseReference(case_id=c) for c in ("case_pos", "case_neg")
    ]
    schedule = _small_schedule(
        case_references, positive_case_ids={"case_pos"}, config=config, policy=policy, total_steps=5
    )
    patch_root = tmp_path / "patches"
    patch_root.mkdir()
    loader = _LazyCaseLoader(positive_ids={"case_pos"})

    result = materialize_definitive_training_patches(
        schedule,
        case_references=case_references,
        positive_case_ids={"case_pos"},
        loader=loader,
        patch_root=patch_root,
        config=config,
    )

    assert len(result.published_patches) == 2 * schedule.total_optimizer_steps
    seen = {(record.step_index, record.role) for record in result.published_patches}
    for step_index in range(schedule.total_optimizer_steps):
        assert (step_index, "positive") in seen
        assert (step_index, "negative") in seen

    for record in result.published_patches:
        assert record.npz_path.exists()
        assert record.metadata_path.exists()
        assert record.metadata.image_shape == config.patch_size
        assert record.metadata.label_shape == config.patch_size

        with np.load(record.npz_path) as archive:
            image = np.asarray(archive["image"])
            label = np.asarray(archive["label"])
        assert image.shape == config.patch_size
        assert label.shape == config.patch_size
        if record.role == "positive":
            assert bool(np.any(label)) is True
        else:
            assert bool(np.any(label)) is False


def test_materialization_no_persistent_cache_across_separate_runs(tmp_path: Path) -> None:
    config = build_definitive_config_v1()
    policy = build_definitive_training_execution_policy_v1()
    case_references = [
        Phase8DefinitiveTrainCaseReference(case_id=c) for c in ("case_pos", "case_neg")
    ]
    schedule = _small_schedule(
        case_references, positive_case_ids={"case_pos"}, config=config, policy=policy, total_steps=4
    )

    root_a = tmp_path / "run_a"
    root_a.mkdir()
    root_b = tmp_path / "run_b"
    root_b.mkdir()

    loader_a = _LazyCaseLoader(positive_ids={"case_pos"})
    loader_b = _LazyCaseLoader(positive_ids={"case_pos"})

    materialize_definitive_training_patches(
        schedule,
        case_references=case_references,
        positive_case_ids={"case_pos"},
        loader=loader_a,
        patch_root=root_a,
        config=config,
    )
    materialize_definitive_training_patches(
        schedule,
        case_references=case_references,
        positive_case_ids={"case_pos"},
        loader=loader_b,
        patch_root=root_b,
        config=config,
    )

    # A fresh call with a fresh loader re-fetches every required case; no
    # module-level or cross-call cache lets the second run skip loading.
    assert loader_a.calls == loader_b.calls
    assert len(loader_b.calls) == len(set(loader_b.calls))


def test_materialization_rejects_positive_case_ids_mismatch(tmp_path: Path) -> None:
    config = build_definitive_config_v1()
    policy = build_definitive_training_execution_policy_v1()
    case_references = [
        Phase8DefinitiveTrainCaseReference(case_id=c) for c in ("case_pos", "case_neg")
    ]
    schedule = _small_schedule(
        case_references, positive_case_ids={"case_pos"}, config=config, policy=policy, total_steps=3
    )
    patch_root = tmp_path / "patches"
    patch_root.mkdir()

    with pytest.raises(Phase8DefinitivePatchMaterializationError):
        materialize_definitive_training_patches(
            schedule,
            case_references=case_references,
            positive_case_ids={"case_neg"},
            loader=_LazyCaseLoader(positive_ids={"case_pos"}),
            patch_root=patch_root,
            config=config,
        )


def test_materialization_rejects_overwrite(tmp_path: Path) -> None:
    config = build_definitive_config_v1()
    policy = build_definitive_training_execution_policy_v1()
    case_references = [
        Phase8DefinitiveTrainCaseReference(case_id=c) for c in ("case_pos", "case_neg")
    ]
    schedule = _small_schedule(
        case_references, positive_case_ids={"case_pos"}, config=config, policy=policy, total_steps=3
    )
    patch_root = tmp_path / "patches"
    patch_root.mkdir()

    materialize_definitive_training_patches(
        schedule,
        case_references=case_references,
        positive_case_ids={"case_pos"},
        loader=_LazyCaseLoader(positive_ids={"case_pos"}),
        patch_root=patch_root,
        config=config,
    )

    with pytest.raises(Phase8DefinitivePatchMaterializationError):
        materialize_definitive_training_patches(
            schedule,
            case_references=case_references,
            positive_case_ids={"case_pos"},
            loader=_LazyCaseLoader(positive_ids={"case_pos"}),
            patch_root=patch_root,
            config=config,
        )


def test_materialization_identities_and_hashes_are_deterministic(tmp_path: Path) -> None:
    config = build_definitive_config_v1()
    policy = build_definitive_training_execution_policy_v1()
    case_references = [
        Phase8DefinitiveTrainCaseReference(case_id=c) for c in ("case_pos", "case_neg")
    ]
    schedule = _small_schedule(
        case_references, positive_case_ids={"case_pos"}, config=config, policy=policy, total_steps=4
    )

    root_a = tmp_path / "run_a"
    root_a.mkdir()
    root_b = tmp_path / "run_b"
    root_b.mkdir()

    result_a = materialize_definitive_training_patches(
        schedule,
        case_references=case_references,
        positive_case_ids={"case_pos"},
        loader=_LazyCaseLoader(positive_ids={"case_pos"}),
        patch_root=root_a,
        config=config,
    )
    result_b = materialize_definitive_training_patches(
        schedule,
        case_references=case_references,
        positive_case_ids={"case_pos"},
        loader=_LazyCaseLoader(positive_ids={"case_pos"}),
        patch_root=root_b,
        config=config,
    )

    by_key_a = {(r.step_index, r.role): r for r in result_a.published_patches}
    by_key_b = {(r.step_index, r.role): r for r in result_b.published_patches}
    assert set(by_key_a) == set(by_key_b)
    for key, record_a in by_key_a.items():
        record_b = by_key_b[key]
        assert record_a.metadata.patch_content_sha256 == record_b.metadata.patch_content_sha256
        assert record_a.metadata.metadata_hash == record_b.metadata.metadata_hash


def test_materialization_metadata_contains_no_raw_identifiers_or_filesystem_paths(
    tmp_path: Path,
) -> None:
    config = build_definitive_config_v1()
    policy = build_definitive_training_execution_policy_v1()
    case_references = [
        Phase8DefinitiveTrainCaseReference(case_id=c) for c in ("case_pos", "case_neg")
    ]
    schedule = _small_schedule(
        case_references, positive_case_ids={"case_pos"}, config=config, policy=policy, total_steps=3
    )
    patch_root = tmp_path / "patches"
    patch_root.mkdir()

    result = materialize_definitive_training_patches(
        schedule,
        case_references=case_references,
        positive_case_ids={"case_pos"},
        loader=_LazyCaseLoader(positive_ids={"case_pos"}),
        patch_root=patch_root,
        config=config,
    )

    for record in result.published_patches:
        metadata_text = record.metadata_path.read_text(encoding="utf-8")
        assert str(patch_root) not in metadata_text
        assert str(tmp_path) not in metadata_text
        assert "/" not in metadata_text.replace(
            '"schema_name":"phase8_definitive_training_patch"', ""
        )


# ---------------------------------------------------------------------------
# P. Training from materialized patches
# ---------------------------------------------------------------------------


def _materialize_small_schedule(
    tmp_path: Path, *, total_steps: int, candidate_checkpoint_steps: tuple[int, ...]
) -> tuple[Phase8DefinitivePatchSchedule, Path]:
    config = build_definitive_config_v1()
    policy = build_definitive_training_execution_policy_v1()
    case_references = [
        Phase8DefinitiveTrainCaseReference(case_id=c) for c in ("case_pos", "case_neg")
    ]
    schedule = _small_schedule(
        case_references,
        positive_case_ids={"case_pos"},
        config=config,
        policy=policy,
        total_steps=total_steps,
    )
    patch_root = tmp_path / "patches"
    patch_root.mkdir()
    materialize_definitive_training_patches(
        schedule,
        case_references=case_references,
        positive_case_ids={"case_pos"},
        loader=_LazyCaseLoader(positive_ids={"case_pos"}),
        patch_root=patch_root,
        config=config,
    )
    return schedule, patch_root


@requires_baseline_environment
def test_training_from_materialized_patches_consumes_exact_step_order_and_snapshots(
    tmp_path: Path,
) -> None:
    config = build_definitive_config_v1()
    schedule, patch_root = _materialize_small_schedule(
        tmp_path, total_steps=4, candidate_checkpoint_steps=(2, 4)
    )

    result = definitive_pipeline._run_definitive_training_from_materialized_patches_core(
        schedule,
        patch_root=patch_root,
        config=config,
        total_optimizer_steps=4,
        candidate_checkpoint_steps=(2, 4),
    )

    assert result.all_steps_finite is True
    assert [step.step_index for step in result.step_results] == [0, 1, 2, 3]
    assert set(result.checkpoint_snapshots.keys()) == {2, 4}


@requires_baseline_environment
def test_training_from_materialized_patches_initializes_model_optimizer_seed_exactly_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import torch  # type: ignore[import-not-found]

    config = build_definitive_config_v1()
    schedule, patch_root = _materialize_small_schedule(
        tmp_path, total_steps=4, candidate_checkpoint_steps=(2, 4)
    )

    seed_calls: list[int] = []
    real_manual_seed = torch.manual_seed

    def _spy_manual_seed(seed: int) -> Any:
        seed_calls.append(seed)
        return real_manual_seed(seed)

    monkeypatch.setattr(torch, "manual_seed", _spy_manual_seed)

    model_build_calls = {"count": 0}
    real_build_model = definitive_pipeline._build_definitive_segresnet_model

    def _spy_build_model(**kwargs: Any) -> Any:
        model_build_calls["count"] += 1
        return real_build_model(**kwargs)

    monkeypatch.setattr(definitive_pipeline, "_build_definitive_segresnet_model", _spy_build_model)

    optimizer_calls = {"count": 0}
    real_adamw = torch.optim.AdamW

    def _spy_adamw(*args: Any, **kwargs: Any) -> Any:
        optimizer_calls["count"] += 1
        return real_adamw(*args, **kwargs)

    monkeypatch.setattr(torch.optim, "AdamW", _spy_adamw)

    result = definitive_pipeline._run_definitive_training_from_materialized_patches_core(
        schedule,
        patch_root=patch_root,
        config=config,
        total_optimizer_steps=4,
        candidate_checkpoint_steps=(2, 4),
    )

    assert len(seed_calls) == 1
    assert model_build_calls["count"] == 1
    assert optimizer_calls["count"] == 1
    assert result.all_steps_finite is True


@requires_baseline_environment
def test_training_from_materialized_patches_stops_on_nonfinite_step(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = build_definitive_config_v1()
    schedule, patch_root = _materialize_small_schedule(
        tmp_path, total_steps=4, candidate_checkpoint_steps=(2, 4)
    )

    def _fake_step(
        *,
        step_index: int,
        positive_patch: Any,
        negative_patch: Any,
        model: Any,
        optimizer: Any,
        loss_function: Any,
        torch: Any,
    ) -> Phase8DefinitiveTrainingStepResult:
        if step_index == 1:
            return Phase8DefinitiveTrainingStepResult(
                step_index=step_index,
                loss_value=float("nan"),
                loss_finite=False,
                gradient_finite=False,
            )
        return Phase8DefinitiveTrainingStepResult(
            step_index=step_index, loss_value=0.1, loss_finite=True, gradient_finite=True
        )

    monkeypatch.setattr(
        definitive_pipeline, "_run_definitive_training_step_from_patches", _fake_step
    )

    result = definitive_pipeline._run_definitive_training_from_materialized_patches_core(
        schedule,
        patch_root=patch_root,
        config=config,
        total_optimizer_steps=4,
        candidate_checkpoint_steps=(2, 4),
    )

    assert result.all_steps_finite is False
    assert len(result.step_results) == 2
    assert result.checkpoint_snapshots == {}


@requires_baseline_environment
def test_run_definitive_training_from_materialized_patches_rejects_tampered_policy_steps(
    tmp_path: Path,
) -> None:
    config = build_definitive_config_v1()
    policy = build_definitive_training_execution_policy_v1()
    tampered_policy = object.__new__(type(policy))
    for field in dataclasses.fields(policy):
        object.__setattr__(tampered_policy, field.name, getattr(policy, field.name))
    object.__setattr__(tampered_policy, "total_optimizer_steps", 3)

    schedule, patch_root = _materialize_small_schedule(
        tmp_path, total_steps=4, candidate_checkpoint_steps=(2, 4)
    )

    with pytest.raises(definitive_pipeline.Phase8DefinitivePipelineConfigError):
        definitive_pipeline.run_definitive_training_from_materialized_patches(
            schedule, patch_root=patch_root, config=config, policy=tampered_policy
        )


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
