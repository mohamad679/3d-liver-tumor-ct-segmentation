"""Unit tests for deterministic ProtoEM optimization orchestration."""

from __future__ import annotations

import inspect
from typing import cast

import numpy as np
import pytest

from protoem_ct.artifacts.hashing import sha256_json
from protoem_ct.models.interfaces import (
    BackgroundPrototype,
    EmbeddingMetadata,
    FeatureEncoding3D,
    FeatureResolution3D,
    ForegroundPrototype,
)
from protoem_ct.protoem import (
    InvalidEStepInputError,
    InvalidMStepInputError,
    NonFiniteObjectiveError,
    OrchestrationInputError,
    ProtoEMConfig,
    ProtoEMInferenceInput,
    ProtoEMInitializationBundle,
    ProtoEMIterationExecution,
    ProtoEMObjectiveWeights,
    ProtoEMOptimizationResult,
    ProtoEMPhase5ArtifactReference,
    ProtoEMPhase5ReferenceBundle,
    ProtoEMPrototypeInput,
    ProtoEMQueryFeatureInput,
    ProtoEMSupportProvenance,
    build_protoem_initialization_bundle,
    run_protoem_optimization,
)
from protoem_ct.retrieval.inference import (
    PROTOTYPE_INFERENCE_SCHEMA_NAME,
    PROTOTYPE_INFERENCE_SCHEMA_VERSION,
    PrototypeInferenceResult,
    hash_prototype_inference_identity,
)
from protoem_ct.retrieval.prototypes import (
    SupportPrototypeMemory,
    hash_support_prototype_identity,
)


def _sha256(seed: str) -> str:
    return (seed * 64)[:64]


def _metadata(
    *,
    input_identity: str,
    feature_channels: int = 2,
    feature_spatial_shape: tuple[int, int, int] = (1, 1, 2),
) -> EmbeddingMetadata:
    return EmbeddingMetadata(
        encoder_identity="segresnet_encoder_v1",
        preprocessing_hash=_sha256("a"),
        checkpoint_hash=_sha256("b"),
        input_identity=input_identity,
        feature_stage="final_encoder",
        normalization_name="l2_channel",
        feature_channels=feature_channels,
        resolution=FeatureResolution3D(
            input_spatial_shape=feature_spatial_shape,
            feature_spatial_shape=feature_spatial_shape,
            downsample_factors=(1, 1, 1),
        ),
    )


def _query_feature_encoding(feature_values: np.ndarray | None = None) -> FeatureEncoding3D:
    values = feature_values
    if values is None:
        values = np.asarray([[[[[1.0, 0.0]]], [[[0.0, 1.0]]]]], dtype=np.float32)
    return FeatureEncoding3D(
        feature_data=values,
        metadata=_metadata(
            input_identity="query_case_001",
            feature_channels=int(values.shape[1]),
            feature_spatial_shape=(
                int(values.shape[2]),
                int(values.shape[3]),
                int(values.shape[4]),
            ),
        ),
    )


def _prototype(
    *,
    kind: str,
    vector: tuple[float, ...],
) -> ForegroundPrototype | BackgroundPrototype:
    prototype_class = ForegroundPrototype if kind == "foreground" else BackgroundPrototype
    content_hash = (
        __import__("hashlib")
        .sha256(np.asarray(vector, dtype=np.float64).tobytes(order="C"))
        .hexdigest()
    )
    draft = prototype_class(
        prototype_kind="ignored",
        feature_channels=len(vector),
        contributing_voxel_count=1,
        source_support_identifiers=("support_case_001",),
        normalization_name="l2_channel",
        prototype_vector=vector,
        encoder_identity="segresnet_encoder_v1",
        preprocessing_hash=_sha256("a"),
        checkpoint_hash=_sha256("b"),
        dataset_manifest_hash=_sha256("d"),
        feature_stage="final_encoder",
        source_patient_ids=("support_patient_001",),
        source_case_ids=("support_case_001",),
        prototype_content_sha256=content_hash,
        prototype_identity_sha256="0" * 64,
    )
    return prototype_class(
        prototype_kind=draft.prototype_kind,
        feature_channels=draft.feature_channels,
        contributing_voxel_count=draft.contributing_voxel_count,
        source_support_identifiers=draft.source_support_identifiers,
        normalization_name=draft.normalization_name,
        prototype_vector=draft.prototype_vector,
        encoder_identity=draft.encoder_identity,
        preprocessing_hash=draft.preprocessing_hash,
        checkpoint_hash=draft.checkpoint_hash,
        dataset_manifest_hash=draft.dataset_manifest_hash,
        feature_stage=draft.feature_stage,
        source_patient_ids=draft.source_patient_ids,
        source_case_ids=draft.source_case_ids,
        prototype_content_sha256=draft.prototype_content_sha256,
        prototype_identity_sha256=hash_support_prototype_identity(draft),
    )


def _prototype_memory() -> SupportPrototypeMemory:
    foreground_prototype = cast(
        ForegroundPrototype, _prototype(kind="foreground", vector=(1.0, 0.0))
    )
    background_prototype = cast(
        BackgroundPrototype, _prototype(kind="background", vector=(0.0, 1.0))
    )
    return SupportPrototypeMemory(
        foreground_prototype=foreground_prototype,
        background_prototype=background_prototype,
    )


def _inference_result(
    *,
    foreground_score_map: np.ndarray | None = None,
    background_score_map: np.ndarray | None = None,
    prediction_mask: np.ndarray | None = None,
    memory: SupportPrototypeMemory | None = None,
) -> PrototypeInferenceResult:
    resolved_memory = memory or _prototype_memory()
    fg_scores = foreground_score_map
    if fg_scores is None:
        fg_scores = np.asarray([[[[[1.0, 0.0]]]]], dtype=np.float64)
    bg_scores = background_score_map
    if bg_scores is None:
        bg_scores = np.asarray([[[[[0.0, 1.0]]]]], dtype=np.float64)
    prediction = prediction_mask
    if prediction is None:
        prediction = np.asarray([[[[[1, 0]]]]], dtype=np.uint8)
    confidence = np.abs(fg_scores - bg_scores)
    draft = PrototypeInferenceResult(
        schema_name=PROTOTYPE_INFERENCE_SCHEMA_NAME,
        schema_version=PROTOTYPE_INFERENCE_SCHEMA_VERSION,
        query_identity="query_case_001",
        query_patient_id="query_patient_001",
        query_case_id="query_case_001",
        encoder_identity="segresnet_encoder_v1",
        preprocessing_hash=_sha256("a"),
        checkpoint_hash=_sha256("b"),
        dataset_manifest_hash=_sha256("d"),
        feature_stage="final_encoder",
        normalization_name="l2_channel",
        feature_shape=(1, 2, 1, 1, 2),
        foreground_prototype_identity=resolved_memory.foreground_prototype.prototype_identity_sha256,
        background_prototype_identity=resolved_memory.background_prototype.prototype_identity_sha256,
        tie_policy="background_on_exact_tie",
        scoring_rule="per_voxel_dual_cosine_similarity",
        foreground_score_map=np.ascontiguousarray(fg_scores),
        background_score_map=np.ascontiguousarray(bg_scores),
        prediction_mask=np.ascontiguousarray(prediction),
        confidence_margin_map=np.ascontiguousarray(confidence),
        foreground_score_content_sha256=__import__("hashlib")
        .sha256(np.ascontiguousarray(fg_scores).tobytes(order="C"))
        .hexdigest(),
        background_score_content_sha256=__import__("hashlib")
        .sha256(np.ascontiguousarray(bg_scores).tobytes(order="C"))
        .hexdigest(),
        prediction_content_sha256=__import__("hashlib")
        .sha256(np.ascontiguousarray(prediction).tobytes(order="C"))
        .hexdigest(),
        confidence_margin_content_sha256=__import__("hashlib")
        .sha256(np.ascontiguousarray(confidence).tobytes(order="C"))
        .hexdigest(),
        inference_identity_sha256="0" * 64,
    )
    return PrototypeInferenceResult(
        schema_name=draft.schema_name,
        schema_version=draft.schema_version,
        query_identity=draft.query_identity,
        query_patient_id=draft.query_patient_id,
        query_case_id=draft.query_case_id,
        encoder_identity=draft.encoder_identity,
        preprocessing_hash=draft.preprocessing_hash,
        checkpoint_hash=draft.checkpoint_hash,
        dataset_manifest_hash=draft.dataset_manifest_hash,
        feature_stage=draft.feature_stage,
        normalization_name=draft.normalization_name,
        feature_shape=draft.feature_shape,
        foreground_prototype_identity=draft.foreground_prototype_identity,
        background_prototype_identity=draft.background_prototype_identity,
        tie_policy=draft.tie_policy,
        scoring_rule=draft.scoring_rule,
        foreground_score_map=draft.foreground_score_map,
        background_score_map=draft.background_score_map,
        prediction_mask=draft.prediction_mask,
        confidence_margin_map=draft.confidence_margin_map,
        foreground_score_content_sha256=draft.foreground_score_content_sha256,
        background_score_content_sha256=draft.background_score_content_sha256,
        prediction_content_sha256=draft.prediction_content_sha256,
        confidence_margin_content_sha256=draft.confidence_margin_content_sha256,
        inference_identity_sha256=hash_prototype_inference_identity(draft),
    )


def _support_provenance() -> ProtoEMSupportProvenance:
    return ProtoEMSupportProvenance(
        schema_name="protoem_support_provenance",
        schema_version="v1",
        support_identifier="support_case_001",
        support_patient_id="support_patient_001",
        support_case_id="support_case_001",
        source_input_identity="support_case_001",
        support_manifest_hash=_sha256("c"),
        support_assignment_index=0,
        dataset_manifest_hash=_sha256("d"),
        encoder_identity="segresnet_encoder_v1",
        preprocessing_hash=_sha256("a"),
        checkpoint_hash=_sha256("b"),
        feature_stage="final_encoder",
        normalization_name="l2_channel",
    )


def _weights() -> ProtoEMObjectiveWeights:
    return ProtoEMObjectiveWeights(
        schema_name="protoem_objective_weights",
        schema_version="v1",
        support=1.0,
        query_entropy=1.0,
        class_balance=1.0,
        consistency=1.0,
        proximal=1.0,
    )


def _config_with_threshold(confidence_threshold: float) -> ProtoEMConfig:
    weights = _weights()
    payload = {
        "schema_name": "protoem_config",
        "schema_version": "v1",
        "explicit_seed": 1729,
        "max_iterations": 5,
        "minimum_iterations": 1,
        "convergence_tolerance": 0.001,
        "temperature": 1.0,
        "confidence_threshold": confidence_threshold,
        "foreground_prior": 0.5,
        "update_schedule": "fixed_em_like",
        "prototype_mode": "single_prototype",
        "objective_weights": {
            "schema_name": weights.schema_name,
            "schema_version": weights.schema_version,
            "support": weights.support,
            "query_entropy": weights.query_entropy,
            "class_balance": weights.class_balance,
            "consistency": weights.consistency,
            "proximal": weights.proximal,
        },
        "phase5_initialization_schema_name": "phase5_run_summary",
        "phase5_initialization_artifact_hash": _sha256("1"),
        "phase5_prototype_schema_name": "prototype_memory",
        "phase5_prototype_artifact_hash": _sha256("2"),
        "phase5_inference_schema_name": "prototype_only_inference_result",
        "phase5_inference_artifact_hash": _sha256("3"),
    }
    return ProtoEMConfig(
        schema_name="protoem_config",
        schema_version="v1",
        config_hash=sha256_json(payload),
        explicit_seed=1729,
        max_iterations=5,
        minimum_iterations=1,
        convergence_tolerance=0.001,
        temperature=1.0,
        confidence_threshold=confidence_threshold,
        foreground_prior=0.5,
        update_schedule="fixed_em_like",
        prototype_mode="single_prototype",
        objective_weights=weights,
        phase5_initialization_schema_name="phase5_run_summary",
        phase5_initialization_artifact_hash=_sha256("1"),
        phase5_prototype_schema_name="prototype_memory",
        phase5_prototype_artifact_hash=_sha256("2"),
        phase5_inference_schema_name="prototype_only_inference_result",
        phase5_inference_artifact_hash=_sha256("3"),
    )


def _references() -> ProtoEMPhase5ReferenceBundle:
    return ProtoEMPhase5ReferenceBundle(
        schema_name="protoem_phase5_reference_bundle",
        schema_version="v1",
        initialization_reference=ProtoEMPhase5ArtifactReference(
            schema_name="protoem_phase5_artifact_reference",
            schema_version="v1",
            artifact_schema_name="phase5_run_summary",
            artifact_hash=_sha256("1"),
        ),
        prototype_reference=ProtoEMPhase5ArtifactReference(
            schema_name="protoem_phase5_artifact_reference",
            schema_version="v1",
            artifact_schema_name="prototype_memory",
            artifact_hash=_sha256("2"),
        ),
        inference_reference=ProtoEMPhase5ArtifactReference(
            schema_name="protoem_phase5_artifact_reference",
            schema_version="v1",
            artifact_schema_name="prototype_only_inference_result",
            artifact_hash=_sha256("3"),
        ),
    )


def _initialization_bundle(
    *,
    feature_values: np.ndarray | None = None,
    foreground_score_map: np.ndarray | None = None,
    background_score_map: np.ndarray | None = None,
    prediction_mask: np.ndarray | None = None,
) -> tuple[ProtoEMInitializationBundle, ProtoEMConfig]:
    config = _config_with_threshold(0.7)
    query = ProtoEMQueryFeatureInput(
        schema_name="protoem_query_feature_input",
        schema_version="v1",
        query_feature_encoding=_query_feature_encoding(feature_values),
        query_identity="query_case_001",
        query_patient_id="query_patient_001",
        query_case_id="query_case_001",
        dataset_manifest_hash=_sha256("d"),
    )
    memory = _prototype_memory()
    prototype = ProtoEMPrototypeInput(
        schema_name="protoem_prototype_input",
        schema_version="v1",
        prototype_memory=memory,
        support_provenance=(_support_provenance(),),
    )
    inference = ProtoEMInferenceInput(
        schema_name="protoem_inference_input",
        schema_version="v1",
        inference_result=_inference_result(
            foreground_score_map=foreground_score_map,
            background_score_map=background_score_map,
            prediction_mask=prediction_mask,
            memory=memory,
        ),
    )
    bundle = build_protoem_initialization_bundle(
        config=config,
        query_input=query,
        prototype_input=prototype,
        inference_input=inference,
        phase5_references=_references(),
    )
    return bundle, config


def _config_override(
    base: ProtoEMConfig,
    *,
    max_iterations: int | None = None,
    minimum_iterations: int | None = None,
    convergence_tolerance: float | None = None,
    confidence_threshold: float | None = None,
    update_schedule: str | None = None,
) -> ProtoEMConfig:
    payload = {
        "schema_name": base.schema_name,
        "schema_version": base.schema_version,
        "explicit_seed": base.explicit_seed,
        "max_iterations": max_iterations if max_iterations is not None else base.max_iterations,
        "minimum_iterations": (
            minimum_iterations if minimum_iterations is not None else base.minimum_iterations
        ),
        "convergence_tolerance": (
            convergence_tolerance
            if convergence_tolerance is not None
            else base.convergence_tolerance
        ),
        "temperature": base.temperature,
        "confidence_threshold": (
            confidence_threshold if confidence_threshold is not None else base.confidence_threshold
        ),
        "foreground_prior": base.foreground_prior,
        "update_schedule": update_schedule if update_schedule is not None else base.update_schedule,
        "prototype_mode": base.prototype_mode,
        "objective_weights": {
            "schema_name": base.objective_weights.schema_name,
            "schema_version": base.objective_weights.schema_version,
            "support": base.objective_weights.support,
            "query_entropy": base.objective_weights.query_entropy,
            "class_balance": base.objective_weights.class_balance,
            "consistency": base.objective_weights.consistency,
            "proximal": base.objective_weights.proximal,
        },
        "phase5_initialization_schema_name": base.phase5_initialization_schema_name,
        "phase5_initialization_artifact_hash": base.phase5_initialization_artifact_hash,
        "phase5_prototype_schema_name": base.phase5_prototype_schema_name,
        "phase5_prototype_artifact_hash": base.phase5_prototype_artifact_hash,
        "phase5_inference_schema_name": base.phase5_inference_schema_name,
        "phase5_inference_artifact_hash": base.phase5_inference_artifact_hash,
    }
    return ProtoEMConfig(
        schema_name=base.schema_name,
        schema_version=base.schema_version,
        config_hash=sha256_json(payload),
        explicit_seed=base.explicit_seed,
        max_iterations=cast(int, payload["max_iterations"]),
        minimum_iterations=cast(int, payload["minimum_iterations"]),
        convergence_tolerance=cast(float, payload["convergence_tolerance"]),
        temperature=base.temperature,
        confidence_threshold=cast(float, payload["confidence_threshold"]),
        foreground_prior=base.foreground_prior,
        update_schedule=cast(str, payload["update_schedule"]),
        prototype_mode=base.prototype_mode,
        objective_weights=base.objective_weights,
        phase5_initialization_schema_name=base.phase5_initialization_schema_name,
        phase5_initialization_artifact_hash=base.phase5_initialization_artifact_hash,
        phase5_prototype_schema_name=base.phase5_prototype_schema_name,
        phase5_prototype_artifact_hash=base.phase5_prototype_artifact_hash,
        phase5_inference_schema_name=base.phase5_inference_schema_name,
        phase5_inference_artifact_hash=base.phase5_inference_artifact_hash,
    )


def _stable_bundle_and_config() -> tuple[ProtoEMInitializationBundle, ProtoEMConfig]:
    bundle, _ = _initialization_bundle()
    return bundle, _config_override(
        _config_with_threshold(0.0),
        max_iterations=5,
        minimum_iterations=2,
        convergence_tolerance=1e9,
        confidence_threshold=0.0,
    )


def test_exact_one_iteration_execution() -> None:
    bundle, config = _initialization_bundle()
    config = _config_override(_config_with_threshold(0.0), max_iterations=1, minimum_iterations=1)

    result = run_protoem_optimization(config=config, initialization_bundle=bundle)

    assert isinstance(result, ProtoEMOptimizationResult)
    assert len(result.iteration_executions) == 1
    assert result.stopping_decision.stop_reason == "max_iterations"
    assert result.final_state.iteration_index == 1


def test_exact_bounded_max_iteration_execution() -> None:
    bundle, config = _initialization_bundle()
    config = _config_override(_config_with_threshold(0.0), max_iterations=1, minimum_iterations=1)

    result = run_protoem_optimization(config=config, initialization_bundle=bundle)

    assert result.stopping_record is not None
    assert result.stopping_record.stop_reason == "max_iterations"
    assert result.stopping_record.failed is False


def test_tolerance_stopping_after_minimum_iterations() -> None:
    bundle, config = _stable_bundle_and_config()

    result = run_protoem_optimization(config=config, initialization_bundle=bundle)

    assert result.stopping_decision.stop_reason == "tolerance_reached"
    assert len(result.iteration_executions) == 2


def test_tolerance_cannot_stop_early() -> None:
    bundle, config = _stable_bundle_and_config()
    config = _config_override(config, minimum_iterations=3)

    result = run_protoem_optimization(config=config, initialization_bundle=bundle)

    assert result.stopping_decision.stop_reason == "tolerance_reached"
    assert len(result.iteration_executions) == 3


def test_foreground_collapse() -> None:
    feature_values = np.asarray([[[[[0.0, 0.0]]], [[[1.0, 1.0]]]]], dtype=np.float32)
    bundle, config = _initialization_bundle(
        feature_values=feature_values,
        foreground_score_map=np.asarray([[[[[0.0, 0.0]]]]], dtype=np.float64),
        background_score_map=np.asarray([[[[[1.0, 1.0]]]]], dtype=np.float64),
        prediction_mask=np.asarray([[[[[0, 0]]]]], dtype=np.uint8),
    )
    config = _config_override(_config_with_threshold(0.0), max_iterations=3, minimum_iterations=1)

    result = run_protoem_optimization(config=config, initialization_bundle=bundle)

    assert result.execution_status == "failed"
    assert result.stopping_decision.stop_reason == "foreground_collapse"
    assert result.collapse_record is not None
    assert result.objective_trace is None


def test_background_collapse() -> None:
    feature_values = np.asarray([[[[[1.0, 1.0]]], [[[0.0, 0.0]]]]], dtype=np.float32)
    bundle, config = _initialization_bundle(
        feature_values=feature_values,
        foreground_score_map=np.asarray([[[[[1.0, 1.0]]]]], dtype=np.float64),
        background_score_map=np.asarray([[[[[0.0, 0.0]]]]], dtype=np.float64),
        prediction_mask=np.asarray([[[[[1, 1]]]]], dtype=np.uint8),
    )
    config = _config_override(_config_with_threshold(0.0), max_iterations=3, minimum_iterations=1)

    result = run_protoem_optimization(config=config, initialization_bundle=bundle)

    assert result.execution_status == "failed"
    assert result.stopping_decision.stop_reason == "background_collapse"
    assert result.collapse_record is not None
    assert result.collapse_record.background_count == 0


def test_no_confident_voxels() -> None:
    bundle, _ = _initialization_bundle(
        foreground_score_map=np.asarray([[[[[0.0, 0.0]]]]], dtype=np.float64),
        background_score_map=np.asarray([[[[[0.0, 0.0]]]]], dtype=np.float64),
        prediction_mask=np.asarray([[[[[0, 0]]]]], dtype=np.uint8),
    )
    config = _config_override(_config_with_threshold(1.0), max_iterations=3, minimum_iterations=1)

    result = run_protoem_optimization(config=config, initialization_bundle=bundle)

    assert result.execution_status == "failed"
    assert result.stopping_decision.stop_reason == "no_confident_voxels"
    assert result.collapse_record is None
    assert result.objective_trace is None


def test_non_finite_objective_handling(monkeypatch: pytest.MonkeyPatch) -> None:
    bundle, config = _initialization_bundle()

    def _raise_non_finite(**_: object) -> object:
        raise NonFiniteObjectiveError("forced non-finite objective")

    monkeypatch.setattr("protoem_ct.protoem.optimize.run_protoem_m_step", _raise_non_finite)
    result = run_protoem_optimization(
        config=_config_override(_config_with_threshold(0.0)), initialization_bundle=bundle
    )

    assert result.execution_status == "failed"
    assert result.stopping_decision.stop_reason == "non_finite_objective"


def test_numerical_e_step_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    bundle, config = _initialization_bundle()

    def _raise_e(**_: object) -> object:
        raise InvalidEStepInputError("forced E-step failure")

    monkeypatch.setattr("protoem_ct.protoem.optimize.run_protoem_e_step", _raise_e)
    result = run_protoem_optimization(
        config=_config_override(_config_with_threshold(0.0)), initialization_bundle=bundle
    )

    assert result.execution_status == "failed"
    assert result.stopping_decision.stop_reason == "numerical_failure"


def test_numerical_m_step_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    bundle, config = _initialization_bundle()

    def _raise_m(**_: object) -> object:
        raise InvalidMStepInputError("forced M-step failure")

    monkeypatch.setattr("protoem_ct.protoem.optimize.run_protoem_m_step", _raise_m)
    result = run_protoem_optimization(
        config=_config_override(_config_with_threshold(0.0)), initialization_bundle=bundle
    )

    assert result.execution_status == "failed"
    assert result.stopping_decision.stop_reason == "numerical_failure"


def test_correct_iteration_index_sequence() -> None:
    bundle, config = _stable_bundle_and_config()
    config = _config_override(config, minimum_iterations=3)

    result = run_protoem_optimization(config=config, initialization_bundle=bundle)

    assert [item.iteration_index for item in result.iteration_executions] == [0, 1, 2]
    assert [item.source_state.iteration_index for item in result.iteration_executions] == [0, 1, 2]
    assert [item.target_state.iteration_index for item in result.iteration_executions] == [1, 2, 3]


def test_completed_iteration_count_equals_trace_length() -> None:
    bundle, config = _stable_bundle_and_config()

    result = run_protoem_optimization(config=config, initialization_bundle=bundle)

    assert result.objective_trace is not None
    assert result.stopping_record is not None
    assert result.stopping_record.completed_iteration_count == len(
        result.objective_trace.iterations
    )


def test_stopping_record_consistency() -> None:
    bundle, config = _stable_bundle_and_config()

    result = run_protoem_optimization(config=config, initialization_bundle=bundle)

    assert result.stopping_record is not None
    assert result.stopping_record.converged is True
    assert result.stopping_record.failed is False


def test_objective_records_contain_every_term() -> None:
    bundle, config = _stable_bundle_and_config()

    result = run_protoem_optimization(config=config, initialization_bundle=bundle)
    record = result.objective_trace.iterations[0]  # type: ignore[union-attr]

    assert record.support_objective >= 0.0
    assert record.query_entropy_objective >= 0.0
    assert record.class_balance_objective >= 0.0
    assert record.consistency_objective >= 0.0
    assert record.proximal_objective >= 0.0


def test_repeated_execution_deterministic() -> None:
    bundle, config = _stable_bundle_and_config()

    first = run_protoem_optimization(config=config, initialization_bundle=bundle)
    second = run_protoem_optimization(config=config, initialization_bundle=bundle)

    assert first.result_identity_hash == second.result_identity_hash
    assert (
        first.stopping_decision.decision_identity_hash
        == second.stopping_decision.decision_identity_hash
    )
    assert first.objective_trace.objective_trace_hash == second.objective_trace.objective_trace_hash  # type: ignore[union-attr]


def test_non_contiguous_input_arrays_produce_identical_identities() -> None:
    base = np.asarray([[[[[1.0, 0.0]]], [[[0.0, 1.0]]]]], dtype=np.float32)
    non_contiguous = base[..., ::-1][..., ::-1]
    first_bundle, config = _initialization_bundle(feature_values=base)
    second_bundle, _ = _initialization_bundle(feature_values=non_contiguous)
    config = _config_override(_config_with_threshold(0.0), max_iterations=1, minimum_iterations=1)

    first = run_protoem_optimization(config=config, initialization_bundle=first_bundle)
    second = run_protoem_optimization(config=config, initialization_bundle=second_bundle)

    assert first.result_identity_hash == second.result_identity_hash


def test_input_arrays_remain_unchanged() -> None:
    features = np.asarray([[[[[1.0, 0.0]]], [[[0.0, 1.0]]]]], dtype=np.float32)
    before = features.copy()
    bundle, _ = _initialization_bundle(feature_values=features)
    config = _config_override(_config_with_threshold(0.0), max_iterations=1, minimum_iterations=1)

    _ = run_protoem_optimization(config=config, initialization_bundle=bundle)

    assert np.array_equal(features, before)


def test_learned_positive_step_is_not_implemented() -> None:
    bundle, config = _initialization_bundle()
    learned = _config_override(_config_with_threshold(0.0), update_schedule="learned_positive_step")

    with pytest.raises(OrchestrationInputError):
        run_protoem_optimization(config=learned, initialization_bundle=bundle)


def test_query_labels_reference_masks_absent_from_public_apis() -> None:
    signature_text = str(inspect.signature(run_protoem_optimization)).lower()
    assert "query_label" not in signature_text
    assert "reference_mask" not in signature_text
    assert "query_label" not in ProtoEMOptimizationResult.__dataclass_fields__
    assert "reference_mask" not in ProtoEMOptimizationResult.__dataclass_fields__
    assert "query_label" not in ProtoEMIterationExecution.__dataclass_fields__
    assert "reference_mask" not in ProtoEMIterationExecution.__dataclass_fields__


def test_no_filesystem_or_model_execution_required() -> None:
    bundle, config = _initialization_bundle()
    result = run_protoem_optimization(
        config=_config_override(
            _config_with_threshold(0.0), max_iterations=1, minimum_iterations=1
        ),
        initialization_bundle=bundle,
    )

    assert result.execution_status == "completed"
