"""Unit tests for deterministic ProtoEM-CT transductive state contracts."""

from __future__ import annotations

import hashlib
import inspect
from typing import cast

import numpy as np
import pytest

from protoem_ct.artifacts.hashing import canonical_json_bytes, sha256_json
from protoem_ct.models.interfaces import (
    BackgroundPrototype,
    EmbeddingMetadata,
    FeatureEncoding3D,
    FeatureResolution3D,
    ForegroundPrototype,
)
from protoem_ct.protoem import (
    AssignmentConsistencyFailureError,
    IncompatibleStateTransitionError,
    PosteriorConsistencyFailureError,
    ProtoEMConfig,
    ProtoEMInferenceInput,
    ProtoEMInitializationBundle,
    ProtoEMObjectiveTerms,
    ProtoEMObjectiveWeights,
    ProtoEMPhase5ArtifactReference,
    ProtoEMPhase5ReferenceBundle,
    ProtoEMPosteriorState,
    ProtoEMPrototypeInput,
    ProtoEMQueryFeatureInput,
    ProtoEMStateTransition,
    ProtoEMSupportProvenance,
    ProtoEMTransductiveState,
    PrototypeStateFailureError,
    build_initial_protoem_transductive_state,
    build_next_protoem_transductive_state,
    build_protoem_assignment_state,
    build_protoem_initialization_bundle,
    build_protoem_posterior_state,
    build_protoem_prototype_state,
    compute_protoem_objective_terms,
    protoem_transductive_state_to_dict,
    protoem_transductive_state_to_json,
    run_protoem_e_step,
    run_protoem_m_step,
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


def _query_feature_encoding(
    *,
    query_identity: str = "query_case_001",
    feature_values: np.ndarray | None = None,
) -> FeatureEncoding3D:
    values = feature_values
    if values is None:
        values = np.asarray([[[[[1.0, 0.0]]], [[[0.0, 1.0]]]]], dtype=np.float32)
    return FeatureEncoding3D(
        feature_data=values,
        metadata=_metadata(
            input_identity=query_identity,
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
    content_hash = hashlib.sha256(
        np.asarray(vector, dtype=np.float64).tobytes(order="C")
    ).hexdigest()
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
        foreground_score_content_sha256=hashlib.sha256(
            np.ascontiguousarray(fg_scores).tobytes(order="C")
        ).hexdigest(),
        background_score_content_sha256=hashlib.sha256(
            np.ascontiguousarray(bg_scores).tobytes(order="C")
        ).hexdigest(),
        prediction_content_sha256=hashlib.sha256(
            np.ascontiguousarray(prediction).tobytes(order="C")
        ).hexdigest(),
        confidence_margin_content_sha256=hashlib.sha256(
            np.ascontiguousarray(confidence).tobytes(order="C")
        ).hexdigest(),
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


def _config() -> ProtoEMConfig:
    return _config_with_threshold(0.7)


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
    config = _config()
    query = ProtoEMQueryFeatureInput(
        schema_name="protoem_query_feature_input",
        schema_version="v1",
        query_feature_encoding=_query_feature_encoding(feature_values=feature_values),
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


def _objective_terms() -> ProtoEMObjectiveTerms:
    return compute_protoem_objective_terms(
        objective_weights=_weights(),
        foreground_posterior_map=np.asarray([[[[[0.75, 0.25]]]]], dtype=np.float64),
        background_posterior_map=np.asarray([[[[[0.25, 0.75]]]]], dtype=np.float64),
        previous_foreground_posterior_map=np.asarray([[[[[0.5, 0.5]]]]], dtype=np.float64),
        previous_background_posterior_map=np.asarray([[[[[0.5, 0.5]]]]], dtype=np.float64),
        updated_foreground_prototype=np.asarray([0.75, 0.25], dtype=np.float64),
        updated_background_prototype=np.asarray([0.25, 0.75], dtype=np.float64),
        support_foreground_reference=np.asarray([1.0, 0.0], dtype=np.float64),
        support_background_reference=np.asarray([0.0, 1.0], dtype=np.float64),
        proximal_foreground_reference=np.asarray([1.0, 0.0], dtype=np.float64),
        proximal_background_reference=np.asarray([0.0, 1.0], dtype=np.float64),
        foreground_prior=0.2,
    )


def test_valid_initial_state_construction() -> None:
    bundle, config = _initialization_bundle()
    state = build_initial_protoem_transductive_state(
        initialization_bundle=bundle,
        config=config,
    )

    assert isinstance(state, ProtoEMTransductiveState)
    assert state.iteration_index == 0
    assert state.current_objective_terms is None
    assert state.previous_objective_total is None
    assert state.convergence_delta is None


def test_initial_state_uses_actual_posterior_conversion_not_raw_phase5_scores() -> None:
    bundle, config = _initialization_bundle(
        foreground_score_map=np.asarray([[[[[1.0, 0.0]]]]], dtype=np.float64),
        background_score_map=np.asarray([[[[[0.0, 1.0]]]]], dtype=np.float64),
        prediction_mask=np.asarray([[[[[1, 0]]]]], dtype=np.uint8),
    )
    state = build_initial_protoem_transductive_state(
        initialization_bundle=bundle,
        config=config,
    )

    assert not np.array_equal(
        state.posterior_state.foreground_posterior_map,
        bundle.initial_foreground_score_map,
    )


def test_exact_tie_resolves_to_background() -> None:
    bundle, config = _initialization_bundle(
        foreground_score_map=np.asarray([[[[[0.0, 0.0]]]]], dtype=np.float64),
        background_score_map=np.asarray([[[[[0.0, 0.0]]]]], dtype=np.float64),
        prediction_mask=np.asarray([[[[[0, 0]]]]], dtype=np.uint8),
    )
    state = build_initial_protoem_transductive_state(
        initialization_bundle=bundle,
        config=config,
    )

    assert int(state.assignment_state.hard_assignment_map[0, 0, 0, 0, 0]) == 0


def test_threshold_boundary_behavior() -> None:
    bundle, config = _initialization_bundle()
    state = build_initial_protoem_transductive_state(
        initialization_bundle=bundle,
        config=config,
    )
    confidence = float(state.posterior_state.confidence_map[0, 0, 0, 0, 0])
    exact_config = _config_with_threshold(confidence)
    exact_state = build_initial_protoem_transductive_state(
        initialization_bundle=bundle,
        config=exact_config,
    )
    assert bool(exact_state.posterior_state.confident_voxel_mask[0, 0, 0, 0, 0])


def test_exact_shapes_and_dtypes() -> None:
    bundle, config = _initialization_bundle()
    state = build_initial_protoem_transductive_state(
        initialization_bundle=bundle,
        config=config,
    )

    assert state.posterior_state.foreground_posterior_map.shape == (1, 1, 1, 1, 2)
    assert state.posterior_state.foreground_posterior_map.dtype == np.float64
    assert state.assignment_state.hard_assignment_map.dtype == np.uint8


def test_posterior_sum_validation() -> None:
    with pytest.raises(PosteriorConsistencyFailureError):
        ProtoEMPosteriorState(
            schema_name="protoem_posterior_state",
            schema_version="v1",
            foreground_posterior_map=np.asarray([[[[[0.8]]]]], dtype=np.float64),
            background_posterior_map=np.asarray([[[[[0.3]]]]], dtype=np.float64),
            confidence_map=np.asarray([[[[[0.8]]]]], dtype=np.float64),
            confident_voxel_mask=np.asarray([[[[[1]]]]], dtype=bool),
            zero_norm_query_voxel_mask=np.asarray([[[[[0]]]]], dtype=bool),
            posterior_state_identity_hash=_sha256("e"),
        )


def test_confidence_mask_consistency() -> None:
    bundle, config = _initialization_bundle()
    state = build_initial_protoem_transductive_state(
        initialization_bundle=bundle,
        config=config,
    )
    expected = state.posterior_state.confidence_map >= config.confidence_threshold
    assert np.array_equal(state.posterior_state.confident_voxel_mask, expected)


def test_assignment_count_consistency() -> None:
    bundle, config = _initialization_bundle()
    state = build_initial_protoem_transductive_state(
        initialization_bundle=bundle,
        config=config,
    )
    mask = state.posterior_state.confident_voxel_mask.astype(bool)
    assert state.assignment_state.confident_voxel_count == int(np.count_nonzero(mask))


def test_deterministic_state_identity() -> None:
    bundle, config = _initialization_bundle()
    first = build_initial_protoem_transductive_state(
        initialization_bundle=bundle,
        config=config,
    )
    second = build_initial_protoem_transductive_state(
        initialization_bundle=bundle,
        config=config,
    )

    assert first.state_identity_hash == second.state_identity_hash


def test_non_contiguous_input_arrays_produce_same_identity() -> None:
    base = np.asarray([[[[[1.0, 0.0]]], [[[0.0, 1.0]]]]], dtype=np.float32)
    non_contiguous = base[..., ::-1][..., ::-1]
    first_bundle, config = _initialization_bundle(feature_values=base)
    second_bundle, _ = _initialization_bundle(feature_values=non_contiguous)
    first = build_initial_protoem_transductive_state(
        initialization_bundle=first_bundle,
        config=config,
    )
    second = build_initial_protoem_transductive_state(
        initialization_bundle=second_bundle,
        config=config,
    )

    assert first.state_identity_hash == second.state_identity_hash


def test_input_arrays_remain_unchanged() -> None:
    features = np.asarray([[[[[1.0, 0.0]]], [[[0.0, 1.0]]]]], dtype=np.float32)
    features_before = features.copy()
    bundle, config = _initialization_bundle(feature_values=features)
    _ = build_initial_protoem_transductive_state(
        initialization_bundle=bundle,
        config=config,
    )
    assert np.array_equal(features, features_before)


def test_config_initialization_mismatch_rejection() -> None:
    bundle, config = _initialization_bundle()
    bad_payload = {
        "schema_name": "protoem_config",
        "schema_version": "v1",
        "explicit_seed": config.explicit_seed,
        "max_iterations": config.max_iterations,
        "minimum_iterations": config.minimum_iterations,
        "convergence_tolerance": config.convergence_tolerance,
        "temperature": config.temperature,
        "confidence_threshold": config.confidence_threshold,
        "foreground_prior": config.foreground_prior,
        "update_schedule": config.update_schedule,
        "prototype_mode": config.prototype_mode,
        "objective_weights": {
            "schema_name": config.objective_weights.schema_name,
            "schema_version": config.objective_weights.schema_version,
            "support": config.objective_weights.support,
            "query_entropy": config.objective_weights.query_entropy,
            "class_balance": config.objective_weights.class_balance,
            "consistency": config.objective_weights.consistency,
            "proximal": config.objective_weights.proximal,
        },
        "phase5_initialization_schema_name": config.phase5_initialization_schema_name,
        "phase5_initialization_artifact_hash": _sha256("9"),
        "phase5_prototype_schema_name": config.phase5_prototype_schema_name,
        "phase5_prototype_artifact_hash": config.phase5_prototype_artifact_hash,
        "phase5_inference_schema_name": config.phase5_inference_schema_name,
        "phase5_inference_artifact_hash": config.phase5_inference_artifact_hash,
    }
    bad_config = ProtoEMConfig(
        schema_name="protoem_config",
        schema_version="v1",
        config_hash=sha256_json(bad_payload),
        explicit_seed=config.explicit_seed,
        max_iterations=config.max_iterations,
        minimum_iterations=config.minimum_iterations,
        convergence_tolerance=config.convergence_tolerance,
        temperature=config.temperature,
        confidence_threshold=config.confidence_threshold,
        foreground_prior=config.foreground_prior,
        update_schedule=config.update_schedule,
        prototype_mode=config.prototype_mode,
        objective_weights=config.objective_weights,
        phase5_initialization_schema_name=config.phase5_initialization_schema_name,
        phase5_initialization_artifact_hash=_sha256("9"),
        phase5_prototype_schema_name=config.phase5_prototype_schema_name,
        phase5_prototype_artifact_hash=config.phase5_prototype_artifact_hash,
        phase5_inference_schema_name=config.phase5_inference_schema_name,
        phase5_inference_artifact_hash=config.phase5_inference_artifact_hash,
    )

    with pytest.raises(IncompatibleStateTransitionError):
        build_initial_protoem_transductive_state(
            initialization_bundle=bundle,
            config=bad_config,
        )


def test_malformed_prototype_state_rejection() -> None:
    with pytest.raises(PrototypeStateFailureError):
        build_protoem_prototype_state(
            foreground_prototypes=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
            background_prototypes=np.asarray([0.0, 1.0], dtype=np.float64),
            prototype_mode="single_prototype",
            channel_count=2,
        )


def test_malformed_posterior_state_rejection() -> None:
    with pytest.raises(PosteriorConsistencyFailureError):
        build_protoem_posterior_state(
            foreground_posterior_map=np.asarray([[[[[0.8]]]]], dtype=np.float64),
            background_posterior_map=np.asarray([[[[[0.4]]]]], dtype=np.float64),
            confidence_map=np.asarray([[[[[0.8]]]]], dtype=np.float64),
            confident_voxel_mask=np.asarray([[[[[1]]]]], dtype=bool),
            zero_norm_query_voxel_mask=np.asarray([[[[[0]]]]], dtype=bool),
        )


def test_malformed_assignment_state_rejection() -> None:
    with pytest.raises(AssignmentConsistencyFailureError):
        build_protoem_assignment_state(
            hard_assignment_map=np.asarray([[[[[2]]]]], dtype=np.uint8),
            confident_voxel_mask=np.asarray([[[[[1]]]]], dtype=bool),
        )


def test_valid_one_step_transition() -> None:
    bundle, config = _initialization_bundle()
    source = build_initial_protoem_transductive_state(
        initialization_bundle=bundle,
        config=config,
    )
    e_step = run_protoem_e_step(
        query_feature_tensor=np.asarray(
            bundle.query_feature_encoding.feature_data, dtype=np.float64
        ),
        foreground_prototypes=np.asarray(
            bundle.foreground_prototype.prototype_vector, dtype=np.float64
        ),
        background_prototypes=np.asarray(
            bundle.background_prototype.prototype_vector, dtype=np.float64
        ),
        temperature=config.temperature,
        confidence_threshold=config.confidence_threshold,
        foreground_prior=config.foreground_prior,
    )
    m_step = run_protoem_m_step(
        query_feature_tensor=np.asarray(
            bundle.query_feature_encoding.feature_data, dtype=np.float64
        ),
        foreground_posterior_map=e_step.foreground_posterior_map,
        background_posterior_map=e_step.background_posterior_map,
        confidence_mask=e_step.confidence_mask_result.confident_mask,
        initial_foreground_prototype=np.asarray(
            bundle.foreground_prototype.prototype_vector, dtype=np.float64
        ),
        initial_background_prototype=np.asarray(
            bundle.background_prototype.prototype_vector, dtype=np.float64
        ),
        previous_foreground_prototype=np.asarray(
            bundle.foreground_prototype.prototype_vector, dtype=np.float64
        ),
        previous_background_prototype=np.asarray(
            bundle.background_prototype.prototype_vector, dtype=np.float64
        ),
        objective_weights=config.objective_weights,
        foreground_prior=config.foreground_prior,
        proximal_foreground_reference=np.asarray(
            bundle.foreground_prototype.prototype_vector, dtype=np.float64
        ),
        proximal_background_reference=np.asarray(
            bundle.background_prototype.prototype_vector, dtype=np.float64
        ),
        previous_foreground_posterior_map=source.posterior_state.foreground_posterior_map,
        previous_background_posterior_map=source.posterior_state.background_posterior_map,
        support_foreground_reference=np.asarray(
            bundle.foreground_prototype.prototype_vector, dtype=np.float64
        ),
        support_background_reference=np.asarray(
            bundle.background_prototype.prototype_vector, dtype=np.float64
        ),
    )
    target, transition = build_next_protoem_transductive_state(
        source_state=source,
        e_step_result=e_step,
        m_step_result=m_step,
        objective_terms=m_step.objective_terms,
    )

    assert isinstance(target, ProtoEMTransductiveState)
    assert isinstance(transition, ProtoEMStateTransition)


def test_iteration_increments_by_exactly_one() -> None:
    bundle, config = _initialization_bundle()
    source = build_initial_protoem_transductive_state(
        initialization_bundle=bundle,
        config=config,
    )
    e_step = run_protoem_e_step(
        query_feature_tensor=np.asarray(
            bundle.query_feature_encoding.feature_data, dtype=np.float64
        ),
        foreground_prototypes=np.asarray(
            bundle.foreground_prototype.prototype_vector, dtype=np.float64
        ),
        background_prototypes=np.asarray(
            bundle.background_prototype.prototype_vector, dtype=np.float64
        ),
        temperature=config.temperature,
        confidence_threshold=config.confidence_threshold,
        foreground_prior=config.foreground_prior,
    )
    m_step = run_protoem_m_step(
        query_feature_tensor=np.asarray(
            bundle.query_feature_encoding.feature_data, dtype=np.float64
        ),
        foreground_posterior_map=e_step.foreground_posterior_map,
        background_posterior_map=e_step.background_posterior_map,
        confidence_mask=e_step.confidence_mask_result.confident_mask,
        initial_foreground_prototype=np.asarray(
            bundle.foreground_prototype.prototype_vector, dtype=np.float64
        ),
        initial_background_prototype=np.asarray(
            bundle.background_prototype.prototype_vector, dtype=np.float64
        ),
        previous_foreground_prototype=np.asarray(
            bundle.foreground_prototype.prototype_vector, dtype=np.float64
        ),
        previous_background_prototype=np.asarray(
            bundle.background_prototype.prototype_vector, dtype=np.float64
        ),
        objective_weights=config.objective_weights,
        foreground_prior=config.foreground_prior,
        proximal_foreground_reference=np.asarray(
            bundle.foreground_prototype.prototype_vector, dtype=np.float64
        ),
        proximal_background_reference=np.asarray(
            bundle.background_prototype.prototype_vector, dtype=np.float64
        ),
        previous_foreground_posterior_map=source.posterior_state.foreground_posterior_map,
        previous_background_posterior_map=source.posterior_state.background_posterior_map,
        support_foreground_reference=np.asarray(
            bundle.foreground_prototype.prototype_vector, dtype=np.float64
        ),
        support_background_reference=np.asarray(
            bundle.background_prototype.prototype_vector, dtype=np.float64
        ),
    )
    target, _ = build_next_protoem_transductive_state(
        source_state=source,
        e_step_result=e_step,
        m_step_result=m_step,
        objective_terms=m_step.objective_terms,
    )

    assert target.iteration_index == source.iteration_index + 1


def test_convergence_delta_calculation() -> None:
    bundle, config = _initialization_bundle()
    source = build_initial_protoem_transductive_state(
        initialization_bundle=bundle,
        config=config,
    )
    first_e = run_protoem_e_step(
        query_feature_tensor=np.asarray(
            bundle.query_feature_encoding.feature_data, dtype=np.float64
        ),
        foreground_prototypes=np.asarray(
            bundle.foreground_prototype.prototype_vector, dtype=np.float64
        ),
        background_prototypes=np.asarray(
            bundle.background_prototype.prototype_vector, dtype=np.float64
        ),
        temperature=config.temperature,
        confidence_threshold=config.confidence_threshold,
        foreground_prior=config.foreground_prior,
    )
    first_m = run_protoem_m_step(
        query_feature_tensor=np.asarray(
            bundle.query_feature_encoding.feature_data, dtype=np.float64
        ),
        foreground_posterior_map=first_e.foreground_posterior_map,
        background_posterior_map=first_e.background_posterior_map,
        confidence_mask=first_e.confidence_mask_result.confident_mask,
        initial_foreground_prototype=np.asarray(
            bundle.foreground_prototype.prototype_vector, dtype=np.float64
        ),
        initial_background_prototype=np.asarray(
            bundle.background_prototype.prototype_vector, dtype=np.float64
        ),
        previous_foreground_prototype=np.asarray(
            bundle.foreground_prototype.prototype_vector, dtype=np.float64
        ),
        previous_background_prototype=np.asarray(
            bundle.background_prototype.prototype_vector, dtype=np.float64
        ),
        objective_weights=config.objective_weights,
        foreground_prior=config.foreground_prior,
        proximal_foreground_reference=np.asarray(
            bundle.foreground_prototype.prototype_vector, dtype=np.float64
        ),
        proximal_background_reference=np.asarray(
            bundle.background_prototype.prototype_vector, dtype=np.float64
        ),
        previous_foreground_posterior_map=source.posterior_state.foreground_posterior_map,
        previous_background_posterior_map=source.posterior_state.background_posterior_map,
        support_foreground_reference=np.asarray(
            bundle.foreground_prototype.prototype_vector, dtype=np.float64
        ),
        support_background_reference=np.asarray(
            bundle.background_prototype.prototype_vector, dtype=np.float64
        ),
    )
    first_target, _ = build_next_protoem_transductive_state(
        source_state=source,
        e_step_result=first_e,
        m_step_result=first_m,
        objective_terms=first_m.objective_terms,
    )
    second_e = run_protoem_e_step(
        query_feature_tensor=np.asarray(
            bundle.query_feature_encoding.feature_data, dtype=np.float64
        ),
        foreground_prototypes=first_m.updated_foreground_prototype,
        background_prototypes=first_m.updated_background_prototype,
        temperature=config.temperature,
        confidence_threshold=0.0,
        foreground_prior=config.foreground_prior,
    )
    second_m = run_protoem_m_step(
        query_feature_tensor=np.asarray(
            bundle.query_feature_encoding.feature_data, dtype=np.float64
        ),
        foreground_posterior_map=second_e.foreground_posterior_map,
        background_posterior_map=second_e.background_posterior_map,
        confidence_mask=second_e.confidence_mask_result.confident_mask,
        initial_foreground_prototype=np.asarray(
            bundle.foreground_prototype.prototype_vector, dtype=np.float64
        ),
        initial_background_prototype=np.asarray(
            bundle.background_prototype.prototype_vector, dtype=np.float64
        ),
        previous_foreground_prototype=first_m.updated_foreground_prototype,
        previous_background_prototype=first_m.updated_background_prototype,
        objective_weights=config.objective_weights,
        foreground_prior=config.foreground_prior,
        proximal_foreground_reference=np.asarray(
            bundle.foreground_prototype.prototype_vector, dtype=np.float64
        ),
        proximal_background_reference=np.asarray(
            bundle.background_prototype.prototype_vector, dtype=np.float64
        ),
        previous_foreground_posterior_map=first_target.posterior_state.foreground_posterior_map,
        previous_background_posterior_map=first_target.posterior_state.background_posterior_map,
        support_foreground_reference=np.asarray(
            bundle.foreground_prototype.prototype_vector, dtype=np.float64
        ),
        support_background_reference=np.asarray(
            bundle.background_prototype.prototype_vector, dtype=np.float64
        ),
    )
    second_target, _ = build_next_protoem_transductive_state(
        source_state=first_target,
        e_step_result=second_e,
        m_step_result=second_m,
        objective_terms=second_m.objective_terms,
    )

    assert second_target.convergence_delta is not None
    assert second_target.convergence_delta == pytest.approx(
        abs(second_m.objective_terms.total - first_m.objective_terms.total),
        rel=0.0,
        abs=1e-12,
    )


def test_incompatible_source_result_rejection() -> None:
    bundle, config = _initialization_bundle()
    source = build_initial_protoem_transductive_state(
        initialization_bundle=bundle,
        config=config,
    )
    bad_e = run_protoem_e_step(
        query_feature_tensor=np.asarray(
            bundle.query_feature_encoding.feature_data, dtype=np.float64
        ),
        foreground_prototypes=np.asarray([0.0, 1.0], dtype=np.float64),
        background_prototypes=np.asarray([1.0, 0.0], dtype=np.float64),
        temperature=config.temperature,
        confidence_threshold=config.confidence_threshold,
        foreground_prior=config.foreground_prior,
    )
    good_m = run_protoem_m_step(
        query_feature_tensor=np.asarray(
            bundle.query_feature_encoding.feature_data, dtype=np.float64
        ),
        foreground_posterior_map=bad_e.foreground_posterior_map,
        background_posterior_map=bad_e.background_posterior_map,
        confidence_mask=bad_e.confidence_mask_result.confident_mask,
        initial_foreground_prototype=np.asarray(
            bundle.foreground_prototype.prototype_vector, dtype=np.float64
        ),
        initial_background_prototype=np.asarray(
            bundle.background_prototype.prototype_vector, dtype=np.float64
        ),
        previous_foreground_prototype=np.asarray(
            bundle.foreground_prototype.prototype_vector, dtype=np.float64
        ),
        previous_background_prototype=np.asarray(
            bundle.background_prototype.prototype_vector, dtype=np.float64
        ),
        objective_weights=config.objective_weights,
        foreground_prior=config.foreground_prior,
        proximal_foreground_reference=np.asarray(
            bundle.foreground_prototype.prototype_vector, dtype=np.float64
        ),
        proximal_background_reference=np.asarray(
            bundle.background_prototype.prototype_vector, dtype=np.float64
        ),
        previous_foreground_posterior_map=source.posterior_state.foreground_posterior_map,
        previous_background_posterior_map=source.posterior_state.background_posterior_map,
        support_foreground_reference=np.asarray(
            bundle.foreground_prototype.prototype_vector, dtype=np.float64
        ),
        support_background_reference=np.asarray(
            bundle.background_prototype.prototype_vector, dtype=np.float64
        ),
    )

    with pytest.raises(IncompatibleStateTransitionError):
        build_next_protoem_transductive_state(
            source_state=source,
            e_step_result=bad_e,
            m_step_result=good_m,
            objective_terms=good_m.objective_terms,
        )


def test_repeated_transition_is_deterministic() -> None:
    bundle, config = _initialization_bundle()
    source = build_initial_protoem_transductive_state(
        initialization_bundle=bundle,
        config=config,
    )
    e_step = run_protoem_e_step(
        query_feature_tensor=np.asarray(
            bundle.query_feature_encoding.feature_data, dtype=np.float64
        ),
        foreground_prototypes=np.asarray(
            bundle.foreground_prototype.prototype_vector, dtype=np.float64
        ),
        background_prototypes=np.asarray(
            bundle.background_prototype.prototype_vector, dtype=np.float64
        ),
        temperature=config.temperature,
        confidence_threshold=config.confidence_threshold,
        foreground_prior=config.foreground_prior,
    )
    m_step = run_protoem_m_step(
        query_feature_tensor=np.asarray(
            bundle.query_feature_encoding.feature_data, dtype=np.float64
        ),
        foreground_posterior_map=e_step.foreground_posterior_map,
        background_posterior_map=e_step.background_posterior_map,
        confidence_mask=e_step.confidence_mask_result.confident_mask,
        initial_foreground_prototype=np.asarray(
            bundle.foreground_prototype.prototype_vector, dtype=np.float64
        ),
        initial_background_prototype=np.asarray(
            bundle.background_prototype.prototype_vector, dtype=np.float64
        ),
        previous_foreground_prototype=np.asarray(
            bundle.foreground_prototype.prototype_vector, dtype=np.float64
        ),
        previous_background_prototype=np.asarray(
            bundle.background_prototype.prototype_vector, dtype=np.float64
        ),
        objective_weights=config.objective_weights,
        foreground_prior=config.foreground_prior,
        proximal_foreground_reference=np.asarray(
            bundle.foreground_prototype.prototype_vector, dtype=np.float64
        ),
        proximal_background_reference=np.asarray(
            bundle.background_prototype.prototype_vector, dtype=np.float64
        ),
        previous_foreground_posterior_map=source.posterior_state.foreground_posterior_map,
        previous_background_posterior_map=source.posterior_state.background_posterior_map,
        support_foreground_reference=np.asarray(
            bundle.foreground_prototype.prototype_vector, dtype=np.float64
        ),
        support_background_reference=np.asarray(
            bundle.background_prototype.prototype_vector, dtype=np.float64
        ),
    )
    first_state, first_transition = build_next_protoem_transductive_state(
        source_state=source,
        e_step_result=e_step,
        m_step_result=m_step,
        objective_terms=m_step.objective_terms,
    )
    second_state, second_transition = build_next_protoem_transductive_state(
        source_state=source,
        e_step_result=e_step,
        m_step_result=m_step,
        objective_terms=m_step.objective_terms,
    )

    assert first_state.state_identity_hash == second_state.state_identity_hash
    assert first_transition.transition_identity_hash == second_transition.transition_identity_hash
    assert protoem_transductive_state_to_dict(first_state) == protoem_transductive_state_to_dict(
        second_state
    )
    assert canonical_json_bytes(
        protoem_transductive_state_to_dict(first_state)
    ) == canonical_json_bytes(protoem_transductive_state_to_dict(second_state))
    assert protoem_transductive_state_to_json(first_state) == protoem_transductive_state_to_json(
        second_state
    )


def test_query_labels_reference_masks_absent_from_public_apis() -> None:
    for func in (
        build_initial_protoem_transductive_state,
        build_next_protoem_transductive_state,
        build_protoem_prototype_state,
        build_protoem_posterior_state,
        build_protoem_assignment_state,
    ):
        signature_text = str(inspect.signature(func)).lower()
        assert "query_label" not in signature_text
        assert "reference_mask" not in signature_text
    assert "query_label" not in ProtoEMTransductiveState.__dataclass_fields__
    assert "reference_mask" not in ProtoEMTransductiveState.__dataclass_fields__
