"""Unit tests for deterministic ProtoEM final inference and execution packaging."""

from __future__ import annotations

import hashlib
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
    FinalInferenceConstructionError,
    ProtoEMConfig,
    ProtoEMExecutionPackage,
    ProtoEMFinalInferenceResult,
    ProtoEMInferenceInput,
    ProtoEMInitializationBundle,
    ProtoEMObjectiveWeights,
    ProtoEMOptimizationResult,
    ProtoEMPhase5ArtifactReference,
    ProtoEMPhase5ReferenceBundle,
    ProtoEMPrototypeInput,
    ProtoEMQueryFeatureInput,
    ProtoEMSupportProvenance,
    build_protoem_ablation_definition,
    build_protoem_execution_package,
    build_protoem_final_inference_result,
    build_protoem_initialization_bundle,
    build_protoem_run_summary,
    protoem_final_inference_identity_payload,
    query_label_not_part_of_final_inference_api,
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
    return SupportPrototypeMemory(
        foreground_prototype=cast(
            ForegroundPrototype, _prototype(kind="foreground", vector=(1.0, 0.0))
        ),
        background_prototype=cast(
            BackgroundPrototype, _prototype(kind="background", vector=(0.0, 1.0))
        ),
    )


def _inference_result(
    *,
    foreground_score_map: np.ndarray | None = None,
    background_score_map: np.ndarray | None = None,
    prediction_mask: np.ndarray | None = None,
    memory: SupportPrototypeMemory | None = None,
) -> PrototypeInferenceResult:
    resolved_memory = memory or _prototype_memory()
    fg_scores = (
        foreground_score_map
        if foreground_score_map is not None
        else np.asarray([[[[[1.0, 0.0]]]]], dtype=np.float64)
    )
    bg_scores = (
        background_score_map
        if background_score_map is not None
        else np.asarray([[[[[0.0, 1.0]]]]], dtype=np.float64)
    )
    prediction = (
        prediction_mask
        if prediction_mask is not None
        else np.asarray([[[[[1, 0]]]]], dtype=np.uint8)
    )
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
        class_balance=0.1,
        consistency=0.2,
        proximal=0.3,
    )


def _config(
    *,
    confidence_threshold: float = 0.6,
    foreground_prior: float = 0.5,
    update_schedule: str = "fixed_em_like",
    prototype_mode: str = "single_prototype",
) -> ProtoEMConfig:
    from protoem_ct.artifacts.hashing import sha256_json

    weights = _weights()
    config_hash = sha256_json(
        {
            "schema_name": "protoem_config",
            "schema_version": "v1",
            "explicit_seed": 7,
            "max_iterations": 2,
            "minimum_iterations": 1,
            "convergence_tolerance": 0.0,
            "temperature": 1.0,
            "confidence_threshold": confidence_threshold,
            "foreground_prior": foreground_prior,
            "update_schedule": update_schedule,
            "prototype_mode": prototype_mode,
            "objective_weights": {
                "schema_name": weights.schema_name,
                "schema_version": weights.schema_version,
                "support": weights.support,
                "query_entropy": weights.query_entropy,
                "class_balance": weights.class_balance,
                "consistency": weights.consistency,
                "proximal": weights.proximal,
            },
            "phase5_initialization_schema_name": "prototype_inference_summary",
            "phase5_initialization_artifact_hash": "1" * 64,
            "phase5_prototype_schema_name": "prototype_memory",
            "phase5_prototype_artifact_hash": "2" * 64,
            "phase5_inference_schema_name": "phase5_run_summary",
            "phase5_inference_artifact_hash": "3" * 64,
        }
    )
    return ProtoEMConfig(
        schema_name="protoem_config",
        schema_version="v1",
        config_hash=config_hash,
        explicit_seed=7,
        max_iterations=2,
        minimum_iterations=1,
        convergence_tolerance=0.0,
        temperature=1.0,
        confidence_threshold=confidence_threshold,
        foreground_prior=foreground_prior,
        update_schedule=update_schedule,
        prototype_mode=prototype_mode,
        objective_weights=weights,
        phase5_initialization_schema_name="prototype_inference_summary",
        phase5_initialization_artifact_hash="1" * 64,
        phase5_prototype_schema_name="prototype_memory",
        phase5_prototype_artifact_hash="2" * 64,
        phase5_inference_schema_name="phase5_run_summary",
        phase5_inference_artifact_hash="3" * 64,
    )


def _phase5_references() -> ProtoEMPhase5ReferenceBundle:
    return ProtoEMPhase5ReferenceBundle(
        schema_name="protoem_phase5_reference_bundle",
        schema_version="v1",
        initialization_reference=ProtoEMPhase5ArtifactReference(
            schema_name="protoem_phase5_artifact_reference",
            schema_version="v1",
            artifact_schema_name="prototype_inference_summary",
            artifact_hash="1" * 64,
        ),
        prototype_reference=ProtoEMPhase5ArtifactReference(
            schema_name="protoem_phase5_artifact_reference",
            schema_version="v1",
            artifact_schema_name="prototype_memory",
            artifact_hash="2" * 64,
        ),
        inference_reference=ProtoEMPhase5ArtifactReference(
            schema_name="protoem_phase5_artifact_reference",
            schema_version="v1",
            artifact_schema_name="phase5_run_summary",
            artifact_hash="3" * 64,
        ),
    )


def _bundle(
    *,
    config: ProtoEMConfig,
    feature_values: np.ndarray | None = None,
    foreground_score_map: np.ndarray | None = None,
    background_score_map: np.ndarray | None = None,
    prediction_mask: np.ndarray | None = None,
) -> ProtoEMInitializationBundle:
    memory = _prototype_memory()
    return build_protoem_initialization_bundle(
        config=config,
        query_input=ProtoEMQueryFeatureInput(
            schema_name="protoem_query_feature_input",
            schema_version="v1",
            query_feature_encoding=_query_feature_encoding(feature_values),
            query_identity="query_case_001",
            query_patient_id="query_patient_001",
            query_case_id="query_case_001",
            dataset_manifest_hash=_sha256("d"),
        ),
        prototype_input=ProtoEMPrototypeInput(
            schema_name="protoem_prototype_input",
            schema_version="v1",
            prototype_memory=memory,
            support_provenance=(_support_provenance(),),
        ),
        inference_input=ProtoEMInferenceInput(
            schema_name="protoem_inference_input",
            schema_version="v1",
            inference_result=_inference_result(
                foreground_score_map=foreground_score_map,
                background_score_map=background_score_map,
                prediction_mask=prediction_mask,
                memory=memory,
            )
        ),
        phase5_references=_phase5_references(),
    )


def _successful_execution() -> tuple[
    ProtoEMConfig, ProtoEMInitializationBundle, ProtoEMOptimizationResult
]:
    config = _config()
    bundle = _bundle(config=config)
    result = run_protoem_optimization(config=config, initialization_bundle=bundle)
    assert result.execution_status == "completed"
    return config, bundle, result


def _build_manual_final_inference(
    *,
    result: ProtoEMOptimizationResult,
    foreground_posterior_map: np.ndarray,
    background_posterior_map: np.ndarray,
    confidence_map: np.ndarray,
    prediction_map: np.ndarray,
    final_foreground_prototypes: np.ndarray,
    final_background_prototypes: np.ndarray,
) -> ProtoEMFinalInferenceResult:
    completed_iteration_count = len(result.iteration_executions)
    stopping_reason = result.stopping_decision.stop_reason or "max_iterations"
    payload = {
        "schema_name": "protoem_final_inference_result",
        "schema_version": "v1",
        "config_hash": result.config_hash,
        "initialization_identity_hash": result.initialization_identity_hash,
        "optimization_result_identity_hash": result.result_identity_hash,
        "final_state_identity_hash": result.final_state.state_identity_hash,
        "foreground_posterior_content_hash": hashlib.sha256(
            np.ascontiguousarray(foreground_posterior_map).tobytes(order="C")
        ).hexdigest(),
        "background_posterior_content_hash": hashlib.sha256(
            np.ascontiguousarray(background_posterior_map).tobytes(order="C")
        ).hexdigest(),
        "confidence_content_hash": hashlib.sha256(
            np.ascontiguousarray(confidence_map).tobytes(order="C")
        ).hexdigest(),
        "prediction_content_hash": hashlib.sha256(
            np.ascontiguousarray(prediction_map).tobytes(order="C")
        ).hexdigest(),
        "final_foreground_prototype_content_hash": hashlib.sha256(
            np.ascontiguousarray(final_foreground_prototypes).tobytes(order="C")
        ).hexdigest(),
        "final_background_prototype_content_hash": hashlib.sha256(
            np.ascontiguousarray(final_background_prototypes).tobytes(order="C")
        ).hexdigest(),
        "stopping_reason": stopping_reason,
        "completed_iteration_count": completed_iteration_count,
    }
    foreground_posterior_content_hash = str(payload["foreground_posterior_content_hash"])
    background_posterior_content_hash = str(payload["background_posterior_content_hash"])
    confidence_content_hash = str(payload["confidence_content_hash"])
    prediction_content_hash = str(payload["prediction_content_hash"])
    final_foreground_prototype_content_hash = str(
        payload["final_foreground_prototype_content_hash"]
    )
    final_background_prototype_content_hash = str(
        payload["final_background_prototype_content_hash"]
    )
    return ProtoEMFinalInferenceResult(
        schema_name="protoem_final_inference_result",
        schema_version="v1",
        inference_identity_hash=sha256_json(payload),
        config_hash=result.config_hash,
        initialization_identity_hash=result.initialization_identity_hash,
        optimization_result_identity_hash=result.result_identity_hash,
        final_state_identity_hash=result.final_state.state_identity_hash,
        foreground_posterior_map=foreground_posterior_map,
        background_posterior_map=background_posterior_map,
        confidence_map=confidence_map,
        prediction_map=prediction_map,
        final_foreground_prototypes=final_foreground_prototypes,
        final_background_prototypes=final_background_prototypes,
        foreground_posterior_content_hash=foreground_posterior_content_hash,
        background_posterior_content_hash=background_posterior_content_hash,
        confidence_content_hash=confidence_content_hash,
        prediction_content_hash=prediction_content_hash,
        final_foreground_prototype_content_hash=final_foreground_prototype_content_hash,
        final_background_prototype_content_hash=final_background_prototype_content_hash,
        stopping_reason=stopping_reason,
        completed_iteration_count=completed_iteration_count,
    )


def test_valid_final_inference_from_successful_optimization() -> None:
    config, bundle, result = _successful_execution()

    final_inference = build_protoem_final_inference_result(optimization_result=result)

    assert final_inference.config_hash == config.config_hash
    assert final_inference.initialization_identity_hash == bundle.initialization_identity_hash
    assert final_inference.optimization_result_identity_hash == result.result_identity_hash
    assert final_inference.final_state_identity_hash == result.final_state.state_identity_hash
    assert final_inference.completed_iteration_count == len(result.iteration_executions)


def test_posterior_prediction_consistency() -> None:
    _, _, result = _successful_execution()

    final_inference = build_protoem_final_inference_result(optimization_result=result)

    expected_prediction = (
        final_inference.foreground_posterior_map > final_inference.background_posterior_map
    ).astype(np.uint8)
    assert np.array_equal(final_inference.prediction_map, expected_prediction)
    assert np.allclose(
        final_inference.confidence_map,
        np.maximum(
            final_inference.foreground_posterior_map,
            final_inference.background_posterior_map,
        ),
        atol=1e-12,
        rtol=0.0,
    )


def test_tie_resolves_to_background() -> None:
    _, _, result = _successful_execution()
    foreground = np.asarray([[[[[0.5, 0.5]]]]], dtype=np.float64)
    background = np.asarray([[[[[0.5, 0.5]]]]], dtype=np.float64)
    prediction = np.zeros((1, 1, 1, 1, 2), dtype=np.uint8)
    confidence = np.asarray([[[[[0.5, 0.5]]]]], dtype=np.float64)
    final_prototypes = result.final_state.prototype_state
    final_inference = _build_manual_final_inference(
        result=result,
        foreground_posterior_map=foreground,
        background_posterior_map=background,
        confidence_map=confidence,
        prediction_map=prediction,
        final_foreground_prototypes=final_prototypes.foreground_prototypes,
        final_background_prototypes=final_prototypes.background_prototypes,
    )

    assert np.count_nonzero(final_inference.prediction_map) == 0


def test_failed_optimization_does_not_fabricate_inference() -> None:
    config = _config(confidence_threshold=0.95, foreground_prior=0.5)
    zero_features = np.zeros((1, 2, 1, 1, 2), dtype=np.float32)
    zero_scores = np.zeros((1, 1, 1, 1, 2), dtype=np.float64)
    bundle = _bundle(
        config=config,
        feature_values=zero_features,
        foreground_score_map=zero_scores,
        background_score_map=zero_scores,
        prediction_mask=np.zeros((1, 1, 1, 1, 2), dtype=np.uint8),
    )
    result = run_protoem_optimization(config=config, initialization_bundle=bundle)

    assert result.execution_status == "failed"
    with pytest.raises(FinalInferenceConstructionError):
        build_protoem_final_inference_result(optimization_result=result)


def test_valid_success_execution_package() -> None:
    config, bundle, result = _successful_execution()
    definition = build_protoem_ablation_definition(variant_name="full_protoem")

    package = build_protoem_execution_package(
        config=config,
        ablation_definition=definition,
        initialization_bundle=bundle,
        optimization_result=result,
    )

    assert isinstance(package, ProtoEMExecutionPackage)
    assert package.run_summary is not None
    assert package.final_inference_result is not None
    assert package.run_summary.final_inference_artifact_hash == (
        package.final_inference_result.inference_identity_hash
    )
    assert package.run_summary.final_output_artifact_hash == result.final_state.state_identity_hash


def test_valid_failed_collapse_execution_package() -> None:
    config = _config(confidence_threshold=0.95, foreground_prior=0.5)
    zero_features = np.zeros((1, 2, 1, 1, 2), dtype=np.float32)
    zero_scores = np.zeros((1, 1, 1, 1, 2), dtype=np.float64)
    bundle = _bundle(
        config=config,
        feature_values=zero_features,
        foreground_score_map=zero_scores,
        background_score_map=zero_scores,
        prediction_mask=np.zeros((1, 1, 1, 1, 2), dtype=np.uint8),
    )
    result = run_protoem_optimization(config=config, initialization_bundle=bundle)
    definition = build_protoem_ablation_definition(variant_name="full_protoem")

    package = build_protoem_execution_package(
        config=config,
        ablation_definition=definition,
        initialization_bundle=bundle,
        optimization_result=result,
    )

    assert package.final_inference_result is None
    assert package.run_summary is None
    assert result.stopping_record is None


def test_run_summary_hash_consistency() -> None:
    config, bundle, result = _successful_execution()
    definition = build_protoem_ablation_definition(variant_name="full_protoem")
    final_inference = build_protoem_final_inference_result(optimization_result=result)

    summary = build_protoem_run_summary(
        config=config,
        ablation_definition=definition,
        initialization_bundle=bundle,
        optimization_result=result,
        final_inference_result=final_inference,
    )

    assert summary.run_summary_hash == summary.run_summary_hash
    assert summary.final_inference_artifact_hash == final_inference.inference_identity_hash


def test_deterministic_identities_and_non_contiguous_arrays() -> None:
    _, _, result = _successful_execution()
    first = build_protoem_final_inference_result(optimization_result=result)
    second = build_protoem_final_inference_result(optimization_result=result)

    assert first.inference_identity_hash == second.inference_identity_hash
    assert protoem_final_inference_identity_payload(
        first
    ) == protoem_final_inference_identity_payload(second)

    foreground = np.asarray([[[[[0.8, 0.2, 0.8, 0.2]]]]], dtype=np.float64)
    background = np.asarray([[[[[0.2, 0.8, 0.2, 0.8]]]]], dtype=np.float64)
    confidence = np.maximum(foreground, background)
    prediction = np.asarray([[[[[1, 0, 1, 0]]]]], dtype=np.uint8)
    bank = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    contiguous = _build_manual_final_inference(
        result=result,
        foreground_posterior_map=foreground[..., ::2],
        background_posterior_map=background[..., ::2],
        confidence_map=confidence[..., ::2],
        prediction_map=prediction[..., ::2],
        final_foreground_prototypes=bank[::2],
        final_background_prototypes=bank[::2],
    )
    transposed = _build_manual_final_inference(
        result=result,
        foreground_posterior_map=np.asfortranarray(contiguous.foreground_posterior_map),
        background_posterior_map=np.asfortranarray(contiguous.background_posterior_map),
        confidence_map=np.asfortranarray(contiguous.confidence_map),
        prediction_map=np.asfortranarray(contiguous.prediction_map),
        final_foreground_prototypes=np.asfortranarray(contiguous.final_foreground_prototypes),
        final_background_prototypes=np.asfortranarray(contiguous.final_background_prototypes),
    )

    assert contiguous.inference_identity_hash == transposed.inference_identity_hash


def test_input_arrays_remain_unchanged() -> None:
    config, bundle, result = _successful_execution()
    original_foreground = result.final_state.posterior_state.foreground_posterior_map.copy()

    _ = build_protoem_execution_package(
        config=config,
        ablation_definition=build_protoem_ablation_definition(variant_name="full_protoem"),
        initialization_bundle=bundle,
        optimization_result=result,
    )

    assert np.array_equal(
        result.final_state.posterior_state.foreground_posterior_map,
        original_foreground,
    )


def test_query_labels_reference_masks_absent_from_public_apis() -> None:
    assert query_label_not_part_of_final_inference_api() is True
    for callable_object in (
        build_protoem_final_inference_result,
        build_protoem_execution_package,
        build_protoem_run_summary,
    ):
        signature_text = str(inspect.signature(callable_object))
        assert "query_label" not in signature_text
        assert "reference_mask" not in signature_text
    assert "query_label" not in ProtoEMFinalInferenceResult.__dataclass_fields__
    assert "query_label" not in ProtoEMExecutionPackage.__dataclass_fields__
