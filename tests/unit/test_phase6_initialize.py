"""Unit tests for deterministic Phase 5 to Phase 6 ProtoEM initialization."""

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
    PROTOEM_INFERENCE_INPUT_SCHEMA_NAME,
    PROTOEM_INFERENCE_INPUT_SCHEMA_VERSION,
    PROTOEM_INITIALIZATION_BUNDLE_SCHEMA_NAME,
    PROTOEM_INITIALIZATION_BUNDLE_SCHEMA_VERSION,
    PROTOEM_PHASE5_ARTIFACT_REFERENCE_SCHEMA_NAME,
    PROTOEM_PHASE5_ARTIFACT_REFERENCE_SCHEMA_VERSION,
    PROTOEM_PHASE5_REFERENCE_BUNDLE_SCHEMA_NAME,
    PROTOEM_PHASE5_REFERENCE_BUNDLE_SCHEMA_VERSION,
    PROTOEM_PROTOTYPE_INPUT_SCHEMA_NAME,
    PROTOEM_PROTOTYPE_INPUT_SCHEMA_VERSION,
    PROTOEM_QUERY_FEATURE_INPUT_SCHEMA_NAME,
    PROTOEM_QUERY_FEATURE_INPUT_SCHEMA_VERSION,
    PROTOEM_SUPPORT_PROVENANCE_SCHEMA_NAME,
    PROTOEM_SUPPORT_PROVENANCE_SCHEMA_VERSION,
    IncompatibleInitializationError,
    InitializationLeakageViolationError,
    InvalidInitialPredictionError,
    InvalidInitialScoreStateError,
    InvalidQueryFeatureError,
    MalformedPhase5ArtifactError,
    ProtoEMConfig,
    ProtoEMInferenceInput,
    ProtoEMInitializationBundle,
    ProtoEMObjectiveWeights,
    ProtoEMPhase5ArtifactReference,
    ProtoEMPhase5ReferenceBundle,
    ProtoEMPrototypeInput,
    ProtoEMQueryFeatureInput,
    ProtoEMSupportProvenance,
    build_protoem_initialization_bundle,
    hash_protoem_initialization_bundle,
    protoem_initialization_bundle_to_dict,
    protoem_initialization_bundle_to_json,
    protoem_initialization_identity_payload,
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
    encoder_identity: str = "segresnet_encoder_v1",
    preprocessing_hash: str = _sha256("a"),
    checkpoint_hash: str = _sha256("b"),
    feature_stage: str = "final_encoder",
    normalization_name: str = "l2_channel",
) -> EmbeddingMetadata:
    return EmbeddingMetadata(
        encoder_identity=encoder_identity,
        preprocessing_hash=preprocessing_hash,
        checkpoint_hash=checkpoint_hash,
        input_identity=input_identity,
        feature_stage=feature_stage,
        normalization_name=normalization_name,
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
    encoder_identity: str = "segresnet_encoder_v1",
    preprocessing_hash: str = _sha256("a"),
    checkpoint_hash: str = _sha256("b"),
    feature_stage: str = "final_encoder",
    normalization_name: str = "l2_channel",
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
            encoder_identity=encoder_identity,
            preprocessing_hash=preprocessing_hash,
            checkpoint_hash=checkpoint_hash,
            feature_stage=feature_stage,
            normalization_name=normalization_name,
        ),
    )


def _prototype(
    *,
    kind: str,
    vector: tuple[float, ...],
    support_identifiers: tuple[str, ...],
    patient_ids: tuple[str, ...],
    case_ids: tuple[str, ...],
    dataset_manifest_hash: str | None = _sha256("d"),
    encoder_identity: str = "segresnet_encoder_v1",
    preprocessing_hash: str = _sha256("a"),
    checkpoint_hash: str = _sha256("b"),
    feature_stage: str = "final_encoder",
    normalization_name: str = "l2_channel",
) -> ForegroundPrototype | BackgroundPrototype:
    prototype_class = ForegroundPrototype if kind == "foreground" else BackgroundPrototype
    content_hash = hashlib.sha256(
        np.asarray(vector, dtype=np.float64).tobytes(order="C")
    ).hexdigest()
    draft = prototype_class(
        prototype_kind="ignored",
        feature_channels=len(vector),
        contributing_voxel_count=len(support_identifiers),
        source_support_identifiers=support_identifiers,
        normalization_name=normalization_name,
        prototype_vector=vector,
        encoder_identity=encoder_identity,
        preprocessing_hash=preprocessing_hash,
        checkpoint_hash=checkpoint_hash,
        dataset_manifest_hash=dataset_manifest_hash,
        feature_stage=feature_stage,
        source_patient_ids=patient_ids,
        source_case_ids=case_ids,
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


def _foreground_prototype(
    *,
    vector: tuple[float, ...],
    support_identifier: str,
    patient_id: str,
    case_id: str,
    dataset_manifest_hash: str | None = _sha256("d"),
) -> ForegroundPrototype:
    return cast(
        ForegroundPrototype,
        _prototype(
            kind="foreground",
            vector=vector,
            support_identifiers=(support_identifier,),
            patient_ids=(patient_id,),
            case_ids=(case_id,),
            dataset_manifest_hash=dataset_manifest_hash,
        ),
    )


def _background_prototype(
    *,
    vector: tuple[float, ...],
    support_identifier: str,
    patient_id: str,
    case_id: str,
    dataset_manifest_hash: str | None = _sha256("d"),
) -> BackgroundPrototype:
    return cast(
        BackgroundPrototype,
        _prototype(
            kind="background",
            vector=vector,
            support_identifiers=(support_identifier,),
            patient_ids=(patient_id,),
            case_ids=(case_id,),
            dataset_manifest_hash=dataset_manifest_hash,
        ),
    )


def _prototype_memory(
    *,
    support_identifier: str = "support_case_001",
    patient_id: str = "support_patient_001",
    case_id: str = "support_case_001",
    dataset_manifest_hash: str | None = _sha256("d"),
) -> SupportPrototypeMemory:
    return SupportPrototypeMemory(
        foreground_prototype=_foreground_prototype(
            vector=(1.0, 0.0),
            support_identifier=support_identifier,
            patient_id=patient_id,
            case_id=case_id,
            dataset_manifest_hash=dataset_manifest_hash,
        ),
        background_prototype=_background_prototype(
            vector=(0.0, 1.0),
            support_identifier=support_identifier,
            patient_id=patient_id,
            case_id=case_id,
            dataset_manifest_hash=dataset_manifest_hash,
        ),
    )


def _prototype_memory_two_supports() -> SupportPrototypeMemory:
    return SupportPrototypeMemory(
        foreground_prototype=cast(
            ForegroundPrototype,
            _prototype(
                kind="foreground",
                vector=(1.0, 0.0),
                support_identifiers=("support_case_001", "support_case_002"),
                patient_ids=("support_patient_001", "support_patient_002"),
                case_ids=("support_case_001", "support_case_002"),
            ),
        ),
        background_prototype=cast(
            BackgroundPrototype,
            _prototype(
                kind="background",
                vector=(0.0, 1.0),
                support_identifiers=("support_case_001", "support_case_002"),
                patient_ids=("support_patient_001", "support_patient_002"),
                case_ids=("support_case_001", "support_case_002"),
            ),
        ),
    )


def _inference_result(
    *,
    query_identity: str = "query_case_001",
    query_patient_id: str = "query_patient_001",
    query_case_id: str = "query_case_001",
    dataset_manifest_hash: str | None = _sha256("d"),
    foreground: ForegroundPrototype | None = None,
    background: BackgroundPrototype | None = None,
    feature_shape: tuple[int, int, int, int, int] = (1, 2, 1, 1, 2),
    foreground_score_map: np.ndarray | None = None,
    background_score_map: np.ndarray | None = None,
    prediction_mask: np.ndarray | None = None,
) -> PrototypeInferenceResult:
    resolved_foreground = foreground or _prototype_memory().foreground_prototype
    resolved_background = background or _prototype_memory().background_prototype
    fg_scores = foreground_score_map
    if fg_scores is None:
        fg_scores = np.asarray([[[[[0.9, 0.1]]]]], dtype=np.float64)
    bg_scores = background_score_map
    if bg_scores is None:
        bg_scores = np.asarray([[[[[0.1, 0.9]]]]], dtype=np.float64)
    prediction = prediction_mask
    if prediction is None:
        prediction = np.asarray([[[[[1, 0]]]]], dtype=np.uint8)
    confidence = np.abs(fg_scores - bg_scores)
    foreground_hash = hashlib.sha256(np.ascontiguousarray(fg_scores).tobytes(order="C")).hexdigest()
    background_hash = hashlib.sha256(np.ascontiguousarray(bg_scores).tobytes(order="C")).hexdigest()
    prediction_hash = hashlib.sha256(
        np.ascontiguousarray(prediction).tobytes(order="C")
    ).hexdigest()
    confidence_hash = hashlib.sha256(
        np.ascontiguousarray(confidence).tobytes(order="C")
    ).hexdigest()
    draft = PrototypeInferenceResult(
        schema_name=PROTOTYPE_INFERENCE_SCHEMA_NAME,
        schema_version=PROTOTYPE_INFERENCE_SCHEMA_VERSION,
        query_identity=query_identity,
        query_patient_id=query_patient_id,
        query_case_id=query_case_id,
        encoder_identity="segresnet_encoder_v1",
        preprocessing_hash=_sha256("a"),
        checkpoint_hash=_sha256("b"),
        dataset_manifest_hash=dataset_manifest_hash,
        feature_stage="final_encoder",
        normalization_name="l2_channel",
        feature_shape=feature_shape,
        foreground_prototype_identity=resolved_foreground.prototype_identity_sha256,
        background_prototype_identity=resolved_background.prototype_identity_sha256,
        tie_policy="background_on_exact_tie",
        scoring_rule="per_voxel_dual_cosine_similarity",
        foreground_score_map=np.ascontiguousarray(fg_scores),
        background_score_map=np.ascontiguousarray(bg_scores),
        prediction_mask=np.ascontiguousarray(prediction),
        confidence_margin_map=np.ascontiguousarray(confidence),
        foreground_score_content_sha256=foreground_hash,
        background_score_content_sha256=background_hash,
        prediction_content_sha256=prediction_hash,
        confidence_margin_content_sha256=confidence_hash,
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


def _support_provenance(
    *,
    support_identifier: str = "support_case_001",
    patient_id: str = "support_patient_001",
    case_id: str = "support_case_001",
    source_input_identity: str | None = None,
    support_assignment_index: int = 0,
    dataset_manifest_hash: str | None = _sha256("d"),
    encoder_identity: str = "segresnet_encoder_v1",
    preprocessing_hash: str = _sha256("a"),
    checkpoint_hash: str = _sha256("b"),
    feature_stage: str = "final_encoder",
    normalization_name: str = "l2_channel",
) -> ProtoEMSupportProvenance:
    return ProtoEMSupportProvenance(
        schema_name=PROTOEM_SUPPORT_PROVENANCE_SCHEMA_NAME,
        schema_version=PROTOEM_SUPPORT_PROVENANCE_SCHEMA_VERSION,
        support_identifier=support_identifier,
        support_patient_id=patient_id,
        support_case_id=case_id,
        source_input_identity=source_input_identity or support_identifier,
        support_manifest_hash=_sha256("c"),
        support_assignment_index=support_assignment_index,
        dataset_manifest_hash=dataset_manifest_hash,
        encoder_identity=encoder_identity,
        preprocessing_hash=preprocessing_hash,
        checkpoint_hash=checkpoint_hash,
        feature_stage=feature_stage,
        normalization_name=normalization_name,
    )


def _config() -> ProtoEMConfig:
    weights = ProtoEMObjectiveWeights(
        schema_name="protoem_objective_weights",
        schema_version="v1",
        support=1.0,
        query_entropy=0.1,
        class_balance=0.1,
        consistency=0.1,
        proximal=0.1,
    )
    payload = {
        "schema_name": "protoem_config",
        "schema_version": "v1",
        "explicit_seed": 1729,
        "max_iterations": 5,
        "minimum_iterations": 1,
        "convergence_tolerance": 0.001,
        "temperature": 1.0,
        "confidence_threshold": 0.9,
        "foreground_prior": 0.2,
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
        confidence_threshold=0.9,
        foreground_prior=0.2,
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
        schema_name=PROTOEM_PHASE5_REFERENCE_BUNDLE_SCHEMA_NAME,
        schema_version=PROTOEM_PHASE5_REFERENCE_BUNDLE_SCHEMA_VERSION,
        initialization_reference=ProtoEMPhase5ArtifactReference(
            schema_name=PROTOEM_PHASE5_ARTIFACT_REFERENCE_SCHEMA_NAME,
            schema_version=PROTOEM_PHASE5_ARTIFACT_REFERENCE_SCHEMA_VERSION,
            artifact_schema_name="phase5_run_summary",
            artifact_hash=_sha256("1"),
        ),
        prototype_reference=ProtoEMPhase5ArtifactReference(
            schema_name=PROTOEM_PHASE5_ARTIFACT_REFERENCE_SCHEMA_NAME,
            schema_version=PROTOEM_PHASE5_ARTIFACT_REFERENCE_SCHEMA_VERSION,
            artifact_schema_name="prototype_memory",
            artifact_hash=_sha256("2"),
        ),
        inference_reference=ProtoEMPhase5ArtifactReference(
            schema_name=PROTOEM_PHASE5_ARTIFACT_REFERENCE_SCHEMA_NAME,
            schema_version=PROTOEM_PHASE5_ARTIFACT_REFERENCE_SCHEMA_VERSION,
            artifact_schema_name="prototype_only_inference_result",
            artifact_hash=_sha256("3"),
        ),
    )


def _inputs() -> tuple[
    ProtoEMConfig,
    ProtoEMQueryFeatureInput,
    ProtoEMPrototypeInput,
    ProtoEMInferenceInput,
    ProtoEMPhase5ReferenceBundle,
]:
    config = _config()
    query = ProtoEMQueryFeatureInput(
        schema_name=PROTOEM_QUERY_FEATURE_INPUT_SCHEMA_NAME,
        schema_version=PROTOEM_QUERY_FEATURE_INPUT_SCHEMA_VERSION,
        query_feature_encoding=_query_feature_encoding(),
        query_identity="query_case_001",
        query_patient_id="query_patient_001",
        query_case_id="query_case_001",
        dataset_manifest_hash=_sha256("d"),
    )
    memory = _prototype_memory()
    prototype = ProtoEMPrototypeInput(
        schema_name=PROTOEM_PROTOTYPE_INPUT_SCHEMA_NAME,
        schema_version=PROTOEM_PROTOTYPE_INPUT_SCHEMA_VERSION,
        prototype_memory=memory,
        support_provenance=(
            _support_provenance(
                support_identifier="support_case_001",
                patient_id="support_patient_001",
                case_id="support_case_001",
            ),
        ),
    )
    inference = ProtoEMInferenceInput(
        schema_name=PROTOEM_INFERENCE_INPUT_SCHEMA_NAME,
        schema_version=PROTOEM_INFERENCE_INPUT_SCHEMA_VERSION,
        inference_result=_inference_result(
            foreground=memory.foreground_prototype,
            background=memory.background_prototype,
        ),
    )
    return config, query, prototype, inference, _references()


def test_successful_initialization_from_valid_phase5_objects() -> None:
    config, query, prototype, inference, references = _inputs()
    bundle = build_protoem_initialization_bundle(
        config=config,
        query_input=query,
        prototype_input=prototype,
        inference_input=inference,
        phase5_references=references,
    )

    assert isinstance(bundle, ProtoEMInitializationBundle)
    assert bundle.schema_name == PROTOEM_INITIALIZATION_BUNDLE_SCHEMA_NAME
    assert bundle.schema_version == PROTOEM_INITIALIZATION_BUNDLE_SCHEMA_VERSION
    assert bundle.contains_cosine_scores is True
    assert bundle.contains_posterior_probabilities is False
    assert bundle.contains_hard_assignments is True


def test_exact_shape_and_dtype_validation() -> None:
    config, query, prototype, inference, references = _inputs()
    bundle = build_protoem_initialization_bundle(
        config=config,
        query_input=query,
        prototype_input=prototype,
        inference_input=inference,
        phase5_references=references,
    )

    assert np.asarray(bundle.query_feature_encoding.feature_data).shape == (1, 2, 1, 1, 2)
    assert bundle.initial_foreground_score_map.shape == (1, 1, 1, 1, 2)
    assert bundle.initial_background_score_map.shape == (1, 1, 1, 1, 2)
    assert bundle.initial_prediction_mask.shape == (1, 1, 1, 1, 2)
    assert bundle.initial_foreground_score_map.dtype == np.float64
    assert bundle.initial_background_score_map.dtype == np.float64
    assert bundle.initial_prediction_mask.dtype == np.uint8


def test_deterministic_identity() -> None:
    config, query, prototype, inference, references = _inputs()
    first = build_protoem_initialization_bundle(
        config=config,
        query_input=query,
        prototype_input=prototype,
        inference_input=inference,
        phase5_references=references,
    )
    second = build_protoem_initialization_bundle(
        config=config,
        query_input=query,
        prototype_input=prototype,
        inference_input=inference,
        phase5_references=references,
    )

    assert first.initialization_identity_hash == second.initialization_identity_hash
    assert hash_protoem_initialization_bundle(first) == hash_protoem_initialization_bundle(second)
    first_payload = protoem_initialization_identity_payload(first)
    second_payload = protoem_initialization_identity_payload(second)
    assert first_payload == second_payload


def test_support_input_order_does_not_alter_identity() -> None:
    config, query, _, _, references = _inputs()
    combined_memory = _prototype_memory_two_supports()
    inference_result = _inference_result(
        foreground=combined_memory.foreground_prototype,
        background=combined_memory.background_prototype,
    )
    provenance_a = _support_provenance(
        support_identifier="support_case_001",
        patient_id="support_patient_001",
        case_id="support_case_001",
        support_assignment_index=0,
    )
    provenance_b = _support_provenance(
        support_identifier="support_case_002",
        patient_id="support_patient_002",
        case_id="support_case_002",
        support_assignment_index=1,
    )
    first = build_protoem_initialization_bundle(
        config=config,
        query_input=query,
        prototype_input=ProtoEMPrototypeInput(
            schema_name=PROTOEM_PROTOTYPE_INPUT_SCHEMA_NAME,
            schema_version=PROTOEM_PROTOTYPE_INPUT_SCHEMA_VERSION,
            prototype_memory=combined_memory,
            support_provenance=(provenance_a, provenance_b),
        ),
        inference_input=ProtoEMInferenceInput(
            schema_name=PROTOEM_INFERENCE_INPUT_SCHEMA_NAME,
            schema_version=PROTOEM_INFERENCE_INPUT_SCHEMA_VERSION,
            inference_result=inference_result,
        ),
        phase5_references=references,
    )
    second = build_protoem_initialization_bundle(
        config=config,
        query_input=query,
        prototype_input=ProtoEMPrototypeInput(
            schema_name=PROTOEM_PROTOTYPE_INPUT_SCHEMA_NAME,
            schema_version=PROTOEM_PROTOTYPE_INPUT_SCHEMA_VERSION,
            prototype_memory=combined_memory,
            support_provenance=(provenance_b, provenance_a),
        ),
        inference_input=ProtoEMInferenceInput(
            schema_name=PROTOEM_INFERENCE_INPUT_SCHEMA_NAME,
            schema_version=PROTOEM_INFERENCE_INPUT_SCHEMA_VERSION,
            inference_result=inference_result,
        ),
        phase5_references=references,
    )

    assert first.initialization_identity_hash == second.initialization_identity_hash


def test_malformed_phase5_artifact_hash_rejection() -> None:
    config, query, prototype, inference, references = _inputs()
    object.__setattr__(inference.inference_result, "inference_identity_sha256", "f" * 64)

    with pytest.raises(MalformedPhase5ArtifactError):
        build_protoem_initialization_bundle(
            config=config,
            query_input=query,
            prototype_input=prototype,
            inference_input=inference,
            phase5_references=references,
        )


def test_query_prototype_metadata_mismatch_rejection() -> None:
    config, query, prototype, inference, references = _inputs()
    mismatch_memory = _prototype_memory()
    object.__setattr__(mismatch_memory.foreground_prototype, "encoder_identity", "other_encoder")
    object.__setattr__(
        mismatch_memory.foreground_prototype,
        "prototype_identity_sha256",
        hash_support_prototype_identity(mismatch_memory.foreground_prototype),
    )

    with pytest.raises(IncompatibleInitializationError):
        build_protoem_initialization_bundle(
            config=config,
            query_input=query,
            prototype_input=ProtoEMPrototypeInput(
                schema_name=PROTOEM_PROTOTYPE_INPUT_SCHEMA_NAME,
                schema_version=PROTOEM_PROTOTYPE_INPUT_SCHEMA_VERSION,
                prototype_memory=mismatch_memory,
                support_provenance=prototype.support_provenance,
            ),
            inference_input=inference,
            phase5_references=references,
        )


def test_patient_leakage_rejection() -> None:
    config, query, prototype, inference, references = _inputs()
    leaking = _support_provenance(patient_id="query_patient_001", case_id="support_case_001")

    with pytest.raises(InitializationLeakageViolationError):
        build_protoem_initialization_bundle(
            config=config,
            query_input=query,
            prototype_input=ProtoEMPrototypeInput(
                schema_name=PROTOEM_PROTOTYPE_INPUT_SCHEMA_NAME,
                schema_version=PROTOEM_PROTOTYPE_INPUT_SCHEMA_VERSION,
                prototype_memory=prototype.prototype_memory,
                support_provenance=(leaking,),
            ),
            inference_input=inference,
            phase5_references=references,
        )


def test_case_leakage_rejection() -> None:
    config, query, prototype, inference, references = _inputs()
    leaking = _support_provenance(patient_id="support_patient_001", case_id="query_case_001")

    with pytest.raises(InitializationLeakageViolationError):
        build_protoem_initialization_bundle(
            config=config,
            query_input=query,
            prototype_input=ProtoEMPrototypeInput(
                schema_name=PROTOEM_PROTOTYPE_INPUT_SCHEMA_NAME,
                schema_version=PROTOEM_PROTOTYPE_INPUT_SCHEMA_VERSION,
                prototype_memory=prototype.prototype_memory,
                support_provenance=(leaking,),
            ),
            inference_input=inference,
            phase5_references=references,
        )


def test_query_identity_in_prototype_provenance_rejection() -> None:
    config, query, _, _, references = _inputs()
    bad_memory = _prototype_memory(
        support_identifier="query_case_001",
        patient_id="support_patient_001",
        case_id="support_case_001",
    )
    bad_inference = ProtoEMInferenceInput(
        schema_name=PROTOEM_INFERENCE_INPUT_SCHEMA_NAME,
        schema_version=PROTOEM_INFERENCE_INPUT_SCHEMA_VERSION,
        inference_result=_inference_result(
            foreground=bad_memory.foreground_prototype,
            background=bad_memory.background_prototype,
        ),
    )
    bad_prototype = ProtoEMPrototypeInput(
        schema_name=PROTOEM_PROTOTYPE_INPUT_SCHEMA_NAME,
        schema_version=PROTOEM_PROTOTYPE_INPUT_SCHEMA_VERSION,
        prototype_memory=bad_memory,
        support_provenance=(
            _support_provenance(
                support_identifier="query_case_001",
                patient_id="support_patient_001",
                case_id="support_case_001",
            ),
        ),
    )

    with pytest.raises(InitializationLeakageViolationError):
        build_protoem_initialization_bundle(
            config=config,
            query_input=query,
            prototype_input=bad_prototype,
            inference_input=bad_inference,
            phase5_references=references,
        )


def test_non_finite_feature_rejection() -> None:
    config, _, prototype, inference, references = _inputs()
    query = ProtoEMQueryFeatureInput(
        schema_name=PROTOEM_QUERY_FEATURE_INPUT_SCHEMA_NAME,
        schema_version=PROTOEM_QUERY_FEATURE_INPUT_SCHEMA_VERSION,
        query_feature_encoding=_query_feature_encoding(
            feature_values=np.asarray([[[[[np.nan, 0.0]]], [[[0.0, 1.0]]]]], dtype=np.float32)
        ),
        query_identity="query_case_001",
        query_patient_id="query_patient_001",
        query_case_id="query_case_001",
        dataset_manifest_hash=_sha256("d"),
    )

    with pytest.raises(InvalidQueryFeatureError):
        build_protoem_initialization_bundle(
            config=config,
            query_input=query,
            prototype_input=prototype,
            inference_input=inference,
            phase5_references=references,
        )


def test_non_finite_score_rejection() -> None:
    config, query, prototype, _, references = _inputs()
    bad_inference = ProtoEMInferenceInput(
        schema_name=PROTOEM_INFERENCE_INPUT_SCHEMA_NAME,
        schema_version=PROTOEM_INFERENCE_INPUT_SCHEMA_VERSION,
        inference_result=_inference_result(
            foreground=prototype.prototype_memory.foreground_prototype,
            background=prototype.prototype_memory.background_prototype,
            foreground_score_map=np.asarray([[[[[np.nan, 0.1]]]]], dtype=np.float64),
        ),
    )

    with pytest.raises(InvalidInitialScoreStateError):
        build_protoem_initialization_bundle(
            config=config,
            query_input=query,
            prototype_input=prototype,
            inference_input=bad_inference,
            phase5_references=references,
        )


def test_nonbinary_prediction_rejection() -> None:
    config, query, prototype, _, references = _inputs()
    bad_inference = ProtoEMInferenceInput(
        schema_name=PROTOEM_INFERENCE_INPUT_SCHEMA_NAME,
        schema_version=PROTOEM_INFERENCE_INPUT_SCHEMA_VERSION,
        inference_result=_inference_result(
            foreground=prototype.prototype_memory.foreground_prototype,
            background=prototype.prototype_memory.background_prototype,
            prediction_mask=np.asarray([[[[[2, 0]]]]], dtype=np.uint8),
        ),
    )

    with pytest.raises(InvalidInitialPredictionError):
        build_protoem_initialization_bundle(
            config=config,
            query_input=query,
            prototype_input=prototype,
            inference_input=bad_inference,
            phase5_references=references,
        )


def test_prototype_channel_mismatch_rejection() -> None:
    config, query, prototype, _, references = _inputs()
    bad_memory = SupportPrototypeMemory(
        foreground_prototype=_foreground_prototype(
            vector=(1.0, 0.0, 0.0),
            support_identifier="support_case_001",
            patient_id="support_patient_001",
            case_id="support_case_001",
        ),
        background_prototype=_background_prototype(
            vector=(0.0, 1.0, 0.0),
            support_identifier="support_case_001",
            patient_id="support_patient_001",
            case_id="support_case_001",
        ),
    )
    bad_inference = ProtoEMInferenceInput(
        schema_name=PROTOEM_INFERENCE_INPUT_SCHEMA_NAME,
        schema_version=PROTOEM_INFERENCE_INPUT_SCHEMA_VERSION,
        inference_result=_inference_result(
            foreground=bad_memory.foreground_prototype,
            background=bad_memory.background_prototype,
            feature_shape=(1, 2, 1, 1, 2),
        ),
    )

    with pytest.raises(IncompatibleInitializationError):
        build_protoem_initialization_bundle(
            config=config,
            query_input=query,
            prototype_input=ProtoEMPrototypeInput(
                schema_name=PROTOEM_PROTOTYPE_INPUT_SCHEMA_NAME,
                schema_version=PROTOEM_PROTOTYPE_INPUT_SCHEMA_VERSION,
                prototype_memory=bad_memory,
                support_provenance=prototype.support_provenance,
            ),
            inference_input=bad_inference,
            phase5_references=references,
        )


def test_missing_required_phase5_reference_rejection() -> None:
    config, query, prototype, inference, references = _inputs()
    bad_references = ProtoEMPhase5ReferenceBundle(
        schema_name=references.schema_name,
        schema_version=references.schema_version,
        initialization_reference=references.initialization_reference,
        prototype_reference=references.prototype_reference,
        inference_reference=ProtoEMPhase5ArtifactReference(
            schema_name=PROTOEM_PHASE5_ARTIFACT_REFERENCE_SCHEMA_NAME,
            schema_version=PROTOEM_PHASE5_ARTIFACT_REFERENCE_SCHEMA_VERSION,
            artifact_schema_name="prototype_only_inference_result",
            artifact_hash=_sha256("9"),
        ),
    )

    with pytest.raises(IncompatibleInitializationError):
        build_protoem_initialization_bundle(
            config=config,
            query_input=query,
            prototype_input=prototype,
            inference_input=inference,
            phase5_references=bad_references,
        )


def test_query_labels_reference_masks_are_absent_from_api() -> None:
    signature = inspect.signature(build_protoem_initialization_bundle)
    signature_text = str(signature)

    assert "label" not in signature_text.lower()
    assert "mask" not in signature_text.lower()
    assert "query_label" not in ProtoEMQueryFeatureInput.__dataclass_fields__
    assert "reference_mask" not in ProtoEMQueryFeatureInput.__dataclass_fields__


def test_repeated_construction_is_byte_identical_at_canonical_mapping_level() -> None:
    config, query, prototype, inference, references = _inputs()
    first = build_protoem_initialization_bundle(
        config=config,
        query_input=query,
        prototype_input=prototype,
        inference_input=inference,
        phase5_references=references,
    )
    second = build_protoem_initialization_bundle(
        config=config,
        query_input=query,
        prototype_input=prototype,
        inference_input=inference,
        phase5_references=references,
    )

    first_mapping = protoem_initialization_bundle_to_dict(first)
    second_mapping = protoem_initialization_bundle_to_dict(second)

    assert first_mapping == second_mapping
    assert canonical_json_bytes(first_mapping) == canonical_json_bytes(second_mapping)
    assert protoem_initialization_bundle_to_json(first) == protoem_initialization_bundle_to_json(
        second
    )
    assert first.initialization_identity_hash == second.initialization_identity_hash
