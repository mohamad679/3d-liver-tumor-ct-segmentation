"""Unit tests for deterministic Phase 5 prototype-only inference."""

from __future__ import annotations

import hashlib
from typing import Literal, overload

import numpy as np
import pytest

from protoem_ct.models.interfaces import EmbeddingMetadata, FeatureEncoding3D, FeatureResolution3D
from protoem_ct.retrieval import (
    IncompatiblePrototypeError,
    InferenceLeakageError,
    InvalidQueryFeatureError,
    ZeroNormPrototypeError,
    hash_prototype_inference_identity,
    infer_query_from_prototypes,
    prototype_inference_identity_payload,
    query_label_not_part_of_inference_api,
)
from protoem_ct.retrieval.prototypes import (
    BackgroundPrototype,
    ForegroundPrototype,
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
    dataset_manifest_hash: str | None = _sha256("d"),
    feature_stage: str = "final_encoder",
    normalization_name: str = "l2_channel",
) -> tuple[EmbeddingMetadata, str | None]:
    return (
        EmbeddingMetadata(
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
        ),
        dataset_manifest_hash,
    )


def _query(
    *,
    input_identity: str,
    feature_values: np.ndarray,
    encoder_identity: str = "segresnet_encoder_v1",
    preprocessing_hash: str = _sha256("a"),
    checkpoint_hash: str = _sha256("b"),
    dataset_manifest_hash: str | None = _sha256("d"),
    feature_stage: str = "final_encoder",
    normalization_name: str = "l2_channel",
) -> tuple[FeatureEncoding3D, str | None]:
    metadata, manifest_hash = _metadata(
        input_identity=input_identity,
        feature_channels=int(feature_values.shape[1]),
        feature_spatial_shape=(
            int(feature_values.shape[2]),
            int(feature_values.shape[3]),
            int(feature_values.shape[4]),
        ),
        encoder_identity=encoder_identity,
        preprocessing_hash=preprocessing_hash,
        checkpoint_hash=checkpoint_hash,
        dataset_manifest_hash=dataset_manifest_hash,
        feature_stage=feature_stage,
        normalization_name=normalization_name,
    )
    return FeatureEncoding3D(feature_data=feature_values, metadata=metadata), manifest_hash


@overload
def _prototype(
    *,
    kind: Literal["foreground"],
    vector: tuple[float, ...],
    support_identifier: str,
    patient_id: str,
    case_id: str,
    encoder_identity: str = "segresnet_encoder_v1",
    preprocessing_hash: str = _sha256("a"),
    checkpoint_hash: str = _sha256("b"),
    dataset_manifest_hash: str | None = _sha256("d"),
    feature_stage: str = "final_encoder",
    normalization_name: str = "l2_channel",
) -> ForegroundPrototype: ...


@overload
def _prototype(
    *,
    kind: Literal["background"],
    vector: tuple[float, ...],
    support_identifier: str,
    patient_id: str,
    case_id: str,
    encoder_identity: str = "segresnet_encoder_v1",
    preprocessing_hash: str = _sha256("a"),
    checkpoint_hash: str = _sha256("b"),
    dataset_manifest_hash: str | None = _sha256("d"),
    feature_stage: str = "final_encoder",
    normalization_name: str = "l2_channel",
) -> BackgroundPrototype: ...


def _prototype(
    *,
    kind: str,
    vector: tuple[float, ...],
    support_identifier: str,
    patient_id: str,
    case_id: str,
    encoder_identity: str = "segresnet_encoder_v1",
    preprocessing_hash: str = _sha256("a"),
    checkpoint_hash: str = _sha256("b"),
    dataset_manifest_hash: str | None = _sha256("d"),
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
        contributing_voxel_count=1,
        source_support_identifiers=(support_identifier,),
        normalization_name=normalization_name,
        prototype_vector=vector,
        encoder_identity=encoder_identity,
        preprocessing_hash=preprocessing_hash,
        checkpoint_hash=checkpoint_hash,
        dataset_manifest_hash=dataset_manifest_hash,
        feature_stage=feature_stage,
        source_patient_ids=(patient_id,),
        source_case_ids=(case_id,),
        prototype_content_sha256=content_hash,
        prototype_identity_sha256="0" * 64,
    )
    return prototype_class(
        prototype_kind="ignored",
        feature_channels=len(vector),
        contributing_voxel_count=1,
        source_support_identifiers=(support_identifier,),
        normalization_name=normalization_name,
        prototype_vector=vector,
        encoder_identity=encoder_identity,
        preprocessing_hash=preprocessing_hash,
        checkpoint_hash=checkpoint_hash,
        dataset_manifest_hash=dataset_manifest_hash,
        feature_stage=feature_stage,
        source_patient_ids=(patient_id,),
        source_case_ids=(case_id,),
        prototype_content_sha256=content_hash,
        prototype_identity_sha256=hash_support_prototype_identity(draft),
    )


def test_exact_known_foreground_background_cosine_scores() -> None:
    query_encoding, manifest_hash = _query(
        input_identity="query_case_001",
        feature_values=np.asarray([[[[[1.0, 0.0]]], [[[0.0, 1.0]]]]], dtype=np.float32),
    )
    foreground = _prototype(
        kind="foreground",
        vector=(1.0, 0.0),
        support_identifier="support_fg",
        patient_id="patient_fg",
        case_id="case_fg",
    )
    background = _prototype(
        kind="background",
        vector=(0.0, 1.0),
        support_identifier="support_bg",
        patient_id="patient_bg",
        case_id="case_bg",
    )

    result = infer_query_from_prototypes(
        query_feature_encoding=query_encoding,
        foreground_prototype=foreground,
        background_prototype=background,
        query_patient_id="query_patient_001",
        query_case_id="query_case_001",
        query_identity="query_case_001",
        query_dataset_manifest_hash=manifest_hash,
    )

    assert result.foreground_score_map.tolist() == [[[[[1.0, 0.0]]]]]
    assert result.background_score_map.tolist() == [[[[[0.0, 1.0]]]]]


def test_clear_foreground_classification() -> None:
    query_encoding, manifest_hash = _query(
        input_identity="query_case_001",
        feature_values=np.asarray([[[[[1.0]]], [[[0.0]]]]], dtype=np.float32),
    )
    result = infer_query_from_prototypes(
        query_feature_encoding=query_encoding,
        foreground_prototype=_prototype(
            kind="foreground",
            vector=(1.0, 0.0),
            support_identifier="support_fg",
            patient_id="patient_fg",
            case_id="case_fg",
        ),
        background_prototype=_prototype(
            kind="background",
            vector=(0.0, 1.0),
            support_identifier="support_bg",
            patient_id="patient_bg",
            case_id="case_bg",
        ),
        query_patient_id="query_patient_001",
        query_case_id="query_case_001",
        query_identity="query_case_001",
        query_dataset_manifest_hash=manifest_hash,
    )

    assert result.prediction_mask.tolist() == [[[[[1]]]]]


def test_clear_background_classification() -> None:
    query_encoding, manifest_hash = _query(
        input_identity="query_case_001",
        feature_values=np.asarray([[[[[0.0]]], [[[1.0]]]]], dtype=np.float32),
    )
    result = infer_query_from_prototypes(
        query_feature_encoding=query_encoding,
        foreground_prototype=_prototype(
            kind="foreground",
            vector=(1.0, 0.0),
            support_identifier="support_fg",
            patient_id="patient_fg",
            case_id="case_fg",
        ),
        background_prototype=_prototype(
            kind="background",
            vector=(0.0, 1.0),
            support_identifier="support_bg",
            patient_id="patient_bg",
            case_id="case_bg",
        ),
        query_patient_id="query_patient_001",
        query_case_id="query_case_001",
        query_identity="query_case_001",
        query_dataset_manifest_hash=manifest_hash,
    )

    assert result.prediction_mask.tolist() == [[[[[0]]]]]


def test_exact_tie_predicts_background() -> None:
    query_encoding, manifest_hash = _query(
        input_identity="query_case_001",
        feature_values=np.asarray([[[[[1.0]]], [[[0.0]]]]], dtype=np.float32),
    )
    tied_vector = (1.0, 0.0)
    result = infer_query_from_prototypes(
        query_feature_encoding=query_encoding,
        foreground_prototype=_prototype(
            kind="foreground",
            vector=tied_vector,
            support_identifier="support_fg",
            patient_id="patient_fg",
            case_id="case_fg",
        ),
        background_prototype=_prototype(
            kind="background",
            vector=tied_vector,
            support_identifier="support_bg",
            patient_id="patient_bg",
            case_id="case_bg",
        ),
        query_patient_id="query_patient_001",
        query_case_id="query_case_001",
        query_identity="query_case_001",
        query_dataset_manifest_hash=manifest_hash,
    )

    assert result.foreground_score_map.tolist() == [[[[[1.0]]]]]
    assert result.background_score_map.tolist() == [[[[[1.0]]]]]
    assert result.prediction_mask.tolist() == [[[[[0]]]]]


def test_zero_norm_query_voxel_uses_background_tie_policy() -> None:
    query_encoding, manifest_hash = _query(
        input_identity="query_case_001",
        feature_values=np.asarray([[[[[0.0]]], [[[0.0]]]]], dtype=np.float32),
    )
    result = infer_query_from_prototypes(
        query_feature_encoding=query_encoding,
        foreground_prototype=_prototype(
            kind="foreground",
            vector=(1.0, 0.0),
            support_identifier="support_fg",
            patient_id="patient_fg",
            case_id="case_fg",
        ),
        background_prototype=_prototype(
            kind="background",
            vector=(0.0, 1.0),
            support_identifier="support_bg",
            patient_id="patient_bg",
            case_id="case_bg",
        ),
        query_patient_id="query_patient_001",
        query_case_id="query_case_001",
        query_identity="query_case_001",
        query_dataset_manifest_hash=manifest_hash,
    )

    assert result.foreground_score_map.tolist() == [[[[[0.0]]]]]
    assert result.background_score_map.tolist() == [[[[[0.0]]]]]
    assert result.prediction_mask.tolist() == [[[[[0]]]]]


def test_zero_norm_foreground_prototype_rejected() -> None:
    query_encoding, manifest_hash = _query(
        input_identity="query_case_001",
        feature_values=np.asarray([[[[[1.0]]], [[[0.0]]]]], dtype=np.float32),
    )
    with pytest.raises(ZeroNormPrototypeError):
        infer_query_from_prototypes(
            query_feature_encoding=query_encoding,
            foreground_prototype=_prototype(
                kind="foreground",
                vector=(0.0, 0.0),
                support_identifier="support_fg",
                patient_id="patient_fg",
                case_id="case_fg",
            ),
            background_prototype=_prototype(
                kind="background",
                vector=(0.0, 1.0),
                support_identifier="support_bg",
                patient_id="patient_bg",
                case_id="case_bg",
            ),
            query_patient_id="query_patient_001",
            query_case_id="query_case_001",
            query_identity="query_case_001",
            query_dataset_manifest_hash=manifest_hash,
        )


def test_zero_norm_background_prototype_rejected() -> None:
    query_encoding, manifest_hash = _query(
        input_identity="query_case_001",
        feature_values=np.asarray([[[[[1.0]]], [[[0.0]]]]], dtype=np.float32),
    )
    with pytest.raises(ZeroNormPrototypeError):
        infer_query_from_prototypes(
            query_feature_encoding=query_encoding,
            foreground_prototype=_prototype(
                kind="foreground",
                vector=(1.0, 0.0),
                support_identifier="support_fg",
                patient_id="patient_fg",
                case_id="case_fg",
            ),
            background_prototype=_prototype(
                kind="background",
                vector=(0.0, 0.0),
                support_identifier="support_bg",
                patient_id="patient_bg",
                case_id="case_bg",
            ),
            query_patient_id="query_patient_001",
            query_case_id="query_case_001",
            query_identity="query_case_001",
            query_dataset_manifest_hash=manifest_hash,
        )


def test_nonfinite_query_rejected() -> None:
    query_encoding, manifest_hash = _query(
        input_identity="query_case_001",
        feature_values=np.asarray([[[[[np.nan]]], [[[0.0]]]]], dtype=np.float32),
    )
    with pytest.raises(InvalidQueryFeatureError):
        infer_query_from_prototypes(
            query_feature_encoding=query_encoding,
            foreground_prototype=_prototype(
                kind="foreground",
                vector=(1.0, 0.0),
                support_identifier="support_fg",
                patient_id="patient_fg",
                case_id="case_fg",
            ),
            background_prototype=_prototype(
                kind="background",
                vector=(0.0, 1.0),
                support_identifier="support_bg",
                patient_id="patient_bg",
                case_id="case_bg",
            ),
            query_patient_id="query_patient_001",
            query_case_id="query_case_001",
            query_identity="query_case_001",
            query_dataset_manifest_hash=manifest_hash,
        )


def test_nonfinite_prototype_rejected() -> None:
    query_encoding, manifest_hash = _query(
        input_identity="query_case_001",
        feature_values=np.asarray([[[[[1.0]]], [[[0.0]]]]], dtype=np.float32),
    )
    foreground = _prototype(
        kind="foreground",
        vector=(1.0, 0.0),
        support_identifier="support_fg",
        patient_id="patient_fg",
        case_id="case_fg",
    )
    object.__setattr__(foreground, "prototype_vector", (float("nan"), 0.0))
    with pytest.raises(IncompatiblePrototypeError):
        infer_query_from_prototypes(
            query_feature_encoding=query_encoding,
            foreground_prototype=foreground,
            background_prototype=_prototype(
                kind="background",
                vector=(0.0, 1.0),
                support_identifier="support_bg",
                patient_id="patient_bg",
                case_id="case_bg",
            ),
            query_patient_id="query_patient_001",
            query_case_id="query_case_001",
            query_identity="query_case_001",
            query_dataset_manifest_hash=manifest_hash,
        )


def test_query_rank_mismatch_rejected() -> None:
    metadata, manifest_hash = _metadata(input_identity="query_case_001")
    with pytest.raises(InvalidQueryFeatureError):
        infer_query_from_prototypes(
            query_feature_encoding=FeatureEncoding3D(
                feature_data=np.asarray([[[[1.0, 0.0]]]], dtype=np.float32),
                metadata=metadata,
            ),
            foreground_prototype=_prototype(
                kind="foreground",
                vector=(1.0, 0.0),
                support_identifier="support_fg",
                patient_id="patient_fg",
                case_id="case_fg",
            ),
            background_prototype=_prototype(
                kind="background",
                vector=(0.0, 1.0),
                support_identifier="support_bg",
                patient_id="patient_bg",
                case_id="case_bg",
            ),
            query_patient_id="query_patient_001",
            query_case_id="query_case_001",
            query_identity="query_case_001",
            query_dataset_manifest_hash=manifest_hash,
        )


def test_batch_size_other_than_one_rejected() -> None:
    query_encoding, manifest_hash = _query(
        input_identity="query_case_001",
        feature_values=np.asarray(
            [
                [[[[1.0]]], [[[0.0]]]],
                [[[[1.0]]], [[[0.0]]]],
            ],
            dtype=np.float32,
        ),
    )
    with pytest.raises(InvalidQueryFeatureError):
        infer_query_from_prototypes(
            query_feature_encoding=query_encoding,
            foreground_prototype=_prototype(
                kind="foreground",
                vector=(1.0, 0.0),
                support_identifier="support_fg",
                patient_id="patient_fg",
                case_id="case_fg",
            ),
            background_prototype=_prototype(
                kind="background",
                vector=(0.0, 1.0),
                support_identifier="support_bg",
                patient_id="patient_bg",
                case_id="case_bg",
            ),
            query_patient_id="query_patient_001",
            query_case_id="query_case_001",
            query_identity="query_case_001",
            query_dataset_manifest_hash=manifest_hash,
        )


def test_channel_mismatch_rejected() -> None:
    query_encoding, manifest_hash = _query(
        input_identity="query_case_001",
        feature_values=np.asarray([[[[[1.0]]], [[[0.0]]]]], dtype=np.float32),
    )
    with pytest.raises(IncompatiblePrototypeError):
        infer_query_from_prototypes(
            query_feature_encoding=query_encoding,
            foreground_prototype=_prototype(
                kind="foreground",
                vector=(1.0, 0.0, 0.0),
                support_identifier="support_fg",
                patient_id="patient_fg",
                case_id="case_fg",
            ),
            background_prototype=_prototype(
                kind="background",
                vector=(0.0, 1.0, 0.0),
                support_identifier="support_bg",
                patient_id="patient_bg",
                case_id="case_bg",
            ),
            query_patient_id="query_patient_001",
            query_case_id="query_case_001",
            query_identity="query_case_001",
            query_dataset_manifest_hash=manifest_hash,
        )


def test_metadata_incompatibility_rejected() -> None:
    query_encoding, manifest_hash = _query(
        input_identity="query_case_001",
        feature_values=np.asarray([[[[[1.0]]], [[[0.0]]]]], dtype=np.float32),
    )
    with pytest.raises(IncompatiblePrototypeError):
        infer_query_from_prototypes(
            query_feature_encoding=query_encoding,
            foreground_prototype=_prototype(
                kind="foreground",
                vector=(1.0, 0.0),
                support_identifier="support_fg",
                patient_id="patient_fg",
                case_id="case_fg",
                encoder_identity="segresnet_encoder_v2",
            ),
            background_prototype=_prototype(
                kind="background",
                vector=(0.0, 1.0),
                support_identifier="support_bg",
                patient_id="patient_bg",
                case_id="case_bg",
                encoder_identity="segresnet_encoder_v2",
            ),
            query_patient_id="query_patient_001",
            query_case_id="query_case_001",
            query_identity="query_case_001",
            query_dataset_manifest_hash=manifest_hash,
        )


def test_query_support_patient_leakage_rejected() -> None:
    query_encoding, manifest_hash = _query(
        input_identity="query_case_001",
        feature_values=np.asarray([[[[[1.0]]], [[[0.0]]]]], dtype=np.float32),
    )
    with pytest.raises(InferenceLeakageError):
        infer_query_from_prototypes(
            query_feature_encoding=query_encoding,
            foreground_prototype=_prototype(
                kind="foreground",
                vector=(1.0, 0.0),
                support_identifier="support_fg",
                patient_id="query_patient_001",
                case_id="case_fg",
            ),
            background_prototype=_prototype(
                kind="background",
                vector=(0.0, 1.0),
                support_identifier="support_bg",
                patient_id="patient_bg",
                case_id="case_bg",
            ),
            query_patient_id="query_patient_001",
            query_case_id="query_case_001",
            query_identity="query_case_001",
            query_dataset_manifest_hash=manifest_hash,
        )


def test_query_support_case_leakage_rejected_when_available() -> None:
    query_encoding, manifest_hash = _query(
        input_identity="query_case_001",
        feature_values=np.asarray([[[[[1.0]]], [[[0.0]]]]], dtype=np.float32),
    )
    with pytest.raises(InferenceLeakageError):
        infer_query_from_prototypes(
            query_feature_encoding=query_encoding,
            foreground_prototype=_prototype(
                kind="foreground",
                vector=(1.0, 0.0),
                support_identifier="support_fg",
                patient_id="patient_fg",
                case_id="query_case_001",
            ),
            background_prototype=_prototype(
                kind="background",
                vector=(0.0, 1.0),
                support_identifier="support_bg",
                patient_id="patient_bg",
                case_id="case_bg",
            ),
            query_patient_id="query_patient_001",
            query_case_id="query_case_001",
            query_identity="query_case_001",
            query_dataset_manifest_hash=manifest_hash,
        )


def test_output_shapes_and_dtypes_are_exact() -> None:
    query_encoding, manifest_hash = _query(
        input_identity="query_case_001",
        feature_values=np.asarray([[[[[1.0, 0.0]]], [[[0.0, 1.0]]]]], dtype=np.float32),
    )
    result = infer_query_from_prototypes(
        query_feature_encoding=query_encoding,
        foreground_prototype=_prototype(
            kind="foreground",
            vector=(1.0, 0.0),
            support_identifier="support_fg",
            patient_id="patient_fg",
            case_id="case_fg",
        ),
        background_prototype=_prototype(
            kind="background",
            vector=(0.0, 1.0),
            support_identifier="support_bg",
            patient_id="patient_bg",
            case_id="case_bg",
        ),
        query_patient_id="query_patient_001",
        query_case_id="query_case_001",
        query_identity="query_case_001",
        query_dataset_manifest_hash=manifest_hash,
    )

    assert result.foreground_score_map.shape == (1, 1, 1, 1, 2)
    assert result.background_score_map.shape == (1, 1, 1, 1, 2)
    assert result.prediction_mask.shape == (1, 1, 1, 1, 2)
    assert result.confidence_margin_map is not None
    assert result.confidence_margin_map.shape == (1, 1, 1, 1, 2)
    assert result.foreground_score_map.dtype == np.float64
    assert result.background_score_map.dtype == np.float64
    assert result.prediction_mask.dtype == np.uint8
    assert result.confidence_margin_map.dtype == np.float64


def test_repeated_inference_produces_identical_arrays_and_identity() -> None:
    query_encoding, manifest_hash = _query(
        input_identity="query_case_001",
        feature_values=np.asarray([[[[[1.0, 0.0]]], [[[0.0, 1.0]]]]], dtype=np.float32),
    )
    foreground = _prototype(
        kind="foreground",
        vector=(1.0, 0.0),
        support_identifier="support_fg",
        patient_id="patient_fg",
        case_id="case_fg",
    )
    background = _prototype(
        kind="background",
        vector=(0.0, 1.0),
        support_identifier="support_bg",
        patient_id="patient_bg",
        case_id="case_bg",
    )

    first = infer_query_from_prototypes(
        query_feature_encoding=query_encoding,
        foreground_prototype=foreground,
        background_prototype=background,
        query_patient_id="query_patient_001",
        query_case_id="query_case_001",
        query_identity="query_case_001",
        query_dataset_manifest_hash=manifest_hash,
    )
    second = infer_query_from_prototypes(
        query_feature_encoding=query_encoding,
        foreground_prototype=foreground,
        background_prototype=background,
        query_patient_id="query_patient_001",
        query_case_id="query_case_001",
        query_identity="query_case_001",
        query_dataset_manifest_hash=manifest_hash,
    )

    assert np.array_equal(first.foreground_score_map, second.foreground_score_map)
    assert np.array_equal(first.background_score_map, second.background_score_map)
    assert np.array_equal(first.prediction_mask, second.prediction_mask)
    assert first.inference_identity_sha256 == second.inference_identity_sha256
    assert prototype_inference_identity_payload(first) == prototype_inference_identity_payload(
        second
    )
    assert hash_prototype_inference_identity(first) == hash_prototype_inference_identity(second)


def test_input_memory_layout_does_not_change_results() -> None:
    contiguous_values = np.asarray([[[[[1.0, 0.0]]], [[[0.0, 1.0]]]]], dtype=np.float32)
    fortran_values = np.asfortranarray(contiguous_values)
    contiguous_query, manifest_hash = _query(
        input_identity="query_case_001",
        feature_values=contiguous_values,
    )
    fortran_query, _ = _query(
        input_identity="query_case_001",
        feature_values=fortran_values,
    )
    foreground = _prototype(
        kind="foreground",
        vector=(1.0, 0.0),
        support_identifier="support_fg",
        patient_id="patient_fg",
        case_id="case_fg",
    )
    background = _prototype(
        kind="background",
        vector=(0.0, 1.0),
        support_identifier="support_bg",
        patient_id="patient_bg",
        case_id="case_bg",
    )

    contiguous_result = infer_query_from_prototypes(
        query_feature_encoding=contiguous_query,
        foreground_prototype=foreground,
        background_prototype=background,
        query_patient_id="query_patient_001",
        query_case_id="query_case_001",
        query_identity="query_case_001",
        query_dataset_manifest_hash=manifest_hash,
    )
    fortran_result = infer_query_from_prototypes(
        query_feature_encoding=fortran_query,
        foreground_prototype=foreground,
        background_prototype=background,
        query_patient_id="query_patient_001",
        query_case_id="query_case_001",
        query_identity="query_case_001",
        query_dataset_manifest_hash=manifest_hash,
    )

    assert np.array_equal(
        contiguous_result.foreground_score_map, fortran_result.foreground_score_map
    )
    assert np.array_equal(
        contiguous_result.background_score_map, fortran_result.background_score_map
    )
    assert np.array_equal(contiguous_result.prediction_mask, fortran_result.prediction_mask)
    assert contiguous_result.inference_identity_sha256 == fortran_result.inference_identity_sha256


def test_no_query_label_appears_in_api() -> None:
    assert query_label_not_part_of_inference_api() is True
