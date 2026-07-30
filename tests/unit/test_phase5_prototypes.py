"""Unit tests for deterministic Phase 5 support prototype construction."""

from __future__ import annotations

import numpy as np
import pytest

from protoem_ct.models import EmbeddingMetadata, FeatureEncoding3D, FeatureResolution3D
from protoem_ct.retrieval import (
    DuplicateSupportError,
    EmptyBackgroundPrototypeError,
    EmptyForegroundPrototypeError,
    IncompatibleSupportSetError,
    InvalidSupportFeatureError,
    InvalidSupportMaskError,
    SupportPrototypeInput,
    build_support_prototype_memory,
    query_label_not_part_of_api,
    support_prototype_identity_payload,
)


def _sha256(seed: str) -> str:
    return (seed * 64)[:64]


def _metadata(
    *,
    input_identity: str,
    feature_channels: int = 2,
    input_spatial_shape: tuple[int, int, int] = (2, 2, 2),
    feature_spatial_shape: tuple[int, int, int] = (2, 2, 2),
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
            input_spatial_shape=input_spatial_shape,
            feature_spatial_shape=feature_spatial_shape,
            downsample_factors=(1, 1, 1),
        ),
    )


def _support(
    *,
    support_identifier: str,
    patient_id: str,
    case_id: str,
    feature_values: np.ndarray,
    mask: np.ndarray,
    dataset_manifest_hash: str | None = _sha256("d"),
    metadata: EmbeddingMetadata | None = None,
) -> SupportPrototypeInput:
    feature_spatial_shape = (
        int(feature_values.shape[2]),
        int(feature_values.shape[3]),
        int(feature_values.shape[4]),
    )
    resolved_metadata = (
        _metadata(
            input_identity=support_identifier,
            feature_channels=int(feature_values.shape[1]),
            input_spatial_shape=feature_spatial_shape,
            feature_spatial_shape=feature_spatial_shape,
        )
        if metadata is None
        else metadata
    )
    return SupportPrototypeInput(
        support_identifier=support_identifier,
        support_patient_id=patient_id,
        support_case_id=case_id,
        dataset_manifest_hash=dataset_manifest_hash,
        feature_encoding=FeatureEncoding3D(
            feature_data=feature_values,
            metadata=resolved_metadata,
        ),
        binary_mask=mask,
    )


def test_exact_known_foreground_mean() -> None:
    features = np.asarray(
        [
            [
                [[[1.0, 2.0], [3.0, 4.0]]],
                [[[10.0, 20.0], [30.0, 40.0]]],
            ]
        ],
        dtype=np.float32,
    )
    mask = np.asarray([[[[1, 0], [1, 0]]]], dtype=np.uint8)
    memory = build_support_prototype_memory(
        supports=(
            _support(
                support_identifier="support_a",
                patient_id="patient_a",
                case_id="case_a",
                feature_values=features,
                mask=mask,
            ),
        )
    )

    assert memory.foreground_prototype.prototype_vector == pytest.approx((2.0, 20.0))


def test_exact_known_background_mean() -> None:
    features = np.asarray(
        [
            [
                [[[1.0, 2.0], [3.0, 4.0]]],
                [[[10.0, 20.0], [30.0, 40.0]]],
            ]
        ],
        dtype=np.float32,
    )
    mask = np.asarray([[[[1, 0], [1, 0]]]], dtype=np.uint8)
    memory = build_support_prototype_memory(
        supports=(
            _support(
                support_identifier="support_a",
                patient_id="patient_a",
                case_id="case_a",
                feature_values=features,
                mask=mask,
            ),
        )
    )

    assert memory.background_prototype.prototype_vector == pytest.approx((3.0, 30.0))


def test_correct_output_shape_c() -> None:
    features = np.asarray([[[[[1.0, 2.0, 3.0]]], [[[4.0, 5.0, 6.0]]]]], dtype=np.float32)
    mask = np.asarray([[[[1, 0, 1]]]], dtype=np.uint8)
    memory = build_support_prototype_memory(
        supports=(
            _support(
                support_identifier="support_a",
                patient_id="patient_a",
                case_id="case_a",
                feature_values=features,
                mask=mask,
            ),
        )
    )

    assert np.asarray(memory.foreground_prototype.prototype_vector).shape == (2,)
    assert np.asarray(memory.background_prototype.prototype_vector).shape == (2,)


def test_nearest_neighbor_mask_alignment() -> None:
    features = np.asarray([[[[[1.0, 2.0]]], [[[10.0, 20.0]]]]], dtype=np.float32)
    mask = np.asarray([[[[1, 1, 0, 0]]]], dtype=np.uint8)
    metadata = _metadata(
        input_identity="support_a",
        feature_channels=2,
        input_spatial_shape=(1, 1, 4),
        feature_spatial_shape=(1, 1, 2),
    )
    memory = build_support_prototype_memory(
        supports=(
            _support(
                support_identifier="support_a",
                patient_id="patient_a",
                case_id="case_a",
                feature_values=features,
                mask=mask,
                metadata=metadata,
            ),
        )
    )

    assert memory.foreground_prototype.prototype_vector == pytest.approx((1.0, 10.0))
    assert memory.background_prototype.prototype_vector == pytest.approx((2.0, 20.0))
    assert memory.foreground_prototype.contributing_voxel_count == 1
    assert memory.background_prototype.contributing_voxel_count == 1


def test_multiple_support_aggregation_by_total_contributing_voxels() -> None:
    first = _support(
        support_identifier="support_a",
        patient_id="patient_a",
        case_id="case_a",
        feature_values=np.asarray([[[[[1.0, 5.0, 5.0]]]]], dtype=np.float32),
        mask=np.asarray([[[[1, 0, 0]]]], dtype=np.uint8),
    )
    second = _support(
        support_identifier="support_b",
        patient_id="patient_b",
        case_id="case_b",
        feature_values=np.asarray([[[[[3.0, 3.0, 3.0]]]]], dtype=np.float32),
        mask=np.asarray([[[[1, 1, 1]]]], dtype=np.uint8),
    )
    memory = build_support_prototype_memory(supports=(first, second))

    expected = (1.0 + 3.0 + 3.0 + 3.0) / 4.0
    assert memory.foreground_prototype.prototype_vector == pytest.approx((expected,))


def test_support_input_order_does_not_change_values_or_identity() -> None:
    first = _support(
        support_identifier="support_b",
        patient_id="patient_b",
        case_id="case_b",
        feature_values=np.asarray([[[[[2.0, 4.0]]]]], dtype=np.float32),
        mask=np.asarray([[[[1, 0]]]], dtype=np.uint8),
    )
    second = _support(
        support_identifier="support_a",
        patient_id="patient_a",
        case_id="case_a",
        feature_values=np.asarray([[[[[6.0, 8.0]]]]], dtype=np.float32),
        mask=np.asarray([[[[0, 1]]]], dtype=np.uint8),
    )
    first_memory = build_support_prototype_memory(supports=(first, second))
    second_memory = build_support_prototype_memory(supports=(second, first))

    assert first_memory == second_memory


def test_deterministic_canonical_source_ordering() -> None:
    memory = build_support_prototype_memory(
        supports=(
            _support(
                support_identifier="support_b",
                patient_id="patient_b",
                case_id="case_b",
                feature_values=np.asarray([[[[[1.0, 0.0]]]]], dtype=np.float32),
                mask=np.asarray([[[[1, 0]]]], dtype=np.uint8),
            ),
            _support(
                support_identifier="support_a",
                patient_id="patient_a",
                case_id="case_a",
                feature_values=np.asarray([[[[[0.0, 1.0]]]]], dtype=np.float32),
                mask=np.asarray([[[[0, 1]]]], dtype=np.uint8),
            ),
        )
    )

    assert memory.foreground_prototype.source_support_identifiers == ("support_a", "support_b")


def test_one_empty_foreground_support_plus_one_valid_support_succeeds() -> None:
    empty = _support(
        support_identifier="support_a",
        patient_id="patient_a",
        case_id="case_a",
        feature_values=np.asarray([[[[[1.0, 2.0]]]]], dtype=np.float32),
        mask=np.asarray([[[[0, 0]]]], dtype=np.uint8),
    )
    valid = _support(
        support_identifier="support_b",
        patient_id="patient_b",
        case_id="case_b",
        feature_values=np.asarray([[[[[3.0, 5.0]]]]], dtype=np.float32),
        mask=np.asarray([[[[1, 0]]]], dtype=np.uint8),
    )
    memory = build_support_prototype_memory(supports=(empty, valid))

    assert memory.foreground_prototype.prototype_vector == pytest.approx((3.0,))
    assert memory.background_prototype.contributing_voxel_count == 3


def test_all_empty_foreground_fails_explicitly() -> None:
    with pytest.raises(EmptyForegroundPrototypeError):
        build_support_prototype_memory(
            supports=(
                _support(
                    support_identifier="support_a",
                    patient_id="patient_a",
                    case_id="case_a",
                    feature_values=np.asarray([[[[[1.0, 2.0]]]]], dtype=np.float32),
                    mask=np.asarray([[[[0, 0]]]], dtype=np.uint8),
                ),
            )
        )


def test_all_empty_background_fails_explicitly() -> None:
    with pytest.raises(EmptyBackgroundPrototypeError):
        build_support_prototype_memory(
            supports=(
                _support(
                    support_identifier="support_a",
                    patient_id="patient_a",
                    case_id="case_a",
                    feature_values=np.asarray([[[[[1.0, 2.0]]]]], dtype=np.float32),
                    mask=np.asarray([[[[1, 1]]]], dtype=np.uint8),
                ),
            )
        )


def test_nonbinary_mask_rejection() -> None:
    with pytest.raises(InvalidSupportMaskError):
        build_support_prototype_memory(
            supports=(
                _support(
                    support_identifier="support_a",
                    patient_id="patient_a",
                    case_id="case_a",
                    feature_values=np.asarray([[[[[1.0, 2.0]]]]], dtype=np.float32),
                    mask=np.asarray([[[[2, 0]]]], dtype=np.uint8),
                ),
            )
        )


def test_nonfinite_mask_rejection() -> None:
    with pytest.raises(InvalidSupportMaskError):
        build_support_prototype_memory(
            supports=(
                _support(
                    support_identifier="support_a",
                    patient_id="patient_a",
                    case_id="case_a",
                    feature_values=np.asarray([[[[[1.0, 2.0]]]]], dtype=np.float32),
                    mask=np.asarray([[[[1.0, np.nan]]]], dtype=np.float32),
                ),
            )
        )


def test_nonfinite_feature_rejection() -> None:
    with pytest.raises(InvalidSupportFeatureError):
        build_support_prototype_memory(
            supports=(
                _support(
                    support_identifier="support_a",
                    patient_id="patient_a",
                    case_id="case_a",
                    feature_values=np.asarray([[[[[1.0, np.nan]]]]], dtype=np.float32),
                    mask=np.asarray([[[[1, 0]]]], dtype=np.uint8),
                ),
            )
        )


def test_rank_shape_mismatch_rejection() -> None:
    with pytest.raises(InvalidSupportMaskError):
        build_support_prototype_memory(
            supports=(
                _support(
                    support_identifier="support_a",
                    patient_id="patient_a",
                    case_id="case_a",
                    feature_values=np.asarray([[[[[1.0, 2.0]]]]], dtype=np.float32),
                    mask=np.asarray([[[1, 0]]], dtype=np.uint8),
                ),
            )
        )


def test_channel_mismatch_rejection() -> None:
    first = _support(
        support_identifier="support_a",
        patient_id="patient_a",
        case_id="case_a",
        feature_values=np.asarray([[[[[1.0, 2.0]]]]], dtype=np.float32),
        mask=np.asarray([[[[1, 0]]]], dtype=np.uint8),
    )
    second = _support(
        support_identifier="support_b",
        patient_id="patient_b",
        case_id="case_b",
        feature_values=np.asarray(
            [[[[[1.0, 2.0]]], [[[3.0, 4.0]]]]],
            dtype=np.float32,
        ),
        mask=np.asarray([[[[1, 0]]]], dtype=np.uint8),
    )
    with pytest.raises(IncompatibleSupportSetError):
        build_support_prototype_memory(supports=(first, second))


def test_metadata_incompatibility_rejection() -> None:
    first = _support(
        support_identifier="support_a",
        patient_id="patient_a",
        case_id="case_a",
        feature_values=np.asarray([[[[[1.0, 2.0]]]]], dtype=np.float32),
        mask=np.asarray([[[[1, 0]]]], dtype=np.uint8),
    )
    second = _support(
        support_identifier="support_b",
        patient_id="patient_b",
        case_id="case_b",
        feature_values=np.asarray([[[[[3.0, 4.0]]]]], dtype=np.float32),
        mask=np.asarray([[[[1, 0]]]], dtype=np.uint8),
        metadata=_metadata(
            input_identity="support_b",
            feature_channels=1,
            input_spatial_shape=(1, 1, 2),
            feature_spatial_shape=(1, 1, 2),
            encoder_identity="segresnet_encoder_v2",
        ),
    )
    with pytest.raises(IncompatibleSupportSetError):
        build_support_prototype_memory(supports=(first, second))


def test_duplicate_support_rejection() -> None:
    first = _support(
        support_identifier="support_a",
        patient_id="patient_a",
        case_id="case_a",
        feature_values=np.asarray([[[[[1.0, 2.0]]]]], dtype=np.float32),
        mask=np.asarray([[[[1, 0]]]], dtype=np.uint8),
    )
    second = _support(
        support_identifier="support_a",
        patient_id="patient_b",
        case_id="case_b",
        feature_values=np.asarray([[[[[3.0, 4.0]]]]], dtype=np.float32),
        mask=np.asarray([[[[1, 0]]]], dtype=np.uint8),
    )
    with pytest.raises(DuplicateSupportError):
        build_support_prototype_memory(supports=(first, second))


def test_query_label_is_not_part_of_api() -> None:
    assert query_label_not_part_of_api() is True


def test_repeated_execution_is_identical() -> None:
    supports = (
        _support(
            support_identifier="support_a",
            patient_id="patient_a",
            case_id="case_a",
            feature_values=np.asarray([[[[[1.0, 2.0]]]]], dtype=np.float32),
            mask=np.asarray([[[[1, 0]]]], dtype=np.uint8),
        ),
        _support(
            support_identifier="support_b",
            patient_id="patient_b",
            case_id="case_b",
            feature_values=np.asarray([[[[[3.0, 4.0]]]]], dtype=np.float32),
            mask=np.asarray([[[[0, 1]]]], dtype=np.uint8),
        ),
    )
    first = build_support_prototype_memory(supports=supports)
    second = build_support_prototype_memory(supports=supports)

    assert first == second
    assert support_prototype_identity_payload(
        first.foreground_prototype
    ) == support_prototype_identity_payload(second.foreground_prototype)
