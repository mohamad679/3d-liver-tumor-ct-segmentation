"""Unit tests for Phase 5 interface and typed retrieval/prototype contracts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError

import pytest

from protoem_ct.models import (
    BackgroundPrototype,
    EmbeddingMetadata,
    FeatureEncoder3D,
    FeatureEncoding3D,
    FeatureResolution3D,
    ForegroundPrototype,
    PromptableSegmenter3D,
    RetrievalCandidate,
    RetrievalContractValidationError,
    RetrievalRequest,
    RetrievalResult,
    RetrievedSupportMatch,
)


def _sha256(seed: str) -> str:
    return (seed * 64)[:64]


def _resolution() -> FeatureResolution3D:
    return FeatureResolution3D(
        input_spatial_shape=(32, 24, 16),
        feature_spatial_shape=(16, 12, 8),
        downsample_factors=(2, 2, 2),
    )


def _metadata(*, input_identity: str = "query_case_001") -> EmbeddingMetadata:
    return EmbeddingMetadata(
        encoder_identity="segresnet_encoder_v1",
        preprocessing_hash=_sha256("a"),
        checkpoint_hash=_sha256("b"),
        input_identity=input_identity,
        feature_stage="decoder_final",
        normalization_name="l2_channel",
        feature_channels=32,
        resolution=_resolution(),
    )


def _candidate(
    *,
    support_identifier: str,
    patient_id: str,
    case_id: str,
    assignment_index: int,
) -> RetrievalCandidate:
    return RetrievalCandidate(
        support_identifier=support_identifier,
        support_patient_id=patient_id,
        support_case_id=case_id,
        support_manifest_hash=_sha256("c"),
        support_assignment_index=assignment_index,
        embedding_metadata=_metadata(input_identity=support_identifier),
    )


def _match(
    *,
    support_identifier: str,
    patient_id: str,
    case_id: str,
    assignment_index: int,
    similarity_score: float,
) -> RetrievedSupportMatch:
    return RetrievedSupportMatch(
        support_identifier=support_identifier,
        support_patient_id=patient_id,
        support_case_id=case_id,
        support_manifest_hash=_sha256("d"),
        support_assignment_index=assignment_index,
        similarity_score=similarity_score,
        embedding_metadata=_metadata(input_identity=support_identifier),
    )


class _DummyEncoder:
    @property
    def encoder_identity(self) -> str:
        return "segresnet_encoder_v1"

    def encode(
        self,
        volume: object,
        *,
        preprocessing_hash: str,
        checkpoint_hash: str,
        input_identity: str,
    ) -> FeatureEncoding3D:
        return FeatureEncoding3D(
            feature_data=volume,
            metadata=EmbeddingMetadata(
                encoder_identity=self.encoder_identity,
                preprocessing_hash=preprocessing_hash,
                checkpoint_hash=checkpoint_hash,
                input_identity=input_identity,
                feature_stage="decoder_final",
                normalization_name="l2_channel",
                feature_channels=16,
                resolution=_resolution(),
            ),
        )


class _DummyPromptableSegmenter:
    def infer_with_prompt(
        self,
        query_features: object,
        *,
        foreground_prototype: ForegroundPrototype,
        background_prototype: BackgroundPrototype,
    ) -> object:
        return (
            query_features,
            foreground_prototype.prototype_kind,
            background_prototype.prototype_kind,
        )


def test_interface_conformance() -> None:
    assert isinstance(_DummyEncoder(), FeatureEncoder3D)
    assert isinstance(_DummyPromptableSegmenter(), PromptableSegmenter3D)


def test_contracts_are_immutable() -> None:
    metadata = _metadata()

    with pytest.raises(FrozenInstanceError):
        metadata.feature_channels = 64  # type: ignore[misc]


def test_canonical_equality_for_identical_metadata() -> None:
    assert _metadata() == _metadata()


def test_retrieval_request_normalizes_candidate_ordering() -> None:
    request = RetrievalRequest(
        query_embedding_metadata=_metadata(),
        support_candidates=(
            _candidate(
                support_identifier="support_case_003",
                patient_id="patient_002",
                case_id="case_003",
                assignment_index=2,
            ),
            _candidate(
                support_identifier="support_case_001",
                patient_id="patient_001",
                case_id="case_001",
                assignment_index=0,
            ),
            _candidate(
                support_identifier="support_case_002",
                patient_id="patient_001",
                case_id="case_002",
                assignment_index=1,
            ),
        ),
        top_k=2,
    )

    assert tuple(item.support_identifier for item in request.support_candidates) == (
        "support_case_001",
        "support_case_002",
        "support_case_003",
    )


def test_retrieval_result_normalizes_match_ordering() -> None:
    result = RetrievalResult(
        query_embedding_metadata=_metadata(),
        similarity_metric="cosine",
        requested_top_k=3,
        ordered_matches=(
            _match(
                support_identifier="support_case_003",
                patient_id="patient_003",
                case_id="case_003",
                assignment_index=2,
                similarity_score=0.75,
            ),
            _match(
                support_identifier="support_case_001",
                patient_id="patient_001",
                case_id="case_001",
                assignment_index=0,
                similarity_score=0.91,
            ),
            _match(
                support_identifier="support_case_002",
                patient_id="patient_002",
                case_id="case_002",
                assignment_index=1,
                similarity_score=0.75,
            ),
        ),
    )

    assert tuple(item.support_identifier for item in result.ordered_matches) == (
        "support_case_001",
        "support_case_002",
        "support_case_003",
    )


def test_prototypes_normalize_source_identifier_ordering() -> None:
    prototype = ForegroundPrototype(
        prototype_kind="ignored",
        feature_channels=32,
        contributing_voxel_count=12,
        source_support_identifiers=(
            "support_case_003",
            "support_case_001",
            "support_case_002",
        ),
        normalization_name="l2_channel",
    )

    assert prototype.prototype_kind == "foreground"
    assert prototype.source_support_identifiers == (
        "support_case_001",
        "support_case_002",
        "support_case_003",
    )


@pytest.mark.parametrize(
    ("factory", "expected_fragment"),
    [
        (
            lambda: EmbeddingMetadata(
                encoder_identity="SegResNet",
                preprocessing_hash=_sha256("a"),
                checkpoint_hash=_sha256("b"),
                input_identity="query_case_001",
                feature_stage="decoder_final",
                normalization_name="l2_channel",
                feature_channels=32,
                resolution=_resolution(),
            ),
            "encoder_identity",
        ),
        (
            lambda: FeatureResolution3D(
                input_spatial_shape=(32, 24, 16),
                feature_spatial_shape=(0, 12, 8),
                downsample_factors=(2, 2, 2),
            ),
            "feature_spatial_shape.depth",
        ),
        (
            lambda: RetrievalRequest(
                query_embedding_metadata=_metadata(),
                support_candidates=(
                    _candidate(
                        support_identifier="support_case_001",
                        patient_id="patient_001",
                        case_id="case_001",
                        assignment_index=0,
                    ),
                ),
                top_k=2,
            ),
            "top_k",
        ),
        (
            lambda: RetrievalResult(
                query_embedding_metadata=_metadata(),
                similarity_metric="cosine",
                requested_top_k=1,
                ordered_matches=(
                    _match(
                        support_identifier="support_case_001",
                        patient_id="patient_001",
                        case_id="case_001",
                        assignment_index=0,
                        similarity_score=1.5,
                    ),
                ),
            ),
            "similarity_score",
        ),
        (
            lambda: BackgroundPrototype(
                prototype_kind="ignored",
                feature_channels=32,
                contributing_voxel_count=4,
                source_support_identifiers=("support_case_001", "support_case_001"),
                normalization_name="l2_channel",
            ),
            "unique",
        ),
    ],
)
def test_invalid_metadata_rejected(
    factory: Callable[[], object],
    expected_fragment: str,
) -> None:
    with pytest.raises(RetrievalContractValidationError, match=expected_fragment):
        factory()
