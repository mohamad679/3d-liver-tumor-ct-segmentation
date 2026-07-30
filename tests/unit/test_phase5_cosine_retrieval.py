"""Unit tests for deterministic Phase 5 exact cosine retrieval."""

from __future__ import annotations

import math

import numpy as np
import pytest

from protoem_ct.models import (
    EmbeddingMetadata,
    FeatureEncoding3D,
    FeatureResolution3D,
    RetrievalCandidate,
    RetrievalRequest,
)
from protoem_ct.retrieval import (
    IncompatibleEmbeddingError,
    InvalidCosineRetrievalRequestError,
    InvalidEmbeddingError,
    RetrievalLeakageError,
    ZeroNormEmbeddingError,
    retrieve_top_k_by_exact_cosine,
)


def _sha256(seed: str) -> str:
    return (seed * 64)[:64]


def _resolution() -> FeatureResolution3D:
    return FeatureResolution3D(
        input_spatial_shape=(1, 1, 4),
        feature_spatial_shape=(1, 1, 4),
        downsample_factors=(1, 1, 1),
    )


def _metadata(*, input_identity: str) -> EmbeddingMetadata:
    return EmbeddingMetadata(
        encoder_identity="segresnet_encoder_v1",
        preprocessing_hash=_sha256("a"),
        checkpoint_hash=_sha256("b"),
        input_identity=input_identity,
        feature_stage="final_encoder",
        normalization_name="l2_channel",
        feature_channels=1,
        resolution=_resolution(),
    )


def _embedding(
    values: list[float], *, input_identity: str, dtype: str = "float32"
) -> FeatureEncoding3D:
    array = np.asarray(values, dtype=np.dtype(dtype)).reshape((1, 1, 1, 1, len(values)))
    return FeatureEncoding3D(
        feature_data=array,
        metadata=_metadata(input_identity=input_identity),
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


def _request(*candidates: RetrievalCandidate, top_k: int) -> RetrievalRequest:
    return RetrievalRequest(
        query_embedding_metadata=_metadata(input_identity="query_case_001"),
        support_candidates=tuple(candidates),
        top_k=top_k,
    )


def test_exact_known_cosine_scores() -> None:
    request = _request(
        _candidate(
            support_identifier="support_case_001",
            patient_id="patient_001",
            case_id="case_001",
            assignment_index=0,
        ),
        _candidate(
            support_identifier="support_case_002",
            patient_id="patient_002",
            case_id="case_002",
            assignment_index=1,
        ),
        top_k=2,
    )
    result = retrieve_top_k_by_exact_cosine(
        request=request,
        query_embedding=_embedding([1.0, 0.0, 1.0, 0.0], input_identity="query_case_001"),
        support_embeddings={
            "support_case_001": _embedding([1.0, 0.0, 1.0, 0.0], input_identity="support_case_001"),
            "support_case_002": _embedding([1.0, 1.0, 0.0, 0.0], input_identity="support_case_002"),
        },
        query_patient_id="query_patient_001",
        query_case_id="query_case_001",
        query_dataset_manifest_hash=_sha256("d"),
        support_dataset_manifest_hashes={
            "support_case_001": _sha256("d"),
            "support_case_002": _sha256("d"),
        },
    )

    assert result.ordered_matches[0].similarity_score == pytest.approx(1.0)
    assert result.ordered_matches[1].similarity_score == pytest.approx(0.5)


def test_identical_vectors_score_one() -> None:
    request = _request(
        _candidate(
            support_identifier="support_case_001",
            patient_id="patient_001",
            case_id="case_001",
            assignment_index=0,
        ),
        top_k=1,
    )
    result = retrieve_top_k_by_exact_cosine(
        request=request,
        query_embedding=_embedding([1.0, 2.0], input_identity="query_case_001"),
        support_embeddings={
            "support_case_001": _embedding([1.0, 2.0], input_identity="support_case_001"),
        },
        query_patient_id="query_patient_001",
        query_case_id="query_case_001",
        query_dataset_manifest_hash=_sha256("d"),
        support_dataset_manifest_hashes={"support_case_001": _sha256("d")},
    )
    assert result.ordered_matches[0].similarity_score == pytest.approx(1.0)


def test_opposite_vectors_score_minus_one() -> None:
    request = _request(
        _candidate(
            support_identifier="support_case_001",
            patient_id="patient_001",
            case_id="case_001",
            assignment_index=0,
        ),
        top_k=1,
    )
    result = retrieve_top_k_by_exact_cosine(
        request=request,
        query_embedding=_embedding([1.0, -2.0], input_identity="query_case_001"),
        support_embeddings={
            "support_case_001": _embedding([-1.0, 2.0], input_identity="support_case_001"),
        },
        query_patient_id="query_patient_001",
        query_case_id="query_case_001",
        query_dataset_manifest_hash=_sha256("d"),
        support_dataset_manifest_hashes={"support_case_001": _sha256("d")},
    )
    assert result.ordered_matches[0].similarity_score == pytest.approx(-1.0)


def test_orthogonal_vectors_score_zero() -> None:
    request = _request(
        _candidate(
            support_identifier="support_case_001",
            patient_id="patient_001",
            case_id="case_001",
            assignment_index=0,
        ),
        top_k=1,
    )
    result = retrieve_top_k_by_exact_cosine(
        request=request,
        query_embedding=_embedding([1.0, 0.0], input_identity="query_case_001"),
        support_embeddings={
            "support_case_001": _embedding([0.0, 1.0], input_identity="support_case_001"),
        },
        query_patient_id="query_patient_001",
        query_case_id="query_case_001",
        query_dataset_manifest_hash=_sha256("d"),
        support_dataset_manifest_hashes={"support_case_001": _sha256("d")},
    )
    assert result.ordered_matches[0].similarity_score == pytest.approx(0.0)


def test_deterministic_descending_ranking() -> None:
    request = _request(
        _candidate(support_identifier="c", patient_id="p3", case_id="c3", assignment_index=2),
        _candidate(support_identifier="a", patient_id="p1", case_id="c1", assignment_index=0),
        _candidate(support_identifier="b", patient_id="p2", case_id="c2", assignment_index=1),
        top_k=3,
    )
    result = retrieve_top_k_by_exact_cosine(
        request=request,
        query_embedding=_embedding([1.0, 0.0], input_identity="query_case_001"),
        support_embeddings={
            "a": _embedding([1.0, 0.0], input_identity="a"),
            "b": _embedding([0.5, 0.0], input_identity="b"),
            "c": _embedding([-1.0, 0.0], input_identity="c"),
        },
        query_patient_id="query_patient_001",
        query_case_id="query_case_001",
        query_dataset_manifest_hash=_sha256("d"),
        support_dataset_manifest_hashes={"a": _sha256("d"), "b": _sha256("d"), "c": _sha256("d")},
    )
    assert tuple(item.support_identifier for item in result.ordered_matches) == ("a", "b", "c")


def test_deterministic_multilevel_tie_breaking() -> None:
    request = _request(
        _candidate(
            support_identifier="z",
            patient_id="patient_002",
            case_id="case_002",
            assignment_index=1,
        ),
        _candidate(
            support_identifier="a",
            patient_id="patient_001",
            case_id="case_002",
            assignment_index=0,
        ),
        _candidate(
            support_identifier="b",
            patient_id="patient_001",
            case_id="case_001",
            assignment_index=2,
        ),
        top_k=3,
    )
    result = retrieve_top_k_by_exact_cosine(
        request=request,
        query_embedding=_embedding([1.0, 1.0], input_identity="query_case_001"),
        support_embeddings={
            "z": _embedding([1.0, 1.0], input_identity="z"),
            "a": _embedding([1.0, 1.0], input_identity="a"),
            "b": _embedding([1.0, 1.0], input_identity="b"),
        },
        query_patient_id="query_patient_0010",
        query_case_id="query_case_001",
        query_dataset_manifest_hash=_sha256("d"),
        support_dataset_manifest_hashes={
            "z": _sha256("d"),
            "a": _sha256("d"),
            "b": _sha256("d"),
        },
    )
    assert tuple(item.support_identifier for item in result.ordered_matches) == ("b", "a", "z")


def test_repeated_execution_is_identical() -> None:
    request = _request(
        _candidate(support_identifier="a", patient_id="p1", case_id="c1", assignment_index=0),
        _candidate(support_identifier="b", patient_id="p2", case_id="c2", assignment_index=1),
        top_k=2,
    )
    query_embedding = _embedding([1.0, 0.5], input_identity="query_case_001")
    support_embeddings = {
        "a": _embedding([1.0, 0.5], input_identity="a"),
        "b": _embedding([0.5, 1.0], input_identity="b"),
    }
    support_dataset_manifest_hashes = {"a": _sha256("d"), "b": _sha256("d")}
    first = retrieve_top_k_by_exact_cosine(
        request=request,
        query_embedding=query_embedding,
        support_embeddings=support_embeddings,
        query_patient_id="query_patient_001",
        query_case_id="query_case_001",
        query_dataset_manifest_hash=_sha256("d"),
        support_dataset_manifest_hashes=support_dataset_manifest_hashes,
    )
    second = retrieve_top_k_by_exact_cosine(
        request=request,
        query_embedding=query_embedding,
        support_embeddings=support_embeddings,
        query_patient_id="query_patient_001",
        query_case_id="query_case_001",
        query_dataset_manifest_hash=_sha256("d"),
        support_dataset_manifest_hashes=support_dataset_manifest_hashes,
    )

    assert first == second


def test_top_k_behavior() -> None:
    request = _request(
        _candidate(support_identifier="a", patient_id="p1", case_id="c1", assignment_index=0),
        _candidate(support_identifier="b", patient_id="p2", case_id="c2", assignment_index=1),
        top_k=1,
    )
    result = retrieve_top_k_by_exact_cosine(
        request=request,
        query_embedding=_embedding([1.0, 0.0], input_identity="query_case_001"),
        support_embeddings={
            "a": _embedding([1.0, 0.0], input_identity="a"),
            "b": _embedding([0.5, 0.0], input_identity="b"),
        },
        query_patient_id="query_patient_001",
        query_case_id="query_case_001",
        query_dataset_manifest_hash=_sha256("d"),
        support_dataset_manifest_hashes={"a": _sha256("d"), "b": _sha256("d")},
    )
    assert len(result.ordered_matches) == 1
    assert result.ordered_matches[0].support_identifier == "a"


def test_zero_query_norm_rejection() -> None:
    request = _request(
        _candidate(support_identifier="a", patient_id="p1", case_id="c1", assignment_index=0),
        top_k=1,
    )
    with pytest.raises(ZeroNormEmbeddingError):
        retrieve_top_k_by_exact_cosine(
            request=request,
            query_embedding=_embedding([0.0, 0.0], input_identity="query_case_001"),
            support_embeddings={"a": _embedding([1.0, 0.0], input_identity="a")},
            query_patient_id="query_patient_001",
            query_case_id="query_case_001",
            query_dataset_manifest_hash=_sha256("d"),
            support_dataset_manifest_hashes={"a": _sha256("d")},
        )


def test_zero_support_norm_rejection() -> None:
    request = _request(
        _candidate(support_identifier="a", patient_id="p1", case_id="c1", assignment_index=0),
        top_k=1,
    )
    with pytest.raises(ZeroNormEmbeddingError):
        retrieve_top_k_by_exact_cosine(
            request=request,
            query_embedding=_embedding([1.0, 0.0], input_identity="query_case_001"),
            support_embeddings={"a": _embedding([0.0, 0.0], input_identity="a")},
            query_patient_id="query_patient_001",
            query_case_id="query_case_001",
            query_dataset_manifest_hash=_sha256("d"),
            support_dataset_manifest_hashes={"a": _sha256("d")},
        )


def test_non_finite_input_rejection() -> None:
    request = _request(
        _candidate(support_identifier="a", patient_id="p1", case_id="c1", assignment_index=0),
        top_k=1,
    )
    bad = np.asarray([1.0, math.nan], dtype=np.float32).reshape((1, 1, 1, 1, 2))
    with pytest.raises(InvalidEmbeddingError):
        retrieve_top_k_by_exact_cosine(
            request=request,
            query_embedding=FeatureEncoding3D(
                feature_data=bad,
                metadata=_metadata(input_identity="query_case_001"),
            ),
            support_embeddings={"a": _embedding([1.0, 0.0], input_identity="a")},
            query_patient_id="query_patient_001",
            query_case_id="query_case_001",
            query_dataset_manifest_hash=_sha256("d"),
            support_dataset_manifest_hashes={"a": _sha256("d")},
        )


def test_rank_mismatch_rejection() -> None:
    request = _request(
        _candidate(support_identifier="a", patient_id="p1", case_id="c1", assignment_index=0),
        top_k=1,
    )
    bad = np.asarray([1.0, 0.0], dtype=np.float32).reshape((1, 2))
    with pytest.raises(InvalidEmbeddingError):
        retrieve_top_k_by_exact_cosine(
            request=request,
            query_embedding=FeatureEncoding3D(
                feature_data=bad, metadata=_metadata(input_identity="query_case_001")
            ),
            support_embeddings={"a": _embedding([1.0, 0.0], input_identity="a")},
            query_patient_id="query_patient_001",
            query_case_id="query_case_001",
            query_dataset_manifest_hash=_sha256("d"),
            support_dataset_manifest_hashes={"a": _sha256("d")},
        )


def test_shape_mismatch_rejection() -> None:
    request = _request(
        _candidate(support_identifier="a", patient_id="p1", case_id="c1", assignment_index=0),
        top_k=1,
    )
    with pytest.raises(IncompatibleEmbeddingError):
        retrieve_top_k_by_exact_cosine(
            request=request,
            query_embedding=_embedding([1.0, 0.0], input_identity="query_case_001"),
            support_embeddings={
                "a": FeatureEncoding3D(
                    feature_data=np.asarray([1.0, 0.0, 0.0], dtype=np.float32).reshape(
                        (1, 1, 1, 1, 3)
                    ),
                    metadata=_metadata(input_identity="a"),
                )
            },
            query_patient_id="query_patient_001",
            query_case_id="query_case_001",
            query_dataset_manifest_hash=_sha256("d"),
            support_dataset_manifest_hashes={"a": _sha256("d")},
        )


def test_metadata_incompatibility_rejection() -> None:
    request = _request(
        _candidate(support_identifier="a", patient_id="p1", case_id="c1", assignment_index=0),
        top_k=1,
    )
    bad_metadata = EmbeddingMetadata(
        encoder_identity="segresnet_encoder_v2",
        preprocessing_hash=_sha256("a"),
        checkpoint_hash=_sha256("b"),
        input_identity="a",
        feature_stage="final_encoder",
        normalization_name="l2_channel",
        feature_channels=1,
        resolution=_resolution(),
    )
    with pytest.raises(IncompatibleEmbeddingError):
        retrieve_top_k_by_exact_cosine(
            request=request,
            query_embedding=_embedding([1.0, 0.0], input_identity="query_case_001"),
            support_embeddings={
                "a": FeatureEncoding3D(
                    feature_data=_embedding(
                        [1.0, 0.0],
                        input_identity="a",
                    ).feature_data,
                    metadata=bad_metadata,
                )
            },
            query_patient_id="query_patient_001",
            query_case_id="query_case_001",
            query_dataset_manifest_hash=_sha256("d"),
            support_dataset_manifest_hashes={"a": _sha256("d")},
        )


def test_duplicate_support_rejection() -> None:
    request = _request(
        _candidate(support_identifier="a", patient_id="p1", case_id="c1", assignment_index=0),
        _candidate(support_identifier="b", patient_id="p1", case_id="c1", assignment_index=1),
        top_k=2,
    )
    with pytest.raises(InvalidCosineRetrievalRequestError):
        retrieve_top_k_by_exact_cosine(
            request=request,
            query_embedding=_embedding([1.0, 0.0], input_identity="query_case_001"),
            support_embeddings={
                "a": _embedding([1.0, 0.0], input_identity="a"),
                "b": _embedding([0.5, 0.0], input_identity="b"),
            },
            query_patient_id="query_patient_001",
            query_case_id="query_case_001",
            query_dataset_manifest_hash=_sha256("d"),
            support_dataset_manifest_hashes={"a": _sha256("d"), "b": _sha256("d")},
        )


def test_patient_leakage_rejection() -> None:
    request = _request(
        _candidate(
            support_identifier="a",
            patient_id="query_patient_001",
            case_id="c1",
            assignment_index=0,
        ),
        top_k=1,
    )
    with pytest.raises(RetrievalLeakageError):
        retrieve_top_k_by_exact_cosine(
            request=request,
            query_embedding=_embedding([1.0, 0.0], input_identity="query_case_001"),
            support_embeddings={"a": _embedding([1.0, 0.0], input_identity="a")},
            query_patient_id="query_patient_001",
            query_case_id="query_case_001",
            query_dataset_manifest_hash=_sha256("d"),
            support_dataset_manifest_hashes={"a": _sha256("d")},
        )


def test_case_leakage_rejection() -> None:
    request = _request(
        _candidate(
            support_identifier="a",
            patient_id="p1",
            case_id="query_case_001",
            assignment_index=0,
        ),
        top_k=1,
    )
    with pytest.raises(RetrievalLeakageError):
        retrieve_top_k_by_exact_cosine(
            request=request,
            query_embedding=_embedding([1.0, 0.0], input_identity="query_case_001"),
            support_embeddings={"a": _embedding([1.0, 0.0], input_identity="a")},
            query_patient_id="query_patient_001",
            query_case_id="query_case_001",
            query_dataset_manifest_hash=_sha256("d"),
            support_dataset_manifest_hashes={"a": _sha256("d")},
        )


def test_input_candidate_order_does_not_affect_final_ranking() -> None:
    first_request = _request(
        _candidate(support_identifier="b", patient_id="p2", case_id="c2", assignment_index=1),
        _candidate(support_identifier="a", patient_id="p1", case_id="c1", assignment_index=0),
        top_k=2,
    )
    second_request = _request(
        _candidate(support_identifier="a", patient_id="p1", case_id="c1", assignment_index=0),
        _candidate(support_identifier="b", patient_id="p2", case_id="c2", assignment_index=1),
        top_k=2,
    )
    query_embedding = _embedding([1.0, 0.0], input_identity="query_case_001")
    support_embeddings = {
        "a": _embedding([1.0, 0.0], input_identity="a"),
        "b": _embedding([0.5, 0.0], input_identity="b"),
    }
    support_dataset_manifest_hashes = {"a": _sha256("d"), "b": _sha256("d")}
    first = retrieve_top_k_by_exact_cosine(
        request=first_request,
        query_embedding=query_embedding,
        support_embeddings=support_embeddings,
        query_patient_id="query_patient_001",
        query_case_id="query_case_001",
        query_dataset_manifest_hash=_sha256("d"),
        support_dataset_manifest_hashes=support_dataset_manifest_hashes,
    )
    second = retrieve_top_k_by_exact_cosine(
        request=second_request,
        query_embedding=query_embedding,
        support_embeddings=support_embeddings,
        query_patient_id="query_patient_001",
        query_case_id="query_case_001",
        query_dataset_manifest_hash=_sha256("d"),
        support_dataset_manifest_hashes=support_dataset_manifest_hashes,
    )
    assert first == second
