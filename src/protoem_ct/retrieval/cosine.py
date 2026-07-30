"""Exact deterministic cosine-similarity retrieval for Phase 5 embeddings."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import numpy as np

from protoem_ct.models.interfaces import (
    EmbeddingMetadata,
    FeatureEncoding3D,
    FeatureResolution3D,
    RetrievalCandidate,
    RetrievalRequest,
    RetrievalResult,
    RetrievedSupportMatch,
)
from protoem_ct.retrieval.cache import CachedEmbedding

_NUMERIC_KINDS: Final[frozenset[str]] = frozenset({"i", "u", "f"})
_FLOATING_KINDS: Final[frozenset[str]] = frozenset({"f"})


class CosineRetrievalError(ValueError):
    """Base error for deterministic exact cosine retrieval."""


class InvalidEmbeddingError(CosineRetrievalError):
    """Raised when one query or support embedding is malformed or non-finite."""


class IncompatibleEmbeddingError(CosineRetrievalError):
    """Raised when one support embedding is incompatible with the query embedding."""


class ZeroNormEmbeddingError(CosineRetrievalError):
    """Raised when one query or support embedding has zero L2 norm."""


class RetrievalLeakageError(CosineRetrievalError):
    """Raised when query/support identity overlap violates the retrieval contract."""


class InvalidCosineRetrievalRequestError(CosineRetrievalError):
    """Raised when one retrieval request or support-embedding map is invalid."""


@dataclass(frozen=True, slots=True)
class _ResolvedEmbedding:
    identifier: str
    patient_id: str
    case_id: str
    dataset_manifest_hash: str | None
    metadata: EmbeddingMetadata
    array: np.ndarray
    dtype_name: str
    batch_size: int
    channel_count: int
    spatial_shape: tuple[int, int, int]


def retrieve_top_k_by_exact_cosine(
    *,
    request: RetrievalRequest,
    query_embedding: FeatureEncoding3D | CachedEmbedding,
    support_embeddings: Mapping[str, FeatureEncoding3D | CachedEmbedding],
    query_patient_id: str,
    query_case_id: str,
    query_dataset_manifest_hash: str | None = None,
    support_dataset_manifest_hashes: Mapping[str, str | None] | None = None,
) -> RetrievalResult:
    """Return exact deterministic top-k cosine retrieval results.

    Query and support arrays are flattened in row-major C-order without pooling or projection.
    In-memory embeddings must provide explicit dataset-manifest hashes through the function
    arguments because ``EmbeddingMetadata`` deliberately excludes that field.
    """

    if request.top_k <= 0:
        raise InvalidCosineRetrievalRequestError("top_k must be a positive integer.")
    _require_identifier(query_patient_id, field_name="query_patient_id")
    _require_identifier(query_case_id, field_name="query_case_id")
    query_resolved = _resolve_query_embedding(
        query_embedding=query_embedding,
        query_patient_id=query_patient_id,
        query_case_id=query_case_id,
        query_dataset_manifest_hash=query_dataset_manifest_hash,
    )
    support_candidates = _validate_and_normalize_candidates(request)
    resolved_support_embeddings = tuple(
        _resolve_support_embedding(
            candidate=candidate,
            embedding=support_embeddings.get(candidate.support_identifier),
            support_dataset_manifest_hash=support_dataset_manifest_hashes.get(
                candidate.support_identifier
            )
            if support_dataset_manifest_hashes is not None
            else None,
        )
        for candidate in support_candidates
    )
    _validate_support_uniqueness(resolved_support_embeddings)
    matches: list[RetrievedSupportMatch] = []
    query_vector = _flatten_embedding(query_resolved)
    query_norm = _l2_norm(query_vector, role="query")
    for resolved_support in resolved_support_embeddings:
        _validate_query_support_disjointness(query_resolved, resolved_support)
        _validate_embedding_compatibility(query_resolved, resolved_support)
        support_vector = _flatten_embedding(resolved_support)
        support_norm = _l2_norm(support_vector, role="support")
        score = _cosine_score(query_vector, query_norm, support_vector, support_norm)
        matches.append(
            RetrievedSupportMatch(
                support_identifier=resolved_support.identifier,
                support_patient_id=resolved_support.patient_id,
                support_case_id=resolved_support.case_id,
                support_manifest_hash=_candidate_manifest_hash(
                    support_candidates=support_candidates,
                    support_identifier=resolved_support.identifier,
                ),
                support_assignment_index=_candidate_assignment_index(
                    support_candidates=support_candidates,
                    support_identifier=resolved_support.identifier,
                ),
                similarity_score=score,
                embedding_metadata=resolved_support.metadata,
            )
        )
    ordered_matches = tuple(
        sorted(
            matches,
            key=lambda item: (
                -item.similarity_score,
                item.support_patient_id,
                item.support_case_id,
                item.support_identifier,
            ),
        )[: request.top_k]
    )
    return RetrievalResult(
        query_embedding_metadata=query_resolved.metadata,
        similarity_metric="cosine",
        requested_top_k=request.top_k,
        ordered_matches=ordered_matches,
    )


def _validate_and_normalize_candidates(
    request: RetrievalRequest,
) -> tuple[RetrievalCandidate, ...]:
    if request.similarity_metric != "cosine":
        raise InvalidCosineRetrievalRequestError("similarity_metric must be 'cosine'.")
    if request.top_k > len(request.support_candidates):
        raise InvalidCosineRetrievalRequestError(
            "top_k cannot exceed the number of available support candidates."
        )
    return request.support_candidates


def _resolve_query_embedding(
    *,
    query_embedding: FeatureEncoding3D | CachedEmbedding,
    query_patient_id: str,
    query_case_id: str,
    query_dataset_manifest_hash: str | None,
) -> _ResolvedEmbedding:
    if isinstance(query_embedding, CachedEmbedding):
        return _resolved_embedding_from_cached(
            identifier=query_embedding.cache_key.input_identity,
            patient_id=query_patient_id,
            case_id=query_case_id,
            embedding=query_embedding,
        )
    if query_dataset_manifest_hash is None:
        raise InvalidCosineRetrievalRequestError(
            "query_dataset_manifest_hash is required for in-memory query embeddings."
        )
    array = _require_embedding_array(
        query_embedding.feature_data,
        role="query",
    )
    return _ResolvedEmbedding(
        identifier=query_embedding.metadata.input_identity,
        patient_id=query_patient_id,
        case_id=query_case_id,
        dataset_manifest_hash=query_dataset_manifest_hash,
        metadata=query_embedding.metadata,
        array=array,
        dtype_name=array.dtype.name,
        batch_size=int(array.shape[0]),
        channel_count=int(array.shape[1]),
        spatial_shape=(int(array.shape[2]), int(array.shape[3]), int(array.shape[4])),
    )


def _resolve_support_embedding(
    *,
    candidate: RetrievalCandidate,
    embedding: FeatureEncoding3D | CachedEmbedding | None,
    support_dataset_manifest_hash: str | None,
) -> _ResolvedEmbedding:
    if embedding is None:
        raise InvalidCosineRetrievalRequestError(
            f"missing support embedding for {candidate.support_identifier!r}."
        )
    if isinstance(embedding, CachedEmbedding):
        resolved = _resolved_embedding_from_cached(
            identifier=candidate.support_identifier,
            patient_id=candidate.support_patient_id,
            case_id=candidate.support_case_id,
            embedding=embedding,
        )
    else:
        if support_dataset_manifest_hash is None:
            raise InvalidCosineRetrievalRequestError(
                "support_dataset_manifest_hashes must provide every in-memory support candidate."
            )
        array = _require_embedding_array(embedding.feature_data, role="support")
        resolved = _ResolvedEmbedding(
            identifier=candidate.support_identifier,
            patient_id=candidate.support_patient_id,
            case_id=candidate.support_case_id,
            dataset_manifest_hash=support_dataset_manifest_hash,
            metadata=embedding.metadata,
            array=array,
            dtype_name=array.dtype.name,
            batch_size=int(array.shape[0]),
            channel_count=int(array.shape[1]),
            spatial_shape=(int(array.shape[2]), int(array.shape[3]), int(array.shape[4])),
        )
    if resolved.metadata != candidate.embedding_metadata:
        raise IncompatibleEmbeddingError(
            f"support embedding metadata does not match candidate {candidate.support_identifier!r}."
        )
    return resolved


def _resolved_embedding_from_cached(
    *,
    identifier: str,
    patient_id: str,
    case_id: str,
    embedding: CachedEmbedding,
) -> _ResolvedEmbedding:
    array = _require_embedding_array(embedding.array, role="cached")
    metadata = EmbeddingMetadata(
        encoder_identity=embedding.artifact.metadata.encoder_identity,
        preprocessing_hash=embedding.artifact.metadata.preprocessing_hash,
        checkpoint_hash=embedding.artifact.metadata.checkpoint_hash,
        input_identity=identifier,
        feature_stage=embedding.artifact.metadata.feature_stage,
        normalization_name=embedding.artifact.metadata.normalization_name,
        feature_channels=int(array.shape[1]),
        resolution=_synthetic_resolution(array.shape),
    )
    return _ResolvedEmbedding(
        identifier=identifier,
        patient_id=patient_id,
        case_id=case_id,
        dataset_manifest_hash=embedding.artifact.metadata.dataset_manifest_hash,
        metadata=metadata,
        array=array,
        dtype_name=array.dtype.name,
        batch_size=int(array.shape[0]),
        channel_count=int(array.shape[1]),
        spatial_shape=(int(array.shape[2]), int(array.shape[3]), int(array.shape[4])),
    )


def _synthetic_resolution(shape: tuple[int, ...]) -> FeatureResolution3D:
    spatial_shape = (int(shape[2]), int(shape[3]), int(shape[4]))
    return FeatureResolution3D(
        input_spatial_shape=spatial_shape,
        feature_spatial_shape=spatial_shape,
        downsample_factors=(1, 1, 1),
    )


def _require_embedding_array(value: object, *, role: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise InvalidEmbeddingError(f"{role} embedding must be a NumPy ndarray.")
    if value.ndim != 5:
        raise InvalidEmbeddingError(f"{role} embedding must have shape [B, C, D, H, W].")
    if value.dtype.kind not in _NUMERIC_KINDS or value.dtype.kind not in _FLOATING_KINDS:
        raise InvalidEmbeddingError(f"{role} embedding must use a finite floating numeric dtype.")
    if not np.isfinite(value).all():
        raise InvalidEmbeddingError(f"{role} embedding must not contain NaN or Infinity.")
    return np.ascontiguousarray(value)


def _flatten_embedding(embedding: _ResolvedEmbedding) -> np.ndarray:
    return embedding.array.reshape(-1, order="C").astype(np.float64, copy=False)


def _l2_norm(vector: np.ndarray, *, role: str) -> float:
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm):
        raise InvalidEmbeddingError(f"{role} embedding norm must be finite.")
    if norm == 0.0:
        raise ZeroNormEmbeddingError(f"{role} embedding must not have zero L2 norm.")
    return norm


def _cosine_score(
    query_vector: np.ndarray,
    query_norm: float,
    support_vector: np.ndarray,
    support_norm: float,
) -> float:
    numerator = float(np.dot(query_vector, support_vector))
    if not math.isfinite(numerator):
        raise InvalidEmbeddingError("cosine numerator must be finite.")
    score = numerator / (query_norm * support_norm)
    if not math.isfinite(score):
        raise InvalidEmbeddingError("cosine similarity must be finite.")
    if score > 1.0 and score <= 1.0 + 1e-12:
        score = 1.0
    elif score < -1.0 and score >= -1.0 - 1e-12:
        score = -1.0
    elif score < -1.0 or score > 1.0:
        raise InvalidEmbeddingError("cosine similarity fell outside [-1, 1].")
    return score


def _validate_support_uniqueness(
    resolved_support_embeddings: tuple[_ResolvedEmbedding, ...],
) -> None:
    identifiers = [item.identifier for item in resolved_support_embeddings]
    if len(set(identifiers)) != len(identifiers):
        raise InvalidCosineRetrievalRequestError("support candidates must be unique by identifier.")
    patient_case_pairs = [(item.patient_id, item.case_id) for item in resolved_support_embeddings]
    if len(set(patient_case_pairs)) != len(patient_case_pairs):
        raise InvalidCosineRetrievalRequestError(
            "support candidates must not duplicate the same patient/case assignment."
        )


def _validate_query_support_disjointness(
    query_embedding: _ResolvedEmbedding,
    support_embedding: _ResolvedEmbedding,
) -> None:
    if query_embedding.patient_id == support_embedding.patient_id:
        raise RetrievalLeakageError("query/support patient overlap is not allowed.")
    if query_embedding.case_id == support_embedding.case_id:
        raise RetrievalLeakageError("query/support case overlap is not allowed.")


def _validate_embedding_compatibility(
    query_embedding: _ResolvedEmbedding,
    support_embedding: _ResolvedEmbedding,
) -> None:
    comparisons = (
        ("batch size", query_embedding.batch_size, support_embedding.batch_size),
        ("channel count", query_embedding.channel_count, support_embedding.channel_count),
        ("spatial shape", query_embedding.spatial_shape, support_embedding.spatial_shape),
        ("dtype", query_embedding.dtype_name, support_embedding.dtype_name),
        (
            "encoder_identity",
            query_embedding.metadata.encoder_identity,
            support_embedding.metadata.encoder_identity,
        ),
        (
            "preprocessing_hash",
            query_embedding.metadata.preprocessing_hash,
            support_embedding.metadata.preprocessing_hash,
        ),
        (
            "checkpoint_hash",
            query_embedding.metadata.checkpoint_hash,
            support_embedding.metadata.checkpoint_hash,
        ),
        (
            "dataset_manifest_hash",
            query_embedding.dataset_manifest_hash,
            support_embedding.dataset_manifest_hash,
        ),
        (
            "feature_stage",
            query_embedding.metadata.feature_stage,
            support_embedding.metadata.feature_stage,
        ),
        (
            "normalization_name",
            query_embedding.metadata.normalization_name,
            support_embedding.metadata.normalization_name,
        ),
    )
    for field_name, expected, observed in comparisons:
        if expected != observed:
            raise IncompatibleEmbeddingError(
                f"support embedding {support_embedding.identifier!r} has incompatible {field_name}."
            )


def _candidate_manifest_hash(
    *,
    support_candidates: tuple[RetrievalCandidate, ...],
    support_identifier: str,
) -> str:
    for item in support_candidates:
        if item.support_identifier == support_identifier:
            return item.support_manifest_hash
    raise InvalidCosineRetrievalRequestError(
        f"missing support manifest hash for {support_identifier!r}."
    )


def _candidate_assignment_index(
    *,
    support_candidates: tuple[RetrievalCandidate, ...],
    support_identifier: str,
) -> int:
    for item in support_candidates:
        if item.support_identifier == support_identifier:
            return item.support_assignment_index
    raise InvalidCosineRetrievalRequestError(
        f"missing support assignment index for {support_identifier!r}."
    )


def _require_identifier(value: str, *, field_name: str) -> None:
    if value == "" or value != value.strip():
        raise InvalidCosineRetrievalRequestError(f"{field_name} must be a nonempty identifier.")


__all__ = [
    "CosineRetrievalError",
    "IncompatibleEmbeddingError",
    "InvalidCosineRetrievalRequestError",
    "InvalidEmbeddingError",
    "RetrievalLeakageError",
    "ZeroNormEmbeddingError",
    "retrieve_top_k_by_exact_cosine",
]
