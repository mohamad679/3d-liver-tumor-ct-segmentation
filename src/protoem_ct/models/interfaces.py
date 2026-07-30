"""Typed Phase 5 model and retrieval interface contracts.

This module defines immutable, validation-backed contracts for deterministic
feature encoding, prompt-aware segmentation, retrieval requests/results, and
support prototypes. It deliberately excludes filesystem access, serialization,
cache logic, retrieval execution, and model execution.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[a-z0-9](?:[a-z0-9_.-]{0,126}[a-z0-9])?$")


class RetrievalContractError(ValueError):
    """Base error for Phase 5 interface and contract validation failures."""


class RetrievalContractValidationError(RetrievalContractError):
    """Raised when one Phase 5 interface contract is malformed."""


def _require_sha256(value: str, *, field_name: str) -> None:
    if not _SHA256_RE.fullmatch(value):
        raise RetrievalContractValidationError(
            f"{field_name} must be a lowercase 64-character SHA-256 hex digest."
        )


def _require_identifier(value: str, *, field_name: str) -> None:
    if not _IDENTIFIER_RE.fullmatch(value):
        raise RetrievalContractValidationError(
            f"{field_name} must match the conservative identifier pattern."
        )


def _require_positive_int(value: int, *, field_name: str) -> None:
    if value <= 0:
        raise RetrievalContractValidationError(f"{field_name} must be a positive integer.")


def _require_nonnegative_int(value: int, *, field_name: str) -> None:
    if value < 0:
        raise RetrievalContractValidationError(f"{field_name} must be a nonnegative integer.")


def _require_spatial_shape(
    value: tuple[int, int, int],
    *,
    field_name: str,
) -> None:
    if len(value) != 3:
        raise RetrievalContractValidationError(f"{field_name} must contain exactly three axes.")
    for axis, item in zip(("depth", "height", "width"), value, strict=True):
        if item <= 0:
            raise RetrievalContractValidationError(
                f"{field_name}.{axis} must be a positive integer."
            )


def _require_probability_score(value: float, *, field_name: str) -> None:
    if not math.isfinite(value):
        raise RetrievalContractValidationError(f"{field_name} must be finite.")
    if value < -1.0 or value > 1.0:
        raise RetrievalContractValidationError(f"{field_name} must lie in [-1.0, 1.0].")


def _require_finite_vector(
    value: tuple[float, ...],
    *,
    field_name: str,
) -> None:
    for index, item in enumerate(value):
        if not math.isfinite(item):
            raise RetrievalContractValidationError(f"{field_name}[{index}] must be finite.")


@dataclass(frozen=True, slots=True)
class FeatureResolution3D:
    """Explicit relation between one input volume and one encoded feature map."""

    input_spatial_shape: tuple[int, int, int]
    feature_spatial_shape: tuple[int, int, int]
    downsample_factors: tuple[int, int, int]

    def __post_init__(self) -> None:
        _require_spatial_shape(self.input_spatial_shape, field_name="input_spatial_shape")
        _require_spatial_shape(self.feature_spatial_shape, field_name="feature_spatial_shape")
        _require_spatial_shape(self.downsample_factors, field_name="downsample_factors")
        for axis, input_size, feature_size, factor in zip(
            ("depth", "height", "width"),
            self.input_spatial_shape,
            self.feature_spatial_shape,
            self.downsample_factors,
            strict=True,
        ):
            if feature_size > input_size:
                raise RetrievalContractValidationError(
                    f"feature_spatial_shape.{axis} cannot exceed input_spatial_shape.{axis}."
                )
            if input_size < factor:
                raise RetrievalContractValidationError(
                    f"downsample_factors.{axis} cannot exceed input_spatial_shape.{axis}."
                )


@dataclass(frozen=True, slots=True)
class EmbeddingMetadata:
    """Deterministic identity and feature metadata for one embedding."""

    encoder_identity: str
    preprocessing_hash: str
    checkpoint_hash: str
    input_identity: str
    feature_stage: str
    normalization_name: str
    feature_channels: int
    resolution: FeatureResolution3D

    def __post_init__(self) -> None:
        _require_identifier(self.encoder_identity, field_name="encoder_identity")
        _require_sha256(self.preprocessing_hash, field_name="preprocessing_hash")
        _require_sha256(self.checkpoint_hash, field_name="checkpoint_hash")
        _require_identifier(self.input_identity, field_name="input_identity")
        _require_identifier(self.feature_stage, field_name="feature_stage")
        _require_identifier(self.normalization_name, field_name="normalization_name")
        _require_positive_int(self.feature_channels, field_name="feature_channels")


@dataclass(frozen=True, slots=True)
class FeatureEncoding3D:
    """One feature payload plus its deterministic embedding metadata."""

    feature_data: object
    metadata: EmbeddingMetadata


@dataclass(frozen=True, slots=True)
class RetrievalCandidate:
    """Support candidate reference for deterministic retrieval planning."""

    support_identifier: str
    support_patient_id: str
    support_case_id: str
    support_manifest_hash: str
    support_assignment_index: int
    embedding_metadata: EmbeddingMetadata

    def __post_init__(self) -> None:
        _require_identifier(self.support_identifier, field_name="support_identifier")
        _require_identifier(self.support_patient_id, field_name="support_patient_id")
        _require_identifier(self.support_case_id, field_name="support_case_id")
        _require_sha256(self.support_manifest_hash, field_name="support_manifest_hash")
        _require_nonnegative_int(
            self.support_assignment_index,
            field_name="support_assignment_index",
        )

    @property
    def ordering_key(self) -> tuple[str, str, int, str]:
        """Return the deterministic ordering key for this candidate."""

        return (
            self.support_patient_id,
            self.support_case_id,
            self.support_assignment_index,
            self.support_identifier,
        )


@dataclass(frozen=True, slots=True)
class RetrievalRequest:
    """Typed request contract for exact deterministic support retrieval."""

    query_embedding_metadata: EmbeddingMetadata
    support_candidates: tuple[RetrievalCandidate, ...]
    similarity_metric: str = "cosine"
    top_k: int = 1

    def __post_init__(self) -> None:
        if self.similarity_metric != "cosine":
            raise RetrievalContractValidationError(
                "similarity_metric must be 'cosine' for Phase 5."
            )
        _require_positive_int(self.top_k, field_name="top_k")
        if not self.support_candidates:
            raise RetrievalContractValidationError(
                "support_candidates must contain at least one item."
            )
        identifiers = [item.support_identifier for item in self.support_candidates]
        if len(set(identifiers)) != len(identifiers):
            raise RetrievalContractValidationError(
                "support_candidates must have unique support_identifier values."
            )
        ordered = tuple(sorted(self.support_candidates, key=lambda item: item.ordering_key))
        object.__setattr__(self, "support_candidates", ordered)
        if self.top_k > len(self.support_candidates):
            raise RetrievalContractValidationError(
                "top_k cannot exceed the number of support_candidates."
            )


@dataclass(frozen=True, slots=True)
class RetrievedSupportMatch:
    """One deterministic support-retrieval match with an exact cosine score."""

    support_identifier: str
    support_patient_id: str
    support_case_id: str
    support_manifest_hash: str
    support_assignment_index: int
    similarity_score: float
    embedding_metadata: EmbeddingMetadata

    def __post_init__(self) -> None:
        _require_identifier(self.support_identifier, field_name="support_identifier")
        _require_identifier(self.support_patient_id, field_name="support_patient_id")
        _require_identifier(self.support_case_id, field_name="support_case_id")
        _require_sha256(self.support_manifest_hash, field_name="support_manifest_hash")
        _require_nonnegative_int(
            self.support_assignment_index,
            field_name="support_assignment_index",
        )
        _require_probability_score(self.similarity_score, field_name="similarity_score")

    @property
    def ordering_key(self) -> tuple[float, str, str, int, str]:
        """Return the deterministic ordering key for retrieval results."""

        return (
            -self.similarity_score,
            self.support_patient_id,
            self.support_case_id,
            self.support_assignment_index,
            self.support_identifier,
        )


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """Deterministically ordered exact-retrieval result contract."""

    query_embedding_metadata: EmbeddingMetadata
    similarity_metric: str
    requested_top_k: int
    ordered_matches: tuple[RetrievedSupportMatch, ...]

    def __post_init__(self) -> None:
        if self.similarity_metric != "cosine":
            raise RetrievalContractValidationError(
                "similarity_metric must be 'cosine' for Phase 5."
            )
        _require_positive_int(self.requested_top_k, field_name="requested_top_k")
        identifiers = [item.support_identifier for item in self.ordered_matches]
        if len(set(identifiers)) != len(identifiers):
            raise RetrievalContractValidationError(
                "ordered_matches must have unique support_identifier values."
            )
        ordered = tuple(sorted(self.ordered_matches, key=lambda item: item.ordering_key))
        object.__setattr__(self, "ordered_matches", ordered)
        if len(self.ordered_matches) > self.requested_top_k:
            raise RetrievalContractValidationError("ordered_matches cannot exceed requested_top_k.")


@dataclass(frozen=True, slots=True)
class SupportPrototype:
    """Base immutable support-prototype contract."""

    prototype_kind: str
    feature_channels: int
    contributing_voxel_count: int
    source_support_identifiers: tuple[str, ...]
    normalization_name: str
    prototype_vector: tuple[float, ...] = ()
    encoder_identity: str = ""
    preprocessing_hash: str = ""
    checkpoint_hash: str = ""
    dataset_manifest_hash: str | None = None
    feature_stage: str = ""
    prototype_content_sha256: str = ""
    prototype_identity_sha256: str = ""

    def __post_init__(self) -> None:
        if self.prototype_kind not in {"foreground", "background"}:
            raise RetrievalContractValidationError(
                "prototype_kind must be either 'foreground' or 'background'."
            )
        _require_positive_int(self.feature_channels, field_name="feature_channels")
        _require_nonnegative_int(
            self.contributing_voxel_count,
            field_name="contributing_voxel_count",
        )
        if not self.source_support_identifiers:
            raise RetrievalContractValidationError(
                "source_support_identifiers must contain at least one identifier."
            )
        normalized_identifiers = tuple(sorted(self.source_support_identifiers))
        for item in normalized_identifiers:
            _require_identifier(item, field_name="source_support_identifiers")
        if len(set(normalized_identifiers)) != len(normalized_identifiers):
            raise RetrievalContractValidationError("source_support_identifiers must be unique.")
        object.__setattr__(self, "source_support_identifiers", normalized_identifiers)
        _require_identifier(self.normalization_name, field_name="normalization_name")
        if self.prototype_vector:
            _require_finite_vector(self.prototype_vector, field_name="prototype_vector")
            if len(self.prototype_vector) != self.feature_channels:
                raise RetrievalContractValidationError(
                    "prototype_vector length must equal feature_channels."
                )
        if self.encoder_identity:
            _require_identifier(self.encoder_identity, field_name="encoder_identity")
        if self.preprocessing_hash:
            _require_sha256(self.preprocessing_hash, field_name="preprocessing_hash")
        if self.checkpoint_hash:
            _require_sha256(self.checkpoint_hash, field_name="checkpoint_hash")
        if self.dataset_manifest_hash is not None:
            _require_sha256(self.dataset_manifest_hash, field_name="dataset_manifest_hash")
        if self.feature_stage:
            _require_identifier(self.feature_stage, field_name="feature_stage")
        if self.prototype_content_sha256:
            _require_sha256(
                self.prototype_content_sha256,
                field_name="prototype_content_sha256",
            )
        if self.prototype_identity_sha256:
            _require_sha256(
                self.prototype_identity_sha256,
                field_name="prototype_identity_sha256",
            )


@dataclass(frozen=True, slots=True)
class ForegroundPrototype(SupportPrototype):
    """Foreground support prototype contract."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "prototype_kind", "foreground")
        SupportPrototype.__post_init__(self)


@dataclass(frozen=True, slots=True)
class BackgroundPrototype(SupportPrototype):
    """Background support prototype contract."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "prototype_kind", "background")
        SupportPrototype.__post_init__(self)


@runtime_checkable
class FeatureEncoder3D(Protocol):
    """Protocol for deterministic 3D feature encoders."""

    @property
    def encoder_identity(self) -> str:
        """Return the stable encoder identity string."""

    def encode(
        self,
        volume: object,
        *,
        preprocessing_hash: str,
        checkpoint_hash: str,
        input_identity: str,
    ) -> FeatureEncoding3D:
        """Encode one preprocessed 3D input deterministically."""


@runtime_checkable
class PromptableSegmenter3D(Protocol):
    """Protocol for prompt-aware inference-only 3D segmenters."""

    def infer_with_prompt(
        self,
        query_features: object,
        *,
        foreground_prototype: ForegroundPrototype,
        background_prototype: BackgroundPrototype,
    ) -> object:
        """Run prompt-aware inference without any learnable update."""


__all__ = [
    "BackgroundPrototype",
    "EmbeddingMetadata",
    "FeatureEncoder3D",
    "FeatureEncoding3D",
    "FeatureResolution3D",
    "ForegroundPrototype",
    "PromptableSegmenter3D",
    "RetrievalCandidate",
    "RetrievalContractError",
    "RetrievalContractValidationError",
    "RetrievalRequest",
    "RetrievalResult",
    "RetrievedSupportMatch",
    "SupportPrototype",
]
