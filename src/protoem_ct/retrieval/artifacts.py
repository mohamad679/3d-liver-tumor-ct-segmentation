"""Deterministic Phase 5 embedding artifact and cache contracts."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from protoem_ct.artifacts.hashing import JsonValue, sha256_json

EMBEDDING_ARTIFACT_SCHEMA_NAME: Final[str] = "retrieval_embedding_artifact"
EMBEDDING_ARTIFACT_SCHEMA_VERSION: Final[str] = "v1"
EMBEDDING_ARTIFACT_METADATA_SCHEMA_NAME: Final[str] = "retrieval_embedding_artifact_metadata"
EMBEDDING_ARTIFACT_METADATA_SCHEMA_VERSION: Final[str] = "v1"
EMBEDDING_TENSOR_DESCRIPTOR_SCHEMA_NAME: Final[str] = "retrieval_embedding_tensor_descriptor"
EMBEDDING_TENSOR_DESCRIPTOR_SCHEMA_VERSION: Final[str] = "v1"
EMBEDDING_CACHE_KEY_SCHEMA_NAME: Final[str] = "retrieval_embedding_cache_key"
EMBEDDING_CACHE_KEY_SCHEMA_VERSION: Final[str] = "v1"
EMBEDDING_CACHE_ENTRY_SCHEMA_NAME: Final[str] = "retrieval_embedding_cache_entry"
EMBEDDING_CACHE_ENTRY_SCHEMA_VERSION: Final[str] = "v1"
EMBEDDING_CACHE_MANIFEST_SCHEMA_NAME: Final[str] = "retrieval_embedding_cache_manifest"
EMBEDDING_CACHE_MANIFEST_SCHEMA_VERSION: Final[str] = "v1"

SUPPORTED_EMBEDDING_TENSOR_DTYPES: Final[frozenset[str]] = frozenset(
    {"bfloat16", "float16", "float32", "float64"}
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[a-z0-9](?:[a-z0-9_.-]{0,126}[a-z0-9])?$")
_RELATIVE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,126}$")
_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")


class RetrievalArtifactError(ValueError):
    """Base error for Phase 5 retrieval artifact contracts."""


class RetrievalArtifactValidationError(RetrievalArtifactError):
    """Raised when one retrieval artifact contract is malformed."""


class RetrievalArtifactSerializationError(RetrievalArtifactError):
    """Raised when one retrieval artifact mapping cannot be reconstructed safely."""


class RetrievalArtifactHashError(RetrievalArtifactError):
    """Raised when a persisted hash does not match the canonical content."""


def hash_embedding_cache_key(cache_key: EmbeddingCacheKey) -> str:
    """Return the canonical SHA-256 hash for one embedding cache key."""

    return sha256_json(embedding_cache_key_identity_payload(cache_key))


def hash_embedding_artifact(artifact: EmbeddingArtifact) -> str:
    """Return the canonical SHA-256 hash for one embedding artifact."""

    return sha256_json(embedding_artifact_identity_payload(artifact))


def hash_embedding_cache_manifest(manifest: EmbeddingCacheManifest) -> str:
    """Return the canonical SHA-256 hash for one embedding cache manifest."""

    return sha256_json(embedding_cache_manifest_identity_payload(manifest))


def embedding_cache_key_identity_payload(cache_key: EmbeddingCacheKey) -> dict[str, JsonValue]:
    """Return the exact canonical identity payload for one embedding cache key."""

    return {
        "checkpoint_hash": cache_key.checkpoint_hash,
        "dataset_manifest_hash": cache_key.dataset_manifest_hash,
        "encoder_identity": cache_key.encoder_identity,
        "feature_shape": list(cache_key.feature_shape),
        "feature_stage": cache_key.feature_stage,
        "input_identity": cache_key.input_identity,
        "normalization_name": cache_key.normalization_name,
        "preprocessing_hash": cache_key.preprocessing_hash,
        "schema_name": cache_key.schema_name,
        "schema_version": cache_key.schema_version,
        "tensor_dtype": cache_key.tensor_dtype,
    }


def embedding_artifact_identity_payload(artifact: EmbeddingArtifact) -> dict[str, JsonValue]:
    """Return the exact canonical identity payload for one embedding artifact."""

    return {
        "cache_key_hash": artifact.cache_key_hash,
        "metadata": embedding_artifact_metadata_to_dict(artifact.metadata),
        "schema_name": artifact.schema_name,
        "schema_version": artifact.schema_version,
        "tensor_descriptor": embedding_tensor_descriptor_to_dict(artifact.tensor_descriptor),
    }


def embedding_cache_manifest_identity_payload(
    manifest: EmbeddingCacheManifest,
) -> dict[str, JsonValue]:
    """Return the exact canonical identity payload for one embedding cache manifest."""

    return {
        "entries": [
            {
                "artifact_hash": item.artifact_hash,
                "cache_key": embedding_cache_key_to_dict(item.cache_key),
                "cache_key_hash": item.cache_key_hash,
                "relative_artifact_path": item.relative_artifact_path,
                "schema_name": item.schema_name,
                "schema_version": item.schema_version,
            }
            for item in manifest.entries
        ],
        "schema_name": manifest.schema_name,
        "schema_version": manifest.schema_version,
    }


@dataclass(frozen=True, slots=True)
class EmbeddingArtifactMetadata:
    """Identity and provenance metadata for one embedding artifact."""

    schema_name: str
    schema_version: str
    encoder_identity: str
    preprocessing_hash: str
    checkpoint_hash: str
    input_identity: str
    dataset_manifest_hash: str | None
    feature_stage: str
    normalization_name: str

    def __post_init__(self) -> None:
        _require_schema_name(
            self.schema_name,
            expected=EMBEDDING_ARTIFACT_METADATA_SCHEMA_NAME,
            field_name="schema_name",
        )
        _require_schema_version(
            self.schema_version,
            expected=EMBEDDING_ARTIFACT_METADATA_SCHEMA_VERSION,
            field_name="schema_version",
        )
        _require_identifier(self.encoder_identity, field_name="encoder_identity")
        _require_sha256(self.preprocessing_hash, field_name="preprocessing_hash")
        _require_sha256(self.checkpoint_hash, field_name="checkpoint_hash")
        _require_identifier(self.input_identity, field_name="input_identity")
        _require_optional_sha256(
            self.dataset_manifest_hash,
            field_name="dataset_manifest_hash",
        )
        _require_identifier(self.feature_stage, field_name="feature_stage")
        _require_identifier(self.normalization_name, field_name="normalization_name")


@dataclass(frozen=True, slots=True)
class EmbeddingTensorDescriptor:
    """Tensor descriptor for one embedding artifact."""

    schema_name: str
    schema_version: str
    feature_shape: tuple[int, int, int, int, int]
    tensor_dtype: str
    tensor_byte_length: int
    tensor_content_sha256: str
    relative_artifact_path: str

    def __post_init__(self) -> None:
        _require_schema_name(
            self.schema_name,
            expected=EMBEDDING_TENSOR_DESCRIPTOR_SCHEMA_NAME,
            field_name="schema_name",
        )
        _require_schema_version(
            self.schema_version,
            expected=EMBEDDING_TENSOR_DESCRIPTOR_SCHEMA_VERSION,
            field_name="schema_version",
        )
        _require_feature_shape(self.feature_shape, field_name="feature_shape")
        if self.tensor_dtype not in SUPPORTED_EMBEDDING_TENSOR_DTYPES:
            raise RetrievalArtifactValidationError(
                f"tensor_dtype must be one of {sorted(SUPPORTED_EMBEDDING_TENSOR_DTYPES)!r}."
            )
        _require_nonnegative_int(self.tensor_byte_length, field_name="tensor_byte_length")
        _require_sha256(self.tensor_content_sha256, field_name="tensor_content_sha256")
        _require_relative_artifact_path(
            self.relative_artifact_path,
            field_name="relative_artifact_path",
        )


@dataclass(frozen=True, slots=True)
class EmbeddingCacheKey:
    """Deterministic compatibility identity for one embedding cache entry."""

    schema_name: str
    schema_version: str
    cache_key_hash: str
    encoder_identity: str
    preprocessing_hash: str
    checkpoint_hash: str
    input_identity: str
    dataset_manifest_hash: str | None
    feature_stage: str
    normalization_name: str
    feature_shape: tuple[int, int, int, int, int]
    tensor_dtype: str

    def __post_init__(self) -> None:
        _require_schema_name(
            self.schema_name,
            expected=EMBEDDING_CACHE_KEY_SCHEMA_NAME,
            field_name="schema_name",
        )
        _require_schema_version(
            self.schema_version,
            expected=EMBEDDING_CACHE_KEY_SCHEMA_VERSION,
            field_name="schema_version",
        )
        _require_sha256(self.cache_key_hash, field_name="cache_key_hash")
        _require_identifier(self.encoder_identity, field_name="encoder_identity")
        _require_sha256(self.preprocessing_hash, field_name="preprocessing_hash")
        _require_sha256(self.checkpoint_hash, field_name="checkpoint_hash")
        _require_identifier(self.input_identity, field_name="input_identity")
        _require_optional_sha256(
            self.dataset_manifest_hash,
            field_name="dataset_manifest_hash",
        )
        _require_identifier(self.feature_stage, field_name="feature_stage")
        _require_identifier(self.normalization_name, field_name="normalization_name")
        _require_feature_shape(self.feature_shape, field_name="feature_shape")
        if self.tensor_dtype not in SUPPORTED_EMBEDDING_TENSOR_DTYPES:
            raise RetrievalArtifactValidationError(
                f"tensor_dtype must be one of {sorted(SUPPORTED_EMBEDDING_TENSOR_DTYPES)!r}."
            )
        expected_hash = hash_embedding_cache_key(self)
        if self.cache_key_hash != expected_hash:
            raise RetrievalArtifactHashError(
                "cache_key_hash does not match the deterministic cache-key content."
            )


@dataclass(frozen=True, slots=True)
class EmbeddingArtifact:
    """Deterministic embedding artifact contract without tensor-byte serialization."""

    schema_name: str
    schema_version: str
    artifact_hash: str
    cache_key_hash: str
    metadata: EmbeddingArtifactMetadata
    tensor_descriptor: EmbeddingTensorDescriptor
    created_at_utc: str | None = None

    def __post_init__(self) -> None:
        _require_schema_name(
            self.schema_name,
            expected=EMBEDDING_ARTIFACT_SCHEMA_NAME,
            field_name="schema_name",
        )
        _require_schema_version(
            self.schema_version,
            expected=EMBEDDING_ARTIFACT_SCHEMA_VERSION,
            field_name="schema_version",
        )
        _require_sha256(self.artifact_hash, field_name="artifact_hash")
        _require_sha256(self.cache_key_hash, field_name="cache_key_hash")
        _require_optional_timestamp(self.created_at_utc, field_name="created_at_utc")
        expected_hash = hash_embedding_artifact(self)
        if self.artifact_hash != expected_hash:
            raise RetrievalArtifactHashError(
                "artifact_hash does not match the deterministic embedding artifact content."
            )


@dataclass(frozen=True, slots=True)
class EmbeddingCacheEntry:
    """One deterministic manifest entry linking a cache key to an embedding artifact."""

    schema_name: str
    schema_version: str
    cache_key: EmbeddingCacheKey
    cache_key_hash: str
    artifact_hash: str
    relative_artifact_path: str
    created_at_utc: str | None = None

    def __post_init__(self) -> None:
        _require_schema_name(
            self.schema_name,
            expected=EMBEDDING_CACHE_ENTRY_SCHEMA_NAME,
            field_name="schema_name",
        )
        _require_schema_version(
            self.schema_version,
            expected=EMBEDDING_CACHE_ENTRY_SCHEMA_VERSION,
            field_name="schema_version",
        )
        _require_sha256(self.cache_key_hash, field_name="cache_key_hash")
        _require_sha256(self.artifact_hash, field_name="artifact_hash")
        _require_relative_artifact_path(
            self.relative_artifact_path,
            field_name="relative_artifact_path",
        )
        _require_optional_timestamp(self.created_at_utc, field_name="created_at_utc")
        if self.cache_key_hash != self.cache_key.cache_key_hash:
            raise RetrievalArtifactHashError(
                "EmbeddingCacheEntry cache_key_hash must match the nested cache key."
            )

    @property
    def ordering_key(self) -> str:
        """Return the deterministic manifest ordering key."""

        return self.cache_key_hash


@dataclass(frozen=True, slots=True)
class EmbeddingCacheManifest:
    """Deterministic embedding cache manifest sorted by cache key hash."""

    schema_name: str
    schema_version: str
    artifact_hash: str
    entries: tuple[EmbeddingCacheEntry, ...]
    created_at_utc: str | None = None

    def __post_init__(self) -> None:
        _require_schema_name(
            self.schema_name,
            expected=EMBEDDING_CACHE_MANIFEST_SCHEMA_NAME,
            field_name="schema_name",
        )
        _require_schema_version(
            self.schema_version,
            expected=EMBEDDING_CACHE_MANIFEST_SCHEMA_VERSION,
            field_name="schema_version",
        )
        _require_sha256(self.artifact_hash, field_name="artifact_hash")
        _require_optional_timestamp(self.created_at_utc, field_name="created_at_utc")
        ordered_entries = tuple(sorted(self.entries, key=lambda item: item.ordering_key))
        object.__setattr__(self, "entries", ordered_entries)
        keys = [item.cache_key_hash for item in self.entries]
        if len(set(keys)) != len(keys):
            raise RetrievalArtifactValidationError(
                "EmbeddingCacheManifest entries must have unique cache_key_hash values."
            )
        for item in self.entries:
            if item.cache_key_hash != item.cache_key.cache_key_hash:
                raise RetrievalArtifactHashError(
                    "EmbeddingCacheManifest entry cache_key_hash does not match its nested key."
                )
        expected_hash = hash_embedding_cache_manifest(self)
        if self.artifact_hash != expected_hash:
            raise RetrievalArtifactHashError(
                "artifact_hash does not match the deterministic cache manifest content."
            )


def embedding_artifact_metadata_to_dict(
    metadata: EmbeddingArtifactMetadata,
) -> dict[str, JsonValue]:
    """Convert one embedding artifact metadata object to a canonical mapping."""

    return {
        "checkpoint_hash": metadata.checkpoint_hash,
        "dataset_manifest_hash": metadata.dataset_manifest_hash,
        "encoder_identity": metadata.encoder_identity,
        "feature_stage": metadata.feature_stage,
        "input_identity": metadata.input_identity,
        "normalization_name": metadata.normalization_name,
        "preprocessing_hash": metadata.preprocessing_hash,
        "schema_name": metadata.schema_name,
        "schema_version": metadata.schema_version,
    }


def embedding_tensor_descriptor_to_dict(
    descriptor: EmbeddingTensorDescriptor,
) -> dict[str, JsonValue]:
    """Convert one embedding tensor descriptor to a canonical mapping."""

    return {
        "feature_shape": list(descriptor.feature_shape),
        "relative_artifact_path": descriptor.relative_artifact_path,
        "schema_name": descriptor.schema_name,
        "schema_version": descriptor.schema_version,
        "tensor_byte_length": descriptor.tensor_byte_length,
        "tensor_content_sha256": descriptor.tensor_content_sha256,
        "tensor_dtype": descriptor.tensor_dtype,
    }


def embedding_cache_key_to_dict(cache_key: EmbeddingCacheKey) -> dict[str, JsonValue]:
    """Convert one embedding cache key to a canonical mapping."""

    return {
        "cache_key_hash": cache_key.cache_key_hash,
        "checkpoint_hash": cache_key.checkpoint_hash,
        "dataset_manifest_hash": cache_key.dataset_manifest_hash,
        "encoder_identity": cache_key.encoder_identity,
        "feature_shape": list(cache_key.feature_shape),
        "feature_stage": cache_key.feature_stage,
        "input_identity": cache_key.input_identity,
        "normalization_name": cache_key.normalization_name,
        "preprocessing_hash": cache_key.preprocessing_hash,
        "schema_name": cache_key.schema_name,
        "schema_version": cache_key.schema_version,
        "tensor_dtype": cache_key.tensor_dtype,
    }


def embedding_artifact_to_dict(artifact: EmbeddingArtifact) -> dict[str, JsonValue]:
    """Convert one embedding artifact to a canonical mapping."""

    return {
        "artifact_hash": artifact.artifact_hash,
        "cache_key_hash": artifact.cache_key_hash,
        "created_at_utc": artifact.created_at_utc,
        "metadata": embedding_artifact_metadata_to_dict(artifact.metadata),
        "schema_name": artifact.schema_name,
        "schema_version": artifact.schema_version,
        "tensor_descriptor": embedding_tensor_descriptor_to_dict(artifact.tensor_descriptor),
    }


def embedding_cache_entry_to_dict(entry: EmbeddingCacheEntry) -> dict[str, JsonValue]:
    """Convert one embedding cache entry to a canonical mapping."""

    return {
        "artifact_hash": entry.artifact_hash,
        "cache_key": embedding_cache_key_to_dict(entry.cache_key),
        "cache_key_hash": entry.cache_key_hash,
        "created_at_utc": entry.created_at_utc,
        "relative_artifact_path": entry.relative_artifact_path,
        "schema_name": entry.schema_name,
        "schema_version": entry.schema_version,
    }


def embedding_cache_manifest_to_dict(manifest: EmbeddingCacheManifest) -> dict[str, JsonValue]:
    """Convert one embedding cache manifest to a canonical mapping."""

    return {
        "artifact_hash": manifest.artifact_hash,
        "created_at_utc": manifest.created_at_utc,
        "entries": [embedding_cache_entry_to_dict(item) for item in manifest.entries],
        "schema_name": manifest.schema_name,
        "schema_version": manifest.schema_version,
    }


def embedding_artifact_metadata_from_mapping(mapping: MappingLike) -> EmbeddingArtifactMetadata:
    """Reconstruct embedding artifact metadata from one strict mapping."""

    _require_exact_fields(
        mapping,
        required_fields=_EMBEDDING_ARTIFACT_METADATA_FIELDS,
        object_name="EmbeddingArtifactMetadata",
    )
    return EmbeddingArtifactMetadata(
        schema_name=_expect_string(mapping["schema_name"], field_name="schema_name"),
        schema_version=_expect_string(mapping["schema_version"], field_name="schema_version"),
        encoder_identity=_expect_string(mapping["encoder_identity"], field_name="encoder_identity"),
        preprocessing_hash=_expect_string(
            mapping["preprocessing_hash"],
            field_name="preprocessing_hash",
        ),
        checkpoint_hash=_expect_string(mapping["checkpoint_hash"], field_name="checkpoint_hash"),
        input_identity=_expect_string(mapping["input_identity"], field_name="input_identity"),
        dataset_manifest_hash=_expect_optional_string(
            mapping["dataset_manifest_hash"],
            field_name="dataset_manifest_hash",
        ),
        feature_stage=_expect_string(mapping["feature_stage"], field_name="feature_stage"),
        normalization_name=_expect_string(
            mapping["normalization_name"],
            field_name="normalization_name",
        ),
    )


def embedding_tensor_descriptor_from_mapping(mapping: MappingLike) -> EmbeddingTensorDescriptor:
    """Reconstruct one tensor descriptor from a strict mapping."""

    _require_exact_fields(
        mapping,
        required_fields=_EMBEDDING_TENSOR_DESCRIPTOR_FIELDS,
        object_name="EmbeddingTensorDescriptor",
    )
    return EmbeddingTensorDescriptor(
        schema_name=_expect_string(mapping["schema_name"], field_name="schema_name"),
        schema_version=_expect_string(mapping["schema_version"], field_name="schema_version"),
        feature_shape=_expect_int_tuple5(mapping["feature_shape"], field_name="feature_shape"),
        tensor_dtype=_expect_string(mapping["tensor_dtype"], field_name="tensor_dtype"),
        tensor_byte_length=_expect_int(
            mapping["tensor_byte_length"],
            field_name="tensor_byte_length",
        ),
        tensor_content_sha256=_expect_string(
            mapping["tensor_content_sha256"],
            field_name="tensor_content_sha256",
        ),
        relative_artifact_path=_expect_string(
            mapping["relative_artifact_path"],
            field_name="relative_artifact_path",
        ),
    )


def embedding_cache_key_from_mapping(mapping: MappingLike) -> EmbeddingCacheKey:
    """Reconstruct one embedding cache key from a strict mapping."""

    _require_exact_fields(
        mapping,
        required_fields=_EMBEDDING_CACHE_KEY_FIELDS,
        object_name="EmbeddingCacheKey",
    )
    return EmbeddingCacheKey(
        schema_name=_expect_string(mapping["schema_name"], field_name="schema_name"),
        schema_version=_expect_string(mapping["schema_version"], field_name="schema_version"),
        cache_key_hash=_expect_string(mapping["cache_key_hash"], field_name="cache_key_hash"),
        encoder_identity=_expect_string(mapping["encoder_identity"], field_name="encoder_identity"),
        preprocessing_hash=_expect_string(
            mapping["preprocessing_hash"],
            field_name="preprocessing_hash",
        ),
        checkpoint_hash=_expect_string(mapping["checkpoint_hash"], field_name="checkpoint_hash"),
        input_identity=_expect_string(mapping["input_identity"], field_name="input_identity"),
        dataset_manifest_hash=_expect_optional_string(
            mapping["dataset_manifest_hash"],
            field_name="dataset_manifest_hash",
        ),
        feature_stage=_expect_string(mapping["feature_stage"], field_name="feature_stage"),
        normalization_name=_expect_string(
            mapping["normalization_name"],
            field_name="normalization_name",
        ),
        feature_shape=_expect_int_tuple5(mapping["feature_shape"], field_name="feature_shape"),
        tensor_dtype=_expect_string(mapping["tensor_dtype"], field_name="tensor_dtype"),
    )


def embedding_artifact_from_mapping(mapping: MappingLike) -> EmbeddingArtifact:
    """Reconstruct one embedding artifact from a strict mapping."""

    _require_exact_fields(
        mapping,
        required_fields=_EMBEDDING_ARTIFACT_FIELDS,
        object_name="EmbeddingArtifact",
    )
    return EmbeddingArtifact(
        schema_name=_expect_string(mapping["schema_name"], field_name="schema_name"),
        schema_version=_expect_string(mapping["schema_version"], field_name="schema_version"),
        artifact_hash=_expect_string(mapping["artifact_hash"], field_name="artifact_hash"),
        cache_key_hash=_expect_string(mapping["cache_key_hash"], field_name="cache_key_hash"),
        metadata=embedding_artifact_metadata_from_mapping(
            _expect_mapping(mapping["metadata"], field_name="metadata")
        ),
        tensor_descriptor=embedding_tensor_descriptor_from_mapping(
            _expect_mapping(mapping["tensor_descriptor"], field_name="tensor_descriptor")
        ),
        created_at_utc=_expect_optional_string(
            mapping["created_at_utc"],
            field_name="created_at_utc",
        ),
    )


def embedding_cache_entry_from_mapping(mapping: MappingLike) -> EmbeddingCacheEntry:
    """Reconstruct one embedding cache entry from a strict mapping."""

    _require_exact_fields(
        mapping,
        required_fields=_EMBEDDING_CACHE_ENTRY_FIELDS,
        object_name="EmbeddingCacheEntry",
    )
    return EmbeddingCacheEntry(
        schema_name=_expect_string(mapping["schema_name"], field_name="schema_name"),
        schema_version=_expect_string(mapping["schema_version"], field_name="schema_version"),
        cache_key=embedding_cache_key_from_mapping(
            _expect_mapping(mapping["cache_key"], field_name="cache_key")
        ),
        cache_key_hash=_expect_string(mapping["cache_key_hash"], field_name="cache_key_hash"),
        artifact_hash=_expect_string(mapping["artifact_hash"], field_name="artifact_hash"),
        relative_artifact_path=_expect_string(
            mapping["relative_artifact_path"],
            field_name="relative_artifact_path",
        ),
        created_at_utc=_expect_optional_string(
            mapping["created_at_utc"],
            field_name="created_at_utc",
        ),
    )


def embedding_cache_manifest_from_mapping(mapping: MappingLike) -> EmbeddingCacheManifest:
    """Reconstruct one embedding cache manifest from a strict mapping."""

    _require_exact_fields(
        mapping,
        required_fields=_EMBEDDING_CACHE_MANIFEST_FIELDS,
        object_name="EmbeddingCacheManifest",
    )
    entries_value = mapping["entries"]
    if not isinstance(entries_value, list):
        raise RetrievalArtifactSerializationError("EmbeddingCacheManifest.entries must be a list.")
    return EmbeddingCacheManifest(
        schema_name=_expect_string(mapping["schema_name"], field_name="schema_name"),
        schema_version=_expect_string(mapping["schema_version"], field_name="schema_version"),
        artifact_hash=_expect_string(mapping["artifact_hash"], field_name="artifact_hash"),
        entries=tuple(
            embedding_cache_entry_from_mapping(
                _expect_mapping(item, field_name=f"entries[{index}]")
            )
            for index, item in enumerate(entries_value)
        ),
        created_at_utc=_expect_optional_string(
            mapping["created_at_utc"],
            field_name="created_at_utc",
        ),
    )


MappingLike = Mapping[str, object]

_EMBEDDING_ARTIFACT_METADATA_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_name",
        "schema_version",
        "encoder_identity",
        "preprocessing_hash",
        "checkpoint_hash",
        "input_identity",
        "dataset_manifest_hash",
        "feature_stage",
        "normalization_name",
    }
)
_EMBEDDING_TENSOR_DESCRIPTOR_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_name",
        "schema_version",
        "feature_shape",
        "tensor_dtype",
        "tensor_byte_length",
        "tensor_content_sha256",
        "relative_artifact_path",
    }
)
_EMBEDDING_CACHE_KEY_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_name",
        "schema_version",
        "cache_key_hash",
        "encoder_identity",
        "preprocessing_hash",
        "checkpoint_hash",
        "input_identity",
        "dataset_manifest_hash",
        "feature_stage",
        "normalization_name",
        "feature_shape",
        "tensor_dtype",
    }
)
_EMBEDDING_ARTIFACT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_name",
        "schema_version",
        "artifact_hash",
        "cache_key_hash",
        "metadata",
        "tensor_descriptor",
        "created_at_utc",
    }
)
_EMBEDDING_CACHE_ENTRY_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_name",
        "schema_version",
        "cache_key",
        "cache_key_hash",
        "artifact_hash",
        "relative_artifact_path",
        "created_at_utc",
    }
)
_EMBEDDING_CACHE_MANIFEST_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_name",
        "schema_version",
        "artifact_hash",
        "entries",
        "created_at_utc",
    }
)


def _require_schema_name(value: str, *, expected: str, field_name: str) -> None:
    if value != expected:
        raise RetrievalArtifactValidationError(f"{field_name} must equal {expected!r}.")


def _require_schema_version(value: str, *, expected: str, field_name: str) -> None:
    if value != expected:
        raise RetrievalArtifactValidationError(f"{field_name} must equal {expected!r}.")


def _require_sha256(value: str, *, field_name: str) -> None:
    if not _SHA256_RE.fullmatch(value):
        raise RetrievalArtifactValidationError(
            f"{field_name} must be a lowercase 64-character SHA-256 hex digest."
        )


def _require_identifier(value: str, *, field_name: str) -> None:
    if not _IDENTIFIER_RE.fullmatch(value):
        raise RetrievalArtifactValidationError(
            f"{field_name} must match the conservative identifier pattern."
        )


def _require_optional_sha256(value: str | None, *, field_name: str) -> None:
    if value is None:
        return
    _require_sha256(value, field_name=field_name)


def _require_feature_shape(
    value: tuple[int, int, int, int, int],
    *,
    field_name: str,
) -> None:
    if len(value) != 5:
        raise RetrievalArtifactValidationError(
            f"{field_name} must contain exactly five axes [B, C, D, H, W]."
        )
    for axis_name, item in zip(("B", "C", "D", "H", "W"), value, strict=True):
        if not isinstance(item, int) or item <= 0:
            raise RetrievalArtifactValidationError(
                f"{field_name}.{axis_name} must be a positive integer."
            )


def _require_nonnegative_int(value: int, *, field_name: str) -> None:
    if not isinstance(value, int) or value < 0:
        raise RetrievalArtifactValidationError(f"{field_name} must be a nonnegative integer.")


def _require_relative_artifact_path(value: str, *, field_name: str) -> None:
    if not value:
        raise RetrievalArtifactValidationError(f"{field_name} must be a nonempty relative path.")
    if "\\" in value:
        raise RetrievalArtifactValidationError(
            f"{field_name} must not contain backslash path separators."
        )
    if value.startswith("/") or value.startswith("~"):
        raise RetrievalArtifactValidationError(
            f"{field_name} must not be absolute or home-relative."
        )
    components = value.split("/")
    if any(component == "" for component in components):
        raise RetrievalArtifactValidationError(
            f"{field_name} must not contain empty path components."
        )
    for component in components:
        if component in {".", ".."}:
            raise RetrievalArtifactValidationError(
                f"{field_name} must not contain traversal components."
            )
        if not _RELATIVE_COMPONENT_RE.fullmatch(component):
            raise RetrievalArtifactValidationError(
                f"{field_name} contains an unsafe relative path component: {component!r}."
            )


def _require_optional_timestamp(value: str | None, *, field_name: str) -> None:
    if value is None:
        return
    if not _TIMESTAMP_RE.fullmatch(value):
        raise RetrievalArtifactValidationError(
            f"{field_name} must use explicit UTC timestamp formatting."
        )


def _expect_mapping(value: object, *, field_name: str) -> MappingLike:
    if not isinstance(value, dict):
        raise RetrievalArtifactSerializationError(f"{field_name} must be a mapping object.")
    if not all(isinstance(key, str) for key in value):
        raise RetrievalArtifactSerializationError(f"{field_name} must use only string keys.")
    return value


def _expect_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise RetrievalArtifactSerializationError(f"{field_name} must be a string.")
    return value


def _expect_optional_string(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _expect_string(value, field_name=field_name)


def _expect_int(value: object, *, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise RetrievalArtifactSerializationError(f"{field_name} must be an integer.")
    return value


def _expect_int_tuple5(value: object, *, field_name: str) -> tuple[int, int, int, int, int]:
    if not isinstance(value, list) or len(value) != 5:
        raise RetrievalArtifactSerializationError(f"{field_name} must be a length-5 JSON array.")
    items = tuple(
        _expect_int(item, field_name=f"{field_name}[{index}]") for index, item in enumerate(value)
    )
    return (
        items[0],
        items[1],
        items[2],
        items[3],
        items[4],
    )


def _require_exact_fields(
    mapping: MappingLike,
    *,
    required_fields: frozenset[str],
    object_name: str,
) -> None:
    unknown_fields = set(mapping) - required_fields
    if unknown_fields:
        names = ", ".join(sorted(unknown_fields))
        raise RetrievalArtifactSerializationError(f"Unknown {object_name} field(s): {names}.")
    missing_fields = required_fields - set(mapping)
    if missing_fields:
        names = ", ".join(sorted(missing_fields))
        raise RetrievalArtifactSerializationError(f"Missing {object_name} field(s): {names}.")


__all__ = [
    "EMBEDDING_ARTIFACT_METADATA_SCHEMA_NAME",
    "EMBEDDING_ARTIFACT_METADATA_SCHEMA_VERSION",
    "EMBEDDING_ARTIFACT_SCHEMA_NAME",
    "EMBEDDING_ARTIFACT_SCHEMA_VERSION",
    "EMBEDDING_CACHE_ENTRY_SCHEMA_NAME",
    "EMBEDDING_CACHE_ENTRY_SCHEMA_VERSION",
    "EMBEDDING_CACHE_KEY_SCHEMA_NAME",
    "EMBEDDING_CACHE_KEY_SCHEMA_VERSION",
    "EMBEDDING_CACHE_MANIFEST_SCHEMA_NAME",
    "EMBEDDING_CACHE_MANIFEST_SCHEMA_VERSION",
    "EMBEDDING_TENSOR_DESCRIPTOR_SCHEMA_NAME",
    "EMBEDDING_TENSOR_DESCRIPTOR_SCHEMA_VERSION",
    "SUPPORTED_EMBEDDING_TENSOR_DTYPES",
    "EmbeddingArtifact",
    "EmbeddingArtifactMetadata",
    "EmbeddingCacheEntry",
    "EmbeddingCacheKey",
    "EmbeddingCacheManifest",
    "EmbeddingTensorDescriptor",
    "RetrievalArtifactError",
    "RetrievalArtifactHashError",
    "RetrievalArtifactSerializationError",
    "RetrievalArtifactValidationError",
    "embedding_artifact_from_mapping",
    "embedding_artifact_identity_payload",
    "embedding_artifact_metadata_from_mapping",
    "embedding_artifact_metadata_to_dict",
    "embedding_artifact_to_dict",
    "embedding_cache_entry_from_mapping",
    "embedding_cache_entry_to_dict",
    "embedding_cache_key_from_mapping",
    "embedding_cache_key_identity_payload",
    "embedding_cache_key_to_dict",
    "embedding_cache_manifest_from_mapping",
    "embedding_cache_manifest_identity_payload",
    "embedding_cache_manifest_to_dict",
    "embedding_tensor_descriptor_from_mapping",
    "embedding_tensor_descriptor_to_dict",
    "hash_embedding_artifact",
    "hash_embedding_cache_key",
    "hash_embedding_cache_manifest",
]
