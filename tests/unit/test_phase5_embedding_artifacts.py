"""Unit tests for Phase 5 embedding artifact and cache contracts."""

from __future__ import annotations

import dataclasses
import sys

import pytest

from protoem_ct.artifacts.hashing import sha256_json
from protoem_ct.retrieval.artifacts import (
    EMBEDDING_ARTIFACT_METADATA_SCHEMA_NAME,
    EMBEDDING_ARTIFACT_METADATA_SCHEMA_VERSION,
    EMBEDDING_ARTIFACT_SCHEMA_NAME,
    EMBEDDING_ARTIFACT_SCHEMA_VERSION,
    EMBEDDING_CACHE_ENTRY_SCHEMA_NAME,
    EMBEDDING_CACHE_ENTRY_SCHEMA_VERSION,
    EMBEDDING_CACHE_KEY_SCHEMA_NAME,
    EMBEDDING_CACHE_KEY_SCHEMA_VERSION,
    EMBEDDING_CACHE_MANIFEST_SCHEMA_NAME,
    EMBEDDING_CACHE_MANIFEST_SCHEMA_VERSION,
    EMBEDDING_TENSOR_DESCRIPTOR_SCHEMA_NAME,
    EMBEDDING_TENSOR_DESCRIPTOR_SCHEMA_VERSION,
    EmbeddingArtifact,
    EmbeddingArtifactMetadata,
    EmbeddingCacheEntry,
    EmbeddingCacheKey,
    EmbeddingCacheManifest,
    EmbeddingTensorDescriptor,
    RetrievalArtifactHashError,
    RetrievalArtifactSerializationError,
    RetrievalArtifactValidationError,
    embedding_artifact_from_mapping,
    embedding_artifact_identity_payload,
    embedding_artifact_metadata_to_dict,
    embedding_artifact_to_dict,
    embedding_cache_entry_to_dict,
    embedding_cache_key_from_mapping,
    embedding_cache_key_to_dict,
    embedding_cache_manifest_from_mapping,
    embedding_cache_manifest_identity_payload,
    embedding_cache_manifest_to_dict,
    hash_embedding_artifact,
    hash_embedding_cache_key,
)


def _sha256(seed: str) -> str:
    return (seed * 64)[:64]


def _cache_key_hash_payload(
    *,
    encoder_identity: str = "segresnet_encoder_v1",
    preprocessing_hash: str = _sha256("a"),
    checkpoint_hash: str = _sha256("b"),
    input_identity: str = "query_case_001",
    dataset_manifest_hash: str | None = _sha256("d"),
    feature_stage: str = "final_encoder",
    normalization_name: str = "l2_channel",
    feature_shape: tuple[int, int, int, int, int] = (1, 32, 16, 12, 8),
    tensor_dtype: str = "float32",
) -> str:
    return sha256_json(
        {
            "checkpoint_hash": checkpoint_hash,
            "dataset_manifest_hash": dataset_manifest_hash,
            "encoder_identity": encoder_identity,
            "feature_shape": list(feature_shape),
            "feature_stage": feature_stage,
            "input_identity": input_identity,
            "normalization_name": normalization_name,
            "preprocessing_hash": preprocessing_hash,
            "schema_name": EMBEDDING_CACHE_KEY_SCHEMA_NAME,
            "schema_version": EMBEDDING_CACHE_KEY_SCHEMA_VERSION,
            "tensor_dtype": tensor_dtype,
        }
    )


def _metadata(*, dataset_manifest_hash: str | None = _sha256("d")) -> EmbeddingArtifactMetadata:
    return EmbeddingArtifactMetadata(
        schema_name=EMBEDDING_ARTIFACT_METADATA_SCHEMA_NAME,
        schema_version=EMBEDDING_ARTIFACT_METADATA_SCHEMA_VERSION,
        encoder_identity="segresnet_encoder_v1",
        preprocessing_hash=_sha256("a"),
        checkpoint_hash=_sha256("b"),
        input_identity="query_case_001",
        dataset_manifest_hash=dataset_manifest_hash,
        feature_stage="final_encoder",
        normalization_name="l2_channel",
    )


def _descriptor(
    *,
    feature_shape: tuple[int, int, int, int, int] = (1, 32, 16, 12, 8),
    tensor_dtype: str = "float32",
    relative_artifact_path: str = "embeddings/query_case_001.bin",
    tensor_content_sha256: str = _sha256("c"),
) -> EmbeddingTensorDescriptor:
    return EmbeddingTensorDescriptor(
        schema_name=EMBEDDING_TENSOR_DESCRIPTOR_SCHEMA_NAME,
        schema_version=EMBEDDING_TENSOR_DESCRIPTOR_SCHEMA_VERSION,
        feature_shape=feature_shape,
        tensor_dtype=tensor_dtype,
        tensor_byte_length=49152,
        tensor_content_sha256=tensor_content_sha256,
        relative_artifact_path=relative_artifact_path,
    )


def _cache_key(
    *,
    encoder_identity: str = "segresnet_encoder_v1",
    preprocessing_hash: str = _sha256("a"),
    checkpoint_hash: str = _sha256("b"),
    input_identity: str = "query_case_001",
    dataset_manifest_hash: str | None = _sha256("d"),
    feature_stage: str = "final_encoder",
    normalization_name: str = "l2_channel",
    feature_shape: tuple[int, int, int, int, int] = (1, 32, 16, 12, 8),
    tensor_dtype: str = "float32",
) -> EmbeddingCacheKey:
    return EmbeddingCacheKey(
        schema_name=EMBEDDING_CACHE_KEY_SCHEMA_NAME,
        schema_version=EMBEDDING_CACHE_KEY_SCHEMA_VERSION,
        cache_key_hash=_cache_key_hash_payload(
            encoder_identity=encoder_identity,
            preprocessing_hash=preprocessing_hash,
            checkpoint_hash=checkpoint_hash,
            input_identity=input_identity,
            dataset_manifest_hash=dataset_manifest_hash,
            feature_stage=feature_stage,
            normalization_name=normalization_name,
            feature_shape=feature_shape,
            tensor_dtype=tensor_dtype,
        ),
        encoder_identity=encoder_identity,
        preprocessing_hash=preprocessing_hash,
        checkpoint_hash=checkpoint_hash,
        input_identity=input_identity,
        dataset_manifest_hash=dataset_manifest_hash,
        feature_stage=feature_stage,
        normalization_name=normalization_name,
        feature_shape=feature_shape,
        tensor_dtype=tensor_dtype,
    )


def _artifact(
    *,
    cache_key_hash: str | None = None,
    metadata: EmbeddingArtifactMetadata | None = None,
    descriptor: EmbeddingTensorDescriptor | None = None,
    created_at_utc: str | None = "2026-07-30T18:10:11Z",
) -> EmbeddingArtifact:
    resolved_metadata = _metadata() if metadata is None else metadata
    resolved_descriptor = _descriptor() if descriptor is None else descriptor
    resolved_cache_key_hash = (
        _cache_key().cache_key_hash if cache_key_hash is None else cache_key_hash
    )
    payload = {
        "cache_key_hash": resolved_cache_key_hash,
        "metadata": embedding_artifact_metadata_to_dict(resolved_metadata),
        "schema_name": EMBEDDING_ARTIFACT_SCHEMA_NAME,
        "schema_version": EMBEDDING_ARTIFACT_SCHEMA_VERSION,
        "tensor_descriptor": {
            "feature_shape": list(resolved_descriptor.feature_shape),
            "relative_artifact_path": resolved_descriptor.relative_artifact_path,
            "schema_name": resolved_descriptor.schema_name,
            "schema_version": resolved_descriptor.schema_version,
            "tensor_byte_length": resolved_descriptor.tensor_byte_length,
            "tensor_content_sha256": resolved_descriptor.tensor_content_sha256,
            "tensor_dtype": resolved_descriptor.tensor_dtype,
        },
    }
    return EmbeddingArtifact(
        schema_name=EMBEDDING_ARTIFACT_SCHEMA_NAME,
        schema_version=EMBEDDING_ARTIFACT_SCHEMA_VERSION,
        artifact_hash=sha256_json(payload),
        cache_key_hash=resolved_cache_key_hash,
        metadata=resolved_metadata,
        tensor_descriptor=resolved_descriptor,
        created_at_utc=created_at_utc,
    )


def _entry(
    *,
    cache_key: EmbeddingCacheKey | None = None,
    artifact_hash: str | None = None,
    relative_artifact_path: str = "embeddings/query_case_001.bin",
    created_at_utc: str | None = "2026-07-30T18:10:12Z",
) -> EmbeddingCacheEntry:
    resolved_cache_key = _cache_key() if cache_key is None else cache_key
    resolved_artifact_hash = _artifact(
        cache_key_hash=resolved_cache_key.cache_key_hash
    ).artifact_hash
    if artifact_hash is not None:
        resolved_artifact_hash = artifact_hash
    return EmbeddingCacheEntry(
        schema_name=EMBEDDING_CACHE_ENTRY_SCHEMA_NAME,
        schema_version=EMBEDDING_CACHE_ENTRY_SCHEMA_VERSION,
        cache_key=resolved_cache_key,
        cache_key_hash=resolved_cache_key.cache_key_hash,
        artifact_hash=resolved_artifact_hash,
        relative_artifact_path=relative_artifact_path,
        created_at_utc=created_at_utc,
    )


def _manifest(*, entries: tuple[EmbeddingCacheEntry, ...]) -> EmbeddingCacheManifest:
    payload = {
        "entries": [
            {
                "artifact_hash": item.artifact_hash,
                "cache_key": embedding_cache_key_to_dict(item.cache_key),
                "cache_key_hash": item.cache_key_hash,
                "relative_artifact_path": item.relative_artifact_path,
                "schema_name": item.schema_name,
                "schema_version": item.schema_version,
            }
            for item in sorted(entries, key=lambda item: item.cache_key_hash)
        ],
        "schema_name": EMBEDDING_CACHE_MANIFEST_SCHEMA_NAME,
        "schema_version": EMBEDDING_CACHE_MANIFEST_SCHEMA_VERSION,
    }
    return EmbeddingCacheManifest(
        schema_name=EMBEDDING_CACHE_MANIFEST_SCHEMA_NAME,
        schema_version=EMBEDDING_CACHE_MANIFEST_SCHEMA_VERSION,
        artifact_hash=sha256_json(payload),
        entries=entries,
        created_at_utc="2026-07-30T18:10:13Z",
    )


def test_canonical_artifact_hash_stability() -> None:
    artifact = _artifact()

    assert hash_embedding_artifact(artifact) == hash_embedding_artifact(_artifact())
    assert artifact.artifact_hash == hash_embedding_artifact(artifact)


def test_canonical_cache_key_hash_stability() -> None:
    cache_key = _cache_key()

    assert hash_embedding_cache_key(cache_key) == hash_embedding_cache_key(_cache_key())
    assert cache_key.cache_key_hash == hash_embedding_cache_key(cache_key)


@pytest.mark.parametrize(
    ("field_name", "builder"),
    [
        ("preprocessing_hash", lambda: _cache_key(preprocessing_hash=_sha256("f"))),
        ("checkpoint_hash", lambda: _cache_key(checkpoint_hash=_sha256("e"))),
        ("encoder_identity", lambda: _cache_key(encoder_identity="segresnet_encoder_v2")),
        ("input_identity", lambda: _cache_key(input_identity="query_case_002")),
        ("dataset_manifest_hash", lambda: _cache_key(dataset_manifest_hash=_sha256("9"))),
        ("feature_shape", lambda: _cache_key(feature_shape=(1, 16, 8, 6, 4))),
        ("tensor_dtype", lambda: _cache_key(tensor_dtype="float16")),
    ],
)
def test_compatibility_critical_field_changes_alter_cache_key(
    field_name: str,
    builder: object,
) -> None:
    baseline = _cache_key()
    variant = builder()  # type: ignore[operator]

    assert baseline.cache_key_hash != variant.cache_key_hash, field_name


def test_created_timestamp_does_not_alter_artifact_identity() -> None:
    first = _artifact(created_at_utc="2026-07-30T18:10:11Z")
    second = _artifact(created_at_utc="2027-01-01T00:00:00Z")

    assert first.artifact_hash == second.artifact_hash
    assert embedding_artifact_identity_payload(first) == embedding_artifact_identity_payload(second)


def test_mapping_round_trip() -> None:
    manifest = _manifest(
        entries=(
            _entry(),
            _entry(cache_key=_cache_key(input_identity="query_case_002")),
        )
    )

    round_tripped = embedding_cache_manifest_from_mapping(
        embedding_cache_manifest_to_dict(manifest)
    )

    assert round_tripped == manifest


def test_unknown_field_is_rejected() -> None:
    payload = embedding_cache_key_to_dict(_cache_key())
    payload["unexpected"] = "value"

    with pytest.raises(RetrievalArtifactSerializationError, match="Unknown EmbeddingCacheKey"):
        embedding_cache_key_from_mapping(payload)


def test_malformed_hash_is_rejected() -> None:
    with pytest.raises(RetrievalArtifactValidationError, match="cache_key_hash"):
        dataclasses.replace(_cache_key(), cache_key_hash="not-a-sha256")


def test_invalid_5d_shape_is_rejected() -> None:
    with pytest.raises(RetrievalArtifactValidationError, match="feature_shape.D"):
        _descriptor(feature_shape=(1, 32, 0, 12, 8))


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "/absolute/path.bin",
        "../escape.bin",
        "embeddings/../escape.bin",
        "embeddings//double.bin",
        "embeddings\\windows.bin",
        "~/home.bin",
    ],
)
def test_unsafe_relative_path_is_rejected(unsafe_path: str) -> None:
    with pytest.raises(RetrievalArtifactValidationError, match="relative_artifact_path"):
        _descriptor(relative_artifact_path=unsafe_path)


def test_duplicate_manifest_entry_is_rejected() -> None:
    entry = _entry()
    payload = {
        "entries": [embedding_cache_entry_to_dict(entry), embedding_cache_entry_to_dict(entry)],
        "schema_name": EMBEDDING_CACHE_MANIFEST_SCHEMA_NAME,
        "schema_version": EMBEDDING_CACHE_MANIFEST_SCHEMA_VERSION,
    }

    with pytest.raises(RetrievalArtifactValidationError, match="unique cache_key_hash"):
        EmbeddingCacheManifest(
            schema_name=EMBEDDING_CACHE_MANIFEST_SCHEMA_NAME,
            schema_version=EMBEDDING_CACHE_MANIFEST_SCHEMA_VERSION,
            artifact_hash=sha256_json(payload),
            entries=(entry, entry),
        )


def test_manifest_entries_are_sorted_deterministically() -> None:
    later = _entry(cache_key=_cache_key(input_identity="query_case_002"))
    earlier = _entry(cache_key=_cache_key(input_identity="query_case_001"))
    manifest = _manifest(entries=(later, earlier))

    assert tuple(item.cache_key_hash for item in manifest.entries) == tuple(
        sorted((later.cache_key_hash, earlier.cache_key_hash))
    )


def test_artifact_self_hash_consistency_is_enforced() -> None:
    artifact = _artifact()
    payload = embedding_artifact_to_dict(artifact)
    payload["artifact_hash"] = _sha256("0")

    with pytest.raises(RetrievalArtifactHashError, match="artifact_hash"):
        embedding_artifact_from_mapping(payload)


def test_cache_key_self_hash_consistency_is_enforced() -> None:
    payload = embedding_cache_key_to_dict(_cache_key())
    payload["cache_key_hash"] = _sha256("1")

    with pytest.raises(RetrievalArtifactHashError, match="cache_key_hash"):
        embedding_cache_key_from_mapping(payload)


def test_manifest_hash_ignores_entry_timestamps() -> None:
    first = _entry(created_at_utc="2026-07-30T18:10:12Z")
    second = _entry(created_at_utc="2030-01-01T00:00:00Z")

    assert first.cache_key_hash == second.cache_key_hash
    manifest_one = _manifest(entries=(first,))
    manifest_two = _manifest(entries=(second,))

    assert manifest_one.artifact_hash == manifest_two.artifact_hash
    assert embedding_cache_manifest_identity_payload(
        manifest_one
    ) == embedding_cache_manifest_identity_payload(manifest_two)


def test_no_torch_dependency_required() -> None:
    assert "torch" not in sys.modules
