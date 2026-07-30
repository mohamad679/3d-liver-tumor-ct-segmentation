"""Unit tests for deterministic Phase 5 embedding-cache publication."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from protoem_ct.artifacts.hashing import canonical_json_bytes, sha256_json
from protoem_ct.retrieval.artifacts import (
    EMBEDDING_ARTIFACT_METADATA_SCHEMA_NAME,
    EMBEDDING_ARTIFACT_METADATA_SCHEMA_VERSION,
    EMBEDDING_ARTIFACT_SCHEMA_NAME,
    EMBEDDING_ARTIFACT_SCHEMA_VERSION,
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
    RetrievalArtifactValidationError,
    embedding_artifact_from_mapping,
    embedding_artifact_to_dict,
    hash_embedding_artifact,
    hash_embedding_cache_manifest,
)
from protoem_ct.retrieval.cache import (
    DEFAULT_CACHE_MANIFEST_RELATIVE_PATH,
    EmbeddingCache,
    EmbeddingCacheIncompatibleEntryError,
    EmbeddingCacheInvalidEntryError,
    EmbeddingCacheUnsafePathError,
)


def _sha256(seed: str) -> str:
    return (seed * 64)[:64]


def _cache_key(
    *,
    input_identity: str = "query_case_001",
    tensor_dtype: str = "float32",
    feature_shape: tuple[int, int, int, int, int] = (1, 2, 3, 4, 5),
) -> EmbeddingCacheKey:
    payload = {
        "checkpoint_hash": _sha256("b"),
        "dataset_manifest_hash": _sha256("d"),
        "encoder_identity": "segresnet_encoder_v1",
        "feature_shape": list(feature_shape),
        "feature_stage": "final_encoder",
        "input_identity": input_identity,
        "normalization_name": "l2_channel",
        "preprocessing_hash": _sha256("a"),
        "schema_name": EMBEDDING_CACHE_KEY_SCHEMA_NAME,
        "schema_version": EMBEDDING_CACHE_KEY_SCHEMA_VERSION,
        "tensor_dtype": tensor_dtype,
    }
    return EmbeddingCacheKey(
        schema_name=EMBEDDING_CACHE_KEY_SCHEMA_NAME,
        schema_version=EMBEDDING_CACHE_KEY_SCHEMA_VERSION,
        cache_key_hash=sha256_json(payload),
        encoder_identity="segresnet_encoder_v1",
        preprocessing_hash=_sha256("a"),
        checkpoint_hash=_sha256("b"),
        input_identity=input_identity,
        dataset_manifest_hash=_sha256("d"),
        feature_stage="final_encoder",
        normalization_name="l2_channel",
        feature_shape=feature_shape,
        tensor_dtype=tensor_dtype,
    )


def _metadata(*, input_identity: str = "query_case_001") -> EmbeddingArtifactMetadata:
    return EmbeddingArtifactMetadata(
        schema_name=EMBEDDING_ARTIFACT_METADATA_SCHEMA_NAME,
        schema_version=EMBEDDING_ARTIFACT_METADATA_SCHEMA_VERSION,
        encoder_identity="segresnet_encoder_v1",
        preprocessing_hash=_sha256("a"),
        checkpoint_hash=_sha256("b"),
        input_identity=input_identity,
        dataset_manifest_hash=_sha256("d"),
        feature_stage="final_encoder",
        normalization_name="l2_channel",
    )


def _array(
    *,
    shape: tuple[int, int, int, int, int] = (1, 2, 3, 4, 5),
    dtype: str = "float32",
) -> np.ndarray:
    total = int(np.prod(shape))
    return np.arange(total, dtype=np.dtype(dtype)).reshape(shape)


def _cache(tmp_path: Path) -> EmbeddingCache:
    return EmbeddingCache((tmp_path / "embedding-cache").resolve())


def _load_artifact(path: Path) -> EmbeddingArtifact:
    payload = cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
    return embedding_artifact_from_mapping(payload)


def _rehash_artifact(
    artifact: EmbeddingArtifact,
    *,
    descriptor: EmbeddingTensorDescriptor | None = None,
) -> EmbeddingArtifact:
    updated_descriptor = artifact.tensor_descriptor if descriptor is None else descriptor
    draft = EmbeddingArtifact.__new__(EmbeddingArtifact)
    object.__setattr__(draft, "schema_name", EMBEDDING_ARTIFACT_SCHEMA_NAME)
    object.__setattr__(draft, "schema_version", EMBEDDING_ARTIFACT_SCHEMA_VERSION)
    object.__setattr__(draft, "artifact_hash", "0" * 64)
    object.__setattr__(draft, "cache_key_hash", artifact.cache_key_hash)
    object.__setattr__(draft, "metadata", artifact.metadata)
    object.__setattr__(draft, "tensor_descriptor", updated_descriptor)
    object.__setattr__(draft, "created_at_utc", artifact.created_at_utc)
    return EmbeddingArtifact(
        schema_name=artifact.schema_name,
        schema_version=artifact.schema_version,
        artifact_hash=hash_embedding_artifact(draft),
        cache_key_hash=artifact.cache_key_hash,
        metadata=artifact.metadata,
        tensor_descriptor=updated_descriptor,
        created_at_utc=artifact.created_at_utc,
    )


def _rewrite_artifact(path: Path, artifact: EmbeddingArtifact) -> None:
    path.write_bytes(canonical_json_bytes(embedding_artifact_to_dict(artifact)))


def _manifest(
    entries: tuple[EmbeddingCacheEntry, ...],
    *,
    created_at_utc: str | None = None,
) -> EmbeddingCacheManifest:
    draft = EmbeddingCacheManifest.__new__(EmbeddingCacheManifest)
    object.__setattr__(draft, "schema_name", EMBEDDING_CACHE_MANIFEST_SCHEMA_NAME)
    object.__setattr__(draft, "schema_version", EMBEDDING_CACHE_MANIFEST_SCHEMA_VERSION)
    object.__setattr__(draft, "artifact_hash", "0" * 64)
    object.__setattr__(
        draft,
        "entries",
        tuple(sorted(entries, key=lambda item: item.cache_key_hash)),
    )
    object.__setattr__(draft, "created_at_utc", created_at_utc)
    return EmbeddingCacheManifest(
        schema_name=EMBEDDING_CACHE_MANIFEST_SCHEMA_NAME,
        schema_version=EMBEDDING_CACHE_MANIFEST_SCHEMA_VERSION,
        artifact_hash=hash_embedding_cache_manifest(draft),
        entries=entries,
        created_at_utc=created_at_utc,
    )


def test_deterministic_cache_path(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    key = _cache_key()
    expected_relative_path = f"embeddings/{key.cache_key_hash[:2]}/{key.cache_key_hash}.npy"

    assert cache.relative_artifact_path_for_key(key) == expected_relative_path
    assert cache.resolve_tensor_path(key) == cache.cache_root / expected_relative_path


def test_successful_publish_read_round_trip(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    key = _cache_key()
    array = _array()

    entry = cache.publish_embedding(cache_key=key, metadata=_metadata(), array=array)
    loaded = cache.read_embedding(key)

    assert entry == loaded.entry
    assert loaded.artifact.cache_key_hash == key.cache_key_hash
    assert loaded.array.dtype == array.dtype
    assert loaded.array.shape == array.shape
    assert np.array_equal(loaded.array, array)


def test_byte_identical_metadata_regeneration(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    key = _cache_key()
    array = _array()

    cache.publish_embedding(cache_key=key, metadata=_metadata(), array=array)
    metadata_path = cache.resolve_metadata_path(key)
    first_bytes = metadata_path.read_bytes()

    cache.publish_embedding(cache_key=key, metadata=_metadata(), array=array)

    assert metadata_path.read_bytes() == first_bytes


def test_cache_hit_and_miss(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    hit_key = _cache_key()
    miss_key = _cache_key(input_identity="query_case_002")
    cache.publish_embedding(cache_key=hit_key, metadata=_metadata(), array=_array())

    assert cache.probe(hit_key).status == "hit"
    assert cache.probe(miss_key).status == "miss"


def test_incompatible_key_rejection(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    source_key = _cache_key()
    requested_key = _cache_key(input_identity="query_case_002")
    cache.publish_embedding(cache_key=source_key, metadata=_metadata(), array=_array())

    source_tensor_path = cache.resolve_tensor_path(source_key)
    source_metadata_path = cache.resolve_metadata_path(source_key)
    requested_tensor_path = cache.resolve_tensor_path(requested_key)
    requested_metadata_path = cache.resolve_metadata_path(requested_key)
    requested_tensor_path.parent.mkdir(parents=True, exist_ok=True)
    requested_tensor_path.write_bytes(source_tensor_path.read_bytes())

    source_artifact = _load_artifact(source_metadata_path)
    rewritten_descriptor = dataclasses.replace(
        source_artifact.tensor_descriptor,
        relative_artifact_path=cache.relative_artifact_path_for_key(requested_key),
    )
    _rewrite_artifact(
        requested_metadata_path,
        _rehash_artifact(source_artifact, descriptor=rewritten_descriptor),
    )

    with pytest.raises(EmbeddingCacheIncompatibleEntryError):
        cache.read_embedding(requested_key)


def test_corrupted_tensor_rejection(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    key = _cache_key()
    cache.publish_embedding(cache_key=key, metadata=_metadata(), array=_array())

    tensor_path = cache.resolve_tensor_path(key)
    tensor_path.write_bytes(b"not-a-npy-file")

    with pytest.raises(EmbeddingCacheInvalidEntryError, match="byte length|content SHA-256"):
        cache.read_embedding(key)


def test_corrupted_metadata_rejection(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    key = _cache_key()
    cache.publish_embedding(cache_key=key, metadata=_metadata(), array=_array())

    metadata_path = cache.resolve_metadata_path(key)
    metadata_path.write_text("{bad json", encoding="utf-8")

    with pytest.raises(EmbeddingCacheInvalidEntryError, match="metadata JSON is malformed"):
        cache.read_embedding(key)


def test_shape_mismatch_rejection(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    key = _cache_key()
    cache.publish_embedding(cache_key=key, metadata=_metadata(), array=_array())

    tensor_path = cache.resolve_tensor_path(key)
    metadata_path = cache.resolve_metadata_path(key)
    changed = _array(shape=(1, 2, 3, 5, 4))
    np.save(tensor_path, changed, allow_pickle=False)

    artifact = _load_artifact(metadata_path)
    updated_descriptor = dataclasses.replace(
        artifact.tensor_descriptor,
        tensor_byte_length=tensor_path.stat().st_size,
        tensor_content_sha256=hashlib.sha256(tensor_path.read_bytes()).hexdigest(),
    )
    _rewrite_artifact(metadata_path, _rehash_artifact(artifact, descriptor=updated_descriptor))

    with pytest.raises(EmbeddingCacheInvalidEntryError, match="shape does not match"):
        cache.read_embedding(key)


def test_dtype_mismatch_rejection(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    key = _cache_key()
    cache.publish_embedding(cache_key=key, metadata=_metadata(), array=_array())

    tensor_path = cache.resolve_tensor_path(key)
    metadata_path = cache.resolve_metadata_path(key)
    changed = _array(dtype="float64")
    np.save(tensor_path, changed, allow_pickle=False)

    artifact = _load_artifact(metadata_path)
    updated_descriptor = dataclasses.replace(
        artifact.tensor_descriptor,
        tensor_byte_length=tensor_path.stat().st_size,
        tensor_content_sha256=hashlib.sha256(tensor_path.read_bytes()).hexdigest(),
    )
    _rewrite_artifact(metadata_path, _rehash_artifact(artifact, descriptor=updated_descriptor))

    with pytest.raises(EmbeddingCacheInvalidEntryError, match="dtype does not match"):
        cache.read_embedding(key)


def test_byte_length_mismatch_rejection(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    key = _cache_key()
    cache.publish_embedding(cache_key=key, metadata=_metadata(), array=_array())

    metadata_path = cache.resolve_metadata_path(key)
    artifact = _load_artifact(metadata_path)
    updated_descriptor = dataclasses.replace(
        artifact.tensor_descriptor,
        tensor_byte_length=artifact.tensor_descriptor.tensor_byte_length + 1,
    )
    _rewrite_artifact(metadata_path, _rehash_artifact(artifact, descriptor=updated_descriptor))

    with pytest.raises(EmbeddingCacheInvalidEntryError, match="byte length"):
        cache.read_embedding(key)


def test_tensor_content_hash_mismatch_rejection(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    key = _cache_key()
    cache.publish_embedding(cache_key=key, metadata=_metadata(), array=_array())

    metadata_path = cache.resolve_metadata_path(key)
    artifact = _load_artifact(metadata_path)
    updated_descriptor = dataclasses.replace(
        artifact.tensor_descriptor,
        tensor_content_sha256=_sha256("f"),
    )
    _rewrite_artifact(metadata_path, _rehash_artifact(artifact, descriptor=updated_descriptor))

    with pytest.raises(EmbeddingCacheInvalidEntryError, match="content SHA-256"):
        cache.read_embedding(key)


def test_unsafe_path_rejection(tmp_path: Path) -> None:
    cache = _cache(tmp_path)

    with pytest.raises(EmbeddingCacheUnsafePathError):
        cache.resolve_manifest_path("../escape.json")


def test_symlink_escape_rejection(tmp_path: Path) -> None:
    real_root = tmp_path / "real"
    real_root.mkdir()
    symlink_root = tmp_path / "cache-link"
    symlink_root.symlink_to(real_root, target_is_directory=True)

    with pytest.raises(EmbeddingCacheUnsafePathError):
        EmbeddingCache(symlink_root)


def test_partial_publication_rejection(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    key = _cache_key()
    tensor_path = cache.resolve_tensor_path(key)
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(tensor_path, _array(), allow_pickle=False)

    with pytest.raises(EmbeddingCacheInvalidEntryError, match="partial"):
        cache.read_embedding(key)


def test_atomic_manifest_replacement_behavior(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    first_entry = cache.publish_embedding(
        cache_key=_cache_key(),
        metadata=_metadata(),
        array=_array(),
    )
    second_entry = cache.publish_embedding(
        cache_key=_cache_key(input_identity="query_case_002"),
        metadata=_metadata(input_identity="query_case_002"),
        array=_array(),
    )
    manifest_path = cache.publish_manifest(_manifest((first_entry,)))
    first_bytes = manifest_path.read_bytes()

    cache.publish_manifest(
        _manifest((second_entry, first_entry), created_at_utc="2026-07-30T18:10:13Z")
    )

    assert manifest_path.read_bytes() != first_bytes
    assert not any(path.name.endswith(".tmp") for path in manifest_path.parent.iterdir())


def test_deterministic_manifest_ordering_and_load(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    later = cache.publish_embedding(
        cache_key=_cache_key(input_identity="query_case_002"),
        metadata=_metadata(input_identity="query_case_002"),
        array=_array(),
    )
    earlier = cache.publish_embedding(cache_key=_cache_key(), metadata=_metadata(), array=_array())
    cache.publish_manifest(_manifest((later, earlier)))

    loaded = cache.load_manifest()

    assert tuple(entry.cache_key_hash for entry in loaded.entries) == tuple(
        sorted((later.cache_key_hash, earlier.cache_key_hash))
    )


def test_duplicate_manifest_rejection(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    entry = cache.publish_embedding(cache_key=_cache_key(), metadata=_metadata(), array=_array())

    with pytest.raises(RetrievalArtifactValidationError, match="unique cache_key_hash"):
        _manifest((entry, entry))


def test_no_pickle_object_array_loading(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    key = _cache_key()
    cache.publish_embedding(cache_key=key, metadata=_metadata(), array=_array())

    tensor_path = cache.resolve_tensor_path(key)
    metadata_path = cache.resolve_metadata_path(key)

    object_array = np.empty((1, 2, 3, 4, 5), dtype=object)
    object_array.fill("x")
    np.save(tensor_path, object_array, allow_pickle=True)

    artifact = _load_artifact(metadata_path)
    descriptor = EmbeddingTensorDescriptor(
        schema_name=EMBEDDING_TENSOR_DESCRIPTOR_SCHEMA_NAME,
        schema_version=EMBEDDING_TENSOR_DESCRIPTOR_SCHEMA_VERSION,
        feature_shape=(1, 2, 3, 4, 5),
        tensor_dtype="float32",
        tensor_byte_length=tensor_path.stat().st_size,
        tensor_content_sha256=hashlib.sha256(tensor_path.read_bytes()).hexdigest(),
        relative_artifact_path=cache.relative_artifact_path_for_key(key),
    )
    _rewrite_artifact(metadata_path, _rehash_artifact(artifact, descriptor=descriptor))

    with pytest.raises(EmbeddingCacheInvalidEntryError, match="pickle|object array"):
        cache.read_embedding(key)


def test_cache_root_must_be_explicit_absolute(tmp_path: Path) -> None:
    with pytest.raises(EmbeddingCacheUnsafePathError, match="absolute"):
        EmbeddingCache(Path("relative-cache"))

    cache = _cache(tmp_path)
    assert cache.resolve_manifest_path(DEFAULT_CACHE_MANIFEST_RELATIVE_PATH).is_absolute()
