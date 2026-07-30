"""Deterministic external embedding-cache publication and validation."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from protoem_ct.artifacts.hashing import canonical_json_bytes, sha256_file, sha256_json
from protoem_ct.retrieval.artifacts import (
    EMBEDDING_ARTIFACT_METADATA_SCHEMA_NAME,
    EMBEDDING_ARTIFACT_METADATA_SCHEMA_VERSION,
    EMBEDDING_ARTIFACT_SCHEMA_NAME,
    EMBEDDING_ARTIFACT_SCHEMA_VERSION,
    EMBEDDING_CACHE_ENTRY_SCHEMA_NAME,
    EMBEDDING_CACHE_ENTRY_SCHEMA_VERSION,
    EMBEDDING_CACHE_KEY_SCHEMA_NAME,
    EMBEDDING_CACHE_KEY_SCHEMA_VERSION,
    EMBEDDING_TENSOR_DESCRIPTOR_SCHEMA_NAME,
    EMBEDDING_TENSOR_DESCRIPTOR_SCHEMA_VERSION,
    EmbeddingArtifact,
    EmbeddingArtifactMetadata,
    EmbeddingCacheEntry,
    EmbeddingCacheKey,
    EmbeddingCacheManifest,
    EmbeddingTensorDescriptor,
    RetrievalArtifactError,
    embedding_artifact_from_mapping,
    embedding_artifact_to_dict,
    embedding_cache_manifest_from_mapping,
    embedding_cache_manifest_to_dict,
)

DEFAULT_CACHE_MANIFEST_RELATIVE_PATH: Final[str] = "manifests/embedding_cache_manifest.json"
_CACHE_TENSOR_DIR: Final[str] = "embeddings"
_METADATA_SUFFIX: Final[str] = ".artifact.json"
_PROBE_STATUSES: Final[frozenset[str]] = frozenset({"hit", "miss", "invalid", "incompatible"})

_DTYPE_BY_NAME: dict[str, np.dtype[np.generic]] = {
    "float16": np.dtype(np.float16),
    "float32": np.dtype(np.float32),
    "float64": np.dtype(np.float64),
}
with suppress(TypeError):
    _DTYPE_BY_NAME["bfloat16"] = np.dtype("bfloat16")


class EmbeddingCacheError(ValueError):
    """Base error for deterministic Phase 5 embedding-cache operations."""


class EmbeddingCacheMissError(EmbeddingCacheError):
    """Raised when no cache entry exists for a requested deterministic cache key."""


class EmbeddingCacheInvalidEntryError(EmbeddingCacheError):
    """Raised when a persisted cache entry is corrupt, partial, or malformed."""


class EmbeddingCacheIncompatibleEntryError(EmbeddingCacheError):
    """Raised when a persisted cache entry does not match the requested cache key."""


class EmbeddingCacheUnsafePathError(EmbeddingCacheError):
    """Raised when a cache root or derived cache path is unsafe."""


class EmbeddingCachePublicationError(EmbeddingCacheError):
    """Raised when deterministic cache publication cannot complete safely."""


@dataclass(frozen=True, slots=True)
class EmbeddingCacheProbeResult:
    """Deterministic cache lookup status for a single cache key."""

    status: str
    cache_key: EmbeddingCacheKey
    tensor_path: Path
    metadata_path: Path
    message: str | None = None

    def __post_init__(self) -> None:
        if self.status not in _PROBE_STATUSES:
            raise EmbeddingCacheError(
                "probe status must be one of hit, miss, invalid, incompatible."
            )


@dataclass(frozen=True, slots=True)
class CachedEmbedding:
    """Validated cached embedding payload plus its deterministic metadata."""

    cache_key: EmbeddingCacheKey
    entry: EmbeddingCacheEntry
    artifact: EmbeddingArtifact
    array: np.ndarray
    tensor_path: Path
    metadata_path: Path


class EmbeddingCache:
    """External deterministic embedding cache rooted at one explicit absolute directory."""

    def __init__(self, cache_root: Path) -> None:
        self._cache_root = _validate_cache_root(cache_root)

    @property
    def cache_root(self) -> Path:
        """Return the validated explicit cache root."""

        return self._cache_root

    def relative_artifact_path_for_key(self, cache_key: EmbeddingCacheKey) -> str:
        """Return the deterministic relative tensor path for a cache key."""

        prefix = cache_key.cache_key_hash[:2]
        return f"{_CACHE_TENSOR_DIR}/{prefix}/{cache_key.cache_key_hash}.npy"

    def resolve_tensor_path(self, cache_key: EmbeddingCacheKey) -> Path:
        """Resolve the deterministic tensor path for a cache key."""

        return self._resolve_relative_path(self.relative_artifact_path_for_key(cache_key))

    def resolve_metadata_path(self, cache_key: EmbeddingCacheKey) -> Path:
        """Resolve the deterministic metadata sidecar path for a cache key."""

        tensor_path = self.resolve_tensor_path(cache_key)
        return tensor_path.with_name(f"{tensor_path.stem}{_METADATA_SUFFIX}")

    def resolve_manifest_path(
        self,
        relative_path: str = DEFAULT_CACHE_MANIFEST_RELATIVE_PATH,
    ) -> Path:
        """Resolve a manifest path beneath the explicit cache root."""

        return self._resolve_relative_path(relative_path)

    def probe(self, cache_key: EmbeddingCacheKey) -> EmbeddingCacheProbeResult:
        """Return hit, miss, invalid, or incompatible status for a cache key."""

        tensor_path = self.resolve_tensor_path(cache_key)
        metadata_path = self.resolve_metadata_path(cache_key)
        try:
            self.validate_entry(cache_key)
        except EmbeddingCacheMissError as exc:
            return EmbeddingCacheProbeResult(
                "miss",
                cache_key,
                tensor_path,
                metadata_path,
                str(exc),
            )
        except EmbeddingCacheInvalidEntryError as exc:
            return EmbeddingCacheProbeResult(
                "invalid",
                cache_key,
                tensor_path,
                metadata_path,
                str(exc),
            )
        except EmbeddingCacheIncompatibleEntryError as exc:
            return EmbeddingCacheProbeResult(
                "incompatible",
                cache_key,
                tensor_path,
                metadata_path,
                str(exc),
            )
        return EmbeddingCacheProbeResult("hit", cache_key, tensor_path, metadata_path, None)

    def publish_embedding(
        self,
        *,
        cache_key: EmbeddingCacheKey,
        metadata: EmbeddingArtifactMetadata,
        array: np.ndarray,
        created_at_utc: str | None = None,
    ) -> EmbeddingCacheEntry:
        """Publish one validated embedding tensor and JSON metadata sidecar."""

        _require_metadata_matches_cache_key(metadata, cache_key)
        normalized_array = _normalize_array(array, cache_key)
        tensor_path = self.resolve_tensor_path(cache_key)
        metadata_path = self.resolve_metadata_path(cache_key)
        relative_path = self.relative_artifact_path_for_key(cache_key)
        artifact = _build_artifact(
            cache_key=cache_key,
            metadata=metadata,
            array=normalized_array,
            relative_artifact_path=relative_path,
            created_at_utc=created_at_utc,
        )

        if tensor_path.exists() or metadata_path.exists():
            try:
                cached = self.read_embedding(cache_key)
            except EmbeddingCacheMissError as exc:
                raise EmbeddingCachePublicationError(
                    "refusing to publish over a partial cache entry."
                ) from exc
            except (EmbeddingCacheInvalidEntryError, EmbeddingCacheIncompatibleEntryError) as exc:
                raise EmbeddingCachePublicationError(
                    "refusing to overwrite an invalid or incompatible existing cache entry."
                ) from exc
            if cached.artifact != artifact:
                raise EmbeddingCachePublicationError(
                    "refusing to overwrite an existing valid incompatible cache artifact."
                )
            return cached.entry

        tensor_bytes = _serialize_array_bytes(normalized_array)
        metadata_bytes = canonical_json_bytes(embedding_artifact_to_dict(artifact))
        try:
            self._publish_bytes_atomically(tensor_path, tensor_bytes, allow_replace=False)
            self._publish_bytes_atomically(metadata_path, metadata_bytes, allow_replace=False)
        except (EmbeddingCachePublicationError, OSError) as exc:
            with suppress(OSError):
                if tensor_path.exists() and not metadata_path.exists():
                    tensor_path.unlink()
            raise EmbeddingCachePublicationError(
                "failed to publish embedding cache entry."
            ) from exc
        return _entry_from_artifact(cache_key=cache_key, artifact=artifact)

    def read_embedding(self, cache_key: EmbeddingCacheKey) -> CachedEmbedding:
        """Read one validated cached embedding for a deterministic cache key."""

        tensor_path = self.resolve_tensor_path(cache_key)
        metadata_path = self.resolve_metadata_path(cache_key)
        _require_file_pair_state(tensor_path, metadata_path)
        artifact = _load_artifact_json(metadata_path)
        expected_relative_path = self.relative_artifact_path_for_key(cache_key)
        if artifact.tensor_descriptor.relative_artifact_path != expected_relative_path:
            raise EmbeddingCacheInvalidEntryError(
                "artifact relative path does not match the deterministic cache path."
            )
        try:
            stored_key = _cache_key_from_artifact(artifact)
        except RetrievalArtifactError as exc:
            raise EmbeddingCacheIncompatibleEntryError(
                "stored artifact metadata is incompatible with its cache-key hash."
            ) from exc
        if stored_key != cache_key:
            raise EmbeddingCacheIncompatibleEntryError(
                "stored embedding artifact is incompatible with the requested cache key."
            )

        if tensor_path.stat().st_size != artifact.tensor_descriptor.tensor_byte_length:
            raise EmbeddingCacheInvalidEntryError(
                "tensor byte length does not match the persisted descriptor."
            )
        if sha256_file(tensor_path) != artifact.tensor_descriptor.tensor_content_sha256:
            raise EmbeddingCacheInvalidEntryError(
                "tensor content SHA-256 does not match the persisted descriptor."
            )
        array = _load_numpy_tensor(tensor_path)
        expected_dtype = _dtype_for_name(cache_key.tensor_dtype)
        if array.dtype != expected_dtype:
            raise EmbeddingCacheInvalidEntryError("tensor dtype does not match the cache contract.")
        if tuple(int(axis) for axis in array.shape) != cache_key.feature_shape:
            raise EmbeddingCacheInvalidEntryError("tensor shape does not match the cache contract.")

        entry = _entry_from_artifact(cache_key=cache_key, artifact=artifact)
        return CachedEmbedding(cache_key, entry, artifact, array, tensor_path, metadata_path)

    def validate_entry(self, cache_key: EmbeddingCacheKey) -> EmbeddingCacheEntry:
        """Validate one persisted cache entry and return its manifest-ready entry."""

        return self.read_embedding(cache_key).entry

    def publish_manifest(
        self,
        manifest: EmbeddingCacheManifest,
        *,
        relative_path: str = DEFAULT_CACHE_MANIFEST_RELATIVE_PATH,
    ) -> Path:
        """Publish one deterministic cache manifest under the explicit cache root."""

        path = self.resolve_manifest_path(relative_path)
        payload = canonical_json_bytes(embedding_cache_manifest_to_dict(manifest))
        try:
            self._publish_bytes_atomically(path, payload, allow_replace=True)
        except (EmbeddingCachePublicationError, OSError) as exc:
            raise EmbeddingCachePublicationError("failed to publish cache manifest.") from exc
        return path

    def load_manifest(
        self,
        *,
        relative_path: str = DEFAULT_CACHE_MANIFEST_RELATIVE_PATH,
    ) -> EmbeddingCacheManifest:
        """Load one deterministic cache manifest from the explicit cache root."""

        path = self.resolve_manifest_path(relative_path)
        if not path.exists():
            raise EmbeddingCacheMissError("cache manifest is missing.")
        if path.is_symlink():
            raise EmbeddingCacheUnsafePathError("cache manifest path must not be a symlink.")
        if not path.is_file():
            raise EmbeddingCacheInvalidEntryError("cache manifest must be a regular file.")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise EmbeddingCacheInvalidEntryError("cache manifest JSON is malformed.") from exc
        try:
            return embedding_cache_manifest_from_mapping(_require_mapping(payload, "manifest"))
        except RetrievalArtifactError as exc:
            raise EmbeddingCacheInvalidEntryError("cache manifest content is invalid.") from exc

    def _resolve_relative_path(self, relative_path: str) -> Path:
        normalized = _validate_relative_path(relative_path)
        root = self._cache_root
        resolved_root = root.resolve(strict=True)
        candidate = root / normalized
        current = root
        for component in Path(normalized).parts[:-1]:
            current = current / component
            if current.exists() and current.is_symlink():
                raise EmbeddingCacheUnsafePathError(
                    "cache path must not traverse symlinked directories."
                )
        try:
            resolved_parent = candidate.parent.resolve(strict=False)
        except OSError as exc:
            raise EmbeddingCacheUnsafePathError("cache path parent cannot be resolved.") from exc
        if resolved_parent != resolved_root and resolved_root not in resolved_parent.parents:
            raise EmbeddingCacheUnsafePathError("cache path escapes the explicit cache root.")
        return candidate

    def _publish_bytes_atomically(self, path: Path, payload: bytes, *, allow_replace: bool) -> None:
        _ensure_parent_directory(self._cache_root, path.parent)
        if path.exists() and not allow_replace:
            raise EmbeddingCachePublicationError(f"cache output already exists: {path}")
        temp_path: Path
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(payload)
        try:
            if path.exists() and not allow_replace:
                raise EmbeddingCachePublicationError(f"cache output already exists: {path}")
            os.replace(temp_path, path)
        finally:
            with suppress(OSError):
                if temp_path.exists():
                    temp_path.unlink()


def _validate_cache_root(cache_root: Path) -> Path:
    if str(cache_root) == "":
        raise EmbeddingCacheUnsafePathError("cache root must not be empty.")
    if not cache_root.is_absolute():
        raise EmbeddingCacheUnsafePathError("cache root must be explicit and absolute.")
    if any(part == ".." for part in cache_root.parts):
        raise EmbeddingCacheUnsafePathError("cache root must not contain parent traversal.")
    if cache_root.exists() and cache_root.is_symlink():
        raise EmbeddingCacheUnsafePathError("cache root must not be a symlink.")
    cache_root.mkdir(parents=True, exist_ok=True)
    resolved_root = cache_root.resolve(strict=True)
    if not resolved_root.is_dir():
        raise EmbeddingCacheUnsafePathError("cache root must resolve to a directory.")
    return resolved_root


def _validate_relative_path(relative_path: str) -> str:
    if relative_path == "":
        raise EmbeddingCacheUnsafePathError("relative cache path must not be empty.")
    if "\\" in relative_path:
        raise EmbeddingCacheUnsafePathError("relative cache path must not contain backslashes.")
    if relative_path.startswith("~"):
        raise EmbeddingCacheUnsafePathError("relative cache path must not be home-relative.")
    candidate = Path(relative_path)
    if candidate.is_absolute():
        raise EmbeddingCacheUnsafePathError("relative cache path must not be absolute.")
    if any(part in {"", ".", ".."} for part in candidate.parts):
        raise EmbeddingCacheUnsafePathError(
            "relative cache path must not contain empty or traversal segments."
        )
    normalized = candidate.as_posix()
    if normalized != relative_path:
        raise EmbeddingCacheUnsafePathError("relative cache path must use normalized POSIX syntax.")
    return normalized


def _ensure_parent_directory(cache_root: Path, parent: Path) -> None:
    resolved_root = cache_root.resolve(strict=True)
    current = cache_root
    for component in parent.relative_to(cache_root).parts:
        current = current / component
        if current.exists():
            if current.is_symlink():
                raise EmbeddingCacheUnsafePathError(
                    "cache publication must not traverse symlinked directories."
                )
            if current.is_file():
                raise EmbeddingCacheUnsafePathError(
                    "cache publication parent must not be a regular file."
                )
            continue
        current.mkdir()
        resolved_current = current.resolve(strict=True)
        if resolved_current != resolved_root and resolved_root not in resolved_current.parents:
            raise EmbeddingCacheUnsafePathError("cache publication path escapes the cache root.")


def _require_metadata_matches_cache_key(
    metadata: EmbeddingArtifactMetadata,
    cache_key: EmbeddingCacheKey,
) -> None:
    if metadata.schema_name != EMBEDDING_ARTIFACT_METADATA_SCHEMA_NAME:
        raise EmbeddingCachePublicationError("metadata schema_name is invalid.")
    if metadata.schema_version != EMBEDDING_ARTIFACT_METADATA_SCHEMA_VERSION:
        raise EmbeddingCachePublicationError("metadata schema_version is invalid.")
    comparisons = (
        ("encoder_identity", metadata.encoder_identity, cache_key.encoder_identity),
        ("preprocessing_hash", metadata.preprocessing_hash, cache_key.preprocessing_hash),
        ("checkpoint_hash", metadata.checkpoint_hash, cache_key.checkpoint_hash),
        ("input_identity", metadata.input_identity, cache_key.input_identity),
        ("dataset_manifest_hash", metadata.dataset_manifest_hash, cache_key.dataset_manifest_hash),
        ("feature_stage", metadata.feature_stage, cache_key.feature_stage),
        ("normalization_name", metadata.normalization_name, cache_key.normalization_name),
    )
    for field_name, observed, expected in comparisons:
        if observed != expected:
            raise EmbeddingCachePublicationError(
                f"metadata {field_name} does not match the cache key."
            )


def _normalize_array(array: np.ndarray, cache_key: EmbeddingCacheKey) -> np.ndarray:
    if not isinstance(array, np.ndarray):
        raise EmbeddingCachePublicationError("embedding tensor must be a NumPy array.")
    if array.ndim != 5:
        raise EmbeddingCachePublicationError("embedding tensor must have shape [B, C, D, H, W].")
    if array.dtype.kind == "O":
        raise EmbeddingCachePublicationError("embedding tensor must not use object dtype.")
    expected_dtype = _dtype_for_name(cache_key.tensor_dtype)
    if array.dtype != expected_dtype:
        raise EmbeddingCachePublicationError("embedding tensor dtype does not match the cache key.")
    if tuple(int(axis) for axis in array.shape) != cache_key.feature_shape:
        raise EmbeddingCachePublicationError("embedding tensor shape does not match the cache key.")
    return np.ascontiguousarray(array)


def _build_artifact(
    *,
    cache_key: EmbeddingCacheKey,
    metadata: EmbeddingArtifactMetadata,
    array: np.ndarray,
    relative_artifact_path: str,
    created_at_utc: str | None,
) -> EmbeddingArtifact:
    tensor_bytes = _serialize_array_bytes(array)
    descriptor = EmbeddingTensorDescriptor(
        schema_name=EMBEDDING_TENSOR_DESCRIPTOR_SCHEMA_NAME,
        schema_version=EMBEDDING_TENSOR_DESCRIPTOR_SCHEMA_VERSION,
        feature_shape=cache_key.feature_shape,
        tensor_dtype=cache_key.tensor_dtype,
        tensor_byte_length=len(tensor_bytes),
        tensor_content_sha256=hashlib.sha256(tensor_bytes).hexdigest(),
        relative_artifact_path=relative_artifact_path,
    )
    artifact_hash = _artifact_hash(
        cache_key_hash=cache_key.cache_key_hash,
        metadata=metadata,
        descriptor=descriptor,
    )
    return EmbeddingArtifact(
        schema_name=EMBEDDING_ARTIFACT_SCHEMA_NAME,
        schema_version=EMBEDDING_ARTIFACT_SCHEMA_VERSION,
        artifact_hash=artifact_hash,
        cache_key_hash=cache_key.cache_key_hash,
        metadata=metadata,
        tensor_descriptor=descriptor,
        created_at_utc=created_at_utc,
    )


def _artifact_hash(
    *,
    cache_key_hash: str,
    metadata: EmbeddingArtifactMetadata,
    descriptor: EmbeddingTensorDescriptor,
) -> str:
    return sha256_json(
        {
            "cache_key_hash": cache_key_hash,
            "metadata": {
                "checkpoint_hash": metadata.checkpoint_hash,
                "dataset_manifest_hash": metadata.dataset_manifest_hash,
                "encoder_identity": metadata.encoder_identity,
                "feature_stage": metadata.feature_stage,
                "input_identity": metadata.input_identity,
                "normalization_name": metadata.normalization_name,
                "preprocessing_hash": metadata.preprocessing_hash,
                "schema_name": metadata.schema_name,
                "schema_version": metadata.schema_version,
            },
            "schema_name": EMBEDDING_ARTIFACT_SCHEMA_NAME,
            "schema_version": EMBEDDING_ARTIFACT_SCHEMA_VERSION,
            "tensor_descriptor": {
                "feature_shape": list(descriptor.feature_shape),
                "relative_artifact_path": descriptor.relative_artifact_path,
                "schema_name": descriptor.schema_name,
                "schema_version": descriptor.schema_version,
                "tensor_byte_length": descriptor.tensor_byte_length,
                "tensor_content_sha256": descriptor.tensor_content_sha256,
                "tensor_dtype": descriptor.tensor_dtype,
            },
        }
    )


def _cache_key_from_artifact(artifact: EmbeddingArtifact) -> EmbeddingCacheKey:
    return EmbeddingCacheKey(
        schema_name=EMBEDDING_CACHE_KEY_SCHEMA_NAME,
        schema_version=EMBEDDING_CACHE_KEY_SCHEMA_VERSION,
        cache_key_hash=artifact.cache_key_hash,
        encoder_identity=artifact.metadata.encoder_identity,
        preprocessing_hash=artifact.metadata.preprocessing_hash,
        checkpoint_hash=artifact.metadata.checkpoint_hash,
        input_identity=artifact.metadata.input_identity,
        dataset_manifest_hash=artifact.metadata.dataset_manifest_hash,
        feature_stage=artifact.metadata.feature_stage,
        normalization_name=artifact.metadata.normalization_name,
        feature_shape=artifact.tensor_descriptor.feature_shape,
        tensor_dtype=artifact.tensor_descriptor.tensor_dtype,
    )


def _entry_from_artifact(
    *,
    cache_key: EmbeddingCacheKey,
    artifact: EmbeddingArtifact,
) -> EmbeddingCacheEntry:
    return EmbeddingCacheEntry(
        schema_name=EMBEDDING_CACHE_ENTRY_SCHEMA_NAME,
        schema_version=EMBEDDING_CACHE_ENTRY_SCHEMA_VERSION,
        cache_key=cache_key,
        cache_key_hash=cache_key.cache_key_hash,
        artifact_hash=artifact.artifact_hash,
        relative_artifact_path=artifact.tensor_descriptor.relative_artifact_path,
        created_at_utc=artifact.created_at_utc,
    )


def _require_file_pair_state(tensor_path: Path, metadata_path: Path) -> None:
    tensor_exists = tensor_path.exists()
    metadata_exists = metadata_path.exists()
    if not tensor_exists and not metadata_exists:
        raise EmbeddingCacheMissError("cache entry is missing.")
    if tensor_exists != metadata_exists:
        raise EmbeddingCacheInvalidEntryError(
            "cache entry is partial because tensor and metadata sidecar do not both exist."
        )
    for path in (tensor_path, metadata_path):
        if path.is_symlink():
            raise EmbeddingCacheUnsafePathError("cache files must not be symlinks.")
        if not path.is_file():
            raise EmbeddingCacheInvalidEntryError("cache files must be regular files.")


def _load_artifact_json(path: Path) -> EmbeddingArtifact:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EmbeddingCacheInvalidEntryError("artifact metadata JSON is malformed.") from exc
    try:
        return embedding_artifact_from_mapping(_require_mapping(payload, "artifact"))
    except RetrievalArtifactError as exc:
        raise EmbeddingCacheInvalidEntryError("artifact metadata content is invalid.") from exc


def _serialize_array_bytes(array: np.ndarray) -> bytes:
    import io

    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def _load_numpy_tensor(path: Path) -> np.ndarray:
    try:
        array = np.load(path, allow_pickle=False)
    except ValueError as exc:
        raise EmbeddingCacheInvalidEntryError(
            "tensor file requires pickle or is malformed."
        ) from exc
    except OSError as exc:
        raise EmbeddingCacheInvalidEntryError("tensor file could not be read.") from exc
    if not isinstance(array, np.ndarray):
        raise EmbeddingCacheInvalidEntryError("tensor file did not contain a NumPy ndarray.")
    if array.dtype.kind == "O":
        raise EmbeddingCacheInvalidEntryError("tensor file must not contain an object array.")
    if array.ndim != 5:
        raise EmbeddingCacheInvalidEntryError("tensor file must preserve [B, C, D, H, W].")
    return array


def _dtype_for_name(dtype_name: str) -> np.dtype[np.generic]:
    if dtype_name not in _DTYPE_BY_NAME:
        raise EmbeddingCachePublicationError(
            f"unsupported deterministic tensor dtype: {dtype_name!r}."
        )
    return _DTYPE_BY_NAME[dtype_name]


def _require_mapping(value: object, field_name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise EmbeddingCacheInvalidEntryError(f"{field_name} JSON payload must be a mapping.")
    if not all(isinstance(key, str) for key in value):
        raise EmbeddingCacheInvalidEntryError(
            f"{field_name} JSON payload must use string keys only."
        )
    return value


__all__ = [
    "CachedEmbedding",
    "DEFAULT_CACHE_MANIFEST_RELATIVE_PATH",
    "EmbeddingCache",
    "EmbeddingCacheError",
    "EmbeddingCacheIncompatibleEntryError",
    "EmbeddingCacheInvalidEntryError",
    "EmbeddingCacheMissError",
    "EmbeddingCacheProbeResult",
    "EmbeddingCachePublicationError",
    "EmbeddingCacheUnsafePathError",
]
