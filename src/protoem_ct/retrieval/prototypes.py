"""Deterministic foreground/background support prototype construction."""

from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass
from typing import Final, cast

import numpy as np

from protoem_ct.artifacts.hashing import JsonValue, sha256_json
from protoem_ct.models.interfaces import (
    BackgroundPrototype,
    EmbeddingMetadata,
    FeatureEncoding3D,
    ForegroundPrototype,
    SupportPrototype,
)

_SHA256_HEX_LENGTH: Final[int] = 64
_FLOAT_DTYPE: Final[np.dtype[np.float64]] = np.dtype(np.float64)
_NUMERIC_KINDS: Final[frozenset[str]] = frozenset({"f", "i", "u"})
_BINARY_VALUES: Final[frozenset[int]] = frozenset({0, 1})


class PrototypeConstructionError(ValueError):
    """Base error for deterministic support prototype construction."""


class InvalidSupportFeatureError(PrototypeConstructionError):
    """Raised when one support feature tensor is malformed or non-finite."""


class InvalidSupportMaskError(PrototypeConstructionError):
    """Raised when one support mask is malformed, non-binary, or ambiguous."""


class IncompatibleSupportSetError(PrototypeConstructionError):
    """Raised when one support set mixes incompatible feature metadata."""


class DuplicateSupportError(PrototypeConstructionError):
    """Raised when one support set contains duplicate identifiers or assignments."""


class EmptyForegroundPrototypeError(PrototypeConstructionError):
    """Raised when all supplied supports are foreground-empty after mask alignment."""


class EmptyBackgroundPrototypeError(PrototypeConstructionError):
    """Raised when all supplied supports are background-empty after mask alignment."""


class PrototypeNumericalError(PrototypeConstructionError):
    """Raised when prototype aggregation produces a non-finite result."""


@dataclass(frozen=True, slots=True)
class SupportPrototypeInput:
    """One support feature tensor plus one explicit binary support mask."""

    support_identifier: str
    support_patient_id: str
    support_case_id: str
    dataset_manifest_hash: str | None
    feature_encoding: FeatureEncoding3D
    binary_mask: np.ndarray


@dataclass(frozen=True, slots=True)
class SupportPrototypeMemory:
    """Deterministic foreground/background prototype memory for one support set."""

    foreground_prototype: ForegroundPrototype
    background_prototype: BackgroundPrototype


@dataclass(frozen=True, slots=True)
class _ResolvedSupport:
    support_identifier: str
    support_patient_id: str
    support_case_id: str
    dataset_manifest_hash: str | None
    metadata: EmbeddingMetadata
    feature_array: np.ndarray
    aligned_mask: np.ndarray


def build_support_prototype_memory(
    *,
    supports: tuple[SupportPrototypeInput, ...],
) -> SupportPrototypeMemory:
    """Build deterministic foreground/background prototypes from support features and masks.

    Supported feature tensor shape is exactly ``[1, C, D, H, W]``.
    Supported mask shapes are exactly ``[1, D, H, W]`` or ``[1, 1, D, H, W]``.
    If the mask spatial shape matches the feature spatial shape it is used directly.
    If it matches ``metadata.resolution.input_spatial_shape`` it is projected to feature resolution
    with deterministic nearest-neighbor interpolation using voxel-center index mapping.
    """

    if not supports:
        raise IncompatibleSupportSetError("supports must contain at least one item.")
    resolved_supports = tuple(_resolve_support(item) for item in supports)
    _validate_support_set(resolved_supports)
    foreground_prototype = _build_foreground_prototype(resolved_supports=resolved_supports)
    background_prototype = _build_background_prototype(resolved_supports=resolved_supports)
    return SupportPrototypeMemory(
        foreground_prototype=foreground_prototype,
        background_prototype=background_prototype,
    )


def support_prototype_identity_payload(prototype: SupportPrototype) -> dict[str, JsonValue]:
    """Return the exact canonical identity payload for one constructed support prototype."""

    if not prototype.prototype_vector:
        raise PrototypeConstructionError(
            "prototype identity requires a populated prototype vector."
        )
    if not prototype.encoder_identity:
        raise PrototypeConstructionError("prototype identity requires explicit encoder metadata.")
    return {
        "channel_count": prototype.feature_channels,
        "checkpoint_hash": prototype.checkpoint_hash,
        "contributing_voxel_count": prototype.contributing_voxel_count,
        "dataset_manifest_hash": prototype.dataset_manifest_hash,
        "encoder_identity": prototype.encoder_identity,
        "feature_stage": prototype.feature_stage,
        "normalization_name": prototype.normalization_name,
        "preprocessing_hash": prototype.preprocessing_hash,
        "prototype_class": prototype.prototype_kind,
        "prototype_content_sha256": prototype.prototype_content_sha256,
        "sorted_support_identifiers": list(prototype.source_support_identifiers),
    }


def hash_support_prototype_identity(prototype: SupportPrototype) -> str:
    """Return the deterministic prototype identity SHA-256 for one support prototype."""

    return sha256_json(support_prototype_identity_payload(prototype))


def _resolve_support(item: SupportPrototypeInput) -> _ResolvedSupport:
    _require_identifier(item.support_identifier, field_name="support_identifier")
    _require_identifier(item.support_patient_id, field_name="support_patient_id")
    _require_identifier(item.support_case_id, field_name="support_case_id")
    if item.dataset_manifest_hash is not None:
        _require_sha256(item.dataset_manifest_hash, field_name="dataset_manifest_hash")
    feature_array = _require_feature_array(item.feature_encoding.feature_data)
    aligned_mask = _align_mask_to_feature_resolution(
        mask=item.binary_mask,
        batch_size=int(feature_array.shape[0]),
        input_spatial_shape=item.feature_encoding.metadata.resolution.input_spatial_shape,
        feature_spatial_shape=item.feature_encoding.metadata.resolution.feature_spatial_shape,
    )
    return _ResolvedSupport(
        support_identifier=item.support_identifier,
        support_patient_id=item.support_patient_id,
        support_case_id=item.support_case_id,
        dataset_manifest_hash=item.dataset_manifest_hash,
        metadata=item.feature_encoding.metadata,
        feature_array=feature_array,
        aligned_mask=aligned_mask,
    )


def _validate_support_set(resolved_supports: tuple[_ResolvedSupport, ...]) -> None:
    identifiers = [item.support_identifier for item in resolved_supports]
    if len(set(identifiers)) != len(identifiers):
        raise DuplicateSupportError("support identifiers must be unique.")
    patient_case_pairs = [
        (item.support_patient_id, item.support_case_id) for item in resolved_supports
    ]
    if len(set(patient_case_pairs)) != len(patient_case_pairs):
        raise DuplicateSupportError("support patient/case assignments must be unique.")
    reference = resolved_supports[0]
    for candidate in resolved_supports[1:]:
        comparisons = (
            ("channel count", reference.feature_array.shape[1], candidate.feature_array.shape[1]),
            (
                "feature spatial shape",
                reference.feature_array.shape[2:],
                candidate.feature_array.shape[2:],
            ),
            (
                "encoder_identity",
                reference.metadata.encoder_identity,
                candidate.metadata.encoder_identity,
            ),
            (
                "preprocessing_hash",
                reference.metadata.preprocessing_hash,
                candidate.metadata.preprocessing_hash,
            ),
            (
                "checkpoint_hash",
                reference.metadata.checkpoint_hash,
                candidate.metadata.checkpoint_hash,
            ),
            (
                "dataset_manifest_hash",
                reference.dataset_manifest_hash,
                candidate.dataset_manifest_hash,
            ),
            (
                "feature_stage",
                reference.metadata.feature_stage,
                candidate.metadata.feature_stage,
            ),
            (
                "normalization_name",
                reference.metadata.normalization_name,
                candidate.metadata.normalization_name,
            ),
        )
        for field_name, expected, observed in comparisons:
            if expected != observed:
                raise IncompatibleSupportSetError(f"support features must match on {field_name}.")


def _build_foreground_prototype(
    *,
    resolved_supports: tuple[_ResolvedSupport, ...],
) -> ForegroundPrototype:
    return cast(
        ForegroundPrototype,
        _build_prototype(resolved_supports=resolved_supports, prototype_kind="foreground"),
    )


def _build_background_prototype(
    *,
    resolved_supports: tuple[_ResolvedSupport, ...],
) -> BackgroundPrototype:
    return cast(
        BackgroundPrototype,
        _build_prototype(resolved_supports=resolved_supports, prototype_kind="background"),
    )


def _build_prototype(
    *,
    resolved_supports: tuple[_ResolvedSupport, ...],
    prototype_kind: str,
) -> ForegroundPrototype | BackgroundPrototype:
    accumulator = np.zeros(
        int(resolved_supports[0].feature_array.shape[1]),
        dtype=_FLOAT_DTYPE,
    )
    contributing_voxel_count = 0
    source_identifiers: list[str] = []
    source_patient_ids: list[str] = []
    source_case_ids: list[str] = []
    for item in resolved_supports:
        feature_vectors = np.moveaxis(item.feature_array, 1, -1)
        selection_mask = item.aligned_mask if prototype_kind == "foreground" else ~item.aligned_mask
        selected_vectors = feature_vectors[selection_mask]
        if selected_vectors.size == 0:
            continue
        contributing_voxel_count += int(selected_vectors.shape[0])
        accumulator += selected_vectors.astype(_FLOAT_DTYPE, copy=False).sum(axis=0)
        source_identifiers.append(item.support_identifier)
        source_patient_ids.append(item.support_patient_id)
        source_case_ids.append(item.support_case_id)
    if contributing_voxel_count == 0:
        if prototype_kind == "foreground":
            raise EmptyForegroundPrototypeError(
                "all supplied supports are foreground-empty after aligned mask projection."
            )
        raise EmptyBackgroundPrototypeError(
            "all supplied supports are background-empty after aligned mask projection."
        )
    prototype_vector = accumulator / float(contributing_voxel_count)
    if not np.isfinite(prototype_vector).all():
        raise PrototypeNumericalError("prototype vector must be finite.")
    vector_tuple = tuple(float(value) for value in prototype_vector.tolist())
    content_sha256 = hashlib.sha256(
        np.asarray(vector_tuple, dtype=_FLOAT_DTYPE).tobytes(order="C")
    ).hexdigest()
    feature_channels = int(resolved_supports[0].feature_array.shape[1])
    source_support_identifiers = tuple(sorted(source_identifiers))
    normalization_name = resolved_supports[0].metadata.normalization_name
    encoder_identity = resolved_supports[0].metadata.encoder_identity
    preprocessing_hash = resolved_supports[0].metadata.preprocessing_hash
    checkpoint_hash = resolved_supports[0].metadata.checkpoint_hash
    dataset_manifest_hash = resolved_supports[0].dataset_manifest_hash
    feature_stage = resolved_supports[0].metadata.feature_stage
    sorted_source_patient_ids = tuple(sorted(source_patient_ids))
    sorted_source_case_ids = tuple(sorted(source_case_ids))
    if prototype_kind == "foreground":
        foreground_draft = ForegroundPrototype(
            prototype_kind="ignored",
            feature_channels=feature_channels,
            contributing_voxel_count=contributing_voxel_count,
            source_support_identifiers=source_support_identifiers,
            normalization_name=normalization_name,
            prototype_vector=vector_tuple,
            encoder_identity=encoder_identity,
            preprocessing_hash=preprocessing_hash,
            checkpoint_hash=checkpoint_hash,
            dataset_manifest_hash=dataset_manifest_hash,
            feature_stage=feature_stage,
            source_patient_ids=sorted_source_patient_ids,
            source_case_ids=sorted_source_case_ids,
            prototype_content_sha256=content_sha256,
            prototype_identity_sha256="0" * _SHA256_HEX_LENGTH,
        )
        return ForegroundPrototype(
            prototype_kind="ignored",
            feature_channels=feature_channels,
            contributing_voxel_count=contributing_voxel_count,
            source_support_identifiers=source_support_identifiers,
            normalization_name=normalization_name,
            prototype_vector=vector_tuple,
            encoder_identity=encoder_identity,
            preprocessing_hash=preprocessing_hash,
            checkpoint_hash=checkpoint_hash,
            dataset_manifest_hash=dataset_manifest_hash,
            feature_stage=feature_stage,
            source_patient_ids=sorted_source_patient_ids,
            source_case_ids=sorted_source_case_ids,
            prototype_content_sha256=content_sha256,
            prototype_identity_sha256=hash_support_prototype_identity(foreground_draft),
        )
    background_draft = BackgroundPrototype(
        prototype_kind="ignored",
        feature_channels=feature_channels,
        contributing_voxel_count=contributing_voxel_count,
        source_support_identifiers=source_support_identifiers,
        normalization_name=normalization_name,
        prototype_vector=vector_tuple,
        encoder_identity=encoder_identity,
        preprocessing_hash=preprocessing_hash,
        checkpoint_hash=checkpoint_hash,
        dataset_manifest_hash=dataset_manifest_hash,
        feature_stage=feature_stage,
        source_patient_ids=sorted_source_patient_ids,
        source_case_ids=sorted_source_case_ids,
        prototype_content_sha256=content_sha256,
        prototype_identity_sha256="0" * _SHA256_HEX_LENGTH,
    )
    return BackgroundPrototype(
        prototype_kind="ignored",
        feature_channels=feature_channels,
        contributing_voxel_count=contributing_voxel_count,
        source_support_identifiers=source_support_identifiers,
        normalization_name=normalization_name,
        prototype_vector=vector_tuple,
        encoder_identity=encoder_identity,
        preprocessing_hash=preprocessing_hash,
        checkpoint_hash=checkpoint_hash,
        dataset_manifest_hash=dataset_manifest_hash,
        feature_stage=feature_stage,
        source_patient_ids=sorted_source_patient_ids,
        source_case_ids=sorted_source_case_ids,
        prototype_content_sha256=content_sha256,
        prototype_identity_sha256=hash_support_prototype_identity(background_draft),
    )


def _require_feature_array(value: object) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise InvalidSupportFeatureError("support feature tensor must be a NumPy ndarray.")
    if value.ndim != 5:
        raise InvalidSupportFeatureError("support feature tensor must have shape [B, C, D, H, W].")
    if value.shape[0] != 1:
        raise InvalidSupportFeatureError(
            "support feature tensor must use batch size 1 for unambiguous support identity."
        )
    if value.dtype.kind not in _NUMERIC_KINDS:
        raise InvalidSupportFeatureError("support feature tensor must use a numeric dtype.")
    if not np.isfinite(value).all():
        raise InvalidSupportFeatureError("support feature tensor must not contain NaN or Infinity.")
    return np.ascontiguousarray(value)


def _align_mask_to_feature_resolution(
    *,
    mask: np.ndarray,
    batch_size: int,
    input_spatial_shape: tuple[int, int, int],
    feature_spatial_shape: tuple[int, int, int],
) -> np.ndarray:
    boolean_mask = _require_mask(mask, batch_size=batch_size)
    source_spatial_shape = boolean_mask.shape[1:]
    if source_spatial_shape == feature_spatial_shape:
        return boolean_mask
    if source_spatial_shape != input_spatial_shape:
        raise InvalidSupportMaskError(
            "support mask spatial shape must match either input or feature resolution exactly."
        )
    depth_indices = _nearest_indices(
        source_size=input_spatial_shape[0],
        target_size=feature_spatial_shape[0],
    )
    height_indices = _nearest_indices(
        source_size=input_spatial_shape[1],
        target_size=feature_spatial_shape[1],
    )
    width_indices = _nearest_indices(
        source_size=input_spatial_shape[2],
        target_size=feature_spatial_shape[2],
    )
    projected = boolean_mask[:, depth_indices, :, :]
    projected = projected[:, :, height_indices, :]
    projected = projected[:, :, :, width_indices]
    return projected


def _require_mask(mask: np.ndarray, *, batch_size: int) -> np.ndarray:
    if not isinstance(mask, np.ndarray):
        raise InvalidSupportMaskError("support mask must be a NumPy ndarray.")
    if mask.ndim == 5:
        if mask.shape[1] != 1:
            raise InvalidSupportMaskError(
                "5D support masks must use a singleton channel dimension."
            )
        normalized = mask[:, 0, :, :, :]
    elif mask.ndim == 4:
        normalized = mask
    else:
        raise InvalidSupportMaskError(
            "support mask must have shape [1, D, H, W] or [1, 1, D, H, W]."
        )
    if normalized.shape[0] != batch_size:
        raise InvalidSupportMaskError("support mask batch size must match the feature tensor.")
    if not np.isfinite(normalized).all():
        raise InvalidSupportMaskError("support mask must not contain NaN or Infinity.")
    unique_values = {int(value) for value in np.unique(normalized)}
    if unique_values - _BINARY_VALUES:
        raise InvalidSupportMaskError("support mask must contain binary values only.")
    return normalized.astype(bool, copy=False)


def _nearest_indices(*, source_size: int, target_size: int) -> np.ndarray:
    if source_size <= 0 or target_size <= 0:
        raise InvalidSupportMaskError("mask alignment sizes must be positive.")
    target_positions = np.arange(target_size, dtype=_FLOAT_DTYPE)
    source_positions = ((target_positions + 0.5) * source_size / target_size) - 0.5
    source_positions = np.rint(source_positions)
    indices = np.clip(source_positions.astype(np.int64), 0, source_size - 1)
    return indices


def _require_identifier(value: str, *, field_name: str) -> None:
    if value == "" or value != value.strip():
        raise PrototypeConstructionError(f"{field_name} must be a nonempty identifier.")


def _require_sha256(value: str, *, field_name: str) -> None:
    if len(value) != _SHA256_HEX_LENGTH or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise PrototypeConstructionError(
            f"{field_name} must be a lowercase 64-character SHA-256 hex digest."
        )


def query_label_not_part_of_api() -> bool:
    """Return True when the public builder surface excludes any query-label parameter."""

    return all(
        "query" not in parameter.name
        for parameter in inspect.signature(build_support_prototype_memory).parameters.values()
    )


__all__ = [
    "BackgroundPrototype",
    "DuplicateSupportError",
    "EmptyBackgroundPrototypeError",
    "EmptyForegroundPrototypeError",
    "ForegroundPrototype",
    "IncompatibleSupportSetError",
    "InvalidSupportFeatureError",
    "InvalidSupportMaskError",
    "PrototypeConstructionError",
    "PrototypeNumericalError",
    "SupportPrototypeInput",
    "SupportPrototypeMemory",
    "build_support_prototype_memory",
    "hash_support_prototype_identity",
    "query_label_not_part_of_api",
    "support_prototype_identity_payload",
]
