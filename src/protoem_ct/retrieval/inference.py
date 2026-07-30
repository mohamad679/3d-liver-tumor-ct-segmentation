"""Deterministic prototype-only query inference for Phase 5."""

from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass
from typing import Final

import numpy as np

from protoem_ct.artifacts.hashing import JsonValue, sha256_json
from protoem_ct.models.interfaces import (
    BackgroundPrototype,
    FeatureEncoding3D,
    ForegroundPrototype,
)
from protoem_ct.retrieval.prototypes import (
    PrototypeConstructionError,
    hash_support_prototype_identity,
    support_prototype_identity_payload,
)

PROTOTYPE_INFERENCE_SCHEMA_NAME: Final[str] = "prototype_only_inference_result"
PROTOTYPE_INFERENCE_SCHEMA_VERSION: Final[str] = "v1"
_FLOAT_DTYPE: Final[np.dtype[np.float64]] = np.dtype(np.float64)
_PREDICTION_DTYPE: Final[np.dtype[np.uint8]] = np.dtype(np.uint8)
_FLOATING_KINDS: Final[frozenset[str]] = frozenset({"f"})
_TIE_POLICY: Final[str] = "background_on_exact_tie"
_SCORING_RULE: Final[str] = "per_voxel_dual_cosine_similarity"


class PrototypeInferenceError(ValueError):
    """Base error for deterministic prototype-only inference."""


class InvalidQueryFeatureError(PrototypeInferenceError):
    """Raised when the query feature tensor is malformed or non-finite."""


class IncompatiblePrototypeError(PrototypeInferenceError):
    """Raised when prototype/query metadata or identities are incompatible."""


class ZeroNormPrototypeError(PrototypeInferenceError):
    """Raised when one prototype vector has zero L2 norm."""


class InferenceLeakageError(PrototypeInferenceError):
    """Raised when query/prototype provenance overlap violates the inference contract."""


class NumericalInferenceError(PrototypeInferenceError):
    """Raised when cosine scoring produces a non-finite or out-of-range result."""


class InvalidTargetShapeError(PrototypeInferenceError):
    """Raised when one requested output spatial shape is malformed."""


@dataclass(frozen=True, slots=True)
class PrototypeInferenceResult:
    """Deterministic prototype-only binary inference output."""

    schema_name: str
    schema_version: str
    query_identity: str
    query_patient_id: str
    query_case_id: str
    encoder_identity: str
    preprocessing_hash: str
    checkpoint_hash: str
    dataset_manifest_hash: str | None
    feature_stage: str
    normalization_name: str
    feature_shape: tuple[int, int, int, int, int]
    foreground_prototype_identity: str
    background_prototype_identity: str
    tie_policy: str
    scoring_rule: str
    foreground_score_map: np.ndarray
    background_score_map: np.ndarray
    prediction_mask: np.ndarray
    confidence_margin_map: np.ndarray | None
    foreground_score_content_sha256: str
    background_score_content_sha256: str
    prediction_content_sha256: str
    confidence_margin_content_sha256: str | None
    inference_identity_sha256: str


def infer_query_from_prototypes(
    *,
    query_feature_encoding: FeatureEncoding3D,
    foreground_prototype: ForegroundPrototype,
    background_prototype: BackgroundPrototype,
    query_patient_id: str,
    query_case_id: str,
    query_identity: str,
    query_dataset_manifest_hash: str | None,
    target_output_spatial_shape: tuple[int, int, int] | None = None,
    emit_confidence_margin: bool = True,
) -> PrototypeInferenceResult:
    """Infer one deterministic binary query mask from foreground/background prototypes."""

    _require_identifier(query_patient_id, field_name="query_patient_id")
    _require_identifier(query_case_id, field_name="query_case_id")
    _require_identifier(query_identity, field_name="query_identity")
    if query_dataset_manifest_hash is not None:
        _require_sha256(query_dataset_manifest_hash, field_name="query_dataset_manifest_hash")
    query_array = _require_query_array(query_feature_encoding.feature_data)
    metadata = query_feature_encoding.metadata
    if metadata.input_identity != query_identity:
        raise IncompatiblePrototypeError(
            "query_identity must match query_feature_encoding.metadata.input_identity."
        )
    if int(query_array.shape[0]) != 1:
        raise InvalidQueryFeatureError("query feature tensor must use batch size 1.")
    feature_spatial_shape = (
        int(query_array.shape[2]),
        int(query_array.shape[3]),
        int(query_array.shape[4]),
    )
    if feature_spatial_shape != metadata.resolution.feature_spatial_shape:
        raise InvalidQueryFeatureError(
            "query feature tensor spatial shape must match "
            "metadata.resolution.feature_spatial_shape."
        )
    if int(query_array.shape[1]) != metadata.feature_channels:
        raise InvalidQueryFeatureError(
            "query feature tensor channel count must match embedding metadata."
        )
    if query_dataset_manifest_hash != foreground_prototype.dataset_manifest_hash:
        raise IncompatiblePrototypeError(
            "query dataset_manifest_hash must match the foreground prototype dataset_manifest_hash."
        )
    if query_dataset_manifest_hash != background_prototype.dataset_manifest_hash:
        raise IncompatiblePrototypeError(
            "query dataset_manifest_hash must match the background prototype dataset_manifest_hash."
        )
    _validate_prototype(
        prototype=foreground_prototype,
        expected_kind="foreground",
        query_identity=query_identity,
        query_patient_id=query_patient_id,
        query_case_id=query_case_id,
    )
    _validate_prototype(
        prototype=background_prototype,
        expected_kind="background",
        query_identity=query_identity,
        query_patient_id=query_patient_id,
        query_case_id=query_case_id,
    )
    _validate_compatibility(
        query_feature_encoding=query_feature_encoding,
        foreground_prototype=foreground_prototype,
        background_prototype=background_prototype,
    )
    if target_output_spatial_shape is not None:
        _require_spatial_shape(
            target_output_spatial_shape, field_name="target_output_spatial_shape"
        )

    foreground_vector = _prototype_vector(foreground_prototype, role="foreground")
    background_vector = _prototype_vector(background_prototype, role="background")
    foreground_norm = _l2_norm(foreground_vector, role="foreground prototype")
    background_norm = _l2_norm(background_vector, role="background prototype")

    query_vectors = np.moveaxis(query_array.astype(_FLOAT_DTYPE, copy=False), 1, -1)
    query_norms = np.linalg.norm(query_vectors, axis=-1)
    if not np.isfinite(query_norms).all():
        raise NumericalInferenceError("query voxel norms must be finite.")

    foreground_scores = np.zeros(query_norms.shape, dtype=_FLOAT_DTYPE)
    background_scores = np.zeros(query_norms.shape, dtype=_FLOAT_DTYPE)
    positive_mask = query_norms > 0.0
    if np.any(positive_mask):
        foreground_scores[positive_mask] = _cosine_scores_for_mask(
            query_vectors=query_vectors[positive_mask],
            query_norms=query_norms[positive_mask],
            prototype_vector=foreground_vector,
            prototype_norm=foreground_norm,
        )
        background_scores[positive_mask] = _cosine_scores_for_mask(
            query_vectors=query_vectors[positive_mask],
            query_norms=query_norms[positive_mask],
            prototype_vector=background_vector,
            prototype_norm=background_norm,
        )
    prediction = (foreground_scores > background_scores).astype(_PREDICTION_DTYPE, copy=False)
    confidence_margin = np.abs(foreground_scores - background_scores)

    foreground_score_map = foreground_scores[:, np.newaxis, :, :, :]
    background_score_map = background_scores[:, np.newaxis, :, :, :]
    prediction_map = prediction[:, np.newaxis, :, :, :]
    confidence_margin_map = (
        confidence_margin[:, np.newaxis, :, :, :] if emit_confidence_margin else None
    )

    if (
        target_output_spatial_shape is not None
        and target_output_spatial_shape != feature_spatial_shape
    ):
        foreground_score_map = _upsample_nearest(
            array=foreground_score_map,
            target_spatial_shape=target_output_spatial_shape,
        )
        background_score_map = _upsample_nearest(
            array=background_score_map,
            target_spatial_shape=target_output_spatial_shape,
        )
        prediction_map = _upsample_nearest(
            array=prediction_map,
            target_spatial_shape=target_output_spatial_shape,
        ).astype(_PREDICTION_DTYPE, copy=False)
        if confidence_margin_map is not None:
            confidence_margin_map = _upsample_nearest(
                array=confidence_margin_map,
                target_spatial_shape=target_output_spatial_shape,
            )

    foreground_hash = _array_content_sha256(foreground_score_map)
    background_hash = _array_content_sha256(background_score_map)
    prediction_hash = _array_content_sha256(prediction_map)
    confidence_hash = (
        _array_content_sha256(confidence_margin_map) if confidence_margin_map is not None else None
    )
    feature_shape = (
        int(query_array.shape[0]),
        int(query_array.shape[1]),
        int(query_array.shape[2]),
        int(query_array.shape[3]),
        int(query_array.shape[4]),
    )
    draft = PrototypeInferenceResult(
        schema_name=PROTOTYPE_INFERENCE_SCHEMA_NAME,
        schema_version=PROTOTYPE_INFERENCE_SCHEMA_VERSION,
        query_identity=query_identity,
        query_patient_id=query_patient_id,
        query_case_id=query_case_id,
        encoder_identity=metadata.encoder_identity,
        preprocessing_hash=metadata.preprocessing_hash,
        checkpoint_hash=metadata.checkpoint_hash,
        dataset_manifest_hash=query_dataset_manifest_hash,
        feature_stage=metadata.feature_stage,
        normalization_name=metadata.normalization_name,
        feature_shape=feature_shape,
        foreground_prototype_identity=foreground_prototype.prototype_identity_sha256,
        background_prototype_identity=background_prototype.prototype_identity_sha256,
        tie_policy=_TIE_POLICY,
        scoring_rule=_SCORING_RULE,
        foreground_score_map=np.ascontiguousarray(foreground_score_map),
        background_score_map=np.ascontiguousarray(background_score_map),
        prediction_mask=np.ascontiguousarray(prediction_map),
        confidence_margin_map=(
            np.ascontiguousarray(confidence_margin_map)
            if confidence_margin_map is not None
            else None
        ),
        foreground_score_content_sha256=foreground_hash,
        background_score_content_sha256=background_hash,
        prediction_content_sha256=prediction_hash,
        confidence_margin_content_sha256=confidence_hash,
        inference_identity_sha256="0" * 64,
    )
    return PrototypeInferenceResult(
        schema_name=draft.schema_name,
        schema_version=draft.schema_version,
        query_identity=draft.query_identity,
        query_patient_id=draft.query_patient_id,
        query_case_id=draft.query_case_id,
        encoder_identity=draft.encoder_identity,
        preprocessing_hash=draft.preprocessing_hash,
        checkpoint_hash=draft.checkpoint_hash,
        dataset_manifest_hash=draft.dataset_manifest_hash,
        feature_stage=draft.feature_stage,
        normalization_name=draft.normalization_name,
        feature_shape=draft.feature_shape,
        foreground_prototype_identity=draft.foreground_prototype_identity,
        background_prototype_identity=draft.background_prototype_identity,
        tie_policy=draft.tie_policy,
        scoring_rule=draft.scoring_rule,
        foreground_score_map=draft.foreground_score_map,
        background_score_map=draft.background_score_map,
        prediction_mask=draft.prediction_mask,
        confidence_margin_map=draft.confidence_margin_map,
        foreground_score_content_sha256=draft.foreground_score_content_sha256,
        background_score_content_sha256=draft.background_score_content_sha256,
        prediction_content_sha256=draft.prediction_content_sha256,
        confidence_margin_content_sha256=draft.confidence_margin_content_sha256,
        inference_identity_sha256=hash_prototype_inference_identity(draft),
    )


def prototype_inference_identity_payload(
    result: PrototypeInferenceResult,
) -> dict[str, JsonValue]:
    """Return the exact canonical identity payload for one inference result."""

    return {
        "background_prototype_identity": result.background_prototype_identity,
        "background_score_content_sha256": result.background_score_content_sha256,
        "checkpoint_hash": result.checkpoint_hash,
        "confidence_margin_content_sha256": result.confidence_margin_content_sha256,
        "dataset_manifest_hash": result.dataset_manifest_hash,
        "encoder_identity": result.encoder_identity,
        "feature_shape": list(result.feature_shape),
        "feature_stage": result.feature_stage,
        "foreground_prototype_identity": result.foreground_prototype_identity,
        "foreground_score_content_sha256": result.foreground_score_content_sha256,
        "normalization_name": result.normalization_name,
        "prediction_content_sha256": result.prediction_content_sha256,
        "preprocessing_hash": result.preprocessing_hash,
        "query_case_id": result.query_case_id,
        "query_identity": result.query_identity,
        "query_patient_id": result.query_patient_id,
        "schema_name": result.schema_name,
        "schema_version": result.schema_version,
        "scoring_rule": result.scoring_rule,
        "tie_policy": result.tie_policy,
    }


def hash_prototype_inference_identity(result: PrototypeInferenceResult) -> str:
    """Return the deterministic SHA-256 identity hash for one inference result."""

    return sha256_json(prototype_inference_identity_payload(result))


def query_label_not_part_of_inference_api() -> bool:
    """Return True when the public inference surface excludes query labels."""

    return all(
        "label" not in parameter.name
        for parameter in inspect.signature(infer_query_from_prototypes).parameters.values()
    )


def _validate_prototype(
    *,
    prototype: ForegroundPrototype | BackgroundPrototype,
    expected_kind: str,
    query_identity: str,
    query_patient_id: str,
    query_case_id: str,
) -> None:
    if prototype.prototype_kind != expected_kind:
        raise IncompatiblePrototypeError(
            f"expected {expected_kind} prototype_kind, got {prototype.prototype_kind}."
        )
    if not prototype.prototype_vector:
        raise IncompatiblePrototypeError("prototype_vector must be populated for inference.")
    if not prototype.encoder_identity:
        raise IncompatiblePrototypeError("prototype encoder_identity must be populated.")
    if not prototype.preprocessing_hash:
        raise IncompatiblePrototypeError("prototype preprocessing_hash must be populated.")
    if not prototype.checkpoint_hash:
        raise IncompatiblePrototypeError("prototype checkpoint_hash must be populated.")
    if not prototype.feature_stage:
        raise IncompatiblePrototypeError("prototype feature_stage must be populated.")
    if not prototype.prototype_content_sha256:
        raise IncompatiblePrototypeError("prototype_content_sha256 must be populated.")
    if not prototype.prototype_identity_sha256:
        raise IncompatiblePrototypeError("prototype_identity_sha256 must be populated.")
    if query_identity in prototype.source_support_identifiers:
        raise InferenceLeakageError(
            "query identity must not appear in prototype source identifiers."
        )
    if prototype.source_patient_ids and query_patient_id in prototype.source_patient_ids:
        raise InferenceLeakageError("query patient ID must not overlap prototype source patients.")
    if prototype.source_case_ids and query_case_id in prototype.source_case_ids:
        raise InferenceLeakageError("query case ID must not overlap prototype source cases.")
    vector_hash = _array_content_sha256(np.asarray(prototype.prototype_vector, dtype=_FLOAT_DTYPE))
    if vector_hash != prototype.prototype_content_sha256:
        raise IncompatiblePrototypeError(
            "prototype_content_sha256 does not match prototype_vector."
        )
    try:
        recomputed_identity = hash_support_prototype_identity(prototype)
    except PrototypeConstructionError as error:
        raise IncompatiblePrototypeError(str(error)) from error
    if recomputed_identity != prototype.prototype_identity_sha256:
        raise IncompatiblePrototypeError(
            "prototype_identity_sha256 does not match the canonical prototype identity payload."
        )
    support_prototype_identity_payload(prototype)


def _validate_compatibility(
    *,
    query_feature_encoding: FeatureEncoding3D,
    foreground_prototype: ForegroundPrototype,
    background_prototype: BackgroundPrototype,
) -> None:
    metadata = query_feature_encoding.metadata
    comparisons = (
        ("channel count", metadata.feature_channels, foreground_prototype.feature_channels),
        ("channel count", metadata.feature_channels, background_prototype.feature_channels),
        ("encoder identity", metadata.encoder_identity, foreground_prototype.encoder_identity),
        ("encoder identity", metadata.encoder_identity, background_prototype.encoder_identity),
        (
            "preprocessing hash",
            metadata.preprocessing_hash,
            foreground_prototype.preprocessing_hash,
        ),
        (
            "preprocessing hash",
            metadata.preprocessing_hash,
            background_prototype.preprocessing_hash,
        ),
        ("checkpoint hash", metadata.checkpoint_hash, foreground_prototype.checkpoint_hash),
        ("checkpoint hash", metadata.checkpoint_hash, background_prototype.checkpoint_hash),
        ("feature stage", metadata.feature_stage, foreground_prototype.feature_stage),
        ("feature stage", metadata.feature_stage, background_prototype.feature_stage),
        (
            "normalization name",
            metadata.normalization_name,
            foreground_prototype.normalization_name,
        ),
        (
            "normalization name",
            metadata.normalization_name,
            background_prototype.normalization_name,
        ),
        (
            "dataset manifest hash",
            foreground_prototype.dataset_manifest_hash,
            background_prototype.dataset_manifest_hash,
        ),
    )
    for field_name, expected, observed in comparisons:
        if expected != observed:
            raise IncompatiblePrototypeError(f"query and prototypes must match on {field_name}.")


def _require_query_array(value: object) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise InvalidQueryFeatureError("query feature tensor must be a NumPy ndarray.")
    if value.ndim != 5:
        raise InvalidQueryFeatureError("query feature tensor must have shape [1, C, D, H, W].")
    if value.dtype.kind not in _FLOATING_KINDS:
        raise InvalidQueryFeatureError("query feature tensor must use a floating dtype.")
    if not np.isfinite(value).all():
        raise InvalidQueryFeatureError("query feature tensor must not contain NaN or Infinity.")
    return np.ascontiguousarray(value)


def _prototype_vector(
    prototype: ForegroundPrototype | BackgroundPrototype,
    *,
    role: str,
) -> np.ndarray:
    vector = np.asarray(prototype.prototype_vector, dtype=_FLOAT_DTYPE)
    if vector.ndim != 1:
        raise IncompatiblePrototypeError(f"{role} prototype vector must be one-dimensional.")
    if vector.shape[0] != prototype.feature_channels:
        raise IncompatiblePrototypeError(
            f"{role} prototype vector length must equal feature_channels."
        )
    if not np.isfinite(vector).all():
        raise IncompatiblePrototypeError(f"{role} prototype vector must be finite.")
    return np.ascontiguousarray(vector)


def _l2_norm(vector: np.ndarray, *, role: str) -> float:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm):
        raise NumericalInferenceError(f"{role} norm must be finite.")
    if norm == 0.0:
        raise ZeroNormPrototypeError(f"{role} must not have zero L2 norm.")
    return norm


def _cosine_scores_for_mask(
    *,
    query_vectors: np.ndarray,
    query_norms: np.ndarray,
    prototype_vector: np.ndarray,
    prototype_norm: float,
) -> np.ndarray:
    numerators = np.sum(query_vectors * prototype_vector, axis=-1, dtype=_FLOAT_DTYPE)
    if not np.isfinite(numerators).all():
        raise NumericalInferenceError("cosine numerators must be finite.")
    scores = numerators / (query_norms * prototype_norm)
    if not np.isfinite(scores).all():
        raise NumericalInferenceError("cosine scores must be finite.")
    scores = np.where((scores > 1.0) & (scores <= 1.0 + 1e-12), 1.0, scores)
    scores = np.where((scores < -1.0) & (scores >= -1.0 - 1e-12), -1.0, scores)
    if np.any(scores < -1.0) or np.any(scores > 1.0):
        raise NumericalInferenceError("cosine scores fell outside [-1, 1].")
    return scores.astype(_FLOAT_DTYPE, copy=False)


def _upsample_nearest(
    *,
    array: np.ndarray,
    target_spatial_shape: tuple[int, int, int],
) -> np.ndarray:
    if array.ndim != 5:
        raise InvalidTargetShapeError("upsampling expects one [B, C, D, H, W] array.")
    source_depth, source_height, source_width = (
        int(array.shape[2]),
        int(array.shape[3]),
        int(array.shape[4]),
    )
    if (source_depth, source_height, source_width) == target_spatial_shape:
        return np.ascontiguousarray(array)
    depth_indices = _nearest_indices(source_size=source_depth, target_size=target_spatial_shape[0])
    height_indices = _nearest_indices(
        source_size=source_height,
        target_size=target_spatial_shape[1],
    )
    width_indices = _nearest_indices(source_size=source_width, target_size=target_spatial_shape[2])
    projected = array[:, :, depth_indices, :, :]
    projected = projected[:, :, :, height_indices, :]
    projected = projected[:, :, :, :, width_indices]
    return np.ascontiguousarray(projected)


def _nearest_indices(*, source_size: int, target_size: int) -> np.ndarray:
    if source_size <= 0 or target_size <= 0:
        raise InvalidTargetShapeError("upsampling sizes must be positive.")
    target_positions = np.arange(target_size, dtype=_FLOAT_DTYPE)
    source_positions = ((target_positions + 0.5) * source_size / target_size) - 0.5
    source_positions = np.rint(source_positions)
    return np.clip(source_positions.astype(np.int64), 0, source_size - 1)


def _array_content_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def _require_identifier(value: str, *, field_name: str) -> None:
    if value == "" or value != value.strip():
        raise PrototypeInferenceError(f"{field_name} must be a nonempty identifier.")


def _require_sha256(value: str, *, field_name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise PrototypeInferenceError(
            f"{field_name} must be a lowercase 64-character SHA-256 hex digest."
        )


def _require_spatial_shape(value: tuple[int, int, int], *, field_name: str) -> None:
    if len(value) != 3:
        raise InvalidTargetShapeError(f"{field_name} must contain exactly three axes.")
    for axis, item in zip(("depth", "height", "width"), value, strict=True):
        if item <= 0:
            raise InvalidTargetShapeError(f"{field_name}.{axis} must be a positive integer.")


__all__ = [
    "InvalidQueryFeatureError",
    "IncompatiblePrototypeError",
    "InvalidTargetShapeError",
    "InferenceLeakageError",
    "NumericalInferenceError",
    "PROTOTYPE_INFERENCE_SCHEMA_NAME",
    "PROTOTYPE_INFERENCE_SCHEMA_VERSION",
    "PrototypeInferenceError",
    "PrototypeInferenceResult",
    "ZeroNormPrototypeError",
    "hash_prototype_inference_identity",
    "infer_query_from_prototypes",
    "prototype_inference_identity_payload",
    "query_label_not_part_of_inference_api",
]
