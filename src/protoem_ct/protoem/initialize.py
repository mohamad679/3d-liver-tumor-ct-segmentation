"""Deterministic Phase 5 to Phase 6 ProtoEM initialization bundle construction."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, cast

import numpy as np

from protoem_ct.artifacts.hashing import JsonValue, canonical_json_bytes, sha256_json
from protoem_ct.models.interfaces import (
    BackgroundPrototype,
    FeatureEncoding3D,
    ForegroundPrototype,
)
from protoem_ct.protoem.artifacts import ProtoEMConfig
from protoem_ct.protoem.contracts import (
    ProtoEMInferenceInput,
    ProtoEMPhase5ArtifactReference,
    ProtoEMPhase5ReferenceBundle,
    ProtoEMPrototypeInput,
    ProtoEMQueryFeatureInput,
    ProtoEMSupportProvenance,
    protoem_phase5_artifact_reference_to_dict,
    protoem_support_provenance_to_dict,
)
from protoem_ct.retrieval.inference import (
    PrototypeInferenceResult,
    hash_prototype_inference_identity,
)
from protoem_ct.retrieval.prototypes import hash_support_prototype_identity

PROTOEM_INITIALIZATION_BUNDLE_SCHEMA_NAME: Final[str] = "protoem_initialization_bundle"
PROTOEM_INITIALIZATION_BUNDLE_SCHEMA_VERSION: Final[str] = "v1"
_FLOATING_KINDS: Final[frozenset[str]] = frozenset({"f"})
_NUMERIC_KINDS: Final[frozenset[str]] = frozenset({"f", "i", "u"})
_PREDICTION_DTYPE: Final[np.dtype[np.uint8]] = np.dtype(np.uint8)
_BINARY_VALUES: Final[frozenset[int]] = frozenset({0, 1})
_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z0-9](?:[a-z0-9_.-]{0,126}[a-z0-9])?$")


class ProtoEMInitializationError(ValueError):
    """Base error for deterministic ProtoEM initialization."""


class MalformedPhase5ArtifactError(ProtoEMInitializationError):
    """Raised when one consumed Phase 5 artifact is malformed or self-hash-invalid."""


class IncompatibleInitializationError(ProtoEMInitializationError):
    """Raised when Phase 5 inputs are incompatible with ProtoEM initialization."""


class InitializationLeakageViolationError(ProtoEMInitializationError):
    """Raised when query/support leakage is detected at the initialization boundary."""


class InvalidQueryFeatureError(ProtoEMInitializationError):
    """Raised when the Phase 5 query feature tensor is malformed for ProtoEM."""


class InvalidInitialPredictionError(ProtoEMInitializationError):
    """Raised when the Phase 5 initial hard prediction is malformed."""


class InvalidInitialScoreStateError(ProtoEMInitializationError):
    """Raised when the Phase 5 initial score maps are malformed or incompatible."""


class InitializationIdentityMismatchError(ProtoEMInitializationError):
    """Raised when a ProtoEM initialization self-hash does not match its content."""


@dataclass(frozen=True, slots=True)
class ProtoEMInitializationBundle:
    """Deterministic ProtoEM initialization state derived only from Phase 5 outputs."""

    schema_name: str
    schema_version: str
    initialization_identity_hash: str
    query_feature_encoding: FeatureEncoding3D
    query_identity: str
    query_patient_id: str
    query_case_id: str
    query_feature_content_sha256: str
    foreground_prototype: ForegroundPrototype
    background_prototype: BackgroundPrototype
    initial_foreground_score_map: np.ndarray
    initial_background_score_map: np.ndarray
    initial_prediction_mask: np.ndarray
    initial_posterior_probability_map: np.ndarray | None
    contains_cosine_scores: bool
    contains_posterior_probabilities: bool
    contains_hard_assignments: bool
    ordered_support_provenance: tuple[ProtoEMSupportProvenance, ...]
    phase5_initialization_reference: ProtoEMPhase5ArtifactReference
    phase5_prototype_reference: ProtoEMPhase5ArtifactReference
    phase5_inference_reference: ProtoEMPhase5ArtifactReference
    phase5_inference_identity_hash: str
    encoder_identity: str
    preprocessing_hash: str
    checkpoint_hash: str
    dataset_manifest_hash: str | None
    feature_stage: str
    normalization_name: str

    def __post_init__(self) -> None:
        if self.schema_name != PROTOEM_INITIALIZATION_BUNDLE_SCHEMA_NAME:
            raise IncompatibleInitializationError(
                f"schema_name must equal {PROTOEM_INITIALIZATION_BUNDLE_SCHEMA_NAME!r}."
            )
        if self.schema_version != PROTOEM_INITIALIZATION_BUNDLE_SCHEMA_VERSION:
            raise IncompatibleInitializationError(
                f"schema_version must equal {PROTOEM_INITIALIZATION_BUNDLE_SCHEMA_VERSION!r}."
            )
        _require_sha256(
            self.initialization_identity_hash,
            field_name="initialization_identity_hash",
        )
        if self.query_feature_encoding.metadata.input_identity != self.query_identity:
            raise IncompatibleInitializationError(
                "query_identity must match query_feature_encoding.metadata.input_identity."
            )
        _require_identifier(self.query_identity, field_name="query_identity")
        _require_identifier(self.query_patient_id, field_name="query_patient_id")
        _require_identifier(self.query_case_id, field_name="query_case_id")
        _require_sha256(
            self.query_feature_content_sha256,
            field_name="query_feature_content_sha256",
        )
        _require_sha256(
            self.phase5_inference_identity_hash,
            field_name="phase5_inference_identity_hash",
        )
        _require_identifier(self.encoder_identity, field_name="encoder_identity")
        _require_sha256(self.preprocessing_hash, field_name="preprocessing_hash")
        _require_sha256(self.checkpoint_hash, field_name="checkpoint_hash")
        _require_optional_sha256(
            self.dataset_manifest_hash,
            field_name="dataset_manifest_hash",
        )
        _require_identifier(self.feature_stage, field_name="feature_stage")
        _require_identifier(self.normalization_name, field_name="normalization_name")
        if not self.contains_cosine_scores:
            raise IncompatibleInitializationError(
                "Phase 5 ProtoEM initialization must contain cosine scores."
            )
        if self.contains_posterior_probabilities:
            raise IncompatibleInitializationError(
                "Phase 5 initialization must not fabricate posterior probabilities."
            )
        if not self.contains_hard_assignments:
            raise IncompatibleInitializationError(
                "Phase 5 ProtoEM initialization must contain hard assignments."
            )
        if self.initial_posterior_probability_map is not None:
            raise IncompatibleInitializationError(
                "Phase 6 substage 2 does not accept posterior probability maps."
            )

        ordered = tuple(sorted(self.ordered_support_provenance, key=lambda item: item.ordering_key))
        object.__setattr__(self, "ordered_support_provenance", ordered)
        if not ordered:
            raise IncompatibleInitializationError(
                "ordered_support_provenance must contain at least one support record."
            )

        expected_identity_hash = hash_protoem_initialization_bundle(self)
        if self.initialization_identity_hash != expected_identity_hash:
            raise InitializationIdentityMismatchError(
                "initialization_identity_hash does not match deterministic bundle content."
            )


def build_protoem_initialization_bundle(
    *,
    config: ProtoEMConfig,
    query_input: ProtoEMQueryFeatureInput,
    prototype_input: ProtoEMPrototypeInput,
    inference_input: ProtoEMInferenceInput,
    phase5_references: ProtoEMPhase5ReferenceBundle,
) -> ProtoEMInitializationBundle:
    """Build one deterministic ProtoEM initialization bundle from validated Phase 5 outputs."""

    _require_phase5_reference_match(
        expected_schema_name=config.phase5_initialization_schema_name,
        expected_hash=config.phase5_initialization_artifact_hash,
        observed=phase5_references.initialization_reference,
        field_name="initialization_reference",
    )
    _require_phase5_reference_match(
        expected_schema_name=config.phase5_prototype_schema_name,
        expected_hash=config.phase5_prototype_artifact_hash,
        observed=phase5_references.prototype_reference,
        field_name="prototype_reference",
    )
    _require_phase5_reference_match(
        expected_schema_name=config.phase5_inference_schema_name,
        expected_hash=config.phase5_inference_artifact_hash,
        observed=phase5_references.inference_reference,
        field_name="inference_reference",
    )

    query_array = _validated_query_feature_array(query_input.query_feature_encoding)
    feature_spatial_shape = _spatial_shape(query_array)
    foreground = cast(
        ForegroundPrototype,
        _validated_prototype(
            prototype=prototype_input.prototype_memory.foreground_prototype,
            role="foreground",
        ),
    )
    background = cast(
        BackgroundPrototype,
        _validated_prototype(
            prototype=prototype_input.prototype_memory.background_prototype,
            role="background",
        ),
    )
    inference = _validated_inference_result(inference_input.inference_result)
    _validate_query_identity_alignment(query_input=query_input, inference=inference)
    _validate_phase5_metadata_compatibility(
        query_feature_encoding=query_input.query_feature_encoding,
        foreground=foreground,
        background=background,
        inference=inference,
        support_provenance=prototype_input.support_provenance,
        query_dataset_manifest_hash=query_input.dataset_manifest_hash,
    )
    _validate_leakage_constraints(
        query_identity=query_input.query_identity,
        query_patient_id=query_input.query_patient_id,
        query_case_id=query_input.query_case_id,
        foreground=foreground,
        background=background,
        support_provenance=prototype_input.support_provenance,
    )

    foreground_scores = _validated_score_map(
        inference.foreground_score_map,
        field_name="foreground_score_map",
        expected_spatial_shape=feature_spatial_shape,
    )
    background_scores = _validated_score_map(
        inference.background_score_map,
        field_name="background_score_map",
        expected_spatial_shape=feature_spatial_shape,
    )
    prediction_mask = _validated_prediction_mask(
        inference.prediction_mask,
        expected_spatial_shape=feature_spatial_shape,
    )
    if (
        inference.confidence_margin_map is not None
        and not np.isfinite(inference.confidence_margin_map).all()
    ):
        raise InvalidInitialScoreStateError("confidence_margin_map must be finite when present.")

    _validate_phase5_hash_consistency(
        inference=inference,
        foreground_scores=foreground_scores,
        background_scores=background_scores,
        prediction_mask=prediction_mask,
    )

    ordered_support_provenance = tuple(
        sorted(prototype_input.support_provenance, key=lambda item: item.ordering_key)
    )
    _validate_prototype_source_alignment(
        foreground=foreground,
        background=background,
        support_provenance=ordered_support_provenance,
    )

    query_feature_copy = np.ascontiguousarray(query_array)
    foreground_score_copy = np.ascontiguousarray(foreground_scores)
    background_score_copy = np.ascontiguousarray(background_scores)
    prediction_copy = np.ascontiguousarray(prediction_mask)
    query_feature_hash = _array_content_sha256(query_feature_copy)
    initialization_identity_hash = _hash_initialization_identity_from_parts(
        schema_name=PROTOEM_INITIALIZATION_BUNDLE_SCHEMA_NAME,
        schema_version=PROTOEM_INITIALIZATION_BUNDLE_SCHEMA_VERSION,
        query_identity=query_input.query_identity,
        query_patient_id=query_input.query_patient_id,
        query_case_id=query_input.query_case_id,
        encoder_identity=query_input.query_feature_encoding.metadata.encoder_identity,
        preprocessing_hash=query_input.query_feature_encoding.metadata.preprocessing_hash,
        checkpoint_hash=query_input.query_feature_encoding.metadata.checkpoint_hash,
        dataset_manifest_hash=query_input.dataset_manifest_hash,
        feature_stage=query_input.query_feature_encoding.metadata.feature_stage,
        normalization_name=query_input.query_feature_encoding.metadata.normalization_name,
        query_feature_content_hash=query_feature_hash,
        foreground_prototype_identity_hash=foreground.prototype_identity_sha256,
        background_prototype_identity_hash=background.prototype_identity_sha256,
        phase5_inference_identity_hash=inference.inference_identity_sha256,
        initial_prediction_content_hash=_array_content_sha256(prediction_copy),
        foreground_score_content_hash=_array_content_sha256(foreground_score_copy),
        background_score_content_hash=_array_content_sha256(background_score_copy),
        ordered_support_provenance=ordered_support_provenance,
        contains_cosine_scores=True,
        contains_posterior_probabilities=False,
        contains_hard_assignments=True,
    )
    return ProtoEMInitializationBundle(
        schema_name=PROTOEM_INITIALIZATION_BUNDLE_SCHEMA_NAME,
        schema_version=PROTOEM_INITIALIZATION_BUNDLE_SCHEMA_VERSION,
        initialization_identity_hash=initialization_identity_hash,
        query_feature_encoding=FeatureEncoding3D(
            feature_data=query_feature_copy,
            metadata=query_input.query_feature_encoding.metadata,
        ),
        query_identity=query_input.query_identity,
        query_patient_id=query_input.query_patient_id,
        query_case_id=query_input.query_case_id,
        query_feature_content_sha256=query_feature_hash,
        foreground_prototype=foreground,
        background_prototype=background,
        initial_foreground_score_map=foreground_score_copy,
        initial_background_score_map=background_score_copy,
        initial_prediction_mask=prediction_copy,
        initial_posterior_probability_map=None,
        contains_cosine_scores=True,
        contains_posterior_probabilities=False,
        contains_hard_assignments=True,
        ordered_support_provenance=ordered_support_provenance,
        phase5_initialization_reference=phase5_references.initialization_reference,
        phase5_prototype_reference=phase5_references.prototype_reference,
        phase5_inference_reference=phase5_references.inference_reference,
        phase5_inference_identity_hash=inference.inference_identity_sha256,
        encoder_identity=query_input.query_feature_encoding.metadata.encoder_identity,
        preprocessing_hash=query_input.query_feature_encoding.metadata.preprocessing_hash,
        checkpoint_hash=query_input.query_feature_encoding.metadata.checkpoint_hash,
        dataset_manifest_hash=query_input.dataset_manifest_hash,
        feature_stage=query_input.query_feature_encoding.metadata.feature_stage,
        normalization_name=query_input.query_feature_encoding.metadata.normalization_name,
    )


def protoem_initialization_identity_payload(
    bundle: ProtoEMInitializationBundle,
) -> dict[str, JsonValue]:
    """Return the exact scientific identity payload for one initialization bundle."""

    support_payload: list[JsonValue] = [
        cast(JsonValue, protoem_support_provenance_to_dict(item))
        for item in bundle.ordered_support_provenance
    ]
    return {
        "schema_name": bundle.schema_name,
        "schema_version": bundle.schema_version,
        "query_identity": bundle.query_identity,
        "query_patient_id": bundle.query_patient_id,
        "query_case_id": bundle.query_case_id,
        "encoder_identity": bundle.encoder_identity,
        "preprocessing_hash": bundle.preprocessing_hash,
        "checkpoint_hash": bundle.checkpoint_hash,
        "dataset_manifest_hash": bundle.dataset_manifest_hash,
        "feature_stage": bundle.feature_stage,
        "normalization_name": bundle.normalization_name,
        "query_feature_content_hash": bundle.query_feature_content_sha256,
        "foreground_prototype_identity_hash": (
            bundle.foreground_prototype.prototype_identity_sha256
        ),
        "background_prototype_identity_hash": (
            bundle.background_prototype.prototype_identity_sha256
        ),
        "phase5_inference_identity_hash": bundle.phase5_inference_identity_hash,
        "initial_prediction_content_hash": _array_content_sha256(bundle.initial_prediction_mask),
        "foreground_score_content_hash": _array_content_sha256(bundle.initial_foreground_score_map),
        "background_score_content_hash": _array_content_sha256(bundle.initial_background_score_map),
        "canonical_ordered_support_provenance": support_payload,
        "contains_cosine_scores": bundle.contains_cosine_scores,
        "contains_posterior_probabilities": bundle.contains_posterior_probabilities,
        "contains_hard_assignments": bundle.contains_hard_assignments,
    }


def hash_protoem_initialization_bundle(bundle: ProtoEMInitializationBundle) -> str:
    """Return the deterministic initialization identity hash."""

    return sha256_json(protoem_initialization_identity_payload(bundle))


def _hash_initialization_identity_from_parts(
    *,
    schema_name: str,
    schema_version: str,
    query_identity: str,
    query_patient_id: str,
    query_case_id: str,
    encoder_identity: str,
    preprocessing_hash: str,
    checkpoint_hash: str,
    dataset_manifest_hash: str | None,
    feature_stage: str,
    normalization_name: str,
    query_feature_content_hash: str,
    foreground_prototype_identity_hash: str,
    background_prototype_identity_hash: str,
    phase5_inference_identity_hash: str,
    initial_prediction_content_hash: str,
    foreground_score_content_hash: str,
    background_score_content_hash: str,
    ordered_support_provenance: Sequence[ProtoEMSupportProvenance],
    contains_cosine_scores: bool,
    contains_posterior_probabilities: bool,
    contains_hard_assignments: bool,
) -> str:
    payload = _initialization_identity_payload_from_parts(
        schema_name=schema_name,
        schema_version=schema_version,
        query_identity=query_identity,
        query_patient_id=query_patient_id,
        query_case_id=query_case_id,
        encoder_identity=encoder_identity,
        preprocessing_hash=preprocessing_hash,
        checkpoint_hash=checkpoint_hash,
        dataset_manifest_hash=dataset_manifest_hash,
        feature_stage=feature_stage,
        normalization_name=normalization_name,
        query_feature_content_hash=query_feature_content_hash,
        foreground_prototype_identity_hash=foreground_prototype_identity_hash,
        background_prototype_identity_hash=background_prototype_identity_hash,
        phase5_inference_identity_hash=phase5_inference_identity_hash,
        initial_prediction_content_hash=initial_prediction_content_hash,
        foreground_score_content_hash=foreground_score_content_hash,
        background_score_content_hash=background_score_content_hash,
        ordered_support_provenance=ordered_support_provenance,
        contains_cosine_scores=contains_cosine_scores,
        contains_posterior_probabilities=contains_posterior_probabilities,
        contains_hard_assignments=contains_hard_assignments,
    )
    return sha256_json(payload)


def _initialization_identity_payload_from_parts(
    *,
    schema_name: str,
    schema_version: str,
    query_identity: str,
    query_patient_id: str,
    query_case_id: str,
    encoder_identity: str,
    preprocessing_hash: str,
    checkpoint_hash: str,
    dataset_manifest_hash: str | None,
    feature_stage: str,
    normalization_name: str,
    query_feature_content_hash: str,
    foreground_prototype_identity_hash: str,
    background_prototype_identity_hash: str,
    phase5_inference_identity_hash: str,
    initial_prediction_content_hash: str,
    foreground_score_content_hash: str,
    background_score_content_hash: str,
    ordered_support_provenance: Sequence[ProtoEMSupportProvenance],
    contains_cosine_scores: bool,
    contains_posterior_probabilities: bool,
    contains_hard_assignments: bool,
) -> dict[str, JsonValue]:
    support_payload: list[JsonValue] = [
        cast(JsonValue, protoem_support_provenance_to_dict(item))
        for item in ordered_support_provenance
    ]
    return {
        "schema_name": schema_name,
        "schema_version": schema_version,
        "query_identity": query_identity,
        "query_patient_id": query_patient_id,
        "query_case_id": query_case_id,
        "encoder_identity": encoder_identity,
        "preprocessing_hash": preprocessing_hash,
        "checkpoint_hash": checkpoint_hash,
        "dataset_manifest_hash": dataset_manifest_hash,
        "feature_stage": feature_stage,
        "normalization_name": normalization_name,
        "query_feature_content_hash": query_feature_content_hash,
        "foreground_prototype_identity_hash": foreground_prototype_identity_hash,
        "background_prototype_identity_hash": background_prototype_identity_hash,
        "phase5_inference_identity_hash": phase5_inference_identity_hash,
        "initial_prediction_content_hash": initial_prediction_content_hash,
        "foreground_score_content_hash": foreground_score_content_hash,
        "background_score_content_hash": background_score_content_hash,
        "canonical_ordered_support_provenance": support_payload,
        "contains_cosine_scores": contains_cosine_scores,
        "contains_posterior_probabilities": contains_posterior_probabilities,
        "contains_hard_assignments": contains_hard_assignments,
    }


def protoem_initialization_bundle_to_dict(
    bundle: ProtoEMInitializationBundle,
) -> dict[str, JsonValue]:
    """Convert one ProtoEM initialization bundle to a canonical mapping."""

    query_array = _feature_array(bundle.query_feature_encoding)
    support_payload: list[JsonValue] = [
        cast(JsonValue, protoem_support_provenance_to_dict(item))
        for item in bundle.ordered_support_provenance
    ]
    return {
        "schema_name": bundle.schema_name,
        "schema_version": bundle.schema_version,
        "initialization_identity_hash": bundle.initialization_identity_hash,
        "query_identity": bundle.query_identity,
        "query_patient_id": bundle.query_patient_id,
        "query_case_id": bundle.query_case_id,
        "query_feature_shape": cast(JsonValue, list(query_array.shape)),
        "query_feature_dtype": str(query_array.dtype),
        "query_feature_content_sha256": bundle.query_feature_content_sha256,
        "foreground_prototype_identity_hash": (
            bundle.foreground_prototype.prototype_identity_sha256
        ),
        "background_prototype_identity_hash": (
            bundle.background_prototype.prototype_identity_sha256
        ),
        "phase5_inference_identity_hash": bundle.phase5_inference_identity_hash,
        "phase5_initialization_reference": cast(
            JsonValue,
            protoem_phase5_artifact_reference_to_dict(bundle.phase5_initialization_reference),
        ),
        "phase5_prototype_reference": cast(
            JsonValue,
            protoem_phase5_artifact_reference_to_dict(bundle.phase5_prototype_reference),
        ),
        "phase5_inference_reference": cast(
            JsonValue,
            protoem_phase5_artifact_reference_to_dict(bundle.phase5_inference_reference),
        ),
        "encoder_identity": bundle.encoder_identity,
        "preprocessing_hash": bundle.preprocessing_hash,
        "checkpoint_hash": bundle.checkpoint_hash,
        "dataset_manifest_hash": bundle.dataset_manifest_hash,
        "feature_stage": bundle.feature_stage,
        "normalization_name": bundle.normalization_name,
        "initial_foreground_score_shape": cast(
            JsonValue, list(bundle.initial_foreground_score_map.shape)
        ),
        "initial_background_score_shape": cast(
            JsonValue, list(bundle.initial_background_score_map.shape)
        ),
        "initial_prediction_shape": cast(JsonValue, list(bundle.initial_prediction_mask.shape)),
        "initial_foreground_score_dtype": str(bundle.initial_foreground_score_map.dtype),
        "initial_background_score_dtype": str(bundle.initial_background_score_map.dtype),
        "initial_prediction_dtype": str(bundle.initial_prediction_mask.dtype),
        "contains_cosine_scores": bundle.contains_cosine_scores,
        "contains_posterior_probabilities": bundle.contains_posterior_probabilities,
        "contains_hard_assignments": bundle.contains_hard_assignments,
        "ordered_support_provenance": support_payload,
    }


def protoem_initialization_bundle_to_json(bundle: ProtoEMInitializationBundle) -> bytes:
    """Serialize one initialization bundle summary surface to canonical JSON bytes."""

    return canonical_json_bytes(protoem_initialization_bundle_to_dict(bundle)) + b"\n"


def _validated_query_feature_array(query_feature_encoding: FeatureEncoding3D) -> np.ndarray:
    array = _feature_array(query_feature_encoding)
    if array.ndim != 5:
        raise InvalidQueryFeatureError("query feature tensor must have rank 5 [1,C,D,H,W].")
    if int(array.shape[0]) != 1:
        raise InvalidQueryFeatureError("query feature tensor batch dimension must equal 1.")
    if array.dtype.kind not in _FLOATING_KINDS:
        raise InvalidQueryFeatureError("query feature tensor must use a floating dtype.")
    if not np.isfinite(array).all():
        raise InvalidQueryFeatureError("query feature tensor must be finite.")

    metadata = query_feature_encoding.metadata
    if int(array.shape[1]) != metadata.feature_channels:
        raise InvalidQueryFeatureError(
            "query feature tensor channel dimension must match metadata.feature_channels."
        )
    if _spatial_shape(array) != metadata.resolution.feature_spatial_shape:
        raise InvalidQueryFeatureError(
            "query feature tensor spatial shape must match "
            "metadata.resolution.feature_spatial_shape."
        )
    return np.ascontiguousarray(array)


def _validated_prototype(
    *,
    prototype: ForegroundPrototype | BackgroundPrototype,
    role: str,
) -> ForegroundPrototype | BackgroundPrototype:
    expected_hash = hash_support_prototype_identity(prototype)
    if prototype.prototype_identity_sha256 != expected_hash:
        raise MalformedPhase5ArtifactError(
            f"{role} prototype identity hash does not match deterministic content."
        )
    vector = np.asarray(prototype.prototype_vector, dtype=np.float64)
    if vector.ndim != 1 or vector.size != prototype.feature_channels:
        raise IncompatibleInitializationError(
            f"{role} prototype vector length must equal feature_channels."
        )
    if not np.isfinite(vector).all():
        raise IncompatibleInitializationError(f"{role} prototype vector must be finite.")
    if math.isclose(float(np.linalg.norm(vector)), 0.0):
        raise IncompatibleInitializationError(f"{role} prototype vector must have nonzero norm.")
    return prototype


def _validated_inference_result(result: PrototypeInferenceResult) -> PrototypeInferenceResult:
    if result.inference_identity_sha256 != hash_prototype_inference_identity(result):
        raise MalformedPhase5ArtifactError(
            "Phase 5 inference identity hash does not match deterministic content."
        )
    return result


def _validate_query_identity_alignment(
    *,
    query_input: ProtoEMQueryFeatureInput,
    inference: PrototypeInferenceResult,
) -> None:
    if inference.query_identity != query_input.query_identity:
        raise IncompatibleInitializationError(
            "Phase 5 inference query_identity must match the query feature input."
        )
    if inference.query_patient_id != query_input.query_patient_id:
        raise IncompatibleInitializationError(
            "Phase 5 inference query_patient_id must match the query feature input."
        )
    if inference.query_case_id != query_input.query_case_id:
        raise IncompatibleInitializationError(
            "Phase 5 inference query_case_id must match the query feature input."
        )


def _validate_phase5_metadata_compatibility(
    *,
    query_feature_encoding: FeatureEncoding3D,
    foreground: ForegroundPrototype,
    background: BackgroundPrototype,
    inference: PrototypeInferenceResult,
    support_provenance: Sequence[ProtoEMSupportProvenance],
    query_dataset_manifest_hash: str | None,
) -> None:
    metadata = query_feature_encoding.metadata
    comparisons: list[tuple[str, str | None, str | None]] = [
        ("encoder_identity", metadata.encoder_identity, foreground.encoder_identity),
        ("encoder_identity", metadata.encoder_identity, background.encoder_identity),
        ("encoder_identity", metadata.encoder_identity, inference.encoder_identity),
        ("preprocessing_hash", metadata.preprocessing_hash, foreground.preprocessing_hash),
        ("preprocessing_hash", metadata.preprocessing_hash, background.preprocessing_hash),
        ("preprocessing_hash", metadata.preprocessing_hash, inference.preprocessing_hash),
        ("checkpoint_hash", metadata.checkpoint_hash, foreground.checkpoint_hash),
        ("checkpoint_hash", metadata.checkpoint_hash, background.checkpoint_hash),
        ("checkpoint_hash", metadata.checkpoint_hash, inference.checkpoint_hash),
        (
            "dataset_manifest_hash",
            query_dataset_manifest_hash,
            foreground.dataset_manifest_hash,
        ),
        (
            "dataset_manifest_hash",
            query_dataset_manifest_hash,
            background.dataset_manifest_hash,
        ),
        (
            "dataset_manifest_hash",
            query_dataset_manifest_hash,
            inference.dataset_manifest_hash,
        ),
        ("feature_stage", metadata.feature_stage, foreground.feature_stage),
        ("feature_stage", metadata.feature_stage, background.feature_stage),
        ("feature_stage", metadata.feature_stage, inference.feature_stage),
        ("normalization_name", metadata.normalization_name, foreground.normalization_name),
        ("normalization_name", metadata.normalization_name, background.normalization_name),
        ("normalization_name", metadata.normalization_name, inference.normalization_name),
    ]
    for field_name, expected, observed in comparisons:
        if expected != observed:
            raise IncompatibleInitializationError(
                f"Phase 5 initialization inputs must match on {field_name}."
            )

    if foreground.feature_channels != metadata.feature_channels:
        raise IncompatibleInitializationError(
            "foreground prototype feature_channels must equal query feature channels."
        )
    if background.feature_channels != metadata.feature_channels:
        raise IncompatibleInitializationError(
            "background prototype feature_channels must equal query feature channels."
        )

    if tuple(int(item) for item in inference.feature_shape) != tuple(
        int(item) for item in _feature_array(query_feature_encoding).shape
    ):
        raise IncompatibleInitializationError(
            "inference feature_shape must match the query feature tensor shape."
        )
    if foreground.prototype_identity_sha256 != inference.foreground_prototype_identity:
        raise IncompatibleInitializationError(
            "foreground prototype identity must match the Phase 5 inference record."
        )
    if background.prototype_identity_sha256 != inference.background_prototype_identity:
        raise IncompatibleInitializationError(
            "background prototype identity must match the Phase 5 inference record."
        )

    for item in support_provenance:
        item_comparisons: list[tuple[str, str | None, str | None]] = [
            ("encoder_identity", metadata.encoder_identity, item.encoder_identity),
            ("preprocessing_hash", metadata.preprocessing_hash, item.preprocessing_hash),
            ("checkpoint_hash", metadata.checkpoint_hash, item.checkpoint_hash),
            (
                "dataset_manifest_hash",
                query_dataset_manifest_hash,
                item.dataset_manifest_hash,
            ),
            ("feature_stage", metadata.feature_stage, item.feature_stage),
            ("normalization_name", metadata.normalization_name, item.normalization_name),
        ]
        for field_name, expected, observed in item_comparisons:
            if expected != observed:
                raise IncompatibleInitializationError(
                    f"support provenance must match on {field_name}."
                )


def _validate_leakage_constraints(
    *,
    query_identity: str,
    query_patient_id: str,
    query_case_id: str,
    foreground: ForegroundPrototype,
    background: BackgroundPrototype,
    support_provenance: Sequence[ProtoEMSupportProvenance],
) -> None:
    for item in support_provenance:
        if item.support_patient_id == query_patient_id:
            raise InitializationLeakageViolationError(
                "query/support patient overlap is not allowed."
            )
        if item.support_case_id == query_case_id:
            raise InitializationLeakageViolationError("query/support case overlap is not allowed.")
        if (
            item.support_identifier == query_identity
            or item.source_input_identity == query_identity
        ):
            raise InitializationLeakageViolationError(
                "query identity must not appear in support provenance."
            )

    if query_identity in foreground.source_support_identifiers:
        raise InitializationLeakageViolationError(
            "query identity must not appear in foreground prototype provenance."
        )
    if query_identity in background.source_support_identifiers:
        raise InitializationLeakageViolationError(
            "query identity must not appear in background prototype provenance."
        )
    if (
        query_patient_id in foreground.source_patient_ids
        or query_patient_id in background.source_patient_ids
    ):
        raise InitializationLeakageViolationError(
            "query patient ID must not appear in prototype source_patient_ids."
        )
    if query_case_id in foreground.source_case_ids or query_case_id in background.source_case_ids:
        raise InitializationLeakageViolationError(
            "query case ID must not appear in prototype source_case_ids."
        )


def _validated_score_map(
    array: np.ndarray,
    *,
    field_name: str,
    expected_spatial_shape: tuple[int, int, int],
) -> np.ndarray:
    if not isinstance(array, np.ndarray):
        raise InvalidInitialScoreStateError(f"{field_name} must be a NumPy array.")
    if array.ndim != 5 or tuple(int(item) for item in array.shape[:2]) != (1, 1):
        raise InvalidInitialScoreStateError(f"{field_name} must have shape [1,1,D,H,W].")
    if _spatial_shape(array) != expected_spatial_shape:
        raise InvalidInitialScoreStateError(
            f"{field_name} spatial shape must match the query feature spatial shape."
        )
    if array.dtype.kind not in _NUMERIC_KINDS:
        raise InvalidInitialScoreStateError(f"{field_name} must use a numeric dtype.")
    if not np.isfinite(array).all():
        raise InvalidInitialScoreStateError(f"{field_name} must contain only finite values.")
    return np.ascontiguousarray(array.astype(np.float64, copy=False))


def _validated_prediction_mask(
    array: np.ndarray,
    *,
    expected_spatial_shape: tuple[int, int, int],
) -> np.ndarray:
    if not isinstance(array, np.ndarray):
        raise InvalidInitialPredictionError("prediction_mask must be a NumPy array.")
    if array.ndim != 5 or tuple(int(item) for item in array.shape[:2]) != (1, 1):
        raise InvalidInitialPredictionError("prediction_mask must have shape [1,1,D,H,W].")
    if _spatial_shape(array) != expected_spatial_shape:
        raise InvalidInitialPredictionError(
            "prediction_mask spatial shape must match the query feature spatial shape."
        )
    if not np.isfinite(array).all():
        raise InvalidInitialPredictionError("prediction_mask must contain only finite values.")
    if set(int(item) for item in np.unique(array).tolist()) - _BINARY_VALUES:
        raise InvalidInitialPredictionError("prediction_mask must be binary.")
    return np.ascontiguousarray(array.astype(_PREDICTION_DTYPE, copy=False))


def _validate_phase5_hash_consistency(
    *,
    inference: PrototypeInferenceResult,
    foreground_scores: np.ndarray,
    background_scores: np.ndarray,
    prediction_mask: np.ndarray,
) -> None:
    if _array_content_sha256(foreground_scores) != inference.foreground_score_content_sha256:
        raise MalformedPhase5ArtifactError(
            "foreground_score_content_sha256 does not match the supplied score map."
        )
    if _array_content_sha256(background_scores) != inference.background_score_content_sha256:
        raise MalformedPhase5ArtifactError(
            "background_score_content_sha256 does not match the supplied score map."
        )
    if _array_content_sha256(prediction_mask) != inference.prediction_content_sha256:
        raise MalformedPhase5ArtifactError(
            "prediction_content_sha256 does not match the supplied prediction mask."
        )


def _validate_prototype_source_alignment(
    *,
    foreground: ForegroundPrototype,
    background: BackgroundPrototype,
    support_provenance: Sequence[ProtoEMSupportProvenance],
) -> None:
    provenance_identifiers = tuple(item.support_identifier for item in support_provenance)
    provenance_patient_ids = tuple(sorted(item.support_patient_id for item in support_provenance))
    provenance_case_ids = tuple(sorted(item.support_case_id for item in support_provenance))
    if foreground.source_support_identifiers != provenance_identifiers:
        raise IncompatibleInitializationError(
            "foreground prototype source_support_identifiers must match support provenance."
        )
    if background.source_support_identifiers != provenance_identifiers:
        raise IncompatibleInitializationError(
            "background prototype source_support_identifiers must match support provenance."
        )
    if foreground.source_patient_ids != provenance_patient_ids:
        raise IncompatibleInitializationError(
            "foreground prototype source_patient_ids must match support provenance."
        )
    if background.source_patient_ids != provenance_patient_ids:
        raise IncompatibleInitializationError(
            "background prototype source_patient_ids must match support provenance."
        )
    if foreground.source_case_ids != provenance_case_ids:
        raise IncompatibleInitializationError(
            "foreground prototype source_case_ids must match support provenance."
        )
    if background.source_case_ids != provenance_case_ids:
        raise IncompatibleInitializationError(
            "background prototype source_case_ids must match support provenance."
        )


def _require_phase5_reference_match(
    *,
    expected_schema_name: str,
    expected_hash: str,
    observed: ProtoEMPhase5ArtifactReference,
    field_name: str,
) -> None:
    if (
        observed.artifact_schema_name != expected_schema_name
        or observed.artifact_hash != expected_hash
    ):
        raise IncompatibleInitializationError(
            f"{field_name} must match the exact Phase 5 reference required by ProtoEMConfig."
        )


def _feature_array(feature_encoding: FeatureEncoding3D) -> np.ndarray:
    array = feature_encoding.feature_data
    if not isinstance(array, np.ndarray):
        raise InvalidQueryFeatureError("query feature_data must be a NumPy array.")
    return array


def _array_content_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def _require_sha256(value: str, *, field_name: str) -> None:
    if not _SHA256_RE.fullmatch(value):
        raise IncompatibleInitializationError(
            f"{field_name} must be a lowercase 64-character SHA-256 hex digest."
        )


def _require_identifier(value: str, *, field_name: str) -> None:
    if not _IDENTIFIER_RE.fullmatch(value):
        raise IncompatibleInitializationError(f"{field_name} must be a conservative identifier.")


def _require_optional_sha256(value: str | None, *, field_name: str) -> None:
    if value is None:
        return
    _require_sha256(value, field_name=field_name)


def _spatial_shape(array: np.ndarray) -> tuple[int, int, int]:
    return (int(array.shape[2]), int(array.shape[3]), int(array.shape[4]))


__all__ = [
    "PROTOEM_INITIALIZATION_BUNDLE_SCHEMA_NAME",
    "PROTOEM_INITIALIZATION_BUNDLE_SCHEMA_VERSION",
    "IncompatibleInitializationError",
    "InitializationIdentityMismatchError",
    "InitializationLeakageViolationError",
    "InvalidInitialPredictionError",
    "InvalidInitialScoreStateError",
    "InvalidQueryFeatureError",
    "MalformedPhase5ArtifactError",
    "ProtoEMInitializationBundle",
    "ProtoEMInitializationError",
    "build_protoem_initialization_bundle",
    "hash_protoem_initialization_bundle",
    "protoem_initialization_bundle_to_dict",
    "protoem_initialization_bundle_to_json",
    "protoem_initialization_identity_payload",
]
