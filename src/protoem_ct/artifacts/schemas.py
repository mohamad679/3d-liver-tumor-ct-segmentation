"""Typed artifact schemas and deterministic JSON serialization."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any, ClassVar, NoReturn, TypeAlias, TypeVar, cast, overload

from protoem_ct.artifacts.hashing import JsonValue

ARTIFACT_SCHEMA_VERSION = "1"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ArtifactError(ValueError):
    """Base error for project-specific artifact contract failures."""


class ArtifactSchemaVersionError(ArtifactError):
    """Raised when artifact JSON uses an unsupported schema version."""


class ArtifactStageError(ArtifactError):
    """Raised when artifact JSON has the wrong stage for its schema."""


class ArtifactValidationError(ArtifactError):
    """Raised when artifact content violates the project artifact contract."""


class ArtifactSerializationError(ArtifactError):
    """Raised when artifact JSON cannot be parsed or serialized."""


@dataclass(frozen=True, slots=True)
class _ArtifactBase:
    """Common metadata required by every persisted artifact."""

    schema_version: str
    stage: str
    created_at_utc: str
    git_commit: str
    config_hash: str
    manifest_hash: str

    _expected_stage: ClassVar[str]

    def __post_init__(self) -> None:
        _require_exact_schema_version(self.schema_version)
        _require_stage(self.stage, self._expected_stage)
        _require_nonempty_string(self.created_at_utc, "created_at_utc")
        _require_nonempty_string(self.git_commit, "git_commit")
        _require_sha256(self.config_hash, "config_hash")
        _require_sha256(self.manifest_hash, "manifest_hash")


@dataclass(frozen=True, slots=True)
class RunMetadata(_ArtifactBase):
    """Metadata describing one reproducible local pipeline run."""

    run_id: str
    phase: str
    command: tuple[str, ...]

    _expected_stage: ClassVar[str] = "run-metadata"

    def __post_init__(self) -> None:
        _ArtifactBase.__post_init__(self)
        _require_nonempty_string(self.run_id, "run_id")
        _require_nonempty_string(self.phase, "phase")
        _require_string_tuple(self.command, "command", allow_empty=False)


@dataclass(frozen=True, slots=True)
class SyntheticManifest(_ArtifactBase):
    """Manifest for deterministic synthetic image-label inputs."""

    dataset_id: str
    seed: int
    case_ids: tuple[str, ...]
    image_paths: tuple[str, ...]
    label_paths: tuple[str, ...]
    shape: tuple[int, int, int]
    spacing: tuple[float, float, float]

    _expected_stage: ClassVar[str] = "synthetic-manifest"

    def __post_init__(self) -> None:
        _ArtifactBase.__post_init__(self)
        _require_nonempty_string(self.dataset_id, "dataset_id")
        _require_nonnegative_int(self.seed, "seed")
        _require_string_tuple(self.case_ids, "case_ids", allow_empty=False)
        if len(set(self.case_ids)) != len(self.case_ids):
            msg = "case_ids must be unique"
            raise ArtifactValidationError(msg)
        _require_string_tuple(self.image_paths, "image_paths", allow_empty=False)
        _require_string_tuple(self.label_paths, "label_paths", allow_empty=False)
        if len(self.case_ids) != len(self.image_paths) or len(self.case_ids) != len(
            self.label_paths
        ):
            msg = "case_ids, image_paths, and label_paths must have equal lengths"
            raise ArtifactValidationError(msg)
        for path in (*self.image_paths, *self.label_paths):
            _require_relative_project_path(path)
        _require_shape_3d(self.shape, "shape")
        _require_spacing_3d(self.spacing, "spacing")


@dataclass(frozen=True, slots=True)
class ValidationArtifact(_ArtifactBase):
    """Artifact produced by validation over a synthetic manifest."""

    valid_case_count: int
    invalid_case_count: int
    validated_case_ids: tuple[str, ...]

    _expected_stage: ClassVar[str] = "validate"

    def __post_init__(self) -> None:
        _ArtifactBase.__post_init__(self)
        _require_nonnegative_int(self.valid_case_count, "valid_case_count")
        _require_nonnegative_int(self.invalid_case_count, "invalid_case_count")
        _require_string_tuple(self.validated_case_ids, "validated_case_ids", allow_empty=True)


@dataclass(frozen=True, slots=True)
class PreprocessArtifact(_ArtifactBase):
    """Artifact produced by synthetic preprocessing."""

    output_case_ids: tuple[str, ...]
    output_image_paths: tuple[str, ...]
    output_label_paths: tuple[str, ...]
    target_spacing: tuple[float, float, float]
    preprocessing_parameters: Mapping[str, JsonValue]

    _expected_stage: ClassVar[str] = "preprocess"

    def __post_init__(self) -> None:
        _ArtifactBase.__post_init__(self)
        _require_string_tuple(self.output_case_ids, "output_case_ids", allow_empty=False)
        _require_string_tuple(self.output_image_paths, "output_image_paths", allow_empty=False)
        _require_string_tuple(self.output_label_paths, "output_label_paths", allow_empty=False)
        if len(self.output_case_ids) != len(self.output_image_paths) or len(
            self.output_case_ids
        ) != len(self.output_label_paths):
            msg = (
                "output_case_ids, output_image_paths, and output_label_paths must have "
                "equal lengths"
            )
            raise ArtifactValidationError(msg)
        for path in (*self.output_image_paths, *self.output_label_paths):
            _require_relative_project_path(path)
        _require_spacing_3d(self.target_spacing, "target_spacing")
        _require_json_mapping(self.preprocessing_parameters, "preprocessing_parameters")


@dataclass(frozen=True, slots=True)
class InferenceArtifact(_ArtifactBase):
    """Artifact produced by deterministic dummy inference."""

    prediction_case_ids: tuple[str, ...]
    method: str
    deterministic_seed: int

    _expected_stage: ClassVar[str] = "infer-dummy"

    def __post_init__(self) -> None:
        _ArtifactBase.__post_init__(self)
        _require_string_tuple(self.prediction_case_ids, "prediction_case_ids", allow_empty=False)
        if self.method != "dummy":
            msg = "method must be fixed to 'dummy'"
            raise ArtifactValidationError(msg)
        _require_nonnegative_int(self.deterministic_seed, "deterministic_seed")


@dataclass(frozen=True, slots=True)
class EvaluationArtifact(_ArtifactBase):
    """Artifact produced by metric evaluation over persisted predictions."""

    metric_name: str
    per_case_values: Mapping[str, float]
    aggregate_value: float
    valid_case_count: int

    _expected_stage: ClassVar[str] = "evaluate"

    def __post_init__(self) -> None:
        _ArtifactBase.__post_init__(self)
        _require_nonempty_string(self.metric_name, "metric_name")
        _require_float_mapping(self.per_case_values, "per_case_values", allow_empty=False)
        _require_finite_float(self.aggregate_value, "aggregate_value")
        _require_nonnegative_int(self.valid_case_count, "valid_case_count")


@dataclass(frozen=True, slots=True)
class ReportArtifact(_ArtifactBase):
    """Artifact describing a report generated only from persisted JSON inputs."""

    source_artifact_paths: tuple[str, ...]
    report_path: str
    reported_metric_names: tuple[str, ...]

    _expected_stage: ClassVar[str] = "report"

    def __post_init__(self) -> None:
        _ArtifactBase.__post_init__(self)
        _require_string_tuple(
            self.source_artifact_paths,
            "source_artifact_paths",
            allow_empty=False,
        )
        for path in self.source_artifact_paths:
            _require_relative_project_path(path)
        _require_relative_project_path(self.report_path)
        _require_string_tuple(
            self.reported_metric_names,
            "reported_metric_names",
            allow_empty=False,
        )


Artifact: TypeAlias = (
    RunMetadata
    | SyntheticManifest
    | ValidationArtifact
    | PreprocessArtifact
    | InferenceArtifact
    | EvaluationArtifact
    | ReportArtifact
)
ArtifactT = TypeVar("ArtifactT", bound=Artifact)

_ARTIFACT_TYPES: tuple[type[Artifact], ...] = (
    RunMetadata,
    SyntheticManifest,
    ValidationArtifact,
    PreprocessArtifact,
    InferenceArtifact,
    EvaluationArtifact,
    ReportArtifact,
)
_ARTIFACT_STAGES = {
    artifact_type._expected_stage: artifact_type for artifact_type in _ARTIFACT_TYPES
}
_SEQUENCE_FIELDS = {
    "command",
    "case_ids",
    "image_paths",
    "label_paths",
    "shape",
    "spacing",
    "validated_case_ids",
    "output_case_ids",
    "output_image_paths",
    "output_label_paths",
    "target_spacing",
    "prediction_case_ids",
    "source_artifact_paths",
    "reported_metric_names",
}
_MAPPING_FIELDS = {"preprocessing_parameters", "per_case_values"}


def artifact_to_dict(artifact: Artifact) -> dict[str, JsonValue]:
    """Convert a known artifact dataclass into a JSON-compatible dictionary."""
    if not isinstance(artifact, _ARTIFACT_TYPES):
        msg = f"unsupported artifact type: {type(artifact).__name__}"
        raise ArtifactSerializationError(msg)
    return cast(dict[str, JsonValue], _jsonify_dataclass(artifact))


def artifact_to_json(artifact: Artifact, path: Path | None = None) -> str:
    """Serialize an artifact with deterministic key ordering and a final newline."""
    try:
        text = json.dumps(
            artifact_to_dict(artifact),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        msg = f"failed to serialize artifact: {exc}"
        raise ArtifactSerializationError(msg) from exc

    text_with_newline = f"{text}\n"
    if path is not None:
        path.write_text(text_with_newline, encoding="utf-8", newline="\n")
    return text_with_newline


@overload
def artifact_from_json(json_text: str, artifact_type: type[ArtifactT]) -> ArtifactT: ...


@overload
def artifact_from_json(json_text: str, artifact_type: None = None) -> Artifact: ...


def artifact_from_json(
    json_text: str,
    artifact_type: type[ArtifactT] | None = None,
) -> ArtifactT | Artifact:
    """Parse artifact JSON into a typed artifact with explicit schema and stage checks."""
    try:
        raw_value = json.loads(
            json_text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_bad_json_constant,
        )
    except json.JSONDecodeError as exc:
        msg = f"invalid artifact JSON: {exc}"
        raise ArtifactSerializationError(msg) from exc

    if not isinstance(raw_value, dict):
        msg = "artifact JSON must contain an object"
        raise ArtifactSerializationError(msg)

    raw = cast(dict[str, object], raw_value)
    schema_version = raw.get("schema_version")
    stage = raw.get("stage")
    _require_exact_schema_version(schema_version)

    if artifact_type is None:
        if not isinstance(stage, str) or stage not in _ARTIFACT_STAGES:
            msg = f"unsupported artifact stage: {stage!r}"
            raise ArtifactStageError(msg)
        resolved_artifact_type = _ARTIFACT_STAGES[stage]
    else:
        resolved_artifact_type = cast(type[Artifact], artifact_type)
        _require_stage(stage, resolved_artifact_type._expected_stage)

    return _construct_artifact(raw, resolved_artifact_type)


def _construct_artifact(raw: dict[str, object], artifact_type: type[Artifact]) -> Artifact:
    expected_fields = {field.name for field in fields(artifact_type)}
    extra_fields = set(raw) - expected_fields
    missing_fields = expected_fields - set(raw)
    if extra_fields:
        msg = f"unexpected fields for {artifact_type.__name__}: {sorted(extra_fields)}"
        raise ArtifactValidationError(msg)
    if missing_fields:
        msg = f"missing fields for {artifact_type.__name__}: {sorted(missing_fields)}"
        raise ArtifactValidationError(msg)

    kwargs: dict[str, object] = {}
    for key, value in raw.items():
        if key in _SEQUENCE_FIELDS:
            if not isinstance(value, list):
                msg = f"{key} must be a JSON array"
                raise ArtifactValidationError(msg)
            kwargs[key] = tuple(value)
        elif key in _MAPPING_FIELDS:
            if not isinstance(value, dict):
                msg = f"{key} must be a JSON object"
                raise ArtifactValidationError(msg)
            kwargs[key] = MappingProxyType(dict(value))
        else:
            kwargs[key] = value

    return cast(Artifact, cast(Any, artifact_type)(**kwargs))


def _jsonify_dataclass(value: Artifact) -> JsonValue:
    result: dict[str, JsonValue] = {}
    for field in fields(value):
        result[field.name] = _jsonify(getattr(value, field.name), field.name)
    return result


def _jsonify(value: object, location: str) -> JsonValue:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            msg = f"{location} must not contain NaN or Infinity"
            raise ArtifactSerializationError(msg)
        return value
    if isinstance(value, tuple):
        return [_jsonify(item, f"{location}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, Mapping):
        result: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                msg = f"{location} contains a non-string dictionary key: {key!r}"
                raise ArtifactSerializationError(msg)
            result[key] = _jsonify(item, f"{location}.{key}")
        return result
    msg = f"{location} is not JSON-compatible: {type(value).__name__}"
    raise ArtifactSerializationError(msg)


def _require_exact_schema_version(value: object) -> None:
    if value != ARTIFACT_SCHEMA_VERSION:
        msg = f"unsupported schema_version: {value!r}; expected {ARTIFACT_SCHEMA_VERSION!r}"
        raise ArtifactSchemaVersionError(msg)


def _require_stage(value: object, expected_stage: str) -> None:
    if value != expected_stage:
        msg = f"incorrect artifact stage: {value!r}; expected {expected_stage!r}"
        raise ArtifactStageError(msg)


def _require_nonempty_string(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value:
        msg = f"{field_name} must be a nonempty string"
        raise ArtifactValidationError(msg)


def _require_sha256(value: object, field_name: str) -> None:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        msg = f"{field_name} must be a lowercase SHA-256 hexadecimal string"
        raise ArtifactValidationError(msg)


def _require_nonnegative_int(value: object, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        msg = f"{field_name} must be a nonnegative integer"
        raise ArtifactValidationError(msg)


def _require_finite_float(value: object, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, float) or not math.isfinite(value):
        msg = f"{field_name} must be a finite float"
        raise ArtifactValidationError(msg)


def _require_string_tuple(value: object, field_name: str, *, allow_empty: bool) -> None:
    if not isinstance(value, tuple):
        msg = f"{field_name} must be a tuple of strings"
        raise ArtifactValidationError(msg)
    if not allow_empty and not value:
        msg = f"{field_name} must not be empty"
        raise ArtifactValidationError(msg)
    if any(not isinstance(item, str) or not item for item in value):
        msg = f"{field_name} must contain only nonempty strings"
        raise ArtifactValidationError(msg)


def _require_shape_3d(value: object, field_name: str) -> None:
    if (
        not isinstance(value, tuple)
        or len(value) != 3
        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in value)
    ):
        msg = f"{field_name} must contain exactly three positive integers"
        raise ArtifactValidationError(msg)


def _require_spacing_3d(value: object, field_name: str) -> None:
    if (
        not isinstance(value, tuple)
        or len(value) != 3
        or any(
            isinstance(item, bool)
            or not isinstance(item, float)
            or not math.isfinite(item)
            or item <= 0
            for item in value
        )
    ):
        msg = f"{field_name} must contain exactly three positive finite floats"
        raise ArtifactValidationError(msg)


def _require_json_mapping(value: object, field_name: str) -> None:
    if not isinstance(value, Mapping):
        msg = f"{field_name} must be a JSON object"
        raise ArtifactValidationError(msg)
    _jsonify(value, field_name)


def _require_float_mapping(value: object, field_name: str, *, allow_empty: bool) -> None:
    if not isinstance(value, Mapping):
        msg = f"{field_name} must be a JSON object"
        raise ArtifactValidationError(msg)
    if not allow_empty and not value:
        msg = f"{field_name} must not be empty"
        raise ArtifactValidationError(msg)
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            msg = f"{field_name} must use nonempty string keys"
            raise ArtifactValidationError(msg)
        _require_finite_float(item, f"{field_name}.{key}")


def _require_relative_project_path(value: object) -> None:
    if not isinstance(value, str) or not value:
        msg = "artifact paths must be nonempty strings"
        raise ArtifactValidationError(msg)
    if (
        value.startswith("~")
        or PurePosixPath(value).is_absolute()
        or PureWindowsPath(value).is_absolute()
    ):
        msg = f"artifact path must be relative and project-local: {value}"
        raise ArtifactValidationError(msg)
    posix_parts = PurePosixPath(value).parts
    windows_parts = PureWindowsPath(value).parts
    if ".." in posix_parts or ".." in windows_parts:
        msg = f"artifact path must not contain parent traversal: {value}"
        raise ArtifactValidationError(msg)


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            msg = f"duplicate JSON object key: {key}"
            raise ArtifactSerializationError(msg)
        result[key] = value
    return result


def _bad_json_constant(value: str) -> NoReturn:
    msg = f"artifact JSON must not contain {value}"
    raise ArtifactSerializationError(msg)
