"""Deterministic saved-prediction manifest contracts for Phase 3 baselines."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, cast

from protoem_ct.artifacts.hashing import canonical_json_bytes, sha256_json

BASELINE_PREDICTION_RECORD_VERSION = "baseline_prediction_record_v1"
BASELINE_PREDICTION_MANIFEST_VERSION = "baseline_prediction_manifest_v1"
SUPPORTED_BASELINE_PREDICTION_FAMILIES: tuple[str, ...] = ("nnunet_v2", "monai_segresnet")
BASELINE_PREDICTION_FOREGROUND_LABEL = 1

_SAFE_IDENTIFIER_RE = re.compile(r"^[a-z0-9](?:[a-z0-9_-]{0,62}[a-z0-9])?$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_WINDOWS_ABSOLUTE_PATH_RE = re.compile(r"^[A-Za-z]:[\\/]")
_MANIFEST_FIELDS = {
    "artifact_hash",
    "baseline_family",
    "checkpoint_sha256",
    "contract_version",
    "dataset_manifest_sha256",
    "development_split_sha256",
    "prediction_records",
    "run_identifier",
}
_RECORD_FIELDS = {
    "affine",
    "byte_size",
    "case_identifier",
    "contract_version",
    "foreground_label",
    "prediction_path",
    "prediction_sha256",
    "shape",
    "voxel_spacing_mm",
}


class BaselinePredictionError(ValueError):
    """Base error for saved-prediction manifest contracts."""


class BaselinePredictionValidationError(BaselinePredictionError):
    """Raised when saved-prediction data violates the contract."""


class BaselinePredictionSerializationError(BaselinePredictionError):
    """Raised when saved-prediction JSON cannot be parsed safely."""


class BaselinePredictionVersionError(BaselinePredictionError):
    """Raised when an unsupported saved-prediction contract version is encountered."""


class BaselinePredictionHashError(BaselinePredictionError):
    """Raised when a persisted saved-prediction artifact hash does not match its content."""


@dataclass(frozen=True, slots=True)
class BaselinePredictionRecord:
    """Immutable metadata for one saved binary prediction file."""

    contract_version: str
    case_identifier: str
    prediction_path: str
    prediction_sha256: str
    byte_size: int
    shape: tuple[int, int, int]
    voxel_spacing_mm: tuple[float, float, float]
    affine: tuple[tuple[float, float, float, float], ...]
    foreground_label: int = BASELINE_PREDICTION_FOREGROUND_LABEL

    def __post_init__(self) -> None:
        _require_contract_version(self.contract_version, BASELINE_PREDICTION_RECORD_VERSION)
        _require_case_identifier(self.case_identifier)
        _require_relative_prediction_path(self.prediction_path)
        _require_sha256(self.prediction_sha256, field_name="prediction_sha256")
        if self.byte_size <= 0:
            raise BaselinePredictionValidationError("byte_size must be positive.")
        shape = _normalize_shape(self.shape, field_name="shape")
        spacing = _normalize_spacing(self.voxel_spacing_mm, field_name="voxel_spacing_mm")
        affine = _normalize_affine(self.affine, field_name="affine")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "voxel_spacing_mm", spacing)
        object.__setattr__(self, "affine", affine)
        if self.foreground_label != BASELINE_PREDICTION_FOREGROUND_LABEL:
            raise BaselinePredictionValidationError("foreground_label must equal 1.")
        _require_no_absolute_paths(_record_payload(self))


@dataclass(frozen=True, slots=True)
class BaselinePredictionManifest:
    """Immutable deterministic manifest for saved baseline predictions."""

    artifact_hash: str
    contract_version: str
    baseline_family: str
    run_identifier: str
    dataset_manifest_sha256: str
    development_split_sha256: str
    checkpoint_sha256: str
    prediction_records: tuple[BaselinePredictionRecord, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _require_contract_version(self.contract_version, BASELINE_PREDICTION_MANIFEST_VERSION)
        _require_sha256(self.artifact_hash, field_name="artifact_hash")
        if self.baseline_family not in SUPPORTED_BASELINE_PREDICTION_FAMILIES:
            raise BaselinePredictionValidationError(
                f"Unsupported baseline family: {self.baseline_family!r}."
            )
        _require_case_identifier(self.run_identifier)
        _require_sha256(self.dataset_manifest_sha256, field_name="dataset_manifest_sha256")
        _require_sha256(self.development_split_sha256, field_name="development_split_sha256")
        _require_sha256(self.checkpoint_sha256, field_name="checkpoint_sha256")
        sorted_records = tuple(
            sorted(self.prediction_records, key=lambda item: item.case_identifier)
        )
        if self.prediction_records != sorted_records:
            raise BaselinePredictionValidationError(
                "prediction_records must be sorted by ascending anonymous case identifier."
            )
        _require_unique_records(self.prediction_records)
        _require_no_absolute_paths(_manifest_payload(self))
        expected_hash = sha256_json(_manifest_payload_without_hash(self))
        if self.artifact_hash != expected_hash:
            raise BaselinePredictionHashError(
                "artifact_hash does not match the deterministic prediction-manifest content."
            )


def baseline_prediction_manifest_to_json(manifest: BaselinePredictionManifest) -> bytes:
    """Serialize a saved-prediction manifest deterministically."""

    return canonical_json_bytes(_manifest_payload(manifest)) + b"\n"


def baseline_prediction_manifest_from_json(
    payload: bytes | str,
) -> BaselinePredictionManifest:
    """Parse and validate a saved-prediction manifest JSON payload."""

    parsed = _parse_json_object(payload, context="baseline prediction manifest")
    _require_exact_fields(
        parsed,
        expected=_MANIFEST_FIELDS,
        context="baseline prediction manifest",
    )
    prediction_records_value = parsed["prediction_records"]
    if not isinstance(prediction_records_value, list):
        raise BaselinePredictionSerializationError("prediction_records must be a JSON array.")
    records = tuple(
        _prediction_record_from_json_value(prediction_record)
        for prediction_record in prediction_records_value
    )
    manifest = BaselinePredictionManifest(
        artifact_hash=_expect_string(parsed["artifact_hash"], field_name="artifact_hash"),
        contract_version=_expect_string(
            parsed["contract_version"],
            field_name="contract_version",
        ),
        baseline_family=_expect_string(parsed["baseline_family"], field_name="baseline_family"),
        run_identifier=_expect_string(parsed["run_identifier"], field_name="run_identifier"),
        dataset_manifest_sha256=_expect_string(
            parsed["dataset_manifest_sha256"],
            field_name="dataset_manifest_sha256",
        ),
        development_split_sha256=_expect_string(
            parsed["development_split_sha256"],
            field_name="development_split_sha256",
        ),
        checkpoint_sha256=_expect_string(
            parsed["checkpoint_sha256"],
            field_name="checkpoint_sha256",
        ),
        prediction_records=records,
    )
    expected_hash = sha256_json(_manifest_payload_without_hash(manifest))
    if manifest.artifact_hash != expected_hash:
        raise BaselinePredictionHashError(
            "artifact_hash does not match the parsed prediction-manifest content."
        )
    return manifest


def build_baseline_prediction_manifest(
    *,
    baseline_family: str,
    run_identifier: str,
    dataset_manifest_sha256: str,
    development_split_sha256: str,
    checkpoint_sha256: str,
    prediction_records: Sequence[BaselinePredictionRecord],
) -> BaselinePredictionManifest:
    """Build a deterministic saved-prediction manifest from validated records."""

    payload_without_hash = {
        "baseline_family": baseline_family,
        "checkpoint_sha256": checkpoint_sha256,
        "contract_version": BASELINE_PREDICTION_MANIFEST_VERSION,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "development_split_sha256": development_split_sha256,
        "prediction_records": [
            _record_payload(prediction_record)
            for prediction_record in sorted(
                tuple(prediction_records),
                key=lambda item: item.case_identifier,
            )
        ],
        "run_identifier": run_identifier,
    }
    return BaselinePredictionManifest(
        artifact_hash=sha256_json(payload_without_hash),
        contract_version=BASELINE_PREDICTION_MANIFEST_VERSION,
        baseline_family=baseline_family,
        run_identifier=run_identifier,
        dataset_manifest_sha256=dataset_manifest_sha256,
        development_split_sha256=development_split_sha256,
        checkpoint_sha256=checkpoint_sha256,
        prediction_records=tuple(
            sorted(tuple(prediction_records), key=lambda item: item.case_identifier)
        ),
    )


def _manifest_payload(manifest: BaselinePredictionManifest) -> dict[str, Any]:
    return {
        "artifact_hash": manifest.artifact_hash,
        "baseline_family": manifest.baseline_family,
        "checkpoint_sha256": manifest.checkpoint_sha256,
        "contract_version": manifest.contract_version,
        "dataset_manifest_sha256": manifest.dataset_manifest_sha256,
        "development_split_sha256": manifest.development_split_sha256,
        "prediction_records": [
            _record_payload(prediction_record) for prediction_record in manifest.prediction_records
        ],
        "run_identifier": manifest.run_identifier,
    }


def _manifest_payload_without_hash(manifest: BaselinePredictionManifest) -> dict[str, Any]:
    payload = _manifest_payload(manifest)
    del payload["artifact_hash"]
    return payload


def _record_payload(record: BaselinePredictionRecord) -> dict[str, Any]:
    return {
        "affine": [list(row) for row in record.affine],
        "byte_size": record.byte_size,
        "case_identifier": record.case_identifier,
        "contract_version": record.contract_version,
        "foreground_label": record.foreground_label,
        "prediction_path": record.prediction_path,
        "prediction_sha256": record.prediction_sha256,
        "shape": list(record.shape),
        "voxel_spacing_mm": list(record.voxel_spacing_mm),
    }


def _prediction_record_from_json_value(value: Any) -> BaselinePredictionRecord:
    if not isinstance(value, dict):
        raise BaselinePredictionSerializationError(
            "prediction_records entries must be JSON objects."
        )
    _require_exact_fields(
        value,
        expected=_RECORD_FIELDS,
        context="baseline prediction record",
    )
    return BaselinePredictionRecord(
        contract_version=_expect_string(value["contract_version"], field_name="contract_version"),
        case_identifier=_expect_string(value["case_identifier"], field_name="case_identifier"),
        prediction_path=_expect_string(value["prediction_path"], field_name="prediction_path"),
        prediction_sha256=_expect_string(
            value["prediction_sha256"],
            field_name="prediction_sha256",
        ),
        byte_size=_expect_int(value["byte_size"], field_name="byte_size"),
        shape=_expect_int_tuple3(value["shape"], field_name="shape"),
        voxel_spacing_mm=_expect_float_tuple3(
            value["voxel_spacing_mm"],
            field_name="voxel_spacing_mm",
        ),
        affine=_expect_affine(value["affine"], field_name="affine"),
        foreground_label=_expect_int(value["foreground_label"], field_name="foreground_label"),
    )


def _require_unique_records(records: Sequence[BaselinePredictionRecord]) -> None:
    case_identifiers = [record.case_identifier for record in records]
    if len(set(case_identifiers)) != len(case_identifiers):
        raise BaselinePredictionValidationError(
            "prediction_records must not contain duplicate case identifiers."
        )
    prediction_paths = [record.prediction_path for record in records]
    if len(set(prediction_paths)) != len(prediction_paths):
        raise BaselinePredictionValidationError(
            "prediction_records must not contain duplicate prediction_path values."
        )


def _require_contract_version(value: str, expected: str) -> None:
    if value != expected:
        raise BaselinePredictionVersionError(f"Unsupported prediction contract version: {value!r}.")


def _require_case_identifier(value: str) -> None:
    if not _SAFE_IDENTIFIER_RE.fullmatch(value):
        raise BaselinePredictionValidationError(
            "anonymous identifiers must match the conservative ASCII identifier grammar."
        )
    _require_no_absolute_path_string(value, field_name="case_identifier")


def _require_relative_prediction_path(value: str) -> None:
    if not value or "\\" in value:
        raise BaselinePredictionValidationError(
            "prediction_path must be a relative POSIX path without backslashes."
        )
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise BaselinePredictionValidationError(
            "prediction_path must not contain control characters."
        )
    pure_path = PurePosixPath(value)
    if pure_path.is_absolute():
        raise BaselinePredictionValidationError("prediction_path must not be absolute.")
    if ".." in pure_path.parts:
        raise BaselinePredictionValidationError(
            "prediction_path must not contain parent-directory traversal."
        )
    if pure_path.suffixes != [".nii", ".gz"]:
        raise BaselinePredictionValidationError("prediction_path must end with .nii.gz.")
    _require_no_absolute_path_string(value, field_name="prediction_path")


def _require_sha256(value: str, *, field_name: str) -> None:
    if not _SHA256_RE.fullmatch(value):
        raise BaselinePredictionValidationError(
            f"{field_name} must be exactly 64 lowercase hexadecimal characters."
        )


def _normalize_shape(
    value: Sequence[int],
    *,
    field_name: str,
) -> tuple[int, int, int]:
    if len(value) != 3:
        raise BaselinePredictionValidationError(f"{field_name} must contain exactly three values.")
    normalized = cast(tuple[int, int, int], tuple(value))
    for item in normalized:
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise BaselinePredictionValidationError(
                f"{field_name} must contain only positive integers."
            )
    return normalized


def _normalize_spacing(
    value: Sequence[float],
    *,
    field_name: str,
) -> tuple[float, float, float]:
    if len(value) != 3:
        raise BaselinePredictionValidationError(f"{field_name} must contain exactly three values.")
    normalized = cast(tuple[float, float, float], tuple(float(item) for item in value))
    for item in normalized:
        if not math.isfinite(item) or item <= 0.0:
            raise BaselinePredictionValidationError(
                f"{field_name} must contain only finite positive values."
            )
    return normalized


def _normalize_affine(
    value: Sequence[Sequence[float]],
    *,
    field_name: str,
) -> tuple[tuple[float, float, float, float], ...]:
    if len(value) != 4:
        raise BaselinePredictionValidationError(f"{field_name} must be a 4x4 matrix.")
    rows: list[tuple[float, float, float, float]] = []
    for row in value:
        if len(row) != 4:
            raise BaselinePredictionValidationError(f"{field_name} must be a 4x4 matrix.")
        normalized_row = cast(
            tuple[float, float, float, float],
            tuple(float(item) for item in row),
        )
        if any(not math.isfinite(item) for item in normalized_row):
            raise BaselinePredictionValidationError(f"{field_name} must be finite.")
        rows.append(normalized_row)
    return cast(tuple[tuple[float, float, float, float], ...], tuple(rows))


def _parse_json_object(payload: bytes | str, *, context: str) -> dict[str, Any]:
    text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
    try:
        parsed = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_bad_json_constant,
        )
    except Exception as error:
        raise BaselinePredictionSerializationError(
            f"Failed to parse {context} JSON: {error}"
        ) from error
    if not isinstance(parsed, dict):
        raise BaselinePredictionSerializationError(f"{context} JSON must encode an object.")
    return parsed


def _require_exact_fields(
    payload: Mapping[str, Any],
    *,
    expected: set[str],
    context: str,
) -> None:
    unknown_fields = set(payload) - expected
    if unknown_fields:
        raise BaselinePredictionSerializationError(
            f"Unknown {context} field(s): {', '.join(sorted(unknown_fields))}."
        )
    missing_fields = expected - set(payload)
    if missing_fields:
        raise BaselinePredictionSerializationError(
            f"Missing {context} field(s): {', '.join(sorted(missing_fields))}."
        )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BaselinePredictionSerializationError(f"Duplicate JSON object key: {key!r}.")
        result[key] = value
    return result


def _bad_json_constant(constant: str) -> Any:
    raise BaselinePredictionSerializationError(f"Invalid JSON constant: {constant!r}.")


def _expect_string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise BaselinePredictionSerializationError(f"{field_name} must be a string.")
    return value


def _expect_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BaselinePredictionSerializationError(f"{field_name} must be an integer.")
    return value


def _expect_number(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BaselinePredictionSerializationError(f"{field_name} must be a number.")
    scalar = float(value)
    if not math.isfinite(scalar):
        raise BaselinePredictionSerializationError(f"{field_name} must be finite.")
    return scalar


def _expect_int_tuple3(value: Any, *, field_name: str) -> tuple[int, int, int]:
    if not isinstance(value, list) or len(value) != 3:
        raise BaselinePredictionSerializationError(f"{field_name} must be a length-3 array.")
    return tuple(_expect_int(item, field_name=field_name) for item in value)  # type: ignore[return-value]


def _expect_float_tuple3(value: Any, *, field_name: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise BaselinePredictionSerializationError(f"{field_name} must be a length-3 array.")
    return tuple(_expect_number(item, field_name=field_name) for item in value)  # type: ignore[return-value]


def _expect_affine(
    value: Any,
    *,
    field_name: str,
) -> tuple[tuple[float, float, float, float], ...]:
    if not isinstance(value, list) or len(value) != 4:
        raise BaselinePredictionSerializationError(f"{field_name} must be a 4x4 array.")
    rows: list[tuple[float, float, float, float]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != 4:
            raise BaselinePredictionSerializationError(f"{field_name} must be a 4x4 array.")
        rows.append(tuple(_expect_number(item, field_name=field_name) for item in row))  # type: ignore[arg-type]
    return tuple(rows)


def _require_no_absolute_paths(value: Any) -> None:
    if isinstance(value, str):
        _require_no_absolute_path_string(value, field_name="persisted string")
        return
    if isinstance(value, Mapping):
        for item in value.values():
            _require_no_absolute_paths(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _require_no_absolute_paths(item)


def _require_no_absolute_path_string(value: str, *, field_name: str) -> None:
    if value.startswith("/") or _WINDOWS_ABSOLUTE_PATH_RE.match(value):
        raise BaselinePredictionValidationError(f"{field_name} must not contain an absolute path.")
