"""Baseline-neutral saved-prediction import and evaluation for Phase 3."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

import nibabel as nib
import numpy as np

from protoem_ct.artifacts.hashing import sha256_file
from protoem_ct.baselines.metrics import (
    BaselineMetricReport,
    baseline_metric_report_to_json,
    build_baseline_metric_report,
    compute_baseline_case_metrics,
)
from protoem_ct.baselines.predictions import (
    BaselinePredictionManifest,
    BaselinePredictionRecord,
    baseline_prediction_manifest_to_json,
    build_baseline_prediction_manifest,
)
from protoem_ct.baselines.synthetic import baseline_synthetic_fixture_from_json

_SAFE_IDENTIFIER_RE = __import__("re").compile(r"^[a-z0-9](?:[a-z0-9_-]{0,62}[a-z0-9])?$")
_SHA256_RE = __import__("re").compile(r"^[0-9a-f]{64}$")
_WINDOWS_ABSOLUTE_PATH_RE = __import__("re").compile(r"^[A-Za-z]:[\\/]")
_ALLOWED_BASELINE_FAMILIES = {"nnunet_v2", "monai_segresnet"}
_REFERENCE_MAPPING_ALLOWED_TOP_LEVEL_KEYS = {"label_paths_by_case"}


class BaselineEvaluationError(ValueError):
    """Base error for Phase 3 saved-prediction import and evaluation."""


class BaselineEvaluationValidationError(BaselineEvaluationError):
    """Raised when saved predictions or references violate the import contract."""


class BaselineEvaluationSerializationError(BaselineEvaluationError):
    """Raised when an external mapping artifact cannot be parsed safely."""


class BaselineEvaluationPublishError(BaselineEvaluationError):
    """Raised when deterministic evaluation artifacts cannot be published safely."""


@dataclass(frozen=True, slots=True)
class BaselineEvaluationFailure(BaselineEvaluationError):
    """Typed import failure carrying a machine-readable failure code."""

    failure_code: str

    def __str__(self) -> str:
        return self.failure_code


@dataclass(frozen=True, slots=True)
class BaselinePredictionImportResult:
    """Deterministic outputs from importing and evaluating saved predictions."""

    prediction_manifest: BaselinePredictionManifest
    metric_report: BaselineMetricReport


def import_saved_baseline_predictions(
    *,
    baseline_family: str,
    run_identifier: str,
    prediction_directory: Path,
    label_paths_by_case: Mapping[str, Path],
    dataset_manifest_sha256: str,
    development_split_sha256: str,
    checkpoint_sha256: str,
    metric_config_sha256: str,
    nsd_tolerance_mm: float,
    affine_tolerance_mm: float,
    prediction_manifest_output_path: Path,
    metric_report_output_path: Path,
) -> BaselinePredictionImportResult:
    """Import saved binary predictions, evaluate them, and publish deterministic artifacts."""

    _require_baseline_family(baseline_family)
    _require_identifier(run_identifier, field_name="run_identifier")
    _require_sha256(dataset_manifest_sha256, field_name="dataset_manifest_sha256")
    _require_sha256(development_split_sha256, field_name="development_split_sha256")
    _require_sha256(checkpoint_sha256, field_name="checkpoint_sha256")
    _require_sha256(metric_config_sha256, field_name="metric_config_sha256")
    if not math.isfinite(nsd_tolerance_mm) or nsd_tolerance_mm < 0.0:
        raise BaselineEvaluationValidationError(
            "nsd_tolerance_mm must be finite and greater than or equal to zero."
        )
    if not math.isfinite(affine_tolerance_mm) or affine_tolerance_mm < 0.0:
        raise BaselineEvaluationValidationError(
            "affine_tolerance_mm must be finite and greater than or equal to zero."
        )

    prediction_root = _validate_existing_external_directory(
        prediction_directory,
        field_name="prediction_directory",
    )
    normalized_label_paths = _normalize_label_paths_by_case(label_paths_by_case)
    prediction_manifest_output = _validate_external_output_path(
        prediction_manifest_output_path,
        field_name="prediction_manifest_output_path",
    )
    metric_report_output = _validate_external_output_path(
        metric_report_output_path,
        field_name="metric_report_output_path",
    )
    if prediction_manifest_output == metric_report_output:
        raise BaselineEvaluationValidationError(
            "prediction_manifest_output_path and metric_report_output_path must differ."
        )

    discovered_predictions = _discover_prediction_files(prediction_root)
    expected_case_identifiers = tuple(sorted(normalized_label_paths))
    if set(discovered_predictions) != set(expected_case_identifiers):
        missing = sorted(set(expected_case_identifiers) - set(discovered_predictions))
        unexpected = sorted(set(discovered_predictions) - set(expected_case_identifiers))
        if missing:
            raise BaselineEvaluationFailure("missing_prediction")
        if unexpected:
            raise BaselineEvaluationFailure("unexpected_prediction")
        raise BaselineEvaluationFailure("prediction_case_mismatch")

    case_metrics = []
    prediction_records = []
    for case_identifier in expected_case_identifiers:
        prediction_path = discovered_predictions[case_identifier]
        label_path = normalized_label_paths[case_identifier]
        prediction_image = cast(Any, nib.load(str(prediction_path)))
        label_image = cast(Any, nib.load(str(label_path)))

        prediction_shape = cast(
            tuple[int, int, int],
            tuple(int(value) for value in prediction_image.shape),
        )
        label_shape = cast(
            tuple[int, int, int],
            tuple(int(value) for value in label_image.shape),
        )
        if prediction_shape != label_shape:
            raise BaselineEvaluationFailure("shape_mismatch")
        if len(prediction_shape) != 3:
            raise BaselineEvaluationFailure("invalid_prediction_shape")

        prediction_affine = np.asarray(prediction_image.affine, dtype=np.float64)
        label_affine = np.asarray(label_image.affine, dtype=np.float64)
        if not np.all(np.isfinite(prediction_affine)) or not np.all(np.isfinite(label_affine)):
            raise BaselineEvaluationFailure("nonfinite_affine")
        if not np.allclose(
            prediction_affine,
            label_affine,
            rtol=0.0,
            atol=affine_tolerance_mm,
        ):
            raise BaselineEvaluationFailure("affine_mismatch")

        prediction_spacing = cast(
            tuple[float, float, float],
            tuple(float(value) for value in prediction_image.header.get_zooms()[:3]),
        )
        label_spacing = cast(
            tuple[float, float, float],
            tuple(float(value) for value in label_image.header.get_zooms()[:3]),
        )
        _require_spacing(prediction_spacing, field_name="prediction_spacing")
        _require_spacing(label_spacing, field_name="label_spacing")
        if not np.allclose(
            np.asarray(prediction_spacing, dtype=np.float64),
            np.asarray(label_spacing, dtype=np.float64),
            rtol=0.0,
            atol=affine_tolerance_mm,
        ):
            raise BaselineEvaluationFailure("spacing_mismatch")

        prediction_data = np.asanyarray(prediction_image.dataobj)
        label_data = np.asanyarray(label_image.dataobj)
        if not np.all(np.isfinite(prediction_data)):
            raise BaselineEvaluationFailure("nonfinite_prediction_values")
        if not np.all(np.isfinite(label_data)):
            raise BaselineEvaluationFailure("nonfinite_reference_values")
        _require_binary_array(prediction_data, field_name="prediction_values")
        _require_binary_array(label_data, field_name="reference_values")

        case_metrics.append(
            compute_baseline_case_metrics(
                case_identifier=case_identifier,
                ground_truth_mask=label_data,
                prediction_mask=prediction_data,
                voxel_spacing_mm=prediction_spacing,
                nsd_tolerance_mm=nsd_tolerance_mm,
            )
        )
        prediction_records.append(
            BaselinePredictionRecord(
                contract_version="baseline_prediction_record_v1",
                case_identifier=case_identifier,
                prediction_path=PurePosixPath(prediction_path.name).as_posix(),
                prediction_sha256=sha256_file(prediction_path),
                byte_size=prediction_path.stat().st_size,
                shape=prediction_shape,
                voxel_spacing_mm=prediction_spacing,
                affine=_affine_to_tuple(prediction_affine),
                foreground_label=1,
            )
        )

    prediction_manifest = build_baseline_prediction_manifest(
        baseline_family=baseline_family,
        run_identifier=run_identifier,
        dataset_manifest_sha256=dataset_manifest_sha256,
        development_split_sha256=development_split_sha256,
        checkpoint_sha256=checkpoint_sha256,
        prediction_records=tuple(prediction_records),
    )
    metric_report = build_baseline_metric_report(
        baseline_family=baseline_family,
        run_identifier=run_identifier,
        dataset_manifest_sha256=dataset_manifest_sha256,
        development_split_sha256=development_split_sha256,
        prediction_manifest_sha256=prediction_manifest.artifact_hash,
        metric_config_sha256=metric_config_sha256,
        nsd_tolerance_mm=nsd_tolerance_mm,
        case_records=tuple(case_metrics),
    )
    _publish_artifacts(
        prediction_manifest_output_path=prediction_manifest_output,
        metric_report_output_path=metric_report_output,
        prediction_manifest=prediction_manifest,
        metric_report=metric_report,
    )
    return BaselinePredictionImportResult(
        prediction_manifest=prediction_manifest,
        metric_report=metric_report,
    )


def load_reference_label_mapping_artifact(mapping_artifact_path: Path) -> dict[str, Path]:
    """Parse a deterministic external reference-label mapping artifact."""

    path = _validate_existing_external_file(
        mapping_artifact_path,
        field_name="mapping_artifact_path",
    )
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_bad_json_constant,
        )
    except Exception as error:
        raise BaselineEvaluationSerializationError(
            f"Failed to parse reference mapping artifact JSON: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise BaselineEvaluationSerializationError(
            "Reference mapping artifact JSON must encode an object."
        )
    unknown_keys = set(payload) - _REFERENCE_MAPPING_ALLOWED_TOP_LEVEL_KEYS
    if unknown_keys:
        raise BaselineEvaluationSerializationError(
            f"Unknown reference mapping artifact field(s): {', '.join(sorted(unknown_keys))}."
        )
    if set(payload) != _REFERENCE_MAPPING_ALLOWED_TOP_LEVEL_KEYS:
        raise BaselineEvaluationSerializationError(
            "Reference mapping artifact must contain exactly label_paths_by_case."
        )
    mapping_value = payload["label_paths_by_case"]
    if not isinstance(mapping_value, dict):
        raise BaselineEvaluationSerializationError("label_paths_by_case must be a JSON object.")
    return {
        case_identifier: _validate_existing_external_file(Path(label_path), field_name="label_path")
        for case_identifier, label_path in _normalize_mapping_object(mapping_value).items()
    }


def build_reference_label_mapping_from_synthetic_fixture(
    fixture_artifact_path: Path,
    *,
    fixture_root: Path,
    split: str = "test",
) -> dict[str, Path]:
    """Build a case-to-label-path mapping from a saved synthetic fixture artifact."""

    artifact = baseline_synthetic_fixture_from_json(fixture_artifact_path.read_bytes())
    root = _validate_existing_external_directory(fixture_root, field_name="fixture_root")
    allowed_roles = {"test_label"} if split == "test" else {"training_label"}
    mapping: dict[str, Path] = {}
    for file_record in artifact.file_records:
        if file_record.role not in allowed_roles:
            continue
        if file_record.case_identifier is None:
            raise BaselineEvaluationValidationError(
                "Synthetic fixture label record is missing case_identifier."
            )
        mapping[file_record.case_identifier] = root / file_record.relative_path
    return mapping


def _discover_prediction_files(prediction_root: Path) -> dict[str, Path]:
    discovered: dict[str, Path] = {}
    for prediction_path in sorted(prediction_root.rglob("*.nii.gz")):
        if not prediction_path.is_file():
            continue
        relative_path = prediction_path.relative_to(prediction_root)
        if len(relative_path.parts) != 1:
            raise BaselineEvaluationFailure("unexpected_prediction")
        case_identifier = prediction_path.name.removesuffix(".nii.gz")
        _require_identifier(case_identifier, field_name="case_identifier")
        if case_identifier in discovered:
            raise BaselineEvaluationFailure("duplicate_prediction")
        discovered[case_identifier] = prediction_path
    return discovered


def _normalize_label_paths_by_case(label_paths_by_case: Mapping[str, Path]) -> dict[str, Path]:
    normalized: dict[str, Path] = {}
    for case_identifier, label_path in label_paths_by_case.items():
        _require_identifier(case_identifier, field_name="case_identifier")
        if case_identifier in normalized:
            raise BaselineEvaluationFailure("duplicate_reference_case")
        normalized[case_identifier] = _validate_existing_external_file(
            Path(label_path),
            field_name="label_path",
        )
    if not normalized:
        raise BaselineEvaluationValidationError("label_paths_by_case must not be empty.")
    return dict(sorted(normalized.items()))


def _publish_artifacts(
    *,
    prediction_manifest_output_path: Path,
    metric_report_output_path: Path,
    prediction_manifest: BaselinePredictionManifest,
    metric_report: BaselineMetricReport,
) -> None:
    created_paths: list[Path] = []
    temporary_paths: list[Path] = []
    try:
        manifest_temp = _write_temp_bytes(
            prediction_manifest_output_path,
            baseline_prediction_manifest_to_json(prediction_manifest),
        )
        temporary_paths.append(manifest_temp)
        report_temp = _write_temp_bytes(
            metric_report_output_path,
            baseline_metric_report_to_json(metric_report),
        )
        temporary_paths.append(report_temp)
        os.replace(manifest_temp, prediction_manifest_output_path)
        created_paths.append(prediction_manifest_output_path)
        temporary_paths.remove(manifest_temp)
        os.replace(report_temp, metric_report_output_path)
        created_paths.append(metric_report_output_path)
        temporary_paths.remove(report_temp)
    except Exception as error:
        for temporary_path in temporary_paths:
            if temporary_path.exists():
                temporary_path.unlink()
        for created_path in reversed(created_paths):
            if created_path.exists():
                created_path.unlink()
        if isinstance(error, BaselineEvaluationError):
            raise
        raise BaselineEvaluationPublishError(
            f"Failed to publish baseline evaluation artifacts: {type(error).__name__}."
        ) from error


def _write_temp_bytes(output_path: Path, payload: bytes) -> Path:
    if output_path.exists():
        raise BaselineEvaluationFailure("artifact_output_exists")
    temporary_directory = output_path.parent
    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{output_path.name}.tmp-",
        suffix=".json",
        dir=temporary_directory,
        delete=False,
    ) as temporary_handle:
        temporary_handle.write(payload)
        temporary_handle.flush()
        return Path(temporary_handle.name)


def _validate_existing_external_directory(path: Path, *, field_name: str) -> Path:
    candidate = _validate_external_runtime_path(path, field_name=field_name)
    if not candidate.exists():
        raise BaselineEvaluationValidationError(f"{field_name} must exist.")
    if not candidate.is_dir():
        raise BaselineEvaluationValidationError(f"{field_name} must be a directory.")
    return candidate


def _validate_existing_external_file(path: Path, *, field_name: str) -> Path:
    candidate = _validate_external_runtime_path(path, field_name=field_name)
    if not candidate.exists():
        raise BaselineEvaluationValidationError(f"{field_name} must exist.")
    if not candidate.is_file():
        raise BaselineEvaluationValidationError(f"{field_name} must be a regular file.")
    return candidate


def _validate_external_output_path(path: Path, *, field_name: str) -> Path:
    candidate = _validate_external_runtime_path(path, field_name=field_name)
    if candidate.exists():
        raise BaselineEvaluationValidationError(f"{field_name} must not already exist.")
    if not candidate.parent.exists():
        raise BaselineEvaluationValidationError(
            f"{field_name} parent directory must already exist."
        )
    if not candidate.parent.is_dir():
        raise BaselineEvaluationValidationError(f"{field_name} parent must be a directory.")
    return candidate


def _validate_external_runtime_path(path: Path, *, field_name: str) -> Path:
    repository_root = Path(__file__).resolve().parents[3]
    candidate = Path(path)
    if not candidate.is_absolute():
        raise BaselineEvaluationValidationError(f"{field_name} must be an absolute path.")
    normalized = os.path.normpath(os.fspath(candidate))
    if ".." in Path(normalized).parts:
        raise BaselineEvaluationValidationError(
            f"{field_name} must not contain parent-directory traversal."
        )
    for component in candidate.parts[1:]:
        if component == "":
            raise BaselineEvaluationValidationError(
                f"{field_name} must not contain empty path components."
            )
        if any(ord(character) < 32 or ord(character) == 127 for character in component):
            raise BaselineEvaluationValidationError(
                f"{field_name} must not contain control characters."
            )
    _require_no_symlink_components(candidate.parent)
    resolved = candidate.resolve(strict=False)
    if resolved == repository_root or _is_relative_to(resolved, repository_root):
        raise BaselineEvaluationValidationError(
            f"{field_name} must resolve outside the repository root."
        )
    return resolved


def _require_no_symlink_components(path: Path) -> None:
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current = current / component
        if current.is_symlink():
            raise BaselineEvaluationValidationError(
                "Runtime paths must not traverse symlinked path components."
            )


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _require_binary_array(array: np.ndarray[Any, Any], *, field_name: str) -> None:
    if array.dtype == np.dtype(object):
        raise BaselineEvaluationFailure("invalid_prediction_values")
    unique_values = np.unique(array)
    allowed_values = {0, 1}
    if any(item not in allowed_values for item in unique_values.tolist()):
        if field_name == "prediction_values":
            raise BaselineEvaluationFailure("invalid_prediction_values")
        raise BaselineEvaluationFailure("invalid_reference_values")


def _affine_to_tuple(affine: np.ndarray[Any, Any]) -> tuple[tuple[float, float, float, float], ...]:
    return cast(
        tuple[tuple[float, float, float, float], ...],
        tuple(tuple(float(value) for value in row) for row in affine.tolist()),
    )


def _normalize_mapping_object(value: dict[str, Any]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str):
            raise BaselineEvaluationSerializationError(
                "label_paths_by_case must map strings to strings."
            )
        _require_identifier(key, field_name="case_identifier")
        normalized[key] = item
    return normalized


def _require_baseline_family(value: str) -> None:
    if value not in _ALLOWED_BASELINE_FAMILIES:
        raise BaselineEvaluationValidationError(f"Unsupported baseline family: {value!r}.")


def _require_identifier(value: str, *, field_name: str) -> None:
    if not _SAFE_IDENTIFIER_RE.fullmatch(value):
        raise BaselineEvaluationValidationError(
            f"{field_name} must match the conservative ASCII identifier grammar."
        )
    if value.startswith("/") or _WINDOWS_ABSOLUTE_PATH_RE.match(value):
        raise BaselineEvaluationValidationError(f"{field_name} must not contain an absolute path.")


def _require_sha256(value: str, *, field_name: str) -> None:
    if not _SHA256_RE.fullmatch(value):
        raise BaselineEvaluationValidationError(
            f"{field_name} must be exactly 64 lowercase hexadecimal characters."
        )


def _require_spacing(value: tuple[float, float, float], *, field_name: str) -> None:
    if len(value) != 3 or any(not math.isfinite(item) or item <= 0.0 for item in value):
        raise BaselineEvaluationValidationError(
            f"{field_name} must contain exactly three finite positive values."
        )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BaselineEvaluationSerializationError(f"Duplicate JSON object key: {key!r}.")
        result[key] = value
    return result


def _bad_json_constant(constant: str) -> Any:
    raise BaselineEvaluationSerializationError(
        f"Invalid JSON constant in reference mapping artifact: {constant!r}."
    )
