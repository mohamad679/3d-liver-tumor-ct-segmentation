"""Phase 3 local MLflow metadata logging helpers."""

from __future__ import annotations

import re
import sqlite3
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

_ALLOWED_PARAMETER_KEYS: frozenset[str] = frozenset(
    {
        "amp_enabled",
        "baseline_family",
        "checkpoint_sha256",
        "config_sha256",
        "dataset_manifest_sha256",
        "metric_report_sha256",
        "prediction_manifest_sha256",
        "run_identifier",
        "seed",
        "selected_device",
        "status",
        "step_count",
        "strategy_identifier",
    }
)
_ALLOWED_METRIC_KEYS: frozenset[str] = frozenset({"best_loss", "final_loss", "initial_loss"})
_DATABASE_FILENAME = "tracking.db"
_EXPERIMENT_ARTIFACT_ROOT_URI = "mlflow-artifacts:/phase3"
_DEFAULT_ARTIFACT_ROOT_URI = "mlflow-artifacts:/"
_APPROVED_NONLOCAL_URI_PREFIXES: tuple[str, ...] = ("mlflow-artifacts:/",)
_POSIX_ABSOLUTE_PATH_RE = re.compile(r"(?<![:A-Za-z0-9_])/(?:[^\\s'\"`<>|]+)")
_WINDOWS_ABSOLUTE_PATH_RE = re.compile(r"[A-Za-z]:[\\\\/]")
_TEXTUAL_SQL_TYPE_MARKERS: tuple[str, ...] = ("CHAR", "CLOB", "TEXT", "VARCHAR")
_EMPTY_MODEL_TABLES: tuple[str, ...] = (
    "logged_model_metrics",
    "logged_model_params",
    "logged_model_tags",
    "logged_models",
    "model_definitions",
    "model_versions",
    "registered_models",
)

_SqlAlchemyStoreConstructor = Callable[[str, str], Any]
_RunTagConstructor = Callable[[str, str], Any]


class _ParamConstructor(Protocol):
    """Typed adapter for MLflow's keyword-only parameter construction boundary."""

    def __call__(self, *, key: str, value: str) -> Any:
        """Construct and return the installed MLflow parameter entity."""


class Phase3MlflowMetadataError(ValueError):
    """Raised when Phase 3 local MLflow metadata logging fails."""


@dataclass(frozen=True, slots=True)
class Phase3MlflowVerificationResult:
    """Immutable postcondition for one metadata-only local MLflow run."""

    database_filename: str
    database_byte_size: int
    experiment_name: str
    run_identifier: str
    baseline_family: str
    parameter_count: int
    metric_count: int
    verified: bool

    def __post_init__(self) -> None:
        if self.database_filename != _DATABASE_FILENAME:
            raise Phase3MlflowMetadataError("Unexpected Phase 3 MLflow database filename.")
        if self.database_byte_size <= 0:
            raise Phase3MlflowMetadataError("Phase 3 MLflow database must be nonempty.")
        if not self.verified:
            raise Phase3MlflowMetadataError("Phase 3 MLflow verification must pass.")


def log_phase3_metadata_only_mlflow_run(
    *,
    tracking_directory: Path,
    experiment_name: str,
    run_name: str,
    parameters: Mapping[str, Any],
    metrics: Mapping[str, float],
) -> Phase3MlflowVerificationResult:
    """Log metadata-only Phase 3 baseline information to a local MLflow SQLite store."""

    _require_tracking_directory(tracking_directory)
    _require_safe_identifier(experiment_name, field_name="experiment_name")
    _require_safe_identifier(run_name, field_name="run_name")
    _validate_parameters(parameters)
    _validate_metrics(metrics)
    normalized_parameters = {
        name: _normalize_parameter_value(parameters[name]) for name in sorted(parameters)
    }
    normalized_metrics = {name: float(metrics[name]) for name in sorted(metrics)}

    try:
        store = _create_tracking_store(tracking_directory=tracking_directory)
        experiment_id = _get_or_create_experiment_id(
            store=store,
            experiment_name=experiment_name,
        )
        run = _create_run(
            store=store,
            experiment_id=experiment_id,
            run_name=run_name,
        )
        _log_metadata(
            store=store,
            run_id=run.info.run_id,
            parameters=parameters,
            metrics=metrics,
        )
        _mark_run_finished(store=store, run_id=run.info.run_id, run_name=run_name)
        return verify_phase3_mlflow_tracking_directory(
            tracking_directory=tracking_directory,
            experiment_name=experiment_name,
            run_name=run_name,
            expected_parameters=normalized_parameters,
            expected_metrics=normalized_metrics,
        )
    except Exception as exc:
        if isinstance(exc, Phase3MlflowMetadataError):
            raise
        raise Phase3MlflowMetadataError("Phase 3 local MLflow metadata logging failed.") from exc


def audit_phase3_mlflow_tracking_directory(*, tracking_directory: Path) -> None:
    """Require persisted Phase 3 MLflow metadata to remain path-free and metadata-only."""

    _require_tracking_directory(tracking_directory)
    database_path = tracking_directory / _DATABASE_FILENAME
    if database_path.is_symlink() or not database_path.is_file():
        raise Phase3MlflowMetadataError("Phase 3 MLflow tracking database is missing.")
    if database_path.stat().st_size <= 0:
        raise Phase3MlflowMetadataError("Phase 3 MLflow tracking database is empty.")
    unexpected_files = [path for path in tracking_directory.rglob("*") if path.is_file()]
    if any(path != database_path for path in unexpected_files):
        raise Phase3MlflowMetadataError("Phase 3 MLflow tracking must remain metadata-only.")
    with sqlite3.connect(database_path) as connection:
        cursor = connection.cursor()
        table_names = [
            row[0]
            for row in cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
        ]
        for table_name in table_names:
            columns = cursor.execute(f"PRAGMA table_info({table_name})").fetchall()
            text_columns = [
                column[1]
                for column in columns
                if isinstance(column[2], str)
                and any(marker in column[2].upper() for marker in _TEXTUAL_SQL_TYPE_MARKERS)
            ]
            for column_name in text_columns:
                values = cursor.execute(f"SELECT {column_name} FROM {table_name}").fetchall()
                for row in values:
                    value = row[0]
                    if value is None:
                        continue
                    if not isinstance(value, str):
                        continue
                    _reject_persisted_path_text(value)
                    if "MLFLOW_ALLOW_FILE_STORE" in value:
                        raise Phase3MlflowMetadataError(
                            "Phase 3 MLflow metadata must not persist control environment names."
                        )
        for table_name in _EMPTY_MODEL_TABLES:
            if table_name in table_names:
                row_count = cursor.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
                if row_count != 0:
                    raise Phase3MlflowMetadataError(
                        "Phase 3 MLflow tracking must not contain model metadata."
                    )


def verify_phase3_mlflow_tracking_directory(
    *,
    tracking_directory: Path,
    experiment_name: str,
    run_name: str,
    expected_parameters: Mapping[str, str],
    expected_metrics: Mapping[str, float],
) -> Phase3MlflowVerificationResult:
    """Verify one finished allowlisted run in a local Phase 3 SQLite database."""

    audit_phase3_mlflow_tracking_directory(tracking_directory=tracking_directory)
    database_path = tracking_directory / _DATABASE_FILENAME
    with sqlite3.connect(database_path) as connection:
        experiment_rows = connection.execute(
            "SELECT experiment_id FROM experiments WHERE name = ? AND lifecycle_stage = 'active'",
            (experiment_name,),
        ).fetchall()
        if len(experiment_rows) != 1:
            raise Phase3MlflowMetadataError("Expected Phase 3 MLflow experiment is invalid.")
        run_rows = connection.execute(
            "SELECT run_uuid, status FROM runs WHERE experiment_id = ? AND name = ?",
            (experiment_rows[0][0], run_name),
        ).fetchall()
        if len(run_rows) != 1 or run_rows[0][1] != "FINISHED":
            raise Phase3MlflowMetadataError("Expected Phase 3 MLflow run is invalid.")
        run_id = run_rows[0][0]
        persisted_parameters = dict(
            connection.execute(
                "SELECT key, value FROM params WHERE run_uuid = ? ORDER BY key",
                (run_id,),
            ).fetchall()
        )
        if persisted_parameters != dict(expected_parameters):
            raise Phase3MlflowMetadataError("Phase 3 MLflow parameters do not match.")
        persisted_metrics = dict(
            connection.execute(
                "SELECT key, value FROM metrics WHERE run_uuid = ? ORDER BY key",
                (run_id,),
            ).fetchall()
        )
        if persisted_metrics != dict(expected_metrics):
            raise Phase3MlflowMetadataError("Phase 3 MLflow metrics do not match.")
        persisted_tags = dict(
            connection.execute(
                "SELECT key, value FROM tags WHERE run_uuid = ? ORDER BY key",
                (run_id,),
            ).fetchall()
        )
        if persisted_tags != {"mlflow.runName": run_name}:
            raise Phase3MlflowMetadataError("Phase 3 MLflow tags do not match.")
        unrelated_parameter_count = connection.execute(
            "SELECT COUNT(*) FROM params WHERE run_uuid != ?",
            (run_id,),
        ).fetchone()[0]
        unrelated_tag_count = connection.execute(
            "SELECT COUNT(*) FROM tags WHERE run_uuid != ?",
            (run_id,),
        ).fetchone()[0]
        if unrelated_parameter_count != 0 or unrelated_tag_count != 0:
            raise Phase3MlflowMetadataError("Unexpected Phase 3 MLflow metadata exists.")
    baseline_family = expected_parameters.get("baseline_family")
    if baseline_family is None:
        raise Phase3MlflowMetadataError("baseline_family metadata is required.")
    return Phase3MlflowVerificationResult(
        database_filename=_DATABASE_FILENAME,
        database_byte_size=database_path.stat().st_size,
        experiment_name=experiment_name,
        run_identifier=run_name,
        baseline_family=baseline_family,
        parameter_count=len(expected_parameters),
        metric_count=len(expected_metrics),
        verified=True,
    )


def _create_tracking_store(*, tracking_directory: Path) -> Any:
    from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore

    database_path = tracking_directory / _DATABASE_FILENAME
    constructor = cast(_SqlAlchemyStoreConstructor, SqlAlchemyStore)
    return constructor(f"sqlite:///{database_path}", _DEFAULT_ARTIFACT_ROOT_URI)


def _get_or_create_experiment_id(*, store: Any, experiment_name: str) -> str:
    existing = store.get_experiment_by_name(experiment_name)
    if existing is not None:
        return str(existing.experiment_id)
    artifact_location = f"{_EXPERIMENT_ARTIFACT_ROOT_URI}/{experiment_name}"
    return str(store.create_experiment(experiment_name, artifact_location=artifact_location))


def _create_run(*, store: Any, experiment_id: str, run_name: str) -> Any:
    from mlflow.entities import RunTag

    run_tag_constructor = cast(_RunTagConstructor, RunTag)
    return store.create_run(
        experiment_id,
        user_id="phase3_local",
        start_time=int(time.time() * 1000),
        tags=[run_tag_constructor("mlflow.runName", run_name)],
        run_name=run_name,
    )


def _log_metadata(
    *,
    store: Any,
    run_id: str,
    parameters: Mapping[str, Any],
    metrics: Mapping[str, float],
) -> None:
    from mlflow.entities import Metric, Param

    metric_timestamp = int(time.time() * 1000)
    param_constructor = cast(_ParamConstructor, Param)
    store.log_batch(
        run_id,
        metrics=[
            Metric(key=name, value=float(metrics[name]), timestamp=metric_timestamp, step=0)
            for name in sorted(metrics)
        ],
        params=[
            param_constructor(key=name, value=_normalize_parameter_value(parameters[name]))
            for name in sorted(parameters)
        ],
        tags=[],
    )


def _mark_run_finished(*, store: Any, run_id: str, run_name: str) -> None:
    from mlflow.entities import RunStatus

    store.update_run_info(
        run_id,
        RunStatus.FINISHED,
        int(time.time() * 1000),
        run_name,
    )


def _validate_parameters(parameters: Mapping[str, Any]) -> None:
    unexpected = set(parameters) - _ALLOWED_PARAMETER_KEYS
    if unexpected:
        names = ", ".join(sorted(unexpected))
        raise Phase3MlflowMetadataError(f"Unsupported Phase 3 MLflow parameter(s): {names}.")
    for key, value in parameters.items():
        normalized = _normalize_parameter_value(value)
        _reject_persisted_path_text(normalized)
        if key == "amp_enabled" and normalized != "False":
            raise Phase3MlflowMetadataError("amp_enabled must remain false.")
        if key == "selected_device" and normalized != "cpu":
            raise Phase3MlflowMetadataError("selected_device must remain cpu.")


def _validate_metrics(metrics: Mapping[str, float]) -> None:
    unexpected = set(metrics) - _ALLOWED_METRIC_KEYS
    if unexpected:
        names = ", ".join(sorted(unexpected))
        raise Phase3MlflowMetadataError(f"Unsupported Phase 3 MLflow metric(s): {names}.")
    for value in metrics.values():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise Phase3MlflowMetadataError(
                "Phase 3 MLflow metrics must be finite numeric scalars."
            )
        numeric = float(value)
        if numeric != numeric or numeric in (float("inf"), float("-inf")):
            raise Phase3MlflowMetadataError(
                "Phase 3 MLflow metrics must be finite numeric scalars."
            )


def _normalize_parameter_value(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, str):
        return value
    raise Phase3MlflowMetadataError("Phase 3 MLflow parameter values must be bool, int, or str.")


def _require_tracking_directory(tracking_directory: Path) -> None:
    if not tracking_directory.is_absolute() or not tracking_directory.is_dir():
        raise Phase3MlflowMetadataError(
            "tracking_directory must be an absolute existing directory."
        )


def _require_safe_identifier(value: str, *, field_name: str) -> None:
    if not value:
        raise Phase3MlflowMetadataError(f"{field_name} must not be empty.")
    if any(character.isspace() for character in value):
        raise Phase3MlflowMetadataError(f"{field_name} must not contain whitespace.")
    _reject_persisted_path_text(value)


def _reject_persisted_path_text(value: str) -> None:
    if any(value.startswith(prefix) for prefix in _APPROVED_NONLOCAL_URI_PREFIXES):
        return
    if _POSIX_ABSOLUTE_PATH_RE.search(value) is not None:
        raise Phase3MlflowMetadataError("Phase 3 MLflow metadata must not contain absolute paths.")
    if _WINDOWS_ABSOLUTE_PATH_RE.search(value) is not None:
        raise Phase3MlflowMetadataError("Phase 3 MLflow metadata must not contain absolute paths.")
    if "file://" in value or "sqlite:///" in value:
        raise Phase3MlflowMetadataError("Phase 3 MLflow metadata must not contain path URIs.")
