from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from protoem_ct.baselines._mlflow_local import (
    Phase3MlflowMetadataError,
    Phase3MlflowVerificationResult,
    audit_phase3_mlflow_tracking_directory,
    log_phase3_metadata_only_mlflow_run,
    verify_phase3_mlflow_tracking_directory,
)


def _logged_parameters(
    *,
    baseline_family: str = "nnunet_v2",
    run_identifier: str = "phase3_gate3_test_nnunet_v2",
) -> dict[str, object]:
    parameters: dict[str, object] = {
        "amp_enabled": False,
        "baseline_family": baseline_family,
        "checkpoint_sha256": "1" * 64,
        "config_sha256": "2" * 64,
        "dataset_manifest_sha256": "3" * 64,
        "metric_report_sha256": "4" * 64,
        "prediction_manifest_sha256": "5" * 64,
        "run_identifier": run_identifier,
        "seed": 1729,
        "selected_device": "cpu",
        "status": "completed",
        "step_count": 12,
    }
    if baseline_family == "nnunet_v2":
        parameters["strategy_identifier"] = (
            "official_plan_preprocess_plus_trainer_network_tiny_cpu_v1"
        )
    return parameters


def _logged_metrics() -> dict[str, float]:
    return {"best_loss": 0.5, "final_loss": 0.5, "initial_loss": 1.0}


def _normalized_parameters(parameters: dict[str, object]) -> dict[str, str]:
    return {
        key: ("True" if value is True else "False" if value is False else str(value))
        for key, value in parameters.items()
    }


def _collect_database_text_values(database_path: Path) -> list[str]:
    text_values: list[str] = []
    with sqlite3.connect(database_path) as connection:
        cursor = connection.cursor()
        tables = cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        for (table_name,) in tables:
            columns = cursor.execute(f"PRAGMA table_info({table_name})").fetchall()
            text_columns = [
                column[1]
                for column in columns
                if isinstance(column[2], str)
                and any(
                    marker in column[2].upper() for marker in ("CHAR", "CLOB", "TEXT", "VARCHAR")
                )
            ]
            for column_name in text_columns:
                rows = cursor.execute(f"SELECT {column_name} FROM {table_name}").fetchall()
                text_values.extend(value for (value,) in rows if isinstance(value, str))
    return text_values


def _log_run(
    tracking_directory: Path,
    *,
    baseline_family: str,
    experiment_name: str,
    run_identifier: str,
) -> Phase3MlflowVerificationResult:
    return log_phase3_metadata_only_mlflow_run(
        tracking_directory=tracking_directory,
        experiment_name=experiment_name,
        run_name=run_identifier,
        parameters=_logged_parameters(
            baseline_family=baseline_family,
            run_identifier=run_identifier,
        ),
        metrics=_logged_metrics(),
    )


def test_phase3_sqlite_run_has_exact_finished_metadata_and_no_artifacts(tmp_path: Path) -> None:
    pytest.importorskip("mlflow")
    tracking_directory = tmp_path / "mlflow"
    tracking_directory.mkdir()
    run_identifier = "phase3_gate3_test_nnunet_v2"

    result = _log_run(
        tracking_directory,
        baseline_family="nnunet_v2",
        experiment_name="phase3_nnunet_tiny",
        run_identifier=run_identifier,
    )

    assert result.verified
    assert result.database_filename == "tracking.db"
    assert result.database_byte_size > 0
    assert result.parameter_count == len(_logged_parameters())
    assert result.metric_count == len(_logged_metrics())
    database_path = tracking_directory / "tracking.db"
    assert database_path.is_file()
    assert not database_path.is_symlink()
    assert sorted(path.name for path in tracking_directory.iterdir()) == ["tracking.db"]

    with sqlite3.connect(database_path) as connection:
        assert connection.execute(
            "SELECT name FROM experiments WHERE name = 'phase3_nnunet_tiny'"
        ).fetchall() == [("phase3_nnunet_tiny",)]
        assert connection.execute(
            "SELECT name, status FROM runs WHERE name = ?",
            (run_identifier,),
        ).fetchall() == [(run_identifier, "FINISHED")]
        assert dict(connection.execute("SELECT key, value FROM params").fetchall()) == (
            _normalized_parameters(_logged_parameters())
        )
        assert dict(connection.execute("SELECT key, value FROM metrics").fetchall()) == (
            _logged_metrics()
        )
        assert dict(connection.execute("SELECT key, value FROM tags").fetchall()) == {
            "mlflow.runName": run_identifier
        }
        for table_name in (
            "logged_model_metrics",
            "logged_model_params",
            "logged_model_tags",
            "logged_models",
            "model_definitions",
            "model_versions",
            "registered_models",
        ):
            assert connection.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0] == 0

    joined = "\n".join(_collect_database_text_values(database_path))
    assert "MLFLOW_ALLOW_FILE_STORE" not in joined
    assert "file://" not in joined
    assert "sqlite:///" not in joined
    assert str(tracking_directory) not in joined
    assert str(Path.home()) not in joined
    assert str(Path(__file__).resolve().parents[2]) not in joined


def test_sequential_nnunet_and_monai_runs_create_distinct_nonempty_databases(
    tmp_path: Path,
) -> None:
    pytest.importorskip("mlflow")
    nnunet_directory = tmp_path / "nnunet_v2_run" / "mlflow"
    monai_directory = tmp_path / "monai_segresnet_run" / "mlflow"
    nnunet_directory.mkdir(parents=True)
    monai_directory.mkdir(parents=True)

    nnunet_result = _log_run(
        nnunet_directory,
        baseline_family="nnunet_v2",
        experiment_name="phase3_nnunet_tiny",
        run_identifier="phase3_gate3_test_nnunet_v2",
    )
    monai_result = _log_run(
        monai_directory,
        baseline_family="monai_segresnet",
        experiment_name="phase3_monai_segresnet",
        run_identifier="phase3_gate3_test_monai_segresnet",
    )

    nnunet_database = nnunet_directory / "tracking.db"
    monai_database = monai_directory / "tracking.db"
    assert nnunet_result.verified and monai_result.verified
    assert nnunet_database.is_file() and nnunet_database.stat().st_size > 0
    assert monai_database.is_file() and monai_database.stat().st_size > 0
    assert nnunet_database != monai_database
    with sqlite3.connect(nnunet_database) as connection:
        assert connection.execute("SELECT name FROM runs").fetchall() == [
            ("phase3_gate3_test_nnunet_v2",)
        ]
    with sqlite3.connect(monai_database) as connection:
        assert connection.execute("SELECT name FROM runs").fetchall() == [
            ("phase3_gate3_test_monai_segresnet",)
        ]


@pytest.mark.parametrize("database_state", ["missing", "empty"])
def test_phase3_mlflow_verification_rejects_missing_or_empty_database(
    tmp_path: Path,
    database_state: str,
) -> None:
    tracking_directory = tmp_path / "mlflow"
    tracking_directory.mkdir()
    if database_state == "empty":
        (tracking_directory / "tracking.db").touch()

    with pytest.raises(Phase3MlflowMetadataError):
        verify_phase3_mlflow_tracking_directory(
            tracking_directory=tracking_directory,
            experiment_name="phase3_monai_segresnet",
            run_name="phase3_gate3_test_monai_segresnet",
            expected_parameters={},
            expected_metrics={},
        )


def test_phase3_mlflow_verification_rejects_unexpected_parameter(tmp_path: Path) -> None:
    pytest.importorskip("mlflow")
    tracking_directory = tmp_path / "mlflow"
    tracking_directory.mkdir()
    run_identifier = "phase3_gate3_test_monai_segresnet"
    parameters = _logged_parameters(
        baseline_family="monai_segresnet",
        run_identifier=run_identifier,
    )
    _log_run(
        tracking_directory,
        baseline_family="monai_segresnet",
        experiment_name="phase3_monai_segresnet",
        run_identifier=run_identifier,
    )
    with sqlite3.connect(tracking_directory / "tracking.db") as connection:
        run_id = connection.execute("SELECT run_uuid FROM runs").fetchone()[0]
        connection.execute(
            "INSERT INTO params (key, value, run_uuid) VALUES (?, ?, ?)",
            ("unexpected", "value", run_id),
        )

    with pytest.raises(Phase3MlflowMetadataError):
        verify_phase3_mlflow_tracking_directory(
            tracking_directory=tracking_directory,
            experiment_name="phase3_monai_segresnet",
            run_name=run_identifier,
            expected_parameters=_normalized_parameters(parameters),
            expected_metrics=_logged_metrics(),
        )


@pytest.mark.parametrize(
    "forbidden_value",
    ["/private/runtime/path", "MLFLOW_ALLOW_FILE_STORE"],
)
def test_phase3_mlflow_audit_rejects_forbidden_sqlite_text(
    tmp_path: Path,
    forbidden_value: str,
) -> None:
    pytest.importorskip("mlflow")
    tracking_directory = tmp_path / "mlflow"
    tracking_directory.mkdir()
    _log_run(
        tracking_directory,
        baseline_family="monai_segresnet",
        experiment_name="phase3_monai_segresnet",
        run_identifier="phase3_gate3_test_monai_segresnet",
    )
    with sqlite3.connect(tracking_directory / "tracking.db") as connection:
        experiment_id = connection.execute(
            "SELECT experiment_id FROM experiments WHERE name = 'phase3_monai_segresnet'"
        ).fetchone()[0]
        connection.execute(
            "INSERT INTO experiment_tags (key, value, experiment_id) VALUES (?, ?, ?)",
            ("forbidden", forbidden_value, experiment_id),
        )

    with pytest.raises(Phase3MlflowMetadataError):
        audit_phase3_mlflow_tracking_directory(tracking_directory=tracking_directory)


def test_phase3_mlflow_rejects_path_like_or_control_parameter_values(tmp_path: Path) -> None:
    pytest.importorskip("mlflow")
    tracking_directory = tmp_path / "mlflow"
    tracking_directory.mkdir()

    for forbidden_value in (str(tmp_path / "leak"), "completed_MLFLOW_ALLOW_FILE_STORE"):
        with pytest.raises(Phase3MlflowMetadataError):
            log_phase3_metadata_only_mlflow_run(
                tracking_directory=tracking_directory,
                experiment_name="phase3_nnunet_tiny",
                run_name="phase3_gate3_test_nnunet_v2",
                parameters={
                    **_logged_parameters(),
                    "status": forbidden_value,
                },
                metrics={"initial_loss": 1.0},
            )
