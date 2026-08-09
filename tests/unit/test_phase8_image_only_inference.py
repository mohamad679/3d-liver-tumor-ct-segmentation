"""Synthetic tests for the Phase 8 Package F external image-only inference driver.

All tests operate on small, synthetic, in-memory data and temp-path fixtures only. No real
``/Volumes`` path, real checkpoint, or real 3D-IRCADb-01 file is referenced anywhere in this file.
"""

from __future__ import annotations

import importlib.util
import json
import struct
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from protoem_ct.artifacts.hashing import canonical_json_bytes, sha256_file, sha256_json
from protoem_ct.external.image_only_inference import (
    REQUIRED_CHECKPOINT_SHA256,
    REQUIRED_DEFINITIVE_CONFIG_HASH,
    REQUIRED_FREEZE_ARTIFACT_SHA256,
    REQUIRED_PREREGISTRATION_HASH,
    REQUIRED_TARGET_SPACING_XYZ_MM,
    REQUIRED_THRESHOLD,
    Phase8ExternalInferenceCaseError,
    Phase8ExternalInferenceIdentityError,
    Phase8ExternalInferenceOutputRootError,
    Phase8ExternalPredictionLock,
    Phase8ExternalPredictionRecord,
    load_and_preprocess_external_case,
    phase8_external_prediction_record_to_dict,
    run_phase8_external_image_only_inference,
    verify_phase8_frozen_identities,
)

requires_baseline_environment = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None or importlib.util.find_spec("monai") is None,
    reason="Phase 8 Package F inference tests require the isolated CPU torch/MONAI environment.",
)

_FAKE_SHA256 = "a" * 64


# ---------------------------------------------------------------------------
# Synthetic DICOM/IRCADb fixture helpers
# ---------------------------------------------------------------------------


def _write_patient_zip(path: Path, slices: list[bytes]) -> None:
    with zipfile.ZipFile(path, "w") as handle:
        for index, data in enumerate(slices):
            handle.writestr(f"slice-{index:03d}.dcm", data)


def _dicom_slice(
    index: int,
    *,
    rows: int = 32,
    columns: int = 32,
    pixel_spacing: tuple[float, float] = (0.8, 0.8),
    slice_spacing: float = 1.5,
) -> bytes:
    position = (0.0, 0.0, float(index) * slice_spacing)
    orientation = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    pixel_values = (np.full((rows, columns), 50 + index, dtype=np.uint16)).tobytes()
    elements: list[tuple[int, int, str, bytes]] = [
        (0x0002, 0x0010, "UI", _ui("1.2.840.10008.1.2.1")),
        (0x0008, 0x0018, "UI", _ui(f"1.2.826.0.1.3680043.10.543.{index}")),
        (0x0008, 0x0060, "CS", _text("CT")),
        (0x0020, 0x000E, "UI", _ui("1.2.826.0.1.3680043.10.543.100")),
        (0x0020, 0x0032, "DS", _ds(position)),
        (0x0020, 0x0037, "DS", _ds(orientation)),
        (0x0028, 0x0010, "US", struct.pack("<H", rows)),
        (0x0028, 0x0011, "US", struct.pack("<H", columns)),
        (0x0028, 0x0030, "DS", _ds(pixel_spacing)),
        (0x0028, 0x0100, "US", struct.pack("<H", 16)),
        (0x0028, 0x0103, "US", struct.pack("<H", 0)),
        (0x0028, 0x1052, "DS", _text("-1000.0")),
        (0x0028, 0x1053, "DS", _text("1.0")),
        (0x7FE0, 0x0010, "OW", pixel_values),
    ]
    body = b"".join(_element(group, elem, vr, value) for group, elem, vr, value in elements)
    return b"\x00" * 128 + b"DICM" + body


def _element(group: int, element: int, vr: str, value: bytes) -> bytes:
    padded = value + (b" " if len(value) % 2 else b"")
    if vr in {"OB", "OW"}:
        return struct.pack("<HH2sxxI", group, element, vr.encode("ascii"), len(padded)) + padded
    return struct.pack("<HH2sH", group, element, vr.encode("ascii"), len(padded)) + padded


def _text(value: str) -> bytes:
    return value.encode("ascii")


def _ui(value: str) -> bytes:
    return value.encode("ascii")


def _ds(values: tuple[float, ...]) -> bytes:
    return "\\".join(str(v) for v in values).encode("ascii")


def _build_ircadb_root(
    root: Path,
    *,
    case_count: int = 20,
    slices_per_case: int = 40,
    rows: int = 32,
    columns: int = 32,
) -> Path:
    dataset_root = root / "3Dircadb1"
    dataset_root.mkdir(parents=True)
    for ordinal in range(1, case_count + 1):
        case_dir = dataset_root / f"3Dircadb1.{ordinal}"
        case_dir.mkdir()
        _write_patient_zip(
            case_dir / "PATIENT_DICOM.zip",
            [
                _dicom_slice(index, rows=rows, columns=columns)
                for index in range(slices_per_case)
            ],
        )
        # Adversarial: prohibited label-bearing files present with garbage content. Inference
        # must never open these; garbage content must not raise any error.
        (case_dir / "MASKS_DICOM.zip").write_bytes(b"not-a-real-zip-garbage-mask-bytes")
        (case_dir / "LABELLED_DICOM.zip").write_bytes(b"not-a-real-zip-garbage-label-bytes")
    return dataset_root


def _build_frozen_artifacts(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Build a minimal, self-consistent freeze/preregistration/checkpoint fixture triple."""

    from protoem_ct.external.artifacts import (
        PHASE8_REQUIRED_DECISION_CATEGORIES,
        FrozenDecisionReference,
        Phase8DecisionFreeze,
        phase8_decision_freeze_to_dict,
    )
    from protoem_ct.external.preregistration import (
        Phase8ArtifactReference,
        Phase8ExternalPreregistration,
        Phase8RobustnessUncertaintyInclusion,
        phase8_external_preregistration_identity_payload,
        phase8_external_preregistration_to_dict,
    )

    references = tuple(
        FrozenDecisionReference(
            schema_name="phase8_frozen_decision_reference",
            schema_version="v1",
            category=category,
            source_schema_name="phase8_fake_source",
            source_schema_version="v1",
            source_phase="phase8_test",
            source_artifact_hash=_FAKE_SHA256,
            rationale_code="synthetic-test-fixture",
            provenance_reference=None,
            frozen=True,
            freeze_state="frozen",
        )
        for category in sorted(PHASE8_REQUIRED_DECISION_CATEGORIES)
    )
    freeze_payload = {
        "decision_references": [
            {
                "category": r.category,
                "freeze_state": r.freeze_state,
                "frozen": r.frozen,
                "provenance_reference": r.provenance_reference,
                "rationale_code": r.rationale_code,
                "schema_name": r.schema_name,
                "schema_version": r.schema_version,
                "source_artifact_hash": r.source_artifact_hash,
                "source_phase": r.source_phase,
                "source_schema_name": r.source_schema_name,
                "source_schema_version": r.source_schema_version,
            }
            for r in sorted(references, key=lambda item: item.category)
        ],
        "freeze_state": "frozen",
        "frozen_before_external_evaluation": True,
        "schema_name": "phase8_decision_freeze",
        "schema_version": "v1",
    }
    freeze = Phase8DecisionFreeze(
        schema_name="phase8_decision_freeze",
        schema_version="v1",
        freeze_inventory_hash=sha256_json(freeze_payload),
        freeze_state="frozen",
        frozen_before_external_evaluation=True,
        decision_references=references,
    )
    freeze_path = tmp_path / "phase8_decision_freeze.json"
    freeze_path.write_bytes(canonical_json_bytes(phase8_decision_freeze_to_dict(freeze)) + b"\n")

    def _reference(resolved: bool) -> Phase8ArtifactReference:
        return Phase8ArtifactReference(
            schema_name="phase8_artifact_reference",
            schema_version="v1",
            referenced_schema_name="phase8_fake_referenced",
            referenced_schema_version="v1",
            artifact_hash=_FAKE_SHA256 if resolved else None,
            reference_state="resolved" if resolved else "unresolved",
        )

    prereg_kwargs = dict(
        schema_name="phase8_external_preregistration",
        schema_version="v1",
        lifecycle_state="draft",
        external_cohort_identifier="3d_ircadb_01",
        frozen_decision_inventory_hash=freeze.freeze_inventory_hash,
        anonymous_manifest_reference=_reference(False),
        eligibility_exclusion_policy_reference=_reference(True),
        label_mapping_policy_reference=_reference(True),
        domain_shift_record_reference=_reference(False),
        preregistered_segmentation_metrics=("tumor_dice",),
        bootstrap_policy_config_reference=_reference(True),
        internal_external_comparison_policy_reference=_reference(True),
        qualitative_output_policy_reference=_reference(True),
        robustness_uncertainty_inclusion=Phase8RobustnessUncertaintyInclusion(
            schema_name="phase8_robustness_uncertainty_inclusion",
            schema_version="v1",
            inclusion_state="not_included",
            policy_reference=None,
        ),
        permitted_deviation_policy="no_deviations_without_versioned_addendum",
        no_tuning_declaration=True,
        external_label_access_state="unavailable_before_prediction_lock",
        prediction_lock_requirement="required_before_label_access",
    )
    prereg_hash = sha256_json(
        phase8_external_preregistration_identity_payload(
            SimpleNamespace(preregistration_hash="0" * 64, **prereg_kwargs)  # type: ignore[arg-type]
        )
    )
    prereg = Phase8ExternalPreregistration(
        preregistration_hash=prereg_hash,
        **prereg_kwargs,  # type: ignore[arg-type]
    )
    prereg_path = tmp_path / "phase8_external_preregistration.json"
    prereg_path.write_bytes(
        canonical_json_bytes(phase8_external_preregistration_to_dict(prereg)) + b"\n"
    )

    checkpoint_path = tmp_path / "phase8_definitive_checkpoint_step_500.pt"
    checkpoint_path.write_bytes(b"not-a-real-checkpoint")

    return freeze_path, prereg_path, checkpoint_path


# ---------------------------------------------------------------------------
# Frozen-identity verification
# ---------------------------------------------------------------------------


def test_verify_frozen_identities_rejects_freeze_mismatch(tmp_path: Path) -> None:
    freeze_path, prereg_path, checkpoint_path = _build_frozen_artifacts(tmp_path)
    freeze_path.write_bytes(freeze_path.read_bytes() + b" ")

    with pytest.raises(Phase8ExternalInferenceIdentityError):
        verify_phase8_frozen_identities(
            freeze_path=freeze_path,
            preregistration_path=prereg_path,
            checkpoint_path=checkpoint_path,
        )


def test_verify_frozen_identities_rejects_preregistration_mismatch(tmp_path: Path) -> None:
    freeze_path, prereg_path, checkpoint_path = _build_frozen_artifacts(tmp_path)
    tampered = json.loads(prereg_path.read_text())
    tampered["external_cohort_identifier"] = "some_other_cohort"
    prereg_path.write_text(json.dumps(tampered))

    with pytest.raises((Phase8ExternalInferenceIdentityError, ValueError)):
        verify_phase8_frozen_identities(
            freeze_path=freeze_path,
            preregistration_path=prereg_path,
            checkpoint_path=checkpoint_path,
        )


def test_verify_frozen_identities_rejects_checkpoint_mismatch(tmp_path: Path) -> None:
    freeze_path, prereg_path, checkpoint_path = _build_frozen_artifacts(tmp_path)
    checkpoint_path.write_bytes(b"different-bytes")

    with pytest.raises(Phase8ExternalInferenceIdentityError):
        verify_phase8_frozen_identities(
            freeze_path=freeze_path,
            preregistration_path=prereg_path,
            checkpoint_path=checkpoint_path,
        )


def test_required_frozen_constants_are_real_sha256_hex() -> None:
    for value in (
        REQUIRED_FREEZE_ARTIFACT_SHA256,
        REQUIRED_PREREGISTRATION_HASH,
        REQUIRED_CHECKPOINT_SHA256,
        REQUIRED_DEFINITIVE_CONFIG_HASH,
    ):
        assert len(value) == 64
        assert all(ch in "0123456789abcdef" for ch in value)
    assert REQUIRED_THRESHOLD == 0.5
    assert REQUIRED_TARGET_SPACING_XYZ_MM == (0.767578125, 0.767578125, 1.0)


# ---------------------------------------------------------------------------
# Output-root overwrite protection
# ---------------------------------------------------------------------------


def test_run_inference_rejects_existing_output_root(tmp_path: Path) -> None:
    dataset_root = _build_ircadb_root(tmp_path / "dataset")
    freeze_path, prereg_path, checkpoint_path = _build_frozen_artifacts(tmp_path)
    output_root = tmp_path / "output"
    output_root.mkdir()

    with pytest.raises(Phase8ExternalInferenceOutputRootError):
        run_phase8_external_image_only_inference(
            dataset_root=dataset_root,
            freeze_path=freeze_path,
            preregistration_path=prereg_path,
            checkpoint_path=checkpoint_path,
            output_root=output_root,
            repository_root=tmp_path,
            git_commit="a" * 40,
        )


def test_run_inference_rejects_identity_mismatch_before_touching_output_root(
    tmp_path: Path,
) -> None:
    dataset_root = _build_ircadb_root(tmp_path / "dataset")
    freeze_path, prereg_path, checkpoint_path = _build_frozen_artifacts(tmp_path)
    checkpoint_path.write_bytes(b"wrong-bytes")
    output_root = tmp_path / "output"

    with pytest.raises(Phase8ExternalInferenceIdentityError):
        run_phase8_external_image_only_inference(
            dataset_root=dataset_root,
            freeze_path=freeze_path,
            preregistration_path=prereg_path,
            checkpoint_path=checkpoint_path,
            output_root=output_root,
            repository_root=tmp_path,
            git_commit="a" * 40,
        )
    assert not output_root.exists()


# ---------------------------------------------------------------------------
# Image-only loading never touches label files, and uses frozen preprocessing
# ---------------------------------------------------------------------------


@requires_baseline_environment
def test_load_and_preprocess_never_reads_prohibited_label_files(tmp_path: Path) -> None:
    dataset_root = _build_ircadb_root(tmp_path / "dataset", case_count=1)
    from protoem_ct.external import definitive_pipeline as pipeline

    config = pipeline.build_definitive_config_v1()
    patient_archive = dataset_root / "3Dircadb1.1" / "PATIENT_DICOM.zip"

    # The case directory contains garbage-bytes MASKS_DICOM.zip / LABELLED_DICOM.zip. If the
    # loader ever attempted to open them, zipfile would raise BadZipFile immediately -- so
    # succeeding here is itself proof the label archives were never opened.
    preprocessed = load_and_preprocess_external_case(
        patient_dicom_zip=patient_archive,
        anonymous_case_id="ext-ircadb-001",
        source_case_ordinal=1,
        config=config,
    )
    assert preprocessed.anonymous_case_id == "ext-ircadb-001"
    # Frozen preprocessing was applied: RAS-oriented, resampled to the frozen target spacing.
    resampled_spacing = tuple(
        float(v)
        for v in np.sqrt(np.sum(preprocessed.resampled_affine_mm[:3, :3] ** 2, axis=0))
    )
    for observed, expected in zip(resampled_spacing, REQUIRED_TARGET_SPACING_XYZ_MM, strict=True):
        assert observed == pytest.approx(expected, abs=1e-6)


@requires_baseline_environment
def test_full_pipeline_is_deterministic_for_identical_synthetic_input(tmp_path: Path) -> None:
    import torch  # type: ignore[import-not-found]

    from protoem_ct.external import definitive_pipeline as pipeline
    from protoem_ct.external.image_only_inference import run_frozen_image_only_inference

    torch.manual_seed(1729)
    config = pipeline.build_definitive_config_v1()
    dataset_root = _build_ircadb_root(tmp_path / "dataset", case_count=1, slices_per_case=24)
    patient_archive = dataset_root / "3Dircadb1.1" / "PATIENT_DICOM.zip"
    preprocessed = load_and_preprocess_external_case(
        patient_dicom_zip=patient_archive,
        anonymous_case_id="ext-ircadb-001",
        source_case_ordinal=1,
        config=config,
    )

    monai_module = pytest.importorskip("monai")
    model = monai_module.networks.nets.SegResNet(
        spatial_dims=config.spatial_dims,
        init_filters=config.init_filters,
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        dropout_prob=config.dropout_prob,
        blocks_down=config.blocks_down,
        blocks_up=config.blocks_up,
        upsample_mode=config.upsample_mode,
    ).to(torch.device("cpu"))
    model.eval()

    first = run_frozen_image_only_inference(model, preprocessed, config=config)
    second = run_frozen_image_only_inference(model, preprocessed, config=config)

    assert first.prediction_finite
    assert second.prediction_finite
    assert np.array_equal(first.prediction_mask, second.prediction_mask)


def test_no_metric_module_is_imported_by_image_only_inference() -> None:
    import protoem_ct.external.image_only_inference as module

    source = Path(module.__file__).read_text()
    assert "evaluation.metrics" not in source
    assert "compute_binary_confusion_counts" not in source
    assert "dice_from_counts" not in source


# ---------------------------------------------------------------------------
# Prediction record / lock contracts
# ---------------------------------------------------------------------------


def _fake_record(**overrides: object) -> Phase8ExternalPredictionRecord:
    base = dict(
        schema_name="phase8_external_prediction_record",
        schema_version="v1",
        anonymous_case_id="ext-ircadb-001",
        source_case_ordinal=1,
        prediction_file_relative_path="predictions/ext-ircadb-001.npy",
        prediction_sha256=_FAKE_SHA256,
        prediction_shape_zyx=(4, 4, 4),
        prediction_dtype="uint8",
        binary_semantics="1_if_tumor_probability_ge_threshold_else_0",
        threshold=0.5,
        source_image_archive_sha256=_FAKE_SHA256,
        original_volume_shape_zyx=(4, 4, 4),
        original_voxel_spacing_row_col_slice_mm=(0.8, 0.8, 1.5),
        ras_reoriented_shape_zyx=(4, 4, 4),
        ras_reoriented_affine_mm=tuple(tuple(row) for row in np.eye(4).tolist()),
        resampled_shape_zyx=(4, 4, 4),
        resampled_affine_mm=tuple(tuple(row) for row in np.eye(4).tolist()),
        target_spacing_xyz_mm=(0.767578125, 0.767578125, 1.0),
        preprocessing_orientation="ras",
        preprocessing_image_interpolation="trilinear",
        checkpoint_sha256=_FAKE_SHA256,
        definitive_config_hash=_FAKE_SHA256,
        freeze_artifact_sha256=_FAKE_SHA256,
        preregistration_hash=_FAKE_SHA256,
        support_policy="no_support",
        external_label_access=False,
    )
    base.update(overrides)
    from protoem_ct.external.image_only_inference import (
        phase8_external_prediction_record_identity_payload,
    )

    stand_in_hash = sha256_json(
        phase8_external_prediction_record_identity_payload(
            SimpleNamespace(record_hash="0" * 64, **base)  # type: ignore[arg-type]
        )
    )
    return Phase8ExternalPredictionRecord(record_hash=stand_in_hash, **base)  # type: ignore[arg-type]


def test_prediction_record_rejects_wrong_threshold() -> None:
    with pytest.raises(Exception):  # noqa: B017, PT011 -- self-hash mismatch or field validation
        _fake_record(threshold=0.6)


def test_prediction_record_rejects_label_access_true() -> None:
    with pytest.raises(Exception):  # noqa: B017, PT011
        _fake_record(external_label_access=True)


def test_prediction_record_metadata_contains_no_label_path_tokens() -> None:
    record = _fake_record()
    payload_text = json.dumps(phase8_external_prediction_record_to_dict(record)).lower()
    for forbidden in ("mask", "labelled_dicom", "meshes_vtk", "liver_"):
        assert forbidden not in payload_text


def _fake_lock(**overrides: object) -> Phase8ExternalPredictionLock:
    base = dict(
        schema_name="phase8_external_prediction_lock",
        schema_version="v1",
        cohort_identifier="3d_ircadb_01",
        preregistration_hash=_FAKE_SHA256,
        freeze_artifact_sha256=_FAKE_SHA256,
        checkpoint_sha256=_FAKE_SHA256,
        definitive_config_hash=_FAKE_SHA256,
        frozen_threshold=0.5,
        support_policy="no_support",
        target_spacing_xyz_mm=(0.767578125, 0.767578125, 1.0),
        observed_image_case_count=2,
        ordered_case_ids=("ext-ircadb-001", "ext-ircadb-002"),
        case_record_hashes=(_FAKE_SHA256, _FAKE_SHA256),
        prediction_sha256_by_case=(
            ("ext-ircadb-001", _FAKE_SHA256),
            ("ext-ircadb-002", _FAKE_SHA256),
        ),
        inference_completion_state="completed",
        external_label_access=False,
        no_tuning=True,
        git_commit="a" * 40,
    )
    base.update(overrides)
    from protoem_ct.external.image_only_inference import (
        phase8_external_prediction_lock_identity_payload,
    )

    stand_in_hash = sha256_json(
        phase8_external_prediction_lock_identity_payload(
            SimpleNamespace(lock_hash="0" * 64, **base)  # type: ignore[arg-type]
        )
    )
    return Phase8ExternalPredictionLock(lock_hash=stand_in_hash, **base)  # type: ignore[arg-type]


def test_prediction_lock_accepts_well_formed_complete_inventory() -> None:
    lock = _fake_lock()
    assert lock.observed_image_case_count == 2
    assert lock.external_label_access is False


def test_prediction_lock_rejects_duplicate_case_ids() -> None:
    with pytest.raises(Exception):  # noqa: B017, PT011
        _fake_lock(
            ordered_case_ids=("ext-ircadb-001", "ext-ircadb-001"),
            case_record_hashes=(_FAKE_SHA256, _FAKE_SHA256),
            prediction_sha256_by_case=(
                ("ext-ircadb-001", _FAKE_SHA256),
                ("ext-ircadb-001", _FAKE_SHA256),
            ),
        )


def test_prediction_lock_rejects_missing_case_count_mismatch() -> None:
    with pytest.raises(Exception):  # noqa: B017, PT011
        _fake_lock(observed_image_case_count=5)


def test_prediction_lock_rejects_incomplete_prediction_coverage() -> None:
    with pytest.raises(Exception):  # noqa: B017, PT011
        _fake_lock(
            prediction_sha256_by_case=(("ext-ircadb-001", _FAKE_SHA256),),
        )


def test_prediction_lock_rejects_external_label_access_true() -> None:
    with pytest.raises(Exception):  # noqa: B017, PT011
        _fake_lock(external_label_access=True)


def test_prediction_lock_rejects_incomplete_state() -> None:
    with pytest.raises(Exception):  # noqa: B017, PT011
        _fake_lock(inference_completion_state="in_progress")


# ---------------------------------------------------------------------------
# Full end-to-end tiny run (real torch/monai SegResNet, synthetic checkpoint)
# ---------------------------------------------------------------------------


@requires_baseline_environment
def test_end_to_end_tiny_run_publishes_lock_and_fails_on_second_run(tmp_path: Path) -> None:
    from protoem_ct.external import definitive_pipeline as pipeline
    from protoem_ct.external import image_only_inference as target

    config = pipeline.build_definitive_config_v1()
    torch = pytest.importorskip("torch")
    monai_module = pytest.importorskip("monai")
    model = monai_module.networks.nets.SegResNet(
        spatial_dims=config.spatial_dims,
        init_filters=config.init_filters,
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        dropout_prob=config.dropout_prob,
        blocks_down=config.blocks_down,
        blocks_up=config.blocks_up,
        upsample_mode=config.upsample_mode,
    ).to(torch.device("cpu"))
    state_dict_path = tmp_path / "checkpoint.pt"
    torch.save(model.state_dict(), state_dict_path)

    # discover_phase8_ircadb_image_layout hard-requires exactly the real cohort's 20 ordinals, so
    # this end-to-end fixture must supply all 20 (kept tiny: 12x12x6 voxels each) to exercise the
    # real image-only discovery contract unmodified.
    dataset_root = _build_ircadb_root(
        tmp_path / "dataset", case_count=20, slices_per_case=6, rows=12, columns=12
    )
    freeze_path, prereg_path, _unused_checkpoint = _build_frozen_artifacts(tmp_path)
    repository_root = tmp_path / "repo"
    repository_root.mkdir()

    # Monkeypatch the required identities to this synthetic checkpoint/config for an isolated,
    # self-consistent end-to-end run (the real frozen hashes are verified separately above).
    monkeypatch_targets = {
        "REQUIRED_CHECKPOINT_SHA256": sha256_file(state_dict_path),
        "REQUIRED_FREEZE_ARTIFACT_SHA256": sha256_file(freeze_path),
        "REQUIRED_PREREGISTRATION_HASH": json.loads(prereg_path.read_text())[
            "preregistration_hash"
        ],
    }
    original_values = {
        name: getattr(target, name) for name in monkeypatch_targets
    }
    for name, value in monkeypatch_targets.items():
        setattr(target, name, value)
    try:
        output_root = tmp_path / "output"
        result = target.run_phase8_external_image_only_inference(
            dataset_root=dataset_root,
            freeze_path=freeze_path,
            preregistration_path=prereg_path,
            checkpoint_path=state_dict_path,
            output_root=output_root,
            repository_root=repository_root,
            git_commit="a" * 40,
        )
        assert result.lock.observed_image_case_count == 20
        assert result.lock.external_label_access is False
        assert (output_root / target.PHASE8_PREDICTION_LOCK_FILENAME).is_file()
        for ordinal in range(1, 21):
            case_id = f"ext-ircadb-{ordinal:03d}"
            prediction_path = output_root / "predictions" / f"{case_id}.npy"
            assert prediction_path.is_file()
            expected_sha256 = dict(result.lock.prediction_sha256_by_case)[case_id]
            assert sha256_file(prediction_path) == expected_sha256

        # A second run against the same, now-existing output root must fail closed.
        with pytest.raises(Phase8ExternalInferenceOutputRootError):
            target.run_phase8_external_image_only_inference(
                dataset_root=dataset_root,
                freeze_path=freeze_path,
                preregistration_path=prereg_path,
                checkpoint_path=state_dict_path,
                output_root=output_root,
                repository_root=repository_root,
                git_commit="a" * 40,
            )
    finally:
        for name, value in original_values.items():
            setattr(target, name, value)


@requires_baseline_environment
def test_run_inference_case_error_is_raised_for_unreadable_image(tmp_path: Path) -> None:
    """A corrupted PATIENT_DICOM.zip must fail closed via Phase8ExternalInferenceCaseError.

    Uses the same required-identity monkeypatch as the end-to-end test so this run genuinely
    reaches image loading (past identity verification and dataset discovery) before failing on
    the corrupted archive -- otherwise the real, hardcoded frozen hashes would reject the
    synthetic fixtures before the corrupted-image code path is ever exercised.
    """

    import torch

    from protoem_ct.external import definitive_pipeline as pipeline
    from protoem_ct.external import image_only_inference as target

    config = pipeline.build_definitive_config_v1()
    monai_module = pytest.importorskip("monai")
    model = monai_module.networks.nets.SegResNet(
        spatial_dims=config.spatial_dims,
        init_filters=config.init_filters,
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        dropout_prob=config.dropout_prob,
        blocks_down=config.blocks_down,
        blocks_up=config.blocks_up,
        upsample_mode=config.upsample_mode,
    ).to(torch.device("cpu"))
    state_dict_path = tmp_path / "checkpoint.pt"
    torch.save(model.state_dict(), state_dict_path)

    dataset_root = _build_ircadb_root(
        tmp_path / "dataset", case_count=20, slices_per_case=6, rows=12, columns=12
    )
    (dataset_root / "3Dircadb1.1" / "PATIENT_DICOM.zip").write_bytes(b"not-a-zip-at-all")
    freeze_path, prereg_path, _unused_checkpoint = _build_frozen_artifacts(tmp_path)
    repository_root = tmp_path / "repo"
    repository_root.mkdir()

    monkeypatch_targets = {
        "REQUIRED_CHECKPOINT_SHA256": sha256_file(state_dict_path),
        "REQUIRED_FREEZE_ARTIFACT_SHA256": sha256_file(freeze_path),
        "REQUIRED_PREREGISTRATION_HASH": json.loads(prereg_path.read_text())[
            "preregistration_hash"
        ],
    }
    original_values = {name: getattr(target, name) for name in monkeypatch_targets}
    for name, value in monkeypatch_targets.items():
        setattr(target, name, value)
    try:
        output_root = tmp_path / "output"
        with pytest.raises(Phase8ExternalInferenceCaseError):
            target.run_phase8_external_image_only_inference(
                dataset_root=dataset_root,
                freeze_path=freeze_path,
                preregistration_path=prereg_path,
                checkpoint_path=state_dict_path,
                output_root=output_root,
                repository_root=repository_root,
                git_commit="a" * 40,
            )
        assert not (output_root / target.PHASE8_PREDICTION_LOCK_FILENAME).exists()
    finally:
        for name, value in original_values.items():
            setattr(target, name, value)
