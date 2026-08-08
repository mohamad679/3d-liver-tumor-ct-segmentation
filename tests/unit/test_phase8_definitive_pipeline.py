"""Unit tests for the Phase 8 definitive development pipeline (Package A).

All tests operate on small, synthetic, in-memory NumPy arrays only. No real
dataset, checkpoint, or ``/Volumes`` path is referenced anywhere in this
file.
"""

from __future__ import annotations

import dataclasses
import importlib.util
from typing import Any, cast

import nibabel as nib
import numpy as np
import pytest

from protoem_ct.external.definitive_pipeline import (
    PHASE8_DEFINITIVE_CONFIG_DESIGN_IDENTIFIER,
    Phase8DefinitiveConfig,
    Phase8DefinitivePipelineCheckpointError,
    Phase8DefinitivePipelineConfigError,
    Phase8DefinitivePipelineRuntimeError,
    Phase8DefinitivePipelineSamplingError,
    Phase8DefinitiveTrainCase,
    Phase8DefinitiveValidationCase,
    binarize_tumor_label,
    build_definitive_checkpoint_metadata,
    build_definitive_config_v1,
    clip_and_scale_intensity,
    compute_median_train_spacing,
    reorient_volume_to_ras,
    resample_volume_to_spacing,
    run_definitive_training,
    run_definitive_validation,
    sample_foreground_aware_patches,
)
from protoem_ct.external.internal_evidence import ArtifactReference

requires_baseline_environment = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None or importlib.util.find_spec("monai") is None,
    reason="Phase 8 definitive pipeline torch/monai tests require the isolated baseline "
    "environment.",
)

_FAKE_SHA256 = "a" * 64
_FAKE_GIT_COMMIT = "b" * 40


def _package_environment_reference() -> ArtifactReference:
    return ArtifactReference(
        schema_name="phase3_cpu_environment",
        schema_version="v1",
        artifact_hash=_FAKE_SHA256,
        artifact_role="package_environment",
    )


# ---------------------------------------------------------------------------
# A. Locked configuration
# ---------------------------------------------------------------------------


def test_build_definitive_config_v1_is_valid_and_self_verifies() -> None:
    config = build_definitive_config_v1()
    assert config.design_identifier == PHASE8_DEFINITIVE_CONFIG_DESIGN_IDENTIFIER
    assert config.patch_size == (64, 64, 32)
    assert config.sliding_window_roi_size == (96, 96, 64)
    assert config.learning_rate == 1e-4
    assert config.weight_decay == 1e-5
    assert config.device_type == "cpu"
    assert config.amp_enabled is False
    assert config.seed == 1729
    assert config.inference_threshold == 0.5
    assert len(config.config_hash) == 64


@pytest.mark.parametrize(
    "field_name,bad_value",
    [
        ("patch_size", (32, 32, 16)),
        ("learning_rate", 1e-3),
        ("device_type", "cuda"),
        ("inference_threshold", 0.6),
    ],
)
def test_tampered_locked_field_fails_closed(field_name: str, bad_value: object) -> None:
    config = build_definitive_config_v1()
    payload = {f.name: getattr(config, f.name) for f in dataclasses.fields(config)}
    payload[field_name] = bad_value
    with pytest.raises(Phase8DefinitivePipelineConfigError):
        Phase8DefinitiveConfig(**payload)


def test_tampered_config_hash_fails_closed() -> None:
    config = build_definitive_config_v1()
    payload = {f.name: getattr(config, f.name) for f in dataclasses.fields(config)}
    payload["config_hash"] = "0" * 64
    with pytest.raises(Phase8DefinitivePipelineConfigError):
        Phase8DefinitiveConfig(**payload)


# ---------------------------------------------------------------------------
# B. Preprocessing
# ---------------------------------------------------------------------------


def test_reorient_volume_to_ras_from_lps() -> None:
    shape = (10, 12, 14)
    volume = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    # LPS affine: negate the first two diagonal entries relative to RAS.
    affine = np.diag([-1.0, -1.0, 1.0, 1.0])
    axcodes_before = cast(Any, nib).aff2axcodes(affine)
    assert axcodes_before == ("L", "P", "S")

    reoriented_volume, reoriented_affine = reorient_volume_to_ras(volume, affine)

    axcodes_after = cast(Any, nib).aff2axcodes(reoriented_affine)
    assert axcodes_after == ("R", "A", "S")
    assert reoriented_volume.shape == shape


def test_compute_median_train_spacing() -> None:
    spacings = [
        (1.0, 1.0, 2.0),
        (2.0, 1.5, 2.5),
        (3.0, 2.0, 3.0),
    ]
    median = compute_median_train_spacing(spacings)
    assert median == (2.0, 1.5, 2.5)


def test_compute_median_train_spacing_raises_on_empty() -> None:
    with pytest.raises(Phase8DefinitivePipelineConfigError):
        compute_median_train_spacing([])


def test_resample_volume_to_spacing_changes_shape_and_uses_expected_interpolation() -> None:
    shape = (10, 10, 10)
    rng = np.random.default_rng(0)
    image = rng.uniform(-500.0, 500.0, size=shape).astype(np.float32)
    label = np.zeros(shape, dtype=np.float32)
    label[4:6, 4:6, 4:6] = 1.0

    source_spacing = (1.0, 1.0, 1.0)
    target_spacing = (0.5, 0.5, 0.5)  # upsample by 2x

    resampled_image = resample_volume_to_spacing(
        image, source_spacing=source_spacing, target_spacing=target_spacing, is_label=False
    )
    resampled_label = resample_volume_to_spacing(
        label, source_spacing=source_spacing, target_spacing=target_spacing, is_label=True
    )

    assert resampled_image.shape[0] > shape[0]
    assert resampled_label.shape == resampled_image.shape
    # Nearest-neighbor resampling of a {0, 1}-valued label must not introduce
    # any intermediate values.
    assert set(np.unique(resampled_label).tolist()) <= {0.0, 1.0}


def test_clip_and_scale_intensity_bounds_output() -> None:
    config = build_definitive_config_v1()
    volume = np.array([-5000.0, -1000.0, 0.0, 1000.0, 5000.0], dtype=np.float32)
    scaled = clip_and_scale_intensity(volume, config=config)
    assert scaled.min() >= -1.0 - 1e-6
    assert scaled.max() <= 1.0 + 1e-6
    assert scaled[0] == pytest.approx(-1.0)
    assert scaled[-1] == pytest.approx(1.0)
    assert scaled[2] == pytest.approx(0.0)


def test_binarize_tumor_label_isolates_raw_value_two_only() -> None:
    config = build_definitive_config_v1()
    label_array = np.array([0, 1, 2, 1, 2, 0], dtype=np.int64)
    binary = binarize_tumor_label(label_array, config=config)
    np.testing.assert_array_equal(binary, np.array([False, False, True, False, True, False]))


# ---------------------------------------------------------------------------
# C. Foreground-aware patch sampling
# ---------------------------------------------------------------------------


def _synthetic_volume_and_label(
    shape: tuple[int, int, int] = (160, 160, 80),
) -> tuple[np.ndarray, np.ndarray]:
    # The volume is deliberately much larger than the locked (64, 64, 32)
    # patch size, and the foreground is placed in one corner, so that many
    # background-centered crops (elsewhere in the volume) do not overlap it
    # -- otherwise an all-background patch could never be found.
    rng = np.random.default_rng(42)
    image = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
    label = np.zeros(shape, dtype=bool)
    label[140:144, 140:144, 70:74] = True
    return image, label


def test_sample_foreground_aware_patches_positive_and_negative() -> None:
    config = build_definitive_config_v1()
    image, label = _synthetic_volume_and_label()
    rng = np.random.default_rng(1729)

    patches = sample_foreground_aware_patches(
        image, label, config=config, positive_count=5, negative_count=5, rng=rng
    )
    positives = [p for p in patches if p.is_positive]
    negatives = [p for p in patches if not p.is_positive]
    assert len(positives) == 5
    assert len(negatives) == 5
    for patch in positives:
        assert patch.image_patch.shape == config.patch_size
        assert bool(np.any(patch.label_patch))
    for patch in negatives:
        assert patch.image_patch.shape == config.patch_size
        assert not bool(np.any(patch.label_patch))


def test_sample_foreground_aware_patches_raises_when_no_foreground() -> None:
    config = build_definitive_config_v1()
    shape = (80, 80, 40)
    image, _ = _synthetic_volume_and_label(shape)
    empty_label = np.zeros(shape, dtype=bool)
    rng = np.random.default_rng(0)

    with pytest.raises(Phase8DefinitivePipelineSamplingError):
        sample_foreground_aware_patches(
            image, empty_label, config=config, positive_count=1, negative_count=0, rng=rng
        )


def test_sample_foreground_aware_patches_raises_when_volume_smaller_than_patch() -> None:
    config = build_definitive_config_v1()
    small_shape = (32, 32, 16)  # smaller than patch_size (64, 64, 32) in every dim
    image = np.zeros(small_shape, dtype=np.float32)
    label = np.zeros(small_shape, dtype=bool)
    label[1, 1, 1] = True
    rng = np.random.default_rng(0)

    with pytest.raises(Phase8DefinitivePipelineSamplingError):
        sample_foreground_aware_patches(
            image, label, config=config, positive_count=1, negative_count=0, rng=rng
        )


# ---------------------------------------------------------------------------
# D/E/F. torch/monai-dependent training, validation, checkpoint metadata
# ---------------------------------------------------------------------------


@requires_baseline_environment
def test_run_definitive_training_produces_finite_losses() -> None:
    config = build_definitive_config_v1()
    shape = (160, 160, 80)

    rng = np.random.default_rng(7)
    positive_image = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
    positive_label = np.zeros(shape, dtype=bool)
    positive_label[140:144, 140:144, 70:74] = True

    empty_image = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
    empty_label = np.zeros(shape, dtype=bool)

    train_cases = [
        Phase8DefinitiveTrainCase(
            case_id="positive_case", image=positive_image, label_binary=positive_label
        ),
        Phase8DefinitiveTrainCase(
            case_id="empty_case", image=empty_image, label_binary=empty_label
        ),
    ]

    result = run_definitive_training(train_cases, config=config, max_steps=3)

    assert result.all_steps_finite is True
    assert len(result.step_results) == 3
    for step_result in result.step_results:
        assert step_result.loss_finite is True
        assert step_result.gradient_finite is True
        assert np.isfinite(step_result.loss_value)
    assert result.model_state_dict is not None


@requires_baseline_environment
def test_run_definitive_training_requires_at_least_one_case_with_foreground() -> None:
    config = build_definitive_config_v1()
    shape = (80, 80, 40)
    empty_image = np.zeros(shape, dtype=np.float32)
    empty_label = np.zeros(shape, dtype=bool)
    train_cases = [
        Phase8DefinitiveTrainCase(case_id="only_empty", image=empty_image, label_binary=empty_label)
    ]
    with pytest.raises(Phase8DefinitivePipelineRuntimeError):
        run_definitive_training(train_cases, config=config, max_steps=1)


@requires_baseline_environment
def test_run_definitive_validation_returns_finite_metrics_in_unit_interval() -> None:
    config = build_definitive_config_v1()
    shape = (160, 160, 80)

    rng = np.random.default_rng(11)
    image = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
    label = np.zeros(shape, dtype=bool)
    label[140:144, 140:144, 70:74] = True

    train_cases = [Phase8DefinitiveTrainCase(case_id="train_case", image=image, label_binary=label)]
    training_result = run_definitive_training(train_cases, config=config, max_steps=1)
    assert training_result.all_steps_finite is True

    import monai  # type: ignore[import-not-found]
    import torch  # type: ignore[import-not-found]

    from protoem_ct.external.definitive_pipeline import _build_definitive_segresnet_model

    model = _build_definitive_segresnet_model(torch=torch, monai=monai)
    model.load_state_dict(training_result.model_state_dict)

    validation_case = Phase8DefinitiveValidationCase(
        case_id="validation_case", image=image, label_binary=label
    )
    result = run_definitive_validation(model, validation_case, config=config)

    assert result.prediction_finite is True
    assert 0.0 <= result.tumor_dice <= 1.0
    assert 0.0 <= result.tumor_iou <= 1.0
    assert np.isfinite(result.tumor_dice)
    assert np.isfinite(result.tumor_iou)


@requires_baseline_environment
def test_build_definitive_checkpoint_metadata_enforces_not_freeze_eligible() -> None:
    config = build_definitive_config_v1()

    metadata = build_definitive_checkpoint_metadata(
        checkpoint_sha256=_FAKE_SHA256,
        checkpoint_byte_size=1024,
        config=config,
        development_manifest_hash=_FAKE_SHA256,
        development_split_hash=_FAKE_SHA256,
        originating_git_commit=_FAKE_GIT_COMMIT,
        package_environment_reference=_package_environment_reference(),
        completion_status="completed",
        preprocessing_evidence_hash=_FAKE_SHA256,
    )
    assert metadata.freeze_eligible is False
    assert metadata.training_config_hash == config.config_hash
    assert metadata.no_external_data is True


def test_build_definitive_checkpoint_metadata_rejects_freeze_eligible_true() -> None:
    config = build_definitive_config_v1()
    with pytest.raises(Phase8DefinitivePipelineCheckpointError):
        build_definitive_checkpoint_metadata(
            checkpoint_sha256=_FAKE_SHA256,
            checkpoint_byte_size=1024,
            config=config,
            development_manifest_hash=_FAKE_SHA256,
            development_split_hash=_FAKE_SHA256,
            originating_git_commit=_FAKE_GIT_COMMIT,
            package_environment_reference=_package_environment_reference(),
            completion_status="completed",
            preprocessing_evidence_hash=_FAKE_SHA256,
            freeze_eligible=True,
        )
