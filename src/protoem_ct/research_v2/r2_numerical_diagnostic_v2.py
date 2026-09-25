"""Pure helpers for the corrected R2 numerical-stability diagnostic v2."""

from __future__ import annotations

from typing import Final, Literal

R2_DIAGNOSTIC_FAILURE_STEP: Final[int] = 227
R2_DIAGNOSTIC_REPLAY_STEPS: Final[int] = 226
R2_GRAD_SCALER_INITIAL_SCALE: Final[float] = 65536.0
R2_GRAD_SCALER_GROWTH_FACTOR: Final[float] = 2.0
R2_GRAD_SCALER_BACKOFF_FACTOR: Final[float] = 0.5
R2_GRAD_SCALER_GROWTH_INTERVAL: Final[int] = 2000
R2_SINGLE_BACKOFF_SCALE: Final[float] = 32768.0
R2_REPLAY_LOSS_ATOL: Final[float] = 1e-5
R2_REPLAY_PATCH_DICE_ATOL: Final[float] = 1e-6

ProbeClassification = Literal[
    "AMP_BACKOFF_RECOVERY_SUPPORTED",
    "AMP_OVERFLOW_PERSISTS_AT_BACKOFF_SCALE",
    "FP32_INSTABILITY_OBSERVED",
    "FAILURE_NOT_REPRODUCED",
]


def expected_loss_scale_before_step(
    step: int,
    *,
    initial_scale: float = R2_GRAD_SCALER_INITIAL_SCALE,
    growth_factor: float = R2_GRAD_SCALER_GROWTH_FACTOR,
    growth_interval: int = R2_GRAD_SCALER_GROWTH_INTERVAL,
) -> float:
    """Return expected GradScaler scale before a step with no prior overflows."""

    if step <= 0:
        raise ValueError("R2 diagnostic step must be positive")
    if initial_scale <= 0.0 or growth_factor <= 0.0 or growth_interval <= 0:
        raise ValueError("R2 GradScaler parameters must be positive")
    completed_steps = step - 1
    completed_growth_intervals = completed_steps // growth_interval
    return float(initial_scale * (growth_factor**completed_growth_intervals))


def classify_probe(
    *,
    observed_scale_amp_gradients_finite: bool,
    backoff_scale_amp_gradients_finite: bool,
    fp32_gradients_finite: bool,
) -> ProbeClassification:
    """Classify the corrected step-227 AMP-backoff-vs-FP32 probe."""

    if observed_scale_amp_gradients_finite:
        return "FAILURE_NOT_REPRODUCED"
    if not fp32_gradients_finite:
        return "FP32_INSTABILITY_OBSERVED"
    if backoff_scale_amp_gradients_finite:
        return "AMP_BACKOFF_RECOVERY_SUPPORTED"
    return "AMP_OVERFLOW_PERSISTS_AT_BACKOFF_SCALE"


__all__ = [
    "ProbeClassification",
    "R2_DIAGNOSTIC_FAILURE_STEP",
    "R2_DIAGNOSTIC_REPLAY_STEPS",
    "R2_GRAD_SCALER_BACKOFF_FACTOR",
    "R2_GRAD_SCALER_GROWTH_FACTOR",
    "R2_GRAD_SCALER_GROWTH_INTERVAL",
    "R2_GRAD_SCALER_INITIAL_SCALE",
    "R2_REPLAY_LOSS_ATOL",
    "R2_REPLAY_PATCH_DICE_ATOL",
    "R2_SINGLE_BACKOFF_SCALE",
    "classify_probe",
    "expected_loss_scale_before_step",
]
