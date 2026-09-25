"""Pure helpers for the R2 numerical-stability diagnostic."""

from __future__ import annotations

from typing import Final, Literal

R2_DIAGNOSTIC_FAILURE_STEP: Final[int] = 227
R2_DIAGNOSTIC_REPLAY_STEPS: Final[int] = 226
R2_GRAD_SCALER_INITIAL_SCALE: Final[float] = 65536.0
R2_GRAD_SCALER_GROWTH_FACTOR: Final[float] = 2.0
R2_GRAD_SCALER_BACKOFF_FACTOR: Final[float] = 0.5
R2_GRAD_SCALER_GROWTH_INTERVAL: Final[int] = 200
R2_REPLAY_LOSS_ATOL: Final[float] = 1e-5
R2_REPLAY_PATCH_DICE_ATOL: Final[float] = 1e-6

ProbeClassification = Literal[
    "SCALE_GROWTH_OVERFLOW_SUPPORTED",
    "AMP_NUMERICAL_OVERFLOW_SUPPORTED",
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
    """Return the expected GradScaler scale before a step with no prior overflows."""

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
    baseline_scale_amp_gradients_finite: bool,
    fp32_gradients_finite: bool,
) -> ProbeClassification:
    """Classify the preregistered step-227 AMP-vs-FP32 gradient probe."""

    if observed_scale_amp_gradients_finite:
        return "FAILURE_NOT_REPRODUCED"
    if not fp32_gradients_finite:
        return "FP32_INSTABILITY_OBSERVED"
    if baseline_scale_amp_gradients_finite:
        return "SCALE_GROWTH_OVERFLOW_SUPPORTED"
    return "AMP_NUMERICAL_OVERFLOW_SUPPORTED"


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
    "classify_probe",
    "expected_loss_scale_before_step",
]
