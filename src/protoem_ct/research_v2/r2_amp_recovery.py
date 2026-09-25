"""Locked helpers for R2 corrective AMP overflow recovery."""

from __future__ import annotations

from typing import Final

R2_AMP_INITIAL_SCALE: Final[float] = 65536.0
R2_AMP_GROWTH_FACTOR: Final[float] = 2.0
R2_AMP_BACKOFF_FACTOR: Final[float] = 0.5
R2_AMP_GROWTH_INTERVAL: Final[int] = 2000
R2_AMP_MAX_OVERFLOW_RETRIES_PER_STEP: Final[int] = 8
R2_DIAGNOSTIC_V2_JSON_SHA256: Final[str] = (
    "961b754e40811a67ef8818f4abe93d50295e1604b19c52e1a872bed9e88678b0"
)


def backed_off_scale(scale: float, *, overflow_count: int = 1) -> float:
    """Return the scale after the locked number of standard 0.5 backoffs."""

    if scale <= 0.0:
        raise ValueError("AMP scale must be positive")
    if overflow_count < 0:
        raise ValueError("overflow_count must be non-negative")
    return float(scale * (R2_AMP_BACKOFF_FACTOR**overflow_count))


def overflow_retry_allowed(completed_overflow_retries: int) -> bool:
    """Return whether another retry is allowed after an overflow."""

    if completed_overflow_retries < 0:
        raise ValueError("completed_overflow_retries must be non-negative")
    return completed_overflow_retries < R2_AMP_MAX_OVERFLOW_RETRIES_PER_STEP


__all__ = [
    "R2_AMP_BACKOFF_FACTOR",
    "R2_AMP_GROWTH_FACTOR",
    "R2_AMP_GROWTH_INTERVAL",
    "R2_AMP_INITIAL_SCALE",
    "R2_AMP_MAX_OVERFLOW_RETRIES_PER_STEP",
    "R2_DIAGNOSTIC_V2_JSON_SHA256",
    "backed_off_scale",
    "overflow_retry_allowed",
]
