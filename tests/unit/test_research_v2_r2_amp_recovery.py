from __future__ import annotations

import pytest

from protoem_ct.research_v2.r2_amp_recovery import (
    R2_AMP_MAX_OVERFLOW_RETRIES_PER_STEP,
    backed_off_scale,
    overflow_retry_allowed,
)


def test_backed_off_scale_matches_diagnostic_v2() -> None:
    assert backed_off_scale(65536.0) == 32768.0
    assert backed_off_scale(65536.0, overflow_count=2) == 16384.0


def test_backed_off_scale_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        backed_off_scale(0.0)
    with pytest.raises(ValueError):
        backed_off_scale(65536.0, overflow_count=-1)


def test_overflow_retry_limit_is_fail_closed() -> None:
    assert overflow_retry_allowed(0)
    assert overflow_retry_allowed(R2_AMP_MAX_OVERFLOW_RETRIES_PER_STEP - 1)
    assert not overflow_retry_allowed(R2_AMP_MAX_OVERFLOW_RETRIES_PER_STEP)
    with pytest.raises(ValueError):
        overflow_retry_allowed(-1)
