from __future__ import annotations

import pytest

from protoem_ct.research_v2.r2_numerical_diagnostic import (
    classify_probe,
    expected_loss_scale_before_step,
)


def test_expected_loss_scale_before_step_matches_default_growth_schedule() -> None:
    assert expected_loss_scale_before_step(1) == 65536.0
    assert expected_loss_scale_before_step(200) == 65536.0
    assert expected_loss_scale_before_step(201) == 131072.0
    assert expected_loss_scale_before_step(227) == 131072.0


def test_expected_loss_scale_before_step_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        expected_loss_scale_before_step(0)
    with pytest.raises(ValueError):
        expected_loss_scale_before_step(1, growth_interval=0)


@pytest.mark.parametrize(
    ("observed_finite", "baseline_finite", "fp32_finite", "expected"),
    [
        (False, True, True, "SCALE_GROWTH_OVERFLOW_SUPPORTED"),
        (False, False, True, "AMP_NUMERICAL_OVERFLOW_SUPPORTED"),
        (False, True, False, "FP32_INSTABILITY_OBSERVED"),
        (True, True, True, "FAILURE_NOT_REPRODUCED"),
    ],
)
def test_classify_probe(
    observed_finite: bool,
    baseline_finite: bool,
    fp32_finite: bool,
    expected: str,
) -> None:
    assert (
        classify_probe(
            observed_scale_amp_gradients_finite=observed_finite,
            baseline_scale_amp_gradients_finite=baseline_finite,
            fp32_gradients_finite=fp32_finite,
        )
        == expected
    )
