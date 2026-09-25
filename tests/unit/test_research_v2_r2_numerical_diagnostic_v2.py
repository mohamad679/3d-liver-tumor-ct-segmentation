from protoem_ct.research_v2.r2_numerical_diagnostic_v2 import (
    R2_DIAGNOSTIC_FAILURE_STEP,
    R2_SINGLE_BACKOFF_SCALE,
    classify_probe,
    expected_loss_scale_before_step,
)


def test_step_227_scale_uses_torch_grad_scaler_default_growth_interval() -> None:
    assert expected_loss_scale_before_step(R2_DIAGNOSTIC_FAILURE_STEP) == 65536.0
    assert expected_loss_scale_before_step(2001) == 131072.0
    assert R2_SINGLE_BACKOFF_SCALE == 32768.0


def test_classify_amp_backoff_recovery() -> None:
    assert (
        classify_probe(
            observed_scale_amp_gradients_finite=False,
            backoff_scale_amp_gradients_finite=True,
            fp32_gradients_finite=True,
        )
        == "AMP_BACKOFF_RECOVERY_SUPPORTED"
    )


def test_classify_persistent_amp_overflow() -> None:
    assert (
        classify_probe(
            observed_scale_amp_gradients_finite=False,
            backoff_scale_amp_gradients_finite=False,
            fp32_gradients_finite=True,
        )
        == "AMP_OVERFLOW_PERSISTS_AT_BACKOFF_SCALE"
    )


def test_classify_fp32_instability() -> None:
    assert (
        classify_probe(
            observed_scale_amp_gradients_finite=False,
            backoff_scale_amp_gradients_finite=False,
            fp32_gradients_finite=False,
        )
        == "FP32_INSTABILITY_OBSERVED"
    )


def test_classify_failure_not_reproduced() -> None:
    assert (
        classify_probe(
            observed_scale_amp_gradients_finite=True,
            backoff_scale_amp_gradients_finite=True,
            fp32_gradients_finite=True,
        )
        == "FAILURE_NOT_REPRODUCED"
    )
