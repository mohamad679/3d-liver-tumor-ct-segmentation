from __future__ import annotations

import numpy as np
import pytest

from protoem_ct.research_v2.r2_overfit import (
    binary_dice,
    choose_random_center,
    choose_random_positive_center,
    positive_case_index_for_step,
    threshold_probability,
    training_role_for_step,
)


def test_training_schedule_is_exactly_three_positive_then_one_empty() -> None:
    observed = [training_role_for_step(step) for step in range(1, 9)]
    assert observed == [
        "positive",
        "positive",
        "positive",
        "empty",
        "positive",
        "positive",
        "positive",
        "empty",
    ]


def test_positive_case_indices_cycle_q20_q50_q80() -> None:
    steps = [1, 2, 3, 5, 6, 7, 9, 10, 11]
    observed = [positive_case_index_for_step(step) for step in steps]
    assert observed == [0, 1, 2, 0, 1, 2, 0, 1, 2]


def test_positive_case_index_rejects_empty_step() -> None:
    with pytest.raises(ValueError, match="non-positive"):
        positive_case_index_for_step(4)


def test_random_positive_center_is_on_tumor_and_seeded() -> None:
    mask = np.zeros((9, 8, 7), dtype=bool)
    mask[1, 2, 3] = True
    mask[7, 6, 5] = True
    first = choose_random_positive_center(mask, rng=np.random.default_rng(1729))
    second = choose_random_positive_center(mask, rng=np.random.default_rng(1729))
    assert first == second
    assert bool(mask[first])


def test_random_center_stays_inside_shape() -> None:
    center = choose_random_center((5, 6, 7), rng=np.random.default_rng(1729))
    assert 0 <= center[0] < 5
    assert 0 <= center[1] < 6
    assert 0 <= center[2] < 7


def test_binary_dice_conventions() -> None:
    empty = np.zeros((3, 3, 3), dtype=bool)
    assert binary_dice(empty, empty) == 1.0

    ground_truth = empty.copy()
    prediction = empty.copy()
    ground_truth[0, 0, 0] = True
    prediction[0, 0, 0] = True
    prediction[1, 1, 1] = True
    assert binary_dice(ground_truth, prediction) == pytest.approx(2.0 / 3.0)


def test_threshold_probability_is_locked_at_point_five() -> None:
    probability = np.array([[[0.49, 0.5, 0.9]]], dtype=np.float32)
    prediction = threshold_probability(probability)
    assert prediction.tolist() == [[[False, True, True]]]

    with pytest.raises(ValueError, match="locked"):
        threshold_probability(probability, threshold=0.4)
