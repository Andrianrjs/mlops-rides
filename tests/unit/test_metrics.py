"""The three success metrics (macro F1, Reckless recall, false alarm rate) and the per-ride vote."""

import numpy as np
import pytest

from rides.data import CLASSES
from rides.train import metrics, per_ride

SAFE, SAFE_OVERALL, BAD_WEATHER, TANDEM, RECKLESS = range(len(CLASSES))


def test_always_safe_baseline_detects_nothing():
    y = np.array([SAFE] * 80 + [SAFE_OVERALL] * 10 + [RECKLESS] * 10)
    m = metrics(y, np.zeros_like(y), "baseline")
    assert m["baseline_reckless_recall"] == 0
    assert m["baseline_false_alarm_rate"] == 0
    assert m["baseline_macro_f1"] < 0.2  # high accuracy, but only one class out of 5 is right


def test_perfect_predictions():
    y = np.array([SAFE, SAFE_OVERALL, BAD_WEATHER, TANDEM, RECKLESS] * 3)
    m = metrics(y, y.copy(), "p")
    assert m == pytest.approx({"p_macro_f1": 1.0, "p_reckless_recall": 1.0, "p_false_alarm_rate": 0.0})


def test_false_alarm_rate_counts_borderline_rides_flagged_reckless():
    y = np.array([SAFE_OVERALL, BAD_WEATHER, TANDEM, TANDEM, SAFE])
    pred = np.array([RECKLESS, BAD_WEATHER, RECKLESS, TANDEM, RECKLESS])
    # 2 of the 4 borderline windows are flagged; the Safe one flagged Reckless does not count here
    assert metrics(y, pred, "m")["m_false_alarm_rate"] == pytest.approx(0.5)


def test_false_alarm_rate_is_zero_without_borderline_rides():
    y = np.array([SAFE, RECKLESS])
    assert metrics(y, np.array([RECKLESS, RECKLESS]), "m")["m_false_alarm_rate"] == 0.0


def test_per_ride_averages_window_probabilities():
    # ride 7: two windows lean Safe, one is surely Reckless -> mean Safe 0.33, mean Reckless 0.47
    probas = np.array(
        [
            [0.5, 0.1, 0.1, 0.1, 0.2],
            [0.5, 0.1, 0.1, 0.1, 0.2],
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.1, 0.1, 0.6, 0.1, 0.1],  # ride 9
        ]
    )
    y = np.array([RECKLESS, RECKLESS, RECKLESS, BAD_WEATHER])
    rides = np.array([7, 7, 7, 9])
    y_true, y_pred = per_ride(probas, y, rides)
    assert y_true.tolist() == [RECKLESS, BAD_WEATHER]
    assert y_pred.tolist() == [RECKLESS, BAD_WEATHER]
