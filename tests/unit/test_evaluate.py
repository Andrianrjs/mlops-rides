"""Evaluation gate: when does a new model version get the @staging alias?"""

import pytest

from rides.evaluate import should_promote

BASELINE = {"macro_f1": 0.18, "reckless_recall": 0.0, "false_alarm_rate": 0.0}
MIN_GAIN = 0.05
MAX_FALSE_ALARM_RATE = 0.10


def m(f1: float, recall: float = 0.4, false_alarm: float = 0.05) -> dict:
    return {"macro_f1": f1, "reckless_recall": recall, "false_alarm_rate": false_alarm}


@pytest.mark.parametrize(
    "candidate, champion, expected",
    [
        (m(0.25), None, True),  # beats baseline by 0.07, nothing in @staging
        (m(0.20), None, False),  # beats baseline by only 0.02
        (m(0.10), None, False),  # worse than the baseline
        (m(0.25, recall=0.0), None, False),  # never finds Reckless
        (m(0.25, false_alarm=0.30), None, False),  # flags too many borderline rides as Reckless
        (m(0.25, false_alarm=0.10), None, True),  # false alarm rate exactly at the limit
        (m(0.25), m(0.30), False),  # worse than the current @staging
        (m(0.30), m(0.30), True),  # as good as the current @staging
    ],
)
def test_gate(candidate, champion, expected):
    promote, reason = should_promote(candidate, BASELINE, champion, MIN_GAIN, MAX_FALSE_ALARM_RATE)
    assert promote is expected
    assert reason


def test_reason_shows_a_negative_gain_with_its_sign():
    _, reason = should_promote(m(0.10), BASELINE, None, MIN_GAIN, MAX_FALSE_ALARM_RATE)
    assert "-0.080" in reason
