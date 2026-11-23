"""Evaluation gate: when does a new model version get the @staging alias?"""
import pytest

from rides.evaluate import should_promote

BASELINE = {"macro_f1": 0.18, "reckless_recall": 0.0}


def m(f1, recall=0.4):
    return {"macro_f1": f1, "reckless_recall": recall}


@pytest.mark.parametrize("candidate, champion, expected", [
    (m(0.25), None, True),              # beats baseline by 0.07, nothing in @staging
    (m(0.20), None, False),             # beats baseline by only 0.02
    (m(0.25, recall=0.0), None, False),  # never finds Reckless
    (m(0.25), m(0.30), False),          # worse than the current @staging
    (m(0.30), m(0.30), True),           # as good as the current @staging
])
def test_gate(candidate, champion, expected):
    promote, reason = should_promote(candidate, BASELINE, champion, min_gain=0.05)
    assert promote is expected
    assert reason
