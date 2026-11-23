"""Ride split: train / val / test must never share a ride and must keep every class."""
import pandas as pd
import pytest

from rides.data import CLASSES, split_rides

# same class counts as GBG500: 416 Safe, 38 Safe overall, 22 Bad weather, 18 Tandem, 6 Reckless
COUNTS = [416, 38, 22, 18, 6]


@pytest.fixture
def labels() -> pd.DataFrame:
    y = [label for label, count in enumerate(COUNTS) for _ in range(count)]
    return pd.DataFrame({"ride_id": range(len(y)), "y": y})


def test_splits_are_disjoint_and_cover_every_ride(labels):
    splits = split_rides(labels, test_size=0.2, val_size=0.2, seed=42)
    train, val, test = splits["train"], splits["val"], splits["test"]
    assert not (train & val or train & test or val & test)
    assert train | val | test == set(labels["ride_id"])


def test_every_split_contains_every_class(labels):
    splits = split_rides(labels, test_size=0.2, val_size=0.2, seed=42)
    y_by_ride = labels.set_index("ride_id")["y"]
    for ids in splits.values():
        assert set(y_by_ride[list(ids)]) == set(range(len(CLASSES)))


def test_test_split_does_not_depend_on_val_size(labels):
    small_val = split_rides(labels, test_size=0.2, val_size=0.1, seed=42)
    large_val = split_rides(labels, test_size=0.2, val_size=0.3, seed=42)
    assert small_val["test"] == large_val["test"]
