"""Windowing and scaling: what prepare and train feed to the model."""

import numpy as np
import pandas as pd
import pytest

from rides.data import SENSORS, Scaler, make_windows


def fake_rides(lengths: dict[int, int]) -> pd.DataFrame:
    """One row per time step; sensor values encode (ride_id, step) so windows can be traced back."""
    rows = [
        {"ride_id": ride_id, "time_index": t, **{s: ride_id * 1000 + t for s in SENSORS}}
        for ride_id, n in lengths.items()
        for t in range(n)
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def labels() -> pd.DataFrame:
    return pd.DataFrame({"ride_id": [1, 2, 3], "y": [0, 4, 2]})


def test_window_shape_and_dtype(labels):
    x, y, groups = make_windows(fake_rides({1: 250}), labels, window=100, stride=50)
    assert x.shape == (4, 100, len(SENSORS))  # starts at 0, 50, 100, 150
    assert x.dtype == np.float32
    assert len(y) == len(groups) == len(x)


def test_windows_never_mix_two_rides(labels):
    x, _, groups = make_windows(fake_rides({1: 230, 2: 180}), labels, window=100, stride=50)
    ride_of_each_step = x[..., 0] // 1000
    assert (ride_of_each_step == groups[:, None]).all()


def test_each_window_inherits_its_ride_label(labels):
    _, y, groups = make_windows(fake_rides({1: 200, 2: 200, 3: 200}), labels, window=100, stride=100)
    expected = labels.set_index("ride_id")["y"]
    assert (y == expected[groups].to_numpy()).all()


def test_ride_shorter_than_a_window_gives_no_window(labels):
    _, _, groups = make_windows(fake_rides({1: 99, 2: 100}), labels, window=100, stride=50)
    assert set(groups) == {2}


def test_scaler_standardises_each_sensor_on_training_data():
    rng = np.random.default_rng(0)
    x = rng.normal(loc=[0, 0, 9.81, 0, 0, 0], scale=[1, 2, 3, 0.1, 0.2, 0.3], size=(50, 100, 6))
    scaled = Scaler.fit(x).transform(x)
    flat = scaled.reshape(-1, 6)
    np.testing.assert_allclose(flat.mean(axis=0), 0, atol=1e-5)
    np.testing.assert_allclose(flat.std(axis=0), 1, atol=1e-4)
    assert scaled.dtype == np.float32


def test_scaler_survives_a_constant_sensor():
    x = np.ones((10, 100, 6), dtype=np.float32)
    assert np.isfinite(Scaler.fit(x).transform(x)).all()
