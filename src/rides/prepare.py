"""DVC stage `prepare`: split rides, cut them into raw windows and save them for train and evaluate.

Windows are saved raw (not normalised): train fits the scaler, and evaluate feeds raw values
to the registered model, exactly like the API does.
"""
from pathlib import Path

import numpy as np

from rides.config import load_params
from rides.data import CLASSES, load_rides, make_windows, split_rides

WINDOWS_FILE = Path("data/processed/windows.npz")


def load_windows(path: Path = WINDOWS_FILE) -> dict[str, np.ndarray]:
    """x_* (n, window, 6) float32 raw readings, y_* class index, rides_* ride id of each window."""
    with np.load(path) as f:
        return dict(f)


def main():
    p = load_params("prepare")
    rides, labels = load_rides()
    train_ids, test_ids = split_rides(labels, test_size=p["test_size"], seed=p["seed"])

    arrays = {}
    for name, ids in [("train", train_ids), ("test", test_ids)]:
        x, y, groups = make_windows(rides[rides["ride_id"].isin(ids)], labels, p["window"], p["stride"])
        arrays |= {f"x_{name}": x, f"y_{name}": y, f"rides_{name}": groups}
        print(f"{name}: {len(ids)} rides, {len(x)} windows", dict(zip(CLASSES, np.bincount(y, minlength=len(CLASSES)).tolist())))

    WINDOWS_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(WINDOWS_FILE, **arrays)


if __name__ == "__main__":
    main()
