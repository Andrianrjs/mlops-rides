"""Load the GBG500 rides, split them by ride and cut them into fixed-size windows."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DIR = Path("data/raw")
SENSORS = ["ax", "ay", "az", "rx", "ry", "rz"]
CLASSES = ["Safe", "Safe overall", "Bad weather", "Tandem", "Reckless"]  # spelling from the labels file
# Window length, stride and split settings live in params.yaml (section `prepare`), not here.


def load_rides(raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read sensor readings and labels; fill the few missing readings."""
    rides = pd.read_parquet(raw_dir / "GBG500.parquet").sort_values(["ride_id", "time_index"])
    labels = pd.read_csv(raw_dir / "GBG500_labels.csv").rename(columns={"RIDE_ID": "ride_id", "LABEL": "label"})

    # ~0.7 % of readings are missing: interpolate from neighbours, within each ride only
    rides[SENSORS] = rides.groupby("ride_id")[SENSORS].transform(
        lambda s: s.interpolate(limit_direction="both")
    )
    labels["y"] = labels["label"].map(CLASSES.index)  # fails loudly on an unknown label
    return rides, labels


def split_rides(labels: pd.DataFrame, test_size: float, seed: int) -> tuple[set, set]:
    """Split ride ids (not windows!) so no ride appears in both train and test."""
    train_ids, test_ids = train_test_split(
        labels["ride_id"], test_size=test_size, stratify=labels["y"], random_state=seed
    )
    return set(train_ids), set(test_ids)


def make_windows(rides: pd.DataFrame, labels: pd.DataFrame, window: int, stride: int):
    """Cut every ride into overlapping windows; each window inherits its ride's label."""
    y_by_ride = labels.set_index("ride_id")["y"]
    xs, ys, groups = [], [], []
    for ride_id, ride in rides.groupby("ride_id", sort=False):
        values = ride[SENSORS].to_numpy(np.float32)
        for start in range(0, len(values) - window + 1, stride):
            xs.append(values[start:start + window])
            ys.append(y_by_ride[ride_id])
            groups.append(ride_id)
    return np.stack(xs), np.array(ys), np.array(groups)


@dataclass
class Scaler:
    """Per-sensor standardisation, fitted on training data only."""
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Scaler":
        flat = x.reshape(-1, x.shape[-1])
        return cls(flat.mean(axis=0), flat.std(axis=0) + 1e-8)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)
