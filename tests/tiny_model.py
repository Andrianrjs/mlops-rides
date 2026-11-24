"""Untrained RidesClassifier saved in the same MLflow format as train.py (pickle + signature).

Used by the integration tests, and by the CD workflow to give the API container a model for the
e2e tests without any MLflow server:

    docker run --rm -i --user root -v "$PWD/.e2e:/out" <image> python - /out/model < tests/tiny_model.py

Its predictions are meaningless: the tests check the API contract, not the model quality.
"""

import sys
from pathlib import Path

import mlflow.pytorch
import numpy as np
import torch
from mlflow.models import Model, ModelSignature
from mlflow.types import Schema, TensorSpec

from rides.data import CLASSES, SENSORS
from rides.model import RidesClassifier

RUN_ID = "tiny-test-model"  # the API reports metadata.run_id as model_version


def save_tiny_model(path: Path) -> Path:
    torch.manual_seed(0)
    model = RidesClassifier(np.zeros(len(SENSORS)), np.ones(len(SENSORS)), channels=8).eval()
    signature = ModelSignature(
        inputs=Schema([TensorSpec(np.dtype(np.float32), (-1, -1, len(SENSORS)))]),
        outputs=Schema([TensorSpec(np.dtype(np.float32), (-1, len(CLASSES)))]),
    )
    mlflow.pytorch.save_model(model, str(path), serialization_format="pickle", signature=signature)
    # A model saved outside a run has no run_id; registry models always have one.
    meta = Model.load(str(path))
    meta.run_id = RUN_ID
    meta.save(str(path / "MLmodel"))
    return path


if __name__ == "__main__":
    print(save_tiny_model(Path(sys.argv[1] if len(sys.argv) > 1 else "tiny-model")))
