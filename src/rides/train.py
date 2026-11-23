"""DVC stage `train`: train the 1D CNN, log the run to MLflow and register it as rides-classifier."""
import copy
import json
import random
import shutil
import tempfile
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import torch
from mlflow.models import ModelSignature
from mlflow.types import Schema, TensorSpec
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from rides.config import load_params
from rides.data import CLASSES, SENSORS, Scaler
from rides.model import RidesClassifier
from rides.prepare import load_windows

MODEL_NAME = "rides-classifier"
TRAIN_INFO = Path("reports/train_info.json")  # tells evaluate which model version this run registered
RECKLESS = CLASSES.index("Reckless")
BORDERLINE = [CLASSES.index(c) for c in ("Safe overall", "Bad weather", "Tandem")]
MODEL_MODULES = ("__init__.py", "data.py", "model.py")  # all the pickled model needs to load


def metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> dict[str, float]:
    """The three success metrics from the README."""
    borderline = np.isin(y_true, BORDERLINE)
    return {
        f"{prefix}_macro_f1": f1_score(y_true, y_pred, average="macro", labels=range(len(CLASSES)), zero_division=0),
        f"{prefix}_reckless_recall": recall_score(y_true == RECKLESS, y_pred == RECKLESS, zero_division=0),
        f"{prefix}_false_alarm_rate": float((y_pred[borderline] == RECKLESS).mean()) if borderline.any() else 0.0,
    }


def set_seed(seed: int) -> None:
    """Seed every random generator used in training, and ask torch for deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


@torch.no_grad()
def predict_proba(model: RidesClassifier, x: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=1024)
    return torch.cat([torch.softmax(model.logits(xb.to(device)), dim=-1).cpu() for (xb,) in loader]).numpy()


def per_ride(probas: np.ndarray, y: np.ndarray, rides: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Average window probabilities over each ride -> one prediction per ride."""
    ids = np.unique(rides)
    y_pred = np.array([probas[rides == r].mean(axis=0).argmax() for r in ids])
    y_true = np.array([y[rides == r][0] for r in ids])
    return y_true, y_pred


def train_one_epoch(model: RidesClassifier, loader: DataLoader, loss_fn: nn.Module,
                    optimizer: torch.optim.Optimizer, device: str) -> float:
    """One pass over the training windows; returns the mean loss."""
    model.train()
    total = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model.logits(xb), yb)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(xb)
    return total / len(loader.dataset)


def bundle_model_code(target: Path) -> Path:
    """Copy only the modules the pickled model needs into target/rides (no api/, no __pycache__)."""
    package = target / "rides"
    package.mkdir()
    for module in MODEL_MODULES:
        shutil.copy(Path(__file__).parent / module, package)
    return package


def main():
    params = load_params("train")  # hyperparameters live in params.yaml
    prep = load_params("prepare")
    if params["epochs"] < 1:
        raise ValueError("train.epochs must be at least 1")
    set_seed(params["seed"])
    device = params["device"]

    # raw windows from the prepare stage; the scaler is fitted on training data only
    w = load_windows()
    scaler = Scaler.fit(w["x_train"])
    x_train, y_train = scaler.transform(w["x_train"]), w["y_train"]
    x_val, y_val = scaler.transform(w["x_val"]), w["y_val"]
    x_test, y_test, rides_test = scaler.transform(w["x_test"]), w["y_test"], w["rides_test"]

    # class imbalance: each class weighs inversely to its frequency (Reckless ~58x Safe)
    counts = np.bincount(y_train, minlength=len(CLASSES))
    class_weights = torch.tensor(counts.sum() / (len(CLASSES) * counts), dtype=torch.float32)

    model = RidesClassifier(scaler.mean, scaler.std, params["channels"], params["dropout"]).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=params["batch_size"], shuffle=True,
    )

    mlflow.set_experiment("rides")
    with mlflow.start_run() as run:
        mlflow.log_params({
            **params, "window": prep["window"], "stride": prep["stride"], "test_size": prep["test_size"],
            "val_size": prep["val_size"], "split_seed": prep["seed"], "sensors": ",".join(SENSORS),
        })

        # keep the epoch with the best validation macro F1; the test set stays for the final score and the gate
        best_f1, best_epoch, best_state = -1.0, 0, None
        for epoch in range(params["epochs"]):
            train_loss = train_one_epoch(model, loader, loss_fn, optimizer, device)
            val_metrics = metrics(y_val, predict_proba(model, x_val, device).argmax(1), "val")
            mlflow.log_metrics({"train_loss": train_loss, **val_metrics}, step=epoch)
            scores = " ".join(f"{k}={v:.3f}" for k, v in val_metrics.items())
            print(f"epoch {epoch:2d} train_loss={train_loss:.3f} {scores}")
            if val_metrics["val_macro_f1"] > best_f1:
                best_f1, best_epoch = val_metrics["val_macro_f1"], epoch
                best_state = copy.deepcopy(model.state_dict())
        model.load_state_dict(best_state)
        print(f"best epoch: {best_epoch} (val macro F1 {best_f1:.3f})")

        probas = predict_proba(model, x_test, device)
        y_ride, y_ride_pred = per_ride(probas, y_test, rides_test)
        mlflow.log_metrics({
            "best_epoch": best_epoch,
            **metrics(y_test, probas.argmax(1), "window"),
            **metrics(y_ride, y_ride_pred, "ride"),
            **metrics(y_test, np.zeros_like(y_test), "baseline"),  # always "Safe"
        })
        cm = confusion_matrix(y_test, probas.argmax(1), labels=range(len(CLASSES)))
        mlflow.log_text(f"rows=true, cols=pred, order={CLASSES}\n{cm}", "confusion_matrix.txt")

        model.cpu().eval()
        # Any batch size and any T >= 10 must work (API contract), which needs two things:
        # - an explicit signature, otherwise MLflow infers (-1, 100, 6) from the example;
        # - pickle format: the default "pt2" (torch.export) freezes batch=1 and T=100 into the graph.
        #   code_paths ships the modules the model needs with it, so it loads outside this repo too.
        signature = ModelSignature(
            inputs=Schema([TensorSpec(np.dtype(np.float32), (-1, -1, len(SENSORS)))]),  # any number of steps
            outputs=Schema([TensorSpec(np.dtype(np.float32), (-1, len(CLASSES)))]),
        )
        with tempfile.TemporaryDirectory() as tmp:
            info = mlflow.pytorch.log_model(
                model,
                name="model",
                serialization_format="pickle",
                code_paths=[str(bundle_model_code(Path(tmp)))],
                signature=signature,
                input_example=w["x_test"][:1],  # raw readings, shape (1, T, C): what the API sends
                registered_model_name=MODEL_NAME,
            )

    TRAIN_INFO.parent.mkdir(parents=True, exist_ok=True)
    TRAIN_INFO.write_text(json.dumps({
        "run_id": run.info.run_id, "model_name": MODEL_NAME, "model_version": info.registered_model_version,
    }, indent=2))
    print(f"registered {MODEL_NAME} v{info.registered_model_version} (run {run.info.run_id})")


if __name__ == "__main__":
    main()
