"""Train the 1D CNN, log the run to MLflow and register it as rides-classifier."""
import mlflow
import mlflow.pytorch
import numpy as np
import torch
from mlflow.models import ModelSignature
from mlflow.types import Schema, TensorSpec
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from rides.data import CLASSES, SENSORS, WINDOW, prepare
from rides.model import RidesClassifier

PARAMS = dict(epochs=15, batch_size=128, lr=1e-3, channels=64, dropout=0.3, seed=42)
RECKLESS = CLASSES.index("Reckless")
BORDERLINE = [CLASSES.index(c) for c in ("Safe overall", "Bad weather", "Tandem")]


def metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> dict[str, float]:
    """The three success metrics from the README."""
    borderline = np.isin(y_true, BORDERLINE)
    return {
        f"{prefix}_macro_f1": f1_score(y_true, y_pred, average="macro", labels=range(len(CLASSES)), zero_division=0),
        f"{prefix}_reckless_recall": recall_score(y_true == RECKLESS, y_pred == RECKLESS, zero_division=0),
        f"{prefix}_false_alarm_rate": float((y_pred[borderline] == RECKLESS).mean()) if borderline.any() else 0.0,
    }


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


def main():
    torch.manual_seed(PARAMS["seed"])
    device = "mps" if torch.backends.mps.is_available() else "cpu"  # Apple GPU when available
    ds = prepare(seed=PARAMS["seed"])

    # class imbalance: each class weighs inversely to its frequency (Reckless ~58x Safe)
    counts = np.bincount(ds.y_train, minlength=len(CLASSES))
    class_weights = torch.tensor(counts.sum() / (len(CLASSES) * counts), dtype=torch.float32)

    model = RidesClassifier(ds.scaler.mean, ds.scaler.std, PARAMS["channels"], PARAMS["dropout"]).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=PARAMS["lr"])
    loader = DataLoader(
        TensorDataset(torch.from_numpy(ds.x_train), torch.from_numpy(ds.y_train)),
        batch_size=PARAMS["batch_size"], shuffle=True,
    )

    mlflow.set_experiment("rides")
    with mlflow.start_run():
        mlflow.log_params({**PARAMS, "window": WINDOW, "sensors": ",".join(SENSORS), "device": device})

        for epoch in range(PARAMS["epochs"]):
            model.train()
            total = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = loss_fn(model.logits(xb), yb)
                loss.backward()
                optimizer.step()
                total += loss.item() * len(xb)

            # test set used for monitoring only, no model selection on it
            probas = predict_proba(model, ds.x_test, device)
            epoch_metrics = {"train_loss": total / len(loader.dataset), **metrics(ds.y_test, probas.argmax(1), "window")}
            mlflow.log_metrics(epoch_metrics, step=epoch)
            print(f"epoch {epoch:2d} " + " ".join(f"{k}={v:.3f}" for k, v in epoch_metrics.items()))

        y_ride, y_ride_pred = per_ride(probas, ds.y_test, ds.rides_test)
        mlflow.log_metrics({
            **metrics(ds.y_test, probas.argmax(1), "window"),
            **metrics(y_ride, y_ride_pred, "ride"),
            **metrics(ds.y_test, np.zeros_like(ds.y_test), "baseline"),  # always "Safe"
        })
        cm = confusion_matrix(ds.y_test, probas.argmax(1), labels=range(len(CLASSES)))
        mlflow.log_text(f"rows=true, cols=pred, order={CLASSES}\n{cm}", "confusion_matrix.txt")

        model.cpu().eval()
        signature = ModelSignature(
            inputs=Schema([TensorSpec(np.dtype(np.float32), (-1, -1, len(SENSORS)))]),  # any number of steps
            outputs=Schema([TensorSpec(np.dtype(np.float32), (-1, len(CLASSES)))]),
        )
        mlflow.pytorch.log_model(
            model, artifact_path="model", signature=signature,
            code_paths=["src/rides"],  # ships the RidesClassifier class with the model
            registered_model_name="rides-classifier",
        )


if __name__ == "__main__":
    main()