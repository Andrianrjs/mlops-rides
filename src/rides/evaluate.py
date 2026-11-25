"""DVC stage `evaluate`: score the version train just registered, then promote it to @staging if it passes the gate.

The model is loaded from the registry with mlflow.pyfunc and fed raw readings, exactly like the API.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from rides.config import load_params
from rides.data import CLASSES
from rides.prepare import load_windows
from rides.train import TRAIN_INFO, metrics, per_ride

EVAL_FILE = Path("reports/eval.json")                # dvc metrics
PREDICTIONS_FILE = Path("reports/predictions.csv")  # dvc plots (confusion matrix)


def should_promote(candidate: dict, baseline: dict, champion: dict | None,
                   min_gain: float, max_false_alarm_rate: float) -> tuple[bool, str]:
    """Evaluation gate. Each dict holds window-level metrics: macro_f1, reckless_recall, false_alarm_rate.

    Promote only if the candidate beats the "always Safe" baseline by min_gain on macro F1,
    detects Reckless better than it, rarely flags the borderline classes as Reckless,
    and is not worse than the model currently in @staging.
    """
    gain = candidate["macro_f1"] - baseline["macro_f1"]
    if gain < min_gain:
        return False, (f"macro F1 {candidate['macro_f1']:.3f} is {gain:+.3f} vs the baseline "
                       f"({baseline['macro_f1']:.3f}), at least +{min_gain} needed")
    if candidate["reckless_recall"] <= baseline["reckless_recall"]:
        return False, f"Reckless recall {candidate['reckless_recall']:.3f} is not above the baseline"
    if candidate["false_alarm_rate"] > max_false_alarm_rate:
        return False, (f"false alarm rate {candidate['false_alarm_rate']:.3f} on borderline classes "
                       f"is above {max_false_alarm_rate}")
    if champion is not None and candidate["macro_f1"] < champion["macro_f1"]:
        return False, f"macro F1 {candidate['macro_f1']:.3f} is below the current @staging ({champion['macro_f1']:.3f})"
    return True, f"macro F1 {candidate['macro_f1']:.3f} beats the baseline by {gain:.3f}" + (
        f" and is >= the current @staging ({champion['macro_f1']:.3f})" if champion else " (no other model in @staging)"
    )


def score(model_uri: str, x: np.ndarray) -> np.ndarray:
    """Class probabilities (n, 5) from a registered model, through the same pyfunc path as the API."""
    model = mlflow.pyfunc.load_model(model_uri)
    return np.concatenate([np.asarray(model.predict(x[i:i + 1024])) for i in range(0, len(x), 1024)])


def window_metrics(probas: np.ndarray, y: np.ndarray) -> dict:
    return {k.removeprefix("window_"): v for k, v in metrics(y, probas.argmax(1), "window").items()}


def main():
    p = load_params("evaluate")
    info = json.loads(TRAIN_INFO.read_text())
    # str: the MLflow server returns versions as strings, a local file store as ints
    name, version, run_id = info["model_name"], str(info["model_version"]), info["run_id"]
    client = MlflowClient()

    w = load_windows()
    x, y, rides = w["x_test"].astype(np.float32), w["y_test"], w["rides_test"]

    # candidate: the version train just registered
    probas = score(f"models:/{name}/{version}", x)
    candidate = window_metrics(probas, y)
    y_ride, y_ride_pred = per_ride(probas, y, rides)
    ride = {k.removeprefix("ride_"): v for k, v in metrics(y_ride, y_ride_pred, "ride").items()}
    baseline = window_metrics(np.eye(len(CLASSES))[np.zeros_like(y)], y)  # always "Safe"

    # champion: the model in @staging now, re-scored on the same test set so the comparison is fair
    champion, champion_version = None, None
    try:
        champion_version = client.get_model_version_by_alias(name, p["alias"]).version
    except MlflowException:
        pass  # no @staging yet
    if champion_version is not None and champion_version != version:
        champion = window_metrics(score(f"models:/{name}@{p['alias']}", x), y)

    promote, reason = should_promote(candidate, baseline, champion, p["min_gain"], p["max_false_alarm_rate"])
    print(f"{name} v{version}: {'PROMOTED to @' + p['alias'] if promote else 'not promoted'} - {reason}")

    # log to the training run of this version, and tag the version in the registry
    cm = confusion_matrix(y, probas.argmax(1), labels=range(len(CLASSES)))
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(ax=ax, xticks_rotation=30, colorbar=False)
    ax.set_title(f"{name} v{version} - test windows")
    fig.tight_layout()
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics({
            **{f"eval_window_{k}": v for k, v in candidate.items()},
            **{f"eval_ride_{k}": v for k, v in ride.items()},
            "eval_promoted": float(promote),
        })
        mlflow.log_figure(fig, "eval/confusion_matrix.png")
    plt.close(fig)
    for key, value in {
        "eval_window_macro_f1": f"{candidate['macro_f1']:.4f}",
        "eval_window_reckless_recall": f"{candidate['reckless_recall']:.4f}",
        "eval_window_false_alarm_rate": f"{candidate['false_alarm_rate']:.4f}",
        "eval_gate": "passed" if promote else "failed",
        "eval_reason": reason,
    }.items():
        client.set_model_version_tag(name, version, key, value)
    if promote:
        client.set_registered_model_alias(name, p["alias"], version)

    # files for dvc metrics / dvc plots, committed to git: always LF, so their hash in dvc.lock
    # is the same on Windows and Mac/Linux (Python would write CRLF on Windows)
    EVAL_FILE.parent.mkdir(parents=True, exist_ok=True)
    EVAL_FILE.write_text(json.dumps({
        "model_version": int(version), "promoted": promote, "reason": reason,
        "window": candidate, "ride": ride, "baseline": baseline,
        "champion": {"version": int(champion_version), **champion} if champion else None,
    }, indent=2) + "\n", newline="\n")
    pd.DataFrame({"actual": np.array(CLASSES)[y], "predicted": np.array(CLASSES)[probas.argmax(1)]}).to_csv(
        PREDICTIONS_FILE, index=False, lineterminator="\n"
    )


if __name__ == "__main__":
    main()
