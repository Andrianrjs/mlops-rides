import os
from contextlib import asynccontextmanager

import mlflow
import numpy as np
from fastapi import FastAPI, HTTPException

from rides.api.schemas import Label, Prediction, Ride

LABELS: list[str] = list(Label.__args__)
MODEL_URI = os.getenv("MODEL_URI", "models:/rides-classifier@staging")
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        state["model"] = mlflow.pyfunc.load_model(MODEL_URI)
        state["version"] = state["model"].metadata.run_id
    except Exception as exc:  # pas de modèle : l'API démarre en mode dégradé
        state["error"] = str(exc)
    yield
    state.clear()


app = FastAPI(title="Rides classifier", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health():
    if "model" not in state:
        raise HTTPException(503, detail={"status": "degraded", "error": state.get("error")})
    return {"status": "ok", "model_version": state["version"]}


@app.post("/v1/predict", response_model=Prediction)
def predict(ride: Ride):
    if "model" not in state:
        raise HTTPException(503, detail="model not loaded")
    x = np.asarray(ride.readings, dtype=np.float32)[np.newaxis, ...]  # (1, T, C)
    probas = np.asarray(state["model"].predict(x))[0]
    return Prediction(
        label=LABELS[int(probas.argmax())],
        probabilities={lbl: float(p) for lbl, p in zip(LABELS, probas)},
        model_version=state["version"],
    )