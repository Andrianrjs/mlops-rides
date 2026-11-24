"""Prometheus metrics of the API, one family per monitoring layer (exposed on GET /metrics)."""

import time

import numpy as np
from fastapi import FastAPI, Request
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app

SENSORS = ["ax", "ay", "az", "rx", "ry", "rz"]  # contract order (README)

# Layer 1: infrastructure
REQUESTS = Counter("api_requests_total", "HTTP requests", ["endpoint", "status"])
LATENCY = Histogram(
    "api_request_duration_seconds",
    "Request latency",
    ["endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)

# Layer 2: input distribution (a unit bug, e.g. g instead of m/s2, shows up on az first)
INPUT_STEPS = Histogram(
    "input_ride_steps", "Number of time steps per request", buckets=(10, 50, 100, 200, 500, 1000, 5000, 20000)
)
INPUT_SENSOR_MEAN = Histogram(
    "input_sensor_mean",
    "Mean value of each sensor over a request",
    ["sensor"],
    buckets=(-20, -10, -5, -2, -1, -0.5, 0, 0.5, 1, 2, 5, 10, 20),
)

# Layer 3: prediction distribution
PREDICTIONS = Counter("prediction_total", "Predicted class", ["label"])
RECKLESS_PROBA = Histogram(
    "prediction_reckless_probability",
    "Predicted probability of the Reckless class",
    buckets=(0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0),
)

MODEL_INFO = Gauge("model_info", "Model version being served (value 1)", ["version"])


def instrument(app: FastAPI) -> None:
    """Expose /metrics and count every request (except /metrics itself) with its latency."""
    app.mount("/metrics", make_asgi_app())

    @app.middleware("http")
    async def observe(request: Request, call_next):
        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            path = request.url.path
            if not path.startswith("/metrics"):
                REQUESTS.labels(endpoint=path, status=str(status)).inc()
                LATENCY.labels(endpoint=path).observe(time.perf_counter() - start)


def record_prediction(x: np.ndarray, label: str, probabilities: dict[str, float]) -> None:
    """x: the (1, T, C) input sent to the model."""
    INPUT_STEPS.observe(x.shape[1])
    for sensor, mean in zip(SENSORS, x[0].mean(axis=0)):
        INPUT_SENSOR_MEAN.labels(sensor=sensor).observe(float(mean))
    PREDICTIONS.labels(label=label).inc()
    RECKLESS_PROBA.observe(probabilities.get("Reckless", 0.0))
