"""Bonus monitoring: GET /metrics exposes what Prometheus scrapes and Grafana plots.

Skipped until src/rides/api/metrics.py (and the prometheus-client dependency) is merged.
"""

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("prometheus_client")
pytest.importorskip("rides.api.metrics")

import rides.api.main as api


@pytest.fixture
def client(tiny_model_uri, monkeypatch):
    monkeypatch.setattr(api, "MODEL_URI", tiny_model_uri)
    with TestClient(api.app) as c:
        yield c


def metric_value(text: str, line_prefix: str) -> float:
    for line in text.splitlines():
        if line.startswith(line_prefix):
            return float(line.rsplit(" ", 1)[1])
    return 0.0


def test_metrics_endpoint_is_prometheus_text(client):
    r = client.get("/metrics/")
    assert r.status_code == 200
    assert "api_requests_total" in r.text
    assert 'model_info{version="tiny-test-model"} 1.0' in r.text


def test_a_prediction_moves_every_layer(client, ride):
    before = client.get("/metrics/").text
    body = client.post("/v1/predict", json={"readings": ride}).json()
    after = client.get("/metrics/").text

    requests = 'api_requests_total{endpoint="/v1/predict",status="200"}'
    assert metric_value(after, requests) == metric_value(before, requests) + 1
    label = f'prediction_total{{label="{body["label"]}"}}'
    assert metric_value(after, label) == metric_value(before, label) + 1
    assert metric_value(after, "input_ride_steps_count") == metric_value(before, "input_ride_steps_count") + 1
    assert 'input_sensor_mean_count{sensor="az"}' in after


def test_metrics_scrapes_are_not_counted(client):
    client.get("/metrics/")
    assert 'endpoint="/metrics' not in client.get("/metrics/").text
