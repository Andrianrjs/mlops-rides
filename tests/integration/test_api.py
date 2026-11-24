"""API + a real model in the MLflow format, loaded through the lifespan exactly as in production."""

import pytest
from fastapi.testclient import TestClient

import rides.api.main as api


@pytest.fixture
def client(tiny_model_uri, monkeypatch):
    monkeypatch.setattr(api, "MODEL_URI", tiny_model_uri)
    with TestClient(api.app, raise_server_exceptions=False) as c:  # runs the lifespan
        yield c


@pytest.fixture
def client_without_model(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "MODEL_URI", str(tmp_path / "missing-model"))
    with TestClient(api.app, raise_server_exceptions=False) as c:
        yield c


def test_health_reports_the_loaded_model(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_version": "tiny-test-model"}


@pytest.mark.parametrize("steps", [10, 100, 3000])  # one window, or a whole ride (T >= 10)
def test_predict_returns_a_class_and_5_probabilities(client, ride, steps):
    readings = (ride * (steps // len(ride) + 1))[:steps]
    r = client.post("/v1/predict", json={"readings": readings})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] in api.LABELS
    assert list(body["probabilities"]) == api.LABELS
    assert sum(body["probabilities"].values()) == pytest.approx(1, abs=1e-4)
    assert body["label"] == max(body["probabilities"], key=body["probabilities"].get)
    assert body["model_version"] == "tiny-test-model"


def test_same_ride_same_answer(client, ride):
    first = client.post("/v1/predict", json={"readings": ride}).json()
    assert client.post("/v1/predict", json={"readings": ride}).json() == first


@pytest.mark.parametrize(
    "payload",
    [
        {"readings": [[0.0] * 6] * 9},  # shorter than 10 steps
        {"readings": []},
        {},  # missing field
        {"readings": [[0.0] * 6] * 10, "extra": 1},  # unknown field (extra="forbid")
        {"readings": [["a"] * 6] * 10},  # not numbers
    ],
)
def test_invalid_payload_is_a_422(client, payload):
    assert client.post("/v1/predict", json=payload).status_code == 422


@pytest.mark.xfail(strict=True, reason="schemas.Ride does not check 6 values per step yet: the model fails -> 500")
@pytest.mark.parametrize(
    "readings",
    [
        [[0.0] * 3] * 10,  # 3 sensors instead of 6
        [[0.0] * 6, [0.0] * 5] * 5,  # ragged rows
    ],
)
def test_wrong_number_of_sensors_is_a_422(client, readings):
    assert client.post("/v1/predict", json={"readings": readings}).status_code == 422


def test_without_model_the_api_starts_degraded(client_without_model, ride):
    health = client_without_model.get("/health")
    assert health.status_code == 503
    assert health.json()["detail"]["status"] == "degraded"
    assert client_without_model.post("/v1/predict", json={"readings": ride}).status_code == 503


def test_swagger_documents_the_predict_route(client):
    schema = client.get("/openapi.json").json()
    assert "/v1/predict" in schema["paths"]
    assert "/health" in schema["paths"]
