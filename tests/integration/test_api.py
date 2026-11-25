from fastapi.testclient import TestClient
from rides.api.main import app, state
from unittest.mock import MagicMock
import numpy as np

client = TestClient(app)

def test_health_degraded_when_no_model():
    state.clear()
    state["error"] = "No model found"
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["detail"]["status"] == "degraded"

def test_health_ok_when_model_loaded():
    state.clear()
    state["model"] = MagicMock()
    state["version"] = "run-12345"
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_version": "run-12345"}

def test_predict_success():
    mock_model = MagicMock()
    # 5 classes: Safe, Safe Overall, Bad Weather, Tandem, Reckless
    mock_model.predict.return_value = np.array([[0.85, 0.05, 0.04, 0.04, 0.02]])
    state["model"] = mock_model
    state["version"] = "run-12345"

    sample_readings = [[0.01, -0.02, 9.81, 0.0, 0.0, 0.0]] * 10
    response = client.post("/v1/predict", json={"readings": sample_readings})
    
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "Safe"
    assert "probabilities" in data
    assert data["probabilities"]["Safe"] == 0.85
    assert data["model_version"] == "run-12345"

def test_predict_too_short():
    # min_length is 10, so 5 should fail validation
    sample_readings = [[0.01, -0.02, 9.81, 0.0, 0.0, 0.0]] * 5
    response = client.post("/v1/predict", json={"readings": sample_readings})
    assert response.status_code == 422
