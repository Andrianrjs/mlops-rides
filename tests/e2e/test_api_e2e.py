"""End-to-end: the API running in its Docker image (or on Kubernetes) answers over HTTP.

Needs only pytest and httpx. Skipped unless E2E_BASE_URL is set, e.g.:

    E2E_BASE_URL=http://localhost:8001 uv run pytest tests/e2e
"""

import os
import time

import httpx
import pytest

BASE_URL = os.getenv("E2E_BASE_URL")
LABELS = {"Safe", "Safe overall", "Bad weather", "Tandem", "Reckless"}  # same names as rides.api.schemas.Label

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(not BASE_URL, reason="E2E_BASE_URL not set: no running API to test"),
]


@pytest.fixture(scope="module")
def http() -> httpx.Client:
    with httpx.Client(base_url=BASE_URL, timeout=30) as client:
        deadline = time.monotonic() + float(os.getenv("E2E_WAIT_SECONDS", "120"))
        while True:  # the container loads torch and the model before /health answers 200
            try:
                if client.get("/health").status_code == 200:
                    break
            except httpx.TransportError:
                pass
            if time.monotonic() > deadline:
                pytest.fail(f"{BASE_URL}/health never answered 200")
            time.sleep(2)
        yield client


def test_health(http):
    body = http.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_version"]


def test_predict_one_window(http):
    readings = [[0.05 * (i % 7), -0.02, 9.81, 0.01, 0.0, -0.01] for i in range(100)]
    r = http.post("/v1/predict", json={"readings": readings})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] in LABELS
    assert set(body["probabilities"]) == LABELS
    assert sum(body["probabilities"].values()) == pytest.approx(1, abs=1e-4)


def test_bad_request_is_rejected(http):
    assert http.post("/v1/predict", json={"readings": [[0.0] * 6] * 5}).status_code == 422


def test_swagger_is_served(http):
    assert http.get("/docs").status_code == 200
