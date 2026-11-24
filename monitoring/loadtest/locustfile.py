"""Load test of the API: a baseline run to calibrate the alert thresholds (latency, error rate).

    uvx locust -f monitoring/loadtest/locustfile.py --host http://localhost:8000 \
        --headless -u 20 -r 5 -t 2m --csv reports/load

Watch the Grafana dashboard while it runs; reports/load_stats.csv holds the p50 / p95 / p99.
"""

import random

from locust import HttpUser, between, task

GRAVITY = 9.81


def ride(steps: int) -> list[list[float]]:
    """Plausible raw IMU readings: small accelerations, gravity on az, small rotations."""
    return [
        [
            random.gauss(0, 0.5),
            random.gauss(0, 0.5),
            random.gauss(GRAVITY, 0.8),
            random.gauss(0, 0.1),
            random.gauss(0, 0.1),
            random.gauss(0, 0.1),
        ]
        for _ in range(steps)
    ]


class RideUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(10)
    def predict_window(self):  # the usual call: one 10 s window
        self.client.post("/v1/predict", json={"readings": ride(100)}, name="/v1/predict [T=100]")

    @task(2)
    def predict_whole_ride(self):  # a whole ride, about 3 min
        self.client.post("/v1/predict", json={"readings": ride(2000)}, name="/v1/predict [T=2000]")

    @task(1)
    def health(self):
        self.client.get("/health")
