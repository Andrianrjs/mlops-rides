"""Shared fixtures. Heavy imports (torch, mlflow) stay inside fixtures, so the e2e tests,
which only need pytest and httpx, can run without the project's dependencies."""

import pytest


@pytest.fixture(scope="session")
def tiny_model_uri(tmp_path_factory) -> str:
    """Path of an untrained model in the same MLflow format as the registered one."""
    from tiny_model import save_tiny_model  # tests/ is on sys.path because this conftest lives there

    return str(save_tiny_model(tmp_path_factory.mktemp("mlflow") / "model"))


@pytest.fixture
def ride() -> list[list[float]]:
    """100 steps x 6 sensors (ax, ay, az, rx, ry, rz): a plausible slow ride, gravity on az."""
    return [[0.05 * (i % 7), -0.02, 9.81, 0.01, 0.0, -0.01] for i in range(100)]
