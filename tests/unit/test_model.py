"""The model honours the contract the API relies on: raw (batch, T, C) in, (batch, 5) probabilities out."""

import numpy as np
import pytest
import torch

from rides.data import CLASSES, SENSORS
from rides.model import RidesClassifier


@pytest.fixture(scope="module")
def model() -> RidesClassifier:
    torch.manual_seed(0)
    mean = np.array([0, 0, 9.81, 0, 0, 0])
    std = np.array([1, 1, 2, 0.5, 0.5, 0.5])
    return RidesClassifier(mean, std, channels=8).eval()


@pytest.mark.parametrize("batch, steps", [(1, 10), (1, 100), (4, 100), (2, 3000)])
@torch.no_grad()
def test_output_is_a_probability_per_class(model, batch, steps):
    x = torch.randn(batch, steps, len(SENSORS))
    probas = model(x)
    assert probas.shape == (batch, len(CLASSES))
    assert (probas >= 0).all()
    torch.testing.assert_close(probas.sum(dim=-1), torch.ones(batch))


@torch.no_grad()
def test_normalisation_lives_inside_the_model(model):
    """forward(raw) must equal softmax(logits(normalised)): the API sends raw readings."""
    raw = torch.randn(3, 100, len(SENSORS)) * 2 + torch.tensor([0, 0, 9.81, 0, 0, 0])
    normalised = (raw - model.mean) / model.std
    torch.testing.assert_close(model(raw), torch.softmax(model.logits(normalised), dim=-1))


def test_scaler_stats_are_saved_with_the_weights(model):
    state = model.state_dict()
    assert {"mean", "std"} <= state.keys()
    assert state["mean"].shape == (len(SENSORS),)
