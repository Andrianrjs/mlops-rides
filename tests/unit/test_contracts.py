"""Contracts between team parts (README, "Contracts between parts"): a change here breaks someone else."""

from rides.api.main import LABELS
from rides.config import load_params
from rides.data import CLASSES, SENSORS


def test_sensor_columns_and_order():
    assert SENSORS == ["ax", "ay", "az", "rx", "ry", "rz"]


def test_window_length_is_100_steps():
    assert load_params("prepare")["window"] == 100


def test_api_labels_follow_the_model_output_order():
    """The API names probabilities[i] with LABELS[i]: same classes, same order as the model."""
    assert [label.lower() for label in LABELS] == [c.lower() for c in CLASSES]


def test_model_name_and_alias_served_by_the_api():
    evaluate = load_params("evaluate")
    assert (evaluate["model_name"], evaluate["alias"]) == ("rides-classifier", "staging")
