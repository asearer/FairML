# tests/test_models.py

# flake8: noqa: E402
import os
import sys

# Add src to Python path
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")),
)

from models.example_model import train_example_model, load_model


def test_model_training_and_loading():
    """Train and load the model, ensuring both steps succeed."""
    train_example_model()
    model = load_model()
    assert model is not None
