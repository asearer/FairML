"""
test_models.py
--------------
Tests for model training and loading in FairML.
"""

from models.example_model import train_example_model, load_model


def test_model_training_loading():
    """Test that the example model can be trained and loaded."""
    train_example_model()
    model = load_model()
    assert model is not None
