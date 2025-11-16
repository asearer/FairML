"""
test_models.py
--------------
Tests for model training and loading in FairML.
"""

import os
import sys

# Ensure src is on the Python path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
)

from models.example_model import train_example_model, load_model


def test_model_training_loading():
    """Test that the example model can be trained and loaded."""
    # Train the example model
    train_example_model()

    # Load the model
    model = load_model("""
test_models.py
--------------
Tests for model training and loading in FairML.
"""

import pytest

from models.example_model import train_example_model, load_model


def test_model_training_loading():
    """Test that the example model can be trained and loaded."""
    # Train the example model
    train_example_model()

    # Load the model
    model = load_model()

    # Assert that the model was loaded successfully
    assert model is not None
)

    # Assert that the model was loaded successfully
    assert model is not None
