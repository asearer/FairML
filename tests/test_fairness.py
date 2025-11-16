"""
test_fairness.py
----------------
Tests for fairness metrics in FairML.
"""

import os
import sys
import pandas as pd

# Ensure src is on the Python path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
)

from fairness_tests.fairness import demographic_parity_difference
from models.example_model import train_example_model, load_model


def test_dp_difference_range():
    """Test that demographic parity difference is between 0 and 1."""
    # Train and load the model
    train_example_model()
    model = load_model()

    # Build path to CSV file relative to this test file
    data_path = os.path.join(
        os.path.dirname(__file__), "../src/data/synthetic_data.csv"
    )
    data = pd.read_csv(data_path)

    # Compute demographic parity difference
    dp = demographic_parity_difference(
        model,
        data.drop("target", axis=1),
        data["target"],
        "feature_0"
    )

    # Check that dp is between 0 and 1
    assert 0 <= dp <= 1
