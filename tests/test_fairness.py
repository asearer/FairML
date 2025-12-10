# tests/test_fairness.py

import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from fairness_tests.fairness import demographic_parity_difference

def test_dp_difference_calculation():
    # Create simple synthetic data
    X = pd.DataFrame({
        "group": ["A", "A", "B", "B"],
        "val": [1, 2, 3, 4]
    })
    y = pd.Series([0, 1, 0, 1])
    
    # Model that always predicts 1
    model = DummyClassifier(strategy="constant", constant=1)
    model.fit(X, y)
    
    # Both groups get 100% positive rate, diff should be 0
    diff = demographic_parity_difference(model, X, y, "group")
    assert diff == 0.0

def test_dp_difference_disparity():
    X = pd.DataFrame({
        "group": ["A", "A", "B", "B"],
        "val": [1, 1, 1, 1]
    })
    y = pd.Series([0, 0, 0, 0])
    
    # Mock model
    class MockModel:
        def predict(self, X):
            # Predict 1 for group A, 0 for group B
            return (X["group"] == "A").astype(int)
            
    model = MockModel()
    
    diff = demographic_parity_difference(model, X, y, "group")
    assert diff == 1.0 # 1.0 - 0.0

def test_missing_column():
    X = pd.DataFrame({"col1": [1]})
    y = pd.Series([1])
    model = DummyClassifier()
    
    with pytest.raises(ValueError):
        demographic_parity_difference(model, X, y, "missing_col")
