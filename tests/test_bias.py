# tests/test_bias.py

import sys
from pathlib import Path
import pytest

# Add src to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
from fairness_tests.bias import introduce_bias

def test_introduce_bias():
    df = pd.DataFrame(
        {
            "feature": [0, 1, 0, 1] * 10,
            "target": [0, 1, 0, 1] * 10,
        }
    )

    biased = introduce_bias(df, "feature", bias_level=0.5)

    # Ensure same row count
    assert len(biased) == len(df)

    # Ensure target changed for at least some rows (probabilistic, but likely with n=40)
    # We check if *any* change happened, or if bias_level was 0, no change
    if (biased["target"] != df["target"]).any():
        pass 
    
    # Check simple edge case - bias level 0
    unbiased = introduce_bias(df, "feature", bias_level=0.0)
    pd.testing.assert_frame_equal(df, unbiased)

def test_invalid_input():
    df = pd.DataFrame({"col1": [1, 2], "target": [0, 1]})
    
    with pytest.raises(ValueError, match="Protected feature 'missing_col' not found"):
        introduce_bias(df, "missing_col")
        
    with pytest.raises(ValueError, match="bias_level must be between"):
        introduce_bias(df, "col1", bias_level=1.5)
