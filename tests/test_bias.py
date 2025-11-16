"""
test_bias.py
------------
Tests for synthetic bias functions in FairML.
"""

import pandas as pd
from fairness_tests.bias import introduce_bias


def test_bias_shape():
    """Test that introducing bias does not change the DataFrame shape."""
    data = pd.DataFrame({
        "feature_0": [0, 1, 0, 1],
        "target": [0, 1, 0, 1]
    })

    biased = introduce_bias(data, "feature_0", 0.5)
    assert biased.shape == data.shape
