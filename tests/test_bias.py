import os
import sys
import pandas as pd

# Ensure src is on the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Now imports will work
from fairness_tests.bias import introduce_bias

def test_bias_shape():
    # Create a small test DataFrame
    data = pd.DataFrame({
        'feature_0': [0, 1, 0, 1],
        'target': [0, 1, 0, 1]
    })

    # Apply bias
    biased = introduce_bias(data, 'feature_0', 0.5)

    # Assert that the shape is unchanged
    assert biased.shape == data.shape
