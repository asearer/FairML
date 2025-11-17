# tests/test_bias.py

import os
import sys
import pandas as pd

# Ensure src is on the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from fairness_tests.bias import introduce_bias


def test_introduce_bias():
    df = pd.DataFrame({
        "feature": [0, 1, 0, 1],
        "target": [0, 1, 0, 1],
    })

    biased = introduce_bias(df, "feature", bias_amount=0.2)

    # Ensure same row count
    assert len(biased) == len(df)

    # Ensure target changed for at least some rows
    assert (biased["target"] != df["target"]).any()
