# tests/test_bias.py

# flake8: noqa: E402
import os
import sys

# Add src to Python path for imports
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")),
)

import pandas as pd
from fairness_tests.bias import introduce_bias


def test_introduce_bias():
    df = pd.DataFrame(
        {
            "feature": [0, 1, 0, 1],
            "target": [0, 1, 0, 1],
        }
    )

    biased = introduce_bias(df, "feature", bias_level=0.5)


    # Ensure same row count
    assert len(biased) == len(df)

    # Ensure target changed for at least some rows
    assert (biased["target"] != df["target"]).any()
