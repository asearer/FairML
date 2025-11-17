# tests/test_bias.py

from src.fairness_tests.bias import introduce_bias
import pandas as pd


def test_introduce_bias_runs():
    df = pd.DataFrame({"f": [1, 2, 3]})
    biased = introduce_bias(df, "f")
    assert len(biased) == len(df)
