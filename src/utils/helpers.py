"""
helpers.py
----------
Utility functions for FairML
"""

import pandas as pd

def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize numeric columns to [0,1]."""
    return (df - df.min()) / (df.max() - df.min())
