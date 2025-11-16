"""
bias.py
-------
Module for synthetic bias generation and testing.
"""

import pandas as pd
import numpy as np

def introduce_bias(X: pd.DataFrame, protected_feature: str, bias_level: float = 0.1) -> pd.DataFrame:
    """
    Introduce synthetic bias by flipping a fraction of labels in one group.

    Args:
        X (pd.DataFrame): Features including target
        protected_feature (str): Column name for protected attribute
        bias_level (float): Fraction of labels to flip

    Returns:
        pd.DataFrame: Biased dataset
    """
    X_biased = X.copy()
    group = X_biased[protected_feature].unique()[0]
    idx = X_biased[protected_feature] == group
    n_flip = int(bias_level * idx.sum())
    flip_indices = np.random.choice(X_biased[idx].index, n_flip, replace=False)
    X_biased.loc[flip_indices, 'target'] = 1 - X_biased.loc[flip_indices, 'target']
    return X_biased
