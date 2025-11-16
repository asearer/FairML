"""
fairness.py
-----------
Module for evaluating model fairness metrics.
"""

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score

def demographic_parity_difference(model: BaseEstimator, X: pd.DataFrame, y: pd.Series, protected_feature: str) -> float:
    """
    Calculate difference in positive prediction rates across groups.

    Args:
        model (BaseEstimator): Trained model
        X (pd.DataFrame): Features
        y (pd.Series): Labels (unused)
        protected_feature (str): Column name for protected group

    Returns:
        float: Absolute difference in positive prediction rates
    """
    groups = X[protected_feature].unique()
    rates = []
    for g in groups:
        idx = X[protected_feature] == g
        preds = model.predict(X[idx])
        rate = (preds == 1).mean()
        rates.append(rate)
    return abs(max(rates) - min(rates))
