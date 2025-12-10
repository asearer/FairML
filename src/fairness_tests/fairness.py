"""
fairness.py
-----------
Module for evaluating model fairness metrics.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from logger import get_logger

logger = get_logger(__name__)

def demographic_parity_difference(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    protected_feature: str
) -> float:
    """
    Calculate the difference in positive prediction rates across groups (Demographic Parity).

    Args:
        model (BaseEstimator): Trained model.
        X (pd.DataFrame): features.
        y (pd.Series): True labels (unused in calculation but kept for API consistency).
        protected_feature (str): Column name for protected group.

    Returns:
        float: The absolute difference between the max and min positive prediction rates.
        Returns 0.0 if only one group exists.

    Raises:
        ValueError: If protected_feature is missing from X.
    """
    if protected_feature not in X.columns:
        logger.error(f"Protected feature '{protected_feature}' not found in X.")
        raise ValueError(f"Protected feature '{protected_feature}' not found in X columns.")

    groups = X[protected_feature].unique()
    rates = []
    
    logger.debug(f"Calculating demographic parity for groups: {groups}")

    for group in groups:
        idx = X[protected_feature] == group
        if idx.sum() == 0:
            continue
            
        # Predict on the subset
        try:
            subset = X[idx]
            preds = model.predict(subset)
            # Check for binary output. If proba, this needs thresholding. 
            # Assuming predict returns class labels 0/1.
            rate = (preds == 1).mean()
            rates.append(rate)
        except Exception as e:
            logger.error(f"Prediction failed for group {group}: {e}")
            raise

    if not rates:
        logger.warning("No valid groups found for parity calculation.")
        return 0.0

    diff = abs(max(rates) - min(rates))
    logger.info(f"Demographic Parity Difference: {diff}")
    return float(diff)
