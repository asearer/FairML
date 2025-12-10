"""
bias.py
-------
Module for synthetic bias generation and testing.
"""

import pandas as pd
import numpy as np
from logger import get_logger

logger = get_logger(__name__)

def introduce_bias(
    X: pd.DataFrame,
    protected_feature: str,
    bias_level: float = 0.1
) -> pd.DataFrame:
    """
    Introduce synthetic bias by flipping a fraction of labels in one group.

    Args:
        X (pd.DataFrame): Features including target.
        protected_feature (str): Column name for protected attribute.
        bias_level (float): Fraction of labels to flip (0.0 to 1.0).

    Returns:
        pd.DataFrame: Biased dataset.

    Raises:
        ValueError: If protected_feature is not in columns or bias_level is invalid.
    """
    if protected_feature not in X.columns:
        logger.error(f"Protected feature '{protected_feature}' not found in DataFrame.")
        raise ValueError(f"Protected feature '{protected_feature}' not found in DataFrame columns: {X.columns.tolist()}")

    if not 0 <= bias_level <= 1:
        logger.error(f"Invalid bias_level: {bias_level}")
        raise ValueError("bias_level must be between 0.0 and 1.0")

    X_biased = X.copy()
    
    # Identify the groups
    groups = X_biased[protected_feature].unique()
    if len(groups) == 0:
        logger.warning("No groups found in protected feature.")
        return X_biased

    # simplistic choice: pick the first group found to introduce bias against
    # In a real scenario, this might be parameterized
    target_group = groups[0]
    idx = X_biased[protected_feature] == target_group
    
    group_size = idx.sum()
    n_flip = int(bias_level * group_size)
    
    logger.info(f"Introducing bias against group '{target_group}': flipping {n_flip} labels ({bias_level*100}%)")

    if n_flip > 0:
        flip_indices = np.random.choice(
            X_biased[idx].index,
            n_flip,
            replace=False
        )
        
        # Safe flip for 0/1 targets
        X_biased.loc[flip_indices, "target"] = 1 - X_biased.loc[flip_indices, "target"]

    return X_biased
