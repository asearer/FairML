"""
helpers.py
----------
Utility functions for FairML.
"""

import pandas as pd
from logger import get_logger

logger = get_logger(__name__)

def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize numeric columns to [0,1] range.

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Normalized dataframe.
        
    Raises:
        ValueError: If input is not a pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        logger.error(f"Expected pandas DataFrame, got {type(df)}")
        raise ValueError("Input must be a pandas DataFrame")

    if df.empty:
        logger.warning("Input DataFrame is empty. returning empty DataFrame.")
        return df

    numeric_cols = df.select_dtypes(include=['number']).columns
    if numeric_cols.empty:
        logger.warning("No numeric columns found to normalize.")
        return df

    logger.debug(f"Normalizing columns: {numeric_cols.tolist()}")
    
    result = df.copy()
    
    # Avoid division by zero if max == min
    for col in numeric_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        if max_val - min_val == 0:
             result[col] = 0.0
        else:
             result[col] = (df[col] - min_val) / (max_val - min_val)
             
    return result
