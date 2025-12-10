"""
synthetic_data.py
-----------------
Module for generating synthetic datasets for FairML examples.
"""

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from typing import Tuple
from config import config
from logger import get_logger

logger = get_logger(__name__)

def generate_classification_data(
    n_samples: int = 1000,
    n_features: int = 10,
    n_informative: int = 8,
    random_state: int = config.RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Generate a synthetic classification dataset.

    Args:
        n_samples (int): Number of samples.
        n_features (int): Number of features.
        n_informative (int): Number of informative features.
        random_state (int): Random seed for reproducibility.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
            X_train, X_test, y_train, y_test
    """
    logger.info(f"Generating synthetic data: samples={n_samples}, features={n_features}")
    
    try:
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            random_state=random_state
        )

        df = pd.DataFrame(
            X, columns=[f"feature_{i}" for i in range(X.shape[1])]
        )
        df["target"] = y

        # Save the full dataset
        output_path = config.DATA_DIR / "synthetic_data.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Saved synthetic data to {output_path}")

        X_train, X_test, y_train, y_test = train_test_split(
            df.drop("target", axis=1),
            df["target"],
            test_size=0.2,
            random_state=random_state
        )
        
        logger.info("Data split completed")
        return X_train, X_test, y_train, y_test

    except Exception as e:
        logger.error(f"Failed to generate synthetic data: {e}")
        raise
