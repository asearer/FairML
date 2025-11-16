"""
example_model.py
----------------
Example ML model for FairML.
"""

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "example_model.pkl"

def train_example_model():
    """
    Train a simple RandomForest classifier on synthetic data.
    """
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=8, random_state=42)
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    df['target'] = y
    X_train, X_test, y_train, y_test = train_test_split(df.drop('target', axis=1), df['target'], test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)
    df.to_csv(Path(__file__).parent.parent / "data" / "synthetic_data.csv", index=False)
    return model

def load_model():
    """Load the trained example model."""
    return joblib.load(MODEL_PATH)
