"""
example_model.py
----------------
Example ML model for FairML.
"""

import joblib
from sklearn.ensemble import RandomForestClassifier
from config import config
from logger import get_logger
from data.synthetic_data import generate_classification_data

logger = get_logger(__name__)

def train_example_model() -> RandomForestClassifier:
    """
    Train a simple RandomForest classifier on synthetic data.

    Returns:
        RandomForestClassifier: The trained model.
    """
    logger.info("Starting model training...")
    try:
        X_train, X_test, y_train, y_test = generate_classification_data()

        model = RandomForestClassifier(random_state=config.RANDOM_SEED)
        model.fit(X_train, y_train)
        logger.info("Model training completed")

        joblib.dump(model, config.MODEL_DIR / "example_model.pkl")
        logger.info(f"Model saved to {config.MODEL_DIR / 'example_model.pkl'}")

        return model

    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise

def load_model() -> RandomForestClassifier:
    """
    Load the trained example model.
    
    Returns:
        RandomForestClassifier: Loaded model instance.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    model_path = config.MODEL_DIR / "example_model.pkl"
    if not model_path.exists():
        logger.error(f"Model file not found at {model_path}")
        raise FileNotFoundError(f"Model not found at {model_path}. Please run train_example_model() first.")
    
    try:
        model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
