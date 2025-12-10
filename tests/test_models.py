# tests/test_models.py

import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
import joblib
from models.example_model import train_example_model, load_model
from config import config

def test_training_pipeline(tmp_path):
    """Test full training pipeline with temporary config override."""
    # Override paths for testing to avoid writing to actual source dir
    original_model_dir = config.MODEL_DIR
    original_data_dir = config.DATA_DIR
    
    config.MODEL_DIR = tmp_path / "models"
    config.DATA_DIR = tmp_path / "data"
    
    config.MODEL_DIR.mkdir()
    config.DATA_DIR.mkdir()
    
    try:
        model = train_example_model()
        assert model is not None
        assert (config.MODEL_DIR / "example_model.pkl").exists()
        assert (config.DATA_DIR / "synthetic_data.csv").exists()
        
        loaded_model = load_model()
        assert loaded_model is not None
        
    finally:
        # Restore paths
        config.MODEL_DIR = original_model_dir
        config.DATA_DIR = original_data_dir

def test_load_model_not_found(tmp_path):
    """Test loading a non-existent model raises correct error."""
    # Point to empty temp dir
    original_model_dir = config.MODEL_DIR
    config.MODEL_DIR = tmp_path
    
    try:
        with pytest.raises(FileNotFoundError):
            load_model()
    finally:
        config.MODEL_DIR = original_model_dir
