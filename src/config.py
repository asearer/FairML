"""
config.py
---------
Centralized configuration management for the FairML project.
"""

import os
from pathlib import Path
from dataclasses import dataclass

@dataclass
class Config:
    """
    Configuration settings for FairML.
    
    Attributes:
        BASE_DIR (Path): Absolute path to the project root directory.
        DATA_DIR (Path): Directory for storing data files.
        MODEL_DIR (Path): Directory for storing trained models.
        LOG_DIR (Path): Directory for storing log files.
        RANDOM_SEED (int): Global random seed for reproducibility.
        LOG_LEVEL (str): Logging level (e.g., 'INFO', 'DEBUG').
    """
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "src" / "data"
    MODEL_DIR: Path = BASE_DIR / "src" / "models"
    LOG_DIR: Path = BASE_DIR / "logs"
    RANDOM_SEED: int = 42
    LOG_LEVEL: str = "INFO"

    def __post_init__(self):
        """Ensure necessary directories exist."""
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)

# Global configuration instance
config = Config()
