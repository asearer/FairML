"""
logger.py
---------
Centralized logging configuration for FairML.
"""

import logging
import sys
from pathlib import Path
from config import config

def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger instance.

    Args:
        name (str): The name of the logger (usually __name__).

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(config.LOG_LEVEL)

    # Check if handlers are already added to avoid duplication
    if not logger.handlers:
        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(config.LOG_LEVEL)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # File Handler
        log_file = config.LOG_DIR / "fairml.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(config.LOG_LEVEL)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger
