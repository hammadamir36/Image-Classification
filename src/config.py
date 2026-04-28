"""Configuration settings for the ML project."""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Model settings
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
TEST_SPLIT = 0.2
VAL_SPLIT = 0.1

# Model paths
BASELINE_MODEL_PATH = MODELS_DIR / "baseline_model.pt"
ADVANCED_MODEL_PATH = MODELS_DIR / "advanced_model.pt"

# Dataset settings
DATASET_NAME = "cifar100"  # Using CIFAR-100 as it's non-trivial
NUM_CLASSES = 100

# API settings
API_HOST = "0.0.0.0"
API_PORT = 5000
API_DEBUG = True
