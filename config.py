"""
Configuration file for the Smart Crop Health Monitoring System
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATASET_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
TRAINING_DIR = PROJECT_ROOT / "training"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for dir_path in [MODELS_DIR, TRAINING_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Model configuration - environment specific
ENVIRONMENT = os.getenv("TRAINING_ENVIRONMENT", "laptop")  # laptop or colab

if ENVIRONMENT == "colab":
    YOLO_MODEL_SIZE = "yolov8m.pt"  # Medium model for Colab
    BATCH_SIZE = 32
    EPOCHS = 125
    WORKERS = 8
else:
    YOLO_MODEL_SIZE = "yolov8m.pt"  # Medium model for laptop
    BATCH_SIZE = 8  # Reduced for medium model
    EPOCHS = 10
    WORKERS = 4

IMAGE_SIZE = 640
DEVICE = "cuda" if os.getenv("CUDA_AVAILABLE", "false").lower() == "true" else "cpu"

# Data paths
DATA_YAML_PATH = DATASET_DIR / "data.yaml"

# API Keys
PLANTNET_API_KEY = os.getenv("PLANTNET_API_KEY", "2b10XDHQvxhONByCDVW2nsXsu")
SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")

# Health scoring configuration
CONFIDENCE_THRESHOLD = 0.25
DISEASE_IMPACT_WEIGHTS = {
    "high_severity": 1.5,
    "medium_severity": 1.2,
    "low_severity": 1.0
}

# Environmental thresholds
OPTIMAL_RANGES = {
    "temperature": (18, 30),
    "humidity": (40, 70),
    "soil_moisture": (30, 70),
    "ph": (6.0, 7.5)
}

# API configuration
API_HOST = "0.0.0.0"
API_PORT = 8000

# PlantNet API configuration (v2)
PLANTNET_BASE_URL = "https://my-api.plantnet.org/v2"
PLANTNET_PROJECT = "all"  # Changed from "weurope" to "all"
PLANTNET_ORGANS = ["leaf", "flower", "fruit", "bark"]

# Alternative PlantNet configurations for v2
PLANTNET_BACKUP_PROJECTS = [
    "all",              # All available flora (best coverage)
    "weurope",
    "useful",
    "k-world-flora",
    "the-plant-list"
]

# Training optimization based on environment
if ENVIRONMENT == "colab":
    TRAINING_CONFIG = {
        "patience": 50,
        "save_period": 10,
        "amp": True,
        "cache": True,
        "rect": False,
        "cos_lr": True,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "warmup_epochs": 3
    }
else:
    TRAINING_CONFIG = {
        "patience": 25,
        "save_period": 5,
        "amp": False,
        "cache": True,  # Reduce memory usage on laptop
        "rect": True,   # Rectangular training for speed
        "cos_lr": False,
        "optimizer": "SGD",
        "lr0": 0.01,
        "warmup_epochs": 2
    }
