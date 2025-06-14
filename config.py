"""
Configuration file for the Smart Crop Health Monitoring System
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATASET_DIR = PROJECT_ROOT / "dataset" / "dataset"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
TRAINING_DIR = PROJECT_ROOT / "training"
UTILS_DIR = PROJECT_ROOT / "utils"
RASPBERRY_API_DIR = PROJECT_ROOT / "raspberry_api"
ADVICE_DIR = PROJECT_ROOT / "advice"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, TRAINING_DIR, UTILS_DIR, RASPBERRY_API_DIR, ADVICE_DIR]:
    dir_path.mkdir(exist_ok=True)

# Model configuration
YOLO_MODEL_SIZE = "yolov8s.pt"  # Options: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
IMAGE_SIZE = 640
BATCH_SIZE = 8
EPOCHS = 30
DEVICE = "cuda" if os.getenv("CUDA_AVAILABLE", "false").lower() == "true" else "cpu"

# API Keys (to be set via environment variables)
PLANTNET_API_KEY = os.getenv("PLANTNET_API_KEY", "")
PLANTNET_PROJECT = "k-world-flora"  # or other available projects

# PlantNet API configuration
PLANTNET_BASE_URL = "https://my-api.plantnet.org/v1"
PLANTNET_ORGANS = ["leaf", "flower", "fruit", "bark"]  # Image organs to consider

# Health scoring weights
HEALTH_WEIGHTS = {
    "disease_score": 0.4,      # Disease detection impact
    "pest_score": 0.3,         # Pest detection impact
    "environmental": 0.3       # Environmental factors (temp, humidity, soil)
}

# Environmental thresholds (example ranges - can be plant-specific)
OPTIMAL_RANGES = {
    "temperature": (18, 30),    # Celsius
    "humidity": (40, 70),       # Percentage
    "soil_moisture": (30, 70),  # Percentage
    "ph": (6.0, 7.5)           # pH scale
}

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FILE = PROJECT_ROOT / "logs" / "crop_health.log"
LOG_FILE.parent.mkdir(exist_ok=True)

# FastAPI configuration
API_HOST = "0.0.0.0"
API_PORT = 8000

# Raspberry Pi configuration
CAMERA_RESOLUTION = (1024, 768)
SENSOR_READ_INTERVAL = 300  # seconds (5 minutes)
