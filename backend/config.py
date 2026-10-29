"""
Configuration module for the Live Prediction FastAPI backend.
"""

import os
from pathlib import Path

# Resolve base directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ML_MODEL_DIR = PROJECT_ROOT / "ML_model"
MODELS_DIR = ML_MODEL_DIR / "models"
CONFIGS_DIR = ML_MODEL_DIR / "configs"

# Model and schema paths
DEFAULT_MODEL_PATH = MODELS_DIR / "best_model.joblib"
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))

DEFAULT_SCHEMA_PATH = CONFIGS_DIR / "feature_schema.json"
SCHEMA_PATH = Path(os.getenv("FEATURE_SCHEMA_PATH", str(DEFAULT_SCHEMA_PATH)))

DEFAULT_LABEL_PATH = CONFIGS_DIR / "label_mapping.json"
LABEL_PATH = Path(os.getenv("LABEL_MAPPING_PATH", str(DEFAULT_LABEL_PATH)))

DEFAULT_METADATA_PATH = MODELS_DIR / "model_metadata.json"
METADATA_PATH = Path(os.getenv("MODEL_METADATA_PATH", str(DEFAULT_METADATA_PATH)))

# Database configuration
DATA_DIR = PROJECT_ROOT / "backend" / "data"
DEFAULT_DB_PATH = DATA_DIR / "noise_history.db"
DB_PATH = Path(os.getenv("DB_PATH", str(DEFAULT_DB_PATH)))

# Server binding configuration
# Default to 8080 to seamlessly replace the frontend's expected target port
SERVER_HOST = os.getenv("BACKEND_HOST", "127.0.0.1")
SERVER_PORT = int(os.getenv("PORT", os.getenv("BACKEND_PORT", "8080")))

# Sound Pressure Level (SPL) Safety Thresholds (in decibels dB SPL)
# IMPORTANT: These values are provisional unvalidated demo thresholds inspired by
# general workplace noise awareness guidelines (e.g., OSHA 85 dBA 8-hr action level
# and EPA community noise guidelines). They are NOT certified legal or clinical limits.
SPL_SAFE_THRESHOLD = float(os.getenv("SPL_SAFE_THRESHOLD", "70.0"))
SPL_MODERATE_THRESHOLD = float(os.getenv("SPL_MODERATE_THRESHOLD", "85.0"))
