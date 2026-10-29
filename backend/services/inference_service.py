"""
Inference Service for Environmental Noise Classification.

Loads the trained machine-learning model (LightGBM) from ML_model/models/best_model.joblib
and the associated feature schema & label mappings. Provides verified, deterministic
in-memory inference without fabricating predictions.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

from backend.config import (
    MODEL_PATH,
    SCHEMA_PATH,
    LABEL_PATH,
    METADATA_PATH,
)

logger = logging.getLogger("noise-inference-service")


class InferenceService:
    """
    Singleton service managing the lifecycle and prediction execution of the ML model.
    """
    _instance: Optional["InferenceService"] = None

    def __init__(self):
        self.model: Any = None
        self.is_loaded: bool = False
        self.load_error: Optional[str] = None
        self.feature_names: List[str] = []
        self.class_id_to_name: Dict[int, str] = {}
        self.name_to_class_id: Dict[str, int] = {}
        self.model_metadata: Dict[str, Any] = {}
        self.model_name: str = "Unknown"

    @classmethod
    def get_instance(cls) -> "InferenceService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_artifacts(self) -> bool:
        """
        Loads the feature schema, label mapping, and trained model binary.
        Returns True if successful, False otherwise.
        """
        # 1. Load Feature Schema
        try:
            if not SCHEMA_PATH.is_file():
                raise FileNotFoundError(f"Feature schema file not found at: {SCHEMA_PATH}")
            with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
                schema_data = json.load(f)
            self.feature_names = schema_data.get("feature_names", [])
            if len(self.feature_names) != 62:
                raise ValueError(
                    f"Feature schema mismatch: expected 62 feature names, found {len(self.feature_names)}"
                )
        except Exception as e:
            self.load_error = f"Failed to load feature schema: {e}"
            logger.error(self.load_error)
            self.is_loaded = False
            return False

        # 2. Load Label Mapping
        try:
            if not LABEL_PATH.is_file():
                raise FileNotFoundError(f"Label mapping file not found at: {LABEL_PATH}")
            with open(LABEL_PATH, "r", encoding="utf-8") as f:
                label_data = json.load(f)
            raw_id_map = label_data.get("class_id_to_name", {})
            self.class_id_to_name = {int(k): v for k, v in raw_id_map.items()}
            self.name_to_class_id = label_data.get("name_to_class_id", {})
        except Exception as e:
            self.load_error = f"Failed to load label mapping: {e}"
            logger.error(self.load_error)
            self.is_loaded = False
            return False

        # 3. Load Model Metadata (optional info)
        try:
            if METADATA_PATH.is_file():
                with open(METADATA_PATH, "r", encoding="utf-8") as f:
                    self.model_metadata = json.load(f)
                self.model_name = self.model_metadata.get("best_model_name", "LightGBM Classifier")
            else:
                self.model_name = "LightGBM Classifier (Artifact)"
        except Exception as e:
            logger.warning("Could not read model metadata: %s", e)
            self.model_name = "LightGBM Classifier"

        # 4. Load Trained Model Binary
        try:
            if not MODEL_PATH.is_file():
                raise FileNotFoundError(f"Model artifact not found at: {MODEL_PATH}")
            logger.info("Loading model artifact from %s ...", MODEL_PATH)
            self.model = joblib.load(MODEL_PATH)
            self.is_loaded = True
            self.load_error = None
            logger.info("Successfully loaded ML model (%s).", type(self.model).__name__)
            return True
        except Exception as e:
            self.load_error = f"Failed to deserialize model binary: {e}"
            logger.error(self.load_error)
            self.is_loaded = False
            self.model = None
            return False

    def predict(self, features: List[float]) -> Dict[str, Any]:
        """
        Executes inference against the loaded model using the verified 62-feature vector.
        
        Args:
            features: List of exactly 62 finite numerical floats in canonical order.
            
        Returns:
            Dictionary containing sound_category, class_id, confidence, probabilities list.
            
        Raises:
            RuntimeError: If model is not loaded.
            ValueError: If feature count is not 62.
        """
        if not self.is_loaded or self.model is None:
            err = self.load_error or "Model is not loaded."
            raise RuntimeError(f"ML Model is currently unavailable: {err}")

        if len(features) != 62:
            raise ValueError(f"Feature vector must contain exactly 62 features, got {len(features)}")

        # Convert to 2D numpy array [1, 62] with float32
        x = np.asarray(features, dtype=np.float32).reshape(1, 62)

        # Execute prediction
        raw_pred = self.model.predict(x)
        class_id = int(raw_pred[0])
        sound_category = self.class_id_to_name.get(class_id, f"class_{class_id}")

        confidence: Optional[float] = None
        probabilities: Optional[List[Dict[str, Any]]] = None

        # Extract prediction probabilities if supported by model
        if hasattr(self.model, "predict_proba"):
            try:
                proba_vector = self.model.predict_proba(x)[0]
                confidence = round(float(proba_vector[class_id]), 4)
                
                # Build formatted probabilities list
                prob_items = []
                for cid, p in enumerate(proba_vector):
                    cat_name = self.class_id_to_name.get(cid, f"class_{cid}")
                    prob_items.append({
                        "class_id": cid,
                        "category": cat_name,
                        "confidence": round(float(p), 4)
                    })
                # Sort probabilities descending by confidence
                prob_items.sort(key=lambda item: item["confidence"], reverse=True)
                probabilities = prob_items
            except Exception as pe:
                logger.warning("predict_proba failed: %s", pe)

        return {
            "class_id": class_id,
            "sound_category": sound_category,
            "confidence": confidence,
            "probabilities": probabilities
        }

    def get_status(self) -> Dict[str, Any]:
        """
        Returns model status metadata.
        """
        return {
            "model_loaded": self.is_loaded,
            "model_name": self.model_name,
            "model_type": type(self.model).__name__ if self.model else None,
            "num_features_expected": len(self.feature_names) if self.feature_names else 62,
            "feature_schema": self.feature_names,
            "label_mapping": {str(k): v for k, v in self.class_id_to_name.items()},
            "error": self.load_error
        }
