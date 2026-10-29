"""
Pydantic data models and request/response validation schemas for the prediction API.
"""

import math
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator


class PredictRequest(BaseModel):
    """
    Input payload for POST /predict.
    Expected to originate from the sensor DSP node or manual test injector.
    """
    model_config = ConfigDict(extra="forbid")

    node_id: str = Field(
        ...,
        description="Sensor or edge device node identifier",
        examples=["NODE_001"]
    )
    spl_db: float = Field(
        ...,
        description="Sound pressure level measured by transducer in decibels (dB SPL)",
        examples=[72.45]
    )
    timestamp: Union[int, float] = Field(
        ...,
        description="Epoch timestamp of sensor measurement",
        examples=[1793200000]
    )
    features: List[float] = Field(
        ...,
        description="Preprocessed acoustic feature vector. Must contain exactly 62 finite numerical features in canonical order."
    )

    @field_validator("node_id")
    @classmethod
    def validate_node_id(cls, v: str) -> str:
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("node_id must not be empty or whitespace only.")
        return trimmed

    @field_validator("spl_db")
    @classmethod
    def validate_spl_db(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("spl_db must be a finite numerical value (NaN and Inf are not permitted).")
        return v

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, v: Union[int, float]) -> Union[int, float]:
        if not math.isfinite(v):
            raise ValueError("timestamp must be a finite numerical value (NaN and Inf are not permitted).")
        return v

    @field_validator("features")
    @classmethod
    def validate_features(cls, v: List[float]) -> List[float]:
        if len(v) != 62:
            raise ValueError(
                f"Invalid feature vector length: expected exactly 62 features matching canonical schema, got {len(v)}."
            )
        for i, val in enumerate(v):
            if not isinstance(val, (int, float)) or not math.isfinite(val):
                raise ValueError(
                    f"Feature at index {i} is not a valid finite number (found '{val}'). NaN and Inf are not allowed."
                )
        return [float(x) for x in v]


class ProbabilityItem(BaseModel):
    """
    Probability score item for a candidate sound class.
    """
    class_id: int = Field(..., description="Integer class index (0-9)")
    category: str = Field(..., description="Canonical category name")
    confidence: float = Field(..., description="Classification probability (0.0 to 1.0)")


class PredictResponse(BaseModel):
    """
    Response returned by POST /predict.
    """
    model_config = ConfigDict(populate_by_name=True)

    status: str = Field("success", description="Request execution status")
    node_id: str = Field(..., description="Echoed node identifier")
    timestamp: Union[int, float] = Field(..., description="Echoed sensor measurement timestamp")
    sound_category: str = Field(..., description="Predicted environmental noise category")
    class_id: int = Field(..., description="Numeric class identifier (0-9)")
    confidence: Optional[float] = Field(
        None,
        description="Top-class prediction probability score (0.0 to 1.0) when supported by model"
    )
    probabilities: Optional[List[ProbabilityItem]] = Field(
        None,
        description="Class-wise prediction probability distribution sorted by descending confidence"
    )
    spl_db: float = Field(..., description="Original transducer SPL decibel reading")
    spl_status: str = Field(..., description="SPL health risk classification: 'SAFE', 'MODERATE', or 'VERY_HARMFUL'")
    spectrum_513: Optional[List[float]] = Field(
        None,
        description="513-bin FFT magnitude spectrum. Reserved for future DSP integration; null until DSP code is integrated."
    )
    message: Optional[str] = Field(
        None,
        description="Contextual note or warnings"
    )


class HealthResponse(BaseModel):
    """
    Response for GET /health liveness and readiness probe.
    """
    status: str = Field(..., description="'healthy' or 'degraded'")
    service: str = Field("Environmental Noise Live Prediction API", description="Service title")
    version: str = Field("1.0.0", description="Backend version")
    model_loaded: bool = Field(..., description="Whether the ML model is currently loaded in memory")
    db_connected: bool = Field(..., description="Whether the SQLite history database is accessible")
    dsp_integrated: bool = Field(
        False,
        description="Indicates whether raw-audio DSP feature extraction is available (False until DSP is provided)"
    )
    note: str = Field(..., description="Operational status note")


class ModelStatusResponse(BaseModel):
    """
    Detailed model inspection response for GET /model/status.
    """
    model_loaded: bool = Field(..., description="Whether model is loaded and ready for inference")
    model_name: Optional[str] = Field(None, description="Human-readable model name")
    model_type: Optional[str] = Field(None, description="Python class type of loaded model")
    num_features_expected: int = Field(62, description="Number of numerical features expected in payload")
    feature_schema: List[str] = Field(default_factory=list, description="Ordered canonical feature column names")
    label_mapping: Dict[str, str] = Field(default_factory=dict, description="Numeric class ID to category name mapping")
    error: Optional[str] = Field(None, description="Error message if model loading failed")


class HistoryRecord(BaseModel):
    """
    Single persisted prediction history item.
    """
    id: int
    timestamp: float
    node_id: str
    predicted_category: str
    confidence: Optional[float]
    spl_db: float
    spl_status: str
    created_at: str
