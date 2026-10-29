"""
Prediction and model inspection router.
"""

import logging
from fastapi import APIRouter, HTTPException, status
from backend.models.schemas import (
    PredictRequest,
    PredictResponse,
    ProbabilityItem,
    ModelStatusResponse,
)
from backend.services.inference_service import InferenceService
from backend.services.spl_service import classify_spl
from backend.services.history_service import HistoryService

logger = logging.getLogger("noise-predict-router")

router = APIRouter(tags=["Prediction"])


@router.post(
    "/predict",
    response_model=PredictResponse,
    status_code=status.HTTP_200_OK,
    summary="Perform live environmental noise classification",
    description="""
    Accepts a preprocessed acoustic telemetry payload containing `node_id`, `spl_db`,
    `timestamp`, and exactly 62 continuous numerical features in the canonical feature schema order.
    
    Executes inference against the verified production LightGBM model, classifies the sound pressure
    level into a safety status (SAFE, MODERATE, VERY_HARMFUL), logs the event in SQLite history,
    and returns full classification details with prediction confidence and class probabilities.
    """
)
def predict_noise(request: PredictRequest) -> PredictResponse:
    inference_svc = InferenceService.get_instance()

    # 1. Verify model readiness
    if not inference_svc.is_loaded:
        err_msg = inference_svc.load_error or "ML model is not loaded in memory."
        logger.error("Inference requested but model is unavailable: %s", err_msg)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Classification engine unavailable: {err_msg}"
        )

    # 2. Compute SPL status based on transducer dB reading
    spl_status = classify_spl(request.spl_db)

    # 3. Execute model inference
    try:
        pred_result = inference_svc.predict(request.features)
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid feature schema: {ve}"
        )
    except Exception as exc:
        logger.exception("Inference execution failure: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference computation error: {exc}"
        )

    class_id = pred_result["class_id"]
    sound_category = pred_result["sound_category"]
    confidence = pred_result["confidence"]
    raw_probs = pred_result.get("probabilities")

    prob_items = None
    if raw_probs:
        prob_items = [
            ProbabilityItem(
                class_id=p["class_id"],
                category=p["category"],
                confidence=p["confidence"]
            )
            for p in raw_probs
        ]

    # 4. Persist to SQLite history
    try:
        HistoryService.save_prediction(
            timestamp=float(request.timestamp),
            node_id=request.node_id,
            predicted_category=sound_category,
            confidence=confidence,
            spl_db=request.spl_db,
            spl_status=spl_status
        )
    except Exception as dbe:
        logger.warning("Failed to persist prediction history to database: %s", dbe)

    return PredictResponse(
        status="success",
        node_id=request.node_id,
        timestamp=request.timestamp,
        sound_category=sound_category,
        class_id=class_id,
        confidence=confidence,
        probabilities=prob_items,
        spl_db=request.spl_db,
        spl_status=spl_status,
        spectrum_513=None,  # Explicitly None until DSP FFT stream is integrated
        message="Prediction generated using production LightGBM model on verified 62-D feature vector."
    )


@router.get(
    "/model/status",
    response_model=ModelStatusResponse,
    summary="Get current ML model status and metadata",
    description="Inspects loaded model architecture, expected feature schema, and label mappings."
)
def get_model_status() -> ModelStatusResponse:
    inference_svc = InferenceService.get_instance()
    info = inference_svc.get_status()
    return ModelStatusResponse(
        model_loaded=info["model_loaded"],
        model_name=info["model_name"],
        model_type=info["model_type"],
        num_features_expected=info["num_features_expected"],
        feature_schema=info["feature_schema"],
        label_mapping=info["label_mapping"],
        error=info["error"]
    )
