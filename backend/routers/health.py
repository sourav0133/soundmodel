"""
Health and status check router.
"""

from fastapi import APIRouter
from backend.models.schemas import HealthResponse
from backend.services.inference_service import InferenceService
from backend.database import is_db_healthy

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health and readiness probe",
    description="Reports the live operational readiness of the prediction engine, model, and database."
)
def health_check() -> HealthResponse:
    inference_svc = InferenceService.get_instance()
    model_ok = inference_svc.is_loaded
    db_ok = is_db_healthy()

    is_healthy = model_ok and db_ok
    status_str = "healthy" if is_healthy else "degraded"

    note = (
        "Prediction engine active with trained LightGBM model. "
        "DSP source integration pending: raw audio feature extraction unavailable; "
        "62-D preprocessed JSON prediction pipeline fully functional."
    )
    if not model_ok:
        note = f"Model artifact could not be loaded: {inference_svc.load_error}"

    return HealthResponse(
        status=status_str,
        service="Environmental Noise Live Prediction API",
        version="1.0.0",
        model_loaded=model_ok,
        db_connected=db_ok,
        dsp_integrated=False,
        note=note,
    )


@router.get("/", tags=["Health"])
def root_status():
    """
    Root status check with links to Swagger UI documentation.
    """
    return {
        "service": "Environmental Noise Live Prediction API",
        "status": "online",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_url": "/health",
        "predict_url": "/predict",
        "history_url": "/history",
    }
