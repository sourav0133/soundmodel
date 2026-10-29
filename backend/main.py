"""
Main entrypoint for the Environmental Noise Live Prediction FastAPI backend.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import SERVER_HOST, SERVER_PORT
from backend.database import init_db
from backend.routers.health import router as health_router
from backend.routers.history import router as history_router
from backend.routers.predict import router as predict_router
from backend.services.inference_service import InferenceService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("noise-backend-main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle management: runs database initialization and model loading on server startup.
    """
    logger.info("Initializing database...")
    init_db()

    logger.info("Loading ML classification model and schemas...")
    inference_svc = InferenceService.get_instance()
    success = inference_svc.load_artifacts()
    if success:
        logger.info("Model '%s' successfully loaded into memory.", inference_svc.model_name)
    else:
        logger.warning(
            "Model failed to load at startup: %s. Requests to /predict will report error.",
            inference_svc.load_error
        )

    yield

    logger.info("Shutting down prediction backend.")


app = FastAPI(
    title="Environmental Noise Live Prediction API",
    description="""
    Live Prediction & SPL Health Monitoring API for Environmental Sound Classification.
    
    ### Capabilities:
    - **POST /predict**: Ingests 62-D acoustic features and SPL decibels from sensor nodes and executes live LightGBM classification.
    - **GET /model/status**: Details loaded model specifications, 62-feature ordering, and label mappings.
    - **GET /history**: Queries stored acoustic events and classification logs from SQLite.
    - **GET /health**: Liveness and model availability probe.
    """,
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for Streamlit and external frontend callers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routes
app.include_router(health_router)
app.include_router(predict_router)
app.include_router(history_router)

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting prediction API on %s:%d", SERVER_HOST, SERVER_PORT)
    uvicorn.run("backend.main:app", host=SERVER_HOST, port=SERVER_PORT, reload=True)
