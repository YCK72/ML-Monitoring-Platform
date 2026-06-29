"""
service.py
----------
FastAPI app exposing a single POST /predict endpoint.

Request flow:
  1. Validate incoming JSON against PredictionRequest schema.
  2. Load the cached XGBoost model via model_loader.get_model().
  3. Run inference → prediction (int) + probability (float).
  4. Build a PredictionEvent and publish it to Kafka (non-blocking).
  5. Return PredictionResponse to the caller — target p95 < 200 ms.

Run locally:
    uvicorn src.prediction.service:app --reload --port 8001

In Docker, this runs on port 8001 alongside the main FastAPI service
(port 8000) so the two concerns stay cleanly separated.
"""

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from src.prediction.kafka_producer import (
    close_producer,
    get_producer,
    is_connected,
    publish_prediction,
)
from src.prediction.schemas import (
    HealthResponse,
    PredictionEvent,
    PredictionRequest,
    PredictionResponse,
)
from src.training.model_loader import get_model

logger = logging.getLogger(__name__)

MODEL_NAME: str = os.getenv("MLFLOW_MODEL_NAME", "drift-detector-model")
MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


# ── Lifespan: warm up connections at startup ──────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load the model and open the Kafka producer before the first request."""
    logger.info("Starting prediction service …")
    try:
        get_model(model_name=MODEL_NAME, tracking_uri=MLFLOW_TRACKING_URI)
        logger.info("Model pre-loaded ✓")
    except Exception as exc:
        logger.error("Could not pre-load model: %s", exc)

    try:
        get_producer()
        logger.info("Kafka producer initialised ✓")
    except Exception as exc:
        logger.warning("Kafka not available at startup: %s", exc)

    yield  # app is running

    close_producer()
    logger.info("Prediction service stopped.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ML Monitoring — Prediction Service",
    description="Runs XGBoost inference and publishes events to Kafka.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5173").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness + readiness probe used by Docker Compose and load balancers."""
    model_loaded = True
    try:
        get_model(model_name=MODEL_NAME, tracking_uri=MLFLOW_TRACKING_URI)
    except Exception:
        model_loaded = False

    kafka_ok = is_connected()

    return HealthResponse(
        status="ok" if (model_loaded and kafka_ok) else "degraded",
        kafka_connected=kafka_ok,
        model_loaded=model_loaded,
        details={
            "model_name": MODEL_NAME,
            "mlflow_uri": MLFLOW_TRACKING_URI,
            "kafka_topic": os.getenv("KAFKA_TOPIC", "prediction-events"),
        },
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["inference"],
)
def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Run inference on a single feature vector and log the event to Kafka.

    - Model loading is cached; the first call warms up, subsequent calls
      return in microseconds.
    - Kafka publishing is fire-and-forget and never delays the response.
    - Returns HTTP 422 if required features are missing or contain NaN.
    - Returns HTTP 503 if the model cannot be loaded.
    """
    # 1. Load model (cached after first call)
    try:
        model = get_model(model_name=MODEL_NAME, tracking_uri=MLFLOW_TRACKING_URI)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model unavailable: {exc}",
        )

    # 2. Build feature DataFrame in the order the model expects
    try:
        X = pd.DataFrame([request.features])
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not build feature matrix: {exc}",
        )

    # 3. Inference
    try:
        prediction: int = int(model.predict(X)[0])
        probability: float = float(model.predict_proba(X)[0][1])
    except Exception as exc:
        logger.error("Inference failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {exc}",
        )

    now = datetime.now(timezone.utc)

    # 4. Resolve model version for the event payload
    # Use a lightweight string tag; avoid an MLflow round-trip per request.
    model_version = os.getenv("MODEL_VERSION", "unknown")

    # 5. Publish to Kafka (non-blocking, errors are swallowed + logged)
    event = PredictionEvent(
        features=request.features,
        prediction=prediction,
        probability=probability,
        model_name=MODEL_NAME,
        model_version=model_version,
        timestamp=now,
    )
    publish_prediction(event)

    # 6. Return response
    return PredictionResponse(
        prediction=prediction,
        probability=probability,
        model_version=model_version,
        timestamp=now,
    )