"""
main.py (api)
-------------
The main FastAPI application for the ML Monitoring Platform's REST API.

Run locally:
    uvicorn src.api.main:app --reload --port 8000

Endpoints (see individual routers for details):
    GET  /health
    GET  /models
    GET  /models/{id}
    GET  /drift/reports
    GET  /drift/reports/{id}
    GET  /drift/summary
    GET  /drift/reports/{id}/html
    GET  /metrics/history
    GET  /alerts
"""

import logging
import os

from fastapi import FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from src.api.routers import alerts, drift, metrics, models
from src.api.schemas import HealthResponse

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

API_KEY: str = os.getenv("API_KEY", "dev-local-key")
CORS_ORIGINS: list[str] = os.getenv(
    "CORS_ORIGINS", "http://localhost:5173"
).split(",")

app = FastAPI(
    title="ML Monitoring Platform — API",
    description="REST endpoints for drift reports, model metadata, "
                 "metrics history, and alert events.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ── API key auth ──────────────────────────────────────────────────────────────

EXEMPT_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}


@app.middleware("http")
async def api_key_middleware(request, call_next):
    """
    Require X-API-Key on every request except /health and the
    auto-generated docs endpoints. Returns HTTP 401 on missing/invalid key.
    """
    if request.url.path in EXEMPT_PATHS:
        return await call_next(request)

    provided_key = request.headers.get("X-API-Key")
    if provided_key != API_KEY:
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"detail": "Invalid or missing X-API-Key header"},
        )

    return await call_next(request)


# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(models.router, tags=["models"])
app.include_router(drift.router, tags=["drift"])
app.include_router(metrics.router, tags=["metrics"])
app.include_router(alerts.router, tags=["alerts"])


# ── Health (no auth required) ─────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    return HealthResponse(status="ok")