"""
schemas.py (api)
----------------
Pydantic response models for every REST endpoint exposed by the FastAPI
service. Keeping these separate from src/prediction/schemas.py and
src/monitoring/models.py keeps the API's public contract decoupled from
internal ORM/event shapes — the API can evolve its response shape
without forcing changes to the database schema or Kafka event format.
"""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"


# ── Models ────────────────────────────────────────────────────────────────────

class ModelVersionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    version: str
    stage: str
    training_metrics: dict[str, Any]
    mlflow_run_id: Optional[str] = None
    created_at: datetime


class ModelVersionListResponse(BaseModel):
    items: list[ModelVersionResponse]
    total: int


# ── Drift reports ─────────────────────────────────────────────────────────────

class DriftReportSummaryResponse(BaseModel):
    """Lightweight shape for list views — excludes the full feature_scores blob."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    overall_severity: str
    window_start: datetime
    window_end: datetime
    created_at: datetime
    html_report_path: Optional[str] = None


class DriftReportDetailResponse(BaseModel):
    """Full shape for single-report views — includes per-feature breakdown."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    model_version_id: Optional[int] = None
    overall_severity: str
    window_start: datetime
    window_end: datetime
    created_at: datetime
    feature_scores: dict[str, Any]
    html_report_path: Optional[str] = None


class DriftReportListResponse(BaseModel):
    items: list[DriftReportSummaryResponse]
    total: int
    page: int
    page_size: int


class FeatureStatusCard(BaseModel):
    """One status card per feature for the dashboard Overview page."""

    feature_name: str
    severity: str
    ks_p_value: Optional[float] = None
    psi_score: Optional[float] = None


class DriftSummaryResponse(BaseModel):
    """
    Snapshot of the most recent drift report, restructured for the
    dashboard's Overview page (per-feature status cards).
    """

    report_id: Optional[int] = None
    overall_severity: str = "Green"
    evaluated_at: Optional[datetime] = None
    feature_cards: list[FeatureStatusCard] = []
    prediction_drift: Optional[dict[str, Any]] = None


# ── Metrics history ───────────────────────────────────────────────────────────

class MetricsHistoryPoint(BaseModel):
    timestamp: datetime
    drift_report_id: int
    overall_severity: str


class MetricsHistoryResponse(BaseModel):
    points: list[MetricsHistoryPoint]


# ── Alerts ────────────────────────────────────────────────────────────────────

class AlertEventResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    drift_report_id: Optional[int] = None
    feature_name: str
    severity: str
    channel: str
    notified_at: datetime


class AlertListResponse(BaseModel):
    items: list[AlertEventResponse]
    total: int
    page: int
    page_size: int