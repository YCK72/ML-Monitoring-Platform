"""
routers/metrics.py
-------------------
GET /metrics/history — time-series of drift report severities over time.

Note: the design doc describes this as AUC/F1/accuracy history, but those
are training-time metrics that don't change between drift evaluations
(they're fixed per model version). What genuinely changes over time is
drift severity per evaluation window — that's what this endpoint surfaces
for the dashboard's Model Performance / trend charts. Training metrics
themselves are available per-version via GET /models/{id}.
"""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from src.api.schemas import MetricsHistoryPoint, MetricsHistoryResponse
from src.monitoring import repository as repo
from src.monitoring.database import get_db

router = APIRouter(prefix="/metrics")


@router.get("/history", response_model=MetricsHistoryResponse)
def get_metrics_history(
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db),
) -> MetricsHistoryResponse:
    """
    Return up to *limit* recent drift reports as a time-series of
    (timestamp, severity) points, oldest first — ready for a line chart.
    """
    page_size = limit
    reports = repo.list_drift_reports(db, page=1, page_size=page_size)

    # list_drift_reports returns newest-first; reverse for a left-to-right
    # chronological chart.
    points = [
        MetricsHistoryPoint(
            timestamp=r.created_at,
            drift_report_id=r.id,
            overall_severity=r.overall_severity,
        )
        for r in reversed(reports)
    ]

    return MetricsHistoryResponse(points=points)