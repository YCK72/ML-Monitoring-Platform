"""
routers/alerts.py
------------------
GET /alerts — filterable, paginated list of alert events.

Query params:
  severity:      filter by exact severity ("Red" | "Yellow" | "Green")
  feature_name:  filter by exact feature name
  page, page_size: pagination
"""

from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from src.api.schemas import AlertEventResponse, AlertListResponse
from src.monitoring import repository as repo
from src.monitoring.database import get_db

router = APIRouter()


@router.get("/alerts", response_model=AlertListResponse)
def list_alerts(
    severity: Optional[str] = Query(None, description="Filter by severity: Red, Yellow, Green"),
    feature_name: Optional[str] = Query(None, description="Filter by exact feature name"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
) -> AlertListResponse:
    """Paginated, filterable list of alert events, newest first."""
    alerts = repo.list_alerts(
        db,
        severity=severity,
        feature_name=feature_name,
        page=page,
        page_size=page_size,
    )

    items = [AlertEventResponse.model_validate(a) for a in alerts]

    # Total count across all pages matching the same filters (for pagination UI)
    all_matching = repo.list_alerts(
        db, severity=severity, feature_name=feature_name, page=1, page_size=10_000
    )

    return AlertListResponse(
        items=items, total=len(all_matching), page=page, page_size=page_size
    )