"""
routers/drift.py
-----------------
GET /drift/reports             — paginated list of drift reports
GET /drift/reports/{id}        — single report with full per-feature breakdown
GET /drift/summary             — latest report, restructured for the Overview page
GET /drift/reports/{id}/html   — serve the saved Evidently HTML report
"""

import os
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from src.api.schemas import (
    DriftReportDetailResponse,
    DriftReportListResponse,
    DriftReportSummaryResponse,
    DriftSummaryResponse,
    FeatureStatusCard,
)
from src.monitoring import repository as repo
from src.monitoring.database import get_db

router = APIRouter(prefix="/drift")


@router.get("/reports", response_model=DriftReportListResponse)
def list_drift_reports(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
) -> DriftReportListResponse:
    """Paginated list of drift reports, newest first."""
    reports = repo.list_drift_reports(db, page=page, page_size=page_size)
    total = repo.count_drift_reports(db)

    items = [DriftReportSummaryResponse.model_validate(r) for r in reports]
    return DriftReportListResponse(items=items, total=total, page=page, page_size=page_size)


@router.get("/reports/{report_id}", response_model=DriftReportDetailResponse)
def get_drift_report(report_id: int, db: Session = Depends(get_db)) -> DriftReportDetailResponse:
    """Full detail for a single drift report, including per-feature scores."""
    report = repo.get_drift_report(db, report_id)
    if report is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Drift report {report_id} not found",
        )
    return DriftReportDetailResponse.model_validate(report)


@router.get("/summary", response_model=DriftSummaryResponse)
def get_drift_summary(db: Session = Depends(get_db)) -> DriftSummaryResponse:
    """
    Restructure the latest drift report into per-feature status cards —
    the shape the dashboard's Overview page renders directly.

    Returns a default "Green, no data yet" response if no reports exist,
    rather than a 404 — the dashboard should render cleanly on a fresh
    install before the first scheduled evaluation has run.
    """
    latest = repo.get_latest_drift_report(db)
    if latest is None:
        return DriftSummaryResponse()

    feature_scores = latest.feature_scores or {}
    features_dict = feature_scores.get("features", {})

    cards: list[FeatureStatusCard] = []
    for feature_name, tests in features_dict.items():
        ks_result = tests.get("ks", {})
        psi_result = tests.get("psi", {})

        # Worst severity across this feature's tests (Red > Yellow > Green)
        severities = [
            t.get("severity") for t in tests.values() if t.get("severity")
        ]
        severity_rank = {"Red": 0, "Yellow": 1, "Green": 2}
        worst = min(severities, key=lambda s: severity_rank.get(s, 2)) if severities else "Green"

        cards.append(
            FeatureStatusCard(
                feature_name=feature_name,
                severity=worst,
                ks_p_value=ks_result.get("p_value"),
                psi_score=psi_result.get("statistic"),
            )
        )

    return DriftSummaryResponse(
        report_id=latest.id,
        overall_severity=latest.overall_severity,
        evaluated_at=latest.created_at,
        feature_cards=cards,
        prediction_drift=feature_scores.get("prediction_drift"),
    )


@router.get("/reports/{report_id}/html")
def get_drift_report_html(report_id: int, db: Session = Depends(get_db)) -> FileResponse:
    """
    Serve the saved Evidently HTML report file for download/viewing.
    Returns 404 if the report doesn't exist or its HTML file is missing
    from disk (e.g. container restarted without a persistent volume).
    """
    report = repo.get_drift_report(db, report_id)
    if report is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Drift report {report_id} not found",
        )

    if not report.html_report_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Drift report {report_id} has no associated HTML report",
        )

    html_path = Path(report.html_report_path)
    if not html_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"HTML file not found on disk: {html_path}",
        )

    return FileResponse(
        path=str(html_path),
        media_type="text/html",
        filename=html_path.name,
    )