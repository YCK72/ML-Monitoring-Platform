"""
repository.py
-------------
CRUD functions for the metric store. This is the single place that talks
to the database — the drift consumer, scheduler, and FastAPI routers all
go through these functions rather than writing raw queries inline.

All functions accept a SQLAlchemy Session as their first argument so they
work equally well under FastAPI's get_db() dependency or a plain script.
"""

from datetime import datetime
from typing import Optional, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.monitoring.models import (
    AlertEvent,
    DriftReport,
    ModelVersion,
    PredictionRecord,
)


# ── ModelVersion ──────────────────────────────────────────────────────────────

def upsert_model_version(
    db: Session,
    name: str,
    version: str,
    stage: str,
    training_metrics: dict,
    mlflow_run_id: Optional[str] = None,
) -> ModelVersion:
    """Insert a model version row, or update it if (name, version) already exists."""
    existing = db.execute(
        select(ModelVersion).where(
            ModelVersion.name == name, ModelVersion.version == version
        )
    ).scalar_one_or_none()

    if existing:
        existing.stage = stage
        existing.training_metrics = training_metrics
        existing.mlflow_run_id = mlflow_run_id
        db.commit()
        db.refresh(existing)
        return existing

    mv = ModelVersion(
        name=name,
        version=version,
        stage=stage,
        training_metrics=training_metrics,
        mlflow_run_id=mlflow_run_id,
    )
    db.add(mv)
    db.commit()
    db.refresh(mv)
    return mv


def get_model_version(db: Session, model_version_id: int) -> Optional[ModelVersion]:
    return db.get(ModelVersion, model_version_id)


def list_model_versions(db: Session) -> Sequence[ModelVersion]:
    return db.execute(
        select(ModelVersion).order_by(ModelVersion.created_at.desc())
    ).scalars().all()


# ── PredictionRecord ──────────────────────────────────────────────────────────

def create_prediction_record(
    db: Session,
    features: dict,
    prediction: int,
    probability: float,
    model_version_id: Optional[int] = None,
    created_at: Optional[datetime] = None,
) -> PredictionRecord:
    record = PredictionRecord(
        features=features,
        prediction=prediction,
        probability=probability,
        model_version_id=model_version_id,
        **({"created_at": created_at} if created_at else {}),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def bulk_create_prediction_records(
    db: Session,
    records: list[dict],
) -> int:
    """
    Insert many prediction records in one transaction.
    Used by the Kafka consumer when flushing a batch of events.
    Returns the number of rows inserted.
    """
    objects = [PredictionRecord(**r) for r in records]
    db.add_all(objects)
    db.commit()
    return len(objects)


def get_recent_predictions(
    db: Session,
    limit: int = 1000,
    since: Optional[datetime] = None,
) -> Sequence[PredictionRecord]:
    """
    Return the most recent prediction records, optionally filtered to
    those created after *since*. Used to build the drift evaluation window.
    """
    stmt = select(PredictionRecord).order_by(PredictionRecord.created_at.desc())
    if since is not None:
        stmt = stmt.where(PredictionRecord.created_at >= since)
    stmt = stmt.limit(limit)
    return db.execute(stmt).scalars().all()


# ── DriftReport ───────────────────────────────────────────────────────────────

def create_drift_report(
    db: Session,
    window_start: datetime,
    window_end: datetime,
    feature_scores: dict,
    overall_severity: str,
    model_version_id: Optional[int] = None,
    html_report_path: Optional[str] = None,
) -> DriftReport:
    report = DriftReport(
        model_version_id=model_version_id,
        window_start=window_start,
        window_end=window_end,
        feature_scores=feature_scores,
        overall_severity=overall_severity,
        html_report_path=html_report_path,
    )
    db.add(report)
    db.commit()
    db.refresh(report)
    return report


def get_drift_report(db: Session, report_id: int) -> Optional[DriftReport]:
    return db.get(DriftReport, report_id)


def get_latest_drift_report(db: Session) -> Optional[DriftReport]:
    return db.execute(
        select(DriftReport).order_by(DriftReport.created_at.desc()).limit(1)
    ).scalar_one_or_none()


def list_drift_reports(
    db: Session,
    page: int = 1,
    page_size: int = 20,
) -> Sequence[DriftReport]:
    offset = (page - 1) * page_size
    return db.execute(
        select(DriftReport)
        .order_by(DriftReport.created_at.desc())
        .offset(offset)
        .limit(page_size)
    ).scalars().all()


def count_drift_reports(db: Session) -> int:
    return db.execute(select(DriftReport)).scalars().all().__len__()


# ── AlertEvent ────────────────────────────────────────────────────────────────

def create_alert_event(
    db: Session,
    feature_name: str,
    severity: str,
    channel: str,
    drift_report_id: Optional[int] = None,
) -> AlertEvent:
    alert = AlertEvent(
        drift_report_id=drift_report_id,
        feature_name=feature_name,
        severity=severity,
        channel=channel,
    )
    db.add(alert)
    db.commit()
    db.refresh(alert)
    return alert


def get_most_recent_alert_for_feature(
    db: Session,
    feature_name: str,
) -> Optional[AlertEvent]:
    """
    Used by the alert engine's cool-down/dedup logic — find the last time
    *feature_name* fired an alert, regardless of which drift report.
    """
    return db.execute(
        select(AlertEvent)
        .where(AlertEvent.feature_name == feature_name)
        .order_by(AlertEvent.notified_at.desc())
        .limit(1)
    ).scalar_one_or_none()


def list_alerts(
    db: Session,
    severity: Optional[str] = None,
    feature_name: Optional[str] = None,
    page: int = 1,
    page_size: int = 50,
) -> Sequence[AlertEvent]:
    stmt = select(AlertEvent).order_by(AlertEvent.notified_at.desc())
    if severity:
        stmt = stmt.where(AlertEvent.severity == severity)
    if feature_name:
        stmt = stmt.where(AlertEvent.feature_name == feature_name)
    offset = (page - 1) * page_size
    stmt = stmt.offset(offset).limit(page_size)
    return db.execute(stmt).scalars().all()