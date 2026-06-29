"""
models.py
---------
SQLAlchemy 2.0 declarative ORM models for the metric store.

Four tables:
  - model_versions     — one row per registered MLflow model version
  - prediction_records — one row per logged inference event (from Kafka)
  - drift_reports      — one row per scheduled drift evaluation
  - alert_events       — one row per notification fired (Slack/email)

Relationships:
  ModelVersion 1───* PredictionRecord
  ModelVersion 1───* DriftReport
  DriftReport  1───* AlertEvent
"""

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    """Timezone-aware UTC timestamp used as a default for all created_at columns."""
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


# ── model_versions ────────────────────────────────────────────────────────────

class ModelVersion(Base):
    """
    Mirrors a registered MLflow model version. One row is inserted/updated
    whenever a new version is trained and registered.
    """

    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    version: Mapped[str] = mapped_column(String(50), nullable=False)
    stage: Mapped[str] = mapped_column(String(50), nullable=False, default="None")
    training_metrics: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    mlflow_run_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )

    prediction_records: Mapped[list["PredictionRecord"]] = relationship(
        back_populates="model_version", cascade="all, delete-orphan"
    )
    drift_reports: Mapped[list["DriftReport"]] = relationship(
        back_populates="model_version", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<ModelVersion {self.name} v{self.version} ({self.stage})>"


# ── prediction_records ────────────────────────────────────────────────────────

class PredictionRecord(Base):
    """
    A single inference event consumed from the 'prediction-events' Kafka
    topic and persisted by the drift consumer (Day 6).
    """

    __tablename__ = "prediction_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    features: Mapped[dict] = mapped_column(JSON, nullable=False)
    prediction: Mapped[int] = mapped_column(Integer, nullable=False)
    probability: Mapped[float] = mapped_column(Float, nullable=False)
    model_version_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("model_versions.id", ondelete="CASCADE"), nullable=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False, index=True
    )

    model_version: Mapped[Optional["ModelVersion"]] = relationship(
        back_populates="prediction_records"
    )

    def __repr__(self) -> str:
        return f"<PredictionRecord id={self.id} pred={self.prediction} p={self.probability:.3f}>"


# ── drift_reports ─────────────────────────────────────────────────────────────

class DriftReport(Base):
    """
    Output of a single scheduled drift evaluation run. feature_scores stores
    the full per-feature breakdown (KS, PSI, chi-squared, severity) as JSON
    so the schema doesn't need to change when new statistical tests are added.
    """

    __tablename__ = "drift_reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_version_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("model_versions.id", ondelete="CASCADE"), nullable=True, index=True
    )
    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    window_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    feature_scores: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    overall_severity: Mapped[str] = mapped_column(String(20), nullable=False, default="Green")
    html_report_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False, index=True
    )

    model_version: Mapped[Optional["ModelVersion"]] = relationship(
        back_populates="drift_reports"
    )
    alert_events: Mapped[list["AlertEvent"]] = relationship(
        back_populates="drift_report", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<DriftReport id={self.id} severity={self.overall_severity}>"


# ── alert_events ──────────────────────────────────────────────────────────────

class AlertEvent(Base):
    """
    A single notification fired by the alert engine (Day 8) when a
    feature's drift score crosses its configured threshold.
    """

    __tablename__ = "alert_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    drift_report_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("drift_reports.id", ondelete="CASCADE"), nullable=True, index=True
    )
    feature_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    severity: Mapped[str] = mapped_column(String(20), nullable=False)
    channel: Mapped[str] = mapped_column(String(20), nullable=False)  # "slack" | "email"
    notified_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False, index=True
    )

    drift_report: Mapped[Optional["DriftReport"]] = relationship(
        back_populates="alert_events"
    )

    def __repr__(self) -> str:
        return f"<AlertEvent feature={self.feature_name} severity={self.severity}>"