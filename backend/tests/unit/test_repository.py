"""
test_repository.py
-------------------
Unit tests for src/monitoring/repository.py.

Each test is isolated via the `db` fixture (fresh in-memory SQLite per test).
Covers every CRUD function across all four tables, plus the relationship
and dedup-lookup behavior that the drift evaluator and alert engine rely on.
"""

from datetime import datetime, timedelta, timezone

import pytest

from src.monitoring import repository as repo


# ── ModelVersion ──────────────────────────────────────────────────────────────

def test_upsert_model_version_creates_new(db):
    mv = repo.upsert_model_version(
        db,
        name="drift-detector-model",
        version="1",
        stage="Staging",
        training_metrics={"auc": 0.91, "f1": 0.88},
        mlflow_run_id="run123",
    )

    assert mv.id is not None
    assert mv.name == "drift-detector-model"
    assert mv.version == "1"
    assert mv.stage == "Staging"
    assert mv.training_metrics == {"auc": 0.91, "f1": 0.88}
    assert mv.mlflow_run_id == "run123"
    assert mv.created_at is not None


def test_upsert_model_version_updates_existing(db):
    repo.upsert_model_version(
        db, name="m", version="1", stage="Staging", training_metrics={"auc": 0.9}
    )

    updated = repo.upsert_model_version(
        db, name="m", version="1", stage="Production", training_metrics={"auc": 0.95}
    )

    all_versions = repo.list_model_versions(db)
    assert len(all_versions) == 1, "Upsert should not create a duplicate row"
    assert updated.stage == "Production"
    assert updated.training_metrics == {"auc": 0.95}


def test_get_model_version_by_id(db):
    mv = repo.upsert_model_version(
        db, name="m", version="1", stage="Staging", training_metrics={}
    )
    fetched = repo.get_model_version(db, mv.id)
    assert fetched is not None
    assert fetched.id == mv.id


def test_get_model_version_missing_returns_none(db):
    assert repo.get_model_version(db, 9999) is None


def test_list_model_versions_orders_newest_first(db):
    repo.upsert_model_version(db, name="m", version="1", stage="Staging", training_metrics={})
    repo.upsert_model_version(db, name="m", version="2", stage="Staging", training_metrics={})

    versions = repo.list_model_versions(db)
    assert len(versions) == 2
    assert versions[0].version == "2"  # most recently created comes first


# ── PredictionRecord ──────────────────────────────────────────────────────────

def test_create_prediction_record(db):
    record = repo.create_prediction_record(
        db,
        features={"feature_income": 0.5, "feature_age": -1.2},
        prediction=1,
        probability=0.82,
    )

    assert record.id is not None
    assert record.features == {"feature_income": 0.5, "feature_age": -1.2}
    assert record.prediction == 1
    assert record.probability == 0.82
    assert record.model_version_id is None


def test_create_prediction_record_links_to_model_version(db):
    mv = repo.upsert_model_version(
        db, name="m", version="1", stage="Staging", training_metrics={}
    )
    record = repo.create_prediction_record(
        db, features={"f": 1.0}, prediction=0, probability=0.1,
        model_version_id=mv.id,
    )
    assert record.model_version_id == mv.id


def test_bulk_create_prediction_records(db):
    records = [
        {"features": {"f": float(i)}, "prediction": i % 2, "probability": 0.5}
        for i in range(10)
    ]
    count = repo.bulk_create_prediction_records(db, records)
    assert count == 10
    assert len(repo.get_recent_predictions(db, limit=100)) == 10


def test_get_recent_predictions_respects_limit(db):
    for i in range(5):
        repo.create_prediction_record(
            db, features={"f": float(i)}, prediction=0, probability=0.1
        )
    recent = repo.get_recent_predictions(db, limit=3)
    assert len(recent) == 3


def test_get_recent_predictions_filters_by_since(db):
    old_time = datetime.now(timezone.utc) - timedelta(days=2)
    recent_time = datetime.now(timezone.utc)

    repo.create_prediction_record(
        db, features={"f": 1.0}, prediction=0, probability=0.1, created_at=old_time
    )
    repo.create_prediction_record(
        db, features={"f": 2.0}, prediction=1, probability=0.9, created_at=recent_time
    )

    cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
    results = repo.get_recent_predictions(db, limit=100, since=cutoff)

    assert len(results) == 1
    assert results[0].probability == 0.9


# ── DriftReport ───────────────────────────────────────────────────────────────

def test_create_drift_report(db):
    now = datetime.now(timezone.utc)
    report = repo.create_drift_report(
        db,
        window_start=now - timedelta(minutes=5),
        window_end=now,
        feature_scores={"feature_income": {"p_value": 0.03, "severity": "Red"}},
        overall_severity="Red",
        html_report_path="/reports/abc.html",
    )

    assert report.id is not None
    assert report.overall_severity == "Red"
    assert report.feature_scores["feature_income"]["severity"] == "Red"
    assert report.html_report_path == "/reports/abc.html"


def test_create_drift_report_links_to_model_version(db):
    mv = repo.upsert_model_version(
        db, name="m", version="1", stage="Staging", training_metrics={}
    )
    now = datetime.now(timezone.utc)
    report = repo.create_drift_report(
        db, window_start=now, window_end=now, feature_scores={},
        overall_severity="Green", model_version_id=mv.id,
    )
    assert report.model_version_id == mv.id
    assert report in mv.drift_reports


def test_get_drift_report_by_id(db):
    now = datetime.now(timezone.utc)
    report = repo.create_drift_report(
        db, window_start=now, window_end=now, feature_scores={}, overall_severity="Green"
    )
    fetched = repo.get_drift_report(db, report.id)
    assert fetched is not None
    assert fetched.id == report.id


def test_get_latest_drift_report_returns_most_recent(db):
    now = datetime.now(timezone.utc)
    repo.create_drift_report(
        db, window_start=now, window_end=now, feature_scores={}, overall_severity="Green"
    )
    second = repo.create_drift_report(
        db, window_start=now, window_end=now, feature_scores={}, overall_severity="Red"
    )

    latest = repo.get_latest_drift_report(db)
    assert latest is not None
    assert latest.id == second.id
    assert latest.overall_severity == "Red"


def test_get_latest_drift_report_empty_returns_none(db):
    assert repo.get_latest_drift_report(db) is None


def test_list_drift_reports_pagination(db):
    now = datetime.now(timezone.utc)
    for _ in range(5):
        repo.create_drift_report(
            db, window_start=now, window_end=now, feature_scores={}, overall_severity="Green"
        )

    page1 = repo.list_drift_reports(db, page=1, page_size=2)
    page2 = repo.list_drift_reports(db, page=2, page_size=2)

    assert len(page1) == 2
    assert len(page2) == 2
    assert {r.id for r in page1}.isdisjoint({r.id for r in page2})


# ── AlertEvent ────────────────────────────────────────────────────────────────

def test_create_alert_event(db):
    alert = repo.create_alert_event(
        db, feature_name="feature_income", severity="Red", channel="slack"
    )
    assert alert.id is not None
    assert alert.feature_name == "feature_income"
    assert alert.severity == "Red"
    assert alert.channel == "slack"
    assert alert.notified_at is not None


def test_create_alert_event_links_to_drift_report(db):
    now = datetime.now(timezone.utc)
    report = repo.create_drift_report(
        db, window_start=now, window_end=now, feature_scores={}, overall_severity="Red"
    )
    alert = repo.create_alert_event(
        db, feature_name="f", severity="Red", channel="email", drift_report_id=report.id
    )
    assert alert.drift_report_id == report.id
    assert alert in report.alert_events


def test_get_most_recent_alert_for_feature(db):
    repo.create_alert_event(db, feature_name="feature_income", severity="Yellow", channel="slack")
    newest = repo.create_alert_event(db, feature_name="feature_income", severity="Red", channel="slack")
    # Different feature — should never be returned by the lookup below
    repo.create_alert_event(db, feature_name="feature_age", severity="Red", channel="slack")

    most_recent = repo.get_most_recent_alert_for_feature(db, "feature_income")
    assert most_recent is not None
    assert most_recent.id == newest.id
    assert most_recent.severity == "Red"


def test_get_most_recent_alert_for_feature_none_when_no_history(db):
    assert repo.get_most_recent_alert_for_feature(db, "never_alerted_feature") is None


def test_list_alerts_filters_by_severity(db):
    repo.create_alert_event(db, feature_name="f1", severity="Red", channel="slack")
    repo.create_alert_event(db, feature_name="f2", severity="Yellow", channel="slack")
    repo.create_alert_event(db, feature_name="f3", severity="Red", channel="email")

    red_alerts = repo.list_alerts(db, severity="Red")
    assert len(red_alerts) == 2
    assert all(a.severity == "Red" for a in red_alerts)


def test_list_alerts_filters_by_feature_name(db):
    repo.create_alert_event(db, feature_name="feature_income", severity="Red", channel="slack")
    repo.create_alert_event(db, feature_name="feature_age", severity="Red", channel="slack")

    results = repo.list_alerts(db, feature_name="feature_income")
    assert len(results) == 1
    assert results[0].feature_name == "feature_income"


def test_list_alerts_pagination(db):
    for i in range(5):
        repo.create_alert_event(db, feature_name=f"f{i}", severity="Red", channel="slack")

    page1 = repo.list_alerts(db, page=1, page_size=2)
    page2 = repo.list_alerts(db, page=2, page_size=2)

    assert len(page1) == 2
    assert len(page2) == 2


# ── Cascade delete behavior ───────────────────────────────────────────────────

def test_deleting_model_version_cascades_to_predictions_and_reports(db):
    mv = repo.upsert_model_version(
        db, name="m", version="1", stage="Staging", training_metrics={}
    )
    repo.create_prediction_record(
        db, features={"f": 1.0}, prediction=1, probability=0.9, model_version_id=mv.id
    )
    now = datetime.now(timezone.utc)
    repo.create_drift_report(
        db, window_start=now, window_end=now, feature_scores={},
        overall_severity="Green", model_version_id=mv.id,
    )

    db.delete(mv)
    db.commit()

    assert len(repo.get_recent_predictions(db, limit=100)) == 0
    assert repo.get_latest_drift_report(db) is None


def test_deleting_drift_report_cascades_to_alert_events(db):
    now = datetime.now(timezone.utc)
    report = repo.create_drift_report(
        db, window_start=now, window_end=now, feature_scores={}, overall_severity="Red"
    )
    repo.create_alert_event(
        db, feature_name="f", severity="Red", channel="slack", drift_report_id=report.id
    )

    db.delete(report)
    db.commit()

    assert repo.list_alerts(db) == []