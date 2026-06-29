"""
evaluator.py
------------
run_drift_evaluation(): the single function the scheduler calls on every
tick. Ties together:

  1. Snapshot the current in-memory window (from consumer.py).
  2. Run DriftDetector (scipy-based KS/PSI/chi-squared/Wasserstein).
  3. Generate an Evidently HTML report alongside it.
  4. Persist a DriftReport row to Postgres.
  5. Evaluate the persisted report for alert-worthy features and dispatch
     notifications (Day 8).
"""

import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from src.alerting.config import load_alert_config
from src.alerting.dispatcher import dispatch
from src.alerting.engine import AlertEngine
from src.drift.consumer import drift_window
from src.drift.detector import DriftDetector
from src.drift.evidently_report import generate_evidently_report
from src.monitoring.database import SessionLocal
from src.monitoring import repository as repo

logger = logging.getLogger(__name__)

REFERENCE_DATA_PATH: str = os.getenv("REFERENCE_DATA_PATH", "data/raw/reference.parquet")
REPORTS_DIR: str = os.getenv("REPORTS_DIR", "reports")

# Minimum window size required before running an evaluation — avoids
# generating statistically meaningless reports on a near-empty window.
MIN_WINDOW_SIZE_FOR_EVAL: int = int(os.getenv("MIN_WINDOW_SIZE_FOR_EVAL", "30"))

# Cached reference data + detector — loaded once per process.
_reference_df: pd.DataFrame | None = None
_reference_probabilities = None
_detector: DriftDetector | None = None


def _load_reference() -> tuple[pd.DataFrame, "DriftDetector"]:
    """
    Lazily load reference.parquet and build a DriftDetector instance.
    Cached at module level — the reference dataset doesn't change between
    scheduler ticks within the same process lifetime.
    """
    global _reference_df, _reference_probabilities, _detector

    if _detector is not None:
        return _reference_df, _detector

    ref_path = Path(REFERENCE_DATA_PATH)
    if not ref_path.exists():
        raise FileNotFoundError(
            f"Reference dataset not found at {ref_path}. "
            "Run `python -m src.training.generate_data` first."
        )

    df = pd.read_parquet(ref_path)
    if "target" in df.columns:
        df = df.drop(columns=["target"])

    _reference_df = df
    # No reference probabilities available from generate_data.py by default;
    # prediction drift detection is skipped unless this is wired up later
    # (e.g. by scoring the reference set through the trained model once).
    _reference_probabilities = None
    _detector = DriftDetector(reference_df=df, reference_probabilities=_reference_probabilities)

    logger.info(
        "Reference data loaded — %d rows, %d features",
        len(df), df.shape[1],
    )
    return _reference_df, _detector


def _run_alerting(db, drift_report_id: int, feature_scores: dict) -> None:
    """
    Evaluate the just-persisted drift report for alert-worthy features,
    dispatch notifications, and persist one alert_events row per channel
    actually notified successfully.

    Wrapped so any alerting failure (Slack down, bad config, etc.) never
    breaks drift evaluation — the report has already been saved
    successfully by the time this runs.
    """
    try:
        config = load_alert_config()  # cached after first call
        cooldown_lookup = lambda feature_name: repo.get_most_recent_alert_for_feature(db, feature_name)
        engine = AlertEngine(config=config, cooldown_lookup=cooldown_lookup)

        candidates = engine.evaluate(feature_scores, drift_report_id=drift_report_id)

        for candidate in candidates:
            results = dispatch(candidate)
            for result in results:
                if result.success:
                    repo.create_alert_event(
                        db,
                        feature_name=candidate.feature_name,
                        severity=candidate.severity,
                        channel=result.channel,
                        drift_report_id=candidate.drift_report_id,
                    )
                else:
                    logger.warning(
                        "Alert delivery failed — feature=%s severity=%s channel=%s error=%s",
                        candidate.feature_name, candidate.severity, result.channel, result.error,
                    )

        if candidates:
            logger.info(
                "Alerting: %d candidate(s) evaluated for drift report #%d",
                len(candidates), drift_report_id,
            )

    except Exception:  # noqa: BLE001
        logger.exception("Alerting pipeline failed for drift report #%d", drift_report_id)


def run_drift_evaluation() -> dict | None:
    """
    The function called by the APScheduler job on every tick.

    Returns the persisted DriftReport's data as a dict, or None if the
    window doesn't yet have enough data to evaluate.
    """
    window_size = len(drift_window)
    if window_size < MIN_WINDOW_SIZE_FOR_EVAL:
        logger.info(
            "Skipping drift evaluation — window has %d events, need at least %d",
            window_size, MIN_WINDOW_SIZE_FOR_EVAL,
        )
        return None

    logger.info("Running drift evaluation on window of %d events …", window_size)

    try:
        reference_df, detector = _load_reference()
    except FileNotFoundError as exc:
        logger.error("Cannot run drift evaluation: %s", exc)
        return None

    features_list, probabilities = drift_window.snapshot()
    current_df = pd.DataFrame(features_list)

    window_end = datetime.now(timezone.utc)
    window_start = window_end - timedelta(minutes=int(os.getenv("DRIFT_EVAL_INTERVAL_MINUTES", "5")))

    # 1. Run our own statistical tests (source of truth for severity)
    result = detector.compute_drift(current_df, current_probabilities=None)

    # 2. Generate the Evidently HTML report alongside it
    html_path = None
    try:
        timestamp_str = window_end.strftime("%Y%m%dT%H%M%S")
        output_path = Path(REPORTS_DIR) / f"{timestamp_str}.html"
        evidently_summary = generate_evidently_report(
            reference_df=reference_df,
            current_df=current_df,
            output_html_path=output_path,
            numerical_columns=detector.numerical_features,
            categorical_columns=detector.categorical_features or None,
        )
        html_path = evidently_summary["html_path"]
    except Exception as exc:  # noqa: BLE001
        logger.error("Evidently report generation failed (non-fatal): %s", exc)

    # 3. Persist the DriftReport
    feature_scores = result.to_dict()

    db = SessionLocal()
    try:
        report = repo.create_drift_report(
            db,
            window_start=window_start,
            window_end=window_end,
            feature_scores=feature_scores,
            overall_severity=result.overall_severity.value,
            html_report_path=html_path,
        )
        logger.info(
            "Drift report #%d persisted — overall severity: %s",
            report.id, report.overall_severity,
        )

        # 4. Evaluate for alerts and dispatch notifications (Day 8)
        _run_alerting(db, report.id, feature_scores)

        return {
            "id": report.id,
            "overall_severity": report.overall_severity,
            "feature_scores": feature_scores,
            "html_report_path": html_path,
            "window_start": window_start.isoformat(),
            "window_end": window_end.isoformat(),
        }
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to persist drift report: %s", exc)
        db.rollback()
        return None
    finally:
        db.close()