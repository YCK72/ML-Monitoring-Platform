"""
engine.py (alerting)

REPLACES the previous version entirely — that draft assumed a
DriftReportLike interface with .feature_results / .metric / .score that
doesn't match your real code. This version operates directly on
DriftResult.to_dict()'s actual shape (src/drift/detector.py):

    {
      "features": {
        "feature_income": {
          "ks":  {"feature_name": ..., "test_name": "ks",  "statistic": ..., "p_value": ..., "severity": "Red"},
          "psi": {"feature_name": ..., "test_name": "psi", "statistic": ..., "p_value": None, "severity": "Yellow"},
        },
        ...
      },
      "prediction_drift": {"statistic": ..., "raw_distance": ..., "severity": "Green"} | None,
      "overall_severity": "Red",
    }

A feature can have multiple tests (ks + psi for numerical features) with
different severities — this module takes the WORST severity across a
feature's tests, same reduction as worst_severity() in severity.py.

Severity is never recomputed here — it was already decided by
src/drift/severity.py upstream. This module only decides: does this
severity warrant a notification right now, given cooldown state?

Cooldown lookup matches your real repository.py exactly:
    repo.get_most_recent_alert_for_feature(db, feature_name) -> AlertEvent | None
This function has no time filter — it just returns the single most recent
alert ever fired for that feature, regardless of age. This module computes
elapsed time itself (now - recent.notified_at) and compares to the
configured cooldown_minutes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, Protocol

from src.alerting.config import AlertConfig, NotifyChannel

_SEVERITY_RANK = {"Green": 2, "Yellow": 1, "Red": 0}  # lower rank = worse, matches severity.py


@dataclass
class AlertCandidate:
    """A feature (or prediction-drift) whose severity warrants notification."""

    drift_report_id: Optional[int]
    feature_name: str  # use "__prediction__" for prediction drift
    severity: str       # "Yellow" | "Red"
    worst_test_name: str    # e.g. "ks", "psi" — for the Slack/email message only, not persisted
    worst_statistic: float  # for the message only, not persisted
    notify_channels: list[NotifyChannel]
    cooldown_minutes: int


class RecentAlertLike(Protocol):
    severity: str
    notified_at: datetime


class CooldownLookup(Protocol):
    """
    Matches repo.get_most_recent_alert_for_feature(db, feature_name) exactly
    — no time filter, just "the most recent alert ever for this feature."
    Wrap your db session in a closure/adapter when constructing AlertEngine,
    e.g.:
        cooldown_lookup = lambda feature_name: repo.get_most_recent_alert_for_feature(db, feature_name)
    """

    def __call__(self, feature_name: str) -> Optional[RecentAlertLike]:
        ...


def _worst_test_for_feature(tests: dict) -> tuple[str, str, float]:
    """
    Given a feature's {"ks": {...}, "psi": {...}} (or just {"chi_squared": {...}}),
    return (worst_severity, worst_test_name, worst_statistic).
    """
    worst_severity = "Green"
    worst_test_name = ""
    worst_statistic = 0.0

    for test_name, result in tests.items():
        severity = result["severity"]
        if _SEVERITY_RANK[severity] < _SEVERITY_RANK[worst_severity]:
            worst_severity = severity
            worst_test_name = test_name
            worst_statistic = result["statistic"]

    return worst_severity, worst_test_name, worst_statistic


class AlertEngine:
    def __init__(self, config: AlertConfig, cooldown_lookup: Optional[CooldownLookup] = None):
        self.config = config
        self.cooldown_lookup = cooldown_lookup

    def evaluate(
        self, feature_scores: dict, drift_report_id: Optional[int], now: Optional[datetime] = None
    ) -> list[AlertCandidate]:
        """
        feature_scores is exactly DriftResult.to_dict()'s output.
        Returns the AlertCandidates that should actually fire (severity
        warrants it AND not currently suppressed by cooldown).
        """
        now = now or datetime.now(timezone.utc)
        candidates: list[AlertCandidate] = []

        for feature_name, tests in feature_scores.get("features", {}).items():
            severity, test_name, statistic = _worst_test_for_feature(tests)
            candidate = self._maybe_build_candidate(
                feature_name, severity, test_name, statistic, drift_report_id, now
            )
            if candidate:
                candidates.append(candidate)

        prediction_drift = feature_scores.get("prediction_drift")
        if prediction_drift is not None:
            candidate = self._maybe_build_candidate(
                "__prediction__",
                prediction_drift["severity"],
                "wasserstein",
                prediction_drift["statistic"],
                drift_report_id,
                now,
            )
            if candidate:
                candidates.append(candidate)

        return candidates

    def _maybe_build_candidate(
        self,
        feature_name: str,
        severity: str,
        test_name: str,
        statistic: float,
        drift_report_id: Optional[int],
        now: datetime,
    ) -> Optional[AlertCandidate]:
        severity_cfg = self.config.config_for(severity)
        if severity_cfg is None:
            return None  # Green — never alerts

        if self._is_suppressed(feature_name, severity, severity_cfg.cooldown_minutes, now):
            return None

        return AlertCandidate(
            drift_report_id=drift_report_id,
            feature_name=feature_name,
            severity=severity,
            worst_test_name=test_name,
            worst_statistic=statistic,
            notify_channels=severity_cfg.channels,
            cooldown_minutes=severity_cfg.cooldown_minutes,
        )

    def _is_suppressed(self, feature_name: str, severity: str, cooldown_minutes: int, now: datetime) -> bool:
        if self.cooldown_lookup is None:
            return False

        recent = self.cooldown_lookup(feature_name)
        if recent is None:
            return False

        notified_at = recent.notified_at
        if notified_at.tzinfo is None:
            notified_at = notified_at.replace(tzinfo=timezone.utc)

        elapsed = now - notified_at
        if elapsed >= timedelta(minutes=cooldown_minutes):
            return False  # cooldown has expired -> allow

        is_escalation = _SEVERITY_RANK[severity] < _SEVERITY_RANK[recent.severity]
        if is_escalation and self.config.allow_escalation_during_cooldown:
            return False  # escalation always gets through

        return True  # within cooldown and not an escalation -> suppress