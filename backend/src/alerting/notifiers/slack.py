from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import requests

from src.alerting.engine import AlertCandidate

logger = logging.getLogger(__name__)

SLACK_TIMEOUT_SECONDS = 5


@dataclass
class NotificationResult:
    channel: str
    success: bool
    error: str | None = None


def _format_message(candidate: AlertCandidate, report_url: str | None) -> dict:
    emoji = {"Yellow": ":large_yellow_circle:", "Red": ":red_circle:"}.get(candidate.severity, ":warning:")
    feature_label = "Prediction distribution" if candidate.feature_name == "__prediction__" else candidate.feature_name

    lines = [
        f"{emoji} *Drift Alert — {candidate.severity.upper()}*",
        f"*Feature:* {feature_label}",
        f"*Test:* {candidate.worst_test_name} = {candidate.worst_statistic:.4f}",
    ]
    if candidate.drift_report_id is not None:
        lines.append(f"*Drift report:* #{candidate.drift_report_id}")
    if report_url:
        lines.append(f"<{report_url}|View full drift report>")

    return {"text": "\n".join(lines)}


def send_slack_alert(candidate: AlertCandidate, webhook_url: str | None = None, report_url: str | None = None) -> NotificationResult:
    webhook_url = webhook_url or os.environ.get("SLACK_WEBHOOK_URL")
    if not webhook_url:
        logger.warning("SLACK_WEBHOOK_URL not configured; skipping Slack notification")
        return NotificationResult(channel="slack", success=False, error="webhook_url_not_configured")

    payload = _format_message(candidate, report_url)

    try:
        response = requests.post(webhook_url, json=payload, timeout=SLACK_TIMEOUT_SECONDS)
        response.raise_for_status()
        return NotificationResult(channel="slack", success=True)
    except requests.RequestException as exc:
        logger.error("Slack notification failed for feature=%s severity=%s: %s", candidate.feature_name, candidate.severity, exc)
        return NotificationResult(channel="slack", success=False, error=str(exc))