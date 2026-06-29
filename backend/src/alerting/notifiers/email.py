from __future__ import annotations

import logging
import os
import smtplib
from email.message import EmailMessage

from src.alerting.engine import AlertCandidate
from src.alerting.notifiers.slack import NotificationResult

logger = logging.getLogger(__name__)

SMTP_TIMEOUT_SECONDS = 10


def _build_email(candidate: AlertCandidate, from_addr: str, to_addr: str, report_url: str | None) -> EmailMessage:
    feature_label = "Prediction distribution" if candidate.feature_name == "__prediction__" else candidate.feature_name

    msg = EmailMessage()
    msg["Subject"] = f"[Drift Alert][{candidate.severity.upper()}] {feature_label}"
    msg["From"] = from_addr
    msg["To"] = to_addr

    body_lines = [
        f"Severity: {candidate.severity.upper()}",
        f"Feature: {feature_label}",
        f"Test: {candidate.worst_test_name} = {candidate.worst_statistic:.4f}",
    ]
    if candidate.drift_report_id is not None:
        body_lines.append(f"Drift report ID: {candidate.drift_report_id}")
    if report_url:
        body_lines.append(f"Full report: {report_url}")

    msg.set_content("\n".join(body_lines))
    return msg


def send_email_alert(candidate: AlertCandidate, report_url: str | None = None) -> NotificationResult:
    host = os.environ.get("SMTP_HOST")
    port = int(os.environ.get("SMTP_PORT", "587"))
    user = os.environ.get("SMTP_USER")
    password = os.environ.get("SMTP_PASSWORD")
    from_addr = os.environ.get("ALERT_EMAIL_FROM")
    to_addr = os.environ.get("ALERT_EMAIL_TO")

    missing = [n for n, v in [("SMTP_HOST", host), ("ALERT_EMAIL_FROM", from_addr), ("ALERT_EMAIL_TO", to_addr)] if not v]
    if missing:
        logger.warning("Email notification skipped; missing env vars: %s", ", ".join(missing))
        return NotificationResult(channel="email", success=False, error=f"missing_config:{','.join(missing)}")

    msg = _build_email(candidate, from_addr, to_addr, report_url)

    try:
        with smtplib.SMTP(host, port, timeout=SMTP_TIMEOUT_SECONDS) as server:
            server.starttls()
            if user and password:
                server.login(user, password)
            server.send_message(msg)
        return NotificationResult(channel="email", success=True)
    except (smtplib.SMTPException, OSError) as exc:
        logger.error("Email notification failed for feature=%s severity=%s: %s", candidate.feature_name, candidate.severity, exc)
        return NotificationResult(channel="email", success=False, error=str(exc))