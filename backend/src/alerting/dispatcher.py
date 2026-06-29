"""
Dispatches an AlertCandidate to its configured channels, with Slack-first /
email-fallback semantics: if "slack" is in the candidate's notify_channels
and the Slack send fails, email is attempted automatically — even if
"email" wasn't explicitly listed — since the design doc frames email as a
fallback channel, not a parallel one.

If a candidate's channels are e.g. [email] only (no slack), Slack is never
attempted and email is sent directly.
"""

from __future__ import annotations

from src.alerting.config import NotifyChannel
from src.alerting.engine import AlertCandidate
from src.alerting.notifiers import NotificationResult
from src.alerting.notifiers.email import send_email_alert
from src.alerting.notifiers.slack import send_slack_alert


def dispatch(candidate: AlertCandidate, report_url: str | None = None) -> list[NotificationResult]:
    results: list[NotificationResult] = []
    channels = set(candidate.notify_channels)

    slack_failed = False
    if NotifyChannel.SLACK in channels:
        slack_result = send_slack_alert(candidate, report_url=report_url)
        results.append(slack_result)
        slack_failed = not slack_result.success

    should_send_email = NotifyChannel.EMAIL in channels or slack_failed
    if should_send_email:
        results.append(send_email_alert(candidate, report_url=report_url))

    return results


def overall_delivery_status(results: list[NotificationResult]) -> str:
    """Collapse a list of per-channel results into a single status string
    suitable for the alert_events.delivery_status column."""
    if not results:
        return "skipped"
    if all(r.success for r in results):
        return "delivered"
    if any(r.success for r in results):
        return "partial"
    return "failed"