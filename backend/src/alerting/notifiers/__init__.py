"""Notification backends for the alerting pipeline (Slack, email)."""

from dataclasses import dataclass


@dataclass
class NotificationResult:
    """Outcome of a single notifier send attempt."""

    channel: str  # "slack" | "email"
    success: bool
    error: str | None = None