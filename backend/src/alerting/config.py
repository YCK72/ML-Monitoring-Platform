"""
config.py (alerting)

Severity (Green/Yellow/Red) is already computed upstream by
src/drift/severity.py's classify_p_value() and classify_psi(), and baked
into every FeatureDriftResult before it ever reaches this module. There is
no raw score/threshold comparison left to do here.

What's actually left for this config to decide:
  - which channels to notify for a given severity (Yellow vs Red)
  - how long to suppress repeat alerts for the same feature (cooldown)
  - whether a severity escalation should bypass an active cooldown
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class NotifyChannel(str, Enum):
    SLACK = "slack"
    EMAIL = "email"


class SeverityChannelConfig(BaseModel):
    channels: list[NotifyChannel] = Field(default_factory=lambda: [NotifyChannel.SLACK])
    cooldown_minutes: int = 30


class AlertConfig(BaseModel):
    yellow: SeverityChannelConfig = Field(default_factory=SeverityChannelConfig)
    red: SeverityChannelConfig = Field(
        default_factory=lambda: SeverityChannelConfig(
            channels=[NotifyChannel.SLACK, NotifyChannel.EMAIL], cooldown_minutes=30
        )
    )
    allow_escalation_during_cooldown: bool = True

    def config_for(self, severity: str) -> SeverityChannelConfig | None:
        """severity is one of 'Green' | 'Yellow' | 'Red' (matches severity.py's Severity enum values)."""
        if severity == "Yellow":
            return self.yellow
        if severity == "Red":
            return self.red
        return None  # Green never alerts


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Alert config not found at {path}")
    with path.open("r") as f:
        raw = yaml.safe_load(f)
    return raw or {}


@lru_cache(maxsize=1)
def load_alert_config(path: str = "config/alert_thresholds.yaml") -> AlertConfig:
    raw = _read_yaml(Path(path))
    return AlertConfig.model_validate(raw)