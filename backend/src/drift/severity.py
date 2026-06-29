"""
severity.py
-----------
Maps a statistical test's p-value to a severity tier.

Thresholds (per the design doc):
  Green  — p > 0.10   (no action)
  Yellow — 0.05 < p <= 0.10  (monitor)
  Red    — p <= 0.05  (alert)

Kept as a standalone module so the thresholds are defined once and reused
by every statistical test (KS, chi-squared) and by the alert engine (Day 8).
"""

from enum import Enum


class Severity(str, Enum):
    GREEN = "Green"
    YELLOW = "Yellow"
    RED = "Red"


# Ordered worst-to-best for easy "pick the worst" reduction across features
_SEVERITY_RANK: dict[str, int] = {
    Severity.RED: 0,
    Severity.YELLOW: 1,
    Severity.GREEN: 2,
}


def classify_p_value(p_value: float) -> Severity:
    """
    Classify a single statistical test's p-value into a severity tier.

    A LOW p-value means the current distribution is statistically
    different from the reference — i.e. drift is more likely — so low
    p-values map to worse (Red) severity.
    """
    if p_value <= 0.05:
        return Severity.RED
    if p_value <= 0.10:
        return Severity.YELLOW
    return Severity.GREEN


def classify_psi(psi_score: float) -> Severity:
    """
    Classify a PSI (Population Stability Index) score into a severity tier.

    Conventional PSI thresholds (industry standard, e.g. credit risk modeling):
      < 0.1   — no significant shift (Green)
      0.1-0.25 — moderate shift, monitor (Yellow)
      > 0.25  — major shift, alert (Red)
    """
    if psi_score >= 0.25:
        return Severity.RED
    if psi_score >= 0.10:
        return Severity.YELLOW
    return Severity.GREEN


def worst_severity(severities: list[Severity]) -> Severity:
    """
    Reduce a list of per-feature severities to the single worst one.
    Used to compute overall_severity for a DriftReport — if any feature
    is Red, the whole report is Red.
    """
    if not severities:
        return Severity.GREEN
    return min(severities, key=lambda s: _SEVERITY_RANK[s])