"""
detector.py
-----------
DriftDetector: compares a current window of predictions against a
reference (training-time) dataset using four statistical tests:

  - Kolmogorov-Smirnov (KS) test  — numerical feature drift
  - Population Stability Index (PSI) — numerical feature drift
  - Chi-squared test              — categorical feature drift
  - Wasserstein distance          — prediction (probability) drift

Each numerical feature gets both a KS and a PSI score; categorical
features get a chi-squared score. The model's output probability
distribution is compared separately via Wasserstein distance.

Usage:
    detector = DriftDetector(reference_df=ref_df)
    result = detector.compute_drift(current_df, current_probabilities)
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from src.drift.severity import Severity, classify_p_value, classify_psi, worst_severity

# Number of bins used for PSI's discretized distribution comparison.
# 10 is the conventional choice in industry PSI implementations.
PSI_BIN_COUNT = 10

# Reference distribution range used to normalize Wasserstein distance
# into a roughly comparable scale across runs (probabilities live in [0, 1]).
PROBABILITY_RANGE = 1.0


@dataclass
class FeatureDriftResult:
    """Per-feature drift result for a single statistical test."""

    feature_name: str
    test_name: str          # "ks", "psi", "chi_squared"
    statistic: float
    p_value: float | None   # None for PSI, which has no p-value
    severity: Severity

    def to_dict(self) -> dict:
        return {
            "feature_name": self.feature_name,
            "test_name": self.test_name,
            "statistic": round(self.statistic, 6),
            "p_value": round(self.p_value, 6) if self.p_value is not None else None,
            "severity": self.severity.value,
        }


@dataclass
class DriftResult:
    """
    Full output of a single drift evaluation run.

    feature_results: one or more FeatureDriftResult per feature (numerical
    features get both KS and PSI; categorical features get chi-squared).
    prediction_drift: Wasserstein-based result for the model's output
    probability distribution, or None if probabilities weren't provided.
    overall_severity: the worst severity across all feature + prediction results.
    """

    feature_results: list[FeatureDriftResult] = field(default_factory=list)
    prediction_drift: dict | None = None
    overall_severity: Severity = Severity.GREEN

    def to_dict(self) -> dict:
        """
        Serialise to the JSON shape stored in DriftReport.feature_scores.
        Grouped by feature name so the API/dashboard can look up a
        feature's full result set in one access.
        """
        by_feature: dict[str, dict] = {}
        for r in self.feature_results:
            by_feature.setdefault(r.feature_name, {})[r.test_name] = r.to_dict()

        return {
            "features": by_feature,
            "prediction_drift": self.prediction_drift,
            "overall_severity": self.overall_severity.value,
        }


class DriftDetector:
    """
    Compares a current window of data against a fixed reference dataset.

    Parameters
    ----------
    reference_df:
        The baseline DataFrame (typically loaded from reference.parquet).
        Must contain the same feature columns as the data later passed to
        compute_drift(). May optionally include a 'target' or probability
        column — these are ignored; only matching feature columns are used.
    numerical_features:
        Explicit list of numerical feature column names. If None, inferred
        automatically from reference_df's dtypes.
    categorical_features:
        Explicit list of categorical feature column names. If None, inferred
        automatically from reference_df's dtypes.
    reference_probabilities:
        Optional array of reference prediction probabilities, used as the
        baseline for Wasserstein-based prediction drift detection.
    """

    def __init__(
        self,
        reference_df: pd.DataFrame,
        numerical_features: list[str] | None = None,
        categorical_features: list[str] | None = None,
        reference_probabilities: np.ndarray | None = None,
    ) -> None:
        self.reference_df = reference_df.copy()
        self.reference_probabilities = (
            np.asarray(reference_probabilities)
            if reference_probabilities is not None
            else None
        )

        if numerical_features is not None:
            self.numerical_features = list(numerical_features)
        else:
            self.numerical_features = list(
                reference_df.select_dtypes(include=[np.number]).columns
            )

        if categorical_features is not None:
            self.categorical_features = list(categorical_features)
        else:
            self.categorical_features = list(
                reference_df.select_dtypes(include=["object", "category"]).columns
            )

    # ── Public API ────────────────────────────────────────────────────────

    def compute_drift(
        self,
        current_df: pd.DataFrame,
        current_probabilities: np.ndarray | None = None,
    ) -> DriftResult:
        """
        Run all configured statistical tests against current_df (and
        current_probabilities, if prediction drift detection is enabled).

        Returns a DriftResult with per-feature results and an aggregated
        overall_severity.
        """
        feature_results: list[FeatureDriftResult] = []

        for col in self.numerical_features:
            if col not in current_df.columns or col not in self.reference_df.columns:
                continue
            feature_results.append(self._run_ks_test(col, current_df))
            feature_results.append(self._run_psi(col, current_df))

        for col in self.categorical_features:
            if col not in current_df.columns or col not in self.reference_df.columns:
                continue
            feature_results.append(self._run_chi_squared(col, current_df))

        prediction_drift = None
        if current_probabilities is not None and self.reference_probabilities is not None:
            prediction_drift = self._run_wasserstein(current_probabilities)

        all_severities = [r.severity for r in feature_results]
        if prediction_drift is not None:
            all_severities.append(Severity(prediction_drift["severity"]))

        overall = worst_severity(all_severities)

        return DriftResult(
            feature_results=feature_results,
            prediction_drift=prediction_drift,
            overall_severity=overall,
        )

    # ── Individual statistical tests ─────────────────────────────────────

    def _run_ks_test(self, feature: str, current_df: pd.DataFrame) -> FeatureDriftResult:
        """
        Kolmogorov-Smirnov two-sample test.
        Null hypothesis: both samples are drawn from the same distribution.
        Low p-value -> reject null -> distributions differ -> drift.
        """
        ref_values = self.reference_df[feature].dropna().to_numpy()
        cur_values = current_df[feature].dropna().to_numpy()

        if len(ref_values) == 0 or len(cur_values) == 0:
            return FeatureDriftResult(feature, "ks", 0.0, 1.0, Severity.GREEN)

        statistic, p_value = stats.ks_2samp(ref_values, cur_values)
        severity = classify_p_value(p_value)
        return FeatureDriftResult(feature, "ks", float(statistic), float(p_value), severity)

    def _run_psi(self, feature: str, current_df: pd.DataFrame) -> FeatureDriftResult:
        """
        Population Stability Index — bins both distributions using the
        reference data's bin edges, then sums the divergence contribution
        of each bin. No p-value; severity is classified directly from the
        PSI magnitude using industry-standard thresholds.
        """
        ref_values = self.reference_df[feature].dropna().to_numpy()
        cur_values = current_df[feature].dropna().to_numpy()

        if len(ref_values) == 0 or len(cur_values) == 0:
            return FeatureDriftResult(feature, "psi", 0.0, None, Severity.GREEN)

        psi_score = self._compute_psi_score(ref_values, cur_values)
        severity = classify_psi(psi_score)
        return FeatureDriftResult(feature, "psi", float(psi_score), None, severity)

    @staticmethod
    def _compute_psi_score(ref_values: np.ndarray, cur_values: np.ndarray) -> float:
        """
        Bin edges are derived from the reference distribution's quantiles
        so each reference bin starts with roughly equal population —
        this is the standard PSI binning approach.
        """
        # Use quantile-based bin edges from the reference set
        quantiles = np.linspace(0, 1, PSI_BIN_COUNT + 1)
        bin_edges = np.unique(np.quantile(ref_values, quantiles))

        if len(bin_edges) < 3:
            # Degenerate case: reference has near-constant values
            return 0.0

        # Extend outer edges so out-of-range current values still get binned
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        ref_counts, _ = np.histogram(ref_values, bins=bin_edges)
        cur_counts, _ = np.histogram(cur_values, bins=bin_edges)

        # Convert to proportions, with a small epsilon to avoid log(0) / div-by-0
        eps = 1e-6
        ref_props = ref_counts / max(len(ref_values), 1) + eps
        cur_props = cur_counts / max(len(cur_values), 1) + eps

        psi = np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))
        return float(psi)

    def _run_chi_squared(self, feature: str, current_df: pd.DataFrame) -> FeatureDriftResult:
        """
        Chi-squared goodness-of-fit test for categorical features.
        Compares the current category frequency distribution against the
        reference's, using the reference proportions scaled to the
        current sample size as "expected" counts.
        """
        ref_counts = self.reference_df[feature].dropna().value_counts()
        cur_counts = current_df[feature].dropna().value_counts()

        if ref_counts.empty or cur_counts.empty:
            return FeatureDriftResult(feature, "chi_squared", 0.0, 1.0, Severity.GREEN)

        # Align both series over the union of categories seen in either set
        categories = sorted(set(ref_counts.index) | set(cur_counts.index))
        ref_aligned = ref_counts.reindex(categories, fill_value=0).to_numpy(dtype=float)
        cur_aligned = cur_counts.reindex(categories, fill_value=0).to_numpy(dtype=float)

        # Expected counts = reference proportions scaled to current sample size
        total_current = cur_aligned.sum()
        total_reference = ref_aligned.sum()
        if total_reference == 0 or total_current == 0:
            return FeatureDriftResult(feature, "chi_squared", 0.0, 1.0, Severity.GREEN)

        expected = (ref_aligned / total_reference) * total_current
        # Avoid divide-by-zero for categories absent from the reference set
        expected = np.where(expected == 0, 1e-6, expected)

        statistic, p_value = stats.chisquare(f_obs=cur_aligned, f_exp=expected)
        severity = classify_p_value(p_value)
        return FeatureDriftResult(
            feature, "chi_squared", float(statistic), float(p_value), severity
        )

    def _run_wasserstein(self, current_probabilities: np.ndarray) -> dict:
        """
        Wasserstein (earth mover's) distance between the reference and
        current model output probability distributions.

        Normalized by PROBABILITY_RANGE (1.0, since probabilities live in
        [0, 1]) to give a 0-1-ish interpretable distance. Severity is
        classified using the same PSI-style thresholds, since Wasserstein
        distance — like PSI — has no associated p-value.
        """
        cur_probs = np.asarray(current_probabilities)
        distance = stats.wasserstein_distance(self.reference_probabilities, cur_probs)
        normalized = float(distance / PROBABILITY_RANGE)
        severity = classify_psi(normalized)

        return {
            "statistic": round(normalized, 6),
            "raw_distance": round(float(distance), 6),
            "severity": severity.value,
        }