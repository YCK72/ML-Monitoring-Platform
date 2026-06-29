"""
evidently_report.py
--------------------
Wraps Evidently AI's DataDriftPreset to generate polished HTML drift
reports alongside our own scipy-based DriftDetector (detector.py).

Why both? DriftDetector gives us full control over the exact statistical
tests and severity thresholds described in the design doc (KS, PSI,
chi-squared, Wasserstein). Evidently AI is layered on top purely to
produce the downloadable HTML report and an independent JSON summary
for the dashboard's "Report Export" feature (FR #10) — it is not used
to drive alerting decisions.

Tested against evidently==0.7.21. Evidently's public API has changed
significantly across major versions; if you upgrade evidently, re-verify
this module against the new API before relying on it.
"""

import logging
from pathlib import Path

import pandas as pd

from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset

logger = logging.getLogger(__name__)


def generate_evidently_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    output_html_path: str | Path,
    numerical_columns: list[str] | None = None,
    categorical_columns: list[str] | None = None,
) -> dict:
    """
    Run Evidently's DataDriftPreset against reference_df vs current_df,
    save the result as an HTML file, and return the JSON-serialisable
    summary dict.

    Parameters
    ----------
    reference_df:
        Baseline DataFrame (same shape as used by DriftDetector).
    current_df:
        Current window DataFrame to compare against the baseline.
    output_html_path:
        Where to write the HTML report. Parent directories are created
        if they don't exist.
    numerical_columns / categorical_columns:
        Explicit column lists. If omitted, Evidently infers types from
        the DataFrame's dtypes — explicit lists are safer for consistency
        with DriftDetector's own feature classification.

    Returns
    -------
    dict with keys:
      - "drifted_columns_count": dict with count/share of drifted columns
      - "column_results": dict mapping column name -> {method, value (p-value
        or score), drifted: bool}
      - "html_path": str path to the saved HTML file
    """
    output_html_path = Path(output_html_path)
    output_html_path.parent.mkdir(parents=True, exist_ok=True)

    definition = DataDefinition(
        numerical_columns=numerical_columns,
        categorical_columns=categorical_columns,
    )

    reference_dataset = Dataset.from_pandas(reference_df, data_definition=definition)
    current_dataset = Dataset.from_pandas(current_df, data_definition=definition)

    report = Report([DataDriftPreset()])
    snapshot = report.run(reference_data=reference_dataset, current_data=current_dataset)

    snapshot.save_html(str(output_html_path))
    logger.info("Evidently HTML report saved → %s", output_html_path)

    raw = snapshot.dict()
    summary = _parse_evidently_result(raw)
    summary["html_path"] = str(output_html_path)

    return summary


def _parse_evidently_result(raw: dict) -> dict:
    """
    Flatten Evidently's metrics list into a simpler, dashboard-friendly
    shape: {drifted_columns_count: {...}, column_results: {col: {...}}}.

    Evidently's raw metric_name strings look like:
      "DriftedColumnsCount(drift_share=0.5)"
      "ValueDrift(column=feature_a,method=K-S p_value,threshold=0.05)"
    We parse these defensively since Evidently does not guarantee a
    stable structured schema across versions.
    """
    drifted_columns_count: dict = {}
    column_results: dict[str, dict] = {}

    for metric in raw.get("metrics", []):
        name = metric.get("metric_name", "")
        value = metric.get("value")
        config = metric.get("config", {})

        if name.startswith("DriftedColumnsCount"):
            if isinstance(value, dict):
                drifted_columns_count = {
                    "count": value.get("count"),
                    "share": value.get("share"),
                }
        elif name.startswith("ValueDrift"):
            column = config.get("column")
            method = config.get("method")
            threshold = config.get("threshold")
            if column is not None:
                drifted = (
                    isinstance(value, (int, float))
                    and threshold is not None
                    and value < threshold
                )
                column_results[column] = {
                    "method": method,
                    "value": value,
                    "threshold": threshold,
                    "drifted": bool(drifted),
                }

    return {
        "drifted_columns_count": drifted_columns_count,
        "column_results": column_results,
    }