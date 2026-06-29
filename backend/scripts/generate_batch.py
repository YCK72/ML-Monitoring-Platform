"""
Generate synthetic prediction-request batches for smoke testing the
drift/alerting pipeline — calibrated against your ACTUAL reference dataset
(data/raw/reference.parquet), not assumed distributions.

Confirmed against src/training/generate_data.py and src/prediction/schemas.py:
  - POST /predict body shape: {"features": {name: float, ...}}
  - 15 real feature names (see FEATURE_NAMES below)
  - data comes from sklearn.make_classification, NOT standard normal —
    so "normal" mode bootstraps real reference rows rather than sampling
    a fabricated distribution. This guarantees the "normal" batch looks
    exactly like training data, by construction.

Modes:
  normal — bootstrap real reference rows verbatim (zero expected drift)
  mild   — bootstrap real rows, then shift every feature by
           +MILD_SHIFT_STDS standard deviations (should land Yellow)
  heavy  — bootstrap real rows, shift by +HEAVY_SHIFT_STDS standard
           deviations AND inflate variance by HEAVY_VARIANCE_MULTIPLIER
           (should land Red)

Usage:
    python scripts/generate_batch.py --mode normal --n 100  --out normal_batch.json
    python scripts/generate_batch.py --mode mild   --n 1000 --out mild_drift_batch.json
    python scripts/generate_batch.py --mode heavy  --n 1000 --out heavy_drift_batch.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_NAMES: list[str] = [
    "feature_income",
    "feature_age",
    "feature_credit_score",
    "feature_debt_ratio",
    "feature_employment_years",
    "feature_num_accounts",
    "feature_loan_amount",
    "feature_payment_history",
    "feature_utilization_rate",
    "feature_recent_inquiries",
    "feature_derived_a",
    "feature_derived_b",
    "feature_derived_c",
    "feature_copy_income",
    "feature_copy_age",
]

MILD_SHIFT_STDS = 1.5      # shift mean by 1.5 std-devs -> aimed at Yellow severity
HEAVY_SHIFT_STDS = 4.0     # shift mean by 4 std-devs -> aimed at Red severity
HEAVY_VARIANCE_MULTIPLIER = 2.0


def load_reference_stats(reference_path: Path) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    if not reference_path.exists():
        raise FileNotFoundError(
            f"Reference dataset not found at {reference_path}. "
            "Run src/training/generate_data.py first, or pass --reference "
            "pointing at your actual reference.parquet location."
        )
    df = pd.read_parquet(reference_path)
    df = df[FEATURE_NAMES]  # drop 'target' and enforce column order
    means = df.mean()
    stds = df.std()
    return df, means, stds


def build_payload(row: dict) -> dict:
    """Matches PredictionRequest in src/prediction/schemas.py exactly."""
    return {"features": row}


def generate_batch(
    mode: str,
    n: int,
    reference_df: pd.DataFrame,
    means: pd.Series,
    stds: pd.Series,
    rng: np.random.Generator,
) -> list[dict]:
    # Bootstrap n real rows with replacement as the base for every mode.
    sampled = reference_df.sample(
        n=n, replace=True, random_state=rng.integers(0, 2**31 - 1)
    ).reset_index(drop=True)

    if mode == "normal":
        result_df = sampled

    elif mode == "mild":
        shift = MILD_SHIFT_STDS * stds
        result_df = sampled + shift

    elif mode == "heavy":
        shift = HEAVY_SHIFT_STDS * stds
        # inflate variance: deviation-from-mean * multiplier, then re-add the (shifted) mean
        deviation = sampled - means
        inflated = deviation * HEAVY_VARIANCE_MULTIPLIER
        result_df = means + shift + inflated

    else:
        raise ValueError(f"Unknown mode: {mode}")

    rows = result_df.to_dict(orient="records")
    return [build_payload(row) for row in rows]


def main():
    parser = argparse.ArgumentParser(description="Generate a synthetic prediction batch from real reference data")
    parser.add_argument("--mode", choices=["normal", "mild", "heavy"], required=True)
    parser.add_argument("--n", type=int, default=100, help="number of rows to generate")
    parser.add_argument("--out", required=True, help="output JSON file path")
    parser.add_argument("--reference", default="data/raw/reference.parquet", help="path to reference.parquet")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    reference_df, means, stds = load_reference_stats(Path(args.reference))

    batch = generate_batch(args.mode, args.n, reference_df, means, stds, rng)

    with open(args.out, "w") as f:
        json.dump(batch, f, indent=2)

    print(f"Wrote {args.n} '{args.mode}' rows to {args.out}")
    if args.mode != "normal":
        shift_stds = MILD_SHIFT_STDS if args.mode == "mild" else HEAVY_SHIFT_STDS
        print(f"  (mean shift: +{shift_stds} std-devs per feature, relative to reference data)")


if __name__ == "__main__":
    main()