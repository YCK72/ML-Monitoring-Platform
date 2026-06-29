import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── Feature names ────────────────────────────────────────────────────────────

# 10 informative + 3 redundant + 2 repeated = 15 total features.
# Naming them semantically makes drift reports much easier to read.
FEATURE_NAMES: list[str] = [
    # Informative (10)
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
    # Redundant — linear combos of informative features (3)
    "feature_derived_a",
    "feature_derived_b",
    "feature_derived_c",
    # Repeated — exact copies of informative features (2)
    "feature_copy_income",
    "feature_copy_age",
]

assert len(FEATURE_NAMES) == 15, "Must have exactly 15 feature names"


# ── Core generation logic ────────────────────────────────────────────────────

def generate_dataset(
    n_samples: int = 10_000,
    random_state: int = 42,
) -> pd.DataFrame:

    logger.info(
        "Generating %d samples with 15 features (10 informative, "
        "3 redundant, 2 repeated) …",
        n_samples,
    )

    X, y = make_classification(
        n_samples=n_samples,
        n_features=15,
        n_informative=10,
        n_redundant=3,
        n_repeated=2,
        n_classes=2,
        class_sep=1.0,       # moderate separability — not trivially easy
        flip_y=0.03,         # 3 % label noise for realism
        random_state=random_state,
    )

    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["target"] = y.astype(int)

    logger.info(
        "Dataset shape: %s | class balance: %.1f%% positive",
        df.shape,
        df["target"].mean() * 100,
    )
    return df


def split_dataset(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split into train / val / test preserving class balance (stratified).

    Default split: 70 % train, 15 % val, 15 % test.
    """
    test_frac = round(1.0 - train_frac - val_frac, 10)
    if test_frac <= 0:
        raise ValueError(
            f"train_frac ({train_frac}) + val_frac ({val_frac}) must be < 1.0"
        )

    # First cut: train vs (val + test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(val_frac + test_frac),
        stratify=df["target"],
        random_state=random_state,
    )

    # Second cut: val vs test, keeping equal halves of the remainder
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_frac / (val_frac + test_frac),
        stratify=temp_df["target"],
        random_state=random_state,
    )

    logger.info(
        "Split sizes — train: %d | val: %d | test: %d",
        len(train_df),
        len(val_df),
        len(test_df),
    )
    return train_df, val_df, test_df


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    """
    Persist each split as a Parquet file and copy the training split as
    reference.parquet (the drift baseline).

    Returns a mapping of logical name → file path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {
        "train": output_dir / "train.parquet",
        "val": output_dir / "val.parquet",
        "test": output_dir / "test.parquet",
        "reference": output_dir / "reference.parquet",
    }

    train_df.to_parquet(paths["train"], index=False)
    val_df.to_parquet(paths["val"], index=False)
    test_df.to_parquet(paths["test"], index=False)

    # reference.parquet is identical to train.parquet at generation time.
    # It is intentionally kept as a separate file so it can be versioned
    # and replaced independently (e.g. after model retraining).
    train_df.to_parquet(paths["reference"], index=False)

    for name, path in paths.items():
        size_kb = path.stat().st_size / 1024
        logger.info("Saved %-12s → %s  (%.1f KB)", name, path, size_kb)

    return paths


def validate_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """
    Lightweight sanity checks. Raises AssertionError on failure.
    """
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        assert not df.isnull().any().any(), f"{name} split contains nulls"
        assert "target" in df.columns, f"{name} split missing 'target' column"
        assert set(df["target"].unique()).issubset(
            {0, 1}
        ), f"{name} split has unexpected target values"
        assert list(df.columns[:-1]) == FEATURE_NAMES, (
            f"{name} split has unexpected feature columns"
        )

    total = len(train_df) + len(val_df) + len(test_df)
    assert total > 0, "Total dataset is empty"

    logger.info("All validation checks passed ✓")


# ── CLI entry point ──────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic dataset for the ML monitoring platform."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory where Parquet files will be written (default: data/raw)",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=10_000,
        help="Total number of rows to generate (default: 10000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    df = generate_dataset(n_samples=args.rows, random_state=args.seed)
    train_df, val_df, test_df = split_dataset(df, random_state=args.seed)
    validate_splits(train_df, val_df, test_df)
    paths = save_splits(train_df, val_df, test_df, args.output_dir)

    logger.info("Done. Reference baseline saved to: %s", paths["reference"])


if __name__ == "__main__":
    main()