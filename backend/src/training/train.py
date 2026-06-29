import argparse
import logging
from pathlib import Path

import mlflow
import mlflow.xgboost
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


PARAMS: dict[str, int | float | str] = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma": 0.1,
    "reg_alpha": 0.1,       # L1
    "reg_lambda": 1.0,      # L2
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "use_label_encoder": False,
    "random_state": 42,
    "n_jobs": -1,
}


# ── Data loading ─────────────────────────────────────────────────────────────

def load_split(data_dir: Path, split: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load a Parquet split and return (X, y)."""
    path = data_dir / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Expected {path}. Run `python -m src.training.generate_data` first."
        )
    df = pd.read_parquet(path)
    X = df.drop(columns=["target"])
    y = df["target"]
    logger.info("Loaded %-10s split — %d rows, %d features", split, len(df), X.shape[1])
    return X, y


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(
    model: XGBClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    prefix: str,
) -> dict[str, float]:
    """
    Compute AUC, F1 (macro), and accuracy for a given split.

    Parameters
    ----------
    prefix:
        Short label prepended to metric keys, e.g. "val" → "val_auc".
    """
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)

    metrics = {
        f"{prefix}_auc": roc_auc_score(y, y_prob),
        f"{prefix}_f1": f1_score(y, y_pred, average="macro"),
        f"{prefix}_accuracy": accuracy_score(y, y_pred),
    }

    for key, value in metrics.items():
        logger.info("  %-22s %.4f", key, value)

    return metrics


# ── Training ─────────────────────────────────────────────────────────────────

def train(
    data_dir: Path,
    experiment_name: str,
    mlflow_tracking_uri: str,
    registered_model_name: str,
) -> str:
    """
    Run a single training experiment and return the MLflow run ID.

    Steps
    -----
    1. Load train / val / test splits.
    2. Fit XGBClassifier with early stopping on the val set.
    3. Evaluate on val and test.
    4. Log everything to MLflow (params, metrics, model, reference artifact).
    5. Register the model in the MLflow Model Registry as `Staging`.
    """
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    # ── Load splits ──────────────────────────────────────────────────────────
    X_train, y_train = load_split(data_dir, "train")
    X_val,   y_val   = load_split(data_dir, "val")
    X_test,  y_test  = load_split(data_dir, "test")

    reference_path = data_dir / "reference.parquet"
    if not reference_path.exists():
        raise FileNotFoundError(
            f"reference.parquet not found at {reference_path}. "
            "Run generate_data.py first."
        )

    # ── MLflow run ───────────────────────────────────────────────────────────
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info("MLflow run started — run_id: %s", run_id)

        # 1. Log all hyperparameters
        for key, value in PARAMS.items():
            mlflow.log_param(key, value)
        mlflow.log_param("data_dir", str(data_dir))
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("val_rows", len(X_val))
        mlflow.log_param("test_rows", len(X_test))
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("feature_names", list(X_train.columns))

        # 2. Fit the model with early stopping evaluated on the val set
        model = XGBClassifier(**PARAMS)

        logger.info("Fitting XGBClassifier …")
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # 3. Evaluate
        logger.info("Val metrics:")
        val_metrics = compute_metrics(model, X_val, y_val, prefix="val")

        logger.info("Test metrics:")
        test_metrics = compute_metrics(model, X_test, y_test, prefix="test")

        all_metrics = {**val_metrics, **test_metrics}

        # 4. Log metrics to MLflow
        for key, value in all_metrics.items():
            mlflow.log_metric(key, value)

        # 5. Log model artifact
        # input_example lets the MLflow UI show a sample request schema
        input_example = X_train.iloc[:3]

        model_info = mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
            input_example=input_example,
            registered_model_name=registered_model_name,
        )
        logger.info("Model artifact logged — URI: %s", model_info.model_uri)

        # 6. Log reference.parquet so the drift engine can always retrieve
        #    the baseline directly from the MLflow artifact store, even if
        #    the local data/ directory is unavailable (e.g. inside Docker).
        mlflow.log_artifact(str(reference_path), artifact_path="reference_data")
        logger.info("Reference baseline logged as artifact.")

        # 7. Log feature importance (useful for the dashboard's Model page)
        importance = pd.Series(
            model.feature_importances_,
            index=X_train.columns,
            name="importance",
        ).sort_values(ascending=False)
        importance_path = data_dir / "feature_importance.csv"
        importance.to_csv(importance_path, header=True)
        mlflow.log_artifact(str(importance_path), artifact_path="analysis")
        logger.info("Feature importance:\n%s", importance.to_string())

    logger.info("MLflow run complete — run_id: %s", run_id)
    return run_id


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train XGBoost model and log run to MLflow."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing train/val/test/reference Parquet files",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="ml-monitoring-platform",
        help="MLflow experiment name (created if it doesn't exist)",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default="http://localhost:5000",
        help="MLflow tracking server URI",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="drift-detector-model",
        help="Name to register the model under in the MLflow Model Registry",
    )
    args = parser.parse_args()

    run_id = train(
        data_dir=args.data_dir,
        experiment_name=args.experiment,
        mlflow_tracking_uri=args.tracking_uri,
        registered_model_name=args.model_name,
    )
    logger.info("Done. To inspect: http://localhost:5000 — run_id: %s", run_id)


if __name__ == "__main__":
    main()