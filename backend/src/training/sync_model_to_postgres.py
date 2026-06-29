"""
sync_model_to_postgres.py
--------------------------
Reads model versions from the MLflow registry and writes/updates matching
rows in the Postgres metric store's model_versions table, so the
dashboard's /models endpoint reflects what's actually registered.

This is intentionally a separate, standalone script rather than baked
into train.py — it can be re-run any time (e.g. after manually promoting
a model to Production in the MLflow UI) to resync without retraining.

Usage:
    python -m src.training.sync_model_to_postgres
    python -m src.training.sync_model_to_postgres --model-name drift-detector-model
"""

import argparse
import logging

import mlflow

from src.monitoring.database import SessionLocal
from src.monitoring import repository as repo
from src.training.model_loader import DEFAULT_MODEL_NAME, DEFAULT_TRACKING_URI, get_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def sync_all_versions(
    model_name: str = DEFAULT_MODEL_NAME,
    tracking_uri: str = DEFAULT_TRACKING_URI,
) -> int:
    """
    Fetch every version of *model_name* from the MLflow registry and
    upsert each one into Postgres's model_versions table.

    Returns the number of versions synced.
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = get_client(tracking_uri)

    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        logger.warning("No versions found for model '%s' in MLflow registry.", model_name)
        return 0

    db = SessionLocal()
    synced = 0
    try:
        for v in versions:
            # Pull the run's logged metrics, if the run still exists
            training_metrics: dict = {}
            try:
                run = client.get_run(v.run_id)
                training_metrics = dict(run.data.metrics)
            except Exception as exc:
                logger.warning(
                    "Could not fetch metrics for run %s (version %s): %s",
                    v.run_id, v.version, exc,
                )

            repo.upsert_model_version(
                db,
                name=v.name,
                version=str(v.version),
                stage=v.current_stage,
                training_metrics=training_metrics,
                mlflow_run_id=v.run_id,
            )
            logger.info(
                "Synced '%s' v%s (stage: %s, %d metrics)",
                v.name, v.version, v.current_stage, len(training_metrics),
            )
            synced += 1
    finally:
        db.close()

    return synced


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync MLflow registered model versions into Postgres."
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    args = parser.parse_args()

    count = sync_all_versions(model_name=args.model_name, tracking_uri=args.tracking_uri)
    logger.info("Done — synced %d version(s) to Postgres.", count)


if __name__ == "__main__":
    main()