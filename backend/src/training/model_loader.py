"""
model_loader.py
---------------
Two responsibilities:

1.  Registry helper — transition a registered model version between
    MLflow lifecycle stages (None → Staging → Production → Archived).
    Called once after training to promote a freshly registered version.

2.  Model loader — load the latest Production (or Staging fallback) model
    from the MLflow registry and cache it in memory so every prediction
    request reuses the same object without hitting the tracking server
    again.

Usage (registry promotion after training):
    python -m src.training.model_loader \
        --model-name drift-detector-model \
        --version 1 \
        --stage Staging

Usage (as an import inside the prediction service):
    from src.training.model_loader import get_model
    model = get_model()           # cached after first call
    proba = model.predict_proba(X)
"""

import argparse
import logging
import os
from functools import lru_cache
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── Defaults (overridden by environment variables in Docker) ─────────────────

DEFAULT_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
DEFAULT_MODEL_NAME   = os.getenv("MLFLOW_MODEL_NAME",   "drift-detector-model")

# Stage preference order: try Production first, fall back to Staging.
# This lets you promote a new version to Production without touching code.
_STAGE_PREFERENCE = ["Production", "Staging"]


# ── Registry helpers ─────────────────────────────────────────────────────────

def get_client(tracking_uri: str = DEFAULT_TRACKING_URI) -> MlflowClient:
    mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient()


def transition_model_version(
    model_name: str,
    version: int | str,
    target_stage: str,
    archive_existing: bool = True,
    tracking_uri: str = DEFAULT_TRACKING_URI,
) -> None:
    """
    Move *version* of *model_name* to *target_stage*.

    Parameters
    ----------
    model_name:
        Registered model name (e.g. "drift-detector-model").
    version:
        Integer version number returned by mlflow.xgboost.log_model.
    target_stage:
        One of "Staging", "Production", "Archived", "None".
    archive_existing:
        When True and target_stage is "Production", automatically archive
        any existing Production version so only one is live at a time.
    tracking_uri:
        MLflow tracking server URI. Defaults to MLFLOW_TRACKING_URI env var.
    """
    valid_stages = {"Staging", "Production", "Archived", "None"}
    if target_stage not in valid_stages:
        raise ValueError(f"target_stage must be one of {valid_stages}")

    client = get_client(tracking_uri)

    # Confirm the version exists before transitioning
    mv = client.get_model_version(name=model_name, version=str(version))
    current_stage = mv.current_stage
    logger.info(
        "Model '%s' v%s — current stage: %s → transitioning to: %s",
        model_name, version, current_stage, target_stage,
    )

    client.transition_model_version_stage(
        name=model_name,
        version=str(version),
        stage=target_stage,
        archive_existing_versions=archive_existing,
    )
    logger.info("Transition complete ✓")


def get_latest_version_number(
    model_name: str,
    tracking_uri: str = DEFAULT_TRACKING_URI,
) -> int:
    """Return the highest registered version number for *model_name*."""
    client = get_client(tracking_uri)
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise RuntimeError(
            f"No versions found for model '{model_name}'. "
            "Run train.py first."
        )
    latest = max(versions, key=lambda v: int(v.version))
    logger.info(
        "Latest version of '%s': v%s (stage: %s)",
        model_name, latest.version, latest.current_stage,
    )
    return int(latest.version)


def list_model_versions(
    model_name: str,
    tracking_uri: str = DEFAULT_TRACKING_URI,
) -> list[dict[str, Any]]:
    """
    Return a summary of all registered versions as a list of dicts.
    Used by the FastAPI /models endpoint on Day 7.
    """
    client = get_client(tracking_uri)
    versions = client.search_model_versions(f"name='{model_name}'")
    return [
        {
            "name": v.name,
            "version": int(v.version),
            "stage": v.current_stage,
            "run_id": v.run_id,
            "created_at": v.creation_timestamp,
        }
        for v in sorted(versions, key=lambda v: int(v.version), reverse=True)
    ]


# ── Cached model loader ───────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_model(
    model_name: str,
    stage: str,
    tracking_uri: str,
) -> Any:
    """
    Internal — loads and caches a model by (name, stage, uri).
    lru_cache means the model object is created exactly once per unique
    combination of arguments for the lifetime of the process.
    """
    mlflow.set_tracking_uri(tracking_uri)
    model_uri = f"models:/{model_name}/{stage}"
    logger.info("Loading model from registry — URI: %s", model_uri)
    model = mlflow.xgboost.load_model(model_uri)
    logger.info("Model loaded and cached ✓")
    return model


@lru_cache(maxsize=8)
def get_model(
    model_name: str = DEFAULT_MODEL_NAME,
    tracking_uri: str = DEFAULT_TRACKING_URI,
) -> Any:
    """
    Public loader used by the prediction service and drift consumer.

    Tries Production first; falls back to Staging if no Production version
    exists. Raises RuntimeError if neither stage has a registered version.

    This entire function is cached via @lru_cache (keyed on model_name +
    tracking_uri) — not just the model download. Without this, every call
    would re-query the MLflow registry's search_model_versions() API over
    the network even when the model is already loaded, adding ~1-2s of
    latency per prediction request. With caching, only the very first call
    per process hits the network; every subsequent call is an in-memory
    dict lookup.

    Example
    -------
    .. code-block:: python

        from src.training.model_loader import get_model
        model = get_model()
        proba = model.predict_proba(X_df)[:, 1]
    """
    client = get_client(tracking_uri)

    # search_model_versions replaces the deprecated get_latest_versions (MLflow 2.9+)
    all_versions = client.search_model_versions(f"name='{model_name}'")

    if not all_versions:
        raise RuntimeError(
            f"No model named '{model_name}' registered. "
            f"Run train.py first."
        )

    # Try Production then Staging, pick the highest version number in each stage
    for stage in _STAGE_PREFERENCE:
        in_stage = [v for v in all_versions if v.current_stage == stage]
        if in_stage:
            best = max(in_stage, key=lambda v: int(v.version))
            logger.info(
                "Found '%s' in stage '%s' (v%s)",
                model_name, stage, best.version,
            )
            return _load_model(model_name, stage, tracking_uri)

    # Neither Production nor Staging — tell the user which stages exist
    found_stages = {v.current_stage for v in all_versions}
    raise RuntimeError(
        f"No model named '{model_name}' found in Production or Staging. "
        f"Registered versions are in stages: {{found_stages}}. "
        f"Run: python -m src.training.model_loader --stage Staging"
    )


def invalidate_model_cache() -> None:
    """
    Force the next get_model() call to re-load from the registry.
    Call this after promoting a new model version to Production so the
    prediction service picks up the update without restarting.
    """
    get_model.cache_clear()
    _load_model.cache_clear()
    logger.info("Model cache cleared — next call will reload from registry.")


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transition a registered MLflow model to a target stage."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
    )
    parser.add_argument(
        "--version",
        type=int,
        default=None,
        help="Version number to transition. Defaults to the latest version.",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="Staging",
        choices=["Staging", "Production", "Archived", "None"],
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=DEFAULT_TRACKING_URI,
    )
    args = parser.parse_args()

    version = args.version or get_latest_version_number(
        args.model_name, args.tracking_uri
    )

    transition_model_version(
        model_name=args.model_name,
        version=version,
        target_stage=args.stage,
        tracking_uri=args.tracking_uri,
    )

    logger.info(
        "Done — '%s' v%s is now in stage '%s'.",
        args.model_name, version, args.stage,
    )


if __name__ == "__main__":
    main()