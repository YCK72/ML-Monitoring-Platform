"""
routers/models.py
------------------
GET /models        — list all model versions (from Postgres metric store)
GET /models/{id}    — single model version detail
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from src.api.schemas import ModelVersionListResponse, ModelVersionResponse
from src.monitoring import repository as repo
from src.monitoring.database import get_db

router = APIRouter()


@router.get("/models", response_model=ModelVersionListResponse)
def list_models(db: Session = Depends(get_db)) -> ModelVersionListResponse:
    """
    Return all model versions recorded in the metric store, newest first.

    Note: this reads from Postgres (populated by train.py via
    repo.upsert_model_version), not directly from the MLflow registry —
    keeping the API decoupled from MLflow availability.
    """
    versions = repo.list_model_versions(db)
    items = [ModelVersionResponse.model_validate(v) for v in versions]
    return ModelVersionListResponse(items=items, total=len(items))


@router.get("/models/{model_id}", response_model=ModelVersionResponse)
def get_model_detail(model_id: int, db: Session = Depends(get_db)) -> ModelVersionResponse:
    """Return a single model version's metadata, or 404 if not found."""
    version = repo.get_model_version(db, model_id)
    if version is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model version {model_id} not found",
        )
    return ModelVersionResponse.model_validate(version)