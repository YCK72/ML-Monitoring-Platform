from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, model_validator


# ── REST request ─────────────────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    """
    Payload sent to POST /predict.

    features must be a flat dict of feature_name → float value.
    Example:
        {
          "features": {
            "feature_income": 0.52,
            "feature_age": -1.3,
            ...
          }
        }
    """

    features: dict[str, float] = Field(
        ...,
        description="Feature name → float value mapping. "
                    "Must contain all 15 features the model was trained on.",
        min_length=1,
    )

    @model_validator(mode="after")
    def check_no_nan(self) -> "PredictionRequest":
        bad = [k for k, v in self.features.items() if v != v]  # NaN check
        if bad:
            raise ValueError(f"NaN values not allowed. Offending keys: {bad}")
        return self


# ── Kafka event ───────────────────────────────────────────────────────────────

class PredictionEvent(BaseModel):
    """
    Structured event published to the 'prediction-events' Kafka topic
    for every inference request.

    The drift consumer deserialises this schema from JSON and writes it
    to PostgreSQL via the repository layer.
    """

    features: dict[str, float] = Field(
        ...,
        description="Raw input features exactly as received in the request.",
    )
    prediction: int = Field(
        ...,
        ge=0,
        le=1,
        description="Binary class prediction (0 or 1).",
    )
    probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Predicted probability of class 1.",
    )
    model_name: str = Field(
        ...,
        description="Registered model name (e.g. 'drift-detector-model').",
    )
    model_version: str = Field(
        ...,
        description="Model version string as returned by the MLflow registry.",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of the inference request.",
    )

    def to_json(self) -> str:
        """Serialise to a JSON string suitable for Kafka message value."""
        return self.model_dump_json()

    @classmethod
    def from_json(cls, raw: bytes | str) -> "PredictionEvent":
        """Deserialise from a Kafka message value."""
        return cls.model_validate_json(raw)


# ── REST response ─────────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    """
    JSON body returned by POST /predict.
    Deliberately minimal — callers only need the prediction and probability.
    """

    prediction: int = Field(..., description="0 or 1")
    probability: float = Field(..., description="P(class=1), range [0, 1]")
    model_version: str
    timestamp: datetime

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}


# ── Health check response ─────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    kafka_connected: bool
    model_loaded: bool
    details: dict[str, Any] = Field(default_factory=dict)