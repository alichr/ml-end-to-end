"""Pydantic models for API request/response schemas."""

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    """Response from a single image prediction."""

    predicted_class: str = Field(description="Predicted class: 'cat' or 'dog'")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence of the prediction")
    probabilities: dict[str, float] = Field(
        description="Class probabilities {'cat': ..., 'dog': ...}"
    )
    model_version: str = Field(description="Model version used for prediction")
    latency_ms: float = Field(ge=0.0, description="Inference latency in milliseconds")


class BatchPredictionResponse(BaseModel):
    """Response from batch image prediction."""

    predictions: list[PredictionResponse]
    total_latency_ms: float = Field(ge=0.0, description="Total batch latency in milliseconds")


class HealthResponse(BaseModel):
    """Response from health check endpoint."""

    status: str = Field(description="'healthy' or 'unhealthy'")
    model_loaded: bool
    model_version: str
    uptime_seconds: float = Field(ge=0.0)


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    detail: str
    request_id: str | None = None
