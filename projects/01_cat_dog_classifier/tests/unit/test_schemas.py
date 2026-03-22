"""Unit tests for API schemas."""

import pytest
from pydantic import ValidationError

from src.serving.schemas import (
    BatchPredictionResponse,
    ErrorResponse,
    HealthResponse,
    PredictionResponse,
)


class TestPredictionResponse:
    def test_valid_response(self) -> None:
        resp = PredictionResponse(
            predicted_class="cat",
            confidence=0.95,
            probabilities={"cat": 0.95, "dog": 0.05},
            model_version="1.0.0",
            latency_ms=5.0,
        )
        assert resp.predicted_class == "cat"
        assert resp.confidence == 0.95

    def test_confidence_must_be_between_0_and_1(self) -> None:
        with pytest.raises(ValidationError):
            PredictionResponse(
                predicted_class="cat",
                confidence=1.5,
                probabilities={"cat": 1.5, "dog": -0.5},
                model_version="1.0.0",
                latency_ms=5.0,
            )

    def test_negative_confidence_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PredictionResponse(
                predicted_class="dog",
                confidence=-0.1,
                probabilities={"cat": 0.0, "dog": -0.1},
                model_version="1.0.0",
                latency_ms=5.0,
            )

    def test_negative_latency_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PredictionResponse(
                predicted_class="cat",
                confidence=0.5,
                probabilities={"cat": 0.5, "dog": 0.5},
                model_version="1.0.0",
                latency_ms=-1.0,
            )


class TestHealthResponse:
    def test_healthy_response(self) -> None:
        resp = HealthResponse(
            status="healthy",
            model_loaded=True,
            model_version="1.0.0",
            uptime_seconds=120.5,
        )
        assert resp.status == "healthy"
        assert resp.model_loaded is True

    def test_unhealthy_response(self) -> None:
        resp = HealthResponse(
            status="unhealthy",
            model_loaded=False,
            model_version="none",
            uptime_seconds=0.0,
        )
        assert resp.status == "unhealthy"
        assert resp.model_loaded is False


class TestBatchPredictionResponse:
    def test_batch_response(self) -> None:
        pred = PredictionResponse(
            predicted_class="dog",
            confidence=0.9,
            probabilities={"cat": 0.1, "dog": 0.9},
            model_version="1.0.0",
            latency_ms=3.0,
        )
        batch = BatchPredictionResponse(predictions=[pred, pred], total_latency_ms=6.0)
        assert len(batch.predictions) == 2


class TestErrorResponse:
    def test_error_response(self) -> None:
        resp = ErrorResponse(error="Not found", detail="Image not found", request_id="abc-123")
        assert resp.error == "Not found"
        assert resp.request_id == "abc-123"

    def test_error_response_without_request_id(self) -> None:
        resp = ErrorResponse(error="Bad request", detail="Invalid input")
        assert resp.request_id is None
