"""Shared test fixtures."""

import io
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image


@pytest.fixture()
def sample_image() -> Image.Image:
    """Create a small random RGB image for testing."""
    arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    return Image.fromarray(arr)


@pytest.fixture()
def sample_image_bytes(sample_image: Image.Image) -> bytes:
    """Return sample image as JPEG bytes."""
    buf = io.BytesIO()
    sample_image.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture()
def mock_predictor() -> MagicMock:
    """Return a mock Predictor that returns a canned result."""
    predictor = MagicMock()
    predictor.model_version = "1.0.0-test"
    predictor.predict.return_value = {
        "predicted_class": "cat",
        "confidence": 0.95,
        "probabilities": {"cat": 0.95, "dog": 0.05},
        "model_version": "1.0.0-test",
        "latency_ms": 5.0,
    }
    predictor.predict_batch.return_value = [
        {
            "predicted_class": "cat",
            "confidence": 0.95,
            "probabilities": {"cat": 0.95, "dog": 0.05},
            "model_version": "1.0.0-test",
            "latency_ms": 2.5,
        },
        {
            "predicted_class": "dog",
            "confidence": 0.88,
            "probabilities": {"cat": 0.12, "dog": 0.88},
            "model_version": "1.0.0-test",
            "latency_ms": 2.5,
        },
    ]
    return predictor
