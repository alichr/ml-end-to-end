"""Integration tests for the FastAPI application."""

import io
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.serving import app as app_module
from src.serving.app import app


def _make_jpeg_bytes(width: int = 100, height: int = 100) -> bytes:
    """Create JPEG bytes for a random image."""
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture()
def client(mock_predictor: MagicMock) -> TestClient:
    """Create a test client with a mocked predictor."""
    app_module.predictor = mock_predictor
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True

    def test_health_when_no_model(self) -> None:
        app_module.predictor = None
        c = TestClient(app)
        resp = c.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "unhealthy"
        assert data["model_loaded"] is False


class TestPredictEndpoint:
    def test_predict_returns_200_on_valid_image(self, client: TestClient) -> None:
        data = _make_jpeg_bytes()
        resp = client.post("/predict", files={"file": ("test.jpg", data, "image/jpeg")})
        assert resp.status_code == 200
        result = resp.json()
        assert result["predicted_class"] in ("cat", "dog")
        assert 0 <= result["confidence"] <= 1
        assert "cat" in result["probabilities"]
        assert "dog" in result["probabilities"]

    def test_predict_returns_correct_schema(self, client: TestClient) -> None:
        data = _make_jpeg_bytes()
        resp = client.post("/predict", files={"file": ("test.jpg", data, "image/jpeg")})
        result = resp.json()
        assert "predicted_class" in result
        assert "confidence" in result
        assert "probabilities" in result
        assert "model_version" in result
        assert "latency_ms" in result

    def test_predict_returns_422_on_invalid_file_type(self, client: TestClient) -> None:
        resp = client.post("/predict", files={"file": ("test.txt", b"not an image", "text/plain")})
        assert resp.status_code == 422

    def test_predict_returns_422_on_empty_file(self, client: TestClient) -> None:
        resp = client.post("/predict", files={"file": ("test.jpg", b"", "image/jpeg")})
        assert resp.status_code == 422

    def test_predict_returns_422_on_oversized_file(self, client: TestClient) -> None:
        big_data = b"\x00" * (6 * 1024 * 1024)  # 6 MB
        resp = client.post("/predict", files={"file": ("big.jpg", big_data, "image/jpeg")})
        assert resp.status_code == 422

    def test_predict_returns_503_when_no_model(self) -> None:
        app_module.predictor = None
        c = TestClient(app)
        data = _make_jpeg_bytes()
        resp = c.post("/predict", files={"file": ("test.jpg", data, "image/jpeg")})
        assert resp.status_code == 503

    def test_predict_png_accepted(self, client: TestClient) -> None:
        arr = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        resp = client.post("/predict", files={"file": ("test.png", buf.getvalue(), "image/png")})
        assert resp.status_code == 200


class TestBatchPredictEndpoint:
    def test_batch_predict_multiple_images(self, client: TestClient) -> None:
        files = [
            ("files", ("img1.jpg", _make_jpeg_bytes(), "image/jpeg")),
            ("files", ("img2.jpg", _make_jpeg_bytes(), "image/jpeg")),
        ]
        resp = client.post("/predict/batch", files=files)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["predictions"]) == 2
        assert "total_latency_ms" in data

    def test_batch_predict_rejects_too_many(self, client: TestClient) -> None:
        files = [
            ("files", (f"img{i}.jpg", _make_jpeg_bytes(), "image/jpeg"))
            for i in range(20)  # > MAX_BATCH_SIZE (16)
        ]
        resp = client.post("/predict/batch", files=files)
        assert resp.status_code == 422


class TestMetricsEndpoint:
    def test_metrics_returns_200(self, client: TestClient) -> None:
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "prediction_requests_total" in resp.text


class TestRequestID:
    def test_response_has_request_id(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert "X-Request-ID" in resp.headers
