"""FastAPI application for cat vs dog image classification."""

import io
import os
import time

import structlog
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.responses import Response

from src.serving.middleware import (
    MAX_BATCH_SIZE,
    RateLimitMiddleware,
    RequestIDMiddleware,
    validate_image_upload,
)
from src.serving.schemas import (
    BatchPredictionResponse,
    ErrorResponse,
    HealthResponse,
    PredictionResponse,
)
from src.serving.predict import Predictor

logger = structlog.get_logger()

# --- Prometheus Metrics ---
PREDICTION_REQUESTS = Counter(
    "prediction_requests_total",
    "Total prediction requests",
    ["predicted_class", "status"],
)
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction latency in seconds",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0],
)
PREDICTION_CONFIDENCE = Histogram(
    "prediction_confidence",
    "Prediction confidence distribution",
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0],
)
ACTIVE_REQUESTS = Gauge("active_requests", "Number of active requests")
MODEL_INFO = Gauge("model_info", "Model information", ["version"])

# --- App Setup ---
MODEL_PATH = os.environ.get("MODEL_PATH", "models/model.onnx")
START_TIME = time.time()

app = FastAPI(
    title="Cat vs Dog Classifier API",
    description="Binary image classifier using MobileNetV2 + ONNX Runtime",
    version="1.0.0",
)

# Middleware (order matters: first added = outermost)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup
predictor: Predictor | None = None


@app.on_event("startup")
async def startup() -> None:
    global predictor  # noqa: PLW0603
    try:
        predictor = Predictor(MODEL_PATH)
        MODEL_INFO.labels(version=predictor.model_version).set(1)
        logger.info("Model loaded", path=MODEL_PATH, version=predictor.model_version)
    except FileNotFoundError:
        logger.error("Model file not found", path=MODEL_PATH)


# --- Exception Handler ---
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):  # type: ignore[no-untyped-def]
    request_id = getattr(request.state, "request_id", None)
    logger.error("Unhandled exception", error=str(exc), request_id=request_id)
    error = ErrorResponse(
        error="Internal server error",
        detail="An unexpected error occurred.",
        request_id=request_id,
    )
    return JSONResponse(status_code=500, content=error.model_dump())


# --- Endpoints ---


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse | JSONResponse:
    """Classify a single image as cat or dog."""
    ACTIVE_REQUESTS.inc()
    try:
        if predictor is None:
            return JSONResponse(
                status_code=503,
                content=ErrorResponse(
                    error="Service unavailable", detail="Model not loaded."
                ).model_dump(),
            )

        # Validate input
        data, error = await validate_image_upload(file)
        if error:
            PREDICTION_REQUESTS.labels(predicted_class="none", status="invalid").inc()
            return JSONResponse(
                status_code=422,
                content=ErrorResponse(error="Invalid input", detail=error).model_dump(),
            )

        # Run inference
        image = Image.open(io.BytesIO(data))
        result = predictor.predict(image)

        # Record metrics
        PREDICTION_REQUESTS.labels(
            predicted_class=result["predicted_class"], status="success"
        ).inc()
        PREDICTION_LATENCY.observe(result["latency_ms"] / 1000)
        PREDICTION_CONFIDENCE.observe(result["confidence"])

        logger.info(
            "prediction",
            predicted_class=result["predicted_class"],
            confidence=round(result["confidence"], 4),
            latency_ms=result["latency_ms"],
        )
        return PredictionResponse(**result)
    finally:
        ACTIVE_REQUESTS.dec()


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    files: list[UploadFile] = File(...),
) -> BatchPredictionResponse | JSONResponse:
    """Classify multiple images in a single request."""
    ACTIVE_REQUESTS.inc()
    try:
        if predictor is None:
            return JSONResponse(
                status_code=503,
                content=ErrorResponse(
                    error="Service unavailable", detail="Model not loaded."
                ).model_dump(),
            )

        if len(files) > MAX_BATCH_SIZE:
            return JSONResponse(
                status_code=422,
                content=ErrorResponse(
                    error="Invalid input",
                    detail=f"Maximum {MAX_BATCH_SIZE} images per batch.",
                ).model_dump(),
            )

        # Validate and load all images
        images: list[Image.Image] = []
        for f in files:
            data, error = await validate_image_upload(f)
            if error:
                PREDICTION_REQUESTS.labels(predicted_class="none", status="invalid").inc()
                return JSONResponse(
                    status_code=422,
                    content=ErrorResponse(
                        error="Invalid input", detail=f"{f.filename}: {error}"
                    ).model_dump(),
                )
            images.append(Image.open(io.BytesIO(data)))

        # Run batch inference
        start = time.perf_counter()
        results = predictor.predict_batch(images)
        total_ms = (time.perf_counter() - start) * 1000

        for r in results:
            PREDICTION_REQUESTS.labels(predicted_class=r["predicted_class"], status="success").inc()
            PREDICTION_LATENCY.observe(r["latency_ms"] / 1000)
            PREDICTION_CONFIDENCE.observe(r["confidence"])

        return BatchPredictionResponse(
            predictions=[PredictionResponse(**r) for r in results],
            total_latency_ms=round(total_ms, 2),
        )
    finally:
        ACTIVE_REQUESTS.dec()


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if predictor is not None else "unhealthy",
        model_loaded=predictor is not None,
        model_version=predictor.model_version if predictor else "none",
        uptime_seconds=round(time.time() - START_TIME, 1),
    )


@app.get("/metrics")
async def metrics() -> Response:
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type="text/plain; charset=utf-8")
