"""API middleware: rate limiting, request ID, input validation."""

import time
import uuid
from collections import defaultdict
from typing import Any

from fastapi import Request, Response, UploadFile
from starlette.middleware.base import BaseHTTPMiddleware

from src.serving.schemas import ErrorResponse

# --- Constants ---
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg"}
MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB
MAX_BATCH_SIZE = 16
RATE_LIMIT_PER_MINUTE = 100


# --- Input Validation ---


async def validate_image_upload(file: UploadFile) -> tuple[bytes, str | None]:
    """Validate an uploaded file is a valid image.

    Returns (file_bytes, error_message). error_message is None if valid.
    """
    # Check content type
    content_type = file.content_type or ""
    if content_type not in ALLOWED_CONTENT_TYPES:
        return b"", f"Invalid file type: '{content_type}'. Allowed: JPEG, PNG."

    # Read and check size
    data = await file.read()
    if len(data) == 0:
        return b"", "Empty file uploaded."
    if len(data) > MAX_FILE_SIZE_BYTES:
        size_mb = len(data) / (1024 * 1024)
        return b"", f"File too large: {size_mb:.1f} MB. Maximum: 5 MB."

    return data, None


# --- Request ID Middleware ---


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Add a unique request ID to every request/response."""

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


# --- Rate Limiting Middleware ---


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiter per client IP."""

    def __init__(self, app: Any, max_requests: int = RATE_LIMIT_PER_MINUTE) -> None:
        super().__init__(app)
        self.max_requests = max_requests
        self.requests: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        window_start = now - 60.0

        # Clean old entries and add current request
        self.requests[client_ip] = [t for t in self.requests[client_ip] if t > window_start]

        if len(self.requests[client_ip]) >= self.max_requests:
            error = ErrorResponse(
                error="Rate limit exceeded",
                detail=f"Maximum {self.max_requests} requests per minute.",
                request_id=getattr(request.state, "request_id", None),
            )
            return Response(
                content=error.model_dump_json(),
                status_code=429,
                media_type="application/json",
            )

        self.requests[client_ip].append(now)
        return await call_next(request)
