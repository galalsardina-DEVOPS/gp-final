from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    image_b64: str = Field(..., description="Base64-encoded image payload.")
    source_id: str | None = Field(default=None, description="Camera or device identifier.")
    metadata: dict[str, Any] = Field(default_factory=dict)


class PredictResponse(BaseModel):
    label: Literal["alert", "drowsy"]
    score: float
    threshold: float
    route: Literal["local", "remote"]
    backend: str
    online: bool
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class HealthResponse(BaseModel):
    service_mode: str
    local_model_backend: str
    local_model_path: str
    remote_endpoint_url: str | None
    online: bool

