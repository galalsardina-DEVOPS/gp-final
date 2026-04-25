from __future__ import annotations

from pathlib import Path
import sys

from fastapi import FastAPI, HTTPException

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drowsy_platform.config import ServiceSettings
from drowsy_platform.hybrid import HybridClassifier
from drowsy_platform.models.base import ModelUnavailableError
from drowsy_platform.schemas import HealthResponse, PredictRequest, PredictResponse

settings = ServiceSettings.from_env()
classifier = HybridClassifier(settings)

app = FastAPI(title="Hybrid Drowsy Driver API", version="1.0.0")


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    return HealthResponse(
        service_mode=settings.service_mode,
        local_model_backend=settings.local_model_backend,
        local_model_path=str(settings.local_model_path),
        remote_endpoint_url=settings.remote_endpoint_url,
        online=classifier.is_online(),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    try:
        result = classifier.predict_encoded(payload.image_b64, metadata=payload.metadata)
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return PredictResponse(
        label=result.label,
        score=result.score,
        threshold=result.threshold,
        route=result.route,
        backend=result.backend,
        online=result.online,
    )
