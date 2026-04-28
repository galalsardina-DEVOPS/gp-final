from __future__ import annotations

from pathlib import Path
import sys

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drowsy_platform.config import ServiceSettings
from drowsy_platform.event_logger import DrowsyEvent, EventLogger
from drowsy_platform.hybrid import HybridClassifier
from drowsy_platform.models.base import ModelUnavailableError
from drowsy_platform.schemas import HealthResponse, PredictRequest, PredictResponse

settings = ServiceSettings.from_env()
classifier = HybridClassifier(settings)
event_logger = EventLogger(settings.event_db_path)

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

    event_logger.log(
        DrowsyEvent(
            source=payload.source_id or "api",
            label=result.label,
            score=result.score,
            route=result.route,
            backend=result.backend,
            metadata=payload.metadata,
        )
    )

    return PredictResponse(
        label=result.label,
        score=result.score,
        threshold=result.threshold,
        route=result.route,
        backend=result.backend,
        online=result.online,
    )


@app.get("/metrics")
def metrics() -> dict:
    return event_logger.summary()


@app.get("/events")
def events(limit: int = 50) -> list[dict]:
    return event_logger.recent(limit=limit)


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> str:
    summary = event_logger.summary()
    recent_events = event_logger.recent(limit=20)
    rows = "\n".join(
        f"""
        <tr>
          <td>{event['created_at']}</td>
          <td>{event['label']}</td>
          <td>{event.get('route') or '-'}</td>
          <td>{event.get('score') if event.get('score') is not None else '-'}</td>
          <td>{event.get('ear') if event.get('ear') is not None else '-'}</td>
          <td>{event.get('mar') if event.get('mar') is not None else '-'}</td>
          <td>{event.get('s3_uri') or event.get('frame_path') or '-'}</td>
        </tr>
        """
        for event in recent_events
    )

    return f"""
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <meta http-equiv="refresh" content="5">
        <title>Drowsy Driver Dashboard</title>
        <style>
          body {{
            margin: 0;
            font-family: Verdana, Geneva, sans-serif;
            background: #f5f7f2;
            color: #172017;
          }}
          main {{
            max-width: 1180px;
            margin: 0 auto;
            padding: 28px;
          }}
          h1 {{
            margin: 0 0 18px;
            font-size: 28px;
          }}
          .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 12px;
            margin-bottom: 22px;
          }}
          .card {{
            border: 1px solid #ccd6c8;
            border-radius: 8px;
            padding: 16px;
            background: #ffffff;
          }}
          .value {{
            display: block;
            margin-top: 8px;
            font-size: 32px;
            font-weight: 700;
          }}
          table {{
            width: 100%;
            border-collapse: collapse;
            background: #ffffff;
            border: 1px solid #ccd6c8;
          }}
          th, td {{
            padding: 10px;
            border-bottom: 1px solid #e2e7df;
            text-align: left;
            font-size: 13px;
            word-break: break-word;
          }}
          th {{
            background: #e9efe5;
          }}
        </style>
      </head>
      <body>
        <main>
          <h1>Drowsy Driver Dashboard</h1>
          <section class="grid">
            <div class="card">Total events<span class="value">{summary['total_events']}</span></div>
            <div class="card">Drowsy events<span class="value">{summary['drowsy_events']}</span></div>
            <div class="card">Alert predictions<span class="value">{summary['alert_events']}</span></div>
            <div class="card">Remote route<span class="value">{summary['remote_events']}</span></div>
            <div class="card">Local route<span class="value">{summary['local_events']}</span></div>
          </section>
          <table>
            <thead>
              <tr>
                <th>Time</th>
                <th>Label</th>
                <th>Route</th>
                <th>Score</th>
                <th>EAR</th>
                <th>MAR</th>
                <th>Frame / S3</th>
              </tr>
            </thead>
            <tbody>{rows}</tbody>
          </table>
        </main>
      </body>
    </html>
    """
