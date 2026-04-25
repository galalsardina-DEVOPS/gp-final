from __future__ import annotations

from typing import Any

import requests

from drowsy_platform.models.base import ModelUnavailableError, PredictionResult


class AwsRemoteClassifier:
    def __init__(self, endpoint_url: str, timeout_seconds: float, api_key: str | None = None) -> None:
        self.endpoint_url = endpoint_url
        self.timeout_seconds = timeout_seconds
        self.api_key = api_key

    def predict_encoded(self, image_b64: str, metadata: dict[str, Any] | None = None) -> PredictionResult:
        headers: dict[str, str] = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        payload = {
            "image_b64": image_b64,
            "metadata": metadata or {},
        }

        try:
            response = requests.post(
                self.endpoint_url,
                json=payload,
                headers=headers,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise ModelUnavailableError(f"Remote AWS inference failed: {exc}") from exc

        data = response.json()
        return PredictionResult(
            label=data["label"],
            score=float(data["score"]),
            threshold=float(data["threshold"]),
            route="remote",
            backend=data.get("backend", "aws-remote"),
            online=True,
        )

