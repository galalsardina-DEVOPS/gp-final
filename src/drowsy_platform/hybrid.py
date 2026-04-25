from __future__ import annotations

from typing import Any

import numpy as np

from drowsy_platform.config import ServiceSettings
from drowsy_platform.connectivity import NetworkStatusProbe
from drowsy_platform.image_utils import decode_base64_image, encode_numpy_image
from drowsy_platform.models.aws_remote import AwsRemoteClassifier
from drowsy_platform.models.base import ModelUnavailableError, PredictionResult
from drowsy_platform.models.local import LocalClassifier


class HybridClassifier:
    def __init__(self, settings: ServiceSettings) -> None:
        self.settings = settings
        self.local_classifier = LocalClassifier(settings)
        self.remote_classifier = (
            AwsRemoteClassifier(
                endpoint_url=settings.remote_endpoint_url,
                timeout_seconds=settings.remote_timeout_seconds,
                api_key=settings.remote_api_key,
            )
            if settings.remote_endpoint_url
            else None
        )
        self.network_probe = NetworkStatusProbe(
            probe_url=settings.network_probe_url,
            timeout_seconds=settings.network_probe_timeout_seconds,
            ttl_seconds=settings.network_probe_ttl_seconds,
        )

    def is_online(self) -> bool:
        return self.network_probe.is_online()

    def predict_encoded(self, image_b64: str, metadata: dict[str, Any] | None = None) -> PredictionResult:
        image = decode_base64_image(image_b64)
        return self.predict_array(image=image, image_b64=image_b64, metadata=metadata)

    def predict_array(
        self,
        image: np.ndarray,
        image_b64: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PredictionResult:
        mode = self.settings.service_mode
        if mode == "remote_only":
            return self._predict_remote(image, image_b64=image_b64, metadata=metadata)
        if mode == "local_only":
            return self.local_classifier.predict(image, metadata=metadata)
        if mode != "hybrid":
            raise ModelUnavailableError(
                f"Unsupported SERVICE_MODE '{mode}'. Use 'hybrid', 'remote_only', or 'local_only'."
            )

        if self.remote_classifier and self.is_online():
            try:
                return self._predict_remote(image, image_b64=image_b64, metadata=metadata)
            except ModelUnavailableError:
                pass

        return self.local_classifier.predict(image, metadata=metadata)

    def _predict_remote(
        self,
        image: np.ndarray,
        image_b64: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PredictionResult:
        if not self.remote_classifier:
            raise ModelUnavailableError("REMOTE_ENDPOINT_URL is not configured.")
        encoded = image_b64 or encode_numpy_image(image)
        return self.remote_classifier.predict_encoded(encoded, metadata=metadata)

