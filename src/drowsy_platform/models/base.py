from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


class ModelUnavailableError(RuntimeError):
    """Raised when a model backend cannot serve a prediction."""


@dataclass(slots=True)
class PredictionResult:
    label: str
    score: float
    threshold: float
    route: str
    backend: str
    online: bool


class ArrayModel(Protocol):
    def predict(self, image: np.ndarray, metadata: dict[str, Any] | None = None) -> PredictionResult:
        ...


class EncodedModel(Protocol):
    def predict_encoded(self, image_b64: str, metadata: dict[str, Any] | None = None) -> PredictionResult:
        ...

