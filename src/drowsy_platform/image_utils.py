from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image


def decode_base64_image(image_b64: str) -> np.ndarray:
    raw = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.asarray(image)


def encode_numpy_image(image: np.ndarray) -> str:
    rgb_image = Image.fromarray(image.astype("uint8")).convert("RGB")
    buffer = io.BytesIO()
    rgb_image.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def preprocess_for_model(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    resized = Image.fromarray(image.astype("uint8")).convert("RGB").resize(size)
    normalized = np.asarray(resized, dtype="float32") / 255.0
    return normalized

