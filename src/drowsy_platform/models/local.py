from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from drowsy_platform.config import ServiceSettings
from drowsy_platform.image_utils import preprocess_for_model
from drowsy_platform.models.base import ModelUnavailableError, PredictionResult


class LocalClassifier:
    def __init__(self, settings: ServiceSettings) -> None:
        self.settings = settings
        self.backend = settings.local_model_backend
        self.model_path = settings.local_model_path
        self.threshold = settings.local_model_threshold
        self._model: Any | None = None
        self._interpreter: Any | None = None

    def predict(self, image: np.ndarray, metadata: dict[str, Any] | None = None) -> PredictionResult:
        if not self.model_path.exists():
            raise ModelUnavailableError(f"Local model not found at {self.model_path}")

        if self.backend == "keras":
            score = self._predict_keras(image)
        elif self.backend == "tflite":
            score = self._predict_tflite(image)
        else:
            raise ModelUnavailableError(
                f"Unsupported LOCAL_MODEL_BACKEND '{self.backend}'. Use 'keras' or 'tflite'."
            )

        label = "drowsy" if score >= self.threshold else "alert"
        return PredictionResult(
            label=label,
            score=score,
            threshold=self.threshold,
            route="local",
            backend=f"local-{self.backend}",
            online=False,
        )

    def _predict_keras(self, image: np.ndarray) -> float:
        if self._model is None:
            import tensorflow as tf

            self._model = self._load_keras_model(tf)

        batch = np.expand_dims(preprocess_for_model(image, self.settings.input_size), axis=0)
        prediction = self._model.predict(batch, verbose=0)
        return float(np.squeeze(prediction))

    def _load_keras_model(self, tf: Any) -> Any:
        try:
            return tf.keras.models.load_model(self.model_path)
        except Exception as exc:
            if "DepthwiseConv2D" not in str(exc) or "groups" not in str(exc):
                raise

        class CompatibleDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                kwargs.pop("groups", None)
                super().__init__(*args, **kwargs)

        return tf.keras.models.load_model(
            self.model_path,
            custom_objects={"DepthwiseConv2D": CompatibleDepthwiseConv2D},
            compile=False,
        )

    def _predict_tflite(self, image: np.ndarray) -> float:
        if self._interpreter is None:
            self._interpreter = self._build_tflite_interpreter(self.model_path)
            self._interpreter.allocate_tensors()

        input_details = self._interpreter.get_input_details()[0]
        output_details = self._interpreter.get_output_details()[0]
        batch = np.expand_dims(preprocess_for_model(image, self.settings.input_size), axis=0).astype(
            input_details["dtype"]
        )

        self._interpreter.set_tensor(input_details["index"], batch)
        self._interpreter.invoke()
        prediction = self._interpreter.get_tensor(output_details["index"])
        return float(np.squeeze(prediction))

    @staticmethod
    def _build_tflite_interpreter(model_path: Path) -> Any:
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            import tensorflow as tf

            Interpreter = tf.lite.Interpreter

        return Interpreter(model_path=str(model_path))
