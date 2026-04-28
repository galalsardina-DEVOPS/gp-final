from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _int_env(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _float_env(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _optional_env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name, default)
    if value is None:
        return None
    value = value.strip()
    if value.lower() in {"", "none", "null"}:
        return None
    return value


@dataclass(frozen=True)
class ServiceSettings:
    service_mode: str
    remote_endpoint_url: str | None
    remote_api_key: str | None
    remote_timeout_seconds: float
    network_probe_url: str
    network_probe_timeout_seconds: float
    network_probe_ttl_seconds: int
    local_model_backend: str
    local_model_path: Path
    local_model_threshold: float
    model_input_width: int
    model_input_height: int
    frame_skip: int
    cnn_every_n_frames: int
    ear_closed_threshold: float
    ear_cooldown_step: int
    ear_alarm_frames: int
    mar_open_threshold: float
    yawn_cooldown_step: int
    yawn_alarm_frames: int
    fusion_ear_gate: float
    alarm_audio_path: Path
    event_db_path: Path
    drowsy_event_cooldown_seconds: float
    drowsy_frame_capture_enabled: bool
    drowsy_frame_dir: Path
    s3_upload_enabled: bool
    s3_bucket: str | None
    s3_prefix: str

    @property
    def input_size(self) -> tuple[int, int]:
        return (self.model_input_width, self.model_input_height)

    @classmethod
    def from_env(cls) -> "ServiceSettings":
        return cls(
            service_mode=os.getenv("SERVICE_MODE", "hybrid").strip().lower(),
            remote_endpoint_url=os.getenv("REMOTE_ENDPOINT_URL"),
            remote_api_key=os.getenv("REMOTE_API_KEY"),
            remote_timeout_seconds=_float_env("REMOTE_TIMEOUT_SECONDS", 3.0),
            network_probe_url=os.getenv("NETWORK_PROBE_URL", "https://aws.amazon.com"),
            network_probe_timeout_seconds=_float_env("NETWORK_PROBE_TIMEOUT_SECONDS", 1.2),
            network_probe_ttl_seconds=_int_env("NETWORK_PROBE_TTL_SECONDS", 5),
            local_model_backend=os.getenv("LOCAL_MODEL_BACKEND", "keras").strip().lower(),
            local_model_path=Path(os.getenv("LOCAL_MODEL_PATH", "./drowsy_model.h5")).resolve(),
            local_model_threshold=_float_env("LOCAL_MODEL_THRESHOLD", 0.65),
            model_input_width=_int_env("MODEL_INPUT_WIDTH", 224),
            model_input_height=_int_env("MODEL_INPUT_HEIGHT", 224),
            frame_skip=_int_env("FRAME_SKIP", 3),
            cnn_every_n_frames=_int_env("CNN_EVERY_N_FRAMES", 25),
            ear_closed_threshold=_float_env("EAR_CLOSED_THRESHOLD", 0.19),
            ear_cooldown_step=_int_env("EAR_COOLDOWN_STEP", 3),
            ear_alarm_frames=_int_env("EAR_ALARM_FRAMES", 35),
            mar_open_threshold=_float_env("MAR_OPEN_THRESHOLD", 0.65),
            yawn_cooldown_step=_int_env("YAWN_COOLDOWN_STEP", 2),
            yawn_alarm_frames=_int_env("YAWN_ALARM_FRAMES", 20),
            fusion_ear_gate=_float_env("FUSION_EAR_GATE", 0.23),
            alarm_audio_path=Path(os.getenv("ALARM_AUDIO_PATH", "./alarm.wav")).resolve(),
            event_db_path=Path(os.getenv("EVENT_DB_PATH", "./artifacts/events.db")).resolve(),
            drowsy_event_cooldown_seconds=_float_env("DROWSY_EVENT_COOLDOWN_SECONDS", 10.0),
            drowsy_frame_capture_enabled=_bool_env("DROWSY_FRAME_CAPTURE_ENABLED", True),
            drowsy_frame_dir=Path(os.getenv("DROWSY_FRAME_DIR", "./artifacts/drowsy_frames")).resolve(),
            s3_upload_enabled=_bool_env("S3_UPLOAD_ENABLED", False),
            s3_bucket=_optional_env("S3_BUCKET"),
            s3_prefix=os.getenv("S3_PREFIX", "drowsy-events").strip().strip("/"),
        )


@dataclass(frozen=True)
class TrainingSettings:
    dataset_path: Path
    output_root: Path
    model_type: str
    base_weights: str | None
    epochs: int
    batch_size: int
    validation_split: float
    learning_rate: float
    random_seed: int
    image_width: int
    image_height: int
    min_promote_accuracy: float
    promote_delta: float

    @property
    def image_size(self) -> tuple[int, int]:
        return (self.image_height, self.image_width)

    @classmethod
    def from_env(cls) -> "TrainingSettings":
        return cls(
            dataset_path=Path(os.getenv("DATASET_PATH", "./data/train_data")).resolve(),
            output_root=Path(os.getenv("TRAIN_OUTPUT_ROOT", "./artifacts/runs")).resolve(),
            model_type=os.getenv("TRAIN_MODEL_TYPE", "simple").strip().lower(),
            base_weights=_optional_env("TRAIN_BASE_WEIGHTS", "none"),
            epochs=_int_env("TRAIN_EPOCHS", 10),
            batch_size=_int_env("TRAIN_BATCH_SIZE", 32),
            validation_split=_float_env("TRAIN_VALIDATION_SPLIT", 0.2),
            learning_rate=_float_env("TRAIN_LEARNING_RATE", 0.001),
            random_seed=_int_env("TRAIN_RANDOM_SEED", 42),
            image_width=_int_env("MODEL_INPUT_WIDTH", 224),
            image_height=_int_env("MODEL_INPUT_HEIGHT", 224),
            min_promote_accuracy=_float_env("TRAIN_MIN_PROMOTE_ACCURACY", 0.75),
            promote_delta=_float_env("TRAIN_PROMOTE_DELTA", 0.01),
        )
