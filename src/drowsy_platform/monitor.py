from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any
import wave

import cv2
import mediapipe as mp
import numpy as np

from drowsy_platform.config import ServiceSettings
from drowsy_platform.event_logger import DrowsyEvent, EventLogger
from drowsy_platform.hybrid import HybridClassifier
from drowsy_platform.models.base import PredictionResult
from drowsy_platform.s3_uploader import S3Uploader

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [13, 14, 78, 308]


def eye_aspect_ratio(eye: np.ndarray) -> float:
    a = np.linalg.norm(eye[1] - eye[5])
    b = np.linalg.norm(eye[2] - eye[4])
    c = np.linalg.norm(eye[0] - eye[3])
    return float((a + b) / (2.0 * c))


def mouth_aspect_ratio(mouth: np.ndarray) -> float:
    a = np.linalg.norm(mouth[0] - mouth[1])
    b = np.linalg.norm(mouth[2] - mouth[3])
    return float(a / b)


@dataclass
class MonitorState:
    counter_eye: int = 0
    counter_yawn: int = 0
    frame_count: int = 0
    cnn_result: bool = False
    last_prediction: PredictionResult | None = None
    last_event_at: float = 0.0
    ear_history: list[float] = field(default_factory=list)


class DriverMonitor:
    def __init__(self, classifier: HybridClassifier, settings: ServiceSettings) -> None:
        self.classifier = classifier
        self.settings = settings
        self.state = MonitorState()
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
        )
        self._alarm_loaded = False
        self._alarm_enabled = False
        self._last_bell_at = 0.0
        self._last_aplay_at = 0.0
        self._aplay_path = shutil.which("aplay")
        self.event_logger = EventLogger(settings.event_db_path)
        self.s3_uploader = (
            S3Uploader(settings.s3_bucket, settings.s3_prefix)
            if settings.s3_upload_enabled and settings.s3_bucket
            else None
        )
        self._load_alarm()

    def run(self, source: int | str = 0) -> None:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open source: {source}")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                annotated = self._process_frame(frame)
                cv2.imshow("Hybrid Driver Monitoring", annotated)

                if cv2.waitKey(25) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        self.state.frame_count += 1
        resized = cv2.resize(frame, (320, 240))

        if self.state.frame_count % self.settings.frame_skip != 0:
            return resized

        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return resized

        face_landmarks = results.multi_face_landmarks[0]
        left_eye, right_eye, mouth = self._extract_points(face_landmarks, resized)

        ear = self._smooth_ear((eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0)
        mar = mouth_aspect_ratio(mouth)

        face_roi = self._extract_face_roi(face_landmarks, resized)
        if face_roi is not None and self.state.frame_count % self.settings.cnn_every_n_frames == 0:
            rgb_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            self.state.last_prediction = self.classifier.predict_array(
                rgb_face,
                metadata={"frame_count": self.state.frame_count},
            )
            self.state.cnn_result = self.state.last_prediction.label == "drowsy"

        ear_drowsy = self._update_eye_counter(ear)
        yawn_detected = self._update_yawn_counter(mar)
        drowsy = ear_drowsy or yawn_detected or (
            self.state.cnn_result and ear < self.settings.fusion_ear_gate
        )

        self._draw_overlay(resized, ear, mar, drowsy)
        if drowsy:
            self._play_alarm_once()
            self._record_drowsy_event(resized, ear, mar)

        return resized

    def _extract_points(self, face_landmarks: Any, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        h, w, _ = frame.shape
        left_eye = []
        right_eye = []
        mouth = []

        for point_id in LEFT_EYE:
            left_eye.append(self._landmark_to_point(face_landmarks, point_id, w, h))
        for point_id in RIGHT_EYE:
            right_eye.append(self._landmark_to_point(face_landmarks, point_id, w, h))
        for point_id in MOUTH:
            mouth.append(self._landmark_to_point(face_landmarks, point_id, w, h))

        return np.array(left_eye), np.array(right_eye), np.array(mouth)

    @staticmethod
    def _landmark_to_point(face_landmarks: Any, point_id: int, width: int, height: int) -> list[int]:
        x = int(face_landmarks.landmark[point_id].x * width)
        y = int(face_landmarks.landmark[point_id].y * height)
        return [x, y]

    def _extract_face_roi(self, face_landmarks: Any, frame: np.ndarray) -> np.ndarray | None:
        h, w, _ = frame.shape
        xs = [landmark.x for landmark in face_landmarks.landmark]
        ys = [landmark.y for landmark in face_landmarks.landmark]

        x1 = max(0, int(min(xs) * w) - 10)
        y1 = max(0, int(min(ys) * h) - 10)
        x2 = min(w, int(max(xs) * w) + 10)
        y2 = min(h, int(max(ys) * h) + 10)

        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]

    def _smooth_ear(self, ear: float) -> float:
        self.state.ear_history.append(ear)
        if len(self.state.ear_history) > 7:
            self.state.ear_history.pop(0)
        return float(np.mean(self.state.ear_history))

    def _update_eye_counter(self, ear: float) -> bool:
        if ear < self.settings.ear_closed_threshold:
            self.state.counter_eye += 1
        else:
            self.state.counter_eye = max(0, self.state.counter_eye - self.settings.ear_cooldown_step)
        return self.state.counter_eye > self.settings.ear_alarm_frames

    def _update_yawn_counter(self, mar: float) -> bool:
        if mar > self.settings.mar_open_threshold:
            self.state.counter_yawn += 1
        else:
            self.state.counter_yawn = max(0, self.state.counter_yawn - self.settings.yawn_cooldown_step)
        return self.state.counter_yawn > self.settings.yawn_alarm_frames

    def _draw_overlay(self, frame: np.ndarray, ear: float, mar: float, drowsy: bool) -> None:
        status = "DROWSY" if drowsy else "ALERT"
        color = (0, 0, 255) if drowsy else (0, 255, 0)
        route = self.state.last_prediction.route if self.state.last_prediction else "local"
        backend = self.state.last_prediction.backend if self.state.last_prediction else "n/a"

        cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        cv2.putText(frame, f"EAR:{ear:.2f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"MAR:{mar:.2f}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(
            frame,
            f"EYE CNT:{self.state.counter_eye}",
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            frame,
            f"ROUTE:{route} ({backend})",
            (20, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )

    def _load_alarm(self) -> None:
        try:
            import pygame
        except ImportError:
            self._alarm_enabled = False
            return

        try:
            pygame.mixer.init()
            self._alarm_enabled = True
            self._pygame = pygame
        except Exception:
            self._alarm_enabled = False

    def _play_alarm_once(self) -> None:
        if self._alarm_enabled and self.settings.alarm_audio_path.exists():
            if not self._alarm_loaded:
                self._pygame.mixer.music.load(str(self.settings.alarm_audio_path))
                self._alarm_loaded = True

            if not self._pygame.mixer.music.get_busy():
                self._pygame.mixer.music.play()
            return

        if self._play_with_aplay():
            return

        self._terminal_bell()

    def _terminal_bell(self) -> None:
        now = time.monotonic()
        if now - self._last_bell_at < 1.5:
            return
        print("\a", end="", flush=True)
        self._last_bell_at = now

    def _play_with_aplay(self) -> bool:
        if not self._aplay_path:
            return False

        now = time.monotonic()
        if now - self._last_aplay_at < 1.5:
            return True

        path = self.settings.alarm_audio_path
        if not path.exists():
            path = self.settings.event_db_path.parent / "generated_alarm.wav"
            self._ensure_generated_alarm(path)

        try:
            subprocess.Popen(
                [self._aplay_path, str(path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._last_aplay_at = now
            return True
        except OSError:
            return False

    @staticmethod
    def _ensure_generated_alarm(path: Path) -> None:
        if path.exists():
            return

        path.parent.mkdir(parents=True, exist_ok=True)
        sample_rate = 8000
        duration_seconds = 0.45
        frequency = 880
        amplitude = 16000
        sample_count = int(sample_rate * duration_seconds)

        with wave.open(str(path), "w") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            for index in range(sample_count):
                phase = (index * frequency // sample_rate) % 2
                value = amplitude if phase == 0 else -amplitude
                wav_file.writeframesraw(int(value).to_bytes(2, byteorder="little", signed=True))

    def _record_drowsy_event(self, frame: np.ndarray, ear: float, mar: float) -> None:
        now = time.monotonic()
        if now - self.state.last_event_at < self.settings.drowsy_event_cooldown_seconds:
            return

        self.state.last_event_at = now
        frame_path = self._save_drowsy_frame(frame) if self.settings.drowsy_frame_capture_enabled else None
        s3_uri = self._upload_frame(frame_path) if frame_path else None
        prediction = self.state.last_prediction

        self.event_logger.log(
            DrowsyEvent(
                source="monitor",
                label="drowsy",
                score=prediction.score if prediction else None,
                route=prediction.route if prediction else "local",
                backend=prediction.backend if prediction else None,
                ear=ear,
                mar=mar,
                frame_path=str(frame_path) if frame_path else None,
                s3_uri=s3_uri,
                metadata={"frame_count": self.state.frame_count},
            )
        )

    def _save_drowsy_frame(self, frame: np.ndarray) -> Path | None:
        self.settings.drowsy_frame_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        path = self.settings.drowsy_frame_dir / f"drowsy_{timestamp}_{self.state.frame_count}.jpg"
        if cv2.imwrite(str(path), frame):
            return path
        return None

    def _upload_frame(self, frame_path: Path | None) -> str | None:
        if not frame_path or not self.s3_uploader:
            return None
        try:
            return self.s3_uploader.upload_file(frame_path)
        except Exception as exc:
            print(f"S3 upload failed: {exc}")
            return None

