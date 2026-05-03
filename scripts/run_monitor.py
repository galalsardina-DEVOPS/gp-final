from __future__ import annotations

import argparse
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drowsy_platform.config import ServiceSettings
from drowsy_platform.hybrid import HybridClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the hybrid drowsy-driver monitor.")
    parser.add_argument("--source", default="0", help="Camera index or video file path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = int(args.source) if str(args.source).isdigit() else args.source
    if isinstance(source, int) and not _camera_index_available(source):
        print(f"Camera index {source} is not available in this environment.")
        print("If you are using WSL, run the camera script on Windows or use an IP camera URL.")
        return

    from drowsy_platform.monitor import DriverMonitor

    settings = ServiceSettings.from_env()
    classifier = HybridClassifier(settings)
    monitor = DriverMonitor(classifier, settings)
    monitor.run(source=source)


def _camera_index_available(source: int) -> bool:
    import cv2

    cap = cv2.VideoCapture(source, cv2.CAP_DSHOW) if sys.platform.startswith("win") else cv2.VideoCapture(source)
    ok, frame = cap.read()
    cap.release()
    return bool(ok and frame is not None)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        crash_log = ROOT / "artifacts" / "monitor_crash.log"
        crash_log.parent.mkdir(parents=True, exist_ok=True)
        crash_text = traceback.format_exc()
        crash_log.write_text(crash_text, encoding="utf-8")
        print(f"Monitor crashed. Full traceback written to: {crash_log}")
        print(crash_text)
        sys.exit(1)
