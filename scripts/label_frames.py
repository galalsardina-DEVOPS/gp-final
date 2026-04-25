from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create alert/drowsy training data from a video.")
    parser.add_argument("--source", required=True, help="Video path or camera index.")
    parser.add_argument("--output", default="data/train_data", help="Output dataset folder.")
    parser.add_argument("--every-n", type=int, default=10, help="Show one frame every N frames.")
    parser.add_argument("--width", type=int, default=640, help="Preview width.")
    parser.add_argument("--label", choices=["alert", "drowsy"], help="Auto-save shown frames with this label.")
    parser.add_argument("--max-images", type=int, default=0, help="Stop after saving this many images. 0 means no limit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = int(args.source) if str(args.source).isdigit() else args.source
    output_root = Path(args.output).resolve()
    alert_dir = output_root / "alert"
    drowsy_dir = output_root / "drowsy"
    alert_dir.mkdir(parents=True, exist_ok=True)
    drowsy_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    saved = 0
    frame_index = 0
    print("Keys: a=save alert, d=save drowsy, s=skip, q=quit")
    print(f"Saving into: {output_root}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_index += 1
            if frame_index % args.every_n != 0:
                continue

            frame = _resize_preview(frame, args.width)
            label = args.label

            if label is None:
                preview = frame.copy()
                cv2.putText(
                    preview,
                    "a=alert  d=drowsy  s=skip  q=quit",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow("Label training frames", preview)
                key = cv2.waitKey(0) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    continue
                if key == ord("a"):
                    label = "alert"
                elif key == ord("d"):
                    label = "drowsy"
                else:
                    continue

            save_path = _save_frame(frame, output_root / label, label, frame_index)
            saved += 1
            print(f"saved {label}: {save_path}")

            if args.max_images and saved >= args.max_images:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"Done. Saved {saved} images.")


def _resize_preview(frame, width: int):
    height = int(frame.shape[0] * (width / frame.shape[1]))
    return cv2.resize(frame, (width, height))


def _save_frame(frame, label_dir: Path, label: str, frame_index: int) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    path = label_dir / f"{label}_{stamp}_{frame_index}.jpg"
    cv2.imwrite(str(path), frame)
    return path


if __name__ == "__main__":
    main()
