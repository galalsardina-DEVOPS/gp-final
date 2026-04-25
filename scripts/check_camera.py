from __future__ import annotations

import argparse

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check which OpenCV camera indexes are available.")
    parser.add_argument("--max-index", type=int, default=5, help="Highest camera index to test.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    found = False

    for index in range(args.max_index + 1):
        cap = cv2.VideoCapture(index)
        ok, frame = cap.read()
        cap.release()

        if ok and frame is not None:
            found = True
            print(f"Camera index {index}: OK ({frame.shape[1]}x{frame.shape[0]})")
        else:
            print(f"Camera index {index}: not available")

    if not found:
        print("No camera was detected by OpenCV in this environment.")


if __name__ == "__main__":
    main()
