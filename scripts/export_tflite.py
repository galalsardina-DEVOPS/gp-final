from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a Keras model to TFLite for offline fallback.")
    parser.add_argument("--input", required=True, help="Path to .keras or .h5 model")
    parser.add_argument("--output", required=True, help="Path to output .tflite file")
    return parser.parse_args()


def main() -> None:
    import tensorflow as tf

    args = parse_args()
    model = tf.keras.models.load_model(args.input)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(tflite_model)
    print(f"TFLite model exported to: {output_path}")


if __name__ == "__main__":
    main()
