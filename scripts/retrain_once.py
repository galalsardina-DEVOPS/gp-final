from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drowsy_platform.config import TrainingSettings
from drowsy_platform.training import promote_candidate, train_and_export


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a candidate model and promote it if metrics pass.")
    parser.add_argument("--production-dir", default=str(ROOT / "artifacts" / "production"))
    parser.add_argument("--skip-promote", action="store_true", help="Only train and write a candidate manifest.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = TrainingSettings.from_env()

    manifest_path = train_and_export(settings)
    print(f"Candidate manifest written to: {manifest_path}")

    if args.skip_promote:
        print("Promotion skipped.")
        return

    promoted = promote_candidate(
        candidate_manifest_path=manifest_path,
        production_dir=Path(args.production_dir).resolve(),
        min_accuracy=settings.min_promote_accuracy,
        min_delta=settings.promote_delta,
    )

    if promoted:
        print(f"Candidate promoted to: {Path(args.production_dir).resolve()}")
    else:
        print("Candidate rejected. Promotion thresholds were not met.")


if __name__ == "__main__":
    main()
