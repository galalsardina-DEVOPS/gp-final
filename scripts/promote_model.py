from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drowsy_platform.config import TrainingSettings
from drowsy_platform.training import promote_candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote a candidate model into production.")
    parser.add_argument("--candidate", required=True, help="Path to candidate manifest.json")
    parser.add_argument("--production-dir", default=str(ROOT / "artifacts" / "production"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = TrainingSettings.from_env()
    promoted = promote_candidate(
        candidate_manifest_path=Path(args.candidate).resolve(),
        production_dir=Path(args.production_dir).resolve(),
        min_accuracy=settings.min_promote_accuracy,
        min_delta=settings.promote_delta,
    )
    if promoted:
        print("Candidate promoted to production.")
    else:
        print("Candidate rejected. Thresholds not met.")


if __name__ == "__main__":
    main()

