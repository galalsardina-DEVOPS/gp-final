from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drowsy_platform.config import TrainingSettings
from drowsy_platform.training import train_and_export


def main() -> None:
    settings = TrainingSettings.from_env()
    manifest_path = train_and_export(settings)
    print(f"Candidate manifest written to: {manifest_path}")


if __name__ == "__main__":
    main()

