#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_PATH="$ROOT_DIR/artifacts/retrain.log"
CRON_LINE="0 2 * * 1 cd $ROOT_DIR && docker compose run --rm retrain >> $LOG_PATH 2>&1"

mkdir -p "$ROOT_DIR/artifacts"

(crontab -l 2>/dev/null | grep -v "scripts/retrain_once.py" | grep -v "docker compose run --rm retrain" || true; echo "$CRON_LINE") | crontab -

echo "Weekly retraining cron installed:"
echo "$CRON_LINE"
echo "Check logs with:"
echo "tail -n 100 $LOG_PATH"

