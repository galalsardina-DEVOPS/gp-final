#!/usr/bin/env bash
set -euo pipefail

if [ "${1:-}" = "" ]; then
  echo "Usage: bash scripts/sync_model_from_s3.sh s3://your-bucket/path/production"
  exit 1
fi

S3_URI="$1"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST_DIR="$ROOT_DIR/artifacts/production"

mkdir -p "$DEST_DIR"
aws s3 sync "$S3_URI" "$DEST_DIR" --delete

echo "Synced production model artifacts into: $DEST_DIR"
echo "Set LOCAL_MODEL_PATH=./artifacts/production/model.keras in .env to use it."

