#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/geeyang/workspace/training-signal-processing"
LOG_DIR="$REPO_ROOT/exports/logs"
LOCK_FILE="$REPO_ROOT/exports/.fineweb2_lite_malay_filtered_snapshot.lock"

mkdir -p "$LOG_DIR"

cd "$REPO_ROOT"

exec /usr/bin/flock -n "$LOCK_FILE" \
  /home/geeyang/.local/bin/uv run --group fineweb_2_lite \
  python scripts/export-fineweb2-lite-malay-filtered-snapshot.py \
  --query src/data-storage/scripts/queries/fineweb2_lite_malay_filtered_snapshot.sql \
  --output-dir exports/fineweb2_lite_malay_filtered_snapshot \
  --rows-per-shard 100000
