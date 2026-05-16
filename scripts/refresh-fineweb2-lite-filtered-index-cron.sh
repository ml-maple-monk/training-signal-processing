#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/geeyang/workspace/training-signal-processing"
LOCK_FILE="$REPO_ROOT/exports/.fineweb2_lite_filtered_index_refresh.lock"

cd "$REPO_ROOT"

exec /usr/bin/flock -n "$LOCK_FILE" \
  /home/geeyang/.local/bin/uv run --group fineweb_2_lite \
  python scripts/refresh-fineweb2-lite-filtered-index.py \
  --mode incremental \
  --query src/data-storage/scripts/queries/fineweb2_lite_filtered_document_index.sql \
  --profile-name fineweb2_lite_all_lid_v1 \
  --run-id fineweb2-lite-metadata-full-20260515T183115Z
