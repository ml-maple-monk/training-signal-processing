#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${SUPERBPE_RUN_ID:-20260501T233054Z-superbpe-balanced-1to1-50k}"
CONFIG_PATH="${SUPERBPE_CONFIG:-config/tokenizer_training.superbpe_balanced_50k.sample.yaml}"
LOG_ROOT="${SUPERBPE_LOG_ROOT:-.runtime/tokenizers/remote_logs}"
LOG_PATH="${LOG_ROOT}/${RUN_ID}.log"
STATUS_PATH="${LOG_ROOT}/${RUN_ID}.status.json"
CORPUS_DIR="${SUPERBPE_CORPUS_DIR:-.runtime/tokenizers/corpora/${RUN_ID}}"
STAGE1_DIR="${SUPERBPE_STAGE1_DIR:-.runtime/tokenizers/${RUN_ID}.staging/stage1_bpe}"
REQUIRE_STAGE1="${SUPERBPE_REQUIRE_STAGE1:-true}"

mkdir -p "$LOG_ROOT"

write_status() {
  local phase="$1"
  local status="$2"
  python - "$STATUS_PATH" "$phase" "$status" "$RUN_ID" "$LOG_PATH" <<'PY'
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "updated_at": datetime.now(timezone.utc).isoformat(),
    "phase": sys.argv[2],
    "status": sys.argv[3],
    "run_id": sys.argv[4],
    "log_path": sys.argv[5],
}
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

on_exit() {
  local code=$?
  if [ "$code" -ne 0 ]; then
    write_status "${SUPERBPE_FAILED_PHASE:-failed}" "failed"
  fi
}
trap on_exit EXIT

require_path() {
  local path="$1"
  if [ ! -e "$path" ]; then
    echo "[superbpe-remote] missing required path: $path" >&2
    return 1
  fi
}

verify_corpus_manifest() {
  python - "$CORPUS_DIR/corpus_manifest.json" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
shard_paths = [Path(path) for path in manifest.get("shard_paths", [])]
shard_bytes = [int(size) for size in manifest.get("shard_bytes", [])]
missing = []
mismatched = []
present_bytes = 0
for index, path in enumerate(shard_paths):
    expected = shard_bytes[index] if index < len(shard_bytes) else None
    if not path.exists():
        missing.append(str(path))
        continue
    actual = path.stat().st_size
    present_bytes += actual
    if expected is not None and actual != expected:
        mismatched.append(f"{path}: actual={actual} expected={expected}")
expected_total = int(manifest.get("corpus_file_bytes") or sum(shard_bytes))
print(
    "[superbpe-remote] corpus manifest: "
    f"shards={len(shard_paths)} present_bytes={present_bytes} expected_bytes={expected_total}"
)
if missing or mismatched:
    preview = missing[:5] + mismatched[:5]
    raise SystemExit(
        "Corpus transfer is incomplete: "
        + "; ".join(preview)
        + (f"; +{len(missing) + len(mismatched) - len(preview)} more" if len(missing) + len(mismatched) > len(preview) else "")
    )
if present_bytes != expected_total:
    raise SystemExit(
        f"Corpus byte total mismatch: actual={present_bytes} expected={expected_total}"
    )
PY
}

SUPERBPE_FAILED_PHASE="preflight"
write_status "preflight" "running"
{
  echo "[superbpe-remote] started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "[superbpe-remote] run_id=${RUN_ID}"
  echo "[superbpe-remote] config=${CONFIG_PATH}"
  echo "[superbpe-remote] hostname=$(hostname)"
  echo "[superbpe-remote] cpus=$(nproc)"
  free -h
  df -h .
  require_path "$CONFIG_PATH"
  require_path "$CORPUS_DIR/corpus_manifest.json"
  if [ "$REQUIRE_STAGE1" = "true" ]; then
    require_path "$STAGE1_DIR/tokenizer.json"
    require_path "$STAGE1_DIR/vocab.json"
    require_path "$STAGE1_DIR/merges.txt"
    require_path "$STAGE1_DIR/meta.json"
    du -sh "$CORPUS_DIR" "$STAGE1_DIR"
  else
    echo "[superbpe-remote] Stage 1 preflight disabled; Stage 1 will train if missing."
    du -sh "$CORPUS_DIR"
  fi
  verify_corpus_manifest

  export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
  export RAYON_NUM_THREADS="${SUPERBPE_RAYON_NUM_THREADS:-$(nproc)}"
  export TOKENIZERS_PARALLELISM="${SUPERBPE_TOKENIZERS_PARALLELISM:-true}"
  export MALLOC_ARENA_MAX="${SUPERBPE_MALLOC_ARENA_MAX:-8}"
  echo "[superbpe-remote] RAYON_NUM_THREADS=${RAYON_NUM_THREADS}"
  echo "[superbpe-remote] TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM}"
  echo "[superbpe-remote] MALLOC_ARENA_MAX=${MALLOC_ARENA_MAX}"

  SUPERBPE_FAILED_PHASE="training"
  write_status "training" "running"
  if [ -x /usr/bin/time ]; then
    /usr/bin/time -v uv run --no-sync --group tokenizer_training \
      python -m training_signal_processing.main tokenizer-training-run \
      --config "$CONFIG_PATH" \
      --set "output.run_id=${RUN_ID}" \
      --set "checkpoint.enabled=false" \
      --set "budget.max_wall_seconds=0" \
      --set "budget.max_memory_gib=0" \
      --set "training.superbpe.stage1_num_bytes=0" \
      --set "training.superbpe.stage2_num_bytes=0"
  else
    uv run --no-sync --group tokenizer_training \
      python -m training_signal_processing.main tokenizer-training-run \
      --config "$CONFIG_PATH" \
      --set "output.run_id=${RUN_ID}" \
      --set "checkpoint.enabled=false" \
      --set "budget.max_wall_seconds=0" \
      --set "budget.max_memory_gib=0" \
      --set "training.superbpe.stage1_num_bytes=0" \
      --set "training.superbpe.stage2_num_bytes=0"
  fi
  SUPERBPE_FAILED_PHASE="complete"
  write_status "complete" "success"
  echo "[superbpe-remote] finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} 2>&1 | tee -a "$LOG_PATH"
