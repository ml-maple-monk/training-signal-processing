#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${SUPERBPE_RUN_ID:-20260501T233054Z-superbpe-balanced-1to1-50k}"
LOG_ROOT="${SUPERBPE_LOG_ROOT:-.runtime/tokenizers/remote_logs}"
LOG_PATH="${LOG_ROOT}/${RUN_ID}.log"
STATUS_PATH="${LOG_ROOT}/${RUN_ID}.status.json"
RUN_DIR=".runtime/tokenizers/${RUN_ID}"
STAGING_DIR=".runtime/tokenizers/${RUN_ID}.staging"
CORPUS_DIR=".runtime/tokenizers/corpora/${RUN_ID}"

echo "[superbpe-status] hostname=$(hostname)"
echo "[superbpe-status] cpus=$(nproc)"
echo "[superbpe-status] launch defaults: RAYON_NUM_THREADS=$(nproc) TOKENIZERS_PARALLELISM=true MALLOC_ARENA_MAX=8"
free -h
df -h .
echo
echo "[superbpe-status] processes"
ps -eo pid,ppid,stat,etime,%cpu,%mem,rss,cmd \
  | grep -E 'train_tokenizer|tokenizer-training-run|uv run --no-sync' \
  | grep -v grep || true
echo
echo "[superbpe-status] paths"
du -sh "$CORPUS_DIR" "$STAGING_DIR" "$RUN_DIR" 2>/dev/null || true
find "$RUN_DIR" "$STAGING_DIR" -maxdepth 2 -type f \
  \( -name 'tokenizer.json' -o -name 'vocab.json' -o -name 'merges.txt' -o -name 'meta.json' -o -name 'training_summary.json' \) \
  -printf '%p %s bytes\n' 2>/dev/null | sort || true
if [ -f "$CORPUS_DIR/corpus_manifest.json" ]; then
  python - "$CORPUS_DIR/corpus_manifest.json" <<'PY' || true
from __future__ import annotations

import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
shard_paths = [Path(path) for path in manifest.get("shard_paths", [])]
shard_bytes = [int(size) for size in manifest.get("shard_bytes", [])]
complete = 0
present = 0
present_bytes = 0
for index, path in enumerate(shard_paths):
    if not path.exists():
        continue
    present += 1
    actual = path.stat().st_size
    present_bytes += actual
    if index < len(shard_bytes) and actual == shard_bytes[index]:
        complete += 1
expected_bytes = int(manifest.get("corpus_file_bytes") or sum(shard_bytes))
percent = (present_bytes / expected_bytes * 100.0) if expected_bytes else 0.0
print(
    "[superbpe-status] corpus "
    f"present_shards={present}/{len(shard_paths)} "
    f"complete_shards={complete}/{len(shard_paths)} "
    f"present_bytes={present_bytes} expected_bytes={expected_bytes} "
    f"percent={percent:.2f}"
)
PY
fi
echo
echo "[superbpe-status] status"
cat "$STATUS_PATH" 2>/dev/null || true
echo
echo "[superbpe-status] log tail"
tail -80 "$LOG_PATH" 2>/dev/null || true
