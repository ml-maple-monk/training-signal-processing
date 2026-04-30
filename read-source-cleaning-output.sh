#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-${SOURCE_CLEANING_RUN_ID:-source-cleaning-full-20260429T171532Z}}"
REMOTE_ROOT="${RCLONE_REMOTE_ROOT:-ocrresults:gpu-poor}"
PREFIX="${REMOTE_ROOT}/dataset/processed/source-cleaning/${RUN_ID}"

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

list_prefix() {
    local remote_path="$1"
    local output_path="$2"
    rclone lsf -R "$remote_path" 2>/dev/null | sed '/\/$/d' >"$output_path" || true
}

count_matches() {
    local pattern="$1"
    local path="$2"
    grep -Ec "$pattern" "$path" || true
}

print_source_counts() {
    local label="$1"
    local suffix="$2"
    local path="$3"

    echo
    echo "$label by source:"
    awk -F/ -v suffix="$suffix" '
        length($0) >= length(suffix) && substr($0, length($0) - length(suffix) + 1) == suffix {
            source = $1
            sub(/^source=/, "", source)
            counts[source] += 1
        }
        END {
            if (length(counts) == 0) {
                print "  none"
                exit
            }
            for (source in counts) {
                print "  " source ": " counts[source]
            }
        }
    ' "$path" | sort
}

list_prefix "$PREFIX/control" "$tmpdir/control.txt"
list_prefix "$PREFIX/done" "$tmpdir/done.txt"
list_prefix "$PREFIX/metrics" "$tmpdir/metrics.txt"
list_prefix "$PREFIX/errors" "$tmpdir/errors.txt"
list_prefix "$PREFIX/source_shards" "$tmpdir/source_shards.txt"
list_prefix "$PREFIX/unified" "$tmpdir/unified.txt"

manifest_rows="$(
    rclone cat "$PREFIX/control/input_manifest.jsonl" 2>/dev/null | wc -l | tr -d ' '
)"
control_files="$(wc -l <"$tmpdir/control.txt" | tr -d ' ')"
done_files="$(count_matches '\.done\.json$' "$tmpdir/done.txt")"
metric_files="$(count_matches '\.metrics\.json$' "$tmpdir/metrics.txt")"
error_files="$(count_matches '\.error\.json$' "$tmpdir/errors.txt")"
source_parquet_files="$(count_matches '\.parquet$' "$tmpdir/source_shards.txt")"
unified_parquet_files="$(count_matches '\.parquet$' "$tmpdir/unified.txt")"

echo "Source cleaning run: $RUN_ID"
echo "Prefix: $PREFIX"
echo
echo "Manifest row groups: $manifest_rows"
echo "Done sentinels: $done_files"
echo "Metric sidecars: $metric_files"
echo "Error sidecars: $error_files"
echo "Source parquet shards: $source_parquet_files"
echo "Unified parquet shards: $unified_parquet_files"
echo "Control files: $control_files"
echo "Completion source of truth: done sentinels"

if [ "$manifest_rows" -gt 0 ]; then
    python - "$manifest_rows" "$done_files" <<'PY'
import sys

total = int(sys.argv[1])
done = int(sys.argv[2])
print(f"Done completion: {done / total * 100:.2f}%")
print(f"Remaining row groups: {max(total - done, 0)}")
PY
fi

print_source_counts "Done sentinels" ".done.json" "$tmpdir/done.txt"
print_source_counts "Metric sidecars" ".metrics.json" "$tmpdir/metrics.txt"
print_source_counts "Error sidecars" ".error.json" "$tmpdir/errors.txt"
print_source_counts "Source parquet shards" ".parquet" "$tmpdir/source_shards.txt"
print_source_counts "Unified parquet shards" ".parquet" "$tmpdir/unified.txt"

echo
echo "Sample unified shard keys:"
if [ "$unified_parquet_files" -gt 0 ]; then
    sed 's#^#  #; 5q' "$tmpdir/unified.txt"
else
    echo "  none"
fi

if [ "${AGGREGATE_METRICS:-0}" = "1" ]; then
    echo
    echo "Aggregating metrics sidecars..."
    mkdir -p "$tmpdir/metric_files"
    rclone copy "$PREFIX/metrics" "$tmpdir/metric_files" \
        --include "*.metrics.json" \
        --transfers "${RCLONE_TRANSFERS:-16}" \
        --checkers "${RCLONE_CHECKERS:-16}" \
        --fast-list >/dev/null
    python - "$tmpdir/metric_files" <<'PY'
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1])
counts = Counter()
rows = Counter()
chars = Counter()
cleaned = Counter()
removed = Counter()
dropped = Counter()

for path in root.rglob("*.metrics.json"):
    metric = json.loads(path.read_text(encoding="utf-8"))
    source = str(metric.get("cleaning_source") or metric.get("source_name") or "unknown")
    counts[source] += 1
    rows[source] += int(metric.get("filtered_row_count") or 0)
    chars[source] += int(metric.get("original_total_characters") or 0)
    cleaned[source] += int(metric.get("cleaned_total_characters") or 0)
    removed[source] += int(metric.get("removed_total_characters") or 0)
    dropped[source] += int(metric.get("dropped_row_count") or 0)

print(f"Aggregated metric files: {sum(counts.values())}")
print(f"Rows: {sum(rows.values())}")
print(f"Original chars: {sum(chars.values())}")
print(f"Cleaned chars: {sum(cleaned.values())}")
print(f"Removed chars: {sum(removed.values())}")
print(f"Dropped rows: {sum(dropped.values())}")
print()
print("Aggregated metrics by source:")
for source in sorted(counts):
    print(
        "  "
        f"{source}: files={counts[source]} rows={rows[source]} "
        f"original_chars={chars[source]} removed_chars={removed[source]} "
        f"dropped_rows={dropped[source]}"
    )
PY
fi
