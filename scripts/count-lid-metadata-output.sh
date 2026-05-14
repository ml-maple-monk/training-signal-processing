#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-${LID_METADATA_RUN_ID:-20260429T015335Z}}"
REMOTE_ROOT="${RCLONE_REMOTE_ROOT:-ocrresults:gpu-poor}"
PREFIX="${REMOTE_ROOT}/dataset/processed/lid-metadata/${RUN_ID}"

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
            counts[$1] += 1
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

list_prefix "$PREFIX/shards" "$tmpdir/shards.txt"
list_prefix "$PREFIX/checkpoints" "$tmpdir/checkpoints.txt"
list_prefix "$PREFIX/errors" "$tmpdir/errors.txt"
list_prefix "$PREFIX/control" "$tmpdir/control.txt"

manifest_rows="$(
    rclone cat "$PREFIX/control/input_manifest.jsonl" 2>/dev/null | wc -l | tr -d ' '
)"
parquet_shards="$(count_matches '\.parquet$' "$tmpdir/shards.txt")"
metric_sidecars="$(count_matches '\.metrics\.json$' "$tmpdir/shards.txt")"
error_sidecars="$(count_matches '\.error\.json$' "$tmpdir/shards.txt")"
checkpoints="$(count_matches '\.json$' "$tmpdir/checkpoints.txt")"
control_files="$(wc -l <"$tmpdir/control.txt" | tr -d ' ')"
pipeline_errors="$(wc -l <"$tmpdir/errors.txt" | tr -d ' ')"

echo "LID metadata run: $RUN_ID"
echo "Prefix: $PREFIX"
echo
echo "Manifest row groups: $manifest_rows"
echo "Completed parquet shards: $parquet_shards"
echo "Metrics sidecars: $metric_sidecars"
echo "Error sidecars: $error_sidecars"
echo "Checkpoints: $checkpoints"
echo "Pipeline error files: $pipeline_errors"
echo "Control files: $control_files"

if [ "$manifest_rows" -gt 0 ]; then
    python - "$manifest_rows" "$parquet_shards" <<'PY'
import sys

total = int(sys.argv[1])
done = int(sys.argv[2])
print(f"Parquet completion: {done / total * 100:.2f}%")
print(f"Remaining row groups: {max(total - done, 0)}")
PY
fi

print_source_counts "Completed parquet shards" ".parquet" "$tmpdir/shards.txt"
print_source_counts "Checkpoints" ".json" "$tmpdir/checkpoints.txt"
print_source_counts "Error sidecars" ".error.json" "$tmpdir/shards.txt"
