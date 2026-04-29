#!/usr/bin/env bash
set -euo pipefail

RAY_HEAD_PORT="${RAY_HEAD_PORT:-6379}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
RAY_CLUSTER_WAIT_SECONDS="${RAY_CLUSTER_WAIT_SECONDS:-900}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-$(nproc)}"
LID_METADATA_CONFIG_PATH="${LID_METADATA_CONFIG_PATH:-/app/config/lid_metadata.sample.yaml}"

require_env() {
    local name="$1"
    if [ -z "${!name:-}" ]; then
        echo "[lid-dstack] ERROR: required env var is missing: $name" >&2
        exit 2
    fi
}

require_remote_job_env() {
    require_env LID_METADATA_RUN_ID
    require_env LID_METADATA_CONFIG_OBJECT_KEY
    require_env LID_METADATA_INPUT_MANIFEST_KEY
    require_env LID_METADATA_UPLOADED_ITEMS
}

wait_for_port() {
    local host="$1"
    local port="$2"
    local timeout="$3"
    python - "$host" "$port" "$timeout" <<'PY'
import socket
import sys
import time

host = sys.argv[1]
port = int(sys.argv[2])
deadline = time.time() + float(sys.argv[3])
while time.time() < deadline:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(2)
        try:
            sock.connect((host, port))
        except OSError:
            time.sleep(2)
        else:
            sys.exit(0)
raise SystemExit(f"Timed out waiting for {host}:{port}")
PY
}

wait_for_ray_nodes() {
    local address="$1"
    local expected="$2"
    local timeout="$3"
    uv run --no-sync --group lid_metadata python - "$address" "$expected" "$timeout" <<'PY'
import sys
import time

import ray

address = sys.argv[1]
expected = int(sys.argv[2])
deadline = time.time() + float(sys.argv[3])
ray.init(address=address, ignore_reinit_error=True)
while time.time() < deadline:
    alive = [node for node in ray.nodes() if node.get("Alive")]
    print(f"[lid-dstack] Ray alive nodes: {len(alive)}/{expected}", flush=True)
    if len(alive) >= expected:
        sys.exit(0)
    time.sleep(10)
raise SystemExit(f"Timed out waiting for {expected} Ray nodes")
PY
}

run_canary() {
    echo "[lid-dstack] Running canary checks on Ray head"
    uv run --no-sync --group lid_metadata python -m training_signal_processing.main \
        lid-metadata-validate \
        --config "$LID_METADATA_CONFIG_PATH" \
        --set ray.concurrency=2 \
        --set ray.target_num_blocks=2
    python - <<'PY'
import os

required = [
    "R2_BUCKET",
    "R2_ACCESS_KEY_ID",
    "R2_SECRET_ACCESS_KEY",
    "R2_REGION",
    "R2_ENDPOINT_URL",
]
missing = [name for name in required if not os.environ.get(name)]
if missing:
    raise SystemExit(f"Missing R2 env vars: {', '.join(missing)}")
print("[lid-dstack] R2 environment keys are present", flush=True)
PY
    uv run --no-sync --group lid_metadata python - <<'PY'
from training_signal_processing.pipelines.lid_metadata.ops import (
    build_lingua_detector,
    build_malaya_runtime,
)

build_lingua_detector()
print("[lid-dstack] Lingua detector initialized", flush=True)
build_malaya_runtime(quantized=True)
print("[lid-dstack] Malaya runtime initialized", flush=True)
PY
}

start_ray_head() {
    echo "[lid-dstack] Starting Ray head at ${DSTACK_MASTER_NODE_IP}:${RAY_HEAD_PORT}"
    uv run --no-sync --group lid_metadata ray stop --force >/dev/null 2>&1 || true
    uv run --no-sync --group lid_metadata ray start \
        --head \
        --node-ip-address="$DSTACK_MASTER_NODE_IP" \
        --port="$RAY_HEAD_PORT" \
        --dashboard-host=0.0.0.0 \
        --dashboard-port="$RAY_DASHBOARD_PORT" \
        --num-cpus="$RAY_NUM_CPUS"
}

start_ray_worker() {
    echo "[lid-dstack] Waiting for Ray head ${DSTACK_MASTER_NODE_IP}:${RAY_HEAD_PORT}"
    wait_for_port "$DSTACK_MASTER_NODE_IP" "$RAY_HEAD_PORT" "$RAY_CLUSTER_WAIT_SECONDS"
    uv run --no-sync --group lid_metadata ray stop --force >/dev/null 2>&1 || true
    echo "[lid-dstack] Joining Ray head ${DSTACK_MASTER_NODE_IP}:${RAY_HEAD_PORT}"
    uv run --no-sync --group lid_metadata ray start \
        --address="${DSTACK_MASTER_NODE_IP}:${RAY_HEAD_PORT}" \
        --num-cpus="$RAY_NUM_CPUS" \
        --block
}

main() {
    DSTACK_NODE_RANK="${DSTACK_NODE_RANK:-0}"
    DSTACK_NODES_NUM="${DSTACK_NODES_NUM:-1}"
    DSTACK_MASTER_NODE_IP="${DSTACK_MASTER_NODE_IP:-127.0.0.1}"

    if [ "$DSTACK_NODE_RANK" != "0" ]; then
        start_ray_worker
        exit 0
    fi

    start_ray_head
    wait_for_ray_nodes "${DSTACK_MASTER_NODE_IP}:${RAY_HEAD_PORT}" "$DSTACK_NODES_NUM" \
        "$RAY_CLUSTER_WAIT_SECONDS"

    export RAY_ADDRESS="${DSTACK_MASTER_NODE_IP}:${RAY_HEAD_PORT}"
    if [ "${LID_METADATA_CANARY_ONLY:-0}" = "1" ]; then
        run_canary
        exit 0
    fi

    require_remote_job_env
    echo "[lid-dstack] Starting LID metadata run ${LID_METADATA_RUN_ID}"
    uv run --no-sync --group lid_metadata python -m training_signal_processing.main \
        lid-metadata-remote-job \
        --run-id "$LID_METADATA_RUN_ID" \
        --config-object-key "$LID_METADATA_CONFIG_OBJECT_KEY" \
        --input-manifest-key "$LID_METADATA_INPUT_MANIFEST_KEY" \
        --uploaded-items "$LID_METADATA_UPLOADED_ITEMS"
}

main "$@"
