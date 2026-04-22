#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
export RUNPOD_API_KEY="$(<credentials/runpod/runpod_api_key)"
python3 - "${RUNPOD_NAME:-training-signal-processing-5090-ssh}" <<'PY'
import json, os, pathlib, sys, urllib.error, urllib.request

name = sys.argv[1]
headers = {"Authorization": f"Bearer {os.environ['RUNPOD_API_KEY']}", "Content-Type": "application/json"}
machine_files = [pathlib.Path(".current-machine"), pathlib.Path("current-machine")]

def call(method, path):
    req = urllib.request.Request(f"https://rest.runpod.io/v1{path}", headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            return json.load(resp) if resp.length else None
    except urllib.error.HTTPError as exc:
        raise SystemExit(exc.read().decode() or str(exc)) from exc

pods = [pod for pod in call("GET", "/pods") if pod.get("name") == name]
for pod in pods:
    call("DELETE", f"/pods/{pod['id']}")
    print(f"deleted {pod['id']}")
if pods:
    for path in machine_files:
        path.unlink(missing_ok=True)
if not pods:
    print(f"no running pods named {name}")
PY
