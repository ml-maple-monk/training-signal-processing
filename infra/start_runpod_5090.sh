#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
export RUNPOD_API_KEY="$(<credentials/runpod/runpod_api_key)" RUNPOD_SSH_PUBLIC_KEY="$(<"$HOME/.ssh/id_ed25519.pub")"
python3 - "${RUNPOD_NAME:-training-signal-processing-5090-ssh}" "${RUNPOD_COUNT:-1}" <<'PY'
import json, os, pathlib, sys, time, urllib.error, urllib.request

name, count = sys.argv[1], int(sys.argv[2])
headers = {"Authorization": f"Bearer {os.environ['RUNPOD_API_KEY']}", "Content-Type": "application/json"}
machine_files = [pathlib.Path(".current-machine"), pathlib.Path("current-machine")]

def call(method, path, payload=None):
    req = urllib.request.Request(
        f"https://rest.runpod.io/v1{path}",
        data=None if payload is None else json.dumps(payload).encode(),
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as exc:
        raise SystemExit(exc.read().decode() or str(exc)) from exc

payload = {
    "name": name,
    "computeType": "GPU",
    "cloudType": "COMMUNITY",
    "gpuCount": 1,
    "gpuTypeIds": ["NVIDIA GeForce RTX 5090"],
    "gpuTypePriority": "custom",
    "imageName": "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04",
    "containerDiskInGb": 50,
    "volumeInGb": 0,
    "ports": ["22/tcp"],
    "supportPublicIp": True,
    "env": {
        "PUBLIC_KEY": os.environ["RUNPOD_SSH_PUBLIC_KEY"],
        "SSH_PUBLIC_KEY": os.environ["RUNPOD_SSH_PUBLIC_KEY"],
    },
}

for pod_id in [call("POST", "/pods", payload)["id"] for _ in range(count)]:
    for _ in range(60):
        pod = call("GET", f"/pods/{pod_id}")
        ip, port = pod.get("publicIp"), (pod.get("portMappings") or {}).get("22")
        if ip and port:
            ssh = f"ssh -i ~/.ssh/id_ed25519 -p {port} root@{ip}"
            for path in machine_files:
                path.write_text(f"{ssh}\n")
            print(f"{pod_id}\t{ssh}")
            break
        time.sleep(5)
    else:
        raise SystemExit(f"Timed out waiting for SSH on {pod_id}")
PY
