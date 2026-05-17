#!/usr/bin/env bash
# Serve Qwen/Qwen3-14B-AWQ with vLLM for the FineWeb-Edu → Malay translator.
#
# Replicates the proven recipe from
#   /home/geeyang/workspace/minimind-mfu-working/evaluation/run_pretrained_suite.sh
#   /home/geeyang/workspace/minimind-mfu-working/evaluation/sea_helm_ms/README.md
# (do NOT improvise — this Qwen3 + vLLM + flashinfer env is quirky):
#   - isolated uv venv; `uv pip install vllm transformers ...` pulls a
#     self-consistent torch + flashinfer (do NOT install standalone flash-attn)
#   - --enforce-eager ALWAYS (stability)
#   - NO --reasoning-parser: the full <think>…</think> must stay in `content`;
#     the runner strips it (README.md:80-81)
#   - NO VLLM_ATTENTION_BACKEND override: flashinfer is implicit
#   - AWQ 4-bit via --quantization awq_marlin (fp4 needs Blackwell; README:189)
#   - cleanup must also kill the VLLM::EngineCore child or the GPU stays pinned
#
# Run (terminal A), then run translate_fineweb_edu_malay.py in terminal B:
#   bash src/data-storage/scripts/serve_qwen3_translate.sh
#
# Overridable via env: MODEL, PORT, API_KEY, MAX_MODEL_LEN, GPU_UTIL, VENV, LOG.
set -euo pipefail

# repo root = scripts/ -> data-storage/ -> src/ -> <root>
cd "$(dirname "$0")/../../.."

MODEL="${MODEL:-Qwen/Qwen3-14B-AWQ}"
PORT="${PORT:-8000}"
API_KEY="${API_KEY:-translate}"
# Qwen3-14B native long context. YaRN/--rope-scaling to 131072 is an optional,
# env-fighting opt-in — NOT enabled here by default.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_UTIL="${GPU_UTIL:-0.90}"
VENV="${VENV:-.venv-qwen-translate}"
LOG="${LOG:-/tmp/qwen3_translate_vllm.log}"

if [ ! -x "$VENV/bin/vllm" ]; then
  echo "[setup] creating $VENV and installing vllm transformers openai asyncpg"
  uv venv "$VENV" --python 3.12
  # vllm pulls a self-consistent torch + flashinfer; do NOT add flash-attn.
  # ninja is required: flashinfer JIT-compiles its sampling kernel at runtime
  # (needs ninja + nvcc; the box has /usr/local/cuda nvcc).
  uv pip install --python "$VENV" vllm transformers openai asyncpg ninja
fi

VLLM="$VENV/bin/vllm"

# flashinfer JIT-compiles its sampling kernel at runtime via a bare `ninja`
# subprocess and nvcc. We invoke $VENV/bin/vllm without activating the venv, so
# put the venv bin (ninja) AND a torch-matching CUDA toolkit on PATH. torch is
# built for cu130, so pin CUDA_HOME to the 13.0 toolkit (the /usr/local/cuda
# symlink points at 12.8 here, which would mismatch).
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
export CUDA_HOME
export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH"

SERVE_PID=""
cleanup() {
  # kill the launcher, its process group, and the vLLM EngineCore child
  # (a spawned subprocess that does NOT match "vllm serve" and otherwise
  # keeps the whole GPU pool reserved after the parent exits).
  [ -n "$SERVE_PID" ] && kill "$SERVE_PID" 2>/dev/null || true
  [ -n "$SERVE_PID" ] && kill -9 -- "-$SERVE_PID" 2>/dev/null || true
  pkill -9 -f "vllm serve $MODEL" 2>/dev/null || true
  pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
  for _ in 1 2 3 4 5; do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
        2>/dev/null | head -1)
    [ -n "${u:-}" ] && [ "$u" -lt 800 ] && break
    sleep 2
  done
}
trap cleanup EXIT INT TERM

echo "[1/2] starting $MODEL on :$PORT (awq_marlin, max-model-len=$MAX_MODEL_LEN, util=$GPU_UTIL)"
"$VLLM" serve "$MODEL" \
  --quantization awq_marlin \
  --host 127.0.0.1 --port "$PORT" --api-key "$API_KEY" \
  --gpu-memory-utilization "$GPU_UTIL" --max-model-len "$MAX_MODEL_LEN" \
  --enforce-eager > "$LOG" 2>&1 &
SERVE_PID=$!

echo "[2/2] waiting for endpoint (pid=$SERVE_PID, log=$LOG)"
for _ in $(seq 1 240); do
  if curl -sf -H "Authorization: Bearer $API_KEY" \
       "http://127.0.0.1:$PORT/v1/models" -o /dev/null 2>/dev/null; then
    echo "      vLLM ready on http://127.0.0.1:$PORT/v1 (model=$MODEL)"
    echo "      run: python3 src/data-storage/scripts/translate_fineweb_edu_malay.py \\"
    echo "             --base-url http://127.0.0.1:$PORT/v1 --api-key $API_KEY --limit 5"
    break
  fi
  if ! kill -0 "$SERVE_PID" 2>/dev/null; then
    echo "ERROR: vLLM died during startup:"; tail -40 "$LOG"; exit 1
  fi
  sleep 5
done
if ! curl -sf -H "Authorization: Bearer $API_KEY" \
     "http://127.0.0.1:$PORT/v1/models" -o /dev/null 2>/dev/null; then
  echo "ERROR: vLLM not ready after 20 min:"; tail -40 "$LOG"; exit 1
fi

# Keep the server in the foreground; Ctrl-C triggers the cleanup trap.
wait "$SERVE_PID"
