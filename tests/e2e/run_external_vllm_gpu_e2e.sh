#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
E2E_DIR="${E2E_DIR:-${RUNNER_TEMP:-/tmp}/nemo-gym-gpu-e2e}"
VLLM_IMAGE="${VLLM_IMAGE:?Set VLLM_IMAGE to a pinned vLLM OpenAI image.}"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
GPU_DEVICE="${GPU_DEVICE:-0}"
HF_CACHE_DIR="${HF_CACHE_DIR:-${HF_HOME:-$HOME/.cache/huggingface}}"
VLLM_PORT="${VLLM_PORT:-18000}"
INDEX_PORT="${INDEX_PORT:-28888}"
HEAD_PORT="${HEAD_PORT:-11000}"
MODEL_API_KEY="${MODEL_API_KEY:-not-a-real-key}" # pragma: allowlist secret
VLLM_CONTAINER="${VLLM_CONTAINER:-nemo-gym-vllm-${GITHUB_RUN_ID:-$$}}"
RESULTS_DIR="$E2E_DIR/results"
VENV_DIR="$E2E_DIR/venv"
INDEX_PID=""
GYM_PID=""

cleanup() {
  if [[ -n "$GYM_PID" ]]; then
    kill "$GYM_PID" 2>/dev/null || true
    wait "$GYM_PID" 2>/dev/null || true
  fi
  docker logs "$VLLM_CONTAINER" > "$RESULTS_DIR/vllm.log" 2>&1 || true
  docker rm --force "$VLLM_CONTAINER" >/dev/null 2>&1 || true
  if [[ -n "$INDEX_PID" ]]; then
    kill "$INDEX_PID" 2>/dev/null || true
    wait "$INDEX_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if [[ "$E2E_DIR" == "/" ]]; then
  echo "E2E_DIR cannot be the filesystem root." >&2
  exit 1
fi
rm -rf -- "$E2E_DIR"
mkdir -p "$RESULTS_DIR" "$E2E_DIR/dist" "$E2E_DIR/index/nemo-gym" "$E2E_DIR/workspace" "$HF_CACHE_DIR"

uv build "$ROOT_DIR" --wheel --out-dir "$E2E_DIR/dist"
cp "$E2E_DIR"/dist/*.whl "$E2E_DIR/index/nemo-gym/"
for wheel in "$E2E_DIR"/index/nemo-gym/*.whl; do
  wheel_name="$(basename "$wheel")"
  printf '<a href="%s">%s</a>\n' "$wheel_name" "$wheel_name"
done > "$E2E_DIR/index/nemo-gym/index.html"
printf '<a href="nemo-gym/">nemo-gym</a>\n' > "$E2E_DIR/index/index.html"

python3 -m http.server "$INDEX_PORT" --bind 127.0.0.1 --directory "$E2E_DIR/index" \
  > "$RESULTS_DIR/package-index.log" 2>&1 &
INDEX_PID=$!
for _ in $(seq 1 10); do
  if curl --fail --silent "http://127.0.0.1:${INDEX_PORT}/nemo-gym/" >/dev/null; then
    break
  fi
  sleep 1
done
curl --fail --silent "http://127.0.0.1:${INDEX_PORT}/nemo-gym/" >/dev/null

uv venv "$VENV_DIR"
uv pip install --python "$VENV_DIR/bin/python" "$E2E_DIR"/dist/*.whl
export NEMO_GYM_ALLOW_PRERELEASE=true
export UV_INDEX_URL="http://127.0.0.1:${INDEX_PORT}/"
export UV_EXTRA_INDEX_URL="https://pypi.org/simple/"

docker run --detach --rm \
  --name "$VLLM_CONTAINER" \
  --gpus "device=${GPU_DEVICE}" \
  --ipc host \
  --publish "127.0.0.1:${VLLM_PORT}:8000" \
  --volume "$HF_CACHE_DIR:/root/.cache/huggingface" \
  "$VLLM_IMAGE" \
  "$MODEL" \
  --served-model-name "$MODEL" \
  --dtype half \
  --enforce-eager \
  --gpu-memory-utilization 0.5 \
  --max-model-len 2048 \
  > "$RESULTS_DIR/vllm-container-id.txt"

for _ in $(seq 1 300); do
  if curl --fail --silent "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null; then
    break
  fi
  if [[ "$(docker inspect "$VLLM_CONTAINER" --format '{{.State.Running}}' 2>/dev/null || true)" != "true" ]]; then
    docker logs "$VLLM_CONTAINER" > "$RESULTS_DIR/vllm.log" 2>&1 || true
    echo "vLLM exited before becoming ready." >&2
    exit 1
  fi
  sleep 1
done
curl --fail --silent "http://127.0.0.1:${VLLM_PORT}/v1/models" > "$RESULTS_DIR/vllm-models.json"

cd "$E2E_DIR/workspace"
"$VENV_DIR/bin/gym" env start \
  --config "$ROOT_DIR/resources_servers/example_single_tool_call/configs/example_single_tool_call.yaml" \
  --config "$ROOT_DIR/responses_api_models/vllm_model/configs/vllm_model.yaml" \
  --model-url "http://127.0.0.1:${VLLM_PORT}/v1" \
  --model-api-key "$MODEL_API_KEY" \
  --model "$MODEL" \
  > "$RESULTS_DIR/gym.log" 2>&1 &
GYM_PID=$!
"$ROOT_DIR/scripts/wait_for_servers.sh" "$GYM_PID" "$HEAD_PORT" 180

"$VENV_DIR/bin/gym" eval run \
  --no-serve \
  --agent example_single_tool_call_simple_agent \
  --input "$ROOT_DIR/tests/e2e/gpu_smoke.jsonl" \
  --output "$RESULTS_DIR/rollouts.jsonl" \
  --limit 1 \
  --max-output-tokens 32

"$VENV_DIR/bin/python" "$ROOT_DIR/tests/e2e/verify_gpu_rollout.py" \
  --rollouts "$RESULTS_DIR/rollouts.jsonl"

docker logs "$VLLM_CONTAINER" > "$RESULTS_DIR/vllm.log" 2>&1
