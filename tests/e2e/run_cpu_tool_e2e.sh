#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
E2E_DIR="${E2E_DIR:-${RUNNER_TEMP:-/tmp}/nemo-gym-cpu-e2e}"
PROVIDER_PORT="${PROVIDER_PORT:-19999}"
INDEX_PORT="${INDEX_PORT:-18888}"
HEAD_PORT="${HEAD_PORT:-11000}"
MODEL_API_KEY="${MODEL_API_KEY:-not-a-real-key}" # pragma: allowlist secret
RESULTS_DIR="$E2E_DIR/results"
VENV_DIR="$E2E_DIR/venv"
PROVIDER_PID=""
INDEX_PID=""
GYM_PID=""

cleanup() {
  if [[ -n "$GYM_PID" ]]; then
    kill "$GYM_PID" 2>/dev/null || true
    wait "$GYM_PID" 2>/dev/null || true
  fi
  if [[ -n "$PROVIDER_PID" ]]; then
    kill "$PROVIDER_PID" 2>/dev/null || true
    wait "$PROVIDER_PID" 2>/dev/null || true
  fi
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
mkdir -p "$RESULTS_DIR" "$E2E_DIR/dist" "$E2E_DIR/index/nemo-gym" "$E2E_DIR/workspace"

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

"$VENV_DIR/bin/python" "$ROOT_DIR/tests/e2e/scripted_provider.py" \
  --port "$PROVIDER_PORT" \
  --events "$RESULTS_DIR/provider-events.jsonl" \
  > "$RESULTS_DIR/provider.log" 2>&1 &
PROVIDER_PID=$!

for _ in $(seq 1 30); do
  if curl --fail --silent "http://127.0.0.1:${PROVIDER_PORT}/healthz" >/dev/null; then
    break
  fi
  if ! kill -0 "$PROVIDER_PID" 2>/dev/null; then
    echo "Scripted provider exited before becoming ready." >&2
    exit 1
  fi
  sleep 1
done
curl --fail --silent "http://127.0.0.1:${PROVIDER_PORT}/healthz" >/dev/null

cd "$E2E_DIR/workspace"
"$VENV_DIR/bin/gym" env start \
  --config "$ROOT_DIR/resources_servers/example_single_tool_call/configs/example_single_tool_call.yaml" \
  --config "$ROOT_DIR/responses_api_models/vllm_model/configs/vllm_model.yaml" \
  --model-url "http://127.0.0.1:${PROVIDER_PORT}/v1" \
  --model-api-key "$MODEL_API_KEY" \
  --model scripted-model \
  > "$RESULTS_DIR/gym.log" 2>&1 &
GYM_PID=$!
"$ROOT_DIR/scripts/wait_for_servers.sh" "$GYM_PID" "$HEAD_PORT" 180

"$VENV_DIR/bin/gym" eval run \
  --no-serve \
  --agent example_single_tool_call_simple_agent \
  --input "$ROOT_DIR/resources_servers/example_single_tool_call/data/example.jsonl" \
  --output "$RESULTS_DIR/rollouts.jsonl" \
  --limit 1

"$VENV_DIR/bin/python" "$ROOT_DIR/tests/e2e/verify_tool_rollout.py" \
  --rollouts "$RESULTS_DIR/rollouts.jsonl" \
  --provider-events "$RESULTS_DIR/provider-events.jsonl"
