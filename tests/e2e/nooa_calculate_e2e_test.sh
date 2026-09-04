#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
E2E_DIR="${E2E_DIR:-${RUNNER_TEMP:-/tmp}/nemo-gym-nooa-e2e}"
RESULTS_DIR="$E2E_DIR/results"
MODEL_PORT="${MODEL_PORT:-18080}"
HEAD_PORT="${HEAD_PORT:-11000}"
MODEL_PID=""
GYM_PID=""
GYM_BIN="${GYM_BIN:-$ROOT_DIR/.venv/bin/gym}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
SERVER_VENV="${SERVER_VENV:-}"

show_log_tail() {
  local label="$1" log_path="$2"
  if [[ -f "$log_path" ]]; then
    echo "===== Last 200 lines of $label =====" >&2
    tail -n 200 "$log_path" >&2
  fi
}

stop_process() {
  local pid="$1" signal="${2:-TERM}"
  if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then return; fi
  kill "-$signal" "$pid" 2>/dev/null || true
  for _ in $(seq 1 10); do
    if ! kill -0 "$pid" 2>/dev/null; then wait "$pid" 2>/dev/null || true; return; fi
    sleep 1
  done
  kill -KILL "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}

cleanup() {
  local exit_code=$?
  trap - EXIT
  stop_process "$GYM_PID" INT
  stop_process "$MODEL_PID"
  if [[ "$exit_code" -ne 0 ]]; then
    show_log_tail "Gym log" "$RESULTS_DIR/gym.log"
    show_log_tail "model log" "$RESULTS_DIR/model.log"
  fi
  exit "$exit_code"
}
trap cleanup EXIT

wait_for_url() {
  local name="$1" url="$2" pid="$3" deadline=$((SECONDS + 60))
  until curl --connect-timeout 2 --max-time 5 --fail --silent "$url" >/dev/null; do
    kill -0 "$pid" 2>/dev/null || { echo "$name exited before readiness" >&2; return 1; }
    (( SECONDS < deadline )) || { echo "$name readiness timeout" >&2; return 1; }
    sleep 1
  done
}

for command in curl timeout; do
  command -v "$command" >/dev/null || { echo "Required command is not installed: $command" >&2; exit 1; }
done
[[ -x "$GYM_BIN" ]] || { echo "Gym executable is not available: $GYM_BIN" >&2; exit 1; }
[[ -x "$PYTHON_BIN" ]] || { echo "Python executable is not available: $PYTHON_BIN" >&2; exit 1; }
[[ "$E2E_DIR" == /* && "$E2E_DIR" != "/" ]] || { echo "E2E_DIR must be an absolute non-root path" >&2; exit 2; }
rm -rf -- "$E2E_DIR"
mkdir -p "$RESULTS_DIR" "$E2E_DIR/workspace"

# By default Gym creates isolated component environments from their committed
# requirements. These overrides are only for coordinated local development.
NOOA_SOURCE_DIR="${NOOA_SOURCE_DIR:-}"
VENV_ROOT="$E2E_DIR/venvs"
if [[ -n "$SERVER_VENV" ]]; then
  for component in \
    responses_api_agents/nooa_agent \
    resources_servers/nooa_capability \
    responses_api_models/openai_model; do
    mkdir -p "$VENV_ROOT/$component"
    ln -s "$SERVER_VENV" "$VENV_ROOT/$component/.venv"
  done
  export NOOA_E2E_SKIP_VENV=true
fi
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
if [[ -n "$NOOA_SOURCE_DIR" ]]; then
  [[ -d "$NOOA_SOURCE_DIR/src/nooa" ]] || { echo "Invalid NOOA_SOURCE_DIR: $NOOA_SOURCE_DIR" >&2; exit 2; }
  export PYTHONPATH="$NOOA_SOURCE_DIR/src:$PYTHONPATH"
fi
# The test may itself be launched by `uv run`; child server cwd differs from the root project.
unset UV_RUN_RECURSION_DEPTH
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export NOOA_E2E_CAPTURE_DIR="$RESULTS_DIR/model-calls"
mkdir -p "$NOOA_E2E_CAPTURE_DIR"

"$PYTHON_BIN" "$ROOT_DIR/tests/e2e/deterministic_nooa_responses_server.py" --port "$MODEL_PORT" \
  > "$RESULTS_DIR/model.log" 2>&1 &
MODEL_PID=$!
wait_for_url "deterministic model" "http://127.0.0.1:${MODEL_PORT}/v1/models" "$MODEL_PID"

cd "$E2E_DIR/workspace"
"$PYTHON_BIN" -c \
  "import os, signal, sys; signal.signal(signal.SIGINT, signal.SIG_DFL); os.execvp(sys.argv[1], sys.argv[1:])" \
  "$GYM_BIN" env start \
  --config "$ROOT_DIR/tests/e2e/nooa_calculate_e2e.yaml" \
  --model-url "http://127.0.0.1:${MODEL_PORT}/v1" \
  --model-api-key not-a-real-key \
  --model deterministic-nooa \
  "++head_server.host=127.0.0.1" \
  "++head_server.port=$HEAD_PORT" \
  "++uv_venv_dir=$VENV_ROOT" \
  "+nemo_gym_log_dir=$RESULTS_DIR/component-logs" \
  > "$RESULTS_DIR/gym.log" 2>&1 &
GYM_PID=$!
"$ROOT_DIR/scripts/wait_for_servers.sh" "$GYM_PID" "$HEAD_PORT" 180

timeout --signal=INT --kill-after=30s 180 "$GYM_BIN" eval run \
  --no-serve \
  --agent nooa_calculate_capability \
  --input "$ROOT_DIR/responses_api_agents/nooa_agent/data/capability_calculate.jsonl" \
  --output "$RESULTS_DIR/rollouts.jsonl" \
  --limit 2 \
  --concurrency 1 \
  --temperature 0 \
  --max-output-tokens 64 \
  "++head_server.host=127.0.0.1" \
  "++head_server.port=$HEAD_PORT"

"$PYTHON_BIN" "$ROOT_DIR/tests/e2e/verify_nooa_calculate_rollout.py" \
  --rollouts "$RESULTS_DIR/rollouts.jsonl"
