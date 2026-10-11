#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
E2E_DIR="${E2E_DIR:-$(mktemp -d "${RUNNER_TEMP:-/tmp}/nemo-gym-nooa-stateful.XXXXXX")}"
RESULTS_DIR="$E2E_DIR/results"
WORKSPACE_DIR="$E2E_DIR/workspace"
VENV_ROOT="$E2E_DIR/venvs"
HEAD_PORT="${HEAD_PORT:-11000}"
SERVER_TIMEOUT_SECS="${SERVER_TIMEOUT_SECS:-900}"
GYM_BIN="${GYM_BIN:-gym}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
POLICY_MODEL_TYPE="${POLICY_MODEL_TYPE:-vllm_model}"
GYM_PID=""

: "${POLICY_BASE_URL:?Set POLICY_BASE_URL to the real model endpoint}"
: "${POLICY_API_KEY:?Set POLICY_API_KEY for the real model endpoint}"
: "${POLICY_MODEL_NAME:?Set POLICY_MODEL_NAME to the served model name}"

show_log_tail() {
  local log_path="$1"
  if [[ -f "$log_path" ]]; then
    echo "===== Last 200 lines of Gym log =====" >&2
    tail -n 200 "$log_path" >&2
  fi
}

stop_process() {
  local pid="$1"
  if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
    return
  fi
  kill -INT "$pid" 2>/dev/null || true
  for _ in $(seq 1 10); do
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid" 2>/dev/null || true
      return
    fi
    sleep 1
  done
  kill -KILL "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}

cleanup() {
  local exit_code=$?
  trap - EXIT
  stop_process "$GYM_PID"
  if [[ "$exit_code" -ne 0 ]]; then
    show_log_tail "$RESULTS_DIR/gym.log"
  fi
  exit "$exit_code"
}
trap cleanup EXIT

command -v "$GYM_BIN" >/dev/null || { echo "Gym executable is not available: $GYM_BIN" >&2; exit 1; }
command -v "$PYTHON_BIN" >/dev/null || { echo "Python executable is not available: $PYTHON_BIN" >&2; exit 1; }
[[ "$E2E_DIR" == /* && "$E2E_DIR" != "/" ]] || { echo "E2E_DIR must be an absolute non-root path" >&2; exit 2; }

mkdir -p "$RESULTS_DIR" "$WORKSPACE_DIR" "$VENV_ROOT"
unset UV_RUN_RECURSION_DEPTH
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0

cd "$WORKSPACE_DIR"
"$PYTHON_BIN" -c \
  "import os, signal, sys; signal.signal(signal.SIGINT, signal.SIG_DFL); os.execvp(sys.argv[1], sys.argv[1:])" \
  "$GYM_BIN" env start \
  --config "$ROOT_DIR/resources_servers/example_session_state_mgmt/configs/example_session_state_mgmt_nooa.yaml" \
  --model-type "$POLICY_MODEL_TYPE" \
  --model-url "$POLICY_BASE_URL" \
  --model-api-key "$POLICY_API_KEY" \
  --model "$POLICY_MODEL_NAME" \
  "++head_server.host=127.0.0.1" \
  "++head_server.port=$HEAD_PORT" \
  "++uv_venv_dir=$VENV_ROOT" \
  "+nemo_gym_log_dir=$RESULTS_DIR/component-logs" \
  > "$RESULTS_DIR/gym.log" 2>&1 &
GYM_PID=$!

"$ROOT_DIR/scripts/wait_for_servers.sh" "$GYM_PID" "$HEAD_PORT" "$SERVER_TIMEOUT_SECS"

"$GYM_BIN" eval run \
  --no-serve \
  --agent example_session_state_mgmt_nooa_agent \
  --input "$ROOT_DIR/resources_servers/example_session_state_mgmt/data/example.jsonl" \
  --output "$RESULTS_DIR/rollouts.jsonl" \
  --limit 1 \
  --concurrency 1 \
  --temperature 0 \
  --max-output-tokens 2048 \
  "++head_server.host=127.0.0.1" \
  "++head_server.port=$HEAD_PORT"

"$PYTHON_BIN" "$ROOT_DIR/responses_api_agents/nooa_agent/smoke/verify_stateful_rollout.py" \
  --rollouts "$RESULTS_DIR/rollouts.jsonl"

echo "NOOA stateful smoke passed; artifacts retained at $RESULTS_DIR"
