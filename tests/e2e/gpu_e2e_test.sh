#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
E2E_DIR="${E2E_DIR:-${RUNNER_TEMP:-/tmp}/nemo-gym-gpu-e2e}"
RESULTS_DIR="${RESULTS_DIR:-$E2E_DIR/results}"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
MODEL_REVISION="${MODEL_REVISION:-7ae557604adf67be50417f59c2c2f167def9a775}"
EXPECTED_ANSWER="${EXPECTED_ANSWER:-Paris}"
GPU_DEVICE="${GPU_DEVICE:-0}"
VLLM_PORT="${VLLM_PORT:-18000}"
HEAD_PORT="${HEAD_PORT:-11000}"
MODEL_API_KEY="${MODEL_API_KEY:-not-a-real-key}" # pragma: allowlist secret
VLLM_STARTUP_TIMEOUT_SECONDS="${VLLM_STARTUP_TIMEOUT_SECONDS:-300}"
GYM_STARTUP_TIMEOUT_SECONDS="${GYM_STARTUP_TIMEOUT_SECONDS:-180}"
EVAL_TIMEOUT_SECONDS="${EVAL_TIMEOUT_SECONDS:-300}"
VLLM_PID=""
GYM_PID=""

if [[ -z "${HF_HOME:-}" ]]; then
  if [[ -n "${TEST_DATA_PATH:-}" ]]; then
    HF_HOME="$TEST_DATA_PATH/HF_HOME"
  else
    HF_HOME="$HOME/.cache/huggingface"
  fi
fi

show_log_tail() {
  local label="$1"
  local log_path="$2"

  if [[ -f "$log_path" ]]; then
    echo "===== Last 200 lines of $label =====" >&2
    tail -n 200 "$log_path" >&2
  fi
}

stop_process() {
  local pid="$1"
  local signal="${2:-TERM}"

  if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
    return
  fi

  kill "-$signal" "$pid" 2>/dev/null || true
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

  stop_process "$GYM_PID" INT
  stop_process "$VLLM_PID"

  if [[ "$exit_code" -ne 0 ]]; then
    show_log_tail "Gym log" "$RESULTS_DIR/gym.log"
    show_log_tail "vLLM log" "$RESULTS_DIR/vllm.log"
  fi

  exit "$exit_code"
}
trap cleanup EXIT

wait_for_url() {
  local name="$1"
  local url="$2"
  local pid="$3"
  local timeout_seconds="$4"
  local log_path="$5"
  local deadline=$((SECONDS + timeout_seconds))

  echo "Waiting up to ${timeout_seconds}s for $name at $url ..."
  until curl --connect-timeout 2 --max-time 5 --fail --silent "$url" >/dev/null; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "$name exited before becoming ready." >&2
      show_log_tail "$name log" "$log_path"
      return 1
    fi
    if ((SECONDS >= deadline)); then
      echo "$name did not become ready within ${timeout_seconds}s." >&2
      show_log_tail "$name log" "$log_path"
      return 1
    fi
    sleep 2
  done
  echo "$name is ready."
}

for command in curl gym nvidia-smi python python3 timeout uv vllm; do
  if ! command -v "$command" >/dev/null; then
    echo "Required command is not installed: $command" >&2
    exit 1
  fi
done

for directory in "$E2E_DIR" "$RESULTS_DIR" "$HF_HOME"; do
  if [[ "$directory" != /* ]]; then
    echo "E2E_DIR, RESULTS_DIR, and HF_HOME must be absolute paths: $directory" >&2
    exit 1
  fi
  mkdir -p "$directory"
  if [[ "$(cd "$directory" && pwd -P)" == "/" ]]; then
    echo "E2E_DIR, RESULTS_DIR, and HF_HOME cannot resolve to the filesystem root." >&2
    exit 1
  fi
done

WORKSPACE_DIR="$(mktemp -d "$E2E_DIR/workspace.XXXXXX")"

export CUDA_VISIBLE_DEVICES="$GPU_DEVICE"
export HF_HOME
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export NEMO_GYM_VLLM_TRANSPORT_LOG="$RESULTS_DIR/vllm-transport.jsonl"

nvidia-smi | tee "$RESULTS_DIR/nvidia-smi.txt"
bash "$ROOT_DIR/docker/install_codec_deps.sh"

vllm serve "$MODEL" \
  --revision "$MODEL_REVISION" \
  --tokenizer-revision "$MODEL_REVISION" \
  --served-model-name "$MODEL" \
  --host 127.0.0.1 \
  --port "$VLLM_PORT" \
  --dtype half \
  --enforce-eager \
  --gpu-memory-utilization 0.5 \
  --max-model-len 2048 \
  --tensor-parallel-size 1 \
  > "$RESULTS_DIR/vllm.log" 2>&1 &
VLLM_PID=$!

wait_for_url \
  "vLLM" \
  "http://127.0.0.1:${VLLM_PORT}/v1/models" \
  "$VLLM_PID" \
  "$VLLM_STARTUP_TIMEOUT_SECONDS" \
  "$RESULTS_DIR/vllm.log"
curl --connect-timeout 2 --max-time 5 --fail --silent \
  "http://127.0.0.1:${VLLM_PORT}/v1/models" \
  > "$RESULTS_DIR/vllm-models.json"

cd "$WORKSPACE_DIR"
# Bash starts asynchronous commands with SIGINT ignored. Reset it before exec so
# Gym can catch the cleanup interrupt and gracefully stop its child servers.
python3 -c \
  "import os, signal, sys; signal.signal(signal.SIGINT, signal.SIG_DFL); os.execvp(sys.argv[1], sys.argv[1:])" \
  gym env start \
  --config "$ROOT_DIR/tests/e2e/gpu_e2e.yaml" \
  --model-url "http://127.0.0.1:${VLLM_PORT}/v1" \
  --model-api-key "$MODEL_API_KEY" \
  --model "$MODEL" \
  "++head_server.host=127.0.0.1" \
  "++head_server.port=$HEAD_PORT" \
  "+nemo_gym_log_dir=$RESULTS_DIR/component-logs" \
  > "$RESULTS_DIR/gym.log" 2>&1 &
GYM_PID=$!
"$ROOT_DIR/scripts/wait_for_servers.sh" "$GYM_PID" "$HEAD_PORT" "$GYM_STARTUP_TIMEOUT_SECONDS"

timeout --signal=INT --kill-after=30s "$EVAL_TIMEOUT_SECONDS" gym eval run \
  --no-serve \
  --agent string_match_simple_agent \
  --input "$ROOT_DIR/tests/e2e/gpu_smoke.jsonl" \
  --output "$RESULTS_DIR/rollouts.jsonl" \
  --limit 1 \
  --concurrency 1 \
  --temperature 0 \
  --max-output-tokens 64

python3 "$ROOT_DIR/tests/e2e/verify_gpu_rollout.py" \
  --rollouts "$RESULTS_DIR/rollouts.jsonl" \
  --expected-model "$MODEL" \
  --expected-answer "$EXPECTED_ANSWER"
