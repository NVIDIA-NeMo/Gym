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
E2E_PROVIDER_CONFIG="${E2E_PROVIDER_CONFIG:?Set E2E_PROVIDER_CONFIG to an inference provider config path.}"
E2E_MODEL="${E2E_MODEL:?Set E2E_MODEL to the hosted model identifier.}"
MODEL_API_KEY="${MODEL_API_KEY:?Set MODEL_API_KEY to the hosted provider credential.}" # pragma: allowlist secret
E2E_DIR="${E2E_DIR:-${RUNNER_TEMP:-/tmp}/nemo-gym-inference-provider-e2e}"
INDEX_PORT="${INDEX_PORT:-18888}"
HEAD_PORT="${HEAD_PORT:-11000}"
RESULTS_DIR="$E2E_DIR/results"
VENV_DIR="$E2E_DIR/venv"
INDEX_PID=""
GYM_PID=""

cleanup() {
  if [[ -n "$GYM_PID" ]]; then
    kill "$GYM_PID" 2>/dev/null || true
    wait "$GYM_PID" 2>/dev/null || true
  fi
  if [[ -n "$INDEX_PID" ]]; then
    kill "$INDEX_PID" 2>/dev/null || true
    wait "$INDEX_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if [[ "$E2E_DIR" != /* || "$E2E_DIR" == "/" ]]; then
  echo "E2E_DIR must be an absolute non-root path: $E2E_DIR" >&2
  exit 2
fi

if [[ "$E2E_PROVIDER_CONFIG" == /* ]]; then
  PROVIDER_CONFIG_PATH="$E2E_PROVIDER_CONFIG"
else
  PROVIDER_CONFIG_PATH="$ROOT_DIR/$E2E_PROVIDER_CONFIG"
fi
if [[ ! -f "$PROVIDER_CONFIG_PATH" ]]; then
  echo "Inference provider config does not exist: $E2E_PROVIDER_CONFIG" >&2
  exit 2
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

cd "$E2E_DIR/workspace"
"$VENV_DIR/bin/gym" env start \
  --config "$ROOT_DIR/tests/e2e/inference_provider_env.yaml" \
  --config "$ROOT_DIR/resources_servers/example_single_tool_call/configs/example_single_tool_call.yaml" \
  --config "$PROVIDER_CONFIG_PATH" \
  > "$RESULTS_DIR/gym.log" 2>&1 &
GYM_PID=$!
"$ROOT_DIR/scripts/wait_for_servers.sh" "$GYM_PID" "$HEAD_PORT" 180

"$VENV_DIR/bin/gym" eval run \
  --no-serve \
  --agent example_single_tool_call_simple_agent \
  --input "$ROOT_DIR/tests/e2e/inference_provider_smoke.jsonl" \
  --output "$RESULTS_DIR/rollouts.jsonl" \
  --limit 1 \
  --temperature 0 \
  --max-output-tokens 4096

"$VENV_DIR/bin/python" "$ROOT_DIR/tests/e2e/verify_inference_provider_rollout.py" \
  --rollouts "$RESULTS_DIR/rollouts.jsonl"
