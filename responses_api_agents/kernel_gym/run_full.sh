#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

: "${NVIDIA_API_KEY:?Set NVIDIA_API_KEY}"
: "${OPENSANDBOX_API_KEY:?Set OPENSANDBOX_API_KEY}"
: "${OPENSANDBOX_DOMAIN:?Set OPENSANDBOX_DOMAIN}"

ROOT=$(git rev-parse --show-toplevel)
IMAGE=${KERNEL_GYM_IMAGE:-ttl.sh/anykernel-ba8202ec-f689-40db-b1c9-35ae8612125c:24h}
OUTPUT=${KERNEL_GYM_OUTPUT:-results/kernel_gym_super_v3_full_250.jsonl}
LOG=${KERNEL_GYM_SERVER_LOG:-results/kernel_gym_server.log}
CONCURRENCY=${KERNEL_GYM_CONCURRENCY:-48}
STATUS_POLL_TIMEOUT=${KERNEL_GYM_STATUS_POLL_TIMEOUT:-10}
MAX_PASSES=${KERNEL_GYM_MAX_PASSES:-5}
export NEMO_GYM_MAX_ROLLOUT_ATTEMPTS=${NEMO_GYM_MAX_ROLLOUT_ATTEMPTS:-10}
cd "$ROOT"

.venv/bin/python responses_api_agents/kernel_gym/prepare.py \
  --kernelbench ../KernelBench --image "$IMAGE" --all

export NEMO_GYM_SANDBOX_MODEL_BASE_URL=https://inference-api.nvidia.com/v1
.venv/bin/gym env start \
  --config responses_api_agents/kernel_gym/configs/kernel_gym.yaml \
  --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml \
  --model-type vllm_model \
  --model nvidia/nvidia/nemotron-3-super-v3 \
  --model-url "$NEMO_GYM_SANDBOX_MODEL_BASE_URL" \
  --model-api-key "$NVIDIA_API_KEY" \
  ++sandbox.opensandbox.operations.status_poll_timeout_s="$STATUS_POLL_TIMEOUT" \
  ++sandbox.opensandbox.operations.retries=10 \
  ++head_server.host=127.0.0.1 ++head_server.port=11100 >"$LOG" 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true' EXIT

until grep -q 'All 2 / 2 servers ready' "$LOG"; do
  kill -0 "$server_pid"
  sleep 2
done

INPUT=responses_api_agents/kernel_gym/data/kernelbench.jsonl
expected=$(wc -l < "$INPUT" | tr -d ' ')
[[ -n ${KERNEL_GYM_LIMIT:-} ]] && expected=$KERNEL_GYM_LIMIT

for ((pass = 1; pass <= MAX_PASSES; pass++)); do
  args=(--no-serve --agent kernel_gym --input "$INPUT" --output "$OUTPUT")
  [[ $pass -gt 1 || ${KERNEL_GYM_RESUME:-0} == 1 ]] && args+=(--resume)
  [[ -n ${KERNEL_GYM_LIMIT:-} ]] && args+=(--limit "$KERNEL_GYM_LIMIT")
  echo "kernel_gym pass $pass/$MAX_PASSES"
  .venv/bin/gym eval run "${args[@]}" \
    --concurrency "$CONCURRENCY" --max-output-tokens 32768 --no-health-check \
    ++route_failures_to_sidecar=true \
    ++sandbox.opensandbox.operations.status_poll_timeout_s="$STATUS_POLL_TIMEOUT" \
    ++sandbox.opensandbox.operations.retries=10 \
    ++head_server.host=127.0.0.1 ++head_server.port=11100 || true

  completed=$(wc -l < "$OUTPUT" | tr -d ' ')
  if [[ $completed -ge $expected ]]; then
    echo "kernel_gym complete: $completed/$expected"
    exit 0
  fi
  echo "kernel_gym incomplete: $completed/$expected; resuming"
done

echo "kernel_gym incomplete after $MAX_PASSES passes" >&2
exit 1
