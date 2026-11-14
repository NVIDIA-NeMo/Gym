#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
umask 077
gym_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$gym_root"
export PATH="$gym_root/.tools/uv/bin:$gym_root/.venv/bin:$PATH"
export UV_CACHE_DIR="$gym_root/cache/uv"
export PYTHONDONTWRITEBYTECODE=1
export NEMO_GYM_MAX_ROLLOUT_ATTEMPTS=1
unset NEMO_GYM_EXTRA_ROOTS PYTHONPATH
mkdir -p results/partner_if/ngc-recorded-context
run_dir="$(mktemp -d "$gym_root/results/partner_if/ngc-recorded-context/run-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")"
printf 'replay_artifacts: %s/captures\nmodel_call_capture_dir: %s/model_calls\nobservability_enabled: true\nmlflow_tracking_required: false\nmlflow_tracking_uri: null\nmlflow_tracking_token: null\n' "$run_dir" "$run_dir" > "$run_dir/run.yaml"
echo "Results: $run_dir"
# Prepare with the registry credentials before disabling evaluation-result delivery.
.venv/bin/gym eval prepare \
  --config benchmarks/partner_if/config_ngc_replay.yaml --config env.yaml
exec .venv/bin/gym eval run \
  --config benchmarks/partner_if/config_ngc_replay.yaml \
  --config env.yaml --config "$run_dir/run.yaml" \
  --agent partner_if_dispatcher --split benchmark --model-type vllm_model \
  --limit 1 --num-repeats 1 --concurrency 1 \
  --output "$run_dir/rollouts.jsonl" "$@"
