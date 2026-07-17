#!/usr/bin/env bash
# NeMo Lens Observability Example — sandbox-backed eval
#
# Prerequisites:
#   opensandbox-server running at localhost:8080
#   Python 3.13 venv with nemo-lens installed:
#     uv venv --python 3.13 && uv sync --extra dev --extra lens
#
# Usage:
#   OPENSANDBOX_API_KEY=<key> OPENSANDBOX_DOMAIN=localhost:8080 \
#     bash example_environments/example_nemo_lens_observability/run_example.sh
#
# IMPORTANT: The observability env vars must be present in the gym env start
# process (not just gym eval run) so that mini_swe_agent_2 can forward them
# to the Ray workers that actually execute sandbox operations.

set -euo pipefail

OBS_DIR="${LENS_SANDBOX_OBSERVABILITY_DIR:-/tmp/gym_nemo_lens_obs}"
SERVICE_NAME="${OTEL_SERVICE_NAME:-nemo-gym-sandbox-rollout}"
LIMIT="${EVAL_LIMIT:-1}"
OPENSANDBOX_API_KEY="${OPENSANDBOX_API_KEY:?Set OPENSANDBOX_API_KEY}"
OPENSANDBOX_DOMAIN="${OPENSANDBOX_DOMAIN:-localhost:8080}"

echo "=== NeMo Lens Observability Example ==="
echo "Artifacts will be written to: ${OBS_DIR}"
echo ""

rm -rf "${OBS_DIR}"
mkdir -p "${OBS_DIR}"

# ------------------------------------------------------------------ #
# 1. Start gym env with observability env vars                         #
#    (mini_swe_agent_2 inherits these and forwards to Ray workers)    #
# ------------------------------------------------------------------ #

echo "Starting gym env servers..."
OPENSANDBOX_API_KEY="${OPENSANDBOX_API_KEY}" \
OPENSANDBOX_DOMAIN="${OPENSANDBOX_DOMAIN}" \
LENS_SANDBOX_OBSERVABILITY_DIR="${OBS_DIR}" \
LENS_SANDBOX_OBSERVABILITY_ARTIFACTS=1 \
LENS_SANDBOX_OBSERVABILITY_EXPORT_OTLP_JSON=1 \
LENS_SANDBOX_OBSERVABILITY_RENDER_HTML=0 \
LENS_SANDBOX_OBSERVABILITY_RENDER_PNG=0 \
OTEL_SERVICE_NAME="${SERVICE_NAME}" \
  gym env start \
    --config responses_api_agents/mini_swe_agent_2/configs/mini_swe_agent_2.yaml \
    --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml \
    --model-type local_vllm_model \
    --model Qwen/Qwen3-1.7B \
    '++policy_model.responses_api_models.local_vllm_model.vllm_serve_env_vars.VLLM_RAY_DP_PACK_STRATEGY=strict' \
    '++policy_model.responses_api_models.local_vllm_model.vllm_serve_kwargs.enable_auto_tool_choice=true' \
    '++policy_model.responses_api_models.local_vllm_model.vllm_serve_kwargs.tool_call_parser=qwen3_coder' \
    > /tmp/gym_env_obs.log 2>&1 &
GYM_ENV_PID=$!
trap 'kill ${GYM_ENV_PID} 2>/dev/null || true' EXIT

# Wait for all servers to be ready
echo "Waiting for servers to be ready..."
until grep -q "All.*servers ready" /tmp/gym_env_obs.log 2>/dev/null; do
    if ! kill -0 "${GYM_ENV_PID}" 2>/dev/null; then
        echo "ERROR: gym env start exited unexpectedly. See /tmp/gym_env_obs.log"
        exit 1
    fi
    sleep 5
done
echo "Servers ready."
echo ""

# ------------------------------------------------------------------ #
# 2. Run eval against running servers (--no-serve reuses them)        #
# ------------------------------------------------------------------ #

echo "Running evaluation (limit=${LIMIT} task(s))..."
LENS_SANDBOX_OBSERVABILITY_DIR="${OBS_DIR}" \
LENS_SANDBOX_OBSERVABILITY_ARTIFACTS=1 \
LENS_SANDBOX_OBSERVABILITY_EXPORT_OTLP_JSON=1 \
LENS_SANDBOX_OBSERVABILITY_RENDER_HTML=0 \
LENS_SANDBOX_OBSERVABILITY_RENDER_PNG=0 \
OTEL_SERVICE_NAME="${SERVICE_NAME}" \
  gym eval run \
    --config responses_api_agents/mini_swe_agent_2/configs/mini_swe_agent_2.yaml \
    --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml \
    --model-type local_vllm_model \
    --model Qwen/Qwen3-1.7B \
    --no-serve \
    --input responses_api_agents/mini_swe_agent_2/data/example.jsonl \
    --output "${OBS_DIR}/rollouts.jsonl" \
    --limit "${LIMIT}" \
    '+agent_name=mini_swe_agent_2' \
    '++policy_model.responses_api_models.local_vllm_model.vllm_serve_env_vars.VLLM_RAY_DP_PACK_STRATEGY=strict' \
    '++policy_model.responses_api_models.local_vllm_model.vllm_serve_kwargs.enable_auto_tool_choice=true' \
    '++policy_model.responses_api_models.local_vllm_model.vllm_serve_kwargs.tool_call_parser=qwen3_coder' \
    || true   # don't exit on eval errors — still show artifact paths

# ------------------------------------------------------------------ #
# 3. Print artifact paths                                              #
# ------------------------------------------------------------------ #

echo ""
echo "=== Observability Artifacts ==="
echo "  Events JSONL  : ${OBS_DIR}/events.jsonl"
echo "  OTel JSON     : ${OBS_DIR}/traces/otel_traces.json"
echo "  Chrome trace  : ${OBS_DIR}/traces/chrome_trace.json"
echo "  Summary       : ${OBS_DIR}/summary.json"
echo ""
echo "Tip: Open ${OBS_DIR}/traces/chrome_trace.json in https://ui.perfetto.dev"
echo "     for an offline timeline of sandbox create / exec / destroy spans."

if [ -f "${OBS_DIR}/summary.json" ]; then
    echo ""
    echo "=== Quick summary ==="
    python3 -c "
import json, sys
s = json.load(open('${OBS_DIR}/summary.json'))
for op, stats in sorted(s.get('durations_by_name', {}).items()):
    print(f\"  {op:25s} count={stats['count']:3d}  p50={stats['p50']*1000:.0f}ms  p95={stats['p95']*1000:.0f}ms\")
" 2>/dev/null || true
fi
