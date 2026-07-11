#!/usr/bin/env bash
# Run Nemotron 3 Nano Omni against an OpenAI-compatible vLLM endpoint.
# Gym and OSWorld may share the model host or connect to a remote endpoint.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GYM_ROOT="${GYM_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

export OMNI_MINI_VLLM_BASE_URL="${OMNI_MINI_VLLM_BASE_URL:-http://127.0.0.1:8000/v1}"
export OMNI_MINI_VLLM_API_KEY="${OMNI_MINI_VLLM_API_KEY:-local-vllm}"
export OMNI_MINI_VLLM_MODEL="${OMNI_MINI_VLLM_MODEL:-nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16}"
# Use one model name for the vLLM request and rollout metadata.
export POLICY_MODEL_NAME="${POLICY_MODEL_NAME:-${OMNI_MINI_VLLM_MODEL}}"
export OSWORLD_POLICY_MODEL_NAME="${OSWORLD_POLICY_MODEL_NAME:-${POLICY_MODEL_NAME}}"

vllm_host="${OMNI_MINI_VLLM_BASE_URL#*://}"
vllm_host="${vllm_host%%[:/]*}"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${vllm_host}"
export no_proxy="${no_proxy:+${no_proxy},}${vllm_host}"

PREFLIGHT="${PREFLIGHT:-1}"
OMNI_MINI_PREFLIGHT_IMAGE_COUNT="${OMNI_MINI_PREFLIGHT_IMAGE_COUNT:-3}"
if [[ "${PREFLIGHT}" == "1" && "${DRY_RUN:-0}" != "1" ]]; then
  python3 "${SCRIPT_DIR}/probe_omni_mini_vllm.py" \
    --base-url "${OMNI_MINI_VLLM_BASE_URL}" \
    --api-key "${OMNI_MINI_VLLM_API_KEY}" \
    --model "${OMNI_MINI_VLLM_MODEL}" \
    --image-count "${OMNI_MINI_PREFLIGHT_IMAGE_COUNT}"
fi

export GYM_ROOT
export RUNNER_NAME="${RUNNER_NAME:-nemotron_v3_nano_omni_agent}"
export INPUT_JSONL="${INPUT_JSONL:-responses_api_agents/osworld_agent/data/example.jsonl}"
export LIMIT="${LIMIT:-5}"
export NUM_ENVS="${NUM_ENVS:-1}"
export NUM_SAMPLES_IN_PARALLEL="${NUM_SAMPLES_IN_PARALLEL:-${NUM_ENVS}}"
export RESUME_FROM_CACHE="${RESUME_FROM_CACHE:-0}"
export MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-4096}"
export TEMPERATURE="${TEMPERATURE:-0.6}"
export RECORD_VIDEO="${RECORD_VIDEO:-0}"
export CONFIG_PATHS="${CONFIG_PATHS:-responses_api_agents/osworld_agent/configs/osworld_agent.yaml,responses_api_agents/osworld_agent/configs/osworld_agent_omni_mini.yaml,responses_api_models/vllm_model/configs/vllm_model_omni_mini.yaml}"

exec bash "${SCRIPT_DIR}/run_multienv_osworld_agent.sh"
