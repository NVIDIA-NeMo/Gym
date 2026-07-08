#!/usr/bin/env bash
# Run the internal-osworld-adapter-nano-omni public-BF16 TP8 scaffold against external vLLM.
# Gym + clean OSWorld run on the Docker/KVM host; vLLM may run locally or on a
# reachable Prenyx B200 node.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GYM_ROOT="${GYM_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

export OMNI_MINI_VLLM_BASE_URL="${OMNI_MINI_VLLM_BASE_URL:-http://127.0.0.1:8000/v1}"
export OMNI_MINI_VLLM_API_KEY="${OMNI_MINI_VLLM_API_KEY:-local-vllm}"
export OMNI_MINI_VLLM_MODEL="${OMNI_MINI_VLLM_MODEL:-nvidia/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning}"
# Keep Gym response metadata and the model sent by the vLLM adapter on the
# same source of truth. This overrides stale policy_model_name values that may
# survive in a copied env.yaml from an earlier Claude/Pointer run.
export POLICY_MODEL_NAME="${POLICY_MODEL_NAME:-${OMNI_MINI_VLLM_MODEL}}"
export OSWORLD_POLICY_MODEL_NAME="${OSWORLD_POLICY_MODEL_NAME:-${POLICY_MODEL_NAME}}"

vllm_host="${OMNI_MINI_VLLM_BASE_URL#*://}"
vllm_host="${vllm_host%%[:/]*}"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${vllm_host}"
export no_proxy="${no_proxy:+${no_proxy},}${vllm_host}"

PREFLIGHT="${PREFLIGHT:-1}"
if [[ "${PREFLIGHT}" == "1" && "${DRY_RUN:-0}" != "1" ]]; then
  python3 "${SCRIPT_DIR}/probe_omni_mini_vllm.py" \
    --base-url "${OMNI_MINI_VLLM_BASE_URL}" \
    --api-key "${OMNI_MINI_VLLM_API_KEY}" \
    --model "${OMNI_MINI_VLLM_MODEL}"
fi

export GYM_ROOT
export RUNNER_NAME="${RUNNER_NAME:-omni_mini_agent}"
export INPUT_JSONL="${INPUT_JSONL:-responses_api_agents/osworld_agent/data/test_nogdrive.jsonl}"
export LIMIT="${LIMIT:-361}"
export NUM_ENVS="${NUM_ENVS:-4}"
export NUM_SAMPLES_IN_PARALLEL="${NUM_SAMPLES_IN_PARALLEL:-${NUM_ENVS}}"
export RESUME_FROM_CACHE="${RESUME_FROM_CACHE:-1}"
export MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-4096}"
export TEMPERATURE="${TEMPERATURE:-0.6}"
export RECORD_VIDEO="${RECORD_VIDEO:-sample}"
export CONFIG_PATHS="${CONFIG_PATHS:-responses_api_agents/osworld_agent/configs/osworld_agent.yaml,responses_api_agents/osworld_agent/configs/osworld_agent_omni_mini_local_reference.yaml,responses_api_models/vllm_model/configs/vllm_model_omni_mini.yaml}"

# Deliberately leave MAX_STEPS unset. The parity overlay is the single owner
# of max_steps=100, preventing a repeat of the earlier accidental 15-step run.
unset MAX_STEPS

exec bash "${SCRIPT_DIR}/run_multienv_osworld_agent.sh"
