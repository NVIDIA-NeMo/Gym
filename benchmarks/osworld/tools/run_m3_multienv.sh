#!/usr/bin/env bash
# Run MiniMax M3 with OSWorld's official M3Agent scaffold through NeMo Gym.
#
# Example smoke run:
#   LIMIT=4 NUM_ENVS=1 \
#     bash benchmarks/osworld/tools/run_m3_multienv.sh
#
# For a full no-GDrive run, set INPUT_JSONL to the materialized 361-task JSONL,
# LIMIT=null, and increase NUM_ENVS only after the endpoint is stable.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUNNER_NAME="${RUNNER_NAME:-m3_agent}"
export POLICY_MODEL_NAME="${POLICY_MODEL_NAME:-nvidia/minimaxai/minimax-m3}"
export MAX_STEPS="${MAX_STEPS:-100}"
export MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-8192}"
export TEMPERATURE="${TEMPERATURE:-0.6}"
export NUM_ENVS="${NUM_ENVS:-1}"
export CONFIG_PATHS="${CONFIG_PATHS:-responses_api_agents/osworld_agent/configs/osworld_agent.yaml,benchmarks/osworld/configs/osworld_agent_m3.yaml,responses_api_models/openai_model/configs/openai_model.yaml}"

exec bash "${SCRIPT_DIR}/run_multienv_osworld_agent.sh"
