#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_portable_python.sh"

: "${DEPS_DIR:?DEPS_DIR must be set}"
: "${NEMO_GYM_ROOT:?NEMO_GYM_ROOT must be set}"

AGENT_DIR="$NEMO_GYM_ROOT/responses_api_agents/simple_strands_agent"
SSA_SPEC="${SSA_SPEC:-$(grep '^simple-strands-agent @ ' "$AGENT_DIR/requirements.txt")}"

install_portable_python
install_nemo_gym_deps

# The agent's overrides.txt keeps Gym's openai/numpy/pyarrow pins over SSA's.
"$DEPS_DIR/bin/python3" -m pip install uv
"$DEPS_DIR/bin/python3" -m uv pip install --python "$DEPS_DIR/bin/python3" --override "$AGENT_DIR/overrides.txt" "$SSA_SPEC"
"$DEPS_DIR/bin/python3" -c "import ssa, strands; print('ssa ok')"
