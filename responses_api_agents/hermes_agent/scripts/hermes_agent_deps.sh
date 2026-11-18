#!/bin/bash
# Install hermes_agent deps into $DEPS_DIR (mounted read-only at /agent_deps_mount).
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${PORTABLE_PYTHON_SH:-$SCRIPT_DIR/_portable_python.sh}"

: "${DEPS_DIR:?DEPS_DIR must be set}"
: "${NEMO_GYM_ROOT:?NEMO_GYM_ROOT must be set}"

install_portable_python
install_nemo_gym_deps

# Build the separate Hermes interpreter on the host, including for cross-platform task images.
args=(--prefix "$DEPS_DIR" --target "$ARCH")
if [[ -n "${HERMES_SPEC:-}" ]]; then
    args+=(--source "$HERMES_SPEC")
fi
uv run --no-dev --locked --directory "$NEMO_GYM_ROOT" python -m responses_api_agents.hermes_agent.scripts.prepare_portable "${args[@]}"
