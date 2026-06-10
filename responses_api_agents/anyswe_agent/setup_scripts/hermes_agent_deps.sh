#!/bin/bash
# Install agent deps for hermes_agent into $DEPS_DIR (a self-contained tree
# mounted read-only into the SWEBench container at /agent_deps_mount).
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_portable_python.sh"

: "${DEPS_DIR:?DEPS_DIR must be set}"
: "${NEMO_GYM_ROOT:?NEMO_GYM_ROOT must be set}"
HERMES_SPEC="${HERMES_SPEC:-hermes-agent}"

install_portable_python
install_nemo_gym_deps

echo "Installing hermes-agent ($HERMES_SPEC)"
"$DEPS_DIR/bin/python3" -m pip install "$HERMES_SPEC"

# Sanity check the imports hermes_agent/app.py performs at module load.
"$DEPS_DIR/bin/python3" -c "import model_tools; from run_agent import AIAgent; print('hermes-agent OK')"

echo "hermes_agent deps ready at $DEPS_DIR"
