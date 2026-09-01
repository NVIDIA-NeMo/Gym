#!/bin/bash
# Install hermes_agent deps into $DEPS_DIR (mounted read-only at /agent_deps_mount).
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${PORTABLE_PYTHON_SH:-$SCRIPT_DIR/_portable_python.sh}"

: "${DEPS_DIR:?DEPS_DIR must be set}"
: "${NEMO_GYM_ROOT:?NEMO_GYM_ROOT must be set}"

# Pin must match hermes_agent/app.py's AIAgent API; override only for experiments.
HERMES_REQ="$NEMO_GYM_ROOT/responses_api_agents/hermes_agent/requirements.txt"
HERMES_SPEC="${HERMES_SPEC:-$(sed -n 's/^hermes-agent @ //p' "$HERMES_REQ")}"
: "${HERMES_SPEC:?could not read hermes-agent pin from $HERMES_REQ}"

install_portable_python
install_nemo_gym_deps

echo "Installing hermes-agent ($HERMES_SPEC)"
# HERMES_NIX_BUILD=1 bypasses the wheel-build restriction added in some branches.
HERMES_NIX_BUILD=1 "$DEPS_DIR/bin/python3" -m pip install --force-reinstall --no-deps "$HERMES_SPEC"
HERMES_NIX_BUILD=1 "$DEPS_DIR/bin/python3" -m pip install "$HERMES_SPEC"
# hermes-agent pins openai==2.24.0, which conflicts with nemo-gym's own openai==2.44.0 pin
# (see pyproject.toml); the install above downgrades it. Reinstall nemo-gym's pinned openai
# last, with --no-deps, so it wins without re-triggering the same resolver conflict.
OPENAI_PIN="$(grep -oE '"openai==[0-9.]+"' "$NEMO_GYM_ROOT/pyproject.toml" | tr -d '"')"
: "${OPENAI_PIN:?could not read openai pin from $NEMO_GYM_ROOT/pyproject.toml}"
"$DEPS_DIR/bin/python3" -m pip install --force-reinstall --no-deps "$OPENAI_PIN"

"$DEPS_DIR/bin/python3" -c "import model_tools; from run_agent import AIAgent; print('hermes-agent OK')"

echo "$HERMES_SPEC" > "$DEPS_DIR/.hermes_spec"
echo "hermes_agent deps ready at $DEPS_DIR"
