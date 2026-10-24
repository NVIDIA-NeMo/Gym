#!/bin/bash
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_portable_python.sh"

: "${DEPS_DIR:?DEPS_DIR must be set}"
: "${NEMO_GYM_ROOT:?NEMO_GYM_ROOT must be set}"

# Pin must match hermes_agent/app.py's AIAgent API; override only for experiments.
HERMES_PROJECT="$NEMO_GYM_ROOT/responses_api_agents/hermes_agent/pyproject.toml"
if [ -z "${HERMES_SPEC:-}" ]; then
    # Read this canonical single-line TOML dependency before Python is provisioned.
    HERMES_SPEC="$(sed -n '/^[[]project[]]$/,/^[[]/ {
        s/^[[:space:]]*"hermes-agent @ \([^"]*\)"[,[:space:]]*$/\1/p
    }' "$HERMES_PROJECT")"
fi
: "${HERMES_SPEC:?could not read hermes-agent pin from $HERMES_PROJECT}"

install_portable_python
install_nemo_gym_deps

echo "Installing hermes-agent ($HERMES_SPEC)"
"$DEPS_DIR/bin/python3" -m pip install --force-reinstall --no-deps "$HERMES_SPEC"
"$DEPS_DIR/bin/python3" -m pip install "$HERMES_SPEC"

"$DEPS_DIR/bin/python3" -c "import model_tools; from run_agent import AIAgent; print('hermes-agent OK')"

echo "hermes_agent deps ready at $DEPS_DIR"
