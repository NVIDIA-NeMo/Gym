#!/bin/bash
# Install Terminus-2 deps into $DEPS_DIR for anyterminal_agent.
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${PORTABLE_PYTHON_SH:-$SCRIPT_DIR/_portable_python.sh}"

: "${DEPS_DIR:?DEPS_DIR must be set}"
: "${NEMO_GYM_ROOT:?NEMO_GYM_ROOT must be set}"

TERMINUS_PROJECT="$NEMO_GYM_ROOT/responses_api_agents/terminus_2_agent/pyproject.toml"
if [ -z "${HARBOR_SPEC:-}" ]; then
    # Read this canonical single-line TOML dependency before Python is provisioned.
    HARBOR_SPEC="$(sed -n '/^[[]project[]]$/,/^[[]/ {
        s/^[[:space:]]*"harbor @ \([^"]*\)"[,[:space:]]*$/\1/p
    }' "$TERMINUS_PROJECT")"
fi
: "${HARBOR_SPEC:?could not read the Harbor pin from $TERMINUS_PROJECT}"

install_portable_python
install_nemo_gym_deps

echo "Installing Harbor Terminus-2 ($HARBOR_SPEC)"
"$DEPS_DIR/bin/python3" -m pip install "$HARBOR_SPEC"
"$DEPS_DIR/bin/python3" -c \
    "from harbor.agents.terminus_2.terminus_2 import Terminus2; print(Terminus2.name())"

echo "terminus_2_agent deps ready at $DEPS_DIR"
