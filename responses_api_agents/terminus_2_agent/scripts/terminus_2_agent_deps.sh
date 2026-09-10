#!/bin/bash
# Install Terminus-2 deps into $DEPS_DIR for anyterminal_agent.
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${PORTABLE_PYTHON_SH:-$SCRIPT_DIR/_portable_python.sh}"

: "${DEPS_DIR:?DEPS_DIR must be set}"
: "${NEMO_GYM_ROOT:?NEMO_GYM_ROOT must be set}"

install_portable_python
install_python_packages "$NEMO_GYM_ROOT[terminus-2]"
"$DEPS_DIR/bin/python3" -c \
    "from harbor.agents.terminus_2.terminus_2 import Terminus2; print(Terminus2.name())"

echo "terminus_2_agent deps ready at $DEPS_DIR"
