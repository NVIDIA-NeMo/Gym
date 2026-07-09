#!/bin/bash
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${PORTABLE_PYTHON_SH:-$SCRIPT_DIR/_portable_python.sh}"

: "${DEPS_DIR:?DEPS_DIR must be set}"
: "${NEMO_GYM_ROOT:?NEMO_GYM_ROOT must be set}"

NODE_VERSION="${NODE_VERSION:-22.19.0}"
OPENCLAW_SPEC="${OPENCLAW_SPEC:-openclaw}"

install_portable_python
install_nemo_gym_deps

if [ ! -x "$DEPS_DIR/bin/node" ]; then
    node_url="https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-x64.tar.xz"
    echo "Downloading portable node: $node_url"
    curl -fsSL "$node_url" | tar xJ -C "$DEPS_DIR" --strip-components=1
fi

export PATH="$DEPS_DIR/bin:$PATH"
export PYTHONPATH="$NEMO_GYM_ROOT${PYTHONPATH:+:$PYTHONPATH}"
echo "Installing openclaw ($OPENCLAW_SPEC)"
npm install -g --prefix "$DEPS_DIR" "$OPENCLAW_SPEC"

"$DEPS_DIR/bin/openclaw" --version
"$DEPS_DIR/bin/python3" -c "from responses_api_agents.openclaw_agent.app import OpenClawAgent; print('openclaw_agent OK')"

echo "openclaw_agent deps ready at $DEPS_DIR"
