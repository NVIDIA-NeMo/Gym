#!/bin/bash
# Install cline_agent deps into $DEPS_DIR: portable Node + the cline CLI on PATH.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_portable_python.sh"

: "${DEPS_DIR:?DEPS_DIR must be set}"
: "${NEMO_GYM_ROOT:?NEMO_GYM_ROOT must be set}"

NODE_VERSION="${NODE_VERSION:-22.15.0}"
# Keep in sync with cline_version in responses_api_agents/cline_agent/configs/cline_agent.yaml.
CLINE_VERSION="${CLINE_VERSION:-3.0.55}"

install_portable_python
install_nemo_gym_deps

if [ ! -x "$DEPS_DIR/bin/node" ]; then
    node_url="https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-x64.tar.xz"
    curl -fsSL "$node_url" | tar xJ -C "$DEPS_DIR" --strip-components=1
fi

export PATH="$DEPS_DIR/bin:$PATH"
npm install -g --prefix "$DEPS_DIR" "cline@${CLINE_VERSION}"
"$DEPS_DIR/bin/cline" --version
