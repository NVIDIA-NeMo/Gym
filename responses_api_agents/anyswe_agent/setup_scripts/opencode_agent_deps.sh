#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_portable_python.sh"

: "${DEPS_DIR:?DEPS_DIR must be set}"
: "${NEMO_GYM_ROOT:?NEMO_GYM_ROOT must be set}"

NODE_VERSION="${NODE_VERSION:-22.15.0}"
OPENCODE_VERSION="${OPENCODE_VERSION:-1.17.8}"

install_portable_python
install_nemo_gym_deps

if [ ! -x "$DEPS_DIR/bin/node" ]; then
    node_url="https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-x64.tar.xz"
    curl -fsSL "$node_url" | tar xJ -C "$DEPS_DIR" --strip-components=1
fi

export PATH="$DEPS_DIR/bin:$PATH"
npm install -g --prefix "$DEPS_DIR" "opencode-ai@${OPENCODE_VERSION}"
"$DEPS_DIR/bin/opencode" --version
