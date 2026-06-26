#!/bin/bash
# install opencode_agent deps into $DEPS_DIR: portable python, nemo_gym, node, and the opencode cli
set -euo pipefail
set -x
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_portable_python.sh"
: "${DEPS_DIR:?}"
: "${NEMO_GYM_ROOT:?}"
NODE_VERSION="${NODE_VERSION:-22.9.0}"
install_portable_python
install_nemo_gym_deps
if [ ! -x "$DEPS_DIR/bin/node" ]; then
  curl -fsSL "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-x64.tar.xz" | tar xJ -C "$DEPS_DIR" --strip-components=1
fi
export PATH="$DEPS_DIR/bin:$PATH"
npm install -g --prefix "$DEPS_DIR" "opencode-ai"
"$DEPS_DIR/bin/opencode" --version 2>/dev/null || echo "warn: opencode --version failed"
echo "opencode_agent deps ready at $DEPS_DIR"
