#!/bin/bash
# Install claude_code_agent deps into $DEPS_DIR (mounted read-only at /agent_deps_mount):
# portable CPython + NeMo-Gym runtime deps + portable Node + the claude CLI on PATH.
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${PORTABLE_PYTHON_SH:-$SCRIPT_DIR/_portable_python.sh}"

: "${DEPS_DIR:?DEPS_DIR must be set}"
: "${NEMO_GYM_ROOT:?NEMO_GYM_ROOT must be set}"
NODE_VERSION="${NODE_VERSION:-20.18.1}"
CLAUDE_SPEC="${CLAUDE_SPEC:-@anthropic-ai/claude-code}"

install_portable_python
install_nemo_gym_deps

case "$(uname -m)" in
    x86_64|amd64) node_arch="x64" ;;
    arm64|aarch64) node_arch="arm64" ;;
    *)
        echo "Unsupported architecture for portable Node: $(uname -m)" >&2
        exit 1
        ;;
esac

if ! [ -x "$DEPS_DIR/bin/node" ] || ! "$DEPS_DIR/bin/node" --version >/dev/null 2>&1; then
    node_url="https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-${node_arch}.tar.xz"
    echo "Downloading portable node: $node_url"
    curl -fsSL "$node_url" | tar xJ -C "$DEPS_DIR" --strip-components=1
fi

export PATH="$DEPS_DIR/bin:$PATH"
echo "Installing claude-code ($CLAUDE_SPEC)"
npm install -g --prefix "$DEPS_DIR" "$CLAUDE_SPEC"

"$DEPS_DIR/bin/claude" --version || echo "warning: claude --version failed (may need runtime env)"

echo "claude_code_agent deps ready at $DEPS_DIR"
