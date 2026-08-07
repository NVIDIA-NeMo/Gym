#!/bin/bash
# Install claude_code_agent deps into $DEPS_DIR: portable Node + the claude CLI on PATH.
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_portable_python.sh"

: "${DEPS_DIR:?DEPS_DIR must be set}"
: "${NEMO_GYM_ROOT:?NEMO_GYM_ROOT must be set}"
NODE_VERSION="${NODE_VERSION:-20.18.1}"
CLAUDE_SPEC="${CLAUDE_SPEC:-@anthropic-ai/claude-code}"
if [ -z "${NODE_ARCH:-}" ]; then
    case "$(uname -m)" in
        x86_64) NODE_ARCH="linux-x64" ;;
        aarch64 | arm64) NODE_ARCH="linux-arm64" ;;
        *) echo "unsupported node architecture: $(uname -m)" >&2; exit 1 ;;
    esac
fi

install_portable_python
install_nemo_gym_deps

if [ ! -x "$DEPS_DIR/bin/node" ]; then
    node_url="https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-${NODE_ARCH}.tar.xz"
    echo "Downloading portable node: $node_url"
    curl -fsSL "$node_url" | tar xJ -C "$DEPS_DIR" --strip-components=1
fi

export PATH="$DEPS_DIR/bin:$PATH"
echo "Installing claude-code ($CLAUDE_SPEC)"
npm install -g --prefix "$DEPS_DIR" "$CLAUDE_SPEC"

"$DEPS_DIR/bin/claude" --version || echo "warning: claude --version failed (may need runtime env)"

echo "claude_code_agent deps ready at $DEPS_DIR"
