#!/bin/bash
# Install agent deps for claude_code_agent into $DEPS_DIR. Claude Code is an npm
# CLI, so we bring a portable Node alongside the portable Python and install the
# claude binary into $DEPS_DIR/bin (which the runner puts on PATH inside the
# container, so claude_code_agent.setup_claude_code.ensure_claude_code finds it).
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_portable_python.sh"

: "${DEPS_DIR:?DEPS_DIR must be set}"
: "${NEMO_GYM_ROOT:?NEMO_GYM_ROOT must be set}"
NODE_VERSION="${NODE_VERSION:-20.18.1}"
CLAUDE_SPEC="${CLAUDE_SPEC:-@anthropic-ai/claude-code}"

install_portable_python
install_nemo_gym_deps

if [ ! -x "$DEPS_DIR/bin/node" ]; then
    node_url="https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-x64.tar.xz"
    echo "Downloading portable node: $node_url"
    curl -fsSL "$node_url" | tar xJ -C "$DEPS_DIR" --strip-components=1
fi

export PATH="$DEPS_DIR/bin:$PATH"
echo "Installing claude-code ($CLAUDE_SPEC)"
npm install -g --prefix "$DEPS_DIR" "$CLAUDE_SPEC"

"$DEPS_DIR/bin/claude" --version || echo "warning: claude --version failed (may need runtime env)"

echo "claude_code_agent deps ready at $DEPS_DIR"
