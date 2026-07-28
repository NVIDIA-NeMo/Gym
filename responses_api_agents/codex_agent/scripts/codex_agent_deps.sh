#!/bin/bash
# Install Codex agent dependencies into a portable prefix mounted in a task sandbox.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${PORTABLE_PYTHON_SH:-$SCRIPT_DIR/_portable_python.sh}"

: "${DEPS_DIR:?DEPS_DIR must be set}"
: "${NEMO_GYM_ROOT:?NEMO_GYM_ROOT must be set}"
NODE_VERSION="${NODE_VERSION:-22.15.0}"
: "${CODEX_SPEC:?CODEX_SPEC must select a pinned @openai/codex version}"
if [ -z "${NODE_ARCH:-}" ]; then
    case "$(uname -m)" in
        x86_64) NODE_ARCH="x64" ;;
        aarch64|arm64) NODE_ARCH="arm64" ;;
        *)
            echo "Unsupported Node architecture: $(uname -m)" >&2
            exit 1
            ;;
    esac
fi

install_portable_python
install_nemo_gym_deps

if [ ! -x "$DEPS_DIR/bin/node" ]; then
    node_url="https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-${NODE_ARCH}.tar.gz"
    curl -fsSL "$node_url" | tar xz -C "$DEPS_DIR" --strip-components=1
fi

export PATH="$DEPS_DIR/bin:$PATH"
npm install -g --prefix "$DEPS_DIR" "$CODEX_SPEC"
"$DEPS_DIR/bin/codex" --version
