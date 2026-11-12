#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_portable_python.sh"

: "${DEPS_DIR:?DEPS_DIR must be set}"
: "${NEMO_GYM_ROOT:?NEMO_GYM_ROOT must be set}"

NODE_VERSION="${NODE_VERSION:-22.19.0}"
# Node.js build; defaults to the host architecture. Override NODE_ARCH (e.g. linux-arm64) to cross-prepare.
if [ -z "${NODE_ARCH:-}" ]; then
    case "$(uname -m)" in
        x86_64) NODE_ARCH="linux-x64" ;;
        aarch64 | arm64) NODE_ARCH="linux-arm64" ;;
        *) echo "unsupported node architecture: $(uname -m)" >&2; exit 1 ;;
    esac
fi
PI_VERSION="${PI_VERSION:-0.80.2}"

install_portable_python
install_nemo_gym_deps

if [ "$("$DEPS_DIR/bin/node" --version 2>/dev/null || true)" != "v${NODE_VERSION}" ]; then
    node_url="https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-${NODE_ARCH}.tar.xz"
    curl -fsSL "$node_url" | tar xJ -C "$DEPS_DIR" --strip-components=1
fi

export PATH="$DEPS_DIR/bin:$PATH"
npm install -g --prefix "$DEPS_DIR" "@earendil-works/pi-coding-agent@${PI_VERSION}"
"$DEPS_DIR/bin/pi" --version
