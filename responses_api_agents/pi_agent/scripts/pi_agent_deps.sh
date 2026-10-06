#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Install Pi plus a portable Python/Gym runtime for containerized harness adapters.
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${PORTABLE_PYTHON_SH:-$SCRIPT_DIR/_portable_python.sh}"

: "${DEPS_DIR:?DEPS_DIR must be set}"
: "${NEMO_GYM_ROOT:?NEMO_GYM_ROOT must be set}"
# Pi 0.80.2 declares Node >=22.19.0.
NODE_VERSION="${NODE_VERSION:-22.19.0}"
PI_VERSION="${PI_VERSION:-0.80.2}"

install_portable_python
install_nemo_gym_deps

if [ "$("$DEPS_DIR/bin/node" --version 2>/dev/null || true)" != "v${NODE_VERSION}" ]; then
    node_url="https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-x64.tar.xz"
    echo "Downloading portable node: $node_url"
    curl -fsSL "$node_url" | tar xJ -C "$DEPS_DIR" --strip-components=1
fi

export PATH="$DEPS_DIR/bin:$PATH"
export PYTHONPATH="$NEMO_GYM_ROOT${PYTHONPATH:+:$PYTHONPATH}"
echo "Installing Pi ${PI_VERSION}"
npm install -g --prefix "$DEPS_DIR" "@earendil-works/pi-coding-agent@${PI_VERSION}"

"$DEPS_DIR/bin/pi" --version
"$DEPS_DIR/bin/python3" -c "from responses_api_agents.pi_agent.app import PiAgent; print('pi_agent OK')"

echo "pi_agent deps ready at $DEPS_DIR"
