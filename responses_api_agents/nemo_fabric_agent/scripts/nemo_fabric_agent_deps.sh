#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHON_VERSION="${PYTHON_VERSION:-3.13.14}"
export PBS_RELEASE="${PBS_RELEASE:-20260610}"
source "${PORTABLE_PYTHON_SH:-$SCRIPT_DIR/_portable_python.sh}"

: "${DEPS_DIR:?DEPS_DIR must be set}"
: "${NEMO_GYM_ROOT:?NEMO_GYM_ROOT must be set}"

install_portable_python
install_nemo_gym_deps

# Keep Gym in the same resolution so Fabric cannot upgrade beyond its SDK pins.
if [[ -n "${NEMO_FABRIC_SPEC:-}" ]]; then
    install_python_packages "$NEMO_GYM_ROOT" "$NEMO_FABRIC_SPEC"
else
    install_python_packages "$NEMO_GYM_ROOT" -r "$SCRIPT_DIR/../runtime-requirements.txt"
fi
# Mini's LiteLLM tool path requires MCP 2, isolated from Gym's MCP 1 controller.
if portable_python_can_run; then
    "$DEPS_DIR/bin/python3" -m pip install --no-deps --upgrade --target "$DEPS_DIR/mini-mcp-site" \
        "mcp==2.3.0" "mcp-types==2.3.0"
else
    uv pip install --no-deps --target "$DEPS_DIR/mini-mcp-site" --python-version "$PYTHON_VERSION" \
        --python-platform "$ARCH" --reinstall "mcp==2.3.0" "mcp-types==2.3.0"
fi
if portable_python_can_run; then
    "$DEPS_DIR/bin/python3" -c "from nemo_fabric import Fabric, FabricConfig; print('nemo-fabric OK')"
fi
