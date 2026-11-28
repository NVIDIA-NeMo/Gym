#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

build_root="/app/stirrup-build"
runtime_root="/app/stirrup-runtime"
requirements="${build_root}/stirrup-requirements.txt"

test ! -e "${runtime_root}/bin/python"
uv venv --python /usr/bin/python3.13 "${runtime_root}"
uv pip install \
    --python "${runtime_root}/bin/python" \
    --requirements "${requirements}"

"${runtime_root}/bin/python" -c \
    'from stirrup import Agent; from stirrup.tools.mcp import MCPToolProvider; print("Stirrup runtime OK")'

