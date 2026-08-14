#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

build_root="/app/apex-harness-build"
runtime_root="/app/apex-harness-runtime"
requirements="${build_root}/harness-requirements.txt"
source_archive="${build_root}/apex-harness-source.tar.gz"

test ! -e "${runtime_root}/bin/python"
uv venv --python /usr/bin/python3.13 "${runtime_root}"
uv pip install \
    --python "${runtime_root}/bin/python" \
    --requirements "${requirements}" \
    "${source_archive}"

"${runtime_root}/bin/python" -c \
    'from apex_harness.apex_agent.agent import ApexAgent; from apex_harness.environments.archipelago.environment import ArchipelagoMCPEnvironment; print("Apex harness OK")'
