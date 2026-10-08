#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PORTABLE_PYTHON_SH="$SCRIPT_DIR/_portable_python.sh"
exec bash "$NEMO_GYM_ROOT/responses_api_agents/nemo_fabric_agent/scripts/nemo_fabric_agent_deps.sh"
