#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Install the existing Hermes adapter's dependencies in the task container.
set -euo pipefail

if [[ ! -f /work/gym_mount/pyproject.toml ]]; then
    echo "Hermes setup requires a Gym checkout or a gym_source archive with build metadata." >&2
    exit 1
fi

if ! command -v curl >/dev/null || ! command -v git >/dev/null; then
    apt-get update
    apt-get install -y --no-install-recommends ca-certificates curl git
fi
curl --fail --location --silent --show-error https://astral.sh/uv/0.12.9/install.sh \
    | env UV_INSTALL_DIR=/work/bin UV_NO_MODIFY_PATH=1 sh
/work/bin/uv venv --python 3.13.14 --allow-existing /work/hermes-venv
mkdir -p /work/gym_mount/cache
cd /work/gym_mount/responses_api_agents/hermes_agent
/work/bin/uv pip install --python /work/hermes-venv/bin/python -r requirements.txt
/work/hermes-venv/bin/python -c 'import model_tools; from run_agent import AIAgent'
