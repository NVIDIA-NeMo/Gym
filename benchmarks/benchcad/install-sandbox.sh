#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Run only while building the disposable Python 3.12 container, as container root.
set -euo pipefail
source_root="${1:?Pass the mounted Gym checkout}"
apt-get update
apt-get install -y --no-install-recommends \
    ca-certificates curl git xz-utils ripgrep libosmesa6 libgl1 libglx-mesa0 libxrender1 libxext6 libsm6
python -m pip install --no-cache-dir uv==0.10.12
git clone https://github.com/BenchCAD/BenchCAD-main.git /opt/benchcad
git -C /opt/benchcad checkout 77fd16a80a4e5fc39964ee23b6cc3225beefac8f
uv venv --python /usr/local/bin/python /opt/benchcad/.venv
uv pip install --python /opt/benchcad/.venv/bin/python \
    -r "$source_root/responses_api_agents/benchcad_agent/cad-requirements.txt" \
    --override "$source_root/responses_api_agents/benchcad_agent/cad-constraints.txt"
cp /opt/benchcad/docker/sitecustomize.py /opt/benchcad/.venv/lib/python3.12/site-packages/
find /opt/benchcad -mindepth 1 -maxdepth 1 ! -name .venv ! -name LICENSE -exec rm -rf {} +
curl -fsSL https://opencode.ai/install -o /tmp/install-opencode
VERSION=1.17.11 bash /tmp/install-opencode
mv /root/.opencode/bin/opencode /usr/local/bin/opencode
opencode --version
mkdir -p /workspace
