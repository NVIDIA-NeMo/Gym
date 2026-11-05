# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Sandbox image for Agentic-MME's OpenCode track (benchmarks/agentic_mme/opencode.yaml): Python 3 with OpenCV, NumPy
# and Pillow for the harness's own image work. opencode_agent installs OpenCode when the session starts; its installer
# needs bash, curl, tar and gzip, and uses ripgrep when present.
#
#   docker build --platform linux/amd64 -t <registry>/agentic-mme-sandbox:<tag> resources_servers/agentic_mme/sandbox_image
FROM python:3.12-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends bash ca-certificates curl tar gzip ripgrep \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "opencv-python-headless==4.10.0.84" \
    "numpy==2.1.3" \
    "pillow==11.0.0"

WORKDIR /workspace
