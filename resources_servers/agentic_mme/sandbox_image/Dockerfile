# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Sandbox image for Agentic-MME's OpenCode track (benchmarks/agentic_mme/opencode.yaml): Python 3 with OpenCV, NumPy
# and Pillow for the harness's own image work. OpenCode is preinstalled under /opt/opencode/bin, so a sandbox with no
# internet access can run it (benchmarks/agentic_mme/opencode.yaml sets sandbox_executable_path and its
# SHA-256). Without that setting, opencode_agent downloads OpenCode when the session starts; its installer needs bash,
# curl, tar and gzip, and uses ripgrep when present.
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

# The glibc x86_64 build that opencode_agent's installer picks for this image. Its SHA-256 is pinned in
# benchmarks/agentic_mme/opencode.yaml, so a new version here needs a new pin there.
ARG OPENCODE_VERSION=1.17.11
RUN mkdir -p /opt/opencode/bin \
    && curl -fsSL "https://github.com/anomalyco/opencode/releases/download/v${OPENCODE_VERSION}/opencode-linux-x64-baseline.tar.gz" \
        | tar -xz -C /opt/opencode/bin opencode \
    && test "$(/opt/opencode/bin/opencode --version)" = "${OPENCODE_VERSION}"

WORKDIR /workspace
