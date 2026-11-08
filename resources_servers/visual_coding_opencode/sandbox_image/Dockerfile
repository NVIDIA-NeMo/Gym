# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Sandbox image for visual coding in OpenCode sandbox sessions (configs/sandbox_sessions.yaml). The model's shell has the
# same Python image toolkit as opencode_vision_agent's venv: Pillow, NumPy, SciPy and OpenCV, at the versions that
# component's uv.lock pins. OpenCode is preinstalled under /opt/opencode/bin, so a sandbox that reaches only the model
# server can run it. sandbox_sessions.yaml pins that binary's SHA-256, so a new OpenCode version here needs a new pin
# there.
#
#   docker build --platform linux/amd64 -t <registry>/visual-coding-opencode-sandbox:<tag> \
#       resources_servers/visual_coding_opencode/sandbox_image
FROM python:3.13-slim-bookworm

# bash for OpenCode's shell tool, ripgrep for its grep and glob tools, curl and tar for the OpenCode download below.
RUN apt-get update \
    && apt-get install -y --no-install-recommends bash ca-certificates curl tar gzip ripgrep \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "pillow==13.0.0" \
    "numpy==2.5.4" \
    "scipy==1.18.1" \
    "opencv-python-headless==5.0.0.93"

# One thread per numeric library, as opencode_vision_agent sets for the model's shell.
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    OPENCV_FOR_THREADS_NUM=1

# The glibc x86_64 build that opencode_agent's installer picks for this image.
ARG OPENCODE_VERSION=1.17.11
RUN mkdir -p /opt/opencode/bin \
    && curl -fsSL "https://github.com/anomalyco/opencode/releases/download/v${OPENCODE_VERSION}/opencode-linux-x64-baseline.tar.gz" \
        | tar -xz -C /opt/opencode/bin opencode \
    && test "$(/opt/opencode/bin/opencode --version)" = "${OPENCODE_VERSION}"

WORKDIR /workspace
