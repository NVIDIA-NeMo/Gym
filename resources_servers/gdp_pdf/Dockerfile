# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
FROM python:3.13.14-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash curl git ripgrep poppler-utils tesseract-ocr libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir pymupdf==1.26.7 pillow==12.0.0 pandas==2.3.3
WORKDIR /workspace
# Reuse Gym's installer at build time: task sandboxes have no download egress.
COPY responses_api_agents/opencode_agent/install_opencode_runtime.sh /tmp/install-opencode.sh
RUN bash /tmp/install-opencode.sh /tmp/nemo-gym-opencode-runtime-1.17.11 1.17.11 \
    && rm /tmp/install-opencode.sh
