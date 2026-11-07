# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
FROM python:3.13.14-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash curl git ripgrep poppler-utils tesseract-ocr libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir pymupdf==1.26.7 pillow==12.0.0 pandas==2.3.3
WORKDIR /workspace
# The shared OpenCode adapter installs its configured, pinned runtime during session setup.
