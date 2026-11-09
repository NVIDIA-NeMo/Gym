# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Scientific package selection follows VIALS's MIT-licensed Harbor adapter.
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash curl git python3 python3-venv tesseract-ocr ca-certificates && \
    rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/agent-python && \
    /opt/agent-python/bin/pip install --no-cache-dir \
    "numpy>=1.26" "scipy>=1.11" "sympy>=1.12" "pillow>=10.0" \
    "imageio>=2.31" "opencv-python-headless>=4.9" "scikit-image>=0.22" \
    "matplotlib>=3.8" "pandas>=2.1" "networkx>=3.2" "pytesseract>=0.3.10"
RUN useradd -m -u 10001 -s /bin/bash agent && \
    mkdir -p /task && chown agent:agent /task
ENV PATH=/opt/agent-python/bin:$PATH
WORKDIR /task
USER agent
