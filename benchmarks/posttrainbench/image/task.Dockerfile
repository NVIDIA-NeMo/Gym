# Copyright (c) 2025 Ben Rank, Hardik Bhatnagar, Matthias Bethge, Maksym Andriushchenko
# Originally MIT License (https://github.com/aisa-group/PostTrainBench/blob/ea8d65bc375e61c8ac77a7aea86effcc995378fb/LICENSE), adapted from https://github.com/aisa-group/PostTrainBench/blob/ea8d65bc375e61c8ac77a7aea86effcc995378fb/src/harbor_adapter/template/environment/Dockerfile
#
# SPDX-FileCopyrightText: Modifications Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# PostTrainBench task image: one image for all 24 tasks. It is upstream's Harbor task image
# (aisa-group/PostTrainBench@ea8d65bc375e61c8ac77a7aea86effcc995378fb,
# src/harbor_adapter/template/environment/Dockerfile) without the per-task files: same base, packages,
# pins and shared files, with the RUN steps grouped so the builder takes fewer filesystem snapshots.
# Every benchmark's task files sit under /opt/ptb/benchmarks/<benchmark>/; each task's overlay Dockerfile
# (Gym benchmarks/posttrainbench/prepare.py) copies its own into place, writes its metadata.json and
# timer.sh, and deletes /opt/ptb before the agent starts (NVBug 6978505).
FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/.local/bin:$PATH"
ENV NO_PROXY="localhost,127.0.0.1"
ENV no_proxy="localhost,127.0.0.1"
ENV VLLM_API_KEY="inspectai"
ENV PYTHONNOUSERSITE="1"
ENV NUM_GPUS="1"

# System packages, python symlinks, Node.js 22.x and uv.
RUN apt-get update && apt-get install -y \
        python3.10 \
        python3-dev \
        python3-pip \
        git \
        wget \
        curl \
        build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh

# vllm (torch backend pinned to CUDA 12.9), the pinned ML deps and flash-attn.
COPY requirements-direct.txt /opt/requirements-direct.txt
RUN uv pip install --system --no-cache vllm==0.11.0 --torch-backend=cu129 \
    && uv pip install --system --no-cache -r /opt/requirements-direct.txt \
    && uv pip install --system --no-cache flash-attn==2.8.3 --no-build-isolation

# Agent CLIs and the inspect stacks, pinned as upstream pins them.
RUN npm install -g \
        @anthropic-ai/claude-code@2.1.219 \
        @openai/codex@0.144.0 \
        @google/gemini-cli@0.18.4 \
        opencode-ai@1.17.18 \
    && mkdir -p /opt \
    && cd /opt \
    && git clone https://github.com/UKGovernmentBEIS/inspect_evals.git \
    && cd /opt/inspect_evals \
    && git checkout 06001a83e6d7c709c2ede0570dce7f1031a0bad8 \
    && uv pip install --system --no-cache . \
    && cd /opt \
    && git clone https://github.com/rank-and-file/inspect_ai_vllm_stdout.git \
    && cd inspect_ai_vllm_stdout \
    && uv pip install --system --no-cache .

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
COPY system_monitor.sh /usr/local/bin/system_monitor.sh
COPY ptb_collect.sh /usr/local/bin/ptb_collect.sh
COPY contamination_check.py /home/agent/contamination_check.py
COPY benchmarks /opt/ptb/benchmarks
RUN chmod +x /usr/local/bin/entrypoint.sh /usr/local/bin/system_monitor.sh /usr/local/bin/ptb_collect.sh \
    && mkdir -p /home/agent/workspace

WORKDIR /home/agent/workspace
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
