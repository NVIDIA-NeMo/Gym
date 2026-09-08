# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
FROM pytorch/pytorch:2.9.0-cuda13.0-cudnn9-runtime

ARG KERNELBENCH_COMMIT=423217d9fda91e0c2d67e4a43bf62f96f6d104f1
RUN apt-get update && apt-get install -y --no-install-recommends curl git && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/ScalingIntelligence/KernelBench.git /opt/KernelBench \
    && git -C /opt/KernelBench checkout "$KERNELBENCH_COMMIT" \
    && pip install --no-cache-dir ninja pydra-config python-dotenv openai \
    && pip install --no-cache-dir --no-deps --ignore-requires-python -e /opt/KernelBench

ENV PYTHONPATH=/opt/KernelBench/src
WORKDIR /workspace
