# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
FROM pytorch/pytorch:2.9.0-cuda13.0-cudnn9-runtime

ARG KERNELBENCH_COMMIT=423217d9fda91e0c2d67e4a43bf62f96f6d104f1
RUN apt-get update && apt-get install -y --no-install-recommends curl git xz-utils && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/ScalingIntelligence/KernelBench.git /opt/KernelBench \
    && git -C /opt/KernelBench checkout "$KERNELBENCH_COMMIT" \
    && pip install --no-cache-dir ninja pydra-config python-dotenv openai \
    && pip install --no-cache-dir --no-deps --ignore-requires-python -e /opt/KernelBench

COPY pyproject.toml README.md LICENSE /opt/Gym/
COPY nemo_gym /opt/Gym/nemo_gym
COPY responses_api_agents/claude_code_agent /opt/Gym/responses_api_agents/claude_code_agent
COPY responses_api_agents/anyterminal_agent/setup_scripts/_portable_python.sh /opt/Gym/_portable_python.sh
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mkdir -p /opt/Gym/cache \
    && /root/.local/bin/uv venv --seed --python 3.13.14 /opt/agent \
    && PORTABLE_PYTHON_SH=/opt/Gym/_portable_python.sh DEPS_DIR=/opt/agent NEMO_GYM_ROOT=/opt/Gym \
        bash /opt/Gym/responses_api_agents/claude_code_agent/scripts/claude_code_agent_deps.sh

ENV PYTHONPATH=/opt/KernelBench/src
WORKDIR /workspace
