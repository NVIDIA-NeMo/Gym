# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Build context: Gym checkout. Override BASE_IMAGE with your approved Gym driver image.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
USER root
COPY responses_api_agents/agentvista_agent/requirements-worker.txt /opt/agentvista/requirements-worker.txt
COPY benchmarks/agentvista/setup.py benchmarks/agentvista/source-pin.json /opt/agentvista/bootstrap/
# OpenCode CLI 1.17.11 must be supplied as a verified build input.
COPY opencode /opt/agentvista/bin/opencode
RUN uv venv --python 3.13 /opt/agentvista-worker \
    && uv pip install --python /opt/agentvista-worker/bin/python -r /opt/agentvista/requirements-worker.txt \
    && python /opt/agentvista/bootstrap/setup.py --source /opt/agentvista/source \
    && test "$(/opt/agentvista/bin/opencode --version)" = 1.17.11 \
    && /opt/agentvista-worker/bin/python -c 'import qwen_vl_utils, torch, jupyter_client, ipykernel, mcp' \
    && PYTHONPATH=/opt/agentvista/source /opt/agentvista-worker/bin/python -c 'import infer; from tools import list_tools; assert {"web_search","image_search","visit","code_interpreter"} <= set(list_tools())'
ENV AGENTVISTA_ROOT=/opt/agentvista/source \
    AGENTVISTA_WORKER_PYTHON=/opt/agentvista-worker/bin/python \
    AGENTVISTA_OPENCODE=/opt/agentvista/bin/opencode \
    PYTHONDONTWRITEBYTECODE=1
