# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Build context: Gym checkout. Override BASE_IMAGE with your approved Gym driver image.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
USER root
ARG OPENCODE_SHA256
COPY responses_api_agents/agentvista_agent/requirements-worker.txt /opt/agentvista/requirements-worker.txt
COPY benchmarks/agentvista/setup.py /opt/agentvista/bootstrap/setup.py
COPY responses_api_agents/agentvista_agent/__init__.py /opt/agentvista/bootstrap/responses_api_agents/agentvista_agent/__init__.py
COPY responses_api_agents/agentvista_agent/runtime/source.py responses_api_agents/agentvista_agent/runtime/source-pin.json /opt/agentvista/bootstrap/responses_api_agents/agentvista_agent/runtime/
# OpenCode CLI 1.17.11 must be supplied as a verified build input.
COPY opencode /opt/agentvista/bin/opencode
RUN uv venv --python 3.13 /opt/agentvista-worker \
    && uv --no-config pip install --python /opt/agentvista-worker/bin/python --index-strategy unsafe-best-match --require-hashes -r /opt/agentvista/requirements-worker.txt \
    && uv pip freeze --python /opt/agentvista-worker/bin/python > /opt/agentvista/installed-packages.txt \
    && PYTHONPATH=/opt/agentvista/bootstrap python /opt/agentvista/bootstrap/setup.py --source /opt/agentvista/source \
    && test "$(/opt/agentvista/bin/opencode --version)" = 1.17.11 \
    && test -n "$OPENCODE_SHA256" && echo "$OPENCODE_SHA256  /opt/agentvista/bin/opencode" | sha256sum -c - \
    && /opt/agentvista-worker/bin/python -c 'import qwen_vl_utils, torch, jupyter_client, ipykernel, mcp' \
    && PYTHONPATH=/opt/agentvista/source /opt/agentvista-worker/bin/python -c 'import infer; from tools import list_tools; assert {"web_search","image_search","visit","code_interpreter"} <= set(list_tools())'
ENV AGENTVISTA_ROOT=/opt/agentvista/source \
    AGENTVISTA_WORKER_PYTHON=/opt/agentvista-worker/bin/python \
    AGENTVISTA_OPENCODE=/opt/agentvista/bin/opencode \
    PYTHONDONTWRITEBYTECODE=1
