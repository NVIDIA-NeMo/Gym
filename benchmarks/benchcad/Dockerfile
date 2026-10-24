# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
FROM python:3.12-slim
COPY responses_api_agents/benchcad_agent/cad-requirements.txt /gym/responses_api_agents/benchcad_agent/
COPY responses_api_agents/benchcad_agent/cad-constraints.txt /gym/responses_api_agents/benchcad_agent/
COPY benchmarks/benchcad/install-sandbox.sh /gym/benchmarks/benchcad/
RUN bash /gym/benchmarks/benchcad/install-sandbox.sh /gym \
    && rm -rf /gym /var/lib/apt/lists/*
ENV PATH="/opt/benchcad/.venv/bin:/usr/local/bin:${PATH}"
WORKDIR /workspace
