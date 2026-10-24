# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Pinned Codex/Hermes runtime. The trial image adds only its own skill catalog.
FROM python:3.14-slim-bookworm@sha256:48b13b003dda20b16f9442b8475aa05fe21bf6579a8c881db92ffb4d8fd20f83

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash git ripgrep ca-certificates procps curl \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv==0.9.4

RUN git init /opt/hermes \
    && git -C /opt/hermes fetch --depth 1 https://github.com/NousResearch/hermes-agent.git \
       d923a5eb487a17c0a68801bba9ef0f56c81040a8 \
    && git -C /opt/hermes checkout --detach FETCH_HEAD \
    && test "$(git -C /opt/hermes rev-parse HEAD)" = d923a5eb487a17c0a68801bba9ef0f56c81040a8
WORKDIR /opt/hermes
RUN uv sync --frozen --no-dev --no-default-groups

RUN curl --fail --location --retry 3 \
    https://github.com/openai/codex/releases/download/rust-v0.156.1/codex-x86_64-unknown-linux-musl.tar.gz \
    -o /tmp/codex.tar.gz \
    && echo 'aff46539a83aff86e3c62c592bce2c50d95391f9df289afaf03a50c01d14533d  /tmp/codex.tar.gz' | sha256sum -c - \
    && mkdir /tmp/codex \
    && tar -xzf /tmp/codex.tar.gz -C /tmp/codex \
    && install /tmp/codex/codex-x86_64-unknown-linux-musl /usr/local/bin/codex \
    && test "$(codex --version)" = 'codex-cli 0.156.1' \
    && rm -rf /tmp/codex /tmp/codex.tar.gz

RUN useradd --create-home --uid 1000 agent \
    && mkdir /workspace \
    && chown agent:agent /workspace \
    && chmod -R a+rX /opt/hermes
ENV PATH=/opt/hermes/.venv/bin:$PATH PYTHONPATH=/opt/hermes PYTHONUNBUFFERED=1
USER agent
WORKDIR /workspace
CMD ["sleep", "infinity"]
