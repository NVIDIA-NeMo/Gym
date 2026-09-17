# syntax=docker/dockerfile:1.7
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Build from the Gym repository root. SCIENCE_IMAGE is an immutable digest in deployments.
ARG SCIENCE_IMAGE
FROM python:3.13.14-bookworm@sha256:7a6b87c02e1f4d6bb572e379235cbbdd7892add0572442c9b0b860a3f5aa9857 AS gym-python
FROM ${SCIENCE_IMAGE} AS runtime
COPY --from=gym-python /usr/local /opt/gym-python

# Gym's interpreter is isolated from the scientific Python on the model's PATH.
COPY pyproject.toml uv.lock /opt/gym-build/
RUN uv venv --python /opt/gym-python/bin/python3 /opt/gym-runtime && \
    cd /opt/gym-build && uv export --frozen --no-dev --no-emit-project --format requirements-txt -o requirements.lock && \
    uv --no-config pip install --python /opt/gym-runtime/bin/python --no-deps --require-hashes -r requirements.lock

RUN curl -fL --retry 3 https://nodejs.org/dist/v22.23.2/node-v22.23.2-linux-x64.tar.xz -o /tmp/node.tar.xz && \
    echo 'd60acfe00a2932254bb0ad20e01b0d74397a0875595de719654b214f4b03f307  /tmp/node.tar.xz' | sha256sum -c - && \
    mkdir /opt/node && tar -xf /tmp/node.tar.xz -C /opt/node --strip-components=1 && rm /tmp/node.tar.xz
ENV PATH=/opt/node/bin:/opt/pi/node_modules/.bin:$PATH
COPY responses_api_agents/harness_agent/image/package.json responses_api_agents/harness_agent/image/package-lock.json /opt/pi/
RUN cd /opt/pi && npm ci --ignore-scripts --omit=dev --no-audit --no-fund && \
    pi --version && opencode --version

# Preinstall the exact dependency OpenCode otherwise bootstraps on every startup.
COPY responses_api_agents/harness_agent/image/opencode/package.json responses_api_agents/harness_agent/image/opencode/package-lock.json /root/.config/opencode/
RUN cd /root/.config/opencode && npm ci --ignore-scripts --omit=dev --no-audit --no-fund

# Host-side Gym stages its exact source revision for each run; no runtime pip/npm installs.
WORKDIR /workspace
