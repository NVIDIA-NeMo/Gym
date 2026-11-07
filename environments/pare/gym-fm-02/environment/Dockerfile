# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

FROM ghcr.io/astral-sh/uv:0.12.5 AS uv-tools
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash build-essential ca-certificates curl git procps \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv-tools /uv /uvx /usr/local/bin/
ENV PATH="/root/.local/bin:${PATH}"

# Gym needs a newer Python than Ubuntu's system Python.
ARG PYTHON_VERSION=3.13.14
RUN uv python install "${PYTHON_VERSION}"

# Install Gym and its bundled examples from the selected full commit SHA.
# The default commit is Gym v0.7.0.
ARG GYM_COMMIT=1f8afc71d45fac14872d86fe300374afa2001e7a
RUN git init /opt/gym \
    && git -C /opt/gym remote add origin https://github.com/NVIDIA-NeMo/Gym.git \
    && git -C /opt/gym fetch --depth 1 origin "${GYM_COMMIT}" \
    && git -C /opt/gym checkout --detach FETCH_HEAD \
    && test "$(git -C /opt/gym rev-parse HEAD)" = "${GYM_COMMIT}"

WORKDIR /opt/gym
RUN uv sync --locked --no-dev --python "${PYTHON_VERSION}" \
    && .venv/bin/gym --version \
    && test -s resources_servers/workplace_assistant/data/example.jsonl
ENV PATH="/opt/gym/.venv/bin:${PATH}"

# Capture the selected Gym version's bundled outputs before the agent can edit
# its samples. Keep this verifier data outside Gym's source and the workspace.
COPY generate_sample_hashes.py /opt/gym-fixture/generate_sample_hashes.py
RUN python /opt/gym-fixture/generate_sample_hashes.py \
    /opt/gym /opt/gym-fixture/bundled-output-hashes.json

RUN mkdir -p /workspace

# Credentials arrive through Harbor's runtime agent or solution environment.
WORKDIR /workspace
CMD ["sleep", "infinity"]
