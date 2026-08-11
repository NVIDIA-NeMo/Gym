# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

ARG BASE_IMAGE=766267172432.dkr.ecr.us-east-1.amazonaws.com/nemo-autobot/agent-sandbox@sha256:cad1aadc9f89dca5b616d11b54a26ebdbb449f98cefde7d63a86934d9a4b4004
FROM ${BASE_IMAGE}

ARG GYM_SOURCE_SHA=5346c8ffc9e4438959955e5b958a12733ed9abc0
ARG RUNTIME_UID=65532
ARG RUNTIME_GID=65532

USER root
RUN install -d -o "${RUNTIME_UID}" -g "${RUNTIME_GID}" /opt/repository-e2e-gym/bin && \
    /usr/bin/curl -LsSf https://astral.sh/uv/0.11.19/install.sh | \
      env UV_UNMANAGED_INSTALL=/opt/repository-e2e-gym/bin sh && \
    test "$(/opt/repository-e2e-gym/bin/uv --version | awk '{print $2}')" = 0.11.19

RUN git init /tmp/repository-e2e-gym-source && \
    git -C /tmp/repository-e2e-gym-source remote add origin https://github.com/ko3n1g/Gym.git && \
    git -c credential.helper= -c http.https://github.com/.extraheader= \
      -C /tmp/repository-e2e-gym-source fetch --depth=1 origin "${GYM_SOURCE_SHA}" && \
    git -C /tmp/repository-e2e-gym-source checkout --detach FETCH_HEAD && \
    test "$(git -C /tmp/repository-e2e-gym-source rev-parse HEAD)" = "${GYM_SOURCE_SHA}" && \
    git -c credential.helper= -c http.https://github.com/.extraheader= \
      -C /tmp/repository-e2e-gym-source submodule update --init --recursive

RUN cd /tmp/repository-e2e-gym-source && \
    UV_CACHE_DIR=/opt/repository-e2e-gym/uv-cache \
      GYM_CI_DEV_VENV_DIR=/opt/repository-e2e-gym/dev-venv \
      bash scripts/ci/setup_dev.sh && \
    python -m venv /opt/repository-e2e-gym/pre-commit-3.6.0 && \
    /opt/repository-e2e-gym/pre-commit-3.6.0/bin/python -m pip install \
      --disable-pip-version-check pre-commit==3.6.0 && \
    PRE_COMMIT_HOME=/opt/repository-e2e-gym/pre-commit-home \
      /opt/repository-e2e-gym/pre-commit-3.6.0/bin/pre-commit install-hooks

COPY docker/repository-e2e-constraints.txt /opt/repository-e2e-gym/constraints.txt
COPY docker/repository-e2e-overrides.txt /opt/repository-e2e-gym/overrides.txt
ENV UV_CONSTRAINT=/opt/repository-e2e-gym/constraints.txt \
    UV_OVERRIDE=/opt/repository-e2e-gym/overrides.txt

# Run every native functional shard while package-index egress is available.
# This validates the dependency closure and fills the immutable uv cache without
# coupling the image contents to the contract digest's sampled shard indices.
RUN set -eu; \
    cd /tmp/repository-e2e-gym-source; \
    for shard in 0 1 2 3 4 5 6 7; do \
      git clean -ffdx; \
      UV_CACHE_DIR=/opt/repository-e2e-gym/uv-cache \
        GYM_CI_UV_VENV_DIR=/tmp/repository-e2e-gym-venvs \
        bash scripts/ci/server_tests.sh "${shard}" 8; \
    done; \
    rm -rf /tmp/repository-e2e-gym-venvs /tmp/repository-e2e-gym-source; \
    chown -R "${RUNTIME_UID}:${RUNTIME_GID}" /opt/repository-e2e-gym

RUN install -d -o "${RUNTIME_UID}" -g "${RUNTIME_GID}" /opt/nemo-gym /workspace

COPY --chmod=0755 docker/repository-e2e-curl /opt/repository-e2e-gym/bin/curl

ENV PATH=/opt/repository-e2e-gym/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    PRE_COMMIT_HOME=/opt/repository-e2e-gym/pre-commit-home \
    UV_CACHE_DIR=/opt/repository-e2e-gym/uv-cache \
    UV_OFFLINE=1

USER ${RUNTIME_UID}:${RUNTIME_GID}
RUN test "$(uv --version | awk '{print $2}')" = 0.11.19 && \
    test -x /opt/repository-e2e-gym/dev-venv/bin/python && \
    test -x /opt/repository-e2e-gym/pre-commit-3.6.0/bin/pre-commit && \
    test -w /opt/repository-e2e-gym/uv-cache && \
    touch /opt/nemo-gym/.repository-e2e-write-probe && \
    rm /opt/nemo-gym/.repository-e2e-write-probe && \
    touch /workspace/.repository-e2e-write-probe && \
    rm /workspace/.repository-e2e-write-probe
