# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

ARG BASE_IMAGE=nvcr.io/nvidia/cuda-dl-base@sha256:fe75077262fd045ba67c6ada947a72ede195fb054aa1a7d2c498bd21fe4c95bd
FROM ${BASE_IMAGE}

ARG GYM_SOURCE_SHA=5346c8ffc9e4438959955e5b958a12733ed9abc0
ARG HERMES_AGENT_SHA=26bb847a88493342ca1b194e0455b479073ae21d
ARG HUMAN_EVAL_INFILLING_SHA=e0a239b08710e3fea48b03f4e326b3161bd650f0
ARG LLM_AZURE_OPENAI_SHA=48045f787f5480734711029d99a47973c38a5a46
ARG LLM_SHA=6de2c7ba5c5f53e95834915814c4d91a3797eeae
ARG OPENENV_SHA=5359534c6f003f81f375482ec783e80dd48b46d4
ARG TALE_SUITE_SHA=ef349ba7cfdebf339e9aedcd09d89d6c917f86e5
ARG TAU2_SHA=346f74d9752f80af8ca3e083467bafcee961bd74
ARG VECALIGN_SHA=f37262758955133d0c9ef1fdff45eba25842a62c
ARG VERIFIABLE_INSTRUCTIONS_SHA=f46a5ac87b1400a4f8973039844b6be9b56e3faf
ARG VERIFIERS_SHA=2fbd2b7fab0236cb039c4aa5afb25ad9c5f17134
ARG RUNTIME_UID=65532
ARG RUNTIME_GID=65532

USER root
RUN install -d -o "${RUNTIME_UID}" -g "${RUNTIME_GID}" /opt/repository-e2e-gym/bin && \
    /usr/bin/curl --fail --location --silent --show-error \
      https://astral.sh/uv/0.11.19/install.sh --output /tmp/uv-install.sh && \
    echo "ef8cf0575d37cf3c72e05f153dd72a845a87a7bb9be86184d5fe931b8c426250  /tmp/uv-install.sh" | \
      sha256sum --check --strict && \
    env UV_UNMANAGED_INSTALL=/opt/repository-e2e-gym/bin sh /tmp/uv-install.sh && \
    rm /tmp/uv-install.sh && \
    test "$(/opt/repository-e2e-gym/bin/uv --version | awk '{print $2}')" = 0.11.19

RUN git init /tmp/repository-e2e-gym-source && \
    git -C /tmp/repository-e2e-gym-source remote add origin https://github.com/NVIDIA-NeMo/Gym.git && \
    git -c credential.helper= -c http.https://github.com/.extraheader= \
      -C /tmp/repository-e2e-gym-source fetch --depth=1 origin "${GYM_SOURCE_SHA}" && \
    git -C /tmp/repository-e2e-gym-source checkout --detach FETCH_HEAD && \
    test "$(git -C /tmp/repository-e2e-gym-source rev-parse HEAD)" = "${GYM_SOURCE_SHA}" && \
    git -c credential.helper= -c http.https://github.com/.extraheader= \
      -C /tmp/repository-e2e-gym-source submodule update --init --recursive && \
    echo "aaaea0000b7c59b6ddc5a37146ce47c6d16bdeb62ca234f738be5d880c0fbacd  /tmp/repository-e2e-gym-source/responses_api_agents/verifiers_agent/requirements.txt" | \
      sha256sum --check --strict && \
    sed -i '\|^--extra-index-url https://hub.primeintellect.ai/primeintellect/simple/$|d' \
      /tmp/repository-e2e-gym-source/responses_api_agents/verifiers_agent/requirements.txt && \
    ! grep -R --include='requirements*.txt' -- '--extra-index-url' \
      /tmp/repository-e2e-gym-source

# uv cannot resolve mutable Git requirements in offline mode, even when their
# previously fetched objects remain in its cache. Bundle every mutable source at
# its exact resolved revision and redirect those requirements to local sources.
RUN set -eu; \
    fetch_source() { \
      name="$1"; url="$2"; revision="$3"; \
      target="/opt/repository-e2e-gym/git-sources/${name}"; \
      git init "${target}"; \
      git -C "${target}" remote add origin "${url}"; \
      git -c credential.helper= -c http.https://github.com/.extraheader= \
        -C "${target}" fetch --depth=1 origin "${revision}"; \
      git -C "${target}" checkout --detach FETCH_HEAD; \
      test "$(git -C "${target}" rev-parse HEAD)" = "${revision}"; \
    }; \
    install -d /opt/repository-e2e-gym/git-sources; \
    fetch_source hermes-agent https://github.com/cmunley1/hermes-agent \
      "${HERMES_AGENT_SHA}"; \
    fetch_source human-eval-infilling \
      https://github.com/wasiahmad/human-eval-infilling.git \
      "${HUMAN_EVAL_INFILLING_SHA}"; \
    fetch_source llm https://github.com/MarcCote/llm.git \
      "${LLM_SHA}"; \
    fetch_source llm-azure-openai https://github.com/MarcCote/llm-azure-openai.git \
      "${LLM_AZURE_OPENAI_SHA}"; \
    fetch_source openenv https://github.com/meta-pytorch/OpenEnv.git \
      "${OPENENV_SHA}"; \
    fetch_source tale-suite https://github.com/microsoft/tale-suite.git \
      "${TALE_SUITE_SHA}"; \
    fetch_source tau2 https://github.com/bxyu-nvidia/tau2-bench \
      "${TAU2_SHA}"; \
    fetch_source vecalign https://github.com/thompsonb/vecalign \
      "${VECALIGN_SHA}"; \
    fetch_source verifiable-instructions \
      https://github.com/abukharin-nv/verifiable-instructions.git \
      "${VERIFIABLE_INSTRUCTIONS_SHA}"; \
    fetch_source verifiers https://github.com/PrimeIntellect-ai/verifiers.git \
      "${VERIFIERS_SHA}"; \
    git config --file /opt/repository-e2e-gym/gitconfig \
      url."file:///opt/repository-e2e-gym/git-sources/human-eval-infilling/".insteadOf \
      https://github.com/wasiahmad/human-eval-infilling.git

COPY docker/repository-e2e-constraints.txt /opt/repository-e2e-gym/constraints.txt
COPY docker/repository-e2e-overrides.txt /opt/repository-e2e-gym/overrides.txt
COPY --chmod=0755 docker/repository-e2e-curl /opt/repository-e2e-gym/bin/curl

RUN install -d -o "${RUNTIME_UID}" -g "${RUNTIME_GID}" /opt/nemo-gym /workspace && \
    chown -R "${RUNTIME_UID}:${RUNTIME_GID}" \
      /opt/repository-e2e-gym /tmp/repository-e2e-gym-source

ENV UV_CONSTRAINT=/opt/repository-e2e-gym/constraints.txt \
    UV_OVERRIDE=/opt/repository-e2e-gym/overrides.txt \
    GIT_CONFIG_GLOBAL=/opt/repository-e2e-gym/gitconfig \
    HOME=/opt/repository-e2e-gym/home \
    XDG_CACHE_HOME=/opt/repository-e2e-gym/home/.cache \
    XDG_CONFIG_HOME=/opt/repository-e2e-gym/home/.config \
    XDG_DATA_HOME=/opt/repository-e2e-gym/home/.local/share

USER ${RUNTIME_UID}:${RUNTIME_GID}

RUN cd /tmp/repository-e2e-gym-source && \
    UV_CACHE_DIR=/opt/repository-e2e-gym/uv-cache \
      GYM_CI_DEV_VENV_DIR=/opt/repository-e2e-gym/dev-venv \
      bash scripts/ci/setup_dev.sh && \
    /opt/repository-e2e-gym/dev-venv/bin/python -m venv \
      /opt/repository-e2e-gym/pre-commit-3.6.0 && \
    /opt/repository-e2e-gym/pre-commit-3.6.0/bin/python -m pip install \
      --disable-pip-version-check pre-commit==3.6.0 && \
    PRE_COMMIT_HOME=/opt/repository-e2e-gym/pre-commit-home \
      /opt/repository-e2e-gym/pre-commit-3.6.0/bin/pre-commit install-hooks

# Run every native functional shard concurrently in isolated source and venv
# trees while package-index egress is available. This validates the dependency
# closure and fills the shared, lock-safe uv cache without coupling the image
# contents to the contract digest's sampled shard indices.
RUN set -eu; \
    pids=""; \
    for shard in 0 1 2 3 4 5 6 7; do \
      shard_root="/tmp/repository-e2e-gym-shard-${shard}"; \
      cp -a --no-preserve=ownership \
        /tmp/repository-e2e-gym-source "${shard_root}"; \
      ( \
        cd "${shard_root}"; \
        git clean -ffdx; \
        UV_CACHE_DIR=/opt/repository-e2e-gym/uv-cache \
          GYM_CI_UV_VENV_DIR="/tmp/repository-e2e-gym-venvs-${shard}" \
          bash scripts/ci/server_tests.sh "${shard}" 8 \
      ) & \
      pids="${pids} $!"; \
    done; \
    status=0; \
    for pid in ${pids}; do \
      wait "${pid}" || status=1; \
    done; \
    rm -rf \
      /tmp/repository-e2e-gym-shard-* \
      /tmp/repository-e2e-gym-venvs-* \
      /tmp/repository-e2e-gym-source; \
    test "${status}" -eq 0

ENV PATH=/opt/repository-e2e-gym/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    PRE_COMMIT_HOME=/opt/repository-e2e-gym/pre-commit-home \
    UV_CACHE_DIR=/opt/repository-e2e-gym/uv-cache \
    UV_OFFLINE=1

RUN test "$(uv --version | awk '{print $2}')" = 0.11.19 && \
    test -x /opt/repository-e2e-gym/dev-venv/bin/python && \
    test -x /opt/repository-e2e-gym/pre-commit-3.6.0/bin/pre-commit && \
    test -w /opt/repository-e2e-gym/uv-cache && \
    touch /opt/nemo-gym/.repository-e2e-write-probe && \
    rm /opt/nemo-gym/.repository-e2e-write-probe && \
    touch /workspace/.repository-e2e-write-probe && \
    rm /workspace/.repository-e2e-write-probe
