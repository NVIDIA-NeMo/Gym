# syntax=docker/dockerfile:1.7
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# HLE-Diamond reference sandbox for OpenCode, linux/amd64: upstream's Python 3.11.16 environment
# (centerforaisafety/hle docs/evaluation-with-tools.md, section 2) plus the pinned OpenCode binary,
# installed exactly as ../offline_science_image does. Nothing else: no SageMath, Lean, Z3, RDKit, OCR
# or chess engines, which that protocol forbids.
FROM ghcr.io/astral-sh/uv:0.12.7@sha256:95f2aa1fe59274951cfe9b0cbc7972e879ff1004bc8945d130a32eb0dbd85945 AS uv
FROM python:3.11.16-slim-bookworm@sha256:a36c24f9cbdf4fd0f52d67f0823eeac19c2028c637cecc392d97f980d4fec56b AS base

ARG DEBIAN_SNAPSHOT=20260916T000000Z
RUN rm -f /etc/apt/sources.list /etc/apt/sources.list.d/debian.sources && \
    printf 'deb [check-valid-until=no] https://snapshot.debian.org/archive/debian/%s/ bookworm main\ndeb [check-valid-until=no] https://snapshot.debian.org/archive/debian-security/%s/ bookworm-security main\n' "$DEBIAN_SNAPSHOT" "$DEBIAN_SNAPSHOT" > /etc/apt/sources.list
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 MPLBACKEND=Agg \
    OPENCODE_DISABLE_AUTOUPDATE=1 OPENCODE_DISABLE_SHARE=1 OPENCODE_DISABLE_MODELS_FETCH=true

# Build-time downloads only; curl and xz never reach the runtime image.
FROM base AS fetch
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl xz-utils && \
    rm -rf /var/lib/apt/lists/*

# The OpenCode binary, pinned to the same release and checksum as ../offline_science_image.
FROM fetch AS opencode-binary
RUN curl -fL --retry 3 https://github.com/anomalyco/opencode/releases/download/v1.17.11/opencode-linux-x64.tar.gz -o /tmp/opencode.tar.gz && \
    tar -xzf /tmp/opencode.tar.gz -C /usr/local/bin opencode && rm /tmp/opencode.tar.gz && \
    echo '0254a429cd0e6cf0ba53fc01672cf98e4a8dc728f7fa94be88f1e4b3645e6ded  /usr/local/bin/opencode' | sha256sum -c - && \
    chmod 0755 /usr/local/bin/opencode

# OpenCode's preinstalled plugin dependency (@opencode-ai/plugin), resolved offline from the same lock as
# ../offline_science_image/opencode so the sandbox never fetches it. Node is needed only to run npm here.
FROM fetch AS opencode-dependencies
RUN curl -fL --retry 3 https://nodejs.org/dist/v22.23.2/node-v22.23.2-linux-x64.tar.xz -o /tmp/node.tar.xz && \
    echo 'd60acfe00a2932254bb0ad20e01b0d74397a0875595de719654b214f4b03f307  /tmp/node.tar.xz' | sha256sum -c - && \
    mkdir /opt/node && tar -xf /tmp/node.tar.xz -C /opt/node --strip-components=1 && rm /tmp/node.tar.xz
ENV PATH=/opt/node/bin:$PATH
COPY opencode/package.json opencode/package-lock.json /root/.config/opencode/
RUN cd /root/.config/opencode && npm ci --ignore-scripts --omit=dev --no-audit --no-fund

# Upstream's pinned requirements (requirements-tools.txt, verbatim) through a hashed resolution of their
# transitive closure (requirements.lock, see README.md). Installed into the image's own Python 3.11.16.
FROM base AS python-tools
COPY --from=uv /uv /opt/uv
COPY requirements-tools.txt requirements.lock /opt/image-provenance/
RUN --mount=type=cache,target=/root/.cache/uv \
    /opt/uv --no-config pip install --system --require-hashes -r /opt/image-provenance/requirements.lock && \
    /opt/uv --no-config pip check --system

FROM base AS runtime
# ripgrep backs OpenCode's glob/grep tools; without it OpenCode tries to download rg, which the
# reference network policy forbids. The only Debian package added to the base image.
RUN apt-get update && apt-get install -y --no-install-recommends ripgrep && \
    rm -rf /var/lib/apt/lists/*
COPY --from=opencode-binary /usr/local/bin/opencode /usr/local/bin/opencode
COPY --from=opencode-dependencies /root/.config/opencode /root/.config/opencode
COPY --from=python-tools /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=python-tools /usr/local/bin /usr/local/bin
COPY requirements-tools.txt requirements.lock /opt/image-provenance/
RUN test "$(python --version)" = 'Python 3.11.16' && \
    test "$(opencode --version)" = '1.17.11' && \
    pip freeze > /opt/image-provenance/python-installed.txt && \
    dpkg-query -W > /opt/image-provenance/debian-installed.txt

USER root
WORKDIR /workspace
CMD ["sleep", "infinity"]
