# syntax=docker/dockerfile:1.7
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# OpenCode and offline scientific tools, linux/amd64.
FROM ghcr.io/astral-sh/uv:0.12.7@sha256:95f2aa1fe59274951cfe9b0cbc7972e879ff1004bc8945d130a32eb0dbd85945 AS uv
FROM mambaorg/micromamba:2.9.0@sha256:5681ae3caa12844c41d31e1d70f636fbaba31b01bdfd1b5b521742d7e490614d AS micromamba
FROM python:3.12.14-bookworm@sha256:581429e3df12d76e6af4be5ab7d0e7fc2013eb57dc23d2de691411c8efdbb970 AS base

ARG DEBIAN_SNAPSHOT=20260916T000000Z
RUN rm -f /etc/apt/sources.list /etc/apt/sources.list.d/debian.sources && \
    printf 'deb [check-valid-until=no] https://snapshot.debian.org/archive/debian/%s/ bookworm main\ndeb [check-valid-until=no] https://snapshot.debian.org/archive/debian-security/%s/ bookworm-security main\n' "$DEBIAN_SNAPSHOT" "$DEBIAN_SNAPSHOT" > /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
      bash git curl ripgrep build-essential gfortran cmake pkg-config zstd \
      libgmp-dev libmpfr-dev libmpc-dev libgl1 libglib2.0-0 libsndfile1 \
      libgeos-dev libglpk-dev poppler-utils tesseract-ocr && \
    rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /usr/local/bin/uv
COPY --from=micromamba /bin/micromamba /usr/local/bin/micromamba
ENV UV_SYSTEM_PYTHON=1 UV_COMPILE_BYTECODE=0 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 MPLBACKEND=Agg \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    MAMBA_ROOT_PREFIX=/opt/mamba \
    OPENCODE_DISABLE_AUTOUPDATE=1 OPENCODE_DISABLE_SHARE=1 OPENCODE_DISABLE_MODELS_FETCH=true

FROM base AS sage
COPY sage-linux-64.lock /opt/image-provenance/sage-linux-64.lock
RUN micromamba create -y -p /opt/sage -f /opt/image-provenance/sage-linux-64.lock && \
    micromamba clean --all --yes

FROM base AS lean
# Mathlib v4.34.0 and the exact Lean toolchain required by its lean-toolchain file.
ARG MATHLIB_COMMIT=5ed2965256430c3649e86755f9576b54eca72435
RUN curl -fL --retry 3 https://github.com/leanprover/lean4/releases/download/v4.34.0/lean-4.34.0-linux.tar.zst -o /tmp/lean.tar.zst && \
    echo 'caaa98356098c85dc0fcbbd28e1ec66f39eb6551829972b752ff20e1286b646b  /tmp/lean.tar.zst' | sha256sum -c - && \
    mkdir -p /opt/lean && tar --zstd -xf /tmp/lean.tar.zst -C /opt/lean --strip-components=1 && \
    rm /tmp/lean.tar.zst
ENV PATH=/opt/lean/bin:$PATH
RUN git init /opt/mathlib && cd /opt/mathlib && \
    git remote add origin https://github.com/leanprover-community/mathlib4.git && \
    git fetch --depth 1 origin "$MATHLIB_COMMIT" && git checkout --detach FETCH_HEAD && \
    test "$(cat lean-toolchain)" = 'leanprover/lean4:v4.34.0' && \
    lake exe cache get && \
    mkdir -p /opt/image-provenance && \
    git rev-parse HEAD > /opt/image-provenance/mathlib-commit.txt && \
    cp lean-toolchain lake-manifest.json /opt/image-provenance/ && \
    rm -rf /root/.cache/mathlib
RUN --network=none cd /opt/mathlib && lake build Mathlib

FROM base AS python-tools
COPY requirements.in requirements.lock /opt/image-provenance/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv --no-config pip install --system --require-hashes --torch-backend cpu -r /opt/image-provenance/requirements.lock && \
    uv --no-config pip check --system

FROM base AS runtime
COPY --from=sage /opt/sage /opt/sage
COPY --from=lean /opt/lean /opt/lean
COPY --from=lean /opt/mathlib /opt/mathlib
COPY --from=lean /opt/image-provenance/ /opt/image-provenance/
COPY --from=python-tools /usr/local /usr/local
COPY requirements.in requirements.lock sage-environment.yaml sage-linux-64.lock /opt/image-provenance/
ENV PATH=/opt/lean/bin:$PATH
# Sage's environment is separate; ordinary python/python3 keep the scientific stack.
RUN printf '#!/bin/sh\nexec micromamba run -p /opt/sage sage "$@"\n' > /usr/local/bin/sage && \
    chmod 0755 /usr/local/bin/sage && \
    mkdir -p /opt/mamba /root/.astropy/config && \
    printf '[utils.iers.iers]\nauto_download = False\n' > /root/.astropy/config/astropy.cfg

RUN curl -fL --retry 3 https://github.com/anomalyco/opencode/releases/download/v1.17.11/opencode-linux-x64.tar.gz -o /tmp/opencode.tar.gz && \
    tar -xzf /tmp/opencode.tar.gz -C /usr/local/bin opencode && rm /tmp/opencode.tar.gz && \
    echo '0254a429cd0e6cf0ba53fc01672cf98e4a8dc728f7fa94be88f1e4b3645e6ded  /usr/local/bin/opencode' | sha256sum -c - && \
    chmod 0755 /usr/local/bin/opencode
COPY tools.md /opt/science/README.md
RUN uv pip freeze --system > /opt/image-provenance/python-installed.txt && \
    dpkg-query -W > /opt/image-provenance/debian-installed.txt

USER root
WORKDIR /workspace
CMD ["sleep", "infinity"]
