# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Build from the repository root, after building resources_servers/deepcad/Dockerfile.
# Use the existing stock OpenCode plugin lock and reproducible archive recipe.
FROM node:24.15.0-bookworm-slim AS plugin-dependencies
WORKDIR /build/opencode
COPY resources_servers/visual_coding_opencode/sandbox_image/opencode/package.json resources_servers/visual_coding_opencode/sandbox_image/opencode/package-lock.json ./
RUN npm ci --ignore-scripts --no-audit --no-fund

FROM debian:bullseye-slim AS plugin-archive
ARG TARGETARCH
COPY --from=plugin-dependencies /build/opencode /build/opencode
RUN tar -C /build/opencode --sort=name --mtime=@1792627200 --owner=0 --group=0 --numeric-owner \
        --mode=u+rwX,go+rX,go-w --hard-dereference --format=gnu -cf - package.json package-lock.json node_modules \
        | gzip -n -9 > /opencode-1.17.11-plugin-dependencies.tar.gz \
    && case "$TARGETARCH" in \
        amd64) digest=77b03eba45c8d7fe3d0a58bfc215f6fb99a1db4dd18a7e3d91fd29981fddafb1 ;; \
        arm64) digest=12ce5f79b27aa8886b64082c1f5d23ab4bc7410e193b7d98146f5df1a61f243a ;; \
        *) echo "Unsupported architecture: $TARGETARCH" >&2; exit 1 ;; \
       esac \
    && echo "$digest  /opencode-1.17.11-plugin-dependencies.tar.gz" | sha256sum -c -

FROM deepcad-cadquery:2.8.0
ARG TARGETARCH
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends bash ca-certificates curl tar gzip ripgrep \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir 'numpy>=2.2' 'pillow>=11'
RUN case "$TARGETARCH" in \
        amd64) archive=opencode-linux-x64-baseline.tar.gz; digest=0254a429cd0e6cf0ba53fc01672cf98e4a8dc728f7fa94be88f1e4b3645e6ded ;; \
        arm64) archive=opencode-linux-arm64.tar.gz; digest=e51b4a9c0e8dfa90c3387f0aa41e7c362d56b1939eec102ae76c1aac0d911186 ;; \
        *) echo "Unsupported architecture: $TARGETARCH" >&2; exit 1 ;; \
       esac \
    && mkdir -p /opt/opencode/bin \
    && curl -fsSL "https://github.com/anomalyco/opencode/releases/download/v1.17.11/$archive" \
        | tar -xz -C /opt/opencode/bin opencode \
    && echo "$digest  /opt/opencode/bin/opencode" | sha256sum -c - \
    && test "$(/opt/opencode/bin/opencode --version)" = "1.17.11"
COPY --from=plugin-archive /opencode-1.17.11-plugin-dependencies.tar.gz /opt/opencode/
COPY resources_servers/deepcad/sandbox_image/cad_render.py /workspace/.cad/cad_render.py
COPY resources_servers/deepcad/dataset/prepare_gym.py resources_servers/deepcad/dataset/augment_inputs.py /workspace/.cad/resources_servers/deepcad/dataset/
RUN mkdir -p /opt/nemo-gym-nooa \
    && chown -R cad:cad /workspace /opt/nemo-gym-nooa
ENV OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MESA_SHADER_CACHE_DISABLE=true
USER cad
WORKDIR /workspace
