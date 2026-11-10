# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Sandbox image for Agentic-MME's OpenCode track (benchmarks/agentic_mme/opencode.yaml): Python 3 with OpenCV, NumPy
# and Pillow for the harness's own image work. OpenCode is preinstalled under /opt/opencode/bin, so a sandbox with no
# internet access can run it (benchmarks/agentic_mme/opencode.yaml sets sandbox_executable_path and its
# SHA-256). Without that setting, opencode_agent downloads OpenCode when the session starts; its installer needs bash,
# curl, tar and gzip, and uses ripgrep when present.
#
# The image also carries OpenCode's plugin dependencies (/opt/opencode/opencode-1.17.11-plugin-dependencies.tar.gz).
# opencode_agent always configures a plugin, and OpenCode 1.17.11 then installs @opencode-ai/plugin from npm at each
# start unless the dependencies are present; offline, the start stalls about 70 s before the install fails. The first
# two stages build the archive from the lockfile in opencode/ (a copy of benchmarks/agentic_if/opencode) with the
# recipe in benchmarks/agentic_if/README.md, so its bytes match the SHA-256 that benchmarks/agentic_mme/opencode.yaml
# pins.
#
#   docker build --platform linux/amd64 -t <registry>/agentic-mme-sandbox:<tag> resources_servers/agentic_mme/sandbox_image

# npm 11.12.1 installs the dependencies that the lockfile pins. npm records the folder name, so it stays "opencode".
FROM node:24.15.0-bookworm-slim AS plugin-dependencies
WORKDIR /build/opencode
COPY opencode/package.json opencode/package-lock.json ./
RUN npm ci --ignore-scripts --no-audit --no-fund

# GNU tar 1.34 and GNU gzip 1.10 (Debian 11), the versions of the pinned digest: another gzip build changes the
# compressed bytes. The build fails here if the archive does not match the pin.
FROM debian:bullseye-slim AS plugin-archive
COPY --from=plugin-dependencies /build/opencode /build/opencode
RUN tar -C /build/opencode --sort=name --mtime=@1792627200 --owner=0 --group=0 --numeric-owner \
        --mode=u+rwX,go+rX,go-w --hard-dereference --format=gnu -cf - package.json package-lock.json node_modules \
        | gzip -n -9 > /opencode-1.17.11-plugin-dependencies.tar.gz \
    && echo "77b03eba45c8d7fe3d0a58bfc215f6fb99a1db4dd18a7e3d91fd29981fddafb1  /opencode-1.17.11-plugin-dependencies.tar.gz" \
        | sha256sum -c -

FROM python:3.12-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends bash ca-certificates curl tar gzip ripgrep \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "opencv-python-headless==4.10.0.84" \
    "numpy==2.1.3" \
    "pillow==11.0.0"

# The glibc x86_64 build that opencode_agent's installer picks for this image. Its SHA-256 is pinned in
# benchmarks/agentic_mme/opencode.yaml, so a new version here needs a new pin there.
ARG OPENCODE_VERSION=1.17.11
RUN mkdir -p /opt/opencode/bin \
    && curl -fsSL "https://github.com/anomalyco/opencode/releases/download/v${OPENCODE_VERSION}/opencode-linux-x64-baseline.tar.gz" \
        | tar -xz -C /opt/opencode/bin opencode \
    && test "$(/opt/opencode/bin/opencode --version)" = "${OPENCODE_VERSION}"
COPY --from=plugin-archive /opencode-1.17.11-plugin-dependencies.tar.gz /opt/opencode/

WORKDIR /workspace
