# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Sandbox image for visual coding in OpenCode sandbox sessions (configs/visual_coding_opencode.yaml). The model's shell
# has a Python image toolkit: Pillow, NumPy, SciPy and OpenCV. OpenCode is preinstalled under /opt/opencode/bin, so a
# sandbox that reaches only the model server can run it. visual_coding_opencode.yaml pins that binary's SHA-256, and
# the build checks it, so a new OpenCode version here needs a new pin in both places.
#
# The image also carries OpenCode's plugin dependencies (/opt/opencode/opencode-1.17.11-plugin-dependencies.tar.gz).
# opencode_agent always configures a plugin, and OpenCode 1.17.11 then installs @opencode-ai/plugin from npm at each
# start unless the dependencies are present; offline, the start stalls about 70 s before the install fails. The first
# two stages build the archive from the lockfile in opencode/ (a copy of benchmarks/agentic_if/opencode) with the
# recipe in benchmarks/agentic_if/README.md, so its bytes match the SHA-256 that visual_coding_opencode.yaml pins.
#
#   docker build --platform linux/amd64 -t <registry>/visual-coding-opencode-sandbox:<tag> \
#       resources_servers/visual_coding_opencode/sandbox_image

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

FROM python:3.13-slim-bookworm

# bash for OpenCode's shell tool, ripgrep for its grep and glob tools, curl and tar for the OpenCode download below.
RUN apt-get update \
    && apt-get install -y --no-install-recommends bash ca-certificates curl tar gzip ripgrep \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "pillow==13.0.0" \
    "numpy==2.5.4" \
    "scipy==1.18.1" \
    "opencv-python-headless==5.0.0.93"

# One thread per numeric library, so parallel tasks on one host do not oversubscribe its CPUs.
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    OPENCV_FOR_THREADS_NUM=1

# The glibc x86_64 build that opencode_agent's installer picks for this image. The build fails here unless the binary
# matches the SHA-256 that visual_coding_opencode.yaml pins (sandbox_executable_sha256), so a mismatched image fails at
# build time instead of at the first episode.
RUN mkdir -p /opt/opencode/bin \
    && curl -fsSL "https://github.com/anomalyco/opencode/releases/download/v1.17.11/opencode-linux-x64-baseline.tar.gz" \
        | tar -xz -C /opt/opencode/bin opencode \
    && echo "0254a429cd0e6cf0ba53fc01672cf98e4a8dc728f7fa94be88f1e4b3645e6ded  /opt/opencode/bin/opencode" \
        | sha256sum -c - \
    && test "$(/opt/opencode/bin/opencode --version)" = "1.17.11"
COPY --from=plugin-archive /opencode-1.17.11-plugin-dependencies.tar.gz /opt/opencode/

WORKDIR /workspace
