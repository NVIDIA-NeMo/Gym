# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Sandbox image for mm_coding_training. One image serves two jobs:
# - The grading sandbox runs renderer.py, which the server uploads to each fresh sandbox. It renders Matplotlib code,
#   SVG (CairoSVG), LaTeX (pdflatex and pdftoppm) and HTML (Playwright Chromium) to a PNG. `renderer.py --check` is
#   the dependency preflight that each render runs first. The build runs it too, so an image that fails it does not
#   build.
# - The OpenCode task sandbox of mm_coding_multi.yaml runs OpenCode 1.17.11 from /opt/opencode/bin/opencode. Set
#   MM_CODING_OPENCODE_PATH to that path and MM_CODING_OPENCODE_SHA256 to the SHA-256 below. The server delivers
#   ripgrep and the OpenCode plugin dependencies from the host (MM_CODING_RG_SOURCE, MM_CODING_PLUGIN_BUNDLE_SOURCE).
#
#   docker build --platform linux/amd64 -t <registry>/mm-coding-sandbox:<tag> \
#       -f resources_servers/mm_coding_training/sandbox_image/Dockerfile resources_servers/mm_coding_training
#
# Push the image, then set MM_CODING_SANDBOX_IMAGE to <registry>/mm-coding-sandbox@sha256:<digest>. The server accepts
# only a digest reference.

FROM python:3.13-slim-bookworm

# bash for OpenCode's shell tool, curl and tar for the OpenCode download below, TeX Live and Poppler for LaTeX answers,
# Cairo for SVG answers, and fonts for all renders.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash ca-certificates curl tar gzip file \
        texlive-latex-base texlive-latex-extra texlive-fonts-recommended texlive-science texlive-pictures \
        poppler-utils libcairo2 fonts-dejavu fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

# Chart2Code answers run as Python with Matplotlib, NumPy and pandas. Playwright installs Chromium and its system
# libraries for HTML answers. A fixed browser folder that every user can read lets a sandbox that runs as another user
# or with another HOME find Chromium.
ENV PLAYWRIGHT_BROWSERS_PATH=/opt/ms-playwright
RUN pip install --no-cache-dir \
        "matplotlib==3.11.2" \
        "numpy==2.5.4" \
        "pandas==3.1.0" \
        "pillow==13.0.0" \
        "cairosvg==2.9.1" \
        "playwright==1.64.1" \
    && playwright install --with-deps chromium \
    && chmod -R a+rX "$PLAYWRIGHT_BROWSERS_PATH" \
    && rm -rf /var/lib/apt/lists/*

# One thread per numeric library, so parallel tasks on one host do not oversubscribe its CPUs.
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    MPLBACKEND=Agg

# The glibc x86_64 build of OpenCode 1.17.11, the same binary and SHA-256 as the visual_coding_opencode image.
RUN mkdir -p /opt/opencode/bin \
    && curl -fsSL "https://github.com/anomalyco/opencode/releases/download/v1.17.11/opencode-linux-x64-baseline.tar.gz" \
        | tar -xz -C /opt/opencode/bin opencode \
    && echo "0254a429cd0e6cf0ba53fc01672cf98e4a8dc728f7fa94be88f1e4b3645e6ded  /opt/opencode/bin/opencode" \
        | sha256sum -c - \
    && test "$(/opt/opencode/bin/opencode --version)" = "1.17.11"

# The build fails here if a renderer dependency or a LaTeX package that renderer.py loads is missing.
COPY renderer.py /tmp/renderer.py
RUN python /tmp/renderer.py --check && rm /tmp/renderer.py

WORKDIR /workspace
