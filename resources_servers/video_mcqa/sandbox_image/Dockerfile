# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Sandbox image of the agentic video MCQA setting (configs/*_agentic.yaml of videomme_v2, videomme_v1 and lvbench).
# It carries OpenCode 1.17.11, FFmpeg 7.0.2 and the video tools: the OpenCode plugin video_tools.mjs, which adds
# watch_video and view_image, and the video_tools program that the plugin runs for each call. OpenCode is preinstalled
# under /opt/opencode/bin, so a sandbox that reaches only the model server can run it; the agentic configs pin that
# binary's SHA-256 and the plugin dependencies' (the pins of visual_coding_opencode). The resources server puts each
# episode's video under /testbed and checks at each seed that /opt/video-mcqa holds this repository's video tools, so
# rebuild the image after changing them.
#
# The image is the full node:22 Debian bookworm image, pinned by digest, that our agentic video training data was
# generated in, so that evaluation and RL run in the same system: its shell, ImageMagick, git, compilers and other
# programs. The tools stage installs the programs above, and the last stage copies them into that image and adds the
# URW fonts that ImageMagick's labels default to. Unlike in that sandbox, the model's drawtext, montage and
# convert -annotate commands work, and the video tools also read the files the model makes under /tmp/opencode.
#
# The first two stages build OpenCode's plugin dependencies from the lockfile in opencode/ (a copy of
# benchmarks/agentic_if/opencode) as visual_coding_opencode/sandbox_image/Dockerfile does, so the archive matches its
# pinned SHA-256. Without it, OpenCode tries to install them from npm at each start, and an offline start waits about
# 70 s. The video tools import nothing from it.
#
#   docker build --platform linux/amd64 -t <registry>/video-mcqa-sandbox:<tag> resources_servers/video_mcqa/sandbox_image

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

# The static FFmpeg 7.0.2 build of mwader/static-ffmpeg, pinned by image digest here and by the binaries' SHA-256
# below. Its decoders and area scaler give watch_video the same frames as the johnvansickle.com 7.0.2 build this image
# used before (identical JPEG bytes on 50 requests); it also has libharfbuzz and fontconfig, so drawtext works.
FROM mwader/static-ffmpeg:7.0.2@sha256:d9b22f65c2049d73bf8ae556bb4c5c7d45e9fb85939d87951ed98f22b0f19105 AS ffmpeg

# Python 3.13 (the python image pinned by digest, so its files stay the same) with the tools' packages, ripgrep,
# OpenCode and the video tools.
FROM python:3.13-slim-bookworm@sha256:a5a7017efc5969e00ae2b91c8eb0631049a569652aa4ffc4501769049ae45c72 AS tools

# bash for OpenCode's shell tool, ripgrep for its grep and glob tools, curl and tar for the OpenCode download below.
RUN apt-get update \
    && apt-get install -y --no-install-recommends bash ca-certificates curl tar gzip ripgrep \
    && rm -rf /var/lib/apt/lists/*

# Pillow encodes the JPEG frames, so its version fixes their bytes.
RUN pip install --no-cache-dir \
    "numpy==2.4.6" \
    "pillow==12.3.0" \
    "pydantic==2.13.5"

# One thread per numeric library, so parallel tasks on one host do not oversubscribe its CPUs.
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1

COPY --from=ffmpeg /ffmpeg /ffprobe /usr/local/bin/
RUN echo "babb8d45fab2016595558852d2935f69086bb1143a6a2a898390940ecca8bcd2  /usr/local/bin/ffmpeg" | sha256sum -c - \
    && echo "66b8cb3851ac3ae77bb47b479315c62e1fc183fedab1fc7d4f33f941acde076f  /usr/local/bin/ffprobe" | sha256sum -c -

# The glibc x86_64 build that opencode_agent's installer picks for this image. The build fails here unless the binary
# matches the SHA-256 that the agentic configs pin (sandbox_executable_sha256).
RUN mkdir -p /opt/opencode/bin \
    && curl -fsSL "https://github.com/anomalyco/opencode/releases/download/v1.17.11/opencode-linux-x64-baseline.tar.gz" \
        | tar -xz -C /opt/opencode/bin opencode \
    && echo "0254a429cd0e6cf0ba53fc01672cf98e4a8dc728f7fa94be88f1e4b3645e6ded  /opt/opencode/bin/opencode" \
        | sha256sum -c - \
    && test "$(/opt/opencode/bin/opencode --version)" = "1.17.11"
COPY --from=plugin-archive /opencode-1.17.11-plugin-dependencies.tar.gz /opt/opencode/

# The video tools; a .pth file puts video_tools on the path of `python3 -I`, the plugin's command.
COPY video_tools.mjs /opt/video-mcqa/video_tools.mjs
COPY video_tools /opt/video-mcqa/video_tools
RUN python3 -c "import pathlib, site; pathlib.Path(site.getsitepackages()[0], 'video-mcqa.pth').write_text('/opt/video-mcqa\n')" \
    && python3 -I -c "import video_tools.cli" \
    && mkdir -p /testbed/videos /testbed/work

# node:22.23.3-bookworm as it was when the training data was generated (the tag has been rebuilt since), with the tools
# copied in. Its PATH puts /usr/local/bin first, so python3 is the tools' Python 3.13.
FROM node:22.23.3-bookworm@sha256:363e1587494626837fa7f9a23bdb453d13b0ff3c67c705c2805cfc69c2d2fad7

COPY --from=tools --chown=0:0 /usr/local /usr/local
COPY --from=tools --chown=0:0 /opt/opencode /opt/opencode
COPY --from=tools --chown=0:0 /opt/video-mcqa /opt/video-mcqa
COPY --from=tools /usr/bin/rg /usr/bin/rg

# The tools stage's thread limits, and its locale (the python image's).
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    LANG=C.UTF-8

# The URW fonts that ImageMagick's labels and annotations default to (Helvetica): without them, its montage and
# convert -annotate abort. The build fails here unless the model's own commands work: drawtext with FFmpeg's default
# font (DejaVu, which the image has), ripgrep, montage with a label and convert -annotate. The drawtext check leaves the
# font cache of FFmpeg's own fontconfig, so the model's first drawtext does not scan the fonts.
RUN apt-get update \
    && apt-get install -y --no-install-recommends fonts-urw-base35 \
    && rm -rf /var/lib/apt/lists/* \
    && ffmpeg -hide_banner -loglevel error -f lavfi -i color=c=black:s=64x32:d=1 -vf drawtext=text=ok \
        -frames:v 1 -f null - \
    && rg --version > /dev/null \
    && convert -size 8x8 xc:red /tmp/check.png \
    && montage -label '%f' /tmp/check.png /tmp/check.png -tile 2x1 -geometry +1+1 /tmp/check-montage.png \
    && convert /tmp/check.png -annotate +0+4 ok /tmp/check-annotate.png \
    && rm /tmp/check.png /tmp/check-montage.png /tmp/check-annotate.png

RUN mkdir -m 1777 /testbed && mkdir /testbed/work

WORKDIR /testbed/work
