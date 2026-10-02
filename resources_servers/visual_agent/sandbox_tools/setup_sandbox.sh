#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# One-time setup of a policy or grader sandbox (run by the visual_agent resources server).
# Base image: apify/actor-python-playwright (Debian, Python 3.12, Playwright + Chromium). Adds what
# rendering and grading need: numpy, Pillow, ffmpeg, Node.js, curl (to download OpenCode when no cached
# binary is configured), and a set of web fonts. It also maps
# generic font families to them (the image maps `sans-serif` to a Thai font).
set -euo pipefail

# OpenCode runs in the image WORKDIR, which in the Apify images is /usr/src/app and holds the
# image's own build files. Empty it and make /workspace an alias of it, so the agent's project
# directory is the task workspace. (Replacing the directory itself breaks later commands: the
# sandbox exec daemon keeps its handle to the deleted directory.)
if [ "$(pwd)" = /usr/src/app ] && [ ! -e /workspace ]; then
    find /usr/src/app -mindepth 1 -delete
    ln -s /usr/src/app /workspace
fi
mkdir -p /workspace/output /workspace/task

python3 -c 'import numpy, PIL' 2>/dev/null || pip install -q --no-cache-dir numpy pillow

packages="ffmpeg nodejs curl ca-certificates fonts-dejavu-core fonts-inter fonts-roboto-unhinted fonts-open-sans fonts-lato"
if ! command -v ffmpeg >/dev/null || ! command -v node >/dev/null || ! command -v curl >/dev/null \
    || ! fc-list | grep -q Inter; then
    for attempt in 1 2 3; do
        if (apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends $packages) > /tmp/apt.log 2>&1; then
            break
        fi
        echo "apt attempt $attempt failed" >&2
        tail -5 /tmp/apt.log >&2
        sleep 10
    done
fi

cat > /etc/fonts/local.conf <<'EOF'
<?xml version="1.0"?>
<!DOCTYPE fontconfig SYSTEM "fonts.dtd">
<fontconfig>
  <alias><family>sans-serif</family><prefer><family>Inter</family><family>DejaVu Sans</family></prefer></alias>
  <alias><family>system-ui</family><prefer><family>Inter</family></prefer></alias>
  <alias><family>serif</family><prefer><family>DejaVu Serif</family></prefer></alias>
  <alias><family>monospace</family><prefer><family>DejaVu Sans Mono</family></prefer></alias>
  <alias><family>Helvetica</family><prefer><family>Liberation Sans</family></prefer></alias>
  <alias><family>Georgia</family><prefer><family>DejaVu Serif</family></prefer></alias>
</fontconfig>
EOF
fc-cache -f > /dev/null

command -v ffmpeg > /dev/null
command -v node > /dev/null
python3 -c 'import numpy, PIL, playwright'
