#!/bin/sh
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Runs the image's real Chrome with the flags a root container with a small /dev/shm needs.
# Point CHROME_BIN / CHROMIUM_BIN at it; Karma customLaunchers bypass it.
for b in /opt/google/chrome/google-chrome /opt/google/chrome/chrome \
    /usr/bin/google-chrome-stable /usr/bin/google-chrome \
    /usr/lib/chromium/chrome /usr/bin/chromium-browser /usr/bin/chromium; do
  if [ -x "$b" ] && [ "$(realpath "$b")" != "$(realpath "$0")" ]; then
    exec "$b" --no-sandbox --disable-dev-shm-usage "$@"
  fi
done
echo 'chrome-wrapper.sh: no real chrome binary found' >&2
exit 1
