# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Make sure the Playwright browser binary exists before anything tries to launch it.

The `playwright` wheel ships the Node driver but not the browsers; those come from
`playwright install`. A fresh environment therefore imports cleanly and then fails at
launch time with an executable-not-found deep inside a rollout. Installing on first
use turns that into a one-off startup cost.

Only the local backend needs this. `RemoteCDPBackend` connects to a browser someone
else is running, so it never calls in here.
"""

from __future__ import annotations

import logging
import os
import pathlib
import subprocess
import sys
import threading


LOGGER = logging.getLogger(__name__)

_INSTALL_TIMEOUT_S = 600.0
_lock = threading.Lock()
_done = False


def _browsers_dir() -> pathlib.Path:
    """Where `playwright install` puts its browsers on this platform."""
    override = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    if override:
        return pathlib.Path(override)
    if sys.platform == "darwin":
        return pathlib.Path.home() / "Library" / "Caches" / "ms-playwright"
    if sys.platform == "win32":
        return pathlib.Path(os.environ.get("LOCALAPPDATA", "")) / "ms-playwright"
    return pathlib.Path.home() / ".cache" / "ms-playwright"


def _executable_present() -> bool:
    """True when a Chromium build is already installed.

    Reads the cache directory rather than asking Playwright: `sync_playwright()`
    would answer too, but it starts the Node driver and refuses to run inside a
    running event loop -- which is where a resources server calls this from.
    """
    root = _browsers_dir()
    if not root.is_dir():
        return False
    return any(
        any(d.rglob(name))
        for d in root.glob("chromium*")
        if d.is_dir()
        for name in ("chrome-headless-shell", "headless_shell", "chrome")
    )


def ensure_chromium() -> None:
    """Install Chromium if it is missing. Safe to call repeatedly and from threads.

    Raises rather than returning quietly when the install fails: a launch a moment
    later would fail anyway, and this message says what actually went wrong.
    """
    global _done
    if _done:
        return
    with _lock:
        if _done:
            return
        if _executable_present():
            _done = True
            return

        LOGGER.info("Playwright Chromium is missing; installing it (one-off, may take a minute)")
        result = subprocess.run(
            [sys.executable, "-m", "playwright", "install", "chromium"],
            capture_output=True,
            text=True,
            timeout=_INSTALL_TIMEOUT_S,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Could not install Playwright Chromium, which the local browser backend needs. "
                f"`python -m playwright install chromium` exited {result.returncode}.\n"
                f"{result.stderr.strip()[-2000:]}"
            )
        _done = True
