# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Admit the exact headed Chromium expected by the selected Playwright driver."""

from __future__ import annotations

import fcntl
import logging
import os
import subprocess
import sys
import time
from pathlib import Path


LOG = logging.getLogger(__name__)


def _chromium_executable() -> Path:
    # Called on a setup thread, never inside the server's async event loop.
    # No browser is launched and PLAYWRIGHT_BROWSERS_PATH=0 is handled by the
    # driver's own resolver, not a guessed cache directory.
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        return Path(playwright.chromium.executable_path)


def ensure_chromium(*, channel: str | None = None, allow_install: bool = True, timeout_seconds: float = 600) -> None:
    """Check the driver-selected full browser; serialize missing-build installs.

    Named system-browser channels are externally provisioned. Headless shell
    is not a substitute for this runtime's Xvfb/PyAutoGUI desktop browser.
    Offline deployments set allow_install=False and prepopulate the cache.
    """

    if channel not in (None, "chromium"):
        return
    executable = _chromium_executable()

    def present() -> bool:
        return executable.is_file() and os.access(executable, os.X_OK)

    if present():
        return
    if not allow_install:
        raise RuntimeError(
            f"required Chromium is missing: {executable}; preinstall with python -m playwright install chromium"
        )
    # The cache root is derived from the expected revision directory. macOS
    # bundles have additional nested directories, so find that revision rather
    # than assuming a fixed number of parent levels.
    revision_dir = next((p for p in executable.parents if p.name.startswith("chromium-")), None)
    if revision_dir is None:
        raise RuntimeError(f"cannot identify Playwright Chromium cache root for {executable}")
    cache = revision_dir.parent
    deadline = time.monotonic() + timeout_seconds
    try:
        cache.mkdir(parents=True, exist_ok=True)
        with (cache / ".gym-chromium-install.lock").open("a") as lock:
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise RuntimeError(f"timed out waiting for Chromium install lock at {cache}") from None
                    time.sleep(0.1)
            if present():
                return
            LOG.info("Installing driver-matched Chromium at %s", executable)
            result = subprocess.run(
                [sys.executable, "-m", "playwright", "install", "chromium"],
                capture_output=True,
                text=True,
                errors="replace",
                timeout=max(0.01, deadline - time.monotonic()),
            )
            if result.returncode or not present():
                raise RuntimeError(
                    f"Chromium install failed: exit={result.returncode}, expected={executable}; "
                    f"stderr={result.stderr[-2000:]}"
                )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(
            f"cannot prepare Chromium at {executable}: {type(exc).__name__}; preinstall the browser cache"
        ) from exc
