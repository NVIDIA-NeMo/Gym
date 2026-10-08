# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Test fixtures, including a real CDP endpoint so the remote backend is tested.

The endpoint is a plain Chromium started with `--remote-debugging-port` — the
same thing a browser container exposes — so `RemoteCDPBackend` is exercised in
CI without any third-party service or SDK.
"""

import shutil
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request

import pytest


def pytest_configure(config):
    """Install the browser once, before any test tries to launch or spawn one."""
    from browser._chromium import ensure_chromium

    ensure_chromium()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_until_ready(url: str, timeout_s: float = 30.0, proc=None) -> None:
    """Poll the CDP version endpoint until the browser answers.

    The last connection error is carried into the failure: without it a browser
    that died at startup reads as a bare "connection refused" 30 seconds later,
    which says nothing about why.
    """
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/json/version", timeout=2) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, OSError) as exc:  # not up yet
            last_error = exc
        time.sleep(0.2)
    detail = ""
    if proc is not None and proc.poll() is not None:
        stderr = (proc.stderr.read() if proc.stderr else "") or ""
        detail = f"; browser exited {proc.returncode}: {stderr.strip()[-2000:]}"
    raise RuntimeError(f"CDP endpoint {url} never became ready within {timeout_s:.0f}s: {last_error!r}{detail}")


@pytest.fixture(scope="session")
def cdp_endpoint():
    """A local Chromium listening for CDP, as `http://127.0.0.1:<port>`.

    Spawns the headless shell, not `chromium.executable_path`. That path is the
    full browser, which on a CI runner needs user namespaces AppArmor restricts
    and libraries the image does not carry; the shell is the binary
    `chromium.launch()` uses, and every other test here proves it runs on the
    runner. `--no-sandbox` for the same reason Playwright passes it there.

    Spawned rather than launched through `sync_playwright()`: this fixture is
    session-scoped and the async tests run inside a loop the sync API refuses to
    share.
    """
    shell = _headless_shell_path()
    port = _free_port()
    profile_dir = tempfile.mkdtemp(prefix="interactive-browser-cdp-")
    proc = subprocess.Popen(
        [
            shell,
            f"--remote-debugging-port={port}",
            f"--user-data-dir={profile_dir}",
            "--remote-allow-origins=*",
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-gpu",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    url = f"http://127.0.0.1:{port}"
    try:
        _wait_until_ready(url, proc=proc)
        yield url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        shutil.rmtree(profile_dir, ignore_errors=True)


def _headless_shell_path() -> str:
    """The headless shell inside the Playwright browser cache."""
    from browser._chromium import _browsers_dir, ensure_chromium

    ensure_chromium()
    for d in sorted(_browsers_dir().glob("chromium_headless_shell-*"), reverse=True):
        for name in ("chrome-headless-shell", "headless_shell"):
            found = next(d.rglob(name), None)
            if found is not None:
                return str(found)
    raise RuntimeError(f"No headless shell under {_browsers_dir()}; run `playwright install chromium`.")
