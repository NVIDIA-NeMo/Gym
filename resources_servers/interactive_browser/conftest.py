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


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_until_ready(url: str, timeout_s: float = 30.0) -> None:
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
    raise RuntimeError(f"CDP endpoint {url} never became ready: {last_error!r}")


@pytest.fixture(scope="session")
def cdp_endpoint():
    """A local Chromium listening for CDP, as `http://127.0.0.1:<port>`."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        executable = p.chromium.executable_path

    port = _free_port()
    profile_dir = tempfile.mkdtemp(prefix="interactive-browser-cdp-")
    proc = subprocess.Popen(
        [
            executable,
            "--headless=new",
            f"--remote-debugging-port={port}",
            f"--user-data-dir={profile_dir}",
            "--remote-allow-origins=*",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-gpu",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}"
    try:
        _wait_until_ready(url)
        yield url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        shutil.rmtree(profile_dir, ignore_errors=True)
