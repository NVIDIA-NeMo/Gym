# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Chromium's system libraries, which `playwright install` does not provide.

The arm64 gym-runtime image runs as root but lacks libnss3 and the other libraries Chromium needs, so Playwright
refuses to launch it there. Where this process is root and has apt-get, install them with
`playwright install-deps chromium`; elsewhere say what to run.
"""

import logging
import os
import shutil
import subprocess
import sys
from typing import Optional


LOGGER = logging.getLogger(__name__)

_INSTALL_DEPS_TIMEOUT_S = 1800
_LAUNCH_TIMEOUT_S = 120
# A subprocess, because the sync API refuses to run inside the server's event loop.
_LAUNCH_PROBE = (
    "from playwright.sync_api import sync_playwright\nwith sync_playwright() as p:\n    p.chromium.launch().close()\n"
)


def missing_host_dependencies() -> Optional[str]:
    """Playwright's report when Chromium cannot launch for want of system libraries; None when it launches."""
    result = subprocess.run(
        [sys.executable, "-c", _LAUNCH_PROBE],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=_LAUNCH_TIMEOUT_S,
    )
    if result.returncode == 0:
        return None
    if "missing dependencies" in result.stderr or "error while loading shared libraries" in result.stderr:
        return result.stderr.strip()[-2000:]
    raise RuntimeError(f"Playwright Chromium does not launch:\n{result.stderr.strip()[-2000:]}")


def ensure_chromium_host_libraries() -> None:
    """Install Chromium's missing system libraries when this process may; raise when it may not or it fails."""
    report = missing_host_dependencies()
    if report is None:
        return
    if os.geteuid() != 0 or shutil.which("apt-get") is None:
        raise RuntimeError(
            "Chromium lacks system libraries. Install them as root with "
            f"`python -m playwright install-deps chromium`.\n{report}"
        )
    LOGGER.info("Chromium lacks system libraries; installing them with `playwright install-deps chromium`")
    result = subprocess.run(
        [sys.executable, "-m", "playwright", "install-deps", "chromium"],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=_INSTALL_DEPS_TIMEOUT_S,
    )
    still_missing = missing_host_dependencies()
    if result.returncode != 0 or still_missing is not None:
        raise RuntimeError(
            f"`playwright install-deps chromium` exited {result.returncode} and Chromium still cannot launch.\n"
            f"{result.stderr.strip()[-2000:]}\n{still_missing or ''}"
        )
