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
"""Install the Chromium build the pinned Playwright drives, once per process."""

import logging
import subprocess
import sys


logger = logging.getLogger(__name__)

_INSTALL_TIMEOUT_S = 600
_installed = False


def ensure_chromium() -> None:
    """Run ``playwright install chromium``, which is a fast no-op when the build is present.

    The Playwright wheel ships the driver but not the browser, and each driver version needs
    its own browser build, so a fresh server venv would otherwise fail at its first rollout.
    On Linux the browser also needs system libraries; ``playwright install-deps chromium``
    (root required) installs them.
    """
    global _installed
    if _installed:
        return
    result = subprocess.run(
        [sys.executable, "-m", "playwright", "install", "chromium"],
        capture_output=True,
        timeout=_INSTALL_TIMEOUT_S,
    )
    if result.returncode != 0:
        output = (result.stderr or result.stdout).decode(errors="replace").strip()
        raise RuntimeError(f"`playwright install chromium` failed (exit {result.returncode}): {output[-2000:]}")
    _installed = True
