# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Auto-install Playwright Chromium browser on server startup.
Called in model_post_init() of the resource server.
"""

import logging
import shutil
import subprocess
import sys


logger = logging.getLogger(__name__)


def _chromium_already_installed() -> bool:
    """Check if Playwright Chromium is already downloaded."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "playwright", "install", "--dry-run", "chromium"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and "chromium" not in result.stdout.lower():
            return True
    except Exception:
        pass

    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            path = p.chromium.executable_path
            if path and shutil.which(path):
                return True
    except Exception:
        pass

    return False


def ensure_playwright():
    """Ensure Playwright and Chromium browser are installed. Skips if already present."""
    try:
        import playwright  # noqa: F401
    except ImportError:
        logger.info("Installing playwright package...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "playwright"], stdout=subprocess.DEVNULL)

    if _chromium_already_installed():
        logger.info("Playwright Chromium already installed — skipping download")
        return

    logger.info("Installing Playwright Chromium browser...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "playwright", "install", "chromium", "--with-deps"],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            logger.warning(f"Playwright install with --with-deps failed: {result.stderr}")
            subprocess.run(
                [sys.executable, "-m", "playwright", "install", "chromium"],
                capture_output=True,
                text=True,
                timeout=600,
            )
        logger.info("Playwright Chromium installed successfully")
    except subprocess.TimeoutExpired:
        logger.warning("Playwright install timed out after 600s")
    except Exception as e:
        logger.warning(f"Playwright install failed: {e}")
