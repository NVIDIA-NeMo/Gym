# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Auto-install GnuCOBOL (cobc) if not already available."""

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path


LOG = logging.getLogger(__name__)

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_PREFIX = _SCRIPT_DIR / ".gnucobol"
_INSTALL_SCRIPT = _SCRIPT_DIR / "scripts" / "install_gnucobol.sh"


def ensure_cobc() -> None:
    """Ensure the GnuCOBOL compiler (cobc) is available.

    Checks PATH first. If not found, installs automatically:
    - macOS: via Homebrew
    - Linux: builds from source using install_gnucobol.sh
    """
    if shutil.which("cobc"):
        LOG.info("cobc found: %s", shutil.which("cobc"))
        return

    LOG.warning("cobc not found on PATH — attempting auto-install")

    if sys.platform == "darwin":
        _install_macos()
    elif sys.platform == "linux":
        _install_linux()
    else:
        raise NotImplementedError(f"Unsupported platform: {sys.platform}")

    _verify_cobc()


def _install_macos() -> None:
    """Install GnuCOBOL via Homebrew on macOS."""
    if not shutil.which("brew"):
        raise RuntimeError(
            "Homebrew is required to auto-install GnuCOBOL on macOS. Install it from https://brew.sh/ and retry."
        )
    LOG.info("Installing GnuCOBOL via Homebrew ...")
    subprocess.run(["brew", "install", "gnu-cobol"], check=True)


def _install_linux() -> None:
    """Build GnuCOBOL from source on Linux (no root required)."""
    for tool in ("gcc", "make"):
        if not shutil.which(tool):
            raise RuntimeError(
                f"'{tool}' is required to build GnuCOBOL from source but was not found on PATH. Install it and retry."
            )

    prefix = os.environ.get("GNUCOBOL_PREFIX", str(_DEFAULT_PREFIX))
    LOG.info("Building GnuCOBOL from source into %s ...", prefix)

    env = os.environ.copy()
    env["GNUCOBOL_PREFIX"] = prefix

    subprocess.run(
        ["bash", str(_INSTALL_SCRIPT)],
        check=True,
        env=env,
    )

    # Add the newly built binaries/libraries to the current process env
    bin_dir = str(Path(prefix) / "bin")
    lib_dir = str(Path(prefix) / "lib")

    os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
    os.environ["LD_LIBRARY_PATH"] = lib_dir + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")


def _verify_cobc() -> None:
    """Verify that cobc is now callable."""
    cobc = shutil.which("cobc")
    if not cobc:
        raise RuntimeError("GnuCOBOL installation completed but 'cobc' is still not on PATH.")
    result = subprocess.run([cobc, "--version"], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"cobc --version failed (exit {result.returncode}): {result.stderr}")
    LOG.info("cobc ready: %s", result.stdout.splitlines()[0])
