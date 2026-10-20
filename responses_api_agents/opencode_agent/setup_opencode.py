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

import logging
import os
import platform
import shutil
import subprocess
import tarfile
import time
import urllib.request
from pathlib import Path

from nemo_gym.agent_utils.install_lock import adopt_recorded_install, install_lock, record_install
from responses_api_agents.opencode_agent.runtime import OPENCODE_VERSION


LOG = logging.getLogger(__name__)

_OPENCODE_PKG = "opencode-ai"
_NODE_VERSION = "22.15.0"
_NODE_ARCH = "arm64" if platform.machine().lower() in ("aarch64", "arm64") else "x64"
_NODE_DIST_URL = f"https://nodejs.org/dist/v{_NODE_VERSION}/node-v{_NODE_VERSION}-linux-{_NODE_ARCH}.tar.xz"
_LOCAL_PREFIX = Path(__file__).parent / ".opencode_node"
# Beside the toolchain directory rather than in it, because an interrupted install's cleanup removes that directory.
_INSTALL_LOCK = _LOCAL_PREFIX.with_name(_LOCAL_PREFIX.name + ".lock")
_INSTALL_RECORD = _LOCAL_PREFIX.with_name(_LOCAL_PREFIX.name + ".installed")


def _npm_install(npm_bin: str, version: str | None) -> None:
    pkg = f"{_OPENCODE_PKG}@{version}" if version else f"{_OPENCODE_PKG}@latest"
    for attempt in range(1, 4):
        try:
            subprocess.run([npm_bin, "install", "-g", pkg], check=True)
            return
        except subprocess.CalledProcessError:
            if attempt == 3:
                raise
            LOG.warning("npm install %s failed (attempt %d/3), retrying", pkg, attempt)
            time.sleep(2 * attempt)


def _npm_global_bin(npm_bin: str) -> str | None:
    prefix = subprocess.run([npm_bin, "prefix", "-g"], capture_output=True, text=True).stdout.strip()
    return str(Path(prefix) / "bin") if prefix else None


def _install_node_locally() -> Path:
    complete = _LOCAL_PREFIX / ".complete"
    if complete.is_file():
        return _LOCAL_PREFIX / "bin"

    # The marker is written last, so a toolchain without it is left from an interrupted install.
    shutil.rmtree(_LOCAL_PREFIX, ignore_errors=True)
    _LOCAL_PREFIX.mkdir(parents=True, exist_ok=True)
    tarball = _LOCAL_PREFIX / "node.tar.xz"

    LOG.info("downloading Node.js %s", _NODE_VERSION)
    urllib.request.urlretrieve(_NODE_DIST_URL, tarball)  # noqa: S310

    with tarfile.open(tarball, "r:xz") as tf:
        tf.extractall(_LOCAL_PREFIX, filter="data")

    nested = next(p for p in _LOCAL_PREFIX.iterdir() if p.is_dir() and p.name.startswith("node-"))
    for item in nested.iterdir():
        item.rename(_LOCAL_PREFIX / item.name)
    nested.rmdir()
    tarball.unlink(missing_ok=True)
    complete.touch()
    return _LOCAL_PREFIX / "bin"


def installed_opencode_version() -> str | None:
    """Return the ``opencode --version`` output, or ``None`` when it cannot be determined."""
    exe = shutil.which("opencode")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=60)  # noqa: S603
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    lines = (out.stdout.strip() or out.stderr.strip()).splitlines()
    return lines[-1].strip() if lines else None


def _require_version(version: str) -> None:
    found = installed_opencode_version()
    if found != version:
        raise RuntimeError(f"Expected OpenCode {version}, found {found!r}; install opencode-ai@{version}")


def ensure_opencode(version: str = OPENCODE_VERSION) -> None:
    """Install OpenCode if absent and require the configured release."""
    if shutil.which("opencode"):
        _require_version(version)
        return

    # Check ~/.local/bin
    local_bin = Path.home() / ".local" / "bin"
    if (local_bin / "opencode").is_file():
        os.environ["PATH"] = str(local_bin) + os.pathsep + os.environ.get("PATH", "")
        _require_version(version)
        return

    # With num_workers > 1, every worker installs on its own first local call, and those calls can arrive together.
    # One worker installs while the others wait here, then they reuse the install it recorded.
    with install_lock(_INSTALL_LOCK):
        if not adopt_recorded_install(_INSTALL_RECORD, "opencode"):
            _install_opencode(version, local_bin)
            record_install(_INSTALL_RECORD, Path(shutil.which("opencode")).parent)

    _require_version(version)
    LOG.info("opencode is ready at %s", shutil.which("opencode"))


def _install_opencode(version: str, local_bin: Path) -> None:
    npm = shutil.which("npm")
    if npm:
        LOG.info("installing opencode via system npm (%s)", npm)
        _npm_install(npm, version)
    else:
        LOG.info("npm not found; installing local Node.js")
        bin_dir = _install_node_locally()
        os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
        npm = shutil.which("npm")
        if not npm:
            raise RuntimeError(f"npm not found after local Node.js install in {bin_dir}")
        _npm_install(npm, version)

    if not shutil.which("opencode"):
        npm_bin_dir = _npm_global_bin(shutil.which("npm") or "npm")
        if npm_bin_dir and Path(npm_bin_dir).is_dir():
            os.environ["PATH"] = npm_bin_dir + os.pathsep + os.environ.get("PATH", "")

    if not shutil.which("opencode") and (local_bin / "opencode").is_file():
        os.environ["PATH"] = str(local_bin) + os.pathsep + os.environ.get("PATH", "")

    if not shutil.which("opencode"):
        raise RuntimeError("opencode install appeared to succeed but 'opencode' is still not on PATH")
