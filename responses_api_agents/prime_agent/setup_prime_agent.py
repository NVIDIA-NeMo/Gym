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
import shutil
import subprocess
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path


LOG = logging.getLogger(__name__)

_INSTALL_URL = "https://app.primeintellect.ai/prime-agent/install.sh"
_NODE_VERSION = "22.15.0"
_NODE_DIST_URL = f"https://nodejs.org/dist/v{_NODE_VERSION}/node-v{_NODE_VERSION}-linux-x64.tar.xz"
_LOCAL_PREFIX = Path(__file__).parent / ".prime_agent_node"


def _npm_global_bin(npm_bin: str) -> str | None:
    prefix = subprocess.run([npm_bin, "prefix", "-g"], capture_output=True, text=True).stdout.strip()
    return str(Path(prefix) / "bin") if prefix else None


def _install_node_locally() -> Path:
    node_bin = _LOCAL_PREFIX / "bin" / "node"
    if node_bin.is_file():
        return _LOCAL_PREFIX / "bin"

    _LOCAL_PREFIX.mkdir(parents=True, exist_ok=True)
    tarball = _LOCAL_PREFIX / "node.tar.xz"
    LOG.info("downloading Node.js %s", _NODE_VERSION)
    urllib.request.urlretrieve(_NODE_DIST_URL, tarball)  # noqa: S310

    with tarfile.open(tarball, "r:xz") as archive:
        archive.extractall(_LOCAL_PREFIX, filter="data")

    nested = next(path for path in _LOCAL_PREFIX.iterdir() if path.is_dir() and path.name.startswith("node-"))
    for item in nested.iterdir():
        item.rename(_LOCAL_PREFIX / item.name)
    nested.rmdir()
    tarball.unlink(missing_ok=True)
    return _LOCAL_PREFIX / "bin"


def _run_installer(version: str | None) -> None:
    with tempfile.TemporaryDirectory(prefix="prime-agent-install-") as temp_dir:
        script = Path(temp_dir) / "install.sh"
        subprocess.run(["curl", "-fsSL", _INSTALL_URL, "-o", str(script)], check=True)
        env = {
            **os.environ,
            "PRIME_AGENT_INSTALLER_PLAIN": "1",
            "PRIME_AGENT_BOOTSTRAP_KERNEL_ON_INSTALL": "0",
        }
        cmd = ["sh", str(script)]
        if version:
            cmd.append(version)
        subprocess.run(cmd, check=True, env=env, stdin=subprocess.DEVNULL, start_new_session=True)


def _verify_version(command: str, expected: str | None) -> None:
    if expected is None:
        return
    result = subprocess.run([command, "--version"], check=True, capture_output=True, text=True)
    actual = (result.stdout or result.stderr).strip().removeprefix("v")
    if actual != expected.removeprefix("v"):
        raise RuntimeError(f"Prime Agent version {actual!r} does not match configured version {expected!r}")


def ensure_prime_agent(version: str | None = None) -> None:
    command = shutil.which("prime-agent")
    if command:
        _verify_version(command, version)
        return

    local_bin = Path.home() / ".local" / "bin"
    if (local_bin / "prime-agent").is_file():
        os.environ["PATH"] = str(local_bin) + os.pathsep + os.environ.get("PATH", "")
        _verify_version(str(local_bin / "prime-agent"), version)
        return

    if shutil.which("npm") is None:
        LOG.info("npm not found. Installing local Node.js")
        bin_dir = _install_node_locally()
        os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")

    for attempt in range(1, 4):
        try:
            LOG.info("installing Prime Agent with the official installer")
            _run_installer(version)
            break
        except (OSError, subprocess.CalledProcessError):
            if attempt == 3:
                raise
            LOG.warning("Prime Agent install failed (attempt %d/3), retrying", attempt)
            time.sleep(2 * attempt)

    if not shutil.which("prime-agent"):
        npm_bin_dir = _npm_global_bin(shutil.which("npm") or "npm")
        if npm_bin_dir and Path(npm_bin_dir).is_dir():
            os.environ["PATH"] = npm_bin_dir + os.pathsep + os.environ.get("PATH", "")

    if not shutil.which("prime-agent") and (local_bin / "prime-agent").is_file():
        os.environ["PATH"] = str(local_bin) + os.pathsep + os.environ.get("PATH", "")

    if not shutil.which("prime-agent"):
        raise RuntimeError("Prime Agent install appeared to succeed but 'prime-agent' is still not on PATH")

    command = shutil.which("prime-agent")
    assert command is not None
    _verify_version(command, version)
    LOG.info("Prime Agent is ready at %s", command)
