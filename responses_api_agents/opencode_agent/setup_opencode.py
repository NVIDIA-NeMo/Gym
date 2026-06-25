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
import urllib.request
from pathlib import Path


LOG = logging.getLogger(__name__)

_OPENCODE_PKG = "opencode-ai"
_NODE_VERSION = "22.15.0"
_NODE_DIST_URL = f"https://nodejs.org/dist/v{_NODE_VERSION}/node-v{_NODE_VERSION}-linux-x64.tar.xz"
_LOCAL_PREFIX = Path(__file__).parent / ".opencode_node"


def _npm_install(npm_bin: str, version: str | None) -> None:
    pkg = f"{_OPENCODE_PKG}@{version}" if version else f"{_OPENCODE_PKG}@latest"
    subprocess.run([npm_bin, "install", "-g", pkg], check=True)


def _install_node_locally() -> Path:
    node_bin = _LOCAL_PREFIX / "bin" / "node"
    if node_bin.is_file():
        return _LOCAL_PREFIX / "bin"

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
    return _LOCAL_PREFIX / "bin"


def ensure_opencode(version: str | None = None) -> None:
    """Ensure ``opencode`` is on PATH, installing it via npm if necessary."""
    if shutil.which("opencode"):
        return

    # Check ~/.local/bin
    local_bin = Path.home() / ".local" / "bin"
    if (local_bin / "opencode").is_file():
        os.environ["PATH"] = str(local_bin) + os.pathsep + os.environ.get("PATH", "")
        return

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
        npm_bin_dir = subprocess.run(
            [shutil.which("npm") or "npm", "bin", "-g"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        if npm_bin_dir and Path(npm_bin_dir).is_dir():
            os.environ["PATH"] = npm_bin_dir + os.pathsep + os.environ.get("PATH", "")

    if not shutil.which("opencode") and (local_bin / "opencode").is_file():
        os.environ["PATH"] = str(local_bin) + os.pathsep + os.environ.get("PATH", "")

    if not shutil.which("opencode"):
        raise RuntimeError("opencode install appeared to succeed but 'opencode' is still not on PATH")

    LOG.info("opencode is ready at %s", shutil.which("opencode"))
