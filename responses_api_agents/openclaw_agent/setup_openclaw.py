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

_OPENCLAW_PKG = "openclaw"
# openclaw requires Node >= 22.19; the system Node may be older (the cluster had 22.15), so install a
# local Node and put it first on PATH when needed. version-keyed prefix so bumping re-installs cleanly.
_NODE_VERSION = "22.20.0"
_MIN_NODE = (22, 19)
_NODE_DIST_URL = f"https://nodejs.org/dist/v{_NODE_VERSION}/node-v{_NODE_VERSION}-linux-x64.tar.xz"
_LOCAL_PREFIX = Path(__file__).parent / f".openclaw_node_{_NODE_VERSION}"


def _npm_install(npm_bin: str, version: str | None) -> None:
    pkg = f"{_OPENCLAW_PKG}@{version}" if version else f"{_OPENCLAW_PKG}@latest"
    subprocess.run([npm_bin, "install", "-g", pkg], check=True)


def _node_ok() -> bool:
    """True if a Node.js >= _MIN_NODE is on PATH (openclaw refuses to run on older)."""
    node = shutil.which("node")
    if not node:
        return False
    try:
        ver = subprocess.run([node, "--version"], capture_output=True, text=True).stdout.strip().lstrip("v")
        return tuple(int(x) for x in ver.split(".")[:2]) >= _MIN_NODE
    except Exception:
        return False


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


def ensure_openclaw(version: str | None = None) -> None:
    """Ensure ``openclaw`` is on PATH and runs against Node >= 22.19, installing both if necessary."""
    if shutil.which("openclaw") and _node_ok():
        return

    # openclaw needs Node >= 22.19. if the system Node is older or missing, install a local Node and
    # put it first on PATH so both the npm install and the openclaw runtime use it (not the old system
    # Node). a too-old system Node was why openclaw exited 1 with no model call on the cluster.
    if not _node_ok():
        LOG.info("Node.js < %s on PATH; installing local Node.js %s for openclaw", _MIN_NODE, _NODE_VERSION)
        bin_dir = _install_node_locally()
        os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")

    npm = shutil.which("npm")
    if not npm:
        raise RuntimeError("npm not found after ensuring Node.js for openclaw")
    LOG.info("installing openclaw via npm (%s)", npm)
    _npm_install(npm, version)

    # npm install -g may put the binary in a prefix not yet on PATH
    if not shutil.which("openclaw"):
        npm_bin_dir = subprocess.run(
            [shutil.which("npm") or "npm", "bin", "-g"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        if npm_bin_dir and Path(npm_bin_dir).is_dir():
            os.environ["PATH"] = npm_bin_dir + os.pathsep + os.environ.get("PATH", "")

    local_bin = Path.home() / ".local" / "bin"
    if not shutil.which("openclaw") and (local_bin / "openclaw").is_file():
        os.environ["PATH"] = str(local_bin) + os.pathsep + os.environ.get("PATH", "")

    if not shutil.which("openclaw"):
        raise RuntimeError("openclaw install appeared to succeed but 'openclaw' is still not on PATH")

    LOG.info("openclaw is ready at %s", shutil.which("openclaw"))
