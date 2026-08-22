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
import re
import shutil
import subprocess
import tarfile
import time
import urllib.request
from pathlib import Path


LOG = logging.getLogger(__name__)

# The Cline CLI is published to npm as `cline` and installs a `cline` binary. The version is pinned
# in the agent config (cline_version) and must also match an existing binary: task images can carry
# an older Cline on PATH, and silently reusing it invalidates the event-stream contract.
_CLINE_PKG = "cline"
_NODE_VERSION = "22.15.0"
_NODE_DIST_URL = f"https://nodejs.org/dist/v{_NODE_VERSION}/node-v{_NODE_VERSION}-linux-x64.tar.xz"
_LOCAL_PREFIX = Path(__file__).parent / ".cline_node"


def _cline_version(cline_bin: str) -> str | None:
    result = subprocess.run([cline_bin, "--version"], capture_output=True, text=True)
    if result.returncode != 0:
        return None
    match = re.search(r"\b\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?\b", result.stdout + result.stderr)
    return match.group(0) if match else None


def _version_matches(cline_bin: str, requested: str | None) -> bool:
    return requested is None or _cline_version(cline_bin) == requested


def _npm_install(npm_bin: str, version: str | None) -> None:
    pkg = f"{_CLINE_PKG}@{version}" if version else f"{_CLINE_PKG}@latest"
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


def ensure_cline(version: str | None = None) -> None:
    """Ensure ``cline`` is on PATH at the requested version, installing it if necessary."""
    existing = shutil.which("cline")
    if existing and _version_matches(existing, version):
        return
    if existing:
        LOG.info(
            "cline at %s is version %s; installing requested version %s", existing, _cline_version(existing), version
        )

    # npm installs the binary here but may not have it on PATH in a fresh shell; add it and reuse.
    local_bin = Path.home() / ".local" / "bin"
    local_cline = local_bin / "cline"
    if local_cline.is_file() and _version_matches(str(local_cline), version):
        os.environ["PATH"] = str(local_bin) + os.pathsep + os.environ.get("PATH", "")
        return

    npm = shutil.which("npm")
    if npm:
        LOG.info("installing cline via system npm (%s)", npm)
        _npm_install(npm, version)
    else:
        LOG.info("npm not found; installing local Node.js")
        bin_dir = _install_node_locally()
        os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
        npm = shutil.which("npm")
        if not npm:
            raise RuntimeError(f"npm not found after local Node.js install in {bin_dir}")
        _npm_install(npm, version)

    # Put the install target ahead of any stale image-provided binary before validating the pin.
    npm_bin_dir = _npm_global_bin(shutil.which("npm") or "npm")
    if npm_bin_dir and Path(npm_bin_dir).is_dir():
        os.environ["PATH"] = npm_bin_dir + os.pathsep + os.environ.get("PATH", "")

    if not shutil.which("cline") and (local_bin / "cline").is_file():
        os.environ["PATH"] = str(local_bin) + os.pathsep + os.environ.get("PATH", "")

    installed = shutil.which("cline")
    if not installed:
        raise RuntimeError("cline install appeared to succeed but 'cline' is still not on PATH")
    installed_version = _cline_version(installed)
    if version and installed_version != version:
        raise RuntimeError(
            f"cline install appeared to succeed, but {installed} reports {installed_version!r}; expected {version!r}"
        )

    LOG.info("cline %s is ready at %s", installed_version or "unknown", installed)
