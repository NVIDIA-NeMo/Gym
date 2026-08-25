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

import hashlib
import logging
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path


LOG = logging.getLogger(__name__)

HERMES_RELEASE = "v2026.8.19"
HERMES_VERSION = "0.20.5"
HERMES_COMMIT = "fcbd1076a93841fa88855acce810e342a5b78101"  # pragma: allowlist secret
HERMES_INSTALLER_SHA256 = (
    "0582d9b1562efcb6e0ac62f445102166"  # pragma: allowlist secret
    "7830b830a72ce7d91eaea9fee8b6c09b"  # pragma: allowlist secret
)

_INSTALL_ROOT = Path(
    os.environ.get(
        "NEMO_GYM_HERMES_INSTALL_ROOT",
        Path.home() / ".cache" / "nemo-gym" / "hermes-agent" / HERMES_VERSION,
    )
)
_INSTALL_DIR = _INSTALL_ROOT / "source"
_BOOTSTRAP_HOME = _INSTALL_ROOT / "bootstrap-home"
_INSTALLER = _INSTALL_ROOT / "install.sh"
_LOCK_SYNC_MARKER = _INSTALL_ROOT / f".lock-synced-{HERMES_COMMIT}"


def _is_ready() -> bool:
    python = _INSTALL_DIR / "venv" / "bin" / "python"
    if not python.is_file():
        return False

    revision = subprocess.run(
        ["git", "-C", str(_INSTALL_DIR), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if revision.returncode != 0 or revision.stdout.strip() != HERMES_COMMIT:
        return False

    installed_version = subprocess.run(
        [str(python), "-c", "from importlib.metadata import version; print(version('hermes-agent'))"],
        capture_output=True,
        text=True,
        check=False,
    )
    return installed_version.returncode == 0 and installed_version.stdout.strip() == HERMES_VERSION


def _download_installer() -> None:
    url = f"https://raw.githubusercontent.com/NousResearch/hermes-agent/{HERMES_COMMIT}/scripts/install.sh"
    with urllib.request.urlopen(url) as response:  # noqa: S310
        content = response.read()
    digest = hashlib.sha256(content).hexdigest()
    if digest != HERMES_INSTALLER_SHA256:
        raise RuntimeError(f"Hermes installer checksum mismatch: expected {HERMES_INSTALLER_SHA256}, got {digest}")
    _INSTALLER.write_bytes(content)
    _INSTALLER.chmod(0o755)


def _run_installer_stage(stage: str) -> None:
    env = os.environ.copy()
    env.update(
        {
            "HOME": str(_BOOTSTRAP_HOME),
            "HERMES_HOME": str(_BOOTSTRAP_HOME / ".hermes"),
        }
    )
    subprocess.run(
        [
            "bash",
            str(_INSTALLER),
            "--stage",
            stage,
            "--non-interactive",
            "--branch",
            HERMES_RELEASE,
            "--commit",
            HERMES_COMMIT,
            "--force-commit",
            "--dir",
            str(_INSTALL_DIR),
            "--hermes-home",
            str(_BOOTSTRAP_HOME / ".hermes"),
            "--skip-browser",
            "--skip-computer-use",
            "--no-skills",
        ],
        env=env,
        check=True,
    )


def _sync_locked_runtime() -> None:
    managed_uv = _BOOTSTRAP_HOME / ".hermes" / "bin" / "uv"
    uv = str(managed_uv) if managed_uv.is_file() else shutil.which("uv")
    if not uv:
        raise RuntimeError("Hermes installer completed without leaving uv available")

    env = os.environ.copy()
    env["UV_PROJECT_ENVIRONMENT"] = str(_INSTALL_DIR / "venv")
    subprocess.run(
        [uv, "sync", "--extra", "all", "--frozen"],
        cwd=_INSTALL_DIR,
        env=env,
        check=True,
    )
    _LOCK_SYNC_MARKER.write_text(HERMES_COMMIT + "\n", encoding="utf-8")


def ensure_hermes() -> Path:
    """Install the pinned upstream Hermes runtime and return its Python interpreter."""
    if sys.platform == "win32":
        raise RuntimeError("The managed Hermes installer currently requires macOS or Linux")

    _INSTALL_ROOT.mkdir(parents=True, exist_ok=True)
    _BOOTSTRAP_HOME.mkdir(parents=True, exist_ok=True)

    lock_path = _INSTALL_ROOT / ".install.lock"
    with lock_path.open("w") as lock:
        try:
            import fcntl

            fcntl.flock(lock, fcntl.LOCK_EX)
        except ImportError:
            pass

        if not _is_ready():
            LOG.info("installing Hermes Agent %s (%s)", HERMES_VERSION, HERMES_COMMIT)
            _download_installer()
            for stage in ("repository", "venv", "python-deps"):
                _run_installer_stage(stage)

        # The upstream installer intentionally falls back to a fresh resolve if
        # its locked sync fails. Reconcile with --frozen so Gym always runs the
        # dependency graph recorded by this exact release.
        if not _LOCK_SYNC_MARKER.is_file():
            _sync_locked_runtime()

        if not _is_ready() or not _LOCK_SYNC_MARKER.is_file():
            raise RuntimeError(f"Hermes Agent {HERMES_VERSION} installation did not produce a valid runtime")

    return _INSTALL_DIR / "venv" / "bin" / "python"
