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
"""Idempotent AppWorld setup, run once on resources-server startup.

Three things happen here, all skipped when their artifacts already exist:

1. **A dedicated venv for AppWorld.** ``appworld`` cannot share this server's
   venv: it pins ``python-multipart>=0.0.9,<0.0.10`` while nemo-gym requires
   ``>=0.0.22``, an unsatisfiable pair. That is fine, because nothing in this
   server imports ``appworld`` — episodes run in ``appworld serve environment``
   subprocesses (see worker_pool.py), so the package only has to be *runnable*,
   not importable. Isolating it also insulates gym from any future pin drift in
   a benchmark that depends on the same FastAPI stack gym does.
2. **``appworld install``** — unpacks AppWorld's encrypted ``.bundle`` files into
   that venv. Purely local: the decryption key is a constant in
   ``appworld.common.constants``, so there is no network call, key file or
   credential involved.
3. **``appworld download data``** — fetches the ~193 MB task/DB corpus into
   ``$APPWORLD_ROOT/data``.

AppWorld's apps, APIs and tasks are Apache 2.0 with the additional requirement
that public redistribution stay encrypted, so none of it is vendored into this
repo; it is fetched at setup time instead.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent

# Default root, gitignored. Holds ``data/`` (the downloaded corpus),
# ``experiments/outputs/`` (per-episode scratch DBs) and worker logs.
DEFAULT_APPWORLD_ROOT = _HERE / ".appworld_root"
# Default location of the isolated AppWorld venv, gitignored.
DEFAULT_APPWORLD_VENV = _HERE / ".appworld_venv"

# Pinned to the 0.1.x line: 0.1.3 is the release whose environment-server API
# (/initialize, /execute, /task_completed, /evaluate, /close) this server speaks.
APPWORLD_REQUIREMENT = "appworld>=0.1.3,<0.2"

_LOCK_TIMEOUT_SECS = 1800
_SUBPROCESS_TIMEOUT_SECS = 1800


@dataclass(frozen=True)
class AppWorldInstall:
    """Where AppWorld lives once setup has run."""

    root: str
    executable: str


def _run(args: list[str], cwd: str, root: str) -> None:
    env = os.environ.copy()
    env["APPWORLD_ROOT"] = root
    logger.info("appworld setup: %s", " ".join(args))
    result = subprocess.run(
        args,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=_SUBPROCESS_TIMEOUT_SECS,
    )
    if result.returncode != 0:
        raise RuntimeError(f"`{' '.join(args)}` failed (exit {result.returncode}):\n{result.stdout}\n{result.stderr}")


def venv_executable(venv_dir: Path, name: str) -> Path:
    return Path(venv_dir) / "bin" / name


def is_installed(venv_dir: Path) -> bool:
    """True once the venv exists and ``appworld install`` has unpacked the bundles.

    The import check mirrors upstream's own ``verify_fully_installed``: the app
    modules only exist after the encrypted bundles are unpacked.
    """
    python = venv_executable(venv_dir, "python")
    if not python.is_file() or not venv_executable(venv_dir, "appworld").is_file():
        return False
    probe = subprocess.run(
        [str(python), "-c", "import appworld; appworld.apps.api_docs"],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=120,
    )
    return probe.returncode == 0


def data_exists(root: str | os.PathLike[str]) -> bool:
    """True once ``appworld download data`` has populated ``<root>/data``."""
    return (Path(root) / "data" / "datasets" / "train.txt").is_file()


def _create_venv(venv_dir: Path) -> None:
    """Build the isolated venv and install AppWorld into it.

    ``--no-config`` is load-bearing, not tidiness. uv reads ``[tool.uv]`` from the
    nearest ``pyproject.toml``, and gym's lists ``exclude-dependencies`` (to trim
    mlflow's transitive tree) containing ``cryptography``, ``sqlalchemy`` and
    ``wcwidth`` — all of which AppWorld genuinely needs (bundle decryption, its
    databases, and ipython's prompt_toolkit). Installing from inside the repo
    without ``--no-config`` silently drops them and AppWorld fails to import.
    Gym's own constraints are irrelevant here anyway: this venv is deliberately
    independent of gym's dependency set.
    """
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    if shutil.which("uv"):
        _run(
            ["uv", "venv", "--no-config", "--allow-existing", "--python", python_version, str(venv_dir)],
            str(_HERE),
            str(_HERE),
        )
        _run(
            [
                "uv",
                "pip",
                "install",
                "--no-config",
                "--python",
                str(venv_executable(venv_dir, "python")),
                APPWORLD_REQUIREMENT,
            ],
            str(_HERE),
            str(_HERE),
        )
        return
    _run([sys.executable, "-m", "venv", str(venv_dir)], str(_HERE), str(_HERE))
    _run(
        [str(venv_executable(venv_dir, "python")), "-m", "pip", "install", APPWORLD_REQUIREMENT],
        str(_HERE),
        str(_HERE),
    )


def ensure_appworld(
    root: str | os.PathLike[str] | None = None,
    venv_dir: str | os.PathLike[str] | None = None,
) -> AppWorldInstall:
    """Make AppWorld runnable and its data present; return where both live.

    Also exports ``APPWORLD_ROOT`` for this process, so the worker subprocesses
    we spawn inherit the same data directory.
    """
    resolved_root = Path(root or os.environ.get("APPWORLD_ROOT") or DEFAULT_APPWORLD_ROOT).expanduser().resolve()
    resolved_venv = Path(venv_dir or os.environ.get("APPWORLD_VENV") or DEFAULT_APPWORLD_VENV).expanduser().resolve()
    resolved_root.mkdir(parents=True, exist_ok=True)
    os.environ["APPWORLD_ROOT"] = str(resolved_root)
    install = AppWorldInstall(
        root=str(resolved_root),
        executable=str(venv_executable(resolved_venv, "appworld")),
    )

    if is_installed(resolved_venv) and data_exists(resolved_root):
        logger.info("appworld already set up (venv=%s, root=%s)", resolved_venv, resolved_root)
        return install

    from filelock import FileLock  # noqa: PLC0415 — only needed on the slow path

    # Guards concurrent startups (several gym servers, or pytest-xdist workers)
    # sharing one root.
    with FileLock(str(resolved_root / ".nemo_gym_setup.lock"), timeout=_LOCK_TIMEOUT_SECS):
        if not is_installed(resolved_venv):
            if not venv_executable(resolved_venv, "appworld").is_file():
                _create_venv(resolved_venv)
            _run([install.executable, "install"], str(resolved_root), str(resolved_root))
            if not is_installed(resolved_venv):
                raise RuntimeError(
                    f"`appworld install` completed but {resolved_venv} still cannot import appworld.apps."
                )
        if not data_exists(resolved_root):
            _run([install.executable, "download", "data"], str(resolved_root), str(resolved_root))
            if not data_exists(resolved_root):
                raise RuntimeError(f"`appworld download data` completed but {resolved_root}/data is still empty.")

    logger.info("appworld ready (venv=%s, root=%s)", resolved_venv, resolved_root)
    return install


def load_task_ids(root: str | os.PathLike[str], split: str) -> list[str]:
    """Task ids for ``split`` (train | dev | test_normal | test_challenge).

    Reads the split file directly rather than importing ``appworld.load_task_ids``,
    which is not importable from this venv by design.
    """
    split_fpath = Path(root) / "data" / "datasets" / f"{split}.txt"
    if not split_fpath.is_file():
        raise FileNotFoundError(
            f"No AppWorld split file at {split_fpath}. Call ensure_appworld() first, "
            f"or run `appworld download data` with APPWORLD_ROOT={root}."
        )
    return [line.strip() for line in split_fpath.read_text().splitlines() if line.strip()]
