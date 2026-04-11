# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional per-``environment`` dependency install, invoked from ``app.py`` before the server imports heavy deps.

NeMo-Gym exports ``NEMO_GYM_CONFIG_DICT`` (merged YAML) and ``NEMO_GYM_CONFIG_PATH`` (this server's top-level
config key) before starting the process. We read ``environment`` from the matching server block and, if
``environments/<environment>/requirements.txt`` exists, run ``uv pip install --python <venv> -r …`` once per
changed file (tracked via a stamp under ``.venv``).
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import subprocess
import sys
from io import StringIO
from pathlib import Path
from typing import Any, Optional

from omegaconf import OmegaConf

from nemo_gym.global_config import NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, NEMO_GYM_CONFIG_PATH_ENV_VAR_NAME


logger = logging.getLogger(__name__)

_STAMP_FILENAME = ".cube_environment_extras.stamp"
_ENV_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")


def _cube_root() -> Path:
    return Path(__file__).resolve().parent


def _stamp_path() -> Path:
    """Stamp under the server venv root so it is per-server venv and not committed.

    Do not derive this from ``Path(sys.executable).resolve()``: the venv's ``python`` is often a
    symlink to the base interpreter; resolving would leave the venv and yield a path like
    ``/usr``, causing permission errors when writing the stamp.
    """
    base_prefix = getattr(sys, "base_prefix", sys.prefix)
    if sys.prefix != base_prefix:
        return Path(sys.prefix) / _STAMP_FILENAME
    cube_venv = _cube_root() / ".venv"
    if cube_venv.is_dir():
        return cube_venv / _STAMP_FILENAME
    return Path(sys.executable).parent.parent / _STAMP_FILENAME


def _requirements_fingerprint(req_file: Path) -> str:
    data = req_file.read_bytes()
    return hashlib.sha256(data).hexdigest()


def _parse_merged_config(yaml_blob: str) -> Any:
    return OmegaConf.load(StringIO(yaml_blob))


def _server_leaf_for_path(cfg: Any, top_key: str) -> Optional[dict[str, Any]]:
    """Match ``ng_run`` nesting: ``<top_key> -> <one> -> <one> -> { entrypoint, environment, ... }``."""
    branch = cfg.get(top_key) if hasattr(cfg, "get") else None
    if branch is None:
        return None
    for mid in branch.values() if hasattr(branch, "values") else []:
        if mid is None:
            continue
        for leaf in mid.values() if hasattr(mid, "values") else []:
            if leaf is not None and hasattr(leaf, "get") and leaf.get("entrypoint") is not None:
                plain = OmegaConf.to_container(leaf, resolve=True)
                return plain if isinstance(plain, dict) else None
    return None


def _resolve_environment(leaf: dict[str, Any]) -> Optional[str]:
    raw = leaf.get("environment", "osworld")
    if raw is None:
        return "osworld"
    if not isinstance(raw, str):
        return None
    if not _ENV_NAME_PATTERN.fullmatch(raw):
        logger.warning("cube bootstrap: ignoring unsafe environment name %r", raw)
        return None
    return raw


def _pip_install_requirements(req_file: Path, *, verbose: bool) -> None:
    """Install into the interpreter that is running this bootstrap (the server venv)."""
    exe = sys.executable
    uv = shutil.which("uv")
    cmd: list[str]
    if uv:
        cmd = [uv, "pip", "install", "--python", exe, "-r", str(req_file)]
        if verbose:
            cmd.insert(2, "-v")
    else:
        cmd = [exe, "-m", "pip", "install", "-r", str(req_file)]
        if verbose:
            cmd.append("-v")
        logger.warning("cube bootstrap: `uv` not on PATH; falling back to %s -m pip", exe)

    logger.info("cube bootstrap: installing environment extras: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def maybe_install_environment_extras() -> None:
    yaml_blob = os.environ.get(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME)
    path_key = os.environ.get(NEMO_GYM_CONFIG_PATH_ENV_VAR_NAME)
    if not yaml_blob or not path_key:
        logger.debug(
            "cube bootstrap: %s / %s unset — skipping extras (direct `python app.py` run).",
            NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME,
            NEMO_GYM_CONFIG_PATH_ENV_VAR_NAME,
        )
        return

    cfg = _parse_merged_config(yaml_blob)
    leaf = _server_leaf_for_path(cfg, path_key)
    if leaf is None:
        logger.warning("cube bootstrap: no server block under %r; skipping extras.", path_key)
        return

    environment = _resolve_environment(leaf)
    if not environment:
        return

    cube_root = _cube_root()
    req_file = cube_root / "environments" / environment / "requirements.txt"
    if not req_file.is_file():
        logger.debug("cube bootstrap: no extras file at %s; skipping.", req_file)
        return

    fingerprint = _requirements_fingerprint(req_file)
    stamp = _stamp_path()
    stamp_line = f"{environment}\n{fingerprint}\n"
    if stamp.is_file() and stamp.read_text() == stamp_line:
        logger.debug("cube bootstrap: extras unchanged (%s); skipping install.", environment)
        return

    verbose = os.environ.get("NEMO_GYM_CUBE_BOOTSTRAP_VERBOSE", "").lower() in ("1", "true", "yes")
    _pip_install_requirements(req_file, verbose=verbose)
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.write_text(stamp_line)
