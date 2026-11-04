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
"""Access to SWE-rebench's own test-log parsers.

Each dataset row names the parser that reads its test output (``install_config.log_parser``),
and the dataset spans 34 of them across 20 languages. Reimplementing that set here would be a
standing correctness risk: a parser that drifts from upstream silently misreads test results and
turns a passing patch into a failure.

So the upstream module is fetched and used directly. It is MIT licensed. It is fetched rather
than vendored because it is ~3.5k lines of third-party code whose only stable contract is the
``NAME_TO_PARSER`` mapping, and a stale copy is worse than an explicit dependency.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Callable


UPSTREAM_REPO = "https://github.com/SWE-rebench/SWE-rebench-V2.git"
# Pinned so a parser change upstream cannot silently move scores between runs. Bump
# deliberately, and re-baseline when you do.
UPSTREAM_COMMIT = "main"

# Upstream has moved this file between releases; accept both layouts rather than pinning one.
_PARSER_RELATIVE_PATHS = (Path("lib") / "agent" / "log_parsers.py", Path("agent") / "log_parsers.py")

_MODULE: ModuleType | None = None


def _clone_locked(destination: Path, timeout_s: float = 600.0) -> None:
    """Clone once across concurrent workers, using a mkdir lock.

    ``gym env start`` may bring several server workers up at once, and an unsynchronised clone
    into a shared path yields a half-written tree that imports with a confusing error.
    """
    if _parser_path(destination) is not None:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.parent / f".{destination.name}.clone.lockdir"
    waited = 0.0
    while True:
        try:
            lock_path.mkdir()
            break
        except FileExistsError:
            if _parser_path(destination) is not None:
                return
            if waited > timeout_s:
                raise TimeoutError(f"timed out waiting for the SWE-rebench clone lock at {lock_path}")
            time.sleep(2.0)
            waited += 2.0
    try:
        if _parser_path(destination) is not None:
            return
        shutil.rmtree(destination, ignore_errors=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", UPSTREAM_REPO, str(destination)],
            check=True,
            capture_output=True,
        )
        if UPSTREAM_COMMIT != "main":
            subprocess.run(["git", "fetch", "--depth", "1", "origin", UPSTREAM_COMMIT], cwd=destination, check=True)
            subprocess.run(["git", "checkout", UPSTREAM_COMMIT], cwd=destination, check=True)
    finally:
        shutil.rmtree(lock_path, ignore_errors=True)


def _parser_path(repo_dir: Path) -> Path | None:
    for relative in _PARSER_RELATIVE_PATHS:
        candidate = repo_dir / relative
        if candidate.exists():
            return candidate
    return None


def load_parsers(cache_dir: Path) -> ModuleType:
    """Import upstream's ``log_parsers`` module, cloning it on first use."""
    global _MODULE
    if _MODULE is not None:
        return _MODULE

    repo_dir = Path(cache_dir) / "SWE-rebench-V2"
    _clone_locked(repo_dir)
    parser_path = _parser_path(repo_dir)
    if parser_path is None:
        raise FileNotFoundError(f"SWE-rebench log_parsers.py not found under {repo_dir}; upstream layout changed")

    # The module does `from lib.agent...`, so the repo root has to be importable.
    added = [p for p in (str(repo_dir), str(repo_dir / "lib")) if p not in sys.path]
    sys.path[:0] = added
    try:
        spec = importlib.util.spec_from_file_location("_swe_rebench_log_parsers", str(parser_path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for path in added:
            try:
                sys.path.remove(path)
            except ValueError:
                pass
    _MODULE = module
    return module


def resolve_parser(name: str, cache_dir: Path) -> Callable[[str], dict[str, str]]:
    """Return the named parser.

    Raises rather than falling back to a default: a wrong parser produces a plausible-looking
    empty result, which grades as "no tests passed" and is indistinguishable from a real
    failure.
    """
    module = load_parsers(cache_dir)
    parser = (getattr(module, "NAME_TO_PARSER", {}) or {}).get(name) or getattr(module, name, None)
    if parser is None:
        available = sorted(getattr(module, "NAME_TO_PARSER", {}) or {})
        raise KeyError(f"unknown SWE-rebench log parser {name!r}; upstream exposes {len(available)}: {available[:8]}…")
    return parser


def default_cache_dir() -> Path:
    return Path(os.environ.get("NEMO_GYM_CACHE_DIR") or (Path.home() / ".cache" / "nemo_gym")) / "swe_rebench"
