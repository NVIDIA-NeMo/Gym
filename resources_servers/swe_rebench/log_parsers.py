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
# Pinned so a parser change upstream cannot silently move scores between runs.
# Bump it deliberately, and re-baseline when you do.
# c71902a8 (2026-03-12) is the tree the swe_agents harness graded with.
UPSTREAM_COMMIT = "c71902a8cf8d2b725f63d51f199f4d3e56f68d2d"  # pragma: allowlist secret

# Upstream has moved this file between releases; accept both layouts rather than pinning one.
_PARSER_RELATIVE_PATHS = (Path("lib") / "agent" / "log_parsers.py", Path("agent") / "log_parsers.py")

_MODULE: ModuleType | None = None


def _clone_locked(destination: Path, timeout_s: float = 600.0) -> None:
    """Clone once across concurrent workers, using a mkdir lock.

    ``gym env start`` may bring several server workers up at once.
    An unsynchronised clone into a shared path would yield a half-written tree that imports with a confusing error.
    The tree is built in a private staging directory and published with one rename.
    A parser file under ``destination`` therefore always belongs to a complete checkout of ``UPSTREAM_COMMIT``.
    A worker killed while it holds the lock leaves the lock behind, and later callers time out naming it.
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
                raise TimeoutError(
                    f"timed out after {timeout_s:.0f}s waiting for the SWE-rebench clone lock at {lock_path}. "
                    "A worker killed while fetching leaves this lock behind; "
                    "if no other worker is fetching, remove it and retry."
                )
            time.sleep(2.0)
            waited += 2.0
    try:
        if _parser_path(destination) is not None:
            return
        shutil.rmtree(destination, ignore_errors=True)
        # Only the lock holder creates staging trees, so any found here belong to a holder that was killed.
        # Such a holder also leaves the lock behind, so this runs once that lock has been removed.
        for orphan in destination.parent.glob(f".{destination.name}.tmp-*"):
            shutil.rmtree(orphan, ignore_errors=True)
        staging = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
        try:
            _git("init", "-q", str(staging))
            _git("fetch", "--depth", "1", UPSTREAM_REPO, UPSTREAM_COMMIT, cwd=staging)
            _git("checkout", "-q", "FETCH_HEAD", cwd=staging)
            os.rename(staging, destination)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
    finally:
        shutil.rmtree(lock_path, ignore_errors=True)


def _git(*args: str, cwd: Path | None = None) -> None:
    """Run one git command, and raise with its stderr if it fails.

    ``verify()`` reports ``str(exc)`` as the failure reason.
    ``CalledProcessError`` leaves stderr out of that string.
    Without stderr, a masked row would not say whether the fetch hit an unknown commit, a DNS failure, or a proxy error.
    """
    try:
        subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode(errors="replace").strip() if exc.stderr else ""
        raise RuntimeError(f"git {args[0]} of {UPSTREAM_REPO} at {UPSTREAM_COMMIT} failed: {stderr}") from exc


def _repo_dir_name() -> str:
    """Key the clone by commit, so a cache filled under another pin (or main) is never reused."""
    return "SWE-rebench-V2" if UPSTREAM_COMMIT == "main" else f"SWE-rebench-V2-{UPSTREAM_COMMIT[:12]}"


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

    repo_dir = Path(cache_dir) / _repo_dir_name()
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
