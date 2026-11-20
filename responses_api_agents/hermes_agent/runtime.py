# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-side helpers for running Hermes locally and building the runtime uploaded to task sandboxes."""

import ast
import functools
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import tomllib
from pathlib import Path

from nemo_gym import CACHE_DIR
from nemo_gym.config import CACHE_DIR_KEY_NAME, maybe_get_global_config_dict


# Wheel metadata reports 0.0.0 and wheels omit runtime assets.
with Path(__file__).with_name("pyproject.toml").open("rb") as metadata:
    _RUNTIME = tomllib.load(metadata)["tool"]["nemo-gym"]["hermes"]
HERMES_SOURCE = _RUNTIME["source"]
HERMES_PYTHON_VERSION = _RUNTIME["python-version"]
HERMES_COMMIT = HERMES_SOURCE.rsplit("@", 1)[1]


@functools.cache
def source_checkout(source_spec: str = HERMES_SOURCE) -> Path:
    """Publish the verified checkout atomically in the host cache for local workers and sandbox bundle builds."""
    import fcntl

    cache = Path((maybe_get_global_config_dict() or {}).get(CACHE_DIR_KEY_NAME) or CACHE_DIR)
    repository, separator, revision = source_spec.removeprefix("git+").rpartition("@")
    if not separator or not repository.startswith("https://") or not revision or revision.startswith("-"):
        raise ValueError(f"Expected a git+https Hermes checkout with an explicit revision, got {source_spec!r}")
    cache_key = hashlib.sha256(source_spec.encode()).hexdigest()[:20]
    directory = cache / "hermes" / cache_key
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / "source"
    with (directory / ".setup.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not source.exists():
            with tempfile.TemporaryDirectory(dir=directory) as temporary:
                checkout = Path(temporary) / "source"
                subprocess.run(["git", "init", "-q", str(checkout)], check=True)
                subprocess.run(["git", "fetch", "--depth=1", repository, revision], cwd=checkout, check=True)
                subprocess.run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=checkout, check=True)
                checkout.rename(source)
        if (
            len(revision) == 40
            and subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip() != revision
        ):
            raise RuntimeError(f"Unexpected Hermes source revision in {source}")
    return source


_HOST_PYTHON_LOCK = threading.Lock()


def host_python() -> Path:
    """Return the host interpreter used by the local worker.

    Reuse a prebuilt Hermes environment or run ``uv sync`` on the host with the upstream lock and MCP extra.
    Concurrent first calls (one per request thread) share one installation.
    """
    with _HOST_PYTHON_LOCK:
        return _host_python()


@functools.cache
def _host_python() -> Path:
    bundled = Path(sys.prefix) / "hermes-runtime"
    if (bundled / "hermes-source").is_file():
        recorded = json.loads((bundled / "hermes-source").read_text())
        expected = os.environ.get("HERMES_SPEC") or recorded["source"]
        if recorded["default"] != HERMES_SOURCE or recorded["source"] != expected:
            raise RuntimeError(f"Portable Hermes at {bundled} was built from a different source")
        return bundled / "bin/python3"
    source = source_checkout()
    env = os.environ.copy()
    environment = source.parent / f"venv-{HERMES_PYTHON_VERSION}"
    env["UV_PROJECT_ENVIRONMENT"] = str(environment)
    subprocess.run(
        ["uv", "sync", "--frozen", "--no-dev", "--extra", "mcp", "--python", HERMES_PYTHON_VERSION],
        check=True,
        cwd=source,
        env=env,
    )
    return environment / "bin/python"


@functools.cache
def builtin_toolsets() -> frozenset[str]:
    """Read the built-in toolset names from the pinned source, so the Agent Server can detect conflicting MCP grants.

    This parses ``toolsets.py`` instead of importing it, so a session that runs Hermes only in its sandbox does not
    install Hermes on the Agent Server host.
    """
    tree = ast.parse((source_checkout() / "toolsets.py").read_text())
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "TOOLSETS" for target in node.targets)
            and isinstance(node.value, ast.Dict)
            and all(isinstance(key, ast.Constant) and isinstance(key.value, str) for key in node.value.keys)
        ):
            return frozenset(key.value for key in node.value.keys)
    raise RuntimeError(f"Could not read the TOOLSETS literal from Hermes {HERMES_COMMIT} toolsets.py")


def stage_source(source: Path, site_packages: Path) -> None:
    """Copy the host checkout and its runtime assets into the bundle's local site-packages staging directory."""
    tree = site_packages / "hermes-source"
    shutil.copytree(source, tree, symlinks=True, ignore=shutil.ignore_patterns(".git", ".venv", "__pycache__"))
    (site_packages / "hermes-source.pth").write_text("hermes-source\n")
