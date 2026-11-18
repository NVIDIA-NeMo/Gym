# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Install the upstream Hermes source layout and its own frozen Python environment."""

import functools
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

from nemo_gym import CACHE_DIR
from nemo_gym.config.global_config import CACHE_DIR_KEY_NAME, maybe_get_global_config_dict


# Wheel metadata reports 0.0.0 and wheels omit runtime assets.
with Path(__file__).with_name("pyproject.toml").open("rb") as metadata:
    HERMES_SOURCE = tomllib.load(metadata)["tool"]["nemo-gym"]["hermes"]["source"]
HERMES_COMMIT = HERMES_SOURCE.rsplit("@", 1)[1]


@functools.cache
def source_checkout(source_spec: str = HERMES_SOURCE) -> Path:
    """Publish the verified checkout atomically, shared by host and sandbox builders."""
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


@functools.cache
def host_python() -> Path:
    """Sync the release lock, including its MCP client, outside Gym's interpreter."""
    bundled = Path(sys.prefix) / "hermes-runtime"
    if (bundled / "hermes-source").is_file():
        recorded = json.loads((bundled / "hermes-source").read_text())
        expected = os.environ.get("HERMES_SPEC") or recorded["source"]
        if recorded["default"] != HERMES_SOURCE or recorded["source"] != expected:
            raise RuntimeError(f"Portable Hermes at {bundled} was built from a different source")
        return bundled / "bin/python3"
    source = source_checkout()
    env = os.environ.copy()
    env["UV_PROJECT_ENVIRONMENT"] = str(source / ".venv")
    subprocess.run(
        ["uv", "sync", "--frozen", "--no-dev", "--extra", "mcp", "--python", "3.14"],
        check=True,
        cwd=source,
        env=env,
    )
    return source / ".venv/bin/python"


@functools.cache
def builtin_toolsets() -> frozenset[str]:
    """Read the toolset names from the exact runtime that will receive the grants."""
    output = subprocess.check_output(
        [
            str(host_python()),
            "-I",
            "-c",
            "import json; from toolsets import TOOLSETS; print(json.dumps(sorted(TOOLSETS)))",
        ],
        text=True,
    )
    return frozenset(json.loads(output))


def stage_source(source: Path, site_packages: Path) -> None:
    """Stage the release with the data files Hermes resolves relative to its source."""
    tree = site_packages / "hermes-source"
    shutil.copytree(source, tree, symlinks=True, ignore=shutil.ignore_patterns(".git", ".venv", "__pycache__"))
    (site_packages / "hermes-source.pth").write_text("hermes-source\n")
