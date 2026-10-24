# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
COMPONENT_ROOTS = ("resources_servers", "responses_api_agents", "responses_api_models", "environment_servers")


def _tracked_files() -> set[str]:
    return set(subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines())


def _servers(tracked: set[str]) -> list[str]:
    return sorted(
        path.removesuffix("/app.py")
        for path in tracked
        if path.endswith("/app.py") and path.split("/")[0] in COMPONENT_ROOTS
    )


def test_servers_declare_locked_dependencies() -> None:
    tracked = _tracked_files()
    servers = _servers(tracked)
    assert servers
    missing = [
        f"{server}/{name}"
        for server in servers
        for name in ("pyproject.toml", "uv.lock")
        if f"{server}/{name}" not in tracked
    ]
    assert not missing, "Servers must declare dependencies in a committed pyproject.toml and uv.lock:\n" + "\n".join(
        missing
    )


def test_servers_do_not_use_requirements_txt() -> None:
    tracked = _tracked_files()
    legacy = [f"{server}/requirements.txt" for server in _servers(tracked) if f"{server}/requirements.txt" in tracked]
    assert not legacy, "Move these dependencies into pyproject.toml and commit uv.lock:\n" + "\n".join(legacy)
