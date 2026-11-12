# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Package paths named in configs resolve like Gym's other read paths."""

from pathlib import Path

from pydantic import BaseModel

from nemo_gym import PARENT_DIR
from resources_servers.partner_if.package import RepoPath, resolve_repo_path


def test_relative_package_paths_resolve_under_a_plugin_root_then_the_gym_root(tmp_path, monkeypatch):
    class Settings(BaseModel):
        path: RepoPath

    # Servers start in their own directories; the result must not depend on the cwd.
    monkeypatch.chdir(tmp_path)
    example = Path("resources_servers/partner_if/data/example.jsonl")
    assert resolve_repo_path(example) == PARENT_DIR / example
    assert Settings(path=str(example)).path == PARENT_DIR / example
    assert resolve_repo_path(tmp_path) == tmp_path
    # A benchmark kept outside Gym is found under its root (`--search-dir` / NEMO_GYM_EXTRA_ROOTS).
    plugin = tmp_path / "plugin"
    package = Path("benchmarks/synthetic/data/package")
    (plugin / package).mkdir(parents=True)
    monkeypatch.setenv("NEMO_GYM_EXTRA_ROOTS", str(plugin))
    assert resolve_repo_path(package) == plugin / package
    assert Settings(path=str(package / "task.json")).path == plugin / package / "task.json"
