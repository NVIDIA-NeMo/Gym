# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reject the stock Gymnasium package, which has the same name and version."""

import json
import tomllib
from importlib.metadata import distribution
from pathlib import Path


def test_installed_gymnasium_is_the_pinned_visgym_fork() -> None:
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text())
    source = project["tool"]["uv"]["sources"]["gymnasium"]
    installed = distribution("gymnasium")
    direct_url = installed.read_text("direct_url.json")
    assert direct_url, "VisGym must be installed from the pinned Git source, not stock Gymnasium."
    origin = json.loads(direct_url)
    assert origin["url"] == source["git"]
    assert origin["vcs_info"]["vcs"] == "git"
    assert origin["vcs_info"]["commit_id"] == source["rev"]
    for module in ("maze_2d/maze_2d.py", "patch_reassembly/patch_reassembly.py"):
        assert installed.locate_file(f"gymnasium/envs/{module}").is_file()
