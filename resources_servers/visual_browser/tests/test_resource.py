# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Component packaging contract; pure browser policy tests live in the core suite."""

import tomllib
from pathlib import Path


COMPONENT_ROOT = Path(__file__).resolve().parents[1]


def test_component_packages_only_its_own_resource_boundary() -> None:
    project = tomllib.loads((COMPONENT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert "nemo-gym" in project["project"]["dependencies"]
    assert project["tool"]["setuptools"]["packages"]["find"]["include"] == ["resources_servers.visual_browser*"]
