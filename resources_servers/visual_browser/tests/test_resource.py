# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Component packaging contract; pure browser policy tests live in the core suite."""

import tomllib
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nemo_gym.client import ServerClient
from nemo_gym.server_runtime.base_server import refuse_unsupported_num_workers
from resources_servers.visual_browser.app import VisualBrowserResourcesServer
from resources_servers.visual_browser.config import VisualBrowserResourcesServerConfig


COMPONENT_ROOT = Path(__file__).resolve().parents[1]


def test_component_packages_only_its_own_resource_boundary() -> None:
    project = tomllib.loads((COMPONENT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert "nemo-gym" in project["project"]["dependencies"]
    assert project["tool"]["setuptools"]["packages"]["find"]["include"] == ["resources_servers.visual_browser*"]


def test_several_workers_are_refused() -> None:
    config = VisualBrowserResourcesServerConfig(
        name="visual-browser", host="localhost", port=8010, entrypoint="app.py", domain="agent", num_workers=2
    )
    server = VisualBrowserResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    with pytest.raises(ValueError, match="needs num_workers=1: PyAutoGUI drives the one X display"):
        refuse_unsupported_num_workers(server)
