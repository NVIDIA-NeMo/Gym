# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Portable runtime sentinels must track the metadata actually used to build the bundle."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from responses_api_agents.anyterminal_agent import app


@pytest.mark.parametrize(
    "metadata",
    ["component/pyproject.toml", "component/uv.lock", "pyproject.toml", "uv.lock", "setup.py"],
)
def test_setup_rebuilds_after_dependency_metadata_changes(tmp_path: Path, metadata: str) -> None:
    root = tmp_path / "root"
    component = root / "responses_api_agents/hermes_agent"
    (component / "scripts").mkdir(parents=True)
    (component / "scripts/hermes_agent_deps.sh").write_text("script")
    host = tmp_path / "host"
    (host / "setup_scripts").mkdir(parents=True)
    (host / "setup_scripts/_portable_python.sh").write_text("helper")
    path = component / metadata.removeprefix("component/") if metadata.startswith("component/") else root / metadata
    path.write_text("before")
    processor = app.GymAgentHarnessProcessor(
        config=SimpleNamespace(agent_server_module="responses_api_agents.hermes_agent.app")
    )
    process = MagicMock()
    process.wait.return_value = 0
    with (
        patch.object(app, "PARENT_DIR", root),
        patch.object(type(processor), "_parent", new_callable=PropertyMock, return_value=host),
        patch.object(app, "Popen", return_value=process) as popen,
    ):
        deps = processor.setup()
        before = (deps / ".installed").read_text()
        assert popen.call_count == 1
        assert processor.setup() == deps
        assert popen.call_count == 1
        path.write_text("after")
        assert processor.setup() == deps
        assert popen.call_count == 2
        assert (deps / ".installed").read_text() != before
