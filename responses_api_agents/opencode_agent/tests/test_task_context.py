# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from responses_api_agents.opencode_agent.task_context import TaskToolContext, read_task_context


def test_native_mcp_and_skills_configuration():
    context = TaskToolContext(
        mcp_servers=[
            {"name": "browser", "transport": "sse", "url": "http://sidecar:3080/sse"},
            {"name": "local", "transport": "stdio", "command": "python", "args": ["/app/server.py"]},
        ],
        skills_dir="/app/.agents/skills",
    )
    config = context.opencode_config()
    assert config["mcp"]["browser"] == {
        "type": "remote",
        "url": "http://sidecar:3080/sse",
        "enabled": True,
        "oauth": False,
    }
    assert config["mcp"]["local"]["command"] == ["python", "/app/server.py"]
    assert config["skills"]["paths"] == ["/app/.agents/skills"]


@pytest.mark.parametrize(
    "value",
    [
        {"skills_dir": "relative"},
        {"mcp_servers": [{"name": "invalid", "transport": "stdio"}]},
        {"mcp_servers": [{"name": "duplicate", "url": "http://host"}] * 2},
    ],
)
def test_invalid_declarations_fail(value):
    with pytest.raises(ValidationError):
        TaskToolContext.model_validate(value)


async def test_provider_failure_does_not_look_like_missing_task_context():
    sandbox = SimpleNamespace(
        exec=AsyncMock(return_value=SimpleNamespace(return_code=0, error_type="timeout", stderr=""))
    )
    with pytest.raises(RuntimeError, match="Unable to read"):
        await read_task_context(sandbox)


async def test_missing_skills_directory_fails_setup():
    sandbox = SimpleNamespace(
        exec=AsyncMock(
            side_effect=[
                SimpleNamespace(return_code=0, error_type=None, stdout='{"skills_dir":"/app/missing"}'),
                SimpleNamespace(return_code=1, error_type=None),
            ]
        )
    )
    with pytest.raises(RuntimeError, match="skills directory"):
        await read_task_context(sandbox)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required for the OpenCode plugin")
@pytest.mark.parametrize("connected", [True, False])
def test_required_mcp_receipt_requires_a_connected_server(tmp_path, connected):
    plugin = Path(__file__).parents[1] / "task_environment.mjs"
    shutil.copyfile(plugin, tmp_path / plugin.name)
    (tmp_path / "task-env.json").write_text(
        json.dumps(
            {key: "/task" for key in ["HOME", "XDG_DATA_HOME", "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_STATE_HOME"]}
        )
    )
    script = """import { TaskEnvironment } from "./task_environment.mjs";
const hooks = await TaskEnvironment({client:{mcp:{status:async()=>({data:{browser:{status:process.argv[2]}}})}}});
await hooks["chat.params"]();
"""
    (tmp_path / "probe.mjs").write_text(script)
    result = subprocess.run(
        [shutil.which("node"), str(tmp_path / "probe.mjs"), "connected" if connected else "failed"],
        env=os.environ | {"NEMO_GYM_REQUIRED_MCP_SERVERS": '["browser"]'},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert (result.returncode == 0) is connected
    assert (tmp_path / "task-tools.ready").exists() is connected
