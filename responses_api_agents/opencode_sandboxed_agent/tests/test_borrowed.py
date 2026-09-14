# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.sandbox import SandboxExecResult
from nemo_gym.sandbox import agent as lifecycle
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from responses_api_agents.opencode_sandboxed_agent import borrowed
from responses_api_agents.opencode_sandboxed_agent.app import (
    OpenCodeSandboxedAgent,
    OpenCodeSandboxedAgentConfig,
    OpenCodeSandboxedAgentRunRequest,
)


@pytest.mark.parametrize(
    "outcome",
    ["completed", "timeout", "nonzero_exit", "missing_export", "observations", "snapshot_error", "export_error"],
)
async def test_handoff_exec_waits_exports_and_preserves_grade(tmp_path, monkeypatch, outcome):
    monkeypatch.chdir(tmp_path)
    params = {"input": []}
    seed = {
        "session_id": "session",
        "sandbox": {"provider": "cpu", "sandbox_id": "box", "workdir": "/task"},
        "instruction": "Official instruction",
        "user": "agent",
        "agent_timeout_sec": 9,
        "setup_timeout_sec": 5,
        "skills_dir": "/skills",
        "mcp_servers": [
            {"name": "local", "transport": "stdio", "command": "tool", "args": ["serve"]},
            {"name": "remote", "transport": "sse", "url": "http://service/mcp"},
        ],
    }
    requests = []

    async def post(**kwargs):
        requests.append(kwargs)
        path = kwargs["url_path"]
        if path == "/seed_session":
            value = seed
        elif path == "/start_session":
            value = {"agent_timeout_sec": 8}
        else:
            value = {**kwargs["json"], "reward": 1, "evaluation_completed": True}
        return SimpleNamespace(cookies={}, value=value)

    client = MagicMock(spec=ServerClient)
    client.post = post
    server = OpenCodeSandboxedAgent(
        config=OpenCodeSandboxedAgentConfig(
            name="agent",
            host="localhost",
            port=1,
            entrypoint="app.py",
            resources_server={"type": "resources_servers", "name": "resources"},
            model_server={"type": "responses_api_models", "name": "model"},
            opencode_version="1.17.11",
            opencode_max_context_window=1000,
            sandbox_provider="unused",
            sandbox_config={},
            sandbox_timeout=9,
            resources_handoff=True,
            sandbox_providers={"cpu": {"local": {}}},
        ),
        server_client=client,
    )
    monkeypatch.setattr(
        OpenCodeSandboxedAgent,
        "_create_opencode_config",
        AsyncMock(return_value={"agent": {"build": {"steps": None}}}),
    )
    monkeypatch.setattr(lifecycle, "get_global_config_dict", lambda: {})
    monkeypatch.setattr(lifecycle, "create_provider", lambda _: MagicMock())
    monkeypatch.setattr(lifecycle, "raise_for_status", AsyncMock())

    async def decode(response):
        return response.value

    monkeypatch.setattr(lifecycle, "get_response_json", decode)
    commands = []
    observations = outcome in {"observations", "snapshot_error"}
    monkeypatch.setattr(OpenCodeSandboxedAgent, "rollout_id_from_run", lambda *_: "rollout" if observations else None)
    parse = MagicMock(return_value=SimpleNamespace(model_dump=lambda **_: {"source": "opencode", "records": []}))
    monkeypatch.setattr(borrowed, "parse_opencode_observations", parse)
    export = {
        "messages": [
            {"info": {"role": "user"}, "parts": [{"type": "text", "text": "Official instruction"}]},
            {
                "info": {
                    "role": "assistant",
                    "tokens": {"input": 10, "output": 2, "reasoning": 0, "cache": {"read": 0}, "total": 12},
                },
                "parts": [{"type": "text", "text": "Done"}],
            },
        ]
    }

    async def execute(command, **kwargs):
        commands.append((command, kwargs))
        if "exec opencode run" in command:
            return SandboxExecResult(
                stdout="installer noise\n" + json.dumps({"sessionID": "opencode-session"}),
                stderr="",
                return_code=1 if outcome == "nonzero_exit" else 0,
                error_type="timeout" if outcome == "timeout" else None,
            )
        if ("source.backup" in command and outcome == "snapshot_error") or (
            "opencode export" in command and outcome == "export_error"
        ):
            return SandboxExecResult(stdout="", stderr="failed", return_code=1)
        return SandboxExecResult(stdout="", stderr="", return_code=0)

    async def download(remote, local):
        if outcome == "missing_export":
            raise FileNotFoundError(remote)
        local.write_text(json.dumps(export))

    sandbox = SimpleNamespace(exec=execute, download=download, release=AsyncMock())
    connect = AsyncMock(return_value=sandbox)
    monkeypatch.setattr(lifecycle.AsyncSandbox, "connect", connect)
    request = SimpleNamespace(cookies={}, session={SESSION_ID_KEY: "client"})
    response = await server.run(request, OpenCodeSandboxedAgentRunRequest(responses_create_params=params))
    run_command, kwargs = next(c for c in commands if "exec opencode run" in c[0])
    assert run_command.startswith("setsid --wait")
    assert kwargs["timeout_s"] == 8
    assert kwargs["user"] == "agent"
    config = json.loads(kwargs["env"]["OPENCODE_CONFIG_CONTENT"])
    assert "steps" not in config["agent"]["build"]
    assert config["mcp"]["local"]["command"] == ["tool", "serve"]
    assert config["mcp"]["remote"]["url"] == "http://service/mcp"
    assert response.reward == 1
    assert response.opencode_export_found == (outcome not in {"missing_export", "export_error"})
    assert response.model_dump()["termination"]["reason"] == (
        outcome if outcome in {"timeout", "nonzero_exit"} else "completed"
    )
    if outcome not in {"missing_export", "export_error"}:
        assert response.response.usage.input_tokens == 10
        assert response.response.output[0].content[0].text == "Done"
    sandbox.release.assert_awaited_once()

    if observations:
        assert any("source.backup" in command for command, _ in commands)
        if outcome == "observations":
            parse.assert_called_once()
            assert response.model_dump()["ng_agent_observations"]["source"] == "opencode"
        else:
            parse.assert_not_called()
            assert (tmp_path / "results/agent/session/missing-observations.txt").is_file()
