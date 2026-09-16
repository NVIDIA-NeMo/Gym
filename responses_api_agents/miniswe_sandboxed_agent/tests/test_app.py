# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.sandbox import SandboxExecResult
from nemo_gym.server_utils import ServerClient
from resources_servers.terminal_bench_4.handoff import SandboxedSeedResponse
from responses_api_agents.miniswe_sandboxed_agent import app as module


@pytest.mark.parametrize("with_mcp", [False, True])
async def test_real_default_agent_loop_uses_gym_model_and_borrowed_commands(tmp_path, monkeypatch, with_mcp):
    monkeypatch.chdir(tmp_path)
    client = MagicMock(spec=ServerClient)
    client.post = AsyncMock(
        return_value=SimpleNamespace(
            value={
                "id": "resp_test",
                "created_at": 0,
                "object": "response",
                "model": "test",
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
                "output": [
                    {
                        "type": "message",
                        "id": "msg_test",
                        "status": "completed",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "```mswea_bash_command\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n```",
                                "annotations": [],
                            }
                        ],
                    }
                ],
                "usage": {
                    "input_tokens": 10,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": 3,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": 13,
                },
            }
        )
    )

    async def decode(r):
        return r.value

    monkeypatch.setattr(module, "get_response_json", decode)
    monkeypatch.setattr(module, "raise_for_status", AsyncMock())
    commands = []

    async def execute(command, **kwargs):
        commands.append(command)
        if "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in command:
            return SandboxExecResult("COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n", "", 0)
        return SandboxExecResult('{"browser": []}', "", 0)

    sandbox = SimpleNamespace(exec=execute, upload=AsyncMock())
    seed = SandboxedSeedResponse(
        session_id="task",
        sandbox={"provider": "cpu", "sandbox_id": "box"},
        instruction="Official task instruction",
        agent_timeout_sec=5,
        setup_timeout_sec=5,
        skills_dir="/skills",
        mcp_servers=[{"name": "browser", "transport": "streamable-http", "url": "http://sidecar/mcp"}]
        if with_mcp
        else [],
    )

    async def lifecycle(agent, request, body, *, setup, execute):
        await setup(sandbox, seed)
        response, termination, extra = await execute(sandbox, seed, 5)
        return {
            **body.model_dump(),
            "response": response.model_dump(),
            "session_id": "task",
            "termination": termination.model_dump(),
            "reward": 1,
            "evaluation_completed": True,
            **extra,
        }

    monkeypatch.setattr(module, "run_borrowed", lifecycle)
    server = module.MiniSWESandboxedAgent(
        config=module.MiniSWESandboxedConfig(
            name="agent",
            host="localhost",
            port=1,
            entrypoint="app.py",
            resources_server={"type": "resources_servers", "name": "resources"},
            model_server={"type": "responses_api_models", "name": "model"},
            sandbox_providers={},
        ),
        server_client=client,
    )
    result = await server.run(
        SimpleNamespace(cookies={}), module.MiniSWERunRequest(responses_create_params={"input": []})
    )
    assert result.reward == 1
    assert result.response.usage.total_tokens == 13
    assert result.termination.reason == "completed"
    assert any(c.startswith("setsid --wait") for c in commands)
    assert client.post.await_args.kwargs["json"]["input"][1]["content"].startswith(seed.instruction)
    if with_mcp:
        assert any("setsid --fork" in c and "server.sock" in c for c in commands)
        assert sandbox.upload.await_count == 2
