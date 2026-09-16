# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from omegaconf import OmegaConf

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import SandboxExecResult
from nemo_gym.server_utils import ServerClient
from resources_servers.terminal_bench_4.handoff import SandboxedSeedResponse
from responses_api_agents.miniswe_sandboxed_agent import app as module


@pytest.mark.parametrize("with_mcp,step_timeout", [(False, 600), (True, 30)])
async def test_real_default_agent_loop_uses_gym_model_and_borrowed_commands(
    tmp_path, monkeypatch, with_mcp, step_timeout
):
    monkeypatch.chdir(tmp_path)
    client = MagicMock(spec=ServerClient)
    first_command = (
        '/tmp/task-mcp/bin/python /tmp/task-mcp/client.py call browser navigate \'{"url":"http://app"}\''
        if with_mcp
        else "echo first"
    )
    client.post = AsyncMock(
        side_effect=[
            SimpleNamespace(
                value={
                    "id": "resp_test",
                    "created_at": 0,
                    "object": "response",
                    "model": "test",
                    "parallel_tool_calls": False,
                    "tool_choice": "auto",
                    "tools": [],
                    "output": [
                        {"type": "reasoning", "id": f"rs_{index}", "summary": []},
                        {
                            "type": "function_call",
                            "id": f"fc_{index}",
                            "call_id": f"call_{index}",
                            "name": "bash",
                            "arguments": json.dumps({"command": command}),
                            "status": "completed",
                        },
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
            for index, command in enumerate(
                [first_command, "echo large", "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"]
            )
        ]
    )

    async def decode(r):
        return r.value

    monkeypatch.setattr(module, "get_response_json", decode)
    monkeypatch.setattr(module, "raise_for_status", AsyncMock())
    commands = []
    schemas = {
        "browser": [{"name": "navigate", "inputSchema": {"type": "object", "properties": {"url": {"type": "string"}}}}]
    }
    full_output = "start" + "x" * 6000 + "MIDDLE_MUST_SURVIVE" + "y" * 6000 + "end"

    async def execute(command, **kwargs):
        commands.append((command, kwargs))
        if command.startswith("uname"):
            return SandboxExecResult("Linux\n6.1\nTask kernel\nx86_64\n", "", 0)
        if "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in command:
            return SandboxExecResult("COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n", "", 0)
        if "echo large" in command:
            return SandboxExecResult(full_output, "", -1, error_type="timeout")
        if command.startswith("setsid --wait"):
            return SandboxExecResult("MCP navigation succeeded" if with_mcp else "first", "", 0)
        return SandboxExecResult(json.dumps(schemas), "", 0)

    sandbox = SimpleNamespace(exec=execute, upload=AsyncMock())
    seed = SandboxedSeedResponse(
        session_id="task",
        sandbox={"provider": "cpu", "sandbox_id": "box"},
        instruction="Official task instruction",
        agent_timeout_sec=900,
        setup_timeout_sec=5,
        skills_dir="/skills",
        mcp_servers=[{"name": "browser", "transport": "streamable-http", "url": "http://sidecar/mcp"}]
        if with_mcp
        else [],
    )

    async def lifecycle(agent, request, body, *, setup, execute):
        await setup(sandbox, seed)
        response, termination, extra = await execute(sandbox, seed, 900)
        assert extra["harness_version"] == "2.4.6"
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
            step_timeout_sec=step_timeout,
        ),
        server_client=client,
    )
    result = await server.run(
        SimpleNamespace(cookies={}),
        module.MiniSWERunRequest(responses_create_params={"input": [], "tool_choice": "required"}),
    )
    assert result.reward == 1
    assert result.response.usage.total_tokens == 39
    assert result.termination.reason == "completed"
    requests = [call.kwargs["json"] for call in client.post.await_args_list]
    for request in requests:
        NeMoGymResponseCreateParamsNonStreaming.model_validate(request)
        assert request["tools"] == [{"type": "function", **module.BASH_TOOL["function"], "strict": False}]
        assert request["tool_choice"] == "required"
    assert requests[0]["input"][0]["content"] == module.MINI_CONFIG["agent"]["system_template"].rstrip("\n")
    prompt = requests[0]["input"][1]["content"]
    assert "Please solve this issue: " + seed.instruction in prompt
    assert "## Recommended Workflow" in prompt
    assert "Linux 6.1 Task kernel x86_64" in prompt
    assert "Task skills are in /skills" in prompt
    tool_outputs = [item for item in requests[-1]["input"] if item.get("type") == "function_call_output"]
    assert [item["call_id"] for item in tool_outputs] == ["call_0", "call_1"]
    assert tool_outputs[-1]["output"] == "<returncode>-1</returncode>\n" + full_output
    calls = [item for item in requests[-1]["input"] if item.get("type") == "function_call"]
    assert [item["call_id"] for item in calls] == ["call_0", "call_1"]
    assert [item["id"] for item in requests[-1]["input"] if item.get("type") == "reasoning"] == ["rs_0", "rs_1"]
    actions = [(command, kwargs) for command, kwargs in commands if command.startswith("setsid --wait")]
    assert len(actions) == 3
    assert all(kwargs["timeout_s"] == step_timeout for _, kwargs in actions)
    assert all(kwargs["env"] == module.MINI_CONFIG["environment"]["env"] for _, kwargs in actions)
    if with_mcp:
        assert json.dumps(schemas) in prompt
        assert "call SERVER TOOL 'JSON_ARGUMENTS'" in prompt
        assert "client.py call browser navigate" in actions[0][0]
        assert tool_outputs[0]["output"].endswith("MCP navigation succeeded")
        assert any("setsid --fork" in command and "server.sock" in command for command, _ in commands)
        assert sandbox.upload.await_count == 2


@pytest.mark.parametrize(
    "overrides,steps,timeout", [({}, 500, 30), ({"tb4_max_steps": 0, "tb4_step_timeout_sec": 45}, 0, 45)]
)
def test_benchmark_limits_resolve_defaults_and_client_overrides(overrides, steps, timeout):
    root = Path(module.__file__).resolve().parents[2]
    config = OmegaConf.merge(OmegaConf.load(root / "benchmarks/terminal_bench_4/miniswe.yaml"), overrides)
    agent = config.terminal_bench_4_miniswe.responses_api_agents.miniswe_sandboxed_agent
    assert agent.step_limit == steps
    assert agent.step_timeout_sec == timeout
    assert agent.datasets[0].num_repeats == 1
    assert module.MiniSWESandboxedConfig.model_fields["step_timeout_sec"].default == 600
