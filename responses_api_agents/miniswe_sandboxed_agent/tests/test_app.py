# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import SandboxExecResult
from nemo_gym.server_utils import ServerClient
from responses_api_agents.miniswe_sandboxed_agent import harness as module


@pytest.mark.parametrize("custom_directory", [False, True])
@pytest.mark.parametrize("with_mcp,step_timeout", [(False, 600), (True, 30)])
async def test_real_default_agent_loop_uses_injected_model_and_existing_sandbox(
    tmp_path, monkeypatch, with_mcp, step_timeout, custom_directory
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
                    "id": f"resp_test_{index}",
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
    seed = module.HarnessContext(
        session_id="task",
        workdir="/task",
        instruction="Official task instruction",
        setup_timeout_sec=5,
        skills_dir="/skills",
        mcp_servers=[{"name": "browser", "transport": "streamable-http", "url": "http://sidecar/mcp"}]
        if with_mcp
        else [],
    )

    async def query(params):
        return NeMoGymResponse.model_validate(await decode(await client.post(json=params)))

    directory = tmp_path / "custom" / "artifacts" if custom_directory else Path("results/agent/task")
    harness = module.MiniSWEHarness(
        sandbox=sandbox,
        context=seed,
        config=module.MiniSWEConfig(step_timeout_sec=step_timeout),
        params=NeMoGymResponseCreateParamsNonStreaming(input=[], tool_choice="required"),
        query=query,
        model_name="model",
        directory=directory,
    )
    await harness.setup()
    response, termination, extra = await harness.execute(900)
    assert extra["harness_version"] == "2.4.6"
    result = SimpleNamespace(response=response, termination=termination)
    assert (directory / "trajectory.json").is_file()
    assert result.termination.artifacts == [str(directory / "trajectory.json")]
    assert result.response.model == "model"
    assert result.response.tool_choice == "required"
    assert len(result.response.output) == 9
    assert result.response.usage.total_tokens == 39
    assert result.termination.reason == "completed"
    observations = extra["ng_agent_observations"]
    invocation = observations["records"][0]
    assert invocation["invocation_id"] == seed.session_id
    assert invocation["status"] == "completed"
    assert [ref["response_id"] for ref in invocation["model_calls"]] == [f"resp_test_{i}" for i in range(3)]
    tools = observations["records"][1:]
    assert [tool["status"] for tool in tools] == ["completed", "timeout", "completed"]
    assert [tool["tool_call_id"] for tool in tools] == [f"call_{i}" for i in range(3)]
    assert all(tool["duration_ms"] >= 0 and tool["completed_at"] >= tool["started_at"] for tool in tools)
    assert all(left["completed_at"] <= right["started_at"] for left, right in zip(tools, tools[1:]))
    outcomes = [item for item in response.output if item.type == "function_call_output"]
    assert len(outcomes) == 3
    assert "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in outcomes[-1].output
    turns = extra["ng_trajectory"]["turns"]
    assert [turn["step_count"] for turn in turns] == [1, 2, 3]
    assert all(turn["resolved"] is None for turn in turns)
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
    assert all(kwargs["cwd"] == "/task" for _, kwargs in actions)
    assert all(kwargs["timeout_s"] == step_timeout for _, kwargs in actions)
    assert all(kwargs["env"] == module.MINI_CONFIG["environment"]["env"] for _, kwargs in actions)
    if with_mcp:
        assert json.dumps(schemas) in prompt
        assert "call SERVER TOOL 'JSON_ARGUMENTS'" in prompt
        assert "client.py call browser navigate" in actions[0][0]
        assert tool_outputs[0]["output"].endswith("MCP navigation succeeded")
        assert any("setsid --fork" in command and "server.sock" in command for command, _ in commands)
        assert sandbox.upload.await_count == 2
        assert json.loads((directory / "mcp.json").read_text()) == seed.mcp_servers
        assert sandbox.upload.await_args_list[1].args[0] == directory / "mcp.json"
