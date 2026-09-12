# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import pytest

from nemo_gym.openai_utils import NeMoGymEasyInputMessage, NeMoGymFunctionCallOutput
from responses_api_agents.mini_swe_agent_sandboxed_agent.app import (
    SUBMIT_MARKER,
    NeMoGymMiniSweAgent,
    NeMoGymResponsesModel,
    NeMoGymSandboxShellEnvironment,
    _load_mini_config_file,
)


class FakeSandbox:
    """Scripted sandbox: maps the command passed to `<shell> -c` onto a canned result."""

    def __init__(self, results):
        self.results = results
        self.calls = []

    async def exec(self, command, **kwargs):
        self.calls.append((command, kwargs))
        if command.startswith("uname"):
            return SimpleNamespace(
                stdout="Linux\n6.5.0\n#1 SMP\nx86_64\nbox\n", stderr=None, return_code=0, error_type=None
            )
        for needle, result in self.results:
            if needle in command:
                return result
        return SimpleNamespace(stdout="", stderr=None, return_code=0, error_type=None)


def _ok(stdout, return_code=0):
    return SimpleNamespace(stdout=stdout, stderr=None, return_code=return_code, error_type=None)


class FakeClient:
    """Returns scripted Responses-API payloads and records the request bodies."""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.requests = []

    async def create_response(self, **kwargs):
        self.requests.append(kwargs)
        output = self.outputs.pop(0)
        return {
            "id": f"resp_{len(self.requests)}",
            "created_at": 0,
            "model": "policy_model",
            "object": "response",
            "output": output,
            "tool_choice": "auto",
            "tools": [],
            "parallel_tool_calls": True,
            "usage": {
                "input_tokens": 10,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 5,
                "output_tokens_details": {"reasoning_tokens": 3},
                "total_tokens": 15,
            },
        }


def _reasoning(text, idx):
    return {"id": f"rs_{idx}", "type": "reasoning", "summary": [{"type": "summary_text", "text": text}]}


def _message(text, idx):
    return {
        "id": f"msg_{idx}",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def _bash(command, idx):
    return {
        "id": f"fc_{idx}",
        "type": "function_call",
        "call_id": f"call_{idx}",
        "name": "bash",
        "arguments": json.dumps({"command": command}),
        "status": "completed",
    }


def _build(client, sandbox, **agent_overrides):
    mini = _load_mini_config_file(None)
    model = NeMoGymResponsesModel(
        client,
        "policy_model",
        observation_template=mini["model"]["observation_template"],
        format_error_template=mini["model"]["format_error_template"],
        model_kwargs={},
        call_timeout_s=10,
        max_attempts=1,
        replay_reasoning_items=False,
    )
    env = NeMoGymSandboxShellEnvironment(sandbox, timeout=30, env=mini["environment"]["env"], shell="/bin/sh")
    kwargs = dict(
        system_template=mini["agent"]["system_template"],
        instance_template=mini["agent"]["instance_template"],
        step_limit=500,
        cost_limit=0.0,
        wall_time_limit_seconds=0,
        max_consecutive_format_errors=3,
    )
    kwargs.update(agent_overrides)
    return model, env, NeMoGymMiniSweAgent(model, env, **kwargs)


@pytest.mark.asyncio
async def test_submit_flow_records_full_trajectory_and_replays_without_reasoning():
    sandbox = FakeSandbox([("ls -la", _ok("file.py\n")), (SUBMIT_MARKER, _ok(f"{SUBMIT_MARKER}\n"))])
    client = FakeClient(
        [
            [_reasoning("think 1", 1), _message("Listing", 1), _bash("ls -la", 1)],
            [_reasoning("think 2", 2), _bash(f"echo {SUBMIT_MARKER}", 2)],
        ]
    )
    model, env, agent = _build(client, sandbox)
    await env.prepare()
    exit_info = await agent.run("Fix the bug")

    assert exit_info["exit_status"] == "Submitted"
    assert agent.n_calls == 2

    # Prompts are the packaged mini.yaml ones, rendered with the sandbox uname.
    assert agent.messages[0]["role"] == "system"
    assert "Please solve this issue: Fix the bug" in agent.messages[1]["content"]
    assert "Linux 6.5.0 #1 SMP x86_64" in agent.messages[1]["content"]

    # Commands run through `<shell> -c` with stderr merged, under the 30 s timeout, with mini's env.
    shell_calls = [c for c in sandbox.calls if not c[0].startswith("uname")]
    assert shell_calls[0][0] == "/bin/sh -c 'ls -la' 2>&1"
    assert shell_calls[0][1]["timeout_s"] == 30
    assert shell_calls[0][1]["env"]["PAGER"] == "cat"

    # Observation uses mini's JSON observation template.
    obs = json.loads(agent.messages[3]["output"])
    assert obs == {"returncode": 0, "output": "file.py\n"}

    # Second request replays message + function_call + function_call_output but not the reasoning item.
    replayed_types = [item.get("type") or item.get("role") for item in client.requests[1]["input"]]
    assert replayed_types == ["system", "user", "message", "function_call", "function_call_output"]
    assert client.requests[1]["tools"][0]["name"] == "bash"

    # Stored rollout keeps everything, reasoning included, in order.
    kinds = []
    for item in agent.trajectory:
        if isinstance(item, NeMoGymEasyInputMessage):
            kinds.append(item.role)
        elif isinstance(item, NeMoGymFunctionCallOutput):
            kinds.append("function_call_output")
        else:
            kinds.append(item.type)
    assert kinds == [
        "system",
        "user",
        "reasoning",
        "message",
        "function_call",
        "function_call_output",
        "reasoning",
        "function_call",
    ]


@pytest.mark.asyncio
async def test_format_error_then_recovery():
    sandbox = FakeSandbox([(SUBMIT_MARKER, _ok(f"{SUBMIT_MARKER}\n"))])
    client = FakeClient(
        [
            [_message("I forgot the tool call", 1)],
            [_bash(f"echo {SUBMIT_MARKER}", 2)],
        ]
    )
    model, env, agent = _build(client, sandbox)
    exit_info = await agent.run("task")

    assert exit_info["exit_status"] == "Submitted"
    assert agent.n_format_errors == 1
    # Upstream: the bad response is not replayed; the error message is a user turn.
    replayed = client.requests[1]["input"]
    assert [i.get("type") or i.get("role") for i in replayed] == ["system", "user", "message"]
    assert replayed[2]["role"] == "user"
    assert "No tool calls found" in replayed[2]["content"][0]["text"]
    # Stored rollout: system, user, failed message, error user turn, function_call.
    assert isinstance(agent.trajectory[2], type(agent.trajectory[2])) and agent.trajectory[2].type == "message"
    assert isinstance(agent.trajectory[3], NeMoGymEasyInputMessage) and agent.trajectory[3].role == "user"


@pytest.mark.asyncio
async def test_repeated_format_errors_exit():
    client = FakeClient([[_message("nope", i)] for i in range(3)])
    model, env, agent = _build(client, FakeSandbox([]))
    exit_info = await agent.run("task")
    assert exit_info["exit_status"] == "RepeatedFormatError"
    assert agent.n_calls == 3


@pytest.mark.asyncio
async def test_step_limit():
    client = FakeClient([[_bash("true", i)] for i in range(5)])
    model, env, agent = _build(client, FakeSandbox([]), step_limit=2)
    exit_info = await agent.run("task")
    assert exit_info["exit_status"] == "LimitsExceeded"
    assert agent.n_calls == 2


@pytest.mark.asyncio
async def test_timeout_and_sandbox_error_mapping():
    timed_out = SimpleNamespace(stdout="partial", stderr="TimeoutError: killed", return_code=125, error_type="sandbox")
    sandbox = FakeSandbox([("sleep 100", timed_out)])
    env = NeMoGymSandboxShellEnvironment(sandbox, timeout=30, env={}, shell="/bin/sh")
    out = await env.execute({"command": "sleep 100"})
    assert out["returncode"] == -1
    assert out["output"] == "partial"
    assert "timed out after 30 seconds" in out["exception_info"]
    assert out["extra"]["exception_type"] == "TimeoutExpired"

    class Boom(FakeSandbox):
        async def exec(self, command, **kwargs):
            raise RuntimeError("backend gone")

    env = NeMoGymSandboxShellEnvironment(Boom([]), timeout=30, env={}, shell="/bin/sh")
    out = await env.execute({"command": "ls"})
    assert out["returncode"] == -1 and "backend gone" in out["exception_info"]


@pytest.mark.asyncio
async def test_submit_marker_requires_first_line_and_zero_exit():
    sandbox = FakeSandbox([("echo", _ok(f"noise\n{SUBMIT_MARKER}\n"))])
    env = NeMoGymSandboxShellEnvironment(sandbox, timeout=30, env={}, shell="/bin/sh")
    out = await env.execute({"command": "echo x"})  # marker not on the first line -> not a submission
    assert out["returncode"] == 0


async def test_raw_tool_output_kept_when_observation_is_elided():
    long_out = "x" * 12_000 + "\n"
    sandbox = FakeSandbox([("cat big", _ok(long_out)), (SUBMIT_MARKER, _ok(f"{SUBMIT_MARKER}\n"))])
    client = FakeClient([[_bash("cat big", 1)], [_bash(f"echo {SUBMIT_MARKER}", 2)]])
    model, env, agent = _build(client, sandbox)
    await env.prepare()
    await agent.run("Read the big file")

    # What the model saw is upstream's head/tail elision...
    obs = json.loads(agent.messages[3]["output"])
    assert obs["elided_chars"] == 2_001 and "output" not in obs
    # ...but the untouched text is kept alongside, keyed by call_id. The submit command ends the
    # episode before an observation is formatted (upstream behaviour), so it has no raw entry either:
    # per episode, len(raw) == number of function_call_output items == model calls - 1.
    assert [r["call_id"] for r in agent.raw_tool_outputs] == ["call_1"]
    assert agent.raw_tool_outputs[0]["output"] == long_out
    assert agent.raw_tool_outputs[0]["returncode"] == 0
    assert agent.raw_tool_outputs[0]["elided_in_observation"] is True
