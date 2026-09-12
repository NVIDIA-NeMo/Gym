# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from types import SimpleNamespace

import pytest
from minisweagent.exceptions import LimitsExceeded

import responses_api_agents.mini_swe_agent_sandboxed_agent.app as mini
from responses_api_agents.mini_swe_agent_sandboxed_agent.sandbox_identity import SandboxIdentityMismatch, checked_exec
from responses_api_agents.mini_swe_agent_sandboxed_agent.tests.test_reference import (
    FakeClient,
    FakeSandbox,
    _bash,
    _build,
    _ok,
)
from responses_api_agents.mini_swe_agent_sandboxed_agent.tests.test_tb4 import build


def test_input_message_shapes_preserve_instruction_order():
    messages = [
        mini.NeMoGymEasyInputMessage(role="user", content="first"),
        {"content": [{"type": "input_text", "text": "second"}, {"type": "input_image"}]},
        {"content": "third"},
        4,
    ]
    assert mini._instruction(messages) == "first\n\nsecond\n\nthird\n\n4"
    assert mini._instruction(None) == ""
    assert mini._text_of("text") == "text"
    assert mini._text_of([{"text": "a"}, {"text": "b"}]) == "a\nb"
    assert mini._text_of(42) == "42"


async def test_model_deadline_retries_and_exhaustion():
    client = FakeClient([[_bash("pwd", 1)]])
    original = client.create_response
    calls = 0

    async def first_timeout(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            await asyncio.sleep(1)
        return await original(**kwargs)

    client.create_response = first_timeout
    model, _, _ = _build(client, FakeSandbox([]))
    model._call_timeout_s = 0.001
    model._max_attempts = 2
    assert (await model.query([]))["extra"]["actions"][0]["command"] == "pwd"
    assert model.calls_gt_timeout == 1 and calls == 2
    calls = 0
    model._max_attempts = 1
    with pytest.raises(TimeoutError, match="query model endpoint"):
        await model.query([])
    assert model.calls_gt_timeout == 2


@pytest.mark.parametrize("kind", ["context_window_exceeded", "unknown_failure"])
async def test_preserved_model_errors_remain_in_capture(kind):
    client = FakeClient([[]])
    original = client.create_response

    async def preserved(**kwargs):
        response = await original(**kwargs)
        response["metadata"] = {"upstream_error": json.dumps({"kind": kind, "raw_response_body": "limit details"})}
        return response

    client.create_response = preserved
    model, _, _ = _build(client, FakeSandbox([]))
    with pytest.raises(LimitsExceeded if kind == "context_window_exceeded" else RuntimeError):
        await model.query([])
    assert len(model.responses) == 1 and model.responses[0].metadata["upstream_error"]


async def test_wall_limit_and_archive_write(tmp_path, monkeypatch):
    model, _, agent = _build(FakeClient([]), FakeSandbox([]), wall_time_limit_seconds=1)
    monkeypatch.setattr(mini.time, "time", lambda: agent._start_time + 2)
    result = await agent.run("synthetic")
    assert result["exit_status"] == "TimeExceeded"
    target = tmp_path / "nested/trajectory.json"
    serialized = agent.save(target)
    assert json.loads(target.read_text()) == serialized
    assert agent.save(None) == serialized
    assert not model.responses


async def test_command_identity_check_is_atomic_and_keeps_exit97():
    calls = []

    async def execute(command, **kwargs):
        calls.append(command)
        return _ok("", 97)

    sandbox = SimpleNamespace(exec=execute)
    assert (await checked_exec(sandbox, "exit 97", "owned-0")).return_code == 97
    assert calls[-1].index("/bin/uname -n") < calls[-1].index("exit 97", calls[-1].index("fi\n"))

    async def mismatch(command, **kwargs):
        return SimpleNamespace(
            stdout="", stderr="MINISWE_SANDBOX_IDENTITY_MISMATCH expected=owned-0 actual=other-0", return_code=97
        )

    sandbox.exec = mismatch
    with pytest.raises(SandboxIdentityMismatch, match="actual=other-0"):
        await checked_exec(sandbox, "touch /app/file", "owned-0")


async def test_identity_mismatch_does_not_start_verifier(tmp_path, monkeypatch):
    agent, request, body, resource, sandbox, _, created, calls = build(tmp_path, monkeypatch)

    async def mismatch(command, **kwargs):
        raise SandboxIdentityMismatch("wrong sandbox")

    sandbox.exec = mismatch
    result = await agent.run(request, body)
    assert result.mini_swe_exit_status == "SandboxIdentityMismatch"
    assert result.evaluation_completed is False
    assert result.mini_swe_error == "wrong sandbox"
    assert [x["role"] for x in created] == ["agent"]
    assert calls[-1] == "/cancel_session" and not resource._sessions


async def test_runtime_exception_and_unwritable_capture_are_explicit(tmp_path, monkeypatch):
    agent, request, body, _, sandbox, _, _, _ = build(tmp_path, monkeypatch)
    original = sandbox.exec

    async def broken(command, **kwargs):
        if command.startswith("uname"):
            raise RuntimeError("synthetic sandbox failure")
        return await original(command, **kwargs)

    sandbox.exec = broken
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("file")
    agent.config.dump_trajectory_dir = str(blocked)
    result = await agent.run(request, body)
    assert result.mini_swe_exit_status == "RuntimeError"
    assert result.mini_swe_error == "synthetic sandbox failure"
    assert result.mini_swe_n_model_calls == 0
    assert result.evaluation_completed is True and result.reward == 0


async def test_user_and_tool_environment_errors_are_preserved():
    sandbox = FakeSandbox(
        [("bad", SimpleNamespace(stdout="partial", stderr="server failed", return_code=1, error_type="transport"))]
    )
    env = mini.NeMoGymSandboxShellEnvironment(sandbox, timeout=30, env={}, shell="/bin/sh", user=1000)
    output = await env.execute({"command": "bad"})
    assert output["extra"]["exception_type"] == "SandboxExecError"
    assert output["output"] == "partial"
    assert sandbox.calls[-1][1]["user"] == 1000
    model, _, agent = _build(FakeClient([]), FakeSandbox([]))

    async def broken_query(**kwargs):
        raise RuntimeError("model unavailable")

    model._client.create_response = broken_query
    with pytest.raises(RuntimeError, match="model unavailable"):
        await agent.run("synthetic")
    assert agent.messages[-1]["extra"]["exit_status"] == "RuntimeError"
