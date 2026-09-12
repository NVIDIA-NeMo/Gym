# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the mini-SWE loop through the real TB4 artifact/verifier lifecycle."""

import asyncio
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import responses_api_agents.mini_swe_agent_sandboxed_agent.app as mini
from nemo_gym.sandbox import SandboxExecResult
from resources_servers.terminal_bench_4.app import TerminalBench4SeedSessionRequest, TerminalBench4VerifyRequest
from resources_servers.terminal_bench_4.tests.test_app import (
    FakeSandbox,
    fake_request,
    make_server,
    make_task_dir,
    row,
    verifier_with_tests,
    wire_sandboxes,
    write_reward,
)
from responses_api_agents.mini_swe_agent_sandboxed_agent.tests.test_reference import FakeClient, _bash, _reasoning


class AgentSandbox(FakeSandbox):
    async def exec(self, command, **kwargs):
        if command.startswith(": ng-tb4-"):
            return await super().exec(command, **kwargs)
        assert not self.stopped, "agent actions must finish before teardown"
        assert kwargs.get("user") is None, "retain the native TB4 image user"
        if command.startswith("uname"):
            return SandboxExecResult(stdout="Linux\n6.5\nSMP\nx86_64\nsb-agent-0\n", stderr="", return_code=0)
        if "printf 42" in command:
            self.files["/app/answer.txt"] = b"42"
            return SandboxExecResult(stdout="", stderr="", return_code=0)
        if mini.SUBMIT_MARKER in command:
            return SandboxExecResult(stdout=mini.SUBMIT_MARKER + "\ndone\n", stderr="", return_code=0)
        raise AssertionError(command)


def build(tmp_path, monkeypatch, *, connect_failure=False, verify_failure=False, slow_model=False):
    task = make_task_dir(tmp_path, artifacts='["/app/answer.txt"]')
    resource = make_server(tmp_path)
    sandbox = AgentSandbox(name="agent", files={"/app/private.txt": b"not an artifact"})

    def grade(verifier):
        assert sandbox.stopped
        assert "/app/private.txt" not in verifier.files
        assert "/tests/test.sh" not in sandbox.files
        reward = "1" if verifier.files.get("/app/answer.txt") == b"42" else "0"
        return write_reward(reward)(verifier)

    verifier = verifier_with_tests(on_run_tests=grade)
    created = wire_sandboxes(resource, sandbox, verifier)
    request = fake_request()
    payload = row(task) | {
        "agent_timeout_sec": 12,
        "responses_create_params": {"input": "Write 42 to /app/answer.txt"},
    }
    body = mini.MiniSweAgentRunRequest.model_validate(payload)

    async def request_json():
        return payload

    request.json = request_json
    calls = []

    async def post(*, server_name, url_path, json, cookies):
        calls.append(url_path)
        if url_path == "/seed_session":
            result = await resource.seed_session(request, TerminalBench4SeedSessionRequest.model_validate(json))
        else:
            assert cookies["seed"] == "retained"
            if url_path == "/cancel_session":
                result = await resource.cancel_session(request)
            else:
                assert url_path == "/verify"
                if verify_failure:
                    raise RuntimeError("verifier transport failure")
                result = await resource.verify(request, TerminalBench4VerifyRequest.model_validate(json))

        async def response_json():
            return result.model_dump(mode="json") if hasattr(result, "model_dump") else result

        return SimpleNamespace(cookies={"seed": "retained"}, json=response_json)

    async def ok(_):
        pass

    async def get_json(response):
        return await response.json()

    monkeypatch.setattr(mini, "raise_for_status", ok)
    monkeypatch.setattr(mini, "get_response_json", get_json)
    monkeypatch.setattr(mini, "get_server_url", lambda _: "http://model.invalid")
    model = FakeClient(
        [
            [_reasoning("write the artifact", 1), _bash("printf 42 > /app/answer.txt", 1)],
            [_bash("echo " + mini.SUBMIT_MARKER, 2)],
        ]
    )
    if slow_model:

        async def slow(**kwargs):
            await asyncio.sleep(10)

        model.create_response = slow
    monkeypatch.setattr(mini, "NeMoGymAsyncOpenAI", lambda **kwargs: model)
    config = mini.MiniSweAgentSandboxedConfig(
        host="",
        port=0,
        entrypoint="",
        name="mini_test",
        resources_server={"type": "resources_servers", "name": "tb4_test"},
        model_server={"type": "responses_api_models", "name": "policy_model"},
        sandbox_provider="",
        sandbox_timeout=10800,
        cancel_session_on_error=True,
        dump_trajectory_dir=str(tmp_path / "trajectories"),
        replay_reasoning_items=True,
    )
    agent = mini.MiniSweAgentSandboxedAgent(config=config, server_client=resource.server_client)
    agent.server_client = SimpleNamespace(post=post)

    async def connect(_):
        if connect_failure:
            raise RuntimeError("connect failure")
        return sandbox

    agent._connect_sandbox = connect
    return agent, request, body, resource, sandbox, verifier, created, calls


async def test_mini_swe_separate_verifier_and_full_capture(tmp_path, monkeypatch):
    agent, request, body, resource, sandbox, verifier, created, calls = build(tmp_path, monkeypatch)
    result = await agent.run(request, body)
    assert result.reward == 1 and result.evaluation_completed
    assert result.mini_swe_completed and result.mini_swe_submission == "done\n"
    assert result.mini_swe_agent_timeout_s == 12
    assert result.response.usage.output_tokens == 10
    assert result.response.usage.output_tokens_details.reasoning_tokens == 6
    assert result.mini_swe_shell_records[-1]["output"].endswith("done\n")
    assert [x["role"] for x in created] == ["agent", "verifier"]
    assert sandbox.stopped and verifier.stopped
    assert not resource._sessions and not agent._session_sandboxes
    assert calls == ["/seed_session", "/verify"]


@pytest.mark.parametrize("failure", ["connect_failure", "verify_failure"])
async def test_failure_releases_seeded_tb4_session(tmp_path, monkeypatch, failure):
    agent, request, body, resource, sandbox, _, created, calls = build(tmp_path, monkeypatch, **{failure: True})
    with pytest.raises(RuntimeError, match="failure"):
        await agent.run(request, body)
    assert calls[-1] == "/cancel_session"
    assert not resource._sessions and not agent._session_sandboxes
    assert sandbox.stopped
    assert [x["role"] for x in created] == ["agent"]


async def test_row_timeout_still_runs_separate_verifier(tmp_path, monkeypatch):
    agent, request, body, _, sandbox, verifier, _, _ = build(tmp_path, monkeypatch, slow_model=True)
    body.agent_timeout_sec = 0.01
    result = await agent.run(request, body)
    assert result.mini_swe_exit_status == "SandboxTimeout"
    assert not result.mini_swe_completed
    assert result.reward == 0 and result.evaluation_completed
    assert sandbox.stopped and verifier.stopped


@pytest.mark.parametrize("budget", [0, -1, float("inf"), float("nan")])
def test_invalid_row_budgets_rejected(budget):
    with pytest.raises(ValidationError):
        mini.MiniSweAgentRunRequest.model_validate(
            {"agent_timeout_sec": budget, "responses_create_params": {"input": "x"}}
        )


async def test_cancel_is_cookie_scoped_and_idempotent(tmp_path):
    server = make_server(tmp_path)
    task = make_task_dir(tmp_path)
    sandbox = AgentSandbox(name="agent")
    wire_sandboxes(server, sandbox, verifier_with_tests())
    request = fake_request("owned")
    await server.seed_session(request, TerminalBench4SeedSessionRequest.model_validate(row(task)))
    assert not (await server.cancel_session(fake_request("other")))["session_found"]
    assert not sandbox.stopped
    assert (await server.cancel_session(request))["agent_sandbox_stopped"]
    assert not (await server.cancel_session(request))["session_found"]
