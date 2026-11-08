# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The flat /run path borrows the resource sandbox without changing its verifier protocol."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException, Request

from nemo_gym.config import SKILLS_REF_KEY_NAME
from nemo_gym.core import encode_model_sampling
from nemo_gym.server_runtime import CHECKPOINT_CLOSE_FOLLOWS_HEADER
from responses_api_agents.claude_code_agent import app
from responses_api_agents.claude_code_agent.sandbox import ClaudeCodeArtifacts, ClaudeCodeSandboxSession
from responses_api_agents.claude_code_agent.tests.test_app import FakeAioHTTPResponse
from responses_api_agents.claude_code_agent.tests.test_sandbox import agent, result


@pytest.fixture
def run_case(monkeypatch, tmp_path):
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"env": {"DISABLE_COMPACT": "1", "CLAUDE_CODE_MAX_CONTEXT_TOKENS": "32768"}}))
    instance = agent(
        resources_server={"type": "resources_servers", "name": "tasks"},
        model_server={"type": "responses_api_models", "name": "policy"},
        sandbox_provider="sandbox",
        concurrency=1,
        settings=str(settings),
        thinking="enabled",
    )
    instance.server_client.global_config_dict.update(
        policy={"responses_api_models": {"vllm_model": {"host": "model.example", "port": 8000}}},
        tasks={"resources_servers": {"swe_rebench": {"host": "resources.example", "port": 8001}}},
        token_id_capture={"enabled": True, "all_agents": True},
        skills={"path": "must-not-replace-run-skills"},
    )
    instance.server_client._build_server_base_url.side_effect = lambda cfg: f"http://{cfg['host']}:{cfg['port']}"
    request = Request({"type": "http", "headers": [(b"cookie", b"upstream=kept")]})
    body = app.ClaudeCodeAgentRunRequest.model_validate(
        {
            "responses_create_params": {"input": [{"role": "user", "content": "Repair the task"}]},
            "_ng_rollout_id": "task-rollout-a2",
            "_ng_model_sampling": {"temperature": 1.0, "top_p": 1.0, "max_output_tokens": 1024},
            "instance_id": "owner__repository-12",
            SKILLS_REF_KEY_NAME: {"path": "run-specific-skills"},
        }
    )
    context = SimpleNamespace(
        instance=instance,
        request=request,
        body=body,
        events=[],
        payload=None,
        state=None,
        failure=None,
        close_error=False,
        seed={"sandbox_descriptor": {"sandbox_id": "task-sandbox", "workdir": "/task/repo"}},
    )
    remote = SimpleNamespace(exec=AsyncMock(return_value=result()), upload=AsyncMock(), start=AsyncMock())
    provider = SimpleNamespace(name="local", aclose=AsyncMock())
    context.remote = remote
    context.provider = provider

    async def connect(descriptor, *, provider):
        context.events.append("connect")
        assert descriptor == context.seed["sandbox_descriptor"]
        assert provider is context.provider
        return remote

    constructor = MagicMock(connect=AsyncMock(side_effect=connect))
    monkeypatch.setattr(app, "AsyncSandbox", constructor)
    monkeypatch.setattr(app, "create_provider", lambda config: provider)

    async def install(*args, **kwargs):
        context.events.append("install")
        assert args == (remote,)
        assert kwargs["version"] == "2.1.108"
        if context.failure == "install":
            raise RuntimeError("install failed")

    monkeypatch.setattr(app, "install_sandbox_runtime", install)

    async def execute(state, payload, **kwargs):
        context.events.append("execute")
        context.state = state
        context.payload = payload
        if context.failure == "execute":
            raise RuntimeError("execute failed")
        if context.failure == "cancel":
            raise asyncio.CancelledError
        state.session.cleanup = {"timed_out": False, "return_code": 0, "cleanup_confirmed": True}
        return ClaudeCodeArtifacts(
            events="\n".join(
                json.dumps(event)
                for event in [
                    {
                        "type": "assistant",
                        "message": {"id": "served-root", "content": [{"type": "text", "text": "done"}]},
                    },
                    {
                        "type": "result",
                        "subtype": "success",
                        "is_error": False,
                        "usage": {"input_tokens": 9, "output_tokens": 1},
                    },
                ]
            )
        )

    async def close(state, timeout):
        context.events.append("agent_close")
        if context.failure == "agent_close":
            raise RuntimeError("agent close failed")

    monkeypatch.setattr(ClaudeCodeSandboxSession, "execute", execute)
    context.close = ClaudeCodeSandboxSession.close
    monkeypatch.setattr(ClaudeCodeSandboxSession, "close", close)
    context.host = AsyncMock(side_effect=AssertionError("sandbox task must not execute on the host"))
    monkeypatch.setattr(app.ClaudeCodeAgent, "_create_response", context.host)

    async def post(*, server_name, url_path, json, cookies, headers=None):
        context.events.append(url_path)
        assert server_name == "tasks"
        assert cookies["upstream"] == "kept"
        if url_path == "/seed_session":
            return FakeAioHTTPResponse(context.seed, {"session": "resource-owner"})
        assert cookies["session"] == "resource-owner"
        if url_path == "/verify":
            assert headers == {CHECKPOINT_CLOSE_FOLLOWS_HEADER: "1"}
            assert context.events[-2] == "agent_close"
            assert json["instance_id"] == "owner__repository-12"
            if context.failure == "verify":
                raise RuntimeError("verify failed")
            return FakeAioHTTPResponse(json | {"reward": 1.0, "evaluation_completed": True}, {"verifier": "kept"})
        assert url_path == "/close_session"
        assert json == {}
        if context.failure is None:
            assert cookies["verifier"] == "kept"
        if context.close_error:
            raise RuntimeError("resource close failed")
        return FakeAioHTTPResponse({"resources_session_id": "resource-owner"})

    instance.server_client.post = AsyncMock(side_effect=post)
    return context


@pytest.mark.asyncio
async def test_run_uses_seeded_workspace_and_the_session_runtime_without_reacquiring_concurrency(run_case):
    case = run_case
    case.seed["mcp"] = {"server_name": "tools", "url_path": "/mcp", "headers": {"Authorization": "fixture"}}
    output = await asyncio.wait_for(case.instance.run(case.request, case.body), timeout=1)
    assert output.reward == 1
    assert output.response.id == "served-root"
    assert output.response.output[0].content[0].text == "done"
    assert output.ng_agent_observations is not None
    assert case.events == [
        "/seed_session",
        "connect",
        "install",
        "execute",
        "agent_close",
        "/verify",
        "/close_session",
    ]
    case.host.assert_not_awaited()
    case.remote.start.assert_not_awaited()
    assert case.state.session.owns_sandbox is False
    assert case.state.request.episode_id.capture_key == "task-rollout-a2"
    assert case.payload["cwd"] == "/task/repo"
    assert case.payload["command"][0] == "/tmp/nemo-gym-claude-runtime-2.1.108/claude"
    assert case.payload["settings"]["env"]["DISABLE_COMPACT"] == "1"
    assert case.payload["settings"]["env"]["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "32768"
    assert case.payload["skills_path"] == "run-specific-skills"
    assert case.payload["mcp_config"]["mcpServers"]["tools"]["headers"] == {"Authorization": "fixture"}
    assert case.payload["env"]["ANTHROPIC_BASE_URL"] == (
        "http://model.example:8000/ng-rollout/task-rollout-a2/training-token-capture/ng-sampling/"
        + encode_model_sampling(case.body.model_sampling)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["install", "execute", "verify", "agent_close", "cancel"])
async def test_failed_run_closes_resources_and_never_falls_back_to_host(run_case, failure):
    case = run_case
    case.failure = failure
    error = asyncio.CancelledError if failure == "cancel" else RuntimeError
    with pytest.raises(error):
        await asyncio.wait_for(case.instance.run(case.request, case.body), timeout=1)
    assert case.events[-1] == "/close_session"
    assert "agent_close" in case.events
    if failure != "verify":
        assert "/verify" not in case.events
    case.host.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "seed",
    [
        {"sandbox_handle": "task-sandbox", "workdir": "/task/repo"},
        {"sandbox_descriptor": {}},
        {"sandbox_descriptor": {"sandbox_id": "task-sandbox"}},
    ],
)
async def test_unusable_sandbox_seed_is_rejected_and_resource_session_closed(run_case, seed):
    case = run_case
    case.seed = seed
    case.failure = "invalid_seed"
    with pytest.raises(HTTPException) as error:
        await case.instance.run(case.request, case.body)
    assert error.value.status_code == 422
    assert case.events == ["/seed_session", "/close_session"]
    case.host.assert_not_awaited()


@pytest.mark.asyncio
async def test_resource_cleanup_failure_does_not_return_a_successful_reward(run_case):
    case = run_case
    case.close_error = True
    with pytest.raises(RuntimeError, match="resource close failed"):
        await case.instance.run(case.request, case.body)
    assert case.events[-2:] == ["/verify", "/close_session"]


@pytest.mark.asyncio
async def test_failed_runtime_cleanup_is_retried_before_resource_close(run_case, monkeypatch):
    case = run_case
    case.failure = "install"
    closes = 0

    async def close(state, timeout):
        nonlocal closes
        closes += 1
        case.events.append("agent_close")
        if closes == 1:
            raise RuntimeError("transport release failed")

    monkeypatch.setattr(ClaudeCodeSandboxSession, "close", close)
    with pytest.raises(RuntimeError, match="install failed"):
        await case.instance.run(case.request, case.body)
    assert closes == 2
    assert case.events[-3:] == ["agent_close", "agent_close", "/close_session"]


@pytest.mark.asyncio
async def test_a_seed_without_a_sandbox_retains_host_execution_and_verification(run_case, monkeypatch):
    case = run_case
    case.body.capture_rollout_id = None
    response = case.instance._response_from_output(case.body.responses_create_params, [], "model", {})
    host = AsyncMock(return_value=response)
    monkeypatch.setattr(app.ClaudeCodeAgent, "_create_response", host)

    async def post(*, server_name, url_path, json, cookies, **kwargs):
        case.events.append(url_path)
        if url_path == "/seed_session":
            return FakeAioHTTPResponse({})
        assert url_path == "/verify"
        assert "headers" not in kwargs
        return FakeAioHTTPResponse(json | {"reward": 1.0})

    case.instance.server_client.post = AsyncMock(side_effect=post)
    output = await case.instance.run(case.request, case.body)
    assert output.reward == 1
    assert case.events == ["/seed_session", "/verify"]
    host.assert_awaited_once()
    assert host.await_args.kwargs["skills_path"] == "run-specific-skills"


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [b'{"sandbox_descriptor":', b"[]", b"null"])
async def test_undecodable_or_non_object_seed_closes_the_owner_with_returned_cookies(run_case, payload):
    case = run_case

    async def post(*, server_name, url_path, json, cookies, **kwargs):
        case.events.append(url_path)
        if url_path == "/seed_session":
            response = FakeAioHTTPResponse({}, {"session": "allocated-owner"})
            response.read = AsyncMock(return_value=payload)
            return response
        assert url_path == "/close_session"
        assert json == {}
        assert cookies == {"upstream": "kept", "session": "allocated-owner"}
        return FakeAioHTTPResponse({"resources_session_id": "allocated-owner"})

    case.instance.server_client.post = AsyncMock(side_effect=post)
    with pytest.raises(ValueError):
        await case.instance.run(case.request, case.body)
    assert case.events == ["/seed_session", "/close_session"]
    case.host.assert_not_awaited()
    case.remote.exec.assert_not_awaited()


@pytest.mark.asyncio
async def test_borrowed_connection_is_disconnected_before_verification(run_case, monkeypatch):
    case = run_case
    case.remote.disconnect = AsyncMock(side_effect=lambda: case.events.append("disconnect"))
    case.remote.stop = AsyncMock()

    async def close(state, timeout):
        await case.close(state, timeout)
        case.events.append("agent_close")

    monkeypatch.setattr(ClaudeCodeSandboxSession, "close", close)
    await case.instance.run(case.request, case.body)
    case.remote.disconnect.assert_awaited_once()
    case.remote.stop.assert_not_awaited()
    assert case.events.index("disconnect") < case.events.index("/verify") < case.events.index("/close_session")
