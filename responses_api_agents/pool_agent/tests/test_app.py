# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise session binding, setup failures, replay, and trajectory conversion without model compute."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from pydantic import ValidationError
from starlette.requests import Request

from nemo_gym.agent_utils.sandbox_session import SandboxSession
from nemo_gym.base_responses_api_agent import AgentSeedSessionRequest, AgentSessionSetupError
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox.providers.base import SandboxExecResult
from nemo_gym.server_utils import ServerClient
from responses_api_agents.pool_agent.app import PoolAgent, PoolAgentConfig, prompt_from_request
from responses_api_agents.pool_agent.configuration import pool_config
from responses_api_agents.pool_agent.sandbox import PoolArtifacts, PoolSession
from responses_api_agents.pool_agent.trajectory import parse_trajectory


APP = "responses_api_agents.pool_agent.app"


def agent(**kwargs) -> PoolAgent:
    return PoolAgent(
        config=PoolAgentConfig(
            host="localhost",
            port=8080,
            name="pool_agent",
            entrypoint="app.py",
            model="test",
            model_server={"type": "responses_api_models", "name": "policy_model"},
            **kwargs,
        ),
        server_client=MagicMock(spec=ServerClient),
    )


def seed(*, borrowed=True, workdir="/repo", tools=None) -> AgentSeedSessionRequest:
    return AgentSeedSessionRequest(
        agent_session_id="session-1",
        episode_id={"rollout_id": "rollout-1", "attempt": 2},
        task_id={"taskset": "test", "task_id": "task"},
        tool_accesses=tools or [],
        sandbox_access={
            "connection": {"kind": "direct", "provider_config_ref": "sandbox", "descriptor": {"id": "task"}},
            "workdir": workdir,
        }
        if borrowed
        else None,
    )


def request(rollout="rollout-1-a2") -> Request:
    return Request(
        {"type": "http", "session": {"agent_session_id": "session-1"}, "path_params": {"rollout_id": rollout}}
    )


def state() -> PoolSession:
    return PoolSession(
        request=seed(),
        provider_name="docker",
        sandbox_id="task",
        session=SandboxSession(
            sandbox=AsyncMock(),
            session_dir="/tmp/nemo-gym-pool-sessions/test",
            workdir="/repo",
            harness="Pool",
        ),
    )


def events(*items) -> str:
    return "\n".join(json.dumps(item) for item in items)


def event(kind, **payload) -> dict:
    return {"type": kind, kind.replace(".", "_"): payload}


def complete() -> str:
    return events(
        event("assistant_message.end", assistant_message="done"), event("session.exit", reason="exit_tool_called")
    )


def test_rejected_tool_call_uses_recorded_validation_error():
    parsed = parse_trajectory(
        events(
            event(
                "tool_call.parsed",
                id="invalid",
                name="shell",
                raw_args='{"timeout":"120"}',
                validation_error={"detail": 'validating root: unexpected additional properties ["timeout"]'},
            ),
            event("session.exit", reason="exit_tool_called"),
        )
    )
    assert not parsed.gaps
    assert parsed.output[0].arguments == '{"timeout":"120"}'
    assert parsed.output[1].call_id == "invalid"
    assert parsed.output[1].output == 'validating root: unexpected additional properties ["timeout"]'


def test_tool_ids_survive_out_of_order_results_and_errors():
    parsed = parse_trajectory(
        events(
            event("tool_call.parsed", id="a", name="shell", args={"command": "false"}),
            event("tool_call.parsed", id="b", name="read", args={"path": "x"}),
            event("tool_call.result", id="b", observation="contents"),
            event("tool_call.result", id="a", observation="exit 1", is_error=True),
            event("session.exit", reason="exit_tool_called"),
        )
    )
    assert [item.call_id for item in parsed.output] == ["a", "b", "b", "a"]
    assert parsed.output[-1].output == "exit 1"
    assert not parsed.gaps


@pytest.mark.parametrize("separator", ["\x85", "\u2028", "\u2029"])
def test_unicode_line_separators_inside_tool_output_are_preserved(separator):
    observation = f"before{separator}after"
    text = "\n".join(
        json.dumps(item, ensure_ascii=False)
        for item in (
            event("tool_call.parsed", id="a", name="read", args={"path": "x"}),
            event("tool_call.result", id="a", observation=observation),
            event("session.exit", reason="exit_tool_called"),
        )
    )
    parsed = parse_trajectory(text)
    assert parsed.output[1].output == observation
    assert not parsed.gaps


def test_truncated_output_never_fabricates_successful_tool_result():
    parsed = parse_trajectory(events(event("tool_call.parsed", id="a", name="shell", args={})) + '\n{"type":')
    assert len(parsed.output) == 1
    assert set(parsed.gaps) == {"truncated_trajectory_event", "tool_result_unavailable", "session_exit_unavailable"}


def test_reasoning_is_separate_and_raw_arguments_preserved():
    parsed = parse_trajectory(
        events(
            event("thought.end", thought="reasoning"),
            event("assistant_message.end", assistant_message="answer"),
            event("tool_call.parsed", id="a", name="read", args={"path": "x"}, raw_args='{ "path" : "x" }'),
        )
    )
    assert parsed.output[0].type == "reasoning"
    assert parsed.output[1].content[0].text == "answer"
    assert parsed.output[2].arguments == '{ "path" : "x" }'


@pytest.mark.parametrize(
    "text",
    [
        "bad\n{}",
        "[]",
        events(event("tool_call.result", id="missing")),
        events(event("tool_call.parsed", id="a", name="read"), event("tool_call.parsed", id="a", name="read")),
    ],
)
def test_malformed_or_ambiguous_trajectory_is_rejected(text):
    with pytest.raises(ValueError):
        parse_trajectory(text)


def test_duplicate_persisted_event_is_not_counted_twice():
    item = {"id": "event-1", **event("assistant_message.end", assistant_message="once")}
    assert len(parse_trajectory(events(item, item)).output) == 1


def test_config_preserves_rollout_routing_and_disables_streaming():
    config = pool_config(base_url="http://model/ng-rollout/r-a2/v1", model="test", context_window=32000, timeout=21600)
    provider = config["model"]["provider"]["openai"]
    assert provider["base_url"] == "http://model/ng-rollout/r-a2/v1"
    assert provider["use_streaming"] is False
    assert config["http_timeout"] == "21600s"
    assert config["memory"]["compact"]["TriggerCompressionTokenCount"] == 25600


@pytest.mark.parametrize("version", ["latest", "v1.0.16; echo injected", ""])
def test_version_must_be_pinned(version):
    with pytest.raises(ValidationError):
        agent(pool_version=version)


@pytest.mark.parametrize(
    "body",
    [
        {"input": ""},
        {"input": "x", "temperature": 0.3},
        {"input": [{"role": "assistant", "content": "old"}, {"role": "user", "content": "next"}]},
        {
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_image", "image_url": "http://example.com/a", "detail": "auto"}],
                }
            ]
        },
    ],
)
def test_unsupported_inputs_fail_before_execution(body):
    with pytest.raises(HTTPException) as error:
        prompt_from_request(NeMoGymResponseCreateParamsNonStreaming(**body))
    assert error.value.status_code == 422


def test_text_parts_and_instructions_are_preserved():
    body = NeMoGymResponseCreateParamsNonStreaming(
        input=[
            {"role": "developer", "content": "policy"},
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "first"}, {"type": "input_text", "text": "second"}],
            },
        ],
        instructions="instructions",
    )
    assert prompt_from_request(body) == "instructions\n\npolicy\n\nfirst\n\nsecond"


@pytest.mark.asyncio
@pytest.mark.parametrize("workdir", ["/", "/tmp", "/tmp/nemo-gym-pool-sessions", "/repo/../tmp"])
async def test_overlapping_workdir_rejected_before_connect(workdir):
    with patch(APP + ".create_provider") as create, pytest.raises(HTTPException):
        await agent()._seed_agent_session_state(seed(workdir=workdir))
    create.assert_not_called()


@pytest.mark.asyncio
async def test_required_tool_grant_rejected_before_connect():
    access = {"kind": "direct_http", "name": "tools", "required": True, "base_url": "http://tools"}
    with patch(APP + ".create_provider") as create, pytest.raises(HTTPException):
        await agent()._seed_agent_session_state(seed(tools=[access]))
    create.assert_not_called()


@pytest.mark.asyncio
async def test_borrowed_setup_failure_disconnects_without_destroying_sandbox():
    sandbox = AsyncMock()
    sandbox.sandbox_id = "task"
    sandbox.exec.return_value = SandboxExecResult(stdout="", stderr="", return_code=0)
    provider = MagicMock(name="provider")
    provider.name = "docker"
    with (
        patch(APP + ".get_global_config_dict", return_value={}),
        patch(APP + ".resolve_provider_config"),
        patch(APP + ".create_provider", return_value=provider),
        patch(APP + ".AsyncSandbox.connect", return_value=sandbox),
        patch.object(PoolSession, "install", side_effect=RuntimeError("install failed")),
    ):
        with pytest.raises(RuntimeError, match="install failed"):
            await agent()._seed_agent_session_state(seed())
    sandbox.stop.assert_not_called()
    sandbox.disconnect.assert_awaited_once()


@pytest.mark.asyncio
async def test_failed_setup_cleanup_retains_state_for_retry():
    sandbox = AsyncMock()
    sandbox.sandbox_id = "task"
    sandbox.exec.return_value = SandboxExecResult(stdout="", stderr="", return_code=0)
    provider = MagicMock()
    provider.name = "docker"
    with (
        patch(APP + ".get_global_config_dict", return_value={}),
        patch(APP + ".resolve_provider_config"),
        patch(APP + ".create_provider", return_value=provider),
        patch(APP + ".AsyncSandbox.connect", return_value=sandbox),
        patch.object(PoolSession, "install", side_effect=RuntimeError("install failed")),
        patch.object(PoolSession, "close", side_effect=RuntimeError("disconnect failed")),
    ):
        with pytest.raises(AgentSessionSetupError) as error:
            await agent()._seed_agent_session_state(seed())
    assert isinstance(error.value.state, PoolSession)
    assert error.value.state.session.sandbox is sandbox


@pytest.mark.asyncio
async def test_owned_seed_merges_metadata_and_sets_ttl():
    sandbox = AsyncMock()
    sandbox.sandbox_id = "owned"
    sandbox.exec.return_value = SandboxExecResult(stdout="", stderr="", return_code=0)
    provider = MagicMock()
    provider.name = "docker"
    with (
        patch(APP + ".get_global_config_dict", return_value={}),
        patch(APP + ".resolve_provider_config"),
        patch(APP + ".resolve_provider_metadata", return_value={"x": "default", "y": "default"}),
        patch(APP + ".create_provider", return_value=provider),
        patch(APP + ".AsyncSandbox", return_value=sandbox),
        patch.object(PoolSession, "install", new_callable=AsyncMock),
    ):
        session = await agent(
            sandbox_provider="sandbox", sandbox_config={"image": "test", "metadata": {"x": "custom"}}
        )._seed_agent_session_state(seed(borrowed=False))
    spec = sandbox.start.call_args.args[0]
    assert spec.metadata == {"x": "custom", "y": "default", "agent_session_id": "session-1"}
    assert spec.ttl_s > 1800
    assert session.session.owns_sandbox


@pytest.mark.asyncio
async def test_identical_requests_share_activation_and_conflicting_retry_fails():
    server, session = agent(), state()
    body = NeMoGymResponseCreateParamsNonStreaming(input="task")
    response = MagicMock()
    with (
        patch.object(PoolAgent, "_require_agent_session", return_value=session),
        patch.object(PoolAgent, "_activate", new_callable=AsyncMock, return_value=response) as activate,
    ):
        await asyncio.gather(server.responses(request(), body), server.responses(request(), body))
        activate.assert_awaited_once()
        with pytest.raises(HTTPException) as error:
            await server.responses(request(), body.model_copy(update={"input": "different"}))
        assert error.value.status_code == 409


@pytest.mark.asyncio
async def test_disconnected_waiter_does_not_cancel_shared_activation():
    server, session = agent(), state()
    started, release = asyncio.Event(), asyncio.Event()

    async def activate(*args):
        started.set()
        await release.wait()
        return MagicMock()

    with (
        patch.object(PoolAgent, "_require_agent_session", return_value=session),
        patch.object(PoolAgent, "_activate", side_effect=activate),
    ):
        waiter = asyncio.create_task(
            server.responses(request(), NeMoGymResponseCreateParamsNonStreaming(input="task"))
        )
        await started.wait()
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert not session.task.done()
        release.set()
        await session.task


@pytest.mark.asyncio
async def test_wrong_rollout_cannot_start_activation():
    server, session = agent(), state()
    with (
        patch.object(PoolAgent, "_require_agent_session", return_value=session),
        pytest.raises(HTTPException) as error,
    ):
        await server.responses(request("another-rollout"), NeMoGymResponseCreateParamsNonStreaming(input="task"))
    assert error.value.status_code == 409
    assert session.task is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exit_code,reason,timed_out",
    [(1, "exit_tool_called", False), (0, "model_next_tool_call_http_error", False)],
)
async def test_runtime_failure_is_not_returned_as_a_gradable_response(exit_code, reason, timed_out):
    server, session = agent(), state()
    session.session.cleanup = {
        "cleanup_confirmed": True,
        "return_code": exit_code,
        "error": None,
        "timed_out": timed_out,
    }
    artifacts = PoolArtifacts(events(event("session.exit", reason=reason)), "", "", exit_code)
    with (
        patch.object(PoolAgent, "model_endpoint", return_value=MagicMock(base_url="http://model/v1")),
        patch.object(SandboxSession, "execute", return_value=artifacts),
        pytest.raises((RuntimeError, TimeoutError)),
    ):
        await server._activate(session, NeMoGymResponseCreateParamsNonStreaming(input="task"))
    assert session.failure


@pytest.mark.asyncio
async def test_successful_activation_uses_task_budget_and_explicit_unknown_usage():
    server, session = agent(), state()
    session.request.agent_timeout_sec = 12
    session.session.cleanup = receipt()
    with (
        patch.object(PoolAgent, "model_endpoint", return_value=MagicMock(base_url="http://model/v1")),
        patch.object(SandboxSession, "execute", return_value=PoolArtifacts(complete(), "", "", 0)) as execute,
    ):
        response = await server._activate(session, NeMoGymResponseCreateParamsNonStreaming(input="task"))
    assert execute.call_args.kwargs["timeout"] == 12
    assert response.output[0].content[0].text == "done"
    assert response.usage is None


@pytest.mark.asyncio
async def test_explicit_pool_task_failure_still_reaches_external_verifier():
    server, session = agent(), state()
    session.session.cleanup = receipt(return_code=4)
    with (
        patch.object(PoolAgent, "model_endpoint", return_value=MagicMock(base_url="http://model/v1")),
        patch.object(SandboxSession, "execute", return_value=PoolArtifacts(complete(), "", "", 4)),
    ):
        response = await server._activate(session, NeMoGymResponseCreateParamsNonStreaming(input="task"))
    assert response.status == "completed"


def test_partial_conversation_is_retained_after_runtime_failure():
    server, session = agent(), state()
    session.failure = "TimeoutError"
    session.activation_request = NeMoGymResponseCreateParamsNonStreaming(input="task")
    session.session.artifacts = PoolArtifacts(
        events(
            event("assistant_message.end", assistant_message="partial answer"),
            event("tool_call.parsed", id="unfinished", name="shell", args={}),
        ),
        "",
        "",
        None,
    )
    bundle = server._observations(session)
    invocation = bundle.records[0]
    assert invocation.status == "failed"
    assert len(invocation.conversation) == 3
    assert invocation.conversation[1].content[0].text == "partial answer"
    assert "tool_result_unavailable" in {gap.code for gap in bundle.gaps}


@pytest.mark.asyncio
async def test_failed_connect_releases_provider():
    provider = AsyncMock()
    with (
        patch(APP + ".get_global_config_dict", return_value={}),
        patch(APP + ".resolve_provider_config"),
        patch(APP + ".create_provider", return_value=provider),
        patch(APP + ".AsyncSandbox.connect", side_effect=RuntimeError("connection failed")),
    ):
        with pytest.raises(RuntimeError, match="connection failed"):
            await agent()._seed_agent_session_state(seed())
    provider.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_user_override_is_not_silently_ignored():
    body = seed()
    body.sandbox_access.user = "task-user"
    with patch(APP + ".create_provider") as create, pytest.raises(HTTPException):
        await agent()._seed_agent_session_state(body)
    create.assert_not_called()


def receipt(**overrides):
    return {"cleanup_confirmed": True, "return_code": 0, "error": None, "timed_out": False} | overrides


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["", events(event("tool_call.parsed", id="a", name="shell", args={})), complete()])
async def test_confirmed_timeout_keeps_partial_output_gradable(text):
    server, session = agent(), state()
    session.session.cleanup = receipt(timed_out=True, return_code=-15)
    artifacts = PoolArtifacts(text, "", "", None)
    session.session.artifacts = artifacts
    with (
        patch.object(PoolAgent, "model_endpoint", return_value=MagicMock(base_url="http://model/v1")),
        patch.object(SandboxSession, "execute", return_value=artifacts),
    ):
        response = await server._activate(session, NeMoGymResponseCreateParamsNonStreaming(input="task"))
    assert response.status == "incomplete"
    assert response.output == parse_trajectory(text).output
    assert session.failure is None
    assert server._observations(session).records[0].status == "incomplete"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cleanup,exit_code,text",
    [
        (None, 0, complete()),
        (receipt(cleanup_confirmed=False), 0, complete()),
        (receipt(error="spawn failed", timed_out=True), None, ""),
        (receipt(return_code=1), None, ""),
        (receipt(return_code=1), 0, complete()),
        (receipt(), None, complete()),
        (receipt(timed_out=True), None, events(event("session.error", error="HTTP 401"))),
        (receipt(timed_out=True), None, "bad\n{}"),
    ],
)
async def test_runtime_errors_cannot_be_disguised_as_gradable_partials(cleanup, exit_code, text):
    server, session = agent(), state()
    session.session.cleanup = cleanup
    with (
        patch.object(PoolAgent, "model_endpoint", return_value=MagicMock(base_url="http://model/v1")),
        patch.object(SandboxSession, "execute", return_value=PoolArtifacts(text, "", "", exit_code)),
        pytest.raises((RuntimeError, ValueError)),
    ):
        await server._activate(session, NeMoGymResponseCreateParamsNonStreaming(input="task"))
    assert session.failure


@pytest.mark.asyncio
async def test_owned_start_and_cleanup_failure_retain_state_for_retry():
    sandbox = AsyncMock()
    sandbox.sandbox_id = "allocated-before-upload-failed"
    sandbox.start.side_effect = RuntimeError("initial upload failed")
    sandbox.stop.side_effect = [RuntimeError("stop failed"), None]
    provider = MagicMock()
    provider.name = "docker"
    with (
        patch(APP + ".get_global_config_dict", return_value={}),
        patch(APP + ".resolve_provider_config"),
        patch(APP + ".resolve_provider_metadata", return_value={}),
        patch(APP + ".create_provider", return_value=provider),
        patch(APP + ".AsyncSandbox", return_value=sandbox),
        pytest.raises(AgentSessionSetupError) as error,
    ):
        await agent(sandbox_provider="sandbox")._seed_agent_session_state(seed(borrowed=False))
    retained = error.value.state
    assert retained.session.sandbox is sandbox
    await retained.close(timeout=1)
    assert sandbox.stop.await_count == 2
    assert retained.session.cleanup_confirmed


@pytest.mark.asyncio
async def test_cancelled_installer_waiter_does_not_release_or_duplicate_installation():
    session = state()
    started, release = asyncio.Event(), asyncio.Event()

    async def install(**kwargs):
        started.set()
        await release.wait()

    with (
        patch.object(PoolSession, "_install", side_effect=install) as installer,
        patch.object(SandboxSession, "close", new_callable=AsyncMock) as close,
    ):
        waiter = asyncio.create_task(session.install(version="v1.0.16", timeout=1))
        await asyncio.wait_for(started.wait(), timeout=1)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        with pytest.raises(TimeoutError):
            await session.close(timeout=0.01)
        close.assert_not_awaited()
        assert not session.installation_task.done()
        retry = asyncio.create_task(session.install(version="v1.0.16", timeout=1))
        release.set()
        await asyncio.wait_for(retry, timeout=1)
        await session.close(timeout=1)
    installer.assert_awaited_once()
    close.assert_awaited_once()


def test_child_observations_are_explicitly_unknown():
    session = state()
    session.session.artifacts = PoolArtifacts(complete(), "", "", 0)
    observations = agent()._observations(session)
    assert len([r for r in observations.records if r.kind == "agent_invocation"]) == 1
    assert "child_invocations_unavailable" in {g.code for g in observations.gaps}
