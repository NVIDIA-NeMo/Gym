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
    kwargs.setdefault("failure_artifacts_dir", None)
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
        await asyncio.wait_for(release.wait(), timeout=5)
        return MagicMock()

    with (
        patch.object(PoolAgent, "_require_agent_session", return_value=session),
        patch.object(PoolAgent, "_activate", side_effect=activate),
    ):
        waiter = asyncio.create_task(
            server.responses(request(), NeMoGymResponseCreateParamsNonStreaming(input="task"))
        )
        await asyncio.wait_for(started.wait(), timeout=5)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(waiter, timeout=5)
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
@pytest.mark.parametrize("text", [events(event("tool_call.parsed", id="a", name="shell", args={})), complete()])
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
        await asyncio.wait_for(release.wait(), timeout=5)

    with (
        patch.object(PoolSession, "_install", side_effect=install) as installer,
        patch.object(SandboxSession, "close", new_callable=AsyncMock) as close,
    ):
        waiter = asyncio.create_task(session.install(version="v1.0.16", timeout=1))
        await asyncio.wait_for(started.wait(), timeout=1)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(waiter, timeout=5)
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


@pytest.mark.asyncio
@pytest.mark.parametrize("budget,expected", [("12", 12), ("120", 60), ("nan", 60), ("0", 60), ("bad", 60)])
async def test_materialized_budget_caps_execution_and_model_requests(budget, expected):
    server, session = agent(), state()
    session.request.agent_timeout_sec = 60
    session.session.cleanup = receipt()
    body = NeMoGymResponseCreateParamsNonStreaming(input="task", metadata={"agent_timeout_sec": budget})
    assert prompt_from_request(body) == "task"

    async def execute(**kwargs):
        await kwargs["stage_activation"]()
        return PoolArtifacts(complete(), "", "", 0)

    with (
        patch.object(PoolAgent, "model_endpoint", return_value=MagicMock(base_url="http://model/v1")),
        patch.object(SandboxSession, "execute", side_effect=execute) as run,
        patch.object(PoolSession, "stage", new_callable=AsyncMock) as stage,
    ):
        await server._activate(session, body)
    assert run.call_args.kwargs["timeout"] == expected
    assert float(json.loads(stage.call_args.kwargs["config"])["http_timeout"][:-1]) == expected


def test_custom_metadata_stays_unsupported():
    with pytest.raises(HTTPException, match="metadata"):
        prompt_from_request(NeMoGymResponseCreateParamsNonStreaming(input="task", metadata={"custom": "x"}))


@pytest.mark.asyncio
async def test_recorded_model_failure_overrides_timeout_and_preserves_diagnostics(caplog):
    server, session = agent(), state()
    session.session.cleanup = receipt(timed_out=True, return_code=-15)
    artifacts = PoolArtifacts(
        events(event("session.exit", reason="model_next_tool_call_http_error")), "", "upstream HTTP 503", None
    )
    session.session.artifacts = artifacts
    with (
        patch.object(PoolAgent, "model_endpoint", return_value=MagicMock(base_url="http://model/v1")),
        patch.object(SandboxSession, "execute", return_value=artifacts),
        pytest.raises(RuntimeError, match="model request failed"),
    ):
        await server._activate(session, NeMoGymResponseCreateParamsNonStreaming(input="task"))
    assert session.failure == "RuntimeError"
    assert "upstream HTTP 503" in caplog.text


def fresh_request():
    req = request()
    req.scope["session"] = {}
    return req


@pytest.mark.asyncio
@pytest.mark.parametrize("owned", [False, True])
async def test_lifespan_closes_sessions_using_their_ownership(owned):
    server, session = agent(), state()
    session.session.owns_sandbox = owned
    session.session.sandbox.exec.return_value = SandboxExecResult(stdout="", stderr="", return_code=0)
    app = server.setup_webserver()
    with patch.object(PoolAgent, "_seed_agent_session_state", return_value=session):
        async with app.router.lifespan_context(app):
            await server.seed_agent_session(fresh_request(), session.request)
    assert session.session.sandbox.stop.await_count == int(owned)
    assert session.session.sandbox.disconnect.await_count == int(not owned)
    with pytest.raises(HTTPException):
        server._require_agent_session(session.request.agent_session_id)
    with pytest.raises(HTTPException) as error:
        await server.seed_agent_session(
            fresh_request(), seed().model_copy(update={"agent_session_id": "fresh-after-shutdown"})
        )
    assert error.value.status_code == 503
    with pytest.raises(HTTPException) as error:
        await server.responses(request(), NeMoGymResponseCreateParamsNonStreaming(input="task"))
    assert error.value.status_code == 503


@pytest.mark.asyncio
async def test_shutdown_continues_after_failed_close_and_retains_retry_state():
    from nemo_gym.base_responses_api_agent import AgentCloseSessionRequest

    server = agent()
    failed, healthy = state(), state()
    failed.session.owns_sandbox = healthy.session.owns_sandbox = True
    healthy.request.agent_session_id = "session-2"
    failed.session.sandbox.stop.side_effect = [RuntimeError("stop unavailable"), None]
    with patch.object(PoolAgent, "_seed_agent_session_state", side_effect=[failed, healthy]):
        await server.seed_agent_session(fresh_request(), failed.request)
        await server.seed_agent_session(fresh_request(), healthy.request)
    await server._shutdown_sessions()
    healthy.session.sandbox.stop.assert_awaited_once()
    assert server._session_records[failed.request.agent_session_id].state is failed
    await server.close_agent_session(
        fresh_request(),
        AgentCloseSessionRequest(
            agent_session_id=failed.request.agent_session_id, episode_id=failed.request.episode_id
        ),
    )
    assert failed.session.sandbox.stop.await_count == 2
    assert server._session_records[failed.request.agent_session_id].state is None


def test_harbor_materialized_request_passes_validation():
    from types import SimpleNamespace

    from nemo_gym.tasks.harbor.materialize import materialize_task

    task = SimpleNamespace(
        task_id="task",
        instruction="Inspect the repository.",
        digest="a" * 64,
        config=SimpleNamespace(agent=SimpleNamespace(timeout_sec=12)),
    )
    row = materialize_task(task, "test")
    body = NeMoGymResponseCreateParamsNonStreaming.model_validate(row.task_input.responses_create_params)
    assert prompt_from_request(body) == task.instruction
    assert body.metadata == {"agent_timeout_sec": "12"}


@pytest.mark.asyncio
async def test_collect_and_response_preserve_runner_identity():
    server, session = agent(), state()
    files = {
        "stdout.jsonl": "output",
        "stderr.log": "diagnostic",
        "exit.json": "0",
        "runtime.json": '{"hostname":"sandbox-worker","pid":123}',
        "trajectory.ndjson": complete(),
    }

    async def read(sandbox, *, path):
        return files[path.rsplit("/", 1)[-1]]

    session.session.sandbox.exec.return_value = SandboxExecResult(stdout="", stderr="", return_code=0)
    with patch("responses_api_agents.pool_agent.sandbox.read_text", side_effect=read):
        artifacts = await session.collect()
    assert artifacts.stderr == "diagnostic"
    session.session.cleanup = receipt()
    with (
        patch.object(PoolAgent, "model_endpoint", return_value=MagicMock(base_url="http://model/v1")),
        patch.object(SandboxSession, "execute", return_value=artifacts),
    ):
        response = await server._activate(session, NeMoGymResponseCreateParamsNonStreaming(input="task"))
    assert response.metadata["runner_hostname"] == "sandbox-worker"
    assert response.metadata["runner_pid"] == "123"


@pytest.mark.asyncio
async def test_shutdown_drains_failed_setup_state():
    server, session = agent(), state()
    session.session.owns_sandbox = True
    with (
        patch.object(
            PoolAgent,
            "_seed_agent_session_state",
            side_effect=AgentSessionSetupError(session, error=RuntimeError("installation failed")),
        ),
        pytest.raises(RuntimeError, match="installation failed"),
    ):
        await server.seed_agent_session(fresh_request(), session.request)
    await server._shutdown_sessions()
    session.session.sandbox.stop.assert_awaited_once()
    assert server._session_records[session.request.agent_session_id].state is None


@pytest.mark.asyncio
async def test_shutdown_during_installation_joins_before_stopping_sandbox():
    server = agent(sandbox_provider="sandbox", sandbox_config={"image": "test", "workdir": "/repo"})
    sandbox = AsyncMock()
    sandbox.sandbox_id = "owned"
    sandbox.exec.return_value = SandboxExecResult(stdout="", stderr="", return_code=0)
    provider = MagicMock()
    provider.name = "docker"
    entered, release = asyncio.Event(), asyncio.Event()

    async def install(*args, **kwargs):
        entered.set()
        await asyncio.wait_for(release.wait(), timeout=5)

    with (
        patch(APP + ".get_global_config_dict", return_value={}),
        patch(APP + ".resolve_provider_config"),
        patch(APP + ".resolve_provider_metadata", return_value={}),
        patch(APP + ".create_provider", return_value=provider),
        patch(APP + ".AsyncSandbox", return_value=sandbox),
        patch.object(PoolSession, "_install", side_effect=install),
    ):
        seeding = asyncio.create_task(server.seed_agent_session(fresh_request(), seed(borrowed=False)))
        await asyncio.wait_for(entered.wait(), timeout=5)
        shutdown = asyncio.create_task(server._shutdown_sessions())
        await asyncio.sleep(0)
        sandbox.stop.assert_not_awaited()
        release.set()
        with pytest.raises(HTTPException, match="shut down"):
            await asyncio.wait_for(seeding, timeout=5)
        await asyncio.wait_for(shutdown, timeout=5)
    sandbox.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_shutdown_setup_deadline_retains_installation_for_later_close():
    from nemo_gym.base_responses_api_agent import AgentCloseSessionRequest

    server = agent(
        setup_timeout=0.01,
        session_close_timeout_seconds=0.01,
        sandbox_provider="sandbox",
        sandbox_config={"image": "test", "workdir": "/repo"},
    )
    sandbox = AsyncMock()
    sandbox.sandbox_id = "owned"
    sandbox.exec.return_value = SandboxExecResult(stdout="", stderr="", return_code=0)
    provider = MagicMock()
    provider.name = "docker"
    entered, release = asyncio.Event(), asyncio.Event()

    async def install(*args, **kwargs):
        entered.set()
        await asyncio.wait_for(release.wait(), timeout=5)

    with (
        patch(APP + ".get_global_config_dict", return_value={}),
        patch(APP + ".resolve_provider_config"),
        patch(APP + ".resolve_provider_metadata", return_value={}),
        patch(APP + ".create_provider", return_value=provider),
        patch(APP + ".AsyncSandbox", return_value=sandbox),
        patch.object(PoolSession, "_install", side_effect=install),
    ):
        body = seed(borrowed=False)
        seeding = asyncio.create_task(server.seed_agent_session(fresh_request(), body))
        await asyncio.wait_for(entered.wait(), timeout=5)
        await asyncio.wait_for(server._shutdown_sessions(), timeout=1)
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(seeding, timeout=5)
        sandbox.stop.assert_not_awaited()
        session = server._session_records[body.agent_session_id].state
        assert isinstance(session, PoolSession)
        release.set()
        await asyncio.wait_for(session.installation_task, timeout=5)
        await server.close_agent_session(
            fresh_request(),
            AgentCloseSessionRequest(agent_session_id=body.agent_session_id, episode_id=body.episode_id),
        )
    sandbox.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_failure_artifacts_survive_close_without_duplicate_files(tmp_path):
    server, session = agent(failure_artifacts_dir=tmp_path), state()
    session.session.cleanup = receipt(timed_out=True, return_code=-15)
    raw = events(event("session.exit", reason="model_next_tool_call_http_error"))
    session.session.artifacts = PoolArtifacts(raw, "raw stdout", "HTTP 503", None)
    with (
        patch.object(PoolAgent, "model_endpoint", return_value=MagicMock(base_url="http://model/v1")),
        patch.object(SandboxSession, "execute", return_value=session.session.artifacts),
        pytest.raises(RuntimeError, match="model request failed"),
    ):
        await server._activate(session, NeMoGymResponseCreateParamsNonStreaming(input="task"))
    with patch.object(PoolSession, "close"):
        await server._close_agent_session_state(session)
    files = list(tmp_path.iterdir())
    assert len(files) == 1
    saved = json.loads(files[0].read_text())
    assert saved["artifacts"]["trajectory"] == raw
    assert saved["artifacts"]["stdout"] == "raw stdout"
    assert saved["artifacts"]["stderr"] == "HTTP 503"
    assert saved["failure"] == "RuntimeError"
    assert saved["cleanup"]["timed_out"] is True
    assert files[0].stat().st_mode & 0o777 == 0o600


@pytest.mark.asyncio
async def test_failure_artifact_write_error_keeps_original_error_and_can_retry(tmp_path, caplog):
    server, session = agent(failure_artifacts_dir=tmp_path), state()
    session.session.artifacts = PoolArtifacts("partial", "", "", None)
    with (
        patch.object(PoolAgent, "model_endpoint", return_value=MagicMock(base_url="http://model/v1")),
        patch.object(SandboxSession, "execute", side_effect=RuntimeError("original failure")),
        patch.object(PoolSession, "save_failure_artifacts", side_effect=OSError("disk full")),
        pytest.raises(RuntimeError, match="original failure"),
    ):
        await server._activate(session, NeMoGymResponseCreateParamsNonStreaming(input="task"))
    assert "Could not preserve Pool failure artifacts" in caplog.text
    assert session.failure_artifact_task is None
    with patch.object(PoolSession, "close"):
        await server._close_agent_session_state(session)
    assert json.loads(session.failure_artifact_path.read_text())["artifacts"]["trajectory"] == "partial"


@pytest.mark.asyncio
async def test_close_preserves_artifacts_collected_after_activation_failure(tmp_path):
    server, session = agent(failure_artifacts_dir=tmp_path), state()
    session.failure = "CancelledError"
    await server._preserve_failure(session)
    assert not list(tmp_path.iterdir())
    # Close can collect output that was unavailable when activation was cancelled.
    session.session.artifacts = PoolArtifacts("late output", "", "", None)
    with patch.object(PoolSession, "close"):
        await server._close_agent_session_state(session)
    assert json.loads(session.failure_artifact_path.read_text())["artifacts"]["trajectory"] == "late output"


@pytest.mark.asyncio
@pytest.mark.parametrize("completed_calls", [0, 1])
async def test_timeout_without_trajectory_requires_completed_model_response(completed_calls):
    server, session = agent(), state()
    session.session.cleanup = receipt(timed_out=True, return_code=-15)
    artifacts = PoolArtifacts("", "", "", None, completed_model_calls=completed_calls)
    with (
        patch.object(PoolAgent, "model_endpoint", return_value=MagicMock(base_url="http://model/v1")),
        patch.object(SandboxSession, "execute", return_value=artifacts),
    ):
        if not completed_calls:
            with pytest.raises(RuntimeError, match="before any model response"):
                await server._activate(session, NeMoGymResponseCreateParamsNonStreaming(input="task"))
            assert session.failure == "RuntimeError"
        else:
            response = await server._activate(session, NeMoGymResponseCreateParamsNonStreaming(input="task"))
            assert response.status == "incomplete" and response.output == []


@pytest.mark.asyncio
async def test_timeout_does_not_hide_unknown_terminal_failure():
    server, session = agent(), state()
    session.session.cleanup = receipt(timed_out=True, return_code=-15)
    artifacts = PoolArtifacts(events(event("session.exit", reason="some_other_failure")), "", "", None)
    with (
        patch.object(PoolAgent, "model_endpoint", return_value=MagicMock(base_url="http://model/v1")),
        patch.object(SandboxSession, "execute", return_value=artifacts),
        pytest.raises(RuntimeError, match="some_other_failure"),
    ):
        await server._activate(session, NeMoGymResponseCreateParamsNonStreaming(input="task"))


@pytest.mark.asyncio
@pytest.mark.parametrize("runtime", ["bad", "[]", '{"hostname": 1, "pid": true}', '{"pid": -1}'])
async def test_optional_runtime_schema_does_not_discard_trajectory(runtime):
    session = state()
    files = {
        "trajectory.ndjson": complete(),
        "exit.json": "0",
        "runtime.json": runtime,
        "model_calls.json": '{"completed": 1}',
    }

    async def read(sandbox, *, path):
        return files.get(path.rsplit("/", 1)[-1], "")

    session.session.sandbox.exec.return_value = SandboxExecResult(stdout="", stderr="", return_code=0)
    with patch("responses_api_agents.pool_agent.sandbox.read_text", side_effect=read):
        artifacts = await session.collect()
    assert artifacts.runner_hostname is None and artifacts.runner_pid is None
    assert artifacts.trajectory == complete() and artifacts.exit_code == 0
    assert artifacts.completed_model_calls == 1
