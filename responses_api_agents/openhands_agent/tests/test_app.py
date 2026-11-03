# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException, Request

from nemo_gym.agents import (
    AGENT_OUTCOME_METADATA_KEY,
    AGENT_SESSION_COOKIE_KEY,
    AgentSeedSessionRequest,
    AgentSessionSetupError,
    HarnessTerminationError,
    SandboxSession,
    check_agent_seed_rejected,
)
from nemo_gym.client import ServerClient
from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_runtime import MCPToolAccess
from responses_api_agents.openhands_agent import app
from responses_api_agents.openhands_agent import sandbox as adapter


GLOBAL_CONFIG = {
    "sandbox": {"local": {}},
    "remote_sandbox": {"opensandbox": {}},
    "policy_model": {"responses_api_models": {"vllm_model": {"host": "10.0.0.5", "port": 9000}}},
}


def seed(*, borrowed=True, workdir="/app", provider="sandbox", tool_accesses=()):
    return AgentSeedSessionRequest(
        agent_session_id="session",
        episode_id={"rollout_id": "rollout"},
        task_id={"taskset": "fixture", "task_id": "task"},
        tool_accesses=list(tool_accesses),
        sandbox_access={
            "connection": {"provider_config_ref": provider, "descriptor": {"id": "borrowed"}},
            "workdir": workdir,
        }
        if borrowed
        else None,
    )


def agent(**kwargs):
    config = {"model_server": {"type": "responses_api_models", "name": "policy_model"}, **kwargs}
    instance = app.OpenHandsAgent(
        config=app.OpenHandsAgentConfig(
            name="openhands", host="127.0.0.1", port=8080, entrypoint="app.py", model="served-model", **config
        ),
        server_client=MagicMock(spec=ServerClient, global_config_dict=GLOBAL_CONFIG),
    )
    instance.server_client._build_server_base_url = lambda server: f"http://{server['host']}:{server['port']}"
    return instance


def result(stdout="", *, code=0):
    return SimpleNamespace(stdout=stdout, stderr="diagnostic" if code else "", return_code=code, error_type=None)


def request(session_id="session", rollout="rollout"):
    session = {AGENT_SESSION_COOKIE_KEY: session_id} if session_id else {}
    return Request({"type": "http", "session": session, "path_params": {"rollout_id": rollout}})


def mcp_grant(*, required):
    return MCPToolAccess(
        name="task_tools",
        required=required,
        connection={"transport": "streamable_http", "url": "http://resources:8080/mcp"},
    )


@pytest.mark.parametrize(
    "instance,body,reason",
    [
        (agent(), seed(tool_accesses=[mcp_grant(required=True)]), "required tool grants: task_tools"),
        (agent(model_server=None), seed(), "model_server"),
        (agent(), seed(borrowed=False), "sandbox_access or a configured sandbox_provider"),
        (agent(), seed(workdir="relative"), "workdir"),
        (agent(), seed(workdir="/tmp/nemo-gym-openhands-sessions/x"), "workdir"),
    ],
)
def test_unservable_seeds_are_final_refusals(instance, body, reason):
    assert reason in check_agent_seed_rejected(instance.setup_webserver(), body)


def test_loopback_model_server_is_refused_for_a_remote_sandbox():
    instance = agent()
    instance.server_client._build_server_base_url = lambda server: "http://127.0.0.1:9000"
    assert "loopback" in check_agent_seed_rejected(instance.setup_webserver(), seed(provider="remote_sandbox"))


def test_llm_settings_reject_connection_and_unknown_fields():
    for name in ("base_url", "api_key", "extra_headers", "log_completions", "fallback_strategy", "stream", "typo"):
        with pytest.raises(ValueError, match=name):
            agent(llm_settings={name: True})
    assert agent(llm_settings={"temperature": 0.0}).config.llm_settings == {"temperature": 0.0}


@pytest.mark.asyncio
@pytest.mark.parametrize("borrowed", [True, False])
async def test_seed_installs_the_runtime_and_keeps_its_files_outside_the_task(monkeypatch, borrowed):
    remote = SimpleNamespace(exec=AsyncMock(return_value=result()), upload=AsyncMock(), start=AsyncMock())
    constructor = MagicMock(return_value=remote)
    constructor.connect = AsyncMock(return_value=remote)
    monkeypatch.setattr(app, "AsyncSandbox", constructor)
    provider = SimpleNamespace(aclose=AsyncMock())
    monkeypatch.setattr(app, "create_provider", lambda config: provider)
    prepare = AsyncMock(return_value="/tmp/nemo-gym-openhands-runtime/fp-x86_64/interp/bin/openhands-gym")
    monkeypatch.setattr(app, "prepare_runtime", prepare)
    instance = agent(sandbox_provider="sandbox", sandbox_config={"workdir": "/app"})
    state = await instance._seed_agent_session_state(seed(borrowed=borrowed))
    assert prepare.await_args.args == (remote,)
    assert state.interpreter.endswith("/openhands-gym")
    assert state.session.session_dir.startswith("/tmp/nemo-gym-openhands-sessions/")
    assert state.session.workdir == "/app"
    uploaded = [call.args[1] for call in remote.upload.await_args_list]
    assert uploaded == [
        f"{state.session.session_dir}/harness_runner.sh",
        f"{state.session.session_dir}/openhands_runner",
    ]
    assert state.session.owns_sandbox is not borrowed
    assert remote.start.await_count == (0 if borrowed else 1)
    commands = [call.args[0] for call in remote.exec.await_args_list]
    if borrowed:
        constructor.connect.assert_awaited_once_with({"id": "borrowed"}, provider=provider)
        # A borrowed task is never prepared by the agent.
        assert not any(command.startswith("mkdir -p -- /app") for command in commands)


@pytest.mark.asyncio
async def test_failed_seed_releases_and_failed_release_retains_state(monkeypatch):
    remote = SimpleNamespace(exec=AsyncMock(return_value=result()), upload=AsyncMock())
    monkeypatch.setattr(app.AsyncSandbox, "connect", AsyncMock(return_value=remote))
    monkeypatch.setattr(app, "create_provider", lambda config: object())
    monkeypatch.setattr(app, "prepare_runtime", AsyncMock(side_effect=RuntimeError("pip failed")))
    close = AsyncMock()
    monkeypatch.setattr(adapter.OpenHandsSandboxSession, "close", close)
    with pytest.raises(RuntimeError, match="pip failed"):
        await agent()._seed_agent_session_state(seed())
    close.assert_awaited_once()
    close.side_effect = RuntimeError("release failed")
    with pytest.raises(AgentSessionSetupError) as error:
        await agent()._seed_agent_session_state(seed())
    assert error.value.state.request == seed()


def session_state(**session):
    defaults = {
        "session_dir": "/tmp/nemo-gym-openhands-sessions/s",
        "workdir": "/app",
        "cleanup": {"timed_out": False, "return_code": 0, "cleanup_confirmed": True, "error": None},
        "artifacts": None,
        "expired": False,
    }
    return adapter.OpenHandsSandboxSession(
        request=seed(), session=SimpleNamespace(**{**defaults, **session}), interpreter="/rt/interp/bin/openhands-gym"
    )


def artifacts(events, run_result):
    lines = "".join(json.dumps([100.0 + index, event]) + "\n" for index, event in enumerate(events))
    return adapter.OpenHandsArtifacts(
        events=lines, result=None if run_result is None else json.dumps(run_result), stderr="trace"
    )


FINISHED_EVENTS = [
    {
        "kind": "ActionEvent",
        "tool_name": "terminal",
        "tool_call_id": "c1",
        "tool_call": {"arguments": '{"command": "make"}'},
        "llm_response_id": "r1",
    },
    {
        "kind": "ObservationEvent",
        "tool_name": "terminal",
        "tool_call_id": "c1",
        "observation": {"content": [{"type": "text", "text": "built"}], "exit_code": 0},
    },
    {
        "kind": "ActionEvent",
        "tool_name": "finish",
        "tool_call_id": "c2",
        "tool_call": {"arguments": '{"message": "done"}'},
        "llm_response_id": "r2",
    },
    {
        "kind": "ObservationEvent",
        "tool_name": "finish",
        "tool_call_id": "c2",
        "observation": {"content": [{"type": "text", "text": "done"}]},
    },
]


def run_result(**overrides):
    usages = [{"response_id": r, "prompt_tokens": 10, "completion_tokens": 2} for r in ("r1", "r2")]
    return {
        "status": "returned",
        "execution_status": "finished",
        "error": None,
        "started_at": 1.0,
        "ended_at": 2.0,
        "stats": {"usage_to_metrics": {"agent": {"token_usages": usages}}},
        **overrides,
    }


@pytest.mark.asyncio
async def test_activation_routes_every_model_call_through_the_rollout_prefixed_model_server(monkeypatch):
    instance = agent(system_prompt="Configured", llm_settings={"reasoning_effort": None, "temperature": 0.0})
    state = session_state()
    execute = AsyncMock(return_value=artifacts(FINISHED_EVENTS, run_result()))
    monkeypatch.setattr(adapter.OpenHandsSandboxSession, "execute", execute)
    body = NeMoGymResponseCreateParamsNonStreaming(
        input=[{"role": "system", "content": "System"}, {"role": "user", "content": "Build it"}],
        metadata={"agent_timeout_sec": "60"},
    )
    prompt, system = instance._validate_request(body)
    response = await instance._activate(state, body, prompt=prompt, system=system)
    runner_input = execute.await_args.args[0]
    assert runner_input["prompt"] == "Build it"
    assert runner_input["system_message_suffix"] == "Configured\n\nSystem"
    assert runner_input["llm"] == {
        "reasoning_effort": None,
        "temperature": 0.0,
        "model": "openai/served-model",
        "base_url": "http://10.0.0.5:9000/ng-rollout/rollout/v1",
        "api_key": "gym",
        "usage_id": "agent",
        "extra_headers": {"x-session-id": "rollout"},
    }
    assert runner_input["workdir"] == "/app"
    for name in ("state_dir", "events_path", "result_path"):
        assert runner_input[name].startswith("/tmp/nemo-gym-openhands-sessions/s/")
    assert execute.await_args.kwargs["timeout"] == 60
    assert response.status == "completed"
    assert response.output_text == "done"
    assert response.usage.total_tokens == 24
    assert response.metadata["harness_execution"] == "sandbox"
    assert response.metadata["stop_reason"] == "finished"
    assert state.observations.records[0].model_calls[0].response_id == "r1"


@pytest.mark.asyncio
async def test_model_failure_is_classified_and_context_limit_stays_gradable(monkeypatch):
    instance = agent()
    error = [{"type": "ConversationRunError", "message": "x"}, {"type": "LLMRateLimitError", "message": "slow down"}]
    failed = run_result(status="error", execution_status="error", error=error)
    monkeypatch.setattr(
        adapter.OpenHandsSandboxSession, "execute", AsyncMock(return_value=artifacts(FINISHED_EVENTS[:2], failed))
    )
    body = NeMoGymResponseCreateParamsNonStreaming(input="Build it")
    response = await instance._activate(session_state(), body, prompt="Build it", system="")
    assert response.status == "failed"
    outcome = json.loads(response.metadata[AGENT_OUTCOME_METADATA_KEY])
    assert outcome == {"failure_kind": "model_request_error", "status_code": 429}

    overflow = run_result(status="error", error=[{"type": "LLMContextWindowExceedError", "message": "too long"}])
    monkeypatch.setattr(
        adapter.OpenHandsSandboxSession, "execute", AsyncMock(return_value=artifacts(FINISHED_EVENTS[:2], overflow))
    )
    response = await instance._activate(session_state(), body, prompt="Build it", system="")
    assert (response.status, response.error) == ("incomplete", None)
    assert json.loads(response.metadata[AGENT_OUTCOME_METADATA_KEY])["failure_kind"] == "model_context_limit"


@pytest.mark.asyncio
async def test_deadline_keeps_partial_work_as_incomplete(monkeypatch):
    instance = agent()
    cleanup = {"timed_out": True, "return_code": 143, "cleanup_confirmed": True, "error": None}
    captured = artifacts(FINISHED_EVENTS[:2], run_result(status="terminated", execution_status="running"))
    state = session_state(cleanup=cleanup, artifacts=captured)
    monkeypatch.setattr(adapter.OpenHandsSandboxSession, "execute", AsyncMock(side_effect=TimeoutError("provider")))
    body = NeMoGymResponseCreateParamsNonStreaming(input="Build it")
    response = await instance._activate(state, body, prompt="Build it", system="")
    assert response.status == "incomplete"
    assert response.metadata["stop_reason"] == "timeout"
    assert [item.type for item in response.output] == ["function_call", "function_call_output"]
    sandbox = state.observations.records[-1]
    assert (sandbox.kind, sandbox.outcome, sandbox.exit_code) == ("sandbox", "timeout", 143)


@pytest.mark.asyncio
async def test_unconfirmed_termination_fails_the_activation_with_observations(monkeypatch):
    instance = agent()
    state = session_state(cleanup=None)
    monkeypatch.setattr(
        adapter.OpenHandsSandboxSession, "execute", AsyncMock(side_effect=RuntimeError("termination unconfirmed"))
    )
    body = NeMoGymResponseCreateParamsNonStreaming(input="Build it")
    with pytest.raises(RuntimeError, match="termination unconfirmed"):
        await instance._activate(state, body, prompt="Build it", system="")
    assert "agent_artifact_unavailable" in [gap.code for gap in state.observations.gaps]


@pytest.mark.asyncio
async def test_missing_result_is_a_runner_failure(monkeypatch):
    instance = agent()
    monkeypatch.setattr(
        adapter.OpenHandsSandboxSession, "execute", AsyncMock(return_value=artifacts(FINISHED_EVENTS[:2], None))
    )
    body = NeMoGymResponseCreateParamsNonStreaming(input="Build it")
    with pytest.raises(RuntimeError, match="before writing a result: trace"):
        await instance._activate(session_state(), body, prompt="Build it", system="")


@pytest.mark.parametrize(
    "fields",
    [
        {"temperature": 0.5},
        {"max_output_tokens": 100},
        {"tools": [{"type": "function", "name": "f", "parameters": {}, "strict": False}]},
        {"metadata": {"other": "x"}},
        {"model": "other-model"},
        {"input": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]},
        {"input": [{"role": "user", "content": [{"type": "input_image", "image_url": "http://x", "detail": "auto"}]}]},
    ],
)
def test_unsupported_request_controls_are_rejected(fields):
    with pytest.raises(HTTPException) as error:
        agent()._validate_request(NeMoGymResponseCreateParamsNonStreaming(**{"input": "Build it", **fields}))
    assert error.value.status_code == 422


@pytest.mark.asyncio
async def test_retry_shares_the_activation_and_rejects_a_changed_body(monkeypatch):
    instance = agent()
    state = session_state(closing=False)
    async with instance._locked_agent_session("session") as record:
        record.state = state
    response = MagicMock()
    response.model_copy.return_value = response
    activate = AsyncMock(return_value=response)
    monkeypatch.setattr(app.OpenHandsAgent, "_activate", activate)
    body = NeMoGymResponseCreateParamsNonStreaming(input="Build it")
    assert await instance.responses(request(), body) is response
    assert await instance.responses(request(), body) is response
    assert activate.await_count == 1
    for changed_request, changed_body in [
        (request(), NeMoGymResponseCreateParamsNonStreaming(input="Changed")),
        (request(rollout="other"), body),
        (request("unknown"), body),
    ]:
        with pytest.raises(HTTPException) as error:
            await instance.responses(changed_request, changed_body)
        assert error.value.status_code == 409
    assert activate.await_count == 1


@pytest.mark.asyncio
async def test_sessionless_requests_and_run_are_refused():
    instance = agent()
    with pytest.raises(HTTPException) as error:
        await instance.responses(request(session_id=None), NeMoGymResponseCreateParamsNonStreaming(input="x"))
    assert error.value.status_code == 422
    with pytest.raises(HTTPException) as error:
        await instance.run(MagicMock())
    assert error.value.status_code == 422


@pytest.mark.asyncio
@pytest.mark.parametrize("confirmed", [True, False])
async def test_close_releases_a_borrowed_sandbox_and_reports_termination(confirmed):
    remote = SimpleNamespace(exec=AsyncMock(return_value=result()), disconnect=AsyncMock())
    session = SandboxSession(sandbox=remote, session_dir="/tmp/session", workdir="/app", harness="OpenHands")
    session.launch_started = True
    receipt = {
        "cleanup_confirmed": confirmed,
        "timed_out": False,
        "return_code": 0,
        "error": None if confirmed else "process survived",
    }
    if confirmed:
        session.cleanup = receipt
    else:
        session.termination_error = HarnessTerminationError(harness="OpenHands", receipt=receipt)
    state = adapter.OpenHandsSandboxSession(request=seed(), session=session)
    close = await agent()._close_agent_session_state(state)
    remote.disconnect.assert_awaited_once()
    assert close.cleanup_confirmed is confirmed
    assert close.agent_observations.gaps[0].code == "agent_activation_interrupted"


@pytest.mark.asyncio
async def test_close_bounds_an_activation_that_ignores_cancellation():
    release = asyncio.Event()

    async def activation():
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            await release.wait()

    session = SimpleNamespace(close=AsyncMock(), artifacts=None, cleanup_confirmed=True)
    state = adapter.OpenHandsSandboxSession(request=seed(), session=session)
    state.task = asyncio.create_task(activation())
    await asyncio.sleep(0)
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(state.close(timeout=0.05), timeout=5)
    release.set()
    await asyncio.wait_for(state.task, timeout=5)
