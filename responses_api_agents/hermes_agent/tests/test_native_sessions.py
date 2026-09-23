# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError

from nemo_gym.base_responses_api_agent import AgentCloseSessionRequest, AgentSeedSessionRequest
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_observability import AgentEpisode, AgentObservationBundle
from nemo_gym.server_utils import ServerClient
from responses_api_agents.hermes_agent.app import (
    HermesAgent,
    HermesAgentConfig,
    HermesAgentRunRequest,
    HermesAgentSessionState,
    RunnerCleanup,
    SessionPhase,
)


@pytest.fixture
def agent():
    return HermesAgent(
        config=HermesAgentConfig(
            host="127.0.0.1",
            port=8080,
            name="hermes",
            entrypoint="app.py",
            resources_server={"type": "resources_servers", "name": "resources"},
            model_server={"type": "responses_api_models", "name": "model"},
            enabled_toolsets=["terminal"],
            max_tokens=500,
            temperature=0.7,
        ),
        server_client=MagicMock(spec=ServerClient),
    )


@pytest.fixture
def state():
    sandbox = AsyncMock()
    sandbox.exec.return_value = SimpleNamespace(return_code=0, stdout="", stderr="")
    return HermesAgentSessionState(
        request=AgentSeedSessionRequest(
            agent_session_id="session",
            episode_id=EpisodeId(rollout_id="native", attempt=1),
            task_id=TaskId(taskset="test", task_id="task"),
            sandbox_access={
                "connection": {
                    "kind": "direct",
                    "provider_config_ref": "provider",
                    "descriptor": {"sandbox_id": "task-box"},
                },
                "workdir": "/app",
            },
        ),
        sandbox=sandbox,
        workdir="/app",
        session_dir="/tmp/nemo-gym-hermes-sessions/session",
    )


def request(state):
    return SimpleNamespace(
        session={"agent_session_id": "session"}, path_params={"rollout_id": state.request.episode_id.capture_key}
    )


def episode(agent):
    body = NeMoGymResponseCreateParamsNonStreaming(input="task")
    return AgentEpisode(
        response=agent._response_from_result(
            body=body,
            result={"completed": True, "messages": [{"role": "assistant", "content": "done"}]},
            model_name="model",
            fail_on_error=False,
        ),
        observations=AgentObservationBundle(source="hermes"),
    )


def test_http_close_retry_and_stale_activation_never_fall_back(agent, state):
    agent._initialize_agent_session_state = AsyncMock(return_value=state)
    agent._run_sandbox_episode = AsyncMock(return_value=episode(agent))
    agent._create_response = AsyncMock(side_effect=AssertionError("host fallback"))
    agent._create_episode = AsyncMock(side_effect=AssertionError("host fallback"))
    with TestClient(agent.setup_webserver()) as client:
        seed = client.post("/v1/agent_sessions", json=state.request.model_dump(mode="json"))
        assert seed.status_code == 200
        assert state.phase is SessionPhase.READY
        assert state.runner_cleanup is RunnerCleanup.IDLE
        assert client.post("/v1/agent_sessions", json=state.request.model_dump(mode="json")).status_code == 200
        path = f"/ng-rollout/{state.request.episode_id.capture_key}/v1/responses"
        assert client.post(path, json={"input": "task"}).status_code == 200
        assert state.task is not None
        assert state.task.done()
        assert state.phase is SessionPhase.ACTIVATED
        assert client.post(path, json={"input": "task"}).status_code == 409
        close = {
            "agent_session_id": seed.json()["agent_session_id"],
            "episode_id": state.request.episode_id.model_dump(),
        }
        first = client.post("/v1/agent_sessions/close", json=close)
        retry = client.post("/v1/agent_sessions/close", json=close)
        assert first.status_code == retry.status_code == 200
        assert state.phase is SessionPhase.CLOSING
        assert first.json() == retry.json()
        assert client.post(path, json={"input": "task"}).status_code == 409
        assert client.post("/run", json={"responses_create_params": {"input": "task"}}).status_code == 409
        wrong = dict(close, episode_id={"rollout_id": "other"})
        assert client.post("/v1/agent_sessions/close", json=wrong).status_code == 409
    assert agent._run_sandbox_episode.await_count == 1
    state.sandbox.disconnect.assert_awaited_once()
    state.sandbox.stop.assert_not_awaited()


async def test_invalid_activation_keeps_session_ready(agent: HermesAgent, state: HermesAgentSessionState) -> None:
    agent._agent_sessions["session"] = state
    agent._run_sandbox_episode = AsyncMock(return_value=episode(agent))
    with pytest.raises(HTTPException) as error:
        await agent.responses(request(state), NeMoGymResponseCreateParamsNonStreaming(input="task", top_p=0.9))
    assert error.value.status_code == 422
    assert state.phase is SessionPhase.READY
    assert state.task is None
    agent._run_sandbox_episode.assert_not_awaited()

    await agent.responses(request(state), NeMoGymResponseCreateParamsNonStreaming(input="task"))
    assert state.phase is SessionPhase.ACTIVATED
    agent._run_sandbox_episode.assert_awaited_once()


def test_http_close_retry_survives_other_session_closes(agent, state, monkeypatch):
    monkeypatch.setattr("responses_api_agents.hermes_agent.app.monotonic", lambda: 100.0)

    async def initialize(agent_session_id, body):
        return replace(state, request=body, close_lock=asyncio.Lock())

    agent._initialize_agent_session_state = AsyncMock(side_effect=initialize)
    with TestClient(agent.setup_webserver()) as client:

        def seed_and_close(index):
            client.cookies.clear()
            body = state.request.model_dump(mode="json")
            body["agent_session_id"] = f"session-{index}"
            body["episode_id"] = {"rollout_id": f"episode-{index}"}
            seed = client.post("/v1/agent_sessions", json=body)
            assert seed.status_code == 200
            cookies = dict(client.cookies)
            close = {"agent_session_id": seed.json()["agent_session_id"], "episode_id": body["episode_id"]}
            result = client.post("/v1/agent_sessions/close", json=close)
            assert result.status_code == 200
            return cookies, close, result.json()

        cookies, close, first = seed_and_close(0)
        # A's close succeeds, but its caller loses the reply while unrelated sessions finish.
        for index in range(1, 66):
            seed_and_close(index)
        client.cookies.clear()
        client.cookies.update(cookies)
        retry = client.post("/v1/agent_sessions/close", json=close)
        assert retry.status_code == 200
        assert retry.json() == first
    assert state.sandbox.disconnect.await_count == 66  # No second cleanup for A.


async def test_close_receipt_expires_without_extending_on_retry(agent, state, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("responses_api_agents.hermes_agent.app.monotonic", lambda: clock[0])
    agent.config.session_close_retry_window_seconds = 10
    agent._agent_sessions["session"] = state
    close = AgentCloseSessionRequest(agent_session_id="session", episode_id=state.request.episode_id)
    first = await agent.close_agent_session(request(state), close)
    clock[0] = 109.0
    assert await agent.close_agent_session(request(state), close) == first
    clock[0] = 110.0
    with pytest.raises(HTTPException) as error:
        await agent.close_agent_session(request(state), close)
    assert error.value.status_code == 409
    assert not agent._closed_agent_sessions
    agent._create_episode = AsyncMock(side_effect=AssertionError("host fallback"))
    with pytest.raises(HTTPException) as error:
        await agent.responses(request(state), NeMoGymResponseCreateParamsNonStreaming(input="task"))
    assert error.value.status_code == 409
    state.sandbox.disconnect.assert_awaited_once()


async def test_close_retry_window_starts_after_cleanup(agent, state, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("responses_api_agents.hermes_agent.app.monotonic", lambda: clock[0])
    agent.config.session_close_retry_window_seconds = 10
    agent._agent_sessions["session"] = state

    async def disconnect():
        clock[0] = 200.0  # Cleanup itself takes longer than the retry window.

    state.sandbox.disconnect.side_effect = disconnect
    close = AgentCloseSessionRequest(agent_session_id="session", episode_id=state.request.episode_id)
    first = await agent.close_agent_session(request(state), close)
    clock[0] = 209.0
    assert await agent.close_agent_session(request(state), close) == first
    state.sandbox.disconnect.assert_awaited_once()


@pytest.mark.parametrize("window", [0, -1, float("inf")])
def test_close_retry_window_must_be_positive_and_finite(agent, window):
    config = agent.config.model_dump() | {"session_close_retry_window_seconds": window}
    with pytest.raises(ValidationError, match="session_close_retry_window_seconds"):
        HermesAgentConfig.model_validate(config)


async def test_close_cancels_activation_and_rejects_duplicate(agent, state):
    started = asyncio.Event()
    stopped = asyncio.Event()

    async def activate(**kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    agent._agent_sessions["session"] = state
    agent._run_sandbox_episode = AsyncMock(side_effect=activate)
    body = NeMoGymResponseCreateParamsNonStreaming(input="task")
    running = asyncio.create_task(agent.responses(request(state), body))
    await asyncio.wait_for(started.wait(), 5)
    assert state.phase is SessionPhase.ACTIVATED
    with pytest.raises(HTTPException) as error:
        await agent.responses(request(state), body)
    assert error.value.status_code == 409
    close = AgentCloseSessionRequest(agent_session_id="session", episode_id=state.request.episode_id)
    first, second = await asyncio.gather(
        agent.close_agent_session(request(state), close), agent.close_agent_session(request(state), close)
    )
    assert first == second
    assert stopped.is_set()
    assert state.phase is SessionPhase.CLOSING
    with pytest.raises(asyncio.CancelledError):
        await running
    state.sandbox.disconnect.assert_awaited_once()


@pytest.mark.parametrize("runner_started", [False, True], ids=["not-launched", "cleanup-confirmed"])
async def test_close_failure_keeps_session_for_retry(
    agent: HermesAgent, state: HermesAgentSessionState, runner_started: bool
) -> None:
    agent._agent_sessions["session"] = state
    if runner_started:
        state.runner_cleanup = RunnerCleanup.UNCONFIRMED
        state.runner_session = AsyncMock()
        state.runner_exit_task = asyncio.create_task(asyncio.sleep(0, result=0))
        await state.runner_exit_task
        agent._download_json = AsyncMock(return_value={"cleanup_confirmed": True})
    state.sandbox.exec.side_effect = [SimpleNamespace(return_code=1), SimpleNamespace(return_code=0)]
    close = AgentCloseSessionRequest(agent_session_id="session", episode_id=state.request.episode_id)
    with pytest.raises(RuntimeError, match="session files"):
        await agent.close_agent_session(request(state), close)
    assert agent._agent_sessions["session"] is state
    assert state.phase is SessionPhase.CLOSING
    assert state.runner_cleanup is (RunnerCleanup.CONFIRMED if runner_started else RunnerCleanup.IDLE)
    assert state.runner_session is None
    state.sandbox.disconnect.assert_not_awaited()
    with pytest.raises(HTTPException):
        await agent.responses(request(state), NeMoGymResponseCreateParamsNonStreaming(input="task"))
    await agent.close_agent_session(request(state), close)
    state.sandbox.disconnect.assert_awaited_once()
    if runner_started:
        agent._download_json.assert_awaited_once()


@pytest.mark.parametrize("failure", [TimeoutError, asyncio.CancelledError])
async def test_unknown_launch_blocks_close(agent, state, failure):
    state.sandbox.pty.create.side_effect = failure("launch status unavailable")
    agent._upload_json = AsyncMock()
    with pytest.raises(failure):
        await agent._run_sandbox_episode(
            request=request(state),
            body=NeMoGymResponseCreateParamsNonStreaming(input="task"),
            agent_session_id="session",
            state=state,
        )
    assert state.runner_cleanup is RunnerCleanup.UNCONFIRMED
    assert state.runner_session is None
    with pytest.raises(RuntimeError, match="launch outcome is unknown"):
        await agent._close_agent_session_state(state)
    state.sandbox.disconnect.assert_not_awaited()


@pytest.mark.parametrize("receipt", [{}, {"cleanup_confirmed": False}, {"cleanup_confirmed": "true"}])
async def test_runner_exit_without_cleanup_receipt_blocks_close(agent, state, receipt):
    state.runner_cleanup = RunnerCleanup.UNCONFIRMED
    state.runner_session = AsyncMock()
    state.runner_exit_task = asyncio.create_task(asyncio.sleep(0, result=0))
    await state.runner_exit_task
    agent._download_json = AsyncMock(return_value=receipt)
    with pytest.raises(RuntimeError, match="cleanup was not confirmed"):
        await agent._close_agent_session_state(state)
    assert state.runner_cleanup is RunnerCleanup.UNCONFIRMED
    state.runner_session.close.assert_not_awaited()
    state.sandbox.disconnect.assert_not_awaited()
    agent._download_json.return_value = {"cleanup_confirmed": True}
    await agent._close_agent_session_state(state)
    assert state.runner_cleanup is RunnerCleanup.CONFIRMED
    state.sandbox.disconnect.assert_awaited_once()


@pytest.mark.parametrize("overrides", [{}, {"max_output_tokens": 32, "temperature": 0.0}])
async def test_native_prompt_and_limits_reach_runner(agent, state, overrides):
    agent.config.system_prompt = "Configured instruction"
    agent._upload_json = AsyncMock()
    state.sandbox.pty.create.side_effect = NotImplementedError
    body = NeMoGymResponseCreateParamsNonStreaming(
        input="Fix the bug", instructions="Request instruction", **overrides
    )
    with pytest.raises(HTTPException):
        await agent._run_sandbox_episode(request=request(state), body=body, agent_session_id="session", state=state)
    payload = agent._upload_json.await_args.args[2]
    assert payload["user_message"] == "Fix the bug"
    assert payload["history"] == []
    assert payload["system_message"] == "Configured instruction\n\nRequest instruction"
    assert payload["max_tokens"] == overrides.get("max_output_tokens", 500)
    assert payload["temperature"] == overrides.get("temperature", 0.7)
    assert body.input == "Fix the bug"  # Do not mutate the caller's request.


@pytest.mark.parametrize("output_available", [True, False], ids=["late-output", "missing-output"])
async def test_runner_exit_rechecks_output_after_stale_probe(
    agent: HermesAgent, state: HermesAgentSessionState, output_available: bool
) -> None:
    agent._upload_json = AsyncMock()
    runner = AsyncMock()
    state.sandbox.pty.create.return_value = runner
    published = asyncio.Event()

    async def wait_exit() -> int:
        await published.wait()
        return 0

    runner.wait_exit.side_effect = wait_exit
    probes = []

    async def execute(command: str, **kwargs) -> SimpleNamespace:
        if command.startswith("if [ -f "):
            probes.append(command)
            if len(probes) == 1:
                # The remote probe saw no output, but the runner publishes its
                # result and exits before that probe's reply reaches the agent.
                published.set()
                await state.runner_exit_task
                return SimpleNamespace(stdout="running\n", stderr="", return_code=0)
            assert len(probes) == 2
            return SimpleNamespace(stdout="output\n" if output_available else "exited\n", return_code=0)
        assert command.startswith("cat ")
        return SimpleNamespace(stdout="runner stderr", stderr="", return_code=0)

    state.sandbox.exec.side_effect = execute

    async def download(sandbox, path: str) -> dict:
        if path.endswith("/cleanup.json"):
            return {"cleanup_confirmed": True}
        assert output_available and path.endswith("/output.json")
        return {
            "result": {
                "completed": True,
                "messages": [
                    {"role": "user", "content": "task"},
                    {"role": "assistant", "content": "Patch done"},
                ],
            },
            "runtime": {"pid": 123},
        }

    agent._download_json = AsyncMock(side_effect=download)
    activation = agent._run_sandbox_episode(
        request=request(state),
        body=NeMoGymResponseCreateParamsNonStreaming(input="task"),
        agent_session_id="session",
        state=state,
    )
    if output_available:
        result = await activation
        assert result.response.status == "completed"
        assert result.response.output[-1].content[0].text == "Patch done"
        assert result.observations.records[0].status == "completed"
    else:
        with pytest.raises(RuntimeError, match="runner exited without output: runner stderr"):
            await activation
    assert len(probes) == 2
    assert state.runner_cleanup is RunnerCleanup.CONFIRMED
    runner.send_signal.assert_not_awaited()
    runner.close.assert_awaited_once()
    assert state.runner_session is None


@pytest.mark.parametrize(
    "override",
    [
        {"top_p": 0.8},
        {"reasoning": {"effort": "low"}},
        {"previous_response_id": "previous"},
        {"tool_choice": "none"},
        {"parallel_tool_calls": False},
        {"background": True},
        {
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_image", "image_url": "https://example.com/x.png", "detail": "auto"}],
                }
            ]
        },
        {"input": [{"type": "function_call_output", "call_id": "call", "output": "result"}]},
    ],
)
def test_unsupported_requests_are_rejected(agent, override):
    body = NeMoGymResponseCreateParamsNonStreaming.model_validate({"input": "task"} | override)
    with pytest.raises(HTTPException) as error:
        agent._validate_sandbox_request(body)
    assert error.value.status_code == 422


def test_text_history_and_system_message_are_preserved(agent):
    body = NeMoGymResponseCreateParamsNonStreaming(
        input=[
            {"role": "system", "content": "System instruction"},
            {"role": "user", "content": [{"type": "input_text", "text": "First question"}]},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Follow-up"},
        ]
    )
    assert agent._validate_sandbox_request(body).input == body.input


async def test_required_resources_tools_are_rejected_before_connect(agent, state):
    from nemo_gym.tool_access import DirectHTTPToolAccess

    state.request.tool_accesses = [DirectHTTPToolAccess(name="required", required=True, base_url="http://tools")]
    with pytest.raises(HTTPException, match="required HTTP/MCP"):
        await agent._initialize_agent_session_state("session", state.request)


async def test_seed_binds_full_payload_and_is_serialized(agent, state):
    agent._initialize_agent_session_state = AsyncMock(return_value=state)
    requests = [SimpleNamespace(session={}), SimpleNamespace(session={})]
    result = await asyncio.gather(*(agent.seed_agent_session(req, state.request) for req in requests))
    assert [item.agent_session_id for item in result] == [state.request.agent_session_id] * 2
    agent._initialize_agent_session_state.assert_awaited_once()
    changed = state.request.model_copy(deep=True)
    changed.sandbox_access.workdir = "/other"
    with pytest.raises(HTTPException, match="another seed"):
        await agent.seed_agent_session(requests[1], changed)
    close = AgentCloseSessionRequest(agent_session_id="session", episode_id=state.request.episode_id)
    result = await agent.close_agent_session(SimpleNamespace(session={}), close)
    assert await agent.close_agent_session(SimpleNamespace(session={}), close) == result
    assert not agent._agent_sessions
    state.sandbox.disconnect.assert_awaited_once()


async def test_unknown_close_tombstone_and_locks_expire(agent, state, monkeypatch):
    agent.config.session_lifetime_seconds = 20
    clock = [100.0]
    monkeypatch.setattr("responses_api_agents.hermes_agent.app.monotonic", lambda: clock[0])
    agent.config.session_close_retry_window_seconds = 10
    close = AgentCloseSessionRequest(agent_session_id="session", episode_id=state.request.episode_id)
    stale_request = SimpleNamespace(session={})
    await agent.close_agent_session(stale_request, close)
    with pytest.raises(HTTPException, match="already closed"):
        await agent.seed_agent_session(SimpleNamespace(session={}), state.request)
    clock[0] = 110.0
    agent._expire_closed_agent_sessions()
    assert not agent._closed_agent_sessions
    with pytest.raises(HTTPException, match="already closed"):
        await agent.seed_agent_session(SimpleNamespace(session={}), state.request)
    with pytest.raises(HTTPException, match="expired"):
        await agent.close_agent_session(SimpleNamespace(session={}), close)
    clock[0] = 120.0
    agent._expire_closed_agent_sessions()
    assert not agent._closed_agent_session_ids
    assert not agent._agent_session_locks
    with pytest.raises(HTTPException, match="expired"):
        await agent.seed_agent_session(stale_request, state.request)
    with pytest.raises(HTTPException, match="expired"):
        await agent.close_agent_session(stale_request, close)


@pytest.mark.parametrize("marker", [None, "", 0, [], {}])
async def test_malformed_cookie_cannot_fall_back_or_seed(agent, state, marker):
    malformed = SimpleNamespace(session={"agent_session_id": marker})
    agent._create_response = AsyncMock(side_effect=AssertionError("host fallback"))
    with pytest.raises(HTTPException, match="Invalid Hermes"):
        await agent.responses(malformed, NeMoGymResponseCreateParamsNonStreaming(input="task"))
    with pytest.raises(HTTPException, match="Invalid Hermes"):
        await agent.seed_agent_session(malformed, state.request)
    with pytest.raises(HTTPException, match="Invalid Hermes"):
        await agent.close_agent_session(
            malformed, AgentCloseSessionRequest(agent_session_id="session", episode_id=state.request.episode_id)
        )
    with pytest.raises(HTTPException, match="Invalid Hermes"):
        await agent.run(malformed, HermesAgentRunRequest(responses_create_params={"input": "task"}))


@pytest.mark.parametrize("cleanup_fails", [False, True])
async def test_abandoned_session_is_closed_or_retained_fail_closed(agent, state, cleanup_fails, caplog):
    agent.config.session_lifetime_seconds = 0.001
    agent._initialize_agent_session_state = AsyncMock(return_value=state)
    await agent.seed_agent_session(SimpleNamespace(session={}), state.request)
    if cleanup_fails:
        state.sandbox.disconnect.side_effect = RuntimeError("disconnect unavailable")
    reaper = agent._session_reapers["session"]
    await asyncio.wait_for(asyncio.shield(reaper), timeout=1)
    if cleanup_fails:
        assert agent._agent_sessions["session"].phase is SessionPhase.CLOSING
        assert "owner recovery required" in caplog.text
        with pytest.raises(HTTPException):
            await agent.responses(request(state), NeMoGymResponseCreateParamsNonStreaming(input="task"))
        state.sandbox.disconnect.side_effect = None
        await agent.close_agent_session(
            SimpleNamespace(session={}),
            AgentCloseSessionRequest(agent_session_id="session", episode_id=state.request.episode_id),
        )
    assert not agent._agent_sessions
    assert not agent._session_reapers
    assert "session" in agent._closed_agent_sessions
    state.sandbox.stop.assert_not_awaited()


@pytest.mark.parametrize("owns_sandbox", [False, True])
@pytest.mark.parametrize("cleanup_fails", [False, True])
async def test_failed_setup_retains_handle_until_cleanup_confirmed(
    agent, state, monkeypatch, owns_sandbox, cleanup_fails
):
    import responses_api_agents.hermes_agent.app as module

    body = state.request.model_copy(deep=True)
    if owns_sandbox:
        body.sandbox_access = None
        agent.config.sandbox_provider = "runtime"
        agent.config.sandbox_config = {"workdir": "/fallback"}
    sandbox = state.sandbox
    factory = MagicMock(return_value=sandbox)
    factory.connect = AsyncMock(return_value=sandbox)
    monkeypatch.setattr(module, "AsyncSandbox", factory)
    monkeypatch.setattr(module, "get_global_config_dict", lambda: {})
    monkeypatch.setattr(module, "resolve_provider_config", lambda *args: {})
    monkeypatch.setattr(module, "create_provider", lambda config: AsyncMock())
    monkeypatch.setattr(module.shutil, "which", lambda name: "/test/uv")
    ok = SimpleNamespace(return_code=0, stdout="", stderr="")
    failed = SimpleNamespace(return_code=1, stdout="", stderr="installer failed")
    sandbox.exec.side_effect = [ok, failed, ok]
    cleanup = sandbox.stop if owns_sandbox else sandbox.disconnect
    if cleanup_fails:
        cleanup.side_effect = RuntimeError("cleanup unavailable")
    with pytest.raises(RuntimeError, match="installer failed"):
        await agent.seed_agent_session(SimpleNamespace(session={}), body)
    if cleanup_fails:
        assert agent._agent_sessions["session"].phase is SessionPhase.CLOSING
        with pytest.raises(HTTPException, match="closing"):
            await agent.seed_agent_session(SimpleNamespace(session={}), body)
    else:
        assert not agent._agent_sessions
    sandbox.exec.side_effect = None
    cleanup.side_effect = None
    receipt = await agent.close_agent_session(
        SimpleNamespace(session={}),
        AgentCloseSessionRequest(agent_session_id="session", episode_id=body.episode_id),
    )
    assert receipt.agent_session_id == "session"
    assert not agent._agent_sessions
    assert not agent._session_reapers
    if owns_sandbox:
        sandbox.disconnect.assert_not_awaited()
    else:
        sandbox.stop.assert_not_awaited()
