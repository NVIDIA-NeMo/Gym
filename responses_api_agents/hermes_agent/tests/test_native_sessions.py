# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from nemo_gym.base_responses_api_agent import AgentCloseSessionRequest, AgentSeedSessionRequest
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_observability import AgentEpisode, AgentObservationBundle
from nemo_gym.server_utils import ServerClient
from responses_api_agents.hermes_agent.app import HermesAgent, HermesAgentConfig, HermesAgentSessionState


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
        assert client.post("/v1/agent_sessions", json=state.request.model_dump(mode="json")).status_code == 409
        path = f"/ng-rollout/{state.request.episode_id.capture_key}/v1/responses"
        assert client.post(path, json={"input": "task"}).status_code == 200
        assert client.post(path, json={"input": "task"}).status_code == 409
        close = {
            "agent_session_id": seed.json()["agent_session_id"],
            "episode_id": state.request.episode_id.model_dump(),
        }
        first = client.post("/v1/agent_sessions/close", json=close)
        retry = client.post("/v1/agent_sessions/close", json=close)
        assert first.status_code == retry.status_code == 200
        assert first.json() == retry.json()
        assert client.post(path, json={"input": "task"}).status_code == 409
        wrong = dict(close, episode_id={"rollout_id": "other"})
        assert client.post("/v1/agent_sessions/close", json=wrong).status_code == 409
    assert agent._run_sandbox_episode.await_count == 1
    state.sandbox.disconnect.assert_awaited_once()
    state.sandbox.stop.assert_not_awaited()


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
    with pytest.raises(HTTPException) as error:
        await agent.responses(request(state), body)
    assert error.value.status_code == 409
    close = AgentCloseSessionRequest(agent_session_id="session", episode_id=state.request.episode_id)
    first, second = await asyncio.gather(
        agent.close_agent_session(request(state), close), agent.close_agent_session(request(state), close)
    )
    assert first == second
    assert stopped.is_set()
    with pytest.raises(asyncio.CancelledError):
        await running
    state.sandbox.disconnect.assert_awaited_once()


async def test_close_failure_keeps_session_for_retry(agent, state):
    agent._agent_sessions["session"] = state
    state.sandbox.exec.side_effect = [SimpleNamespace(return_code=1), SimpleNamespace(return_code=0)]
    close = AgentCloseSessionRequest(agent_session_id="session", episode_id=state.request.episode_id)
    with pytest.raises(RuntimeError, match="session files"):
        await agent.close_agent_session(request(state), close)
    assert agent._agent_sessions["session"] is state
    state.sandbox.disconnect.assert_not_awaited()
    with pytest.raises(HTTPException):
        await agent.responses(request(state), NeMoGymResponseCreateParamsNonStreaming(input="task"))
    await agent.close_agent_session(request(state), close)
    state.sandbox.disconnect.assert_awaited_once()


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
    assert state.launch_started
    with pytest.raises(RuntimeError, match="launch outcome is unknown"):
        await agent._close_agent_session_state(state)
    state.sandbox.disconnect.assert_not_awaited()


@pytest.mark.parametrize("receipt", [{}, {"cleanup_confirmed": False}, {"cleanup_confirmed": "true"}])
async def test_runner_exit_without_cleanup_receipt_blocks_close(agent, state, receipt):
    state.launch_started = True
    state.runner_session = AsyncMock()
    state.runner_exit_task = asyncio.create_task(asyncio.sleep(0, result=0))
    await state.runner_exit_task
    agent._download_json = AsyncMock(return_value=receipt)
    with pytest.raises(RuntimeError, match="cleanup was not confirmed"):
        await agent._close_agent_session_state(state)
    state.runner_session.close.assert_not_awaited()
    state.sandbox.disconnect.assert_not_awaited()
    agent._download_json.return_value = {"cleanup_confirmed": True}
    await agent._close_agent_session_state(state)
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
