# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NOOA session integration with the existing single-agent-turn environment."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import orjson
import pytest
from nooa import Agent
from omegaconf import OmegaConf

from environment_servers.single_agent_turn.app import (
    SingleAgentTurnEnvironmentServer,
    SingleAgentTurnEnvironmentServerConfig,
)
from nemo_gym.base_responses_api_agent import AgentCloseSessionRequest, AgentSeedSessionRequest
from nemo_gym.client import ServerClient
from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming, current_rollout_id
from nemo_gym.rollout_collection import _episode_record
from nemo_gym.single_agent_turn_types import SingleAgentTurnRequest
from nemo_gym.trajectory import AgentInvocation, ToolCallObservation
from responses_api_agents.nooa_agent import resource_tools
from responses_api_agents.nooa_agent.app import NOOAAgent
from responses_api_agents.nooa_agent.result import finalize_run_result
from responses_api_agents.nooa_agent.runner import NOOARunFailure
from responses_api_agents.nooa_agent.termination import NOOATerminationReason
from responses_api_agents.nooa_agent.tests.test_app import agent, result
from responses_api_agents.nooa_agent.tests.test_app import request as agent_request
from responses_api_agents.nooa_agent.tests.test_config import agent_config, invocation_config
from responses_api_agents.nooa_agent.tests.test_resource_tools import FakeResponse, weather_tool


class _Response:
    ok = True

    def __init__(self, body: dict, *, cookies: dict[str, str] | None = None) -> None:
        self.body = body
        self.headers: dict[str, str] = {}
        self.cookies = {key: SimpleNamespace(value=value) for key, value in (cookies or {}).items()}

    async def read(self) -> bytes:
        return orjson.dumps(self.body)


def _request() -> SingleAgentTurnRequest:
    return SingleAgentTurnRequest.model_validate(
        {
            "episode_id": {"rollout_id": "rollout", "attempt": 2},
            "task": {
                "task_id": {"taskset": "tests", "task_id": "task"},
                "task_input": {
                    "responses_create_params": {"input": "task"},
                    "task_data": {"instance_id": "task"},
                },
            },
        }
    )


def _environment(instance: NOOAAgent) -> tuple[SingleAgentTurnEnvironmentServer, MagicMock]:
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = OmegaConf.create(
        {
            "agent": {"responses_api_agents": {"nooa_agent": {"entrypoint": "app.py", "token_id_capture": True}}},
            "token_id_capture": {"enabled": True},
        }
    )
    client._resolve_base_url.return_value = "http://resources:8000"
    request = agent_request(capture=True)

    async def post(*, server_name: str, url_path: str, **kwargs) -> _Response:
        assert current_rollout_id() == "rollout-a2"
        body = kwargs["json"]
        if server_name == "agent":
            if url_path == "/v1/agent_sessions":
                response = await instance.seed_agent_session(request, AgentSeedSessionRequest.model_validate(body))
            elif url_path.endswith("/v1/responses"):
                response = await instance.responses(
                    request, NeMoGymResponseCreateParamsNonStreaming.model_validate(body)
                )
            else:
                assert url_path == "/v1/agent_sessions/close"
                response = await instance.close_agent_session(request, AgentCloseSessionRequest.model_validate(body))
            return _Response(response.model_dump(mode="json"), cookies={"agent": "seed"})
        assert server_name == "resources"
        if url_path == "/seed_session":
            return _Response({"resources_session_id": body["resources_session_id"]}, cookies={"resource": "seed"})
        if url_path == "/verify":
            return _Response({**body, "reward": 1.0, "benchmark_field": "preserved"})
        if url_path == "/close_session":
            return _Response({"resources_session_id": body["resources_session_id"]})
        raise AssertionError(url_path)

    client.post = AsyncMock(side_effect=post)
    config = SingleAgentTurnEnvironmentServerConfig(
        name="nooa_environment",
        host="localhost",
        port=8002,
        entrypoint="app.py",
        resources_server={"type": "resources_servers", "name": "resources"},
        agent_server={"type": "responses_api_agents", "name": "agent"},
        cleanup_timeout_seconds=1,
        resources_tool_transports=["direct_http"],
    )
    return SingleAgentTurnEnvironmentServer(config=config, server_client=client), client


def _paths(client: MagicMock) -> list[str]:
    return [call.kwargs["url_path"] for call in client.post.call_args_list]


async def correct_resource_call(agent: Agent, request: NeMoGymResponseCreateParamsNonStreaming) -> str:
    feedback = await agent.get_weather(city="unknown")
    city = feedback["retry_city"] if isinstance(feedback, dict) else feedback.removeprefix("retry with city ")
    weather = await agent.get_weather(city=city)
    return f"{weather['weather']} in {city}"


async def swallow_resource_failure(agent: Agent, request: NeMoGymResponseCreateParamsNonStreaming) -> str:
    try:
        await agent.get_weather(city="unknown")
    except aiohttp.ClientError:
        pass
    weather = await agent.get_weather(city="Paris")
    return f"{weather['weather']} in Paris"


def _resource_environment(adapter: str) -> tuple[NOOAAgent, SingleAgentTurnEnvironmentServer, MagicMock]:
    instance = NOOAAgent(
        config=agent_config(nooa=invocation_config(invocation_adapter=f"{__name__}:{adapter}").model_dump()),
        server_client=MagicMock(spec=ServerClient),
    )
    environment, client = _environment(instance)
    return instance, environment, client


@pytest.mark.parametrize(
    ("status", "feedback"),
    [
        (400, {"detail": "unknown city", "retry_city": "Paris"}),
        (401, {"detail": "retry with refreshed session", "retry_city": "Paris"}),
        (403, {"detail": "choose an allowed city", "retry_city": "Paris"}),
        (429, "retry with city Paris"),
    ],
)
async def test_resource_client_error_feedback_allows_correction_and_verification(
    monkeypatch: pytest.MonkeyPatch, status: int, feedback: dict[str, str] | str
) -> None:
    body = feedback.encode() if isinstance(feedback, str) else feedback
    outbound = AsyncMock(
        side_effect=[
            FakeResponse(body, status=status, cookie=("session", "feedback")),
            FakeResponse({"weather": "sunny"}, cookie=("session", "corrected")),
        ]
    )
    monkeypatch.setattr(resource_tools, "request", outbound)
    instance, environment, client = _resource_environment("correct_resource_call")
    request = _request()
    request.task.task_input.responses_create_params = NeMoGymResponseCreateParamsNonStreaming(
        input="task", tools=[{**weather_tool(), "strict": False}]
    )

    response = await environment.run_request(request)

    assert response.failure is None
    assert response.result.reward == 1
    assert response.result.response.output[-1].content[0].text == "sunny in Paris"
    outputs = [item for item in response.result.response.output if item.type == "function_call_output"]
    assert (outputs[0].output if isinstance(feedback, str) else orjson.loads(outputs[0].output)) == feedback
    assert orjson.loads(outputs[1].output) == {"weather": "sunny"}
    assert [call.kwargs["json"]["city"] for call in outbound.await_args_list] == ["unknown", "Paris"]
    assert outbound.await_args_list[0].kwargs["cookies"] == {"resource": "seed"}
    assert outbound.await_args_list[1].kwargs["cookies"] == {"resource": "seed", "session": "feedback"}
    assert _paths(client)[-3:] == ["/v1/agent_sessions/close", "/verify", "/close_session"]
    receipt = next(iter(instance._closed_session_records.values())).close_response
    assert receipt.resources_cookies == {"resource": "seed", "session": "corrected"}
    assert client.post.call_args_list[-2].kwargs["cookies"] == receipt.resources_cookies
    assert client.post.call_args_list[-1].kwargs["cookies"] == receipt.resources_cookies
    observations = response.result.ng_agent_observations
    assert observations == receipt.agent_observations
    assert {gap.code for gap in observations.gaps} == {"non_trainable_terminal_output"}
    tools = [record for record in observations.records if isinstance(record, ToolCallObservation)]
    assert [(tool.status, tool.error_type) for tool in tools] == [("failed", f"http_{status}"), ("completed", None)]


@pytest.mark.parametrize("failure", [503, 599, "transport"])
async def test_swallowed_resource_failure_still_skips_verification_and_closes_with_evidence(
    monkeypatch: pytest.MonkeyPatch, failure: int | str
) -> None:
    first = (
        FakeResponse({"detail": "unavailable"}, status=failure, cookie=("session", "feedback"))
        if isinstance(failure, int)
        else aiohttp.ClientConnectionError("resource connection lost")
    )
    outbound = AsyncMock(side_effect=[first, FakeResponse({"weather": "sunny"}, cookie=("session", "corrected"))])
    monkeypatch.setattr(resource_tools, "request", outbound)
    instance, environment, client = _resource_environment("swallow_resource_failure")
    request = _request()
    request.task.task_input.responses_create_params = NeMoGymResponseCreateParamsNonStreaming(
        input="task", tools=[{**weather_tool(), "strict": False}]
    )

    response = await environment.run_request(request)

    assert response.failure.stage == "agent"
    assert "503: NOOA dependency request failed" in response.failure.failure_reason
    assert response.result is None
    assert [call.kwargs["json"]["city"] for call in outbound.await_args_list] == ["unknown", "Paris"]
    assert outbound.await_args_list[0].kwargs["cookies"] == {"resource": "seed"}
    expected_retry_cookies = (
        {"resource": "seed", "session": "feedback"} if isinstance(failure, int) else {"resource": "seed"}
    )
    assert outbound.await_args_list[1].kwargs["cookies"] == expected_retry_cookies
    assert "/verify" not in _paths(client)
    assert _paths(client)[-2:] == ["/v1/agent_sessions/close", "/close_session"]
    receipt = next(iter(instance._closed_session_records.values())).close_response
    assert receipt.resources_cookies == {"resource": "seed", "session": "corrected"}
    assert client.post.call_args_list[-1].kwargs["cookies"] == receipt.resources_cookies
    observations = receipt.agent_observations
    assert NOOATerminationReason.INFRASTRUCTURE_ERROR in {gap.code for gap in observations.gaps}
    tools = [record for record in observations.records if isinstance(record, ToolCallObservation)]
    error_type = "ClientResponseError" if isinstance(failure, int) else "ClientConnectionError"
    assert [(tool.status, tool.error_type) for tool in tools] == [("failed", error_type), ("completed", None)]
    outputs = [
        item
        for record in observations.records
        if isinstance(record, AgentInvocation)
        for item in record.conversation
        if item.type == "function_call_output"
    ]
    assert orjson.loads(outputs[0].output)["error_type"] == error_type
    assert orjson.loads(outputs[1].output) == {"weather": "sunny"}


async def test_nooa_close_precedes_verification_with_final_cookies_and_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance, runner_run = agent(monkeypatch)
    environment, client = _environment(instance)
    request = _request()
    response = await environment.run_request(request)
    assert _paths(client) == [
        "/seed_session",
        "/v1/agent_sessions",
        "/ng-rollout/rollout-a2/training-token-capture/v1/responses",
        "/v1/agent_sessions/close",
        "/verify",
        "/close_session",
    ]
    assert response.failure is None
    assert response.result.response.output[-1].content[0].text == "answer"
    assert response.result.reward == 1
    assert response.result.ng_agent_observations.source == "nooa"
    calls = client.post.call_args_list
    assert calls[1].kwargs["json"]["tool_accesses"][0]["cookies"] == {"resource": "seed"}
    assert calls[2].kwargs["cookies"] == {"agent": "seed"}
    assert calls[4].kwargs["cookies"] == calls[5].kwargs["cookies"] == {"session": "resource-cookie"}
    assert calls[4].kwargs["json"]["instance_id"] == "task"
    run_request = runner_run.call_args.args[0]
    assert run_request.resource_cookies == {"resource": "seed"}
    assert run_request.model_url_path == "/ng-rollout/rollout-a2/training-token-capture/v1/responses"
    assert run_request.rollout_id == "rollout-a2"
    assert run_request.task_id == "task"
    runner_run.assert_awaited_once()
    record = _episode_record(response.model_dump(mode="json"))
    assert record["ng_agent_observations"]["source"] == "nooa"
    assert record["benchmark_field"] == "preserved"
    assert "ng_trajectory" not in record
    assert "agent_timed_out" not in record
    assert "agent_timeout_seconds" not in record
    assert request.task.task_input.task_data == {"instance_id": "task"}


async def test_failed_nooa_activation_skips_verification_and_closes_with_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance, runner_run = agent(monkeypatch)
    failure = NOOARunFailure(ValueError("adapter defect"), result())
    runner_run.side_effect = failure
    environment, client = _environment(instance)
    response = await environment.run_request(_request())
    assert response.failure.stage == "agent"
    assert response.result is None
    assert "/verify" not in _paths(client)
    assert _paths(client)[-2:] == ["/v1/agent_sessions/close", "/close_session"]
    receipt = next(iter(instance._closed_session_records.values())).close_response
    assert NOOATerminationReason.INFRASTRUCTURE_ERROR in {gap.code for gap in receipt.agent_observations.gaps}


@pytest.mark.parametrize("cancel", [False, True], ids=["episode-timeout", "caller-cancellation"])
async def test_interrupted_nooa_episode_stops_execution_without_verification(
    monkeypatch: pytest.MonkeyPatch, cancel: bool
) -> None:
    instance, runner_run = agent(monkeypatch)
    partial = result()
    partial.return_value = None
    started, stopped = asyncio.Event(), asyncio.Event()

    async def run(_):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError as error:
            error.nooa_result = partial
            stopped.set()
            raise

    runner_run.side_effect = run
    environment, client = _environment(instance)
    environment.config.default_episode_timeout_seconds = None if cancel else 0.05
    task = asyncio.create_task(environment.run_request(_request()))
    await asyncio.wait_for(started.wait(), 1)
    if cancel:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        response = await asyncio.wait_for(task, 1)
        assert response.result is None
        assert response.failure.failure_reason == "Episode timed out"
        assert response.failure.partial_response is None
    assert stopped.is_set()
    assert "/verify" not in _paths(client)
    assert _paths(client)[-2:] == ["/v1/agent_sessions/close", "/close_session"]
    receipt = next(iter(instance._closed_session_records.values())).close_response
    assert NOOATerminationReason.CANCELLED in {gap.code for gap in receipt.agent_observations.gaps}
    if cancel:
        assert partial.termination_reason is NOOATerminationReason.CANCELLED
        projected_response, observations = finalize_run_result(partial)
        assert projected_response.status == "cancelled"
        assert projected_response.error is None
        assert projected_response.output == partial.episode.response.output
        assert observations == receipt.agent_observations
