# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import json
from http.cookies import SimpleCookie
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest
from fastapi import HTTPException, Response

from nemo_gym.base_resources_server import AggregateMetricsRequest
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.rollout_collection import NG_FAILURE_CLASS_KEY, NG_TERMINAL_KEY
from nemo_gym.rollout_observability import AgentEpisode, AgentObservationBundle
from nemo_gym.server_utils import ServerClient
from responses_api_agents.nooa_agent.app import (
    NOOA_TERMINATION_ERROR_KEY,
    NOOA_TERMINATION_REASON_KEY,
    NOOAAgent,
    NOOAAgentRunRequest,
)
from responses_api_agents.nooa_agent.config import NOOAAgentConfig
from responses_api_agents.nooa_agent.runner import NOOARunResult


class FakeHTTPResponse:
    ok = True

    def __init__(self, payload: dict, *, status: int = 200, cookie: tuple[str, str] | None = None) -> None:
        self._payload = json.dumps(payload).encode()
        self.status = status
        self.cookies = SimpleCookie()
        if cookie:
            self.cookies[cookie[0]] = cookie[1]

    async def read(self) -> bytes:
        return self._payload

    def raise_for_status(self) -> None:
        if not self.ok:
            raise RuntimeError(f"HTTP {self.status}")


def config(**overrides: object) -> NOOAAgentConfig:
    values: dict[str, object] = {
        "name": "nooa_agent",
        "host": "127.0.0.1",
        "port": 9000,
        "entrypoint": "app.py",
        "resources_server": {"type": "resources_servers", "name": "resources"},
        "model_server": {"type": "responses_api_models", "name": "policy"},
        "nooa": {
            "agent_class": "responses_api_agents.nooa_agent.example_agent:WeatherAgent",
            "entrypoint": "answer",
            "arguments": {
                "question": {
                    "source": "responses_create_params.input",
                    "transform": "latest_user_text",
                }
            },
        },
        "run_timeout_secs": 1,
    }
    values.update(overrides)
    return NOOAAgentConfig.model_validate(values)


def body(**extras: object) -> NOOAAgentRunRequest:
    return NOOAAgentRunRequest.model_validate(
        {
            "responses_create_params": {"input": [{"role": "user", "content": "Weather in Paris?"}]},
            "task_id": "task-1",
        }
        | extras
    )


def request(
    cookie: str = "incoming",
    *,
    rollout_id: str | None = None,
    token_capture: bool = False,
) -> SimpleNamespace:
    prefix = f"/ng-rollout/{rollout_id}" if rollout_id is not None else ""
    if token_capture:
        prefix += "/training-token-capture"
    return SimpleNamespace(
        cookies={"session": cookie},
        path_params={"rollout_id": rollout_id} if rollout_id is not None else {},
        url=SimpleNamespace(path=f"{prefix}/v1/responses"),
    )


def episode() -> AgentEpisode:
    return AgentEpisode(
        response=NeMoGymResponse(
            id="nooa-test",
            created_at=0,
            model="nooa",
            object="response",
            output=[],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        ),
        observations=AgentObservationBundle(source="nooa", records=[], gaps=[]),
    )


def runner_result(run_request: object) -> NOOARunResult:
    run_request.model_cookies["model"] = "model-cookie"
    run_request.resource_cookies["session"] = "tool-cookie"
    return NOOARunResult(
        episode=episode(),
        return_value="The weather is cold.",
        model_cookies=run_request.model_cookies,
        resource_cookies=run_request.resource_cookies,
    )


def server_client() -> ServerClient:
    return ServerClient.model_construct(head_server_config=MagicMock(), global_config_dict={})


def make_agent(*, verify_reward: float = 1.0) -> tuple[NOOAAgent, ServerClient]:
    client = server_client()
    seed = FakeHTTPResponse({}, cookie=("session", "seed-cookie"))
    verify = FakeHTTPResponse(
        {
            "responses_create_params": {"input": [{"role": "user", "content": "Weather in Paris?"}]},
            "task_id": "task-1",
            "response": {
                "id": "verified",
                "created_at": 0,
                "model": "nooa",
                "object": "response",
                "output": [],
                "parallel_tool_calls": False,
                "tool_choice": "none",
                "tools": [],
            },
            "reward": verify_reward,
        },
        cookie=("verified", "yes"),
    )
    object.__setattr__(client, "post", AsyncMock(side_effect=[seed, verify]))
    agent = NOOAAgent(config=config(), server_client=client)
    agent.runner = MagicMock()
    agent.runner.run = AsyncMock(side_effect=runner_result)
    return agent, client


@pytest.mark.asyncio
async def test_run_uses_complete_row_seed_tool_and_verify_cookie_lifecycle() -> None:
    agent, client = make_agent()
    outgoing = Response()

    result = await agent.run(request(), outgoing, body(customer_id="customer-42"))

    run_request = agent.runner.run.await_args.args[0]
    assert run_request.row.customer_id == "customer-42"
    assert run_request.resource_cookies == {"session": "tool-cookie", "verified": "yes"}
    assert client.post.await_args_list[0].kwargs["cookies"] == {
        "session": "tool-cookie",
        "verified": "yes",
    }
    assert client.post.await_args_list[1].kwargs["cookies"] == {
        "session": "tool-cookie",
        "verified": "yes",
    }
    assert result.reward == 1.0
    assert result.ng_agent_observations is not None
    assert result.ng_agent_observations.gaps[0].code == "non_trainable_fallback_output"
    assert "session=tool-cookie" in outgoing.headers.get("set-cookie", "")


@pytest.mark.asyncio
async def test_direct_responses_reports_missing_top_level_mapping() -> None:
    agent, _ = make_agent()
    agent.runner.run = AsyncMock(side_effect=ValueError("source 'customer_id' does not exist"))

    with pytest.raises(HTTPException, match="customer_id") as error:
        await agent.responses(
            request(),
            Response(),
            body().responses_create_params,
        )

    assert error.value.status_code == 422


@pytest.mark.asyncio
async def test_direct_responses_preserves_inbound_rollout_prefix() -> None:
    agent, _ = make_agent()

    await agent.responses(
        request(rollout_id="direct-rollout", token_capture=True),
        Response(),
        body().responses_create_params,
    )

    run_request = agent.runner.run.await_args.args[0]
    assert run_request.model_url_path == "/ng-rollout/direct-rollout/training-token-capture/v1/responses"


@pytest.mark.asyncio
async def test_direct_responses_returns_atif_episode_without_verifier_fallback() -> None:
    agent, _ = make_agent()

    response = await agent.responses(request(), Response(), body().responses_create_params)

    assert response.output == []


@pytest.mark.asyncio
async def test_unexpected_harness_failure_remains_legitimate() -> None:
    agent, _ = make_agent()
    agent.runner.run = AsyncMock(side_effect=RuntimeError("model unavailable"))

    result = await agent.run(request(), Response(), body())

    assert result.reward == 0
    assert result.model_extra[NG_FAILURE_CLASS_KEY] == "legitimate"
    assert "model unavailable" in result.model_extra["error"]


@pytest.mark.parametrize(
    "error",
    [
        TimeoutError("downstream timeout"),
        aiohttp.ClientConnectionError("connection reset"),
        aiohttp.ClientResponseError(
            request_info=MagicMock(real_url="http://resources/verify"),
            history=(),
            status=503,
        ),
    ],
)
@pytest.mark.asyncio
async def test_downstream_infrastructure_failure_is_transient(error: Exception) -> None:
    client = server_client()
    object.__setattr__(client, "post", AsyncMock(side_effect=error))
    agent = NOOAAgent(config=config(), server_client=client)
    agent.runner = MagicMock()
    agent.runner.run = AsyncMock()

    result = await agent.run(request(), Response(), body())

    assert result.reward == 0
    assert result.model_extra[NG_FAILURE_CLASS_KEY] == "transient"
    assert NG_TERMINAL_KEY not in result.model_extra
    agent.runner.run.assert_not_awaited()


@pytest.mark.asyncio
async def test_mid_episode_downstream_timeout_is_not_the_episode_budget() -> None:
    agent, _ = make_agent()
    agent.runner.run = AsyncMock(side_effect=TimeoutError("NOOA queue read timed out"))

    result = await agent.run(request(), Response(), body())

    assert result.model_extra[NG_FAILURE_CLASS_KEY] == "transient"
    assert NG_TERMINAL_KEY not in result.model_extra
    assert "queue read timed out" in result.model_extra["error"]


@pytest.mark.asyncio
async def test_policy_budget_exhaustion_is_verified_and_counted() -> None:
    agent, client = make_agent(verify_reward=0.0)

    def budget_exhausted(run_request: object) -> NOOARunResult:
        result = runner_result(run_request)
        result.return_value = None
        result.termination_reason = "policy_budget_exceeded"
        result.termination_error = "NOOA policy call budget exhausted after 1 calls"
        return result

    agent.runner.run = AsyncMock(side_effect=budget_exhausted)

    result = await agent.run(request(), Response(), body())

    assert result.reward == 0.0
    assert NG_FAILURE_CLASS_KEY not in result.model_extra
    assert result.model_extra[NOOA_TERMINATION_REASON_KEY] == "policy_budget_exceeded"
    assert "exhausted after 1 calls" in result.model_extra[NOOA_TERMINATION_ERROR_KEY]
    assert result.ng_agent_observations.gaps[0].code == "policy_budget_exceeded"
    assert [call.kwargs["url_path"] for call in client.post.await_args_list] == ["/seed_session", "/verify"]


@pytest.mark.asyncio
async def test_invalid_policy_output_is_verified_and_counted() -> None:
    agent, client = make_agent(verify_reward=0.0)

    def invalid_output(run_request: object) -> NOOARunResult:
        result = runner_result(run_request)
        result.return_value = None
        result.termination_reason = "invalid_policy_output"
        result.termination_error = "Gym model returned invalid Answer JSON"
        return result

    agent.runner.run = AsyncMock(side_effect=invalid_output)

    result = await agent.run(request(), Response(), body())

    assert result.reward == 0.0
    assert NG_FAILURE_CLASS_KEY not in result.model_extra
    assert result.model_extra[NOOA_TERMINATION_REASON_KEY] == "invalid_policy_output"
    assert result.ng_agent_observations.gaps[0].code == "invalid_policy_output"
    assert [call.kwargs["url_path"] for call in client.post.await_args_list] == ["/seed_session", "/verify"]


@pytest.mark.asyncio
async def test_whole_run_timeout_is_terminal() -> None:
    agent, _ = make_agent()
    agent.config.run_timeout_secs = 0.001

    async def blocked(*args: object, **kwargs: object) -> object:
        await asyncio.sleep(1)
        return runner_result(args[0])

    agent.runner.run = AsyncMock(side_effect=blocked)

    result = await agent.run(request(), Response(), body())

    assert result.model_extra[NG_FAILURE_CLASS_KEY] == "timeout_exceeded"
    assert result.model_extra[NG_TERMINAL_KEY] is True


@pytest.mark.asyncio
async def test_slow_verification_is_outside_episode_timeout_budget() -> None:
    agent, client = make_agent()
    agent.config.run_timeout_secs = 0.001
    responses = client.post.side_effect

    async def delayed_verify(*args: object, **kwargs: object) -> FakeHTTPResponse:
        if kwargs["url_path"] == "/verify":
            await asyncio.sleep(0.01)
        return next(responses)

    object.__setattr__(client, "post", AsyncMock(side_effect=delayed_verify))

    result = await agent.run(request(), Response(), body())

    assert result.reward == 1.0
    assert NG_FAILURE_CLASS_KEY not in result.model_extra


@pytest.mark.asyncio
async def test_skip_verification_and_aggregate_metrics_proxy() -> None:
    client = server_client()
    object.__setattr__(
        client,
        "post",
        AsyncMock(
            side_effect=[
                FakeHTTPResponse({}),
                FakeHTTPResponse(
                    {
                        "mean_reward": 0.5,
                    }
                ),
            ]
        ),
    )
    agent = NOOAAgent(config=config(skip_verification=True, skip_verification_reward=0.25), server_client=client)
    agent.runner = MagicMock()
    agent.runner.run = AsyncMock(side_effect=runner_result)

    result = await agent.run(request(), Response(), body())

    assert result.reward == 0.25
    assert result.model_extra["verification_skipped"] is True
    # Skip mode uses Gym's local aggregate implementation and must not make another server call.
    await agent.aggregate_metrics(AggregateMetricsRequest(verify_responses=[]))
    assert client.post.await_count == 1
