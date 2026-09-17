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
from nooa import Agent, PredictStrategy, strategy
from nooa.config import PredictConfig
from pydantic import BaseModel

from nemo_gym.base_resources_server import AggregateMetricsRequest
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseOutputMessageForTraining,
    NeMoGymResponseOutputText,
)
from nemo_gym.rollout_collection import NG_FAILURE_CLASS_KEY, NG_TERMINAL_KEY
from nemo_gym.rollout_observability import AgentEpisode, AgentObservationBundle
from nemo_gym.server_utils import ServerClient
from responses_api_agents.nooa_agent.app import (
    NOOA_TERMINATION_ERROR_KEY,
    NOOA_TERMINATION_REASON_KEY,
    NOOAAgent,
    NOOAAgentRunRequest,
    NOOACookieConflictError,
    _identity,
    _merge_downstream_cookies,
)
from responses_api_agents.nooa_agent.config import NOOAAgentConfig, NOOAInvocationConfig
from responses_api_agents.nooa_agent.runner import (
    EmbeddedNOOARunner,
    NOOARunFailure,
    NOOARunResult,
)
from responses_api_agents.nooa_agent.tests.test_gym_llm import model_response
from responses_api_agents.nooa_agent.tests.test_runner import FailingAgent, WaitingAgent, policy_runner


async def invoke(agent: object, request: NeMoGymResponseCreateParamsNonStreaming) -> object:
    return agent, request


async def invoke_text(agent: Any, request: NeMoGymResponseCreateParamsNonStreaming) -> object:
    assert isinstance(request.input, list)
    content = request.input[-1].content
    assert isinstance(content, str)
    return await agent.analyze(content)


class FakeHTTPResponse:
    def __init__(self, payload: dict, *, status: int = 200, cookie: tuple[str, str] | None = None) -> None:
        self._payload = json.dumps(payload).encode()
        self.status = status
        self.ok = status < 400
        self.content = SimpleNamespace(read=self.read)
        self.cookies = SimpleCookie()
        if cookie:
            self.cookies[cookie[0]] = cookie[1]

    async def read(self) -> bytes:
        return self._payload

    def raise_for_status(self) -> None:
        if not self.ok:
            raise aiohttp.ClientResponseError(
                request_info=MagicMock(real_url="http://resources.test"),
                history=(),
                status=self.status,
                message=f"HTTP {self.status}",
            )


class StructuredAnswer(BaseModel):
    result: str


class StructuredRetryAgent(Agent):
    @strategy(PredictStrategy(config=PredictConfig(max_retries=3)))
    async def analyze(self, text: str) -> StructuredAnswer:
        """Return a structured answer for the supplied text."""

        ...


def config(**overrides: object) -> NOOAAgentConfig:
    values: dict[str, object] = {
        "name": "nooa_agent",
        "host": "127.0.0.1",
        "port": 9000,
        "entrypoint": "app.py",
        "resources_server": {"type": "resources_servers", "name": "resources"},
        "model_server": {"type": "responses_api_models", "name": "policy"},
        "nooa": {
            "agent_class": "responses_api_agents.nooa_agent.example_agent:GymResourceAgent",
            "invocation_adapter": f"{__name__}:invoke",
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
    run_request.model_cookies["session"] = "tool-cookie"
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


def make_structured_retry_agent(outputs: list[str], *, max_policy_calls: int) -> tuple[NOOAAgent, list[object]]:
    client = server_client()
    model_calls: list[object] = []
    remaining = iter(outputs)

    async def post(*, server_name: str, url_path: str, json: object, **_: object) -> FakeHTTPResponse:
        if server_name == "resources":
            if url_path == "/seed_session":
                return FakeHTTPResponse({})
            assert url_path == "/verify"
            assert isinstance(json, dict)
            return FakeHTTPResponse(json | {"reward": 0.75})
        assert server_name == "policy"
        model_calls.append(json)
        index = len(model_calls)
        message = NeMoGymResponseOutputMessageForTraining(
            id=f"msg-{index}",
            content=[NeMoGymResponseOutputText(annotations=[], text=next(remaining), logprobs=[])],
            prompt_token_ids=[index],
            generation_token_ids=[index + 10],
            generation_log_probs=[-0.1],
        )
        return FakeHTTPResponse(model_response(message, response_id=f"response-{index}"))

    object.__setattr__(client, "post", AsyncMock(side_effect=post))
    invocation = NOOAInvocationConfig(
        agent_class=f"{__name__}:StructuredRetryAgent",
        invocation_adapter=f"{__name__}:invoke_text",
    )
    agent = NOOAAgent(config=config(), server_client=client)
    agent.runner = EmbeddedNOOARunner(
        invocation=invocation,
        server_client=client,
        model_server_name="policy",
        resources_server_name="resources",
        max_policy_calls=max_policy_calls,
    )
    return agent, model_calls


def test_merge_downstream_cookies_allows_distinct_names_and_identical_values() -> None:
    assert _merge_downstream_cookies(
        {"model": "one", "shared": "same"},
        {"resource": "two", "shared": "same"},
    ) == {"model": "one", "resource": "two", "shared": "same"}


def test_merge_downstream_cookies_rejects_conflicting_values() -> None:
    with pytest.raises(NOOACookieConflictError, match="conflicting values.*'shared'"):
        _merge_downstream_cookies({"shared": "model"}, {"shared": "resource"})


def conflicting_runner_result(run_request: object) -> NOOARunResult:
    result = runner_result(run_request)
    result.model_cookies["shared"] = "model"
    result.resource_cookies["shared"] = "resource"
    return result


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["responses", "run"])
async def test_agent_endpoints_reject_conflicting_downstream_cookies(endpoint: str) -> None:
    agent, _ = make_agent()
    agent.runner.run = AsyncMock(side_effect=conflicting_runner_result)

    with pytest.raises(NOOACookieConflictError, match="'shared'"):
        if endpoint == "responses":
            await agent.responses(request(), Response(), body().responses_create_params)
        else:
            await agent.run(request(), Response(), body())


@pytest.mark.asyncio
async def test_partial_failure_rejects_conflicting_downstream_cookies() -> None:
    agent, _ = make_agent()

    def fail_with_partial(run_request: object) -> None:
        partial = conflicting_runner_result(run_request)
        raise NOOARunFailure(RuntimeError("agent failed"), partial)

    agent.runner.run = AsyncMock(side_effect=fail_with_partial)

    with pytest.raises(NOOACookieConflictError, match="'shared'"):
        await agent.run(request(), Response(), body())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("outputs", "max_policy_calls", "termination_reason", "expected_calls"),
    [
        pytest.param(["not json", '{"result":"done"}'], 3, None, 2, id="invalid-then-success"),
        pytest.param(
            ["not json", "still not json", "invalid again"],
            3,
            "invalid_policy_output",
            3,
            id="retries-exhausted",
        ),
        pytest.param(["not json"], 1, "policy_budget_exceeded", 1, id="budget-exhausted-during-retry"),
    ],
)
async def test_real_structured_output_retries_are_counted_with_partial_evidence(
    outputs: list[str],
    max_policy_calls: int,
    termination_reason: str | None,
    expected_calls: int,
) -> None:
    agent, model_calls = make_structured_retry_agent(outputs, max_policy_calls=max_policy_calls)

    result = await agent.run(request(), Response(), body())

    assert result.reward == 0.75
    assert NG_FAILURE_CLASS_KEY not in result.model_extra
    assert result.model_extra.get(NOOA_TERMINATION_REASON_KEY) == termination_reason
    assert len(model_calls) == expected_calls
    trajectory = result.model_extra["ng_trajectory"]
    assert len(trajectory["turns"]) == expected_calls
    assert all(turn["answer"] for turn in trajectory["turns"])
    assert [turn["model_calls"][0]["response_id"] for turn in trajectory["turns"]] == [
        f"response-{index}" for index in range(1, expected_calls + 1)
    ]
    assert len(result.response.output) == expected_calls


@pytest.mark.asyncio
async def test_run_uses_complete_row_seed_tool_and_verify_cookie_lifecycle() -> None:
    agent, client = make_agent()
    outgoing = Response()
    responses = iter(client.post.side_effect)
    requests: list[tuple[str, dict[str, str]]] = []

    async def post(**kwargs: object) -> FakeHTTPResponse:
        requests.append((str(kwargs["url_path"]), dict(kwargs["cookies"])))
        return next(responses)

    object.__setattr__(client, "post", AsyncMock(side_effect=post))

    result = await agent.run(request(), outgoing, body(customer_id="customer-42"))

    run_request = agent.runner.run.await_args.args[0]
    assert run_request.responses_create_params == body().responses_create_params
    assert run_request.resource_cookies == {"session": "tool-cookie", "verified": "yes"}
    assert requests == [
        ("/seed_session", {"session": "incoming"}),
        ("/verify", {"session": "tool-cookie"}),
    ]
    assert result.reward == 1.0
    assert result.ng_agent_observations is not None
    assert result.ng_agent_observations.gaps[0].code == "non_trainable_terminal_output"
    set_cookies = outgoing.headers.getlist("set-cookie")
    assert any("model=model-cookie" in header for header in set_cookies)
    assert any("session=tool-cookie" in header for header in set_cookies)
    assert any("verified=yes" in header for header in set_cookies)


@pytest.mark.asyncio
async def test_direct_responses_propagates_unrelated_value_error() -> None:
    agent, _ = make_agent()
    agent.runner.run = AsyncMock(side_effect=ValueError("agent implementation failed"))

    with pytest.raises(ValueError, match="agent implementation failed"):
        await agent.responses(
            request(),
            Response(),
            body().responses_create_params,
        )


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
async def test_direct_responses_forwards_model_and_resource_cookies() -> None:
    agent, _ = make_agent()
    outgoing = Response()

    await agent.responses(request(), outgoing, body().responses_create_params)

    set_cookies = outgoing.headers.getlist("set-cookie")
    assert any("model=model-cookie" in header for header in set_cookies)
    assert any("session=tool-cookie" in header for header in set_cookies)


@pytest.mark.parametrize(
    ("row", "rollout_id", "expected"),
    [
        (
            {
                "task_id": "task",
                "problem_id": "problem",
                "instance_id": "instance",
                "_ng_task_index": 3,
                "_ng_rollout_index": 4,
                "_ng_rollout_id": "body-rollout",
            },
            "path-rollout",
            {"task_id": "task", "rollout_id": "path-rollout"},
        ),
        (
            {
                "task_id": None,
                "problem_id": "problem",
                "instance_id": "instance",
                "_ng_task_index": 3,
                "_ng_rollout_index": 4,
            },
            None,
            {"task_id": "problem", "rollout_id": "3-4"},
        ),
    ],
)
def test_identity_uses_explicit_field_precedence(
    row: dict[str, object], rollout_id: str | None, expected: dict[str, str]
) -> None:
    assert _identity(body(**row), rollout_id) == expected


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
async def test_seed_http_failure_is_transient_and_skips_episode() -> None:
    client = server_client()
    object.__setattr__(client, "post", AsyncMock(return_value=FakeHTTPResponse({"error": "down"}, status=503)))
    agent = NOOAAgent(config=config(), server_client=client)
    agent.runner = MagicMock()
    agent.runner.run = AsyncMock()

    result = await agent.run(request(), Response(), body())

    assert result.model_extra[NG_FAILURE_CLASS_KEY] == "transient"
    assert "HTTP 503" in result.model_extra["error"]
    agent.runner.run.assert_not_awaited()


@pytest.mark.asyncio
async def test_verify_http_failure_preserves_completed_episode() -> None:
    agent, client = make_agent()
    object.__setattr__(
        client,
        "post",
        AsyncMock(side_effect=[FakeHTTPResponse({}), FakeHTTPResponse({"error": "down"}, status=503)]),
    )

    result = await agent.run(request(), Response(), body())

    assert result.model_extra[NG_FAILURE_CLASS_KEY] == "transient"
    assert result.response.output == []
    assert result.ng_agent_observations is not None
    assert [call.kwargs["url_path"] for call in client.post.await_args_list] == ["/seed_session", "/verify"]


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


@pytest.mark.asyncio
async def test_aggregate_metrics_proxies_success() -> None:
    client = server_client()
    response = FakeHTTPResponse(
        {
            "group_level_metrics": [{"task_id": "task-1", "reward": 1.0}],
            "agent_metrics": {"mean_reward": 1.0},
            "key_metrics": {"mean_reward": 1.0},
        }
    )
    object.__setattr__(client, "post", AsyncMock(return_value=response))
    agent = NOOAAgent(config=config(), server_client=client)
    request_body = AggregateMetricsRequest(verify_responses=[{"_ng_task_index": 0, "reward": 1.0}])

    result = await agent.aggregate_metrics(request_body)

    assert result.agent_metrics == {"mean_reward": 1.0}
    assert result.key_metrics == {"mean_reward": 1.0}
    client.post.assert_awaited_once_with(
        server_name="resources",
        url_path="/aggregate_metrics",
        json=request_body,
    )


@pytest.mark.asyncio
async def test_aggregate_metrics_enforces_timeout() -> None:
    client = server_client()

    async def delayed_post(**_: object) -> FakeHTTPResponse:
        await asyncio.sleep(1)
        return FakeHTTPResponse({})

    object.__setattr__(client, "post", AsyncMock(side_effect=delayed_post))
    agent = NOOAAgent(config=config(run_timeout_secs=0.001), server_client=client)

    with pytest.raises(TimeoutError):
        await agent.aggregate_metrics(AggregateMetricsRequest(verify_responses=[]))


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [ConnectionError("verifier unavailable"), FakeHTTPResponse({"reward": "invalid"})])
async def test_verifier_failure_preserves_completed_episode(failure: object) -> None:
    agent, client = make_agent()
    agent.runner, _ = policy_runner()
    object.__setattr__(client, "post", AsyncMock(side_effect=[FakeHTTPResponse({}), failure]))

    result = await agent.run(request(), Response(), body())

    assert result.reward == 0
    assert result.model_extra[NG_FAILURE_CLASS_KEY] == (
        "transient" if isinstance(failure, ConnectionError) else "legitimate"
    )
    assert len(result.model_extra["ng_trajectory"]["turns"]) == 1
    assert result.response.output
    assert result.ng_agent_observations.records


@pytest.mark.asyncio
async def test_agent_failure_preserves_completed_call_and_skips_verification() -> None:
    agent, client = make_agent()
    agent.runner, _ = policy_runner(FailingAgent)
    object.__setattr__(client, "post", AsyncMock(return_value=FakeHTTPResponse({})))

    result = await agent.run(request(), Response(), body())

    assert result.reward == 0
    assert result.model_extra[NG_FAILURE_CLASS_KEY] == "legitimate"
    assert len(result.model_extra["ng_trajectory"]["turns"]) == 1
    assert result.response.output
    assert client.post.await_count == 1


@pytest.mark.asyncio
async def test_episode_timeout_preserves_completed_call() -> None:
    agent, client = make_agent()
    agent.runner, _ = policy_runner(WaitingAgent)
    object.__setattr__(client, "post", AsyncMock(return_value=FakeHTTPResponse({})))
    agent.config.run_timeout_secs = 0.2
    WaitingAgent.ready = asyncio.Event()
    try:
        result = await agent.run(request(), Response(), body())
        assert WaitingAgent.ready.is_set()
    finally:
        WaitingAgent.ready = None

    assert result.model_extra[NG_FAILURE_CLASS_KEY] == "timeout_exceeded"
    assert result.model_extra[NG_TERMINAL_KEY] is True
    assert len(result.model_extra["ng_trajectory"]["turns"]) == 1
    assert result.response.output
    assert client.post.await_count == 1


@pytest.mark.asyncio
async def test_verifier_reply_does_not_drop_input_row_fields() -> None:
    agent, _ = make_agent()
    result = await agent.run(request(), Response(), body(agent_inputs={"customer": "alice"}))
    assert result.agent_inputs == {"customer": "alice"}
