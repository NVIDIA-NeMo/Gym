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

import json
from http.cookies import SimpleCookie
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from nooa import Agent
from nooa.agentdoc import doc
from nooa.runtime.hooks import get_hooks, set_hooks
from nooa.tracing import get_session
from pydantic import BaseModel, ConfigDict

from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCallForTraining,
)
from responses_api_agents.nooa_agent.config import NOOAInvocationConfig
from responses_api_agents.nooa_agent.gym_llm import InvalidPolicyOutputError, PolicyCallBudgetExceeded
from responses_api_agents.nooa_agent.runner import EmbeddedNOOARunner, NOOARunFailure, NOOARunRequest


class ValidAgent(Agent):
    def __init__(self, *, llm: Any, label: str) -> None:
        super().__init__(llm=llm)
        self.label = label

    async def analyze(self, text: str, customer_id: str) -> str: ...


class FakeAgent:
    instances = 0

    def __init__(self, *, llm: Any, label: str) -> None:
        FakeAgent.instances += 1
        self.llm = llm
        self.label = label
        self.event_manager = FakeEventManager()

    async def analyze(self, text: str, customer_id: str) -> str:
        weather = await self.weather.get_weather(city=customer_id)
        return f"{text}: {weather['weather']}"


class ExplodingAgent(FakeAgent):
    async def analyze(self, text: str, customer_id: str) -> str:
        await self.weather.get_weather(city=customer_id)
        raise RuntimeError("agent exploded")


class BudgetExhaustedAgent(FakeAgent):
    async def analyze(self, text: str, customer_id: str) -> str:
        await self.weather.get_weather(city=customer_id)
        raise PolicyCallBudgetExceeded("NOOA policy call budget exhausted after 1 calls")


class InvalidOutputAgent(FakeAgent):
    async def analyze(self, text: str, customer_id: str) -> str:
        raise InvalidPolicyOutputError("Gym model returned invalid Answer JSON")


class FakeEventManager:
    def on(self, event_type: str, handler: Any) -> Any:
        return lambda: None


class Row(BaseModel):
    model_config = ConfigDict(extra="allow")

    responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    customer_id: str


class FakeContent:
    async def read(self) -> bytes:
        return json.dumps({"weather": "cold"}).encode()


class FakeResponse:
    ok = True
    status = 200
    content = FakeContent()
    cookies = SimpleCookie()

    async def read(self) -> bytes:
        return await self.content.read()


def make_runner() -> tuple[EmbeddedNOOARunner, MagicMock]:
    invocation = NOOAInvocationConfig.model_validate(
        {
            "agent_class": f"{__name__}:ValidAgent",
            "entrypoint": "analyze",
            "init_kwargs": {"label": "configured"},
            "tool_namespace": "weather",
            "allowed_tools": ["get_weather"],
            "arguments": {
                "text": {
                    "source": "responses_create_params.input",
                    "transform": "latest_user_text",
                },
                "customer_id": {"source": "customer_id"},
            },
        }
    )
    client = MagicMock()
    client.post = AsyncMock(return_value=FakeResponse())
    runner = EmbeddedNOOARunner(
        invocation=invocation,
        server_client=client,
        model_server_name="policy_model",
        resources_server_name="weather_resources",
        max_steps=3,
    )
    runner._agent_class = FakeAgent
    return runner, client


def row(customer_id: str) -> Row:
    return Row(
        customer_id=customer_id,
        responses_create_params={
            "input": [{"role": "user", "content": "Check delivery"}],
            "tools": [
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get weather",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                        "additionalProperties": False,
                    },
                }
            ],
        },
    )


@pytest.mark.asyncio
async def test_embedded_runner_maps_full_row_and_attaches_named_tools() -> None:
    runner, client = make_runner()

    result = await runner.run(
        NOOARunRequest(
            row=row("Paris"),
            rollout_id="rollout-1",
            task_id="task-1",
            model_url_path="/ng-rollout/rollout-1/v1/responses",
            resource_cookies={"session": "one"},
        )
    )

    assert result.return_value == "Check delivery: cold"
    assert result.agent.label == "configured"
    assert result.agent.llm.model == "gym-policy"
    rendered = doc(type(result.agent))
    assert "weather: WeatherTools" in rendered
    assert "async def get_weather(self, city: str)" in rendered
    assert "dispatcher" not in rendered
    assert result.tool_executions[0].name == "get_weather"
    assert client.post.await_args.kwargs["json"] == {"city": "Paris"}


@pytest.mark.asyncio
async def test_constructs_a_fresh_agent_for_every_rollout() -> None:
    runner, _ = make_runner()
    FakeAgent.instances = 0

    first = await runner.run(
        NOOARunRequest(
            row=row("Paris"),
            rollout_id="one",
            task_id="task",
            model_url_path="/one/v1/responses",
        )
    )
    second = await runner.run(
        NOOARunRequest(
            row=row("Berlin"),
            rollout_id="two",
            task_id="task",
            model_url_path="/two/v1/responses",
        )
    )

    assert FakeAgent.instances == 2
    assert first.agent is not second.agent
    assert first.resource_cookies is not second.resource_cookies


@pytest.mark.asyncio
async def test_policy_budget_exhaustion_returns_partial_execution() -> None:
    runner, _ = make_runner()
    runner._agent_class = BudgetExhaustedAgent

    result = await runner.run(
        NOOARunRequest(
            row=row("Paris"),
            rollout_id="budget-exhausted",
            task_id="task",
            model_url_path="/budget-exhausted/v1/responses",
        )
    )

    assert result.return_value is None
    assert result.termination_reason == "policy_budget_exceeded"
    assert "exhausted after 1 calls" in result.termination_error
    assert result.tool_executions[0].name == "get_weather"
    assert [item.type for item in result.trace.output] == ["function_call", "function_call_output"]


@pytest.mark.asyncio
async def test_invalid_policy_output_returns_counted_termination() -> None:
    runner, _ = make_runner()
    runner._agent_class = InvalidOutputAgent

    result = await runner.run(
        NOOARunRequest(
            row=row("Paris"),
            rollout_id="invalid-output",
            task_id="task",
            model_url_path="/invalid-output/v1/responses",
        )
    )

    assert result.return_value is None
    assert result.termination_reason == "invalid_policy_output"
    assert "invalid Answer JSON" in result.termination_error


@pytest.mark.asyncio
async def test_real_nooa_codeact_rollout_calls_generated_gym_tool() -> None:
    execute_output = NeMoGymResponseFunctionToolCallForTraining(
        id="fc-1",
        call_id="code-1",
        name="execute_python",
        arguments=json.dumps(
            {"code": 'weather = await self.weather.get_weather(city="Paris")\nprint(weather["weather"])'}
        ),
        prompt_token_ids=[1, 2],
        generation_token_ids=[3, 4],
        generation_log_probs=[-0.1, -0.2],
    )
    return_output = NeMoGymResponseFunctionToolCallForTraining(
        id="fc-2",
        call_id="return-1",
        name="return_result",
        arguments=json.dumps({"result": "cold"}),
        prompt_token_ids=[1, 2, 3, 4],
        generation_token_ids=[5],
        generation_log_probs=[-0.05],
    )
    first_response = NeMoGymResponse(
        id="resp-1",
        created_at=0,
        model="policy",
        object="response",
        output=[execute_output],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
    )
    second_response = NeMoGymResponse(
        id="resp-2",
        created_at=0,
        model="policy",
        object="response",
        output=[return_output],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
    )

    class RoutedClient:
        def __init__(self) -> None:
            self.model_requests: list[Any] = []

        async def post(self, *, server_name: str, **kwargs: Any) -> Any:
            if server_name == "policy":
                self.model_requests.append(kwargs["json"])
                selected = first_response if len(self.model_requests) == 1 else second_response
                payload = selected.model_dump(mode="json")

                class ModelContent:
                    async def read(self) -> bytes:
                        return json.dumps(payload).encode()

                model_http = FakeResponse()
                model_http.content = ModelContent()
                return model_http
            return FakeResponse()

    invocation = NOOAInvocationConfig.model_validate(
        {
            "agent_class": "responses_api_agents.nooa_agent.example_agent:WeatherAgent",
            "entrypoint": "answer",
            "tool_namespace": "weather",
            "allowed_tools": ["get_weather"],
            "arguments": {
                "question": {
                    "source": "responses_create_params.input",
                    "transform": "latest_user_text",
                }
            },
        }
    )
    client = RoutedClient()
    runner = EmbeddedNOOARunner(
        invocation=invocation,
        server_client=client,
        model_server_name="policy",
        resources_server_name="weather",
        max_steps=2,
    )

    result = await runner.run(
        NOOARunRequest(
            row=row("Paris"),
            rollout_id="real-rollout",
            task_id="real-task",
            model_url_path="/ng-rollout/real-rollout/v1/responses",
        )
    )

    assert result.return_value == "cold"
    assert result.model_requests == client.model_requests
    assert result.model_requests is not client.model_requests
    assert [response.id for response in result.model_responses] == ["resp-1", "resp-2"]
    assert result.model_responses[0].output[0].generation_token_ids == [3, 4]
    prior_call = next(
        item for item in client.model_requests[1].input if item.type == "function_call" and item.call_id == "code-1"
    )
    assert prior_call.generation_token_ids == [3, 4]
    code_output = next(
        item
        for item in client.model_requests[1].input
        if item.type == "function_call_output" and item.call_id == "code-1"
    )
    assert code_output.output == "status: complete"
    python_output = next(
        item
        for item in result.model_requests[1].input
        if item.type == "message" and "PythonOutput" in str(item.content) and "cold" in str(item.content)
    )
    assert "cold" in str(python_output.content)
    assert result.tool_executions[0].name == "get_weather"
    assert result.trace.invocations
    root = next(invocation for invocation in result.trace.invocations if invocation.parent_invocation_id is None)
    assert root.status == "completed"
    assert [reference.response_id for reference in root.model_calls] == ["resp-1", "resp-2"]
    assert result.tool_executions[0].invocation_id == root.invocation_id
    assert [item.type for item in result.trace.output] == [
        "function_call",
        "function_call",
        "function_call_output",
        "function_call",
    ]


@pytest.mark.asyncio
async def test_real_runner_composes_and_restores_preexisting_nooa_hooks() -> None:
    calls: list[str] = []

    class ExistingHooks:
        def before_agent_call(self, **kwargs: Any) -> str:
            calls.append(f"before:{kwargs['method_name']}:{get_session()}")
            return kwargs["call_id"]

        def after_agent_call(self, *, context: Any, **kwargs: Any) -> None:
            calls.append(f"after:{kwargs['method_name']}:{context}")

        def before_generation(self, **kwargs: Any) -> None:
            return None

        def after_generation(self, **kwargs: Any) -> None:
            return None

        def before_code_execution(self, **kwargs: Any) -> None:
            return None

        def after_code_execution(self, **kwargs: Any) -> None:
            return None

        def before_method_invocation(self, **kwargs: Any) -> None:
            return None

        def after_method_invocation(self, **kwargs: Any) -> None:
            return None

        def before_tool_execution(self, **kwargs: Any) -> None:
            return None

        def after_tool_execution(self, **kwargs: Any) -> None:
            return None

        def on_messages_built(self, **kwargs: Any) -> None:
            return None

    execute_output = NeMoGymResponseFunctionToolCallForTraining(
        id="fc-1",
        call_id="return-1",
        name="return_result",
        arguments=json.dumps({"result": "cold"}),
        prompt_token_ids=[1],
        generation_token_ids=[2],
        generation_log_probs=[-0.1],
    )
    model_result = NeMoGymResponse(
        id="resp-1",
        created_at=0,
        model="policy",
        object="response",
        output=[execute_output],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
    )

    class RoutedClient:
        async def post(self, *, server_name: str, **kwargs: Any) -> Any:
            payload = model_result.model_dump(mode="json")

            class ModelContent:
                async def read(self) -> bytes:
                    return json.dumps(payload).encode()

            response = FakeResponse()
            response.content = ModelContent()
            return response

    invocation = NOOAInvocationConfig.model_validate(
        {
            "agent_class": "responses_api_agents.nooa_agent.example_agent:WeatherAgent",
            "entrypoint": "answer",
            "tool_namespace": "weather",
            "allowed_tools": ["get_weather"],
            "arguments": {
                "question": {
                    "source": "responses_create_params.input",
                    "transform": "latest_user_text",
                }
            },
        }
    )
    runner = EmbeddedNOOARunner(
        invocation=invocation,
        server_client=RoutedClient(),
        model_server_name="policy",
        resources_server_name="weather",
        max_steps=1,
    )
    existing = ExistingHooks()
    set_hooks(existing)  # type: ignore[arg-type]
    try:
        result = await runner.run(
            NOOARunRequest(
                row=row("Paris"),
                rollout_id="composed-hooks",
                task_id="task",
                model_url_path="/ng-rollout/composed-hooks/v1/responses",
            )
        )
        assert result.return_value == "cold"
        assert calls[0] == "before:answer:composed-hooks"
        assert calls[-1].startswith("after:answer:")
        assert get_hooks() is existing
    finally:
        set_hooks(None)


@pytest.mark.asyncio
async def test_unexpected_failure_carries_partial_tool_evidence_and_cookies() -> None:
    runner, _ = make_runner()
    runner._agent_class = ExplodingAgent
    request = NOOARunRequest(
        row=row("Paris"),
        rollout_id="failed-run",
        task_id="task",
        model_url_path="/failed-run/v1/responses",
        resource_cookies={"session": "seeded"},
    )

    with pytest.raises(NOOARunFailure, match="agent exploded") as raised:
        await runner.run(request)

    partial = raised.value.result
    assert partial.return_value is None
    assert partial.tool_executions[0].name == "get_weather"
    assert [item.type for item in partial.trace.output] == ["function_call", "function_call_output"]
    assert partial.resource_cookies == {"session": "seeded"}


def test_runner_rejects_tool_namespace_collision_at_startup() -> None:
    invocation = NOOAInvocationConfig.model_validate(
        {
            "agent_class": f"{__name__}:ValidAgent",
            "entrypoint": "analyze",
            "init_kwargs": {"label": "configured"},
            "tool_namespace": "analyze",
            "arguments": {
                "text": {"source": "responses_create_params.input", "transform": "latest_user_text"},
                "customer_id": {"source": "customer_id"},
            },
        }
    )

    with pytest.raises(ValueError, match="collides with an existing agent attribute"):
        EmbeddedNOOARunner(
            invocation=invocation,
            server_client=MagicMock(),
            model_server_name="policy",
            resources_server_name="resources",
            max_steps=1,
        )
