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

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_observability import AgentInvocation, ToolCallObservation
from responses_api_agents.nooa_agent.config import NOOAInvocationConfig
from responses_api_agents.nooa_agent.runner import EmbeddedNOOARunner, NOOARunRequest


class ValidAgent(Agent):
    def __init__(self, *, llm: Any, label: str) -> None:
        super().__init__(llm=llm)
        self.label = label

    async def analyze(self, text: str, customer_id: str) -> str: ...


class FakeAgent:
    instances = 0
    get_weather: Any

    def __init__(self, *, llm: Any, label: str) -> None:
        FakeAgent.instances += 1
        self.llm = llm
        self.label = label
        self.event_manager = FakeEventManager()

    async def analyze(self, text: str, customer_id: str) -> str:
        weather = await self.get_weather(city=customer_id)
        return f"{text}: {weather['weather']}"


adapter_requests: list[NeMoGymResponseCreateParamsNonStreaming] = []


class FakeEventManager:
    def on(self, event_type: str, handler: Any) -> Any:
        return lambda: None


async def invoke(agent: Any, request: NeMoGymResponseCreateParamsNonStreaming) -> object:
    adapter_requests.append(request)
    assert isinstance(request.input, str)
    text, customer_id = request.input.split("|", maxsplit=1)
    return await agent.analyze(text, customer_id)


class FakeContent:
    async def read(self) -> bytes:
        return json.dumps({"weather": "cold"}).encode()


class FakeResponse:
    status = 200
    content = FakeContent()
    cookies = SimpleCookie()


def make_runner(*, execution_mode: str = "embedded") -> tuple[EmbeddedNOOARunner, MagicMock]:
    invocation = NOOAInvocationConfig.model_validate(
        {
            "agent_class": f"{__name__}:ValidAgent",
            "invocation_adapter": f"{__name__}:invoke",
            "execution_mode": execution_mode,
            "init_kwargs": {"label": "configured"},
        }
    )
    client = MagicMock()
    client.post = AsyncMock(return_value=FakeResponse())
    runner = EmbeddedNOOARunner(
        invocation=invocation,
        server_client=client,
        model_server_name="policy_model",
        resources_server_name="weather_resources",
        max_policy_calls=3,
    )
    runner._agent_class = FakeAgent
    return runner, client


def responses_create_params(customer_id: str) -> NeMoGymResponseCreateParamsNonStreaming:
    return NeMoGymResponseCreateParamsNonStreaming.model_validate(
        {
            "input": f"Check delivery|{customer_id}",
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
        }
    )


@pytest.mark.asyncio
async def test_embedded_runner_invokes_adapter_and_attaches_resource_methods() -> None:
    runner, client = make_runner()
    request = responses_create_params("Paris")
    adapter_requests.clear()

    result = await runner.run(
        NOOARunRequest(
            responses_create_params=request,
            model_url_path="/ng-rollout/rollout-1/v1/responses",
            resource_cookies={"session": "one"},
        )
    )

    assert [item.type for item in result.episode.response.output] == ["function_call", "function_call_output"]
    assert result.return_value == "Check delivery: cold"
    assert result.episode.observations.source == "nooa"
    assert result.episode.observations.gaps == []
    invocation = next(record for record in result.episode.observations.records if isinstance(record, AgentInvocation))
    tool = next(record for record in result.episode.observations.records if isinstance(record, ToolCallObservation))
    assert invocation.conversation[0].content == "Check delivery|Paris"
    assert tool.tool_name == "get_weather"
    assert adapter_requests == [request]
    assert client.post.await_args.kwargs["json"] == {"city": "Paris"}


@pytest.mark.asyncio
async def test_constructs_a_fresh_agent_for_every_rollout() -> None:
    runner, _ = make_runner()
    FakeAgent.instances = 0

    first = await runner.run(
        NOOARunRequest(
            responses_create_params=responses_create_params("Paris"),
            model_url_path="/one/v1/responses",
        )
    )
    second = await runner.run(
        NOOARunRequest(
            responses_create_params=responses_create_params("Berlin"),
            model_url_path="/two/v1/responses",
        )
    )

    assert FakeAgent.instances == 2
    assert first.episode is not second.episode
    assert first.resource_cookies is not second.resource_cookies


def test_sandboxed_execution_mode_fails_during_runner_construction() -> None:
    with pytest.raises(NotImplementedError, match="sandboxed execution is not implemented"):
        make_runner(execution_mode="sandboxed")
