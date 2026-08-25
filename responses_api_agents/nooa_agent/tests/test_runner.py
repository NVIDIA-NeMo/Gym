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
from pydantic import BaseModel, ConfigDict

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.nooa_agent.config import NOOAInvocationConfig
from responses_api_agents.nooa_agent.runner import EmbeddedNOOARunner, NOOARunRequest


class ValidAgent(Agent):
    def __init__(self, *, llm: Any, label: str) -> None:
        super().__init__(llm=llm)
        self.label = label

    async def analyze(self, text: str, customer_id: str) -> str: ...


class FakeAgent:
    instances = 0

    def __init__(self, *, llm: Any, label: str) -> None:
        type(self).instances += 1
        self.llm = llm
        self.label = label
        self.gym_tools: Any = None

    async def analyze(self, text: str, customer_id: str) -> str:
        weather = await self.gym_tools.get_weather(city=customer_id)
        return f"{text}: {weather['weather']}"


class Row(BaseModel):
    model_config = ConfigDict(extra="allow")

    responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    customer_id: str


class FakeContent:
    async def read(self) -> bytes:
        return json.dumps({"weather": "cold"}).encode()


class FakeResponse:
    status = 200
    content = FakeContent()
    cookies = SimpleCookie()


def make_runner() -> tuple[EmbeddedNOOARunner, MagicMock]:
    invocation = NOOAInvocationConfig.model_validate(
        {
            "agent_class": f"{__name__}:ValidAgent",
            "entrypoint": "analyze",
            "init_kwargs": {"label": "configured"},
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
async def test_embedded_runner_maps_full_row_and_attaches_gym_tools() -> None:
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
