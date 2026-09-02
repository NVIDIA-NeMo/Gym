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
import inspect
import json
from http.cookies import SimpleCookie
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from jsonschema import Draft202012Validator
from nooa import Agent

from responses_api_agents.nooa_agent.resource_tools import (
    ResourceToolDispatcher,
    ResourceToolExecution,
    create_agent_class_with_resource_methods,
    validate_agent_resource_method_bindings,
)


class FakeContent:
    def __init__(self, value: object) -> None:
        self._payload = json.dumps(value).encode()

    async def read(self) -> bytes:
        return self._payload


class FakeResponse:
    def __init__(self, value: object, status: int = 200, cookie: tuple[str, str] | None = None) -> None:
        self.content = FakeContent(value)
        self.status = status
        self.cookies = SimpleCookie()
        if cookie:
            self.cookies[cookie[0]] = cookie[1]


class FakeAgent:
    pass


class AgentWithCollision:
    def get_weather(self, city: str) -> str:
        return city


class AgentWithInstanceCollision:
    def __init__(self) -> None:
        self.get_weather = "not a resource method"


def weather_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "name": "get_weather",
        "description": "Return the weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "units": {"type": "string", "default": "celsius"},
            },
            "required": ["city"],
            "additionalProperties": False,
        },
    }


def make_agent(
    response: FakeResponse,
) -> tuple[FakeAgent, MagicMock, dict[str, str], list[ResourceToolExecution]]:
    client = MagicMock()
    client.post = AsyncMock(return_value=response)
    cookies = {"session": "old"}
    executions: list[ResourceToolExecution] = []
    dispatcher = ResourceToolDispatcher(
        server_client=client,
        resources_server_name="weather_resources",
        cookies=cookies,
        executions=executions,
    )
    agent_class = create_agent_class_with_resource_methods(
        FakeAgent,
        dispatcher=dispatcher,
        tools=[weather_tool()],
    )
    agent = agent_class()
    return agent, client, cookies, executions


@pytest.mark.asyncio
async def test_method_is_attached_directly_with_typed_signature_defaults_and_cookies() -> None:
    agent, client, cookies, executions = make_agent(
        FakeResponse({"city": "Paris", "weather": "cold"}, cookie=("session", "new"))
    )

    output = await agent.get_weather("Paris")  # type: ignore[attr-defined]

    assert "get_weather" in vars(type(agent))
    assert "gym_tools" not in vars(agent)
    signature = inspect.signature(agent.get_weather)  # type: ignore[attr-defined]
    assert signature.parameters["city"].annotation is str
    assert signature.parameters["city"].default is inspect.Parameter.empty
    assert signature.parameters["units"].default == "celsius"
    assert agent.get_weather.__doc__ == "Return the weather for a city."  # type: ignore[attr-defined]
    assert output == {"city": "Paris", "weather": "cold"}
    assert client.post.await_args.kwargs == {
        "server_name": "weather_resources",
        "url_path": "/get_weather",
        "json": {"city": "Paris", "units": "celsius"},
        "cookies": {"session": "new"},
    }
    assert cookies == {"session": "new"}
    assert executions[0].status == "completed"
    assert executions[0].arguments == {"city": "Paris", "units": "celsius"}


@pytest.mark.asyncio
async def test_invalid_arguments_are_model_visible_without_http_call() -> None:
    agent, client, _, executions = make_agent(FakeResponse({}))

    output = await agent.get_weather(city=7)  # type: ignore[attr-defined]

    assert "Invalid arguments" in output["error"]
    client.post.assert_not_awaited()
    assert executions[0].status == "failed"
    assert executions[0].error_type == "invalid_arguments"


@pytest.mark.asyncio
async def test_invalid_python_call_raises_type_error_without_http_call() -> None:
    agent, client, _, executions = make_agent(FakeResponse({}))

    with pytest.raises(TypeError, match="Invalid arguments for get_weather"):
        await agent.get_weather()  # type: ignore[attr-defined]

    client.post.assert_not_awaited()
    assert executions == []


@pytest.mark.asyncio
async def test_resource_http_error_is_returned_and_observed() -> None:
    agent, _, _, executions = make_agent(FakeResponse({"detail": "unavailable"}, status=503))

    output = await agent.get_weather(city="Paris")  # type: ignore[attr-defined]

    assert output == {"detail": "unavailable"}
    assert executions[0].status == "failed"
    assert executions[0].error_type == "http_503"


@pytest.mark.asyncio
async def test_resource_calls_are_serialized_per_rollout() -> None:
    active_calls = 0
    max_active_calls = 0

    async def post(**_: Any) -> FakeResponse:
        nonlocal active_calls, max_active_calls
        active_calls += 1
        max_active_calls = max(max_active_calls, active_calls)
        await asyncio.sleep(0.01)
        active_calls -= 1
        return FakeResponse({"ok": True})

    client = MagicMock()
    client.post = AsyncMock(side_effect=post)
    executions: list[ResourceToolExecution] = []
    dispatcher = ResourceToolDispatcher(
        server_client=client,
        resources_server_name="resources",
        cookies={},
        executions=executions,
    )
    validator = Draft202012Validator({"type": "object"})

    await asyncio.gather(
        dispatcher.call(name="first", arguments={}, validator=validator),
        dispatcher.call(name="second", arguments={}, validator=validator),
    )

    assert max_active_calls == 1
    assert [execution.name for execution in executions] == ["first", "second"]


def test_rejects_method_name_that_collides_with_agent() -> None:
    dispatcher = ResourceToolDispatcher(
        server_client=MagicMock(),
        resources_server_name="resources",
        cookies={},
        executions=[],
    )

    with pytest.raises(ValueError, match="conflicting agent method"):
        create_agent_class_with_resource_methods(
            AgentWithCollision,
            dispatcher=dispatcher,
            tools=[weather_tool()],
        )


def test_rejects_instance_field_that_hides_resource_method() -> None:
    dispatcher = ResourceToolDispatcher(
        server_client=MagicMock(),
        resources_server_name="resources",
        cookies={},
        executions=[],
    )
    agent_class = create_agent_class_with_resource_methods(
        AgentWithInstanceCollision,
        dispatcher=dispatcher,
        tools=[weather_tool()],
    )
    agent = agent_class()

    with pytest.raises(ValueError, match="instance field conflicts.*get_weather"):
        validate_agent_resource_method_bindings(agent)


def test_method_can_be_attached_to_real_nooa_agent_instance() -> None:
    dispatcher = ResourceToolDispatcher(
        server_client=MagicMock(),
        resources_server_name="resources",
        cookies={},
        executions=[],
    )
    agent_class = create_agent_class_with_resource_methods(
        Agent,
        dispatcher=dispatcher,
        tools=[weather_tool()],
    )
    agent = agent_class(llm=MagicMock())
    validate_agent_resource_method_bindings(agent)

    assert "get_weather" in vars(type(agent))
    assert inspect.signature(agent.get_weather).parameters["city"].annotation is str  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("tools", "message"),
    [
        ([{"type": "web_search_preview"}], "function tools only"),
        ([weather_tool(), weather_tool()], "duplicate"),
        ([weather_tool() | {"name": "_private"}], "public Python identifier"),
        ([weather_tool() | {"parameters": {"type": "array"}}], "object JSON Schema"),
    ],
)
def test_rejects_unsupported_or_colliding_tool_definitions(tools: list[dict], message: str) -> None:
    dispatcher = ResourceToolDispatcher(
        server_client=MagicMock(),
        resources_server_name="resources",
        cookies={},
        executions=[],
    )

    with pytest.raises(ValueError, match=message):
        create_agent_class_with_resource_methods(FakeAgent, dispatcher=dispatcher, tools=tools)
