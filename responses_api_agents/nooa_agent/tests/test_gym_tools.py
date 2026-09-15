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
from nooa.agentdoc import doc

from responses_api_agents.nooa_agent.gym_tools import GymToolExecution, build_tool_namespace


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


def weather_tool() -> dict:
    return {
        "type": "function",
        "name": "get_weather",
        "description": "Return the weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "units": {"type": "string"},
            },
            "required": ["city"],
            "additionalProperties": False,
        },
    }


def make_tools(response: FakeResponse) -> tuple[Any, MagicMock, dict[str, str], list[GymToolExecution]]:
    client = MagicMock()
    client.post = AsyncMock(return_value=response)
    cookies = {"session": "old"}
    observations: list[GymToolExecution] = []
    tools = build_tool_namespace(
        namespace_name="weather",
        server_client=client,
        resources_server_name="weather_resources",
        tools=[weather_tool()],
        allowed_tools=frozenset({"get_weather"}),
        cookies=cookies,
        observations=observations,
    )
    return tools, client, cookies, observations


@pytest.mark.asyncio
async def test_generated_method_has_typed_signature_and_routes_with_cookies() -> None:
    tools, client, cookies, observations = make_tools(
        FakeResponse({"city": "Paris", "weather": "cold"}, cookie=("session", "new"))
    )

    output = await tools.get_weather("Paris")

    signature = inspect.signature(tools.get_weather)
    assert signature.parameters["city"].annotation is str
    assert signature.parameters["city"].default is inspect.Parameter.empty
    assert signature.parameters["units"].default is None
    assert tools.get_weather.__doc__ == "Return the weather for a city."
    assert output == {"city": "Paris", "weather": "cold"}
    assert client.post.await_args.kwargs == {
        "server_name": "weather_resources",
        "url_path": "/get_weather",
        "json": {"city": "Paris"},
        "cookies": {"session": "new"},
    }
    assert cookies == {"session": "new"}
    assert observations[0].status == "completed"
    assert observations[0].arguments == {"city": "Paris"}


@pytest.mark.asyncio
async def test_invalid_arguments_are_model_visible_without_http_call() -> None:
    tools, client, _, observations = make_tools(FakeResponse({}))

    output = await tools.get_weather(city=7)

    assert "Invalid arguments" in output["error"]
    client.post.assert_not_awaited()
    assert observations[0].status == "failed"
    assert observations[0].error_type == "invalid_arguments"


@pytest.mark.asyncio
async def test_resource_http_error_is_returned_and_observed() -> None:
    tools, _, _, observations = make_tools(FakeResponse({"detail": "unavailable"}, status=503))

    output = await tools.get_weather(city="Paris")

    assert output == {"detail": "unavailable"}
    assert observations[0].status == "failed"
    assert observations[0].error_type == "http_503"


@pytest.mark.parametrize(
    ("tools", "message"),
    [
        ([{"type": "web_search_preview"}], "function tools only"),
        ([weather_tool(), weather_tool()], "duplicate"),
        ([weather_tool() | {"name": "_private"}], "public Python identifier"),
        ([weather_tool() | {"name": "verify"}], "reserved by Gym"),
        ([weather_tool() | {"name": "class"}], "public Python identifier"),
        ([weather_tool() | {"parameters": {"type": "array"}}], "closed object JSON Schema"),
        (
            [weather_tool() | {"parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}],
            "additionalProperties=false",
        ),
    ],
)
def test_rejects_unsupported_or_colliding_tool_definitions(tools: list[dict], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        build_tool_namespace(
            namespace_name="resources",
            server_client=MagicMock(),
            resources_server_name="resources",
            tools=tools,
            allowed_tools=frozenset(str(tool.get("name")) for tool in tools),
            cookies={},
            observations=[],
        )


@pytest.mark.asyncio
async def test_transport_failure_is_recorded_before_propagation() -> None:
    tools, client, _, observations = make_tools(FakeResponse({}))
    client.post.side_effect = ConnectionError("connection lost")

    with pytest.raises(ConnectionError, match="connection lost"):
        await tools.get_weather(city="Paris")

    assert len(observations) == 1
    assert observations[0].status == "failed"
    assert observations[0].error_type == "ConnectionError"
    assert observations[0].output is None


@pytest.mark.asyncio
async def test_cancellation_is_recorded_before_propagation() -> None:
    tools, client, _, observations = make_tools(FakeResponse({}))
    client.post.side_effect = asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await tools.get_weather(city="Paris")

    assert len(observations) == 1
    assert observations[0].status == "cancelled"
    assert observations[0].error_type == "CancelledError"
    assert observations[0].output is None


def test_semantic_namespace_has_clean_agentdoc_surface() -> None:
    tools, _, _, _ = make_tools(FakeResponse({}))

    rendered = doc(tools)

    assert "class WeatherTools:" in rendered
    assert "async def get_weather(self, city: str, units: str = None)" in rendered
    assert "Return the weather for a city." in rendered
    assert "server_client" not in rendered
    assert "cookies" not in rendered
    assert "observations" not in rendered
    assert "dispatcher" not in rendered


def test_generated_method_accepts_positional_arguments() -> None:
    tools, _, _, _ = make_tools(FakeResponse({}))

    signature = inspect.signature(tools.get_weather)

    assert signature.parameters["city"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert signature.parameters["units"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD


def test_rejects_row_tool_not_in_trusted_allowlist() -> None:
    with pytest.raises(ValueError, match="not authorized"):
        build_tool_namespace(
            namespace_name="weather",
            server_client=MagicMock(),
            resources_server_name="resources",
            tools=[weather_tool()],
            allowed_tools=frozenset(),
            cookies={},
            observations=[],
        )


@pytest.mark.asyncio
async def test_concurrent_resource_calls_are_serialized() -> None:
    active = 0
    maximum_active = 0

    async def post(**kwargs: Any) -> FakeResponse:
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        await asyncio.sleep(0)
        active -= 1
        return FakeResponse({"city": kwargs["json"]["city"]})

    client = MagicMock()
    client.post = AsyncMock(side_effect=post)
    tools = build_tool_namespace(
        namespace_name="weather",
        server_client=client,
        resources_server_name="resources",
        tools=[weather_tool()],
        allowed_tools=frozenset({"get_weather"}),
        cookies={},
        observations=[],
    )

    await asyncio.gather(tools.get_weather("Paris"), tools.get_weather("Berlin"))

    assert maximum_active == 1
