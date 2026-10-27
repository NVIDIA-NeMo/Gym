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
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest
from jsonschema import Draft202012Validator
from nooa import Agent

from nemo_gym.tool_access import DirectHTTPToolAccess
from responses_api_agents.nooa_agent import resource_tools
from responses_api_agents.nooa_agent.observability import GymTraceHooks
from responses_api_agents.nooa_agent.resource_tools import (
    ResourceToolDispatcher,
    create_agent_class_with_resource_methods,
    validate_agent_resource_method_bindings,
)


@pytest.fixture(autouse=True)
def outbound(monkeypatch):
    mock = AsyncMock()
    monkeypatch.setattr(resource_tools, "request", mock)
    return mock


class FakeContent:
    def __init__(self, value: object) -> None:
        self._payload = value if isinstance(value, bytes) else json.dumps(value).encode()

    async def read(self) -> bytes:
        return self._payload


class FakeResponse:
    def __init__(self, value: object, status: int = 200, cookie: tuple[str, str] | None = None) -> None:
        self.content = FakeContent(value)
        self.status = status
        self.cookies = SimpleCookie()
        if cookie:
            self.cookies[cookie[0]] = cookie[1]

    @property
    def ok(self) -> bool:
        return self.status < 400

    def raise_for_status(self) -> None:
        if not self.ok:
            raise aiohttp.ClientResponseError(
                SimpleNamespace(url="http://resources", real_url="http://resources", method="POST", headers={}),
                (),
                status=self.status,
                message="unavailable",
            )


def grant() -> DirectHTTPToolAccess:
    return DirectHTTPToolAccess(
        name="resources", required=True, base_url="http://resources/", headers={"X-Grant": "yes"}
    )


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


@pytest.mark.parametrize(
    "schema_type,annotation,value",
    [
        ("integer", int, 1),
        ("number", float, 1.5),
        ("boolean", bool, True),
        ("array", list, [1]),
        ("object", dict, {"x": 1}),
        (None, Any, "untyped"),
    ],
)
@pytest.mark.asyncio
async def test_resource_signature_and_dispatch_for_json_types(schema_type: str, annotation: Any, value: Any) -> None:
    client = MagicMock()
    client.post = resource_tools.request
    client.post.return_value = FakeResponse({"ok": True})
    dispatcher = ResourceToolDispatcher(tool_access=grant(), cookies={})
    tool = weather_tool()
    tool["name"] = "typed"
    tool["parameters"]["properties"] = {"value": {"type": schema_type} if schema_type else {}}
    tool["parameters"]["required"] = ["value"]
    agent = create_agent_class_with_resource_methods(FakeAgent, dispatcher=dispatcher, tools=[tool])()
    assert inspect.signature(agent.typed).parameters["value"].annotation is annotation
    assert await agent.typed(value) == {"ok": True}
    assert client.post.await_args.kwargs["json"] == {"value": value}


@pytest.mark.asyncio
async def test_invalid_resource_arguments_are_observed_without_http() -> None:
    client = MagicMock()
    client.post = resource_tools.request
    trace = GymTraceHooks()
    dispatcher = ResourceToolDispatcher(
        tool_access=grant(),
        cookies={},
        trace_hooks=trace,
    )
    agent = create_agent_class_with_resource_methods(FakeAgent, dispatcher=dispatcher, tools=[weather_tool()])()
    result = await agent.get_weather(city=123)
    assert "Invalid arguments" in result["error"]
    client.post.assert_not_awaited()
    event = trace._events[0].observation
    assert event.status == "failed"
    assert event.error_type == "invalid_arguments"
    assert event.output == result
    assert dispatcher.fatal_error is None


def test_invalid_tool_json_schema_rejected_before_execution() -> None:
    dispatcher = ResourceToolDispatcher(
        tool_access=grant(),
        cookies={},
    )
    tool = weather_tool()
    tool["parameters"]["properties"]["city"] = {"type": "not-a-json-type"}
    with pytest.raises(ValueError, match="invalid JSON Schema"):
        create_agent_class_with_resource_methods(FakeAgent, dispatcher=dispatcher, tools=[tool])


def make_agent(
    response: FakeResponse,
) -> tuple[FakeAgent, MagicMock, dict[str, str]]:
    client = MagicMock()
    client.post = resource_tools.request
    client.post.return_value = response
    cookies = {"session": "old"}
    dispatcher = ResourceToolDispatcher(
        tool_access=grant(),
        cookies=cookies,
    )
    agent_class = create_agent_class_with_resource_methods(
        FakeAgent,
        dispatcher=dispatcher,
        tools=[weather_tool()],
    )
    agent = agent_class()
    return agent, client, cookies


@pytest.mark.asyncio
async def test_method_is_attached_directly_with_typed_signature_defaults_and_cookies() -> None:
    agent, client, cookies = make_agent(FakeResponse({"city": "Paris", "weather": "cold"}, cookie=("session", "new")))

    output = await agent.get_weather("Paris")  # type: ignore[attr-defined]

    assert "get_weather" in vars(type(agent))
    signature = inspect.signature(agent.get_weather)  # type: ignore[attr-defined]
    assert signature.parameters["city"].annotation is str
    assert signature.parameters["city"].default is inspect.Parameter.empty
    assert signature.parameters["units"].default == "celsius"
    assert agent.get_weather.__doc__ == "Return the weather for a city."  # type: ignore[attr-defined]
    assert output == {"city": "Paris", "weather": "cold"}
    assert client.post.await_args.kwargs == {
        "headers": {"X-Grant": "yes"},
        "json": {"city": "Paris", "units": "celsius"},
        "cookies": {"session": "old"},
    }
    assert client.post.await_args.args == ("POST", "http://resources/get_weather")
    assert cookies == {"session": "new"}


@pytest.mark.asyncio
async def test_invalid_arguments_are_model_visible_without_http_call() -> None:
    agent, client, _ = make_agent(FakeResponse({}))

    output = await agent.get_weather(city=7)  # type: ignore[attr-defined]

    assert "Invalid arguments" in output["error"]
    client.post.assert_not_awaited()


@pytest.mark.asyncio
async def test_invalid_python_call_raises_type_error_without_http_call() -> None:
    agent, client, _ = make_agent(FakeResponse({}))

    with pytest.raises(TypeError, match="Invalid arguments for get_weather"):
        await agent.get_weather()  # type: ignore[attr-defined]

    client.post.assert_not_awaited()


@pytest.mark.parametrize(
    ("status", "feedback"),
    [
        (400, {"detail": "invalid request"}),
        (401, "Authentication required"),
        (403, {"detail": "permission denied"}),
        (404, "City not found"),
        (422, {"detail": [{"loc": ["city"], "msg": "unknown city"}]}),
        (429, "Rate limit reached; retry later"),
    ],
)
@pytest.mark.asyncio
async def test_resource_client_errors_return_observed_feedback_and_update_cookies(
    outbound: AsyncMock, status: int, feedback: dict | str
) -> None:
    body = feedback.encode() if isinstance(feedback, str) else feedback
    outbound.side_effect = [
        FakeResponse(body, status=status, cookie=("session", "updated")),
        FakeResponse({"weather": "sunny"}),
    ]
    trace = GymTraceHooks()
    cookies = {"session": "original"}
    dispatcher = ResourceToolDispatcher(tool_access=grant(), cookies=cookies, trace_hooks=trace)
    agent = create_agent_class_with_resource_methods(FakeAgent, dispatcher=dispatcher, tools=[weather_tool()])()

    assert await agent.get_weather(city="Paris") == feedback
    assert dispatcher.fatal_error is None
    assert cookies == {"session": "updated"}
    event = trace._events[0].observation
    assert event.status == "failed"
    assert event.error_type == f"http_{status}"
    assert event.output == feedback

    assert await agent.get_weather(city="London") == {"weather": "sunny"}
    assert dispatcher.fatal_error is None
    assert outbound.await_args_list[0].kwargs["cookies"] == {"session": "original"}
    assert outbound.await_args_list[1].kwargs["cookies"] == {"session": "updated"}
    assert trace._events[1].observation.status == "completed"
    assert trace._events[1].observation.error_type is None


@pytest.mark.parametrize("status", [500, 502, 503, 599])
@pytest.mark.asyncio
async def test_resource_server_errors_raise_and_remain_fatal_after_success(outbound: AsyncMock, status: int) -> None:
    outbound.side_effect = [
        FakeResponse({"detail": "unavailable"}, status=status, cookie=("session", "updated")),
        FakeResponse({"weather": "sunny"}),
    ]
    trace = GymTraceHooks()
    cookies = {"session": "original"}
    dispatcher = ResourceToolDispatcher(tool_access=grant(), cookies=cookies, trace_hooks=trace)
    agent = create_agent_class_with_resource_methods(FakeAgent, dispatcher=dispatcher, tools=[weather_tool()])()

    with pytest.raises(aiohttp.ClientResponseError) as caught:
        await agent.get_weather(city="Paris")
    assert caught.value.status == status
    assert json.loads(caught.value.response_content) == {"detail": "unavailable"}
    assert dispatcher.fatal_error is caught.value
    assert cookies == {"session": "updated"}
    event = trace._events[0].observation
    assert event.status == "failed"
    assert event.error_type == "ClientResponseError"

    assert await agent.get_weather(city="London") == {"weather": "sunny"}
    assert dispatcher.fatal_error is caught.value
    assert outbound.await_args_list[1].kwargs["cookies"] == {"session": "updated"}
    assert trace._events[1].observation.status == "completed"


@pytest.mark.asyncio
async def test_resource_calls_are_serialized_per_rollout() -> None:
    active_calls = 0
    max_active_calls = 0
    call_order: list[str] = []

    async def post(method: str, url: str, **_: Any) -> FakeResponse:
        nonlocal active_calls, max_active_calls
        active_calls += 1
        max_active_calls = max(max_active_calls, active_calls)
        call_order.append(url)
        await asyncio.sleep(0.01)
        active_calls -= 1
        return FakeResponse({"ok": True})

    client = MagicMock()
    client.post = resource_tools.request
    client.post.side_effect = post
    dispatcher = ResourceToolDispatcher(
        tool_access=grant(),
        cookies={},
    )
    validator = Draft202012Validator({"type": "object"})

    await asyncio.gather(
        dispatcher.call(name="first", arguments={}, validator=validator),
        dispatcher.call(name="second", arguments={}, validator=validator),
    )

    assert max_active_calls == 1
    assert call_order == ["http://resources/first", "http://resources/second"]


def test_rejects_method_name_that_collides_with_agent() -> None:
    dispatcher = ResourceToolDispatcher(
        tool_access=grant(),
        cookies={},
    )

    with pytest.raises(ValueError, match="conflicting agent method"):
        create_agent_class_with_resource_methods(
            AgentWithCollision,
            dispatcher=dispatcher,
            tools=[weather_tool()],
        )


def test_rejects_instance_field_that_hides_resource_method() -> None:
    dispatcher = ResourceToolDispatcher(
        tool_access=grant(),
        cookies={},
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
        tool_access=grant(),
        cookies={},
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
        tool_access=grant(),
        cookies={},
    )

    with pytest.raises(ValueError, match=message):
        create_agent_class_with_resource_methods(FakeAgent, dispatcher=dispatcher, tools=tools)


@pytest.mark.parametrize("name", ["seed", "verify", "close", "seed_session", "close_session"])
def test_lifecycle_routes_cannot_be_exposed(name: str) -> None:
    dispatcher = ResourceToolDispatcher(tool_access=grant(), cookies={})
    with pytest.raises(ValueError, match="lifecycle route"):
        create_agent_class_with_resource_methods(
            FakeAgent, dispatcher=dispatcher, tools=[weather_tool() | {"name": name}]
        )


@pytest.mark.parametrize("stage", ["connection", "body"])
@pytest.mark.asyncio
async def test_transport_failure_remains_sticky_after_a_later_success(outbound: AsyncMock, stage: str) -> None:
    if stage == "connection":
        failure = aiohttp.ClientConnectionError("resources unavailable")
        outbound.side_effect = [failure, FakeResponse({"ok": True})]
    else:
        failure = aiohttp.ClientPayloadError("incomplete response body")
        response = FakeResponse({"detail": "invalid request"}, status=422)
        response.content.read = AsyncMock(side_effect=failure)
        outbound.side_effect = [response, FakeResponse({"ok": True})]
    trace = GymTraceHooks()
    dispatcher = ResourceToolDispatcher(tool_access=grant(), cookies={}, trace_hooks=trace)
    validator = Draft202012Validator({"type": "object"})

    with pytest.raises(type(failure)) as caught:
        await dispatcher.call(name="tool", arguments={}, validator=validator)
    assert caught.value is failure
    assert dispatcher.fatal_error is failure
    event = trace._events[0].observation
    assert event.status == "failed"
    assert event.error_type == type(failure).__name__
    assert str(failure) in event.output["error"]
    assert await dispatcher.call(name="tool", arguments={}, validator=validator) == {"ok": True}
    assert dispatcher.fatal_error is failure
    assert trace._events[1].observation.status == "completed"


@pytest.mark.asyncio
async def test_unserializable_resource_arguments_return_feedback_without_transport_or_fatal_error(
    outbound: AsyncMock,
) -> None:
    dispatcher = ResourceToolDispatcher(tool_access=grant(), cookies={})
    output = await dispatcher.call(
        name="tool", arguments={"value": {1, 2}}, validator=Draft202012Validator({"type": "object"})
    )
    assert "Could not serialize arguments for tool" in output["error"]
    outbound.assert_not_awaited()
    assert dispatcher.fatal_error is None
