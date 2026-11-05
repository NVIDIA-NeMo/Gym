# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MCP tool grants, called from the sandbox runner: a real streamable-HTTP MCP server that admits only the
episode's token."""

import json
import socket
import threading
import time
from collections.abc import Iterator

import pytest
import uvicorn
from fastapi import HTTPException
from mcp.server.fastmcp import FastMCP

from nemo_gym.server_runtime import MCPStreamableHTTPConnection, MCPToolAccess
from responses_api_agents.crewai_agent.tests.fakes import completion
from responses_api_agents.crewai_agent.tests.test_app import _agent, _body, _episode, _model, _records, linux_only


pytestmark = linux_only
_TOKEN = "Bearer episode-token"


def _require_token(app):
    async def guarded(scope, receive, send):
        if scope["type"] == "http" and dict(scope["headers"]).get(b"authorization") != _TOKEN.encode():
            await send({"type": "http.response.start", "status": 401, "headers": []})
            await send({"type": "http.response.body", "body": b""})
            return
        await app(scope, receive, send)

    return guarded


def _reject_strict_calls(app):
    """Answer each call of the ``strict`` tool with a JSON-RPC error, as a server does for invalid parameters.

    FastMCP turns an exception in a tool into a tool result marked as an error, so the error answer comes from here.
    """

    async def rejecting(scope, receive, send):
        if scope["type"] != "http":
            await app(scope, receive, send)
            return
        chunks, more = [], True
        while more:
            message = await receive()
            chunks.append(message.get("body", b""))
            more = message.get("more_body", False)
        body = b"".join(chunks)
        request = json.loads(body) if body else None
        if (
            isinstance(request, dict)
            and request.get("method") == "tools/call"
            and request["params"]["name"] == "strict"
        ):
            error = {"code": -32602, "message": "city must not be empty"}
            answer = json.dumps({"jsonrpc": "2.0", "id": request["id"], "error": error}).encode()
            await send(
                {"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"application/json")]}
            )
            await send({"type": "http.response.body", "body": answer})
            return
        replayed = False

        async def replay():
            nonlocal replayed
            if replayed:
                return await receive()
            replayed = True
            return {"type": "http.request", "body": body, "more_body": False}

        await app(scope, replay, send)

    return rejecting


@pytest.fixture(scope="module")
def mcp_url() -> Iterator[str]:
    server = FastMCP("weather", stateless_http=True, json_response=True)

    @server.tool()
    def get_weather(city: str) -> str:
        """Report the weather in a city."""
        return f"The weather in {city} is sunny."

    @server.tool()
    def broken(reason: str) -> str:
        """Always fail."""
        raise ValueError(reason)

    @server.tool()
    def strict(city: str) -> str:
        """Answered with a JSON-RPC error."""
        raise AssertionError("the server's error answer comes first")

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    uvicorn_server = uvicorn.Server(
        uvicorn.Config(
            _require_token(_reject_strict_calls(server.streamable_http_app())),
            host="127.0.0.1",
            port=port,
            log_level="warning",
        )
    )
    thread = threading.Thread(target=uvicorn_server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    while not uvicorn_server.started:
        assert time.monotonic() < deadline, "the MCP server did not start"
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}/mcp"
    uvicorn_server.should_exit = True
    thread.join(15)


def _grant(url: str, *, name: str = "weather", required: bool = True, token: str = _TOKEN) -> MCPToolAccess:
    return MCPToolAccess(
        name=name,
        required=required,
        connection=MCPStreamableHTTPConnection(url=url, headers={"Authorization": token}),
    )


async def test_a_granted_tool_is_discovered_called_and_its_result_reaches_the_model(
    mcp_url, lent_sandbox, runtime, monkeypatch
):
    _, access = lent_sandbox
    model = _model(
        monkeypatch,
        completion(None, ("get_weather", {"city": "Paris"}), ("broken", {"reason": "no"}), id="c1"),
        completion("The weather in Paris is sunny.", id="c2"),
    )
    try:
        response, close = await _episode(
            _agent(), access, _body(input="What is the weather in Paris?"), tool_accesses=[_grant(mcp_url)]
        )
    finally:
        model.close()

    assert response.status == "completed"
    first, second = model.requests
    # Grant tools keep their own names, which verifiers compare.
    assert sorted(tool["function"]["name"] for tool in first["json"]["tools"]) == [
        "broken",
        "get_weather",
        "shell",
        "strict",
    ]
    assert [item.name for item in response.output if item.type == "function_call"] == ["get_weather", "broken"]
    results = {
        message["tool_call_id"]: message["content"]
        for message in second["json"]["messages"]
        if message["role"] == "tool"
    }
    assert results["c1-call-0"] == "The weather in Paris is sunny."
    assert "no" in results["c1-call-1"]
    tools = {record.tool_call_id: record for record in _records(close, "tool_call")}
    assert (tools["c1-call-0"].status, tools["c1-call-1"].status) == ("completed", "failed")
    assert tools["c1-call-1"].error_type == "tool_error"


async def test_a_required_grant_that_does_not_connect_fails_the_episode_before_any_model_call(
    mcp_url, lent_sandbox, runtime, monkeypatch
):
    _, access = lent_sandbox
    model = _model(monkeypatch)
    try:
        response, close = await _episode(_agent(), access, tool_accesses=[_grant(mcp_url, token="Bearer wrong")])
    finally:
        model.close()

    # A 503 lets the Environment Server retry the episode; the crew never ran.
    assert isinstance(response, HTTPException) and response.status_code == 503
    assert model.requests == []
    [record] = _records(close, "sandbox")
    assert (record.outcome, record.error_type) == ("failed", "infrastructure_error")


async def test_an_optional_grant_that_does_not_connect_is_skipped_with_a_gap(
    mcp_url, lent_sandbox, runtime, monkeypatch
):
    _, access = lent_sandbox
    model = _model(monkeypatch, completion("No tools needed.", id="c1"))
    try:
        response, close = await _episode(
            _agent(), access, tool_accesses=[_grant(mcp_url, required=False, token="Bearer wrong")]
        )
    finally:
        model.close()

    assert response.status == "completed"
    assert [tool["function"]["name"] for tool in model.requests[0]["json"]["tools"]] == ["shell"]
    assert "mcp_server_skipped" in {gap.code for gap in close.agent_observations.gaps}


async def test_grants_whose_tool_names_collide_are_refused(mcp_url, lent_sandbox, runtime, monkeypatch):
    _, access = lent_sandbox
    model = _model(monkeypatch)
    try:
        response, _ = await _episode(
            _agent(), access, tool_accesses=[_grant(mcp_url, name="weather"), _grant(mcp_url, name="forecast")]
        )
    finally:
        model.close()

    assert isinstance(response, HTTPException) and response.status_code == 422
    assert "collide" in response.detail and "get_weather" in response.detail
    assert model.requests == []


@pytest.mark.parametrize(
    ("offered", "served"),
    [("get_weather", ["get_weather", "shell"]), ("get_forecast", None)],
)
async def test_request_tools_select_the_grant_tools_the_crew_offers(
    mcp_url, lent_sandbox, runtime, monkeypatch, offered, served
):
    _, access = lent_sandbox
    model = _model(monkeypatch, completion("Done.", id="c1"))
    declared = {"type": "function", "name": offered, "parameters": {"type": "object"}, "strict": False}
    try:
        response, _ = await _episode(
            _agent(), access, _body(input="hi", tools=[declared]), tool_accesses=[_grant(mcp_url)]
        )
    finally:
        model.close()

    if served is None:
        assert isinstance(response, HTTPException) and response.status_code == 422
        assert f"none serves {offered}" in response.detail
    else:
        assert response.status == "completed"
        assert sorted(tool["function"]["name"] for tool in model.requests[0]["json"]["tools"]) == served


async def test_a_call_the_server_answers_with_a_json_rpc_error_reaches_the_model_as_a_failed_call(
    mcp_url, lent_sandbox, runtime, monkeypatch
):
    _, access = lent_sandbox
    model = _model(
        monkeypatch,
        completion(None, ("strict", {"city": ""}), id="c1"),
        completion("I could not get the weather.", id="c2"),
    )
    try:
        response, close = await _episode(
            _agent(), access, _body(input="What is the weather?"), tool_accesses=[_grant(mcp_url)]
        )
    finally:
        model.close()

    # Not an infrastructure failure (503): the model sees the error and carries on.
    assert response.status == "completed"
    assert "city must not be empty" in model.requests[1]["json"]["messages"][-1]["content"]
    [tool] = _records(close, "tool_call")
    assert (tool.tool_call_id, tool.status, tool.error_type) == ("c1-call-0", "failed", "mcp_error")
