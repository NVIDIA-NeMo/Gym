# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MCP tool grants: a real streamable-HTTP MCP server that admits only the episode's token."""

import socket
import threading
import time
from collections.abc import Iterator

import pytest
import uvicorn
from fastapi.testclient import TestClient
from mcp import ClientSession, McpError
from mcp.server.fastmcp import FastMCP
from mcp.types import INVALID_PARAMS, ErrorData

from nemo_gym.agents import check_agent_seed_rejected
from nemo_gym.server_runtime import MCPStreamableHTTPConnection, MCPToolAccess
from responses_api_agents.crewai_agent.tests.test_app import (
    _agent,
    _close,
    _completion,
    _Model,
    _respond,
    _seed,
    _seed_body,
)


_TOKEN = "Bearer episode-token"


@pytest.fixture(autouse=True)
def fresh_http_client(monkeypatch):
    # Each test client runs its own event loop; Gym's shared aiohttp client belongs to the loop that created it.
    monkeypatch.setattr("nemo_gym.client.http_client._GLOBAL_AIOHTTP_CLIENT", None)


def _require_token(app):
    async def guarded(scope, receive, send):
        if scope["type"] == "http" and dict(scope["headers"]).get(b"authorization") != _TOKEN.encode():
            await send({"type": "http.response.start", "status": 401, "headers": []})
            await send({"type": "http.response.body", "body": b""})
            return
        await app(scope, receive, send)

    return guarded


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

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    uvicorn_server = uvicorn.Server(
        uvicorn.Config(_require_token(server.streamable_http_app()), host="127.0.0.1", port=port, log_level="warning")
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


def test_a_granted_tool_is_discovered_called_and_its_result_reaches_the_model(mcp_url):
    model = _Model(
        _completion(None, ("get_weather", {"city": "Paris"}), ("broken", {"reason": "no"}), id="c1"),
        _completion("The weather in Paris is sunny.", id="c2"),
    )
    with TestClient(_agent(model).setup_webserver()) as client:
        assert _seed(client, tool_accesses=[_grant(mcp_url)]).status_code == 200
        response = _respond(client, {"input": "What is the weather in Paris?"})
        closed = _close(client).json()

    assert response.status_code == 200, response.text
    first, second = model.requests
    # Grant tools keep their own names, which verifiers compare.
    assert sorted(tool["function"]["name"] for tool in first["json"]["tools"]) == ["broken", "get_weather"]
    output = response.json()["output"]
    assert [item["name"] for item in output if item["type"] == "function_call"] == ["get_weather", "broken"]
    results = {
        message["tool_call_id"]: message["content"]
        for message in second["json"]["messages"]
        if message["role"] == "tool"
    }
    assert results["c1-call-0"] == "The weather in Paris is sunny."
    assert "no" in results["c1-call-1"]
    tools = {record["tool_call_id"]: record for record in closed["agent_observations"]["records"][1:]}
    assert (tools["c1-call-0"]["status"], tools["c1-call-1"]["status"]) == ("completed", "failed")
    assert tools["c1-call-1"]["error_type"] == "tool_error"


def test_a_call_the_server_answers_with_a_json_rpc_error_reaches_the_model_as_a_failed_call(mcp_url, monkeypatch):
    async def rejected(self, name, arguments=None, *args, **kwargs):
        # The session raises the server's JSON-RPC error answer, as for invalid parameters.
        raise McpError(ErrorData(code=INVALID_PARAMS, message="city must not be empty"))

    monkeypatch.setattr(ClientSession, "call_tool", rejected)
    model = _Model(
        _completion(None, ("get_weather", {"city": ""}), id="c1"),
        _completion("I could not get the weather.", id="c2"),
    )
    with TestClient(_agent(model).setup_webserver()) as client:
        assert _seed(client, tool_accesses=[_grant(mcp_url)]).status_code == 200
        response = _respond(client, {"input": "What is the weather?"})
        closed = _close(client).json()

    # Not an infrastructure failure (503): the model sees the error and carries on.
    assert response.status_code == 200, response.text
    assert "city must not be empty" in model.requests[1]["json"]["messages"][-1]["content"]
    (tool,) = closed["agent_observations"]["records"][1:]
    assert (tool["tool_call_id"], tool["status"], tool["error_type"]) == ("c1-call-0", "failed", "mcp_error")


def test_a_required_grant_that_does_not_connect_fails_the_seed_before_any_model_call(mcp_url):
    model = _Model()
    with TestClient(_agent(model).setup_webserver(), raise_server_exceptions=False) as client:
        seeded = _seed(client, tool_accesses=[_grant(mcp_url, token="Bearer wrong")])

    # A 500 lets the Environment Server retry the episode; the crew never ran.
    assert seeded.status_code == 500
    assert model.requests == []


def test_an_optional_grant_that_does_not_connect_is_skipped_with_a_gap(mcp_url):
    model = _Model(_completion("No tools needed.", id="c1"))
    with TestClient(_agent(model).setup_webserver()) as client:
        seeded = _seed(client, tool_accesses=[_grant(mcp_url, required=False, token="Bearer wrong")])
        _respond(client)
        closed = _close(client).json()

    assert seeded.status_code == 200
    assert "tools" not in model.requests[0]["json"]
    assert "mcp_server_skipped" in {gap["code"] for gap in closed["agent_observations"]["gaps"]}


def test_grants_whose_tool_names_collide_are_refused(mcp_url):
    seed = _seed_body(tool_accesses=[_grant(mcp_url, name="weather"), _grant(mcp_url, name="forecast")])

    detail = check_agent_seed_rejected(_agent(_Model()).setup_webserver(), seed)

    assert "collide" in detail and "get_weather" in detail


def test_request_tools_select_the_grant_tools_the_crew_offers(mcp_url):
    model = _Model(_completion("Done.", id="c1"))
    declared = {"type": "function", "name": "get_weather", "parameters": {"type": "object"}, "strict": False}
    with TestClient(_agent(model).setup_webserver()) as client:
        _seed(client, tool_accesses=[_grant(mcp_url)])
        undeclared = _respond(client, {"input": "hi", "tools": [{**declared, "name": "get_forecast"}]})
        response = _respond(client, {"input": "hi", "tools": [declared]})

    assert undeclared.status_code == 422 and "none serves get_forecast" in undeclared.json()["detail"]
    assert response.status_code == 200
    assert [tool["function"]["name"] for tool in model.requests[0]["json"]["tools"]] == ["get_weather"]
