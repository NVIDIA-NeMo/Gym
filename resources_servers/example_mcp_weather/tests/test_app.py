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
from unittest.mock import MagicMock

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from nemo_gym.base_resources_server import MCPSessionError
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from resources_servers.example_mcp_weather.app import (
    ExampleMCPWeatherResourcesServer,
    ExampleMCPWeatherResourcesServerConfig,
    ExampleMCPWeatherSeedSessionRequest,
    ExampleMCPWeatherVerifyRequest,
)


class FakeMCP:
    """Captures tools registered via the FastMCP-style ``add_tool`` API used by gym_tool."""

    def __init__(self):
        self.tools = {}

    def add_tool(self, fn, name=None, description=None):
        self.tools[name or fn.__name__] = fn


def _server() -> ExampleMCPWeatherResourcesServer:
    config = ExampleMCPWeatherResourcesServerConfig(
        host="127.0.0.1",
        port=12345,
        entrypoint="app.py",
        name="example_mcp_weather",
    )
    return ExampleMCPWeatherResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _request(session_id: str) -> Request:
    request = MagicMock(spec=Request)
    request.session = {SESSION_ID_KEY: session_id}
    return request


def _verify_request(expected_city: str, final_text: str) -> ExampleMCPWeatherVerifyRequest:
    return ExampleMCPWeatherVerifyRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
            input=[NeMoGymEasyInputMessage(role="user", content="use the MCP weather tool")]
        ),
        response=NeMoGymResponse(
            id="resp_1",
            created_at=0,
            model="test",
            object="response",
            output=[
                NeMoGymResponseOutputMessage(
                    id="msg_1",
                    content=[NeMoGymResponseOutputText(text=final_text, annotations=[])],
                    role="assistant",
                    status="completed",
                    type="message",
                )
            ],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        ),
        verifier_metadata={"expected_city": expected_city},
    )


@pytest.mark.asyncio
async def test_verify_rewards_tool_call_from_same_session() -> None:
    server = _server()
    seed = await server.seed_session(
        _request("session-1"), ExampleMCPWeatherSeedSessionRequest(verifier_metadata={"expected_city": "Paris"})
    )
    token = seed.mcp.headers["X-NeMo-Gym-Session-Token"]

    fake_mcp = FakeMCP()
    server.register_mcp_tools(fake_mcp)

    from nemo_gym.base_resources_server import _MCP_SESSION_TOKEN

    # The auto-registered MCP wrapper takes only `city` (session_id is injected from the token) and is
    # async (sync tools are offloaded to a threadpool), so it must be awaited.
    context_token = _MCP_SESSION_TOKEN.set(token)
    try:
        assert await fake_mcp.tools["get_weather"](city="Paris") == "The weather in Paris is sunny and 72 F."
    finally:
        _MCP_SESSION_TOKEN.reset(context_token)

    result = await server.verify(
        _request("session-1"),
        _verify_request("Paris", "The weather in Paris is sunny and 72 F."),
    )

    assert result.reward == 1.0
    assert result.tool_call_seen is True
    assert result.final_response_mentions_weather is True


@pytest.mark.asyncio
async def test_verify_accepts_differently_cased_city() -> None:
    # A correct tool call that used different casing than the seed city must still be rewarded.
    server = _server()
    await server.seed_session(
        _request("session-1"), ExampleMCPWeatherSeedSessionRequest(verifier_metadata={"expected_city": "Paris"})
    )
    server.session_id_to_state["session-1"]["weather_calls"].append(
        {"city": "PARIS", "weather": "The weather in PARIS is sunny and 72 F."}
    )

    result = await server.verify(
        _request("session-1"),
        _verify_request("Paris", "The weather in PARIS is sunny and 72 F."),
    )

    assert result.reward == 1.0
    assert result.tool_call_seen is True
    assert result.final_response_mentions_weather is True


@pytest.mark.asyncio
async def test_verify_rejects_tool_call_from_different_session() -> None:
    server = _server()
    await server.seed_session(
        _request("session-1"), ExampleMCPWeatherSeedSessionRequest(verifier_metadata={"expected_city": "Paris"})
    )
    server.session_id_to_state["session-2"] = {
        "expected_city": "Paris",
        "weather_calls": [{"city": "Paris", "weather": "The weather in Paris is sunny and 72 F."}],
    }

    result = await server.verify(
        _request("session-1"),
        _verify_request("Paris", "The weather in Paris is sunny and 72 F."),
    )

    assert result.reward == 0.0
    assert result.tool_call_seen is False


@pytest.mark.asyncio
async def test_mcp_tool_requires_valid_session_token() -> None:
    server = _server()
    fake_mcp = FakeMCP()
    server.register_mcp_tools(fake_mcp)

    from nemo_gym.base_resources_server import _MCP_SESSION_TOKEN

    context_token = _MCP_SESSION_TOKEN.set("invalid-token")
    try:
        with pytest.raises(MCPSessionError):
            await fake_mcp.tools["get_weather"](city="Paris")
    finally:
        _MCP_SESSION_TOKEN.reset(context_token)


def test_streamable_http_mcp_endpoint_records_same_session() -> None:
    pytest.importorskip("mcp")
    server = _server()
    app = server.setup_webserver()
    rpc_headers = {
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
    }

    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        seed_response = client.post("/seed_session", json={"verifier_metadata": {"expected_city": "Paris"}})
        assert seed_response.status_code == 200
        token = seed_response.json()["mcp"]["headers"]["X-NeMo-Gym-Session-Token"]

        tool_response = client.post(
            "/mcp",
            headers={**rpc_headers, "X-NeMo-Gym-Session-Token": token},
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "get_weather", "arguments": {"city": "Paris"}},
            },
            follow_redirects=False,
        )

        assert tool_response.status_code == 200
        assert tool_response.json()["result"]["structuredContent"]["result"] == (
            "The weather in Paris is sunny and 72 F."
        )

        verify_response = client.post(
            "/verify",
            json=_verify_request("Paris", "The weather in Paris is sunny and 72 F.").model_dump(mode="json"),
        )
        assert verify_response.status_code == 200
        assert verify_response.json()["reward"] == 1.0
        assert verify_response.json()["tool_call_seen"] is True
