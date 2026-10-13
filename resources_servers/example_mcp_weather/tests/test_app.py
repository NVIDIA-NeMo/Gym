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
from unittest.mock import MagicMock

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from resources_servers.example_mcp_weather.app import (
    ExampleMCPWeatherGetWeatherRequest,
    ExampleMCPWeatherResourcesServer,
    ExampleMCPWeatherResourcesServerConfig,
    ExampleMCPWeatherSeedSessionRequest,
    ExampleMCPWeatherVerifyRequest,
)


TOKEN_HEADER = "X-NeMo-Gym-Session-Token"
RPC_HEADERS = {
    "Accept": "application/json, text/event-stream",
    "Content-Type": "application/json",
}


def _server() -> ExampleMCPWeatherResourcesServer:
    config = ExampleMCPWeatherResourcesServerConfig(
        host="127.0.0.1",
        port=12345,
        entrypoint="app.py",
        name="example_mcp_weather",
        expose_tools_over_mcp=True,
    )
    return ExampleMCPWeatherResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _request(session_id: str) -> Request:
    request = MagicMock(spec=Request)
    request.session = {SESSION_ID_KEY: session_id}
    return request


def _verify_request(expected_city: str, final_text: str) -> ExampleMCPWeatherVerifyRequest:
    return ExampleMCPWeatherVerifyRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
            input=[NeMoGymEasyInputMessage(role="user", content="use the weather tool")]
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
    await server.seed_session(
        _request("session-1"), ExampleMCPWeatherSeedSessionRequest(verifier_metadata={"expected_city": "Paris"})
    )

    tool_response = await server.get_weather(_request("session-1"), ExampleMCPWeatherGetWeatherRequest(city="Paris"))
    assert tool_response.weather == "The weather in Paris is sunny and 72 F."

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


def test_http_tool_route_records_same_session() -> None:
    # The plain HTTP door: the session cookie set by /seed_session ties /get_weather to /verify.
    server = _server()
    app = server.setup_webserver()

    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        seed_response = client.post("/seed_session", json={"verifier_metadata": {"expected_city": "Paris"}})
        assert seed_response.status_code == 200

        tool_response = client.post("/get_weather", json={"city": "Paris"})
        assert tool_response.status_code == 200
        assert tool_response.json()["weather"] == "The weather in Paris is sunny and 72 F."

        verify_response = client.post(
            "/verify",
            json=_verify_request("Paris", "The weather in Paris is sunny and 72 F.").model_dump(mode="json"),
        )
        assert verify_response.status_code == 200
        assert verify_response.json()["reward"] == 1.0
        assert verify_response.json()["tool_call_seen"] is True


def _rpc(client: TestClient, method: str, params: dict | None = None, token: str | None = None, rid: int = 1) -> dict:
    headers = dict(RPC_HEADERS)
    if token:
        headers[TOKEN_HEADER] = token
    body = {"jsonrpc": "2.0", "id": rid, "method": method}
    if params is not None:
        body["params"] = params
    return client.post("/mcp", headers=headers, json=body, follow_redirects=False).json()


def _mcp_client(server: ExampleMCPWeatherResourcesServer) -> TestClient:
    from nemo_gym.mcp_auto_exposure import maybe_auto_expose

    app = server.setup_webserver()
    maybe_auto_expose(server, app)
    return TestClient(app, base_url="http://127.0.0.1:8000")


def _handshake(client: TestClient) -> None:
    _rpc(
        client,
        "initialize",
        {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}},
    )
    client.post("/mcp", headers=RPC_HEADERS, json={"jsonrpc": "2.0", "method": "notifications/initialized"})


def test_streamable_http_mcp_endpoint_records_same_session() -> None:
    # The MCP door: /seed_session returns the "mcp" metadata; tools/call carries the per-rollout
    # token, so the tool call lands in the same session /verify scores.
    pytest.importorskip("mcp")
    server = _server()

    with _mcp_client(server) as client:
        seed_response = client.post("/seed_session", json={"verifier_metadata": {"expected_city": "Paris"}})
        assert seed_response.status_code == 200
        metadata = seed_response.json()["mcp"]
        assert metadata["server_name"] == "example_mcp_weather"
        token = metadata["headers"][TOKEN_HEADER]

        _handshake(client)
        result = _rpc(
            client,
            "tools/call",
            {"name": "get_weather", "arguments": {"city": "Paris"}},
            token=token,
            rid=2,
        )["result"]
        assert result.get("isError") is not True, result
        assert json.loads(result["content"][0]["text"])["weather"] == "The weather in Paris is sunny and 72 F."

        verify_response = client.post(
            "/verify",
            json=_verify_request("Paris", "The weather in Paris is sunny and 72 F.").model_dump(mode="json"),
        )
        assert verify_response.status_code == 200
        assert verify_response.json()["reward"] == 1.0
        assert verify_response.json()["tool_call_seen"] is True


def test_mcp_tool_call_requires_session_token() -> None:
    pytest.importorskip("mcp")
    server = _server()

    with _mcp_client(server) as client:
        client.post("/seed_session", json={"verifier_metadata": {"expected_city": "Paris"}})
        _handshake(client)
        result = _rpc(
            client,
            "tools/call",
            {"name": "get_weather", "arguments": {"city": "Paris"}},
            token=None,
            rid=2,
        )["result"]
        assert result.get("isError") is True
