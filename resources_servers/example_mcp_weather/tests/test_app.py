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

from nemo_gym.client import SESSION_ID_KEY, ServerClient
from nemo_gym.core import (
    EpisodeId,
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    TaskId,
)
from nemo_gym.server_runtime import (
    ResourcesCloseSessionRequest,
    ResourcesSeedSessionRequest,
    check_resources_session_contract,
)
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
    from nemo_gym.server_runtime import maybe_auto_expose

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


def _typed_seed(resources_session_id: str = "resources-session", task_data: dict | None = None) -> dict:
    # The body single_agent_turn sends: task_data holds the row's own fields.
    return ResourcesSeedSessionRequest(
        resources_session_id=resources_session_id,
        episode_id=EpisodeId(rollout_id="rollout", attempt=0),
        task_id=TaskId(taskset="example", task_id="0"),
        task_data={"verifier_metadata": {"expected_city": "Paris"}} if task_data is None else task_data,
    ).model_dump(mode="json")


def _typed_close(resources_session_id: str = "resources-session") -> dict:
    return ResourcesCloseSessionRequest(
        resources_session_id=resources_session_id, episode_id=EpisodeId(rollout_id="rollout", attempt=0)
    ).model_dump(mode="json")


@pytest.mark.parametrize("over_mcp", [False, True], ids=["http", "mcp"])
def test_resources_session_contract(over_mcp: bool) -> None:
    if over_mcp:
        pytest.importorskip("mcp")
    server = _server()
    app = server.setup_webserver()
    if over_mcp:
        # The shipped config sets expose_tools_over_mcp, which wraps /seed_session and /verify.
        from nemo_gym.server_runtime import maybe_auto_expose

        maybe_auto_expose(server, app)

    check_resources_session_contract(
        app,
        ResourcesSeedSessionRequest.model_validate(_typed_seed("contract-session")),
        keeps_state=True,
        holds_state=lambda resources_session_id: resources_session_id in server.session_id_to_state,
        verify_body=_verify_request("Paris", "The weather in Paris is sunny and 72 F.").model_dump(mode="json"),
    )


def test_typed_seed_rejects_malformed_task_data() -> None:
    server = _server()

    with TestClient(server.setup_webserver()) as client:
        response = client.post("/seed_session", json=_typed_seed(task_data={"verifier_metadata": "Paris"}))

    assert response.status_code == 422
    assert "task_data is not an example_mcp_weather task" in response.text
    assert server.session_id_to_state == {}


def test_repeated_typed_seed_keeps_recorded_weather_calls() -> None:
    server = _server()

    with TestClient(server.setup_webserver()) as client:
        assert client.post("/seed_session", json=_typed_seed()).status_code == 200
        assert client.post("/get_weather", json={"city": "Paris"}).status_code == 200
        # A seed retried after its reply was lost arrives without the cookie.
        client.cookies.clear()
        assert client.post("/seed_session", json=_typed_seed()).status_code == 200

        assert server.session_id_to_state["resources-session"]["weather_calls"] == [
            {"city": "Paris", "weather": "The weather in Paris is sunny and 72 F."}
        ]
        verify_response = client.post(
            "/verify",
            json=_verify_request("Paris", "The weather in Paris is sunny and 72 F.").model_dump(mode="json"),
        )

    assert verify_response.json()["reward"] == 1.0


def test_tool_call_without_open_session_is_lost() -> None:
    server = _server()

    with TestClient(server.setup_webserver()) as client:
        response = client.post("/get_weather", json={"city": "Paris"})

    assert response.status_code == 404
    assert response.json()["detail"]["failure_kind"] == "session_lost"
    assert server.session_id_to_state == {}


def test_typed_seed_grants_mcp_tools_until_close() -> None:
    # An Environment Server reads the MCP grant from resources_tools and hands its token to the agent; the token's
    # tool calls land in the session /verify scores, and fail once the session is closed.
    pytest.importorskip("mcp")
    server = _server()

    with _mcp_client(server) as client:
        seed_response = client.post("/seed_session", json=_typed_seed())
        assert seed_response.status_code == 200
        seed = seed_response.json()
        assert seed["resources_session_id"] == "resources-session"
        assert "mcp" not in seed
        assert seed["resources_tools"]["server_name"] == "example_mcp_weather"
        token = seed["resources_tools"]["headers"][TOKEN_HEADER]

        _handshake(client)
        result = _rpc(client, "tools/call", {"name": "get_weather", "arguments": {"city": "Paris"}}, token=token)
        assert result["result"].get("isError") is not True, result
        assert server.session_id_to_state["resources-session"]["weather_calls"][0]["city"] == "Paris"

        verify_body = _verify_request("Paris", "The weather in Paris is sunny and 72 F.").model_dump(mode="json")
        verify_response = client.post("/verify", json=verify_body)
        assert verify_response.json()["reward"] == 1.0
        assert verify_response.json()["tool_call_seen"] is True

        assert client.post("/close_session", json=_typed_close()).status_code == 200
        assert server.session_id_to_state == {}

        result = _rpc(
            client, "tools/call", {"name": "get_weather", "arguments": {"city": "Paris"}}, token=token, rid=3
        )
        assert result["result"].get("isError") is True
        assert "HTTP 404" in result["result"]["content"][0]["text"]
        assert server.session_id_to_state == {}

        verify_response = client.post("/verify", json=verify_body)
        assert verify_response.json()["mask_sample"] is True
        assert verify_response.json()["failure_kind"] == "session_lost"


@pytest.mark.parametrize("stage", ["active", "verified", "closed"])
def test_legacy_seed_cannot_overwrite_typed_or_closed_cookie(stage: str) -> None:
    server = _server()
    with TestClient(server.setup_webserver()) as client:
        assert client.post("/seed_session", json=_typed_seed()).status_code == 200
        if stage == "verified":
            assert (
                client.post(
                    "/verify",
                    json=_verify_request("Paris", "The weather in Paris is sunny and 72 F.").model_dump(mode="json"),
                ).status_code
                == 200
            )
        elif stage == "closed":
            assert client.post("/close_session", json=_typed_close()).status_code == 200
        before = dict(server.session_id_to_state)
        assert client.post("/seed_session", json={"verifier_metadata": {"expected_city": "Paris"}}).status_code == 409
        assert server.session_id_to_state == before


def test_malformed_typed_seed_does_not_fall_back_to_legacy() -> None:
    server = _server()
    malformed = dict({"verifier_metadata": {"expected_city": "Paris"}}) | {"resources_session_id": "incomplete-typed"}
    with TestClient(server.setup_webserver()) as client:
        assert client.post("/seed_session", json=malformed).status_code == 422
    assert server.session_id_to_state == {}
    assert server._session_episodes == {}


@pytest.mark.parametrize("typed", [False, True])
def test_checkpoint_verify_releases_state_but_retry_survives_until_expiry(typed: bool, monkeypatch) -> None:
    from omegaconf import DictConfig

    clock = [100.0]
    monkeypatch.setattr("nemo_gym.server_runtime.base_resources_server.monotonic", lambda: clock[0])
    server = _server()
    server.server_client.global_config_dict = DictConfig(
        {"checkpoint": {"enabled": True, "control_auth_token": "test"}}
    )
    prefix = "/ng-rollout/rollout"
    with TestClient(server.setup_webserver()) as client:
        assert (
            client.post(
                prefix + "/seed_session",
                json=_typed_seed() if typed else {"verifier_metadata": {"expected_city": "Paris"}},
            ).status_code
            == 200
        )
        sid = next(iter(server.session_id_to_state))
        assert server._checkpoint.is_open(sid)
        assert client.post(prefix + "/get_weather", json={"city": "Paris"}).status_code == 200
        first = client.post(
            prefix + "/verify",
            json=_verify_request("Paris", "The weather in Paris is sunny and 72 F.").model_dump(mode="json"),
        ).json()
        assert not first["mask_sample"]
        assert not server._checkpoint.is_open(sid)
        assert server.session_id_to_state == {}
        assert server._session_episodes == {}
        retry = client.post(
            prefix + "/verify",
            json=_verify_request("Paris", "The weather in Paris is sunny and 72 F.").model_dump(mode="json"),
        ).json()
        assert retry["reward"] == first["reward"] and retry["mask_sample"] is False
        if typed:
            wrong_owner = _typed_close()
            wrong_owner["episode_id"]["rollout_id"] = "other"
            assert client.post(prefix + "/close_session", json=wrong_owner).status_code == 409
        clock[0] += server.config.session_close_retry_window_seconds + 1
        assert (
            client.post(
                prefix + "/verify",
                json=_verify_request("Paris", "The weather in Paris is sunny and 72 F.").model_dump(mode="json"),
            ).json()["mask_sample"]
            is True
        )
        assert len(server._verified_sessions) == 0


def test_next_legacy_run_gets_new_cookie_and_old_verify_remains_retryable() -> None:
    server = _server()
    with TestClient(server.setup_webserver()) as client:
        assert client.post("/seed_session", json={"verifier_metadata": {"expected_city": "Paris"}}).status_code == 200
        old_cookies = dict(client.cookies)
        old_id = next(iter(server.session_id_to_state))
        first = client.post(
            "/verify", json=_verify_request("Paris", "The weather in Paris is sunny and 72 F.").model_dump(mode="json")
        ).json()
        assert client.post("/seed_session", json={"verifier_metadata": {"expected_city": "Paris"}}).status_code == 200
        new_id = next(iter(server.session_id_to_state))
        assert new_id != old_id
        assert (
            client.post(
                "/verify",
                json=_verify_request("Paris", "The weather in Paris is sunny and 72 F.").model_dump(mode="json"),
            ).json()["mask_sample"]
            is False
        )
        client.cookies.clear()
        client.cookies.update(old_cookies)
        retry = client.post(
            "/verify", json=_verify_request("Paris", "The weather in Paris is sunny and 72 F.").model_dump(mode="json")
        ).json()
        assert retry["reward"] == first["reward"] and retry["mask_sample"] is False


@pytest.mark.parametrize("typed", [False, True])
def test_checkpoint_retire_releases_live_session(typed: bool) -> None:
    from omegaconf import DictConfig

    server = _server()
    server.server_client.global_config_dict = DictConfig(
        {"checkpoint": {"enabled": True, "control_auth_token": "test"}}
    )
    with TestClient(server.setup_webserver()) as client:
        assert client.post("/ng-rollout/rollout/seed_session", json=_typed_seed() if typed else {}).status_code == 200
        client.portal.call(server._checkpoint.retire, EpisodeId(rollout_id="rollout", attempt=0))
        assert server.session_id_to_state == {}
        assert server._session_episodes == {}


def test_typed_seed_cannot_replace_a_live_legacy_session() -> None:
    server = _server()
    with TestClient(server.setup_webserver()) as client:
        assert client.post("/seed_session", json={}).status_code == 200
        sid = next(iter(server.session_id_to_state))
        original = server.session_id_to_state[sid]
        assert client.post("/seed_session", json=_typed_seed(sid)).status_code == 409
        assert server.session_id_to_state[sid] is original
        assert server._session_episodes == {}
