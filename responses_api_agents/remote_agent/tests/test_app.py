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
import json
from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest
from aiohttp import ClientConnectorError, ClientPayloadError, ServerDisconnectedError
from pydantic import ValidationError

import responses_api_agents.remote_agent.app as remote_agent_app
from nemo_gym.config_types import ResourcesServerRef
from nemo_gym.rollout_collection import NG_FAILURE_CLASS_KEY, NG_NO_PERSIST_KEY, NG_TERMINAL_KEY
from nemo_gym.server_utils import ServerClient
from responses_api_agents.remote_agent.app import (
    REMOTE_AGENT_FAILURE_CLASS,
    RESOURCES_URL_HEADER,
    SESSION_COOKIE_HEADER,
    RemoteAgent,
    RemoteAgentConfig,
    RemoteAgentRunRequest,
    cookie_header_value,
    normalize_remote_url,
)


_MINIMAL_TRAJECTORY = {
    "id": "traj_1",
    "created_at": 1.0,
    "model": "their-model",
    "object": "response",
    "output": [
        {
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "id": "msg_1",
            "content": [{"type": "output_text", "text": "the answer is 42", "annotations": []}],
        }
    ],
    "parallel_tool_calls": False,
    "tools": [],
    "tool_choice": "auto",
    "usage": {
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens_details": {"reasoning_tokens": 0},
    },
}


def make_config(**overrides) -> RemoteAgentConfig:
    fields = dict(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="remote_agent",
        agent_base_url="http://localhost:9000",
        resources_server=ResourcesServerRef(type="resources_servers", name="my_env"),
    )
    fields.update(overrides)
    return RemoteAgentConfig(**fields)


def make_agent(server_client=None, **config_overrides) -> RemoteAgent:
    return RemoteAgent(
        config=make_config(**config_overrides),
        server_client=server_client or MagicMock(spec=ServerClient),
    )


def make_row(tools=None, **extras) -> dict:
    row = {
        "responses_create_params": {"input": [{"role": "user", "content": "what is 6 x 7?"}]},
        "verifier_metadata": {"expected_answer": "42"},
    }
    if tools is not None:
        row["responses_create_params"]["tools"] = tools
    row.update(extras)
    return row


def make_request(cookies=None) -> MagicMock:
    request = MagicMock()
    request.cookies = cookies or {}
    return request


class FakeRemoteResponse:
    """Stands in for an aiohttp ClientResponse from the remote service."""

    def __init__(self, status: int, content: bytes, headers=None, read_exc=None):
        self.status = status
        self._content = content
        self.headers = headers or {}
        self._read_exc = read_exc

    @property
    def ok(self) -> bool:
        return self.status < 400

    async def read(self) -> bytes:
        if self._read_exc is not None:
            raise self._read_exc
        return self._content


class FakeServerClientResponse:
    """Stands in for an aiohttp ClientResponse from a Gym server via ServerClient."""

    def __init__(self, body: dict, cookies=None, status: int = 200):
        self._body = body
        self.cookies = cookies or {}
        self.status = status

    @property
    def ok(self) -> bool:
        return self.status < 400

    @property
    def content(self):
        reader = MagicMock()

        async def _read():
            return orjson.dumps(self._body)

        reader.read = _read
        return reader

    def raise_for_status(self):
        if not self.ok:
            raise RuntimeError(f"HTTP {self.status}")

    async def read(self) -> bytes:
        return orjson.dumps(self._body)


def mock_remote(monkeypatch: pytest.MonkeyPatch, request_mock: AsyncMock) -> MagicMock:
    client = MagicMock()
    client.request = request_mock
    monkeypatch.setattr(remote_agent_app, "get_global_aiohttp_client", lambda: client)
    monkeypatch.setattr(remote_agent_app, "_REMOTE_RETRY_SLEEP_SECS", 0)
    return client


def seed_verify_server_client(verify_body=None, seed_cookies=None, seed_status=200, verify_status=200):
    """A ServerClient mock that answers /seed_session and /verify."""
    calls = []

    async def _post(server_name, url_path, json=None, cookies=None, **kwargs):
        calls.append({"server_name": server_name, "url_path": url_path, "json": json, "cookies": cookies})
        if url_path == "/seed_session":
            return FakeServerClientResponse({}, cookies=seed_cookies or {"session": "abc123"}, status=seed_status)
        if url_path == "/verify":
            body = verify_body if verify_body is not None else (json | {"reward": 1.0})
            return FakeServerClientResponse(body, status=verify_status)
        return FakeServerClientResponse({}, status=200)

    server_client = MagicMock(spec=ServerClient)
    server_client.post = AsyncMock(side_effect=_post)
    server_client.calls = calls
    return server_client


class TestConfig:
    def test_sanity_construct_and_semaphore(self) -> None:
        agent = make_agent(concurrency=7)
        assert agent.sem._value == 7

    def test_agent_base_url_normalized(self) -> None:
        assert make_config(agent_base_url="http://localhost:9000/").agent_base_url == "http://localhost:9000"

    @pytest.mark.parametrize(
        "bad_url",
        ["ftp://h:1", "localhost:9000", "http://h:1?token=abc", "http://h:1#frag", "http://user:pass@h:1"],
    )
    def test_agent_base_url_rejected(self, bad_url: str) -> None:
        with pytest.raises(ValidationError):
            make_config(agent_base_url=bad_url)

    def test_normalize_remote_url_never_echoes_credentials(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            normalize_remote_url("http://user:hunter2@h:1")
        assert "hunter2" not in str(exc_info.value)

    def test_cookie_header_value_shapes(self) -> None:
        assert cookie_header_value({}) is None
        assert cookie_header_value({"a": "1", "b": "2"}) == "a=1; b=2"
        morsel = MagicMock()
        morsel.value = "xyz"
        assert cookie_header_value({"session": morsel}) == "session=xyz"


class TestRunHappyPath:
    async def test_seed_then_remote_then_verify(self, monkeypatch: pytest.MonkeyPatch) -> None:
        request_mock = AsyncMock(return_value=FakeRemoteResponse(200, orjson.dumps(_MINIMAL_TRAJECTORY)))
        client = mock_remote(monkeypatch, request_mock)
        server_client = seed_verify_server_client(seed_cookies={"session": "s1"})
        agent = make_agent(server_client=server_client)

        row = make_row()
        result = await agent.run(make_request(), RemoteAgentRunRequest.model_validate(row))

        # Order and payloads: seed first, remote POST in between, verify last on the seed cookies
        assert [c["url_path"] for c in server_client.calls] == ["/seed_session", "/verify"]
        assert server_client.calls[1]["cookies"] == {"session": "s1"}
        assert server_client.calls[1]["json"]["response"]["id"] == "traj_1"
        assert server_client.calls[1]["json"]["verifier_metadata"] == {"expected_answer": "42"}

        args, kwargs = client.request.call_args
        assert args == ("POST", "http://localhost:9000/v1/responses")
        # The remote service receives ONLY create-params: no verifier_metadata, no row keys
        remote_payload = orjson.loads(kwargs["data"])
        assert remote_payload == row["responses_create_params"]
        assert kwargs["allow_redirects"] is False
        assert kwargs["timeout"].total == 1800.0
        # No session forwarding by default
        assert RESOURCES_URL_HEADER not in kwargs["headers"]
        assert SESSION_COOKIE_HEADER not in kwargs["headers"]

        dumped = result.model_dump()
        assert dumped["reward"] == 1.0
        assert NG_FAILURE_CLASS_KEY not in dumped
        assert NG_NO_PERSIST_KEY not in dumped
        assert NG_TERMINAL_KEY not in dumped

    async def test_verify_extras_pass_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_remote(monkeypatch, AsyncMock(return_value=FakeRemoteResponse(200, orjson.dumps(_MINIMAL_TRAJECTORY))))
        row = make_row()
        verify_body = row | {"response": _MINIMAL_TRAJECTORY, "reward": 0.5, "grading_notes": "close enough"}
        agent = make_agent(server_client=seed_verify_server_client(verify_body=verify_body))

        result = await agent.run(make_request(), RemoteAgentRunRequest.model_validate(row))

        assert result.model_dump()["grading_notes"] == "close enough"
        assert result.reward == 0.5


class TestRemoteFailuresBecomeSentinelRows:
    async def _run(self, monkeypatch, request_mock, **config_overrides):
        client = mock_remote(monkeypatch, request_mock)
        server_client = seed_verify_server_client()
        agent = make_agent(server_client=server_client, **config_overrides)
        result = await agent.run(make_request(), RemoteAgentRunRequest.model_validate(make_row()))
        return client, server_client, result.model_dump()

    async def test_timeout_fails_once_without_retry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client, _, result = await self._run(monkeypatch, AsyncMock(side_effect=asyncio.TimeoutError()))
        assert client.request.call_count == 1
        assert result[NG_FAILURE_CLASS_KEY] == REMOTE_AGENT_FAILURE_CLASS
        assert NG_TERMINAL_KEY not in result
        assert "timed out after 1800.0s" in result["error"]
        assert result["reward"] == 0.0

    async def test_connect_exhaustion_after_bounded_retries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        connect_error = ClientConnectorError(MagicMock(), OSError("connection refused"))
        client, server_client, result = await self._run(monkeypatch, AsyncMock(side_effect=connect_error))
        assert client.request.call_count == 3
        assert result[NG_FAILURE_CLASS_KEY] == REMOTE_AGENT_FAILURE_CLASS
        assert "Is your service running at http://localhost:9000?" in result["error"]
        # verify is never reached on a failed remote call
        assert [c["url_path"] for c in server_client.calls] == ["/seed_session"]

    async def test_disconnect_then_success_retries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        request_mock = AsyncMock(
            side_effect=[ServerDisconnectedError(), FakeRemoteResponse(200, orjson.dumps(_MINIMAL_TRAJECTORY))]
        )
        client, _, result = await self._run(monkeypatch, request_mock)
        assert client.request.call_count == 2
        assert NG_FAILURE_CLASS_KEY not in result

    async def test_http_500_with_body_excerpt(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _, _, result = await self._run(monkeypatch, AsyncMock(return_value=FakeRemoteResponse(500, b"kaboom")))
        assert result[NG_FAILURE_CLASS_KEY] == REMOTE_AGENT_FAILURE_CLASS
        assert "HTTP 500" in result["error"] and "kaboom" in result["error"]

    async def test_redirect_rejected_with_location_hint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        response = FakeRemoteResponse(301, b"", headers={"Location": "https://elsewhere"})
        _, _, result = await self._run(monkeypatch, AsyncMock(return_value=response))
        assert "HTTP 301" in result["error"] and "https://elsewhere" in result["error"]

    async def test_invalid_json_body(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _, _, result = await self._run(monkeypatch, AsyncMock(return_value=FakeRemoteResponse(200, b"not json")))
        assert "not valid JSON" in result["error"]

    async def test_non_object_body(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _, _, result = await self._run(monkeypatch, AsyncMock(return_value=FakeRemoteResponse(200, b"[1, 2]")))
        assert "expected a JSON object" in result["error"]

    @pytest.mark.parametrize(
        "read_exc",
        [ClientPayloadError("Response payload is not completed"), asyncio.TimeoutError()],
        ids=["mid-body disconnect", "deadline during body read"],
    )
    async def test_body_read_failure(self, monkeypatch: pytest.MonkeyPatch, read_exc: Exception) -> None:
        response = FakeRemoteResponse(200, b"", read_exc=read_exc)
        _, _, result = await self._run(monkeypatch, AsyncMock(return_value=response))
        assert result[NG_FAILURE_CLASS_KEY] == REMOTE_AGENT_FAILURE_CLASS
        assert "reading the response body failed" in result["error"]

    async def test_unexpected_exception_fails_fast_without_retry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client, _, result = await self._run(monkeypatch, AsyncMock(side_effect=RuntimeError("surprise")))
        assert client.request.call_count == 1
        assert "RuntimeError: surprise" in result["error"]

    async def test_invalid_trajectory_shape_is_terminal_and_skips_verify(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bad = {"id": "x", "object": "response"}  # missing required Responses API fields
        client, server_client, result = await self._run(
            monkeypatch, AsyncMock(return_value=FakeRemoteResponse(200, orjson.dumps(bad)))
        )
        assert result[NG_FAILURE_CLASS_KEY] == REMOTE_AGENT_FAILURE_CLASS
        assert result[NG_TERMINAL_KEY] is True
        assert "invalid Responses API object" in result["error"]
        assert [c["url_path"] for c in server_client.calls] == ["/seed_session"]


class TestGymSideFailuresBecomeSentinelRows:
    async def test_seed_failure_skips_remote_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        request_mock = AsyncMock(return_value=FakeRemoteResponse(200, orjson.dumps(_MINIMAL_TRAJECTORY)))
        client = mock_remote(monkeypatch, request_mock)
        server_client = seed_verify_server_client(seed_status=500)
        agent = make_agent(server_client=server_client)

        result = (await agent.run(make_request(), RemoteAgentRunRequest.model_validate(make_row()))).model_dump()

        assert result[NG_FAILURE_CLASS_KEY] == REMOTE_AGENT_FAILURE_CLASS
        assert "/seed_session" in result["error"]
        assert client.request.call_count == 0

    async def test_verify_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_remote(monkeypatch, AsyncMock(return_value=FakeRemoteResponse(200, orjson.dumps(_MINIMAL_TRAJECTORY))))
        server_client = seed_verify_server_client(verify_status=500)
        agent = make_agent(server_client=server_client)

        result = (await agent.run(make_request(), RemoteAgentRunRequest.model_validate(make_row()))).model_dump()

        assert result[NG_FAILURE_CLASS_KEY] == REMOTE_AGENT_FAILURE_CLASS
        assert "/verify" in result["error"]
        assert result["reward"] == 0.0


class TestToolsGuardAndSessionForwarding:
    _TOOLS = [
        {
            "type": "function",
            "name": "increment_counter",
            "parameters": {
                "type": "object",
                "properties": {"count": {"type": "integer", "description": ""}},
                "required": ["count"],
                "additionalProperties": False,
            },
            "strict": True,
            "description": "",
        }
    ]

    async def test_declared_tools_refused_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = mock_remote(monkeypatch, AsyncMock())
        server_client = seed_verify_server_client()
        agent = make_agent(server_client=server_client)

        row = make_row(tools=self._TOOLS)
        result = (await agent.run(make_request(), RemoteAgentRunRequest.model_validate(row))).model_dump()

        assert result[NG_FAILURE_CLASS_KEY] == REMOTE_AGENT_FAILURE_CLASS
        assert result[NG_TERMINAL_KEY] is True
        assert "forward_session" in result["error"] and "assume_remote_tools" in result["error"]
        # Refused before any network traffic
        assert client.request.call_count == 0
        assert server_client.calls == []

    async def test_assume_remote_tools_skips_guard_without_headers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        request_mock = AsyncMock(return_value=FakeRemoteResponse(200, orjson.dumps(_MINIMAL_TRAJECTORY)))
        client = mock_remote(monkeypatch, request_mock)
        agent = make_agent(server_client=seed_verify_server_client(), assume_remote_tools=True)

        result = await agent.run(make_request(), RemoteAgentRunRequest.model_validate(make_row(tools=self._TOOLS)))

        assert NG_FAILURE_CLASS_KEY not in result.model_dump()
        headers = client.request.call_args.kwargs["headers"]
        assert RESOURCES_URL_HEADER not in headers

    async def test_forward_session_sends_url_and_cookie_headers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        request_mock = AsyncMock(return_value=FakeRemoteResponse(200, orjson.dumps(_MINIMAL_TRAJECTORY)))
        client = mock_remote(monkeypatch, request_mock)
        monkeypatch.setattr(remote_agent_app, "get_server_url", lambda name: f"http://resolved-{name}:1234")
        server_client = seed_verify_server_client(seed_cookies={"session": "cookie-value"})
        agent = make_agent(server_client=server_client, forward_session=True)

        result = await agent.run(make_request(), RemoteAgentRunRequest.model_validate(make_row(tools=self._TOOLS)))

        assert NG_FAILURE_CLASS_KEY not in result.model_dump()
        headers = client.request.call_args.kwargs["headers"]
        assert headers[RESOURCES_URL_HEADER] == "http://resolved-my_env:1234"
        assert headers[SESSION_COOKIE_HEADER] == "session=cookie-value"

    async def test_skills_ref_warns_and_continues(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        mock_remote(monkeypatch, AsyncMock(return_value=FakeRemoteResponse(200, orjson.dumps(_MINIMAL_TRAJECTORY))))
        agent = make_agent(server_client=seed_verify_server_client())

        row = make_row(skills_ref={"path": "/skills", "hash": "abc", "skills": []})
        result = await agent.run(make_request(), RemoteAgentRunRequest.model_validate(row))

        assert NG_FAILURE_CLASS_KEY not in result.model_dump()
        assert "skills_ref" in capsys.readouterr().out


class TestResponseQualityWarnings:
    async def _run_with_trajectory(self, monkeypatch, trajectory):
        mock_remote(monkeypatch, AsyncMock(return_value=FakeRemoteResponse(200, orjson.dumps(trajectory))))
        agent = make_agent(server_client=seed_verify_server_client())
        return await agent.run(make_request(), RemoteAgentRunRequest.model_validate(make_row()))

    async def test_non_terminal_trajectory_warns(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        trajectory = dict(_MINIMAL_TRAJECTORY)
        trajectory["output"] = [
            {
                "type": "function_call",
                "status": "completed",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "increment_counter",
                "arguments": "{}",
            }
        ]
        result = await self._run_with_trajectory(monkeypatch, trajectory)
        assert NG_FAILURE_CLASS_KEY not in result.model_dump()
        assert "does not end with an assistant message" in capsys.readouterr().out

    async def test_missing_usage_warns(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        trajectory = dict(_MINIMAL_TRAJECTORY)
        trajectory.pop("usage")
        result = await self._run_with_trajectory(monkeypatch, trajectory)
        assert NG_FAILURE_CLASS_KEY not in result.model_dump()
        assert "no usage" in capsys.readouterr().out

    async def test_clean_trajectory_no_warnings(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        await self._run_with_trajectory(monkeypatch, _MINIMAL_TRAJECTORY)
        out = capsys.readouterr().out
        assert "WARNING" not in out


class TestRunTimeoutAndSemaphore:
    async def test_run_wallclock_bound_becomes_sentinel(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def slow_request(*args, **kwargs):
            await asyncio.sleep(30)

        mock_remote(monkeypatch, AsyncMock(side_effect=slow_request))
        agent = make_agent(server_client=seed_verify_server_client(), run_timeout_secs=0.05)

        result = (await agent.run(make_request(), RemoteAgentRunRequest.model_validate(make_row()))).model_dump()

        assert result[NG_FAILURE_CLASS_KEY] == REMOTE_AGENT_FAILURE_CLASS
        assert "run_timeout_secs" in result["error"]

    async def test_semaphore_bounds_in_flight_and_releases_on_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        in_flight = 0
        max_in_flight = 0
        release = asyncio.Event()

        async def gated_request(*args, **kwargs):
            nonlocal in_flight, max_in_flight
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            await release.wait()
            in_flight -= 1
            return FakeRemoteResponse(500, b"boom")  # failure path must release the permit too

        mock_remote(monkeypatch, AsyncMock(side_effect=gated_request))
        agent = make_agent(server_client=seed_verify_server_client(), concurrency=2)

        rows = [RemoteAgentRunRequest.model_validate(make_row()) for _ in range(4)]
        tasks = [asyncio.create_task(agent.run(make_request(), row)) for row in rows]
        await asyncio.sleep(0.05)
        assert max_in_flight == 2
        release.set()
        results = await asyncio.gather(*tasks)

        assert all(r.model_dump()[NG_FAILURE_CLASS_KEY] == REMOTE_AGENT_FAILURE_CLASS for r in results)
        assert agent.sem._value == 2  # every permit released despite 4 failures


class TestRoutes:
    def _client_and_mocks(self, monkeypatch, request_mock=None):
        from fastapi.testclient import TestClient

        mock_remote(
            monkeypatch,
            request_mock or AsyncMock(return_value=FakeRemoteResponse(200, orjson.dumps(_MINIMAL_TRAJECTORY))),
        )
        agent = make_agent(server_client=seed_verify_server_client())
        return TestClient(agent.setup_webserver(), raise_server_exceptions=False)

    def test_run_route_happy_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = self._client_and_mocks(monkeypatch)
        response = client.post("/run", json=make_row())
        assert response.status_code == 200
        assert response.json()["reward"] == 1.0

    def test_run_route_failure_serializes_sentinel_with_http_200(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The sentinel body must survive FastAPI response-model serialization: a 500 here
        # would abort the entire collection run instead of routing to the failures sidecar.
        client = self._client_and_mocks(monkeypatch, AsyncMock(side_effect=RuntimeError("remote exploded")))
        response = client.post("/run", json=make_row())
        assert response.status_code == 200
        body = response.json()
        assert body[NG_FAILURE_CLASS_KEY] == REMOTE_AGENT_FAILURE_CLASS
        assert body["reward"] == 0.0
        assert body["response"]["output"][0]["type"] == "message"

    async def test_responses_not_implemented(self) -> None:
        agent = make_agent()
        with pytest.raises(NotImplementedError):
            await agent.responses(body={})


class TestStatefulToolsEndToEnd:
    """The full session contract, in-process: RemoteAgent seeds the counter environment,
    forwards the session to a fake remote service, the service calls the counter tools with
    the forwarded cookie, and verify() scores the mutated session state."""

    def _counter_client(self):
        from fastapi.testclient import TestClient

        from resources_servers.example_session_state_mgmt.app import (
            StatefulCounterResourcesServer,
            StatefulCounterResourcesServerConfig,
        )

        config = StatefulCounterResourcesServerConfig(
            host="0.0.0.0", port=8081, entrypoint="", name="counter", domain="agent"
        )
        server = StatefulCounterResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        return TestClient(server.setup_webserver())

    async def test_counter_env_reward_through_forwarded_session(self, monkeypatch: pytest.MonkeyPatch) -> None:
        counter = self._counter_client()

        async def gym_post(server_name, url_path, json=None, cookies=None, **kwargs):
            response = counter.post(url_path, json=json, cookies=dict(cookies or {}))
            return FakeServerClientResponse(
                response.json(), cookies=dict(response.cookies), status=response.status_code
            )

        server_client = MagicMock(spec=ServerClient)
        server_client.post = AsyncMock(side_effect=gym_post)

        async def remote_service(method, url, data=None, headers=None, **kwargs):
            # The remote service reads the forwarded session and calls the counter tools
            # with the cookie echoed on every call — the contract under test.
            cookie_pair = headers[SESSION_COOKIE_HEADER]
            cookie_name, cookie_value = cookie_pair.split("=", 1)
            tool_cookies = {cookie_name: cookie_value}
            assert headers[RESOURCES_URL_HEADER].startswith("http://")

            assert counter.post("/increment_counter", json={"count": 1}, cookies=tool_cookies).status_code == 200
            assert counter.post("/increment_counter", json={"count": 2}, cookies=tool_cookies).status_code == 200
            count = counter.post("/get_counter_value", json={}, cookies=tool_cookies).json()["count"]

            trajectory = dict(_MINIMAL_TRAJECTORY)
            trajectory["output"] = [
                {
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "id": "msg_1",
                    "content": [{"type": "output_text", "text": f"final count is {count}", "annotations": []}],
                }
            ]
            return FakeRemoteResponse(200, orjson.dumps(trajectory))

        mock_remote(monkeypatch, AsyncMock(side_effect=remote_service))
        monkeypatch.setattr(remote_agent_app, "get_server_url", lambda name: "http://counter-in-process")

        agent = make_agent(server_client=server_client, forward_session=True)
        row = {
            "responses_create_params": {
                "input": [{"role": "user", "content": "add 1 then add 2 then get the count"}],
                "tools": [
                    {
                        "type": "function",
                        "name": "increment_counter",
                        "parameters": {
                            "type": "object",
                            "properties": {"count": {"type": "integer", "description": ""}},
                            "required": ["count"],
                            "additionalProperties": False,
                        },
                        "strict": True,
                        "description": "",
                    }
                ],
            },
            "initial_count": 3,
            "expected_count": 6,
        }

        result = await agent.run(make_request(), RemoteAgentRunRequest.model_validate(row))

        dumped = result.model_dump()
        assert NG_FAILURE_CLASS_KEY not in dumped
        # Reward 1.0 only if seed, both tool calls, and verify all shared ONE session
        assert dumped["reward"] == 1.0
        assert "final count is 6" in dumped["response"]["output"][0]["content"][0]["text"]


class TestCollectorRoundTrip:
    """Drive the real rollout-collection helper against this agent in-process and assert
    the sidecar contract end to end: successes to the main jsonl, sentinel rows to the
    failures sidecar."""

    async def test_success_and_failure_routing(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        from fastapi.testclient import TestClient

        import nemo_gym.rollout_collection
        from nemo_gym.rollout_collection import RolloutCollectionConfig, RolloutCollectionHelper

        # The collector reads the global config for model-call capture dirs; neutralize the
        # Hydra CLI parse it would otherwise attempt under pytest (same as the core tests).
        monkeypatch.setattr(nemo_gym.rollout_collection, "get_global_config_dict", MagicMock(return_value={}))

        async def remote_service(method, url, data=None, headers=None, **kwargs):
            params = orjson.loads(data)
            if "fail" in params["input"][0]["content"]:
                return FakeRemoteResponse(500, b"remote exploded")
            return FakeRemoteResponse(200, orjson.dumps(_MINIMAL_TRAJECTORY))

        mock_remote(monkeypatch, AsyncMock(side_effect=remote_service))
        agent = make_agent(server_client=seed_verify_server_client())
        agent_http = TestClient(agent.setup_webserver(), raise_server_exceptions=False)

        class InProcessHelper(RolloutCollectionHelper):
            def setup_server_client(self, *args, **kwargs):
                async def _post(server_name, url_path, json=None, **kw):
                    response = agent_http.post(url_path, json=json)
                    return FakeServerClientResponse(response.json(), status=response.status_code)

                server_client = MagicMock(spec=ServerClient)
                server_client.post = AsyncMock(side_effect=_post)
                return server_client

            async def _call_aggregate_metrics(self, results, rows, output_fpath):
                return None

        input_fpath = tmp_path / "input.jsonl"
        rows = [
            {"responses_create_params": {"input": [{"role": "user", "content": "please succeed"}]}},
            {"responses_create_params": {"input": [{"role": "user", "content": "please fail"}]}},
        ]
        input_fpath.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(input_fpath),
            output_jsonl_fpath=str(tmp_path / "rollouts.jsonl"),
            agent_name="remote_agent",
            upload_rollouts_to_wandb=False,
        )
        await InProcessHelper().run_from_config(config)

        main_rows = [json.loads(line) for line in (tmp_path / "rollouts.jsonl").open()]
        assert len(main_rows) == 1
        assert main_rows[0]["reward"] == 1.0

        sidecar_rows = [json.loads(line) for line in (tmp_path / "rollouts_failures.jsonl").open()]
        assert len(sidecar_rows) == 1
        assert sidecar_rows[0][NG_FAILURE_CLASS_KEY] == REMOTE_AGENT_FAILURE_CLASS
        assert "HTTP 500" in sidecar_rows[0]["error"]


class TestReviewFindingPins:
    """Regression pins for the adversarial-review findings."""

    _REUSED_ROW_EXTRAS = {
        "reward": 0.75,
        "response": {"stale": True},
        "error": "stale error",
        NG_FAILURE_CLASS_KEY: "stale_class",
        NG_NO_PERSIST_KEY: True,
        NG_TERMINAL_KEY: True,
    }

    async def test_failure_on_reused_rollout_row_still_returns_sentinel(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A rollouts/failures JSONL re-fed as a dataset carries reward/response/error and stale
        # routing keys; the failure path must not TypeError on them (the never-raise contract).
        mock_remote(monkeypatch, AsyncMock(side_effect=RuntimeError("remote exploded")))
        agent = make_agent(server_client=seed_verify_server_client())

        row = make_row(**self._REUSED_ROW_EXTRAS)
        result = (await agent.run(make_request(), RemoteAgentRunRequest.model_validate(row))).model_dump()

        assert result[NG_FAILURE_CLASS_KEY] == REMOTE_AGENT_FAILURE_CLASS
        assert result["reward"] == 0.0
        assert result["response"]["output"][0]["type"] == "message"
        assert "remote exploded" in result["error"]
        # Stale no-persist/terminal flags from the input row must not survive
        assert NG_NO_PERSIST_KEY not in result
        assert NG_TERMINAL_KEY not in result

    def test_failure_on_reused_rollout_row_route_level_stays_200(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from fastapi.testclient import TestClient

        mock_remote(monkeypatch, AsyncMock(side_effect=RuntimeError("remote exploded")))
        agent = make_agent(server_client=seed_verify_server_client())
        client = TestClient(agent.setup_webserver(), raise_server_exceptions=False)

        response = client.post("/run", json=make_row(**self._REUSED_ROW_EXTRAS))

        assert response.status_code == 200
        assert response.json()[NG_FAILURE_CLASS_KEY] == REMOTE_AGENT_FAILURE_CLASS

    async def test_happy_path_reused_row_leaks_no_stale_sentinels(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Stale routing keys on an input row must not echo through verify and misroute a
        # SUCCESS into the failures sidecar.
        mock_remote(monkeypatch, AsyncMock(return_value=FakeRemoteResponse(200, orjson.dumps(_MINIMAL_TRAJECTORY))))
        agent = make_agent(server_client=seed_verify_server_client())

        row = make_row(**self._REUSED_ROW_EXTRAS)
        result = (await agent.run(make_request(), RemoteAgentRunRequest.model_validate(row))).model_dump()

        assert result["reward"] == 1.0
        assert NG_FAILURE_CLASS_KEY not in result
        assert NG_NO_PERSIST_KEY not in result
        assert NG_TERMINAL_KEY not in result

    async def test_run_outer_backstop_never_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        agent = make_agent()

        async def explode(*args, **kwargs):
            raise RuntimeError("internal bug")

        monkeypatch.setattr(agent, "_run_once", explode)
        result = (await agent.run(make_request(), RemoteAgentRunRequest.model_validate(make_row()))).model_dump()

        assert result[NG_FAILURE_CLASS_KEY] == REMOTE_AGENT_FAILURE_CLASS
        assert "internal bug" in result["error"]

    async def test_aggregate_metrics_proxies_to_resources_server(self) -> None:
        agg_body = {
            "agent_metrics": {"mean/reward": 1.0},
            "key_metrics": {"mean/reward": 1.0},
            "group_level_metrics": [],
        }

        async def _post(server_name, url_path, json=None, **kwargs):
            assert url_path == "/aggregate_metrics"
            assert server_name == "my_env"
            return FakeServerClientResponse(agg_body)

        server_client = MagicMock(spec=ServerClient)
        server_client.post = AsyncMock(side_effect=_post)
        agent = make_agent(server_client=server_client)

        from nemo_gym.base_resources_server import AggregateMetricsRequest

        result = await agent.aggregate_metrics(AggregateMetricsRequest(verify_responses=[]))
        assert result.key_metrics == {"mean/reward": 1.0}

    async def test_aggregate_metrics_bounded_when_resources_server_hangs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def hang(*args, **kwargs):
            await asyncio.sleep(60)

        server_client = MagicMock(spec=ServerClient)
        server_client.post = AsyncMock(side_effect=hang)
        agent = make_agent(server_client=server_client)
        monkeypatch.setattr(remote_agent_app, "_AGGREGATE_PROXY_TIMEOUT_SECS", 0.05)

        from nemo_gym.base_resources_server import AggregateMetricsRequest

        with pytest.raises(asyncio.TimeoutError):
            await agent.aggregate_metrics(AggregateMetricsRequest(verify_responses=[]))

    async def test_run_timeout_excludes_semaphore_queue_wait(self, monkeypatch: pytest.MonkeyPatch) -> None:
        release_first = asyncio.Event()

        async def gated(*args, **kwargs):
            await release_first.wait()
            return FakeRemoteResponse(200, orjson.dumps(_MINIMAL_TRAJECTORY))

        mock_remote(monkeypatch, AsyncMock(side_effect=gated))
        agent = make_agent(server_client=seed_verify_server_client(), concurrency=1, run_timeout_secs=0.5)

        first = asyncio.create_task(agent.run(make_request(), RemoteAgentRunRequest.model_validate(make_row())))
        second = asyncio.create_task(agent.run(make_request(), RemoteAgentRunRequest.model_validate(make_row())))
        # Hold the only permit for most of the second task's would-be budget
        await asyncio.sleep(0.4)
        release_first.set()
        results = [r.model_dump() for r in await asyncio.gather(first, second)]

        # If queue wait counted against run_timeout_secs, the second task would time out
        assert all(NG_FAILURE_CLASS_KEY not in r for r in results)

    async def test_forward_session_loopback_warning_for_offhost_remote(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        mock_remote(monkeypatch, AsyncMock(return_value=FakeRemoteResponse(200, orjson.dumps(_MINIMAL_TRAJECTORY))))
        monkeypatch.setattr(remote_agent_app, "get_server_url", lambda name: "http://127.0.0.1:15022")
        agent = make_agent(
            server_client=seed_verify_server_client(),
            forward_session=True,
            agent_base_url="http://gpu-node-7:9000",
        )

        await agent.run(make_request(), RemoteAgentRunRequest.model_validate(make_row()))

        assert "advertised_resources_url" in capsys.readouterr().out

    async def test_advertised_resources_url_overrides_header_and_silences_warning(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        request_mock = AsyncMock(return_value=FakeRemoteResponse(200, orjson.dumps(_MINIMAL_TRAJECTORY)))
        client = mock_remote(monkeypatch, request_mock)
        monkeypatch.setattr(remote_agent_app, "get_server_url", lambda name: "http://127.0.0.1:15022")
        agent = make_agent(
            server_client=seed_verify_server_client(),
            forward_session=True,
            agent_base_url="http://gpu-node-7:9000",
            advertised_resources_url="http://head-node.cluster:15022",
        )

        await agent.run(make_request(), RemoteAgentRunRequest.model_validate(make_row()))

        headers = client.request.call_args.kwargs["headers"]
        assert headers[RESOURCES_URL_HEADER] == "http://head-node.cluster:15022"
        assert "advertised_resources_url" not in capsys.readouterr().out

    async def test_quality_warnings_are_throttled(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        trajectory = dict(_MINIMAL_TRAJECTORY)
        trajectory.pop("usage")
        mock_remote(monkeypatch, AsyncMock(return_value=FakeRemoteResponse(200, orjson.dumps(trajectory))))
        agent = make_agent(server_client=seed_verify_server_client())

        for _ in range(10):
            await agent.run(make_request(), RemoteAgentRunRequest.model_validate(make_row()))

        # Head of 5, then every 100th: 10 rollouts -> exactly 5 printed warnings
        assert capsys.readouterr().out.count("no usage") == 5
