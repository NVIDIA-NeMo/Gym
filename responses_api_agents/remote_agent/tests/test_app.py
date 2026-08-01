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
from http.cookies import SimpleCookie
from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest
from aiohttp import ClientConnectorError, ClientPayloadError, ClientResponseError, ServerDisconnectedError
from fastapi import Response
from pydantic import BaseModel, ValidationError

import responses_api_agents.remote_agent.app as remote_agent_app
from nemo_gym.config_types import ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_collection import NG_FAILURE_CLASS_KEY, NG_NO_PERSIST_KEY, NG_TERMINAL_KEY
from nemo_gym.server_utils import ServerClient
from responses_api_agents.remote_agent.app import (
    REMOTE_AGENT_FAILURE_CLASS,
    RemoteAgent,
    RemoteAgentConfig,
    RemoteAgentRunRequest,
    normalize_remote_url,
)


def msg(text: str, item_id: str = "msg_1") -> dict:
    return {
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "id": item_id,
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def fn_call(call_id: str, name: str, arguments: str) -> dict:
    return {"type": "function_call", "call_id": call_id, "name": name, "arguments": arguments, "id": f"fc_{call_id}"}


def fn_output(call_id: str, output: str) -> dict:
    return {"type": "function_call_output", "call_id": call_id, "output": output}


def traj(output: list, usage: dict | None = "default") -> dict:
    t = {
        "id": "traj_1",
        "created_at": 1.0,
        "model": "their-model",
        "object": "response",
        "output": output,
        "parallel_tool_calls": False,
        "tools": [],
        "tool_choice": "auto",
    }
    if usage == "default":
        usage = {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        }
    if usage is not None:
        t["usage"] = usage
    return t


_MINIMAL_TRAJECTORY = traj([msg("the answer is 42")])

_COUNTER_TOOLS = [
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

    def __init__(self, status: int, content: bytes, headers=None, read_exc=None, set_cookies=None):
        self.status = status
        self._content = content
        self.headers = headers or {}
        self._read_exc = read_exc
        self.cookies = SimpleCookie()
        for k, v in (set_cookies or {}).items():
            self.cookies[k] = v

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
        # Mirror aiohttp so nemo_gym.raise_for_status attaches response_content, the
        # channel run() reads middleware-serialized errors (incl. terminal names) from.
        if not self.ok:
            raise ClientResponseError(request_info=MagicMock(), history=(), status=self.status, message="error")

    async def read(self) -> bytes:
        return orjson.dumps(self._body)


def mock_remote(monkeypatch: pytest.MonkeyPatch, request_mock: AsyncMock) -> MagicMock:
    client = MagicMock()
    client.request = request_mock
    monkeypatch.setattr(remote_agent_app, "get_global_aiohttp_client", lambda: client)
    monkeypatch.setattr(remote_agent_app, "_REMOTE_RETRY_SLEEP_SECS", 0)
    return client


def scripted_service(*turns):
    """AsyncMock remote service returning one canned trajectory per call, recording payloads."""
    received = []

    async def handler(method, url, data=None, headers=None, cookies=None, **kwargs):
        received.append({"payload": orjson.loads(data), "cookies": dict(cookies or {})})
        turn = turns[min(len(received) - 1, len(turns) - 1)]
        if isinstance(turn, FakeRemoteResponse):
            return turn
        return FakeRemoteResponse(200, orjson.dumps(turn))

    request_mock = AsyncMock(side_effect=handler)
    request_mock.received = received
    return request_mock


def make_agent(server_client=None, **config_overrides) -> RemoteAgent:
    return RemoteAgent(
        config=make_config(**config_overrides),
        server_client=server_client or MagicMock(spec=ServerClient),
    )


def wire_gym(
    agent: RemoteAgent,
    verify_body=None,
    seed_cookies=None,
    seed_status=200,
    verify_status=200,
    tool_handler=None,
):
    """A ServerClient mock that emulates the Gym side: seed_session and verify on the
    resources server, tool routes via tool_handler, and — the load-bearing part — the
    /v1/responses self-post routed into the agent's REAL responses() with the exception
    middleware emulated (exceptions become a 500 body carrying repr(e))."""
    calls = []

    async def _post(server_name, url_path, json=None, cookies=None, **kwargs):
        calls.append({"server_name": server_name, "url_path": url_path, "json": json, "cookies": cookies})
        if url_path == "/seed_session":
            return FakeServerClientResponse({}, cookies=seed_cookies or {"session": "abc123"}, status=seed_status)
        if url_path == "/verify":
            body = verify_body if verify_body is not None else (json | {"reward": 1.0})
            return FakeServerClientResponse(body, status=verify_status)
        if url_path.endswith("/v1/responses"):
            wire = json.model_dump(exclude_unset=True) if isinstance(json, BaseModel) else json
            params = NeMoGymResponseCreateParamsNonStreaming.model_validate(wire)
            fastapi_response = Response()
            try:
                result = await agent.responses(make_request(dict(cookies or {})), fastapi_response, params)
            except Exception as e:  # noqa: BLE001 -- emulate SimpleServer's exception middleware
                return FakeServerClientResponse({"error": repr(e)}, status=500)
            out_cookies = SimpleCookie()
            for header_value in fastapi_response.headers.getlist("set-cookie"):
                out_cookies.load(header_value)
            return FakeServerClientResponse(
                result.model_dump(mode="json"), cookies={k: m.value for k, m in out_cookies.items()}
            )
        if tool_handler is not None:
            return await tool_handler(url_path, json, cookies)
        return FakeServerClientResponse({}, status=200)

    server_client = MagicMock(spec=ServerClient)
    server_client.post = AsyncMock(side_effect=_post)
    server_client.calls = calls
    agent.server_client = server_client
    return server_client


def make_wired_agent(monkeypatch, request_mock, *, tool_handler=None, verify_body=None, **kwargs):
    client = mock_remote(monkeypatch, request_mock)
    agent = make_agent(
        **{k: v for k, v in kwargs.items() if k not in ("seed_cookies", "seed_status", "verify_status")}
    )
    server_client = wire_gym(
        agent,
        verify_body=verify_body,
        seed_cookies=kwargs.get("seed_cookies"),
        seed_status=kwargs.get("seed_status", 200),
        verify_status=kwargs.get("verify_status", 200),
        tool_handler=tool_handler,
    )
    return agent, client, server_client


class TestConfig:
    def test_sanity_construct_and_semaphore(self) -> None:
        agent = make_agent(concurrency=7)
        assert agent.sem._value == 7

    def test_agent_base_url_normalized(self) -> None:
        assert make_config(agent_base_url="http://localhost:9000/").agent_base_url == "http://localhost:9000"

    def test_max_steps_defaults_to_none(self) -> None:
        assert make_config().max_steps is None

    @pytest.mark.parametrize(
        "bad_url",
        [
            "ftp://h:1",
            "localhost:9000",
            "http://h:1?token=abc",
            "http://h:1#frag",
            "http://user:pass@h:1",  # pragma: allowlist secret
        ],
    )
    def test_agent_base_url_rejected(self, bad_url: str) -> None:
        with pytest.raises(ValidationError):
            make_config(agent_base_url=bad_url)

    def test_normalize_remote_url_never_echoes_credentials(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            normalize_remote_url("http://user:hunter2@h:1")  # pragma: allowlist secret
        assert "hunter2" not in str(exc_info.value)


class TestRunHappyPath:
    async def test_seed_then_loop_then_verify(self, monkeypatch: pytest.MonkeyPatch) -> None:
        service = scripted_service(_MINIMAL_TRAJECTORY)
        agent, client, server_client = make_wired_agent(monkeypatch, service, seed_cookies={"session": "s1"})

        row = make_row()
        result = await agent.run(make_request(), RemoteAgentRunRequest.model_validate(row))

        paths = [c["url_path"] for c in server_client.calls]
        assert paths == ["/seed_session", "/v1/responses", "/verify"]
        # Seeded cookies reach the loop and verify; verify carries the trajectory + row keys
        assert server_client.calls[1]["cookies"] == {"session": "s1"}
        assert server_client.calls[2]["json"]["response"]["id"] == "traj_1"
        assert server_client.calls[2]["json"]["verifier_metadata"] == {"expected_answer": "42"}

        # The remote service receives ONLY create-params: no verifier_metadata, no row keys
        assert service.received[0]["payload"] == row["responses_create_params"]
        args, request_kwargs = client.request.call_args
        assert args == ("POST", "http://localhost:9000/v1/responses")
        assert request_kwargs["allow_redirects"] is False
        assert request_kwargs["timeout"].total == 1800.0

        dumped = result.model_dump()
        assert dumped["reward"] == 1.0
        assert NG_FAILURE_CLASS_KEY not in dumped
        assert NG_NO_PERSIST_KEY not in dumped
        assert NG_TERMINAL_KEY not in dumped

    async def test_verify_extras_pass_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        row = make_row()
        verify_body = row | {"response": _MINIMAL_TRAJECTORY, "reward": 0.5, "grading_notes": "close enough"}
        agent, _, _ = make_wired_agent(monkeypatch, scripted_service(_MINIMAL_TRAJECTORY), verify_body=verify_body)

        result = await agent.run(make_request(), RemoteAgentRunRequest.model_validate(row))

        assert result.model_dump()["grading_notes"] == "close enough"
        assert result.reward == 0.5


class TestAgentToolLoop:
    """The Gym-driven loop: the service returns unpaired function_calls as asks; Gym
    executes them on the resources server and re-posts the grown conversation."""

    @staticmethod
    async def counter_tool_handler(url_path, body, cookies):
        if url_path == "/increment_counter":
            return FakeServerClientResponse({"success": True}, cookies={"tool_session": "t1"})
        if url_path == "/get_counter_value":
            return FakeServerClientResponse({"count": 6})
        return FakeServerClientResponse({"detail": f"Not Found: {url_path}"}, status=404)

    async def test_multi_turn_tool_execution(self, monkeypatch: pytest.MonkeyPatch) -> None:
        service = scripted_service(
            traj([fn_call("c1", "increment_counter", '{"count": 3}')]),
            traj([msg("done, counter incremented")]),
        )
        agent, _, server_client = make_wired_agent(monkeypatch, service, tool_handler=self.counter_tool_handler)

        row = make_row(tools=_COUNTER_TOOLS)
        result = await agent.run(make_request(), RemoteAgentRunRequest.model_validate(row))

        dumped = result.model_dump()
        assert NG_FAILURE_CLASS_KEY not in dumped
        assert dumped["reward"] == 1.0

        # Gym executed the tool: one resources-server POST to /increment_counter with parsed args
        tool_calls = [c for c in server_client.calls if c["url_path"] == "/increment_counter"]
        assert len(tool_calls) == 1
        assert tool_calls[0]["json"] == {"count": 3}

        # Turn 2 payload = original input + call + tool output
        second_input = service.received[1]["payload"]["input"]
        assert [i.get("type", "message") for i in second_input] == ["message", "function_call", "function_call_output"]
        assert second_input[2]["output"] == '{"success":true}'

        # The final trajectory carries the merged conversation
        types = [o["type"] for o in dumped["response"]["output"]]
        assert types == ["function_call", "function_call_output", "message"]

    async def test_paired_calls_pass_through_unexecuted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # call_a is the service's own internal tool record (paired); call_b is the ask.
        service = scripted_service(
            traj(
                [
                    fn_call("call_a", "web_search", '{"q": "counters"}'),
                    fn_output("call_a", '{"results": []}'),
                    fn_call("call_b", "increment_counter", '{"count": 3}'),
                ]
            ),
            traj([msg("done")]),
        )
        agent, _, server_client = make_wired_agent(monkeypatch, service, tool_handler=self.counter_tool_handler)

        result = await agent.run(make_request(), RemoteAgentRunRequest.model_validate(make_row(tools=_COUNTER_TOOLS)))

        dumped = result.model_dump()
        assert dumped["reward"] == 1.0
        executed = [c["url_path"] for c in server_client.calls if c["url_path"].startswith("/increment")]
        assert executed == ["/increment_counter"]
        # web_search was never sent to the resources server
        assert not any(c["url_path"] == "/web_search" for c in server_client.calls)
        # The paired record survives in the final trajectory
        types = [o["type"] for o in dumped["response"]["output"]]
        assert types.count("function_call") == 2 and types.count("function_call_output") == 2

    async def test_unknown_tool_error_fed_back_not_raised(self, monkeypatch: pytest.MonkeyPatch) -> None:
        service = scripted_service(
            traj([fn_call("c1", "web_search", "{}")]),  # unpaired ask for a tool the env doesn't serve
            traj([msg("recovered")]),
        )
        agent, _, _ = make_wired_agent(monkeypatch, service, tool_handler=self.counter_tool_handler)

        result = await agent.run(make_request(), RemoteAgentRunRequest.model_validate(make_row(tools=_COUNTER_TOOLS)))

        assert NG_FAILURE_CLASS_KEY not in result.model_dump()
        # The 404 body came back to the service as the tool output
        second_input = service.received[1]["payload"]["input"]
        assert "Not Found" in second_input[-1]["output"]

    async def test_malformed_arguments_fed_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        service = scripted_service(
            traj([fn_call("c1", "increment_counter", "not json")]),
            traj([msg("ok")]),
        )
        agent, _, server_client = make_wired_agent(monkeypatch, service, tool_handler=self.counter_tool_handler)

        result = await agent.run(make_request(), RemoteAgentRunRequest.model_validate(make_row(tools=_COUNTER_TOOLS)))

        assert NG_FAILURE_CLASS_KEY not in result.model_dump()
        assert not any(c["url_path"] == "/increment_counter" for c in server_client.calls)
        second_input = service.received[1]["payload"]["input"]
        assert "Invalid tool call arguments" in second_input[-1]["output"]

    async def test_max_steps_bounds_the_loop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        always_ask = traj([fn_call("c1", "increment_counter", '{"count": 1}')])
        service = scripted_service(always_ask, always_ask, always_ask, always_ask)
        agent, _, _ = make_wired_agent(monkeypatch, service, tool_handler=self.counter_tool_handler, max_steps=2)

        result = await agent.run(make_request(), RemoteAgentRunRequest.model_validate(make_row(tools=_COUNTER_TOOLS)))

        assert len(service.received) == 2
        assert NG_FAILURE_CLASS_KEY not in result.model_dump()

    async def test_service_cookies_round_trip(self, monkeypatch: pytest.MonkeyPatch) -> None:
        turn1 = FakeRemoteResponse(
            200,
            orjson.dumps(traj([fn_call("c1", "increment_counter", '{"count": 1}')])),
            set_cookies={"svc_session": "svc1"},
        )
        service = scripted_service(turn1, traj([msg("done")]))
        agent, _, _ = make_wired_agent(monkeypatch, service, tool_handler=self.counter_tool_handler)

        await agent.run(make_request(), RemoteAgentRunRequest.model_validate(make_row(tools=_COUNTER_TOOLS)))

        assert service.received[0]["cookies"] == {}
        assert service.received[1]["cookies"] == {"svc_session": "svc1"}

    async def test_usage_accumulates_across_turns(self, monkeypatch: pytest.MonkeyPatch) -> None:
        service = scripted_service(
            traj([fn_call("c1", "increment_counter", '{"count": 1}')]),
            traj([msg("done")]),
        )
        agent, _, _ = make_wired_agent(monkeypatch, service, tool_handler=self.counter_tool_handler)

        result = await agent.run(make_request(), RemoteAgentRunRequest.model_validate(make_row(tools=_COUNTER_TOOLS)))

        usage = result.model_dump()["response"]["usage"]
        assert usage["input_tokens"] == 20 and usage["output_tokens"] == 10 and usage["total_tokens"] == 30

    async def test_string_input_coerced_to_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        service = scripted_service(_MINIMAL_TRAJECTORY)
        agent, _, _ = make_wired_agent(monkeypatch, service)

        row = {"responses_create_params": {"input": "just a string"}}
        result = await agent.run(make_request(), RemoteAgentRunRequest.model_validate(row))

        assert NG_FAILURE_CLASS_KEY not in result.model_dump()
        assert service.received[0]["payload"]["input"][0]["content"] == "just a string"


class TestRemoteFailuresBecomeSentinelRows:
    async def _run(self, monkeypatch, request_mock, **config_overrides):
        agent, client, server_client = make_wired_agent(monkeypatch, request_mock, **config_overrides)
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
        # verify is never reached on a failed loop
        assert [c["url_path"] for c in server_client.calls] == ["/seed_session", "/v1/responses"]

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
        # Terminal classification survives the HTTP self-post boundary (exception-name match)
        assert result[NG_TERMINAL_KEY] is True
        assert "invalid Responses API object" in result["error"]
        assert [c["url_path"] for c in server_client.calls] == ["/seed_session", "/v1/responses"]

    async def test_mid_loop_failure_becomes_sentinel(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Turn 1 succeeds with a tool ask; turn 2 the service dies — the whole rollout
        # must land in the sidecar, not crash the loop.
        service = scripted_service(
            traj([fn_call("c1", "increment_counter", '{"count": 1}')]),
            FakeRemoteResponse(500, b"died mid-rollout"),
        )
        agent, _, _ = make_wired_agent(monkeypatch, service, tool_handler=TestAgentToolLoop.counter_tool_handler)
        result = (
            await agent.run(make_request(), RemoteAgentRunRequest.model_validate(make_row(tools=_COUNTER_TOOLS)))
        ).model_dump()
        assert result[NG_FAILURE_CLASS_KEY] == REMOTE_AGENT_FAILURE_CLASS
        assert "died mid-rollout" in result["error"]


class TestGymSideFailuresBecomeSentinelRows:
    async def test_seed_failure_skips_remote_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        service = scripted_service(_MINIMAL_TRAJECTORY)
        agent, client, _ = make_wired_agent(monkeypatch, service, seed_status=500)

        result = (await agent.run(make_request(), RemoteAgentRunRequest.model_validate(make_row()))).model_dump()

        assert result[NG_FAILURE_CLASS_KEY] == REMOTE_AGENT_FAILURE_CLASS
        assert "/seed_session" in result["error"]
        assert client.request.call_count == 0

    async def test_verify_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        agent, _, _ = make_wired_agent(monkeypatch, scripted_service(_MINIMAL_TRAJECTORY), verify_status=500)

        result = (await agent.run(make_request(), RemoteAgentRunRequest.model_validate(make_row()))).model_dump()

        assert result[NG_FAILURE_CLASS_KEY] == REMOTE_AGENT_FAILURE_CLASS
        assert "/verify" in result["error"]
        assert result["reward"] == 0.0

    async def test_skills_ref_warns_and_continues(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        agent, _, _ = make_wired_agent(monkeypatch, scripted_service(_MINIMAL_TRAJECTORY))

        row = make_row(skills_ref={"path": "/skills", "hash": "abc", "skills": []})
        result = await agent.run(make_request(), RemoteAgentRunRequest.model_validate(row))

        assert NG_FAILURE_CLASS_KEY not in result.model_dump()
        assert "skills_ref" in capsys.readouterr().out


class TestResponseQualityWarnings:
    async def test_missing_usage_warns(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        agent, _, _ = make_wired_agent(monkeypatch, scripted_service(traj([msg("hi")], usage=None)))
        result = await agent.run(make_request(), RemoteAgentRunRequest.model_validate(make_row()))
        assert NG_FAILURE_CLASS_KEY not in result.model_dump()
        assert "no usage" in capsys.readouterr().out

    async def test_clean_trajectory_no_warnings(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        agent, _, _ = make_wired_agent(monkeypatch, scripted_service(_MINIMAL_TRAJECTORY))
        await agent.run(make_request(), RemoteAgentRunRequest.model_validate(make_row()))
        assert "WARNING" not in capsys.readouterr().out

    async def test_quality_warnings_are_throttled(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        agent, _, _ = make_wired_agent(monkeypatch, scripted_service(traj([msg("hi")], usage=None)))

        for _ in range(10):
            await agent.run(make_request(), RemoteAgentRunRequest.model_validate(make_row()))

        # Head of 5, then every 100th: 10 rollouts -> exactly 5 printed warnings
        assert capsys.readouterr().out.count("no usage") == 5


class TestRunTimeoutAndSemaphore:
    async def test_run_wallclock_bound_becomes_sentinel(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def slow_request(*args, **kwargs):
            await asyncio.sleep(30)

        agent, _, _ = make_wired_agent(monkeypatch, AsyncMock(side_effect=slow_request), run_timeout_secs=0.05)

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

        agent, _, _ = make_wired_agent(monkeypatch, AsyncMock(side_effect=gated_request), concurrency=2)

        rows = [RemoteAgentRunRequest.model_validate(make_row()) for _ in range(4)]
        tasks = [asyncio.create_task(agent.run(make_request(), row)) for row in rows]
        await asyncio.sleep(0.05)
        assert max_in_flight == 2
        release.set()
        results = await asyncio.gather(*tasks)

        assert all(r.model_dump()[NG_FAILURE_CLASS_KEY] == REMOTE_AGENT_FAILURE_CLASS for r in results)
        assert agent.sem._value == 2  # every permit released despite 4 failures

    async def test_queue_wait_does_not_count_against_run_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        release_first = asyncio.Event()
        first_seen = asyncio.Event()
        call_count = 0

        async def gated_request(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                first_seen.set()
                await release_first.wait()
            return FakeRemoteResponse(200, orjson.dumps(_MINIMAL_TRAJECTORY))

        agent, _, _ = make_wired_agent(
            monkeypatch, AsyncMock(side_effect=gated_request), concurrency=1, run_timeout_secs=0.5
        )

        first = asyncio.create_task(agent.run(make_request(), RemoteAgentRunRequest.model_validate(make_row())))
        second = asyncio.create_task(agent.run(make_request(), RemoteAgentRunRequest.model_validate(make_row())))
        await first_seen.wait()
        # Hold the only permit for most of the second task's would-be budget
        await asyncio.sleep(0.4)
        release_first.set()
        results = [r.model_dump() for r in await asyncio.gather(first, second)]

        # If queue wait counted against run_timeout_secs, the second task would time out
        assert all(NG_FAILURE_CLASS_KEY not in r for r in results)


class TestRoutes:
    def _client(self, monkeypatch, request_mock=None):
        from fastapi.testclient import TestClient

        agent, _, _ = make_wired_agent(monkeypatch, request_mock or scripted_service(_MINIMAL_TRAJECTORY))
        return TestClient(agent.setup_webserver(), raise_server_exceptions=False)

    def test_run_route_happy_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = self._client(monkeypatch)
        response = client.post("/run", json=make_row())
        assert response.status_code == 200
        assert response.json()["reward"] == 1.0

    def test_run_route_failure_serializes_sentinel_with_http_200(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The sentinel body must survive FastAPI response-model serialization: a 500 here
        # would abort the entire collection run instead of routing to the failures sidecar.
        client = self._client(monkeypatch, AsyncMock(side_effect=RuntimeError("remote exploded")))
        response = client.post("/run", json=make_row())
        assert response.status_code == 200
        body = response.json()
        assert body[NG_FAILURE_CLASS_KEY] == REMOTE_AGENT_FAILURE_CLASS
        assert body["reward"] == 0.0
        assert body["response"]["output"][0]["type"] == "message"

    def test_responses_route_is_live(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # /v1/responses is a real route now: create-params in, finished trajectory out.
        client = self._client(monkeypatch)
        response = client.post("/v1/responses", json={"input": [{"role": "user", "content": "hi"}]})
        assert response.status_code == 200
        assert response.json()["id"] == "traj_1"


class TestStatefulToolsEndToEnd:
    """The full session contract, in-process: RemoteAgent seeds the real counter environment,
    the service asks for tools via unpaired function_calls, GYM executes them against the
    counter server on the seeded session, and verify() scores the mutated state."""

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

    async def test_counter_env_reward_through_gym_executed_tools(self, monkeypatch: pytest.MonkeyPatch) -> None:
        counter = self._counter_client()

        # The service never sees the counter server: it only returns asks and reads outputs.
        def turn2(received):
            return traj([fn_call("c3", "get_counter_value", "{}")])

        service = scripted_service(
            traj(
                [
                    fn_call("c1", "increment_counter", '{"count": 1}'),
                    fn_call("c2", "increment_counter", '{"count": 2}'),
                ]
            ),
            traj([fn_call("c3", "get_counter_value", "{}")]),
            # Final turn: read the count Gym fed back and answer with it
            traj([msg("final count is 6")]),
        )

        agent = make_agent()

        async def gym_post(server_name, url_path, json=None, cookies=None, **kwargs):
            if url_path.endswith("/v1/responses"):
                wire = json.model_dump(exclude_unset=True) if isinstance(json, BaseModel) else json
                params = NeMoGymResponseCreateParamsNonStreaming.model_validate(wire)
                fastapi_response = Response()
                try:
                    result = await agent.responses(make_request(dict(cookies or {})), fastapi_response, params)
                except Exception as e:  # noqa: BLE001
                    return FakeServerClientResponse({"error": repr(e)}, status=500)
                out_cookies = SimpleCookie()
                for header_value in fastapi_response.headers.getlist("set-cookie"):
                    out_cookies.load(header_value)
                return FakeServerClientResponse(
                    result.model_dump(mode="json"), cookies={k: m.value for k, m in out_cookies.items()}
                )
            # Everything else — seed, tools, verify — hits the REAL counter server
            response = counter.post(url_path, json=json, cookies=dict(cookies or {}))
            return FakeServerClientResponse(
                response.json(), cookies=dict(response.cookies), status=response.status_code
            )

        server_client = MagicMock(spec=ServerClient)
        server_client.post = AsyncMock(side_effect=gym_post)
        agent.server_client = server_client
        mock_remote(monkeypatch, service)

        row = {
            "responses_create_params": {
                "input": [{"role": "user", "content": "add 1 then add 2 then get the count"}],
                "tools": _COUNTER_TOOLS,
            },
            "initial_count": 3,
            "expected_count": 6,
        }

        result = await agent.run(make_request(), RemoteAgentRunRequest.model_validate(row))

        dumped = result.model_dump()
        assert NG_FAILURE_CLASS_KEY not in dumped
        # Reward 1.0 only if seed, both Gym-executed tool calls, and verify shared ONE session
        assert dumped["reward"] == 1.0
        # The service really was fed the counter value Gym read back
        third_input = service.received[2]["payload"]["input"]
        assert any('"count":6' in i.get("output", "") for i in third_input if isinstance(i, dict))


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

        async def remote_service(method, url, data=None, headers=None, cookies=None, **kwargs):
            params = orjson.loads(data)
            if "fail" in params["input"][0]["content"]:
                return FakeRemoteResponse(500, b"remote exploded")
            return FakeRemoteResponse(200, orjson.dumps(_MINIMAL_TRAJECTORY))

        agent, _, _ = make_wired_agent(monkeypatch, AsyncMock(side_effect=remote_service))
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
        agent, _, _ = make_wired_agent(monkeypatch, AsyncMock(side_effect=RuntimeError("remote exploded")))

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

        agent, _, _ = make_wired_agent(monkeypatch, AsyncMock(side_effect=RuntimeError("remote exploded")))
        client = TestClient(agent.setup_webserver(), raise_server_exceptions=False)

        response = client.post("/run", json=make_row(**self._REUSED_ROW_EXTRAS))

        assert response.status_code == 200
        assert response.json()[NG_FAILURE_CLASS_KEY] == REMOTE_AGENT_FAILURE_CLASS

    async def test_happy_path_reused_row_leaks_no_stale_sentinels(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Stale routing keys on an input row must not echo through verify and misroute a
        # SUCCESS into the failures sidecar.
        agent, _, _ = make_wired_agent(monkeypatch, scripted_service(_MINIMAL_TRAJECTORY))

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

        with pytest.raises(asyncio.TimeoutError):
            await agent.aggregate_metrics(
                __import__(
                    "nemo_gym.base_resources_server", fromlist=["AggregateMetricsRequest"]
                ).AggregateMetricsRequest(verify_responses=[])
            )
