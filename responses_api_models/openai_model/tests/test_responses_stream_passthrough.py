# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in relay of streaming Responses calls (for the Codex CLI's own request shape)."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import ClientResponseError, web
from aiohttp.test_utils import TestServer
from fastapi.testclient import TestClient

from nemo_gym.client import ServerClient, http_client
from nemo_gym.models import CaptureStore
from nemo_gym.rollouts import read_model_call_records
from responses_api_models.openai_model.app import (
    NeMoGymAsyncOpenAI,
    SimpleModelServer,
    SimpleModelServerConfig,
)


# A Codex 0.155.1 request for a catalog model. The typed path accepts its `max` effort (OpenAI SDK 2.45.0), but it
# drops the code-mode tools shipped in the `additional_tools` item, which only the relay delivers.
CODEX_REQUEST = {
    "model": "gpt-6-astra",
    "stream": True,
    "store": False,
    "include": ["reasoning.encrypted_content"],
    "reasoning": {"effort": "max", "context": "all_turns"},
    "text": {"verbosity": "low"},
    "tools": [],
    "prompt_cache_key": "thread-1",
    "input": [
        {
            "type": "additional_tools",
            "role": "developer",
            "tools": [
                {
                    "type": "namespace",
                    "name": "functions",
                    "description": "",
                    "tools": [{"type": "custom", "name": "exec", "description": "Run JavaScript"}],
                }
            ],
        },
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Solve the task."}]},
    ],
}
SSE = [
    b'event: response.created\ndata: {"type":"response.created","response":{"id":"resp_1"}}\n\n',
    b'event: response.output_item.done\ndata: {"type":"response.output_item.done","output_index":0,',
    b'"item":{"type":"custom_tool_call","name":"exec","call_id":"c1","input":"1"}}\n\n',
    b'event: response.completed\ndata: {"type":"response.completed","response":{"id":"resp_1",'
    b'"status":"completed","output":[],"usage":{"input_tokens":5,"output_tokens":1,"total_tokens":6}}}\n\n',
]


class FakeUpstream:
    def __init__(self, chunks):
        self.status = 200
        self.headers = {"Content-Type": "text/event-stream; charset=utf-8"}
        self.released = False

        async def iter_any():
            for chunk in chunks:
                yield chunk

        self.content = MagicMock(iter_any=iter_any)

    def release(self):
        self.released = True


def make_server(tmp_path=None, **config) -> SimpleModelServer:
    server = SimpleModelServer(
        config=SimpleModelServerConfig(
            host="0.0.0.0",
            port=8081,
            openai_base_url="https://inference.example/v1",
            openai_api_key="dummy_key",  # pragma: allowlist secret
            openai_model="openai/openai/gpt-6-astra",
            entrypoint="",
            name="policy_model",
            **config,
        ),
        server_client=MagicMock(spec=ServerClient, global_config_dict={}),
    )
    if tmp_path is not None:
        server.server_client.global_config_dict = {
            "observability_enabled": True,
            "model_call_capture_dir": str(tmp_path),
        }
    server._client = MagicMock(spec=NeMoGymAsyncOpenAI)
    return server


def test_relays_the_codex_request_and_stream_unchanged_except_the_model(tmp_path) -> None:
    server = make_server(
        tmp_path, responses_stream_passthrough=True, extra_body={"service_tier": "flex", "store": True}
    )
    upstream = FakeUpstream(SSE)
    server._client.create_response_stream = AsyncMock(return_value=upstream)
    with TestClient(server.setup_webserver()) as client:
        response = client.post("/ng-rollout/rollout-1/v1/responses", json=CODEX_REQUEST)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.content == b"".join(SSE)
    sent = server._client.create_response_stream.await_args.kwargs
    # extra_body fills absent fields; the request's own fields win, as on the typed path.
    assert sent == CODEX_REQUEST | {"model": "openai/openai/gpt-6-astra", "service_tier": "flex"}
    server._client.create_response.assert_not_called()
    assert upstream.released
    calls = read_model_call_records(CaptureStore(tmp_path), "rollout-1")
    assert len(calls) == 1 and calls[0].status_code == 200 and calls[0].error_category is None
    captured = json.loads(next(tmp_path.glob("rollout-1.capture.jsonl")).read_text().splitlines()[0])
    assert '"effort":"max"' in json.dumps(captured, separators=(",", ":"))


def test_non_streaming_requests_keep_the_typed_path() -> None:
    server = make_server(responses_stream_passthrough=True)
    server._client.create_response = AsyncMock(
        return_value={
            "id": "resp_1",
            "created_at": 0,
            "model": "openai/openai/gpt-6-astra",
            "object": "response",
            "output": [],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        }
    )
    server._client.create_response_stream = AsyncMock()
    with TestClient(server.setup_webserver()) as client:
        response = client.post("/v1/responses", json={"input": "hello"})
    assert response.status_code == 200
    server._client.create_response.assert_awaited_once()
    server._client.create_response_stream.assert_not_called()


def test_without_the_opt_in_streaming_requests_are_sanitized_and_typed() -> None:
    server = make_server()
    server._client.create_response_stream = AsyncMock()
    server._client.create_response = AsyncMock(
        return_value={
            "id": "resp_1",
            "created_at": 0,
            "model": "openai/openai/gpt-6-astra",
            "object": "response",
            "output": [],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        }
    )
    with TestClient(server.setup_webserver()) as client:
        response = client.post("/v1/responses", json=CODEX_REQUEST)
    # The typed path makes a non-streaming call and synthesizes the stream from its result.
    assert response.status_code == 200
    server._client.create_response_stream.assert_not_called()
    server._client.create_response.assert_awaited_once()
    sent = server._client.create_response.await_args.kwargs
    # The sanitizer drops the code-mode tools of the additional_tools item, which only the relay delivers.
    assert [item["type"] for item in sent["input"]] == ["message"]
    assert sent["reasoning"]["effort"] == "max"


@pytest.mark.parametrize("propagated,expected", [([400], 400), ([], 500)])
def test_upstream_errors_before_the_stream_keep_the_existing_policy(propagated, expected) -> None:
    server = make_server(responses_stream_passthrough=True, propagate_upstream_http_status_codes=propagated)
    error = ClientResponseError(MagicMock(), (), status=400, message="context_length_exceeded")
    error.response_content = b'{"error":{"code":"context_length_exceeded"}}'
    server._client.create_response_stream = AsyncMock(side_effect=error)
    app = server.setup_webserver()
    server.setup_exception_middleware(app)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/v1/responses", json=CODEX_REQUEST)
    assert response.status_code == expected
    if expected == 400:
        # As on the typed path: the provider's own error body.
        assert response.json() == {"error": {"code": "context_length_exceeded"}}


async def test_relay_holds_the_provider_slot_until_the_stream_ends() -> None:
    server = make_server(responses_stream_passthrough=True, max_concurrent_requests=1)
    upstream = FakeUpstream(SSE)
    server._client.create_response_stream = AsyncMock(return_value=upstream)
    response = await server._relay_responses_stream(dict(CODEX_REQUEST))
    assert server._semaphore.locked()
    chunks = [chunk async for chunk in response.body_iterator]
    assert b"".join(chunks) == b"".join(SSE)
    assert not server._semaphore.locked()
    assert upstream.released


@pytest.mark.parametrize(
    "config",
    [
        {"upstream_retry_policy": {"max_attempts": 2}, "upstream_max_num_tries": 1},
        {"drop_input_reasoning_items": True},
    ],
)
def test_passthrough_rejects_settings_it_cannot_apply(config) -> None:
    with pytest.raises(ValueError, match="relays requests as sent"):
        make_server(responses_stream_passthrough=True, **config)


@pytest.mark.parametrize("jitter,expected_sleeps", [([1, 2], [1.5]), (None, [])])
async def test_relay_waits_out_the_pre_request_jitter_before_taking_a_slot(
    monkeypatch, jitter, expected_sleeps
) -> None:
    # A one-attempt policy's jitter delays every provider attempt, the relayed one included, and the wait happens
    # outside the provider slot so it does not hold capacity.
    policy = {"max_attempts": 1} if jitter is None else {"max_attempts": 1, "pre_request_jitter_seconds": jitter}
    server = make_server(responses_stream_passthrough=True, max_concurrent_requests=1, upstream_retry_policy=policy)
    events = []

    async def fake_sleep(seconds):
        events.append(("sleep", seconds, server._semaphore.locked()))

    async def create_response_stream(**payload):
        events.append(("provider", None, server._semaphore.locked()))
        return FakeUpstream(SSE)

    monkeypatch.setattr("responses_api_models.openai_model.app.asyncio.sleep", fake_sleep)
    monkeypatch.setattr("responses_api_models.openai_model.app.random.uniform", lambda low, high: (low + high) / 2)
    server._client.create_response_stream = create_response_stream

    response = await server._relay_responses_stream(dict(CODEX_REQUEST))
    assert b"".join([chunk async for chunk in response.body_iterator]) == b"".join(SSE)

    assert events == [("sleep", seconds, False) for seconds in expected_sleeps] + [("provider", None, True)]


@pytest.fixture
async def gym_http_client():
    """A fresh shared aiohttp client, as a running Gym server has, for calls to a real upstream."""
    saved = (http_client._GLOBAL_AIOHTTP_CLIENT, http_client._GLOBAL_AIOHTTP_CONTROL_CLIENT)
    http_client._GLOBAL_AIOHTTP_CLIENT = http_client._GLOBAL_AIOHTTP_CONTROL_CLIENT = None
    http_client.set_global_aiohttp_client(http_client.GlobalAIOHTTPAsyncClientConfig())
    try:
        yield
    finally:
        await http_client._GLOBAL_AIOHTTP_CLIENT.close()
        http_client._GLOBAL_AIOHTTP_CLIENT, http_client._GLOBAL_AIOHTTP_CONTROL_CLIENT = saved


async def test_request_timeout_bounds_each_read_of_a_relayed_stream_not_the_whole_stream(gym_http_client) -> None:
    # A passthrough stream lasts as long as the model's answer, which can take longer than the
    # default 30-minute limit, so upstream_request_timeout_seconds bounds each read instead.
    async def stream(request: web.Request) -> web.StreamResponse:
        gaps = (await request.json())["gaps"]
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        for index, gap in enumerate(gaps):
            await asyncio.sleep(gap)
            await response.write(f"data: {index}\n\n".encode())
        await response.write_eof()
        return response

    application = web.Application()
    application.router.add_post("/v1/responses", stream)
    async with TestServer(application) as upstream:
        client = NeMoGymAsyncOpenAI(
            base_url=str(upstream.make_url("/v1")),
            api_key="dummy_key",  # pragma: allowlist secret
            request_timeout_seconds=0.5,
        )
        # Six reads 0.15 s apart: 0.9 s in all, beyond the 0.5-second limit.
        live = await client.create_response_stream(gaps=[0.15] * 6)
        try:
            body = b"".join([chunk async for chunk in live.content.iter_any()])
        finally:
            live.release()
        assert body == b"".join(f"data: {index}\n\n".encode() for index in range(6))
        # A read that waits longer than the limit still ends the stream.
        stalled = await client.create_response_stream(gaps=[0.0, 1.5])
        try:
            with pytest.raises(asyncio.TimeoutError):
                async for _ in stalled.content.iter_any():
                    pass
        finally:
            stalled.release()


@pytest.mark.parametrize("request_timeout_seconds", [0.5, None])
@pytest.mark.parametrize("debug_enabled", [True, False])
async def test_an_upstream_error_keeps_its_status_and_body_with_request_debug_on_or_off(
    gym_http_client, monkeypatch, capsys, debug_enabled, request_timeout_seconds
) -> None:
    # With request debugging on, the diagnostic print of the request must not replace the provider's error, even
    # though a relayed stream's request carries its own ClientTimeout.
    monkeypatch.setattr(http_client, "_GLOBAL_AIOHTTP_CLIENT_REQUEST_DEBUG", debug_enabled)
    error_body = b'{"error":{"code":"context_length_exceeded"}}'

    async def reject(request: web.Request) -> web.Response:
        return web.Response(status=400, body=error_body, content_type="application/json")

    application = web.Application()
    application.router.add_post("/v1/responses", reject)
    async with TestServer(application) as upstream:
        client = NeMoGymAsyncOpenAI(
            base_url=str(upstream.make_url("/v1")),
            api_key="dummy_key",  # pragma: allowlist secret
            request_timeout_seconds=request_timeout_seconds,
        )
        with pytest.raises(ClientResponseError) as raised:
            await client.create_response_stream(**CODEX_REQUEST)

    assert raised.value.status == 400
    assert raised.value.response_content == error_body
    assert ("Request kwargs:" in capsys.readouterr().out) is debug_enabled
