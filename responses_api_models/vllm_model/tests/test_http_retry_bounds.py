# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import time
from unittest.mock import MagicMock

import pytest
from aiohttp import ClientResponseError, ClientSession, web
from pydantic import ValidationError

import nemo_gym.client.http_client as transport
from nemo_gym.client import ServerClient
from nemo_gym.openai_utils import MAX_NUM_TRIES
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig


def config(**overrides):
    return VLLMModelConfig(
        **(
            dict(
                host="127.0.0.1",
                port=8081,
                base_url="http://placeholder.invalid/v1",
                api_key="test",
                model="test",
                entrypoint="",
                name="test",
                return_token_id_information=False,
                uses_reasoning_parser=False,
            )
            | overrides
        )
    )


@pytest.mark.parametrize("rebind", [False, True])
@pytest.mark.parametrize("failure,explicit_transport_bound", [("503", False), ("503", True), ("timeout", True)])
async def test_one_actual_http_attempt(tmp_path, monkeypatch, rebind, failure, explicit_transport_bound):
    received = []

    async def handle(request):
        received.append(await request.json())
        if failure == "timeout":
            await asyncio.sleep(1)
        return web.json_response(
            {"error": {"message": "controlled unavailable"}}, status=503, headers={"Retry-After": "60"}
        )

    app = web.Application()
    app.router.add_post("/v1/chat/completions", handle)
    runner = web.AppRunner(app, shutdown_timeout=0.01)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}/v1"
    endpoint = tmp_path / "endpoint.txt"
    overrides = dict(
        base_url="http://placeholder.invalid/v1" if rebind else url,
        endpoint_file=str(endpoint) if rebind else None,
        max_http_attempts=1,
        max_num_tries=1 if explicit_transport_bound else None,
        request_timeout_seconds=0.04,
    )
    server = VLLMModel(config=config(**overrides), server_client=MagicMock(spec=ServerClient))
    if rebind:
        endpoint.write_text(url)
        server._maybe_rebind_endpoint()
    client = server._clients[0]
    assert client.base_url == url
    assert client.max_rate_limit_wait_seconds == 0
    try:
        async with ClientSession() as session:
            monkeypatch.setattr(transport, "get_global_aiohttp_client", lambda: session)
            started = time.monotonic()
            expected = TimeoutError if failure == "timeout" else ClientResponseError
            with pytest.raises(expected):
                await client.create_chat_completion(model="test", messages=[{"role": "user", "content": "test"}])
            elapsed = time.monotonic() - started
            assert len(received) == 1
            assert elapsed < 0.3
            assert client.max_http_attempts == 1
            assert client.max_num_tries == overrides["max_num_tries"]
    finally:
        await runner.cleanup()


def test_default_retry_controls_are_unchanged(tmp_path):
    endpoint = tmp_path / "endpoint.txt"
    server = VLLMModel(config=config(endpoint_file=str(endpoint)), server_client=MagicMock(spec=ServerClient))
    initial = server._clients[0]
    endpoint.write_text("http://new.invalid/v1")
    server._maybe_rebind_endpoint()
    for client in (initial, server._clients[0]):
        assert client.max_http_attempts == MAX_NUM_TRIES
        assert client.max_num_tries is None
        assert client.max_rate_limit_wait_seconds == 0
        assert client.max_connection_retries == 8
    static = VLLMModel(config=config(), server_client=MagicMock(spec=ServerClient))._clients[0]
    assert static.max_connection_retries is None


@pytest.mark.parametrize(
    "setting,value", [("max_http_attempts", 0), ("max_http_attempts", -1), ("max_num_tries", 0), ("max_num_tries", 2)]
)
def test_invalid_retry_controls_are_rejected(setting, value):
    with pytest.raises(ValidationError):
        config(**{setting: value})
