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
"""Exercise the real Gym HTTP client against local provider and agent sockets."""

import asyncio
import json
from contextlib import AsyncExitStack

import pytest
from aiohttp import ClientError, web

from environments.skill_trigger.budget import TrialBudget
from environments.skill_trigger.gateway import InferenceGateway, response_outcome


@pytest.fixture
async def network(tmp_path, monkeypatch):
    from nemo_gym import server_utils

    # Use Gym's actual client and retry implementation. Isolate only its global
    # lifecycle between pytest event loops, not its HTTP behavior.
    monkeypatch.setattr(server_utils, "_GLOBAL_AIOHTTP_CLIENT", None)
    client = server_utils.set_global_aiohttp_client(server_utils.GlobalAIOHTTPAsyncClientConfig())
    stack = AsyncExitStack()
    providers = []

    async def start(handler, *, clock=None, resolver=None):
        provider = web.Application()
        provider.router.add_post("/v1/{endpoint:.*}", handler)
        runner = web.AppRunner(provider, access_log=None, shutdown_timeout=0.1)
        await runner.setup()
        stack.push_async_callback(runner.cleanup)
        await web.TCPSite(runner, "127.0.0.1", 0).start()
        port = runner.addresses[0][1]
        providers.append(port)
        options = {"clock": clock} if clock else {}
        budget = TrialBudget(model="test-model", root_session_id="root", preserve_payload=False, **options)

        async def root(payload):
            return "root"

        gateway = InferenceGateway(
            budget=budget,
            upstream_url=f"http://127.0.0.1:{port}/v1",
            api_key="test-provider-secret",
            evidence_dir=tmp_path / f"wire-{port}",
            resolve_session=resolver or root,
        )
        await gateway.start(bind_host="127.0.0.1")
        stack.push_async_callback(gateway.close)
        budget.release()
        return gateway

    async def send(gateway, *, endpoint="responses", token=None, payload=None):
        response = await server_utils.request(
            "POST",
            f"http://127.0.0.1:{gateway.port}/v1/{endpoint}",
            json={"model": "test-model", "input": "original prompt"} if payload is None else payload,
            headers={"Authorization": "Bearer " + (gateway.token if token is None else token)},
            _max_num_tries=1,
            _max_connection_retries=0,
            _resend_after_disconnect=False,
        )
        async with response:
            return response.status, await response.read()

    try:
        yield start, send
    finally:
        await stack.aclose()
        await client.close()


async def test_three_requests_forward_fourth_is_preserved_but_blocked(network):
    start, send = network
    received = []

    async def provider(request):
        received.append(await request.json())
        assert request.headers["Authorization"] == "Bearer test-provider-secret"
        return web.json_response({"status": "completed", "output": []})

    gateway = await start(provider)
    for _ in range(3):
        assert (await send(gateway))[0] == 200
    assert (await send(gateway))[0] == 409
    assert len(received) == len(gateway.budget.requests) == 3
    assert all(p["input"] == "original prompt" and p["max_output_tokens"] == 8192 for p in received)
    assert gateway.outcome == "routing_horizon"
    assert (gateway.evidence_dir / "004/request-original.json").is_file()
    assert not (gateway.evidence_dir / "004/request-sent.json").exists()
    metadata = json.loads((gateway.evidence_dir / "004/metadata.json").read_text())
    assert metadata["forwarded"] is False
    assert all("test-provider-secret" not in p.read_text() for p in gateway.evidence_dir.rglob("*") if p.is_file())


async def test_auth_and_endpoint_validation_prevent_forwarding(network):
    start, send = network
    calls = []

    async def provider(request):
        calls.append(request.path)
        return web.json_response({"status": "completed"})

    gateway = await start(provider)
    assert (await send(gateway, token="wrong"))[0] == 403
    assert (await send(gateway, endpoint="files"))[0] == 404
    assert calls == []
    assert list(gateway.evidence_dir.iterdir()) == []


@pytest.mark.parametrize("status", [429, 500, 307])
async def test_provider_errors_and_redirects_are_recorded_without_retry(network, status):
    start, send = network
    calls = []

    async def provider(request):
        calls.append(request.path)
        return web.json_response({"error": "unavailable"}, status=status, headers={"Retry-After": "5"})

    gateway = await start(provider)
    response_status, body = await send(gateway)
    assert response_status == status
    assert json.loads(body) == {"error": "unavailable"}
    assert len(calls) == 1
    assert gateway.outcome == "provider_error"
    meta = json.loads((gateway.evidence_dir / "001/metadata.json").read_text())
    assert meta["http_status"] == status
    assert meta["provider_metadata"]["Retry-After"] == "5"


@pytest.mark.parametrize("complete", [True, False])
async def test_stream_bytes_are_preserved_and_truncation_is_unscored(network, complete):
    start, send = network
    raw = b'data: {"type":"response.output_text.delta","delta":"instructions"}\n\n'
    if complete:
        raw += b'data: {"type":"response.completed","response":{"status":"completed"}}\n\n'

    async def provider(request):
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        for index in range(0, len(raw), 7):
            await response.write(raw[index : index + 7])
        await response.write_eof()
        return response

    gateway = await start(provider)
    status, body = await send(gateway, payload={"model": "test-model", "stream": True})
    assert status == 200 and body == raw
    assert (gateway.evidence_dir / "001/response.body").read_bytes() == raw
    assert gateway.outcome == (None if complete else "trace_error")


async def test_live_stream_cannot_extend_absolute_deadline(network):
    start, send = network
    now = [0.0]
    disconnected = asyncio.Event()

    async def provider(request):
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        try:
            while True:
                await response.write(b'data: {"type":"response.output_text.delta","delta":"x"}\n\n')
                await asyncio.sleep(0.005)
        except (ConnectionError, asyncio.CancelledError):
            disconnected.set()
        return response

    gateway = await start(provider, clock=lambda: now[0])
    now[0] = 119.95
    with pytest.raises(ClientError):
        await send(gateway, payload={"model": "test-model", "stream": True})
    await asyncio.wait_for(disconnected.wait(), 1)
    assert gateway.outcome == "timeout"
    assert len(gateway.budget.requests) == 1
    assert (gateway.evidence_dir / "001/response.body").stat().st_size > 0


async def test_session_attribution_wait_uses_same_deadline(network):
    start, send = network
    now = [0.0]

    async def provider(request):
        pytest.fail("unproven session reached provider")

    async def unresolved(payload):
        await asyncio.Future()

    gateway = await start(provider, clock=lambda: now[0], resolver=unresolved)
    now[0] = 119.98
    assert (await send(gateway))[0] == 409
    assert gateway.outcome == "timeout"
    assert gateway.budget.requests == []


@pytest.mark.parametrize("payload", [[], {"model": "other-model"}])
async def test_invalid_payload_cannot_become_a_scored_miss(network, payload):
    start, send = network

    async def provider(request):
        pytest.fail("invalid request forwarded")

    gateway = await start(provider)
    assert (await send(gateway, payload=payload))[0] == 500
    assert gateway.outcome == "trace_error"
    assert gateway.budget.requests == []


@pytest.mark.parametrize(
    "body,outcome",
    [
        ({"choices": [{"finish_reason": "stop"}]}, None),
        ({"choices": [{"finish_reason": "length"}]}, "output_limit"),
        ({"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}}, "output_limit"),
        ({"status": "incomplete", "incomplete_details": {"reason": "content_filter"}}, "provider_error"),
        ({"error": {"message": "failed"}}, "provider_error"),
        ({"choices": [{"finish_reason": None}]}, "trace_error"),
    ],
)
def test_provider_terminal_outcomes(body, outcome):
    assert response_outcome(json.dumps(body).encode(), streaming=False) == outcome
