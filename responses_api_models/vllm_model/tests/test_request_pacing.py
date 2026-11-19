# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
from contextlib import asynccontextmanager
from time import monotonic
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import ClientResponseError, ClientSession, web
from pydantic import ValidationError
from starlette.requests import Request

import nemo_gym.client.http_client as transport
from nemo_gym.client import ServerClient
from nemo_gym.core import NeMoGymChatCompletionCreateParamsNonStreaming, NeMoGymResponseCreateParamsNonStreaming
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig


INTERVAL = 0.06


def config(**settings):
    return VLLMModelConfig(
        **(
            dict(
                host="127.0.0.1",
                port=8081,
                name="test",
                entrypoint="app.py",
                base_url="http://fixture.invalid/v1",
                api_key="test",
                model="test",
                return_token_id_information=False,
                uses_reasoning_parser=False,
                max_http_attempts=1,
                max_num_tries=1,
                minimum_request_interval_seconds=INTERVAL,
            )
            | settings
        )
    )


def chat_reply():
    return {
        "id": "test",
        "object": "chat.completion",
        "created": 0,
        "model": "test",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "final"}}],
    }


def reply(request):
    if request.path.endswith("/responses"):
        return {
            "id": "test",
            "object": "response",
            "created_at": 0,
            "model": "test",
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "output": [
                {
                    "id": "message",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "final", "annotations": []}],
                }
            ],
        }
    if request.path.endswith("/completions") and not request.path.endswith("/chat/completions"):
        return {
            "id": "test",
            "object": "text_completion",
            "created": 0,
            "model": "test",
            "choices": [{"index": 0, "text": "final", "finish_reason": "stop", "logprobs": None}],
        }
    return chat_reply()


@asynccontextmanager
async def backend(monkeypatch, handler, **settings):
    app = web.Application()
    for path in ("/v1/chat/completions", "/v1/responses", "/v1/completions"):
        app.router.add_post(path, handler)
    runner = web.AppRunner(app, shutdown_timeout=0.01)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}/v1"
    try:
        async with ClientSession() as session:
            monkeypatch.setattr(transport, "get_global_aiohttp_client", lambda: session)
            yield VLLMModel(config=config(base_url=url, **settings), server_client=MagicMock(spec=ServerClient))
    finally:
        await runner.cleanup()


def make_request():
    return Request(
        {
            "type": "http",
            "path": "/v1/responses",
            "scheme": "http",
            "server": ("test", 80),
            "headers": [],
            "session": {"session_id": "paced-test"},
        }
    )


async def send(server, text="test", *, request=None):
    request = request or make_request()
    if server.config.is_responses_native:
        return await server.responses(request, NeMoGymResponseCreateParamsNonStreaming(input=text))
    return await server.chat_completions(
        request, NeMoGymChatCompletionCreateParamsNonStreaming(messages=[{"role": "user", "content": text}])
    )


@pytest.mark.parametrize(
    "overrides", [{"max_http_attempts": 2}, {"max_num_tries": None}, {"max_http_attempts": 2, "max_num_tries": None}]
)
def test_pacing_requires_explicit_single_attempt_transport(overrides):
    with pytest.raises(ValidationError, match="single|pacing"):
        config(**overrides)


@pytest.mark.parametrize("options", [{}, {"is_responses_native": True}, {"use_completions_api": True}])
async def test_pacing_covers_each_upstream_api(monkeypatch, options):
    starts = []

    async def handle(request):
        starts.append(monotonic())
        return web.json_response(reply(request))

    async with backend(monkeypatch, handle, **options) as server:
        await send(server)
        await send(server)
    assert len(starts) == 2
    assert starts[1] - starts[0] >= INTERVAL * 0.9


@pytest.mark.parametrize("options", [{}, {"is_responses_native": True}, {"use_completions_api": True}])
async def test_default_unpaced_calls_remain_concurrent(monkeypatch, options):
    entered = asyncio.Event()
    release = asyncio.Event()
    arrivals = []

    async def handle(request):
        arrivals.append(request.path)
        if len(arrivals) == 1:
            entered.set()
            await release.wait()
        return web.json_response(reply(request))

    assert VLLMModelConfig.model_fields["minimum_request_interval_seconds"].default == 0
    async with backend(monkeypatch, handle, minimum_request_interval_seconds=0, **options) as server:
        first = asyncio.create_task(send(server))
        try:
            await asyncio.wait_for(entered.wait(), 2)
            await asyncio.wait_for(send(server), 2)
            assert len(arrivals) == 2
            assert not first.done()
        finally:
            release.set()
            await asyncio.wait_for(first, 2)


@pytest.mark.parametrize("options", [{}, {"is_responses_native": True}, {"use_completions_api": True}])
async def test_queued_call_resolves_rotated_endpoint_at_dispatch(monkeypatch, tmp_path, options):
    entered = asyncio.Event()
    queued = asyncio.Event()
    release = asyncio.Event()
    arrivals = []
    original_call = VLLMModel._call_endpoint
    dispatches = 0

    async def observe_queue(self, request, create_call, **kwargs):
        nonlocal dispatches
        dispatches += 1
        if dispatches == 2:
            queued.set()
        return await original_call(self, request, create_call, **kwargs)

    async def retired(request):
        arrivals.append("retired")
        if len(arrivals) > 1:
            return web.json_response({"error": {"message": "endpoint retired"}}, status=503)
        entered.set()
        await release.wait()
        return web.json_response(reply(request))

    async def replacement(request):
        arrivals.append("replacement")
        return web.json_response(reply(request))

    monkeypatch.setattr(VLLMModel, "_call_endpoint", observe_queue)
    endpoint_file = tmp_path / "endpoint.txt"
    async with backend(monkeypatch, retired, **options) as server:
        async with backend(monkeypatch, replacement, **options) as successor:
            endpoint_file.write_text(server.config.base_url[0])
            server.config.endpoint_file = str(endpoint_file)
            server.config.endpoint_check_interval_s = 0
            first = asyncio.create_task(send(server))
            tasks = [first]
            try:
                await asyncio.wait_for(entered.wait(), 2)
                second = asyncio.create_task(send(server))
                tasks.append(second)
                await asyncio.wait_for(queued.wait(), 2)
                assert arrivals == ["retired"]
                previous = endpoint_file.stat()
                endpoint_file.write_text(successor.config.base_url[0])
                # Publishers signal a new value with a new mtime, including on coarse filesystems.
                os.utime(endpoint_file, ns=(previous.st_atime_ns, previous.st_mtime_ns + 1_000_000_000))
                # Another caller can observe publication before the queued call is admitted.
                assert server._resolve_client(make_request()).base_url == successor.config.base_url[0]
                release.set()
                await asyncio.wait_for(asyncio.gather(*tasks), 2)
                assert arrivals == ["retired", "replacement"]
            finally:
                release.set()
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)


async def test_slow_payload_preparation_cannot_consume_the_interval(monkeypatch):
    starts = []
    prepare = VLLMModel._prepare_chat_request

    async def delayed(self, request, body):
        if body.messages[0]["content"] == "slow":
            await asyncio.sleep(INTERVAL * 3)
        return await prepare(self, request, body)

    async def handle(request):
        starts.append(monotonic())
        return web.json_response(chat_reply())

    monkeypatch.setattr(VLLMModel, "_prepare_chat_request", delayed)
    async with backend(monkeypatch, handle) as server:
        await send(server, "slow")
        await send(server, "fast")
    assert len(starts) == 2
    assert starts[1] - starts[0] >= INTERVAL * 0.9


async def test_status_failures_are_single_attempt_and_next_call_is_paced(monkeypatch):
    starts = []

    async def handle(request):
        starts.append(monotonic())
        return web.json_response(
            {"error": {"message": "controlled failure"}}, status=503, headers={"Retry-After": "0"}
        )

    async with backend(monkeypatch, handle) as server:
        for _ in range(2):
            with pytest.raises(ClientResponseError) as error:
                await send(server)
            assert error.value.status == 503
    assert len(starts) == 2
    assert starts[1] - starts[0] >= INTERVAL * 0.9


async def test_concurrent_calls_serialize_and_cancelled_waiter_does_not_dispatch(monkeypatch):
    starts = []
    entered = asyncio.Event()
    release = asyncio.Event()

    async def handle(request):
        starts.append(monotonic())
        if len(starts) == 1:
            entered.set()
            await release.wait()
        return web.json_response(chat_reply())

    async with backend(monkeypatch, handle) as server:
        first = asyncio.create_task(send(server))
        tasks = [first]
        try:
            await asyncio.wait_for(entered.wait(), 2)
            cancelled_request = make_request()
            cancelled = asyncio.create_task(send(server, request=cancelled_request))
            third = asyncio.create_task(send(server))
            tasks.extend([cancelled, third])
            await asyncio.sleep(0)
            cancelled.cancel()
            with pytest.raises(asyncio.CancelledError):
                await cancelled
            assert cancelled_request.state.nemo_gym_model_execution["upstream_attempted"] is False
            release.set()
            await asyncio.wait_for(asyncio.gather(first, third), 2)
            await asyncio.wait_for(send(server), 2)
        finally:
            release.set()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
    assert len(starts) == 3
    assert all(second - first >= INTERVAL * 0.9 for first, second in zip(starts, starts[1:]))


async def test_inflight_cancellation_releases_lock_and_preserves_cooldown(monkeypatch):
    starts = []
    entered = asyncio.Event()
    release = asyncio.Event()

    async def handle(request):
        starts.append(monotonic())
        if len(starts) == 1:
            entered.set()
            await release.wait()
        return web.json_response(chat_reply())

    async with backend(monkeypatch, handle) as server:
        first = asyncio.create_task(send(server))
        try:
            await asyncio.wait_for(entered.wait(), 2)
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            await asyncio.wait_for(send(server), 2)
        finally:
            release.set()
            first.cancel()
            await asyncio.gather(first, return_exceptions=True)
    assert len(starts) == 2
    assert starts[1] - starts[0] >= INTERVAL * 0.9


def test_pacing_rejects_multiple_workers():
    with patch.object(VLLMModel, "_post_init"):
        server = VLLMModel.model_construct(config=config(num_workers=2))
    assert server.multiple_workers_unsupported_reason() is not None


async def test_compaction_summary_retries_are_separate_paced_attempts(monkeypatch):
    starts = []

    async def respond(request):
        starts.append(monotonic())
        payload = chat_reply()
        if len(starts) == 2:
            payload["choices"][0]["message"]["content"] = "## Goal\nComplete the test."
        return web.json_response(payload)

    async with backend(
        monkeypatch,
        respond,
        compaction_summary_max_tokens=512,
        compaction_summary_sections=["Goal"],
        compaction_summary_max_attempts=2,
    ) as server:
        request = make_request()
        request.scope["headers"].append((b"x-nemo-gym-opencode-agent", b"compaction"))
        result = await send(server, request=request)
    assert result.choices[0].message.content == "## Goal\nComplete the test."
    assert len(starts) == 2
    assert starts[1] - starts[0] >= INTERVAL * 0.9
