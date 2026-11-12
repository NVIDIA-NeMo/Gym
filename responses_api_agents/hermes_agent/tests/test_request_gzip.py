# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gzip
import hashlib
import json

import httpx
import openai
import pytest

from responses_api_agents.hermes_agent.request_gzip import _compress, install_request_gzip


def test_compressed_request_preserves_serialized_bytes_and_updates_stream_and_headers():
    body = json.dumps({"input": "hello 🌍" * 1024}, ensure_ascii=False).encode()
    request = httpx.Request(
        "POST",
        "http://gym.test/v1/responses",
        content=body,
        headers={"x-test": "kept", "transfer-encoding": "chunked"},
    )

    _compress(request, body)

    assert len(request.content) < len(body)
    assert gzip.decompress(request.content) == body
    assert b"".join(request.stream) == request.content
    assert request.headers["content-encoding"] == "gzip"
    assert request.headers["content-length"] == str(len(request.content))
    assert request.headers["x-test"] == "kept"
    assert "transfer-encoding" not in request.headers


@pytest.mark.parametrize(
    ("body", "headers"),
    [
        pytest.param(b"x" * 4095, {}, id="small-body"),
        # Stored gzip blocks keep this above the small-body cutoff and eligible for further compression.
        pytest.param(
            gzip.compress(b"x" * 8192, compresslevel=0, mtime=0),
            {"content-encoding": "gzip"},
            id="already-gzipped",
        ),
        pytest.param(
            b"".join(hashlib.sha256(str(i).encode()).digest() for i in range(256)), {}, id="no-size-reduction"
        ),
    ],
)
def test_bypassed_requests_keep_their_body_stream_and_headers(body, headers):
    request = httpx.Request("POST", "http://gym.test/v1/responses", content=body, headers=headers)
    original_headers = dict(request.headers)
    original_stream = request.stream

    _compress(request, body)

    assert request.content == body
    assert request.stream is original_stream
    assert dict(request.headers) == original_headers


def test_body_at_minimum_compression_size_is_eligible():
    body = b"x" * 4096
    request = httpx.Request("POST", "http://gym.test/v1/responses", content=body)

    _compress(request, body)

    assert request.headers["content-encoding"] == "gzip"
    assert gzip.decompress(request.content) == body


@pytest.fixture
def installed_hooks(monkeypatch):
    # Restore the original constructors after the production installer wraps them.
    monkeypatch.setattr(openai.OpenAI, "__init__", openai.OpenAI.__init__)
    monkeypatch.setattr(openai.AsyncOpenAI, "__init__", openai.AsyncOpenAI.__init__)
    install_request_gzip("http://gym.test/rollouts/episode/v1")


def test_sync_openai_client_sends_gzip_to_the_model_endpoint(installed_hooks):
    observed = []

    def respond(request):
        observed.append(request)
        return httpx.Response(200, json={"id": "response-test", "object": "response", "output": []})

    with openai.OpenAI(
        api_key="unused-unit-test-key",
        base_url="http://gym.test/rollouts/episode/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
        max_retries=0,
    ) as client:
        response = client.responses.create(model="unit-test-model", input="hello " * 1024)

    assert response.id == "response-test"
    assert len(observed) == 1
    assert observed[0].headers["content-encoding"] == "gzip"
    assert json.loads(gzip.decompress(observed[0].content)) == {"model": "unit-test-model", "input": "hello " * 1024}


async def test_async_openai_client_sends_gzip_to_the_model_endpoint(installed_hooks):
    observed = []

    async def respond(request):
        observed.append(request)
        return httpx.Response(200, json={"id": "response-test", "object": "response", "output": []})

    async with openai.AsyncOpenAI(
        api_key="unused-unit-test-key",
        base_url="http://gym.test/rollouts/episode/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
        max_retries=0,
    ) as client:
        response = await client.responses.create(model="unit-test-model", input="hello " * 1024)

    assert response.id == "response-test"
    assert len(observed) == 1
    assert observed[0].headers["content-encoding"] == "gzip"
    assert json.loads(gzip.decompress(observed[0].content)) == {"model": "unit-test-model", "input": "hello " * 1024}


@pytest.mark.parametrize(
    "url", ["http://other.test/rollouts/episode/v1/responses", "http://gym.test/rollouts/episode/v1/tools"]
)
def test_hooks_leave_non_model_requests_unchanged(installed_hooks, url):
    body = b"x" * 8192
    with openai.OpenAI(api_key="unused-unit-test-key", base_url="http://gym.test/rollouts/episode/v1") as client:
        request = httpx.Request("POST", url, content=body)
        for hook in client._client.event_hooks["request"]:
            hook(request)

    assert request.content == body
    assert "content-encoding" not in request.headers
