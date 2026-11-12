# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gzip

import pytest
from fastapi import FastAPI, Request
from starlette.testclient import TestClient

from nemo_gym.server_runtime.request_handling.request_gzip import GzipRequestMiddleware


@pytest.fixture
def client():
    app = FastAPI()

    @app.post("/v1/responses")
    async def echo(request: Request):
        return {"body": (await request.body()).decode(), "headers": dict(request.headers)}

    app.add_middleware(GzipRequestMiddleware)
    with TestClient(app) as client:
        yield client


def test_gzip_body_is_decoded_and_headers_describe_the_decoded_request(client):
    body = '{"input":"hello 🌍"}'.encode()
    response = client.post(
        "/v1/responses",
        content=gzip.compress(body),
        headers={"Content-Encoding": "gzip", "Transfer-Encoding": "chunked", "X-Test": "preserved"},
    )

    assert response.status_code == 200
    assert response.json()["body"] == body.decode()
    headers = response.json()["headers"]
    assert headers["content-length"] == str(len(body))
    assert headers["x-test"] == "preserved"
    assert "content-encoding" not in headers
    assert "transfer-encoding" not in headers


@pytest.mark.parametrize("encoding", [None, "identity"])
def test_uncompressed_requests_pass_through_without_decoding(client, encoding):
    headers = {} if encoding is None else {"Content-Encoding": encoding}
    response = client.post("/v1/responses", content=b'{"input":"plain"}', headers=headers)

    assert response.status_code == 200
    assert response.json()["body"] == '{"input":"plain"}'
    assert response.json()["headers"].get("content-encoding") == encoding


@pytest.mark.parametrize(
    "body",
    [b"not gzip", gzip.compress(b"hello")[:-4], gzip.compress(b"hello") + b"trailing data"],
    ids=["invalid", "truncated", "trailing-data"],
)
def test_malformed_gzip_returns_400(client, body):
    response = client.post("/v1/responses", content=body, headers={"Content-Encoding": "gzip"})

    assert response.status_code == 400
    assert response.json() == {"error": "Invalid gzip request body"}


@pytest.mark.parametrize("encoding", ["br", "gzip, br"])
def test_unsupported_content_encoding_returns_415(client, encoding):
    response = client.post("/v1/responses", content=b"body", headers={"Content-Encoding": encoding})

    assert response.status_code == 415
    assert response.json() == {"error": "Unsupported Content-Encoding"}


async def test_non_http_events_pass_through_untouched():
    scope = {"type": "lifespan"}
    events = []

    async def receive():
        return {"type": "lifespan.startup"}

    async def send(message):
        events.append(message)

    async def app(actual_scope, actual_receive, actual_send):
        assert actual_scope is scope
        assert actual_receive is receive
        assert actual_send is send
        assert await actual_receive() == {"type": "lifespan.startup"}
        await actual_send({"type": "lifespan.startup.complete"})

    await GzipRequestMiddleware(app)(scope, receive, send)
    assert events == [{"type": "lifespan.startup.complete"}]
