# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gzip

import pytest
from fastapi import FastAPI, Request
from starlette.testclient import TestClient

from nemo_gym.server_runtime.request_handling.request_gzip import GzipRequestMiddleware


MAX_DECODED_BYTES = 64


@pytest.fixture
def client():
    app = FastAPI()

    @app.post("/v1/responses")
    async def echo(request: Request):
        return {"body": (await request.body()).decode(), "headers": dict(request.headers)}

    app.add_middleware(GzipRequestMiddleware, max_decoded_bytes=MAX_DECODED_BYTES)
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


@pytest.mark.parametrize("encoding", [None, "", "identity", "br", "gzip, br"])
def test_requests_without_a_single_gzip_encoding_pass_through(client, encoding):
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
    assert response.json() == {
        "error": {
            "message": "Invalid gzip request body.",
            "type": "invalid_request_error",
            "param": None,
            "code": None,
        }
    }


def test_gzip_body_at_the_decoded_size_limit_is_accepted(client):
    body = b"a" * MAX_DECODED_BYTES
    response = client.post("/v1/responses", content=gzip.compress(body), headers={"Content-Encoding": "gzip"})

    assert response.status_code == 200
    assert response.json()["body"] == body.decode()


def test_gzip_body_above_the_decoded_size_limit_returns_413(client):
    body = b"a" * (MAX_DECODED_BYTES + 1)
    response = client.post("/v1/responses", content=gzip.compress(body), headers={"Content-Encoding": "gzip"})

    assert response.status_code == 413
    assert response.json() == {
        "error": {
            "message": "Decoded gzip request body exceeds 64 bytes.",
            "type": "invalid_request_error",
            "param": None,
            "code": None,
        }
    }


def test_decoded_size_limit_does_not_apply_to_plain_bodies(client):
    body = b"a" * (MAX_DECODED_BYTES + 1)
    response = client.post("/v1/responses", content=body)

    assert response.status_code == 200
    assert response.json()["body"] == body.decode()


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

    await GzipRequestMiddleware(app, max_decoded_bytes=MAX_DECODED_BYTES)(scope, receive, send)
    assert events == [{"type": "lifespan.startup.complete"}]
