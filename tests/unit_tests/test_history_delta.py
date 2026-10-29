# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""History reconstruction, protocol validation and client resynchronization tests."""

import hashlib
from contextlib import ExitStack
from copy import deepcopy
from unittest.mock import Mock

import pytest
from fastapi import FastAPI, Request, Response
from fastapi.testclient import TestClient
from openai import ConflictError

from nemo_gym.models.model_call_capture import _CaptureMiddleware
from nemo_gym.models.history_delta import (
    HistoryBuffer,
    HistoryResync,
    HistoryTransportProtocol,
    _encoded,
    reconstruct_request,
)
from responses_api_agents.hermes_agent.history_delta import HistoryDeltaClient


@pytest.fixture
def envelope():
    def make(messages, *, identity, base=None, mode="full", conversation="root", field="messages"):
        return {
            field: messages,
            "tools": [{"type": "function"}],
            "temperature": 1,
            "_gym_history": {
                "conversation_id": conversation,
                "request_id": identity,
                "base_version": base,
                "mode": mode,
                "messages_sha256": hashlib.sha256(_encoded(messages)).hexdigest(),
            },
        }

    return make


@pytest.fixture
def buffer():
    return HistoryBuffer()


@pytest.fixture
def read_events():
    return []


@pytest.fixture
def received_headers():
    return []


@pytest.fixture
def make_client(read_events, received_headers):
    """Echo requests through the real middleware; record when the body is consumed."""
    with ExitStack() as stack:

        def make(*, enabled=True, protocol=None):
            app = FastAPI()

            @app.post("/v1/chat/completions")
            @app.post("/v1/responses")
            async def echo(request: Request):
                read_events.append("dispatch")
                received_headers.append(dict(request.headers))
                return Response(await request.body(), media_type="application/json")

            app.add_middleware(
                _CaptureMiddleware,
                store=None,
                model_server_name="model",
                history_delta_enabled=enabled,
                history_transport_protocol=protocol,
            )

            async def record_reads(scope, receive, send):
                async def read():
                    read_events.append("read")
                    return await receive()

                await app(scope, read, send)

            return stack.enter_context(TestClient(record_reads))

        yield make


@pytest.fixture
def client(make_client):
    return make_client()


@pytest.fixture
def hermes_backend(buffer):
    """Model-call stub backed by Gym's real history reconstruction."""
    wire = []

    def create(**kwargs):
        request = {k: v for k, v in kwargs.items() if k not in {"extra_body", "extra_headers"}}
        request.update(kwargs["extra_body"])
        wire.append(deepcopy(request))
        try:
            return buffer.reconstruct("rollout", request)
        except HistoryResync:
            raise ConflictError(
                "History resync required",
                response=Mock(status_code=409, request=Mock()),
                body={"error": {"code": "gym_history_resync"}},
            ) from None

    def restart():
        nonlocal buffer
        buffer = HistoryBuffer()

    return create, restart, wire


def test_plain_request_is_forwarded_without_reading_body_first(client, read_events):
    read_events.clear()  # Ignore the test client's lifespan messages.
    response = client.post("/ng-rollout/test/v1/chat/completions", content=b"not JSON")

    assert response.status_code == 200
    assert response.content == b"not JSON"
    assert read_events == ["dispatch", "read"]


def test_custom_protocol_header_selects_reconstruction(make_client, envelope):
    class NativeProtocol(HistoryTransportProtocol):
        def matches_request(self, scope):
            return b"x-native-continuation" in dict(scope.get("headers", []))

        async def reconstruct_request(self, buffer, scope, receive, send, rollout_id, *, history_field):
            return await reconstruct_request(buffer, scope, receive, send, rollout_id, history_field=history_field)

    client = make_client(protocol=NativeProtocol())
    request = envelope([{"role": "user", "content": "task"}], identity="one")
    response = client.post(
        "/ng-rollout/test/v1/chat/completions", json=request, headers={"x-native-continuation": "1"}
    )

    assert response.status_code == 200
    assert response.json() == {key: value for key, value in request.items() if key != "_gym_history"}


def test_hermes_client_preserves_requests_after_append_edit_and_lost_base(hermes_backend):
    create, restart, wire = hermes_backend
    client = HistoryDeltaClient()
    options = {
        "model": "model",
        "tools": [{"type": "function"}],
        "temperature": 1,
        "extra_body": {"metadata": {"unchanged": True}},
    }
    system = {"role": "system", "content": "instructions"}
    user = {"role": "user", "content": [{"type": "text", "text": "café\n雪"}]}
    assistant = {"role": "assistant", "tool_calls": [{"id": "tool-one"}]}
    tool = {"role": "tool", "tool_call_id": "tool-one", "content": "result"}
    compacted = {"role": "user", "content": "compacted history"}

    first = {**options, "messages": [system, user]}
    appended = {**options, "messages": [system, user, assistant, tool]}
    edited = {**options, "messages": [compacted]}
    after_restart = {**options, "messages": [compacted, assistant, tool]}
    requests = [first, appended, edited, after_restart]
    originals = deepcopy(requests)

    responses = [
        client.call(create, **first),
        client.call(create, **appended),
        client.call(create, **edited),
    ]
    restart()
    responses.append(client.call(create, **after_restart))

    expected = [
        {k: v for k, v in request.items() if k != "extra_body"} | request["extra_body"] for request in requests
    ]
    assert responses == expected
    assert requests == originals
    assert [request["_gym_history"]["mode"] for request in wire] == ["full", "append", "full", "append", "full"]
    assert wire[1]["messages"] == [assistant, tool]


def test_invalid_delta_is_rejected(buffer, envelope):
    history = [{"role": "user", "content": "a"}]
    first_request = envelope(history, identity="one")
    buffer.reconstruct("rollout", first_request)

    assistant = {"role": "assistant", "content": "answer"}
    history_digest = hashlib.sha256(_encoded([*history, assistant])).hexdigest()

    valid_delta = {
        "messages": [assistant],
        "_gym_history": {
            "conversation_id": "root",
            "request_id": "two",
            "base_version": "one",
            "mode": "append",
            "messages_sha256": history_digest,
        },
    }
    bad_digest_request = {
        "messages": [assistant],
        "_gym_history": {
            "conversation_id": "root",
            "request_id": "two",
            "base_version": "one",
            "mode": "append",
            "messages_sha256": "0" * 64,
        },
    }
    stale_base_request = {
        "messages": [assistant],
        "_gym_history": {
            "conversation_id": "root",
            "request_id": "two",
            "base_version": "missing",
            "mode": "append",
            "messages_sha256": history_digest,
        },
    }
    reused_identity_request = {
        "messages": [assistant],
        "_gym_history": {
            "conversation_id": "root",
            "request_id": "one",
            "base_version": "one",
            "mode": "append",
            "messages_sha256": history_digest,
        },
    }

    with pytest.raises(HistoryResync, match="Missing or stale history base"):
        buffer.reconstruct("other", valid_delta)

    with pytest.raises(HistoryResync, match="Reconstructed history digest mismatch"):
        buffer.reconstruct("rollout", bad_digest_request)

    with pytest.raises(HistoryResync, match="Missing or stale history base"):
        buffer.reconstruct("rollout", stale_base_request)

    with pytest.raises(ValueError, match="Request identity reused with a different payload"):
        buffer.reconstruct("rollout", reused_identity_request)


def test_hermes_client_does_not_retry_model_errors():
    create = Mock(side_effect=RuntimeError("model failure"))

    with pytest.raises(RuntimeError, match="model failure"):
        HistoryDeltaClient().call(create, messages=[{"role": "user", "content": "task"}])

    assert create.call_count == 1


@pytest.mark.parametrize("loss", ["expiry", "capacity"])
def test_lost_history_requires_resync(envelope, monkeypatch, loss):
    buffer = HistoryBuffer(ttl=1, max_entries=1)
    monkeypatch.setattr("nemo_gym.models.history_delta.time.monotonic", lambda: 0)
    buffer.reconstruct("a", envelope([], identity="one"))
    if loss == "expiry":
        monkeypatch.setattr("nemo_gym.models.history_delta.time.monotonic", lambda: 2)
    else:
        buffer.reconstruct("b", envelope([], identity="other-one"))

    with pytest.raises(HistoryResync):
        buffer.reconstruct("a", envelope([], identity="two", base="one", mode="append"))


@pytest.mark.parametrize("endpoint,field", [("chat/completions", "messages"), ("responses", "input")])
def test_http_requests_are_reconstructed_and_repeated_delivery_is_safe(client, received_headers, endpoint, field):
    url = f"/ng-rollout/test/v1/{endpoint}"
    headers = {"x-gym-history-transport": "v1"}
    user_message = {"role": "user", "content": "task"}
    tool_result = {"type": "function_call_output", "call_id": "one", "output": "tool result"}

    first_request = {
        field: [user_message],
        "tools": [{"type": "function"}],
        "temperature": 1,
        "stream": True,
        "_gym_history": {
            "conversation_id": "root",
            "request_id": "one",
            "base_version": None,
            "mode": "full",
            "messages_sha256": hashlib.sha256(_encoded([user_message])).hexdigest(),
        },
    }
    delta_request = {
        field: [tool_result],
        "tools": [{"type": "function"}],
        "temperature": 1,
        "stream": True,
        "_gym_history": {
            "conversation_id": "root",
            "request_id": "two",
            "base_version": "one",
            "mode": "append",
            "messages_sha256": hashlib.sha256(_encoded([user_message, tool_result])).hexdigest(),
        },
    }

    first_response = client.post(url, json=first_request, headers=headers)
    delta_response = client.post(url, json=delta_request, headers=headers)
    retry_response = client.post(url, json=delta_request, headers=headers)

    assert first_response.status_code == 200
    assert delta_response.status_code == 200
    assert retry_response.status_code == 200
    assert first_response.json() == {
        field: [user_message],
        "tools": [{"type": "function"}],
        "temperature": 1,
        "stream": True,
    }
    expected = {
        field: [user_message, tool_result],
        "tools": [{"type": "function"}],
        "temperature": 1,
        "stream": True,
    }
    assert delta_response.json() == expected
    assert retry_response.json() == expected
    assert all("x-gym-history-transport" not in headers for headers in received_headers)


def test_disabled_server_rejects_transport_before_dispatch(make_client, envelope, received_headers):
    client = make_client(enabled=False)
    response = client.post(
        "/ng-rollout/test/v1/chat/completions",
        json=envelope([], identity="one"),
        headers={"x-gym-history-transport": "v1"},
    )

    assert response.status_code == 400
    assert received_headers == []
