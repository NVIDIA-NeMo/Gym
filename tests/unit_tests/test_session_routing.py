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
"""Session routing across uvicorn workers, with two in-process workers that share a real socket directory."""

import asyncio
import json
import os
import shutil
import socket
import tempfile
from base64 import b64decode
from collections.abc import Iterator
from typing import Optional
from unittest.mock import MagicMock

import itsdangerous
import pytest
from aiohttp import ClientSession, DummyCookieJar, UnixConnector
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

from nemo_gym.base_resources_server import BaseResourcesServerConfig, SimpleResourcesServer
from nemo_gym.mcp_auto_exposure import TOKEN_HEADER, maybe_auto_expose, session_token_serializer
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from nemo_gym.session_routing import (
    SESSION_OWNER_HEADER,
    SESSION_OWNER_KEY,
    SessionRoutingMiddleware,
    install_session_routing,
    session_aliases,
    session_placements,
    worker_socket_path,
)


class StoreRequest(BaseModel):
    value: int


class SessionStateServer(SimpleResourcesServer):
    """Keeps a value per session, like swe_rebench's sandbox map: stored by one request, taken by a later one."""

    label: str = ""
    store: dict[str, int] = Field(default_factory=dict)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/store")(self.put)
        app.post("/take")(self.take)
        app.get("/echo")(self.echo)
        return app

    async def put(self, request: Request, body: StoreRequest) -> dict:
        self.store[request.session[SESSION_ID_KEY]] = body.value
        return {"worker": self.label}

    async def take(self, request: Request) -> dict:
        return {"worker": self.label, "value": self.store.pop(request.session[SESSION_ID_KEY], None)}

    async def echo(self, request: Request) -> dict:
        return {"worker": self.label, "query": request.url.query, "probe": request.headers.get("x-probe")}

    async def verify(self, body):
        pass


def _server(label: str, *, mcp: bool) -> SessionStateServer:
    config = BaseResourcesServerConfig(host="", port=0, entrypoint="", name="state", expose_tools_over_mcp=mcp)
    return SessionStateServer(config=config, server_client=MagicMock(spec=ServerClient), label=label)


class Worker:
    def __init__(self, label: str, socket_dir: Optional[str], *, mcp: bool = False) -> None:
        self.server = _server(label, mcp=mcp)
        self.app = self.server.setup_webserver()
        mcp_routing = {}
        if mcp:
            maybe_auto_expose(self.server, self.app)
            mcp_routing = dict(
                mcp_token_header=TOKEN_HEADER, mcp_token_serializer=session_token_serializer(self.server)
            )
        self.cookie_name = self.server.get_session_middleware_key()
        self.routing_id: Optional[str] = None
        if socket_dir is not None:
            self.routing_id = install_session_routing(
                self.app,
                socket_dir=socket_dir,
                session_cookie=self.cookie_name,
                secret_key=self.cookie_name,
                **mcp_routing,
            )
        self.client = TestClient(self.app)

    def session(self, cookie_value: str) -> dict:
        signer = itsdangerous.TimestampSigner(self.cookie_name)
        return json.loads(b64decode(signer.unsign(cookie_value.encode())))


@pytest.fixture
def socket_dir() -> Iterator[str]:
    # Under /tmp, as in production: AF_UNIX paths are limited to about 100 bytes.
    path = tempfile.mkdtemp(prefix="ng-test-", dir="/tmp")
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def workers(socket_dir: str) -> Iterator[tuple[Worker, Worker]]:
    a, b = Worker("a", socket_dir), Worker("b", socket_dir)
    with a.client, b.client:
        yield a, b


@pytest.fixture
def mcp_workers(socket_dir: str) -> Iterator[tuple[Worker, Worker]]:
    a, b = Worker("a", socket_dir, mcp=True), Worker("b", socket_dir, mcp=True)
    with a.client, b.client:
        yield a, b


def _mcp_call(client: TestClient, name: str, arguments: dict, token: str) -> dict:
    """A tools/call as a CLI harness sends it: the session token header, and no cookies."""
    client.cookies.clear()
    response = client.post(
        "/mcp",
        headers={"accept": "application/json, text/event-stream", TOKEN_HEADER: token},
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": arguments}},
    )
    assert response.status_code == 200, response.text
    return response.json()["result"]


def _mcp_payload(result: dict) -> dict:
    assert result.get("isError") is not True, result
    return json.loads(result["content"][0]["text"])


def _set_cookies(response, name: str) -> list[str]:
    return [value for value in response.headers.get_list("set-cookie") if value.startswith(f"{name}=")]


class TestSessionRouting:
    def test_a_new_session_is_handled_locally_and_owned_by_this_worker(self, workers) -> None:
        a, b = workers

        response = b.client.post("/store", json={"value": 7})

        assert response.json() == {"worker": "b"}
        assert list(b.server.store.values()) == [7] and a.server.store == {}
        assert b.session(response.cookies[b.cookie_name])[SESSION_OWNER_KEY] == b.routing_id

    def test_a_session_owned_by_another_worker_is_forwarded_there(self, workers) -> None:
        a, b = workers
        cookie = a.client.post("/store", json={"value": 7}).cookies[a.cookie_name]

        response = b.client.post("/take", cookies={a.cookie_name: cookie})

        assert response.status_code == 200
        assert response.json() == {"worker": "a", "value": 7}
        assert a.server.store == {} and b.server.store == {}
        # Only the owner's Set-Cookie comes back, and the session stays with the owner.
        [set_cookie] = _set_cookies(response, a.cookie_name)
        assert a.session(response.cookies[a.cookie_name])[SESSION_OWNER_KEY] == a.routing_id
        assert "path=/" in set_cookie

    def test_method_path_query_and_headers_are_forwarded_unchanged(self, workers) -> None:
        a, b = workers
        cookie = a.client.post("/store", json={"value": 1}).cookies[a.cookie_name]

        response = b.client.get("/echo?x=1&y=a%20b", headers={"x-probe": "p"}, cookies={a.cookie_name: cookie})

        assert response.json() == {"worker": "a", "query": "x=1&y=a%20b", "probe": "p"}

    def test_a_request_without_a_session_goes_to_the_worker_its_header_names(self, workers) -> None:
        a, b = workers

        response = b.client.get("/echo", headers={SESSION_OWNER_HEADER: a.routing_id})
        b.client.cookies.clear()
        own = b.client.get("/echo")

        assert response.json()["worker"] == "a"
        assert own.json()["worker"] == "b"

    def test_a_session_of_an_aliased_worker_goes_to_the_worker_that_holds_it_now(self, socket_dir: str) -> None:
        old, a, b = Worker("old", socket_dir), Worker("a", socket_dir), Worker("b", socket_dir)
        with old.client:
            cookie = old.client.post("/store", json={"value": 7}).cookies[old.cookie_name]
        # A restore moved the exited worker's session to a, and every worker's router knows it.
        a.server.store.update(old.server.store)
        for worker in (a, b):
            session_aliases(worker.app)[old.routing_id] = a.routing_id
        with a.client, b.client:
            response = b.client.post("/take", cookies={old.cookie_name: cookie})

        assert response.json() == {"worker": "a", "value": 7}

    def test_a_forwarded_request_is_never_forwarded_again(self, workers) -> None:
        a, b = workers
        cookie = a.client.post("/store", json={"value": 7}).cookies[a.cookie_name]

        async def take_over_private_socket() -> dict:
            connector = UnixConnector(_socket_of(b))
            async with ClientSession(connector=connector, cookie_jar=DummyCookieJar()) as client:
                async with client.post("http://localhost/take", cookies={a.cookie_name: cookie}) as response:
                    return await response.json()

        # Arriving over b's private socket, the request runs on b even though a owns the session.
        assert asyncio.run(take_over_private_socket()) == {"worker": "b", "value": None}
        assert list(a.server.store.values()) == [7]

    def test_a_session_whose_worker_exited_gets_an_explicit_error(self, socket_dir: str) -> None:
        a, b = Worker("a", socket_dir), Worker("b", socket_dir)
        with b.client:
            with a.client:
                cookie = a.client.post("/store", json={"value": 7}).cookies[a.cookie_name]
            assert not os.path.exists(_socket_of(a))

            response = b.client.post("/take", cookies={a.cookie_name: cookie})

        assert response.status_code == 410
        assert "no longer running" in response.json()["detail"]

    def test_a_crashed_worker_left_socket_gets_an_explicit_error(self, socket_dir: str) -> None:
        a, b = Worker("a", socket_dir), Worker("b", socket_dir)
        with a.client:
            cookie = a.client.post("/store", json={"value": 7}).cookies[a.cookie_name]
        # A worker killed outright leaves its socket file behind with nothing listening.
        with socket.socket(socket.AF_UNIX) as stale:
            stale.bind(_socket_of(a))

        with b.client:
            response = b.client.post("/take", cookies={a.cookie_name: cookie})

        assert response.status_code == 410

    def test_a_badly_signed_cookie_is_handled_locally(self, workers) -> None:
        a, b = workers
        forged = itsdangerous.TimestampSigner("another secret").sign(
            json.dumps({SESSION_ID_KEY: "s", SESSION_OWNER_KEY: a.routing_id}).encode()
        )

        response = b.client.post("/store", json={"value": 3}, cookies={a.cookie_name: forged.decode()})

        assert response.json() == {"worker": "b"}
        assert b.session(response.cookies[b.cookie_name])[SESSION_OWNER_KEY] == b.routing_id

    def test_a_session_from_a_single_worker_server_is_adopted_locally(self, workers) -> None:
        a, b = workers
        single = Worker("single", socket_dir=None)
        with single.client:
            cookie = single.client.post("/store", json={"value": 3}).cookies[single.cookie_name]

        response = b.client.post("/store", json={"value": 4}, cookies={b.cookie_name: cookie})

        assert response.json() == {"worker": "b"}
        assert b.session(response.cookies[b.cookie_name])[SESSION_OWNER_KEY] == b.routing_id

    def test_a_single_worker_server_has_no_routing(self) -> None:
        single = Worker("single", socket_dir=None)
        with single.client:
            response = single.client.post("/store", json={"value": 3})

        assert not any(m.cls is SessionRoutingMiddleware for m in single.app.user_middleware)
        assert SESSION_OWNER_KEY not in single.session(response.cookies[single.cookie_name])


class TestMCPSessionRouting:
    def test_an_mcp_call_with_a_token_from_another_worker_is_forwarded_there(self, mcp_workers) -> None:
        a, b = mcp_workers
        token = a.client.post("/seed_session", json={}).json()["mcp"]["headers"][TOKEN_HEADER]
        assert session_token_serializer(a.server).loads(token)[SESSION_OWNER_KEY] == a.routing_id

        assert _mcp_payload(_mcp_call(b.client, "store", {"value": 5}, token)) == {"worker": "a"}
        assert _mcp_payload(_mcp_call(b.client, "take", {}, token)) == {"worker": "a", "value": 5}
        assert a.server.store == {} and b.server.store == {}

    def test_an_mcp_token_without_an_owner_is_handled_locally(self, mcp_workers) -> None:
        a, b = mcp_workers
        # A token minted before this change, or by a single-worker server.
        token = session_token_serializer(a.server).dumps({"sid": "s", "tools": None})

        assert _mcp_payload(_mcp_call(b.client, "store", {"value": 5}, token)) == {"worker": "b"}
        assert b.server.store == {"s": 5}

    def test_a_badly_signed_mcp_token_is_handled_locally_and_refused(self, mcp_workers) -> None:
        a, b = mcp_workers
        forged = itsdangerous.URLSafeSerializer("another secret", salt="x").dumps(
            {"sid": "s", "tools": None, SESSION_OWNER_KEY: a.routing_id}
        )

        result = _mcp_call(b.client, "store", {"value": 5}, forged)

        assert result["isError"] is True and "Invalid Gym MCP session token" in result["content"][0]["text"]
        assert a.server.store == {} and b.server.store == {}

    def test_a_restored_session_without_an_owner_is_found_by_its_session_id(self, socket_dir: str) -> None:
        """A single-worker cookie or MCP token names no worker; a restore places its session by session ID."""
        single = Worker("single", socket_dir=None, mcp=True)
        with single.client:
            seeded = single.client.post("/seed_session", json={})
            token = seeded.json()["mcp"]["headers"][TOKEN_HEADER]
            cookie = seeded.cookies[single.cookie_name]
        session_id = single.session(cookie)[SESSION_ID_KEY]
        a, b = Worker("a", socket_dir, mcp=True), Worker("b", socket_dir, mcp=True)
        # A restore installed the session on a, and every worker's router knows where it is.
        a.server.store[session_id] = 7
        for worker in (a, b):
            session_placements(worker.app)[session_id] = a.routing_id
        with a.client, b.client:
            by_token = _mcp_payload(_mcp_call(b.client, "take", {}, token))
            a.server.store[session_id] = 8
            b.client.cookies.clear()
            by_cookie = b.client.post("/take", cookies={single.cookie_name: cookie}).json()

        assert by_token == {"worker": "a", "value": 7}
        assert by_cookie == {"worker": "a", "value": 8}

    def test_a_single_worker_mcp_token_has_no_owner(self) -> None:
        single = Worker("single", socket_dir=None, mcp=True)
        with single.client:
            token = single.client.post("/seed_session", json={}).json()["mcp"]["headers"][TOKEN_HEADER]

        assert SESSION_OWNER_KEY not in session_token_serializer(single.server).loads(token)


def _socket_of(worker: Worker) -> str:
    [middleware] = [m for m in worker.app.user_middleware if m.cls is SessionRoutingMiddleware]
    return worker_socket_path(middleware.kwargs["socket_dir"], worker.routing_id)
