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
import hashlib
import json
import logging
import os
import shutil
import socket
import tempfile
import time
from base64 import b64decode
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from unittest.mock import MagicMock

import itsdangerous
import pytest
from aiohttp import ClientSession, DummyCookieJar, UnixConnector
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient
from httpx import Response
from pydantic import BaseModel, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    ResourcesCloseSessionRequest,
    ResourcesCloseSessionResponse,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    SimpleResourcesServer,
)
from nemo_gym.base_responses_api_agent import (
    AgentCloseSessionResponse,
    AgentSessionSetupError,
    AgentSessionState,
    BaseResponsesAPIAgentConfig,
    SimpleResponsesAPIAgent,
)
from nemo_gym.mcp_auto_exposure import TOKEN_HEADER, maybe_auto_expose, session_token_serializer
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from nemo_gym.session_routing import (
    AGENT_SESSION_CLAIMS_DIR,
    AGENT_SESSION_ROUTES,
    RESOURCES_SESSION_CLAIMS_DIR,
    SESSION_OWNER_KEY,
    SessionClaims,
    SessionRoutingMiddleware,
    install_session_routing,
    serve_private_socket,
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

    def test_a_single_worker_mcp_token_has_no_owner(self) -> None:
        single = Worker("single", socket_dir=None, mcp=True)
        with single.client:
            token = single.client.post("/seed_session", json={}).json()["mcp"]["headers"][TOKEN_HEADER]

        assert SESSION_OWNER_KEY not in session_token_serializer(single.server).loads(token)


def _socket_of(worker: Worker) -> str:
    [middleware] = [m for m in worker.app.user_middleware if m.cls is SessionRoutingMiddleware]
    return worker_socket_path(middleware.kwargs["socket_dir"], worker.routing_id)


class _SessionAgent(SimpleResponsesAPIAgent):
    label: str = ""
    fail_seed: bool = False
    seeds: list[str] = Field(default_factory=list)
    closes: list[str] = Field(default_factory=list)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        # A close under a rollout prefix, which the router routes like a bare one.
        app.post("/ng-rollout/{rollout_id}/v1/agent_sessions/close")(self.close_agent_session)
        return app

    async def responses(self, body):
        raise NotImplementedError

    async def run(self, body):
        raise NotImplementedError

    async def _seed_agent_session_state(self, body) -> AgentSessionState:
        self.seeds.append(body.agent_session_id)
        if self.fail_seed:
            # Setup started something it could not clean up, so the state stays for the caller's close.
            raise AgentSessionSetupError(AgentSessionState(request=body), error=RuntimeError("setup failed"))
        return AgentSessionState(request=body)

    async def _close_agent_session_state(self, state: AgentSessionState) -> AgentCloseSessionResponse:
        self.closes.append(state.request.agent_session_id)
        return AgentCloseSessionResponse(agent_session_id=f"{state.request.agent_session_id}@{self.label}")


class AgentWorker:
    """One worker of a two-worker agent server, as run_webserver wires it."""

    def __init__(self, label: str, socket_dir: Optional[str], *, close_window_seconds: float = 300) -> None:
        num_workers = 2 if socket_dir is not None else 1
        config = BaseResponsesAPIAgentConfig(
            host="",
            port=0,
            entrypoint="",
            name="agent",
            num_workers=num_workers,
            session_close_retry_window_seconds=close_window_seconds,
        )
        self.agent = _SessionAgent(config=config, server_client=MagicMock(spec=ServerClient), label=label)
        self.app = self.agent.setup_webserver()
        self.cookie_name = self.agent.get_session_middleware_key()
        self.routing_id: Optional[str] = None
        if socket_dir is not None:
            self.routing_id = install_session_routing(
                self.app,
                socket_dir=socket_dir,
                session_cookie=self.cookie_name,
                secret_key=self.cookie_name,
                agent_sessions=self.agent,
            )
        # A failed seed answers 500, as the exception middleware does for a running server.
        self.client = TestClient(self.app, raise_server_exceptions=False)

    def session(self, cookie_value: str) -> dict:
        signer = itsdangerous.TimestampSigner(self.cookie_name)
        return json.loads(b64decode(signer.unsign(cookie_value.encode())))

    def seed(self, agent_session_id: str = "s", **kwargs) -> Response:
        self.client.cookies.clear()
        return self.client.post("/v1/agent_sessions", json=_seed_body(agent_session_id), **kwargs)

    def close(self, agent_session_id: str = "s", **kwargs) -> Response:
        self.client.cookies.clear()
        body = {"agent_session_id": agent_session_id, "episode_id": {"rollout_id": "r"}}
        return self.client.post("/v1/agent_sessions/close", json=body, **kwargs)


def _seed_body(agent_session_id: str) -> dict:
    return {
        "agent_session_id": agent_session_id,
        "episode_id": {"rollout_id": "r"},
        "task_id": {"taskset": "t", "task_id": "0"},
    }


def _claim_path(socket_dir: str, session_id: str, directory: str = AGENT_SESSION_CLAIMS_DIR) -> str:
    return os.path.join(socket_dir, directory, hashlib.sha256(session_id.encode("utf-8")).hexdigest())


def _wait_until(condition, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


@pytest.fixture
def agent_workers(socket_dir: str) -> Iterator[tuple[AgentWorker, AgentWorker]]:
    a, b = AgentWorker("a", socket_dir), AgentWorker("b", socket_dir)
    with a.client, b.client:
        yield a, b


class TestAgentSessionRouting:
    def test_an_agent_session_seeded_on_one_worker_is_closed_on_that_worker(self, agent_workers) -> None:
        a, b = agent_workers
        seeded = a.seed()
        assert seeded.status_code == 200, seeded.text

        closed = b.close(cookies={a.cookie_name: seeded.cookies[a.cookie_name]})

        assert closed.status_code == 200, closed.text
        assert closed.json()["agent_session_id"] == "s@a"
        assert "s" in a.agent._closed_session_records
        assert not b.agent._session_records and not b.agent._closed_session_records

    def test_a_cookieless_close_after_a_failed_seed_reaches_the_owner_and_frees_its_state(self, agent_workers) -> None:
        """A failed seed answers 500 without Set-Cookie, so the caller's cleanup close carries no cookie."""
        a, b = agent_workers
        a.agent.fail_seed = True
        failed = a.seed()
        assert failed.status_code == 500 and a.cookie_name not in failed.cookies

        closed = b.close()

        assert closed.status_code == 200, closed.text
        assert closed.json()["agent_session_id"] == "s@a"
        assert a.agent.closes == ["s"] and a.agent._session_records["s"].state is None
        assert not b.agent._session_records and b.agent.closes == []

    def test_a_cookieless_seed_retry_reaches_the_owner_and_creates_no_duplicate(self, agent_workers) -> None:
        a, b = agent_workers
        assert a.seed().status_code == 200

        retried = b.seed()

        assert retried.status_code == 200, retried.text
        assert a.agent.seeds == ["s"] and b.agent.seeds == []
        assert set(a.agent._session_records) == {"s"} and not b.agent._session_records
        # The retry's cookie names the owner, so later requests go there by cookie.
        assert a.session(retried.cookies[a.cookie_name])[SESSION_OWNER_KEY] == a.routing_id

    def test_a_delayed_seed_after_a_close_of_an_unknown_id_is_refused(self, agent_workers) -> None:
        a, b = agent_workers
        closed = a.close()
        assert closed.status_code == 200 and closed.json()["agent_session_id"] == "s"

        seeded = b.seed()

        assert seeded.status_code == 409 and "already closed" in seeded.json()["detail"]
        assert a.agent.seeds == [] and b.agent.seeds == []
        assert not b.agent._session_records

    def test_a_claim_is_removed_once_the_owner_evicts_the_closed_record(self, socket_dir: str) -> None:
        a = AgentWorker("a", socket_dir, close_window_seconds=0.05)
        b = AgentWorker("b", socket_dir, close_window_seconds=0.05)
        with a.client, b.client:
            assert a.seed().status_code == 200
            assert b.close().json()["agent_session_id"] == "s@a"
            assert os.path.exists(_claim_path(socket_dir, "s"))
            time.sleep(0.1)
            # The next request for any session evicts the expired receipt.
            assert a.seed("other").status_code == 200

            assert "s" not in a.agent._session_records
            assert not os.path.exists(_claim_path(socket_dir, "s"))
            # With the claim gone, the ID is new again to whichever worker sees it first.
            assert b.seed().status_code == 200
        assert b.agent.seeds == ["s"] and a.agent.seeds == ["s", "other"]

    def test_a_failed_seed_that_nobody_closes_is_released_after_the_retry_window(self, socket_dir: str) -> None:
        a = AgentWorker("a", socket_dir, close_window_seconds=0.05)
        b = AgentWorker("b", socket_dir, close_window_seconds=0.05)
        with a.client, b.client:
            a.agent.fail_seed = True
            assert a.seed().status_code == 500
            time.sleep(0.1)
            a.agent.fail_seed = False
            assert a.seed("other").status_code == 200

            _wait_until(lambda: "s" not in a.agent._session_records)
            assert a.agent.closes == ["s"]
            assert not os.path.exists(_claim_path(socket_dir, "s"))
            assert not a.agent._failed_seed_records

    def test_one_worker_has_no_claims(self, socket_dir: str) -> None:
        single = AgentWorker("single", socket_dir=None)
        with single.client:
            assert single.seed().status_code == 200
            assert single.close().status_code == 200

        assert not any(m.cls is SessionRoutingMiddleware for m in single.app.user_middleware)
        assert single.agent._agent_session_claims is None
        assert not os.path.exists(os.path.join(socket_dir, AGENT_SESSION_CLAIMS_DIR))

    def test_two_workers_racing_to_claim_ids_agree_on_one_claimant_each(self, socket_dir: str) -> None:
        holder = MagicMock(spec=SimpleResponsesAPIAgent)
        claims = [
            SessionClaims(socket_dir, routing_id, holder.holds_agent_session, routes=AGENT_SESSION_ROUTES)
            for routing_id in ("a" * 32, "b" * 32)
        ]
        ids = [f"id-{index}" for index in range(500)]

        def claim_all(worker: SessionClaims) -> list[Optional[str]]:
            return [worker.claim(agent_session_id) for agent_session_id in ids]

        with ThreadPoolExecutor(2) as pool:
            first, second = pool.map(claim_all, claims)

        assert first == second
        for agent_session_id, claimant in zip(ids, first):
            with open(_claim_path(socket_dir, agent_session_id)) as claim:
                assert claim.read() == claimant
        for worker in claims:
            assert worker._mine == {
                agent_session_id for agent_session_id, c in zip(ids, first) if c == worker.routing_id
            }
        # Only claim files remain: every temporary file was removed.
        assert len(os.listdir(os.path.join(socket_dir, AGENT_SESSION_CLAIMS_DIR))) == len(ids)

    def test_a_cookieless_close_for_a_session_of_an_exited_worker_gets_an_explicit_error(
        self, socket_dir: str
    ) -> None:
        a, b = AgentWorker("a", socket_dir), AgentWorker("b", socket_dir)
        with b.client:
            with a.client:
                assert a.seed().status_code == 200

            closed = b.close()

        assert closed.status_code == 410
        assert not b.agent._session_records

    def test_a_restarted_server_starts_with_no_claims(self, socket_dir: str) -> None:
        a, b = AgentWorker("a", socket_dir), AgentWorker("b", socket_dir)
        with a.client, b.client:
            assert a.seed().status_code == 200
        # The restarted server has a fresh socket directory, as each server process does.
        fresh_dir = tempfile.mkdtemp(prefix="ng-test-", dir="/tmp")
        try:
            c, d = AgentWorker("c", fresh_dir), AgentWorker("d", fresh_dir)
            with c.client, d.client:
                closed = d.close()
            claimed_fresh = os.path.exists(_claim_path(fresh_dir, "s"))
        finally:
            shutil.rmtree(fresh_dir, ignore_errors=True)

        # As on one worker after a restart: the close of an ID nobody here has seen is answered with a receipt.
        assert closed.status_code == 200 and closed.json()["agent_session_id"] == "s"
        assert claimed_fresh and "s" in d.agent._closed_session_records and not c.agent._session_records

    def test_a_seed_whose_cookie_names_another_worker_than_the_ids_claimant_is_refused(self, agent_workers) -> None:
        a, b = agent_workers
        assert b.seed("taken").status_code == 200
        # A cookie of a session a owns, with no agent session in it yet.
        a.client.cookies.clear()
        cookie = a.client.get("/no-such-path").cookies[a.cookie_name]
        a.client.cookies.clear()

        response = a.client.post("/v1/agent_sessions", json=_seed_body("taken"), cookies={a.cookie_name: cookie})

        assert response.status_code == 409 and "another server worker" in response.json()["detail"]
        assert b.agent.seeds == ["taken"] and a.agent.seeds == []

    @pytest.mark.parametrize(
        "body", [b"not json", b"[1, 2]", b'{"agent_session_id": 5}', b'{"episode_id": {"rollout_id": "r"}}']
    )
    def test_a_body_without_a_string_id_is_refused_by_the_app_and_claims_nothing(
        self, agent_workers, socket_dir: str, body: bytes
    ) -> None:
        a, b = agent_workers

        response = b.client.post("/v1/agent_sessions", content=body, headers={"content-type": "application/json"})

        assert response.status_code == 422
        assert os.listdir(os.path.join(socket_dir, AGENT_SESSION_CLAIMS_DIR)) == []
        assert not a.agent._session_records and not b.agent._session_records

    def test_an_oversized_body_is_refused_without_reading_it_all(self, agent_workers, socket_dir: str) -> None:
        a, b = agent_workers
        body = json.dumps({**_seed_body("s"), "padding": "x" * (9 * 1024 * 1024)}).encode()

        response = b.client.post("/v1/agent_sessions", content=body, headers={"content-type": "application/json"})

        assert response.status_code == 413
        assert os.listdir(os.path.join(socket_dir, AGENT_SESSION_CLAIMS_DIR)) == []

    def test_a_rollout_prefixed_close_is_routed_too(self, agent_workers) -> None:
        a, b = agent_workers
        assert a.seed().status_code == 200
        b.client.cookies.clear()

        closed = b.client.post(
            "/ng-rollout/r/v1/agent_sessions/close",
            json={"agent_session_id": "s", "episode_id": {"rollout_id": "r"}},
        )

        assert closed.status_code == 200, closed.text
        assert closed.json()["agent_session_id"] == "s@a"
        assert not b.agent._session_records


class TestPrivateSocketLogging:
    def test_the_private_socket_leaves_the_workers_uvicorn_logging_as_it_was(self, socket_dir: str) -> None:
        error, access = logging.getLogger("uvicorn.error"), logging.getLogger("uvicorn.access")
        saved = (error.level, access.level, list(access.handlers), access.propagate)
        handler = logging.NullHandler()
        error.setLevel(logging.INFO)
        access.setLevel(logging.INFO)
        access.addHandler(handler)
        try:

            async def serve() -> None:
                async with serve_private_socket(FastAPI(), os.path.join(socket_dir, "private.sock")):
                    pass

            asyncio.run(serve())

            assert error.level == logging.INFO and access.level == logging.INFO
            assert handler in access.handlers
        finally:
            error.setLevel(saved[0])
            access.setLevel(saved[1])
            access.handlers = saved[2]
            access.propagate = saved[3]


class TypedSessionServer(SimpleResourcesServer):
    """Keeps state per resources_session_id, as a server an environment server seeds does."""

    label: str = ""
    fail_seed: bool = False
    seeds: list[str] = Field(default_factory=list)
    closes: list[str] = Field(default_factory=list)
    sessions: dict[str, str] = Field(default_factory=dict)
    closed: set[str] = Field(default_factory=set)

    async def seed_session(self, body: ResourcesSeedSessionRequest | BaseSeedSessionRequest):
        if not isinstance(body, ResourcesSeedSessionRequest):
            self.seeds.append("cookie session")
            return BaseSeedSessionResponse()
        session_id = body.resources_session_id
        if session_id in self.closed:
            raise HTTPException(409, "Resources session is already closed")
        self.seeds.append(session_id)
        if self.fail_seed:
            # Setup started something its close must release.
            self.sessions[session_id] = "partly set up"
            raise HTTPException(500, "setup failed")
        self.sessions.setdefault(session_id, "ready")
        return ResourcesSeedSessionResponse(resources_session_id=session_id)

    async def close_resources_session(self, body: ResourcesCloseSessionRequest) -> ResourcesCloseSessionResponse:
        self.closes.append(body.resources_session_id)
        self.sessions.pop(body.resources_session_id, None)
        self.closed.add(body.resources_session_id)
        return ResourcesCloseSessionResponse(resources_session_id=f"{body.resources_session_id}@{self.label}")

    async def verify(self, body):
        pass


class TypedResourcesWorker:
    """One worker of a two-worker resources server, as run_webserver wires it."""

    def __init__(self, label: str, socket_dir: str, *, close_window_seconds: float = 300) -> None:
        config = BaseResourcesServerConfig(
            host="", port=0, entrypoint="", name="resources", session_close_retry_window_seconds=close_window_seconds
        )
        self.server = TypedSessionServer(config=config, server_client=MagicMock(spec=ServerClient), label=label)
        self.app = self.server.setup_webserver()
        cookie_name = self.server.get_session_middleware_key()
        install_session_routing(
            self.app,
            socket_dir=socket_dir,
            session_cookie=cookie_name,
            secret_key=cookie_name,
            **self.server.claimed_session_routing(),
        )
        self.client = TestClient(self.app, raise_server_exceptions=False)

    def seed(self, session_id: str = "s") -> Response:
        self.client.cookies.clear()
        body = {
            "resources_session_id": session_id,
            "episode_id": {"rollout_id": "r"},
            "task_id": {"taskset": "t", "task_id": "0"},
            "task_data": {},
        }
        return self.client.post("/ng-rollout/r/seed_session", json=body)

    def close(self, session_id: str = "s") -> Response:
        self.client.cookies.clear()
        body = {"resources_session_id": session_id, "episode_id": {"rollout_id": "r"}}
        return self.client.post("/ng-rollout/r/close_session", json=body)


@pytest.fixture
def resources_workers(socket_dir: str) -> Iterator[tuple[TypedResourcesWorker, TypedResourcesWorker]]:
    a, b = TypedResourcesWorker("a", socket_dir), TypedResourcesWorker("b", socket_dir)
    with a.client, b.client:
        yield a, b


class TestTypedResourcesSessionRouting:
    """An environment server seeds and closes a resources session by its ID, without the cookie it gets back."""

    def test_a_duplicate_typed_seed_through_the_other_worker_reaches_the_owner(self, resources_workers) -> None:
        a, b = resources_workers
        assert a.seed().status_code == 200

        retried = b.seed()

        assert retried.status_code == 200, retried.text
        assert a.server.seeds == ["s", "s"] and b.server.seeds == []
        assert set(a.server.sessions) == {"s"} and not b.server.sessions

    def test_a_cookieless_close_after_a_failed_seed_reaches_the_owner(self, resources_workers) -> None:
        a, b = resources_workers
        a.server.fail_seed = True
        assert a.seed().status_code == 500

        closed = b.close()

        assert closed.status_code == 200, closed.text
        assert closed.json()["resources_session_id"] == "s@a"
        assert a.server.closes == ["s"] and not a.server.sessions and b.server.closes == []

    def test_a_delayed_typed_seed_after_a_close_of_an_unknown_id_is_refused(self, resources_workers) -> None:
        a, b = resources_workers
        assert a.close().status_code == 200

        seeded = b.seed()

        assert seeded.status_code == 409 and "already closed" in seeded.json()["detail"]
        assert a.server.seeds == [] and b.server.seeds == []

    def test_a_claim_is_released_a_window_after_the_close(self, socket_dir: str) -> None:
        a = TypedResourcesWorker("a", socket_dir, close_window_seconds=0.05)
        b = TypedResourcesWorker("b", socket_dir, close_window_seconds=0.05)
        with a.client, b.client:
            assert b.seed().status_code == 200
            assert a.close().json()["resources_session_id"] == "s@b"
            claim = _claim_path(socket_dir, "s", RESOURCES_SESSION_CLAIMS_DIR)
            held_after_close = os.path.exists(claim)
            time.sleep(0.1)
            # The next seed or close on the owner releases the claims whose window has passed.
            assert b.seed("other").status_code == 200

        assert held_after_close and not os.path.exists(claim)
        assert os.path.exists(_claim_path(socket_dir, "other", RESOURCES_SESSION_CLAIMS_DIR))

    def test_a_seed_without_a_resources_session_id_is_served_where_it_lands(
        self, resources_workers, socket_dir: str
    ) -> None:
        a, b = resources_workers
        b.client.cookies.clear()

        seeded = b.client.post("/seed_session", json={})

        assert seeded.status_code == 200 and b.server.seeds == ["cookie session"]
        assert os.listdir(os.path.join(socket_dir, RESOURCES_SESSION_CLAIMS_DIR)) == []
