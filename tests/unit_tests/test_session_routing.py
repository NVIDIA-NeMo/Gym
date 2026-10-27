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
"""Session routing across uvicorn workers, with in-process workers that share a real socket directory.

Every test follows one rule: find the session ID, look up its claim, and forward to the claimant or handle locally.
"""

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import shutil
import socket
import tempfile
import threading
import time
from base64 import b64decode, b64encode
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional
from unittest.mock import MagicMock

import itsdangerous
import pytest
from aiohttp import ClientSession, DummyCookieJar, UnixConnector
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient
from httpx import Response
from pydantic import BaseModel, Field

from nemo_gym import session_routing
from nemo_gym._checkpoint.steps import CHECKPOINT_CLOSE_FOLLOWS_HEADER
from nemo_gym.base_resources_server import (
    RESOURCES_SESSIONS,
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
    AGENT_SESSION_COOKIE_KEY,
    AgentCloseSessionResponse,
    AgentSessionSetupError,
    AgentSessionState,
    BaseResponsesAPIAgentConfig,
    SimpleResponsesAPIAgent,
)
from nemo_gym.mcp_auto_exposure import TOKEN_HEADER, maybe_auto_expose, session_token_serializer
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from nemo_gym.session_routing import (
    SESSION_CLAIMS_DIR,
    SESSION_OWNER_HEADER,
    ResourcesSessionLifetimes,
    SessionClaims,
    SessionRoutingMiddleware,
    install_session_routing,
    serve_private_socket,
    worker_socket_path,
)


# The key earlier versions stamped the creating worker's ID under, in cookies and MCP tokens.
OLD_WORKER_STAMP = "nemo_gym_worker"


def _claim_path(socket_dir: str, session_id: str) -> str:
    return os.path.join(socket_dir, SESSION_CLAIMS_DIR, hashlib.sha256(session_id.encode("utf-8")).hexdigest())


def _claimant(socket_dir: str, session_id: str) -> Optional[str]:
    try:
        with open(_claim_path(socket_dir, session_id)) as claim:
            return claim.read()
    except FileNotFoundError:
        return None


def _claim_files(socket_dir: str) -> list[str]:
    directory = os.path.join(socket_dir, SESSION_CLAIMS_DIR)
    return os.listdir(directory) if os.path.isdir(directory) else []


def _wait_until(condition, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


def _sign_cookie(secret: str, session: dict) -> str:
    """A session cookie as Starlette's SessionMiddleware signs it."""
    return itsdangerous.TimestampSigner(secret).sign(b64encode(json.dumps(session).encode())).decode()


@pytest.fixture
def socket_dir() -> Iterator[str]:
    # Under /tmp, as in production: AF_UNIX paths are limited to about 100 bytes.
    path = tempfile.mkdtemp(prefix="ng-test-", dir="/tmp")
    yield path
    shutil.rmtree(path, ignore_errors=True)


# Resources servers.


class StoreRequest(BaseModel):
    value: int


class StateServer(SimpleResourcesServer):
    """Keeps a value per cookie session, and typed sessions per resources_session_id."""

    label: str = ""
    fail_seed: bool = False
    store: dict[str, int] = Field(default_factory=dict)
    seeds: list[str] = Field(default_factory=list)
    seed_cookie_ids: list[str] = Field(default_factory=list)
    closes: list[str] = Field(default_factory=list)
    sessions: dict[str, str] = Field(default_factory=dict)
    closed: set[str] = Field(default_factory=set)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/store")(self.put)
        app.post("/take")(self.take)
        app.get("/echo")(self.echo)
        app.post("/end")(self.end)
        app.post("/ping")(self.ping)
        return app

    async def ping(self) -> dict:
        # Keeps nothing per session, like a checkpoint control route.
        return {"worker": self.label}

    async def put(self, request: Request, body: StoreRequest) -> dict:
        self.store[request.session[SESSION_ID_KEY]] = body.value
        return {"worker": self.label}

    async def take(self, request: Request) -> dict:
        return {"worker": self.label, "value": self.store.pop(request.session[SESSION_ID_KEY], None)}

    async def echo(self, request: Request) -> dict:
        return {"worker": self.label, "query": request.url.query, "probe": request.headers.get("x-probe")}

    async def end(self, request: Request) -> dict:
        # A route that ends a session under another name, such as a Gymnasium terminal step.
        self.checkpoint_session_ended(request)
        return {"worker": self.label}

    async def seed_session(self, request: Request, body: ResourcesSeedSessionRequest | BaseSeedSessionRequest):
        self.seed_cookie_ids.append(request.session[SESSION_ID_KEY])
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
            raise RuntimeError("setup failed")
        self.sessions.setdefault(session_id, "ready")
        return ResourcesSeedSessionResponse(resources_session_id=session_id)

    async def close_resources_session(self, body: ResourcesCloseSessionRequest) -> ResourcesCloseSessionResponse:
        self.closes.append(body.resources_session_id)
        self.sessions.pop(body.resources_session_id, None)
        self.closed.add(body.resources_session_id)
        return ResourcesCloseSessionResponse(resources_session_id=f"{body.resources_session_id}@{self.label}")

    async def verify(self, request: Request) -> dict:
        return {"worker": self.label, "value": self.store.get(request.session[SESSION_ID_KEY])}


class SessionIgnoringSeedServer(StateServer):
    """Seeds with the default ``seed_session``, which never reads the session, as an agent's own ``/run`` seeds."""

    seed_session = SimpleResourcesServer.seed_session


class Worker:
    """One worker of a resources server, wired as run_webserver wires it; no socket directory means one worker."""

    def __init__(
        self,
        label: str,
        socket_dir: Optional[str],
        *,
        mcp: bool = False,
        close_window_seconds: float = 300,
        server_class: type[StateServer] = StateServer,
    ) -> None:
        config = BaseResourcesServerConfig(
            host="",
            port=0,
            entrypoint="",
            name="state",
            expose_tools_over_mcp=mcp,
            session_close_retry_window_seconds=close_window_seconds,
        )
        self.server = server_class(config=config, server_client=MagicMock(spec=ServerClient), label=label)
        self.app = self.server.setup_webserver()
        self.server.setup_liveness(self.app)
        mcp_routing = {}
        if mcp:
            maybe_auto_expose(self.server, self.app)
            mcp_routing = dict(
                mcp_token_header=TOKEN_HEADER, mcp_token_serializer=session_token_serializer(self.server)
            )
        self.cookie_name = self.server.get_session_middleware_key()
        self.socket_dir = socket_dir
        self.routing_id = install_session_routing(
            self.app,
            socket_dir=socket_dir,
            session_cookie=self.cookie_name,
            secret_key=self.cookie_name,
            **mcp_routing,
            **self.server.claimed_session_routing(),
        )
        # A failed seed answers 500, as the exception middleware does for a running server.
        self.client = TestClient(self.app, raise_server_exceptions=False)

    def session(self, response: Response) -> dict:
        signer = itsdangerous.TimestampSigner(self.cookie_name)
        return json.loads(b64decode(signer.unsign(response.cookies[self.cookie_name].encode())))

    def post(self, path: str, *, cookie: Optional[str] = None, **kwargs) -> Response:
        self.client.cookies.clear()
        cookies = {self.cookie_name: cookie} if cookie is not None else None
        return self.client.post(path, cookies=cookies, **kwargs)

    def seed(self, session_id: str = "s", *, cookie: Optional[str] = None, **task_data: Any) -> Response:
        body = {
            "resources_session_id": session_id,
            "episode_id": {"rollout_id": "r"},
            "task_id": {"taskset": "t", "task_id": "0"},
            "task_data": task_data,
        }
        return self.post("/ng-rollout/r/seed_session", json=body, cookie=cookie)

    def close(self, session_id: str = "s", *, cookie: Optional[str] = None) -> Response:
        body = {"resources_session_id": session_id, "episode_id": {"rollout_id": "r"}}
        return self.post("/ng-rollout/r/close_session", json=body, cookie=cookie)


def _socket_of(worker) -> str:
    return worker_socket_path(worker.socket_dir, worker.routing_id)


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


# Agent servers.


class _SessionAgent(SimpleResponsesAPIAgent):
    label: str = ""
    fail_seed: bool = False
    fail_close: bool = False
    seeds: list[str] = Field(default_factory=list)
    closes: list[str] = Field(default_factory=list)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        # A close under a rollout prefix, which the router routes like a bare one.
        app.post("/ng-rollout/{rollout_id}/v1/agent_sessions/close")(self.close_agent_session)
        app.get("/whoami")(self.whoami)
        return app

    async def whoami(self, request: Request) -> dict:
        return {"worker": self.label, "agent_session_id": request.session.get(AGENT_SESSION_COOKIE_KEY)}

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
        if self.fail_close:
            raise RuntimeError("close failed")
        return AgentCloseSessionResponse(agent_session_id=f"{state.request.agent_session_id}@{self.label}")


class AgentWorker:
    """One worker of an agent server, wired as run_webserver wires it; no socket directory means one worker."""

    def __init__(self, label: str, socket_dir: Optional[str], *, close_window_seconds: float = 300) -> None:
        config = BaseResponsesAPIAgentConfig(
            host="",
            port=0,
            entrypoint="",
            name="agent",
            num_workers=2 if socket_dir is not None else 1,
            session_close_retry_window_seconds=close_window_seconds,
        )
        self.agent = _SessionAgent(config=config, server_client=MagicMock(spec=ServerClient), label=label)
        self.app = self.agent.setup_webserver()
        self.cookie_name = self.agent.get_session_middleware_key()
        self.socket_dir = socket_dir
        self.routing_id = install_session_routing(
            self.app,
            socket_dir=socket_dir,
            session_cookie=self.cookie_name,
            secret_key=self.cookie_name,
            **self.agent.claimed_session_routing(),
        )
        self.client = TestClient(self.app, raise_server_exceptions=False)

    def session(self, response: Response) -> dict:
        signer = itsdangerous.TimestampSigner(self.cookie_name)
        return json.loads(b64decode(signer.unsign(response.cookies[self.cookie_name].encode())))

    def seed(self, agent_session_id: str = "s", **kwargs) -> Response:
        self.client.cookies.clear()
        return self.client.post("/v1/agent_sessions", json=_agent_seed_body(agent_session_id), **kwargs)

    def close(self, agent_session_id: str = "s", **kwargs) -> Response:
        self.client.cookies.clear()
        body = {"agent_session_id": agent_session_id, "episode_id": {"rollout_id": "r"}}
        return self.client.post("/v1/agent_sessions/close", json=body, **kwargs)


def _agent_seed_body(agent_session_id: str) -> dict:
    return {
        "agent_session_id": agent_session_id,
        "episode_id": {"rollout_id": "r"},
        "task_id": {"taskset": "t", "task_id": "0"},
    }


@pytest.fixture
def agent_workers(socket_dir: str) -> Iterator[tuple[AgentWorker, AgentWorker]]:
    a, b = AgentWorker("a", socket_dir), AgentWorker("b", socket_dir)
    with a.client, b.client:
        yield a, b


class TestTheSessionIdSources:
    """Each place a request names its session routes it to the worker that created the session."""

    def test_the_resources_cookie(self, workers) -> None:
        a, b = workers
        cookie = a.post("/store", json={"value": 7}).cookies[a.cookie_name]

        response = b.post("/take", cookie=cookie)

        assert response.json() == {"worker": "a", "value": 7}
        assert a.server.store == {} and b.server.store == {}

    def test_the_mcp_session_token(self, mcp_workers) -> None:
        a, b = mcp_workers
        token = a.post("/seed_session", json={}).json()["mcp"]["headers"][TOKEN_HEADER]

        assert _mcp_payload(_mcp_call(b.client, "store", {"value": 5}, token)) == {"worker": "a"}
        assert _mcp_payload(_mcp_call(b.client, "take", {}, token)) == {"worker": "a", "value": 5}
        assert a.server.store == {} and b.server.store == {}

    @pytest.mark.parametrize("route", ["seed", "close"])
    def test_the_typed_resources_seed_and_close_body(self, workers, route: str) -> None:
        a, b = workers
        assert a.seed().status_code == 200

        response = b.seed() if route == "seed" else b.close()

        assert response.status_code == 200, response.text
        assert a.server.seeds == (["s", "s"] if route == "seed" else ["s"]) and b.server.seeds == []
        assert a.server.closes == ([] if route == "seed" else ["s"]) and b.server.closes == []

    def test_the_agent_session_cookie(self, agent_workers) -> None:
        a, b = agent_workers
        seeded = a.seed()
        assert seeded.status_code == 200, seeded.text

        response = b.client.get("/whoami", cookies={a.cookie_name: seeded.cookies[a.cookie_name]})

        assert response.json() == {"worker": "a", "agent_session_id": "s"}

    @pytest.mark.parametrize("route", ["seed", "close"])
    def test_the_agent_seed_and_close_body(self, agent_workers, route: str) -> None:
        a, b = agent_workers
        assert a.seed().status_code == 200

        response = b.seed() if route == "seed" else b.close()

        assert response.status_code == 200, response.text
        assert a.agent.seeds == ["s"] and b.agent.seeds == []
        assert a.agent.closes == ([] if route == "seed" else ["s"]) and b.agent.closes == []
        assert not b.agent._session_records

    def test_an_agent_session_seeded_on_one_worker_is_closed_on_that_worker(self, agent_workers) -> None:
        a, b = agent_workers
        seeded = a.seed()
        assert seeded.status_code == 200, seeded.text

        closed = b.close(cookies={a.cookie_name: seeded.cookies[a.cookie_name]})

        assert closed.status_code == 200, closed.text
        assert closed.json()["agent_session_id"] == "s@a"
        assert "s" in a.agent._closed_session_records
        assert not b.agent._session_records and not b.agent._closed_session_records

    def test_a_typed_seed_gives_the_cookie_its_resources_session_id(self, workers) -> None:
        a, b = workers

        seeded = b.seed("typed")

        assert a.server.seed_cookie_ids == [] and b.server.seed_cookie_ids == ["typed"]
        assert b.session(seeded)[SESSION_ID_KEY] == "typed"
        # A later request with that cookie names the same session.
        assert a.post("/store", json={"value": 1}, cookie=seeded.cookies[b.cookie_name]).json() == {"worker": "b"}
        assert b.server.store == {"typed": 1}


class TestClaimCreation:
    """Whichever worker creates a session claims it, on every path that creates one."""

    @pytest.mark.parametrize(
        "create",
        [
            pytest.param(lambda w: w.seed("s"), id="typed seed"),
            pytest.param(lambda w: w.close("s"), id="close of an unknown id"),
        ],
    )
    def test_a_typed_seed_or_close_claims_its_id(self, workers, socket_dir: str, create) -> None:
        a, b = workers

        assert create(b).status_code == 200

        assert _claimant(socket_dir, "s") == b.routing_id

    def test_a_typed_seed_with_its_own_cookie_keeps_its_claim(self, workers, socket_dir: str) -> None:
        a, b = workers
        seeded = b.seed()

        retried = a.seed(cookie=seeded.cookies[b.cookie_name])

        assert retried.status_code == 200
        assert _claimant(socket_dir, "s") == b.routing_id and b.server.seeds == ["s", "s"]

    @pytest.mark.parametrize("path", ["/seed_session", "/store"])
    def test_a_new_session_is_claimed_by_the_worker_that_mints_its_id(self, workers, socket_dir: str, path) -> None:
        a, b = workers
        body = {} if path == "/seed_session" else {"value": 1}

        response = b.post(path, json=body)

        session_id = b.session(response)[SESSION_ID_KEY]
        assert _claimant(socket_dir, session_id) == b.routing_id
        assert len(_claim_files(socket_dir)) == 1

    def test_a_post_whose_handler_keeps_no_session_leaves_no_claim(self, workers, socket_dir: str) -> None:
        a, b = workers

        assert b.post("/ping").json() == {"worker": "b"}
        metrics = b.post("/aggregate_metrics", json={"verify_responses": []})
        assert metrics.status_code == 200, metrics.text

        assert _claim_files(socket_dir) == []
        assert len(b.app.state.nemo_gym_resources_session_lifetimes) == 0

    def test_a_post_whose_handler_keeps_no_session_gets_no_session_cookie(self, workers) -> None:
        a, b = workers

        for response in (b.post("/ping"), b.post("/aggregate_metrics", json={"verify_responses": []})):
            assert response.status_code == 200, response.text
            # A cookie would name a session nobody claims, whose later requests could reach any worker.
            assert _set_cookies(response, b.cookie_name) == []

    def test_a_seed_whose_handler_ignores_the_session_still_claims_it(self, socket_dir: str) -> None:
        a = Worker("a", socket_dir, server_class=SessionIgnoringSeedServer)
        b = Worker("b", socket_dir, server_class=SessionIgnoringSeedServer)
        with a.client, b.client:
            seeded = a.post("/seed_session", json={})
            cookie = seeded.cookies[a.cookie_name]
            session_id = a.session(seeded)[SESSION_ID_KEY]

            stored = b.post("/store", json={"value": 3}, cookie=cookie)
            taken = b.post("/take", cookie=cookie)

        assert seeded.status_code == 200 and _claimant(socket_dir, session_id) == a.routing_id
        # Every later request of the session reaches the worker that seeded it.
        assert stored.json() == {"worker": "a"}
        assert taken.json() == {"worker": "a", "value": 3}

    def test_a_cookie_whose_claim_ended_is_claimed_again_by_the_worker_that_uses_it(self, socket_dir: str) -> None:
        a = Worker("a", socket_dir, close_window_seconds=0.1)
        b = Worker("b", socket_dir, close_window_seconds=0.1)
        with a.client, b.client:
            stored = a.post("/store", json={"value": 7})
            cookie = stored.cookies[a.cookie_name]
            session_id = a.session(stored)[SESSION_ID_KEY]
            _wait_until(lambda: _claimant(socket_dir, session_id) is None)

            again = b.post("/store", json={"value": 9}, cookie=cookie)
            claimant = _claimant(socket_dir, session_id)
            taken = a.post("/take", cookie=cookie)

        assert again.json() == {"worker": "b"} and claimant == b.routing_id
        # The session's next request reaches the worker that claimed it again, not the one that kept a stale copy.
        assert taken.json() == {"worker": "b", "value": 9}

    def test_a_new_session_is_claimed_before_its_reply_arrives(self, workers, socket_dir: str) -> None:
        a, b = workers

        stored = b.post("/store", json={"value": 5})

        # Claimed by the time the client has the reply, so its next request through another worker finds it.
        assert _claimant(socket_dir, b.session(stored)[SESSION_ID_KEY]) == b.routing_id
        assert a.post("/take", cookie=stored.cookies[b.cookie_name]).json() == {"worker": "b", "value": 5}

    def test_a_minted_id_another_worker_claimed_meanwhile_is_refused(
        self, workers, socket_dir: str, monkeypatch
    ) -> None:
        a, b = workers
        monkeypatch.setattr("nemo_gym.session_routing.uuid4", lambda: "taken-id")
        assert a.app.state.nemo_gym_session_claims.claim("taken-id") == a.routing_id

        response = b.post("/store", json={"value": 5})

        assert response.status_code == 409
        assert _claimant(socket_dir, "taken-id") == a.routing_id

    def test_a_get_without_a_session_claims_nothing(self, workers, socket_dir: str) -> None:
        a, b = workers

        for path in ("/echo", "/health"):
            assert b.client.get(path).status_code == 200

        assert _claim_files(socket_dir) == []

    @pytest.mark.parametrize("create", ["seed", "close"])
    def test_an_agent_seed_or_close_claims_its_id(self, agent_workers, socket_dir: str, create: str) -> None:
        a, b = agent_workers

        response = b.seed() if create == "seed" else b.close()

        assert response.status_code == 200
        assert _claimant(socket_dir, "s") == b.routing_id

    def test_an_agent_request_without_a_session_claims_nothing(self, agent_workers, socket_dir: str) -> None:
        a, b = agent_workers

        assert b.client.get("/whoami").json() == {"worker": "b", "agent_session_id": None}

        assert _claim_files(socket_dir) == []

    def test_two_workers_racing_to_claim_ids_agree_on_one_claimant_each(self, socket_dir: str) -> None:
        claims = [SessionClaims(socket_dir, routing_id, lambda _: False) for routing_id in ("a" * 32, "b" * 32)]
        ids = [f"id-{index}" for index in range(500)]

        def claim_all(worker: SessionClaims) -> list[Optional[str]]:
            return [worker.claim(session_id) for session_id in ids]

        with ThreadPoolExecutor(2) as pool:
            first, second = pool.map(claim_all, claims)

        assert first == second
        for session_id, claimant in zip(ids, first):
            assert _claimant(socket_dir, session_id) == claimant
        for worker in claims:
            assert worker._mine == {session_id for session_id, c in zip(ids, first) if c == worker.routing_id}
        # Only claim files remain: every temporary file was removed.
        assert len(_claim_files(socket_dir)) == len(ids)

    def test_new_sessions_minted_on_two_workers_at_once_are_distinct(self, workers, socket_dir: str) -> None:
        a, b = workers

        def mint(worker: Worker) -> list[str]:
            client = TestClient(worker.app)
            cookies = []
            for _ in range(50):
                client.cookies.clear()
                cookies.append(client.post("/store", json={"value": 1}).cookies[worker.cookie_name])
            return cookies

        with ThreadPoolExecutor(2) as pool:
            cookies_a, cookies_b = pool.map(mint, (a, b))

        signer = itsdangerous.TimestampSigner(a.cookie_name)
        ids = [json.loads(b64decode(signer.unsign(c.encode())))[SESSION_ID_KEY] for c in cookies_a + cookies_b]
        assert len(set(ids)) == 100 and len(_claim_files(socket_dir)) == 100
        assert len(a.server.store) == 50 and len(b.server.store) == 50


class TestClaimLifetime:
    """A claim ends once its session's last record on the claimant is gone; one worker writes no claims."""

    def test_a_typed_session_keeps_its_claim_while_open_and_releases_it_a_window_after_its_close(
        self, socket_dir: str
    ) -> None:
        a = Worker("a", socket_dir, close_window_seconds=0.2)
        b = Worker("b", socket_dir, close_window_seconds=0.2)
        with a.client, b.client:
            assert b.seed().status_code == 200
            time.sleep(0.4)
            open_after_window = _claimant(socket_dir, "s")
            assert a.close().json()["resources_session_id"] == "s@b"
            closed_now = _claimant(socket_dir, "s")
            # No further request: the claimant's sweep releases it.
            _wait_until(lambda: _claimant(socket_dir, "s") is None)

        assert open_after_window == closed_now == b.routing_id
        assert not b.app.state.nemo_gym_session_claims._mine

    def test_a_new_session_keeps_its_claim_until_verify_and_a_window_after(self, socket_dir: str) -> None:
        a = Worker("a", socket_dir, close_window_seconds=0.2)
        b = Worker("b", socket_dir, close_window_seconds=0.2)
        with a.client, b.client:
            seeded = b.post("/seed_session", json={})
            session_id = b.session(seeded)[SESSION_ID_KEY]
            cookie = seeded.cookies[b.cookie_name]
            time.sleep(0.4)
            assert a.post("/store", json={"value": 3}, cookie=cookie).json() == {"worker": "b"}
            verified = a.post("/verify", json={}, cookie=cookie)
            assert verified.json() == {"worker": "b", "value": 3}
            claimed_after_verify = _claimant(socket_dir, session_id)
            _wait_until(lambda: _claimant(socket_dir, session_id) is None)

        assert claimed_after_verify == b.routing_id

    def test_a_verify_that_says_a_close_follows_keeps_the_session_open_until_the_close(self, socket_dir: str) -> None:
        a = Worker("a", socket_dir, close_window_seconds=0.2)
        b = Worker("b", socket_dir, close_window_seconds=0.2)
        with a.client, b.client:
            cookie = b.seed().cookies[b.cookie_name]
            # An environment server verifies, then closes in its final cleanup, which can come much later.
            verified = a.post("/verify", json={}, cookie=cookie, headers={CHECKPOINT_CLOSE_FOLLOWS_HEADER: "1"})
            assert verified.json()["worker"] == "b"
            time.sleep(0.5)
            claimed_before_close = _claimant(socket_dir, "s")

            closed = a.close(cookie=cookie)

            assert closed.json()["resources_session_id"] == "s@b"
            _wait_until(lambda: _claimant(socket_dir, "s") is None)
        assert claimed_before_close == b.routing_id and b.server.closes == ["s"] and a.server.closes == []

    def test_a_session_ended_under_another_route_releases_its_claim(self, socket_dir: str) -> None:
        a = Worker("a", socket_dir, close_window_seconds=0.2)
        b = Worker("b", socket_dir, close_window_seconds=0.2)
        with a.client, b.client:
            seeded = b.post("/seed_session", json={})
            session_id = b.session(seeded)[SESSION_ID_KEY]
            assert a.post("/end", cookie=seeded.cookies[b.cookie_name]).json() == {"worker": "b"}
            _wait_until(lambda: _claimant(socket_dir, session_id) is None)

    def test_a_session_started_by_another_request_keeps_its_claim_a_window_after_each_request(
        self, socket_dir: str
    ) -> None:
        a = Worker("a", socket_dir, close_window_seconds=0.3)
        b = Worker("b", socket_dir, close_window_seconds=0.3)
        with a.client, b.client:
            stored = b.post("/store", json={"value": 1})
            session_id = b.session(stored)[SESSION_ID_KEY]
            cookie = stored.cookies[b.cookie_name]
            for _ in range(4):
                time.sleep(0.15)
                assert a.post("/store", json={"value": 2}, cookie=cookie).json() == {"worker": "b"}
            still_claimed = _claimant(socket_dir, session_id)
            _wait_until(lambda: _claimant(socket_dir, session_id) is None)

        assert still_claimed == b.routing_id

    def test_a_failed_typed_seed_keeps_its_claim_for_the_cleanup_close_then_releases_it(self, socket_dir: str) -> None:
        a = Worker("a", socket_dir, close_window_seconds=0.3)
        b = Worker("b", socket_dir, close_window_seconds=0.3)
        with a.client, b.client:
            a.server.fail_seed = True
            failed = a.seed()
            assert failed.status_code == 500 and a.cookie_name not in failed.cookies

            closed = b.close()

            assert closed.status_code == 200, closed.text
            assert closed.json()["resources_session_id"] == "s@a"
            assert a.server.closes == ["s"] and not a.server.sessions and b.server.closes == []
            _wait_until(lambda: _claimant(socket_dir, "s") is None)

    def test_a_delayed_typed_seed_after_a_close_of_an_unknown_id_is_refused(self, workers) -> None:
        a, b = workers
        assert a.close().status_code == 200

        seeded = b.seed()

        assert seeded.status_code == 409 and "already closed" in seeded.json()["detail"]
        assert a.server.seeds == [] and b.server.seeds == []

    def test_lifetimes_keep_one_heap_entry_per_session(self, monkeypatch) -> None:
        clock = [0.0]
        monkeypatch.setattr("nemo_gym.session_routing.monotonic", lambda: clock[0])
        lifetimes = ResourcesSessionLifetimes(10, starts=("/seed_session",), ends=("/close_session",))

        for step in range(1000):
            clock[0] = step / 100
            lifetimes.served("s", {"path": "/store", "headers": []}, 200)

        assert len(lifetimes._expiries) == 1 and lifetimes.holds("s")
        clock[0] = 19.98
        assert lifetimes.sweep() == [] and lifetimes.holds("s")
        clock[0] = 20.0
        assert lifetimes.sweep() == ["s"] and not lifetimes._expiries and len(lifetimes) == 0

    def test_an_agent_claim_is_released_once_the_owner_drops_the_closed_record(self, socket_dir: str) -> None:
        a = AgentWorker("a", socket_dir, close_window_seconds=0.1)
        b = AgentWorker("b", socket_dir, close_window_seconds=0.1)
        with a.client, b.client:
            assert a.seed().status_code == 200
            assert b.close().json()["agent_session_id"] == "s@a"
            claimed_after_close = _claimant(socket_dir, "s")
            # No further request: the claimant's sweep drops the receipt and releases the claim.
            _wait_until(lambda: _claimant(socket_dir, "s") is None)
            assert "s" not in a.agent._session_records
            # With the claim gone, the ID is new again to whichever worker sees it first.
            assert b.seed().status_code == 200
        assert claimed_after_close == a.routing_id
        assert b.agent.seeds == ["s"] and a.agent.seeds == ["s"]

    def test_an_agent_failed_seed_that_nobody_closes_is_released_after_the_window(self, socket_dir: str) -> None:
        a = AgentWorker("a", socket_dir, close_window_seconds=0.1)
        b = AgentWorker("b", socket_dir, close_window_seconds=0.1)
        with a.client, b.client:
            a.agent.fail_seed = True
            assert a.seed().status_code == 500

            _wait_until(lambda: "s" not in a.agent._session_records)
            assert a.agent.closes == ["s"]
            _wait_until(lambda: _claimant(socket_dir, "s") is None)
            assert not a.agent._failed_seed_records

    @pytest.mark.parametrize("kind", ["resources", "agent"])
    def test_one_worker_writes_no_claims(self, socket_dir: str, kind: str) -> None:
        single = Worker("single", None) if kind == "resources" else AgentWorker("single", None)
        with single.client:
            if kind == "resources":
                assert single.seed().status_code == 200
                assert single.post("/store", json={"value": 1}).status_code == 200
                assert single.close().status_code == 200
                assert not hasattr(single.app.state, "nemo_gym_session_claims")
            else:
                assert single.seed().status_code == 200
                assert single.close().status_code == 200
                assert single.agent._agent_session_claims is None
                # An agent with one worker has nothing to route.
                assert not any(m.cls is SessionRoutingMiddleware for m in single.app.user_middleware)

        assert single.routing_id is None
        assert os.listdir(socket_dir) == []

    def test_one_worker_gives_a_typed_seed_its_id_in_the_cookie(self) -> None:
        single = Worker("single", None)
        with single.client:
            seeded = single.seed("typed")

        assert single.session(seeded)[SESSION_ID_KEY] == "typed" and single.server.seed_cookie_ids == ["typed"]


class TestRollback:
    """A seed that fails to create its session leaves no claim behind for longer than the window."""

    def test_an_agent_seed_that_creates_nothing_releases_its_claim_at_once(
        self, agent_workers, socket_dir: str
    ) -> None:
        a, b = agent_workers
        b.agent._seed_agent_session_state = MagicMock(side_effect=RuntimeError("bad grant"))

        assert b.seed().status_code == 500

        assert _claimant(socket_dir, "s") is None and not b.agent._session_records

    def test_an_agent_seed_whose_session_cannot_be_registered_is_undone(self, agent_workers, socket_dir: str) -> None:
        a, b = agent_workers
        # The checkpoint participant refuses the session after its state was set up, as for a retired attempt.
        b.agent._restart_only = MagicMock()
        b.agent._restart_only.seeding = lambda *args: contextlib.nullcontext()
        b.agent._restart_only.open_session.side_effect = RuntimeError("attempt retired")

        response = b.seed()

        assert response.status_code == 500
        assert b.agent.closes == ["s"]
        assert not b.agent._session_records and _claimant(socket_dir, "s") is None
        # The ID is new again, to either worker.
        assert a.seed().status_code == 200 and a.agent.seeds == ["s"]

    def test_an_undo_whose_close_hook_fails_keeps_the_state_for_the_callers_close(
        self, agent_workers, socket_dir: str
    ) -> None:
        a, b = agent_workers
        b.agent._restart_only = MagicMock()
        b.agent._restart_only.seeding = lambda *args: contextlib.nullcontext()
        b.agent._restart_only.open_session.side_effect = RuntimeError("attempt retired")
        b.agent.fail_close = True

        assert b.seed().status_code == 500

        assert b.agent._session_records["s"].state is not None and "s" in b.agent._failed_seed_records
        assert _claimant(socket_dir, "s") == b.routing_id
        b.agent.fail_close = False
        closed = a.close()
        assert closed.status_code == 200 and closed.json()["agent_session_id"] == "s@b"


class TestConflictsAndLostWorkers:
    def test_a_typed_seed_whose_cookie_names_another_session_is_refused(self, workers) -> None:
        a, b = workers
        other = a.seed("other").cookies[a.cookie_name]

        response = b.seed("s", cookie=other)

        assert response.status_code == 409 and "names another session" in response.json()["detail"]
        assert a.server.seeds == ["other"] and b.server.seeds == []

    def test_one_worker_refuses_a_typed_seed_whose_cookie_names_another_session(self) -> None:
        single = Worker("single", None)
        with single.client:
            other = single.seed("other").cookies[single.cookie_name]

            response = single.seed("s", cookie=other)

        assert response.status_code == 409 and single.server.seeds == ["other"]

    def test_an_agent_seed_whose_cookie_names_another_session_is_refused(self, agent_workers) -> None:
        a, b = agent_workers
        other = a.seed("other").cookies[a.cookie_name]

        response = b.seed("s", cookies={a.cookie_name: other})

        assert response.status_code == 409
        assert a.agent.seeds == ["other"] and b.agent.seeds == []

    def test_a_session_whose_worker_exited_gets_an_explicit_error(self, socket_dir: str) -> None:
        a, b = Worker("a", socket_dir), Worker("b", socket_dir)
        with b.client:
            with a.client:
                cookie = a.post("/store", json={"value": 7}).cookies[a.cookie_name]
            assert not os.path.exists(_socket_of(a))

            response = b.post("/take", cookie=cookie)

        assert response.status_code == 410
        assert "no longer running" in response.json()["detail"]

    def test_a_crashed_worker_left_socket_gets_an_explicit_error(self, socket_dir: str) -> None:
        a, b = Worker("a", socket_dir), Worker("b", socket_dir)
        with a.client:
            cookie = a.post("/store", json={"value": 7}).cookies[a.cookie_name]
        # A worker killed outright leaves its socket file behind with nothing listening.
        with socket.socket(socket.AF_UNIX) as stale:
            stale.bind(_socket_of(a))

        with b.client:
            response = b.post("/take", cookie=cookie)

        assert response.status_code == 410

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

    def test_an_mcp_token_of_an_exited_worker_gets_an_explicit_error(self, socket_dir: str) -> None:
        a, b = Worker("a", socket_dir, mcp=True), Worker("b", socket_dir, mcp=True)
        with b.client:
            with a.client:
                token = a.post("/seed_session", json={}).json()["mcp"]["headers"][TOKEN_HEADER]
            b.client.cookies.clear()
            response = b.client.post(
                "/mcp",
                headers={"accept": "application/json, text/event-stream", TOKEN_HEADER: token},
                json={"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "take", "arguments": {}}},
            )

        assert response.status_code == 410

    def test_a_badly_signed_cookie_is_handled_locally(self, workers, socket_dir: str) -> None:
        a, b = workers
        forged = _sign_cookie("another secret", {SESSION_ID_KEY: "s"})

        response = b.post("/store", json={"value": 3}, cookie=forged)

        assert response.json() == {"worker": "b"}
        assert _claimant(socket_dir, b.session(response)[SESSION_ID_KEY]) == b.routing_id

    def test_a_badly_signed_mcp_token_is_handled_locally_and_refused(self, mcp_workers) -> None:
        a, b = mcp_workers
        forged = itsdangerous.URLSafeSerializer("another secret", salt="x").dumps({"sid": "s", "tools": None})

        result = _mcp_call(b.client, "store", {"value": 5}, forged)

        assert result["isError"] is True and "Invalid Gym MCP session token" in result["content"][0]["text"]
        assert a.server.store == {} and b.server.store == {}


class TestPrivateSockets:
    def test_method_path_query_and_headers_are_forwarded_unchanged(self, workers) -> None:
        a, b = workers
        cookie = a.post("/store", json={"value": 1}).cookies[a.cookie_name]

        b.client.cookies.clear()
        response = b.client.get("/echo?x=1&y=a%20b", headers={"x-probe": "p"}, cookies={a.cookie_name: cookie})

        assert response.json() == {"worker": "a", "query": "x=1&y=a%20b", "probe": "p"}

    def test_only_the_claimants_set_cookie_comes_back(self, workers) -> None:
        a, b = workers
        stored = a.post("/store", json={"value": 7})

        response = b.post("/take", cookie=stored.cookies[a.cookie_name])

        [set_cookie] = _set_cookies(response, a.cookie_name)
        assert "path=/" in set_cookie
        assert a.session(response)[SESSION_ID_KEY] == a.session(stored)[SESSION_ID_KEY]

    def test_a_forwarded_request_is_always_handled_where_it_arrives(self, workers) -> None:
        a, b = workers
        cookie = a.post("/store", json={"value": 7}).cookies[a.cookie_name]

        async def take_over_private_socket() -> dict:
            connector = UnixConnector(_socket_of(b))
            async with ClientSession(connector=connector, cookie_jar=DummyCookieJar()) as client:
                async with client.post("http://localhost/take", cookies={a.cookie_name: cookie}) as response:
                    return await response.json()

        # Arriving over b's private socket, the request runs on b even though a claims the session.
        assert asyncio.run(take_over_private_socket()) == {"worker": "b", "value": None}
        assert list(a.server.store.values()) == [7]

    def test_a_seed_forwarded_after_its_claim_moved_is_refused(self, workers, socket_dir: str) -> None:
        a, b = workers
        # a claims the ID; the seed then reaches b's private socket, as if b had held the claim and released it.
        assert a.seed().status_code == 200
        body = {
            "resources_session_id": "s",
            "episode_id": {"rollout_id": "r"},
            "task_id": {"taskset": "t", "task_id": "0"},
            "task_data": {},
        }

        async def seed_over_private_socket() -> int:
            connector = UnixConnector(_socket_of(b))
            async with ClientSession(connector=connector, cookie_jar=DummyCookieJar()) as client:
                async with client.post("http://localhost/seed_session", json=body) as response:
                    return response.status

        assert asyncio.run(seed_over_private_socket()) == 409
        assert b.server.seeds == [] and a.server.seeds == ["s"]


class TestOldStamps:
    """A cookie or token from before this change still carries a worker ID, which is ignored for its session ID."""

    def test_an_old_cookie_routes_by_its_session_id_not_its_stamp(self, workers) -> None:
        a, b = workers
        stored = a.post("/store", json={"value": 7})
        session_id = a.session(stored)[SESSION_ID_KEY]
        # Stamped with b, while a holds the session.
        old = _sign_cookie(a.cookie_name, {SESSION_ID_KEY: session_id, OLD_WORKER_STAMP: b.routing_id})

        response = b.post("/take", cookie=old)

        assert response.json() == {"worker": "a", "value": 7}

    def test_an_old_cookie_of_a_dead_worker_without_a_claim_is_handled_locally(self, workers) -> None:
        a, b = workers
        old = _sign_cookie(a.cookie_name, {SESSION_ID_KEY: "from-before", OLD_WORKER_STAMP: "f" * 32})

        assert b.post("/store", json={"value": 1}, cookie=old).json() == {"worker": "b"}
        assert b.server.store == {"from-before": 1}

    def test_an_old_mcp_token_routes_by_its_session_id(self, mcp_workers) -> None:
        a, b = mcp_workers
        token = a.post("/seed_session", json={}).json()["mcp"]["headers"][TOKEN_HEADER]
        payload = session_token_serializer(a.server).loads(token)
        assert OLD_WORKER_STAMP not in payload
        old = session_token_serializer(a.server).dumps({**payload, OLD_WORKER_STAMP: b.routing_id})

        assert _mcp_payload(_mcp_call(b.client, "store", {"value": 5}, old)) == {"worker": "a"}

    def test_an_mcp_token_from_a_single_worker_server_is_handled_locally(self, mcp_workers) -> None:
        a, b = mcp_workers
        token = session_token_serializer(a.server).dumps({"sid": "s", "tools": None})

        assert _mcp_payload(_mcp_call(b.client, "store", {"value": 5}, token)) == {"worker": "b"}
        assert b.server.store == {"s": 5}


class TestRaces:
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
            claimed_fresh = _claimant(fresh_dir, "s")
        finally:
            shutil.rmtree(fresh_dir, ignore_errors=True)

        # As on one worker after a restart: the close of an ID nobody here has seen is answered with a receipt.
        assert closed.status_code == 200 and closed.json()["agent_session_id"] == "s"
        assert claimed_fresh == d.routing_id and not c.agent._session_records

    def test_a_seed_and_a_close_of_one_id_on_two_workers_at_once_run_on_one_worker(self, socket_dir: str) -> None:
        for attempt in range(20):
            session_id = f"race-{attempt}"
            a, b = Worker("a", socket_dir), Worker("b", socket_dir)
            with a.client, b.client:
                barrier = threading.Barrier(2)

                def seed() -> int:
                    barrier.wait()
                    return a.seed(session_id).status_code

                def close() -> int:
                    barrier.wait()
                    return b.close(session_id).status_code

                with ThreadPoolExecutor(2) as pool:
                    seeded, closed = pool.submit(seed), pool.submit(close)
                    statuses = (seeded.result(), closed.result())

            served = [w for w in (a, b) if w.server.seeds or w.server.closes]
            assert len(served) == 1, statuses
            assert closed.result() == 200 and seeded.result() in (200, 409)

    def test_a_claim_released_while_a_request_is_forwarded_is_handled_by_the_former_claimant(
        self, socket_dir: str
    ) -> None:
        a = Worker("a", socket_dir, close_window_seconds=0.1)
        b = Worker("b", socket_dir, close_window_seconds=0.1)
        with a.client, b.client:
            stored = a.post("/store", json={"value": 7})
            session_id = a.session(stored)[SESSION_ID_KEY]
            _wait_until(lambda: _claimant(socket_dir, session_id) is None)
            # b read the claim just before the release and forwards to a anyway.
            b.app.state.nemo_gym_session_claims.claimant = lambda _: a.routing_id

            response = b.post("/take", cookie=stored.cookies[a.cookie_name])

        assert response.json() == {"worker": "a", "value": 7}
        # The former claimant served it without claiming the ID again.
        assert _claimant(socket_dir, session_id) is None


class TestCheckpointRestores:
    """A partial-rollout checkpoint restore moves a session to another worker by rewriting its claim."""

    def test_an_old_cookie_follows_its_session_to_the_worker_a_restore_moved_it_to(self, workers, socket_dir) -> None:
        a, b = workers
        stored = a.post("/store", json={"value": 7})
        session_id = a.session(stored)[SESSION_ID_KEY]
        # The restore installs the session's state on b, which claims it; a drops its own claim.
        b.server.store[session_id] = a.server.store.pop(session_id)
        b.app.state.nemo_gym_session_claims.rewrite(session_id)
        a.app.state.nemo_gym_session_claims.disown(session_id)

        taken = a.post("/take", cookie=stored.cookies[a.cookie_name])

        assert _claimant(socket_dir, session_id) == b.routing_id
        assert taken.json() == {"worker": "b", "value": 7}

    def test_a_release_after_a_restore_moved_the_claim_leaves_the_new_claim(self, socket_dir: str) -> None:
        a = Worker("a", socket_dir, close_window_seconds=0.1)
        b = Worker("b", socket_dir)
        with a.client, b.client:
            stored = a.post("/store", json={"value": 7})
            session_id = a.session(stored)[SESSION_ID_KEY]
            # b rewrites the claim before a has dropped it, and a's own window then ends.
            b.app.state.nemo_gym_session_claims.rewrite(session_id)
            _wait_until(lambda: not a.app.state.nemo_gym_session_claims.mine(session_id))

            assert _claimant(socket_dir, session_id) == b.routing_id

    def test_a_release_cannot_delete_a_claim_a_restore_rewrites_while_it_releases(self, socket_dir: str) -> None:
        former = SessionClaims(socket_dir, "a" * 32, lambda _: False)
        holder = SessionClaims(socket_dir, "b" * 32, lambda _: False)
        assert former.claim("s") == former.routing_id
        read_claimant = former.claimant
        rewrites: list[threading.Thread] = []

        def rewritten_right_after_the_check(session_id: str) -> Optional[str]:
            # The release has found its own claim; the restore on another worker rewrites it now, before the unlink.
            found = read_claimant(session_id)
            rewrite = threading.Thread(target=holder.rewrite, args=(session_id,))
            rewrite.start()
            rewrites.append(rewrite)
            # Long enough for the rewrite to finish, unless the release holds it off until it is done.
            rewrite.join(timeout=0.5)
            return found

        former.claimant = rewritten_right_after_the_check
        former.release("s")
        for rewrite in rewrites:
            rewrite.join()

        assert rewrites and _claimant(socket_dir, "s") == holder.routing_id

    async def test_a_claim_on_this_worker_during_a_batched_rewrite_of_the_same_id_does_not_fail_the_rewrite(
        self, socket_dir: str
    ) -> None:
        holder = SessionClaims(socket_dir, "b" * 32, lambda _: True)
        real_exclusive = holder._exclusive

        @contextlib.contextmanager
        def claimed_meanwhile() -> Iterator[None]:
            # A stale seed naming the restored session reaches this worker while the restore's thread rewrites it.
            assert holder.claim("s") == holder.routing_id
            with real_exclusive():
                yield

        holder._exclusive = claimed_meanwhile
        await holder.rewrite_all(["s"])

        assert _claimant(socket_dir, "s") == holder.routing_id
        claims_dir = os.path.dirname(holder._path("s"))
        assert not [name for name in os.listdir(claims_dir) if name.endswith(".tmp")]

    async def test_a_batched_rewrite_whose_caller_is_cancelled_finishes_before_the_worker_counts_its_claims(
        self, socket_dir: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(session_routing, "_REWRITE_BATCH", 10)
        former = SessionClaims(socket_dir, "a" * 32, lambda _: False)
        holder = SessionClaims(socket_dir, "b" * 32, lambda _: True)
        session_ids = [f"s{index}" for index in range(200)]
        for session_id in session_ids:
            former.claim(session_id)
        renamed = threading.Event()
        real_replace = os.replace

        def slow_replace(source: str, target: str) -> None:
            real_replace(source, target)
            renamed.set()
            time.sleep(0.001)

        monkeypatch.setattr(session_routing.os, "replace", slow_replace)
        rewriting = asyncio.create_task(holder.rewrite_all(session_ids))
        await asyncio.to_thread(renamed.wait, 5)
        # The caller is cancelled while the thread is part way through.
        mine_while_rewriting = len(holder)
        rewriting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await rewriting
        monkeypatch.setattr(session_routing.os, "replace", real_replace)

        # The thread ran to the end, and only then did the worker count the claims as its own: every one of them.
        assert mine_while_rewriting == 0
        assert len(holder) == len(session_ids)
        assert all(_claimant(socket_dir, session_id) == holder.routing_id for session_id in session_ids)

    def test_a_request_without_a_session_goes_to_the_worker_its_header_names(self, agent_workers) -> None:
        a, b = agent_workers

        named = b.client.get("/whoami", headers={SESSION_OWNER_HEADER: a.routing_id})
        own = b.client.get("/whoami")
        gone = b.client.get("/whoami", headers={SESSION_OWNER_HEADER: "0" * 32})

        assert named.json()["worker"] == "a" and own.json()["worker"] == "b"
        # A worker that is gone, such as one that exited mid-/run, gives a 410 rather than another worker's answer.
        assert gone.status_code == 410


class TestBodies:
    def test_a_typed_seed_with_large_task_data_is_routed_to_the_claimant(self, workers) -> None:
        a, b = workers
        assert a.seed().status_code == 200

        # Large enough to arrive in several receive messages, as a seed with images in its task data would.
        retried = b.seed(image="x" * (9 * 1024 * 1024))

        assert retried.status_code == 200, retried.text
        assert a.server.seeds == ["s", "s"] and b.server.seeds == [] and not b.server.sessions

    def test_a_large_seed_without_a_resources_session_id_starts_a_session_where_it_lands(
        self, workers, socket_dir: str
    ) -> None:
        a, b = workers
        # An agent's own /run seed sends its whole run request, which a multimodal task can make this large.
        body = json.dumps({"responses_create_params": {"input": "x" * (9 * 1024 * 1024)}}).encode()

        seeded = b.post("/seed_session", content=body, headers={"content-type": "application/json"})

        assert seeded.status_code == 200 and b.server.seeds == ["cookie session"] and a.server.seeds == []
        assert _claimant(socket_dir, b.session(seeded)[SESSION_ID_KEY]) == b.routing_id

    @pytest.mark.parametrize(
        ("body", "status"),
        [
            (b"not json", 422),
            (b"[1, 2]", 422),
            # No typed ID, so the app takes these as seeds of the cookie session.
            (b'{"resources_session_id": 5}', 200),
            (b'{"resources_session_id": ""}', 200),
        ],
    )
    def test_a_seed_without_a_valid_typed_id_is_served_where_it_lands(
        self, socket_dir: str, body: bytes, status: int
    ) -> None:
        a = Worker("a", socket_dir, close_window_seconds=0.1)
        b = Worker("b", socket_dir, close_window_seconds=0.1)
        with a.client, b.client:
            response = b.post("/seed_session", content=body, headers={"content-type": "application/json"})
            assert response.status_code == status
            claims = _claim_files(socket_dir)
            # A seed claims its session even when the app refuses its body, since its reply's cookie names it;
            # a refused seed's claim ends after the window.
            if status != 200:
                _wait_until(lambda: _claim_files(socket_dir) == [])
        assert len(claims) == 1 and a.server.seeds == []

    @pytest.mark.parametrize(
        "body", [b"not json", b"[1, 2]", b'{"agent_session_id": 5}', b'{"episode_id": {"rollout_id": "r"}}']
    )
    def test_an_agent_body_without_a_string_id_is_refused_by_the_app_and_claims_nothing(
        self, agent_workers, socket_dir: str, body: bytes
    ) -> None:
        a, b = agent_workers

        response = b.client.post("/v1/agent_sessions", content=body, headers={"content-type": "application/json"})

        assert response.status_code == 422
        assert _claim_files(socket_dir) == []
        assert not a.agent._session_records and not b.agent._session_records

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


class TestPrivateSocketBacklog:
    @pytest.fixture
    def backlogs(self, monkeypatch: pytest.MonkeyPatch) -> list[int]:
        """The listen backlog of every uvicorn server session routing configures."""
        seen: list[int] = []
        real_config = session_routing.uvicorn.Config

        def config(*args: Any, **kwargs: Any) -> Any:
            built = real_config(*args, **kwargs)
            seen.append(built.backlog)
            return built

        monkeypatch.setattr(session_routing.uvicorn, "Config", config)
        return seen

    def test_the_private_socket_listens_with_the_configured_backlog(
        self, socket_dir: str, backlogs: list[int]
    ) -> None:
        app = FastAPI()
        install_session_routing(
            app,
            kind=RESOURCES_SESSIONS,
            socket_dir=socket_dir,
            session_cookie="c",
            secret_key="c",
            close_window_seconds=1.0,
            backlog=1234,
        )

        with TestClient(app):
            pass

        assert backlogs == [1234]

    def test_the_private_socket_backlog_defaults_to_the_gym_default(
        self, socket_dir: str, backlogs: list[int]
    ) -> None:
        async def serve() -> None:
            async with serve_private_socket(FastAPI(), os.path.join(socket_dir, "private.sock")):
                pass

        asyncio.run(serve())

        assert backlogs == [16384]
