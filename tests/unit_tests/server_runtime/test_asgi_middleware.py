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
"""Gym's request middleware calls the app directly, in the request's own task.

These tests cover what a middleware written that way must still get right.
That is the replies it settles or replaces before they start, failures after a reply starts, and the request's own
cancellation.
It also covers context a route leaves behind on a reused connection.
"""

import asyncio
import re
import socket
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import AsyncIterator, Iterator
from unittest.mock import MagicMock

import pytest
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from starlette.requests import ClientDisconnect

from nemo_gym.client.server_client import SESSION_ID_KEY
from nemo_gym.server_runtime.base_server import SimpleServer
from nemo_gym.server_runtime.request_handling.disconnect import ClientDisconnectCancellationMiddleware
from nemo_gym.server_runtime.request_handling.request_context import install_request_context_isolation
from nemo_gym.server_runtime.request_handling.session_routing import SESSION_ACQUIRE_SCOPE_KEY, SESSION_ID_SCOPE_KEY
from nemo_gym.trajectory.trajectory_record import AgentObservationBundle


REPO = Path(__file__).resolve().parents[3]


def _app_with_gym_middleware() -> FastAPI:
    app = FastAPI()
    server = MagicMock()
    server.get_session_middleware_key.return_value = "TestServer___test"
    # The session hooks keep their defaults, which derive the signing key and the cookie's name from the key above.
    server.get_session_secret.side_effect = lambda: SimpleServer.get_session_secret(server)
    server.get_session_cookie_name.side_effect = lambda: SimpleServer.get_session_cookie_name(server)
    server.config.name = "test"
    SimpleServer.setup_session_middleware(server, app)
    SimpleServer.setup_exception_middleware(server, app)
    return app


def test_a_session_keeps_its_id_across_requests() -> None:
    app = _app_with_gym_middleware()

    @app.get("/id")
    async def session_id(request: Request) -> str:
        return request.session[SESSION_ID_KEY]

    client = TestClient(app)
    first = client.get("/id")
    second = client.get("/id")

    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()
    assert "set-cookie" in first.headers


def test_a_failed_request_under_a_new_session_hands_out_no_cookie() -> None:
    app = _app_with_gym_middleware()

    @app.get("/missing")
    async def missing(request: Request) -> None:
        request.session[SESSION_ID_KEY]
        raise ValueError("nothing here")

    response = TestClient(app).get("/missing")

    assert response.status_code == 500
    assert "set-cookie" not in response.headers


def test_a_new_session_whose_route_replied_with_an_error_hands_out_no_cookie() -> None:
    # The route's own error reply passes through the session layers, unlike a raised exception.
    app = _app_with_gym_middleware()

    @app.get("/refused")
    async def refused(request: Request) -> None:
        request.session[SESSION_ID_KEY]
        raise HTTPException(status_code=404, detail="no such seed")

    response = TestClient(app).get("/refused")

    assert response.status_code == 404
    assert "set-cookie" not in response.headers


def test_a_new_session_takes_the_id_the_session_router_chose() -> None:
    app = _app_with_gym_middleware()

    @app.get("/id")
    async def session_id(request: Request) -> str:
        return request.session[SESSION_ID_KEY]

    gym_app = app.build_middleware_stack()

    async def routed(scope, receive, send):
        await gym_app({**scope, SESSION_ID_SCOPE_KEY: "chosen"}, receive, send)

    assert TestClient(routed).get("/id").json() == "chosen"


def test_a_session_the_router_chose_hands_out_no_cookie_when_its_route_replied_with_an_error() -> None:
    app = _app_with_gym_middleware()

    @app.get("/refused")
    async def refused(request: Request) -> None:
        request.session[SESSION_ID_KEY]
        raise HTTPException(status_code=409, detail="seed belongs to another episode")

    gym_app = app.build_middleware_stack()

    async def routed(scope, receive, send):
        await gym_app({**scope, SESSION_ID_SCOPE_KEY: "another-episode"}, receive, send)

    response = TestClient(routed).get("/refused")

    assert response.status_code == 409
    assert "set-cookie" not in response.headers


def test_a_failed_route_reply_keeps_its_partial_observations() -> None:
    app = _app_with_gym_middleware()

    @app.post("/run")
    async def run(request: Request) -> None:
        request.state.nemo_gym_agent_observations = AgentObservationBundle(source="simple_agent")
        raise ValueError("policy failed")

    response = TestClient(app).post("/run")

    assert response.status_code == 500
    assert response.json()["detail"] == "ValueError('policy failed')"
    assert response.json()["ng_agent_observations"]["source"] == "simple_agent"


def test_a_failed_route_logs_its_own_traceback(capsys: pytest.CaptureFixture[str]) -> None:
    app = _app_with_gym_middleware()

    @app.get("/broken")
    async def broken_route() -> None:
        raise ValueError("broken")

    TestClient(app).get("/broken")

    logged = capsys.readouterr()
    assert "in broken_route" in logged.out + logged.err
    assert "NoneType: None" not in logged.out + logged.err


def test_a_route_that_never_replies_gets_a_500() -> None:
    app = _app_with_gym_middleware()

    async def silent(scope, receive, send):
        return

    app.mount("/silent", silent)
    response = TestClient(app).get("/silent/")

    assert response.status_code == 500
    assert response.json() == "RuntimeError('No response returned.')"


def test_a_route_that_raises_cancelled_error_gets_a_500() -> None:
    app = _app_with_gym_middleware()

    @app.get("/cancelled")
    async def cancelled() -> None:
        raise asyncio.CancelledError()

    response = TestClient(app).get("/cancelled")

    assert response.status_code == 500
    assert response.json() == "Request was cancelled"


def test_a_failure_after_the_reply_started_propagates() -> None:
    app = _app_with_gym_middleware()

    async def body() -> Iterator[bytes]:
        yield b"partial"
        raise RuntimeError("failed mid-stream")

    @app.get("/stream")
    async def stream() -> StreamingResponse:
        return StreamingResponse(body())

    with pytest.raises(RuntimeError, match="failed mid-stream"):
        TestClient(app).get("/stream")


def _http_scope(path: str) -> dict:
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "root_path": "",
        "query_string": b"",
        "headers": [],
        "client": ("127.0.0.1", 1),
        "server": ("127.0.0.1", 80),
        "state": {},
    }


async def test_a_cancelled_request_gets_the_failure_reply_and_the_cancellation_propagates() -> None:
    app = _app_with_gym_middleware()
    entered = asyncio.Event()

    @app.get("/slow")
    async def slow() -> None:
        entered.set()
        await asyncio.Event().wait()

    sent = []

    async def receive():
        await asyncio.Event().wait()

    async def send(message):
        sent.append(message)

    request = asyncio.create_task(app(_http_scope("/slow"), receive, send))
    try:
        # An expired deadline raises TimeoutError here, so it cannot pass for the request's own cancellation.
        with pytest.raises(asyncio.CancelledError):
            async with asyncio.timeout(10):
                await entered.wait()
                request.cancel()
                await request
    finally:
        request.cancel()
    assert sent[0]["status"] == 500
    assert sent[1]["body"] == b'"Request was cancelled"'


def test_a_route_whose_task_other_code_cancels_gets_the_failure_reply() -> None:
    # Checkpoint retirement and the model gate cancel the task serving a route, not the client's connection.
    app = _app_with_gym_middleware()
    app.add_middleware(ClientDisconnectCancellationMiddleware)

    @app.get("/retired")
    async def retired() -> None:
        task = asyncio.current_task()
        asyncio.get_running_loop().call_soon(task.cancel)
        await asyncio.Event().wait()

    response = TestClient(app).get("/retired")

    assert response.status_code == 500
    assert response.json() == "Request was cancelled"


def _app_with_session_owner(acquired: bool, acquires: list[str], streamed: list[str] | None = None) -> FastAPI:
    """Gym's session middleware behind a session router that takes ownership or finds another owner, recording each attempt."""
    streamed = [] if streamed is None else streamed
    app = _app_with_gym_middleware()

    @app.get("/uses-session")
    async def uses_session(request: Request) -> str:
        return request.session[SESSION_ID_KEY]

    @app.get("/ignores-session")
    async def ignores_session() -> str:
        return "ok"

    @app.get("/refuses-session")
    async def refuses_session(request: Request) -> None:
        request.session[SESSION_ID_KEY]
        raise HTTPException(status_code=404, detail="no such seed")

    @app.get("/streams-session")
    async def streams_session(request: Request) -> StreamingResponse:
        request.session[SESSION_ID_KEY]

        async def chunks() -> AsyncIterator[bytes]:
            streamed.append("chunk")
            yield b"chunk"

        return StreamingResponse(chunks())

    gym_app = app.build_middleware_stack()

    def acquire() -> bool:
        acquires.append("acquire")
        return acquired

    async def routed(scope, receive, send):
        await gym_app({**scope, SESSION_ACQUIRE_SCOPE_KEY: acquire}, receive, send)

    return routed


def test_a_session_another_worker_owns_is_refused_with_a_409() -> None:
    acquires = []
    response = TestClient(_app_with_session_owner(False, acquires)).get("/uses-session")

    assert response.status_code == 409
    assert response.json() == {"detail": "This session's ID is already owned by another worker. Retry it."}
    assert acquires == ["acquire"]


def test_a_route_whose_reply_the_ownership_409_replaced_stops_before_streaming() -> None:
    acquires = []
    streamed = []
    response = TestClient(_app_with_session_owner(False, acquires, streamed)).get("/streams-session")

    assert response.status_code == 409
    assert streamed == []


def test_an_owned_new_session_whose_route_replied_with_an_error_hands_out_no_cookie() -> None:
    acquires = []
    response = TestClient(_app_with_session_owner(True, acquires)).get("/refuses-session")

    assert response.status_code == 404
    assert "set-cookie" not in response.headers


def test_an_owned_session_the_route_used_hands_out_its_cookie() -> None:
    acquires = []
    response = TestClient(_app_with_session_owner(True, acquires)).get("/uses-session")

    assert response.status_code == 200
    assert "set-cookie" in response.headers
    assert acquires == ["acquire"]


def test_a_new_session_the_route_never_used_is_neither_owned_nor_handed_out() -> None:
    acquires = []
    response = TestClient(_app_with_session_owner(True, acquires)).get("/ignores-session")

    assert response.status_code == 200
    assert "set-cookie" not in response.headers
    assert response.headers["vary"] == "Cookie"
    assert acquires == []


def test_a_named_session_the_route_never_used_keeps_its_vary_header() -> None:
    acquires = []
    client = TestClient(_app_with_session_owner(True, acquires))
    client.get("/uses-session")
    acquires.clear()

    response = client.get("/ignores-session")

    assert response.status_code == 200
    assert response.headers["vary"] == "Cookie"
    assert acquires == []


_LEFT_BEHIND: ContextVar[str] = ContextVar("left_behind", default="unset")


def _app_leaving_context_behind() -> FastAPI:
    app = FastAPI()

    @app.get("/set")
    async def set_without_reset() -> str:
        _LEFT_BEHIND.set("left behind")
        return "set"

    @app.get("/read")
    async def read() -> str:
        return _LEFT_BEHIND.get()

    return app


@contextmanager
def _serving(app: FastAPI) -> Iterator[int]:
    """Serve ``app`` on a real uvicorn server in a thread, and yield its port."""
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=0, http="httptools", log_level="warning", lifespan="off")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started:
            assert time.monotonic() < deadline, "the test server did not start"
            time.sleep(0.01)
        yield server.servers[0].sockets[0].getsockname()[1]
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def _pipelined_replies(app: FastAPI) -> list[str]:
    """Send ``/set`` then ``/read`` in one write on one connection, so uvicorn pipelines ``/read``."""
    with _serving(app) as port, socket.create_connection(("127.0.0.1", port), timeout=10) as connection:
        connection.sendall(
            b"GET /set HTTP/1.1\r\nHost: test\r\n\r\nGET /read HTTP/1.1\r\nHost: test\r\nConnection: close\r\n\r\n"
        )
        raw = b""
        while chunk := connection.recv(65536):
            raw += chunk
    return re.findall(r'\r\n\r\n"([^"]*)"', raw.decode())


def test_without_isolation_a_pipelined_request_inherits_what_the_previous_route_set() -> None:
    # The control: it shows the pipelined request below really does start from the previous request's context.
    assert _pipelined_replies(_app_leaving_context_behind()) == ["set", "left behind"]


def test_a_pipelined_request_does_not_inherit_what_the_previous_route_set() -> None:
    app = _app_leaving_context_behind()
    install_request_context_isolation(app)

    assert _pipelined_replies(app) == ["set", "unset"]


def test_a_client_that_disconnects_mid_upload_cancels_the_route_before_it_sees_the_disconnect() -> None:
    # A handler that treats a failed body read as an empty body must not go on to act on it.
    app = _app_with_gym_middleware()
    app.add_middleware(ClientDisconnectCancellationMiddleware)
    reading = threading.Event()
    finished = threading.Event()
    outcome = []

    @app.post("/execute")
    async def execute(request: Request) -> str:
        try:
            reading.set()
            await request.json()
        except ClientDisconnect:
            outcome.append("carried on with an empty body")
        finally:
            finished.set()
        return "executed"

    with _serving(app) as port:
        with socket.create_connection(("127.0.0.1", port), timeout=10) as connection:
            connection.sendall(
                b"POST /execute HTTP/1.1\r\nHost: test\r\nContent-Type: application/json\r\n"
                b'Content-Length: 100\r\n\r\n{"code": '
            )
            assert reading.wait(10), "the route never started reading the body"
        assert finished.wait(10), "the route was neither cancelled nor finished"

    assert outcome == []


# Environment-specific bearer-token checks, to be converted separately, by how many each file registers.
_ALLOWED_DECORATED_MIDDLEWARE = {
    "resources_servers/osworld/app.py": 1,
    "resources_servers/osworld/worker.py": 1,
}


def test_gym_servers_add_no_middleware_that_runs_the_app_in_another_task() -> None:
    # `@app.middleware("http")` and `BaseHTTPMiddleware` run the rest of the app in a new task for every request.
    # They also relay each reply message through an in-memory stream, which adds event-loop work on every hop.
    # Write ASGI middleware instead.
    pattern = re.compile(
        r"""\.middleware\(\s*["']http["']\s*\)"""
        r"|\bBaseHTTPMiddleware\b"
        r"|starlette\.middleware\.base\b"
    )
    offending = {}
    for root in (
        "nemo_gym",
        "responses_api_agents",
        "responses_api_models",
        "resources_servers",
        "environment_servers",
    ):
        for path in sorted((REPO / root).rglob("*.py")):
            relative = path.relative_to(REPO).as_posix()
            parts = path.relative_to(REPO).parts
            # Installed environments, such as a server's .venv, hold third-party code that may use either form.
            if "tests" in parts or any(part.startswith(".") or part == "site-packages" for part in parts):
                continue
            count = len(pattern.findall(path.read_text(encoding="utf-8", errors="replace")))
            if count > _ALLOWED_DECORATED_MIDDLEWARE.get(relative, 0):
                offending[relative] = count

    assert offending == {}
