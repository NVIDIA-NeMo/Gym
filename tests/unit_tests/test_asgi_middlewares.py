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
from asyncio.exceptions import CancelledError
from contextlib import asynccontextmanager
from typing import Optional

import pytest
from aiohttp import ClientResponseError
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from omegaconf import DictConfig
from starlette.middleware.sessions import SessionMiddleware
from starlette.types import ASGIApp, Message, Receive, Scope, Send

import nemo_gym.server_utils as server_utils
from nemo_gym.config_types import BaseRunServerInstanceConfig, BaseServerConfig
from nemo_gym.server_utils import (
    SESSION_ID_KEY,
    ExceptionHandlingMiddleware,
    ServerClient,
    SessionIDMiddleware,
    SimpleServer,
)


class _MiddlewareTestServer(SimpleServer):
    def setup_webserver(self) -> FastAPI:
        raise AssertionError("not used in these tests")


def _make_server() -> _MiddlewareTestServer:
    return _MiddlewareTestServer(
        config=BaseRunServerInstanceConfig(name="my_server", host="", port=0, entrypoint=""),
        server_client=ServerClient(
            head_server_config=BaseServerConfig(host="", port=0),
            global_config_dict=DictConfig({}),
        ),
    )


def _make_app(server: SimpleServer, lifespan=None) -> FastAPI:
    """Mirror run_webserver's real order: session middleware first, exception middleware last."""
    app = FastAPI(lifespan=lifespan)
    server.setup_session_middleware(app)
    server.setup_exception_middleware(app)
    return app


def _client_response_error(response_content: Optional[bytes]) -> ClientResponseError:
    error = ClientResponseError(request_info=None, history=(), status=502, message="bad gateway")
    if response_content is not None:
        error.response_content = response_content
    return error


def _http_scope(session: Optional[dict] = None) -> Scope:
    scope: Scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/",
        "raw_path": b"/",
        "query_string": b"",
        "root_path": "",
        "headers": [],
        "client": ("127.0.0.1", 1),
        "server": ("testserver", 80),
    }
    if session is not None:
        scope["session"] = session
    return scope


async def _receive() -> Message:
    return {"type": "http.request", "body": b"", "more_body": False}


async def _run_asgi(app: ASGIApp, scope: Optional[Scope] = None) -> list[Message]:
    sent: list[Message] = []

    async def send(message: Message) -> None:
        sent.append(message)

    await app(scope or _http_scope(), _receive, send)
    return sent


class TestSessionIDMiddleware:
    def test_session_id_and_cookie_payload_stay_stable(self) -> None:
        server = _make_server()
        app = _make_app(server)

        @app.get("/session")
        async def get_session(request: Request) -> dict:
            return {"session_id": request.session[SESSION_ID_KEY]}

        with TestClient(app) as client:
            first = client.get("/session")
            second = client.get("/session")

        first_id = first.json()["session_id"]
        second_id = second.json()["session_id"]
        assert first_id
        assert first_id == second_id

        # Each response persists the stable session ID.
        assert 1 == len(first.headers.get_list("set-cookie"))
        assert 1 == len(second.headers.get_list("set-cookie"))

    async def test_uuid_is_generated_only_when_session_id_is_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        generated: list[str] = []

        def make_uuid() -> str:
            generated.append("generated-id")
            return generated[-1]

        monkeypatch.setattr(server_utils, "uuid4", make_uuid)

        async def downstream(scope: Scope, receive: Receive, send: Send) -> None:
            return None

        middleware = SessionIDMiddleware(downstream)
        existing_session = {SESSION_ID_KEY: "existing-id"}
        missing_session: dict = {}

        await _run_asgi(middleware, _http_scope(existing_session))
        await _run_asgi(middleware, _http_scope(missing_session))

        assert existing_session[SESSION_ID_KEY] == "existing-id"
        assert missing_session[SESSION_ID_KEY] == "generated-id"
        assert generated == ["generated-id"]

    def test_registration_order_populates_session_before_use(self) -> None:
        server = _make_server()
        app = _make_app(server)

        # user_middleware lists request-time order from outermost to innermost.
        assert [ExceptionHandlingMiddleware, SessionMiddleware, SessionIDMiddleware] == [
            m.cls for m in app.user_middleware
        ]

    def test_lifespan_scope_passes_through(self) -> None:
        events: list[str] = []

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            events.append("startup")
            yield
            events.append("shutdown")

        server = _make_server()
        app = _make_app(server, lifespan=lifespan)

        with TestClient(app):
            assert ["startup"] == events
        assert ["startup", "shutdown"] == events

    @pytest.mark.parametrize("scope_type", ["lifespan", "websocket"])
    async def test_non_http_scope_passes_through_unchanged(self, scope_type: str) -> None:
        server = _make_server()
        scope = {"type": scope_type}
        observed: list[tuple[Scope, Receive, Send]] = []

        async def downstream(actual_scope: Scope, receive: Receive, send: Send) -> None:
            observed.append((actual_scope, receive, send))

        async def receive() -> Message:
            return {"type": f"{scope_type}.disconnect"}

        async def send(message: Message) -> None:
            raise AssertionError(f"unexpected message: {message}")

        app = ExceptionHandlingMiddleware(SessionIDMiddleware(downstream), server)
        await app(scope, receive, send)

        assert observed == [(scope, receive, send)]


class TestExceptionHandlingMiddleware:
    def test_normal_response_passes_through(self) -> None:
        server = _make_server()
        app = _make_app(server)

        @app.get("/ok")
        async def ok() -> dict:
            return {"status": "ok"}

        with TestClient(app) as client:
            response = client.get("/ok")
        assert 200 == response.status_code
        assert {"status": "ok"} == response.json()

    async def test_streaming_response_messages_pass_through(self) -> None:
        server = _make_server()
        expected: list[Message] = [
            {"type": "http.response.start", "status": 200, "headers": [(b"x-test", b"stream")]},
            {"type": "http.response.body", "body": b"first", "more_body": True},
            {"type": "http.response.body", "body": b"second", "more_body": False},
        ]

        async def downstream(scope: Scope, receive: Receive, send: Send) -> None:
            for message in expected:
                await send(message)

        sent = await _run_asgi(ExceptionHandlingMiddleware(downstream, server))

        assert sent == expected

    def test_client_response_error_with_response_content(self) -> None:
        server = _make_server()
        app = _make_app(server)

        @app.get("/boom")
        async def boom() -> None:
            raise _client_response_error(b"inner server said no")

        with TestClient(app) as client:
            response = client.get("/boom")

        assert 500 == response.status_code
        expected = (
            f"Hit an exception in {server.get_session_middleware_key()} "
            f"calling an inner server: {b'inner server said no'}"
        )
        assert expected == response.json()

    def test_client_response_error_without_response_content_asserts(self) -> None:
        server = _make_server()
        app = _make_app(server)

        @app.get("/boom")
        async def boom() -> None:
            raise _client_response_error(None)

        with TestClient(app) as client:
            with pytest.raises(AssertionError, match="raise_for_status"):
                client.get("/boom")

    def test_generic_exception_returns_repr(self, capsys: pytest.CaptureFixture) -> None:
        server = _make_server()
        app = _make_app(server)

        @app.get("/boom")
        async def boom() -> None:
            raise ValueError("something went wrong")

        with TestClient(app) as client:
            response = client.get("/boom")

        assert 500 == response.status_code
        assert repr(ValueError("something went wrong")) == response.json()

        printed = capsys.readouterr().out
        assert "Caught an exception printed above in my_server (_MiddlewareTestServer)" in printed
        assert "repr(e): ValueError('something went wrong')" in printed

    def test_bare_except_returns_unknown_error(self, capsys: pytest.CaptureFixture) -> None:
        class NotAnException(BaseException):
            pass

        server = _make_server()
        app = _make_app(server)

        @app.get("/boom")
        async def boom() -> None:
            raise NotAnException()

        with TestClient(app) as client:
            response = client.get("/boom")

        assert 500 == response.status_code
        assert "An unknown error occurred" == response.json()

        captured = capsys.readouterr()
        assert "Caught an unknown exception printed above in my_server (_MiddlewareTestServer)" in captured.out
        assert "NotAnException" in captured.err  # print_exc()

    def test_cancelled_error_returns_unknown_error(self) -> None:
        server = _make_server()
        app = _make_app(server)

        @app.get("/boom")
        async def boom() -> None:
            raise CancelledError()

        with TestClient(app) as client:
            response = client.get("/boom")

        assert 500 == response.status_code
        assert "An unknown error occurred" == response.json()

    def test_exception_after_response_start_is_reraised(self) -> None:
        server = _make_server()
        app = _make_app(server)

        @app.get("/stream")
        async def stream() -> StreamingResponse:
            async def body():
                yield b"partial"
                raise ValueError("mid-stream failure")

            return StreamingResponse(body())

        with TestClient(app) as client:
            with pytest.raises(ValueError, match="mid-stream failure"):
                client.get("/stream")

    async def test_raw_streaming_exception_preserves_partial_response(self) -> None:
        server = _make_server()
        sent: list[Message] = []
        expected: list[Message] = [
            {"type": "http.response.start", "status": 200, "headers": []},
            {"type": "http.response.body", "body": b"partial", "more_body": True},
        ]

        async def downstream(scope: Scope, receive: Receive, send: Send) -> None:
            for message in expected:
                await send(message)
            raise ValueError("mid-stream failure")

        async def send(message: Message) -> None:
            sent.append(message)

        middleware = ExceptionHandlingMiddleware(downstream, server)
        with pytest.raises(ValueError, match="mid-stream failure"):
            await middleware(_http_scope(), _receive, send)

        assert sent == expected
