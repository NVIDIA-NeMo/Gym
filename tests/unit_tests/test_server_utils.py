# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import asyncio
import logging
import os
import pickle
import socket
from unittest.mock import AsyncMock, MagicMock

import uvicorn
from aiohttp import ClientResponseError, RequestInfo
from fastapi import Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from multidict import CIMultiDict, CIMultiDictProxy
from omegaconf import OmegaConf
from pydantic import ValidationError
from pytest import CaptureFixture, LogCaptureFixture, MonkeyPatch, mark, raises
from uvicorn.protocols.http.httptools_impl import HttpToolsProtocol
from yarl import URL

import nemo_gym.client.http_client
import nemo_gym.client.server_client
import nemo_gym.config.global_config
import nemo_gym.server_utils
from nemo_gym.client import ServerClient
from nemo_gym.config.config_types import BaseRunServerInstanceConfig
from nemo_gym.config.global_config import (
    DRY_RUN_KEY_NAME,
    NEMO_GYM_CONFIG_PATH_ENV_VAR_NAME,
)
from nemo_gym.server_utils import (
    BaseServer,
    BaseServerConfig,
    ClientDisconnectCancellationMiddleware,
    DictConfig,
    HeadServer,
    KeepaliveHttpToolsProtocol,
    SimpleServer,
    UvicornProxyHeadersConfig,
    _log_validation_exception,
    _validation_exception_handler,
    initialize_ray,
)
from nemo_gym.utils.server_process import IS_NEMO_GYM_FASTAPI_WORKER_KEY_NAME


# macOS exposes the idle option as TCP_KEEPALIVE rather than TCP_KEEPIDLE.
_TCP_KEEPIDLE_OPT = getattr(socket, "TCP_KEEPIDLE", getattr(socket, "TCP_KEEPALIVE", None))


_TCP_KEEPALIVE_TEST_IDLE = 42
_TCP_KEEPALIVE_TEST_INTERVAL = 7
_TCP_KEEPALIVE_TEST_PROBES = 2


class TestServerUtils:
    def test_BaseServer_load_config_from_global_config(self, monkeypatch: MonkeyPatch) -> None:
        # Clear any lingering env vars.
        monkeypatch.setenv(NEMO_GYM_CONFIG_PATH_ENV_VAR_NAME, "my_server")

        global_config_dict = DictConfig(
            {"my_server": {"a": {"b": {"host": "", "port": 0, "entrypoint": "my entrypoint"}}}}
        )
        get_global_config_dict_mock = MagicMock()
        get_global_config_dict_mock.return_value = global_config_dict
        monkeypatch.setattr(nemo_gym.server_utils, "get_global_config_dict", get_global_config_dict_mock)

        actual_config = BaseServer.load_config_from_global_config()
        assert "" == actual_config.host
        assert 0 == actual_config.port
        assert "my entrypoint" == actual_config.entrypoint

    def test_HeadServer_setup_webserver_sanity(self) -> None:
        head_server = HeadServer(config=BaseServerConfig(host="", port=0))
        head_server.setup_webserver()

    def test_HeadServer_health_reports_readiness_without_changing_liveness(self) -> None:
        from fastapi.testclient import TestClient

        head_server = HeadServer(config=BaseServerConfig(host="", port=0))

        with TestClient(head_server.setup_webserver()) as client:
            for path in ("/", "/livez"):
                response = client.get(path)
                assert response.status_code == 200
                assert response.json() == {"status": "ok"}

            for path in ("/health", "/healthz", "/readyz"):
                response = client.get(path)
                assert response.status_code == 503
                assert response.json() == {"status": "starting"}

            head_server.mark_ready()

            for path in ("/health", "/healthz", "/readyz"):
                response = client.get(path)
                assert response.status_code == 200
                assert response.json() == {"status": "ok"}

    async def test_HeadServer_global_config_dict_yaml(self, monkeypatch: MonkeyPatch) -> None:
        global_config_dict = DictConfig({"a": 2})
        get_global_config_dict_mock = MagicMock()
        get_global_config_dict_mock.return_value = global_config_dict
        monkeypatch.setattr(nemo_gym.server_utils, "get_global_config_dict", get_global_config_dict_mock)

        head_server = HeadServer(config=BaseServerConfig(host="", port=0))
        resp = await head_server.global_config_dict_yaml()

        assert "a: 2\n" == resp

    async def test_HeadServer_global_config_dict_yaml_caches(self, monkeypatch: MonkeyPatch) -> None:
        """Serialize the global config once until the cache is cleared."""
        global_config_dict = DictConfig({"a": 2})
        get_global_config_dict_mock = MagicMock(return_value=global_config_dict)
        monkeypatch.setattr(nemo_gym.server_utils, "get_global_config_dict", get_global_config_dict_mock)

        to_yaml_mock = MagicMock(wraps=OmegaConf.to_yaml)
        monkeypatch.setattr(nemo_gym.server_utils.OmegaConf, "to_yaml", to_yaml_mock)

        head_server = HeadServer(config=BaseServerConfig(host="", port=0))
        first = await head_server.global_config_dict_yaml()
        second = await head_server.global_config_dict_yaml()

        assert first is second
        assert to_yaml_mock.call_count == 1

        head_server.invalidate_global_config_dict_yaml_cache()
        third = await head_server.global_config_dict_yaml()
        assert third == first
        assert to_yaml_mock.call_count == 2

    def _mock_ray_return_value(self, monkeypatch: MonkeyPatch, return_value: bool) -> MagicMock:
        ray_mock = MagicMock()
        ray_mock.is_initialized.return_value = return_value
        monkeypatch.setattr(nemo_gym.server_utils, "_get_ray", MagicMock(return_value=ray_mock))
        return ray_mock.is_initialized

    def _mock_ray_init(self) -> MagicMock:
        return nemo_gym.server_utils._get_ray().init

    def test_initialize_ray_already_initialized(self, monkeypatch: MonkeyPatch) -> None:
        ray_is_initialized_mock = self._mock_ray_return_value(monkeypatch, True)

        get_global_config_dict_mock = MagicMock()
        monkeypatch.setattr(nemo_gym.server_utils, "get_global_config_dict", get_global_config_dict_mock)

        initialize_ray()

        ray_is_initialized_mock.assert_called_once()
        get_global_config_dict_mock.assert_not_called()

    def test_initialize_ray_with_address(self, monkeypatch: MonkeyPatch) -> None:
        ray_is_initialized_mock = self._mock_ray_return_value(monkeypatch, False)

        ray_init_mock = self._mock_ray_init()

        # Mock global config dict with ray_head_node_address
        global_config_dict = DictConfig({"ray_head_node_address": "ray://test-address:10001"})
        get_global_config_dict_mock = MagicMock()
        get_global_config_dict_mock.return_value = global_config_dict
        monkeypatch.setattr(nemo_gym.server_utils, "get_global_config_dict", get_global_config_dict_mock)

        initialize_ray()

        ray_is_initialized_mock.assert_called_once()
        get_global_config_dict_mock.assert_called_once()
        ray_init_mock.assert_called_once_with(address="ray://test-address:10001", ignore_reinit_error=True)

    def test_initialize_ray_without_address(self, monkeypatch: MonkeyPatch) -> None:
        ray_is_initialized_mock = self._mock_ray_return_value(monkeypatch, False)

        ray_init_mock = self._mock_ray_init()

        ray_runtime_context_mock = MagicMock()
        ray_runtime_context_mock.gcs_address = "ray://mock-address:10001"
        ray_get_runtime_context_mock = nemo_gym.server_utils._get_ray().get_runtime_context
        ray_get_runtime_context_mock.return_value = ray_runtime_context_mock

        # Mock global config dict without ray_head_node_address
        global_config_dict = DictConfig({"k": "v"})
        get_global_config_dict_mock = MagicMock()
        get_global_config_dict_mock.return_value = global_config_dict
        monkeypatch.setattr(nemo_gym.server_utils, "get_global_config_dict", get_global_config_dict_mock)

        initialize_ray()

        ray_is_initialized_mock.assert_called_once()
        get_global_config_dict_mock.assert_called_once()
        ray_init_mock.assert_called_once_with(ignore_reinit_error=True)
        ray_get_runtime_context_mock.assert_called_once()

    @mark.parametrize("family", [socket.AF_INET, socket.AF_INET6, socket.AF_UNIX])
    def test_keepalive_httptools_protocol_enables_keepalive_on_tcp_only(
        self, monkeypatch: MonkeyPatch, family: int
    ) -> None:
        parent_connection_made = MagicMock()
        monkeypatch.setattr(HttpToolsProtocol, "__init__", lambda self, *args, **kwargs: None)
        monkeypatch.setattr(HttpToolsProtocol, "connection_made", parent_connection_made)
        protocol = KeepaliveHttpToolsProtocol(
            keepalive=(_TCP_KEEPALIVE_TEST_IDLE, _TCP_KEEPALIVE_TEST_INTERVAL, _TCP_KEEPALIVE_TEST_PROBES)
        )
        sock = socket.socket(family, socket.SOCK_STREAM)
        transport = MagicMock()
        transport.get_extra_info.return_value = sock
        try:
            protocol.connection_made(transport)  # A Unix socket must not raise on TCP-level options.
            keepalive_on = sock.getsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE) != 0
            assert keepalive_on is (family != socket.AF_UNIX)
            if family != socket.AF_UNIX and _TCP_KEEPIDLE_OPT is not None:
                assert sock.getsockopt(socket.IPPROTO_TCP, _TCP_KEEPIDLE_OPT) == _TCP_KEEPALIVE_TEST_IDLE
        finally:
            sock.close()
        parent_connection_made.assert_called_once_with(transport)

    def test_dry_run_skips_webserver_spinup(self, monkeypatch: MonkeyPatch) -> None:
        self._mock_ray_return_value(monkeypatch, True)

        get_global_config_dict_mock = MagicMock()
        monkeypatch.setattr(nemo_gym.server_utils, "get_global_config_dict", get_global_config_dict_mock)

        ServerClient_mock = MagicMock(spec=ServerClient)
        monkeypatch.setattr(nemo_gym.server_utils, "ServerClient", ServerClient_mock)

        class TestSimpleServer(SimpleServer):
            def __init__(self, *args, **kwargs):
                # Only the config: the startup checks before the dry-run return read it.
                object.__setattr__(self, "config", MagicMock(num_workers=None))

            def setup_webserver(self):
                assert False

            @classmethod
            def load_config_from_global_config(cls) -> None:
                pass

        TestSimpleServer.run_webserver()

    def test_dry_run_reports_unsupported_num_workers(self, monkeypatch: MonkeyPatch) -> None:
        self._mock_ray_return_value(monkeypatch, True)
        monkeypatch.setattr(nemo_gym.server_utils, "get_global_config_dict", MagicMock())
        monkeypatch.setattr(nemo_gym.server_utils, "ServerClient", MagicMock(spec=ServerClient))

        class RefusingServer(SimpleServer):
            def __init__(self, *args, **kwargs):
                object.__setattr__(self, "config", MagicMock(num_workers=2))
                self.config.name = "refusing_server"

            def multiple_workers_unsupported_reason(self) -> str:
                return "sessions live in this process"

            def setup_webserver(self):
                assert False

            @classmethod
            def load_config_from_global_config(cls) -> None:
                pass

        with raises(ValueError, match="^refusing_server needs num_workers=1: sessions live in this process$"):
            RefusingServer.run_webserver()

    def test_setup_liveness_exposes_conventional_probe_surface(self) -> None:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        app = FastAPI()
        BaseServer.setup_liveness(MagicMock(), app)

        with TestClient(app) as client:
            for path in ("/", "/health", "/healthz", "/livez", "/readyz"):
                response = client.get(path)
                assert response.status_code == 200
                assert response.json() == {"status": "ok"}

    def test_setup_session_middleware_idempotent(self) -> None:
        from fastapi import FastAPI, Request
        from fastapi.testclient import TestClient
        from starlette.middleware.sessions import SessionMiddleware

        from nemo_gym.client import SESSION_ID_KEY

        class TestSimpleServer(SimpleServer):
            def setup_webserver(self):
                assert False

        server = TestSimpleServer(
            config=BaseRunServerInstanceConfig(name="my_server", host="", port=0, entrypoint=""),
            server_client=ServerClient(
                head_server_config=BaseServerConfig(host="", port=0),
                global_config_dict=DictConfig({}),
            ),
        )

        app = FastAPI()
        server.setup_session_middleware(app)
        server.setup_session_middleware(app)

        session_middlewares = [m for m in app.user_middleware if m.cls is SessionMiddleware]
        assert 1 == len(session_middlewares)
        assert 2 == len(app.user_middleware)

        @app.get("/session")
        async def get_session(request: Request) -> dict:
            return {"session_id": request.session[SESSION_ID_KEY]}

        with TestClient(app) as client:
            response = client.get("/session")
            assert response.json()["session_id"]
            assert 1 == len(response.headers.get_list("set-cookie"))

    def test_cancellation_middleware_preserves_request_body(self) -> None:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        class TestSimpleServer(SimpleServer):
            def setup_webserver(self):
                assert False

        server = TestSimpleServer(
            config=BaseRunServerInstanceConfig(name="my_server", host="", port=0, entrypoint=""),
            server_client=MagicMock(spec=ServerClient),
        )
        app = FastAPI()
        server.setup_cancellation_middleware(app)

        @app.post("/echo")
        async def echo(body: dict) -> dict:
            return body

        with TestClient(app) as client:
            response = client.post("/echo", json={"message": "hello"})

        assert response.status_code == 200
        assert response.json() == {"message": "hello"}

    async def test_cancellation_middleware_cancels_handler_on_disconnect(self) -> None:
        from fastapi import FastAPI, Request

        class TestSimpleServer(SimpleServer):
            def setup_webserver(self):
                assert False

        server = TestSimpleServer(
            config=BaseRunServerInstanceConfig(name="my_server", host="", port=0, entrypoint=""),
            server_client=MagicMock(spec=ServerClient),
        )
        app = FastAPI()
        server.setup_exception_middleware(app)
        server.setup_cancellation_middleware(app)
        handler_started = asyncio.Event()
        handler_cancelled = asyncio.Event()

        @app.post("/work")
        async def work(request: Request) -> None:
            assert await request.json() == {"message": "hello"}
            handler_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                handler_cancelled.set()
                raise

        incoming_messages = asyncio.Queue()
        await incoming_messages.put({"type": "http.request", "body": b'{"message":"hello"}', "more_body": False})

        async def receive():
            return await incoming_messages.get()

        sent_messages = []

        async def send(message):
            sent_messages.append(message)

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/work",
            "raw_path": b"/work",
            "query_string": b"",
            "headers": [(b"content-type", b"application/json")],
            "client": ("127.0.0.1", 1234),
            "server": ("testserver", 80),
        }
        app_task = asyncio.create_task(app(scope, receive, send))
        await asyncio.wait_for(handler_started.wait(), timeout=1)
        await incoming_messages.put({"type": "http.disconnect"})
        await asyncio.wait_for(app_task, timeout=1)

        assert handler_cancelled.is_set()
        assert sent_messages == []

    async def test_cancellation_middleware_ignores_disconnect_after_response_completion(self) -> None:
        response_sent = asyncio.Event()
        finish_cleanup = asyncio.Event()
        cleanup_completed = asyncio.Event()
        handler_cancelled = asyncio.Event()

        async def inner_app(scope, receive, send) -> None:
            assert await receive() == {"type": "http.request", "body": b"", "more_body": False}
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok", "more_body": False})
            try:
                await finish_cleanup.wait()
            except asyncio.CancelledError:
                handler_cancelled.set()
                raise
            cleanup_completed.set()

        middleware = ClientDisconnectCancellationMiddleware(inner_app)
        request_delivered = False

        async def receive():
            nonlocal request_delivered
            if not request_delivered:
                request_delivered = True
                return {"type": "http.request", "body": b"", "more_body": False}

            await response_sent.wait()
            return {"type": "http.disconnect"}

        sent_messages = []

        async def send(message):
            sent_messages.append(message)
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                response_sent.set()

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/work",
            "raw_path": b"/work",
            "query_string": b"",
            "headers": [],
            "client": ("127.0.0.1", 1234),
            "server": ("testserver", 80),
        }
        app_task = asyncio.create_task(middleware(scope, receive, send))
        await asyncio.wait_for(response_sent.wait(), timeout=1)
        await asyncio.sleep(0)
        finish_cleanup.set()
        await asyncio.wait_for(app_task, timeout=1)

        assert cleanup_completed.is_set()
        assert not handler_cancelled.is_set()
        assert middleware.num_cancelled == 0
        assert sent_messages == [
            {"type": "http.response.start", "status": 200, "headers": []},
            {"type": "http.response.body", "body": b"ok", "more_body": False},
        ]

    @mark.parametrize(
        "path", ["/v1/chat/completions", "/ng-rollout/3-1/v1/messages", "/run", "/tools/run", "/seed_session"]
    )
    async def test_cancellation_middleware_samples_all_routes(
        self, monkeypatch: MonkeyPatch, capsys: CaptureFixture, path: str
    ) -> None:
        monkeypatch.setattr("nemo_gym.client.http_client._GLOBAL_AIOHTTP_CLIENT_REQUEST_DEBUG", False)
        handler_waiting = asyncio.Event()

        async def inner_app(scope, receive, send) -> None:
            await receive()
            handler_waiting.set()
            await asyncio.Event().wait()

        middleware = ClientDisconnectCancellationMiddleware(inner_app)
        scope = {"type": "http", "method": "POST", "path": path, "client": ("127.0.0.1", 1234)}

        async def send(message):
            pass

        for _ in range(101):
            handler_waiting.clear()
            incoming = asyncio.Queue()
            await incoming.put({"type": "http.request", "body": b"", "more_body": False})
            app_task = asyncio.create_task(middleware(scope, incoming.get, send))
            await asyncio.wait_for(handler_waiting.wait(), timeout=1)
            await incoming.put({"type": "http.disconnect"})
            await asyncio.wait_for(app_task, timeout=1)

        assert middleware.num_cancelled == 101
        assert capsys.readouterr().out == (
            f'127.0.0.1:1234 - "POST {path}" 499 CLIENT DISCONNECTED (100 total for this server worker)\n'
        )

    @mark.parametrize(
        ("response", "phase"),
        [("none", "before_response_start"), ("headers", "response_started"), ("keepalive", "response_started")],
    )
    async def test_cancellation_middleware_debug_phase(
        self, monkeypatch: MonkeyPatch, capsys: CaptureFixture, response: str, phase: str
    ) -> None:
        from types import SimpleNamespace

        monkeypatch.setattr("nemo_gym.client.http_client._GLOBAL_AIOHTTP_CLIENT_REQUEST_DEBUG", True)
        clock = iter([10.0, 12.5])
        monkeypatch.setattr(nemo_gym.server_utils, "time", SimpleNamespace(monotonic=lambda: next(clock)))
        handler_waiting = asyncio.Event()

        async def inner_app(scope, receive, send) -> None:
            await receive()
            if response != "none":
                await send({"type": "http.response.start", "status": 200, "headers": []})
            if response == "keepalive":
                await send({"type": "http.response.body", "body": b": keep-alive\n\n", "more_body": True})
            handler_waiting.set()
            await asyncio.Event().wait()

        middleware = ClientDisconnectCancellationMiddleware(inner_app)
        incoming = asyncio.Queue()
        await incoming.put({"type": "http.request", "body": b"", "more_body": False})

        async def send(message):
            pass

        scope = {"type": "http", "method": "POST", "path": "/v1/chat/completions", "client": ("127.0.0.1", 1234)}
        app_task = asyncio.create_task(middleware(scope, incoming.get, send))
        await asyncio.wait_for(handler_waiting.wait(), timeout=1)
        await incoming.put({"type": "http.disconnect"})
        await asyncio.wait_for(app_task, timeout=1)

        assert middleware.num_cancelled == 1
        assert capsys.readouterr().out == (
            '127.0.0.1:1234 - "POST /v1/chat/completions" 499 CLIENT DISCONNECTED '
            f"after 2.5s {phase} (1 total for this server worker)\n"
        )

    @mark.parametrize(
        ("body", "expected_truncated"),
        [
            (b"", False),
            (b'{"nested":{"value":"small"}}', False),
            (b"not-json\nwith-control-\x00", False),
            (b'{"credentials":{"password":"secret"}}', False),
            (b"x" * 4094, False),
            (b"x" * 4095, True),
            (b"\x00" * 4096, True),
            ("中文".encode() * 4096, True),
            (b"\\" * 4096, True),
            (b'{"payload":"' + b"x" * (2 * 1024 * 1024) + b'"}', True),
        ],
        ids=[
            "empty",
            "small-json",
            "non-json-control",
            "credentials",
            "at-limit",
            "over-limit",
            "escape-boundary",
            "unicode",
            "backslashes",
            "multi-megabyte",
        ],
    )
    async def test_validation_exception_log_bounds_body_before_rendering(
        self, body: bytes, expected_truncated: bool, caplog: LogCaptureFixture, monkeypatch: MonkeyPatch
    ) -> None:
        request = MagicMock(spec=Request)
        request.body = AsyncMock(return_value=body)
        errors = [
            {
                "type": "missing",
                "loc": ("body", "required_field"),
                "msg": "Field required",
                "input": {"password": "value that must not be copied into the error log"},
            }
        ]
        exc = RequestValidationError(errors, body={"original": "body"})
        rendered_body_sizes = []
        escaped_log_prefix = nemo_gym.server_utils._escaped_log_prefix

        def tracking_escaped_log_prefix(value: str, max_chars: int):
            rendered_body_sizes.append(len(value))
            return escaped_log_prefix(value, max_chars)

        monkeypatch.setattr(nemo_gym.server_utils, "_escaped_log_prefix", tracking_escaped_log_prefix)

        with caplog.at_level(logging.WARNING, logger="nemo_gym.server_utils"):
            await _log_validation_exception(request, exc)

        record = caplog.records[-1]
        assert record.request_body_size_bytes == len(body)
        assert len(rendered_body_sizes) == 1
        assert rendered_body_sizes[0] <= nemo_gym.server_utils._VALIDATION_ERROR_LOG_BODY_CHARS
        assert len(record.request_body_prefix) <= nemo_gym.server_utils._VALIDATION_ERROR_LOG_BODY_CHARS
        assert record.request_body_truncated is expected_truncated
        decoded_prefix = nemo_gym.server_utils.json.loads(record.request_body_prefix)
        assert decoded_prefix.endswith("...[truncated]") is expected_truncated
        assert ("...[truncated]" in record.getMessage()) is expected_truncated
        if not expected_truncated:
            assert decoded_prefix == body.decode("utf-8", errors="replace")
        assert "request_body_size_bytes=" in record.getMessage()
        assert "request_body_truncated=" in record.getMessage()
        assert "request_body_prefix=" in record.getMessage()
        assert record.validation_error_count == 1
        assert record.validation_errors == [
            {
                "type": "missing",
                "loc": ["body", "required_field"],
                "msg": "Field required",
            }
        ]
        assert "value that must not be copied into the error log" not in str(record.validation_errors)
        if body == b"not-json\nwith-control-\x00":
            assert "\n" not in record.request_body_prefix
            assert "\x00" not in record.request_body_prefix
            assert "\\n" in record.request_body_prefix
            assert "\\u0000" in record.request_body_prefix

    @mark.parametrize(("location", "body_unavailable"), [("body", False), ("query", False), ("body", True)])
    async def test_validation_exception_log_bounds_error_count(
        self, location: str, body_unavailable: bool, caplog: LogCaptureFixture
    ) -> None:
        request = MagicMock(spec=Request)
        request.body = AsyncMock(return_value=b"{}")
        if body_unavailable:
            request.body.side_effect = RuntimeError("body unavailable")
        errors = [
            {
                "type": "missing",
                "loc": (location, f"field_{index}"),
                "msg": "Field required",
                "input": None,
            }
            for index in range(nemo_gym.server_utils._VALIDATION_ERROR_LOG_MAX_ERRORS + 5)
        ]

        with caplog.at_level(logging.WARNING, logger="nemo_gym.server_utils"):
            await _log_validation_exception(request, RequestValidationError(errors))

        record = caplog.records[-1]
        assert record.validation_error_count == len(errors)
        assert len(record.validation_errors) == nemo_gym.server_utils._VALIDATION_ERROR_LOG_MAX_ERRORS
        assert record.validation_errors_truncated is True
        assert record.getMessage().endswith("...[truncated]")

    async def test_validation_exception_log_marks_omitted_location_items(self, caplog: LogCaptureFixture) -> None:
        request = MagicMock(spec=Request)
        request.body = AsyncMock(return_value=b"{}")
        errors = [
            {
                "type": "missing",
                "loc": ("body", *("field" for _ in range(nemo_gym.server_utils._VALIDATION_ERROR_LOG_LOC_ITEMS))),
                "msg": "Field required",
            }
        ]

        with caplog.at_level(logging.WARNING, logger="nemo_gym.server_utils"):
            await _log_validation_exception(request, RequestValidationError(errors))

        record = caplog.records[-1]
        assert len(record.validation_errors[0]["loc"]) == nemo_gym.server_utils._VALIDATION_ERROR_LOG_LOC_ITEMS
        assert record.validation_errors_truncated is True
        assert record.getMessage().endswith("...[truncated]")

    async def test_validation_exception_detects_body_error_after_error_log_cap(
        self, caplog: LogCaptureFixture
    ) -> None:
        body = b'{"required":null}'
        request = MagicMock(spec=Request)
        request.body = AsyncMock(return_value=body)
        errors = [
            {"type": "missing", "loc": ("query", f"field_{index}"), "msg": "Field required", "input": None}
            for index in range(nemo_gym.server_utils._VALIDATION_ERROR_LOG_MAX_ERRORS + 1)
        ]
        errors.append({"type": "missing", "loc": ("body", "required"), "msg": "Field required", "input": None})

        with caplog.at_level(logging.WARNING, logger="nemo_gym.server_utils"):
            await _log_validation_exception(request, RequestValidationError(errors))

        request.body.assert_awaited_once()
        record = caplog.records[-1]
        assert record.request_body_size_bytes == len(body)
        assert record.request_body_prefix == '"{\\"required\\":null}"'
        assert len(record.validation_errors) == nemo_gym.server_utils._VALIDATION_ERROR_LOG_MAX_ERRORS
        assert record.validation_errors_truncated is True

    async def test_validation_exception_log_bounds_error_fields(self, caplog: LogCaptureFixture) -> None:
        request = MagicMock(spec=Request)
        request.body = AsyncMock(return_value=b"{}")
        large_value = "unsafe\n\x00" + "x" * (2 * 1024 * 1024)
        errors = [
            {
                "type": large_value,
                "loc": ("body", *([large_value] * (nemo_gym.server_utils._VALIDATION_ERROR_LOG_LOC_ITEMS + 1))),
                "msg": large_value,
                "input": None,
            }
        ]

        with caplog.at_level(logging.WARNING, logger="nemo_gym.server_utils"):
            await _log_validation_exception(request, RequestValidationError(errors))

        record = caplog.records[-1]
        summary = record.validation_errors[0]
        assert len(summary["type"]) <= nemo_gym.server_utils._VALIDATION_ERROR_LOG_FIELD_CHARS
        assert len(summary["msg"]) <= nemo_gym.server_utils._VALIDATION_ERROR_LOG_FIELD_CHARS
        assert summary["type"].endswith("...[truncated]")
        assert summary["msg"].endswith("...[truncated]")
        assert len(summary["loc"]) == nemo_gym.server_utils._VALIDATION_ERROR_LOG_LOC_ITEMS
        assert all(
            not isinstance(value, str) or len(value) <= nemo_gym.server_utils._VALIDATION_ERROR_LOG_FIELD_CHARS
            for value in summary["loc"]
        )
        assert "\n" not in str(summary)
        assert "\x00" not in str(summary)
        assert all(value.endswith("...[truncated]") for value in summary["loc"][1:])
        assert record.validation_errors_truncated is True
        assert record.getMessage().endswith("...[truncated]")
        assert len(record.getMessage()) < 5000

    @mark.parametrize(("location", "body_unavailable"), [("body", False), ("query", False), ("body", True)])
    async def test_validation_exception_message_names_failed_fields(
        self, location: str, body_unavailable: bool, caplog: LogCaptureFixture
    ) -> None:
        # The summaries are also attached as ``extra``, which the default formatter drops,
        # so the rendered message itself must say which field failed.
        request = MagicMock(spec=Request)
        request.body = AsyncMock(return_value=b"{}")
        if body_unavailable:
            request.body.side_effect = RuntimeError("body unavailable")
        errors = [
            {
                "type": "extra_forbidden",
                "loc": (location, "aws_region_name"),
                "msg": "Extra inputs are not permitted",
                "input": "must-not-be-logged",
            }
        ]

        with caplog.at_level(logging.WARNING, logger="nemo_gym.server_utils"):
            await _log_validation_exception(request, RequestValidationError(errors))

        message = caplog.records[-1].getMessage()
        expected = {"type": "extra_forbidden", "loc": [location, "aws_region_name"]}
        if location == "body":
            expected["msg"] = "Extra inputs are not permitted"
        rendered = message.split("validation_errors=", 1)[1].split(" validation_errors_truncated=", 1)[0]
        assert nemo_gym.server_utils.json.loads(rendered) == [expected]
        assert "must-not-be-logged" not in message
        assert message.endswith("validation_errors_truncated=False")

    @mark.parametrize("body_unavailable", [False, True])
    async def test_validation_exception_message_bounds_rendered_errors(
        self, body_unavailable: bool, caplog: LogCaptureFixture
    ) -> None:
        request = MagicMock(spec=Request)
        request.body = AsyncMock(return_value=b"{}")
        if body_unavailable:
            request.body.side_effect = RuntimeError("body unavailable")
        long_msg = "m" * nemo_gym.server_utils._VALIDATION_ERROR_LOG_FIELD_CHARS
        errors = [
            {"type": "value_error", "loc": ("body", f"field_{index}"), "msg": long_msg, "input": None}
            for index in range(nemo_gym.server_utils._VALIDATION_ERROR_LOG_MAX_ERRORS)
        ]

        with caplog.at_level(logging.WARNING, logger="nemo_gym.server_utils"):
            await _log_validation_exception(request, RequestValidationError(errors))

        record = caplog.records[-1]
        rendered = record.getMessage().split("validation_errors=", 1)[1].split(" validation_errors_truncated=", 1)[0]
        assert len(rendered) == nemo_gym.server_utils._VALIDATION_ERROR_LOG_ERRORS_CHARS
        assert rendered.endswith("...[truncated]")
        assert record.validation_errors_truncated is False
        assert record.validation_errors == [
            {"type": error["type"], "loc": list(error["loc"]), "msg": long_msg} for error in errors
        ]
        assert record.getMessage().endswith("validation_errors_truncated=True ...[truncated]")

    @mark.parametrize("invalid_body", [False, True])
    def test_validation_exception_custom_header_query_messages_are_not_rendered(
        self, invalid_body: bool, caplog: LogCaptureFixture
    ) -> None:
        from typing import Annotated

        from fastapi import Body, FastAPI, Header, Query
        from fastapi.testclient import TestClient
        from pydantic import AfterValidator

        def check(value: str) -> str:
            raise ValueError(f"invalid api key {value}")

        app = FastAPI()
        app.exception_handler(RequestValidationError)(_validation_exception_handler)

        @app.post("/validate")
        async def validate(
            count: Annotated[int, Body(embed=True)],
            api_key: Annotated[str, Header(), AfterValidator(check)],
            query_key: Annotated[str, Query(), AfterValidator(check)],
        ) -> dict:
            return {"count": count}

        with caplog.at_level(logging.WARNING, logger="nemo_gym.server_utils"), TestClient(app) as client:
            response = client.post(
                "/validate",
                json={"count": "invalid" if invalid_body else 1},
                headers={"api-key": "sk-sensitive-header"},
                params={"query_key": "sk-sensitive-query"},
            )

        assert response.status_code == 422
        record = next(record for record in reversed(caplog.records) if record.name == "nemo_gym.server_utils")
        message = record.getMessage()
        assert "sk-sensitive-header" not in message
        assert "sk-sensitive-query" not in message
        assert "invalid api key" not in message
        rendered = message.split("validation_errors=", 1)[1].split(" validation_errors_truncated=", 1)[0]
        summaries = nemo_gym.server_utils.json.loads(rendered)
        assert {tuple(summary["loc"]) for summary in summaries} == {
            tuple(error["loc"]) for error in response.json()["detail"]
        }
        for summary in summaries:
            assert ("msg" in summary) is (summary["loc"][0] == "body")
        assert record.validation_error_count == (3 if invalid_body else 2)
        assert hasattr(record, "request_body_prefix") is invalid_body
        assert record.validation_errors_truncated is False
        # Structured consumers keep the same bounded metadata, including custom messages.
        assert record.validation_errors == [
            {key: error[key] for key in ("type", "loc", "msg")} for error in response.json()["detail"]
        ]

    @mark.parametrize("location", [("path", "key"), ("cookie", "key"), (), ("unknown", "key"), None])
    async def test_validation_exception_non_body_messages_are_not_rendered(
        self, location: tuple[str, ...] | None, caplog: LogCaptureFixture
    ) -> None:
        request = MagicMock(spec=Request)
        request.body = AsyncMock(return_value=b"must-not-be-read")
        errors = [{"type": "value_error", "loc": location, "msg": "invalid api key sk-sensitive-value"}]

        with caplog.at_level(logging.WARNING, logger="nemo_gym.server_utils"):
            await _log_validation_exception(request, RequestValidationError(errors))

        request.body.assert_not_awaited()
        record = caplog.records[-1]
        assert "sk-sensitive-value" not in record.getMessage()
        assert '"msg"' not in record.getMessage()
        assert record.validation_errors[0]["msg"] == errors[0]["msg"]

    async def test_validation_exception_does_not_log_body_for_query_error(self, caplog: LogCaptureFixture) -> None:
        request = MagicMock(spec=Request)
        request.body = AsyncMock(return_value=b"password=must-not-be-logged")
        errors = [{"type": "missing", "loc": ("query", "required"), "msg": "Field required", "input": None}]

        with caplog.at_level(logging.WARNING, logger="nemo_gym.server_utils"):
            await _log_validation_exception(request, RequestValidationError(errors, body=None))

        request.body.assert_not_awaited()
        record = caplog.records[-1]
        assert not hasattr(record, "request_body_prefix")
        assert "must-not-be-logged" not in record.getMessage()
        assert "must-not-be-logged" not in str(record.validation_errors)

    def test_validation_exception_query_error_does_not_disclose_body(self, caplog: LogCaptureFixture) -> None:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        app = FastAPI()
        app.exception_handler(RequestValidationError)(_validation_exception_handler)

        @app.post("/query")
        async def query(required: str) -> dict:
            return {"required": required}

        with caplog.at_level(logging.WARNING, logger="nemo_gym.server_utils"), TestClient(app) as client:
            response = client.post("/query", content=b"password=must-not-be-logged")

        assert response.status_code == 422
        record = next(record for record in reversed(caplog.records) if record.name == "nemo_gym.server_utils")
        assert not hasattr(record, "request_body_prefix")
        assert "must-not-be-logged" not in record.getMessage()
        assert "must-not-be-logged" not in str(record.validation_errors)

    async def test_validation_exception_body_unavailable_is_logged(self, caplog: LogCaptureFixture) -> None:
        request = MagicMock(spec=Request)
        request.body = AsyncMock(side_effect=RuntimeError("body unavailable"))
        errors = [{"type": "missing", "loc": ("body", "field"), "msg": "Field required", "input": {}}]
        exc = RequestValidationError(errors, body={"field": None})

        with caplog.at_level(logging.WARNING, logger="nemo_gym.server_utils"):
            await _log_validation_exception(request, exc)

        record = caplog.records[-1]
        assert record.getMessage().startswith("Request validation failed; request body unavailable")
        assert record.validation_error_count == 1
        assert exc.errors() == errors
        assert exc.body == {"field": None}

    async def test_validation_exception_logging_failure_does_not_mask_422(self, monkeypatch: MonkeyPatch) -> None:
        request = MagicMock(spec=Request)
        request.body = AsyncMock(return_value=b"{}")
        exc = RequestValidationError(
            [{"type": "missing", "loc": ("body", "field"), "msg": "Field required", "input": None}]
        )
        expected = await request_validation_exception_handler(request, exc)
        monkeypatch.setattr(
            nemo_gym.server_utils.logger, "warning", MagicMock(side_effect=RuntimeError("sink failed"))
        )

        actual = await _validation_exception_handler(request, exc)

        assert actual.status_code == expected.status_code == 422
        assert actual.body == expected.body
        assert actual.headers == expected.headers

    async def test_validation_exception_handler_preserves_fastapi_response(self, caplog: LogCaptureFixture) -> None:
        request = MagicMock(spec=Request)
        request.body = AsyncMock(return_value=b'{"field":null}')
        exc = RequestValidationError(
            [{"type": "missing", "loc": ("body", "required"), "msg": "Field required", "input": None}]
        )
        expected = await request_validation_exception_handler(request, exc)

        with caplog.at_level(logging.WARNING, logger="nemo_gym.server_utils"):
            actual = await _validation_exception_handler(request, exc)

        assert actual.status_code == expected.status_code == 422
        assert actual.body == expected.body
        assert actual.headers == expected.headers

    async def test_exception_middleware_logs_upstream_error_without_debug(
        self, monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]
    ) -> None:
        callbacks = []
        app = MagicMock()

        def register_middleware(middleware_type):
            assert middleware_type == "http"

            def register(callback):
                callbacks.append(callback)
                return callback

            return register

        app.middleware.side_effect = register_middleware
        server = MagicMock()
        server.get_session_middleware_key.return_value = "TestSimpleServer___my_server"
        SimpleServer.setup_exception_middleware(server, app)

        request_info = RequestInfo(
            url=URL("http://policy.test/v1/responses"),
            method="POST",
            headers=CIMultiDictProxy(CIMultiDict()),
            real_url=URL("http://policy.test/v1/responses"),
        )
        error = ClientResponseError(request_info=request_info, history=(), status=500, message="policy failed")
        error.response_content = b"ValueError: actionable inner failure"

        async def fail(_request):
            raise error

        monkeypatch.setattr("nemo_gym.client.http_client._GLOBAL_AIOHTTP_CLIENT_REQUEST_DEBUG", False)
        response = await callbacks[0](MagicMock(), fail)

        assert response.status_code == 500
        captured = capsys.readouterr().out
        assert "[upstream_request_failed]" in captured
        assert "ValueError: actionable inner failure" in captured


_SPOOFED_HOST = "203.0.113.99"
_LOOPBACK = "127.0.0.1"


async def _scope_seen_by_app(*, uvicorn_kwargs: dict, peer: str, forwarded: bool) -> dict:
    """Drive uvicorn's loaded app with one request and return the scope the inner app observed."""
    seen: dict = {}

    async def recorder(scope, receive, send) -> None:
        seen["client"] = scope.get("client")
        seen["scheme"] = scope["scheme"]

    # Take the proxy settings straight from what run_webserver produced, so this exercises
    # Gym's wiring rather than restating uvicorn's defaults.
    config = uvicorn.Config(
        app=recorder,
        proxy_headers=uvicorn_kwargs["proxy_headers"],
        forwarded_allow_ips=uvicorn_kwargs["forwarded_allow_ips"],
    )
    config.load()

    headers = []
    if forwarded:
        headers = [(b"x-forwarded-for", _SPOOFED_HOST.encode()), (b"x-forwarded-proto", b"https")]

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "method": "GET",
        "path": "/",
        "raw_path": b"/",
        "query_string": b"",
        "root_path": "",
        "scheme": "http",
        "headers": headers,
        "client": (peer, 54321),
        "server": (_LOOPBACK, 8000),
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message) -> None:
        return None

    await config.loaded_app(scope, receive, send)
    return seen


class TestUvicornProxyHeadersConfig:
    def test_disabled_by_default(self) -> None:
        config = UvicornProxyHeadersConfig.model_validate({})

        assert config.uvicorn_proxy_headers is False
        assert config.uvicorn_forwarded_allow_ips is None

    def test_unrelated_config_keys_are_ignored(self) -> None:
        config = UvicornProxyHeadersConfig.model_validate({"uvicorn_logging_show_200_ok": True, "port": 1234})

        assert config.uvicorn_proxy_headers is False

    def test_enabling_without_allowlist_is_rejected(self) -> None:
        with raises(ValidationError, match="requires a non-empty uvicorn_forwarded_allow_ips"):
            UvicornProxyHeadersConfig.model_validate({"uvicorn_proxy_headers": True})

    def test_enabling_with_empty_allowlist_is_rejected(self) -> None:
        with raises(ValidationError, match="requires a non-empty uvicorn_forwarded_allow_ips"):
            UvicornProxyHeadersConfig.model_validate(
                {"uvicorn_proxy_headers": True, "uvicorn_forwarded_allow_ips": ["  "]}
            )

    def test_wildcard_allowlist_is_rejected(self) -> None:
        with raises(ValidationError, match="must not be"):
            UvicornProxyHeadersConfig.model_validate(
                {"uvicorn_proxy_headers": True, "uvicorn_forwarded_allow_ips": ["10.0.0.1", "*"]}
            )

    def test_all_address_networks_are_rejected(self) -> None:
        for network in ("0.0.0.0/0", "::/0"):
            with raises(ValidationError, match="covers every address"):
                UvicornProxyHeadersConfig.model_validate(
                    {"uvicorn_proxy_headers": True, "uvicorn_forwarded_allow_ips": [network]}
                )

    def test_unparseable_allowlist_entries_are_rejected(self) -> None:
        """uvicorn keeps an unparseable entry as a literal that never matches a TCP peer, so the
        allowlist would look populated while trusting nobody."""
        for bad in ("10.0.0.5/24", "proxy.internal", "not an ip", "0/0"):
            with raises(ValidationError, match="is not a valid IP address or CIDR range"):
                UvicornProxyHeadersConfig.model_validate(
                    {"uvicorn_proxy_headers": True, "uvicorn_forwarded_allow_ips": [bad]}
                )

    def test_ipv4_mapped_all_address_network_is_rejected(self) -> None:
        """::ffff:0:0/96 has prefixlen 96 but trusts every IPv4 peer on a dual-stack socket."""
        for bad in ("::ffff:0:0/96", "::ffff:0.0.0.0/96"):
            with raises(ValidationError, match="covers every address"):
                UvicornProxyHeadersConfig.model_validate(
                    {"uvicorn_proxy_headers": True, "uvicorn_forwarded_allow_ips": [bad]}
                )

    def test_valid_cidr_ranges_are_accepted(self) -> None:
        config = UvicornProxyHeadersConfig.model_validate(
            {"uvicorn_proxy_headers": True, "uvicorn_forwarded_allow_ips": ["10.0.1.0/24", "10.0.0.1"]}
        )

        assert ["10.0.1.0/24", "10.0.0.1"] == config.uvicorn_forwarded_allow_ips

    def test_allowlist_is_normalized(self) -> None:
        config = UvicornProxyHeadersConfig.model_validate(
            {"uvicorn_proxy_headers": True, "uvicorn_forwarded_allow_ips": [" 10.0.0.1 ", "", "10.0.0.2"]}
        )

        assert ["10.0.0.1", "10.0.0.2"] == config.uvicorn_forwarded_allow_ips


class TestUvicornProxyHeadersBehavior:
    """End-to-end: the kwargs run_webserver builds are fed to uvicorn and the resulting
    middleware stack is driven with a real request."""

    def _kwargs(self, monkeypatch: MonkeyPatch, config_dict: dict) -> dict:
        return TestRunWebserverProxyKwargs()._capture_uvicorn_kwargs(monkeypatch, config_dict, num_workers=1)

    async def test_forwarded_headers_ignored_on_the_default_internal_path(self, monkeypatch: MonkeyPatch) -> None:
        """Gym's default config must leave the real peer and scheme intact despite forged headers."""
        seen = await _scope_seen_by_app(uvicorn_kwargs=self._kwargs(monkeypatch, {}), peer=_LOOPBACK, forwarded=True)

        assert (_LOOPBACK, 54321) == seen["client"]
        assert "http" == seen["scheme"]

    async def test_real_peer_reported_when_no_forwarded_headers_are_sent(self, monkeypatch: MonkeyPatch) -> None:
        seen = await _scope_seen_by_app(uvicorn_kwargs=self._kwargs(monkeypatch, {}), peer=_LOOPBACK, forwarded=False)

        assert (_LOOPBACK, 54321) == seen["client"]
        assert "http" == seen["scheme"]

    async def test_forwarded_headers_honored_for_trusted_proxy(self, monkeypatch: MonkeyPatch) -> None:
        kwargs = self._kwargs(monkeypatch, {"uvicorn_proxy_headers": True, "uvicorn_forwarded_allow_ips": [_LOOPBACK]})
        seen = await _scope_seen_by_app(uvicorn_kwargs=kwargs, peer=_LOOPBACK, forwarded=True)

        assert (_SPOOFED_HOST, 0) == seen["client"]
        assert "https" == seen["scheme"]

    async def test_forwarded_headers_honored_for_trusted_cidr_range(self, monkeypatch: MonkeyPatch) -> None:
        """A CIDR allowlist entry, as the configuration docs advertise, must actually match."""
        kwargs = self._kwargs(
            monkeypatch, {"uvicorn_proxy_headers": True, "uvicorn_forwarded_allow_ips": ["10.0.1.0/24"]}
        )
        seen = await _scope_seen_by_app(uvicorn_kwargs=kwargs, peer="10.0.1.55", forwarded=True)

        assert (_SPOOFED_HOST, 0) == seen["client"]
        assert "https" == seen["scheme"]

    async def test_forwarded_headers_ignored_from_untrusted_peer(self, monkeypatch: MonkeyPatch) -> None:
        """Opt-in enabled, but the caller is not on the allowlist, so its claims are discarded."""
        kwargs = self._kwargs(
            monkeypatch, {"uvicorn_proxy_headers": True, "uvicorn_forwarded_allow_ips": ["10.0.0.1"]}
        )
        seen = await _scope_seen_by_app(uvicorn_kwargs=kwargs, peer=_LOOPBACK, forwarded=True)

        assert (_LOOPBACK, 54321) == seen["client"]
        assert "http" == seen["scheme"]


class TestRunWebserverProxyKwargs:
    """run_webserver must forward the proxy config into uvicorn on both launch paths."""

    def _capture_uvicorn_kwargs(
        self,
        monkeypatch: MonkeyPatch,
        config_dict: dict,
        num_workers: int | None,
        ray_enabled: bool | None = None,
        is_worker: bool = False,
        routes_sessions: bool = False,
        unsupported_reason: str | None = None,
    ) -> dict:
        from fastapi import FastAPI

        global_config = DictConfig({DRY_RUN_KEY_NAME: False, "my_server": {"a": {"b": {}}}, **config_dict})
        ray_mock = MagicMock()
        ray_mock.is_initialized.return_value = True
        self.ray_loader_mock = MagicMock(return_value=ray_mock)
        monkeypatch.setattr(nemo_gym.server_utils, "_get_ray", self.ray_loader_mock)
        monkeypatch.setattr(nemo_gym.server_utils, "get_global_config_dict", MagicMock(return_value=global_config))
        server_client = ServerClient(
            head_server_config=BaseServerConfig(host="", port=0), global_config_dict=DictConfig({})
        )
        server_client_mock = MagicMock(return_value=server_client)
        server_client_mock.load_head_server_config = MagicMock(return_value=BaseServerConfig(host="", port=0))
        monkeypatch.setattr(nemo_gym.server_utils, "ServerClient", server_client_mock)
        monkeypatch.setattr(
            nemo_gym.server_utils,
            "is_nemo_gym_fastapi_worker",
            MagicMock(return_value=is_worker),
        )

        captured: dict = {}
        self.uvicorn_kwargs = captured
        monkeypatch.setattr(nemo_gym.server_utils.uvicorn, "run", lambda **kwargs: captured.update(kwargs))

        server_config = BaseRunServerInstanceConfig(
            name="my_server", host="127.0.0.1", port=8000, entrypoint="app.py", num_workers=num_workers
        )
        ray_setting = ray_enabled

        class TestSimpleServer(SimpleServer):
            ray_enabled = ray_setting
            routes_sessions_to_owner = routes_sessions

            def claimed_session_routing(self) -> dict:
                return {}

            @classmethod
            def load_config_from_global_config(cls):
                return server_config

            def multiple_workers_unsupported_reason(self) -> str | None:
                return unsupported_reason

            def setup_webserver(self) -> FastAPI:
                return FastAPI()

            def setup_telemetry(self) -> None: ...
            def set_ulimit(self) -> None: ...
            def prefix_server_logs(self) -> None: ...
            def setup_exception_middleware(self, app) -> None: ...
            def setup_cancellation_middleware(self, app) -> None: ...
            def instrument_app_for_telemetry(self, app) -> None: ...

        TestSimpleServer.run_webserver()
        return captured

    def test_proxy_headers_disabled_by_default_single_worker(self, monkeypatch: MonkeyPatch) -> None:
        kwargs = self._capture_uvicorn_kwargs(monkeypatch, {}, num_workers=1)

        # Servers that don't declare ray_enabled = True run without Ray.
        self.ray_loader_mock.assert_not_called()
        assert kwargs["proxy_headers"] is False
        assert [] == kwargs["forwarded_allow_ips"]
        # A single worker passes the app object itself rather than an import string.
        assert not isinstance(kwargs["app"], str)
        assert "workers" not in kwargs

    def test_proxy_headers_disabled_by_default_multi_worker(self, monkeypatch: MonkeyPatch) -> None:
        kwargs = self._capture_uvicorn_kwargs(monkeypatch, {}, num_workers=4)

        self.ray_loader_mock.assert_not_called()
        # Multi-worker launches re-import the app, so uvicorn receives an import string.
        assert isinstance(kwargs["app"], str)
        assert kwargs["app"].endswith(":app")
        assert 4 == kwargs["workers"]
        assert kwargs["proxy_headers"] is False
        assert [] == kwargs["forwarded_allow_ips"]

    def test_server_with_a_reason_refuses_several_workers_before_uvicorn_starts(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        # Recorded with monkeypatch, so the variable is restored even if a launch sets it.
        monkeypatch.setenv(IS_NEMO_GYM_FASTAPI_WORKER_KEY_NAME, "")
        monkeypatch.delenv(IS_NEMO_GYM_FASTAPI_WORKER_KEY_NAME)

        with raises(ValueError, match="^my_server needs num_workers=1: sessions live in this process$"):
            self._capture_uvicorn_kwargs(
                monkeypatch, {}, num_workers=2, unsupported_reason="sessions live in this process"
            )

        assert self.uvicorn_kwargs == {}
        # The refusal comes before the multi-worker launch sets up the workers' environment.
        assert IS_NEMO_GYM_FASTAPI_WORKER_KEY_NAME not in os.environ

    def test_server_with_a_reason_still_runs_one_worker(self, monkeypatch: MonkeyPatch) -> None:
        kwargs = self._capture_uvicorn_kwargs(
            monkeypatch, {}, num_workers=1, unsupported_reason="sessions live in this process"
        )

        assert "workers" not in kwargs

    def test_server_without_a_reason_runs_several_workers(self, monkeypatch: MonkeyPatch) -> None:
        kwargs = self._capture_uvicorn_kwargs(monkeypatch, {}, num_workers=2)

        assert 2 == kwargs["workers"]

    def test_undeclared_server_does_not_join_a_configured_cluster(self, monkeypatch: MonkeyPatch) -> None:
        self._capture_uvicorn_kwargs(monkeypatch, {"ray_head_node_address": "10.0.0.1:6379"}, num_workers=1)

        self.ray_loader_mock.assert_not_called()

    def test_ray_disabled_skips_initialization(self, monkeypatch: MonkeyPatch) -> None:
        self._capture_uvicorn_kwargs(monkeypatch, {}, num_workers=1, ray_enabled=False)

        self.ray_loader_mock.assert_not_called()

    def test_multi_worker_child_initializes_ray(self, monkeypatch: MonkeyPatch) -> None:
        self._capture_uvicorn_kwargs(monkeypatch, {}, num_workers=4, ray_enabled=True, is_worker=True)

        self.ray_loader_mock.assert_called_once()

    def test_unrelated_uvicorn_settings_are_unchanged(self, monkeypatch: MonkeyPatch) -> None:
        """The issue calls out parser, keepalive, access-log, and graceful-shutdown as must-not-change."""
        kwargs = self._capture_uvicorn_kwargs(monkeypatch, {}, num_workers=1)

        # Still the httptools parser (never an h11 fallback), now with TCP keepalive on accepted connections.
        assert issubclass(kwargs["http"].func, HttpToolsProtocol)
        assert 30 == kwargs["timeout_keep_alive"]
        assert kwargs["access_log"] is False
        assert 0.5 == kwargs["timeout_graceful_shutdown"]

    def test_server_tcp_keepalive_uses_global_aiohttp_keepalive_config(self, monkeypatch: MonkeyPatch) -> None:
        kwargs = self._capture_uvicorn_kwargs(monkeypatch, {}, num_workers=1)
        assert kwargs["http"].func is KeepaliveHttpToolsProtocol
        assert kwargs["http"].keywords["keepalive"] == (60, 10, 3)

        kwargs = self._capture_uvicorn_kwargs(
            monkeypatch,
            {"global_aiohttp_tcp_keepalive_idle_seconds": 90, "global_aiohttp_tcp_keepalive_probes": 5},
            num_workers=4,
        )
        assert kwargs["http"].keywords["keepalive"] == (90, 10, 5)
        # Multi-worker uvicorn pickles its config into spawned worker processes.
        restored = pickle.loads(pickle.dumps(kwargs["http"]))
        assert restored.func is KeepaliveHttpToolsProtocol
        assert restored.keywords["keepalive"] == (90, 10, 5)

    @mark.skipif(_TCP_KEEPIDLE_OPT is None, reason="platform has no TCP keepalive idle option")
    async def test_uvicorn_applies_tcp_keepalive_to_accepted_connections(self, monkeypatch: MonkeyPatch) -> None:
        """Start a real uvicorn server with the `http` protocol that run_webserver builds and serve one request."""
        http_protocol = self._capture_uvicorn_kwargs(
            monkeypatch, {"global_aiohttp_tcp_keepalive_idle_seconds": 90}, num_workers=1
        )["http"]

        async def app(scope, receive, send) -> None:
            await send({"type": "http.response.start", "status": 200, "headers": [(b"content-length", b"2")]})
            await send({"type": "http.response.body", "body": b"ok"})

        server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=0, http=http_protocol, lifespan="off", log_level="warning")
        )
        serve_task = asyncio.create_task(server.serve())
        try:
            while not server.started:
                await asyncio.sleep(0.01)
            port = server.servers[0].sockets[0].getsockname()[1]
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(b"GET / HTTP/1.1\r\nHost: test\r\n\r\n")
            await writer.drain()
            response = await asyncio.wait_for(reader.readuntil(b"ok"), timeout=5)
            assert response.startswith(b"HTTP/1.1 200")

            # HTTP/1.1 keeps the connection open, so the server side of it can be inspected.
            (connection,) = server.server_state.connections
            accepted_sock = connection.transport.get_extra_info("socket")
            assert accepted_sock.getsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE) != 0
            assert 90 == accepted_sock.getsockopt(socket.IPPROTO_TCP, _TCP_KEEPIDLE_OPT)
            assert 10 == accepted_sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL)
            assert 3 == accepted_sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT)

            writer.close()
            await writer.wait_closed()
        finally:
            server.should_exit = True
            await asyncio.wait_for(serve_task, timeout=5)

    def test_trusted_proxy_opt_in_is_forwarded_to_uvicorn(self, monkeypatch: MonkeyPatch) -> None:
        kwargs = self._capture_uvicorn_kwargs(
            monkeypatch,
            {"uvicorn_proxy_headers": True, "uvicorn_forwarded_allow_ips": ["10.0.0.1"]},
            num_workers=1,
        )

        assert kwargs["proxy_headers"] is True
        assert ["10.0.0.1"] == kwargs["forwarded_allow_ips"]

    def test_enabling_without_allowlist_fails_startup(self, monkeypatch: MonkeyPatch) -> None:
        with raises(ValidationError, match="requires a non-empty uvicorn_forwarded_allow_ips"):
            self._capture_uvicorn_kwargs(monkeypatch, {"uvicorn_proxy_headers": True}, num_workers=1)

    @mark.parametrize(("is_worker", "num_workers"), [(False, None), (False, 1), (False, 4), (True, 4)])
    def test_connection_pool_report_is_printed_only_by_the_main_process(
        self, monkeypatch: MonkeyPatch, is_worker: bool, num_workers: int | None
    ) -> None:
        report = MagicMock()
        monkeypatch.setattr(nemo_gym.server_utils, "report_connection_pool_capacity", report)

        self._capture_uvicorn_kwargs(monkeypatch, {}, num_workers=num_workers, is_worker=is_worker)

        if is_worker:
            report.assert_not_called()
        else:
            report.assert_called_once()
            assert report.call_args.args[1].workers == (num_workers or 1)
            assert report.call_args.kwargs == {"visible": True}

    @mark.parametrize("is_worker", [False, True])
    def test_positive_limit_that_divides_to_zero_fails_before_uvicorn_starts(
        self, monkeypatch: MonkeyPatch, is_worker: bool
    ) -> None:
        report = MagicMock()
        monkeypatch.setattr(nemo_gym.server_utils, "report_connection_pool_capacity", report)

        with raises(ValueError, match="must remain at least 1"):
            self._capture_uvicorn_kwargs(
                monkeypatch, {"global_aiohttp_connector_limit_per_host": 8}, num_workers=16, is_worker=is_worker
            )

        assert self.uvicorn_kwargs == {}
        report.assert_not_called()

    @mark.parametrize("num_workers", [1, 4])
    def test_listen_backlog_defaults_above_uvicorns(self, monkeypatch: MonkeyPatch, num_workers: int) -> None:
        kwargs = self._capture_uvicorn_kwargs(monkeypatch, {}, num_workers=num_workers)

        assert 16384 == kwargs["backlog"]

    def test_listen_backlog_comes_from_the_global_config(self, monkeypatch: MonkeyPatch) -> None:
        kwargs = self._capture_uvicorn_kwargs(monkeypatch, {"uvicorn_backlog": 4096}, num_workers=4)

        assert 4096 == kwargs["backlog"]

    def test_a_workers_private_socket_gets_the_listen_backlog(self, monkeypatch: MonkeyPatch) -> None:
        install = MagicMock()
        monkeypatch.setattr(nemo_gym.server_utils, "install_session_routing", install)

        self._capture_uvicorn_kwargs(
            monkeypatch, {"uvicorn_backlog": 4096}, num_workers=4, is_worker=True, routes_sessions=True
        )

        assert 4096 == install.call_args.kwargs["backlog"]

    def test_a_listen_backlog_below_one_fails_before_uvicorn_starts(self, monkeypatch: MonkeyPatch) -> None:
        with raises(ValidationError, match="uvicorn_backlog"):
            self._capture_uvicorn_kwargs(monkeypatch, {"uvicorn_backlog": 0}, num_workers=1)

        assert self.uvicorn_kwargs == {}


class TestHeadServerProxyKwargs:
    """The independently launched head server must use the same proxy-header policy."""

    @staticmethod
    def _capture_uvicorn_kwargs(monkeypatch: MonkeyPatch, config_dict: dict) -> dict:
        monkeypatch.setattr(
            ServerClient,
            "load_head_server_config",
            MagicMock(return_value=BaseServerConfig(host="127.0.0.1", port=11000)),
        )
        monkeypatch.setattr(
            nemo_gym.server_utils,
            "get_global_config_dict",
            MagicMock(return_value=DictConfig(config_dict)),
        )

        captured: dict = {}

        def capture_config(app, **kwargs):
            captured.update(kwargs)
            return MagicMock()

        monkeypatch.setattr(nemo_gym.server_utils.uvicorn, "Config", capture_config)
        monkeypatch.setattr(nemo_gym.server_utils.uvicorn, "Server", MagicMock(return_value=MagicMock()))
        monkeypatch.setattr(nemo_gym.server_utils, "Thread", MagicMock(return_value=MagicMock()))

        HeadServer.run_webserver()
        return captured

    def test_proxy_headers_are_disabled_by_default(self, monkeypatch: MonkeyPatch) -> None:
        kwargs = self._capture_uvicorn_kwargs(monkeypatch, {})

        assert kwargs["proxy_headers"] is False
        assert kwargs["forwarded_allow_ips"] == []

    def test_trusted_proxy_opt_in_is_forwarded(self, monkeypatch: MonkeyPatch) -> None:
        kwargs = self._capture_uvicorn_kwargs(
            monkeypatch,
            {"uvicorn_proxy_headers": True, "uvicorn_forwarded_allow_ips": ["10.0.0.1"]},
        )

        assert kwargs["proxy_headers"] is True
        assert kwargs["forwarded_allow_ips"] == ["10.0.0.1"]

    def test_listen_backlog_follows_the_global_config(self, monkeypatch: MonkeyPatch) -> None:
        assert 16384 == self._capture_uvicorn_kwargs(monkeypatch, {})["backlog"]
        assert 4096 == self._capture_uvicorn_kwargs(monkeypatch, {"uvicorn_backlog": 4096})["backlog"]
