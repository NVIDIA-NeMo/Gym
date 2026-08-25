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
import errno
import socket
from unittest.mock import AsyncMock, MagicMock

from aiohttp import (
    ClientConnectionResetError,
    ClientConnectorDNSError,
    ClientConnectorError,
    ClientError,
    ClientOSError,
    ClientPayloadError,
    ClientProxyConnectionError,
    ServerDisconnectedError,
)
from aiohttp.client_exceptions import ConnectionTimeoutError, InvalidUrlClientError, SocketTimeoutError
from aiohttp.client_reqrep import ConnectionKey
from pytest import CaptureFixture, MonkeyPatch, raises

import nemo_gym.global_config
import nemo_gym.server_utils
from nemo_gym.global_config import (
    NEMO_GYM_CONFIG_PATH_ENV_VAR_NAME,
)
from nemo_gym.server_utils import (
    BaseServer,
    BaseServerConfig,
    ConnectionError,
    DictConfig,
    GlobalAIOHTTPAsyncClientConfig,
    HeadServer,
    ServerClient,
    SimpleServer,
    _classify_request_failure,
    _make_keepalive_socket_factory,
    _next_failure_report,
    _report_request_failure,
    _request_origin,
    _RequestFailureKind,
    initialize_ray,
)


def _connection_key() -> ConnectionKey:
    return ConnectionKey(
        host="localhost",
        port=8000,
        is_ssl=False,
        ssl=True,
        proxy=None,
        proxy_auth=None,
        proxy_headers_hash=None,
    )


_TCP_KEEPALIVE_TEST_IDLE = 42
_TCP_KEEPALIVE_TEST_INTERVAL = 7
_TCP_KEEPALIVE_TEST_PROBES = 2
_TEST_ADDR_INFO = (
    socket.AF_INET,
    socket.SOCK_STREAM,
    socket.IPPROTO_TCP,
    "",
    ("203.0.113.1", 443),
)


class TestServerUtils:
    def test_global_aiohttp_client_request_debug_enabled(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr(nemo_gym.server_utils, "_GLOBAL_AIOHTTP_CLIENT_REQUEST_DEBUG", False)
        assert not nemo_gym.server_utils.is_global_aiohttp_client_request_debug_enabled()

        monkeypatch.setattr(nemo_gym.server_utils, "_GLOBAL_AIOHTTP_CLIENT_REQUEST_DEBUG", True)
        assert nemo_gym.server_utils.is_global_aiohttp_client_request_debug_enabled()

    def test_ServerClient_load_head_server_config(self, monkeypatch: MonkeyPatch) -> None:
        global_config_dict = DictConfig(
            {
                "head_server": {
                    "host": "",
                    "port": 0,
                }
            }
        )
        get_global_config_dict_mock = MagicMock()
        get_global_config_dict_mock.return_value = global_config_dict
        monkeypatch.setattr(nemo_gym.server_utils, "get_global_config_dict", get_global_config_dict_mock)
        actual_config = ServerClient.load_head_server_config()
        assert actual_config.host == ""
        assert actual_config.port == 0

    def test_ServerClient_load_from_global_config(self, monkeypatch: MonkeyPatch) -> None:
        global_config_dict = DictConfig(
            {
                "head_server": {
                    "host": "",
                    "port": 0,
                }
            }
        )
        get_global_config_dict_mock = MagicMock()
        get_global_config_dict_mock.return_value = global_config_dict
        monkeypatch.setattr(nemo_gym.server_utils, "get_global_config_dict", get_global_config_dict_mock)

        httpx_client_mock = MagicMock()
        httpx_response_mock = MagicMock()
        httpx_client_mock.return_value = httpx_response_mock
        httpx_response_mock.content = b'"a: 2"'
        monkeypatch.setattr(nemo_gym.server_utils.requests, "get", httpx_client_mock)

        actual_client = ServerClient.load_from_global_config()
        assert {"a": 2} == actual_client.global_config_dict

    def test_ServerClient_load_from_global_config_propogate_ConnectionError(self, monkeypatch: MonkeyPatch) -> None:
        global_config_dict = DictConfig(
            {
                "head_server": {
                    "host": "",
                    "port": 0,
                }
            }
        )
        get_global_config_dict_mock = MagicMock()
        get_global_config_dict_mock.return_value = global_config_dict
        monkeypatch.setattr(nemo_gym.server_utils, "get_global_config_dict", get_global_config_dict_mock)

        httpx_client_mock = MagicMock()
        httpx_client_mock.side_effect = ConnectionError
        monkeypatch.setattr(nemo_gym.server_utils.requests, "get", httpx_client_mock)

        with raises(ValueError):
            ServerClient.load_from_global_config()

    async def test_ServerClient_get_post_sanity(self, monkeypatch: MonkeyPatch) -> None:
        server_client = ServerClient(
            head_server_config=BaseServerConfig(host="abcdef", port=12345),
            global_config_dict=DictConfig(
                {
                    "my_server": {
                        "a": {
                            "b": {
                                "host": "xyz",
                                "port": 54321,
                            }
                        }
                    }
                }
            ),
        )

        httpx_client_mock = MagicMock()
        httpx_client_request_mock = AsyncMock()
        httpx_client_request_mock.return_value = "my mock response"
        httpx_client_mock.return_value.request = httpx_client_request_mock
        monkeypatch.setattr(nemo_gym.server_utils, "get_global_aiohttp_client", httpx_client_mock)

        actual_response = await server_client.get(
            server_name="my_server",
            url_path="blah blah",
        )
        assert "my mock response" == actual_response

        actual_response = await server_client.post(
            server_name="my_server",
            url_path="blah blah",
        )
        assert "my mock response" == actual_response

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

    async def test_HeadServer_global_config_dict_yaml(self, monkeypatch: MonkeyPatch) -> None:
        global_config_dict = DictConfig({"a": 2})
        get_global_config_dict_mock = MagicMock()
        get_global_config_dict_mock.return_value = global_config_dict
        monkeypatch.setattr(nemo_gym.server_utils, "get_global_config_dict", get_global_config_dict_mock)

        head_server = HeadServer(config=BaseServerConfig(host="", port=0))
        resp = await head_server.global_config_dict_yaml()

        assert "a: 2\n" == resp

    def _mock_ray_return_value(self, monkeypatch: MonkeyPatch, return_value: bool) -> MagicMock:
        ray_is_initialized_mock = MagicMock()
        ray_is_initialized_mock.return_value = return_value
        monkeypatch.setattr(nemo_gym.server_utils.ray, "is_initialized", ray_is_initialized_mock)
        return ray_is_initialized_mock

    def _mock_ray_init(self, monkeypatch: MonkeyPatch) -> MagicMock:
        ray_init_mock = MagicMock()
        monkeypatch.setattr(nemo_gym.server_utils.ray, "init", ray_init_mock)
        return ray_init_mock

    def test_initialize_ray_already_initialized(self, monkeypatch: MonkeyPatch) -> None:
        ray_is_initialized_mock = self._mock_ray_return_value(monkeypatch, True)

        get_global_config_dict_mock = MagicMock()
        monkeypatch.setattr(nemo_gym.server_utils, "get_global_config_dict", get_global_config_dict_mock)

        initialize_ray()

        ray_is_initialized_mock.assert_called_once()
        get_global_config_dict_mock.assert_not_called()

    def test_initialize_ray_with_address(self, monkeypatch: MonkeyPatch) -> None:
        ray_is_initialized_mock = self._mock_ray_return_value(monkeypatch, False)

        ray_init_mock = self._mock_ray_init(monkeypatch)

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

        ray_init_mock = self._mock_ray_init(monkeypatch)

        ray_runtime_context_mock = MagicMock()
        ray_runtime_context_mock.gcs_address = "ray://mock-address:10001"
        ray_get_runtime_context_mock = MagicMock()
        ray_get_runtime_context_mock.return_value = ray_runtime_context_mock
        monkeypatch.setattr(nemo_gym.server_utils.ray, "get_runtime_context", ray_get_runtime_context_mock)

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

    def test_keepalive_socket_factory_sets_keepalive_sockopts(self, monkeypatch: MonkeyPatch) -> None:
        mock_sock = MagicMock()
        socket_ctor_mock = MagicMock(return_value=mock_sock)
        monkeypatch.setattr(socket, "socket", socket_ctor_mock)

        factory = _make_keepalive_socket_factory(
            idle_seconds=_TCP_KEEPALIVE_TEST_IDLE,
            interval_seconds=_TCP_KEEPALIVE_TEST_INTERVAL,
            probes=_TCP_KEEPALIVE_TEST_PROBES,
        )
        result = factory(_TEST_ADDR_INFO)

        assert result is mock_sock
        socket_ctor_mock.assert_called_once_with(
            family=_TEST_ADDR_INFO[0], type=_TEST_ADDR_INFO[1], proto=_TEST_ADDR_INFO[2]
        )
        mock_sock.setsockopt.assert_any_call(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        for opt_name, opt_value in (
            ("TCP_KEEPIDLE", _TCP_KEEPALIVE_TEST_IDLE),
            ("TCP_KEEPINTVL", _TCP_KEEPALIVE_TEST_INTERVAL),
            ("TCP_KEEPCNT", _TCP_KEEPALIVE_TEST_PROBES),
        ):
            opt = getattr(socket, opt_name, None)
            if opt is not None:
                mock_sock.setsockopt.assert_any_call(socket.IPPROTO_TCP, opt, opt_value)

    def test_keepalive_socket_factory_skips_missing_platform_sockopts(self, monkeypatch: MonkeyPatch) -> None:
        mock_sock = MagicMock()
        socket_ctor_mock = MagicMock(return_value=mock_sock)
        monkeypatch.setattr(socket, "socket", socket_ctor_mock)
        for opt_name in ("TCP_KEEPIDLE", "TCP_KEEPINTVL", "TCP_KEEPCNT"):
            monkeypatch.delattr(socket, opt_name, raising=False)

        factory = _make_keepalive_socket_factory(
            idle_seconds=_TCP_KEEPALIVE_TEST_IDLE,
            interval_seconds=_TCP_KEEPALIVE_TEST_INTERVAL,
            probes=_TCP_KEEPALIVE_TEST_PROBES,
        )
        factory(_TEST_ADDR_INFO)

        mock_sock.setsockopt.assert_called_once_with(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)

    def test_GlobalAIOHTTPAsyncClientConfig_keepalive_defaults(self) -> None:
        cfg = GlobalAIOHTTPAsyncClientConfig()
        assert cfg.global_aiohttp_tcp_keepalive_idle_seconds == 60
        assert cfg.global_aiohttp_tcp_keepalive_interval_seconds == 10
        assert cfg.global_aiohttp_tcp_keepalive_probes == 3

    def test_keepalive_socket_factory_uses_configured_values(self, monkeypatch: MonkeyPatch) -> None:
        mock_sock = MagicMock()
        socket_ctor_mock = MagicMock(return_value=mock_sock)
        monkeypatch.setattr(socket, "socket", socket_ctor_mock)

        cfg = GlobalAIOHTTPAsyncClientConfig(
            global_aiohttp_tcp_keepalive_idle_seconds=123,
            global_aiohttp_tcp_keepalive_interval_seconds=45,
            global_aiohttp_tcp_keepalive_probes=6,
        )
        factory = _make_keepalive_socket_factory(
            idle_seconds=cfg.global_aiohttp_tcp_keepalive_idle_seconds,
            interval_seconds=cfg.global_aiohttp_tcp_keepalive_interval_seconds,
            probes=cfg.global_aiohttp_tcp_keepalive_probes,
        )
        factory(_TEST_ADDR_INFO)

        for opt_name, opt_value in (
            ("TCP_KEEPIDLE", 123),
            ("TCP_KEEPINTVL", 45),
            ("TCP_KEEPCNT", 6),
        ):
            opt = getattr(socket, opt_name, None)
            if opt is not None:
                mock_sock.setsockopt.assert_any_call(socket.IPPROTO_TCP, opt, opt_value)

    def test_dry_run_skips_webserver_spinup(self, monkeypatch: MonkeyPatch) -> None:
        self._mock_ray_return_value(monkeypatch, True)

        get_global_config_dict_mock = MagicMock()
        monkeypatch.setattr(nemo_gym.server_utils, "get_global_config_dict", get_global_config_dict_mock)

        ServerClient_mock = MagicMock(spec=ServerClient)
        monkeypatch.setattr(nemo_gym.server_utils, "ServerClient", ServerClient_mock)

        class TestSimpleServer(SimpleServer):
            def __init__(self, *args, **kwargs):
                pass

            def setup_webserver(self):
                assert False

            @classmethod
            def load_config_from_global_config(cls) -> None:
                pass

        TestSimpleServer.run_webserver()

    def test_setup_session_middleware_idempotent(self) -> None:
        from fastapi import FastAPI, Request
        from fastapi.testclient import TestClient
        from starlette.middleware.sessions import SessionMiddleware

        from nemo_gym.config_types import BaseRunServerInstanceConfig
        from nemo_gym.server_utils import SESSION_ID_KEY

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

    def test_classify_request_failure_never_connected(self) -> None:
        key = _connection_key()

        assert _RequestFailureKind.UNREACHABLE == _classify_request_failure(
            ClientConnectorError(key, OSError(errno.ECONNREFUSED, "refused"))
        )
        assert _RequestFailureKind.UNREACHABLE == _classify_request_failure(
            ClientConnectorDNSError(key, OSError("name resolution failed"))
        )
        # A ClientConnectorError subclass, and easy to miss.
        assert _RequestFailureKind.UNREACHABLE == _classify_request_failure(
            ClientProxyConnectionError(key, OSError(errno.ECONNREFUSED, "refused"))
        )

    def test_classify_request_failure_resource_exhaustion(self) -> None:
        key = _connection_key()

        for err in (errno.EMFILE, errno.ENFILE, errno.ENOBUFS, errno.ENOMEM):
            assert _RequestFailureKind.RESOURCE == _classify_request_failure(
                ClientConnectorError(key, OSError(err, "out of descriptors"))
            )

    def test_classify_request_failure_peer_drop(self) -> None:
        assert _RequestFailureKind.PEER_DROP == _classify_request_failure(ServerDisconnectedError())
        assert _RequestFailureKind.PEER_DROP == _classify_request_failure(ClientConnectionResetError("reset"))
        assert _RequestFailureKind.PEER_DROP == _classify_request_failure(ClientOSError(errno.ECONNRESET, "reset"))
        # An aiohttp error we do not recognise is retried rather than failed fast.
        assert _RequestFailureKind.PEER_DROP == _classify_request_failure(ClientPayloadError("truncated"))

    def test_classify_request_failure_timeout_and_fatal(self) -> None:
        assert _RequestFailureKind.TIMEOUT == _classify_request_failure(ConnectionTimeoutError())
        assert _RequestFailureKind.TIMEOUT == _classify_request_failure(SocketTimeoutError())
        assert _RequestFailureKind.TIMEOUT == _classify_request_failure(asyncio.TimeoutError())

        assert _RequestFailureKind.FATAL == _classify_request_failure(InvalidUrlClientError("http://[bad"))
        assert _RequestFailureKind.FATAL == _classify_request_failure(TypeError("programming error"))
        assert _RequestFailureKind.FATAL == _classify_request_failure(ValueError("programming error"))

    def test_classify_request_failure_subclass_ordering(self) -> None:
        """Both orderings fail silently if broken, and breaking the first reinstates the bug."""
        # ClientConnectorError subclasses ClientOSError; testing the parent first would sort every
        # refused connection into PEER_DROP.
        refused = ClientConnectorError(_connection_key(), OSError(errno.ECONNREFUSED, "refused"))
        assert isinstance(refused, ClientOSError)
        assert _RequestFailureKind.UNREACHABLE == _classify_request_failure(refused)

        # InvalidURL is both a ClientError and a ValueError.
        bad_url = InvalidUrlClientError("http://[bad")
        assert isinstance(bad_url, ClientError) and isinstance(bad_url, ValueError)
        assert _RequestFailureKind.FATAL == _classify_request_failure(bad_url)

    def test_classify_request_failure_tolerates_missing_os_error(self) -> None:
        """`classify` runs inside an `except` block, so raising here would crash a retryable call."""
        # A bare ClientOSError carries no `os_error` attribute at all.
        assert not hasattr(ClientOSError(), "os_error")
        assert _RequestFailureKind.PEER_DROP == _classify_request_failure(ClientOSError())

        # An OSError carrying no errno at all still classifies rather than raising.
        assert _RequestFailureKind.UNREACHABLE == _classify_request_failure(
            ClientConnectorError(_connection_key(), OSError("no errno"))
        )

    def test_request_origin(self) -> None:
        assert "http://localhost:8000" == _request_origin("http://localhost:8000/v1/responses")
        assert "https://api.openai.com" == _request_origin("https://api.openai.com/v1/responses")
        assert "/v1/responses" == _request_origin("/v1/responses")

    def test_next_failure_report_throttles_per_origin_and_kind(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr(nemo_gym.server_utils, "_FAILURE_REPORT_STATE", {})
        interval = nemo_gym.server_utils._FAILURE_REPORT_INTERVAL_SECONDS

        # First sighting prints in full.
        assert 0 == _next_failure_report("http://a:8000", _RequestFailureKind.UNREACHABLE, 0.0)
        # Inside the interval, quiet, but counting.
        assert _next_failure_report("http://a:8000", _RequestFailureKind.UNREACHABLE, 1.0) is None
        assert _next_failure_report("http://a:8000", _RequestFailureKind.UNREACHABLE, 2.0) is None
        # Once it passes, one line carrying the suppressed count.
        assert 3 == _next_failure_report("http://a:8000", _RequestFailureKind.UNREACHABLE, interval)
        # The count restarts after a report.
        assert _next_failure_report("http://a:8000", _RequestFailureKind.UNREACHABLE, interval + 1) is None

        # A different kind at the same origin, and a different origin, are tracked separately.
        assert 0 == _next_failure_report("http://a:8000", _RequestFailureKind.PEER_DROP, 1.0)
        assert 0 == _next_failure_report("http://b:8000", _RequestFailureKind.UNREACHABLE, 1.0)

    def test_report_request_failure_message_matches_the_kind(
        self, monkeypatch: MonkeyPatch, capsys: CaptureFixture
    ) -> None:
        monkeypatch.setattr(nemo_gym.server_utils, "_FAILURE_REPORT_STATE", {})
        key = _connection_key()

        _report_request_failure(
            "http://localhost:8000/v1/responses",
            ClientConnectorError(key, OSError(errno.ECONNREFUSED, "refused")),
            _RequestFailureKind.UNREACHABLE,
            0.0,
        )
        unreachable = capsys.readouterr().out
        assert "http://localhost:8000/v1/responses" in unreachable
        assert "policy_base_url" in unreachable
        # The ulimit and replica advice cannot apply when no socket was opened.
        assert "Increase ulimit" not in unreachable
        assert "adapter server replicas" not in unreachable

        _report_request_failure(
            "http://localhost:8000/v1/responses",
            ClientConnectorError(key, OSError(errno.EMFILE, "too many open files")),
            _RequestFailureKind.RESOURCE,
            0.0,
        )
        assert "Increase ulimit" in capsys.readouterr().out

        _report_request_failure(
            "http://localhost:8000/v1/responses", ServerDisconnectedError(), _RequestFailureKind.PEER_DROP, 0.0
        )
        assert "adapter server replicas" in capsys.readouterr().out

    def test_report_request_failure_bounds_output_volume(
        self, monkeypatch: MonkeyPatch, capsys: CaptureFixture
    ) -> None:
        """500 in-flight requests against one dead endpoint must not print 500 diagnostics."""
        monkeypatch.setattr(nemo_gym.server_utils, "_FAILURE_REPORT_STATE", {})
        refused = ClientConnectorError(_connection_key(), OSError(errno.ECONNREFUSED, "refused"))

        for _ in range(500):
            _report_request_failure(
                "http://localhost:8000/v1/responses", refused, _RequestFailureKind.UNREACHABLE, 0.0
            )

        assert 1 == capsys.readouterr().out.count("[request_retry")

    async def test_request_still_retries_unreachable_endpoints_forever(self, monkeypatch: MonkeyPatch) -> None:
        """M1 changes reporting only. A refused connect must still retry without limit."""
        monkeypatch.setattr(nemo_gym.server_utils, "_FAILURE_REPORT_STATE", {})
        refused = ClientConnectorError(_connection_key(), OSError(errno.ECONNREFUSED, "refused"))

        num_calls = 0

        async def request_mock(**_kwargs) -> str:
            nonlocal num_calls
            num_calls += 1
            if num_calls < 200:
                raise refused
            return "my mock response"

        monkeypatch.setattr(nemo_gym.server_utils.asyncio, "sleep", AsyncMock())
        client_mock = MagicMock()
        client_mock.return_value.request = request_mock
        monkeypatch.setattr(nemo_gym.server_utils, "get_global_aiohttp_client", client_mock)

        assert "my mock response" == await nemo_gym.server_utils.request("POST", "http://localhost:8000/v1/responses")
        assert 200 == num_calls

    async def test_request_still_retries_dropped_connections_forever(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr(nemo_gym.server_utils, "_FAILURE_REPORT_STATE", {})

        num_calls = 0

        async def request_mock(**_kwargs) -> str:
            nonlocal num_calls
            num_calls += 1
            if num_calls < 200:
                raise ServerDisconnectedError()
            return "my mock response"

        monkeypatch.setattr(nemo_gym.server_utils.asyncio, "sleep", AsyncMock())
        client_mock = MagicMock()
        client_mock.return_value.request = request_mock
        monkeypatch.setattr(nemo_gym.server_utils, "get_global_aiohttp_client", client_mock)

        assert "my mock response" == await nemo_gym.server_utils.request("POST", "http://localhost:8000/v1/responses")
        assert 200 == num_calls

    async def test_request_still_raises_unknown_errors_after_max_num_tries(self, monkeypatch: MonkeyPatch) -> None:
        """The generic branch keeps its 3-tries-then-raise behaviour for external calls."""
        monkeypatch.setattr(nemo_gym.server_utils.asyncio, "sleep", AsyncMock())
        client_mock = MagicMock()
        client_mock.return_value.request = AsyncMock(side_effect=TypeError("programming error"))
        monkeypatch.setattr(nemo_gym.server_utils, "get_global_aiohttp_client", client_mock)

        with raises(TypeError):
            await nemo_gym.server_utils.request("POST", "http://localhost:8000/v1/responses")

        assert nemo_gym.server_utils.MAX_NUM_TRIES == client_mock.return_value.request.await_count

    async def test_request_success_path_is_untouched(self, monkeypatch: MonkeyPatch, capsys: CaptureFixture) -> None:
        client_mock = MagicMock()
        client_mock.return_value.request = AsyncMock(return_value="my mock response")
        monkeypatch.setattr(nemo_gym.server_utils, "get_global_aiohttp_client", client_mock)

        assert "my mock response" == await nemo_gym.server_utils.request("POST", "http://localhost:8000/v1/responses")
        assert 1 == client_mock.return_value.request.await_count
        assert "" == capsys.readouterr().out

    async def test_request_propagates_cancellation(self, monkeypatch: MonkeyPatch) -> None:
        """CancelledError is a BaseException, so `except Exception` must never swallow it."""
        client_mock = MagicMock()
        client_mock.return_value.request = AsyncMock(side_effect=asyncio.CancelledError())
        monkeypatch.setattr(nemo_gym.server_utils, "get_global_aiohttp_client", client_mock)

        with raises(asyncio.CancelledError):
            await nemo_gym.server_utils.request("POST", "http://localhost:8000/v1/responses")
