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
import multiprocessing
import socket
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from unittest.mock import AsyncMock, MagicMock

import uvicorn
from aiohttp import ClientOSError, ClientResponseError, RequestInfo
from multidict import CIMultiDict, CIMultiDictProxy
from omegaconf import OmegaConf
from pydantic import ValidationError
from pytest import MonkeyPatch, mark, raises
from uvicorn.protocols.http.httptools_impl import HttpToolsProtocol
from yarl import URL

import nemo_gym.global_config
import nemo_gym.server_utils
from nemo_gym.global_config import (
    NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME,
    NEMO_GYM_CONFIG_PATH_ENV_VAR_NAME,
)
from nemo_gym.server_utils import (
    BaseServer,
    BaseServerConfig,
    ConnectionError,
    DictConfig,
    GlobalAIOHTTPAsyncClientConfig,
    HeadServer,
    KeepaliveHttpToolsProtocol,
    ServerClient,
    SimpleServer,
    _make_keepalive_socket_factory,
    _set_tcp_keepalive,
    initialize_ray,
    raise_for_status,
)


_TCP_KEEPALIVE_TEST_IDLE = 42
_TCP_KEEPALIVE_TEST_INTERVAL = 7
_TCP_KEEPALIVE_TEST_PROBES = 2
# macOS exposes the idle option as TCP_KEEPALIVE rather than TCP_KEEPIDLE.
_TCP_KEEPIDLE_OPT = getattr(socket, "TCP_KEEPIDLE", getattr(socket, "TCP_KEEPALIVE", None))
_TEST_ADDR_INFO = (
    socket.AF_INET,
    socket.SOCK_STREAM,
    socket.IPPROTO_TCP,
    "",
    ("203.0.113.1", 443),
)


def _return_exception_from_child_process(error: ClientResponseError) -> ClientResponseError:
    return error


class TestServerUtils:
    async def test_raise_for_status_preserves_message_across_process_boundary(self) -> None:
        headers = CIMultiDictProxy(
            CIMultiDict(
                [
                    ("x-request-id", "request-123"),
                    ("Retry-After", "10"),
                    ("retry-after", "20"),
                    ("Set-Cookie", "session=abc"),
                    ("Set-Cookie", "preferences=dark"),
                ]
            )
        )
        request_info = RequestInfo(
            url=URL("http://resources-server.test/verify"),
            method="POST",
            headers=headers,
            real_url=URL("http://resources-server.test/verify"),
        )
        original_error = ClientResponseError(
            request_info=request_info,
            history=(),
            status=500,
            message="verifier failed",
            headers=headers,
        )
        response = MagicMock()
        response.ok = False
        response.content.read = AsyncMock(return_value=b'{"detail":"backend unavailable"}')
        response.request_info = request_info
        response.raise_for_status.side_effect = original_error

        with raises(ClientResponseError) as exc_info:
            await raise_for_status(response)

        error = exc_info.value
        assert str(error) == ("500, message='verifier failed', url='http://resources-server.test/verify'")
        assert error.response_content == b'{"detail":"backend unavailable"}'

        with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as executor:
            restored_error = executor.submit(_return_exception_from_child_process, error).result()

        assert isinstance(restored_error, ClientResponseError)
        assert str(restored_error) == str(error)
        assert restored_error.status == 500
        assert restored_error.message == "verifier failed"
        assert restored_error.response_content == error.response_content
        assert restored_error.request_info.method == "POST"
        assert isinstance(restored_error.request_info.headers, CIMultiDict)
        assert restored_error.request_info.headers["X-REQUEST-ID"] == "request-123"
        assert restored_error.request_info.headers.getall("RETRY-AFTER") == ["10", "20"]
        assert restored_error.request_info.headers.getall("set-cookie") == ["session=abc", "preferences=dark"]
        assert isinstance(restored_error.headers, CIMultiDict)
        assert restored_error.headers.getall("retry-after") == ["10", "20"]
        assert restored_error.headers.getall("SET-COOKIE") == ["session=abc", "preferences=dark"]

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
        """Fetch the config from the head server when no config was injected."""
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

        monkeypatch.setattr(nemo_gym.global_config, "_GLOBAL_CONFIG_DICT", None)
        monkeypatch.delenv(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, raising=False)

        httpx_client_mock = MagicMock()
        httpx_response_mock = MagicMock()
        httpx_client_mock.return_value = httpx_response_mock
        httpx_response_mock.content = b'"a: 2"'
        monkeypatch.setattr(nemo_gym.server_utils.requests, "get", httpx_client_mock)

        actual_client = ServerClient.load_from_global_config()
        assert {"a": 2} == actual_client.global_config_dict

    def test_ServerClient_load_from_global_config_fetches_when_config_was_not_injected(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        """Do not treat an unrelated process-local config as the server config."""
        global_config_dict = DictConfig(
            {
                "head_server": {"host": "", "port": 0},
                "my_server": {"a": {"b": {"host": "x", "port": 1}}},
            }
        )
        get_global_config_dict_mock = MagicMock(return_value=global_config_dict)
        monkeypatch.setattr(nemo_gym.server_utils, "get_global_config_dict", get_global_config_dict_mock)

        # `gym eval run --no-serve` initializes a partial local config.
        # It must still fetch the full config from the running head server.
        monkeypatch.setattr(nemo_gym.global_config, "_GLOBAL_CONFIG_DICT", global_config_dict)
        monkeypatch.delenv(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, raising=False)

        response = MagicMock(content=b'"remote_server: {host: remote, port: 1234}"')
        get_mock = MagicMock(return_value=response)
        monkeypatch.setattr(nemo_gym.server_utils.requests, "get", get_mock)

        client = ServerClient.load_from_global_config()
        assert client.global_config_dict == {"remote_server": {"host": "remote", "port": 1234}}
        get_mock.assert_called_once()

    def test_ServerClient_load_from_global_config_fast_path_via_env(self, monkeypatch: MonkeyPatch) -> None:
        """Use the config injected into a Gym-launched server process."""
        global_config_dict = DictConfig({"head_server": {"host": "", "port": 0}})
        get_global_config_dict_mock = MagicMock(return_value=global_config_dict)
        monkeypatch.setattr(nemo_gym.server_utils, "get_global_config_dict", get_global_config_dict_mock)

        monkeypatch.setattr(nemo_gym.global_config, "_GLOBAL_CONFIG_DICT", None)
        monkeypatch.setenv(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, "head_server: {host: '', port: 0}")

        def boom(*args, **kwargs):
            raise AssertionError("requests.get should not be called on the fast path")

        monkeypatch.setattr(nemo_gym.server_utils.requests, "get", boom)

        client = ServerClient.load_from_global_config()
        assert client.global_config_dict is global_config_dict

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

        monkeypatch.setattr(nemo_gym.global_config, "_GLOBAL_CONFIG_DICT", None)
        monkeypatch.delenv(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, raising=False)

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

    async def test_ServerClient_request_uses_base_url_table(self, monkeypatch: MonkeyPatch) -> None:
        """Resolve each server's base URL once."""
        server_client = ServerClient(
            head_server_config=BaseServerConfig(host="head", port=11000),
            global_config_dict=DictConfig({"my_server": {"a": {"b": {"host": "xyz", "port": 54321}}}}),
        )

        httpx_client_mock = MagicMock()
        httpx_client_request_mock = AsyncMock()
        httpx_client_request_mock.return_value = "ok"
        httpx_client_mock.return_value.request = httpx_client_request_mock
        monkeypatch.setattr(nemo_gym.server_utils, "get_global_aiohttp_client", httpx_client_mock)

        await server_client.post(server_name="my_server", url_path="/x")
        assert server_client._server_base_urls == {"my_server": "http://xyz:54321"}

        def boom(*_args, **_kwargs):
            raise AssertionError("get_first_server_config_dict should not be called once the URL is cached")

        monkeypatch.setattr(nemo_gym.server_utils, "get_first_server_config_dict", boom)

        await server_client.post(server_name="my_server", url_path="/y")
        await server_client.get(server_name="my_server", url_path="/z")

        assert httpx_client_request_mock.call_count == 3
        for call in httpx_client_request_mock.call_args_list:
            assert call.kwargs["url"].startswith("http://xyz:54321")

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
        for opt_name in ("TCP_KEEPIDLE", "TCP_KEEPALIVE", "TCP_KEEPINTVL", "TCP_KEEPCNT"):
            monkeypatch.delattr(socket, opt_name, raising=False)

        factory = _make_keepalive_socket_factory(
            idle_seconds=_TCP_KEEPALIVE_TEST_IDLE,
            interval_seconds=_TCP_KEEPALIVE_TEST_INTERVAL,
            probes=_TCP_KEEPALIVE_TEST_PROBES,
        )
        factory(_TEST_ADDR_INFO)

        mock_sock.setsockopt.assert_called_once_with(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)

    def test_keepalive_idle_falls_back_to_macos_tcp_keepalive(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.delattr(socket, "TCP_KEEPIDLE", raising=False)
        monkeypatch.setattr(socket, "TCP_KEEPALIVE", 0x10, raising=False)
        mock_sock = MagicMock()

        _set_tcp_keepalive(
            mock_sock, _TCP_KEEPALIVE_TEST_IDLE, _TCP_KEEPALIVE_TEST_INTERVAL, _TCP_KEEPALIVE_TEST_PROBES
        )

        mock_sock.setsockopt.assert_any_call(socket.IPPROTO_TCP, 0x10, _TCP_KEEPALIVE_TEST_IDLE)

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

    def test_GlobalAIOHTTPAsyncClientConfig_keepalive_defaults(self) -> None:
        cfg = GlobalAIOHTTPAsyncClientConfig()
        assert cfg.global_aiohttp_tcp_keepalive_idle_seconds == 60
        assert cfg.global_aiohttp_tcp_keepalive_interval_seconds == 10
        assert cfg.global_aiohttp_tcp_keepalive_probes == 3

    @mark.parametrize(
        "field, value",
        [
            ("global_aiohttp_tcp_keepalive_idle_seconds", 0),
            ("global_aiohttp_tcp_keepalive_idle_seconds", 32768),
            ("global_aiohttp_tcp_keepalive_interval_seconds", 0),
            ("global_aiohttp_tcp_keepalive_interval_seconds", 32768),
            ("global_aiohttp_tcp_keepalive_probes", 0),
            ("global_aiohttp_tcp_keepalive_probes", 128),
        ],
    )
    def test_GlobalAIOHTTPAsyncClientConfig_rejects_out_of_range_keepalive(self, field: str, value: int) -> None:
        # Linux setsockopt returns EINVAL for these, so they must fail at config load instead.
        with raises(ValidationError):
            GlobalAIOHTTPAsyncClientConfig.model_validate({field: value})

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

    def _mock_global_client(self, monkeypatch: MonkeyPatch, connection_errors: int) -> MagicMock:
        """Global-client stand-in whose request() raises ClientOSError `connection_errors` times, then succeeds."""
        client = MagicMock()
        client.request = AsyncMock(side_effect=[ClientOSError()] * connection_errors + [client.success_response])
        monkeypatch.setattr(nemo_gym.server_utils, "get_global_aiohttp_client", lambda: client)
        monkeypatch.setattr(nemo_gym.server_utils.asyncio, "sleep", AsyncMock())
        return client

    async def test_request_bounded_connection_retries_surface_dead_endpoint(self, monkeypatch: MonkeyPatch) -> None:
        client = self._mock_global_client(monkeypatch, connection_errors=10)
        with raises(ClientOSError):
            await nemo_gym.server_utils.request("POST", "http://dead-host:1/v1", _max_connection_retries=3)
        assert client.request.await_count == 3

    async def test_request_connection_retries_unbounded_by_default(self, monkeypatch: MonkeyPatch) -> None:
        client = self._mock_global_client(monkeypatch, connection_errors=4)
        response = await nemo_gym.server_utils.request("POST", "http://flaky-host:1/v1")
        assert response is client.success_response
        assert client.request.await_count == 5


class TestKeepaliveHttpToolsProtocolServer:
    @mark.skipif(_TCP_KEEPIDLE_OPT is None, reason="platform has no TCP keepalive idle option")
    async def test_uvicorn_applies_tcp_keepalive_to_accepted_connections(self, monkeypatch: MonkeyPatch) -> None:
        """Start a real uvicorn server with the keepalive `http` protocol and serve one request."""
        http_protocol = partial(KeepaliveHttpToolsProtocol, keepalive=(90, 10, 3))

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
