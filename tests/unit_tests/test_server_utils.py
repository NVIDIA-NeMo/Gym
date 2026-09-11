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
from unittest.mock import AsyncMock, MagicMock

import uvicorn
from aiohttp import ClientOSError, ClientResponseError, RequestInfo
from multidict import CIMultiDict, CIMultiDictProxy
from omegaconf import OmegaConf
from pydantic import ValidationError
from pytest import CaptureFixture, MonkeyPatch, raises
from yarl import URL

import nemo_gym.global_config
import nemo_gym.server_utils
from nemo_gym.config_types import BaseRunServerInstanceConfig
from nemo_gym.global_config import (
    DRY_RUN_KEY_NAME,
    NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME,
    NEMO_GYM_CONFIG_PATH_ENV_VAR_NAME,
)
from nemo_gym.server_utils import (
    NEMO_GYM_MODEL_SERVER_BASE_URL_ENV_VAR_NAME,
    NEMO_GYM_MODEL_SERVER_NAME_ENV_VAR_NAME,
    BaseServer,
    BaseServerConfig,
    ClientDisconnectCancellationMiddleware,
    ConnectionError,
    DictConfig,
    GlobalAIOHTTPAsyncClientConfig,
    HeadServer,
    ServerClient,
    SimpleServer,
    UvicornProxyHeadersConfig,
    _format_upstream_error_log,
    _make_keepalive_socket_factory,
    initialize_ray,
    raise_for_status,
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

    async def test_ServerClient_preserves_external_capture_url(self, monkeypatch: MonkeyPatch) -> None:
        server_client = ServerClient(
            head_server_config=BaseServerConfig(host="head", port=12345),
            global_config_dict=DictConfig(
                {"policy_model": {"responses_api_models": {"vllm_model": {"host": "plain-host", "port": 54321}}}}
            ),
        )
        monkeypatch.setenv(NEMO_GYM_MODEL_SERVER_NAME_ENV_VAR_NAME, "policy_model")
        monkeypatch.setenv(
            NEMO_GYM_MODEL_SERVER_BASE_URL_ENV_VAR_NAME,
            "http://model/ng-rollout/rollout-1/training-token-capture",
        )

        request_mock = AsyncMock(return_value="response")
        client_mock = MagicMock()
        client_mock.return_value.request = request_mock
        monkeypatch.setattr(nemo_gym.server_utils, "get_global_aiohttp_client", client_mock)

        response = await server_client.post(
            server_name="policy_model",
            url_path="/v1/chat/completions",
            headers={"x-existing": "value"},
        )

        assert response == "response"
        request_mock.assert_awaited_once_with(
            method="POST",
            url="http://model/ng-rollout/rollout-1/training-token-capture/v1/chat/completions",
            headers={"x-existing": "value"},
        )

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

    def test_upstream_error_log_has_bounded_body_and_redacted_url(self) -> None:
        request_info = RequestInfo(
            url=URL("http://policy.test/v1/responses?api_key=secret"),
            method="POST",
            headers=CIMultiDictProxy(CIMultiDict()),
            real_url=URL("http://policy.test/v1/responses?api_key=secret"),
        )
        error = ClientResponseError(
            request_info=request_info,
            history=(),
            status=500,
            message="policy failed",
        )
        error.response_content = (
            b"Traceback (most recent call last):\nValueError: actionable inner failure\n" + b"x" * 3000
        )

        message = _format_upstream_error_log("TestSimpleServer___my_server", error)

        assert "[upstream_request_failed]" in message
        assert "server=TestSimpleServer___my_server" in message
        assert "method=POST url=http://policy.test/v1/responses status=500" in message
        assert "ValueError: actionable inner failure" in message
        assert "api_key=secret" not in message
        assert message.endswith("…")
        assert len(message) < 2200

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

        monkeypatch.setattr(nemo_gym.server_utils, "_GLOBAL_AIOHTTP_CLIENT_REQUEST_DEBUG", False)
        response = await callbacks[0](MagicMock(), fail)

        assert response.status_code == 500
        captured = capsys.readouterr().out
        assert "[upstream_request_failed]" in captured
        assert "ValueError: actionable inner failure" in captured

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


_SPOOFED_HOST = "203.0.113.99"
_LOOPBACK = "127.0.0.1"


async def _scope_seen_by_app(*, proxy_headers: bool, allow_ips: list[str] | None, peer: str, forwarded: bool) -> dict:
    """Drive uvicorn's loaded app with one request and return the scope the inner app observed."""
    seen: dict = {}

    async def recorder(scope, receive, send) -> None:
        seen["client"] = scope.get("client")
        seen["scheme"] = scope["scheme"]

    config = uvicorn.Config(
        app=recorder,
        proxy_headers=proxy_headers,
        forwarded_allow_ips=allow_ips or [],
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

    def test_allowlist_is_normalized(self) -> None:
        config = UvicornProxyHeadersConfig.model_validate(
            {"uvicorn_proxy_headers": True, "uvicorn_forwarded_allow_ips": [" 10.0.0.1 ", "", "10.0.0.2"]}
        )

        assert ["10.0.0.1", "10.0.0.2"] == config.uvicorn_forwarded_allow_ips


class TestUvicornProxyHeadersBehavior:
    async def test_forwarded_headers_ignored_when_disabled(self) -> None:
        """The documented internal-only default: the real peer and scheme survive a forged header."""
        seen = await _scope_seen_by_app(proxy_headers=False, allow_ips=None, peer=_LOOPBACK, forwarded=True)

        assert (_LOOPBACK, 54321) == seen["client"]
        assert "http" == seen["scheme"]

    async def test_real_peer_reported_when_disabled_without_forwarded_headers(self) -> None:
        seen = await _scope_seen_by_app(proxy_headers=False, allow_ips=None, peer=_LOOPBACK, forwarded=False)

        assert (_LOOPBACK, 54321) == seen["client"]
        assert "http" == seen["scheme"]

    async def test_forwarded_headers_honored_for_trusted_proxy(self) -> None:
        seen = await _scope_seen_by_app(proxy_headers=True, allow_ips=[_LOOPBACK], peer=_LOOPBACK, forwarded=True)

        assert (_SPOOFED_HOST, 0) == seen["client"]
        assert "https" == seen["scheme"]

    async def test_forwarded_headers_ignored_from_untrusted_peer(self) -> None:
        """Enabled, but the caller is not on the allowlist, so its forwarded claims are discarded."""
        seen = await _scope_seen_by_app(proxy_headers=True, allow_ips=["10.0.0.1"], peer=_LOOPBACK, forwarded=True)

        assert (_LOOPBACK, 54321) == seen["client"]
        assert "http" == seen["scheme"]

    def test_proxy_middleware_absent_for_single_and_multi_worker(self) -> None:
        """run_webserver builds one kwargs dict for both launch paths, so the setting must hold for each."""
        from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

        for workers in (1, 4):
            config = uvicorn.Config(
                app=lambda scope, receive, send: None,
                workers=workers,
                proxy_headers=False,
                forwarded_allow_ips=[],
            )
            config.load()

            assert not isinstance(config.loaded_app, ProxyHeadersMiddleware), workers


class TestRunWebserverProxyKwargs:
    """run_webserver must forward the proxy config into uvicorn on both launch paths."""

    def _capture_uvicorn_kwargs(self, monkeypatch: MonkeyPatch, config_dict: dict, num_workers: int) -> dict:
        from fastapi import FastAPI

        global_config = DictConfig({DRY_RUN_KEY_NAME: False, "my_server": {"a": {"b": {}}}, **config_dict})
        monkeypatch.setattr(nemo_gym.server_utils.ray, "is_initialized", MagicMock(return_value=True))
        monkeypatch.setattr(nemo_gym.server_utils, "get_global_config_dict", MagicMock(return_value=global_config))
        server_client = ServerClient(
            head_server_config=BaseServerConfig(host="", port=0), global_config_dict=DictConfig({})
        )
        server_client_mock = MagicMock(return_value=server_client)
        server_client_mock.load_head_server_config = MagicMock(return_value=BaseServerConfig(host="", port=0))
        monkeypatch.setattr(nemo_gym.server_utils, "ServerClient", server_client_mock)
        monkeypatch.setattr(nemo_gym.server_utils, "is_nemo_gym_fastapi_worker", MagicMock(return_value=False))

        captured: dict = {}
        monkeypatch.setattr(nemo_gym.server_utils.uvicorn, "run", lambda **kwargs: captured.update(kwargs))

        server_config = BaseRunServerInstanceConfig(
            name="my_server", host="127.0.0.1", port=8000, entrypoint="app.py", num_workers=num_workers
        )

        class TestSimpleServer(SimpleServer):
            @classmethod
            def load_config_from_global_config(cls):
                return server_config

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

        assert kwargs["proxy_headers"] is False
        assert [] == kwargs["forwarded_allow_ips"]
        # A single worker passes the app object itself rather than an import string.
        assert not isinstance(kwargs["app"], str)
        assert "workers" not in kwargs

    def test_proxy_headers_disabled_by_default_multi_worker(self, monkeypatch: MonkeyPatch) -> None:
        kwargs = self._capture_uvicorn_kwargs(monkeypatch, {}, num_workers=4)

        # Multi-worker launches re-import the app, so uvicorn receives an import string.
        assert isinstance(kwargs["app"], str)
        assert kwargs["app"].endswith(":app")
        assert 4 == kwargs["workers"]
        assert kwargs["proxy_headers"] is False
        assert [] == kwargs["forwarded_allow_ips"]

    def test_unrelated_uvicorn_settings_are_unchanged(self, monkeypatch: MonkeyPatch) -> None:
        """The issue calls out parser, keepalive, access-log, and graceful-shutdown as must-not-change."""
        kwargs = self._capture_uvicorn_kwargs(monkeypatch, {}, num_workers=1)

        assert "httptools" == kwargs["http"]
        assert 30 == kwargs["timeout_keep_alive"]
        assert kwargs["access_log"] is False
        assert 0.5 == kwargs["timeout_graceful_shutdown"]

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
