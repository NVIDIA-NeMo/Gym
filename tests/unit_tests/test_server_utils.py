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
from unittest.mock import AsyncMock, MagicMock

from omegaconf import OmegaConf
from pytest import MonkeyPatch, raises

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
    HeadServer,
    ServerClient,
    SimpleServer,
    initialize_ray,
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
        """HTTP fallback path: simulate an ad-hoc client.py with no local cache."""
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

        # Force the slow path: no local cache and no env var.
        monkeypatch.setattr(nemo_gym.global_config, "_GLOBAL_CONFIG_DICT", None)
        monkeypatch.delenv(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, raising=False)

        httpx_client_mock = MagicMock()
        httpx_response_mock = MagicMock()
        httpx_client_mock.return_value = httpx_response_mock
        httpx_response_mock.content = b'"a: 2"'
        monkeypatch.setattr(nemo_gym.server_utils.requests, "get", httpx_client_mock)

        actual_client = ServerClient.load_from_global_config()
        assert {"a": 2} == actual_client.global_config_dict

    def test_ServerClient_load_from_global_config_prefers_local_cache(self, monkeypatch: MonkeyPatch) -> None:
        """When `_GLOBAL_CONFIG_DICT` is populated locally, no HTTP fetch happens."""
        global_config_dict = DictConfig(
            {
                "head_server": {"host": "", "port": 0},
                "my_server": {"a": {"b": {"host": "x", "port": 1}}},
            }
        )
        get_global_config_dict_mock = MagicMock(return_value=global_config_dict)
        monkeypatch.setattr(nemo_gym.server_utils, "get_global_config_dict", get_global_config_dict_mock)

        # Simulate a worker process: the global config dict is cached locally.
        monkeypatch.setattr(nemo_gym.global_config, "_GLOBAL_CONFIG_DICT", global_config_dict)

        # If the fast path is taken, requests.get is never called. Make it loud
        # if it does get called so the test fails clearly.
        def boom(*args, **kwargs):
            raise AssertionError("requests.get should not be called on the fast path")

        monkeypatch.setattr(nemo_gym.server_utils.requests, "get", boom)

        client = ServerClient.load_from_global_config()
        assert client.global_config_dict is global_config_dict

    def test_ServerClient_load_from_global_config_fast_path_via_env(self, monkeypatch: MonkeyPatch) -> None:
        """`NEMO_GYM_CONFIG_DICT` env var alone is enough to trigger the fast path
        (covers the first call inside a fresh worker before the singleton is hit)."""
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

        # Force the slow path so the ConnectionError can fire.
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
        """Step 2: the immutable config dict is serialized to YAML at most once."""
        global_config_dict = DictConfig({"a": 2})
        get_global_config_dict_mock = MagicMock(return_value=global_config_dict)
        monkeypatch.setattr(nemo_gym.server_utils, "get_global_config_dict", get_global_config_dict_mock)

        to_yaml_mock = MagicMock(wraps=OmegaConf.to_yaml)
        monkeypatch.setattr(nemo_gym.server_utils.OmegaConf, "to_yaml", to_yaml_mock)

        head_server = HeadServer(config=BaseServerConfig(host="", port=0))
        first = await head_server.global_config_dict_yaml()
        second = await head_server.global_config_dict_yaml()

        # Same string object — proves the cache, not just value-equal.
        assert first is second
        assert to_yaml_mock.call_count == 1

        # Explicit invalidation re-serializes.
        head_server.invalidate_global_config_dict_yaml_cache()
        third = await head_server.global_config_dict_yaml()
        assert third == first  # value
        assert to_yaml_mock.call_count == 2

    async def test_ServerClient_request_uses_base_url_table(self, monkeypatch: MonkeyPatch) -> None:
        """Step 3: after the first request to a server_name, `request()` no longer
        walks the OmegaConf DictConfig — the base URL comes from the cached
        `_server_base_urls` table."""
        server_client = ServerClient(
            head_server_config=BaseServerConfig(host="head", port=11000),
            global_config_dict=DictConfig({"my_server": {"a": {"b": {"host": "xyz", "port": 54321}}}}),
        )

        httpx_client_mock = MagicMock()
        httpx_client_request_mock = AsyncMock()
        httpx_client_request_mock.return_value = "ok"
        httpx_client_mock.return_value.request = httpx_client_request_mock
        monkeypatch.setattr(nemo_gym.server_utils, "get_global_aiohttp_client", httpx_client_mock)

        # First call: populates the table.
        await server_client.post(server_name="my_server", url_path="/x")
        assert server_client._server_base_urls == {"my_server": "http://xyz:54321"}

        # Second call: must not touch the OmegaConf walker. Make
        # `get_first_server_config_dict` loud to prove the fast path.
        def boom(*_args, **_kwargs):
            raise AssertionError("get_first_server_config_dict should not be called once the URL is cached")

        monkeypatch.setattr(nemo_gym.server_utils, "get_first_server_config_dict", boom)

        await server_client.post(server_name="my_server", url_path="/y")
        await server_client.get(server_name="my_server", url_path="/z")

        # All three calls dispatched to the same precomputed URL.
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
