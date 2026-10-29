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
"""Tests for `nemo_gym.client.server_client`."""

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
import logging
from unittest.mock import AsyncMock, MagicMock

import orjson
from omegaconf import DictConfig, OmegaConf
from pytest import LogCaptureFixture, MonkeyPatch, mark, raises
from requests.exceptions import ProxyError

import nemo_gym.client.http_client
import nemo_gym.client.server_client
import nemo_gym.config.global_config
import nemo_gym.server_runtime.server_utils
from nemo_gym.client import ServerClient
from nemo_gym.client.server_client import (
    NEMO_GYM_MODEL_SERVER_BASE_URL_ENV_VAR_NAME,
    NEMO_GYM_MODEL_SERVER_NAME_ENV_VAR_NAME,
    ConnectionError,
)
from nemo_gym.config import BaseServerConfig
from nemo_gym.config.config_types import ConfigError, HeadServerUnreachableError
from nemo_gym.config.global_config import (
    NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME,
)
from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming


@mark.parametrize("header", [None, b"X-Other-Harness-Reply"])
def test_model_header_comes_from_its_harness_property(monkeypatch, header):
    import sys
    from types import ModuleType

    harness = ModuleType("responses_api_agents.test_header_harness")
    harness._assistant_message_header = header
    plain = ModuleType("responses_api_agents.test_plain_harness")
    monkeypatch.setitem(sys.modules, harness.__name__, harness)
    monkeypatch.setitem(sys.modules, plain.__name__, plain)
    client = ServerClient(
        head_server_config={"host": "localhost", "port": 0},
        global_config_dict=OmegaConf.create(
            {
                "first": {
                    "responses_api_agents": {
                        "test_header_harness": {
                            "model_server": {"type": "responses_api_models", "name": "first_model"},
                        }
                    }
                },
                "second": {
                    "responses_api_agents": {
                        "test_plain_harness": {
                            "model_server": {"type": "responses_api_models", "name": "second_model"},
                        }
                    }
                },
                "observability_enabled": True,
            }
        ),
    )
    assert client.assistant_message_header("first_model") == (header.lower() if header else None)
    assert client.assistant_message_header("second_model") is None
    assert client.assistant_message_header("unused_model") is None


def test_shared_model_rejects_conflicting_harness_headers(monkeypatch):
    import sys
    from types import ModuleType

    config = {}
    for index, header in enumerate((b"x-one-reply", b"x-two-reply")):
        name = f"test_header_{index}"
        harness = ModuleType(f"responses_api_agents.{name}")
        harness._assistant_message_header = header
        monkeypatch.setitem(sys.modules, harness.__name__, harness)
        config[name] = {
            "responses_api_agents": {
                name: {
                    "model_server": {"type": "responses_api_models", "name": "policy"},
                }
            }
        }
    client = ServerClient(
        head_server_config={"host": "localhost", "port": 0}, global_config_dict=OmegaConf.create(config)
    )
    with raises(ValueError, match="different assistant headers"):
        client.assistant_message_header("policy")


class TestServerClient:
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
        monkeypatch.setattr(nemo_gym.client.server_client, "get_global_config_dict", get_global_config_dict_mock)
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
        monkeypatch.setattr(nemo_gym.client.server_client, "get_global_config_dict", get_global_config_dict_mock)

        monkeypatch.setattr(nemo_gym.config.global_config, "_GLOBAL_CONFIG_DICT", None)
        monkeypatch.delenv(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, raising=False)

        httpx_client_mock = MagicMock()
        httpx_response_mock = MagicMock()
        httpx_client_mock.return_value = httpx_response_mock
        httpx_response_mock.content = b'"a: 2"'
        monkeypatch.setattr(nemo_gym.client.server_client.requests, "get", httpx_client_mock)

        actual_client = ServerClient.load_from_global_config()
        assert {"a": 2} == actual_client.global_config_dict

    def test_ServerClient_uses_loopback_for_wildcard_bind_hosts(self, monkeypatch: MonkeyPatch) -> None:
        global_config_dict = DictConfig(
            {
                "head_server": {
                    "host": "0.0.0.0",
                    "port": 11000,
                }
            }
        )
        monkeypatch.setattr(
            nemo_gym.client.server_client,
            "get_global_config_dict",
            MagicMock(return_value=global_config_dict),
        )
        response = MagicMock(content=b'"head_server: {}"')
        requests_get = MagicMock(return_value=response)
        monkeypatch.setattr(nemo_gym.client.server_client.requests, "get", requests_get)

        client = ServerClient.load_from_global_config()

        requests_get.assert_called_once_with("http://127.0.0.1:11000/global_config_dict_yaml")
        assert client._build_server_base_url(DictConfig({"host": "0.0.0.0", "port": 11000})) == (
            "http://127.0.0.1:11000"
        )
        assert client._build_server_base_url(DictConfig({"host": "::", "port": 11000})) == ("http://[::1]:11000")

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
        monkeypatch.setattr(nemo_gym.client.server_client, "get_global_config_dict", get_global_config_dict_mock)

        # `gym eval run --no-serve` initializes a partial local config.
        # It must still fetch the full config from the running head server.
        monkeypatch.setattr(nemo_gym.config.global_config, "_GLOBAL_CONFIG_DICT", global_config_dict)
        monkeypatch.delenv(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, raising=False)

        response = MagicMock(content=b'"remote_server: {host: remote, port: 1234}"')
        get_mock = MagicMock(return_value=response)
        monkeypatch.setattr(nemo_gym.client.server_client.requests, "get", get_mock)

        client = ServerClient.load_from_global_config()
        assert client.global_config_dict == {"remote_server": {"host": "remote", "port": 1234}}
        get_mock.assert_called_once()

    def test_ServerClient_load_from_global_config_fast_path_via_env(self, monkeypatch: MonkeyPatch) -> None:
        """Use the config injected into a Gym-launched server process."""
        global_config_dict = DictConfig({"head_server": {"host": "", "port": 0}})
        get_global_config_dict_mock = MagicMock(return_value=global_config_dict)
        monkeypatch.setattr(nemo_gym.client.server_client, "get_global_config_dict", get_global_config_dict_mock)

        monkeypatch.setattr(nemo_gym.config.global_config, "_GLOBAL_CONFIG_DICT", None)
        monkeypatch.setenv(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, "head_server: {host: '', port: 0}")

        def boom(*args, **kwargs):
            raise AssertionError("requests.get should not be called on the fast path")

        monkeypatch.setattr(nemo_gym.client.server_client.requests, "get", boom)

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
        monkeypatch.setattr(nemo_gym.client.server_client, "get_global_config_dict", get_global_config_dict_mock)

        monkeypatch.setattr(nemo_gym.config.global_config, "_GLOBAL_CONFIG_DICT", None)
        monkeypatch.delenv(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, raising=False)

        httpx_client_mock = MagicMock()
        httpx_client_mock.side_effect = ConnectionError
        monkeypatch.setattr(nemo_gym.client.server_client.requests, "get", httpx_client_mock)

        with raises(ValueError):
            ServerClient.load_from_global_config()

    def _nothing_listening_on_the_head_server(self, monkeypatch: MonkeyPatch, host: str, port: int) -> ConnectionError:
        """Make `load_from_global_config` take the fetch path and have the head server refuse the connection."""
        global_config_dict = DictConfig({"head_server": {"host": host, "port": port}})
        monkeypatch.setattr(
            nemo_gym.client.server_client, "get_global_config_dict", MagicMock(return_value=global_config_dict)
        )
        monkeypatch.delenv(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, raising=False)

        refused = ConnectionError("[Errno 61] Connection refused")
        monkeypatch.setattr(nemo_gym.client.server_client.requests, "get", MagicMock(side_effect=refused))
        return refused

    def test_ServerClient_load_from_global_config_unreachable_head_server_is_a_ConfigError(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        # An unreachable head server is a user mistake (nothing started, or the wrong port), not a bug:
        # it must be a ConfigError so `exit_cleanly_on_config_error` prints the message instead of a
        # traceback (#2687), and still a ValueError for callers that already catch that.
        refused = self._nothing_listening_on_the_head_server(monkeypatch, host="127.0.0.1", port=11000)

        with raises(HeadServerUnreachableError) as exc_info:
            ServerClient.load_from_global_config()

        assert isinstance(exc_info.value, ConfigError)
        assert isinstance(exc_info.value, ValueError)
        # The low-level cause is not lost, just kept off the user-facing message.
        assert exc_info.value.__cause__ is refused

    def test_ServerClient_load_from_global_config_unreachable_head_server_message_is_actionable(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        # Same wording as `gym env status`: name the address that was tried, then the fix for each of
        # the two likely causes (nothing running -> `gym env start`; running elsewhere -> override).
        self._nothing_listening_on_the_head_server(monkeypatch, host="10.0.0.5", port=9500)

        with raises(HeadServerUnreachableError) as exc_info:
            ServerClient.load_from_global_config()

        message = str(exc_info.value)
        assert message.startswith("Could not connect to the head server at http://10.0.0.5:9500.")
        assert "Is the head server running? Start it with: `gym env start`." in message
        assert "`++head_server.host=<host>` / `++head_server.port=<port>`" in message

    def test_ServerClient_load_from_global_config_unreachable_head_server_logs_the_cause_at_debug(
        self, monkeypatch: MonkeyPatch, caplog: LogCaptureFixture
    ) -> None:
        # A proxy or name-resolution failure is also a requests ConnectionError, and the CLI prints the
        # ConfigError without its cause, so the real reason must still reach `--verbose` (DEBUG) output.
        self._nothing_listening_on_the_head_server(monkeypatch, host="127.0.0.1", port=11000)
        proxy_error = ProxyError("Tunnel connection failed: 407 Proxy Authentication Required")
        monkeypatch.setattr(nemo_gym.client.server_client.requests, "get", MagicMock(side_effect=proxy_error))

        with (
            caplog.at_level(logging.DEBUG, logger="nemo_gym.client.server_client"),
            raises(HeadServerUnreachableError),
        ):
            ServerClient.load_from_global_config()

        (record,) = [record for record in caplog.records if record.name == "nemo_gym.client.server_client"]
        assert record.levelno == logging.DEBUG
        assert "http://127.0.0.1:11000" in record.getMessage()
        assert record.exc_info[1] is proxy_error

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
        monkeypatch.setattr("nemo_gym.client.http_client.get_global_aiohttp_client", httpx_client_mock)

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
        monkeypatch.setattr("nemo_gym.client.http_client.get_global_aiohttp_client", client_mock)

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
        monkeypatch.setattr("nemo_gym.client.http_client.get_global_aiohttp_client", httpx_client_mock)

        await server_client.post(server_name="my_server", url_path="/x")
        assert server_client._server_base_urls == {"my_server": "http://xyz:54321"}

        def boom(*_args, **_kwargs):
            raise AssertionError("get_first_server_config_dict should not be called once the URL is cached")

        monkeypatch.setattr(nemo_gym.client.server_client, "get_first_server_config_dict", boom)

        await server_client.post(server_name="my_server", url_path="/y")
        await server_client.get(server_name="my_server", url_path="/z")

        assert httpx_client_request_mock.call_count == 3
        for call in httpx_client_request_mock.call_args_list:
            assert call.kwargs["url"].startswith("http://xyz:54321")

    async def test_ServerClient_post_encodes_replayed_output_logprobs(self, monkeypatch: MonkeyPatch) -> None:
        """A request body that replays assistant output with logprobs is sent as JSON with the logprobs intact."""
        server_client = ServerClient(
            head_server_config=BaseServerConfig(host="head", port=11000),
            global_config_dict=DictConfig({"my_server": {"a": {"b": {"host": "xyz", "port": 54321}}}}),
        )
        aiohttp_client_mock = MagicMock()
        aiohttp_client_request_mock = AsyncMock(return_value="ok")
        aiohttp_client_mock.return_value.request = aiohttp_client_request_mock
        monkeypatch.setattr("nemo_gym.client.http_client.get_global_aiohttp_client", aiohttp_client_mock)

        logprobs = [{"token": "a", "bytes": [97], "logprob": -0.5, "top_logprobs": []}]
        assistant_message = {
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "a", "annotations": [], "logprobs": logprobs}],
        }
        body = NeMoGymResponseCreateParamsNonStreaming.model_validate({"input": [assistant_message]})

        await server_client.post(server_name="my_server", url_path="/v1/responses", json=body)

        sent = orjson.loads(aiohttp_client_request_mock.call_args.kwargs["data"])
        assert sent["input"][0]["content"][0]["logprobs"] == logprobs
