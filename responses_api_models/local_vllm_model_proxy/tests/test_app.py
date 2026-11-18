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
from unittest.mock import MagicMock

import pytest

import responses_api_models.local_vllm_model_proxy.app
from nemo_gym.client import ServerClient
from responses_api_models.local_vllm_model_proxy.app import (
    LocalVLLMModelProxyServer,
    LocalVLLMModelProxyServerConfig,
)


class TestApp:
    def _setup_server(self, **overrides):
        config = LocalVLLMModelProxyServerConfig(
            host="0.0.0.0",
            port=8081,
            entrypoint="",
            name="",
            return_token_id_information=False,
            uses_reasoning_parser=True,
            model_server={"type": "responses_api_models", "name": "dummy ref"},
            **overrides,
        )
        return LocalVLLMModelProxyServer(
            config=config, server_client=MagicMock(spec=ServerClient, global_config_dict={})
        )

    def test_sanity(self) -> None:
        self._setup_server()

    def test_setup_webserver_sanity(self, monkeypatch) -> None:
        server = self._setup_server()

        sleep_mock = MagicMock()
        monkeypatch.setattr(responses_api_models.local_vllm_model_proxy.app, "sleep", sleep_mock)

        server.server_client.poll_for_status.side_effect = ["error", "error", "success"]
        server.server_client.global_config_dict = {}

        get_first_server_config_dict_mock = MagicMock()
        monkeypatch.setattr(
            responses_api_models.local_vllm_model_proxy.app,
            "get_first_server_config_dict",
            get_first_server_config_dict_mock,
        )

        json_mock = MagicMock()
        json_mock.json.return_value = {
            "base_url": ["abcd", "defg"],
            "api_key": "my api key",  # pragma: allowlist secret
            "model": "my model",
        }
        requests_mock = MagicMock()
        requests_mock.get.return_value = json_mock
        monkeypatch.setattr(responses_api_models.local_vllm_model_proxy.app, "requests", requests_mock)

        server.setup_webserver()

        assert sleep_mock.call_count == 2

        expected_model = "my model"
        actual_model = server.config.model
        assert expected_model == actual_model

        assert len(server._clients) == 2

    def test_a_model_server_that_never_comes_up_fails_startup_within_the_limit(self, monkeypatch) -> None:
        server = self._setup_server(model_server_wait_timeout_seconds=25)
        # Each poll "sleeps" 10 s on a fake clock, so the 25 s limit ends the wait after three polls. If the limit were
        # ignored the polls would end on their own, with a success, and the raises below would fail.
        now = [0.0]
        monkeypatch.setattr(responses_api_models.local_vllm_model_proxy.app, "monotonic", lambda: now[0])
        monkeypatch.setattr(
            responses_api_models.local_vllm_model_proxy.app,
            "sleep",
            lambda seconds: now.__setitem__(0, now[0] + seconds),
        )
        server.server_client.poll_for_status.side_effect = ["error"] * 5 + ["success"]

        with pytest.raises(TimeoutError, match="was not up within 25"):
            server.setup_webserver()

        assert server.server_client.poll_for_status.call_count == 3  # at 0 s, 10 s and 20 s

    def test_a_model_server_that_comes_up_only_after_the_deadline_is_not_accepted(self, monkeypatch) -> None:
        server = self._setup_server(model_server_wait_timeout_seconds=5)
        now = [0.0]
        monkeypatch.setattr(responses_api_models.local_vllm_model_proxy.app, "monotonic", lambda: now[0])
        # The sleep is capped at the 5 s that remain, not the usual 10 s, so the next poll never happens after the limit.
        monkeypatch.setattr(
            responses_api_models.local_vllm_model_proxy.app,
            "sleep",
            lambda seconds: now.__setitem__(0, now[0] + seconds),
        )
        server.server_client.poll_for_status.side_effect = ["error", "success"]

        with pytest.raises(TimeoutError, match="was not up within 5"):
            server.setup_webserver()

        assert server.server_client.poll_for_status.call_count == 1

    def test_the_config_request_carries_the_limit_and_no_limit_means_none(self, monkeypatch) -> None:
        for limit in (None, 30):
            server = self._setup_server(**({} if limit is None else {"model_server_wait_timeout_seconds": limit}))
            server.server_client.poll_for_status.side_effect = ["success"]
            server.server_client.global_config_dict = {}
            monkeypatch.setattr(
                responses_api_models.local_vllm_model_proxy.app, "get_first_server_config_dict", MagicMock()
            )
            reply = MagicMock()
            reply.json.return_value = {
                "base_url": ["abcd"],
                "api_key": "key",
                "model": "m",
            }  # pragma: allowlist secret
            requests_mock = MagicMock()
            requests_mock.get.return_value = reply
            monkeypatch.setattr(responses_api_models.local_vllm_model_proxy.app, "requests", requests_mock)

            server.setup_webserver()

            assert requests_mock.get.call_args.kwargs["timeout"] == limit

    def test_a_wait_limit_of_zero_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            self._setup_server(model_server_wait_timeout_seconds=0)
