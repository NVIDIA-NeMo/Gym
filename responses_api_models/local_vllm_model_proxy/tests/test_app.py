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

import responses_api_models.local_vllm_model_proxy.app
from nemo_gym.server_utils import ServerClient
from responses_api_models.local_vllm_model_proxy.app import (
    LocalVLLMModelProxyServer,
    LocalVLLMModelProxyServerConfig,
)


class TestApp:
    def _setup_server(self):
        config = LocalVLLMModelProxyServerConfig(
            host="0.0.0.0",
            port=8081,
            entrypoint="",
            name="",
            return_token_id_information=False,
            uses_reasoning_parser=True,
            model_server={"type": "responses_api_models", "name": "dummy ref"},
        )
        return LocalVLLMModelProxyServer(config=config, server_client=MagicMock(spec=ServerClient))

    def test_sanity(self) -> None:
        self._setup_server()

    def test_setup_webserver_sanity(self, monkeypatch) -> None:
        server = self._setup_server()

        sleep_mock = MagicMock()
        monkeypatch.setattr(responses_api_models.local_vllm_model_proxy.app, "sleep", sleep_mock)

        server.server_client.poll_for_status.side_effect = ["error", "error", "success"]
        server.server_client.global_config_dict = None

        get_first_server_config_dict_mock = MagicMock()
        monkeypatch.setattr(
            responses_api_models.local_vllm_model_proxy.app,
            "get_first_server_config_dict",
            get_first_server_config_dict_mock,
        )

        json_mock = MagicMock()
        json_mock.json.return_value = {"base_url": ["abcd", "defg"], "api_key": "my api key", "model": "my model"}
        requests_mock = MagicMock()
        requests_mock.get.return_value = json_mock
        monkeypatch.setattr(responses_api_models.local_vllm_model_proxy.app, "requests", requests_mock)

        server.setup_webserver()

        assert sleep_mock.call_count == 2

        expected_model = "my model"
        actual_model = server.config.model
        assert expected_model == actual_model

        assert len(server._clients) == 2
