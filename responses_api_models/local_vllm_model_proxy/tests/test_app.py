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
            model_server={"type": "responses_api_models", "name": "dummy ref"},
        )
        return LocalVLLMModelProxyServer(config=config, server_client=MagicMock(spec=ServerClient))

    async def test_sanity(self) -> None:
        self._setup_server()
