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
"""
This responses_api_models server is only used to proxy to an existing LocalVLLMModel server so we don't need to duplicate GPU resources.
"""

from time import sleep
from typing import List, Union

import requests
from pydantic import Field

from nemo_gym.config_types import ModelServerRef
from nemo_gym.global_config import get_first_server_config_dict
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig


class LocalVLLMModelProxyServerConfig(VLLMModelConfig):
    # We inherit these configs from VLLMModelConfig, but they are set to optional since we will get this information after the referenced LocalVLLMModel spinup
    base_url: Union[str, List[str]] = Field(default_factory=list)
    # Not used on local deployments
    api_key: str = "dummy"  # pragma: allowlist secret
    model: str = "dummy"

    model_server: ModelServerRef


class LocalVLLMModelProxyServer(VLLMModel):
    config: LocalVLLMModelProxyServerConfig

    def setup_webserver(self):
        model_server_name = self.config.model_server.name

        print(f"Waiting for LocalVLLMModelServer `{model_server_name}` spinup")

        while self.server_client.poll_for_status(model_server_name) != "success":
            # Sleep for 10s by default
            sleep(10)

        model_server_config_dict = get_first_server_config_dict(
            self.server_client.global_config_dict, model_server_name
        )
        model_server_base_url = self.server_client._build_server_base_url(model_server_config_dict)
        response = requests.get(
            f"{model_server_base_url}/get_inner_vllm_config",
        )
        assert response.ok

        response_dict = response.json()

        self.config.base_url = response_dict["base_url"]
        self.config.api_key = response_dict["api_key"]
        self.config.model = response_dict["model"]

        return super().setup_webserver()


if __name__ == "__main__":
    LocalVLLMModelProxyServer.run_webserver()
