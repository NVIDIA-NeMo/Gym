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

from time import monotonic, sleep
from typing import List, Optional, Union

import requests
from pydantic import Field, PositiveFloat

from nemo_gym.config import ModelServerRef, get_first_server_config_dict
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig


class LocalVLLMModelProxyServerConfig(VLLMModelConfig):
    # We inherit these configs from VLLMModelConfig, but they are set to optional since we will get this information after the referenced LocalVLLMModel spinup
    base_url: Union[str, List[str]] = Field(default_factory=list)
    # Not used on local deployments
    api_key: str = "dummy"  # pragma: allowlist secret
    model: str = "dummy"

    model_server: ModelServerRef

    # Wait for the referenced LocalVLLMModel server at startup, and for its answer to the config request. None (default)
    # waits without limit. A number of seconds makes startup fail with a TimeoutError if the server is not up by then.
    model_server_wait_timeout_seconds: Optional[PositiveFloat] = None


class LocalVLLMModelProxyServer(VLLMModel):
    config: LocalVLLMModelProxyServerConfig

    def setup_webserver(self):
        model_server_name = self.config.model_server.name

        print(f"Waiting for LocalVLLMModelServer `{model_server_name}` spinup")

        timeout_s = self.config.model_server_wait_timeout_seconds
        deadline = None if timeout_s is None else monotonic() + timeout_s
        while True:
            # Expiry is checked before every poll, so a server that comes up only after the deadline is not accepted.
            remaining = None if deadline is None else deadline - monotonic()
            if remaining is not None and remaining <= 0:
                raise TimeoutError(f"LocalVLLMModelServer `{model_server_name}` was not up within {timeout_s} s")
            if self.server_client.poll_for_status(model_server_name) == "success":
                break
            # Sleep for 10s by default, or less when that would pass the deadline
            sleep(10 if remaining is None else min(10, remaining))

        model_server_config_dict = get_first_server_config_dict(
            self.server_client.global_config_dict, model_server_name
        )
        model_server_base_url = self.server_client._build_server_base_url(model_server_config_dict)
        response = requests.get(
            f"{model_server_base_url}/get_inner_vllm_config",
            timeout=timeout_s,
        )
        assert response.ok

        response_dict = response.json()

        self.config.base_url = response_dict["base_url"]
        self.config.api_key = response_dict["api_key"]
        self.config.model = response_dict["model"]

        # Reset clients after base_url config
        self._post_init()

        return super().setup_webserver()


if __name__ == "__main__":
    LocalVLLMModelProxyServer.run_webserver()
