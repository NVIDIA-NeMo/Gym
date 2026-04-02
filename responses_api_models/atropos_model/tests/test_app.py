# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from unittest.mock import MagicMock

from nemo_gym.server_utils import ServerClient
from responses_api_models.atropos_model.app import AtroposModel
from responses_api_models.vllm_model.app import VLLMModelConfig


class TestApp:
    def test_sanity(self) -> None:
        config = VLLMModelConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            base_url="http://localhost:8000/v1",
            api_key="test",
            model="test-model",
            return_token_id_information=True,
            uses_reasoning_parser=False,
        )
        AtroposModel(config=config, server_client=MagicMock(spec=ServerClient))
