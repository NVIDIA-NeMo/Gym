# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from app import (
    VerifiersAgent,
    VerifiersAgentConfig,
    VerifiersAgentRunRequest,
    VLLMOpenAIClient,
)

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient


class TestApp:
    def test_sanity(self) -> None:
        config = VerifiersAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            resources_server=ResourcesServerRef(type="resources_servers", name=""),
            model_server=ModelServerRef(type="responses_api_models", name=""),
        )
        VerifiersAgent(config=config, server_client=MagicMock(spec=ServerClient))

    def test_vllm_openai_client_init(self) -> None:
        client = VLLMOpenAIClient(base_url="http://localhost:8000/v1")
        assert client.chat is not None
        assert client.chat.completions is not None

    def test_verifiers_agent_run_request(self) -> None:
        req = VerifiersAgentRunRequest(
            task_idx=0,
            vf_env_id="test-env",
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[{"role": "user", "content": "Sort: c, b, a"}]
            ),
            answer="a, b, c",
            task="sort",
            example_id=42,
        )
        assert req.task_idx == 0
        assert req.vf_env_id == "test-env"
        assert req.answer == "a, b, c"
        assert req.task == "sort"
        assert req.example_id == 42

    def test_agent_caches_are_instance_level(self) -> None:
        config = VerifiersAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            resources_server=ResourcesServerRef(type="resources_servers", name=""),
            model_server=ModelServerRef(type="responses_api_models", name=""),
        )
        agent1 = VerifiersAgent(config=config, server_client=MagicMock(spec=ServerClient))
        agent2 = VerifiersAgent(config=config, server_client=MagicMock(spec=ServerClient))

        agent1.envs_cache["test"] = MagicMock()
        assert "test" not in agent2.envs_cache
