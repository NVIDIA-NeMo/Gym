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

from responses_api_agents.gdpval_agent.app import GDPValAgent, GDPValAgentConfig
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.server_utils import ServerClient


class TestGDPValAgent:
    def test_sanity(self) -> None:
        config = GDPValAgentConfig(
            host="127.0.0.1",
            port=8000,
            entrypoint="app.py",
            resources_server=ResourcesServerRef(
                type="resources_servers",
                name="my_resources_server",
            ),
            model_server=ModelServerRef(
                type="responses_api_models",
                name="my_model",
            ),
        )
        GDPValAgent(config=config, server_client=MagicMock(spec=ServerClient))

    def test_config_defaults(self) -> None:
        config = GDPValAgentConfig(
            host="127.0.0.1",
            port=8000,
            entrypoint="app.py",
            resources_server=ResourcesServerRef(
                type="resources_servers",
                name="my_resources_server",
            ),
            model_server=ModelServerRef(
                type="responses_api_models",
                name="my_model",
            ),
        )

        assert config.max_steps == 100
        assert config.max_tokens == 10000
        assert config.context_summarization_cutoff == 0.7
        assert config.step_warning_threshold == 80
