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
from responses_api_agents.cobol_eval_agent.app import (
    CobolEvalAgent,
    CobolEvalAgentConfig,
    ModelServerRef,
    ResourcesServerRef,
)


class TestApp:
    def test_sanity(self) -> None:
        """Agent can be instantiated with valid config."""
        config = CobolEvalAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            resources_server=ResourcesServerRef(
                type="resources_servers",
                name="",
            ),
            model_server=ModelServerRef(
                type="responses_api_models",
                name="",
            ),
        )
        CobolEvalAgent(config=config, server_client=MagicMock(spec=ServerClient))

    def test_config_defaults(self) -> None:
        """Test that config has correct default values."""
        config = CobolEvalAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            resources_server=ResourcesServerRef(
                type="resources_servers",
                name="cobol_compiler",
            ),
            model_server=ModelServerRef(
                type="responses_api_models",
                name="policy_model",
            ),
        )
        assert config.max_correction_turns == 3
        assert config.include_all_attempts is True

    def test_config_custom_values(self) -> None:
        """Test that config accepts custom values."""
        config = CobolEvalAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            resources_server=ResourcesServerRef(
                type="resources_servers",
                name="cobol_compiler",
            ),
            model_server=ModelServerRef(
                type="responses_api_models",
                name="policy_model",
            ),
            max_correction_turns=5,
            include_all_attempts=False,
        )
        assert config.max_correction_turns == 5
        assert config.include_all_attempts is False
