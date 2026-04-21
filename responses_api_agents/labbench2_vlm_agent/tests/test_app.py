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

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.server_utils import ServerClient
from responses_api_agents.labbench2_vlm_agent.app import (
    LabbenchVLMAgent,
    LabbenchVLMAgentConfig,
)


def test_agent_instantiation() -> None:
    config = LabbenchVLMAgentConfig(
        name="labbench2_vlm_simple_agent",
        host="0.0.0.0",
        port=8080,
        entrypoint="app.py",
        resources_server=ResourcesServerRef(type="resources_servers", name="labbench2_vlm"),
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        media_base_dir="resources_servers/labbench2_vlm/data",
        dpi=170,
    )
    agent = LabbenchVLMAgent(config=config, server_client=MagicMock(spec=ServerClient))
    assert agent.config.media_base_dir == "resources_servers/labbench2_vlm/data"
    assert agent.config.dpi == 170
