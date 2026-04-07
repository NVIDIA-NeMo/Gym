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
from responses_api_agents.gymnasium_agent.app import GymnasiumAgent, GymnasiumAgentConfig


def _make_agent():
    config = GymnasiumAgentConfig(
        host="",
        port=0,
        entrypoint="",
        name="test_gymnasium_agent",
        env_server=ResourcesServerRef(type="resources_servers", name="my_env"),
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
    )
    return GymnasiumAgent(config=config, server_client=MagicMock(spec=ServerClient))


class TestGymnasiumAgent:
    def test_setup_webserver(self):
        agent = _make_agent()
        app = agent.setup_webserver()
        routes = {r.path for r in app.routes}
        assert "/run" in routes
        assert "/v1/responses" in routes
        assert "/aggregate_metrics" in routes

    def test_config(self):
        agent = _make_agent()
        assert agent.config.env_server.name == "my_env"
        assert agent.config.model_server.name == "policy_model"
        assert agent.config.max_steps == 10
