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

from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgent,
    BaseResponsesAPIAgentConfig,
    SimpleResponsesAPIAgent,
)
from nemo_gym.global_config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.server_utils import ServerClient


class TestBaseResponsesAPIAgent:
    def test_BaseResponsesAPIAgent(self) -> None:
        config = BaseResponsesAPIAgentConfig(host="", port=0, entrypoint="", name="")
        BaseResponsesAPIAgent(config=config)

    def test_SimpleResponsesAPIAgent(self) -> None:
        config = BaseResponsesAPIAgentConfig(host="", port=0, entrypoint="", name="")

        class TestSimpleResponsesAPIAgent(SimpleResponsesAPIAgent):
            async def responses(self, body=...):
                raise NotImplementedError

            async def run(self, body=...):
                raise NotImplementedError

        agent = TestSimpleResponsesAPIAgent(
            config=config, server_client=MagicMock(spec=ServerClient, global_config_dict=dict())
        )
        agent.setup_webserver()

    def test_resolve_model_call_path(self):
        mock_agent = MagicMock()
        mock_agent._capture_config.should_capture_model_calls = True

        with_id = SimpleResponsesAPIAgent.resolve_model_call_path(
            mock_agent, base_url_or_path="http://my-test-url/v1", body={TASK_INDEX_KEY_NAME: 2}
        )
        assert with_id == "http://my-test-url/v1"

        with_id = SimpleResponsesAPIAgent.resolve_model_call_path(
            mock_agent,
            base_url_or_path="http://my-test-url/v1",
            body={TASK_INDEX_KEY_NAME: 2, ROLLOUT_INDEX_KEY_NAME: 4},
        )
        assert with_id == "http://my-test-url/v1/ng-rollout/2-4"

        with_id = SimpleResponsesAPIAgent.resolve_model_call_path(
            mock_agent, base_url_or_path="/v1/responses", body={TASK_INDEX_KEY_NAME: 2, ROLLOUT_INDEX_KEY_NAME: 4}
        )
        assert with_id == "/v1/responses/ng-rollout/2-4"
