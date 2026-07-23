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

        agent = TestSimpleResponsesAPIAgent(config=config, server_client=MagicMock(spec=ServerClient))
        agent.setup_webserver()

    def _agent(self, global_config: dict, *, token_id_capture: bool = False) -> SimpleResponsesAPIAgent:
        config = BaseResponsesAPIAgentConfig(
            host="", port=0, entrypoint="", name="", token_id_capture=token_id_capture
        )

        class _Agent(SimpleResponsesAPIAgent):
            async def responses(self, body=...):
                raise NotImplementedError

            async def run(self, body=...):
                raise NotImplementedError

        client = MagicMock(spec=ServerClient)
        client.global_config_dict = global_config
        return _Agent(config=config, server_client=client)

    def test_eval_capture_prefix_applies_to_every_agent(self) -> None:
        # Eval capture (observability_enabled) correlates every agent, regardless of the per-agent
        # token-capture opt-in.
        body = {"_ng_task_index": 0, "_ng_rollout_index": 0}
        assert self._agent({}).rollout_id_from_run(body) is None
        assert self._agent({"observability_enabled": True}).rollout_id_from_run(body) == "0-0"

    def test_token_capture_prefix_is_scoped_to_participating_agents(self) -> None:
        # Training token capture correlates a call only when the run-level switch is on AND the agent
        # opted in -- native agents (opt-out) carry token ids inline and must not be correlated here.
        body = {"_ng_task_index": 0, "_ng_rollout_index": 0}
        gc = {"token_id_capture_enabled": True}
        assert self._agent(gc, token_id_capture=False).rollout_id_from_run(body) is None
        assert self._agent(gc, token_id_capture=True).rollout_id_from_run(body) == "0-0"
        # The run-level switch is still required: opting in alone does nothing.
        assert self._agent({}, token_id_capture=True).rollout_id_from_run(body) is None
