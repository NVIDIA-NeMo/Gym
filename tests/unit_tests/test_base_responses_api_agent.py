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

import pytest
from omegaconf import OmegaConf

from nemo_gym.base_resources_server import AggregateMetricsRequest
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgent,
    BaseResponsesAPIAgentConfig,
    ModelEndpoint,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef
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

    async def test_aggregate_metrics_skip_verification_warns_and_returns_empty_metrics(self) -> None:
        config = BaseResponsesAPIAgentConfig(
            host="",
            port=0,
            entrypoint="",
            name="",
            skip_verification=True,
        )

        class TestSimpleResponsesAPIAgent(SimpleResponsesAPIAgent):
            async def responses(self, body=...):
                raise NotImplementedError

            async def run(self, body=...):
                raise NotImplementedError

        agent = TestSimpleResponsesAPIAgent(config=config, server_client=MagicMock(spec=ServerClient))
        body = AggregateMetricsRequest(verify_responses=[])

        with pytest.warns(RuntimeWarning, match="skip_verification=True"):
            result = await agent.aggregate_metrics(body)

        assert result.group_level_metrics == []
        assert result.agent_metrics == {}
        assert result.key_metrics == {}

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
        # Evaluation capture correlates every agent.
        # It does not depend on the agent's training-token opt-in.
        body = {"_ng_task_index": 0, "_ng_rollout_index": 0}
        assert self._agent({}).rollout_id_from_run(body) is None
        assert self._agent({"observability_enabled": True}).rollout_id_from_run(body) == "0-0"

    def test_token_capture_prefix_is_scoped_to_participating_agents(self) -> None:
        # Training-token capture requires both run-level enablement and agent opt-in.
        # Correlated calls preserve ``/ng-rollout/<id>/training-token-capture``.
        # Native agents carry token ids inline and do not opt in.
        body = {"_ng_task_index": 0, "_ng_rollout_index": 0}
        gc = {"token_id_capture": {"enabled": True}}
        assert self._agent(gc, token_id_capture=False).rollout_id_from_run(body) is None
        assert self._agent(gc, token_id_capture=True).rollout_id_from_run(body) == "0-0"
        # Agent opt-in alone does not enable capture.
        assert self._agent({}, token_id_capture=True).rollout_id_from_run(body) is None


class TestModelEndpoint:
    MODEL_SERVER = ModelServerRef(type="responses_api_models", name="policy_model")

    def _agent(self, *, token_capture: bool = False) -> SimpleResponsesAPIAgent:
        config = BaseResponsesAPIAgentConfig(host="", port=0, entrypoint="", name="", token_id_capture=token_capture)

        class _Agent(SimpleResponsesAPIAgent):
            async def responses(self, body=...):
                raise NotImplementedError

            async def run(self, body=...):
                raise NotImplementedError

        client = MagicMock(spec=ServerClient)
        client.global_config_dict = OmegaConf.create(
            {
                "policy_model": {"responses_api_models": {"vllm_model": {"host": "model-host", "port": 8000}}},
                "token_id_capture": {"enabled": token_capture},
            }
        )
        client._build_server_base_url.side_effect = lambda server: ServerClient._build_server_base_url(client, server)
        return _Agent(config=config, server_client=client)

    def test_session_endpoint_wins_over_model_server_and_base_url(self) -> None:
        session_endpoint = ModelEndpoint(base_url="http://127.0.0.1:4000/v1", model="served-model")
        endpoint = self._agent().model_endpoint(
            model_server=self.MODEL_SERVER,
            rollout_id="3-1",
            base_url="https://api.example.com/v1",
            session_endpoint=session_endpoint,
        )
        assert endpoint == session_endpoint

    def test_model_server_carries_the_rollout_prefix(self) -> None:
        endpoint = self._agent().model_endpoint(
            model_server=self.MODEL_SERVER, rollout_id="3-1", base_url="https://api.example.com/v1"
        )
        assert endpoint == ModelEndpoint(base_url="http://model-host:8000/ng-rollout/3-1/v1")

    def test_model_server_without_rollout_id_is_unprefixed(self) -> None:
        endpoint = self._agent().model_endpoint(model_server=self.MODEL_SERVER)
        assert endpoint == ModelEndpoint(base_url="http://model-host:8000/v1")

    def test_model_server_keeps_the_token_capture_route(self) -> None:
        endpoint = self._agent(token_capture=True).model_endpoint(model_server=self.MODEL_SERVER, rollout_id="3-1")
        assert endpoint == ModelEndpoint(base_url="http://model-host:8000/ng-rollout/3-1/training-token-capture/v1")

    def test_configured_base_url_is_used_verbatim(self) -> None:
        endpoint = self._agent(token_capture=True).model_endpoint(
            model_server=None, rollout_id="3-1", base_url="https://api.example.com/v1"
        )
        assert endpoint == ModelEndpoint(base_url="https://api.example.com/v1")

    @pytest.mark.parametrize("base_url", [None, ""])
    def test_no_endpoint_configured_raises(self, base_url: str | None) -> None:
        with pytest.raises(ValueError, match="No model endpoint"):
            self._agent().model_endpoint(model_server=None, rollout_id="3-1", base_url=base_url)
