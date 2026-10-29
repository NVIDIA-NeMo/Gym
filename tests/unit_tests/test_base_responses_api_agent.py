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
import logging
from typing import ClassVar
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

from nemo_gym.base_resources_server import AggregateMetricsRequest
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgent,
    BaseResponsesAPIAgentConfig,
    ModelEndpoint,
    SimpleResponsesAPIAgent,
    assert_model_url_reachable_from_sandbox,
    is_loopback_host,
)
from nemo_gym.client import ServerClient
from nemo_gym.config import ModelServerRef


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


class TestCallsModelFromSandbox:
    def test_default_is_false_and_a_declaration_is_a_class_var_not_a_field(self) -> None:
        class InSandbox(SimpleResponsesAPIAgent):
            calls_model_from_sandbox: ClassVar[bool] = True

        class Inherits(InSandbox):
            pass

        assert BaseResponsesAPIAgent.calls_model_from_sandbox is False
        assert SimpleResponsesAPIAgent.calls_model_from_sandbox is False
        assert InSandbox.calls_model_from_sandbox is True
        assert Inherits.calls_model_from_sandbox is True
        assert "calls_model_from_sandbox" not in InSandbox.model_fields
        assert "calls_model_from_sandbox" not in BaseResponsesAPIAgentConfig.model_fields


class TestAssertModelUrlReachableFromSandbox:
    @pytest.mark.parametrize(
        "url",
        [
            "http://127.0.0.1:8000/v1",
            "http://127.0.0.1:8000/ng-rollout/0-0/v1",
            "http://127.5.6.7:8000/v1",
            "http://localhost:8000/v1",
            "http://LOCALHOST:8000/v1",
            "http://[::1]:8000/v1",
        ],
    )
    @pytest.mark.parametrize("provider_name", ["opensandbox", "e2b", "daytona", None])
    def test_loopback_on_remote_provider_raises_with_fix(self, url: str, provider_name: str | None) -> None:
        with pytest.raises(ValueError, match=r"\+\+use_absolute_ip=true") as excinfo:
            assert_model_url_reachable_from_sandbox(url, provider_name=provider_name)
        assert url in str(excinfo.value)
        assert "reachable model URL" in str(excinfo.value)

    def test_loopback_on_docker_warns_and_passes(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="nemo_gym.base_responses_api_agent"):
            assert_model_url_reachable_from_sandbox("http://127.0.0.1:8000/v1", provider_name="docker")
        warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "use_absolute_ip" in warnings[0].getMessage()
        assert "host network" in warnings[0].getMessage()

    def test_loopback_on_host_local_provider_passes_silently(self, caplog: pytest.LogCaptureFixture) -> None:
        # The ``local`` provider runs on the host itself, so loopback is the host.
        with caplog.at_level(logging.WARNING, logger="nemo_gym.base_responses_api_agent"):
            assert_model_url_reachable_from_sandbox("http://127.0.0.1:8000/v1", provider_name="local")
        assert not caplog.records

    @pytest.mark.parametrize(
        "url",
        [
            "http://10.0.0.5:8000/v1",
            "http://model-server.internal:8000/ng-rollout/0-0/v1",
            "https://api.example.com/v1",
            "http://[fd00::1]:8000/v1",
        ],
    )
    @pytest.mark.parametrize("provider_name", ["opensandbox", "docker", None])
    def test_non_loopback_passes(self, url: str, provider_name: str | None, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="nemo_gym.base_responses_api_agent"):
            assert_model_url_reachable_from_sandbox(url, provider_name=provider_name)
        assert not caplog.records

    @pytest.mark.parametrize("url", ["", "not a url", "http://", "http://[bad"])
    def test_unparseable_hosts_are_not_loopback(self, url: str) -> None:
        assert is_loopback_host(url) is False
        assert_model_url_reachable_from_sandbox(url, provider_name="opensandbox")
