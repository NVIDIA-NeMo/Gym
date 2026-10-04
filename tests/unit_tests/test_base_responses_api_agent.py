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
from unittest.mock import MagicMock

import pytest

from nemo_gym.base_resources_server import AggregateMetricsRequest
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgent,
    BaseResponsesAPIAgentConfig,
    SimpleResponsesAPIAgent,
    assert_model_url_reachable_from_sandbox,
    is_loopback_host,
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

    def test_ipv6_loopback_raises(self) -> None:
        with pytest.raises(ValueError, match="::1"):
            assert_model_url_reachable_from_sandbox("http://[::1]:8000/v1", provider_name="opensandbox")

    @pytest.mark.parametrize("url", ["", "not a url", "http://", "http://[bad"])
    def test_unparseable_hosts_are_not_loopback(self, url: str) -> None:
        assert is_loopback_host(url) is False
        assert_model_url_reachable_from_sandbox(url, provider_name="opensandbox")
