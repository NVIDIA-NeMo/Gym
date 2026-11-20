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
from fastapi import Body, Request
from fastapi.testclient import TestClient
from omegaconf import OmegaConf
from pydantic import ConfigDict, ValidationError

from nemo_gym.agents.base_responses_api_agent import (
    AgentSeedRejectedError,
    BaseResponsesAPIAgent,
    BaseResponsesAPIAgentConfig,
    ModelEndpoint,
    SimpleResponsesAPIAgent,
    StrictResponsesAPIAgentConfig,
    assert_model_url_reachable_from_sandbox,
    is_loopback_host,
)
from nemo_gym.client.server_client import ServerClient
from nemo_gym.config.config_types import ModelServerRef
from nemo_gym.core.model_call_sampling import ModelCallSampling, encode_model_sampling
from nemo_gym.core.responses_api_types import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_runtime.base_resources_server import AggregateMetricsRequest, BaseRunRequest


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

    def test_agent_without_run_starts_and_its_run_answers_422(self) -> None:
        # run() is deprecated: an agent that implements only agent sessions need not define it.
        config = BaseResponsesAPIAgentConfig(host="", port=0, entrypoint="", name="")

        class SessionOnlyAgent(SimpleResponsesAPIAgent):
            async def responses(self, body=...):
                raise NotImplementedError

        agent = SessionOnlyAgent(config=config, server_client=MagicMock(spec=ServerClient))
        with TestClient(agent.setup_webserver()) as client:
            response = client.post("/run", json={"responses_create_params": {"input": "hi"}})

        assert response.status_code == 422
        assert "SessionOnlyAgent does not implement /run" in response.json()["detail"]
        assert "drive it through agent sessions from an environment server" in response.json()["detail"]

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


class TestModelSampling:
    """A trained agent's model URLs carry the rollout's sampling values, whichever way the agent builds them."""

    VALUES = {"temperature": 0.6, "top_p": 0.95, "max_output_tokens": 64}
    SEGMENT = encode_model_sampling(ModelCallSampling(**VALUES))

    def _agent(self, *, observability: bool = True) -> tuple[SimpleResponsesAPIAgent, dict]:
        """An agent that records the model URLs it builds, and the record."""
        config = BaseResponsesAPIAgentConfig(host="", port=0, entrypoint="", name="agent")
        seen: dict = {}

        class _RunRequest(BaseRunRequest):
            model_config = ConfigDict(extra="allow")

        class _Agent(SimpleResponsesAPIAgent):
            async def responses(self, request: Request, body: dict = Body()) -> dict:
                seen["responses"] = {
                    "model_path": self.url_path_for_request("/v1/responses", request),
                    "model_url": self.resolve_model_base_url("policy_model"),
                }
                return {}

            async def run(self, body: _RunRequest) -> dict:
                seen["run"] = {
                    "self_call": self.url_path_for_run("/v1/responses", body),
                    "harness_url": f"{self.base_url_for_run('http://model-host:8000', body)}/v1",
                    "model_url": self.resolve_model_base_url("policy_model", self.rollout_id_from_run(body)),
                }
                seen["untrained"] = {
                    "model_path": self.url_path_for_run("/v1/chat/completions", body, trained=False),
                    "harness_url": f"{self.base_url_for_run('http://model-host:8000', body, trained=False)}/v1",
                    "model_url": self.resolve_model_base_url(
                        "policy_model", self.rollout_id_from_run(body), trained=False
                    ),
                }
                return {}

        client = MagicMock(spec=ServerClient)
        client.global_config_dict = OmegaConf.create(
            {
                "observability_enabled": observability,
                "policy_model": {"responses_api_models": {"vllm_model": {"host": "model-host", "port": 8000}}},
            }
        )
        client._build_server_base_url.side_effect = lambda server: ServerClient._build_server_base_url(client, server)
        return _Agent(config=config, server_client=client), seen

    def test_a_run_with_values_puts_them_in_every_model_url_it_builds(self) -> None:
        agent, seen = self._agent()
        row = {"_ng_task_index": 3, "_ng_rollout_index": 1, "responses_create_params": {"input": "task"}}

        TestClient(agent.setup_webserver()).post("/run", json=row | {"_ng_model_sampling": self.VALUES})

        assert seen["run"] == {
            "self_call": f"/ng-rollout/3-1/ng-sampling/{self.SEGMENT}/v1/responses",
            "harness_url": f"http://model-host:8000/ng-rollout/3-1/ng-sampling/{self.SEGMENT}/v1",
            "model_url": f"http://model-host:8000/ng-rollout/3-1/ng-sampling/{self.SEGMENT}/v1",
        }

        # Without a rollout prefix the segment still travels; without values nothing changes.
        unprefixed, unprefixed_seen = self._agent(observability=False)
        TestClient(unprefixed.setup_webserver()).post("/run", json=row | {"_ng_model_sampling": self.VALUES})
        assert unprefixed_seen["run"]["self_call"] == f"/ng-sampling/{self.SEGMENT}/v1/responses"
        TestClient(unprefixed.setup_webserver()).post("/run", json=row)
        assert unprefixed_seen["run"] == {
            "self_call": "/v1/responses",
            "harness_url": "http://model-host:8000/v1",
            "model_url": "http://model-host:8000/v1",
        }

    def test_a_role_the_framework_does_not_train_gets_null_in_a_run_with_values(self) -> None:
        agent, seen = self._agent()
        row = {"_ng_task_index": 3, "_ng_rollout_index": 1, "responses_create_params": {"input": "task"}}

        TestClient(agent.setup_webserver()).post("/run", json=row | {"_ng_model_sampling": self.VALUES})

        assert seen["untrained"] == {
            "model_path": "/ng-rollout/3-1/ng-sampling/null/v1/chat/completions",
            "harness_url": "http://model-host:8000/ng-rollout/3-1/ng-sampling/null/v1",
            "model_url": "http://model-host:8000/ng-rollout/3-1/ng-sampling/null/v1",
        }

        # Gym-only evaluation sends no values, so no role gets a segment.
        TestClient(agent.setup_webserver()).post("/run", json=row)
        assert seen["untrained"] == {
            "model_path": "/ng-rollout/3-1/v1/chat/completions",
            "harness_url": "http://model-host:8000/ng-rollout/3-1/v1",
            "model_url": "http://model-host:8000/ng-rollout/3-1/v1",
        }

    @pytest.mark.parametrize("capture", ["", "/training-token-capture"])
    def test_an_activation_passes_its_segment_on_to_the_agents_model_calls(self, capture: str) -> None:
        agent, seen = self._agent()

        response = TestClient(agent.setup_webserver()).post(
            f"/ng-rollout/3-1{capture}/ng-sampling/{self.SEGMENT}/v1/responses", json={"input": "task"}
        )

        assert response.status_code == 200, response.text
        assert seen["responses"] == {
            "model_path": f"/ng-rollout/3-1{capture}/ng-sampling/{self.SEGMENT}/v1/responses",
            "model_url": f"http://model-host:8000/ng-sampling/{self.SEGMENT}/v1",
        }

    def test_an_invalid_segment_is_refused(self) -> None:
        agent, _ = self._agent()
        response = TestClient(agent.setup_webserver()).post(
            "/ng-rollout/3-1/ng-sampling/not+valid/v1/responses", json={"input": "task"}
        )

        assert response.status_code == 422


class TestRolloutTokenCapture:
    """A rollout that skips token capture sends none of its model calls with the training tag.

    The run captures every agent (RL's ``all_agents: true``); only the rollout's own setting skips capture.
    """

    def _agent(self) -> SimpleResponsesAPIAgent:
        class _Agent(SimpleResponsesAPIAgent):
            async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> dict:
                # A harness builds its model URL from the agent's own capture setting, as Claude Code does.
                return {"model_base_url": self.resolve_model_base_url("policy_model", "3-1")}

            async def run(self, body: BaseRunRequest = Body()) -> dict:
                # A harness's URL follows the request's capture setting; the run's URL helpers read the row.
                return {
                    "model_url_path": self.url_path_for_run("/v1/responses", body),
                    "model_base_url": self.resolve_model_base_url("policy_model", self.rollout_id_from_run(body)),
                }

        client = MagicMock(spec=ServerClient)
        client.global_config_dict = OmegaConf.create(
            {
                "policy_model": {"responses_api_models": {"vllm_model": {"host": "model-host", "port": 8000}}},
                "token_id_capture": {"enabled": True, "all_agents": True},
            }
        )
        client._build_server_base_url.side_effect = lambda server: ServerClient._build_server_base_url(client, server)
        config = BaseResponsesAPIAgentConfig(host="", port=0, entrypoint="", name="agent")
        return _Agent(config=config, server_client=client)

    def _client(self) -> TestClient:
        return TestClient(self._agent().setup_webserver())

    @pytest.mark.parametrize(
        ("row", "model_url_path", "model_base_url"),
        [
            (
                {},
                "/ng-rollout/3-1/training-token-capture/v1/responses",
                "http://model-host:8000/ng-rollout/3-1/training-token-capture/v1",
            ),
            (
                {"_ng_skip_token_capture": False},
                "/ng-rollout/3-1/training-token-capture/v1/responses",
                "http://model-host:8000/ng-rollout/3-1/training-token-capture/v1",
            ),
            (
                {"_ng_skip_token_capture": None},
                "/ng-rollout/3-1/training-token-capture/v1/responses",
                "http://model-host:8000/ng-rollout/3-1/training-token-capture/v1",
            ),
            # The rollout keeps its prefix and drops only the training tag.
            (
                {"_ng_skip_token_capture": True},
                "/ng-rollout/3-1/v1/responses",
                "http://model-host:8000/ng-rollout/3-1/v1",
            ),
        ],
    )
    def test_a_run_follows_its_rollout_setting(self, row: dict, model_url_path: str, model_base_url: str) -> None:
        reply = self._client().post(
            "/run", json={"responses_create_params": {"input": "task"}, "_ng_rollout_id": "3-1", **row}
        )

        assert reply.status_code == 200, reply.text
        assert reply.json() == {"model_url_path": model_url_path, "model_base_url": model_base_url}

    def test_a_run_with_a_setting_that_is_not_a_boolean_is_refused(self) -> None:
        reply = self._client().post(
            "/run",
            json={
                "responses_create_params": {"input": "task"},
                "_ng_rollout_id": "3-1",
                "_ng_skip_token_capture": "yes",
            },
        )

        assert reply.status_code == 422

    @pytest.mark.parametrize(
        ("activation_path", "model_base_url"),
        [
            (
                "/ng-rollout/3-1/training-token-capture/v1/responses",
                "http://model-host:8000/ng-rollout/3-1/training-token-capture/v1",
            ),
            # The caller that built the path decided not to capture: the agent's own setting does not override it.
            ("/ng-rollout/3-1/v1/responses", "http://model-host:8000/ng-rollout/3-1/v1"),
            # A call that names no rollout keeps the agent's own setting.
            ("/v1/responses", "http://model-host:8000/ng-rollout/3-1/training-token-capture/v1"),
            # The rollout's sampling values ride along either way; only the training tag follows the path.
            (
                f"/ng-rollout/3-1/ng-sampling/{TestModelSampling.SEGMENT}/v1/responses",
                f"http://model-host:8000/ng-rollout/3-1/ng-sampling/{TestModelSampling.SEGMENT}/v1",
            ),
            (
                f"/ng-rollout/3-1/training-token-capture/ng-sampling/{TestModelSampling.SEGMENT}/v1/responses",
                f"http://model-host:8000/ng-rollout/3-1/training-token-capture/ng-sampling/{TestModelSampling.SEGMENT}/v1",
            ),
        ],
    )
    def test_an_activation_follows_the_training_tag_of_its_path(
        self, activation_path: str, model_base_url: str
    ) -> None:
        reply = self._client().post(activation_path, json={"input": "task"})

        assert reply.status_code == 200, reply.text
        assert reply.json() == {"model_base_url": model_base_url}

    def test_every_rollout_route_without_the_training_tag_turns_capture_off(self) -> None:
        # Covers activation routes added later, such as one per agent instance: each must turn capture off.
        app = self._agent().setup_webserver()
        paths = [
            path
            for route in app.routes
            if (path := getattr(route, "path", "")).startswith("/ng-rollout/{rollout_id}/")
            and path.endswith("/v1/responses")
            and "training-token-capture" not in path
        ]
        assert "/ng-rollout/{rollout_id}/v1/responses" in paths
        client = TestClient(app)
        for path in paths:
            url = path.replace("{rollout_id}", "3-1")
            url = "/".join("x" if part.startswith("{") else part for part in url.split("/"))
            reply = client.post(url, json={"input": "task"})

            assert reply.status_code == 200, (path, reply.text)
            assert "training-token-capture" not in reply.json()["model_base_url"], path

    def test_the_run_url_helpers_read_the_row_where_the_run_route_sets_no_context(self) -> None:
        # Some agents register their own /run route; the URL helpers still read the row's setting.
        agent = self._agent()
        captured_row = {"_ng_rollout_id": "3-1"}
        uncaptured_row = {"_ng_rollout_id": "3-1", "_ng_skip_token_capture": True}

        assert agent.url_path_for_run("/v1/responses", captured_row) == (
            "/ng-rollout/3-1/training-token-capture/v1/responses"
        )
        assert agent.url_path_for_run("/v1/responses", uncaptured_row) == "/ng-rollout/3-1/v1/responses"
        assert (
            agent.base_url_for_run("http://model-host:8000", uncaptured_row) == "http://model-host:8000/ng-rollout/3-1"
        )


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
        with pytest.raises(AgentSeedRejectedError, match=r"\+\+use_absolute_ip=true") as excinfo:
            assert_model_url_reachable_from_sandbox(url, provider_name=provider_name)
        # The same configuration fails every attempt, so a seed refuses it with a final 422.
        assert excinfo.value.status_code == 422
        assert isinstance(excinfo.value, ValueError)
        assert url in excinfo.value.detail
        assert "reachable model URL" in excinfo.value.detail

    def test_loopback_on_docker_warns_and_passes(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="nemo_gym.agents.base_responses_api_agent"):
            assert_model_url_reachable_from_sandbox("http://127.0.0.1:8000/v1", provider_name="docker")
        warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "use_absolute_ip" in warnings[0].getMessage()
        assert "host network" in warnings[0].getMessage()

    def test_loopback_on_host_local_provider_passes_silently(self, caplog: pytest.LogCaptureFixture) -> None:
        # The ``local`` provider runs on the host itself, so loopback is the host.
        with caplog.at_level(logging.WARNING, logger="nemo_gym.agents.base_responses_api_agent"):
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
        with caplog.at_level(logging.WARNING, logger="nemo_gym.agents.base_responses_api_agent"):
            assert_model_url_reachable_from_sandbox(url, provider_name=provider_name)
        assert not caplog.records

    @pytest.mark.parametrize("url", ["", "not a url", "http://", "http://[bad"])
    def test_unparseable_hosts_are_not_loopback(self, url: str) -> None:
        assert is_loopback_host(url) is False
        assert_model_url_reachable_from_sandbox(url, provider_name="opensandbox")


class TestStrictResponsesAPIAgentConfig:
    BASE = {"name": "agent", "host": "127.0.0.1", "port": 8000, "entrypoint": "app.py"}

    def test_an_undeclared_key_is_refused(self) -> None:
        # A misspelled network_access would otherwise be dropped and leave the sandbox open.
        with pytest.raises(ValidationError, match="network_acess"):
            StrictResponsesAPIAgentConfig.model_validate({**self.BASE, "network_acess": "model_only"})

    def test_keys_other_components_read_are_accepted(self) -> None:
        config = StrictResponsesAPIAgentConfig.model_validate(
            {
                **self.BASE,
                "datasets": [{"name": "train", "type": "train", "jsonl_fpath": "data/train.jsonl"}],
                "description": "an agent",
                "tool_servers": [{"type": "resources_servers", "name": "tavily"}],
            }
        )
        assert config.description == "an agent"
        assert config.tool_servers == [{"type": "resources_servers", "name": "tavily"}]
