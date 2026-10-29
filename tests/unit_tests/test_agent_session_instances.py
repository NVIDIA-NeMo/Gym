# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Agent sessions are bound to the agent instance they were seeded as."""

from unittest.mock import MagicMock

import pytest
from fastapi import Body, Request
from fastapi.testclient import TestClient
from pydantic import PrivateAttr, ValidationError

from nemo_gym.base_environment_server import require_distinct_agent_instances
from nemo_gym.base_responses_api_agent import (
    AgentCloseSessionResponse,
    AgentSeedSessionRequest,
    AgentSessionState,
    BaseResponsesAPIAgentConfig,
    SimpleResponsesAPIAgent,
)
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_correlation import current_agent_instance
from nemo_gym.server_utils import ServerClient


EPISODE = {"rollout_id": "r", "attempt": 0}
RUN_BODY = {"_ng_rollout_id": "r"}


def _reply() -> NeMoGymResponse:
    return NeMoGymResponse(
        id="reply",
        created_at=0,
        model="test",
        object="response",
        output=[],
        tool_choice="auto",
        parallel_tool_calls=True,
        tools=[],
    )


class _NamedAgent(SimpleResponsesAPIAgent):
    _activations: list[tuple] = PrivateAttr(default_factory=list)
    _seed_urls: list[str] = PrivateAttr(default_factory=list)

    async def responses(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        self._activations.append(
            (
                current_agent_instance(),
                self.url_path_for_request("/v1/responses", request),
                self.base_url_for_run("http://model", RUN_BODY),
                self.base_url_for_run("http://model", RUN_BODY, agent_instance="judge"),
            )
        )
        return _reply()

    async def run(self, body):
        raise NotImplementedError

    async def _seed_agent_session_state(self, body: AgentSeedSessionRequest) -> AgentSessionState:
        # A harness that receives its model URL when the session starts.
        self._seed_urls.append(self.base_url_for_run("http://model", RUN_BODY))
        return AgentSessionState(request=body)

    async def _close_agent_session_state(self, state: AgentSessionState) -> AgentCloseSessionResponse:
        return AgentCloseSessionResponse(agent_session_id=state.request.agent_session_id, cleanup_confirmed=True)


@pytest.fixture
def agent() -> _NamedAgent:
    return _NamedAgent(
        config=BaseResponsesAPIAgentConfig(host="localhost", port=1, entrypoint="app.py", name="test"),
        server_client=MagicMock(spec=ServerClient, global_config_dict={"observability_enabled": True}),
    )


def _seed(client: TestClient, session_id: str, agent_instance: str | None):
    body = AgentSeedSessionRequest(
        agent_session_id=session_id,
        episode_id=EPISODE,
        task_id={"taskset": "test", "task_id": "task"},
        agent_instance=agent_instance,
    )
    return client.post("/v1/agent_sessions", json=body.model_dump(mode="json"))


def _activate(client: TestClient, prefix: str):
    return client.post(f"{prefix}/v1/responses", json={"input": "go"})


def test_an_unnamed_seed_serializes_as_before_and_a_named_one_is_validated():
    unnamed = AgentSeedSessionRequest(
        agent_session_id="s", episode_id=EPISODE, task_id={"taskset": "t", "task_id": "x"}
    )
    assert "agent_instance" not in unnamed.model_dump(mode="json")
    named = unnamed.model_copy(update={"agent_instance": "debater.1"})
    assert AgentSeedSessionRequest.model_validate(named.model_dump(mode="json")).agent_instance == "debater.1"
    with pytest.raises(ValidationError, match="invalid agent instance name"):
        AgentSeedSessionRequest.model_validate({**unnamed.model_dump(mode="json"), "agent_instance": "User"})


def test_a_named_session_s_activation_and_model_calls_carry_its_name(agent):
    client = TestClient(agent.setup_webserver())
    assert _seed(client, "s-user", "user").status_code == 200

    response = _activate(client, "/ng-rollout/r/ng-agent-instance/user")

    assert response.status_code == 200, response.text
    assert agent._seed_urls == ["http://model/ng-rollout/r/ng-agent-instance/user"]
    assert agent._activations == [
        (
            "user",
            "/ng-rollout/r/ng-agent-instance/user/v1/responses",
            "http://model/ng-rollout/r/ng-agent-instance/user",
            # An explicit name wins over the session's.
            "http://model/ng-rollout/r/ng-agent-instance/judge",
        )
    ]
    assert current_agent_instance() is None


@pytest.mark.parametrize(
    "seeded, path, status",
    [
        ("user", "/ng-rollout/r/ng-agent-instance/judge", 409),
        ("user", "/ng-rollout/r", 409),
        ("user", "", 409),
        (None, "/ng-rollout/r/ng-agent-instance/user", 409),
        (None, "/ng-rollout/r", 200),
        ("user", "/ng-rollout/r/ng-agent-instance/user/training-token-capture", 200),
    ],
)
def test_an_activation_must_name_the_session_s_agent_instance(agent, seeded, path, status):
    client = TestClient(agent.setup_webserver())
    assert _seed(client, "s", seeded).status_code == 200

    response = _activate(client, path)

    assert response.status_code == status, response.text
    if status == 409:
        assert agent._activations == []


def test_one_open_session_per_agent_instance_of_an_episode(agent):
    first = TestClient(agent.setup_webserver())
    second = TestClient(first.app)
    assert _seed(first, "s-1", "user").status_code == 200

    assert _seed(second, "s-2", "user").status_code == 409
    assert _seed(TestClient(first.app), "s-3", "assistant").status_code == 200

    closed = first.post("/v1/agent_sessions/close", json={"agent_session_id": "s-1", "episode_id": EPISODE})
    assert closed.status_code == 200, closed.text
    assert _seed(second, "s-2", "user").status_code == 200


def test_an_environment_server_refuses_to_seed_a_name_twice():
    require_distinct_agent_instances(["assistant", "user", None, None])
    with pytest.raises(ValueError, match="seeded twice"):
        require_distinct_agent_instances(["assistant", "debater.1", "debater.1"])


def test_an_inbound_name_wins_over_the_session_s_when_carrying_the_path(agent):
    from types import SimpleNamespace

    from nemo_gym.rollout_correlation import agent_instance_context

    request = SimpleNamespace(scope={"type": "http", "path": "/ng-rollout/r/ng-agent-instance/user/v1/responses"})
    unnamed = SimpleNamespace(scope={"type": "http", "path": "/ng-rollout/r/training-token-capture/v1/responses"})
    with agent_instance_context("assistant"):
        assert agent.url_path_for_request("/v1/x", request) == "/ng-rollout/r/ng-agent-instance/user/v1/x"
        assert (
            agent.url_path_for_request("/v1/x", unnamed)
            == "/ng-rollout/r/ng-agent-instance/assistant/training-token-capture/v1/x"
        )
    assert agent.url_path_for_request("/v1/x", SimpleNamespace(scope={"type": "http", "path": "/v1/x"})) == "/v1/x"
