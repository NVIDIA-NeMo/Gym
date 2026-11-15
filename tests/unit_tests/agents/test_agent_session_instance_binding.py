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
"""The scope of an agent session's agent instance binding: which seeds it refuses and which URLs carry the name."""

from unittest.mock import MagicMock

import pytest
from fastapi import Body, Request
from fastapi.testclient import TestClient
from pydantic import PrivateAttr

from nemo_gym.agents.base_responses_api_agent import (
    AgentCloseSessionResponse,
    AgentSeedSessionRequest,
    AgentSessionState,
    BaseResponsesAPIAgentConfig,
    SimpleResponsesAPIAgent,
)
from nemo_gym.client.server_client import ServerClient
from nemo_gym.core.episode_context import current_agent_instance
from nemo_gym.core.responses_api_types import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming


MODEL_SERVER = {"responses_api_models": {"vllm_model": {"host": "model", "port": 80}}}


class _RecordingAgent(SimpleResponsesAPIAgent):
    _activations: list[tuple] = PrivateAttr(default_factory=list)
    _seed_urls: list[str] = PrivateAttr(default_factory=list)

    async def responses(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        self._activations.append(
            (
                current_agent_instance(),
                self.resolve_model_base_url("policy", "r"),
                self.resolve_model_base_url("policy", "r", agent_instance="judge"),
            )
        )
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

    async def run(self, body):
        raise NotImplementedError

    async def _seed_agent_session_state(self, body: AgentSeedSessionRequest) -> AgentSessionState:
        # A harness that resolves its model URL when the session starts.
        self._seed_urls.append(self.resolve_model_base_url("policy", "r"))
        return AgentSessionState(request=body)

    async def _close_agent_session_state(self, state: AgentSessionState) -> AgentCloseSessionResponse:
        return AgentCloseSessionResponse(agent_session_id=state.request.agent_session_id, cleanup_confirmed=True)


@pytest.fixture
def agent() -> _RecordingAgent:
    server_client = MagicMock(
        spec=ServerClient, global_config_dict={"observability_enabled": True, "policy": MODEL_SERVER}
    )
    server_client._build_server_base_url.return_value = "http://model"
    return _RecordingAgent(
        config=BaseResponsesAPIAgentConfig(host="localhost", port=1, entrypoint="app.py", name="test"),
        server_client=server_client,
    )


def _seed(client: TestClient, session_id: str, agent_instance: str | None, *, rollout_id: str = "r", attempt: int = 0):
    body = AgentSeedSessionRequest(
        agent_session_id=session_id,
        episode_id={"rollout_id": rollout_id, "attempt": attempt},
        task_id={"taskset": "test", "task_id": "task"},
        agent_instance=agent_instance,
    )
    return client.post("/v1/agent_sessions", json=body.model_dump(mode="json"))


def _activate(client: TestClient, prefix: str):
    return client.post(f"{prefix}/v1/responses", json={"input": "go"})


def test_the_same_agent_instance_is_open_in_other_episodes_and_attempts_at_once(agent):
    app = agent.setup_webserver()
    assert _seed(TestClient(app), "s-1", "assistant").status_code == 200

    # The binding is per episode: concurrent episodes on one agent server each seed their own ``assistant``.
    assert _seed(TestClient(app), "s-2", "assistant", rollout_id="other").status_code == 200
    assert _seed(TestClient(app), "s-3", "assistant", attempt=1).status_code == 200
    # Within one of those episodes the name is still taken.
    assert _seed(TestClient(app), "s-4", "assistant", rollout_id="other").status_code == 409


def test_unnamed_sessions_of_one_episode_are_not_bound(agent):
    app = agent.setup_webserver()

    assert _seed(TestClient(app), "s-1", None).status_code == 200
    assert _seed(TestClient(app), "s-2", None).status_code == 200


def test_a_resolved_model_base_url_carries_the_session_s_name_at_seed_and_activation(agent):
    client = TestClient(agent.setup_webserver())
    assert _seed(client, "s-user", "user").status_code == 200

    response = _activate(client, "/ng-rollout/r/ng-agent-instance/user")

    assert response.status_code == 200, response.text
    assert agent._seed_urls == ["http://model/ng-rollout/r/ng-agent-instance/user/v1"]
    # An explicit name wins over the session's.
    assert agent._activations == [
        (
            "user",
            "http://model/ng-rollout/r/ng-agent-instance/user/v1",
            "http://model/ng-rollout/r/ng-agent-instance/judge/v1",
        )
    ]


@pytest.mark.parametrize(
    "seeded, path",
    [
        ("user", "/ng-rollout/r/ng-agent-instance/judge/training-token-capture"),
        ("user", "/ng-rollout/r/training-token-capture"),
        (None, "/ng-rollout/r/ng-agent-instance/user/training-token-capture"),
    ],
)
def test_a_training_tagged_activation_must_name_the_session_s_agent_instance_too(agent, seeded, path):
    client = TestClient(agent.setup_webserver())
    assert _seed(client, "s", seeded).status_code == 200

    response = _activate(client, path)

    assert response.status_code == 409, response.text
    assert agent._activations == []
