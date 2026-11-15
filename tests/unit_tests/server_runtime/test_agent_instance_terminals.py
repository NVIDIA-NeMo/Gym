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
"""An environment server's episode result names each agent instance's terminal from the activations it sent."""

from collections.abc import Callable

import orjson
import pytest
from fastapi import Body
from omegaconf import OmegaConf
from pydantic import BaseModel

from nemo_gym.client.server_client import ServerClient, rollout_path_prefix
from nemo_gym.config.config_types import AggregateMetrics, AggregateMetricsRequest, BaseServerConfig
from nemo_gym.core.episode_context import recording_agent_activations_active
from nemo_gym.core.episode_types import (
    AgentInstanceResult,
    BaseEpisodeRequest,
    BaseEpisodeResponse,
    EpisodeId,
    MaterializedTask,
    TaskId,
)
from nemo_gym.server_runtime.base_environment_server import (
    BaseEnvironmentServer,
    BaseEnvironmentServerConfig,
    CleanupContext,
)


class _TaskInput(BaseModel):
    pass


class _Request(BaseEpisodeRequest[_TaskInput]):
    pass


class _Response(BaseEpisodeResponse[dict]):
    pass


class _Reply:
    """What ``ServerClient`` reads of a served reply: its status and its cached body."""

    def __init__(self, body: bytes, *, ok: bool = True) -> None:
        self.ok = ok
        self._body = body

    async def read(self) -> bytes:
        return self._body


def _served(response_id: object) -> _Reply:
    return _Reply(orjson.dumps({"id": response_id}))


# (server name, path after the rollout prefix's capture key is known, reply)
_Call = tuple[str, Callable[[str], str], _Reply]


def _activation(agent_instance: str | None, *, token_capture: bool = False, query: str = "") -> Callable[[str], str]:
    return lambda key: (
        f"{rollout_path_prefix(key, token_capture=token_capture, agent_instance=agent_instance)}/v1/responses{query}"
    )


class _ScriptedServer(BaseEnvironmentServer[_Request, _Response]):
    """Sends a scripted list of calls, then reports the agent instances its test configured."""

    request_model = _Request
    response_model = _Response
    calls: list = []
    reported: dict[str, AgentInstanceResult] | None = None

    async def run(self, request: _Request, cleanup: CleanupContext) -> _Response:
        for server_name, path, _ in self.calls:
            await self.server_client.post(server_name, path(request.episode_id.capture_key), json={})
        return _Response(
            episode_id=request.episode_id,
            task_id=request.task.task_id,
            result={"reward": 1.0},
            agent_instances=self.reported,
        )

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        return AggregateMetrics()


@pytest.fixture
def run_episode(monkeypatch):
    async def run(calls: list[_Call], reported: dict[str, AgentInstanceResult] | None = None) -> _Response:
        replies = iter([reply for _, _, reply in calls])

        async def transport(method, url, **kwargs):
            return next(replies)

        monkeypatch.setattr("nemo_gym.client.server_client.request", transport)
        client = ServerClient(
            head_server_config=BaseServerConfig(host="head", port=1),
            global_config_dict=OmegaConf.create(
                {"agent": {"responses_api_agents": {"agent": {"host": "agent", "port": 1}}}}
            ),
        )
        config = BaseEnvironmentServerConfig(
            name="environment", host="127.0.0.1", port=1, entrypoint="app.py", cleanup_timeout_seconds=1
        )
        server = _ScriptedServer(config=config, server_client=client)
        server.calls, server.reported = calls, reported
        return await server.run_request(
            _Request(
                episode_id=EpisodeId(rollout_id="r"),
                task=MaterializedTask(task_id=TaskId(taskset="t", task_id="x"), task_input=_TaskInput()),
            )
        )

    return run


async def test_an_activation_carrying_the_training_tag_or_a_query_is_recorded(run_episode):
    response = await run_episode(
        [
            ("agent", _activation("assistant", token_capture=True), _served("resp_a1")),
            ("agent", _activation("user", query="?stream=false"), _served("resp_u1")),
        ]
    )

    assert response.agent_instances == {
        "assistant": AgentInstanceResult(terminal_response_id="resp_a1"),
        "user": AgentInstanceResult(terminal_response_id="resp_u1"),
    }


async def test_a_terminal_and_a_reward_the_environment_set_are_kept(run_episode):
    response = await run_episode(
        [("agent", _activation("assistant"), _served("resp_a1")), ("agent", _activation("user"), _served("resp_u1"))],
        reported={
            "assistant": AgentInstanceResult(terminal_response_id="resp_chosen_by_the_environment", reward=0.5),
            # Reported by the environment and never activated through the client.
            "judge": AgentInstanceResult(reward=0.25),
        },
    )

    assert response.agent_instances == {
        "assistant": AgentInstanceResult(terminal_response_id="resp_chosen_by_the_environment", reward=0.5),
        "judge": AgentInstanceResult(reward=0.25),
        "user": AgentInstanceResult(terminal_response_id="resp_u1"),
    }
    assert response.model_dump(mode="json")["agent_instances"] == {
        "assistant": {"terminal_response_id": "resp_chosen_by_the_environment", "reward": 0.5},
        "judge": {"reward": 0.25},
        "user": {"terminal_response_id": "resp_u1"},
    }


async def test_an_episode_with_only_an_unnamed_activation_reports_no_agent_instances(run_episode):
    response = await run_episode([("agent", _activation(None), _served("resp_1"))])

    assert response.agent_instances is None
    assert "agent_instances" not in response.model_dump(mode="json")
    # Recording ends with the episode.
    assert recording_agent_activations_active() is False
