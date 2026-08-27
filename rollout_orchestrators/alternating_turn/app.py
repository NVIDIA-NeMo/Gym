# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Alternating-turn rollout orchestration for independently hosted agents."""

import asyncio
import json
from time import time
from typing import Any, Optional
from uuid import uuid4

from fastapi import Body, Request
from pydantic import ConfigDict, Field, PrivateAttr, model_validator

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_rollout_orchestrator import BaseRolloutOrchestratorConfig, SimpleRolloutOrchestrator
from nemo_gym.config_types import (
    AgentServerRef,
    AggregateMetrics,
    AggregateMetricScope,
    AggregateMetricsRequest,
    ResourcesServerRef,
)
from nemo_gym.multi_agent import AgentActResponse, AgentTurn, MultiAgentResetResponse, MultiAgentStepResponse
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.reward_profile import compute_aggregate_metrics
from nemo_gym.server_utils import get_response_json, raise_for_status


class AlternatingTurnOrchestratorConfig(BaseRolloutOrchestratorConfig):
    resources_server: ResourcesServerRef
    agents: dict[str, AgentServerRef] = Field(min_length=2)
    focal_agent: str
    max_turns: int = Field(16, ge=1)

    @model_validator(mode="after")
    def validate_focal_agent(self) -> "AlternatingTurnOrchestratorConfig":
        if self.focal_agent not in self.agents:
            raise ValueError(f"focal_agent {self.focal_agent!r} must be present in agents")
        return self


class AlternatingTurnRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class AlternatingTurnRunResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    focal_agent: str
    agent_rewards: dict[str, float]
    agent_trajectories: dict[str, list[AgentTurn]]
    terminated: bool
    truncated: bool
    info: dict[str, Any] = Field(default_factory=dict)


def _episode_response(trajectories: dict[str, list[AgentTurn]]) -> NeMoGymResponse:
    summary = json.dumps(
        {agent_id: [turn.model_dump() for turn in turns] for agent_id, turns in trajectories.items()},
        sort_keys=True,
    )
    return NeMoGymResponse(
        id=f"multi-agent-{uuid4()}",
        created_at=time(),
        model="multi-agent",
        object="response",
        output=[
            NeMoGymResponseOutputMessage(
                id=f"message-{uuid4()}",
                content=[NeMoGymResponseOutputText(annotations=[], text=summary, type="output_text")],
                role="assistant",
                status="completed",
                type="message",
            )
        ],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    )


class AlternatingTurnOrchestrator(SimpleRolloutOrchestrator):
    config: AlternatingTurnOrchestratorConfig
    _episode_lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)

    async def run(self, request: Request, body: AlternatingTurnRunRequest) -> AlternatingTurnRunResponse:
        async with self._episode_lock:
            return await self._run_episode(request, body)

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        overall = compute_aggregate_metrics(
            body.verify_responses,
            compute_metrics_fn=self.compute_metrics,
            get_key_metrics_fn=self.get_key_metrics,
        )
        per_agent_metrics: dict[str, AggregateMetricScope] = {}
        for agent_id in self.config.agents:
            agent_responses = []
            for response in body.verify_responses:
                rewards = response.get("agent_rewards")
                if not isinstance(rewards, dict) or agent_id not in rewards:
                    continue
                agent_responses.append(response | {"reward": rewards[agent_id]})

            if not agent_responses:
                continue
            metrics = compute_aggregate_metrics(
                agent_responses,
                compute_metrics_fn=self.compute_metrics,
                get_key_metrics_fn=self.get_key_metrics,
            )
            per_agent_metrics[agent_id] = AggregateMetricScope(
                group_level_metrics=metrics.group_level_metrics,
                metrics=metrics.agent_metrics,
                key_metrics=metrics.key_metrics,
            )

        return overall.model_copy(update={"per_agent_metrics": per_agent_metrics})

    async def _run_episode(self, request: Request, body: AlternatingTurnRunRequest) -> AlternatingTurnRunResponse:
        trajectories: dict[str, list[AgentTurn]] = {agent_id: [] for agent_id in self.config.agents}
        rewards: dict[str, float] = {agent_id: 0.0 for agent_id in self.config.agents}
        agent_cookies: dict[str, Any] = {agent_id: None for agent_id in self.config.agents}
        env_cookies = request.cookies

        reset_resp = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/reset",
            json=body.model_dump(),
            cookies=env_cookies,
        )
        await raise_for_status(reset_resp)
        reset = MultiAgentResetResponse.model_validate(await get_response_json(reset_resp))
        env_cookies = reset_resp.cookies
        active_agent: Optional[str] = reset.active_agent
        observation: Optional[str] = reset.observation
        terminated = False
        truncated = False
        info = reset.info

        try:
            for _ in range(self.config.max_turns):
                if active_agent not in self.config.agents:
                    raise RuntimeError(f"No agent server configured for active agent {active_agent!r}.")
                if observation is None:
                    raise RuntimeError(f"Environment returned no observation for active agent {active_agent!r}.")

                act_resp = await self.server_client.post(
                    server_name=self.config.agents[active_agent].name,
                    url_path="/act",
                    json={
                        "agent_id": active_agent,
                        "observation": observation,
                        "history": [turn.model_dump() for turn in trajectories[active_agent]],
                    },
                    cookies=agent_cookies[active_agent],
                )
                await raise_for_status(act_resp)
                action = AgentActResponse.model_validate(await get_response_json(act_resp)).action
                agent_cookies[active_agent] = act_resp.cookies
                trajectories[active_agent].append(AgentTurn(observation=observation, action=action))

                step_resp = await self.server_client.post(
                    server_name=self.config.resources_server.name,
                    url_path="/step",
                    json=body.model_dump() | {"agent_id": active_agent, "action": action},
                    cookies=env_cookies,
                )
                await raise_for_status(step_resp)
                step = MultiAgentStepResponse.model_validate(await get_response_json(step_resp))
                env_cookies = step_resp.cookies

                for agent_id, reward in step.rewards.items():
                    rewards[agent_id] = rewards.get(agent_id, 0.0) + reward
                active_agent = step.active_agent
                observation = step.observation
                terminated = step.terminated
                truncated = step.truncated
                info = step.info
                if terminated or truncated:
                    break
            else:
                truncated = True
        finally:
            if not terminated:
                close_resp = await self.server_client.post(
                    server_name=self.config.resources_server.name,
                    url_path="/close",
                    json={},
                    cookies=env_cookies,
                )
                await raise_for_status(close_resp)

        response = _episode_response(trajectories)
        return AlternatingTurnRunResponse(
            responses_create_params=body.responses_create_params,
            response=response,
            reward=rewards[self.config.focal_agent],
            focal_agent=self.config.focal_agent,
            agent_rewards=rewards,
            agent_trajectories=trajectories,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )


if __name__ == "__main__":
    AlternatingTurnOrchestrator.run_webserver()
