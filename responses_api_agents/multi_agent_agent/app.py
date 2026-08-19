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

"""Generic alternating-turn orchestrator with keyboard-controlled agents."""

import asyncio
import json
from time import time
from typing import Any, Literal, Optional, Protocol
from uuid import uuid4

from fastapi import Body, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.config_types import ResourcesServerRef
from nemo_gym.multi_agent import MultiAgentResetResponse, MultiAgentStepResponse
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import get_response_json, raise_for_status


class AgentTurn(BaseModel):
    observation: str
    action: str


class AgentController(Protocol):
    async def act(self, agent_id: str, observation: str, history: list[AgentTurn]) -> str: ...


class KeyboardControllerConfig(BaseModel):
    type: Literal["keyboard"] = "keyboard"
    display_name: Optional[str] = None


class KeyboardController:
    def __init__(self, display_name: str) -> None:
        self.display_name = display_name

    async def act(self, agent_id: str, observation: str, history: list[AgentTurn]) -> str:
        prompt = f"\n--- {self.display_name} ({agent_id}) ---\n{observation}\n{self.display_name}> "
        return await asyncio.to_thread(input, prompt)


class MultiAgentAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    controllers: dict[str, KeyboardControllerConfig]
    focal_agent: str
    max_turns: int = Field(16, ge=1)


class MultiAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class MultiAgentRunResponse(BaseVerifyResponse):
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
        id=f"keyboard-{uuid4()}",
        created_at=time(),
        model="keyboard",
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


class MultiAgentAgent(SimpleResponsesAPIAgent):
    config: MultiAgentAgentConfig
    _interactive_lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)

    async def responses(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        raise HTTPException(status_code=405, detail="Use /run for multi-agent episodes.")

    def _controllers(self) -> dict[str, AgentController]:
        return {
            agent_id: KeyboardController(controller.display_name or agent_id)
            for agent_id, controller in self.config.controllers.items()
        }

    async def run(self, request: Request, body: MultiAgentRunRequest) -> MultiAgentRunResponse:
        async with self._interactive_lock:
            return await self._run_episode(request, body)

    async def _run_episode(self, request: Request, body: MultiAgentRunRequest) -> MultiAgentRunResponse:
        controllers = self._controllers()
        trajectories: dict[str, list[AgentTurn]] = {agent_id: [] for agent_id in controllers}
        rewards: dict[str, float] = {agent_id: 0.0 for agent_id in controllers}
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
                if active_agent not in controllers:
                    raise RuntimeError(f"No controller configured for active agent {active_agent!r}.")
                if observation is None:
                    raise RuntimeError(f"Environment returned no observation for active agent {active_agent!r}.")

                action = await controllers[active_agent].act(
                    active_agent,
                    observation,
                    trajectories[active_agent],
                )
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

        if self.config.focal_agent not in rewards:
            raise RuntimeError(f"Focal agent {self.config.focal_agent!r} is not configured.")

        response = _episode_response(trajectories)
        return MultiAgentRunResponse(
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
    MultiAgentAgent.run_webserver()
