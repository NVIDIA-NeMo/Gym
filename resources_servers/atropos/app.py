# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import uuid
from abc import ABC
from collections import defaultdict
from typing import Any, Dict, Generic, Optional, TypeVar

from fastapi import FastAPI, Request
from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import SimpleResourcesServer
from nemo_gym.integrations.atropos import (
    AtroposAgentVerifyRequest,
    AtroposAgentVerifyResponse,
    AtroposCloseRequest,
    AtroposCloseResponse,
    AtroposResourcesServerConfig,
    AtroposSeedSessionRequest,
    AtroposSeedSessionResponse,
    AtroposStepRequest,
    AtroposStepResponse,
)
from nemo_gym.openai_utils import NeMoGymEasyInputMessage

TEnv = TypeVar("TEnv")


class AtroposEnvironmentState:
    def __init__(self, env: Any, item: Any):
        self.env = env
        self.item = item
        self.messages = []
        self.total_reward = 0.0
        self.done = False
        self.info = {}


class AtroposResourcesServer(SimpleResourcesServer, Generic[TEnv], ABC):
    # to be extended or overridden in subclasses
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: AtroposResourcesServerConfig
    env_id_to_state: Dict[str, AtroposEnvironmentState] = Field(default_factory=dict)
    env_id_to_total_reward: Dict[str, float] = Field(default_factory=lambda: defaultdict(float))

    system_prompt: Optional[str] = None
    item_message_field: Optional[str] = "question"

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/step")(self.step)
        app.post("/close")(self.close)
        return app

    async def env_factory(self, task_idx: int) -> TEnv:
        raise NotImplementedError("Subclasses must implement env_factory()")

    async def get_initial_item(self, env: TEnv, task_idx: int) -> Any:
        if hasattr(env, 'train'):
            return env.train[task_idx % len(env.train)]
        elif hasattr(env, 'get_next_item'):
            return await env.get_next_item()
        else:
            raise NotImplementedError(
                f"Environment {type(env).__name__} has no 'train' dataset or 'get_next_item' method. "
                "Override get_initial_item() in your server."
            )

    async def format_item_as_message(self, item: Any) -> str:
        if self.item_message_field and isinstance(item, dict) and self.item_message_field in item:
            return item[self.item_message_field]
        return str(item)

    async def score_response(
        self,
        env: TEnv,
        item: Any,
        response: str,
        messages: list,
    ) -> tuple[float, bool, Optional[dict]]:
        return 0.0, True, {"message": "Default scoring - override score_response()"}

    async def seed_session(
        self,
        request: Request,
        body: AtroposSeedSessionRequest,
    ) -> AtroposSeedSessionResponse:
        env_id = str(uuid.uuid4())
        env = await self.env_factory(body.task_idx)
        item = await self.get_initial_item(env, body.task_idx)
        user_message_content = await self.format_item_as_message(item)

        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_message_content})

        state = AtroposEnvironmentState(env=env, item=item)
        state.messages = messages
        self.env_id_to_state[env_id] = state

        obs = [NeMoGymEasyInputMessage.model_validate(msg) for msg in messages]

        return AtroposSeedSessionResponse(
            env_id=env_id,
            obs=obs,
            system_prompt=self.system_prompt,
            metadata={"task_idx": body.task_idx},
        )

    async def step(
        self,
        request: Request,
        body: AtroposStepRequest,
    ) -> AtroposStepResponse:
        state = self.env_id_to_state[body.env_id]
        state.messages.append({"role": "assistant", "content": body.action})

        reward, done, info = await self.score_response(
            env=state.env,
            item=state.item,
            response=body.action,
            messages=state.messages,
        )

        state.total_reward += reward
        state.done = done
        state.info = info or {}
        self.env_id_to_total_reward[body.env_id] = state.total_reward

        return AtroposStepResponse(obs=[], reward=reward, done=done, info=info)

    async def verify(
        self,
        request: Request,
        body: AtroposAgentVerifyRequest,
    ) -> AtroposAgentVerifyResponse:
        env_id = body.response.env_id
        state = self.env_id_to_state.get(env_id)
        reward = self.env_id_to_total_reward[env_id]
        info = state.info if state else {}
        return AtroposAgentVerifyResponse(**body.model_dump(), reward=reward, **info)

    async def close(
        self,
        request: Request,
        body: AtroposCloseRequest,
    ) -> AtroposCloseResponse:
        state = self.env_id_to_state.pop(body.env_id, None)
        if state and hasattr(state.env, "close"):
            await state.env.close()
        self.env_id_to_total_reward.pop(body.env_id, None)
        return AtroposCloseResponse(message="Success", success=True)
