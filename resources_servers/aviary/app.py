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
import json
import uuid
from abc import ABC
from collections import defaultdict
from typing import Generic, TypeVar, cast

from fastapi import FastAPI, Request
from openai.types.responses import FunctionToolParam
from pydantic import ConfigDict, Field

from aviary.core import (
    Environment,
    EnvStateMessage,
    Message,
    TaskDataset,
    Tool,
    ToolCall,
    ToolCallFunction,
    ToolRequestMessage,
    ToolResponseMessage,
)
from nemo_gym.base_resources_server import SimpleResourcesServer
from nemo_gym.integrations.aviary import (
    AviaryAgentVerifyRequest,
    AviaryAgentVerifyResponse,
    AviaryCloseRequest,
    AviaryCloseResponse,
    AviaryEnvStateEasyInputMessage,
    AviaryResourcesServerConfig,
    AviarySeedSessionRequest,
    AviarySeedSessionResponse,
    AviaryStepRequest,
    AviaryStepResponse,
)
from nemo_gym.openai_utils import NeMoGymEasyInputMessage, NeMoGymFunctionCallOutput


TEnv = TypeVar("TEnv", bound=Environment)
TDataset = TypeVar("TDataset", bound=TaskDataset)


def tool_to_function_tool_param(tool: Tool) -> FunctionToolParam:
    tool_dump = tool.info.model_dump()
    tool_dump["parameters"].setdefault("additionalProperties", False)
    return FunctionToolParam(type="function", strict=True, **tool_dump)


def obs_msg_to_nemo_gym(obs: Message) -> NeMoGymEasyInputMessage:
    dump = obs.model_dump()
    if isinstance(dump["content"], list):
        type_remap = {k: f"input_{k}" for k in ("text", "image", "file", "audio")}

        def fix_content(c: dict) -> dict:
            if c["type"] == "image_url":
                return {
                    "type": "input_image",
                    "file_id": None,
                    "detail": "auto",
                    "image_url": c["image_url"]["url"],
                }
            else:
                return {**c, "type": type_remap.get(c["type"], c["type"])}

        dump["content"] = [fix_content(c) for c in dump["content"]]
    message_cls = AviaryEnvStateEasyInputMessage if isinstance(obs, EnvStateMessage) else NeMoGymEasyInputMessage
    return message_cls.model_validate(dump)


class AviaryResourcesServer(SimpleResourcesServer, Generic[TEnv, TDataset], ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: AviaryResourcesServerConfig
    dataset: TDataset
    env_id_to_env: dict[str, TEnv] = Field(default_factory=dict)
    env_id_to_total_reward: dict[str, float] = Field(default_factory=lambda: defaultdict(float))

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/step")(self.step)
        app.post("/close")(self.close)
        return app

    async def seed_session(self, request: Request, body: AviarySeedSessionRequest) -> AviarySeedSessionResponse:
        """
        Wraps creation of the Aviary environment and calling reset().
        """
        env_id = str(uuid.uuid4())
        env = cast(Environment, self.dataset.get_new_env_by_idx(body.task_idx))
        self.env_id_to_env[env_id] = env

        obs, tools = await env.reset()
        return AviarySeedSessionResponse(
            env_id=env_id,
            obs=[obs_msg_to_nemo_gym(o) for o in obs],
            tools=[tool_to_function_tool_param(t) for t in tools],
        )

    async def step(self, request: Request, body: AviaryStepRequest) -> AviaryStepResponse:
        """
        Wraps calling step().
        """
        env = self.env_id_to_env[body.env_id]

        action = ToolRequestMessage(
            content=None,
            tool_calls=[
                ToolCall(id=a.call_id, function=ToolCallFunction(name=a.name, arguments=json.loads(a.arguments)))
                for a in body.action
            ],
        )
        obs, reward, done, _ = await env.step(action)

        self.env_id_to_total_reward[body.env_id] += reward

        nemo_obs = [
            NeMoGymFunctionCallOutput(call_id=o.tool_call_id, output=o.content)
            if isinstance(o, ToolResponseMessage)
            else obs_msg_to_nemo_gym(o)
            for o in obs
        ]

        return AviaryStepResponse(obs=nemo_obs, reward=reward, done=done)

    async def verify(self, request: Request, body: AviaryAgentVerifyRequest) -> AviaryAgentVerifyResponse:
        return AviaryAgentVerifyResponse(**body.model_dump(), reward=self.env_id_to_total_reward[body.response.env_id])

    async def close(self, request: Request, body: AviaryCloseRequest) -> AviaryCloseResponse:
        """
        Closes and deregisters body.env_id.
        """
        try:
            await self.env_id_to_env.pop(body.env_id).close()
        except Exception as e:
            return AviaryCloseResponse(message=repr(e), success=False)
        return AviaryCloseResponse(message="Success", success=True)
