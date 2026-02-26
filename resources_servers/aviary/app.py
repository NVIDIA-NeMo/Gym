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
import logging
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
from nemo_gym.openai_utils import NeMoGymEasyInputMessage, NeMoGymFunctionCallOutput

from .schemas import (
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


logger = logging.getLogger(__name__)


TEnv = TypeVar("TEnv", bound=Environment)
TDataset = TypeVar("TDataset", bound=TaskDataset)


def tool_to_function_tool_param(tool: Tool) -> FunctionToolParam:
    tool_dump = tool.info.model_dump()
    tool_dump["parameters"].setdefault("additionalProperties", False)
    return FunctionToolParam(type="function", strict=True, **tool_dump)


def obs_msg_to_nemo_gym(obs: Message) -> list[NeMoGymEasyInputMessage]:
    # This does some Qwen3-specific things:
    # 1. if content is a JSON list, we flatten it to a list of messages. Qwen3's
    #    chat template doesn't support messages with list contents
    # 2. if content contains images (or really any other media), we drop it for now.
    # Most of this is what we'd call a HACK.

    is_env_state = isinstance(obs, EnvStateMessage) or (obs.info or {}).get("is_env_state", False)

    dump = obs.model_dump()
    try:
        content: str | list = json.loads(dump["content"])
    except json.JSONDecodeError:
        content = dump["content"]

    flat_content: list[str] = []
    if isinstance(content, list):
        flat_content = [c["text"] for c in content if c["type"] == "text"]
    else:
        flat_content = [content]

    message_cls = AviaryEnvStateEasyInputMessage if is_env_state else NeMoGymEasyInputMessage
    return [message_cls.model_validate(dump | {"content": c}) for c in flat_content]


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
            obs=[message for o in obs for message in obs_msg_to_nemo_gym(o)],
            tools=[tool_to_function_tool_param(t) for t in tools],
        )

    async def step(self, request: Request, body: AviaryStepRequest) -> AviaryStepResponse:
        """
        Wraps calling step().
        """
        try:
            env = self.env_id_to_env[body.env_id]

            try:
                tool_calls = [
                    ToolCall(id=a.call_id, function=ToolCallFunction(name=a.name, arguments=json.loads(a.arguments)))
                    for a in body.action
                ]
            except json.decoder.JSONDecodeError as e:
                logger.warning("Failed to parse tool call arguments as JSON: %s", e)
                return AviaryStepResponse(
                    obs=[NeMoGymEasyInputMessage(role="user", content=f"Invalid tool call arguments (not valid JSON): {e}")],
                    reward=0.0,
                    done=True,
                )

            action = ToolRequestMessage(content=None, tool_calls=tool_calls)
            obs, reward, done, _ = await env.step(action)

            self.env_id_to_total_reward[body.env_id] += reward

            nemo_obs = [
                message
                for o in obs
                for message in (
                    [NeMoGymFunctionCallOutput(call_id=o.tool_call_id, output=o.content)]
                    if isinstance(o, ToolResponseMessage)
                    else obs_msg_to_nemo_gym(o)
                )
            ]
        except Exception:
            logger.exception("Error in step")
            raise

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
