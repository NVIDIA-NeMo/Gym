# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
from contextlib import asynccontextmanager
from time import time
from typing import Any

import verifiers.v1 as vf
from fastapi import Body, FastAPI, HTTPException
from pydantic import ConfigDict, Field, PrivateAttr
from verifiers.v1.dialects.chat import message_to_wire

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.global_config import POLICY_MODEL_NAME_KEY_NAME
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.responses_converter import ResponsesConverter


class VerifiersAgentConfig(BaseResponsesAPIAgentConfig):
    model_server: ModelServerRef
    verifiers: vf.SingleAgentEnvConfig
    max_tokens: int = 8192
    temperature: float = 1.0
    top_p: float = 1.0


class VerifiersAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    task_idx: int = Field(ge=0)


class VerifiersNeMoGymResponse(NeMoGymResponse):
    reward: float
    metrics: dict[str, float] = Field(default_factory=dict)


class VerifiersAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    response: VerifiersNeMoGymResponse
    metrics: dict[str, float] = Field(default_factory=dict)


class VerifiersAgent(SimpleResponsesAPIAgent):
    config: VerifiersAgentConfig
    _env: vf.SingleAgentEnv = PrivateAttr()
    _tasks: list[vf.Task] = PrivateAttr()
    _converter: ResponsesConverter = PrivateAttr()

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self._env = vf.SingleAgentEnv(self.config.verifiers)
        self._tasks = list(self._env.taskset)
        self._converter = ResponsesConverter(return_token_id_information=True)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        # A V1 rollout needs task_idx, which is part of Gym's /run request.
        app.router.routes[:] = [
            route for route in app.router.routes if not getattr(route, "path", "").endswith("/v1/responses")
        ]
        parent_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            async with parent_lifespan(app), self._env.serving():
                yield

        app.router.lifespan_context = lifespan
        return app

    async def responses(
        self,
        body: VerifiersAgentRunRequest = Body(),
    ) -> VerifiersNeMoGymResponse:
        params = body.responses_create_params
        try:
            task = self._tasks[body.task_idx]
        except IndexError:
            raise HTTPException(
                status_code=422,
                detail=f"task_idx {body.task_idx} is out of range for {len(self._tasks)} tasks",
            ) from None

        sampling = json.loads((params.metadata or {}).get("extra_body", "{}"))
        sampling.update(
            max_tokens=params.max_output_tokens if params.max_output_tokens is not None else self.config.max_tokens,
            temperature=params.temperature if params.temperature is not None else self.config.temperature,
            top_p=params.top_p if params.top_p is not None else self.config.top_p,
        )
        model = str(
            self.server_client.global_config_dict.get(POLICY_MODEL_NAME_KEY_NAME)
            or params.model
            or self.config.model_server.name
        )
        context = vf.ModelContext(
            model=model,
            client=vf.EvalClientConfig(
                base_url=self.resolve_model_base_url(
                    self.config.model_server.name,
                    self.rollout_id_from_run(body),
                ),
                api_key_var="NEMO_GYM_API_KEY",
            ),
            sampling=vf.SamplingConfig.model_validate(sampling),
        )
        episode = await self._env.run_slot(self._env.slots(task)[0], context)
        if not episode.ok:
            error = episode.last_error or (episode.traces[-1].last_error if episode.traces else None)
            raise RuntimeError(error.message if error else "Verifiers rollout failed")

        trace = episode.traces[0]
        branch = trace.branches[-1]
        calls = iter(branch.calls)
        input_items, output = [], []
        items = input_items
        for node in branch.nodes:
            message = node.message
            if node.sampled:
                items = output
                call = next(calls, None)
                if call and call.endpoint == "/responses" and message.provider_state:
                    items.extend(message.provider_state)
                    continue
            wire = message_to_wire(message)
            if isinstance(message, vf.AssistantMessage) and message.reasoning_content:
                wire["content"] = f"<think>{message.reasoning_content}</think>{message.content or ''}"
            items.extend(self._converter.chat_completions_messages_to_responses_items([wire]))

        tools = [
            {
                "type": "function",
                **tool.model_dump(exclude_none=True),
                "strict": bool(tool.strict),
            }
            for tool in trace.tools or []
        ]
        params = params.model_copy(update={"input": input_items, "instructions": None, "tools": tools})
        body.responses_create_params = params

        return VerifiersNeMoGymResponse(
            id=f"resp_{trace.id}",
            created_at=int(time()),
            model=next(
                (call.model for call in reversed(trace.calls) if call.model),
                model,
            ),
            object="response",
            output=output,
            parallel_tool_calls=params.parallel_tool_calls,
            tool_choice=params.tool_choice or "auto",
            tools=tools,
            reward=trace.reward,
            metrics={name: value for name, value in trace.metrics.items() if value is not None},
        )

    async def run(
        self,
        body: VerifiersAgentRunRequest = Body(),
    ) -> VerifiersAgentVerifyResponse:
        result = await self.responses(body)
        return VerifiersAgentVerifyResponse(
            responses_create_params=body.responses_create_params,
            response=result,
            reward=result.reward,
            metrics=result.metrics,
            **{f"vf/{name}": value for name, value in result.metrics.items()},
        )


if __name__ == "__main__":
    VerifiersAgent.run_webserver()
