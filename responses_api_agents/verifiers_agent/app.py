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

from __future__ import annotations

import json
import logging
import traceback
from typing import Any

import verifiers as vf
from fastapi import Body, Request, Response
from openai import AsyncOpenAI
from pydantic import ConfigDict, Field
from verifiers.clients import NeMoRLChatCompletionsClient

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef
from nemo_gym.global_config import get_first_server_config_dict
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseFunctionToolCallForTraining,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputMessageForTraining,
    NeMoGymResponseOutputText,
)


logger = logging.getLogger(__name__)


class VerifiersNeMoGymResponse(NeMoGymResponse):
    env_id: str
    group_id: str
    output: list[dict[str, Any]]
    reward: float
    metrics: dict[str, Any] = Field(default_factory=dict)
    parallel_tool_calls: bool = True
    tool_choice: str = "auto"
    tools: list = Field(default_factory=list)


class VerifiersAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    response: VerifiersNeMoGymResponse
    reward: float


class VerifiersAgentConfig(BaseResponsesAPIAgentConfig):
    model_server: ModelServerRef
    model_name: str = Field(default="", description="Model name")

    vf_env_id: str = Field(default="", description="Verifiers environment ID")
    vf_env_args: dict = Field(default_factory=dict, description="Verifiers environment arguments")

    max_tokens: int = Field(default=8192, description="Max tokens for generation")

    # nemo rl generation_config overrides these
    temperature: float = Field(default=1.0)
    top_p: float = Field(default=1.0)


class VerifiersAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    task_idx: int
    vf_env_id: str | None = Field(default=None, description="Verifiers environment ID")
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming = Field(
        default_factory=lambda: NeMoGymResponseCreateParamsNonStreaming(input=[])
    )
    answer: str = Field(default="", description="Expected answer from dataset")
    task: str = Field(default="default", description="Task type from dataset")
    example_id: int | str = Field(default=0, description="Example ID from dataset")
    info: dict = Field(default_factory=dict, description="Extra info from dataset")


class VerifiersAgent(SimpleResponsesAPIAgent):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    config: VerifiersAgentConfig

    envs_cache: dict[str, Any] = Field(default_factory=dict)
    client_cache: dict[str, NeMoRLChatCompletionsClient] = Field(default_factory=dict)

    def _get_env(self, vf_env_id: str) -> vf.Environment:
        if vf_env_id not in self.envs_cache:
            self.envs_cache[vf_env_id] = vf.load_environment(vf_env_id, **self.config.vf_env_args)
        return self.envs_cache[vf_env_id]

    def _get_client(self) -> NeMoRLChatCompletionsClient:
        cache_key = self.config.model_server.name
        if cache_key not in self.client_cache:
            server_config_dict = get_first_server_config_dict(
                self.server_client.global_config_dict,
                self.config.model_server.name,
            )
            model_server_url = f"http://{server_config_dict.host}:{server_config_dict.port}"

            if not model_server_url.endswith("/v1"):
                model_server_url = model_server_url.rstrip("/") + "/v1"

            openai_client = AsyncOpenAI(
                base_url=model_server_url,
                api_key="EMPTY",  # pragma: allowlist secret
            )
            self.client_cache[cache_key] = NeMoRLChatCompletionsClient(openai_client)

        return self.client_cache[cache_key]

    def _convert_trajectory_to_output(self, rollout_output: dict) -> list:
        def as_dict(m):
            return m if isinstance(m, dict) else {
                k: getattr(m, k, None) for k in ("role", "content", "tool_calls", "tool_call_id", "tokens")
            }

        def text(c):
            return c if isinstance(c, str) else ("" if c is None else json.dumps(c, default=str))

        def tok_kwargs(t):
            return {} if not t else {
                "prompt_token_ids": t.get("prompt_ids", []),
                "generation_token_ids": t.get("completion_ids", []),
                "generation_log_probs": t.get("completion_logprobs", []),
            }

        # One token bundle per assistant turn, in trajectory order.
        a_tokens: list[dict | None] = []
        for step in rollout_output.get("trajectory") or []:
            if not isinstance(step, dict):
                continue
            st = step.get("tokens")
            for m in step.get("completion") or []:
                if as_dict(m).get("role") == "assistant":
                    a_tokens.append(st or as_dict(m).get("tokens"))

        output: list[dict] = []
        ai = 0
        for m in rollout_output.get("completion") or []:
            msg = as_dict(m)
            role = msg.get("role", "user")

            if role == "tool":
                cid = msg.get("tool_call_id") or f"call_{id(m)}"
                output.append(NeMoGymFunctionCallOutput(
                    call_id=cid, id=cid, output=text(msg.get("content")), status="completed",
                ).model_dump())
                continue

            if role == "assistant":
                tokens = a_tokens[ai] if ai < len(a_tokens) else None
                ai += 1
                calls = msg.get("tool_calls") or []
                body = text(msg.get("content"))

                if body:
                    cls = NeMoGymResponseOutputMessageForTraining if (tokens and not calls) else NeMoGymResponseOutputMessage
                    output.append(cls(
                        id=f"msg_{id(m)}",
                        content=[NeMoGymResponseOutputText(text=body, annotations=[])],
                        **(tok_kwargs(tokens) if not calls else {}),
                    ).model_dump())

                for i, tc in enumerate(calls):
                    if isinstance(tc, str):
                        try:
                            tc = json.loads(tc)
                        except json.JSONDecodeError:
                            tc = {"arguments": tc}
                    fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                    cid = tc.get("id") or tc.get("call_id") or f"call_{id(tc)}"
                    t = tokens if i == len(calls) - 1 else None
                    cls = NeMoGymResponseFunctionToolCallForTraining if t else NeMoGymResponseFunctionToolCall
                    output.append(cls(
                        id=cid, call_id=cid,
                        name=tc.get("name") or fn.get("name", ""),
                        arguments=text(tc.get("arguments") or fn.get("arguments") or "{}"),
                        status="completed", **tok_kwargs(t),
                    ).model_dump())

                if not body and not calls:
                    output.append(NeMoGymResponseOutputMessage(
                        id=f"msg_{id(m)}",
                        content=[NeMoGymResponseOutputText(text="", annotations=[])],
                    ).model_dump())
                continue

            output.append(NeMoGymEasyInputMessage(role=role, content=text(msg.get("content"))).model_dump())

        return output

    async def responses(
        self,
        request: Request,
        response: Response,
        body: VerifiersAgentRunRequest = Body(),
    ) -> VerifiersNeMoGymResponse:
        try:
            vf_env_id = body.vf_env_id or self.config.vf_env_id
            if not vf_env_id:
                raise ValueError("vf_env_id must be set on the request or in the agent config")
            vf_env = self._get_env(vf_env_id)
            task_idx = body.task_idx

            prompt_messages = []
            for item in body.responses_create_params.input or []:
                if hasattr(item, "role") and hasattr(item, "content"):
                    prompt_messages.append({"role": item.role, "content": item.content})
                elif isinstance(item, dict):
                    prompt_messages.append({"role": item.get("role", "user"), "content": item.get("content", "")})

            rollout_input = vf.RolloutInput(
                prompt=prompt_messages,
                answer=body.answer,
                task=body.task,
                info=body.info,
                example_id=body.example_id,
            )

            client = self._get_client()

            # prefer NeMo RL generation config set in responses_create_params
            # https://github.com/NVIDIA-NeMo/RL/blob/main/nemo_rl/experience/rollouts.py#L1045-L1046
            sampling_args = {
                "max_tokens": self.config.max_tokens,
                "temperature": getattr(body.responses_create_params, "temperature", None) or self.config.temperature,
                "top_p": getattr(body.responses_create_params, "top_p", None) or self.config.top_p,
            }
            outputs = await vf_env.run_group(
                group_inputs=[rollout_input],
                client=client,
                model=self.config.model_name,
                sampling_args=sampling_args,
                state_columns=["trajectory"],
            )

            rollout_output = outputs[0]
            reward = rollout_output.get("reward", 0.0) or 0.0
            metrics = rollout_output.get("metrics", {}) or {}

            output = self._convert_trajectory_to_output(rollout_output)

            return VerifiersNeMoGymResponse(
                id=f"verifiers-{vf_env_id}-{task_idx}",
                created_at=0,
                model=self.config.model_name,
                object="response",
                output=output,
                env_id=vf_env_id,
                group_id=str(task_idx),
                reward=reward,
                metrics=metrics,
            )
        except Exception as e:
            logger.error(f"Exception in responses(): {type(e).__name__}: {e}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            raise

    async def run(
        self,
        request: Request,
        response: Response,
        body: VerifiersAgentRunRequest = Body(),
    ) -> VerifiersAgentVerifyResponse:
        resp = await self.responses(request, response, body)

        return VerifiersAgentVerifyResponse(
            responses_create_params=body.responses_create_params,
            response=resp,
            reward=resp.reward,
        )


if __name__ == "__main__":
    VerifiersAgent.run_webserver()
