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
import logging
import traceback
import uuid
from typing import Any

import verifiers as vf
from fastapi import Body, Request, Response
from openai import AsyncOpenAI
from openai.resources.chat import AsyncChat
from openai.resources.chat.completions import AsyncCompletions
from openai.types.chat.chat_completion import ChatCompletion
from pydantic import ConfigDict, Field
from verifiers.utils.async_utils import maybe_semaphore

from nemo_gym.base_resources_server import BaseRunRequest
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.global_config import get_first_server_config_dict
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputMessageForTraining,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import get_global_aiohttp_client

from resources_servers.verifiers.schemas import (
    VerifiersAgentVerifyResponse,
    VerifiersNeMoGymResponse,
)
from resources_servers.verifiers.utils import load_verifiers_dataset

logger = logging.getLogger(__name__)

class _VLLMChatCompletions(AsyncCompletions):
    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")

    async def create(self, *args: Any, **kwargs: Any) -> ChatCompletion:
        request_body: dict[str, Any] = {
            "model": kwargs.get("model", ""),
            "messages": kwargs.get("messages", []),
        }
        for key in ("temperature", "max_tokens", "max_completion_tokens", "top_p", "stop", "n", "tools", "tool_choice"):
            if key in kwargs and kwargs[key] is not None:
                request_body[key] = kwargs[key]

        url = f"{self._base_url}/chat/completions"
        try:
            session = get_global_aiohttp_client()
            async with session.post(url, json=request_body) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(f"Request to {url} failed with status {resp.status}: {error_text}")
                    resp.raise_for_status()
                response_dict = await resp.json()
        except Exception as e:
            logger.error(f"Exception calling {url}: {type(e).__name__}: {e}")
            raise

        choice_dict = response_dict["choices"][0]
        message_dict = choice_dict.get("message", {})

        prompt_token_ids = message_dict.pop("prompt_token_ids", [])
        generation_token_ids = message_dict.pop("generation_token_ids", [])
        generation_log_probs = message_dict.pop("generation_log_probs", [])

        if not generation_token_ids:
            logger.warning(f"No generation_token_ids in response! Full message keys were: {list(choice_dict.get('message', {}).keys())}")

        if generation_token_ids and isinstance(generation_token_ids[0], str):
            generation_token_ids = [int(tid) for tid in generation_token_ids]

        if generation_token_ids and generation_log_probs:
            choice_dict["logprobs"] = {
                "content": [
                    {"token": f"token_id:{tid}", "logprob": lp, "top_logprobs": []}
                    for tid, lp in zip(generation_token_ids, generation_log_probs)
                ]
            }

        response = ChatCompletion.model_validate(response_dict)
        setattr(response, "prompt_token_ids", prompt_token_ids)
        setattr(response.choices[0], "token_ids", generation_token_ids)
        return response


class _VLLMChat(AsyncChat):
    def __init__(self, base_url: str) -> None:
        self._completions = _VLLMChatCompletions(base_url)

    @property
    def completions(self) -> AsyncCompletions:
        return self._completions


class VLLMOpenAIClient(AsyncOpenAI):
    def __init__(self, base_url: str) -> None:
        super().__init__(api_key="dummy", base_url=base_url)
        self._chat = _VLLMChat(base_url)

    @property
    def chat(self) -> AsyncChat:
        return self._chat


class VerifiersAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    model_name: str = Field(default="", description="Model name for the vLLM server")

    vf_env_id: str = Field(default="", description="Default verifiers environment ID")
    vf_env_args: dict = Field(default_factory=dict, description="Environment arguments")
    dataset_n: int = Field(default=-1, description="Number of examples to load")
    dataset_seed: int | None = Field(default=None, description="Dataset shuffle seed")

    group_size: int = Field(default=1, description="Number of rollouts per example")
    max_concurrent_generation: int = Field(default=-1, description="Max concurrent generation requests")
    max_concurrent_scoring: int = Field(default=-1, description="Max concurrent scoring requests")

    max_tokens: int = Field(default=512, description="Max tokens for generation")
    temperature: float = Field(default=1.0, description="Sampling temperature")
    top_p: float = Field(default=1.0, description="Top-p sampling")


class VerifiersAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    task_idx: int
    vf_env_id: str | None = Field(default=None, description="Override env ID from config")
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming = Field(
        default_factory=lambda: NeMoGymResponseCreateParamsNonStreaming(input=[])
    )
    answer: str = Field(default="", description="Expected answer")
    task: str = Field(default="default", description="Task type")
    example_id: int | str = Field(default=0, description="Example ID")
    info: dict = Field(default_factory=dict, description="Extra info for scoring")


class VerifiersAgent(SimpleResponsesAPIAgent):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    config: VerifiersAgentConfig

    envs_cache: dict[str, vf.Environment] = Field(default_factory=dict)
    env_ids_cache: dict[str, str] = Field(default_factory=dict)
    dataset_rows_cache: dict[str, list[dict]] = Field(default_factory=dict)
    openai_client_cache: dict[str, VLLMOpenAIClient] = Field(default_factory=dict)

    async def _ensure_env_loaded(self, vf_env_id: str) -> tuple[vf.Environment, str, list[dict]]:
        if vf_env_id in self.envs_cache:
            return self.envs_cache[vf_env_id], self.env_ids_cache[vf_env_id], self.dataset_rows_cache[vf_env_id]

        env_id = f"{vf_env_id}-{uuid.uuid4().hex[:8]}"
        logger.info(f"Loading verifiers environment: {vf_env_id}")

        vf_env = vf.load_environment(vf_env_id, **self.config.vf_env_args)
        dataset_rows = load_verifiers_dataset(vf_env, n=self.config.dataset_n, seed=self.config.dataset_seed)

        self.envs_cache[vf_env_id] = vf_env
        self.env_ids_cache[vf_env_id] = env_id
        self.dataset_rows_cache[vf_env_id] = dataset_rows

        return vf_env, env_id, dataset_rows

    def _get_openai_client(self) -> VLLMOpenAIClient:
        cache_key = self.config.model_server.name
        if cache_key not in self.openai_client_cache:
            server_config_dict = get_first_server_config_dict(
                self.server_client.global_config_dict,
                self.config.model_server.name,
            )
            model_server_url = f"http://{server_config_dict.host}:{server_config_dict.port}"

            if not model_server_url.endswith("/v1"):
                model_server_url = model_server_url.rstrip("/") + "/v1"

            self.openai_client_cache[cache_key] = VLLMOpenAIClient(base_url=model_server_url)

        return self.openai_client_cache[cache_key]

    def _convert_trajectory_to_output(self, state: dict) -> list:
        output = []
        trajectory = state.get("trajectory", [])

        for step in trajectory:
            for msg in step.get("prompt", []):
                if isinstance(msg, dict):
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    output.append(NeMoGymEasyInputMessage(role=role, content=content).model_dump())

            tokens = step.get("tokens")
            for msg in step.get("completion", []):
                if isinstance(msg, dict):
                    content = msg.get("content", "")
                    if tokens:
                        output.append(NeMoGymResponseOutputMessageForTraining(
                            id=f"msg_{id(msg)}",
                            content=[NeMoGymResponseOutputText(text=content, annotations=[])],
                            prompt_token_ids=tokens.get("prompt_ids", []),
                            generation_token_ids=tokens.get("completion_ids", []),
                            generation_log_probs=tokens.get("completion_logprobs", []),
                        ).model_dump())
                    else:
                        output.append(NeMoGymResponseOutputMessage(
                            id=f"msg_{id(msg)}",
                            content=[NeMoGymResponseOutputText(text=content, annotations=[])],
                        ).model_dump())

        return output

    async def responses(
        self,
        request: Request,
        response: Response,
        body: VerifiersAgentRunRequest = Body(),
    ) -> VerifiersNeMoGymResponse:
        try:
            vf_env_id = body.vf_env_id or self.config.vf_env_id
            vf_env, env_id, _ = await self._ensure_env_loaded(vf_env_id)

            task_idx = body.task_idx

            prompt_messages = []
            for item in body.responses_create_params.input or []:
                if hasattr(item, 'role') and hasattr(item, 'content'):
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

            client = self._get_openai_client()

            gen_sem = await maybe_semaphore(self.config.max_concurrent_generation)
            score_sem = await maybe_semaphore(self.config.max_concurrent_scoring)

            sampling_args = {
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
            }
            states = await vf_env.run_group(
                group_inputs=[rollout_input],
                client=client,
                model=self.config.model_name,
                gen_sampling_args=sampling_args,
                gen_sem=gen_sem,
                score_sem=score_sem,
            )

            state = states[0]
            reward = state.get("reward", 0.0) or 0.0
            metrics = state.get("metrics", {}) or {}

            output = self._convert_trajectory_to_output(state)

            return VerifiersNeMoGymResponse(
                id=f"verifiers-{env_id}-{task_idx}",
                created_at=0,
                model=self.config.model_name,
                object="response",
                output=output,
                env_id=env_id,
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
