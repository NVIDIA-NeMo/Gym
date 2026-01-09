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
import logging
from typing import Any

import aiohttp
import verifiers as vf
from openai import AsyncOpenAI
from openai.resources.chat import AsyncChat
from openai.resources.chat.completions import AsyncCompletions
from openai.types.chat.chat_completion import ChatCompletion
from pydantic import ConfigDict, Field
from verifiers.utils.async_utils import maybe_semaphore

from nemo_gym.base_resources_server import BaseRunRequest
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming

from resources_servers.verifiers.schemas import (
    VerifiersAgentVerifyRequest,
    VerifiersAgentVerifyResponse,
    VerifiersNeMoGymResponse,
    VerifiersSeedSessionResponse,
)


logger = logging.getLogger(__name__)


class _VLLMChatCompletions(AsyncCompletions):
    """Wraps vllm_model and injects token IDs as attributes for verifiers."""
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

        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self._base_url}/chat/completions", json=request_body) as resp:
                resp.raise_for_status()
                response_dict = await resp.json()

        # Extract token IDs from vllm_model format 
        choice_dict = response_dict["choices"][0]
        message_dict = choice_dict.get("message", {})
        prompt_token_ids = message_dict.pop("prompt_token_ids", [])
        generation_token_ids = message_dict.pop("generation_token_ids", [])
        generation_log_probs = message_dict.pop("generation_log_probs", [])

        # Reconstruct logprobs.content for verifiers
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
    """OpenAI-compatible client wrapping vllm_model."""
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


class VerifiersAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    task_idx: int
    vf_env_id: str | None = Field(default=None, description="Override env ID from config")
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming = Field(
        default_factory=lambda: NeMoGymResponseCreateParamsNonStreaming(input=[])
    )


class VerifiersAgent(SimpleResponsesAPIAgent):
    """Uses vf_env.run_group() with an AsyncOpenAI client pointing to the vLLM model server."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    config: VerifiersAgentConfig

    _envs: dict[str, vf.Environment] = {}
    _env_ids: dict[str, str] = {}
    _dataset_rows_cache: dict[str, list[dict]] = {}
    _openai_client: VLLMOpenAIClient | None = None

    async def _ensure_env_loaded(self, vf_env_id: str) -> tuple[vf.Environment, str, list[dict]]:
        if vf_env_id in self._envs:
            return self._envs[vf_env_id], self._env_ids[vf_env_id], self._dataset_rows_cache[vf_env_id]

        response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json={
                "vf_env_id": vf_env_id,
                "vf_env_args": self.config.vf_env_args,
                "dataset_n": self.config.dataset_n,
                "dataset_seed": self.config.dataset_seed,
            },
        )
        response.raise_for_status()
        seed_response = VerifiersSeedSessionResponse.model_validate(await response.json())

        env_id = seed_response.env_id
        logger.info(f"Seeded verifiers environment: {seed_response.vf_env_id} with {seed_response.dataset_length} examples")

        vf_env = vf.load_environment(vf_env_id, **self.config.vf_env_args)

        # Try get_dataset first, fall back to eval_dataset/train_dataset for some envs
        # TODO: is there more standard way in verifiers.. check prime rl
        try:
            dataset = vf_env.get_dataset(n=self.config.dataset_n, seed=self.config.dataset_seed)
        except ValueError:
            dataset = None
            for attr in ['dataset', 'train_dataset', 'eval_dataset']:
                ds = getattr(vf_env, attr, None)
                if ds is not None:
                    dataset = ds
                    logger.info(f"Found dataset in vf_env.{attr}")
                    break
            if dataset is None:
                raise ValueError(f"Environment {vf_env_id} does not have a dataset")
            if self.config.dataset_seed is not None:
                dataset = dataset.shuffle(seed=self.config.dataset_seed)
            if self.config.dataset_n > 0:
                dataset = dataset.select(range(min(self.config.dataset_n, len(dataset))))

        dataset_rows = [
            {
                "prompt": dataset["prompt"][i],
                "example_id": dataset["example_id"][i],
                "task": dataset["task"][i],
                **({"answer": dataset["answer"][i]} if "answer" in dataset.column_names else {}),
                **({"info": dataset["info"][i]} if "info" in dataset.column_names else {}),
            }
            for i in range(len(dataset))
        ]

        self._envs[vf_env_id] = vf_env
        self._env_ids[vf_env_id] = env_id
        self._dataset_rows_cache[vf_env_id] = dataset_rows

        return vf_env, env_id, dataset_rows

    def _get_openai_client(self) -> VLLMOpenAIClient:
        if self._openai_client is None:
            from nemo_gym.global_config import get_first_server_config_dict

            server_config_dict = get_first_server_config_dict(
                self.server_client.global_config_dict,
                self.config.model_server.name,
            )
            model_server_url = f"http://{server_config_dict.host}:{server_config_dict.port}"

            if not model_server_url.endswith("/v1"):
                model_server_url = model_server_url.rstrip("/") + "/v1"

            self._openai_client = VLLMOpenAIClient(base_url=model_server_url)
            logger.info(f"Created VLLMOpenAIClient pointing to: {model_server_url}")

        return self._openai_client

    def _convert_trajectory_to_output(self, state: dict) -> list:
        from nemo_gym.openai_utils import (
            NeMoGymEasyInputMessage,
            NeMoGymResponseOutputMessage,
            NeMoGymResponseOutputText,
            NeMoGymEasyInputMessageForTraining,
            NeMoGymResponseOutputMessageForTraining,
        )

        output = []
        trajectory = state.get("trajectory", [])

        for step in trajectory:
            step_output = []

            for msg in step.get("prompt", []):
                if isinstance(msg, dict):
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    step_output.append(NeMoGymEasyInputMessage(role=role, content=content))

            tokens = step.get("tokens")
            for msg in step.get("completion", []):
                if isinstance(msg, dict):
                    content = msg.get("content", "")
                    if tokens:
                        step_output.append(NeMoGymResponseOutputMessageForTraining(
                            id=f"msg_{id(msg)}",
                            content=[NeMoGymResponseOutputText(text=content, annotations=[])],
                            prompt_token_ids=tokens.get("prompt_ids", []),
                            generation_token_ids=tokens.get("completion_ids", []),
                            generation_log_probs=tokens.get("completion_logprobs", []),
                        ))
                    else:
                        step_output.append(NeMoGymResponseOutputMessage(
                            id=f"msg_{id(msg)}",
                            content=[NeMoGymResponseOutputText(text=content, annotations=[])],
                        ))

            output.append(step_output)

        return output

    async def responses(self, req: VerifiersAgentRunRequest) -> VerifiersNeMoGymResponse:
        # Use env_id from request if provided, else fall back to config
        vf_env_id = req.vf_env_id or self.config.vf_env_id
        vf_env, env_id, dataset_rows = await self._ensure_env_loaded(vf_env_id)

        task_idx = req.task_idx
        row = dataset_rows[task_idx]

        rollout_input = vf.RolloutInput(
            prompt=row["prompt"],
            answer=row.get("answer", ""),
            task=row["task"],
            info=row.get("info", {}),
            example_id=row["example_id"],
        )

        client = self._get_openai_client()

        gen_sem = await maybe_semaphore(self.config.max_concurrent_generation)
        score_sem = await maybe_semaphore(self.config.max_concurrent_scoring)

        sampling_args = {
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
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

    async def run(self, body: VerifiersAgentRunRequest) -> VerifiersAgentVerifyResponse:
        response = await self.responses(body)

        return VerifiersAgentVerifyResponse(
            responses_create_params=body.responses_create_params,
            response=response,
            reward=response.reward,
        )


if __name__ == "__main__":
    VerifiersAgent.run_webserver()
