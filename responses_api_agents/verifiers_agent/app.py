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

import verifiers as vf
from openai import AsyncOpenAI
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
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming = Field(
        default_factory=lambda: NeMoGymResponseCreateParamsNonStreaming(input=[])
    )


class VerifiersAgent(SimpleResponsesAPIAgent):
    """Uses vf_env.run_group() with an AsyncOpenAI client pointing to the vLLM model server."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    config: VerifiersAgentConfig

    _vf_env: vf.Environment | None = None
    _env_id: str | None = None
    _dataset_rows: list[dict] | None = None
    _openai_client: AsyncOpenAI | None = None

    async def _ensure_env_loaded(self) -> None:
        if self._vf_env is not None:
            return

        response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json={
                "vf_env_id": self.config.vf_env_id,
                "vf_env_args": self.config.vf_env_args,
                "dataset_n": self.config.dataset_n,
                "dataset_seed": self.config.dataset_seed,
            },
        )
        response.raise_for_status()
        seed_response = VerifiersSeedSessionResponse.model_validate(await response.json())

        self._env_id = seed_response.env_id
        logger.info(f"Seeded verifiers environment: {seed_response.vf_env_id} with {seed_response.dataset_length} examples")

        self._vf_env = vf.load_environment(self.config.vf_env_id, **self.config.vf_env_args)
        dataset = self._vf_env.get_dataset(n=self.config.dataset_n, seed=self.config.dataset_seed)

        self._dataset_rows = [
            {
                "prompt": dataset["prompt"][i],
                "example_id": dataset["example_id"][i],
                "task": dataset["task"][i],
                **({"answer": dataset["answer"][i]} if "answer" in dataset.column_names else {}),
                **({"info": dataset["info"][i]} if "info" in dataset.column_names else {}),
            }
            for i in range(len(dataset))
        ]

    def _get_openai_client(self) -> AsyncOpenAI:
        if self._openai_client is None:
            from nemo_gym.global_config import get_first_server_config_dict

            server_config_dict = get_first_server_config_dict(
                self.server_client.global_config_dict,
                self.config.model_server.name,
            )
            model_server_url = f"http://{server_config_dict.host}:{server_config_dict.port}"

            if not model_server_url.endswith("/v1"):
                model_server_url = model_server_url.rstrip("/") + "/v1"

            self._openai_client = AsyncOpenAI(
                base_url=model_server_url,
                api_key="dummy",  # assuming vLLM for now, probably breaks with openai model
            )
            logger.info(f"Created OpenAI client pointing to: {model_server_url}")

        return self._openai_client

    async def responses(self, req: VerifiersAgentRunRequest) -> VerifiersNeMoGymResponse:
        await self._ensure_env_loaded()

        task_idx = req.task_idx
        row = self._dataset_rows[task_idx]

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

        states = await self._vf_env.run_group(
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

        return VerifiersNeMoGymResponse(
            id=f"verifiers-{self._env_id}-{task_idx}",
            created_at=0,
            model=self.config.model_name,
            object="response",
            output=[],  # Could put trajectory if needed for something
            env_id=self._env_id,
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
