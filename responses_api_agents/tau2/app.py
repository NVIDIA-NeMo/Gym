# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from asyncio import get_event_loop
from functools import partial
from os import environ
from pathlib import Path
from subprocess import run
from typing import Literal


DATA_DIR = Path(__file__).parent / "tau2_data"
environ["TAU2_DATA_DIR"] = str(DATA_DIR)

from fastapi import Body

from nemo_gym.base_resources_server import (
    BaseRunRequest,
    BaseVerifyResponse,
)
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import get_server_url
from tau2.data_model.simulation import SimulationRun, TextRunConfig
from tau2.data_model.tasks import Task
from tau2.evaluator.evaluator import EvaluationType
from tau2.runner.batch import run_single_task


class Tau2Config(BaseResponsesAPIAgentConfig):
    model_server: ModelServerRef
    user_model_server: ModelServerRef
    max_steps: int = None


class Tau2RunRequest(BaseRunRequest):
    config: TextRunConfig
    task: Task
    seed: int
    evaluation_type: EvaluationType
    save_dir: Literal[None]
    user_voice_settings: Literal[None]
    user_persona_config: Literal[None]
    verbose_logs: Literal[False]
    audio_debug: Literal[False]
    audio_taps: Literal[False]
    auto_review: Literal[False]
    review_mode: Literal["full"]
    hallucination_feedback: Literal[None]


class Tau2VerifyResponse(Tau2RunRequest, BaseVerifyResponse):
    result: SimulationRun


class Tau2Agent(SimpleResponsesAPIAgent):
    config: Tau2Config

    def setup_webserver(self):
        cwd = Path(__file__).parent
        if not DATA_DIR.exists():
            run(
                """git clone https://github.com/bxyu-nvidia/tau2-bench \
&& cd tau2-bench \
&& git checkout bxyu/nemo_gym_stable \
&& cd .. \
&& mv tau2-bench/data tau2_data \
&& rm -rf tau2-bench""",
                shell=True,
                cwd=cwd,
                check=True,
                executable="/bin/bash",
            )

        return super().setup_webserver()

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        raise NotImplementedError

    async def run(self, body: Tau2RunRequest) -> Tau2VerifyResponse:
        kwargs = {name: getattr(body, name) for name in Tau2RunRequest.model_fields}
        kwargs.pop("responses_create_params")

        config: TextRunConfig = kwargs["config"]
        # Need `openai/` provider prefix for LiteLLM
        config.llm_user = "openai/dummy user model"
        config.llm_args_user |= {
            "api_base": f"{get_server_url(self.config.model_server.name)}/v1",
            "api_key": "dummy api key",
        }
        # TODO support User LLM temperature parameter. Temperature 0.0 is not supported for GPT 5.2
        config.llm_args_user.pop("temperature")
        # Need `openai/` provider prefix for LiteLLM
        config.llm_agent = "openai/dummy agent model"
        config.llm_args_agent = {
            "api_base": f"{get_server_url(self.config.user_model_server.name)}/v1",
            "api_key": "dummy api key",
        }

        loop = get_event_loop()
        task_fn = partial(run_single_task, **kwargs)
        result = await loop.run_in_executor(None, task_fn)

        return Tau2VerifyResponse.model_validate(
            **body.model_dump(),
            reward=result.reward_info.reward,
            result=result,
        )


if __name__ == "__main__":
    Tau2Agent.run_webserver()
