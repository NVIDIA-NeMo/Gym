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

from os import environ
from pathlib import Path
from subprocess import run
from time import time
from typing import Literal


DATA_DIR = Path(__file__).parent / "tau2_data"
environ["TAU2_DATA_DIR"] = str(DATA_DIR)

from fastapi import Body
from loguru import logger

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
from responses_api_models.vllm_model.app import VLLMConverter, split_responses_input_output_items
from tau2.data_model.simulation import SimulationRun, TextRunConfig
from tau2.data_model.tasks import Task
from tau2.evaluator.evaluator import EvaluationType
from tau2.runner.batch import run_single_task


class Tau2Config(BaseResponsesAPIAgentConfig):
    model_server: ModelServerRef
    user_model_server: ModelServerRef
    debug: bool = False
    print_step_counts: bool = False
    # Tau2 default
    max_steps: int = 200


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

        if not self.config.debug:
            print("Removing loguru logging since `debug=False`")
            logger.remove()

        if self.config.print_step_counts:
            environ["NEMO_GYM_TAU2_STEP_COUNT_PRINT"] = "true"

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

        config.max_steps = self.config.max_steps

        result = await run_single_task(**kwargs)

        body_dict = body.model_dump()
        body_dict.pop("responses_create_params")

        message_dicts = [m.model_dump() for m in result.messages]
        converter = VLLMConverter(return_token_id_information=True)
        all_items = converter.chat_completions_messages_to_responses_items(message_dicts)
        input_items, output_items = split_responses_input_output_items(all_items)
        return Tau2VerifyResponse(
            **body_dict,
            responses_create_params=dict(
                input=input_items,
                model=body.responses_create_params.model or "",
                parallel_tool_calls=body.responses_create_params.parallel_tool_calls,
                tool_choice=body.responses_create_params.tool_choice,
                tools=body.responses_create_params.tools,
            ),
            response=dict(
                id=f"tau2-{body.config.domain}-{body.task.id}",
                created_at=int(time()),
                object="response",
                output=output_items,
                model=body.responses_create_params.model or "",
                parallel_tool_calls=body.responses_create_params.parallel_tool_calls,
                tool_choice=body.responses_create_params.tool_choice,
                tools=body.responses_create_params.tools,
            ),
            reward=result.reward_info.reward,
            result=result,
        )


if __name__ == "__main__":
    Tau2Agent.run_webserver()
