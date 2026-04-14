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

from collections import defaultdict
from os import environ
from pathlib import Path
from subprocess import run
from time import time
from typing import Any, Dict, List, Literal


DATA_DIR = Path(__file__).parent / "tau2_data"
environ["TAU2_DATA_DIR"] = str(DATA_DIR)

from fastapi import Body
from loguru import logger
from pydantic import Field

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
from nemo_gym.server_utils import get_server_url, is_nemo_gym_fastapi_entrypoint
from responses_api_models.vllm_model.app import VLLMConverter, split_responses_input_output_items
from tau2.data_model.simulation import SimulationRun, TextRunConfig
from tau2.data_model.tasks import Task
from tau2.evaluator.evaluator import EvaluationType
from tau2.runner.batch import run_single_task
from tau2.utils.llm_utils import to_litellm_messages


class Tau2Config(BaseResponsesAPIAgentConfig):
    model_server: ModelServerRef
    user_model_server: ModelServerRef
    user_llm_args: dict = Field(default_factory=dict)
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
    duration: float


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
        body_dict = {name: getattr(body, name) for name in Tau2RunRequest.model_fields}
        responses_create_params = body_dict.pop("responses_create_params").model_dump(exclude_unset=True)

        config: TextRunConfig = body_dict["config"]

        # Need `openai/` provider prefix for LiteLLM
        config.llm_user = "openai/dummy user model"
        config.llm_args_user |= {
            "api_base": f"{get_server_url(self.config.user_model_server.name)}/v1",
            "api_key": "dummy api key",  # pragma: allowlist secret
        } | self.config.user_llm_args

        extra_agent_args = {k: v for k, v in responses_create_params.items() if k in ("temperature", "top_p")}
        # Need `openai/` provider prefix for LiteLLM
        config.llm_agent = "openai/dummy agent model"
        config.llm_args_agent = {
            "api_base": f"{get_server_url(self.config.model_server.name)}/v1",
            "api_key": "dummy api key",  # pragma: allowlist secret
        } | extra_agent_args

        config.max_steps = self.config.max_steps

        result = await run_single_task(**body_dict)

        message_dicts = to_litellm_messages(result.messages)
        converter = VLLMConverter(return_token_id_information=True)
        all_items = converter.chat_completions_messages_to_responses_items(message_dicts)
        input_items_1, output_items = split_responses_input_output_items(all_items)
        # Tau starts trajectories with an assistant message
        input_items_1 += output_items[:1]
        input_items_2, output_items = split_responses_input_output_items(output_items[1:])

        return Tau2VerifyResponse(
            **body_dict,
            responses_create_params=dict(
                input=body.responses_create_params.input + input_items_1 + input_items_2,
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
            duration=result.duration,
        )

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Override to select headline metrics for this benchmark.

        Default: all mean/* entries from agent_metrics.
        """
        res = super().get_key_metrics(agent_metrics)
        del (
            res["mean/seed"],
            res["mean/verbose_logs"],
            res["mean/audio_debug"],
            res["mean/audio_taps"],
            res["mean/auto_review"],
        )
        return res

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        domain_to_rewards = defaultdict(list)
        domain_to_unique_samples = defaultdict(int)
        termination_reason_domain_count = defaultdict(int)
        termination_reason_count = defaultdict(int)
        finish_reasons_count = defaultdict(int)
        total_count = 0
        for task_group in tasks:
            for task in task_group:
                domain = task["config"]["domain"]
                domain_to_rewards[domain].append(task["reward"])

                termination_reason = task["result"]["termination_reason"]
                termination_reason_count[f"{termination_reason}/count"] += 1
                termination_reason_domain_count[f"{domain}/{termination_reason}/count"] += 1

                for message in task["result"]["messages"]:
                    if message["role"] != "assistant":
                        continue

                    finish_reason = message["raw_data"]["choices"][0]["finish_reason"]
                    finish_reasons_count[f"{finish_reason}/count"] += 1

                total_count += 1

            domain_to_unique_samples[f"{domain}/num_samples_unique"] += 1

        total_finish_reason = sum(finish_reasons_count.values())
        finish_reasons_pct = {
            f"{k.removesuffix('/count')}/pct": v / total_finish_reason for k, v in finish_reasons_count.items()
        }

        domain_to_average_reward: Dict[str, float] = dict()
        domain_to_counts: Dict[str, int] = dict()
        for domain, rewards in domain_to_rewards.items():
            domain_to_counts[f"{domain}/num_samples_total"] = len(rewards)
            domain_to_average_reward[f"{domain}/reward"] = sum(rewards) / domain_to_counts[domain]

        macro_average = sum(domain_to_average_reward.values()) / len(domain_to_average_reward)

        termination_reason_pct = {
            f"{k.removesuffix('/count')}/pct": v / total_count for k, v in termination_reason_count.items()
        }
        termination_reason_domain_pct = dict()
        for k, v in termination_reason_domain_count.items():
            for domain, domain_count in domain_to_counts.items():
                if k.startswith(domain):
                    termination_reason_domain_pct[f"{k.removesuffix('/count')}/pct"] = v / domain_count
                    break

        return {
            "macro_average": macro_average,
            **domain_to_unique_samples,
            **domain_to_counts,
            **domain_to_average_reward,
            **termination_reason_domain_count,
            **termination_reason_count,
            **termination_reason_pct,
            **termination_reason_domain_pct,
            **finish_reasons_count,
            **finish_reasons_pct,
        }


if __name__ == "__main__":
    Tau2Agent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = Tau2Agent.run_webserver()  # noqa: F401
