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
from typing import Any, Dict, List, Literal, Optional


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
    num_steps: int
    num_agent_calls: int
    min_prompt_tokens: Optional[float]
    min_completion_tokens: Optional[float]
    mean_prompt_tokens: Optional[float]
    mean_completion_tokens: Optional[float]
    max_prompt_tokens: Optional[float]
    max_completion_tokens: Optional[float]


class Tau2Agent(SimpleResponsesAPIAgent):
    config: Tau2Config

    __key_metrics: Optional[List[str]] = None

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

        messages_to_convert = []
        for message in result.messages:
            if message.role == "user" and message.tool_calls:
                continue
            elif message.role == "tool" and message.requestor == "user":
                continue
            messages_to_convert.append(message)

        message_dicts = to_litellm_messages(messages_to_convert)

        converter = VLLMConverter(return_token_id_information=True)
        all_items = converter.chat_completions_messages_to_responses_items(message_dicts)
        input_items_1, output_items = split_responses_input_output_items(all_items)
        # Tau starts trajectories with an assistant message
        input_items_1 += output_items[:1]
        input_items_2, output_items = split_responses_input_output_items(output_items[1:])

        prompt_usages = []
        completion_usages = []
        num_agent_calls = 0
        for message in result.messages:
            if not message.role == "assistant":
                continue

            num_agent_calls += 1
            if message.usage:
                prompt_usages.append(message.usage["prompt_tokens"])
                completion_usages.append(message.usage["completion_tokens"])

        min_prompt_tokens = None
        min_completion_tokens = None
        mean_prompt_tokens = None
        mean_completion_tokens = None
        max_prompt_tokens = None
        max_completion_tokens = None
        if prompt_usages:
            min_prompt_tokens = min(prompt_usages)
            min_completion_tokens = min(completion_usages)
            mean_prompt_tokens = sum(prompt_usages) / len(prompt_usages)
            mean_completion_tokens = sum(completion_usages) / len(completion_usages)
            max_prompt_tokens = max(prompt_usages)
            max_completion_tokens = max(completion_usages)

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
            num_steps=len(result.messages),
            num_agent_calls=num_agent_calls,
            min_prompt_tokens=min_prompt_tokens,
            min_completion_tokens=min_completion_tokens,
            mean_prompt_tokens=mean_prompt_tokens,
            mean_completion_tokens=mean_completion_tokens,
            max_prompt_tokens=max_prompt_tokens,
            max_completion_tokens=max_completion_tokens,
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
        return res | {k: agent_metrics[k] for k in self.__key_metrics}

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        domain_to_rewards = defaultdict(list)
        domain_to_unique_samples = defaultdict(int)
        termination_reason_domain_count = defaultdict(int)
        termination_reason_count = defaultdict(int)
        finish_reasons_count = defaultdict(int)
        hallucination_count = defaultdict(int)
        transfer_to_human_agents = 0
        total_count = 0
        missing_tool_call = 0
        incomplete_reasoning = 0
        telecom_subtask_rewards = defaultdict(list)
        for task_group in tasks:
            for task in task_group:
                domain = task["config"]["domain"]
                domain_to_rewards[domain].append(task["reward"])

                if domain == "telecom":
                    subtask = task["task"]["id"].split("]")[0].removeprefix("[")
                    telecom_subtask_rewards[f"telecom/{subtask}/reward"].append(task["reward"])

                termination_reason = task["result"]["termination_reason"]
                termination_reason_count[f"trajectory_termination_reason/{termination_reason}/count"] += 1
                termination_reason_domain_count[
                    f"{domain}/trajectory_termination_reason/{termination_reason}/count"
                ] += 1

                this_task_transfer_to_human_agents = False
                has_tool_call = False
                for message in task["result"]["messages"]:
                    if message["role"] == "tool":
                        # e.g. `Error: Tool 'run_speed_test' not found.`
                        if "Error: Tool" and "not found" in message["content"]:
                            tool_name = message["content"].removeprefix("Error: Tool '").removesuffix(" not found.")
                            hallucination_count[f"tool_call_hallucination/{tool_name}/count"] += 1

                    if message["role"] != "assistant":
                        continue

                    if message["raw_data"]:
                        finish_reason = message["raw_data"]["choices"][0]["finish_reason"]
                        finish_reasons_count[f"message_finish_reason/{finish_reason}/count"] += 1

                        raw_message = message["raw_data"]["choices"][0]["message"]
                        has_reasoning = raw_message.get("reasoning_content") is not None
                        is_empty = not (raw_message.get("content") or raw_message.get("tool_calls"))
                        incomplete_reasoning += is_empty and has_reasoning

                    if not message.get("tool_calls"):
                        continue

                    has_tool_call = True

                    if message["tool_calls"][0]["name"] == "transfer_to_human_agents":
                        this_task_transfer_to_human_agents = True

                missing_tool_call += not has_tool_call
                transfer_to_human_agents += this_task_transfer_to_human_agents
                total_count += 1

            domain_to_unique_samples[f"{domain}/num_samples_unique"] += 1

        total_num_assistant_messages = sum(finish_reasons_count.values())
        finish_reasons_pct = {
            f"{k.removesuffix('/count')}/pct": v / total_num_assistant_messages
            for k, v in finish_reasons_count.items()
        }

        telecom_subtask_avg_reward = {k: sum(v) / len(v) for k, v in telecom_subtask_rewards.items()}

        domain_to_average_reward: Dict[str, float] = dict()
        domain_to_counts: Dict[str, int] = dict()
        for domain, rewards in domain_to_rewards.items():
            domain_to_counts[f"{domain}/num_samples_total"] = len(rewards)
            domain_to_average_reward[f"{domain}/reward"] = (
                sum(rewards) / domain_to_counts[f"{domain}/num_samples_total"]
            )

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

        res = {
            "macro_average": macro_average,
            **domain_to_unique_samples,
            **domain_to_counts,
            **domain_to_average_reward,
            **telecom_subtask_avg_reward,
            **termination_reason_domain_count,
            **termination_reason_count,
            **termination_reason_pct,
            **termination_reason_domain_pct,
            **finish_reasons_count,
            **finish_reasons_pct,
            "trajectory_transfer_to_human_agents/count": transfer_to_human_agents,
            "trajectory_transfer_to_human_agents/pct": transfer_to_human_agents / total_count,
            **hallucination_count,
            "tool_call_hallucination/count/total": sum(hallucination_count.values()),
            "trajectory_missing_tool_call/count": missing_tool_call,
            "trajectory_missing_tool_call/pct": missing_tool_call / total_count,
            "messages_with_incomplete_reasoning/count": incomplete_reasoning,
            "messages_with_incomplete_reasoning/pct": incomplete_reasoning / total_num_assistant_messages,
        }
        self.__key_metrics = list(res.keys())
        return res


if __name__ == "__main__":
    Tau2Agent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = Tau2Agent.run_webserver()  # noqa: F401
