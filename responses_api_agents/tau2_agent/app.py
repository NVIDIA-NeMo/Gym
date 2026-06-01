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
"""Tau2 agent webserver entrypoint.

Only import names used by this module; keep imports grouped and minimal.
"""

import asyncio
import random
import time
from asyncio import Semaphore
from typing import Any
from uuid import uuid4

from pydantic import ConfigDict, PrivateAttr
from tau2.data_model.simulation import RunConfig

from nemo_gym.base_resources_server import (
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
)
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import ServerClient, get_first_server_config_dict
from responses_api_agents.tau2_agent.utils import (
    convert_trajectory_to_output_items,
    run_domain,
)

# Expose a name compatible with Ray remote function mocking in tests
# Tests patch `responses_api_agents.tau2_agent.app.runner_ray_remote`.
# Provide a simple alias to the local `run_domain` so the attribute exists
# and can be mocked in tests.
runner_ray_remote = run_domain


def get_config_path(path):
    """Return a Path-like object for a config path.

    Lightweight stub so tests can patch `get_config_path` in
    `responses_api_agents.tau2_agent.app`.
    """
    from pathlib import Path

    return Path(path)


class Tau2BenchAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    user_model_server: ModelServerRef = None
    max_steps: int = None
    concurrency: int = 4
    tau2_domain: str = None


class Tau2BenchAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class Tau2BenchAgentVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class Tau2BenchAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


class Tau2BenchAgent(SimpleResponsesAPIAgent):
    config: Tau2BenchAgentConfig
    _sem: Semaphore = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        self._sem = Semaphore(self.config.concurrency)
        print(
            f"\n\n\n============= max_concurrency set to {self.config.concurrency} =============\n\n\n"
        )

    async def responses(
        self, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        raise NotImplementedError

    async def run(self, body: Tau2BenchAgentRunRequest) -> Tau2BenchAgentVerifyResponse:
        async with self._sem:
            # Policy model
            model_server_name = self.config.model_server.name
            global_config_dict = (
                ServerClient.load_from_global_config().global_config_dict
            )
            model_server_config = get_first_server_config_dict(
                global_config_dict,
                model_server_name,
            )
            base_url = (
                f"http://{model_server_config['host']}:{model_server_config['port']}/v1"
            )
            policy_model_name = global_config_dict["policy_model_name"]
            model_name = f"hosted_vllm/{policy_model_name}"

            # User simulator model
            if self.config.user_model_server:
                user_model_server_name = self.config.user_model_server.name
                user_model_server_config = get_first_server_config_dict(
                    global_config_dict,
                    user_model_server_name,
                )
                user_base_url = f"http://{user_model_server_config['host']}:{user_model_server_config['port']}/v1"
                user_model_name = f"hosted_vllm/{user_model_server_config['model']}"
            else:
                user_base_url = base_url
                user_model_name = model_name

            tau2_domain = (
                body.task_domain
                if hasattr(body, "task_domain") and body.task_domain
                else self.config.tau2_domain
            )
            run_config = RunConfig(
                domain=tau2_domain,
                task_ids=[str(body.task_id)] if hasattr(body, "task_id") else None,
                agent="llm_agent",
                user="user_simulator",
                llm_agent=model_name,
                llm_user=user_model_name,
                num_trials=1,
                max_steps=200,  # You can set max_steps if needed
                save_to=None,  # To enable saving, update this save_to in run_domain as well
                max_concurrency=1,
                llm_args_agent={"api_base": base_url, "api_key": "EMPTY"},
                llm_args_user={"api_base": user_base_url, "api_key": "EMPTY"},
            )
            try:
                simulation_results = await asyncio.to_thread(run_domain, run_config)
                # Convert Tau^2 Bench trajectory format to NemoGym format
                output_messages = convert_trajectory_to_output_items(
                    simulation_results.simulations[0].messages
                )

                responses = [
                    NeMoGymResponse(
                        id=simulation_results.simulations[0].id,
                        object="response",
                        model="empty",
                        output=output_messages,
                        created_at=time.time(),
                        parallel_tool_calls=False,
                        tool_choice="auto",
                        tools=[],
                    )
                ]
                rewards = [
                    simulation_results.simulations[i].reward_info.reward
                    for i in range(len(simulation_results.simulations))
                ]

                verify_response = Tau2BenchAgentVerifyResponse(
                    responses_create_params=body.responses_create_params,
                    response=responses[0],  # can only return one rollout trajectory
                    reward=rewards[0],
                    reward_breakdown=simulation_results.simulations[
                        0
                    ].reward_info.reward_breakdown,
                )
            except Exception as e:
                import traceback

                task_id = str(body.task_id) if hasattr(body, "task_id") else "unknown"
                print(
                    f"Error running tau2 domain for task {task_id}: {e}\n{traceback.format_exc()}",
                    flush=True,
                )
                responses = [
                    NeMoGymResponse(
                        id=f"tau2-error-{uuid4()}",
                        object="response",
                        model="empty",
                        output=[],
                        created_at=time.time(),
                        parallel_tool_calls=False,
                        tool_choice="none",
                        tools=[],
                        metadata={
                            "error": str(e),
                            "traceback": str(traceback.format_exc()),
                            "task_id": task_id,
                        },
                    )
                ]
                verify_response = Tau2BenchAgentVerifyResponse(
                    responses_create_params=body.responses_create_params,
                    response=responses[0],
                    reward=0.0,
                    reward_breakdown={},
                )

            return verify_response


if __name__ == "__main__":
    Tau2BenchAgent.run_webserver()
