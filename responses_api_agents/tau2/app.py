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
from typing import Any, Dict

from fastapi import Request

from nemo_gym.base_resources_server import (
    BaseRunRequest,
    BaseVerifyResponse,
)
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from responses_api_agents.tau2.repo_utils import clone_and_checkout
from tau2.data_model.simulation import SimulationRun, TextRunConfig
from tau2.data_model.tasks import Task
from tau2.runner.batch import run_single_task


class Tau2Config(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    max_steps: int = None


class Tau2RunRequest(BaseRunRequest):
    sample_data: Dict[str, Any]


class Tau2VerifyResponse(Tau2RunRequest, BaseVerifyResponse):
    result: SimulationRun


class Tau2Agent(SimpleResponsesAPIAgent):
    config: Tau2Config

    def setup_webserver(self):
        clone_and_checkout()
        return super().setup_webserver()

    async def run(self, request: Request, body: Tau2RunRequest) -> Tau2VerifyResponse:
        kwargs = body.sample_data | {
            "config": TextRunConfig.model_validate(body.sample_data["config"]),
            "task": Task.model_validate(TextRunConfig.model_validate(body.sample_data["task"])),
            "save_dir": None,
        }

        loop = get_event_loop()
        result = await loop.run_in_executor(None, run_single_task, **kwargs)

        return Tau2VerifyResponse.model_validate(
            **body.model_dump(),
            reward=result.reward_info.reward,
            result=result,
        )


if __name__ == "__main__":
    Tau2Agent.run_webserver()
