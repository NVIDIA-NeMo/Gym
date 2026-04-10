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
from pathlib import Path
from subprocess import run
from typing import Any, Dict

from nemo_gym.base_resources_server import (
    BaseRunRequest,
    BaseVerifyResponse,
)
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.server_utils import get_server_url
from tau2.data_model.simulation import SimulationRun, TextRunConfig
from tau2.data_model.tasks import Task
from tau2.runner.batch import run_single_task


class Tau2Config(BaseResponsesAPIAgentConfig):
    model_server: ModelServerRef
    user_model_server: ModelServerRef
    max_steps: int = None


class Tau2RunRequest(BaseRunRequest):
    sample_data: Dict[str, Any]


class Tau2VerifyResponse(Tau2RunRequest, BaseVerifyResponse):
    result: SimulationRun


class Tau2Agent(SimpleResponsesAPIAgent):
    config: Tau2Config

    def setup_webserver(self):
        cwd = Path(__file__).parent
        repo_path = cwd / "tau2-bench"
        if not repo_path.exists():
            run(
                """git clone https://github.com/bxyu-nvidia/tau2-bench""",
                shell=True,
                cwd=cwd,
                check=True,
                executable="/bin/bash",
            )

        run(
            """source .venv/bin/activate \
    && cd tau2-bench \
    && git checkout b76acee2e625b8fb22deeb63cb0a3e756d5f094e \
    && uv sync --active""",
            shell=True,
            cwd=cwd,
            check=True,
            executable="/bin/bash",
        )
        return super().setup_webserver()

    async def run(self, body: Tau2RunRequest) -> Tau2VerifyResponse:
        kwargs = body.sample_data | {
            "config": TextRunConfig.model_validate(body.sample_data["config"]),
            "task": Task.model_validate(TextRunConfig.model_validate(body.sample_data["task"])),
            "save_dir": None,
        }

        config: TextRunConfig = kwargs["config"]
        config.llm_user = "dummy user model"
        config.llm_args_user |= {
            "api_base": f"{get_server_url(self.config.model_server.name)}/v1",
            "api_key": "dummy api key",
        }
        config.llm_agent = "dummy agent model"
        config.llm_args_agent = {
            "api_base": f"{get_server_url(self.config.user_model_server.name)}/v1",
            "api_key": "dummy api key",
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
