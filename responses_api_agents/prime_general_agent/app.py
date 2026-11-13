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
from pathlib import Path

from fastapi import Body, Request, Response

from responses_api_agents.verifiers_agent.app import (
    VerifiersAgent,
    VerifiersAgentRunRequest,
    VerifiersNeMoGymResponse,
)


class PrimeGeneralAgent(VerifiersAgent):
    async def responses(
        self,
        request: Request,
        response: Response,
        body: VerifiersAgentRunRequest = Body(),
    ) -> VerifiersNeMoGymResponse:
        # prepare.py stores info.task_dir relative to the env's tasks dir so rows work in any venv.
        # An absolute task_dir is left as is.
        if "task_dir" in body.info:
            tasks_dir = self._get_env(body.vf_env_id or self.config.vf_env_id).taskset.tasks_dir
            body.info["task_dir"] = str(Path(tasks_dir) / body.info["task_dir"])
        return await super().responses(request, response, body)


if __name__ == "__main__":
    PrimeGeneralAgent.run_webserver()
