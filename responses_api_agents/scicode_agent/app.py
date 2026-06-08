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

"""SciCode multi-step generation agent.

SciCode problems are solved one sub-step at a time: for each sub-step the agent builds a prompt
from the problem description, the code it generated for previous sub-steps, and the current
function header, calls the model, extracts the Python code, and accumulates it. After all
sub-steps it sends the accumulated per-step solutions to the resources server's /verify endpoint.

The full step loop — building each step's prompt, extracting the Python code block,
prefilled-steps handling, context-window-exhaustion handling, and the accumulation/verify
call — is not yet implemented (see run()).

Templated on responses_api_agents/proof_refinement_agent (the multi-turn run() skeleton).
"""

import logging

from fastapi import Request, Response
from pydantic import ConfigDict

from nemo_gym.base_resources_server import BaseRunRequest
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import raise_for_status


LOG = logging.getLogger(__name__)


class ScicodeAgentConfig(BaseResponsesAPIAgentConfig):
    """Configuration for the SciCode multi-step agent."""

    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    # Include each sub-step's scientific background in the prompt (nemo-skills default).
    with_background: bool = True


class ScicodeAgentRunRequest(BaseRunRequest):
    """extra verifier_metadata fields (problem_id, sub_steps,
    required_dependencies) forward through to the resources server."""

    model_config = ConfigDict(extra="allow")


class ScicodeAgent(SimpleResponsesAPIAgent):
    """Agent that drives the SciCode per-sub-step generation + code-accumulation loop."""

    config: ScicodeAgentConfig

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        body = body.model_copy(deep=True)

        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        model_response = await self.server_client.post(
            server_name=self.config.model_server.name,
            url_path="/v1/responses",
            json=body,
            cookies=request.cookies,
        )
        await raise_for_status(model_response)
        model_response_json = await model_response.json()

        for k, v in model_response.cookies.items():
            response.set_cookie(k, v)

        return NeMoGymResponse.model_validate(model_response_json)

    async def run(self, request: Request, body: ScicodeAgentRunRequest):
        """Execute the SciCode multi-step loop, then verify.

        TODO: for cur_step in range(len(sub_steps)):
          - skip prefilled steps; mark remaining steps as out-of-context if the window overflows
          - build the per-step prompt and post to self.config.name /v1/responses
          - extract the Python code block and accumulate deps + previous code + this step
        Then POST the accumulated solutions dict to the resources server /verify and return the result.
        """
        raise NotImplementedError("SciCode multi-step loop not yet implemented")


if __name__ == "__main__":
    ScicodeAgent.run_webserver()
