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
import logging

from fastapi import Request, Response
from pydantic import ConfigDict

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, Body, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import get_response_json, raise_for_status


LOG = logging.getLogger(__name__)


class CritPtAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef


class CritPtAgentRunRequest(BaseRunRequest):
    problem_id: str
    code_template: str


class CritPtAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


class CritPtAgent(SimpleResponsesAPIAgent):
    config: CritPtAgentConfig

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        model_response = await self.server_client.post(
            server_name=self.config.model_server.name,
            url_path="/v1/responses",
            json=body,
            cookies=request.cookies,
        )
        await raise_for_status(model_response)
        for k, v in model_response.cookies.items():
            response.set_cookie(k, v)
        return NeMoGymResponse.model_validate(await get_response_json(model_response))

    async def run(self, request: Request, body: CritPtAgentRunRequest) -> CritPtAgentVerifyResponse:
        cookies = request.cookies

        seed_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(seed_response)
        cookies = seed_response.cookies

        # Turn 1: solve the problem
        turn1_response = await self.server_client.post(
            server_name=self.config.name,
            url_path="/v1/responses",
            json=body.responses_create_params,
            cookies=cookies,
        )
        await raise_for_status(turn1_response)
        cookies = turn1_response.cookies
        turn1_json = await get_response_json(turn1_response)
        turn1_text = _extract_output_text(turn1_json)

        # Turn 2: populate code template using Turn 1 reasoning as context
        turn2_input = list(body.responses_create_params.input) + [
            {"role": "assistant", "content": turn1_text},
            {"role": "user", "content": _build_turn2_user_message(body.code_template)},
        ]
        turn2_params = body.responses_create_params.model_copy(update={"input": turn2_input})

        turn2_response = await self.server_client.post(
            server_name=self.config.name,
            url_path="/v1/responses",
            json=turn2_params,
            cookies=cookies,
        )
        await raise_for_status(turn2_response)
        cookies = turn2_response.cookies
        turn2_json = await get_response_json(turn2_response)

        # Verify Turn 2 output against the Artificial Analysis API
        verify_request_data = body.model_dump() | {"response": turn2_json}
        verify_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=verify_request_data,
            cookies=cookies,
        )
        await raise_for_status(verify_response)
        return CritPtAgentVerifyResponse.model_validate(await get_response_json(verify_response))


def _extract_output_text(response_json: dict) -> str:
    parts = []
    for item in response_json.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                parts.append(content.get("text", ""))
    return "".join(parts)


def _build_turn2_user_message(code_template: str) -> str:
    return (
        "Populate your final answer into the code template provided below.\n"
        "This step is purely for formatting/display purposes. No additional reasoning or derivation should be performed.\n"
        "Do not import any modules or packages beyond what is provided in the template.\n"
        f"```python\n{code_template}\n```"
    )


if __name__ == "__main__":
    CritPtAgent.run_webserver()
