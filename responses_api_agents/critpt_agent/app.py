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
import re
from pathlib import Path
from typing import Any

import yaml
from fastapi import Request, Response
from pydantic import ConfigDict

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, Body, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymEasyInputMessage, NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import get_response_json, raise_for_status


LOG = logging.getLogger(__name__)

_THINK_BLOCK_PATTERN = re.compile(r"<think>.*?</think>|<thinking>.*?</thinking>", re.DOTALL)

# Repo root: responses_api_agents/critpt_agent/app.py -> nemo-gym/
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class CritPtAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    turn2_prompt_fpath: str


class CritPtAgentRunRequest(BaseRunRequest):
    problem_id: str
    code_template: str


class CritPtAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


class CritPtAgent(SimpleResponsesAPIAgent):
    config: CritPtAgentConfig

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        path = Path(self.config.turn2_prompt_fpath)
        if not path.is_absolute():
            path = _REPO_ROOT / path
        prompt_yaml = yaml.safe_load(path.read_text())
        self._turn2_user_template: str = prompt_yaml["user"]

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
        # Strip reasoning blocks: the Turn 2 user message asks the model not to reason again, so
        # the Turn 2 assistant context should contain only the conclusion, not the thinking trace.
        turn1_text = _strip_thinking_blocks(_extract_output_text(turn1_json))

        # Turn 2: populate code template using Turn 1 reasoning as context
        turn2_user_msg = self._turn2_user_template.format(code_template=body.code_template)
        turn2_input = list(body.responses_create_params.input) + [
            NeMoGymEasyInputMessage(role="assistant", content=turn1_text),
            NeMoGymEasyInputMessage(role="user", content=turn2_user_msg),
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


def _strip_thinking_blocks(text: str) -> str:
    return _THINK_BLOCK_PATTERN.sub("", text).strip()


def _extract_output_text(response_json: dict) -> str:
    parts = []
    for item in response_json.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                parts.append(content.get("text", ""))
    return "".join(parts)


if __name__ == "__main__":
    CritPtAgent.run_webserver()
