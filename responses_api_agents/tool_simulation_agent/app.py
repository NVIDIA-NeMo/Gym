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
import json

from fastapi import Body
from pydantic import ConfigDict, ValidationError

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import raise_for_status


class ToolSimulationAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef


class ToolSimulationAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class ToolSimulationAgentVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class ToolSimulationAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


class ToolSimulationAgent(SimpleResponsesAPIAgent):
    config: ToolSimulationAgentConfig

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        model_response = await self.server_client.post(
            server_name=self.config.model_server.name,
            url_path="/v1/responses",
            json=body,
        )

        # Model calls are expected to always succeed.
        await raise_for_status(model_response)
        model_response_json = await model_response.json()

        try:
            return NeMoGymResponse.model_validate(model_response_json)
        except ValidationError as e:
            raise RuntimeError(
                f"Received an invalid response from the model server: {json.dumps(model_response_json)}"
            ) from e

    async def run(self, body: ToolSimulationAgentRunRequest = Body()) -> ToolSimulationAgentVerifyResponse:
        config = self.config
        response = await self.server_client.post(
            server_name=config.name,
            url_path="/v1/responses",
            json=body.responses_create_params,
        )
        await raise_for_status(response)

        verify_request_body = body.model_dump()
        verify_request_body["response"] = await response.json()
        verify_request = ToolSimulationAgentVerifyRequest.model_validate(verify_request_body)

        verify_response = await self.server_client.post(
            server_name=config.resources_server.name,
            url_path="/verify",
            json=verify_request.model_dump(),
        )
        await raise_for_status(verify_response)
        verify_response_json = await verify_response.json()

        return ToolSimulationAgentVerifyResponse.model_validate(verify_response_json)


if __name__ == "__main__":
    ToolSimulationAgent.run_webserver()
