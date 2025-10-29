# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import json
from typing import Any, Dict

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import raise_for_status


class CrowdstrikeLogscaleSyntaxResourcesServerConfig(BaseResourcesServerConfig):
    judge_model: ModelServerRef


class CrowdstrikeLogscaleSyntaxVerifyRequest(BaseVerifyRequest):
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    reference: Dict[str, Any]


class CrowdstrikeLogscaleSyntaxVerifyResponse(BaseVerifyResponse, CrowdstrikeLogscaleSyntaxVerifyRequest):
    judge_response: NeMoGymResponse
    correct_syntax: bool
    used_keyword_arguments: bool


class CrowdstrikeLogscaleSyntaxResourcesServer(SimpleResourcesServer):
    config: CrowdstrikeLogscaleSyntaxResourcesServerConfig

    async def verify(self, body: CrowdstrikeLogscaleSyntaxVerifyRequest) -> CrowdstrikeLogscaleSyntaxVerifyResponse:
        input_content = body.judge_responses_create_params.input[1].content
        input_content = input_content.replace("{candidate_answer}", body.response.output_text)
        body.judge_responses_create_params.input[1].content = input_content

        response = await self.server_client.post(
            server_name=self.config.judge_model.name,
            url_path="/v1/responses",
            json=body.judge_responses_create_params,
        )
        await raise_for_status(response)

        judge_response = NeMoGymResponse.model_validate(await response.json())
        tool_calls = [o for o in judge_response.output if o.type == "function_call"]
        assert tool_calls, "Need a tool call from the judge!"

        results = json.loads(tool_calls[0].arguments)
        correct_syntax = results["correct_syntax"]
        used_keyword_arguments = results["used_keyword_arguments"]

        return CrowdstrikeLogscaleSyntaxVerifyResponse(
            **body.model_dump(),
            reward=1.0 if correct_syntax and used_keyword_arguments else 0.0,
            judge_response=judge_response,
            correct_syntax=correct_syntax,
            used_keyword_arguments=used_keyword_arguments,
        )


if __name__ == "__main__":
    CrowdstrikeLogscaleSyntaxResourcesServer.run_webserver()
