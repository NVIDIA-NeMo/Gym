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

from typing import Any, Dict, Optional

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import get_response_json


class AalcrResourcesServerConfig(BaseResourcesServerConfig):
    judge_model_server: ModelServerRef
    judge_responses_create_params_overrides: Dict[str, Any]


class AALCRVerifyRequest(BaseVerifyRequest):
    document_category: str
    document_set_id: str
    question_id: int
    question: str
    answer: str
    data_source_filenames: str
    data_source_urls: str
    input_tokens: int
    input_tokens_band: str


class AALCRVerifyResponse(AALCRVerifyRequest, BaseVerifyResponse):
    invalid_judge_response: bool
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    judge_response: NeMoGymResponse
    reward_lt_80k: Optional[float] = None
    reward_80k_100k: Optional[float] = None
    reward_100k_110k: Optional[float] = None
    reward_110k_128k: Optional[float] = None
    reward_128k_plus: Optional[float] = None


class AalcrResourcesServer(SimpleResourcesServer):
    config: AalcrResourcesServerConfig

    async def verify(self, body: AALCRVerifyRequest) -> AALCRVerifyResponse:
        candidate_answer = body.response.output_text

        judge_prompt = f"""Assess whether the following CANDIDATE ANSWER is CORRECT or INCORRECT.
For the CANDIDATE ANSWER to be correct, it must be consistent with the OFFICIAL ANSWER.

The question, for reference only: {body.question}
The OFFICIAL ANSWER: {body.answer}
CANDIDATE ANSWER TO ASSESS: {candidate_answer}

Reply only with CORRECT or INCORRECT."""

        judge_responses_create_params = dict(input=[{"role": "user", "content": judge_prompt}])
        judge_responses_create_params |= self.config.judge_responses_create_params_overrides

        http_response = await self.server_client.post(
            server_name=self.config.judge_model_server.name,
            url_path="/v1/responses",
            json=judge_responses_create_params,
        )
        judge_response = NeMoGymResponse.model_validate(await get_response_json(http_response))

        judge_response_text = judge_response.output_text.strip()
        if judge_response_text == "CORRECT":
            invalid_judge_response = False
            reward = 1.0
        elif judge_response_text == "INCORRECT":
            invalid_judge_response = False
            reward = 0.0
        else:
            invalid_judge_response = True
            reward = 0.0

        match body.input_tokens_band:
            case "<80k":
                input_tokens_band_key = "reward_lt_80k"
            case "80k-100k":
                input_tokens_band_key = "reward_80k_100k"
            case "100k-110k":
                input_tokens_band_key = "reward_100k_110k"
            case "110k-128k":
                input_tokens_band_key = "reward_110k_128k"
            case "128k+":
                input_tokens_band_key = "reward_128k_plus"

        return AALCRVerifyResponse(
            **body.model_dump(),
            reward=reward,
            invalid_judge_response=invalid_judge_response,
            judge_responses_create_params=judge_responses_create_params,
            judge_response=judge_response,
            **{input_tokens_band_key: reward},
        )


if __name__ == "__main__":
    AalcrResourcesServer.run_webserver()
