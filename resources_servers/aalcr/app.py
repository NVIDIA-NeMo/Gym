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
import re
from typing import Any, Dict, Optional

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.judge import JudgeError, call_judge
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from resources_servers.aalcr.versions import DEFAULT_VERSION, get_version


VERDICT_CORRECT = "CORRECT"
VERDICT_INCORRECT = "INCORRECT"

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_json_verdict(judge_response_text: str) -> Optional[str]:
    """Extract the verdict from a JSON judge reply, or None if it is unusable.

    The upstream prompt asks for "JSON, with a verdict of CORRECT or INCORRECT" without fixing a schema,
    so accept any JSON object carrying a `verdict` key, tolerating a code fence or surrounding prose.
    """
    match = _JSON_OBJECT_RE.search(judge_response_text)
    if match is None:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None

    for key, value in parsed.items():
        if isinstance(key, str) and key.strip().lower() == "verdict" and isinstance(value, str):
            verdict = value.strip().upper()
            if verdict in (VERDICT_CORRECT, VERDICT_INCORRECT):
                return verdict
    return None


def _parse_bare_verdict(judge_response_text: str) -> Optional[str]:
    if judge_response_text in (VERDICT_CORRECT, VERDICT_INCORRECT):
        return judge_response_text
    return None


class AalcrResourcesServerConfig(BaseResourcesServerConfig):
    judge_model_server: ModelServerRef
    judge_responses_create_params_overrides: Dict[str, Any]
    # Selects the upstream answer keys *and* the judge protocol that grades them; see versions.py.
    dataset_version: str = DEFAULT_VERSION


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
    invalid_model_response: bool
    invalid_judge_response: Optional[bool] = None
    judge_responses_create_params: Optional[NeMoGymResponseCreateParamsNonStreaming] = None
    judge_response: Optional[NeMoGymResponse] = None
    reward_lt_80k: Optional[float] = None
    reward_80k_100k: Optional[float] = None
    reward_100k_110k: Optional[float] = None
    reward_110k_128k: Optional[float] = None
    reward_128k_plus: Optional[float] = None


class AalcrResourcesServer(SimpleResourcesServer):
    config: AalcrResourcesServerConfig

    async def verify(self, body: AALCRVerifyRequest) -> AALCRVerifyResponse:
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

        candidate_answer = body.response.output_text.strip()
        if not candidate_answer:
            reward = 0.0
            return AALCRVerifyResponse(
                **body.model_dump(),
                invalid_model_response=True,
                reward=reward,
                **{input_tokens_band_key: reward},
            )

        version = get_version(self.config.dataset_version)
        judge_prompt = version.judge_user_prompt.format(
            question=body.question, official_answer=body.answer, candidate_answer=candidate_answer
        )

        judge_responses_create_params: Dict[str, Any] = dict(input=[{"role": "user", "content": judge_prompt}])
        if version.judge_system_prompt is not None:
            judge_responses_create_params["instructions"] = version.judge_system_prompt
        judge_responses_create_params |= self.config.judge_responses_create_params_overrides

        judge_response = await call_judge(
            self.server_client,
            server_name=self.config.judge_model_server.name,
            url_path="/v1/responses",
            json=judge_responses_create_params,
            response_model=NeMoGymResponse,
        )
        judge_response_text = judge_response.output_text.strip()
        if not judge_response_text:
            raise JudgeError("empty judge response")

        parse = _parse_json_verdict if version.judge_replies_json else _parse_bare_verdict
        verdict = parse(judge_response_text)
        invalid_judge_response = verdict is None
        reward = 1.0 if verdict == VERDICT_CORRECT else 0.0

        return AALCRVerifyResponse(
            **body.model_dump(),
            reward=reward,
            invalid_model_response=False,
            invalid_judge_response=invalid_judge_response,
            judge_responses_create_params=judge_responses_create_params,
            judge_response=judge_response,
            **{input_tokens_band_key: reward},
        )


if __name__ == "__main__":
    AalcrResourcesServer.run_webserver()
