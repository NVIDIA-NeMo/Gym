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
from typing import Any, Dict, Literal, Optional

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.judge import JudgeError, call_judge
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming


LEGACY_JUDGE_PROTOCOL = "legacy_v1_0"
V1_1_JUDGE_PROTOCOL = "official_v1_1"
JudgeProtocol = Literal["legacy_v1_0", "official_v1_1"]

V1_0_DATASET_REVISION = "bdae010bbce259820c0e34c1d7cce210d966fb75"  # pragma: allowlist secret
V1_1_DATASET_REVISION = "9a77ef56b717057ade24ceab4d273712a0b4f19e"  # pragma: allowlist secret

V1_1_SYSTEM_PROMPT = """Decide whether the CANDIDATE ANSWER is correct or incorrect against the OFFICIAL ANSWER.
Note the following points when assessing correctness:

- Numbers should still match when they are the same value written differently, e.g., a
  percentage, a count of percentage points, and the equivalent decimal fraction are the same
  value: 0.675, "67.5%" and "67.5 percentage points" all match. So do different scales
  (thousand, million, bn) and different notations (thousands separators, currency symbols,
  LaTeX markup, and numbers written as words).
- Where the question asks for a particular format (e.g., a percentage, a number of decimal
  places, a unit, a rounding, or an ordering) the CANDIDATE ANSWER must meet it. If the
  question asks for no particular format, accept any equivalent form.
- In cases where the question asks for an ordered list, a title, honorific or article added
  to an entry in the CANDIDATE ANSWER can change where that entry sorts. Accept the ordering
  if it is correct either with those additions or without them.
- Grade the value the CANDIDATE ANSWER finally commits to, and it must commit to one. Values
  reached while working, and alternatives it considers and sets aside, do not count. If it
  offers several values without selecting one, it is incorrect even if one of them is right.
  Hedging is fine as long as one clearly definitive answer is given."""


class AalcrResourcesServerConfig(BaseResourcesServerConfig):
    judge_model_server: ModelServerRef
    judge_protocol: JudgeProtocol = LEGACY_JUDGE_PROTOCOL
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
    aa_lcr_version: str = "1.0.0"
    aa_lcr_dataset_revision: str = V1_0_DATASET_REVISION
    aa_lcr_judge_protocol: JudgeProtocol = LEGACY_JUDGE_PROTOCOL


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
        _validate_protocol_metadata(body, self.config.judge_protocol)

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

        judge_responses_create_params = dict(
            input=_build_judge_input(
                self.config.judge_protocol,
                question=body.question,
                official_answer=body.answer,
                candidate_answer=candidate_answer,
            )
        )
        judge_responses_create_params |= self.config.judge_responses_create_params_overrides

        judge_response = await call_judge(
            self.server_client,
            server_name=self.config.judge_model_server.name,
            url_path="/v1/responses",
            json=judge_responses_create_params,
            response_model=NeMoGymResponse,
        )
        verdict = _parse_judge_verdict(judge_response.output_text, self.config.judge_protocol)
        if verdict == "CORRECT":
            invalid_judge_response = False
            reward = 1.0
        elif verdict == "INCORRECT":
            invalid_judge_response = False
            reward = 0.0
        else:
            invalid_judge_response = True
            reward = 0.0

        return AALCRVerifyResponse(
            **body.model_dump(),
            reward=reward,
            invalid_model_response=False,
            invalid_judge_response=invalid_judge_response,
            judge_responses_create_params=judge_responses_create_params,
            judge_response=judge_response,
            **{input_tokens_band_key: reward},
        )


def _build_judge_input(
    judge_protocol: JudgeProtocol,
    *,
    question: str,
    official_answer: str,
    candidate_answer: str,
) -> list[dict[str, str]]:
    if judge_protocol == LEGACY_JUDGE_PROTOCOL:
        user_prompt = f"""Assess whether the following CANDIDATE ANSWER is CORRECT or INCORRECT.
For the CANDIDATE ANSWER to be correct, it must be consistent with the OFFICIAL ANSWER.

The question, for reference only: {question}
The OFFICIAL ANSWER: {official_answer}
CANDIDATE ANSWER TO ASSESS: {candidate_answer}

Reply only with CORRECT or INCORRECT."""
        return [{"role": "user", "content": user_prompt}]

    user_prompt = f"""Assess whether the following CANDIDATE ANSWER is CORRECT or INCORRECT.
For the CANDIDATE ANSWER to be correct, it must be consistent with the OFFICIAL ANSWER.

The question, for reference only: START QUESTION {question}

END QUESTION

The OFFICIAL ANSWER: {official_answer}

END OFFICIAL ANSWER

BEGIN CANDIDATE ANSWER TO ASSESS

{candidate_answer}

END CANDIDATE ANSWER TO ASSESS

Reply as JSON, with a verdict of CORRECT or INCORRECT."""
    return [
        {"role": "system", "content": V1_1_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _parse_judge_verdict(judge_response_text: str, judge_protocol: JudgeProtocol) -> Optional[str]:
    text = judge_response_text.strip()
    if judge_protocol == LEGACY_JUDGE_PROTOCOL:
        return text if text in {"CORRECT", "INCORRECT"} else None

    if not text:
        raise JudgeError("empty judge response")

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise JudgeError("AA-LCR v1.1 judge response is not valid JSON") from error
    if not isinstance(payload, dict) or set(payload) != {"verdict"}:
        raise JudgeError("AA-LCR v1.1 judge response must contain only a verdict field")
    verdict = payload["verdict"]
    if verdict not in {"CORRECT", "INCORRECT"}:
        raise JudgeError("AA-LCR v1.1 judge verdict must be CORRECT or INCORRECT")
    return verdict


def _validate_protocol_metadata(body: AALCRVerifyRequest, judge_protocol: JudgeProtocol) -> None:
    expected = {
        LEGACY_JUDGE_PROTOCOL: ("1.0.0", V1_0_DATASET_REVISION),
        V1_1_JUDGE_PROTOCOL: ("1.1", V1_1_DATASET_REVISION),
    }[judge_protocol]
    actual = (body.aa_lcr_version, body.aa_lcr_dataset_revision)
    if body.aa_lcr_judge_protocol != judge_protocol or actual != expected:
        raise ValueError(
            "AA-LCR dataset and judge protocol mismatch: "
            f"server={judge_protocol}/{expected}, row={body.aa_lcr_judge_protocol}/{actual}"
        )


if __name__ == "__main__":
    AalcrResourcesServer.run_webserver()
