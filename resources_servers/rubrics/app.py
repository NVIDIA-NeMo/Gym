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
"""Rubrics resources server.

Port of NeMo-RLVR's `nemo_rl/environments/rubrics_environment.py` to NeMo-Gym.

For each rollout, runs an LLM judge that scores the model answer against a
list of rubrics. Each rubric contributes a positive (must-have) or negative
(pitfall) weight; the final reward is the weighted ratio of passed rubrics
to total positive weight.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Optional, Union

from fastapi import FastAPI
from pydantic import BaseModel

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import get_response_json
from resources_servers.rubrics.verifier_utils import (
    compute_weighted_reward,
    extract_answer_from_response,
    extract_rubrics,
    extract_user_question,
    format_rubrics_for_prompt,
    parse_rubrics_response,
)

logger = logging.getLogger(__name__)


class RubricsResourcesServerConfig(BaseResourcesServerConfig):
    """Configuration for the rubrics resources server."""

    judge_model_server: ModelServerRef
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming

    judge_prompt_template_fpath: str = "prompt_templates/rubrics_verifier.txt"
    judge_endpoint_max_concurrency: Optional[int] = 64


class RubricsVerifyRequest(BaseVerifyRequest):
    # Either a JSON string or dict containing a ``rubrics`` key. The rubrics
    # value is a list of dicts: either ``{criterion, points}`` (new) or
    # ``{title, description, weight}`` (legacy). A doubly-nested
    # ``rubrics: [[{...}, ...]]`` shape is also accepted (RLVR datasets).
    ground_truth: Union[str, dict]


class RubricsJudgeEvaluation(BaseModel):
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    response: NeMoGymResponse
    parsed: Optional[dict[str, Any]] = None


class RubricsVerifyResponse(BaseVerifyResponse):
    extracted_answer: str
    passed_count: int
    total_count: int
    verification_failed: bool
    judge_evaluation: Optional[RubricsJudgeEvaluation] = None


def _final_response_text(body: BaseVerifyRequest) -> str:
    if not body.response.output:
        return ""
    last_output = body.response.output[-1]
    if hasattr(last_output, "content") and last_output.content:
        return last_output.content[0].text
    return ""


class RubricsResourcesServer(SimpleResourcesServer):
    config: RubricsResourcesServerConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.config.judge_endpoint_max_concurrency is not None:
            self._judge_semaphore = asyncio.Semaphore(
                value=self.config.judge_endpoint_max_concurrency
            )
        else:
            self._judge_semaphore = nullcontext()

        template_path = Path(self.config.judge_prompt_template_fpath)
        if not template_path.is_absolute():
            template_path = Path(__file__).resolve().parent / template_path
        with open(template_path, "r") as f:
            self._judge_prompt_template = f.read()

    def setup_webserver(self) -> FastAPI:
        return super().setup_webserver()

    async def verify(self, body: RubricsVerifyRequest) -> RubricsVerifyResponse:
        text = _final_response_text(body)

        # Incomplete <think> -> reward 0, not a verifier failure.
        if "<think>" in text and "</think>" not in text:
            return self._make_response(
                body,
                reward=0.0,
                extracted_answer="",
                passed=0,
                total=0,
                verification_failed=False,
                judge_evaluation=None,
            )

        rubrics = extract_rubrics(body.ground_truth)
        if not rubrics:
            logger.warning("Rubrics ground truth missing or empty.")
            return self._make_response(
                body,
                reward=0.0,
                extracted_answer="",
                passed=0,
                total=0,
                verification_failed=True,
                judge_evaluation=None,
            )

        question = (
            extract_user_question(body.responses_create_params.input or [])
            or ""
        )
        extracted = extract_answer_from_response(text)

        prompt = self._judge_prompt_template.format(
            question=question,
            model_answer=extracted,
            rubrics=format_rubrics_for_prompt(rubrics),
        )
        responses_create_params = self.config.judge_responses_create_params.model_copy(deep=True)
        responses_create_params.input = [
            NeMoGymEasyInputMessage(role="user", content=prompt)
        ]

        async with self._judge_semaphore:
            try:
                http_response = await self.server_client.post(
                    server_name=self.config.judge_model_server.name,
                    url_path="/v1/responses",
                    json=responses_create_params,
                )
                judge_response = NeMoGymResponse.model_validate(
                    await get_response_json(http_response)
                )
            except Exception as e:
                logger.warning("Rubrics judge HTTP call failed: %s", e)
                return self._make_response(
                    body,
                    reward=0.0,
                    extracted_answer=extracted,
                    passed=0,
                    total=len(rubrics),
                    verification_failed=True,
                    judge_evaluation=None,
                )

        evaluation = RubricsJudgeEvaluation(
            responses_create_params=responses_create_params,
            response=judge_response,
            parsed=None,
        )

        try:
            grading_text = judge_response.output[-1].content[-1].text
            parsed = parse_rubrics_response(grading_text, len(rubrics))
            reward, passed, total = compute_weighted_reward(rubrics, parsed)
            evaluation.parsed = parsed
            return self._make_response(
                body,
                reward=reward,
                extracted_answer=extracted,
                passed=passed,
                total=total,
                verification_failed=False,
                judge_evaluation=evaluation,
            )
        except Exception as e:
            logger.warning("Rubrics judge response parse failed: %s", e)
            return self._make_response(
                body,
                reward=0.0,
                extracted_answer=extracted,
                passed=0,
                total=len(rubrics),
                verification_failed=True,
                judge_evaluation=evaluation,
            )

    def _make_response(
        self,
        body: RubricsVerifyRequest,
        *,
        reward: float,
        extracted_answer: str,
        passed: int,
        total: int,
        verification_failed: bool,
        judge_evaluation: Optional[RubricsJudgeEvaluation],
    ) -> RubricsVerifyResponse:
        return RubricsVerifyResponse(
            **body.model_dump(),
            reward=reward,
            extracted_answer=extracted_answer,
            passed_count=passed,
            total_count=total,
            verification_failed=verification_failed,
            judge_evaluation=judge_evaluation,
        )


if __name__ == "__main__":
    RubricsResourcesServer.run_webserver()
