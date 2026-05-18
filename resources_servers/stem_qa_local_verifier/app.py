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
"""StemQA local-verifier resources server.

Port of NeMo-RLVR's `nemo_rl/environments/stem_qa_local_verifier.py` to
NeMo-Gym. Two grading modes selected per row by ``ground_truth.style``:

- ``style == "natural_text"`` — LLM-as-judge using the ``judge_model_server``
  configured on this server. Mirrors RLVR's ``LOOSE_VERIFIER_PROMPT``.
- otherwise (multiple-choice) — regex-extracts ``Answer: (X)`` from the
  response and compares letters case-insensitively.

Single-turn, single-rollout per request. Per-rollout LLM calls go through
``self.server_client.post(...)`` so concurrency is bounded by an
``asyncio.Semaphore``; pass-rate aggregation lives outside the server.
"""
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import nullcontext
from pathlib import Path
from typing import Optional, Union

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
from resources_servers.stem_qa_local_verifier.verifier_utils import (
    extract_answer_from_response,
    extract_mc_answer,
    extract_user_question,
    parse_individual_response,
)

logger = logging.getLogger(__name__)


class StemQALocalVerifierConfig(BaseResourcesServerConfig):
    """Configuration for the StemQA local-verifier server."""

    # Required: which model server to use as the LLM judge for natural-text answers.
    judge_model_server: ModelServerRef
    # Base create params for the judge call. ``input`` is overwritten per request.
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming

    # Path to the verifier prompt template. Placeholders: {question}, {ground_truth}, {response}.
    judge_prompt_template_fpath: str = "prompt_templates/stem_qa_loose_verifier.txt"

    # Concurrency limit for outgoing judge calls. None disables limiting.
    judge_endpoint_max_concurrency: Optional[int] = 64


class StemQAGroundTruth(BaseModel):
    style: str
    value: str


class StemQAVerifyRequest(BaseVerifyRequest):
    # Either a JSON-encoded string or a parsed dict with keys ``style`` and ``value``.
    # ``style`` must be ``"natural_text"`` for LLM-judge grading; any other value
    # triggers multiple-choice (letter) grading.
    ground_truth: Union[str, StemQAGroundTruth, dict]


class StemQAJudgeEvaluation(BaseModel):
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    response: NeMoGymResponse


class StemQAVerifyResponse(BaseVerifyResponse):
    style: str
    extracted_answer: str
    verification_failed: bool
    judge_evaluation: Optional[StemQAJudgeEvaluation] = None


def _normalize_ground_truth(gt: Union[str, StemQAGroundTruth, dict]) -> StemQAGroundTruth:
    if isinstance(gt, StemQAGroundTruth):
        return gt
    if isinstance(gt, str):
        gt = json.loads(gt)
    if not isinstance(gt, dict):
        raise ValueError(f"ground_truth must be dict-like, got {type(gt)!r}")
    return StemQAGroundTruth(style=str(gt["style"]), value=str(gt["value"]))


def _final_response_text(body: BaseVerifyRequest) -> str:
    if not body.response.output:
        return ""
    last_output = body.response.output[-1]
    if hasattr(last_output, "content") and last_output.content:
        return last_output.content[0].text
    return ""


class StemQALocalVerifierServer(SimpleResourcesServer):
    """Resources server that grades StemQA-style rollouts.

    Ports the per-row dispatch from RLVR's ``StemQALocalVerifierEnvironment.step``:
    natural-text answers go through an LLM judge, multiple-choice answers go
    through letter-extraction.
    """

    config: StemQALocalVerifierConfig

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

    async def verify(self, body: StemQAVerifyRequest) -> StemQAVerifyResponse:
        text = _final_response_text(body)

        # Incomplete <think> -> reward 0, not a verifier failure.
        if "<think>" in text and "</think>" not in text:
            try:
                style = _normalize_ground_truth(body.ground_truth).style
            except Exception:
                style = "unknown"
            return self._make_response(
                body,
                reward=0.0,
                style=style,
                extracted_answer="",
                verification_failed=False,
                judge_evaluation=None,
            )

        try:
            gt = _normalize_ground_truth(body.ground_truth)
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning("StemQA ground_truth parse failed: %s", e)
            return self._make_response(
                body,
                reward=0.0,
                style="unknown",
                extracted_answer="",
                verification_failed=True,
                judge_evaluation=None,
            )

        if gt.style == "natural_text":
            return await self._verify_natural_text(body, gt, text)
        return self._verify_multiple_choice(body, gt, text)

    def _make_response(
        self,
        body: StemQAVerifyRequest,
        *,
        reward: float,
        style: str,
        extracted_answer: str,
        verification_failed: bool,
        judge_evaluation: Optional[StemQAJudgeEvaluation],
    ) -> StemQAVerifyResponse:
        return StemQAVerifyResponse(
            **body.model_dump(),
            reward=reward,
            style=style,
            extracted_answer=extracted_answer,
            verification_failed=verification_failed,
            judge_evaluation=judge_evaluation,
        )

    def _verify_multiple_choice(
        self,
        body: StemQAVerifyRequest,
        gt: StemQAGroundTruth,
        text: str,
    ) -> StemQAVerifyResponse:
        predicted = extract_mc_answer(text)
        reward = (
            1.0
            if predicted.strip().upper() == gt.value.strip().upper() and predicted != ""
            else 0.0
        )
        return self._make_response(
            body,
            reward=reward,
            style=gt.style,
            extracted_answer=predicted,
            verification_failed=False,
            judge_evaluation=None,
        )

    async def _verify_natural_text(
        self,
        body: StemQAVerifyRequest,
        gt: StemQAGroundTruth,
        text: str,
    ) -> StemQAVerifyResponse:
        question = (
            extract_user_question(body.responses_create_params.input or [])
            or ""
        )
        extracted = extract_answer_from_response(text)

        prompt = self._judge_prompt_template.format(
            question=question,
            ground_truth=gt.value,
            response=extracted,
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
                logger.warning("StemQA judge HTTP call failed: %s", e)
                return self._make_response(
                    body,
                    reward=0.0,
                    style=gt.style,
                    extracted_answer=extracted,
                    verification_failed=True,
                    judge_evaluation=None,
                )

        evaluation = StemQAJudgeEvaluation(
            responses_create_params=responses_create_params,
            response=judge_response,
        )

        try:
            grading_text = judge_response.output[-1].content[-1].text
            score = parse_individual_response(grading_text)
            reward = float(score)
            verification_failed = False
        except Exception as e:
            logger.warning("StemQA judge response parse failed: %s", e)
            reward = 0.0
            verification_failed = True

        return self._make_response(
            body,
            reward=reward,
            style=gt.style,
            extracted_answer=extracted,
            verification_failed=verification_failed,
            judge_evaluation=evaluation,
        )


if __name__ == "__main__":
    StemQALocalVerifierServer.run_webserver()
