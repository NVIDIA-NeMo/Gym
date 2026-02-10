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
"""
Inverse IF (Instruction Following) Environment Resources Server.

Evaluates model responses on the Inverse IF benchmark using a per-task LLM judge.
Each task is a single-turn instruction-following challenge where the model must
precisely follow complex formatting, content, and structural constraints.

Unlike MultiChallenge (which uses a global judge template), each Inverse IF task
carries its own judge prompt template and system prompt. The judge evaluates one
criterion at a time and returns a structured JSON verdict: {"result": "PASS"/"FAIL"}.

This environment:
1. Loads single-turn tasks with intricate instruction-following requirements
2. Feeds the user prompt to the policy model for generation
3. Retrieves the generated response (excluding thinking/reasoning blocks)
4. For each rubric criterion, queries the LLM judge using the task's own template
5. Aggregates per-criterion PASS/FAIL scores into a final reward
"""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import Any, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
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


# ---------------------------------------------------------------------------
# Enums and data models
# ---------------------------------------------------------------------------


class AggregationMode(str, Enum):
    """How to aggregate rubric scores into a final reward."""

    MEAN = "mean"
    MIN = "min"
    MAX = "max"
    ALL = "all"
    ANY = "any"
    WEIGHTED = "weighted"


class RubricEvaluation(BaseModel):
    """Result of evaluating a single rubric criterion."""

    criterion_id: str
    criteria: str
    judge_prompt: str
    judge_response: str
    verdict: str  # "PASS" or "FAIL"
    explanation: str
    score: float  # 1.0 for PASS, 0.0 for FAIL


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class InverseIFConfig(BaseResourcesServerConfig):
    """Configuration for the Inverse IF environment server."""

    name: str = "inverse_if"

    # Reference to the judge model server
    judge_model_server: ModelServerRef = Field(
        description="Reference to the model server used as the LLM judge"
    )

    # Parameters for judge requests
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming = Field(
        description="Base parameters for judge model requests"
    )

    # Aggregation mode for combining rubric scores
    aggregation_mode: AggregationMode = Field(
        default=AggregationMode.MEAN,
        description="How to aggregate scores from multiple rubric criteria",
    )

    # Default judge prompts (used as fallback for tasks missing their own)
    default_judge_prompt_template: str = Field(
        description="Fallback judge prompt template for tasks missing their own",
    )
    default_judge_system_prompt: str = Field(
        description="Fallback judge system prompt for tasks missing their own",
    )

    # Whether to run rubric evaluations in parallel
    parallel_evaluation: bool = Field(
        default=True,
        description="Whether to evaluate rubric criteria in parallel",
    )


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class InverseIFRunRequest(BaseRunRequest):
    """Run request payload for Inverse IF tasks."""

    model_config = ConfigDict(extra="allow")

    uuid: Optional[str | int] = None
    task_id: Optional[int] = None
    prompt: Optional[str] = None
    rubric: Optional[List[dict]] = None
    reference_response: Optional[str] = None
    judge_prompt_template: Optional[str] = None
    judge_system_prompt: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


class InverseIFVerifyRequest(InverseIFRunRequest, BaseVerifyRequest):
    """Verify request that includes the model's response."""

    pass


class InverseIFVerifyResponse(BaseVerifyResponse):
    """Response with detailed rubric evaluations."""

    model_config = ConfigDict(extra="allow")

    prompt: str
    generated_response: str
    reference_response: str
    rubric_evaluations: List[RubricEvaluation]
    aggregation_mode: str
    num_passed: int
    num_total: int


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _extract_text_from_response(response: NeMoGymResponse, exclude_thinking: bool = True) -> str:
    """Extract text content from the last assistant message, optionally excluding thinking."""
    for output in reversed(response.output):
        if getattr(output, "type", None) == "message" and getattr(output, "role", None) == "assistant":
            content = getattr(output, "content", None)
            if isinstance(content, list):
                texts = []
                for c in content:
                    text = getattr(c, "text", None)
                    if isinstance(text, str):
                        texts.append(text)
                full_text = "\n".join(texts).strip()
            elif isinstance(content, str):
                full_text = content.strip()
            else:
                continue

            if exclude_thinking:
                full_text = re.sub(r"<think>.*?</think>", "", full_text, flags=re.DOTALL)
                full_text = re.sub(r"<thinking>.*?</thinking>", "", full_text, flags=re.DOTALL)
                # Fallback: the opening <think>/<thinking> tag may have been part of
                # the prompt template rather than the model's generation, so
                # generated_response starts with CoT reasoning followed by </think>
                # without a matching opening tag. Strip everything up to and
                # including the unpaired closing tag.
                full_text = re.sub(r"^.*?</think>", "", full_text, flags=re.DOTALL)
                full_text = re.sub(r"^.*?</thinking>", "", full_text, flags=re.DOTALL)

            return full_text.strip()
    return ""


def _extract_verdict(judge_text: str) -> tuple[str, str]:
    """
    Extract PASS/FAIL verdict and explanation from judge response.

    The judge is instructed to return JSON like:
        {"result": "PASS", "explanation": "..."}

    We try multiple strategies:
    1. Parse a JSON block (possibly inside ```json ... ```)
    2. Find JSON-like object anywhere in the text
    3. Fallback to scanning for PASS/FAIL keywords

    Returns:
        (verdict, explanation) where verdict is "PASS" or "FAIL".
    """
    # Strategy 1: Extract from fenced code block
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", judge_text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(1))
            result = parsed.get("result", "").upper().strip()
            explanation = parsed.get("explanation", "")
            if result in ("PASS", "FAIL"):
                return result, explanation
        except (json.JSONDecodeError, AttributeError):
            pass

    # Strategy 2: Find any JSON object with a "result" field
    json_obj_match = re.search(r"\{[^{}]*\"result\"\s*:\s*\"(PASS|FAIL)\"[^{}]*\}", judge_text, re.IGNORECASE)
    if json_obj_match:
        try:
            parsed = json.loads(json_obj_match.group(0))
            result = parsed.get("result", "").upper().strip()
            explanation = parsed.get("explanation", "")
            if result in ("PASS", "FAIL"):
                return result, explanation
        except (json.JSONDecodeError, AttributeError):
            pass

    # Strategy 3: Keyword fallback — scan last few lines
    lines = judge_text.strip().split("\n")
    for line in reversed(lines[-5:]):
        upper = line.strip().upper()
        if "PASS" in upper:
            return "PASS", ""
        if "FAIL" in upper:
            return "FAIL", ""

    # Default to FAIL if we can't determine the verdict
    return "FAIL", ""


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


class InverseIFServer(SimpleResourcesServer):
    """Inverse IF evaluation server."""

    config: InverseIFConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        return app

    async def verify(self, body: InverseIFVerifyRequest) -> InverseIFVerifyResponse:
        """Verify a model response against per-criterion rubric using the LLM judge."""

        # Extract the generated response (excluding thinking blocks)
        generated_response = _extract_text_from_response(body.response, exclude_thinking=True)

        # Resolve task-level fields (top-level or from metadata)
        prompt = body.prompt or ""
        reference_response = body.reference_response or ""
        rubric = body.rubric or []
        judge_prompt_template = body.judge_prompt_template or self.config.default_judge_prompt_template
        judge_system_prompt = body.judge_system_prompt or self.config.default_judge_system_prompt

        # Fallback: pull from metadata if top-level fields are empty
        if body.metadata:
            if not prompt:
                prompt = body.metadata.get("prompt", "")
            if not reference_response:
                reference_response = body.metadata.get("reference_response", "")
            if not rubric:
                rubric = body.metadata.get("rubric", [])

        # Evaluate each criterion
        if self.config.parallel_evaluation and len(rubric) > 1:
            import asyncio

            evaluations = await asyncio.gather(
                *[
                    self._evaluate_criterion(
                        item, prompt, generated_response, reference_response,
                        judge_prompt_template, judge_system_prompt,
                    )
                    for item in rubric
                ]
            )
        else:
            evaluations = []
            for item in rubric:
                result = await self._evaluate_criterion(
                    item, prompt, generated_response, reference_response,
                    judge_prompt_template, judge_system_prompt,
                )
                evaluations.append(result)

        # Aggregate scores
        reward = self._aggregate_scores(evaluations)
        num_passed = sum(1 for e in evaluations if e.score >= 0.99)

        # Build response
        payload = body.model_dump()
        for key in ("prompt", "rubric", "reference_response", "judge_prompt_template", "judge_system_prompt"):
            payload.pop(key, None)

        return InverseIFVerifyResponse(
            **payload,
            reward=reward,
            prompt=prompt,
            generated_response=generated_response,
            reference_response=reference_response,
            rubric_evaluations=evaluations,
            aggregation_mode=self.config.aggregation_mode.value,
            num_passed=num_passed,
            num_total=len(evaluations),
        )

    async def _evaluate_criterion(
        self,
        item: dict,
        prompt: str,
        model_response: str,
        standard_response: str,
        judge_prompt_template: str,
        judge_system_prompt: str,
    ) -> RubricEvaluation:
        """Evaluate a single rubric criterion using the LLM judge."""

        criterion_id = item.get("id", "")
        criteria = item.get("criteria", "")

        # Fill the judge prompt template with canonical placeholders.
        # All variant placeholder names (typos, aliases) are normalised to
        # these four during preprocessing — see dataset_preprocess.py.
        judge_prompt = judge_prompt_template.format(
            prompt=prompt,
            model_response=model_response,
            standard_response=standard_response,
            criteria=criteria,
        )

        # Build messages for the judge
        msgs: List[NeMoGymEasyInputMessage] = []
        if judge_system_prompt:
            msgs.append(NeMoGymEasyInputMessage(role="system", content=judge_system_prompt))
        msgs.append(NeMoGymEasyInputMessage(role="user", content=judge_prompt))

        # Create request parameters
        request_params = self.config.judge_responses_create_params.model_copy(deep=True)
        request_params.input = msgs

        # Call the judge model
        response_obj = await self.server_client.post(
            server_name=self.config.judge_model_server.name,
            url_path="/v1/responses",
            json=request_params,
        )
        judge_response = NeMoGymResponse.model_validate(await response_obj.json())
        judge_text = _extract_text_from_response(judge_response, exclude_thinking=True)

        # Extract verdict
        verdict, explanation = _extract_verdict(judge_text)
        score = 1.0 if verdict == "PASS" else 0.0

        return RubricEvaluation(
            criterion_id=criterion_id,
            criteria=criteria,
            judge_prompt=judge_prompt,
            judge_response=judge_text,
            verdict=verdict,
            explanation=explanation,
            score=score,
        )

    def _aggregate_scores(self, evaluations: List[RubricEvaluation]) -> float:
        """Aggregate rubric scores into a final reward."""
        if not evaluations:
            return 0.0

        scores = [e.score for e in evaluations]
        mode = self.config.aggregation_mode

        if mode == AggregationMode.MEAN:
            return sum(scores) / len(scores)
        elif mode == AggregationMode.MIN:
            return min(scores)
        elif mode == AggregationMode.MAX:
            return max(scores)
        elif mode == AggregationMode.ALL:
            return 1.0 if all(s >= 0.99 for s in scores) else 0.0
        elif mode == AggregationMode.ANY:
            return 1.0 if any(s >= 0.99 for s in scores) else 0.0
        elif mode == AggregationMode.WEIGHTED:
            # Inverse IF rubric items don't carry weights, but support it for generality
            return sum(scores) / len(scores)
        return 0.0


if __name__ == "__main__":
    InverseIFServer.run_webserver()
