"""
MultiChallenge Environment Resources Server.

Evaluates model responses on the MultiChallenge benchmark using an LLM judge.
Each task contains a conversation context and a rubric of yes/no questions
that assess the quality of the final assistant response.

This environment:
1. Loads tasks from configurable splits (e.g., "advanced", "vanilla")
2. Feeds conversation context to the policy model
3. Retrieves the final response (excluding thinking parts)
4. Evaluates against each rubric entry using an LLM judge
5. Aggregates scores using a configurable method
"""

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
from __future__ import annotations

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


class AggregationMode(str, Enum):
    """How to aggregate rubric scores into a final reward."""

    # Average of all rubric scores
    MEAN = "mean"
    # Minimum score across all rubric items (strict)
    MIN = "min"
    # Maximum score across all rubric items (lenient)
    MAX = "max"
    # All rubric items must pass (product of binary scores)
    ALL = "all"
    # Any rubric item passes (max of binary scores)
    ANY = "any"
    # Weighted average (requires weights in rubric items)
    WEIGHTED = "weighted"


class RubricEvaluation(BaseModel):
    """Result of evaluating a single rubric item."""

    question: str
    pass_criteria: str
    judge_prompt: str
    judge_response: str
    verdict: str  # "YES" or "NO"
    score: float  # 1.0 for pass, 0.0 for fail
    weight: float = 1.0


class MultiChallengeConfig(BaseResourcesServerConfig):
    """Configuration for the MultiChallenge environment server."""

    name: str = "multichallenge"

    # Reference to the judge model server
    judge_model_server: ModelServerRef = Field(description="Reference to the model server used as the LLM judge")

    # Parameters for judge requests
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming = Field(
        description="Base parameters for judge model requests"
    )

    # Aggregation mode for combining rubric scores
    aggregation_mode: AggregationMode = Field(
        default=AggregationMode.MEAN, description="How to aggregate scores from multiple rubric items"
    )

    # Template for the judge prompt
    judge_prompt_template: str = Field(
        default="""You are evaluating whether a model's response meets a specific criterion.

CONVERSATION CONTEXT:
{context}

MODEL'S FINAL RESPONSE:
{response}

EVALUATION QUESTION:
{question}

EXPECTED ANSWER: {pass_criteria}

Does the model's response satisfy the criterion described in the evaluation question?
Analyze carefully, then respond with exactly [[YES]] or [[NO]] on the last line.""",
        description="Template for the judge evaluation prompt",
    )

    # System message for the judge
    judge_system_message: Optional[str] = Field(
        default="You are a precise evaluator. Assess responses objectively based on the given criteria.",
        description="Optional system message for the judge",
    )

    # Whether to run rubric evaluations in parallel
    parallel_evaluation: bool = Field(default=True, description="Whether to evaluate rubric items in parallel")

    # Labels for verdict extraction
    yes_label: str = Field(default="[[YES]]", description="Label indicating YES verdict")
    no_label: str = Field(default="[[NO]]", description="Label indicating NO verdict")


class MultiChallengeRunRequest(BaseRunRequest):
    """Run request payload for MultiChallenge tasks."""

    model_config = ConfigDict(extra="allow")

    uuid: Optional[str | int] = None
    task_id: Optional[int] = None
    rubric: Optional[List[dict]] = None
    context: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


class MultiChallengeVerifyRequest(MultiChallengeRunRequest, BaseVerifyRequest):
    """Verify request that includes the model's response."""

    pass


class MultiChallengeVerifyResponse(BaseVerifyResponse):
    """Response with detailed rubric evaluations."""

    model_config = ConfigDict(extra="allow")

    context: str
    generated_response: str
    rubric_evaluations: List[RubricEvaluation]
    aggregation_mode: str
    num_passed: int
    num_total: int


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
                # Remove <think>...</think> blocks
                full_text = re.sub(r"<think>.*?</think>", "", full_text, flags=re.DOTALL)
                # Also remove <thinking>...</thinking> blocks
                full_text = re.sub(r"<thinking>.*?</thinking>", "", full_text, flags=re.DOTALL)

            return full_text.strip()
    return ""


def _build_context_from_messages(messages: List[dict], exclude_thinking: bool = True) -> str:
    """Build a readable context string from the message history."""
    context_parts = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        # Skip thinking messages
        if exclude_thinking and role == "thinking":
            continue

        role_label = role.upper()
        context_parts.append(f"[{role_label}]: {content}")

    return "\n\n".join(context_parts)


def _extract_verdict(response_text: str, yes_label: str, no_label: str) -> str:
    """Extract YES/NO verdict from judge response."""
    # Look for the labels in the response
    yes_pos = response_text.rfind(yes_label)
    no_pos = response_text.rfind(no_label)

    if yes_pos < 0 and no_pos < 0:
        # Fallback: look for plain YES/NO at end of response
        lines = response_text.strip().split("\n")
        last_line = lines[-1].strip().upper() if lines else ""
        if "YES" in last_line:
            return "YES"
        elif "NO" in last_line:
            return "NO"
        return "NO"  # Default to NO if unclear

    # Return whichever appears last (most authoritative)
    if yes_pos > no_pos:
        return "YES"
    return "NO"


class MultiChallengeServer(SimpleResourcesServer):
    """MultiChallenge evaluation server."""

    config: MultiChallengeConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        return app

    async def verify(self, body: MultiChallengeVerifyRequest) -> MultiChallengeVerifyResponse:
        """Verify model response against the rubric using LLM judge."""

        # Extract the generated response (without thinking)
        generated_response = _extract_text_from_response(body.response, exclude_thinking=True)

        # Get context from the request or build from messages if available
        context = body.context or ""
        if not context and body.metadata and "messages" in body.metadata:
            context = _build_context_from_messages(body.metadata["messages"])

        # Get rubric from request
        rubric = body.rubric or []
        if not rubric and body.metadata and "rubric" in body.metadata:
            rubric = body.metadata["rubric"]

        # Evaluate each rubric item
        if self.config.parallel_evaluation and len(rubric) > 1:
            import asyncio

            evaluations = await asyncio.gather(
                *[self._evaluate_rubric_item(item, context, generated_response) for item in rubric]
            )
        else:
            evaluations = []
            for item in rubric:
                eval_result = await self._evaluate_rubric_item(item, context, generated_response)
                evaluations.append(eval_result)

        # Aggregate scores
        reward = self._aggregate_scores(evaluations)
        num_passed = sum(1 for e in evaluations if e.score >= 0.99)

        # Build response
        payload = body.model_dump()
        payload.pop("context", None)
        payload.pop("rubric", None)

        return MultiChallengeVerifyResponse(
            **payload,
            reward=reward,
            context=context,
            generated_response=generated_response,
            rubric_evaluations=evaluations,
            aggregation_mode=self.config.aggregation_mode.value,
            num_passed=num_passed,
            num_total=len(evaluations),
        )

    async def _evaluate_rubric_item(self, item: dict, context: str, response: str) -> RubricEvaluation:
        """Evaluate a single rubric item using the LLM judge."""

        question = item.get("question", "")
        pass_criteria = item.get("pass_criteria", "YES")
        weight = item.get("weight", 1.0)

        # Format the judge prompt
        judge_prompt = self.config.judge_prompt_template.format(
            context=context,
            response=response,
            question=question,
            pass_criteria=pass_criteria,
        )

        # Build messages for judge
        msgs: List[NeMoGymEasyInputMessage] = []
        if self.config.judge_system_message:
            msgs.append(NeMoGymEasyInputMessage(role="system", content=self.config.judge_system_message))
        msgs.append(NeMoGymEasyInputMessage(role="user", content=judge_prompt))

        # Create request parameters
        request_params = self.config.judge_responses_create_params.model_copy(deep=True)
        request_params.input = msgs

        # Call judge model
        response_obj = await self.server_client.post(
            server_name=self.config.judge_model_server.name,
            url_path="/v1/responses",
            json=request_params,
        )
        judge_response = NeMoGymResponse.model_validate(await response_obj.json())
        judge_text = _extract_text_from_response(judge_response, exclude_thinking=True)

        # Extract verdict
        verdict = _extract_verdict(judge_text, self.config.yes_label, self.config.no_label)

        # Score based on whether verdict matches expected criteria
        if pass_criteria.upper() == "YES":
            score = 1.0 if verdict == "YES" else 0.0
        elif pass_criteria.upper() == "NO":
            score = 1.0 if verdict == "NO" else 0.0
        else:
            # For other criteria, treat YES as success
            score = 1.0 if verdict == "YES" else 0.0

        return RubricEvaluation(
            question=question,
            pass_criteria=pass_criteria,
            judge_prompt=judge_prompt,
            judge_response=judge_text,
            verdict=verdict,
            score=score,
            weight=weight,
        )

    def _aggregate_scores(self, evaluations: List[RubricEvaluation]) -> float:
        """Aggregate rubric scores into final reward."""
        if not evaluations:
            return 0.0

        scores = [e.score for e in evaluations]
        weights = [e.weight for e in evaluations]

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
            total_weight = sum(weights)
            if total_weight == 0:
                return 0.0
            weighted_sum = sum(s * w for s, w in zip(scores, weights))
            return weighted_sum / total_weight

        return 0.0


if __name__ == "__main__":
    MultiChallengeServer.run_webserver()
