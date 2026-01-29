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
GenRM Pairwise Comparison Resources Server.

Compares multiple candidate responses using a GenRM model via pairwise comparisons.
The GenRM model expects OpenAI-format messages with special roles 'response_1' and 'response_2'.

Input:
- conversation_history: List of user/assistant messages
- responses: List of N candidate response strings to compare

Output:
- Per-response rewards after pairwise aggregation
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI
from pydantic import BaseModel

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    SimpleResourcesServer,
    BaseVerifyRequest,
    BaseVerifyResponse,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponseCreateParamsNonStreaming,
)
from resources_servers.genrm_compare.utils import (
    GenRMOutputParseError,
    aggregate_scores,
    extract_output_text,
    generate_comparison_pairs,
    parse_genrm_output,
)

logger = logging.getLogger(__name__)


class GenRMCompareConfig(BaseResourcesServerConfig):
    """Configuration for the GenRM compare server.

    Attributes:
        genrm_model_server: Target model server (GenRM model with special chat template)
        genrm_responses_create_params: Base create params for GenRM calls
        comparison_strategy: "all_pairs" or "circular"
        num_judges_per_comparison: Number of judge passes per pair (majority voting)
        aggregator_method: Method for aggregating scores
        reasoning_bonus: Bonus for shortest reasoning content among top performers
        answer_bonus: Bonus for shortest answer among top performers
        top_percentile: Percentile threshold for applying bonuses
        group_reasoning_length_penalty_coeff: Coefficient for reasoning length penalty
        group_answer_length_penalty_coeff: Coefficient for answer length penalty
        group_style_penalty_coeff: Coefficient for style density penalty
        default_score: Default neutral score when parsing fails
        default_ranking: Default neutral ranking when parsing fails
        debug_logging: Enable verbose logging for debugging
        genrm_parse_retries: Number of retries on parse failures
        genrm_parse_retry_sleep_s: Sleep duration between parse retries
        use_principle: Enable principle-based comparison
        default_principle: Default principle when none provided in request
    """

    name: str = "genrm_compare"
    genrm_model_server: ModelServerRef
    genrm_responses_create_params: NeMoGymResponseCreateParamsNonStreaming

    # Comparison strategy
    comparison_strategy: str = "circular"  # "all_pairs" or "circular"
    num_judges_per_comparison: int = 1

    # Principle-based GenRM settings
    use_principle: bool = False
    default_principle: str = (
        "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants "
        "to the user prompt. Begin your evaluation by generating your own answer to the prompt. You must provide "
        "your answer before judging any answers. When evaluating the assistants' answers, compare both assistants' "
        "answers with your answer. You must identify and correct any mistakes or inaccurate information. Then "
        "consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly "
        "responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or more than "
        "one interpretation, it is more helpful and appropriate to ask for clarifications or more information from "
        "the user than providing an answer based on assumptions. Relevant means all parts of the response closely "
        "connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or "
        "excessive. Then consider the creativity and novelty of the assistant's answers when needed. Finally, "
        "identify any missing important information in the assistants' answers that would be beneficial to include "
        "when responding to the user prompt."
    )

    # Aggregator settings (only "simple_tiebreaker" is currently implemented)
    aggregator_method: str = "simple_tiebreaker"

    # Length bonus config (only for simple_tiebreaker)
    reasoning_bonus: float = 0.0
    answer_bonus: float = 0.0
    top_percentile: float = 0.2
    group_reasoning_length_penalty_coeff: float = 0.0
    group_answer_length_penalty_coeff: float = 0.0
    group_style_penalty_coeff: float = 0.0

    # Default neutral scores when parsing fails
    default_score: float = 3.0
    default_ranking: float = 3.5

    # Debug logging
    debug_logging: bool = False

    # Retry config for parse failures
    genrm_parse_retries: int = 3
    genrm_parse_retry_sleep_s: float = 0.2


class GenRMCompareRequest(BaseModel):
    """Request payload for GenRM pairwise comparison."""

    conversation_history: List[Dict[str, str]]  # User/assistant messages before the responses
    response_objs: List[Dict[str, Any]]  # Raw Response API objects from policy model
    principle: Optional[str] = None  # Principle for principle-based GenRM (e.g., "The response should be helpful")


class GenRMCompareResponse(BaseModel):
    """Response payload with per-response rewards."""

    rewards: List[float]  # One reward per response, in same order as input
    comparison_results: Optional[List[Dict[str, Any]]] = None  # Detailed pairwise results
    metrics: Optional[Dict[str, float]] = None  # Aggregation metrics


class GenRMCompareResourcesServer(SimpleResourcesServer):
    """Resources server for GenRM pairwise comparison of multiple responses."""

    config: GenRMCompareConfig

    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        """Stub verify to satisfy abstract base; GenRMCompare uses /compare instead."""
        return BaseVerifyResponse(
            responses_create_params=body.responses_create_params,
            response=body.response,
            reward=self.config.default_score,
        )

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/compare")(self.compare)
        return app

    async def compare(self, body: GenRMCompareRequest) -> GenRMCompareResponse:
        """Compare multiple responses using GenRM pairwise comparisons.

        Args:
            body: Request with conversation_history and response_objs

        Returns:
            GenRMCompareResponse with per-response rewards
        """
        cfg = self.config
        response_objs = body.response_objs
        conversation_history = body.conversation_history
        num_responses = len(response_objs)

        if cfg.debug_logging:
            logger.info(f"[GenRM] Compare request: {num_responses} responses")

        # Single response case - return neutral reward (no comparison possible)
        if num_responses < 2:
            return GenRMCompareResponse(
                rewards=[cfg.default_score],
                comparison_results=None,
                metrics=None,
            )

        # Generate comparison pairs
        try:
            comparison_pairs = generate_comparison_pairs(
                cfg.comparison_strategy, num_responses
            )
            if cfg.debug_logging:
                logger.info(f"[GenRM] Strategy '{cfg.comparison_strategy}': {len(comparison_pairs)} pairs")
        except ValueError as e:
            raise ValueError(f"Configuration error: {e}")

        # Build comparison tasks - one task per (pair, judge) combination
        # Multiple judges per pair enables majority voting for more robust scores
        comparison_tasks = []
        comparison_metadata = []

        for judge_idx in range(cfg.num_judges_per_comparison):
            for i, j in comparison_pairs:
                task = self._run_single_comparison(
                    conversation_history,
                    response_objs[i],
                    response_objs[j],
                    pair_idx=(i, j),
                    principle=body.principle,
                )
                comparison_tasks.append(task)
                comparison_metadata.append((i, j, judge_idx))

        # Run all comparisons concurrently
        comparison_results = await asyncio.gather(*comparison_tasks)

        # Aggregate pairwise scores into per-response rewards
        rewards, metrics, base_rewards, bonuses = aggregate_scores(
            comparison_results=comparison_results,
            comparison_metadata=comparison_metadata,
            response_objs=response_objs,
            aggregator_method=cfg.aggregator_method,
            default_score=cfg.default_score,
            reasoning_bonus=cfg.reasoning_bonus,
            answer_bonus=cfg.answer_bonus,
            top_percentile=cfg.top_percentile,
            group_reasoning_length_penalty_coeff=cfg.group_reasoning_length_penalty_coeff,
            group_answer_length_penalty_coeff=cfg.group_answer_length_penalty_coeff,
            group_style_penalty_coeff=cfg.group_style_penalty_coeff,
        )

        # Format detailed results
        detailed_results = [
            {
                "response_i": i,
                "response_j": j,
                "judge_idx": judge_idx,
                "score_1": score_1,
                "score_2": score_2,
                "ranking": ranking,
            }
            for (score_1, score_2, ranking), (i, j, judge_idx) in zip(
                comparison_results, comparison_metadata
            )
        ]

        if cfg.debug_logging:
            logger.info(f"[GenRM] Final rewards: {[f'{r:.4f}' for r in rewards]}")

        return GenRMCompareResponse(
            rewards=rewards,
            comparison_results=detailed_results,
            metrics=metrics,
        )

    async def _run_single_comparison(
        self,
        conversation_history: List[Dict[str, str]],
        response_obj_1: Dict[str, Any],
        response_obj_2: Dict[str, Any],
        pair_idx: Tuple[int, int] = (0, 0),
        principle: Optional[str] = None,
    ) -> Tuple[float, float, float]:
        """Run a single pairwise comparison via GenRM.

        Args:
            conversation_history: The conversation context
            response_obj_1: First Response API object
            response_obj_2: Second Response API object
            pair_idx: Tuple of (i, j) for logging
            principle: Optional principle for principle-based comparison

        Returns:
            Tuple of (score_1, score_2, ranking)
        """
        cfg = self.config

        # Extract final answer from Response API objects (GenRM only takes the final answer, not reasoning)
        response_1 = extract_output_text(response_obj_1)
        response_2 = extract_output_text(response_obj_2)

        # Format messages for GenRM using special roles 'response_1' and 'response_2'
        # The GenRM model's chat template handles these custom roles
        messages: List[NeMoGymEasyInputMessage] = []
        for msg in conversation_history:
            messages.append(
                NeMoGymEasyInputMessage(
                    role=msg.get("role", "user"),
                    content=msg.get("content", ""),
                    type="message",
                )
            )

        # Add principle message if enabled
        if cfg.use_principle:
            principle_text = principle if principle else cfg.default_principle
            messages.append(
                NeMoGymEasyInputMessage(role="principle", content=principle_text, type="message")
            )

        messages.extend(
            [
                NeMoGymEasyInputMessage(role="response_1", content=response_1, type="message"),
                NeMoGymEasyInputMessage(role="response_2", content=response_2, type="message"),
            ]
        )

        # Build the request params
        responses_create_params = cfg.genrm_responses_create_params.model_copy(deep=True)
        responses_create_params.input = messages

        try:
            # Retry logic for parse failures (not connection errors, which are handled elsewhere)
            max_attempts = max(1, int(cfg.genrm_parse_retries) + 1)

            for attempt_idx in range(max_attempts):
                # Call the GenRM model via /v1/responses endpoint
                response = await self.server_client.post(
                    server_name=cfg.genrm_model_server.name,
                    url_path="/v1/responses",
                    json=responses_create_params,
                )
                raw_response = await response.json()

                # Extract output_text from GenRM response (skip reasoning, only parse the final JSON scores)
                genrm_answer = extract_output_text(raw_response)

                try:
                    score_1, score_2, ranking = parse_genrm_output(
                        genrm_answer,
                        cfg.default_score,
                        cfg.default_ranking,
                        raise_on_fail=True,
                    )
                    return score_1, score_2, ranking

                except GenRMOutputParseError:
                    if attempt_idx < max_attempts - 1:
                        await asyncio.sleep(float(cfg.genrm_parse_retry_sleep_s))
                        continue

                    # Give up: fall back to defaults
                    logger.warning(
                        f"[GenRM] Parse failed for pair {pair_idx} after {max_attempts} attempts; "
                        f"falling back to defaults."
                    )
                    return cfg.default_score, cfg.default_score, cfg.default_ranking

            return cfg.default_score, cfg.default_score, cfg.default_ranking

        except Exception as e:
            logger.error(f"[GenRM] Error in comparison for pair {pair_idx}: {e}")
            return cfg.default_score, cfg.default_score, cfg.default_ranking


if __name__ == "__main__":
    GenRMCompareResourcesServer.run_webserver()
