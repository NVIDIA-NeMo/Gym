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
- response_objs: List of N candidate Response API objects to compare

Output:
- Per-response rewards after pairwise aggregation
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections import defaultdict
from typing import Any, Dict, List, Literal, Optional, Tuple

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.global_config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponseCreateParamsNonStreaming,
)
from resources_servers.genrm_compare.utils import (
    GenRMOutputParseError,
    aggregate_scores,
    extract_from_response_obj,
    extract_output_text,
    generate_comparison_pairs,
    get_prompt_key_from_input,
    parse_genrm_output,
)


logger = logging.getLogger(__name__)

ComparisonResult = Tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
]

# Cohort state for verify(): buffer by prompt_key until num_rollouts_per_prompt received (Difference 1)
_cohort_lock: asyncio.Lock = asyncio.Lock()
_cohort_buffers: Dict[str, List[Tuple[Any, asyncio.Future]]] = defaultdict(list)
_cohort_jit_buffers: Dict[str, Tuple[List[ComparisonResult], List[Tuple[int, int, int]]]] = defaultdict(
    lambda: ([], [])
)

class GenRMCompareConfig(BaseResourcesServerConfig):
    """Configuration for the GenRM compare server.

    Attributes:
        genrm_model_server: Target GenRM model server (default: genrm_model from config)
        genrm_responses_create_params: Base create params for GenRM calls
        comparison_mode: Compare rollouts to each other or to a fixed baseline
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
        score_source: "overall" or "rubric_mean"
        use_principle: Enable principle-based comparison
        default_principle: Default principle when none provided in request
    """

    name: str = "genrm_compare"
    genrm_model_server: ModelServerRef  # Default: genrm_model (see config)
    genrm_responses_create_params: NeMoGymResponseCreateParamsNonStreaming

    # Cohort-based verify: number of rollouts per prompt before running comparison (Difference 1)
    # When > 1, verify() buffers by prompt and runs comparison when cohort is full; rewards are relative to cohort.
    # When <= 1, verify() returns default_score (no comparison).
    num_rollouts_per_prompt: int = 1

    # Comparison strategy
    comparison_mode: Literal["rollout_cohort", "fixed_baseline"] = "rollout_cohort"
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
    score_source: Literal["overall", "rubric_mean"] = "overall"

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


class GenRMCompareVerifyRequest(BaseVerifyRequest):
    """Verify request with optional principle for cohort-based GenRM comparison."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    principle: Optional[str] = None  # Principle for principle-based GenRM; forwarded by agent when provided
    expected_rubric_ids: Optional[Tuple[int, ...]] = None
    task_index: Optional[int] = Field(default=None, alias=TASK_INDEX_KEY_NAME)
    rollout_index: Optional[int] = Field(default=None, alias=ROLLOUT_INDEX_KEY_NAME)
    prompt_id: Optional[str] = None  # Optional stable prompt identifier from the caller


class GenRMCompareVerifyResponse(BaseVerifyResponse):
    reasoning_text: str
    answer_text: str
    reward_score_raw: float
    reward_rubric_mean_clean: Optional[float] = None
    reward_overall_raw: float
    reward_overall_len_adjusted: float
    reward_length_adjustment: float
    genrm_parse_failure_rate_per_group: float = 0.0
    genrm_rubric_parse_failure_rate_per_group: float = 0.0
    genrm_api_error_rate_per_group: float = 0.0
    genrm_input_tokens_per_comparison_mean: Optional[float] = None
    genrm_input_tokens_per_comparison_p50: Optional[float] = None
    genrm_input_tokens_per_comparison_p95: Optional[float] = None
    genrm_output_tokens_per_comparison_mean: Optional[float] = None
    genrm_output_tokens_per_comparison_p50: Optional[float] = None
    genrm_output_tokens_per_comparison_p95: Optional[float] = None
    genrm_output_tokens_total_per_group: Optional[float] = None
    genrm_max_output_tokens_hit_rate_per_group: Optional[float] = None


class GenRMCompareRequest(BaseModel):
    """Request payload for GenRM pairwise comparison."""

    conversation_history: List[Dict[str, str]]  # User/assistant messages before the responses
    response_objs: List[Dict[str, Any]]  # Raw Response API objects from policy model
    principle: Optional[str] = None  # Principle for principle-based GenRM (e.g., "The response should be helpful")
    expected_rubric_ids: Optional[Tuple[int, ...]] = None


class GenRMCompareResponse(BaseModel):
    """Response payload with per-response rewards."""

    rewards: List[float]  # One reward per response, in same order as input
    comparison_results: Optional[List[Dict[str, Any]]] = None  # Detailed pairwise results
    metrics: Optional[Dict[str, float]] = None  # Aggregation metrics


def _input_to_conversation_history(input_messages: Any) -> List[Dict[str, str]]:
    """Convert Response API input messages to conversation_history list of {role, content}."""
    out: List[Dict[str, str]] = []
    items = list(input_messages) if input_messages else []
    for m in items:
        if isinstance(m, dict):
            role = m.get("role", "user")
            content = m.get("content", "")
        else:
            role = getattr(m, "role", "user")
            content = getattr(m, "content", "") or ""
        if isinstance(content, list):
            content = "".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "output_text"
            )
        out.append({"role": str(role), "content": str(content)})
    return out


class GenRMCompareResourcesServer(SimpleResourcesServer):
    """Resources server for GenRM pairwise comparison of multiple responses.

    Supports two modes:
    - Cohort-based verify (Difference 1): When num_rollouts_per_prompt > 1, verify() buffers by prompt;
      when the cohort is full, runs comparison and returns per-rollout rewards. Callers await until
      their cohort is complete and get their reward.
    - Batch /compare: Direct comparison of N response_objs (e.g. for rollout_collection or tests).
    """

    config: GenRMCompareConfig

    async def verify(self, body: GenRMCompareVerifyRequest) -> GenRMCompareVerifyResponse:
        """Buffer one rollout and return its cohort-relative reward."""
        cfg = self.config
        principle = body.principle
        expected_rubric_ids = body.expected_rubric_ids
        response_obj = (
            body.response.model_dump()
            if hasattr(body.response, "model_dump")
            else body.response
        )
        reasoning_text, answer_text = extract_from_response_obj(response_obj)
        if cfg.score_source == "rubric_mean" and not expected_rubric_ids:
            raise ValueError("score_source=rubric_mean requires expected_rubric_ids")
        if cfg.num_rollouts_per_prompt <= 1:
            return GenRMCompareVerifyResponse(
                responses_create_params=body.responses_create_params,
                response=body.response,
                reasoning_text=reasoning_text,
                answer_text=answer_text,
                reward=cfg.default_score,
                reward_score_raw=cfg.default_score,
                reward_overall_raw=cfg.default_score,
                reward_overall_len_adjusted=cfg.default_score,
                reward_length_adjustment=0.0,
            )

        input_messages = getattr(body.responses_create_params, "input", None) or []
        baseline_response = None
        if cfg.comparison_mode == "fixed_baseline":
            baseline_response = (body.responses_create_params.metadata or {}).get("baseline_response")
            if not isinstance(baseline_response, str) or not baseline_response.strip():
                raise ValueError("comparison_mode=fixed_baseline requires metadata.baseline_response")
        prompt_key = self._get_verify_cohort_key(
            body,
            input_messages if isinstance(input_messages, list) else list(input_messages),
            principle,
        )
        if baseline_response is not None:
            # Never mix the same prompt evaluated against different fixed baselines.
            prompt_key = f"{prompt_key}:{hashlib.sha256(baseline_response.encode()).hexdigest()}"
        if expected_rubric_ids is not None:
            # Keep cohorts with different rubric contracts separate.
            prompt_key = f"{prompt_key}:rubrics:{','.join(map(str, expected_rubric_ids))}"
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()

        _cohort_buffers[prompt_key].append((body, future))

        conversation_history = _input_to_conversation_history(getattr(body.responses_create_params, "input", []) or [])
        buf = _cohort_buffers[prompt_key]
        response_objs = [
            (b.response.model_dump() if hasattr(b.response, "model_dump") else b.response) for b, _ in buf
        ]
        principle_val = getattr(body, "principle", None) or principle

        if cfg.comparison_mode == "rollout_cohort":
            existing_results, existing_metadata = _cohort_jit_buffers[prompt_key]
            new_results, new_metadata = await self._run_jit_compare_using_most_recent_response_obj(
                conversation_history,
                response_objs,
                existing_metadata,
                principle_val,
                expected_rubric_ids,
            )
            existing_results.extend(new_results)
            existing_metadata.extend(new_metadata)

        cohort_ready = False
        if len(response_objs) >= cfg.num_rollouts_per_prompt:
            assert len(response_objs) == cfg.num_rollouts_per_prompt
            cohort_ready = True

        if cohort_ready:
            cohort_buf = _cohort_buffers.pop(prompt_key)
            try:
                if cfg.comparison_mode == "rollout_cohort":
                    existing_results, existing_metadata = _cohort_jit_buffers.pop(prompt_key)
                    ordered = sorted(
                        zip(existing_results, existing_metadata),
                        key=lambda pair: (pair[1][2], pair[1][0], pair[1][1]),
                    )
                    results, metadata = zip(*ordered)
                    compare_result = self._aggregate_results(response_objs, results, metadata)
                else:
                    compare_result = await self._run_fixed_baseline_compare(
                        conversation_history,
                        response_objs,
                        baseline_response,
                        principle_val,
                        prompt_key,
                        expected_rubric_ids,
                    )
                (
                    rewards,
                    raw_scores,
                    clean_scores,
                    metrics,
                    _,
                    overall_raw,
                    overall_adjusted,
                    length_adjustments,
                ) = compare_result
            except Exception as error:
                for _, pending in cohort_buf:
                    if not pending.done():
                        pending.set_exception(error)
                raise

            for i, (_, f) in enumerate(cohort_buf):
                if not f.done():
                    f.set_result(
                        {
                            "reward": rewards[i],
                            "reward_score_raw": raw_scores[i],
                            "reward_rubric_mean_clean": clean_scores[i],
                            "reward_overall_raw": overall_raw[i],
                            "reward_overall_len_adjusted": overall_adjusted[i],
                            "reward_length_adjustment": length_adjustments[i],
                            **metrics,
                        }
                    )

        result = await future
        return GenRMCompareVerifyResponse(
            responses_create_params=body.responses_create_params,
            response=body.response,
            reasoning_text=reasoning_text,
            answer_text=answer_text,
            **result,
        )

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/compare")(self.compare)
        return app

    def _aggregate_results(
        self,
        response_objs: List[Dict[str, Any]],
        raw_results: List[ComparisonResult],
        metadata: List[Tuple[int, int, int]],
        trainable_count: Optional[int] = None,
    ) -> tuple:
        cfg = self.config
        # Train on the configured score source while retaining overall scores as diagnostics.
        comparisons = [result[:3] for result in raw_results]
        overall_comparisons = [result[3:6] for result in raw_results]
        count = trainable_count if trainable_count is not None else len(response_objs)

        def aggregate(
            results: List[Tuple[float, float, float]],
            result_metadata: List[Tuple[int, int, int]],
            adjust: bool = False,
        ) -> tuple:
            return aggregate_scores(
                comparison_results=results,
                comparison_metadata=result_metadata,
                response_objs=response_objs,
                aggregator_method=cfg.aggregator_method,
                default_score=cfg.default_score,
                reasoning_bonus=cfg.reasoning_bonus if adjust else 0.0,
                answer_bonus=cfg.answer_bonus if adjust else 0.0,
                top_percentile=cfg.top_percentile,
                group_reasoning_length_penalty_coeff=(cfg.group_reasoning_length_penalty_coeff if adjust else 0.0),
                group_answer_length_penalty_coeff=(cfg.group_answer_length_penalty_coeff if adjust else 0.0),
                group_style_penalty_coeff=cfg.group_style_penalty_coeff if adjust else 0.0,
                adjustment_count=count,
            )

        rewards, _, raw_scores, _ = aggregate(comparisons, metadata, adjust=True)
        overall_adjusted, _, overall_raw, length_adjustments = aggregate(overall_comparisons, metadata, adjust=True)
        rewards, raw_scores = rewards[:count], raw_scores[:count]
        overall_raw, overall_adjusted = overall_raw[:count], overall_adjusted[:count]
        length_adjustments = length_adjustments[:count]
        clean_scores: List[Optional[float]] = [None] * count
        if cfg.score_source == "rubric_mean":
            # A clean rubric metric requires every comparison touching that response to parse.
            valid = [not rubric_failed and not api_error for *_, rubric_failed, api_error in raw_results]
            valid_results = [comparison for comparison, keep in zip(comparisons, valid) if keep]
            valid_metadata = [item for item, keep in zip(metadata, valid) if keep]
            failed_indices = {idx for item, keep in zip(metadata, valid) if not keep for idx in item[:2]}
            if valid_results:
                clean_values = aggregate(valid_results, valid_metadata)[2]
                clean_scores = [clean_values[idx] if idx not in failed_indices else None for idx in range(count)]
        total = max(1, len(raw_results))
        metrics = {
            "genrm_parse_failure_rate_per_group": sum(result[-3] for result in raw_results) / total,
            "genrm_rubric_parse_failure_rate_per_group": (
                sum(result[-2] for result in raw_results) / total if cfg.score_source == "rubric_mean" else 0.0
            ),
            "genrm_api_error_rate_per_group": sum(result[-1] for result in raw_results) / total,
        }
        token_usage = [
            result[6:9]
            for result in raw_results
            if len(result) >= 12 and result[6] >= 0 and result[7] >= 0
        ]
        if token_usage:
            input_tokens = [usage[0] for usage in token_usage]
            output_tokens = [usage[1] for usage in token_usage]

            def percentile(values: List[float], fraction: float) -> float:
                ordered = sorted(values)
                position = (len(ordered) - 1) * fraction
                lower = int(position)
                upper = min(lower + 1, len(ordered) - 1)
                return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

            metrics.update(
                {
                    "genrm_input_tokens_per_comparison_mean": sum(input_tokens) / len(input_tokens),
                    "genrm_input_tokens_per_comparison_p50": percentile(input_tokens, 0.50),
                    "genrm_input_tokens_per_comparison_p95": percentile(input_tokens, 0.95),
                    "genrm_output_tokens_per_comparison_mean": sum(output_tokens) / len(output_tokens),
                    "genrm_output_tokens_per_comparison_p50": percentile(output_tokens, 0.50),
                    "genrm_output_tokens_per_comparison_p95": percentile(output_tokens, 0.95),
                    "genrm_output_tokens_total_per_group": sum(output_tokens),
                    "genrm_max_output_tokens_hit_rate_per_group": (
                        sum(usage[2] for usage in token_usage) / len(token_usage)
                    ),
                }
            )
        return (
            rewards,
            raw_scores,
            clean_scores,
            metrics,
            comparisons,
            overall_raw,
            overall_adjusted,
            length_adjustments,
        )

    def _get_verify_cohort_key(
        self,
        body: GenRMCompareVerifyRequest,
        input_messages: List[Any],
        principle: Optional[str] = None,
    ) -> str:
        """Prefer task-scoped keys when available so identical prompt text from different tasks does not collide."""
        prompt_key = get_prompt_key_from_input(input_messages, principle)
        if body.task_index is not None:
            return f"task_idx::{body.task_index}::{prompt_key}"
        if body.prompt_id is not None:
            return f"prompt_id::{body.prompt_id}::{prompt_key}"
        return prompt_key

    async def _run_jit_compare_using_most_recent_response_obj(
        self,
        conversation_history: List[Dict[str, str]],
        response_objs: List[Dict[str, Any]],
        seen_comparison_metadata: List[Tuple[int, int, int]],
        principle: Optional[str] = None,
        expected_rubric_ids: Optional[Tuple[int, ...]] = None,
    ) -> Tuple[List[ComparisonResult], List[Tuple[int, int, int]]]:
        # Cannot run comparison with only 1 result
        if len(response_objs) == 1:
            return [], []

        cfg = self.config
        this_response_idx = len(response_objs) - 1

        comparison_pairs = generate_comparison_pairs(cfg.comparison_strategy, cfg.num_rollouts_per_prompt)
        comparison_tasks = []
        comparison_metadata: List[Tuple[int, int, int]] = []
        for judge_idx in range(cfg.num_judges_per_comparison):
            for i, j in comparison_pairs:
                # If one of the indices has not yet been run, continue
                if not (i < len(response_objs) and j < len(response_objs)):
                    continue

                # At least one of the indices must be this index
                if i != this_response_idx and j != this_response_idx:
                    continue

                this_comparison_metadata = (i, j, judge_idx)

                # Don't double count since this will trigger when both i and j are finished.
                if this_comparison_metadata in seen_comparison_metadata:
                    continue

                comparison_tasks.append(
                    self._run_single_comparison(
                        conversation_history,
                        response_objs[i],
                        response_objs[j],
                        pair_idx=(i, j),
                        principle=principle,
                        expected_rubric_ids=expected_rubric_ids,
                    )
                )
                comparison_metadata.append(this_comparison_metadata)

        comparison_results = await asyncio.gather(*comparison_tasks)

        return comparison_results, comparison_metadata

    async def _run_compare(
        self,
        conversation_history: List[Dict[str, str]],
        response_objs: List[Dict[str, Any]],
        principle: Optional[str] = None,
        expected_rubric_ids: Optional[Tuple[int, ...]] = None,
    ) -> Tuple[List[float], Dict[str, float], List[Tuple[float, float, float]], List[Tuple[int, int, int]]]:
        """Run pairwise comparison; return rewards, metrics, results, and metadata."""
        cfg = self.config
        num_responses = len(response_objs)
        if num_responses < 2:
            return [cfg.default_score] * num_responses, {}, [], []

        comparison_pairs = generate_comparison_pairs(cfg.comparison_strategy, num_responses)
        comparison_tasks = []
        comparison_metadata: List[Tuple[int, int, int]] = []
        for judge_idx in range(cfg.num_judges_per_comparison):
            for i, j in comparison_pairs:
                comparison_tasks.append(
                    self._run_single_comparison(
                        conversation_history,
                        response_objs[i],
                        response_objs[j],
                        pair_idx=(i, j),
                        principle=principle,
                        expected_rubric_ids=expected_rubric_ids,
                    )
                )
                comparison_metadata.append((i, j, judge_idx))
        raw_results = list(await asyncio.gather(*comparison_tasks))
        comparison_results = [result[:3] for result in raw_results]
        rewards, metrics, _, _ = aggregate_scores(
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
        return rewards, metrics, comparison_results, comparison_metadata

    async def _run_fixed_baseline_compare(
        self,
        conversation_history: List[Dict[str, str]],
        response_objs: List[Dict[str, Any]],
        baseline_response: str,
        principle: Optional[str],
        prompt_key: str,
        expected_rubric_ids: Optional[Tuple[int, ...]] = None,
    ) -> tuple:
        cfg = self.config
        baseline_idx = len(response_objs)
        baseline_obj = {
            "output": [{"type": "message", "content": [{"type": "output_text", "text": baseline_response}]}]
        }

        async def compare_one(response_idx: int, response_obj: Dict[str, Any], judge_idx: int) -> ComparisonResult:
            # Deterministically choose the first order, then alternate judges to reduce position bias.
            order = int(hashlib.sha256(f"{prompt_key}:{response_idx}".encode()).hexdigest(), 16)
            rollout_first = (order + judge_idx) % 2 == 0
            first, second = (response_obj, baseline_obj) if rollout_first else (baseline_obj, response_obj)
            result = await self._run_single_comparison(
                conversation_history,
                first,
                second,
                pair_idx=(response_idx, baseline_idx),
                principle=principle,
                expected_rubric_ids=expected_rubric_ids,
            )
            if rollout_first:
                return result
            score_1, score_2, ranking, overall_1, overall_2, overall_ranking, *flags = result
            return score_2, score_1, 7.0 - ranking, overall_2, overall_1, 7.0 - overall_ranking, *flags

        comparison_tasks = []
        metadata = []
        for response_idx, response in enumerate(response_objs):
            for judge_idx in range(cfg.num_judges_per_comparison):
                comparison_tasks.append(compare_one(response_idx, response, judge_idx))
                metadata.append((response_idx, baseline_idx, judge_idx))
        return self._aggregate_results(
            response_objs + [baseline_obj],
            list(await asyncio.gather(*comparison_tasks)),
            metadata,
            trainable_count=len(response_objs),
        )

    async def compare(self, body: GenRMCompareRequest) -> GenRMCompareResponse:
        """Compare multiple responses using GenRM pairwise comparisons (batch API)."""
        cfg = self.config
        response_objs = body.response_objs
        conversation_history = body.conversation_history
        num_responses = len(response_objs)
        if cfg.debug_logging:
            logger.info(f"[GenRM] Compare request: {num_responses} responses")
        if num_responses < 2:
            return GenRMCompareResponse(
                rewards=[cfg.default_score],
                comparison_results=None,
                metrics=None,
            )
        rewards, metrics, comparison_results, comparison_metadata = await self._run_compare(
            conversation_history,
            response_objs,
            principle=body.principle,
            expected_rubric_ids=body.expected_rubric_ids,
        )
        detailed_results = [
            {
                "response_i": i,
                "response_j": j,
                "judge_idx": judge_idx,
                "score_1": score_1,
                "score_2": score_2,
                "ranking": ranking,
            }
            for (score_1, score_2, ranking), (i, j, judge_idx) in zip(comparison_results, comparison_metadata)
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
        expected_rubric_ids: Optional[Tuple[int, ...]] = None,
    ) -> ComparisonResult:
        """Run a single pairwise comparison via GenRM.

        Args:
            conversation_history: The conversation context
            response_obj_1: First Response API object
            response_obj_2: Second Response API object
            pair_idx: Tuple of (i, j) for logging
            principle: Optional principle for principle-based comparison

        Returns:
            Selected scores, overall scores, and parse/API failure flags.
        """
        cfg = self.config
        if cfg.score_source == "rubric_mean" and not expected_rubric_ids:
            raise ValueError("score_source=rubric_mean requires expected_rubric_ids")

        # Extract final answer from Response API objects (GenRM only takes the final answer, not reasoning)
        response_1 = extract_output_text(response_obj_1)
        response_2 = extract_output_text(response_obj_2)

        # input carries only the conversation history (standard OpenAI roles).
        # The comparison payload is passed via metadata so the request schema stays
        # generic and GenRMModelMixin._preprocess_chat_completion_create_params can
        # inject the GenRM-specific roles (response_1, response_2, principle) server-side.
        messages: List[NeMoGymEasyInputMessage] = [
            NeMoGymEasyInputMessage(
                role=msg.get("role", "user"),
                content=msg.get("content", ""),
                type="message",
            )
            for msg in conversation_history
        ]

        metadata = {"response_1": response_1, "response_2": response_2}
        if cfg.use_principle:
            metadata["principle"] = principle if principle else cfg.default_principle

        # Build the request params
        responses_create_params = cfg.genrm_responses_create_params.model_copy(deep=True)
        responses_create_params.input = messages
        responses_create_params.metadata = metadata

        input_tokens = 0.0
        output_tokens = 0.0
        max_output_tokens_hit = 0.0
        usage_available = False
        try:
            # Retry logic for parse failures (not connection errors, which are handled elsewhere)
            max_attempts = max(1, int(cfg.genrm_parse_retries) + 1)
            for attempt_idx in range(max_attempts):
                # Call the GenRM model via /v1/responses endpoint (server name from config, e.g. genrm_model)
                response = await self.server_client.post(
                    server_name=cfg.genrm_model_server.name,
                    url_path="/v1/responses",
                    json=responses_create_params,
                )
                raw_response = await response.json()

                # Include retry attempts because each one consumes GenRM capacity.
                usage = raw_response.get("usage") or {}
                attempt_input_tokens = usage.get(
                    "input_tokens", usage.get("prompt_tokens")
                )
                attempt_output_tokens = usage.get(
                    "output_tokens", usage.get("completion_tokens")
                )
                if (
                    isinstance(attempt_input_tokens, (int, float))
                    and not isinstance(attempt_input_tokens, bool)
                    and isinstance(attempt_output_tokens, (int, float))
                    and not isinstance(attempt_output_tokens, bool)
                ):
                    input_tokens += float(attempt_input_tokens)
                    output_tokens += float(attempt_output_tokens)
                    usage_available = True
                if (raw_response.get("incomplete_details") or {}).get("reason") == "max_output_tokens":
                    max_output_tokens_hit = 1.0

                # Extract output_text from GenRM response (skip reasoning, only parse the final JSON scores)
                genrm_answer = extract_output_text(raw_response)
                try:
                    overall = parse_genrm_output(
                        genrm_answer,
                        cfg.default_score,
                        cfg.default_ranking,
                        score_source="overall",
                        raise_on_fail=True,
                    )
                    overall_failed = 0.0
                except GenRMOutputParseError:
                    overall = (cfg.default_score, cfg.default_score, cfg.default_ranking)
                    overall_failed = 1.0

                rubric_failed = 0.0
                if cfg.score_source == "rubric_mean":
                    try:
                        selected = parse_genrm_output(
                            genrm_answer,
                            cfg.default_score,
                            cfg.default_ranking,
                            score_source="rubric_mean",
                            expected_rubric_ids=expected_rubric_ids,
                            raise_on_fail=True,
                        )
                    except GenRMOutputParseError:
                        selected = (cfg.default_score, cfg.default_score, cfg.default_ranking)
                        rubric_failed = 1.0
                    selected_failed = rubric_failed
                else:
                    selected = overall
                    selected_failed = overall_failed

                token_metrics = (
                    (input_tokens, output_tokens, max_output_tokens_hit)
                    if usage_available
                    else (-1.0, -1.0, -1.0)
                )
                if not selected_failed:
                    return (*selected, *overall, *token_metrics, overall_failed, rubric_failed, 0.0)

                if attempt_idx < max_attempts - 1:
                    await asyncio.sleep(float(cfg.genrm_parse_retry_sleep_s))
                    continue

                logger.warning(
                    f"[GenRM] {cfg.score_source} parse failed for pair {pair_idx} after "
                    f"{max_attempts} attempts; falling back to defaults."
                )
                return (*selected, *overall, *token_metrics, overall_failed, rubric_failed, 0.0)

        except Exception as e:
            logger.error(f"[GenRM] Error in comparison for pair {pair_idx}: {e}")
            neutral = (cfg.default_score, cfg.default_score, cfg.default_ranking)
            token_metrics = (
                (input_tokens, output_tokens, max_output_tokens_hit)
                if usage_available
                else (-1.0, -1.0, -1.0)
            )
            return (*neutral, *neutral, *token_metrics, 0.0, 0.0, 1.0)


if __name__ == "__main__":
    GenRMCompareResourcesServer.run_webserver()
