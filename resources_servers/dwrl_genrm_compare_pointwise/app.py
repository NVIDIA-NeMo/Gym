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

import hashlib
import json
import re
import asyncio
import math
import logging
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field, model_validator

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
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymChatCompletionSystemMessageParam,
    NeMoGymChatCompletionUserMessageParam,
    NeMoGymChatCompletionAssistantMessageParam,
)
from resources_servers.dwrl_genrm_compare_pointwise.cohort import Cohort, CohortRegistry, MemberJudgmentFailed
from resources_servers.dwrl_genrm_compare_pointwise.utils import (
    GenRMOutputParseError,
    aggregate_scores,
    extract_output_text,
    generate_comparison_pairs,
    get_prompt_key_from_input,
    parse_genrm_output,
)


logger = logging.getLogger(__name__)

# Cohort state for verify(): rollouts are grouped by prompt_key until num_rollouts_per_prompt have arrived.
_cohorts = CohortRegistry()


def _response_digest(response_obj: Any) -> str:
    """Fingerprint of a rollout's response, to tell a repeated request from a different rollout with the same index."""
    return hashlib.sha256(json.dumps(response_obj, sort_keys=True, default=str).encode()).hexdigest()


def flatten_to_single_turn(message_log):
    ret = ""
    for idx, message in enumerate(message_log):
        if message.role == "system":
            ret += "System: " + message.content.strip() + "\n\n"
        elif message.role == "user":
            ret += "User: " + message.content.strip() + "\n\n"
        elif message.role == "assistant":
            resp_no_thinking1 = re.sub(r"(?i)(?s)(\<think\>)(.*?)(\<\/think\>)", "", message.content).strip()
            resp_no_thinking2 = re.sub(r"(?i)(?s)(.*?)(\<\/think\>)", "", resp_no_thinking1).strip()
            ret += "Assistant: " + resp_no_thinking2 + "\n\n"
    
    return ret.strip()

def convert_to_chat_completion_format(message_log):
    ret = []
    for idx, message in enumerate(message_log):
        if message.role == "system":
            ret.append( NeMoGymChatCompletionSystemMessageParam(role="system", content=message.content) )
        elif message.role == "user":
            ret.append( NeMoGymChatCompletionUserMessageParam(role="user", content=message.content) )
        elif message.role == "assistant":
            ret.append( NeMoGymChatCompletionAssistantMessageParam(role="assistant", content=message.content) )
    
    return ret

class GenRMCompareConfig(BaseResourcesServerConfig):
    """Configuration for the GenRM compare server.

    Attributes:
        genrm_model_server: Target GenRM model server (default: genrm_model from config)
        genrm_responses_create_params: Base create params for GenRM calls
        comparison_strategy: "all_pairs" or "circular"
        num_judges_per_comparison: Number of judge passes per pair (majority voting)
        aggregator_method: Method for aggregating scores
        reasoning_bonus: Bonus for shortest reasoning content among top performers
        answer_bonus: Bonus for shortest answer among top performers
        top_percentile: Percentile threshold for applying bonuses
        group_reasoning_length_penalty_coeff: Coefficient for reasoning length penalty
        group_answer_length_penalty_coeff: Coefficient for answer length penalty
        default_score: Default neutral score when parsing fails
        default_ranking: Default neutral ranking when parsing fails
        debug_logging: Enable verbose logging for debugging
        genrm_parse_retries: Number of retries on parse failures
        genrm_parse_retry_sleep_s: Sleep duration between parse retries
        use_principle: Enable principle-based comparison
        default_principle: Default principle when none provided in request
    """

    name: str = "genrm_compare"
    genrm_model_server: ModelServerRef  # Default: genrm_model (see config)
    #genrm_responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    genrm_responses_create_params: NeMoGymChatCompletionCreateParamsNonStreaming

    # Cohort-based verify: number of rollouts per prompt before running comparison (Difference 1)
    # When > 1, verify() buffers by prompt and runs comparison when cohort is full; rewards are relative to cohort.
    # When <= 1, verify() returns default_score (no comparison).
    num_rollouts_per_prompt: int = 1

    # Comparison strategy
    #comparison_strategy: str = "circular"  # "all_pairs" or "circular"
    num_judges_per_comparison: int = Field(default=1, ge=1)

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

    # Default neutral scores when parsing fails
    default_score: float = 3.0
    default_ranking: float = 0.5

    # Debug logging
    debug_logging: bool = False

    # Retry config for parse failures
    genrm_parse_retries: int = 3
    genrm_parse_retry_sleep_s: float = 0.2
    
    # DWRL specific
    score_token: str
    bt_prompt: str
    dwrl_prompt: str

    @model_validator(mode="after")
    def _cohorts_need_a_single_worker_process(self) -> "GenRMCompareConfig":
        # The rollouts of a prompt are grouped in memory (see cohort.py). With several uvicorn worker processes they
        # would be spread over separate groups that never fill up, and every verify() call would wait forever.
        if self.num_rollouts_per_prompt > 1 and self.num_workers is not None and self.num_workers > 1:
            raise ValueError(
                f"num_workers must be 1 when num_rollouts_per_prompt > 1 (got num_workers={self.num_workers}): "
                "rollouts of one prompt have to be handled by the same process."
            )
        return self


class GenRMCompareVerifyRequest(BaseVerifyRequest):
    """Verify request with optional principle for cohort-based GenRM comparison."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    principle: Optional[str] = None  # Principle for principle-based GenRM; forwarded by agent when provided
    task_index: Optional[int] = Field(default=None, alias=TASK_INDEX_KEY_NAME)
    rollout_index: Optional[int] = Field(default=None, alias=ROLLOUT_INDEX_KEY_NAME)
    prompt_id: Optional[str] = None  # Optional stable prompt identifier from the caller


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

    async def verify(self, body: GenRMCompareVerifyRequest) -> BaseVerifyResponse:
        """Verify a single rollout.

        When num_rollouts_per_prompt > 1, rollouts of the same prompt are grouped into a cohort. Each rollout
        is judged as soon as it arrives, and the cohort's rewards are computed by whichever rollout finishes
        judging last (see cohort.py). Every caller awaits until its cohort is complete and gets its reward.

        A request that repeats one that is already being handled (same prompt, same `rollout_index` and the same
        response, e.g. a client retry) takes no slot in the cohort and is not judged again: it waits for the reward
        of the request it repeats. If that request failed, the repeat fails as well.
        """
        cfg = self.config
        principle = body.principle
        if cfg.num_rollouts_per_prompt <= 1:
            return BaseVerifyResponse(
                responses_create_params=body.responses_create_params,
                response=body.response,
                reward=cfg.default_score,
            )

        # Everything that can fail on malformed input is computed before joining the cohort, so a bad request
        # never leaves a member slot that is never filled.
        input_messages = getattr(body.responses_create_params, "input", None) or []
        prompt_key = self._get_verify_cohort_key(
            body,
            input_messages if isinstance(input_messages, list) else list(input_messages),
            principle,
        )
        conversation_history = _input_to_conversation_history(input_messages)
        response_obj = body.response.model_dump() if hasattr(body.response, "model_dump") else body.response

        # A rollout is identified by its index within its task, plus its response so that a different rollout that
        # reuses the index is not mistaken for a repeat. Without an index nothing is treated as a repeat.
        identity = None
        if body.rollout_index is not None:
            identity = (body.rollout_index, _response_digest(response_obj))

        cohort, member_idx, future, is_new = _cohorts.join(
            prompt_key,
            body,
            size=cfg.num_rollouts_per_prompt,
            num_judges=cfg.num_judges_per_comparison,
            identity=identity,
        )

        if not is_new:
            logger.warning(
                f"[GenRM] Repeated verify request for rollout_index={body.rollout_index} (task_index={body.task_index}, "
                f"prompt_id={body.prompt_id}); waiting for the reward of the request that is already being handled"
            )
            # Shielded: cancelling this request must not cancel the future that the other requests are waiting on.
            reward = await asyncio.shield(future)
            return BaseVerifyResponse(
                responses_create_params=body.responses_create_params,
                response=body.response,
                reward=reward,
            )

        error: Optional[BaseException] = None
        try:
            judge_results = await self._judge_rollout(conversation_history, response_obj, principle)
        except BaseException as e:  # Includes cancellation, e.g. the client disconnecting.
            # Fill this member's slot with neutral results so the rest of the cohort is not left waiting.
            error = e
            judge_results = [(cfg.default_score, cfg.default_ranking)] * cfg.num_judges_per_comparison

        if error is None:
            cohort.record(member_idx, judge_results)
        else:
            cohort.record_failure(member_idx, error, judge_results)
        if cohort.is_complete:
            self._finalize_cohort(cohort)

        if error is not None:
            raise error

        # Shielded, because a repeated request may be waiting on the same future (see above).
        reward = await asyncio.shield(future)
        return BaseVerifyResponse(
            responses_create_params=body.responses_create_params,
            response=body.response,
            reward=reward,
        )

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/compare")(self.compare)
        return app

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

    async def _judge_rollout(
        self,
        conversation_history: List[Dict[str, str]],
        response_obj: Dict[str, Any],
        principle: Optional[str] = None,
    ) -> List[Tuple[float, float]]:
        """Judge one rollout `num_judges_per_comparison` times, concurrently. Results are in judge order."""
        results = await asyncio.gather(
            *[
                self._run_single_comparison(conversation_history, response_obj, principle=principle)
                for _ in range(self.config.num_judges_per_comparison)
            ]
        )
        return list(results)

    def _finalize_cohort(self, cohort: Cohort) -> None:
        """Compute the rewards of a fully judged cohort and resolve each member's future.

        Called once, by the verify() call that records the last judge result. There is no await in here, so
        no other task can observe the cohort half-finished.
        """
        cfg = self.config
        # From here on a repeat of one of its members is no longer recognized: it starts a cohort of its own.
        _cohorts.finish(cohort)
        try:
            # Ordered by (judge_idx, member_idx), as in the original `_run_compare` logic.
            comparison_results, comparison_metadata = cohort.ordered()
            response_objs = [
                (b.response.model_dump() if hasattr(b.response, "model_dump") else b.response)
                for b in cohort.payloads
            ]
            rewards, _, _, _ = aggregate_scores(
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
            )
        except Exception as e:
            # Fail every member instead of leaving the other callers waiting forever.
            logger.exception("[GenRM] Failed to aggregate cohort scores")
            for f in cohort.futures:
                if not f.done():
                    f.set_exception(e)
                    f.exception()  # Mark as retrieved: not every future is awaited by someone.
            return

        for member_idx, (reward, f) in enumerate(zip(rewards, cohort.futures)):
            if f.done():
                continue
            if member_idx in cohort.failures:
                # The request that judged this rollout raised its own error. Only a repeat of that request can be
                # waiting here, and it must not be given the neutral reward that stands in for the missing judgment.
                f.set_exception(
                    MemberJudgmentFailed(f"The request judging this rollout failed: {cohort.failures[member_idx]!r}")
                )
                f.exception()  # Mark as retrieved: usually nobody is waiting on it.
            else:
                f.set_result(reward)

    async def _run_compare(
        self,
        conversation_history: List[Dict[str, str]],
        response_objs: List[Dict[str, Any]],
        principle: Optional[str] = None,
    ) -> Tuple[List[float], Dict[str, float], List[Tuple[float, float, float]], List[Tuple[int, int, int]]]:
        """Run pairwise comparison; return (rewards, metrics, comparison_results, comparison_metadata)."""
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
                    )
                )
                comparison_metadata.append((i, j, judge_idx))
        comparison_results = await asyncio.gather(*comparison_tasks)
        rewards, metrics, _, _ = aggregate_scores(
            comparison_results=list(comparison_results),
            comparison_metadata=comparison_metadata,
            response_objs=response_objs,
            aggregator_method=cfg.aggregator_method,
            default_score=cfg.default_score,
            reasoning_bonus=cfg.reasoning_bonus,
            answer_bonus=cfg.answer_bonus,
            top_percentile=cfg.top_percentile,
            group_reasoning_length_penalty_coeff=cfg.group_reasoning_length_penalty_coeff,
            group_answer_length_penalty_coeff=cfg.group_answer_length_penalty_coeff,
        )
        return rewards, metrics, list(comparison_results), comparison_metadata

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
            conversation_history, response_objs, principle=body.principle
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
        #response_obj_2: Dict[str, Any],
        #pair_idx: Tuple[int, int] = (0, 0),
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
        #response_2 = extract_output_text(response_obj_2)

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

        principle_val = principle if principle else cfg.default_principle.strip()
        #print("### PRINCIPLE_INSIDE_COMPARE: ", principle_val, flush=True)
        prompt_template = cfg.dwrl_prompt.strip().format(context=flatten_to_single_turn(messages), response=response_1, principle=principle_val)
        init_msg = [NeMoGymEasyInputMessage(role="user", content=prompt_template, type="message")]

        # Build the request params
        responses_create_params = cfg.genrm_responses_create_params.model_copy(deep=True)
        #responses_create_params.input = init_msg
        #responses_create_params = NeMoGymChatCompletionCreateParamsNonStreaming(messages=convert_to_chat_completion_format(init_msg))
        responses_create_params.messages = convert_to_chat_completion_format(init_msg)
        #responses_create_params.metadata = {"chat_template_kwargs": json.dumps({"enable_thinking": True}), "extra_body": json.dumps({"chat_template_kwargs": {"enable_thinking": True}})}
        #responses_create_params.model = "nvidia/nvidia/Nemotron-3-Nano-30B-A3B"
        #responses_create_params.top_p = 0.95
        #responses_create_params.temperature = 1.0
        #responses_create_params.max_tokens = 16384

        try:
            # Retry logic for parse failures (not connection errors, which are handled elsewhere)
            max_attempts = max(1, int(cfg.genrm_parse_retries) + 1)

            for attempt_idx in range(max_attempts):
                # Call the GenRM model via /v1/responses endpoint (server name from config, e.g. genrm_model)
                response_thought = await self.server_client.post(
                    server_name=cfg.genrm_model_server.name,
                    url_path="/v1/chat/completions",
                    json=responses_create_params,
                )
                raw_response_thought = await response_thought.json()
                #print("### RAW_RESPONSE_THOUGHT: ", raw_response_thought, flush=True)

                # Extract raw string response without any post processing
                #thought_raw_str = raw_response_thought['output'][0]['content'][0]['text']
                thought_raw_str = raw_response_thought['choices'][0]['message']['content']
				
                init_msg.append(NeMoGymEasyInputMessage(
                    role="assistant",
                    content=thought_raw_str,
                    type="message",
                ))
                init_msg.append(NeMoGymEasyInputMessage(
                    role="user",
                    content=cfg.bt_prompt,
                    type="message",
                ))
				
                responses_second_turn = cfg.genrm_responses_create_params.model_copy(deep=True)
                #responses_second_turn = NeMoGymChatCompletionCreateParamsNonStreaming(messages=convert_to_chat_completion_format(init_msg))
                #responses_second_turn.model = "model"
                #responses_second_turn.input = init_msg
                responses_second_turn.messages = convert_to_chat_completion_format(init_msg)
                responses_second_turn.logprobs = True
                responses_second_turn.metadata = {"chat_template_kwargs": json.dumps({"enable_thinking": False}), "extra_body": json.dumps({"chat_template_kwargs": {"enable_thinking": False}})}
                responses_second_turn.top_p = 1.0
                responses_second_turn.temperature = 1.0
                responses_second_turn.max_tokens = 1
                responses_second_turn.top_logprobs = 20
				
                response_score = await self.server_client.post(
                    server_name=cfg.genrm_model_server.name,
                    url_path="/v1/chat/completions",
                    json=responses_second_turn,
                )
                raw_response_score = await response_score.json()
                #print("### RAW_RESPONSE_SCORE: ", raw_response_score, flush=True)
                
                top_logprobs = raw_response_score['choices'][0]['logprobs']['content'][-1]['top_logprobs']
                score = None
                for possible_tokens in top_logprobs:
                    if possible_tokens['token'] == cfg.score_token:
                        score = math.exp(possible_tokens['logprob'])
                        break
                if score is None:
                    for possible_tokens in top_logprobs:
                        if possible_tokens['token'].lower().strip() == cfg.score_token:
                            score = math.exp(possible_tokens['logprob'])
                            break
                if score is None:
                    # A token outside the top-k has a probability of at most 1/k, so 0 is the right order of magnitude.
                    # This is also what happens, for every rollout, if `score_token` does not match the model's tokens.
                    logger.warning(
                        f"[GenRM] Score token {cfg.score_token!r} not found in the top {len(top_logprobs)} logprobs "
                        f"{[t['token'] for t in top_logprobs]}; using a BT score of 0.0"
                    )
                    score = 0.0
                print("### BT_SCORE: ", score, flush=True)

                try:
                    resp_no_thinking1 = re.sub(r"(?i)(?s)(\<think\>)(.*?)(\<\/think\>)", "", thought_raw_str).strip()
                    resp_no_thinking2 = re.sub(r"(?i)(?s)(.*?)(\<\/think\>)", "", resp_no_thinking1).strip()
                    
                    score_1 = parse_genrm_output(
                        resp_no_thinking2,
                        cfg.default_score,
                        raise_on_fail=False,
                    )
                    print(f"Score_1: {score_1}", flush=True)
                    return score_1, score

                except GenRMOutputParseError as e:
                    if attempt_idx < max_attempts - 1:
                        await asyncio.sleep(float(cfg.genrm_parse_retry_sleep_s))
                        continue

                    # Give up: fall back to defaults
                    logger.warning(
                        f"[GenRM] Parse failed after {max_attempts} attempts; "
                        f"falling back to defaults. {e}"
                    )
                    return cfg.default_score, score

            return cfg.default_score, cfg.default_ranking

        except Exception as e:
            logger.error(f"[GenRM] Error in comparison: {e}")
            return cfg.default_score, cfg.default_ranking


if __name__ == "__main__":
    GenRMCompareResourcesServer.run_webserver()
