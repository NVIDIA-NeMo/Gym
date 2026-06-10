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

import json
import re
import asyncio
import math
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

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
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymChatCompletionSystemMessageParam,
    NeMoGymChatCompletionUserMessageParam,
    NeMoGymChatCompletionAssistantMessageParam,
)
from resources_servers.dwrl_genrm_compare_pairwise.utils import (
    GenRMOutputParseError,
    aggregate_scores,
    extract_output_text,
    generate_comparison_pairs,
    get_prompt_key_from_input,
    parse_genrm_output,
)


logger = logging.getLogger(__name__)

# Cohort state for verify(): buffer by prompt_key until num_rollouts_per_prompt received (Difference 1)
_cohort_lock: asyncio.Lock = asyncio.Lock()
_cohort_buffers: Dict[str, List[Tuple[Any, asyncio.Future]]] = defaultdict(list)
_cohort_jit_buffers: Dict[str, Tuple[List[float, float, float], List[int, int, int]]] = defaultdict(lambda: ([], []))


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

    # Default neutral scores when parsing fails
    default_score: float = 3.0
    default_ranking: float = 3.5

    # Debug logging
    debug_logging: bool = False

    # Retry config for parse failures
    genrm_parse_retries: int = 3
    genrm_parse_retry_sleep_s: float = 0.2
    
    # DWRL specific
    score_token: str = "1"
    bt_prompt: str = None
    dwrl_prompt: str


class GenRMCompareVerifyRequest(BaseVerifyRequest):
    """Verify request with optional principle for cohort-based GenRM comparison."""

    principle: Optional[str] = None  # Principle for principle-based GenRM; forwarded by agent when provided


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
        """Verify a single rollout. When num_rollouts_per_prompt > 1, buffers by prompt and runs comparison when cohort is full."""
        cfg = self.config
        principle = body.principle
        if cfg.num_rollouts_per_prompt <= 1:
            return BaseVerifyResponse(
                responses_create_params=body.responses_create_params,
                response=body.response,
                reward=cfg.default_score,
            )

        input_messages = getattr(body.responses_create_params, "input", None) or []
        prompt_key = get_prompt_key_from_input(
            input_messages if isinstance(input_messages, list) else list(input_messages),
            principle,
        )
        future: asyncio.Future[float] = asyncio.get_running_loop().create_future()

        _cohort_buffers[prompt_key].append((body, future))

        conversation_history = _input_to_conversation_history(getattr(body.responses_create_params, "input", []) or [])
        buf = _cohort_buffers[prompt_key]
        response_objs = [
            (b.response.model_dump() if hasattr(b.response, "model_dump") else b.response) for b, _ in buf
        ]
        principle_val = getattr(body, "principle", None) or principle

        existing_results, existing_metadata = _cohort_jit_buffers[prompt_key]
        new_results, new_metadata = await self._run_jit_compare_using_most_recent_response_obj(
            conversation_history, response_objs, existing_metadata, principle_val
        )
        existing_results.extend(new_results)
        existing_metadata.extend(new_metadata)

        cohort_ready = False
        if len(response_objs) >= cfg.num_rollouts_per_prompt:
            assert len(response_objs) == cfg.num_rollouts_per_prompt
            cohort_ready = True

        # Only run for the final response
        if cohort_ready:
            existing_results, existing_metadata = _cohort_jit_buffers.pop(prompt_key)

            # Sort to match the ordering of the original `_run_compare` logic
            existing_results, existing_metadata = zip(
                *sorted(
                    zip(existing_results, existing_metadata), key=lambda pair: (pair[1][2], pair[1][0], pair[1][1])
                )
            )

            rewards, _, _, _ = aggregate_scores(
                comparison_results=existing_results,
                comparison_metadata=existing_metadata,
                response_objs=response_objs,
                aggregator_method=cfg.aggregator_method,
                default_score=cfg.default_score,
                reasoning_bonus=cfg.reasoning_bonus,
                answer_bonus=cfg.answer_bonus,
                top_percentile=cfg.top_percentile,
                group_reasoning_length_penalty_coeff=cfg.group_reasoning_length_penalty_coeff,
                group_answer_length_penalty_coeff=cfg.group_answer_length_penalty_coeff,
            )

            cohort_buf = _cohort_buffers.pop(prompt_key)
            for i, (_, f) in enumerate(cohort_buf):
                if not f.done():
                    f.set_result(rewards[i])

        reward = await future
        return BaseVerifyResponse(
            responses_create_params=body.responses_create_params,
            response=body.response,
            reward=reward,
        )

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/compare")(self.compare)
        return app

    async def _run_jit_compare_using_most_recent_response_obj(
        self,
        conversation_history: List[Dict[str, str]],
        response_objs: List[Dict[str, Any]],
        seen_comparison_metadata: List[Tuple[int, int, int]],
        principle: Optional[str] = None,
    ) -> Tuple[List[Tuple[float, float, float]], List[Tuple[int, int, int]]]:
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
        prompt_template = cfg.dwrl_prompt.strip().format(context=flatten_to_single_turn(messages), response_1=response_1, response_2=response_2)
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
                print("### RAW_RESPONSE_THOUGHT: ", raw_response_thought, flush=True)

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
                print("### RAW_RESPONSE_SCORE: ", raw_response_score, flush=True)
                
                score = -999
                for possible_tokens in raw_response_score['choices'][0]['logprobs']['content'][-1]['top_logprobs']:
                    if possible_tokens['token'] == cfg.score_token:
                        score = math.exp(possible_tokens['logprob'])
                        break
                if score == -999:
                    for possible_tokens in raw_response_score['choices'][0]['logprobs']['content'][-1]['top_logprobs']:
                        if possible_tokens['token'].lower().strip() == cfg.score_token:
                            score = math.exp(possible_tokens['logprob'])
                            break
                print("### BT_SCORE: ", score, flush=True)
                #ranking = 2 if score >= 0.5 else 5

                try:
                    resp_no_thinking1 = re.sub(r"(?i)(?s)(\<think\>)(.*?)(\<\/think\>)", "", thought_raw_str).strip()
                    resp_no_thinking2 = re.sub(r"(?i)(?s)(.*?)(\<\/think\>)", "", resp_no_thinking1).strip()
                    
                    score_1, score_2, _ = parse_genrm_output(
                        resp_no_thinking2,
                        cfg.default_score,
                        cfg.default_ranking,
                        raise_on_fail=True,
                    )
                    print(f"Score_1: {score_1}   Score_2: {score_2}", flush=True)
                    return score_1, score_2, score

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
