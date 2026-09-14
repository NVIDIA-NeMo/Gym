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
import json
import logging
import time
from collections import Counter, OrderedDict
from contextlib import asynccontextmanager
from contextvars import Context
from dataclasses import dataclass, field
from functools import lru_cache
from math import isfinite
from typing import Any, ClassVar, Dict, List, Literal, Optional, Tuple

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.global_config import (
    GROUP_ATTEMPT_KEY_NAME,
    GROUP_ID_KEY_NAME,
    GROUP_MEMBER_INDEX_KEY_NAME,
    ROLLOUT_INDEX_KEY_NAME,
    TASK_INDEX_KEY_NAME,
)
from nemo_gym.judge import JudgeError, reraise_judge_errors
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import raise_for_status
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


@lru_cache(maxsize=1)
def _warn_legacy_attempt() -> None:
    logger.warning("GenRM group attempt omitted; treating legacy requests as group attempt zero")


class CohortEvaluationError(RuntimeError):
    """A terminal scoring failure for this cohort attempt; retry policy belongs to the caller."""


@dataclass
class _CohortMember:
    """One authoritative response for a logical rollout slot."""

    body: Optional["GenRMCompareVerifyRequest"]
    response_digest: str
    waiters: List[asyncio.Future[float]] = field(default_factory=list)


@dataclass
class _CohortState:
    """Process-local state for one prompt cohort."""

    prompt_digest: str
    key: str = ""
    group_id: Optional[str] = None
    group_attempt: int = 0
    members: Dict[int, _CohortMember] = field(default_factory=dict)
    phase: Literal["collecting", "evaluating", "completed", "failed"] = "collecting"
    rewards: Dict[int, float] = field(default_factory=dict)
    failure: Optional[str] = None
    terminal_at: Optional[float] = None
    deadline: float = 0.0
    timeout_handle: Optional[asyncio.TimerHandle] = None
    evaluation_task: Optional[asyncio.Task[None]] = None
    comparisons: Dict[Tuple[int, int, int], asyncio.Task] = field(default_factory=dict)


@dataclass
class _GroupAttemptWatermark:
    """Newest physical attempt observed for one logical prompt group."""

    latest_attempt: int
    prompt_digest: str


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
        group_style_penalty_coeff: Coefficient for style density penalty
        default_score: Default neutral score when parsing fails
        default_ranking: Default neutral ranking when parsing fails
        debug_logging: Enable verbose logging for debugging
        genrm_parse_retries: Number of retries on parse failures
        genrm_parse_retry_sleep_s: Sleep duration between parse retries
        cohort_timeout_s: Finite deadline from first arrival through reward publication
        cohort_result_ttl_s: Retention time for completed and failed cohort tombstones
        max_terminal_cohorts: Maximum number of completed and failed cohort tombstones
        use_principle: Enable principle-based comparison
        default_principle: Default principle when none provided in request
    """

    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.UNSUPPORTED

    name: str = "genrm_compare"
    genrm_model_server: ModelServerRef  # Default: genrm_model (see config)
    genrm_responses_create_params: NeMoGymResponseCreateParamsNonStreaming

    # Cohort verification publishes relative rewards after all identified members
    # and comparisons finish; pairwise judging can overlap remaining generation.
    # When <= 1, verify() returns default_score (no comparison).
    num_rollouts_per_prompt: int = Field(default=1, ge=1)
    cohort_timeout_s: float = Field(default=1800.0, gt=0, allow_inf_nan=False)
    cohort_result_ttl_s: float = Field(default=3600.0, gt=0, allow_inf_nan=False)
    max_terminal_cohorts: int = Field(default=4096, gt=0)

    # Comparison strategy
    comparison_strategy: Literal["all_pairs", "circular"] = "circular"  # "all_pairs" or "circular"
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
    group_style_penalty_coeff: float = 0.0

    # Default neutral scores when parsing fails
    default_score: float = 3.0
    default_ranking: float = 3.5

    # Debug logging
    debug_logging: bool = False

    # Retry config for parse failures
    genrm_parse_retries: int = 3
    genrm_parse_retry_sleep_s: float = 0.2

    @model_validator(mode="after")
    def _validate_cohort_workers(self):
        if self.num_rollouts_per_prompt > 1 and (self.num_workers or 1) > 1:
            raise ValueError("GenRM cohort verification requires one HTTP worker because group state is process-local")
        return self


class GenRMCompareVerifyRequest(BaseVerifyRequest):
    """Verify request with optional principle for cohort-based GenRM comparison."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    principle: Optional[str] = None  # Principle for principle-based GenRM; forwarded by agent when provided
    task_index: Optional[int] = Field(default=None, alias=TASK_INDEX_KEY_NAME)
    group_id: Optional[str] = Field(default=None, alias=GROUP_ID_KEY_NAME, min_length=1, max_length=256)
    group_attempt: int = Field(default=0, alias=GROUP_ATTEMPT_KEY_NAME, ge=0, strict=True)
    rollout_index: Optional[int] = Field(default=None, alias=ROLLOUT_INDEX_KEY_NAME, ge=0, strict=True)
    group_member_index: Optional[int] = Field(default=None, alias=GROUP_MEMBER_INDEX_KEY_NAME, ge=0, strict=True)
    prompt_id: Optional[str] = None  # Optional stable prompt identifier from the caller

    @model_validator(mode="before")
    @classmethod
    def _warn_on_legacy_group_identity(cls, data: Any) -> Any:
        """Treat an omitted group attempt as zero during client migration."""
        if not isinstance(data, dict):
            return data
        group_id = data.get(GROUP_ID_KEY_NAME, data.get("group_id"))
        has_group_attempt = GROUP_ATTEMPT_KEY_NAME in data or "group_attempt" in data
        if group_id is not None and not has_group_attempt:
            _warn_legacy_attempt()
        return data


class GenRMCompareVerifyResponse(BaseVerifyResponse):
    """Verification response that echoes logical cohort coordinates."""

    model_config = ConfigDict(populate_by_name=True)

    group_id: Optional[str] = Field(default=None, alias=GROUP_ID_KEY_NAME, min_length=1, max_length=256)
    group_attempt: int = Field(alias=GROUP_ATTEMPT_KEY_NAME, ge=0)
    group_member_index: Optional[int] = Field(default=None, alias=GROUP_MEMBER_INDEX_KEY_NAME, ge=0, strict=True)
    rollout_index: Optional[int] = Field(default=None, alias=ROLLOUT_INDEX_KEY_NAME, ge=0, strict=True)


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
    items = (
        [{"role": "user", "content": input_messages}] if isinstance(input_messages, str) else input_messages
    ) or []
    for m in items:
        if hasattr(m, "model_dump"):
            m = m.model_dump()
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
                if isinstance(part, dict) and part.get("type") in ("input_text", "output_text")
            )
        out.append({"role": str(role), "content": str(content)})
    return out


class GenRMCompareResourcesServer(SimpleResourcesServer):
    """Resources server for GenRM pairwise comparison of multiple responses.

    Supports two modes:
    - Cohort-based verify: Compare identified members as they arrive; publish
      per-rollout rewards only after the complete cohort finishes judging.
    - Batch /compare: Direct comparison of N response_objs (e.g. for rollout_collection or tests).
    """

    config: GenRMCompareConfig
    _verify_cohorts: Dict[str, _CohortState] = PrivateAttr(default_factory=dict)
    _latest_group_attempts: Dict[str, _GroupAttemptWatermark] = PrivateAttr(default_factory=dict)
    _terminal_cohorts: OrderedDict[str, _CohortState] = PrivateAttr(default_factory=OrderedDict)
    _group_cohort_counts: Counter[str] = PrivateAttr(default_factory=Counter)
    _cohort_tasks: set[asyncio.Task] = PrivateAttr(default_factory=set)
    _closed: bool = PrivateAttr(default=False)

    async def verify(self, body: GenRMCompareVerifyRequest) -> GenRMCompareVerifyResponse:
        """Register a member, then await its result without owning the shared scoring task.

        Registry/member/terminal transitions contain no awaits. They are atomic on
        the server's event loop, including timer callbacks and final publication.
        The registry is process-local: use one HTTP worker for this server.
        """
        if self._closed:
            raise HTTPException(status_code=503, detail="GenRM server is shutting down")
        cfg = self.config
        if cfg.num_rollouts_per_prompt <= 1:
            return self._verify_response(body, cfg.default_score)
        self._validate_logical_coordinates(body)
        input_messages = body.responses_create_params.input or []
        prompt_key = self._get_verify_cohort_key(body, input_messages, body.principle)
        prompt_digest = get_prompt_key_from_input(input_messages, body.principle)
        response_digest = self._response_digest(body.response)
        index = self._member_index(body)
        assert index is not None
        cohort = self._resolve_verify_cohort(body, prompt_key, prompt_digest)
        self._expire_verify_cohort(cohort)
        if cohort.phase == "failed":
            self._raise_cohort_failure(cohort)
        if cohort.phase == "completed":
            raise HTTPException(status_code=409, detail="GenRM group attempt is complete; start a new attempt")
        member = cohort.members.get(index)
        if member is not None:
            if member.response_digest != response_digest:
                raise HTTPException(
                    status_code=409, detail=f"GenRM already has a different response for rollout_index={index}"
                )
        else:
            if cohort.phase != "collecting":
                raise HTTPException(status_code=409, detail=f"GenRM cohort is already {cohort.phase}")
            member = _CohortMember(body=body, response_digest=response_digest)
            cohort.members[index] = member

        future = asyncio.get_running_loop().create_future()
        # The request can be cancelled just as a timer sets the exception. Consume
        # orphaned exceptions without changing what an active waiter receives.
        future.add_done_callback(lambda done: None if done.cancelled() else done.exception())
        member.waiters.append(future)
        try:
            self._schedule_comparisons(cohort)
            if len(cohort.members) == cfg.num_rollouts_per_prompt and cohort.phase == "collecting":
                cohort.phase = "evaluating"
                cohort.evaluation_task = self._own_task(self._evaluate_verify_cohort(cohort), cohort)
            reward = await asyncio.shield(future)
        except CohortEvaluationError:
            self._raise_cohort_failure(cohort)
        finally:
            future.cancel()
            if future in member.waiters:
                member.waiters.remove(future)
            # One disconnected duplicate does not cancel an attached original.
            # Losing every waiter for an accepted member fails the active group:
            # a retried /run may generate a different answer, not retransmit this one.
            if not member.waiters:
                self._fail_verify_cohort(cohort, "GenRM member lost its last waiter", "cancelled")
        return self._verify_response(body, reward)

    @staticmethod
    def _verify_response(body: GenRMCompareVerifyRequest, reward: float) -> GenRMCompareVerifyResponse:
        return GenRMCompareVerifyResponse(
            responses_create_params=body.responses_create_params,
            response=body.response,
            reward=reward,
            group_id=body.group_id,
            group_attempt=body.group_attempt,
            rollout_index=body.rollout_index,
            group_member_index=body.group_member_index,
        )

    @staticmethod
    def _member_index(body: GenRMCompareVerifyRequest) -> Optional[int]:
        return body.group_member_index if body.group_member_index is not None else body.rollout_index

    @staticmethod
    def _raise_cohort_failure(cohort: _CohortState) -> None:
        raise HTTPException(status_code=503, detail=cohort.failure)

    def _validate_logical_coordinates(self, body: GenRMCompareVerifyRequest) -> None:
        if body.task_index is None and not body.group_id and not body.prompt_id:
            raise HTTPException(
                status_code=422, detail=f"either {TASK_INDEX_KEY_NAME}, {GROUP_ID_KEY_NAME}, or prompt_id is required"
            )
        if self._member_index(body) is None:
            raise HTTPException(
                status_code=422, detail=f"{GROUP_MEMBER_INDEX_KEY_NAME} or {ROLLOUT_INDEX_KEY_NAME} is required"
            )
        if not 0 <= self._member_index(body) < self.config.num_rollouts_per_prompt:
            raise HTTPException(
                status_code=422,
                detail=f"GenRM member slot must be in [0, {self.config.num_rollouts_per_prompt})",
            )

    def _resolve_verify_cohort(
        self, body: GenRMCompareVerifyRequest, prompt_key: str, prompt_digest: str
    ) -> _CohortState:
        self._prune_terminal_cohorts()
        # Legacy task-scoped callers get the same attempt fencing. They must not
        # reuse task indices across concurrent runs on the same server; explicit
        # group ids are the migration path for callers sharing a server.
        legacy_id = f"task_idx::{body.task_index}" if body.task_index is not None else f"prompt_id::{body.prompt_id}"
        group_id = body.group_id if body.group_id is not None else f"{legacy_id}::{prompt_digest}"
        # Prefix the identity so an explicit id cannot collide with a legacy id.
        identity = ("explicit::" if body.group_id is not None else "legacy::") + group_id
        now = time.monotonic()
        watermark = self._latest_group_attempts.get(identity)
        if watermark is not None:
            if watermark.prompt_digest != prompt_digest:
                raise HTTPException(
                    status_code=409, detail="GenRM group received inconsistent prompt or principle content"
                )
            if body.group_attempt < watermark.latest_attempt:
                raise HTTPException(
                    status_code=409,
                    detail=f"GenRM attempt {body.group_attempt} was superseded by attempt {watermark.latest_attempt}",
                )
        if watermark is not None and body.group_attempt > watermark.latest_attempt:
            for older in list(self._verify_cohorts.values()):
                if older.group_id == identity and older.group_attempt < body.group_attempt:
                    self._fail_verify_cohort(
                        older,
                        f"GenRM attempt {older.group_attempt} was superseded by attempt {body.group_attempt}",
                        "superseded",
                    )
        if watermark is None or body.group_attempt > watermark.latest_attempt:
            self._latest_group_attempts[identity] = _GroupAttemptWatermark(body.group_attempt, prompt_digest)
        cohort = self._verify_cohorts.get(prompt_key)
        if cohort is None:
            cohort = _CohortState(
                prompt_digest=prompt_digest,
                key=prompt_key,
                group_id=identity,
                group_attempt=body.group_attempt,
                deadline=now + self.config.cohort_timeout_s,
            )
            self._verify_cohorts[prompt_key] = cohort
            self._group_cohort_counts[identity] += 1
            cohort.timeout_handle = asyncio.get_running_loop().call_later(
                self.config.cohort_timeout_s, self._expire_verify_cohort, cohort
            )
        elif cohort.prompt_digest != prompt_digest:
            raise HTTPException(
                status_code=409, detail="GenRM cohort received inconsistent prompt or principle content"
            )
        return cohort

    def _expire_verify_cohort(self, cohort: _CohortState) -> bool:
        if cohort.phase not in ("collecting", "evaluating"):
            return False
        remaining = cohort.deadline - time.monotonic()
        if remaining > 0:
            # Event-loop timers may fire up to one clock-resolution early.
            if cohort.timeout_handle is not None:
                cohort.timeout_handle.cancel()
            cohort.timeout_handle = asyncio.get_running_loop().call_later(
                remaining, self._expire_verify_cohort, cohort
            )
            return False
        return self._fail_verify_cohort(
            cohort,
            f"GenRM cohort deadline exceeded with {len(cohort.members)}/"
            f"{self.config.num_rollouts_per_prompt} members during {cohort.phase}",
            "cohort_incomplete" if cohort.phase == "collecting" else "judge_timeout",
        )

    @staticmethod
    def _response_digest(response: Any) -> str:
        payload = response.model_dump(mode="json") if hasattr(response, "model_dump") else response
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _own_task(self, coroutine, cohort: _CohortState) -> asyncio.Task:
        task = asyncio.create_task(coroutine, context=Context())
        self._cohort_tasks.add(task)

        def finished(done):
            self._cohort_tasks.discard(done)
            if done.cancelled():
                self._fail_verify_cohort(cohort, "GenRM comparison was cancelled", "cancelled")
            elif (error := done.exception()) is not None:
                self._fail_verify_cohort(
                    cohort,
                    f"GenRM evaluation failed: {type(error).__name__}: {str(error)[:1000]}",
                    "evaluation_failed",
                )

        task.add_done_callback(finished)
        return task

    def _schedule_comparisons(self, cohort: _CohortState) -> None:
        # Keep main's overlap between generation and pairwise judging. Every
        # pair is scheduled once by logical slots, including out-of-order arrivals.
        cfg = self.config
        for judge in range(cfg.num_judges_per_comparison):
            for i, j in generate_comparison_pairs(cfg.comparison_strategy, cfg.num_rollouts_per_prompt):
                key = (i, j, judge)
                if i in cohort.members and j in cohort.members and key not in cohort.comparisons:
                    cohort.comparisons[key] = self._own_task(self._compare_members(cohort, i, j), cohort)

    async def _compare_members(self, cohort: _CohortState, i: int, j: int):
        first, second = cohort.members[i].body, cohort.members[j].body
        return await self._run_single_comparison(
            _input_to_conversation_history(first.responses_create_params.input),
            first.response.model_dump(),
            second.response.model_dump(),
            pair_idx=(i, j),
            principle=first.principle,
        )

    async def _evaluate_verify_cohort(self, cohort: _CohortState) -> None:
        try:
            indices = sorted(cohort.members)
            bodies = [cohort.members[index].body for index in indices]
            first = bodies[0]
            if first is None or any(body is None for body in bodies):
                raise RuntimeError("GenRM response discarded before evaluation")
            responses = [body.response.model_dump() for body in bodies]
            metadata = sorted(cohort.comparisons, key=lambda key: (key[2], key[0], key[1]))
            comparisons = await asyncio.gather(*(cohort.comparisons[key] for key in metadata))
            if cohort.phase != "evaluating":
                return
            cfg = self.config
            rewards, _, _, _ = aggregate_scores(
                comparison_results=comparisons,
                comparison_metadata=metadata,
                response_objs=responses,
                aggregator_method=cfg.aggregator_method,
                default_score=cfg.default_score,
                reasoning_bonus=cfg.reasoning_bonus,
                answer_bonus=cfg.answer_bonus,
                top_percentile=cfg.top_percentile,
                group_reasoning_length_penalty_coeff=cfg.group_reasoning_length_penalty_coeff,
                group_answer_length_penalty_coeff=cfg.group_answer_length_penalty_coeff,
                group_style_penalty_coeff=cfg.group_style_penalty_coeff,
            )
            if len(rewards) != len(indices) or any(not isfinite(reward) for reward in rewards):
                raise RuntimeError("GenRM must return one finite reward per cohort member")
            # The deadline is checked at publication too, even if a slow callback
            # prevented the timer from running before the judge completed.
            if self._expire_verify_cohort(cohort) or cohort.phase != "evaluating":
                return
            cohort.rewards = dict(zip(indices, rewards))
            self._finish_verify_cohort(cohort, "completed", "completed")
        except asyncio.CancelledError:
            self._fail_verify_cohort(cohort, "GenRM cohort evaluation was cancelled", "cancelled")
            raise
        except Exception as error:
            self._fail_verify_cohort(
                cohort,
                f"GenRM cohort evaluation failed: {type(error).__name__}: {str(error)[:1000]}",
                "evaluation_failed",
            )

    def _fail_verify_cohort(self, cohort: _CohortState, message: str, reason: str) -> bool:
        if cohort.phase not in ("collecting", "evaluating"):
            return False
        cohort.failure = message
        self._finish_verify_cohort(cohort, "failed", reason)
        return True

    def _finish_verify_cohort(self, cohort: _CohortState, phase: Literal["completed", "failed"], reason: str) -> None:
        cohort.phase = phase
        cohort.terminal_at = time.monotonic()
        self._terminal_cohorts[cohort.key] = cohort
        if cohort.timeout_handle is not None:
            cohort.timeout_handle.cancel()
            cohort.timeout_handle = None
        task = cohort.evaluation_task
        cohort.evaluation_task = None
        if task is not None and task is not asyncio.current_task() and not task.done():
            task.cancel()
        for comparison in cohort.comparisons.values():
            if comparison is not asyncio.current_task() and not comparison.done():
                comparison.cancel()
        cohort.comparisons.clear()
        for index, member in cohort.members.items():
            for waiter in member.waiters:
                if not waiter.done():
                    if phase == "completed":
                        waiter.set_result(cohort.rewards[index])
                    else:
                        waiter.set_exception(CohortEvaluationError(cohort.failure))
            member.body = None
            member.waiters.clear()
        # One bounded event per terminal transition; no prompts, answers or ids.
        log = logger.info if phase == "completed" else logger.warning
        log(
            "GenRM cohort disposition=%s reason=%s arrived=%d expected=%d",
            phase,
            reason,
            len(cohort.members),
            self.config.num_rollouts_per_prompt,
        )
        # Retain only a compact disposition, never answer/reward replay data.
        cohort.members.clear()
        cohort.rewards.clear()
        self._prune_terminal_cohorts()

    def _prune_terminal_cohorts(self) -> None:
        now = time.monotonic()
        # Terminal transitions append in monotonic order. Each entry is removed
        # once; a request within the retention budget only examines the oldest.
        while self._terminal_cohorts:
            key, cohort = next(iter(self._terminal_cohorts.items()))
            if (
                len(self._terminal_cohorts) <= self.config.max_terminal_cohorts
                and now - cohort.terminal_at < self.config.cohort_result_ttl_s
            ):
                break
            self._terminal_cohorts.popitem(last=False)
            self._verify_cohorts.pop(key, None)
            identity = cohort.group_id
            self._group_cohort_counts[identity] -= 1
            if self._group_cohort_counts[identity] == 0:
                del self._group_cohort_counts[identity]
                self._latest_group_attempts.pop(identity, None)

    async def aclose(self) -> None:
        """Settle waiters and drain owned scoring tasks on server shutdown."""
        self._closed = True
        for cohort in list(self._verify_cohorts.values()):
            self._fail_verify_cohort(cohort, "GenRM server is shutting down", "shutdown")
        await asyncio.gather(*self._cohort_tasks, return_exceptions=True)
        self._verify_cohorts.clear()
        self._latest_group_attempts.clear()
        self._terminal_cohorts.clear()
        self._group_cohort_counts.clear()

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        parent_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(app):
            async with parent_lifespan(app) as state:
                try:
                    yield state
                finally:
                    await self.aclose()

        app.router.lifespan_context = lifespan
        app.post("/compare")(self.compare)
        return app

    def _get_verify_cohort_key(
        self,
        body: GenRMCompareVerifyRequest,
        input_messages: List[Any],
        principle: Optional[str] = None,
    ) -> str:
        """Return an attempt-scoped key so replacement cohorts cannot mix with old responses."""
        if body.group_id is not None:
            return f"group_id::{body.group_id}::group_attempt::{body.group_attempt}"

        prompt_key = get_prompt_key_from_input(input_messages, principle)
        if body.task_index is not None:
            logical_key = f"task_idx::{body.task_index}::{prompt_key}"
        elif body.prompt_id is not None:
            logical_key = f"prompt_id::{body.prompt_id}::{prompt_key}"
        else:
            logical_key = prompt_key
        return f"{logical_key}::group_attempt::{body.group_attempt}"

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
        tasks = [asyncio.create_task(comparison) for comparison in comparison_tasks]
        try:
            comparison_results = await asyncio.gather(*tasks)
        finally:
            # gather does not cancel siblings when one comparison raises.
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
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
            group_style_penalty_coeff=cfg.group_style_penalty_coeff,
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
        try:
            rewards, metrics, comparison_results, comparison_metadata = await self._run_compare(
                conversation_history, response_objs, principle=body.principle
            )
        except JudgeError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
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

        # Build the request params
        responses_create_params = cfg.genrm_responses_create_params.model_copy(deep=True)
        responses_create_params.input = messages
        responses_create_params.metadata = metadata

        async def call():
            response = await self.server_client.post(
                server_name=cfg.genrm_model_server.name,
                url_path="/v1/responses",
                json=responses_create_params,
            )
            await raise_for_status(response)
            return await response.json()

        max_attempts = max(1, int(cfg.genrm_parse_retries) + 1)
        for attempt_idx in range(max_attempts):
            raw_response = await reraise_judge_errors(call())
            if not isinstance(raw_response, dict) or raw_response.get("status") not in {None, "completed"}:
                raise JudgeError("GenRM judge returned an unsuccessful response")
            _, genrm_answer = extract_from_response_obj(raw_response)
            if not genrm_answer.strip():
                raise JudgeError("GenRM judge returned no answer")
            try:
                return parse_genrm_output(genrm_answer, cfg.default_score, cfg.default_ranking, raise_on_fail=True)
            except GenRMOutputParseError:
                if attempt_idx < max_attempts - 1:
                    await asyncio.sleep(float(cfg.genrm_parse_retry_sleep_s))
                    continue
                logger.warning(
                    "[GenRM] Parse failed for pair %s after %s attempts; falling back to defaults.",
                    pair_idx,
                    max_attempts,
                )
        return cfg.default_score, cfg.default_score, cfg.default_ranking


if __name__ == "__main__":
    GenRMCompareResourcesServer.run_webserver()
