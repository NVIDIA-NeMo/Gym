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
from contextlib import asynccontextmanager
from contextvars import Context
from dataclasses import dataclass, field
from functools import lru_cache
from math import isfinite
from typing import Any, ClassVar, Dict, List, Literal, Optional, Tuple

from aiohttp import ClientConnectionError, ClientPayloadError, ClientResponseError
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.global_config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.judge import JudgeError
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import get_response_json, raise_for_status
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

# Tuple layout: selected (score_1, score_2, ranking), overall (score_1, score_2, ranking),
# token metrics (input_tokens, output_tokens, max_output_tokens_hit), and failure flags
# (overall_parse_failed, rubric_parse_failed).
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
]
VerifyResult = Dict[str, Any]


def _output_budget_exhausted(raw_response: Any) -> bool:
    """True if a Responses API object was cut off by ``max_output_tokens``.

    The model spent its whole output budget (typically on reasoning) and never emitted a
    verdict: ``status="incomplete"`` with ``incomplete_details.reason="max_output_tokens"``.
    Gym's chat->Responses conversion produces this for ``finish_reason="length"``; hosted
    Responses API backends emit it natively.
    """
    if not isinstance(raw_response, dict) or raw_response.get("status") != "incomplete":
        return False
    details = raw_response.get("incomplete_details") or {}
    return isinstance(details, dict) and details.get("reason") == "max_output_tokens"


GROUP_ID_KEY_NAME = "_ng_group_id"
GROUP_ATTEMPT_KEY_NAME = "_ng_group_attempt"


@lru_cache(maxsize=1)
def _warn_legacy_attempt() -> None:
    logger.warning("GenRM group attempt omitted; treating legacy requests as group attempt zero")


class CohortEvaluationError(RuntimeError):
    """A failed cohort attempt; replacement policy belongs to the caller."""


@dataclass
class _CohortMember:
    """One authoritative response for a logical rollout slot."""

    body: Optional["GenRMCompareVerifyRequest"]
    response_digest: str
    waiters: List[asyncio.Future[VerifyResult]] = field(default_factory=list)


@dataclass
class _CohortState:
    """Process-local state for one prompt cohort."""

    prompt_digest: str
    key: str
    group_id: Optional[str] = None
    group_attempt: int = 0
    members: Dict[int, _CohortMember] = field(default_factory=dict)
    phase: Literal["collecting", "evaluating", "completed", "failed"] = "collecting"
    results: Dict[int, VerifyResult] = field(default_factory=dict)
    failure: Optional[str] = None
    failure_kind: Literal["cohort", "judge"] = "cohort"
    terminal_at: Optional[float] = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    collection_timeout_task: Optional[asyncio.Task[None]] = None
    evaluation_task: Optional[asyncio.Task[None]] = None


@dataclass
class _GroupAttemptWatermark:
    """Newest physical attempt observed for one logical prompt group."""

    latest_attempt: int
    prompt_digest: str
    updated_at: float


class GenRMCompareConfig(BaseResourcesServerConfig):
    """Configuration for the GenRM compare server.

    Attributes:
        genrm_model_server: Target GenRM model server (default: genrm_model from config)
        genrm_responses_create_params: Base create params for GenRM calls
        comparison_mode: Compare rollouts to each other or to a fixed baseline
        comparison_strategy: "all_pairs" or "circular"
        num_judges_per_comparison: Number of judge passes per pair (majority voting)
        aggregator_method: Method for aggregating scores
        score_source: "overall" or "rubric_mean"
        reasoning_bonus: Bonus for shortest reasoning content among top performers
        answer_bonus: Bonus for shortest answer among top performers
        top_percentile: Percentile threshold for applying bonuses
        group_reasoning_length_penalty_coeff: Coefficient for reasoning length penalty
        group_answer_length_penalty_coeff: Coefficient for answer length penalty
        group_style_penalty_coeff: Coefficient for style density penalty
        default_score: Default neutral score when parsing fails
        default_ranking: Default neutral ranking when parsing fails
        debug_logging: Enable verbose logging for debugging
        genrm_parse_retries: Shared retry budget for parse failures and HTTP 408, 429, or 5xx
        genrm_parse_retry_sleep_s: Sleep duration between retry attempts
        cohort_collection_timeout_s: Deadline to collect every logical rollout index
        cohort_evaluation_timeout_s: Separate deadline for all comparisons and aggregation
        judge_request_timeout_s: Deadline for each judge HTTP request, including transport retries
        cohort_result_ttl_s: Optional retention time for completed and failed cohort tombstones
        max_terminal_cohorts: Maximum number of completed and failed cohort tombstones
        use_principle: Enable principle-based comparison
        default_principle: Default principle when none provided in request
    """

    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.UNSUPPORTED

    name: str = "genrm_compare"
    genrm_model_server: ModelServerRef  # Default: genrm_model (see config)
    genrm_responses_create_params: NeMoGymResponseCreateParamsNonStreaming

    # Verification waits for this many rollouts in either comparison mode.
    # A singleton rollout_cohort returns default_score; fixed_baseline still calls the judge.
    num_rollouts_per_prompt: int = Field(default=1, ge=1)
    cohort_collection_timeout_s: float = Field(default=1800.0, gt=0, allow_inf_nan=False)
    cohort_evaluation_timeout_s: float = Field(default=1800.0, gt=0, allow_inf_nan=False)
    judge_request_timeout_s: float = Field(default=1800.0, gt=0, allow_inf_nan=False)
    cohort_result_ttl_s: Optional[float] = Field(default=3600.0, gt=0, allow_inf_nan=False)
    max_terminal_cohorts: int = Field(default=4096, gt=0)

    # Comparison strategy
    comparison_mode: Literal["rollout_cohort", "fixed_baseline"] = "rollout_cohort"
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

    # Shared retry budget for parse failures and transient judge HTTP errors
    genrm_parse_retries: int = 3
    genrm_parse_retry_sleep_s: float = 0.2

    @model_validator(mode="after")
    def _validate_cohort_workers(self):
        if (self.comparison_mode == "fixed_baseline" or self.num_rollouts_per_prompt > 1) and (
            self.num_workers or 1
        ) > 1:
            raise ValueError("GenRM cohort verification requires one HTTP worker because group state is process-local")
        return self


class GenRMCompareVerifyRequest(BaseVerifyRequest):
    """Verify request with optional principle for cohort-based GenRM comparison."""

    model_config = ConfigDict(extra="allow", populate_by_name=True, serialize_by_alias=True)

    principle: Optional[str] = None  # Principle for principle-based GenRM; forwarded by agent when provided
    expected_rubric_ids: Optional[Tuple[int, ...]] = None
    task_index: Optional[int] = Field(default=None, alias=TASK_INDEX_KEY_NAME)
    group_id: Optional[str] = Field(default=None, alias=GROUP_ID_KEY_NAME)
    group_attempt: int = Field(default=0, alias=GROUP_ATTEMPT_KEY_NAME, ge=0)
    rollout_index: Optional[int] = Field(default=None, alias=ROLLOUT_INDEX_KEY_NAME)
    prompt_id: Optional[str] = None  # Optional stable prompt identifier from the caller

    @field_validator("expected_rubric_ids")
    @classmethod
    def _canonicalize_rubric_ids(cls, value: Optional[Tuple[int, ...]]) -> Optional[Tuple[int, ...]]:
        """Rubric contracts are sets; ordering must not split cohorts or replay keys."""
        return tuple(sorted(set(value))) if value is not None else None

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

    group_id: Optional[str] = Field(default=None, alias=GROUP_ID_KEY_NAME)
    group_attempt: int = Field(alias=GROUP_ATTEMPT_KEY_NAME, ge=0)
    rollout_index: Optional[int] = Field(default=None, alias=ROLLOUT_INDEX_KEY_NAME)
    reasoning_text: str
    answer_text: str
    reward_score_raw: float
    # Rubric aggregate including the tiebreaker, before length/style adjustments.
    # None outside rubric_mean mode or if any comparison for this rollout failed rubric parsing.
    reward_rubric_aggregate_valid: Optional[float] = None
    reward_overall_score_raw: float
    reward_overall_score: float
    reward_length_adjustment: float
    genrm_parse_failure_rate_per_group: float = 0.0
    genrm_rubric_parse_failure_rate_per_group: float = 0.0
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
    if isinstance(input_messages, str):
        return [{"role": "user", "content": input_messages}]
    out = []
    for item in input_messages or []:
        item = item.model_dump(mode="json") if hasattr(item, "model_dump") else item
        if not isinstance(item, dict) or item.get("type", "message") != "message":
            continue
        content = item.get("content") or ""
        if isinstance(content, list):
            content = "".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") in ("input_text", "output_text")
            )
        out.append({"role": item.get("role", "user"), "content": str(content)})
    return out


class GenRMCompareResourcesServer(SimpleResourcesServer):
    """Resources server for GenRM pairwise comparison of multiple responses.

    Supports two modes:
    - Cohort-based verify (Difference 1): When num_rollouts_per_prompt > 1, verify() buffers by prompt;
      when the cohort is full, runs comparison and returns per-rollout rewards. Callers await until
      their cohort is complete and get their reward.
    - Batch /compare: Direct comparison of N response_objs (e.g. for rollout_collection or tests).
    """

    ray_enabled = False

    config: GenRMCompareConfig
    _verify_cohorts: Dict[str, _CohortState] = PrivateAttr(default_factory=dict)
    _latest_group_attempts: Dict[str, _GroupAttemptWatermark] = PrivateAttr(default_factory=dict)
    _cohort_registry_lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)

    _cohort_tasks: set[asyncio.Task] = PrivateAttr(default_factory=set)
    _closed: bool = PrivateAttr(default=False)

    def _own_task(self, coro, *, name: str) -> asyncio.Task:
        task = asyncio.create_task(coro, name=name, context=Context())
        self._cohort_tasks.add(task)
        task.add_done_callback(self._cohort_tasks.discard)
        return task

    async def verify(self, body: GenRMCompareVerifyRequest) -> GenRMCompareVerifyResponse:
        """Verify one logical rollout slot as part of a prompt cohort."""
        if self._closed:
            raise HTTPException(status_code=503, detail="GenRM server is shutting down")
        cfg = self.config
        principle = body.principle
        if cfg.score_source == "rubric_mean" and not body.expected_rubric_ids:
            raise ValueError("score_source=rubric_mean requires expected_rubric_ids")
        # A singleton has no peer unless fixed-baseline mode supplies one.
        if cfg.comparison_mode == "rollout_cohort" and cfg.num_rollouts_per_prompt <= 1:
            return self._verify_response(
                body,
                {
                    "reward": cfg.default_score,
                    "reward_score_raw": cfg.default_score,
                    "reward_overall_score_raw": cfg.default_score,
                    "reward_overall_score": cfg.default_score,
                    "reward_length_adjustment": 0.0,
                },
            )

        self._validate_logical_coordinates(body)
        input_messages = getattr(body.responses_create_params, "input", None) or []
        baseline_response = None
        if cfg.comparison_mode == "fixed_baseline":
            baseline_response = (body.responses_create_params.metadata or {}).get("baseline_response")
            if not isinstance(baseline_response, str) or not baseline_response.strip():
                raise ValueError("comparison_mode=fixed_baseline requires metadata.baseline_response")
        prompt_key = self._get_verify_cohort_key(
            body,
            input_messages,
            principle,
        )
        prompt_digest = get_prompt_key_from_input(
            input_messages,
            principle,
        )
        if baseline_response is not None:
            # Never mix the same prompt evaluated against different fixed baselines.
            baseline_digest = hashlib.sha256(baseline_response.encode()).hexdigest()
            prompt_key = f"{prompt_key}:{baseline_digest}"
            prompt_digest = f"{prompt_digest}:{baseline_digest}"
        if body.expected_rubric_ids is not None:
            # Keep cohorts with different rubric contracts separate.
            rubric_contract = ",".join(map(str, body.expected_rubric_ids))
            prompt_key = f"{prompt_key}:rubrics:{rubric_contract}"
            prompt_digest = f"{prompt_digest}:rubrics:{rubric_contract}"
        rollout_index = body.rollout_index
        assert rollout_index is not None  # Validated above; keeps the type narrow below.
        response_digest = self._response_digest(body.response)
        future: asyncio.Future[VerifyResult] = asyncio.get_running_loop().create_future()
        cohort_identity = body.task_index if body.task_index is not None else body.group_id

        cohort = await self._resolve_verify_cohort(
            body=body,
            prompt_key=prompt_key,
            prompt_digest=prompt_digest,
        )
        async with cohort.lock:
            if cohort.prompt_digest != prompt_digest:
                raise HTTPException(
                    status_code=409,
                    detail=(f"GenRM cohort {prompt_key!r} received inconsistent prompt or principle content"),
                )
            if cohort.phase == "failed":
                self._raise_cohort_failure(body, cohort)
            member = cohort.members.get(rollout_index)
            if member is not None:
                if member.response_digest != response_digest:
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            f"GenRM cohort {prompt_key!r} already has a different response for "
                            f"rollout_index={rollout_index}"
                        ),
                    )
                if cohort.phase == "completed":
                    return self._verify_response(body, cohort.results[rollout_index])
                member.waiters.append(future)
            else:
                if cohort.phase != "collecting":
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            f"GenRM cohort {prompt_key!r} is already {cohort.phase}; "
                            f"rollout_index={rollout_index} cannot be added"
                        ),
                    )
                cohort.members[rollout_index] = _CohortMember(
                    body=body,
                    response_digest=response_digest,
                    waiters=[future],
                )
                if cohort.collection_timeout_task is None:
                    cohort.collection_timeout_task = self._own_task(
                        self._expire_collecting_cohort(
                            prompt_key,
                            cohort,
                            cfg.cohort_collection_timeout_s,
                        ),
                        name=f"genrm-cohort-collection-{cohort_identity}-attempt-{body.group_attempt}",
                    )

            if len(cohort.members) == cfg.num_rollouts_per_prompt and cohort.phase == "collecting":
                cohort.phase = "evaluating"
                if cohort.collection_timeout_task is not None:
                    cohort.collection_timeout_task.cancel()
                    cohort.collection_timeout_task = None
                members = dict(cohort.members)
                cohort.evaluation_task = self._own_task(
                    self._evaluate_verify_cohort(prompt_key, cohort, members),
                    name=f"genrm-cohort-evaluation-{cohort_identity}",
                )

        # A disconnected request must not cancel the shared cohort result.
        try:
            result = await asyncio.shield(future)
        except CohortEvaluationError:
            self._raise_cohort_failure(body, cohort)
        except asyncio.CancelledError:
            # The logical member remains registered, but this HTTP request no
            # longer needs a result. Mark its waiter consumed so a later cohort
            # failure does not produce an unobserved Future exception.
            if future.done() and not future.cancelled():
                future.exception()  # Consume a failure racing the disconnect.
            future.cancel()
            await asyncio.shield(self._remove_waiter(cohort, rollout_index, future))
            raise
        return self._verify_response(body, result)

    @staticmethod
    def _raise_cohort_failure(body: GenRMCompareVerifyRequest, cohort: _CohortState) -> None:
        message = cohort.failure or "GenRM cohort evaluation failed"
        if cohort.group_id is None:
            message += (
                " This legacy group has failed; retry requires a fresh _ng_group_id shared by every member."
                " Reusing the task/prompt key could mix delayed answers with a replacement group."
            )
        if cohort.failure_kind == "judge":
            # The shared failsafe preserves the answer and sets both masking contracts.
            raise JudgeError(message)
        raise HTTPException(status_code=503, detail=message)

    @staticmethod
    def _verify_response(body: GenRMCompareVerifyRequest, result: VerifyResult) -> GenRMCompareVerifyResponse:
        response_obj = body.response.model_dump() if hasattr(body.response, "model_dump") else body.response
        reasoning_text, answer_text = extract_from_response_obj(response_obj)
        return GenRMCompareVerifyResponse(
            responses_create_params=body.responses_create_params,
            response=body.response,
            reasoning_text=reasoning_text,
            answer_text=answer_text,
            group_id=body.group_id,
            group_attempt=body.group_attempt,
            rollout_index=body.rollout_index,
            **result,
        )

    def _validate_logical_coordinates(self, body: GenRMCompareVerifyRequest) -> None:
        """Reject malformed cohort members before mutating shared state."""
        expected_size = self.config.num_rollouts_per_prompt
        if body.rollout_index is None:
            raise HTTPException(status_code=422, detail=f"{ROLLOUT_INDEX_KEY_NAME} is required for cohort comparison")
        if not 0 <= body.rollout_index < expected_size:
            raise HTTPException(
                status_code=422,
                detail=(f"{ROLLOUT_INDEX_KEY_NAME} must be in [0, {expected_size}); got {body.rollout_index}"),
            )

    async def _resolve_verify_cohort(
        self,
        *,
        body: GenRMCompareVerifyRequest,
        prompt_key: str,
        prompt_digest: str,
    ) -> _CohortState:
        """Resolve one attempt cohort and retire superseded attempts atomically."""
        if body.group_id is None:
            self._prune_terminal_cohorts()
            cohort = self._verify_cohorts.get(prompt_key)
            if cohort is None:
                cohort = _CohortState(prompt_digest=prompt_digest, key=prompt_key)
                self._verify_cohorts[prompt_key] = cohort
            return cohort

        async with self._cohort_registry_lock:
            if self._closed:
                raise HTTPException(status_code=503, detail="GenRM server is shutting down")
            self._prune_terminal_cohorts()
            now = time.monotonic()
            watermark = self._latest_group_attempts.get(body.group_id)
            if watermark is not None and watermark.prompt_digest != prompt_digest:
                raise HTTPException(
                    status_code=409,
                    detail=(f"GenRM group {body.group_id!r} received inconsistent prompt or principle content"),
                )
            if watermark is not None and body.group_attempt < watermark.latest_attempt:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"GenRM group {body.group_id!r} attempt {body.group_attempt} "
                        f"was superseded by attempt {watermark.latest_attempt}"
                    ),
                )

            if watermark is None or body.group_attempt > watermark.latest_attempt:
                self._latest_group_attempts[body.group_id] = _GroupAttemptWatermark(
                    latest_attempt=body.group_attempt,
                    prompt_digest=prompt_digest,
                    updated_at=now,
                )
                await self._supersede_older_group_attempts(
                    group_id=body.group_id,
                    new_attempt=body.group_attempt,
                )
            else:
                watermark.updated_at = now

            cohort = self._verify_cohorts.get(prompt_key)
            if cohort is None:
                cohort = _CohortState(
                    prompt_digest=prompt_digest,
                    key=prompt_key,
                    group_id=body.group_id,
                    group_attempt=body.group_attempt,
                )
                self._verify_cohorts[prompt_key] = cohort
            return cohort

    async def _supersede_older_group_attempts(
        self,
        *,
        group_id: str,
        new_attempt: int,
    ) -> None:
        """Release waiters and payloads owned by older active attempts."""
        for cohort in self._verify_cohorts.values():
            if (
                cohort.group_id != group_id
                or cohort.group_attempt >= new_attempt
                or cohort.phase not in ("collecting", "evaluating")
            ):
                continue
            old_phase = cohort.phase
            evaluation_task = cohort.evaluation_task
            failed = await self._fail_verify_cohort(
                cohort,
                (f"GenRM group {group_id!r} attempt {cohort.group_attempt} was superseded by attempt {new_attempt}"),
                expected_phase=old_phase,
            )
            if failed and evaluation_task is not None and not evaluation_task.done():
                evaluation_task.cancel()

    async def _expire_collecting_cohort(
        self,
        prompt_key: str,
        cohort: _CohortState,
        timeout_s: float,
    ) -> None:
        """Fail a cohort that never receives all of its logical members."""
        try:
            await asyncio.sleep(timeout_s)
        except asyncio.CancelledError:
            return

        await self._fail_verify_cohort(
            cohort,
            (
                f"GenRM cohort {prompt_key!r} did not collect "
                f"{self.config.num_rollouts_per_prompt} unique rollout indices within "
                f"{timeout_s}s"
            ),
            expected_phase="collecting",
        )

    @staticmethod
    async def _remove_waiter(
        cohort: _CohortState,
        rollout_index: int,
        waiter: asyncio.Future[VerifyResult],
    ) -> None:
        """Detach one transport waiter without retiring its logical member."""
        async with cohort.lock:
            member = cohort.members.get(rollout_index)
            if member is not None and waiter in member.waiters:
                member.waiters.remove(waiter)

    @staticmethod
    def _response_digest(response: Any) -> str:
        """Hash the exact response payload whose tokens will receive the reward."""
        payload = response.model_dump(mode="json") if hasattr(response, "model_dump") else response
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    async def _evaluate_verify_cohort(
        self,
        prompt_key: str,
        cohort: _CohortState,
        members: Dict[int, _CohortMember],
    ) -> None:
        """Evaluate exactly one immutable snapshot and publish results to all waiters."""
        try:
            sorted_indices = sorted(members)
            first_body = members[sorted_indices[0]].body
            if first_body is None:
                raise RuntimeError("GenRM cohort member body was discarded before evaluation")
            conversation_history = _input_to_conversation_history(
                getattr(first_body.responses_create_params, "input", []) or []
            )
            response_objs = []
            for index in sorted_indices:
                member_body = members[index].body
                if member_body is None:
                    raise RuntimeError(f"GenRM cohort member rollout_index={index} was discarded before evaluation")
                if member_body.expected_rubric_ids != first_body.expected_rubric_ids:
                    raise ValueError("GenRM cohort members have inconsistent expected_rubric_ids")
                response_objs.append(
                    member_body.response.model_dump()
                    if hasattr(member_body.response, "model_dump")
                    else member_body.response
                )

            async with asyncio.timeout(self.config.cohort_evaluation_timeout_s):
                if self.config.comparison_mode == "fixed_baseline":
                    baselines = [
                        (member.body.responses_create_params.metadata or {}).get("baseline_response")
                        for member in members.values()
                        if member.body is not None
                    ]
                    baseline_response = baselines[0] if baselines else None
                    if not isinstance(baseline_response, str) or not baseline_response.strip():
                        raise ValueError("comparison_mode=fixed_baseline requires metadata.baseline_response")
                    if any(baseline != baseline_response for baseline in baselines):
                        raise ValueError("GenRM cohort members have inconsistent baseline_response values")
                    compare_result = await self._run_fixed_baseline_compare(
                        conversation_history,
                        response_objs,
                        baseline_response,
                        first_body.principle,
                        prompt_key,
                        first_body.expected_rubric_ids,
                    )
                else:
                    raw_results, metadata = await self._collect_comparisons(
                        conversation_history=conversation_history,
                        response_objs=response_objs,
                        principle=first_body.principle,
                        expected_rubric_ids=first_body.expected_rubric_ids,
                    )
                    compare_result = self._aggregate_results(response_objs, raw_results, metadata)
            (
                rewards,
                raw_scores,
                valid_rubric_scores,
                metrics,
                _,
                overall_raw,
                overall_adjusted,
                length_adjustments,
            ) = compare_result
            if len(rewards) != len(sorted_indices):
                raise RuntimeError(f"GenRM returned {len(rewards)} rewards for {len(sorted_indices)} cohort members")
            if not all(isfinite(reward) for reward in rewards):
                raise ValueError("GenRM returned non-finite cohort rewards")
            results_by_index = {
                index: {
                    "reward": rewards[position],
                    "reward_score_raw": raw_scores[position],
                    "reward_rubric_aggregate_valid": valid_rubric_scores[position],
                    "reward_overall_score_raw": overall_raw[position],
                    "reward_overall_score": overall_adjusted[position],
                    "reward_length_adjustment": length_adjustments[position],
                    **metrics,
                }
                for position, index in enumerate(sorted_indices)
            }
            await self._publish_verify_cohort(prompt_key, cohort, results_by_index)
        except asyncio.CancelledError:
            await asyncio.shield(
                self._fail_verify_cohort(
                    cohort,
                    "GenRM cohort evaluation was cancelled",
                    expected_phase="evaluating",
                )
            )
            raise
        except JudgeError as error:
            await self._fail_verify_cohort(cohort, str(error), expected_phase="evaluating", failure_kind="judge")
        except TimeoutError:
            await self._fail_verify_cohort(
                cohort,
                f"GenRM cohort evaluation deadline exceeded after {self.config.cohort_evaluation_timeout_s}s",
                expected_phase="evaluating",
                failure_kind="judge",
            )
        except Exception as error:
            logger.exception("GenRM cohort evaluation failed for %s", prompt_key)
            await self._fail_verify_cohort(
                cohort,
                f"GenRM cohort evaluation failed: {type(error).__name__}: {str(error)[:1000]}",
                expected_phase="evaluating",
            )

    async def _publish_verify_cohort(
        self,
        prompt_key: str,
        cohort: _CohortState,
        results_by_index: Dict[int, VerifyResult],
    ) -> None:
        """Publish rewards, retiring legacy cohorts and compacting explicit-ID tombstones."""
        async with cohort.lock:
            if cohort.phase == "failed":
                logger.debug("Discarding late GenRM completion for key=%r", cohort.key[:160])
                return
            if cohort.phase != "evaluating":
                raise RuntimeError(f"cannot publish GenRM rewards while cohort is {cohort.phase}")
            cohort.results = results_by_index
            cohort.phase = "completed"
            cohort.terminal_at = time.monotonic()
            cohort.evaluation_task = None
            for index, member in cohort.members.items():
                for waiter in member.waiters:
                    if not waiter.done():
                        waiter.set_result(results_by_index[index])
                # A tombstone only needs the digest and result. Do not retain
                # full response payloads for every completed training cohort.
                member.body = None
                member.waiters.clear()
            if cohort.group_id is None and self._verify_cohorts.get(prompt_key) is cohort:
                self._verify_cohorts.pop(prompt_key, None)
            logger.info(
                "GenRM cohort disposition=completed key=%r attempt=%s members=%s",
                cohort.key[:160],
                cohort.group_attempt,
                len(cohort.members),
            )

    async def _fail_verify_cohort(
        self,
        cohort: _CohortState,
        message: str,
        *,
        expected_phase: Literal["collecting", "evaluating"],
        failure_kind: Literal["cohort", "judge"] = "cohort",
    ) -> bool:
        async with cohort.lock:
            if cohort.phase != expected_phase:
                return False
            cohort.phase = "failed"
            cohort.failure = message
            cohort.failure_kind = failure_kind
            cohort.terminal_at = time.monotonic()
            timeout_task = cohort.collection_timeout_task
            cohort.collection_timeout_task = None
            current_task = asyncio.current_task()
            if timeout_task is not None and timeout_task is not current_task:
                timeout_task.cancel()
            cohort.evaluation_task = None
            for member in cohort.members.values():
                for waiter in member.waiters:
                    if not waiter.done():
                        waiter.set_exception(CohortEvaluationError(message))
                member.body = None
                member.waiters.clear()
            # Keep failed legacy groups fenced too: delayed old members must not
            # join a replacement under the same key. Recovery needs an explicit ID.
            logger.warning(
                "GenRM cohort disposition=failed key=%r attempt=%s members=%s kind=%s reason=%r",
                cohort.key[:160],
                cohort.group_attempt,
                len(cohort.members),
                failure_kind,
                message[:500],
            )
            return True

    def _prune_terminal_cohorts(self) -> None:
        """Bound process-local cohort tombstones and attempt watermarks."""
        now = time.monotonic()
        terminal = [(key, cohort) for key, cohort in self._verify_cohorts.items() if cohort.terminal_at is not None]
        terminal_ttl_s = self.config.cohort_result_ttl_s
        if terminal_ttl_s is not None:
            for key, cohort in terminal:
                if now - cohort.terminal_at >= terminal_ttl_s:
                    self._verify_cohorts.pop(key, None)

        terminal = sorted(
            ((key, cohort) for key, cohort in self._verify_cohorts.items() if cohort.terminal_at is not None),
            key=lambda item: item[1].terminal_at or 0.0,
        )
        for key, _ in terminal[: -self.config.max_terminal_cohorts]:
            self._verify_cohorts.pop(key, None)

        active_group_ids = {
            cohort.group_id
            for cohort in self._verify_cohorts.values()
            if cohort.group_id is not None and cohort.terminal_at is None
        }
        if terminal_ttl_s is not None:
            for group_id, watermark in list(self._latest_group_attempts.items()):
                if group_id not in active_group_ids and now - watermark.updated_at >= terminal_ttl_s:
                    self._latest_group_attempts.pop(group_id, None)

        prunable_watermarks = sorted(
            (
                (group_id, watermark)
                for group_id, watermark in self._latest_group_attempts.items()
                if group_id not in active_group_ids
            ),
            key=lambda item: item[1].updated_at,
        )
        excess = len(self._latest_group_attempts) - self.config.max_terminal_cohorts
        for group_id, _ in prunable_watermarks[: max(0, excess)]:
            self._latest_group_attempts.pop(group_id, None)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/compare")(self.compare)
        previous_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(app):
            async with previous_lifespan(app):
                try:
                    yield
                finally:
                    await self.aclose()

        app.router.lifespan_context = lifespan
        return app

    async def aclose(self) -> None:
        """Fail active cohorts and drain all tasks owned by this server."""
        async with self._cohort_registry_lock:
            self._closed = True
            for cohort in list(self._verify_cohorts.values()):
                if cohort.phase in ("collecting", "evaluating"):
                    await self._fail_verify_cohort(
                        cohort, "GenRM server is shutting down", expected_phase=cohort.phase
                    )
            tasks = list(self._cohort_tasks)
            for task in tasks:
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._verify_cohorts.clear()
        self._latest_group_attempts.clear()

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

        rewards, aggregation_metrics, raw_scores, length_adjustments = aggregate(comparisons, metadata, adjust=True)
        overall_adjusted, _, overall_raw, _ = aggregate(overall_comparisons, metadata, adjust=True)
        rewards, raw_scores = rewards[:count], raw_scores[:count]
        overall_raw, overall_adjusted = overall_raw[:count], overall_adjusted[:count]
        length_adjustments = length_adjustments[:count]

        valid_rubric_scores: List[Optional[float]] = [None] * count
        if cfg.score_source == "rubric_mean":
            # A valid rubric metric requires every comparison touching that response to parse.
            valid = [not result[-1] for result in raw_results]
            valid_results = [comparison for comparison, keep in zip(comparisons, valid) if keep]
            valid_metadata = [item for item, keep in zip(metadata, valid) if keep]
            failed_indices = {index for item, keep in zip(metadata, valid) if not keep for index in item[:2]}
            if valid_results:
                valid_values = aggregate(valid_results, valid_metadata)[2]
                valid_rubric_scores = [
                    valid_values[index] if index not in failed_indices else None for index in range(count)
                ]

        total = max(1, len(raw_results))
        # Preserve the existing score metrics and add GenRM diagnostics alongside them.
        metrics = {
            **aggregation_metrics,
            "genrm_parse_failure_rate_per_group": sum(result[-2] for result in raw_results) / total,
            "genrm_rubric_parse_failure_rate_per_group": (
                sum(result[-1] for result in raw_results) / total if cfg.score_source == "rubric_mean" else 0.0
            ),
        }
        token_usage = [result[6:9] for result in raw_results if result[6] >= 0 and result[7] >= 0]
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
            valid_rubric_scores,
            metrics,
            comparisons,
            overall_raw,
            overall_adjusted,
            length_adjustments,
        )

    async def _collect_comparisons(
        self,
        conversation_history: List[Dict[str, str]],
        response_objs: List[Dict[str, Any]],
        principle: Optional[str] = None,
        expected_rubric_ids: Optional[Tuple[int, ...]] = None,
    ) -> Tuple[List[ComparisonResult], List[Tuple[int, int, int]]]:
        cfg = self.config
        comparison_pairs = generate_comparison_pairs(cfg.comparison_strategy, len(response_objs))
        tasks = []
        metadata: List[Tuple[int, int, int]] = []
        for judge_idx in range(cfg.num_judges_per_comparison):
            for i, j in comparison_pairs:
                tasks.append(
                    asyncio.create_task(
                        self._run_single_comparison(
                            conversation_history,
                            response_objs[i],
                            response_objs[j],
                            pair_idx=(i, j),
                            principle=principle,
                            expected_rubric_ids=expected_rubric_ids,
                        )
                    )
                )
                metadata.append((i, j, judge_idx))
        try:
            return list(await asyncio.gather(*tasks)), metadata
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _run_compare(
        self,
        conversation_history: List[Dict[str, str]],
        response_objs: List[Dict[str, Any]],
        principle: Optional[str] = None,
        expected_rubric_ids: Optional[Tuple[int, ...]] = None,
    ) -> tuple:
        """Run pairwise comparison; return (rewards, metrics, comparison_results, comparison_metadata)."""
        cfg = self.config
        num_responses = len(response_objs)
        if num_responses < 2:
            return [cfg.default_score] * num_responses, {}, [], []

        raw_results, comparison_metadata = await self._collect_comparisons(
            conversation_history,
            response_objs,
            principle,
            expected_rubric_ids,
        )
        aggregate_result = self._aggregate_results(response_objs, raw_results, comparison_metadata)
        rewards, _, _, metrics, comparison_results, _, _, _ = aggregate_result
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
        baseline_index = len(response_objs)
        baseline_obj = {
            "output": [{"type": "message", "content": [{"type": "output_text", "text": baseline_response}]}]
        }

        async def compare_one(response_index: int, response_obj: Dict[str, Any], judge_index: int) -> ComparisonResult:
            # Deterministically choose the first order, then alternate judges to reduce position bias.
            order = int(hashlib.sha256(f"{prompt_key}:{response_index}".encode()).hexdigest(), 16)
            rollout_first = (order + judge_index) % 2 == 0
            first, second = (response_obj, baseline_obj) if rollout_first else (baseline_obj, response_obj)
            result = await self._run_single_comparison(
                conversation_history,
                first,
                second,
                pair_idx=(response_index, baseline_index),
                principle=principle,
                expected_rubric_ids=expected_rubric_ids,
            )
            if rollout_first:
                return result
            score_1, score_2, ranking, overall_1, overall_2, overall_ranking, *diagnostics = result
            return (
                score_2,
                score_1,
                7.0 - ranking,
                overall_2,
                overall_1,
                7.0 - overall_ranking,
                *diagnostics,
            )

        tasks = []
        metadata = []
        for response_index, response in enumerate(response_objs):
            for judge_index in range(cfg.num_judges_per_comparison):
                tasks.append(asyncio.create_task(compare_one(response_index, response, judge_index)))
                metadata.append((response_index, baseline_index, judge_index))
        try:
            raw_results = list(await asyncio.gather(*tasks))
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        # The appended baseline is diagnostic only; adjustments apply to the trainable prefix.
        return self._aggregate_results(
            response_objs + [baseline_obj], raw_results, metadata, trainable_count=len(response_objs)
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
                rewards=[cfg.default_score] * num_responses,
                comparison_results=None,
                metrics=None,
            )
        try:
            rewards, metrics, comparison_results, comparison_metadata = await self._run_compare(
                conversation_history,
                response_objs,
                principle=body.principle,
                expected_rubric_ids=body.expected_rubric_ids,
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
            Selected and overall scores, token metrics, and failure flags.
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

        async def call():
            # The outer deadline also bounds ServerClient's connection retries.
            async with asyncio.timeout(cfg.judge_request_timeout_s):
                response = await self.server_client.post(
                    server_name=cfg.genrm_model_server.name,
                    url_path="/v1/responses",
                    json=responses_create_params,
                )
                await raise_for_status(response)
                return await get_response_json(response)

        call_context = (
            f"GenRM judge {cfg.genrm_model_server.name} /v1/responses pair={pair_idx} "
            f"deadline={cfg.judge_request_timeout_s}s"
        )

        max_attempts = max(1, int(cfg.genrm_parse_retries) + 1)
        saw_completed_answer = False
        budget_exhausted_attempts = 0
        input_tokens = 0.0
        output_tokens = 0.0
        max_output_tokens_hit = 0.0
        usage_available = False
        for attempt_idx in range(max_attempts):
            try:
                raw_response = await call()
            except Exception as error:
                retryable = isinstance(error, (ClientPayloadError, ClientConnectionError)) or (
                    isinstance(error, ClientResponseError)
                    and (error.status in (408, 429) or 500 <= error.status < 600)
                )
                if retryable and attempt_idx < max_attempts - 1:
                    await asyncio.sleep(float(cfg.genrm_parse_retry_sleep_s))
                    continue
                content = getattr(error, "response_content", b"")
                if isinstance(content, bytes):
                    content = content[:1000].decode("utf-8", errors="replace")
                detail = f"; response={str(content)[:1000]}" if content else ""
                raise JudgeError(f"{call_context}: {type(error).__name__}: {str(error)[:1000]}{detail}") from error
            # Missing or malformed usage must not discard a valid judge verdict.
            usage = raw_response.get("usage") if isinstance(raw_response, dict) else None
            if not isinstance(usage, dict):
                usage = {}
            attempt_input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
            attempt_output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
            if (
                isinstance(attempt_input_tokens, (int, float))
                and not isinstance(attempt_input_tokens, bool)
                and isinstance(attempt_output_tokens, (int, float))
                and not isinstance(attempt_output_tokens, bool)
            ):
                # Retry attempts count because each consumes judge capacity.
                input_tokens += float(attempt_input_tokens)
                output_tokens += float(attempt_output_tokens)
                usage_available = True
            _, answer = extract_from_response_obj(raw_response)
            usable = (
                isinstance(raw_response, dict)
                and (raw_response.get("status") or "completed") == "completed"
                and bool(answer.strip())
            )
            saw_completed_answer |= usable
            if _output_budget_exhausted(raw_response):
                budget_exhausted_attempts += 1
                max_output_tokens_hit = 1.0
                logger.warning(
                    "GenRM output budget exhausted for pair %s (attempt %s/%s, max_output_tokens=%s): "
                    "response is incomplete (reason=max_output_tokens) and contains no verdict",
                    pair_idx,
                    attempt_idx + 1,
                    max_attempts,
                    responses_create_params.max_output_tokens,
                )
            token_metrics = (
                (input_tokens, output_tokens, max_output_tokens_hit) if usage_available else (-1.0, -1.0, -1.0)
            )
            if usable:
                try:
                    overall = parse_genrm_output(
                        answer,
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
                            answer,
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
                if not selected_failed:
                    return (*selected, *overall, *token_metrics, overall_failed, rubric_failed)
            else:
                overall = (cfg.default_score, cfg.default_score, cfg.default_ranking)
                selected = overall
                overall_failed = 1.0
                rubric_failed = 1.0 if cfg.score_source == "rubric_mean" else 0.0

            error = GenRMOutputParseError("Judge returned an empty, unsuccessful, or malformed response")
            if attempt_idx < max_attempts - 1:
                await asyncio.sleep(float(cfg.genrm_parse_retry_sleep_s))
                continue
            if not saw_completed_answer:
                message = f"Judge returned no completed answer after {max_attempts} attempts"
                if budget_exhausted_attempts:
                    message += (
                        f" ({budget_exhausted_attempts} of them exhausted max_output_tokens="
                        f"{responses_create_params.max_output_tokens} without emitting a verdict; "
                        "raise the budget or constrain the judge's reasoning)"
                    )
                raise JudgeError(message) from error
            # Preserve main's fallback for completed, nonempty but malformed answers.
            logger.warning(
                "GenRM %s parse failed for pair %s after %s attempts; using defaults",
                cfg.score_source,
                pair_idx,
                max_attempts,
            )
            return (*selected, *overall, *token_metrics, overall_failed, rubric_failed)


if __name__ == "__main__":
    GenRMCompareResourcesServer.run_webserver()
