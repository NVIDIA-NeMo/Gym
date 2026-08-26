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
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.global_config import (
    GROUP_ATTEMPT_KEY_NAME,
    GROUP_ID_KEY_NAME,
    ROLLOUT_INDEX_KEY_NAME,
    TASK_INDEX_KEY_NAME,
)
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponseCreateParamsNonStreaming,
)
from resources_servers.genrm_compare.utils import (
    GenRMOutputParseError,
    aggregate_scores,
    extract_output_text,
    generate_comparison_pairs,
    get_prompt_key_from_input,
    parse_genrm_output,
)


logger = logging.getLogger(__name__)


class CohortEvaluationError(RuntimeError):
    """A cohort-level failure that callers should treat as retriable."""


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
    group_id: Optional[str] = None
    group_attempt: int = 0
    members: Dict[int, _CohortMember] = field(default_factory=dict)
    phase: Literal["collecting", "evaluating", "completed", "failed"] = "collecting"
    rewards: Dict[int, float] = field(default_factory=dict)
    failure: Optional[str] = None
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
        cohort_collection_timeout_s: Optional maximum time to wait for every logical rollout index
        cohort_result_ttl_s: Optional retention time for completed and failed cohort tombstones
        max_terminal_cohorts: Maximum number of completed and failed cohort tombstones
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
    cohort_collection_timeout_s: Optional[float] = Field(default=None, gt=0)
    cohort_result_ttl_s: Optional[float] = Field(default=3600.0, gt=0)
    max_terminal_cohorts: int = Field(default=4096, gt=0)

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


class GenRMCompareVerifyRequest(BaseVerifyRequest):
    """Verify request with optional principle for cohort-based GenRM comparison."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    principle: Optional[str] = None  # Principle for principle-based GenRM; forwarded by agent when provided
    task_index: Optional[int] = Field(default=None, alias=TASK_INDEX_KEY_NAME)
    group_id: Optional[str] = Field(default=None, alias=GROUP_ID_KEY_NAME)
    group_attempt: int = Field(default=0, alias=GROUP_ATTEMPT_KEY_NAME, ge=0)
    rollout_index: Optional[int] = Field(default=None, alias=ROLLOUT_INDEX_KEY_NAME)
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
            warnings.warn(
                f"{GROUP_ATTEMPT_KEY_NAME} was omitted for {GROUP_ID_KEY_NAME}={group_id!r}; "
                "treating this legacy request as group attempt zero",
                UserWarning,
                stacklevel=2,
            )
        return data


class GenRMCompareVerifyResponse(BaseVerifyResponse):
    """Verification response that echoes logical cohort coordinates."""

    model_config = ConfigDict(populate_by_name=True)

    group_id: Optional[str] = Field(default=None, alias=GROUP_ID_KEY_NAME)
    group_attempt: int = Field(alias=GROUP_ATTEMPT_KEY_NAME, ge=0)
    rollout_index: Optional[int] = Field(default=None, alias=ROLLOUT_INDEX_KEY_NAME)


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
    _verify_cohorts: Dict[str, _CohortState] = PrivateAttr(default_factory=dict)
    _latest_group_attempts: Dict[str, _GroupAttemptWatermark] = PrivateAttr(default_factory=dict)
    _cohort_registry_lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)

    async def verify(self, body: GenRMCompareVerifyRequest) -> GenRMCompareVerifyResponse:
        """Verify one logical rollout slot as part of a prompt cohort."""
        cfg = self.config
        principle = body.principle
        if cfg.num_rollouts_per_prompt <= 1:
            return self._verify_response(body, cfg.default_score)

        self._validate_logical_coordinates(body)
        input_messages = getattr(body.responses_create_params, "input", None) or []
        prompt_key = self._get_verify_cohort_key(
            body,
            input_messages if isinstance(input_messages, list) else list(input_messages),
            principle,
        )
        prompt_digest = get_prompt_key_from_input(
            input_messages if isinstance(input_messages, list) else list(input_messages),
            principle,
        )
        rollout_index = body.rollout_index
        assert rollout_index is not None  # Validated above; keeps the type narrow below.
        response_digest = self._response_digest(body.response)
        future: asyncio.Future[float] = asyncio.get_running_loop().create_future()
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
                    reward = cohort.rewards[rollout_index]
                    return self._verify_response(body, reward)
                if cohort.phase == "failed":
                    raise HTTPException(status_code=503, detail=cohort.failure or "GenRM cohort evaluation failed")
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
                if cohort.collection_timeout_task is None and cfg.cohort_collection_timeout_s is not None:
                    cohort.collection_timeout_task = asyncio.create_task(
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
                cohort.evaluation_task = asyncio.create_task(
                    self._evaluate_verify_cohort(prompt_key, cohort, members),
                    name=f"genrm-cohort-evaluation-{cohort_identity}",
                )

        # A disconnected request must not cancel the shared cohort result.
        try:
            reward = await asyncio.shield(future)
        except CohortEvaluationError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        except asyncio.CancelledError:
            # The logical member remains registered, but this HTTP request no
            # longer needs a result. Mark its waiter consumed so a later cohort
            # failure does not produce an unobserved Future exception.
            future.cancel()
            await asyncio.shield(self._remove_waiter(cohort, rollout_index, future))
            raise
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
        )

    def _validate_logical_coordinates(self, body: GenRMCompareVerifyRequest) -> None:
        """Reject malformed cohort members before mutating shared state."""
        expected_size = self.config.num_rollouts_per_prompt
        if body.task_index is None and not body.group_id:
            raise HTTPException(
                status_code=422,
                detail=f"either {TASK_INDEX_KEY_NAME} or {GROUP_ID_KEY_NAME} is required for cohort comparison",
            )
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
                cohort = _CohortState(prompt_digest=prompt_digest)
                self._verify_cohorts[prompt_key] = cohort
            return cohort

        async with self._cohort_registry_lock:
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
        waiter: asyncio.Future[float],
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
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
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
                response_objs.append(
                    member_body.response.model_dump()
                    if hasattr(member_body.response, "model_dump")
                    else member_body.response
                )

            rewards, _, _, _ = await self._run_compare(
                conversation_history=conversation_history,
                response_objs=response_objs,
                principle=first_body.principle,
            )
            if len(rewards) != len(sorted_indices):
                raise RuntimeError(f"GenRM returned {len(rewards)} rewards for {len(sorted_indices)} cohort members")
            reward_by_index = dict(zip(sorted_indices, rewards))
            await self._publish_verify_cohort(cohort, reward_by_index)
        except asyncio.CancelledError:
            await asyncio.shield(
                self._fail_verify_cohort(
                    cohort,
                    "GenRM cohort evaluation was cancelled",
                    expected_phase="evaluating",
                )
            )
            raise
        except Exception as error:
            logger.exception("GenRM cohort evaluation failed for %s", prompt_key)
            await self._fail_verify_cohort(
                cohort,
                f"GenRM cohort evaluation failed: {error}",
                expected_phase="evaluating",
            )

    @staticmethod
    async def _publish_verify_cohort(cohort: _CohortState, reward_by_index: Dict[int, float]) -> None:
        """Atomically publish a complete reward map and compact the tombstone."""
        async with cohort.lock:
            if cohort.phase != "evaluating":
                raise RuntimeError(f"cannot publish GenRM rewards while cohort is {cohort.phase}")
            cohort.rewards = reward_by_index
            cohort.phase = "completed"
            cohort.terminal_at = time.monotonic()
            cohort.evaluation_task = None
            for index, member in cohort.members.items():
                for waiter in member.waiters:
                    if not waiter.done():
                        waiter.set_result(reward_by_index[index])
                # A tombstone only needs the digest and reward. Do not retain
                # full response payloads for every completed training cohort.
                member.body = None
                member.waiters.clear()

    @staticmethod
    async def _fail_verify_cohort(
        cohort: _CohortState,
        message: str,
        *,
        expected_phase: Literal["collecting", "evaluating"],
    ) -> bool:
        async with cohort.lock:
            if cohort.phase != expected_phase:
                return False
            cohort.phase = "failed"
            cohort.failure = message
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

        # Build the request params
        responses_create_params = cfg.genrm_responses_create_params.model_copy(deep=True)
        responses_create_params.input = messages
        responses_create_params.metadata = metadata

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
