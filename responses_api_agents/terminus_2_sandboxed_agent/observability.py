# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evidence recorded at Terminus model-response and decision boundaries."""

from dataclasses import dataclass
from typing import Any

from harbor.llms.base import LLMResponse

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseFunctionToolCall, NeMoGymResponseOutputMessage
from nemo_gym.rollout_observability import (
    ContextCompactionObservation,
    ModelCallRef,
    ObservationGap,
    TrajectoryModelCall,
    TrajectoryModelCallAttempt,
    TrajectoryRecord,
    TrajectoryResponseMetadata,
    TrajectoryTokenStats,
    TrajectoryTurn,
)


@dataclass
class ObservedResponse:
    question: list[dict[str, Any]]
    response: NeMoGymResponse
    timestamp: float
    harbor_response: LLMResponse | None = None
    model_call_id: str | None = None


class TerminusObservations:
    def __init__(self, invocation_id: str, task_id: str, rollout_id: str, model_ref: ModelServerRef):
        self.invocation_id = invocation_id
        self.model_ref = model_ref
        self.trajectory = TrajectoryRecord(task_id=task_id, rollout_id=rollout_id)
        self.responses: dict[str, list[ObservedResponse]] = {}
        self.compactions: list[ContextCompactionObservation] = []
        self.compaction: ContextCompactionObservation | None = None
        self.decision_no = 0
        self.decision_response: LLMResponse | None = None
        self.selected_response_ids: set[str] = set()
        # The shared HTTP transport can retry after server admission without notifying
        # this adapter. Invocation ownership remains exact; full turn-call scope does not.
        self.gap("transport_retry_visibility_unavailable")

    def gap(self, code: str, detail: str | None = None) -> None:
        gap = ObservationGap(code=code, invocation_id=self.invocation_id, detail=detail)
        if gap not in self.trajectory.gaps:
            self.trajectory.gaps.append(gap)
        scope_gap = ObservationGap(code="turn_model_call_scope_incomplete", invocation_id=self.invocation_id)
        if scope_gap not in self.trajectory.gaps:
            self.trajectory.gaps.append(scope_gap)

    @staticmethod
    def _response_kind(response: NeMoGymResponse) -> str:
        for item in response.output:
            if isinstance(item, NeMoGymResponseFunctionToolCall):
                return "tool_call"
        for item in response.output:
            if isinstance(item, NeMoGymResponseOutputMessage) and item.content:
                return "text"
        return "other"

    def record_response(
        self,
        question: list[dict[str, Any]],
        response: NeMoGymResponse,
        timestamp: float,
        *,
        model_call_id: str | None = None,
        started_at: float | None = None,
        model_call_purpose: str | None = None,
        attempts: list[dict[str, Any]] | None = None,
    ) -> ObservedResponse:
        observed = ObservedResponse(
            question=question, response=response, timestamp=timestamp, model_call_id=model_call_id
        )
        same_id = self.responses.setdefault(response.id, [])
        same_id.append(observed)
        if len(same_id) > 1:
            self.gap("model_response_id_reused", response.id)
        if not response.id:
            self.gap("model_response_id_unavailable")
        ref = (
            (
                ModelCallRef(model_call_id=model_call_id, model_ref=self.model_ref, response_id=response.id)
                if model_call_id
                else ModelCallRef(model_ref=self.model_ref, response_id=response.id)
            )
            if response.id
            else None
        )
        if self.compaction is not None and ref is not None:
            self.compaction.model_calls.append(ref)
        if model_call_id is not None and started_at is not None:
            usage = response.usage
            token_stats = TrajectoryTokenStats()
            if usage is not None:
                token_stats = TrajectoryTokenStats(
                    prompt_tokens=usage.input_tokens,
                    completion_tokens=usage.output_tokens,
                    reasoning_tokens=(
                        usage.output_tokens_details.reasoning_tokens
                        if usage.output_tokens_details is not None
                        else None
                    ),
                    total_tokens=usage.total_tokens,
                    cached_tokens=(
                        usage.input_tokens_details.cached_tokens if usage.input_tokens_details is not None else None
                    ),
                )
            self.trajectory.model_calls.append(
                TrajectoryModelCall(
                    model_call_id=model_call_id,
                    started_at=started_at,
                    completed_at=timestamp,
                    duration_ms=(timestamp - started_at) * 1000,
                    response_metadata=TrajectoryResponseMetadata(
                        response_id=response.id,
                        model_ref=self.model_ref,
                    ),
                    token_stats=token_stats,
                    model_call_purpose=model_call_purpose,
                    model_response_kind=self._response_kind(response),
                    attempts=[
                        TrajectoryModelCallAttempt(attempt_index=index, **attempt)
                        for index, attempt in enumerate(attempts or [], start=1)
                    ],
                )
            )
        return observed

    def begin_decision(self) -> None:
        self.decision_no += 1
        self.decision_response = None

    def finish_decision(self, step_count: int) -> None:
        response = self.decision_response
        candidates = (
            self.responses.get(response.response_id, []) if response is not None and response.response_id else []
        )
        if len(candidates) != 1 or candidates[0].harbor_response is not response:
            self.gap("decision_model_response_unavailable", f"decision={self.decision_no}")
            return
        observed = candidates[0]
        self.selected_response_ids.add(observed.response.id)
        ref = (
            ModelCallRef(
                model_call_id=observed.model_call_id,
                model_ref=self.model_ref,
                response_id=observed.response.id,
            )
            if observed.model_call_id
            else ModelCallRef(model_ref=self.model_ref, response_id=observed.response.id)
        )
        self.trajectory.turns.append(
            TrajectoryTurn(
                invocation_id=self.invocation_id,
                task_id=self.trajectory.task_id,
                rollout_id=self.trajectory.rollout_id,
                turn_no=self.decision_no,
                timestamp=observed.timestamp,
                question=observed.question,
                answer=[item.model_dump(mode="json") for item in observed.response.output if item.type != "reasoning"],
                reasoning_content=response.reasoning_content,
                # Completed nonempty terminal-command batch attempts before this decision.
                # A returned timeout counts as a batch attempt, not as all commands having run.
                step_count=step_count,
                model_calls=[ref],
            )
        )

    def finish(self) -> TrajectoryRecord:
        if self.responses.keys() - self.selected_response_ids:
            self.gap("auxiliary_model_calls", "Some model responses were not selected as main-agent decisions.")
        if not self.trajectory.turns:
            self.trajectory.gaps.append(ObservationGap(code="turns_unavailable", invocation_id=self.invocation_id))

        for compaction in self.compactions:
            if compaction.completed_at is None:
                continue
            next_agent_call = next(
                (
                    m
                    for m in self.trajectory.model_calls
                    if m.started_at is not None
                    and m.started_at > compaction.completed_at
                    and m.model_call_purpose not in ("compaction_summary", "compaction_question")
                ),
                None,
            )
            if next_agent_call is not None and next_agent_call.token_stats.prompt_tokens is not None:
                compaction.tokens_after = next_agent_call.token_stats.prompt_tokens

        return self.trajectory
