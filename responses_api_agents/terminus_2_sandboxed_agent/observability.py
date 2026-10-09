# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evidence recorded at Terminus model-response and decision boundaries."""

import socket
from dataclasses import dataclass
from time import perf_counter, time
from typing import Any
from uuid import uuid4

from harbor.llms.base import LLMResponse

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseFunctionToolCall, NeMoGymResponseOutputMessage
from nemo_gym.rollout_observability import (
    COMPACTION_PURPOSES,
    AgentPhaseObservation,
    ContextCompactionObservation,
    ModelCallRef,
    ObservationGap,
    TrajectoryModelCall,
    TrajectoryModelCallAttempt,
    TrajectoryRecord,
    TrajectoryResponseMetadata,
    TrajectoryTokenStats,
    TrajectoryToolCall,
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
        # Wall clock when the latest decision finished parsing; the next command batch was requested then.
        self.decision_parsed_at: float | None = None
        # Tool calls whose result has not yet been handed to a model call.
        self.pending_tool_calls: list[TrajectoryToolCall] = []
        # The last command batch, until the model call that reads its observation names the prompt item.
        self.last_batch_tools: list[TrajectoryToolCall] = []
        self.phases: list[AgentPhaseObservation] = []
        # Every timestamp here is read on this process's clock; durations come from perf_counter.
        self.clock_domain = f"node:{socket.gethostname()}"
        self.last_decision_call_id: str | None = None
        self.last_rejected_call_id: str | None = None
        self._compaction_started_perf: float | None = None
        self._tokens_after_pending: ContextCompactionObservation | None = None
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
        duration_ms: float | None = None,
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
                    client_call_id=model_call_id,
                    invocation_id=self.invocation_id,
                    clock_domain=self.clock_domain,
                    started_at=started_at,
                    completed_at=timestamp,
                    duration_ms=duration_ms if duration_ms is not None else (timestamp - started_at) * 1000,
                    status="ok",
                    response_metadata=TrajectoryResponseMetadata(
                        response_id=response.id,
                        model_ref=self.model_ref,
                    ),
                    token_stats=token_stats,
                    model_call_purpose=model_call_purpose,
                    model_response_kind=self._response_kind(response),
                    attempts=self._attempts(attempts),
                )
            )
            pending = self._tokens_after_pending
            if pending is not None and model_call_purpose not in COMPACTION_PURPOSES:
                if token_stats.prompt_tokens is not None:
                    pending.tokens_after = token_stats.prompt_tokens
                self._tokens_after_pending = None
        return observed

    def _attempts(self, attempts: list[dict[str, Any]] | None) -> list[TrajectoryModelCallAttempt]:
        return [
            TrajectoryModelCallAttempt(attempt_index=index, clock_domain=self.clock_domain, **attempt)
            for index, attempt in enumerate(attempts or [], start=1)
        ]

    def _call(self, model_call_id: str | None) -> TrajectoryModelCall | None:
        return next((c for c in self.trajectory.model_calls if c.model_call_id == model_call_id), None)

    def record_failed_call(
        self,
        *,
        model_call_id: str,
        started_at: float,
        completed_at: float,
        duration_ms: float,
        status: str,
        model_call_purpose: str,
        attempts: list[dict[str, Any]],
    ) -> None:
        """A logical call that never produced a response is still a call the agent waited for."""
        self.trajectory.model_calls.append(
            TrajectoryModelCall(
                model_call_id=model_call_id,
                client_call_id=model_call_id,
                invocation_id=self.invocation_id,
                clock_domain=self.clock_domain,
                started_at=started_at,
                completed_at=completed_at,
                duration_ms=duration_ms,
                status=status,
                model_call_purpose=model_call_purpose,
                attempts=self._attempts(attempts),
            )
        )

    def mark_rejected(self, model_call_id: str) -> None:
        """Gym answered an over-context request with an empty response: the call was rejected, not answered."""
        call = self._call(model_call_id)
        if call is not None:
            call.status = "incomplete"
            call.context_overflow = True
        self.last_rejected_call_id = model_call_id

    def note_decision_result(self, parsed_tool_calls: int, task_complete: bool) -> None:
        call = self._call(self.last_decision_call_id)
        if call is not None:
            call.parsed_tool_calls = parsed_tool_calls
            call.task_complete = task_complete

    def command_record(
        self,
        *,
        batch_id: str,
        batch_index: int,
        keystrokes: str,
        requested_wait_s: float,
        started_at: float,
        started_perf: float,
        status: str,
    ) -> TrajectoryToolCall:
        """One terminal command, timed on its own: send_keys plus the wait the model asked for."""
        return TrajectoryToolCall(
            invocation_id=self.invocation_id,
            tool_call_id=f"cmd_{uuid4().hex[:12]}",
            tool_name="terminal",
            operation=keystrokes.rstrip("\r\n")[:512] if keystrokes else None,
            requested_at=self.decision_parsed_at,
            started_at=started_at,
            completed_at=time(),
            duration_ms=(perf_counter() - started_perf) * 1000,
            requested_wait_s=requested_wait_s,
            batch_id=batch_id,
            batch_index=batch_index,
            timing_source="harness",
            clock_domain=self.clock_domain,
            status=status,
        )

    def record_command_batch(self, tools: list[TrajectoryToolCall], *, output_chars: int) -> None:
        for tool in tools:
            tool.output_chars = output_chars
        self.trajectory.tool_calls.extend(tools)
        self.pending_tool_calls.extend(tools)
        self.last_batch_tools = tools

    def note_observation_index(self, index: int) -> None:
        """The prompt item at ``index`` carries the last batch's observation to the model."""
        for tool in self.last_batch_tools:
            tool.output_trajectory_index = index
        self.last_batch_tools = []

    def begin_compaction(self, *, trigger: str, tokens_before: int | None) -> ContextCompactionObservation:
        rejected = self.last_rejected_call_id if trigger == "overflow" else None
        self.compaction = ContextCompactionObservation(
            invocation_id=self.invocation_id,
            observed_at=time(),
            trigger=trigger,
            tokens_before=tokens_before,
            clock_domain=self.clock_domain,
            before_model_call=ModelCallRef(model_call_id=rejected) if rejected else None,
        )
        self._compaction_started_perf = perf_counter()
        return self.compaction

    def end_compaction(self, outcome: str) -> None:
        compaction = self.compaction
        if compaction is None:
            return
        compaction.outcome = outcome
        compaction.completed_at = time()
        if self._compaction_started_perf is not None:
            compaction.duration_ms = (perf_counter() - self._compaction_started_perf) * 1000
        self.compactions.append(compaction)
        if outcome == "completed":
            self._tokens_after_pending = compaction
        self.compaction = None
        self._compaction_started_perf = None

    def record_phase(
        self,
        name: str,
        *,
        started_at: float,
        started_perf: float,
        status: str = "completed",
        error_type: str | None = None,
    ) -> None:
        self.phases.append(
            AgentPhaseObservation(
                name=name,
                started_at=started_at,
                completed_at=time(),
                duration_ms=(perf_counter() - started_perf) * 1000,
                status=status,
                error_type=error_type,
                clock_domain=self.clock_domain,
            )
        )

    def mark_tool_results_observed(self) -> None:
        """The previous batch's output is in the prompt for the next model call.

        Harbor forms the prompt right after _execute_commands returns and then, at the top of the
        next iteration, runs the proactive-summarization check before the decision. Calling this
        from both hooks and clearing the list means the earliest one wins, so a compaction call in
        between is not folded into tool_observation_delay. Same clock as the batch timings.
        """
        if not self.pending_tool_calls:
            return
        received_at = time()
        for tool in self.pending_tool_calls:
            tool.response_received_at = received_at
        self.pending_tool_calls = []

    def begin_decision(self) -> None:
        self.mark_tool_results_observed()
        self.decision_no += 1
        self.decision_response = None

    def finish_decision(self, step_count: int) -> None:
        self.decision_parsed_at = time()
        response = self.decision_response
        candidates = (
            self.responses.get(response.response_id, []) if response is not None and response.response_id else []
        )
        if len(candidates) != 1 or candidates[0].harbor_response is not response:
            self.gap("decision_model_response_unavailable", f"decision={self.decision_no}")
            return
        observed = candidates[0]
        self.selected_response_ids.add(observed.response.id)
        self.last_decision_call_id = observed.model_call_id
        decision_call = self._call(observed.model_call_id)
        if decision_call is not None:
            decision_call.processed_at = self.decision_parsed_at
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
                    and m.model_call_purpose not in COMPACTION_PURPOSES
                ),
                None,
            )
            if next_agent_call is not None and next_agent_call.token_stats.prompt_tokens is not None:
                compaction.tokens_after = next_agent_call.token_stats.prompt_tokens

        return self.trajectory
