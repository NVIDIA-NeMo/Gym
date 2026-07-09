# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Standardized trajectory telemetry built on Gym's native Responses API contracts.

A :class:`Trajectory` is the standardized rollout record requested by issue #1867: it
carries everything needed for cost estimation, debugging, and rollout analysis in one
versioned, validatable artifact. Conversation content is represented with Gym's native
Responses API types (``NeMoGymResponseInputItem``: messages, function calls, function
call outputs, reasoning items) and per-model-call token stats with the native
``NeMoGymResponseUsage`` — no parallel content schema to keep in sync.

Telemetry the Responses API does not standardize (tool execution timing, provider
request/response identity, tool errors, raw provider usage) lives in
:class:`TrajectorySpan` records, modeled after the OpenAI Agents SDK tracing spans
(``generation_span`` / ``function_span`` with ``started_at`` / ``ended_at`` /
``error``): spans decorate the native items via ``call_id`` / ``response_id`` instead
of polluting them.

Reconstruction semantics (delta / append-only representation):

- ``steps`` is ordered. Each step holds only the *new* items it introduced: a user
  message, or one complete agent turn (one model call) with its output items and the
  resulting ``function_call_output`` items.
- The model-visible input of a step is every item of every earlier step, starting from
  the most recent ``context_boundary`` step (compaction): the boundary's items are the
  summary that replaced prior history. :func:`reconstruct_model_input` resolves this
  into a native input list, and :func:`to_response_create_params` wraps it in a
  ``NeMoGymResponseCreateParamsNonStreaming``.

Agent harnesses should not build these models by hand — parse provider artifacts and
drive a :class:`TrajectoryBuilder` (see ``responses_api_agents/claude_code_agent/
trajectory.py`` for a reference adapter). Mandatory fields are non-optional on the
models; optional telemetry a source cannot provide stays ``None`` — never fabricated.
"""

from datetime import datetime
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field

from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseInputItem,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseReasoningItem,
    NeMoGymResponseUsage,
    NeMoGymSummary,
)


TRAJECTORY_SCHEMA_VERSION = "1.0"


class TrajectorySpanError(BaseModel):
    message: str
    data: Optional[dict[str, Any]] = None


class TrajectorySpan(BaseModel):
    """Execution telemetry for one model call ("generation") or one tool call ("function").

    Spans reference the native items they decorate: function spans via ``call_id``
    (matching a ``function_call`` / ``function_call_output`` item), generation spans via
    ``response_id`` / ``request_id`` (provider message and HTTP request identity).
    ``extra`` holds provider-specific metadata verbatim (e.g. raw usage with cache
    creation stats, sandbox/tool execution details) so no information is dropped even
    when it has no native field.
    """

    type: Literal["generation", "function"]
    call_id: Optional[str] = None
    response_id: Optional[str] = None
    request_id: Optional[str] = None
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    duration_ms: Optional[float] = None
    error: Optional[TrajectorySpanError] = None
    extra: Optional[dict[str, Any]] = None


class TrajectoryStep(BaseModel):
    step_id: int
    type: Literal["user_message", "agent_turn", "context_boundary"]
    timestamp: Optional[str] = None
    # Native Gym Responses items introduced by this step, in order. For an agent turn:
    # reasoning / message / function_call output items followed by the function_call_output
    # items produced by its tool calls.
    items: list[NeMoGymResponseInputItem] = Field(default_factory=list)
    # agent_turn only: 1-based model-call counter and native per-call token stats.
    turn_no: Optional[int] = None
    model: Optional[str] = None
    usage: Optional[NeMoGymResponseUsage] = None
    stop_reason: Optional[str] = None
    spans: list[TrajectorySpan] = Field(default_factory=list)


def _zero_usage() -> NeMoGymResponseUsage:
    return NeMoGymResponseUsage(
        input_tokens=0,
        input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=0),
        output_tokens=0,
        output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
        total_tokens=0,
    )


class Trajectory(BaseModel):
    schema_version: str = TRAJECTORY_SCHEMA_VERSION
    agent: str
    # Which artifact the trajectory was derived from (e.g. "transcript", "stream_json").
    source: str
    session_id: Optional[str] = None
    model: Optional[str] = None
    steps: list[TrajectoryStep] = Field(default_factory=list)
    # Native run totals, summed over the per-turn usage.
    usage: NeMoGymResponseUsage = Field(default_factory=_zero_usage)
    num_turns: Optional[int] = None
    duration_ms: Optional[float] = None
    total_cost_usd: Optional[float] = None
    # The provider's own end-of-run usage report, verbatim, for cross-checking.
    provider_usage: Optional[dict[str, Any]] = None
    # Records the adapter saw but did not represent (e.g. {"sidechain": 3}), so consumers
    # can tell "nothing happened" from "events were dropped".
    dropped_records: dict[str, int] = Field(default_factory=dict)
    extra: Optional[dict[str, Any]] = None


def validate_trajectory(data: dict[str, Any]) -> Trajectory:
    """Validate a serialized trajectory against the versioned schema."""
    return Trajectory.model_validate(data)


def usage_from_provider(raw: dict[str, Any]) -> NeMoGymResponseUsage:
    """Map a provider usage dict (Anthropic or OpenAI dialect) onto the native usage model.

    Unreported detail counters default to 0 per the OpenAI contract; the raw dict should be
    preserved verbatim on the generation span's ``extra`` so nothing is lost.
    """
    input_tokens = int(raw.get("input_tokens") or 0)
    output_tokens = int(raw.get("output_tokens") or 0)
    input_details = raw.get("input_tokens_details") or {}
    output_details = raw.get("output_tokens_details") or {}
    cached = raw.get("cache_read_input_tokens", input_details.get("cached_tokens")) or 0
    reasoning = raw.get("reasoning_tokens", output_details.get("reasoning_tokens")) or 0
    total = int(raw.get("total_tokens") or (input_tokens + output_tokens))
    return NeMoGymResponseUsage(
        input_tokens=input_tokens,
        input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=int(cached)),
        output_tokens=output_tokens,
        output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=int(reasoning)),
        total_tokens=total,
    )


def reconstruct_model_input(
    trajectory: Trajectory, before_step_id: Optional[int] = None
) -> list[NeMoGymResponseInputItem]:
    """Resolve the delta representation into the model-visible input item list.

    Returns the flattened items of every step before ``before_step_id`` (or all steps when
    None), starting from the most recent ``context_boundary`` — i.e. exactly what the model
    call at that step saw, including after compaction.
    """
    steps = [s for s in trajectory.steps if before_step_id is None or s.step_id < before_step_id]
    start = 0
    for index, step in enumerate(steps):
        if step.type == "context_boundary":
            start = index
    return [item for step in steps[start:] for item in step.items]


def to_response_create_params(
    trajectory: Trajectory, before_step_id: Optional[int] = None
) -> NeMoGymResponseCreateParamsNonStreaming:
    """Package a reconstructed model input as native Responses create params."""
    return NeMoGymResponseCreateParamsNonStreaming(
        input=reconstruct_model_input(trajectory, before_step_id=before_step_id),
        model=trajectory.model,
    )


def _parse_timestamp(value: Any) -> Optional[datetime]:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


class TrajectoryBuilder:
    """Incremental, agent-agnostic assembly of a :class:`Trajectory`.

    Adapters parse their provider's artifacts in order and call the ``add_*`` methods;
    the builder owns the shared semantics — turn numbering, model-call deduplication
    (``start_agent_turn`` with the same ``response_id`` continues the current turn
    instead of double counting usage), tool call/observation correlation with
    independent per-call timing, orphan handling, and usage totals.
    """

    def __init__(self, agent: str, source: str) -> None:
        self._trajectory = Trajectory(agent=agent, source=source)
        # call_id -> (step_id that issued the call, its timestamp) for observation correlation.
        self._pending_calls: dict[str, tuple[int, Optional[str]]] = {}
        self._turn_count = 0

    @property
    def steps(self) -> list[TrajectoryStep]:
        return self._trajectory.steps

    def set_session_id(self, session_id: Optional[str]) -> None:
        if self._trajectory.session_id is None and session_id:
            self._trajectory.session_id = session_id

    def add_user_message(self, content: str, timestamp: Optional[str] = None) -> TrajectoryStep:
        step = TrajectoryStep(
            step_id=len(self.steps),
            type="user_message",
            timestamp=timestamp,
            items=[NeMoGymEasyInputMessage(role="user", content=content)],
        )
        self.steps.append(step)
        return step

    def add_context_boundary(self, summary: str = "", timestamp: Optional[str] = None) -> TrajectoryStep:
        """Record a compaction: `summary` is the content that replaced the prior history."""
        items: list[NeMoGymResponseInputItem] = []
        if summary:
            items.append(NeMoGymEasyInputMessage(role="user", content=summary))
        step = TrajectoryStep(step_id=len(self.steps), type="context_boundary", timestamp=timestamp, items=items)
        self.steps.append(step)
        return step

    def start_agent_turn(
        self,
        response_id: Optional[str] = None,
        request_id: Optional[str] = None,
        model: Optional[str] = None,
        timestamp: Optional[str] = None,
        stop_reason: Optional[str] = None,
        provider_usage: Optional[dict[str, Any]] = None,
    ) -> TrajectoryStep:
        """Start (or continue) the step for one model call.

        Providers may emit one record per content block of the same API message; calling
        this again with the current turn's ``response_id`` returns that step so content
        accumulates without double counting usage.
        """
        current = self.steps[-1] if self.steps else None
        if (
            current is not None
            and current.type == "agent_turn"
            and response_id is not None
            and self._generation_span(current).response_id == response_id
        ):
            if stop_reason:
                current.stop_reason = stop_reason
            return current

        self._turn_count += 1
        span = TrajectorySpan(
            type="generation",
            response_id=response_id,
            request_id=request_id,
            # A provider record is written when the message completes; the generation's
            # start is not observable from artifacts, so only ended_at is set.
            ended_at=timestamp,
            extra={"provider_usage": provider_usage} if provider_usage else None,
        )
        step = TrajectoryStep(
            step_id=len(self.steps),
            type="agent_turn",
            timestamp=timestamp,
            turn_no=self._turn_count,
            model=model,
            usage=usage_from_provider(provider_usage) if provider_usage else None,
            stop_reason=stop_reason,
            spans=[span],
        )
        self.steps.append(step)
        if model and self._trajectory.model is None:
            self._trajectory.model = model
        return step

    def _generation_span(self, step: TrajectoryStep) -> TrajectorySpan:
        return next(s for s in step.spans if s.type == "generation")

    def _current_turn(self) -> TrajectoryStep:
        current = self.steps[-1] if self.steps else None
        if current is None or current.type != "agent_turn":
            raise ValueError("no agent turn in progress; call start_agent_turn() first")
        return current

    def add_output_text(self, text: str) -> None:
        step = self._current_turn()
        block = NeMoGymResponseOutputText(annotations=[], text=text)
        last_item = step.items[-1] if step.items else None
        if isinstance(last_item, NeMoGymResponseOutputMessage):
            last_item.content.append(block)
        else:
            step.items.append(
                NeMoGymResponseOutputMessage(id=f"msg-{step.step_id}-{len(step.items)}", content=[block])
            )

    def add_reasoning(self, text: str) -> None:
        step = self._current_turn()
        step.items.append(
            NeMoGymResponseReasoningItem(
                id=f"rs-{step.step_id}-{len(step.items)}",
                summary=[NeMoGymSummary(text=text, type="summary_text")],
            )
        )

    def add_tool_call(self, call_id: str, name: str, arguments: str) -> None:
        step = self._current_turn()
        step.items.append(
            NeMoGymResponseFunctionToolCall(call_id=call_id, name=name, arguments=arguments, status="completed")
        )
        self._pending_calls[call_id] = (step.step_id, step.timestamp)

    def add_tool_result(
        self,
        call_id: str,
        output: str,
        completed_at: Optional[str] = None,
        started_at: Optional[str] = None,
        error: Optional[Union[str, TrajectorySpanError]] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        """Attach a tool observation to the turn that issued ``call_id``.

        ``started_at`` defaults to the issuing turn's timestamp; passing it explicitly
        (e.g. from a provider execution record) overrides that. Each result carries its
        own timing, so parallel tool calls stay independently timed. Results that match
        no known call attach to the latest agent turn, or are counted as dropped when
        there is none.
        """
        step_id, registered_started = self._pending_calls.pop(call_id, (None, None))
        if step_id is None:
            step_id = next((s.step_id for s in reversed(self.steps) if s.type == "agent_turn"), None)
            if step_id is None:
                self.count_dropped("orphan_tool_results")
                return
        step = self.steps[step_id]

        started_at = started_at or registered_started
        duration_ms = None
        started_dt, completed_dt = _parse_timestamp(started_at), _parse_timestamp(completed_at)
        if started_dt is not None and completed_dt is not None:
            duration_ms = (completed_dt - started_dt).total_seconds() * 1000.0

        step.items.append(NeMoGymFunctionCallOutput(call_id=call_id, output=output, status="completed"))
        if isinstance(error, str):
            error = TrajectorySpanError(message=error)
        step.spans.append(
            TrajectorySpan(
                type="function",
                call_id=call_id,
                started_at=started_at,
                ended_at=completed_at,
                duration_ms=duration_ms,
                error=error,
                extra=extra,
            )
        )

    def count_dropped(self, kind: str) -> None:
        self._trajectory.dropped_records[kind] = self._trajectory.dropped_records.get(kind, 0) + 1

    def set_run_totals(
        self,
        num_turns: Optional[int] = None,
        duration_ms: Optional[float] = None,
        total_cost_usd: Optional[float] = None,
        provider_usage: Optional[dict[str, Any]] = None,
    ) -> None:
        if num_turns is not None:
            self._trajectory.num_turns = int(num_turns)
        if duration_ms is not None:
            self._trajectory.duration_ms = float(duration_ms)
        if total_cost_usd is not None:
            self._trajectory.total_cost_usd = float(total_cost_usd)
        if provider_usage is not None:
            self._trajectory.provider_usage = provider_usage

    def build(self) -> Trajectory:
        totals = _zero_usage()
        for step in self.steps:
            if step.usage is None:
                continue
            totals.input_tokens += step.usage.input_tokens
            totals.output_tokens += step.usage.output_tokens
            totals.total_tokens += step.usage.total_tokens
            totals.input_tokens_details.cached_tokens += step.usage.input_tokens_details.cached_tokens
            totals.output_tokens_details.reasoning_tokens += step.usage.output_tokens_details.reasoning_tokens
        self._trajectory.usage = totals
        return self._trajectory
