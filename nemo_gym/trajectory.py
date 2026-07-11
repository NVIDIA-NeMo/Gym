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

"""Agent trajectory support over Gym's native Responses contract (issue #1867).

The trajectory entity **is** the ``NeMoGymResponse``. Its ``output`` item list is the
episode — lossless and in execution order — and the telemetry rides on the contract
itself (see ``nemo_gym.openai_utils``), following the same pattern the ``*ForTraining``
variants use for token IDs:

- **On the items**: model-produced items are the ``*WithAgentTelemetry`` variants
  carrying ``agent_step_no`` (which model call produced/observed them);
  ``function_call_output`` items additionally carry ``execution``
  (:class:`~nemo_gym.openai_utils.NeMoGymToolExecution`: independent per-call timing,
  errors, provider execution metadata); a compaction summary is a
  :class:`~nemo_gym.openai_utils.NeMoGymContextBoundaryMessage`.
- **On the Response**: ``generations`` (one
  :class:`~nemo_gym.openai_utils.NeMoGymGeneration` per agent step: native per-call
  usage, provider identity, ``stop_reason``) and ``agent_telemetry`` (run-level:
  provenance ``source``, ``session_id``, ``num_agent_steps``, duration/cost, verbatim
  provider usage, ``dropped_records``). Model servers leave both ``None``.

Terminology: an **agent step** is one interaction with the environment through the
model — a single LLM generation plus the orchestration of its tool calls and their
outputs (one agent-loop iteration; ``max_steps`` counts these). A **turn** is a full
cycle of control, from user input until the agent hands control back, containing one or
more agent steps; turns are derivable from ``role: "user"`` items in the output.

Nothing is stored twice: the task's initial prompt lives only in
``responses_create_params.input`` (pass it to :func:`reconstruct_model_input` as
``base_input``), and per-step grouping is the ``agent_step_no`` tag rather than a copy
of the items. Because the output is append-only in execution order, any step's exact
model-visible input is derivable — compaction-aware — from the single stored copy.

Capture is contract-agnostic: in-process loops drive :class:`TrajectoryBuilder` as they
execute (exact boundaries, measured tool timing); black-box harness wrappers reconstruct
post-hoc from artifacts (see ``responses_api_agents/claude_code_agent/trajectory.py``).
Fidelity differences surface as data — ``source`` labels provenance, unobservable
telemetry stays ``None`` (never fabricated), ``dropped_records`` counts events seen but
not represented.
"""

from datetime import datetime
from typing import Any, Iterator, Optional, Union

from nemo_gym.openai_utils import (
    NeMoGymAgentTelemetry,
    NeMoGymContextBoundaryMessage,
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutputWithAgentTelemetry,
    NeMoGymGeneration,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCallWithAgentTelemetry,
    NeMoGymResponseInputItem,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputMessageWithAgentTelemetry,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseReasoningItemWithAgentTelemetry,
    NeMoGymResponseUsage,
    NeMoGymSummary,
    NeMoGymToolExecution,
    NeMoGymToolExecutionError,
)


def zero_usage() -> NeMoGymResponseUsage:
    return NeMoGymResponseUsage(
        input_tokens=0,
        input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=0),
        output_tokens=0,
        output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
        total_tokens=0,
    )


def usage_from_provider(raw: dict[str, Any]) -> NeMoGymResponseUsage:
    """Map a provider usage dict (Anthropic or OpenAI dialect) onto the native usage model.

    Unreported detail counters default to 0 per the OpenAI contract; the raw dict is
    preserved verbatim on the agent-step record so nothing is lost.
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


def summed_usage(generations: list[NeMoGymGeneration]) -> NeMoGymResponseUsage:
    """Sum the per-agent-step native usage into run totals."""
    totals = zero_usage()
    for generation in generations:
        if generation.usage is None:
            continue
        totals.input_tokens += generation.usage.input_tokens
        totals.output_tokens += generation.usage.output_tokens
        totals.total_tokens += generation.usage.total_tokens
        totals.input_tokens_details.cached_tokens += generation.usage.input_tokens_details.cached_tokens
        totals.output_tokens_details.reasoning_tokens += generation.usage.output_tokens_details.reasoning_tokens
    return totals


def agent_step_slices(
    output: list[NeMoGymResponseInputItem],
) -> Iterator[tuple[int, list[NeMoGymResponseInputItem]]]:
    """Yield ``(agent_step_no, items)`` for each agent step, from the items' tags."""
    current_step: Optional[int] = None
    current_items: list[NeMoGymResponseInputItem] = []
    for item in output:
        step = getattr(item, "agent_step_no", None)
        if step is None:
            continue
        if step != current_step:
            if current_step is not None:
                yield current_step, current_items
            current_step, current_items = step, []
        current_items.append(item)
    if current_step is not None:
        yield current_step, current_items


def reconstruct_model_input(
    output: list[NeMoGymResponseInputItem],
    agent_step_no: Optional[int] = None,
    base_input: Optional[list[NeMoGymResponseInputItem]] = None,
) -> list[NeMoGymResponseInputItem]:
    """Resolve the model-visible input of an agent step from the output item list.

    With ``agent_step_no`` set, returns what that step's model call saw:
    ``base_input + output[:first item tagged with that step]`` — or, if a
    ``NeMoGymContextBoundaryMessage`` precedes the step, the context restarts at that
    boundary (its summary replaced everything earlier, including the base input). With
    ``agent_step_no=None``, returns the full final context. ``base_input`` is the task's
    original ``responses_create_params.input``, deliberately stored only there.
    """
    if agent_step_no is None:
        end = len(output)
    else:
        # First item belonging to this step or a later one; a step that produced no
        # items (e.g. the run ended mid-step) saw everything recorded before it.
        end = next(
            (i for i, item in enumerate(output) if (getattr(item, "agent_step_no", None) or 0) >= agent_step_no),
            len(output),
        )

    boundary_index: Optional[int] = None
    for index in range(end):
        if isinstance(output[index], NeMoGymContextBoundaryMessage):
            boundary_index = index
    if boundary_index is not None:
        return list(output[boundary_index:end])
    return list(base_input or []) + list(output[:end])


def to_response_create_params(
    output: list[NeMoGymResponseInputItem],
    agent_step_no: Optional[int] = None,
    base_input: Optional[list[NeMoGymResponseInputItem]] = None,
    model: Optional[str] = None,
) -> NeMoGymResponseCreateParamsNonStreaming:
    """Package a reconstructed model input as native Responses create params."""
    return NeMoGymResponseCreateParamsNonStreaming(
        input=reconstruct_model_input(output, agent_step_no=agent_step_no, base_input=base_input),
        model=model,
    )


def _parse_timestamp(value: Any) -> Optional[datetime]:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


class TrajectoryBuilder:
    """Incremental, agent-agnostic assembly of an agent's Response content and telemetry.

    Call the ``add_*`` methods in execution order (from an in-process loop, or from an
    adapter replaying a black-box harness's artifacts). The builder appends the
    telemetry-tagged native items to ``output`` and maintains the per-generation records
    in lockstep: agent-step numbering, model-call deduplication (``start_agent_step``
    with the current step's ``response_id`` continues it instead of double counting
    usage), call/execution correlation with independent per-call timing, orphan
    handling.

    ``build()`` returns ``(output_items, generations, agent_telemetry)`` — everything an
    agent server needs to construct its ``NeMoGymResponse``.
    """

    def __init__(self, agent: str, source: str) -> None:
        self.output: list[NeMoGymResponseInputItem] = []
        self.generations: list[NeMoGymGeneration] = []
        self._telemetry = NeMoGymAgentTelemetry(agent=agent, source=source)
        self._model: Optional[str] = None
        # call_id -> (agent_step_no, issue timestamp) for execution timing correlation.
        self._pending_calls: dict[str, tuple[int, Optional[str]]] = {}
        self._open = False  # whether items may still be appended to the last generation

    def _current_agent_step(self) -> NeMoGymGeneration:
        if not self.generations or not self._open:
            raise ValueError("no agent step in progress; call start_agent_step() first")
        return self.generations[-1]

    def set_session_id(self, session_id: Optional[str]) -> None:
        if self._telemetry.session_id is None and session_id:
            self._telemetry.session_id = session_id

    def add_user_message(self, content: str, timestamp: Optional[str] = None) -> None:
        """Record a mid-episode user message (a turn boundary).

        The task's *initial* input must not be added here — it lives in
        ``responses_create_params.input`` and is passed to reconstruction as ``base_input``.
        """
        self._open = False
        self.output.append(NeMoGymEasyInputMessage(role="user", content=content))

    def add_context_boundary(self, summary: str = "", timestamp: Optional[str] = None) -> None:
        """Record a compaction: `summary` is the content that replaced all prior history."""
        self._open = False
        self.output.append(NeMoGymContextBoundaryMessage(role="user", content=summary, context_boundary=True))

    def start_agent_step(
        self,
        response_id: Optional[str] = None,
        request_id: Optional[str] = None,
        model: Optional[str] = None,
        timestamp: Optional[str] = None,
        stop_reason: Optional[str] = None,
        provider_usage: Optional[dict[str, Any]] = None,
    ) -> NeMoGymGeneration:
        """Start (or continue) the agent step for one model call.

        Providers may emit one record per content block of the same API message; calling
        this again with the current step's ``response_id`` returns that generation so
        content accumulates under its tag without double counting usage.
        """
        current = self.generations[-1] if self.generations else None
        if current is not None and self._open and response_id is not None and current.response_id == response_id:
            if stop_reason:
                current.stop_reason = stop_reason
            return current

        generation = NeMoGymGeneration(
            agent_step_no=len(self.generations) + 1,
            model=model,
            stop_reason=stop_reason,
            response_id=response_id,
            request_id=request_id,
            # A provider record is written when the message completes; the generation's
            # start is not observable from artifacts, so only the completion time is kept.
            ended_at=timestamp,
            usage=usage_from_provider(provider_usage) if provider_usage else None,
            provider_usage=provider_usage,
        )
        self.generations.append(generation)
        self._open = True
        if model and self._model is None:
            self._model = model
        return generation

    def add_output_text(self, text: str) -> None:
        generation = self._current_agent_step()
        block = NeMoGymResponseOutputText(annotations=[], text=text)
        last_item = self.output[-1] if self.output else None
        if (
            isinstance(last_item, NeMoGymResponseOutputMessageWithAgentTelemetry)
            and last_item.agent_step_no == generation.agent_step_no
        ):
            last_item.content.append(block)
        else:
            self.output.append(
                NeMoGymResponseOutputMessageWithAgentTelemetry(
                    id=f"msg-{len(self.output)}",
                    content=[block],
                    agent_step_no=generation.agent_step_no,
                )
            )

    def add_reasoning(self, text: str) -> None:
        generation = self._current_agent_step()
        self.output.append(
            NeMoGymResponseReasoningItemWithAgentTelemetry(
                id=f"rs-{len(self.output)}",
                summary=[NeMoGymSummary(text=text, type="summary_text")],
                agent_step_no=generation.agent_step_no,
            )
        )

    def add_tool_call(self, call_id: str, name: str, arguments: str) -> None:
        generation = self._current_agent_step()
        self.output.append(
            NeMoGymResponseFunctionToolCallWithAgentTelemetry(
                call_id=call_id,
                name=name,
                arguments=arguments,
                status="completed",
                agent_step_no=generation.agent_step_no,
            )
        )
        self._pending_calls[call_id] = (generation.agent_step_no, generation.ended_at)

    def add_tool_result(
        self,
        call_id: str,
        output: str,
        completed_at: Optional[str] = None,
        started_at: Optional[str] = None,
        error: Optional[Union[str, NeMoGymToolExecutionError]] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        """Record a tool observation: a ``function_call_output`` item tagged with the
        issuing agent step and carrying its execution telemetry.

        ``started_at`` defaults to the issuing step's completion timestamp; passing it
        explicitly (e.g. from a provider execution record) overrides that. Each result
        carries its own timing, so parallel tool calls stay independently timed. Results
        with no matching call and no step to attach to are counted as dropped.
        """
        step_no, registered_started = self._pending_calls.pop(call_id, (None, None))
        if step_no is None:
            if not self.generations:
                self.count_dropped("orphan_tool_results")
                return
            step_no = self.generations[-1].agent_step_no

        started_at = started_at or registered_started
        duration_ms = None
        started_dt, completed_dt = _parse_timestamp(started_at), _parse_timestamp(completed_at)
        if started_dt is not None and completed_dt is not None:
            duration_ms = (completed_dt - started_dt).total_seconds() * 1000.0

        if isinstance(error, str):
            error = NeMoGymToolExecutionError(message=error)
        self.output.append(
            NeMoGymFunctionCallOutputWithAgentTelemetry(
                call_id=call_id,
                output=output,
                status="completed",
                agent_step_no=step_no,
                execution=NeMoGymToolExecution(
                    started_at=started_at,
                    completed_at=completed_at,
                    duration_ms=duration_ms,
                    error=error,
                    extra=extra,
                ),
            )
        )

    def count_dropped(self, kind: str) -> None:
        self._telemetry.dropped_records[kind] = self._telemetry.dropped_records.get(kind, 0) + 1

    def set_run_totals(
        self,
        num_agent_steps: Optional[int] = None,
        duration_ms: Optional[float] = None,
        total_cost_usd: Optional[float] = None,
        provider_usage: Optional[dict[str, Any]] = None,
    ) -> None:
        if num_agent_steps is not None:
            self._telemetry.num_agent_steps = int(num_agent_steps)
        if duration_ms is not None:
            self._telemetry.duration_ms = float(duration_ms)
        if total_cost_usd is not None:
            self._telemetry.total_cost_usd = float(total_cost_usd)
        if provider_usage is not None:
            self._telemetry.provider_usage = provider_usage

    @property
    def model(self) -> Optional[str]:
        return self._model

    def build(
        self,
    ) -> tuple[list[NeMoGymResponseInputItem], list[NeMoGymGeneration], NeMoGymAgentTelemetry]:
        return self.output, self.generations, self._telemetry
